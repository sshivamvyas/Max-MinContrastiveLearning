import os
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset # Subset is still imported but not used for subsampling here
import random
import numpy as np

# --- 1. Custom Transform for Two Crops (Two Augmented Views) ---
class TwoCropTransform:
    """
    A custom transform that applies two different augmentations to the same image,
    creating two augmented views (pos_1 and pos_2) for contrastive learning.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

# --- 2. Custom TinyImageNet Dataset to return index ---
class TinyImageNet(ImageFolder):
    """
    Custom TinyImageNet dataset class that inherits from ImageFolder and
    modifies __getitem__ to also return the index of the image.
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        if self.train:
            self.data_path = os.path.join(self.root, 'train')
        else:
            # For TinyImageNet validation, images are in a flat folder and labels are in a file.
            # We'll use ImageFolder assuming a pre-processed val directory structure like:
            # val/
            # ├── images/
            # │   ├── val_0.JPEG
            # │   ├── val_1.JPEG
            # │   └── ...
            # └── val_annotations.txt
            # Ensure your TinyImageNet 'val' directory is structured with subfolders per class
            # for ImageFolder to work out of the box, or adjust this part.
            self.data_path = os.path.join(self.root, 'val')
        
        super().__init__(self.data_path, transform=self.transform)
        # Store targets for easier access, as ImageFolder keeps them internally.
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        """
        Modified __getitem__ to return (image, target, index).
        If the transform is TwoCropTransform, 'image' will be a list of two augmented views.
        """
        path, target = self.samples[index]
        image = self.loader(path) # Default image loader (PIL.Image.Image)

        if self.transform is not None:
            image = self.transform(image) # Apply transform (e.g., TwoCropTransform or single view transform)

        return image, target, index # Return image(s), label, and original index


def get_dataset(dataset_name, dataset_location):
    """
    Loads the full TinyImageNet dataset splits for contrastive learning and evaluation.
    
    Args:
        dataset_name (str): The name of the dataset (e.g., "tiny_imagenet").
        dataset_location (str): The root directory where the dataset is stored.

    Returns:
        tuple: (train_data, memory_data, test_data) - Full datasets:
               - train_data provides two augmented views (from TinyImageNet train split).
               - memory_data provides single views (from TinyImageNet train split, for KNN).
               - test_data provides single views (from TinyImageNet val split, for testing).
    """
    if dataset_name == "tiny_imagenet":
        # Transformations for training (contrastive views)
        train_base_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_transform = TwoCropTransform(train_base_transform)

        # Transformations for memory and test evaluation (single view)
        eval_transform = transforms.Compose([
            transforms.Resize(70),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the full TinyImageNet datasets with appropriate transforms
        # train_data: Full training dataset with two augmented views for contrastive learning
        train_data = TinyImageNet(root=dataset_location, train=True, transform=train_transform)
        
        # memory_data: Full training dataset with single views for the KNN memory bank
        memory_data = TinyImageNet(root=dataset_location, train=True, transform=eval_transform)
        
        # test_data: Full validation dataset with single views for final testing
        test_data = TinyImageNet(root=dataset_location, train=False, transform=eval_transform)

        # No need for index generation or Subset creation as we are using the full datasets directly
        return train_data, memory_data, test_data
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

