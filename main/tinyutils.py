import os
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import random # Import the random module
import numpy as np # Import numpy for shuffling and splitting

# Define the TinyImageNet dataset class if it's not already defined elsewhere
# This is a common way to load TinyImageNet, assuming its structure.
class TinyImageNet(ImageFolder):
    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root (string): Root directory of the TinyImageNet dataset.
            train (bool): If True, loads the training set, otherwise loads the validation set.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version.
        """
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
            # Or, if ImageFolder expects subdirectories per class, you might need to
            # restructure your 'val' directory or use a custom dataset loading logic.
            # For simplicity, we'll assume a structure compatible with ImageFolder
            # if `root/val` itself contains class subdirectories.
            # If not, a custom loading from val_annotations.txt is needed.
            # Let's assume a common setup where 'val' is restructured or a simple ImageFolder works.
            self.data_path = os.path.join(self.root, 'val')
        
        super().__init__(self.data_path, transform=self.transform)
        # Store targets for easier access, as ImageFolder keeps them internally.
        self.targets = [s[1] for s in self.samples]


def get_dataset(dataset_name, dataset_location):
    """
    Loads and subsamples the specified dataset.
    
    Args:
        dataset_name (str): The name of the dataset (e.g., "tiny_imagenet").
        dataset_location (str): The root directory where the dataset is stored.

    Returns:
        tuple: (train_subset, memory_subset, test_subset) - Subsampled datasets.
    """
    if dataset_name == "tiny_imagenet":
        # Define common transformations for TinyImageNet
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(70),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the full TinyImageNet training and validation datasets
        full_train_dataset = TinyImageNet(root=dataset_location, train=True, transform=train_transform)
        full_val_dataset = TinyImageNet(root=dataset_location, train=False, transform=test_transform) # Use val for testing

        # Define the desired number of samples for each subset
        num_train_samples = 5000
        num_memory_samples = 500
        num_test_samples = 500 # This will be taken from the validation set

        # --- Generate indices for subsampling ---
        # Ensure we have enough samples in the full datasets
        if len(full_train_dataset) < (num_train_samples + num_memory_samples):
            raise ValueError(
                f"Not enough samples in training dataset ({len(full_train_dataset)}) "
                f"to create {num_train_samples} train and {num_memory_samples} memory subsets."
            )
        if len(full_val_dataset) < num_test_samples:
            raise ValueError(
                f"Not enough samples in validation dataset ({len(full_val_dataset)}) "
                f"to create {num_test_samples} test subset."
            )

        # Create a shuffled list of indices for the full training dataset
        train_val_indices = list(range(len(full_train_dataset)))
        random.shuffle(train_val_indices) # Shuffle to ensure random sampling

        # Allocate indices for training and memory subsets from the shuffled training indices
        train_indices = train_val_indices[:num_train_samples]
        memory_indices = train_val_indices[num_train_samples : num_train_samples + num_memory_samples]

        # Create indices for the test subset from the validation dataset
        test_indices = list(range(len(full_val_dataset)))
        random.shuffle(test_indices)
        test_indices = test_indices[:num_test_samples]

        # Create the Subsets
        train_data = Subset(full_train_dataset, train_indices)
        memory_data = Subset(full_train_dataset, memory_indices)
        test_data = Subset(full_val_dataset, test_indices) # Test data from validation set

        return train_data, memory_data, test_data
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

