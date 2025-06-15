Max-Min Contrastive Learning
This repository contains the code for the Max-Min Contrastive Learning project.

Setup and Installation
To get started with this project, follow these steps:

Clone the Repository:
Open your terminal or command prompt (or a Kaggle notebook cell) and clone the GitHub repository using the following command:

!git clone https://github.com/sshivamvyas/Max-MinContrastiveLearning.git

Install Dependencies:
Navigate into the cloned directory and install all necessary Python packages. If you're in a Kaggle notebook, you can run this directly in a cell:

!pip install numpy pandas opencv-python torch torchvision tqdm termcolor matplotlib Pillow scikit-learn

Navigate to Project Directory:
Change your current working directory to the project's root:

%cd /kaggle/working/Max-MinContrastiveLearning

Usage
Once you've set up the project, you can run the training scripts for different datasets.

Tiny ImageNet Dataset
Note on Dataset: The Tiny ImageNet dataset is large and cannot be directly included in the GitHub repository. You will need to upload the dataset to your Kaggle environment.

Please ensure the dataset is located at the following path in your Kaggle environment: /kaggle/input/tiny-image-net/tiny-imagenet-200.

You can find the dataset on Kaggle here: Tiny ImageNet Dataset Link

The base paper for this project utilized criterion_to_use=mmcl_pgd for its experiments.

To train the model on the Tiny ImageNet dataset for 100 epochs using mmcl_pgd as the criterion, use the following command:

!python mainTinyNet.py --criterion_to_use=mmcl_pgd --epochs=100

For a quicker test run with fewer epochs (e.g., 2 epochs) and more frequent validation (e.g., every epoch), you can modify the command like this:

!python mainTinyNet.py --criterion_to_use=mmcl_pgd --epochs=2 --val_freq=1

Proposed Methodologies Testing: Hard Negative Mining
For testing the Hard Negative Mining without PGD approach, which replaces expensive PGD optimization with top-k hardest negatives based on similarity within a batch (improving efficiency while still focusing on informative negatives), use the following command:

!python mainTinyNet.py --criterion_to_use=MMCL_HardNegative --epochs=100

CIFAR-100 Dataset
To train the model on the CIFAR-100 dataset for 100 epochs using mmcl_pgd as the criterion, run:

!python main.py --criterion_to_use=mmcl_pgd --epochs=100

If you want to test with fewer epochs (e.g., 2 epochs), use this command:

!python main.py --criterion_to_use=mmcl_pgd --epochs=2 --val_freq=1
