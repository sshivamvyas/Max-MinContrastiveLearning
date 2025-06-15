

# Max-Min Contrastive Learning

This repository contains the implementation of **Max-Min Contrastive Learning (MMCL)** — a contrastive learning approach inspired by margin-based classifiers. MMCL improves contrastive learning by selecting the most informative negative samples using max-margin principles, enhancing both convergence and final performance.

## Setup & Installation

### 1. Clone the Repository

Run the following command in a terminal or Kaggle notebook cell:

```bash
!git clone https://github.com/sshivamvyas/Max-MinContrastiveLearning.git
```

### 2. Install Dependencies

Navigate into the cloned directory and install required Python packages:

```bash
!pip install numpy pandas opencv-python torch torchvision tqdm termcolor matplotlib Pillow scikit-learn
```

### 3. Navigate to Project Directory

```bash
%cd /kaggle/working/Max-MinContrastiveLearning/main
```

---

## Usage

### Training on Tiny ImageNet

#### Dataset Note:

Due to size constraints, the Tiny ImageNet dataset is not included in this repository.
Make sure the dataset is available at the following path in your Kaggle environment:

```
/kaggle/input/tiny-image-net/tiny-imagenet-200
```

You can download the dataset from [Kaggle: Tiny ImageNet Dataset](https://www.kaggle.com/c/tiny-imagenet)

#### Train with PGD-based Max-Min Contrastive Loss

```bash
!python mainTinyNet.py --criterion_to_use=mmcl_pgd --epochs=100
```

#### Quick Test Run (e.g., 2 epochs with validation after every epoch)

```bash
!python mainTinyNet.py --criterion_to_use=mmcl_pgd --epochs=2 --val_freq=1
```

#### Hard Negative Mining (No PGD)

To use the efficient hard negative selection method:

```bash
!python mainTinyNet.py --criterion_to_use=MMCL_HardNegative --epochs=100
```

---

### Training on CIFAR-100

#### Full Training (100 epochs)

```bash
!python main.py --criterion_to_use=mmcl_pgd --epochs=100
```

#### Quick Test (2 epochs)

```bash
!python main.py --criterion_to_use=mmcl_pgd --epochs=2 --val_freq=1
```

---

## Notes

* `mmcl_pgd`: Original MMCL loss with adversarial (PGD-based) negative selection
* `MMCL_HardNegative`: Efficient variant using batch-wise hardest negatives
* You can extend this framework to other datasets by modifying the data loader scripts

---

## Reference

This repository is inspired by the following paper:

**Max-Margin Contrastive Learning**
Anshul Shah\*, Suvrit Sra, Rama Chellappa, Anoop Cherian†
Johns Hopkins University, MIT, Mitsubishi Electric Research Labs
[arXiv:2302.00684](https://arxiv.org/abs/2302.00684)

> Shah, A., Sra, S., Chellappa, R., & Cherian, A. (2023). *Max-Margin Contrastive Learning*. arXiv preprint arXiv:2302.00684.

