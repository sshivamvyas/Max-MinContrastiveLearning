import torch
import torch.nn as nn

# This function remains the same as it's a general kernel computation.
def compute_kernel_new(X, Y, gamma=0.1):
    """
    Computes the Radial Basis Function (RBF) kernel between two sets of features X and Y.
    
    Args:
        X (torch.Tensor): First set of features (e.g., anchor embeddings).
        Y (torch.Tensor): Second set of features (e.g., all embeddings in the batch).
        gamma (float): Kernel parameter controlling the width of the RBF.
                       Larger gamma (smaller sigma) means narrower kernel.

    Returns:
        torch.Tensor: The kernel similarity matrix.
    """
    gamma = 1. / float(gamma)
    # Ensure X and Y are floats for matrix multiplication
    X_f = X.float()
    Y_f = Y.float()
    
    # Calculate squared Euclidean distances (scaled by -gamma)
    # This formula (2 - 2 * X.Y^T) is common when X and Y are L2-normalized.
    distances = -gamma * (2 - 2. * torch.mm(X_f, Y_f.T))
    
    # Apply exponential to get kernel similarities
    kernel = torch.exp(distances)
    return kernel

class MMCL_HardNegative(nn.Module):
    """
    Max-Margin Contrastive Loss with Hard Negative Mining.
    This class replaces the PGD optimization with a direct selection of top-k hardest negatives
    based on similarity within the batch.
    """
    def __init__(self, sigma=0.07, batch_size=256, anchor_count=2, C=1.0, k_negatives=1):
        """
        Initializes the MMCL_HardNegative loss module.

        Args:
            sigma (float): Kernel parameter, inverse of gamma.
            batch_size (int): The number of original samples in a batch.
            anchor_count (int): The number of anchors (views) per original sample.
                                This implementation specifically expects anchor_count=2.
            C (float): Regularization parameter (kept for signature consistency but not
                       directly used for clamping alphas as in the PGD version).
            k_negatives (int): The number of hardest negative examples to select for each anchor.
        """
        super(MMCL_HardNegative, self).__init__()
        self.sigma = sigma
        self.C = C # Not directly used for clamping in this version, but kept for consistency
        self.k_negatives = k_negatives

        if anchor_count != 2:
            raise ValueError(
                "MMCL_HardNegative currently assumes 'anchor_count=2' for proper "
                "positive/negative pair extraction. Each sample must have two views."
            )
        
        self.bs = batch_size # Store batch size for clarity
        # k_negatives must be less than the number of other samples (batch_size - 1)
        if self.k_negatives >= (self.bs - 1):
             print(f"Warning: k_negatives ({self.k_negatives}) is too large for batch size ({self.bs}). Setting to {self.bs - 1}.")
             self.k_negatives = self.bs - 1
        if self.k_negatives == 0:
            print("Warning: k_negatives is 0. No hard negatives will be considered.")
            
    def forward(self, features, labels=None, mask=None):
        """
        Computes the hard negative mining-based contrastive loss.

        Args:
            features (torch.Tensor): Input feature embeddings.
                                     Expected shape: (batch_size, anchor_count, embedding_dim).
            labels (torch.Tensor, optional): Not used in this implementation but kept for API consistency.
            mask (torch.Tensor, optional): Not used in this implementation but kept for API consistency.

        Returns:
            tuple: A tuple containing:
                   -pos_loss (torch.Tensor): Negative of the average positive similarity (to be minimized).
                   hard_neg_loss (torch.Tensor): Average of the hardest negative similarities (to be minimized).
        """
        # features shape: (batch_size, anchor_count, embedding_dim)
        # For anchor_count=2, features[:, 0, :] is the first anchor for each sample,
        # and features[:, 1, :] is the second anchor for each sample.
        
        # Unbind features to get two separate sets of anchors (views)
        # F1 will be (batch_size, embedding_dim), F2 will be (batch_size, embedding_dim)
        F1, F2 = torch.unbind(features, dim=1)

        # Compute kernel similarities between the first set of anchors (F1)
        # and the second set of anchors (F2).
        # K_anchor_pairs[i, j] = similarity(F1[i], F2[j])
        # This matrix will have shape: (batch_size, batch_size)
        K_anchor_pairs = compute_kernel_new(F1, F2, gamma=self.sigma)

        # --- 1. Calculate Positive Loss ---
        # Positive pairs are (F1[i], F2[i]), so their similarities are on the diagonal
        pos_similarities = torch.diag(K_anchor_pairs)
        pos_loss = pos_similarities.mean() # Average similarity of all positive pairs

        # --- 2. Calculate Negative Loss using Hard Negative Mining ---
        # Create an identity mask to identify positive pairs (diagonal elements).
        # We'll use this to set positive similarities to a very small value (-inf)
        # so they are not considered when finding top-k negatives.
        identity_mask = torch.eye(self.bs, device=K_anchor_pairs.device).bool()
        
        # Mask out the positive pairs by setting their similarity to negative infinity.
        # This ensures that `topk` only considers true negative pairs.
        neg_similarities = K_anchor_pairs.masked_fill(identity_mask, -float('inf'))

        # Handle the case where k_negatives might be zero
        if self.k_negatives == 0:
            hard_neg_loss = torch.tensor(0.0, device=K_anchor_pairs.device)
        else:
            # Find the top-k hardest (most similar) negatives for each anchor (row) in F1.
            # `topk` along `dim=1` finds the highest values in each row.
            # For each F1[i], we find the k_negatives F2[j] (where j!=i) that have the highest similarity.
            topk_hard_neg_values, _ = neg_similarities.topk(self.k_negatives, dim=1)
            
            # Calculate the mean of these top-k hard negative similarities across all anchors.
            hard_neg_loss = topk_hard_neg_values.mean()

        # --- 3. Combine Losses ---
        # Following the original structure, the objective is to maximize the margin
        # between negative and positive similarities.
        # Minimizing `hard_neg_loss - pos_loss` means pushing negatives away (reducing hard_neg_loss)
        # and pulling positives closer (increasing pos_loss).
        # We return -pos_loss because minimizing this term is equivalent to maximizing pos_loss.
        return -pos_loss, hard_neg_loss
