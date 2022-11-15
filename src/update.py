import torch
import torch.nn.functional as F


def hierarchical_update(l1, l2, gamma):
    """Update gamma as proposed in `Hierarchical Multi-task Learning 
    for Organization Evaluation of Argumentative Student Essays`_.
    """
    ratios = l1 / l2
    gamma = torch.max(torch.min(torch.cat((ratios * gamma, 0))), 0.01)
    return gamma


def hierarchical_consistency_update(lam_const, p, p_th, alpha=0.5):
    """Update weights of individual loss functions as proposed in 
    `Keeping Consistency of Sentence Generation and Document Classification 
    with Multi-Task Learning`_.
    """
    lam = lam_const * F.sigmoid((p - p_th) / alpha)
    return lam


def pcgrad(gi, gj):
    """PCGrad update as proposed in `Gradient Surgery for Multi-Task Learning`_."""
    n_tasks = gi.size(dim=0)
    
    for i in range(n_tasks):
        # Compute inner product <gi, gj>
        inner_prod = torch.dot(gi, gj)
    
        # Subtract projection of gi onto gj
    
    
    # Sum all task gradients
    g = torch.sum(gi)
    
    return g