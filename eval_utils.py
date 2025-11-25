import torch
from einops import rearrange, repeat
import numpy as np

def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation2, rotation1.transpose(0, 2, 1))
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi

def compute_optimal_alignment(A, B):
    """
    Compute the optimal scale s, rotation R, and translation t that minimizes:
    || A - (s * B @ R + T) || ^ 2

    Reference: Umeyama (TPAMI 91)

    Args:
        A (torch.Tensor): (N, 3).
        B (torch.Tensor): (N, 3).

    Returns:
        s (float): scale.
        R (torch.Tensor): rotation matrix (3, 3).
        t (torch.Tensor): translation (3,).
    """
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    # normally with R @ B, this would be A @ B.T
    H = (B - B_bar).T @ (A - A_bar)
    U, S, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    variance = torch.sum((B - B_bar) ** 2)
    scale = 1 / variance * torch.trace(torch.diag(S) @ S_prime)
    R = U @ S_prime @ Vh
    t = A_bar - scale * B_bar @ R

    A_hat = scale * B @ R + t
    return A_hat, scale, R, t


def get_permutations(num_images):
    permutations = []
    for i in range(0, num_images):
        for j in range(0, num_images):
            if i != j:
                permutations.append((j, i))

    return permutations

def n_to_np_rotations(num_frames, n_rots):
    R_pred_rel = []
    permutations = get_permutations(num_frames)
    for i, j in permutations:
        R_pred_rel.append(n_rots[i].T @ n_rots[j])
    R_pred_rel = torch.stack(R_pred_rel)

    return R_pred_rel

def eval_rot(gt_rot, pred_rot):
    n_frames = len(gt_rot)
    gt_perm = n_to_np_rotations(n_frames, gt_rot)
    pred_perm = n_to_np_rotations(n_frames, pred_rot)
    error = compute_angular_error_batch(gt_perm.detach().cpu().numpy(), pred_perm.detach().cpu().numpy())
    return error.mean().item()


def eval_trans(gt_cc, pred_cc):
    scene_scale = torch.linalg.norm(gt_cc[None] - gt_cc[:, None], dim=-1).max()
    A_hat = compute_optimal_alignment(gt_cc, pred_cc)[0]
    norm = torch.linalg.norm(gt_cc - A_hat, dim=1) / (scene_scale + 1e-6)
    return norm.mean().item()