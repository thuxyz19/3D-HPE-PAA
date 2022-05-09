# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

def euclidean_losses(actual, target):
    """Calculate the average Euclidean loss for multi-point samples.
    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).
    Args:
        actual (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)
    """

    assert actual.size() == target.size(), 'input tensors must have the same size'

    # Calculate Euclidean distances between actual and target locations
    diff = actual - target
    dist_sq = diff.pow(2).sum(-1, keepdim=False)
    dist = dist_sq.sqrt()
    return dist


def pck(actual, expected, included_joints=None, threshold=0.15, valid=None):
    dists = euclidean_losses(actual, expected)
    if included_joints is not None:
        dists = dists.gather(-1, torch.LongTensor(included_joints))
    if valid is not None:
        valid = valid.view(dists.shape[0], 1, 1)
    else:
        valid = torch.ones((dists.shape[0], 1, 1), dtype=torch.float32, device=dists.device)

    return ((dists < threshold).double()*valid).mean().item() * dists.shape[0] / valid.sum()


def auc(actual, expected, included_joints=None, valid=None):
    # This range of thresholds mimics `mpii_compute_3d_pck.m`, which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = torch.linspace(0, 150, 31).tolist()

    pck_values = torch.DoubleTensor(len(thresholds))
    for i, threshold in enumerate(thresholds):
        pck_values[i] = pck(actual, expected, included_joints, threshold=threshold/1000.0, valid=valid)
    return pck_values.mean().item()

def mpjpe(predicted, target, valid=None):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if valid is not None:
        valid = valid.view(valid.shape[0], 1, 1)
    else:
        valid = torch.ones((target.shape[0], 1, 1), dtype=torch.float32, device=target.device)
    err = torch.norm(predicted - target, dim=len(target.shape)-1)
    err = err * valid
    e = err.mean() * err.shape[0] / valid.sum()
    return e

def p_mpjpe(predicted, target, valid=None):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    if valid is not None:
        valid = valid.view(valid.shape[0], 1).cpu().numpy()
    else:
        valid = np.ones((target.shape[0], 1), dtype=np.float32)
    # print(predicted_aligned.shape, target.shape)
    err = np.linalg.norm(predicted_aligned - target,  axis=len(target.shape)-1) * valid
    # return err.mean(-1)
    return np.mean(err) * err.shape[0] / np.sum(valid)
    

