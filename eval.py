# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from src.cfg import load_config
import torch
import torch.nn as nn
import os
from src.camera import *
from src.model import *
from src.loss import *
from src.cps import compute_CP_list
from src.generators import UnchunkedGenerator
import random
from src.arguments import parse_args

args = parse_args()
cfg = load_config(args.config)
SEED = cfg.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print('Loading dataset...')
dataset_path = 'data/data_3d_' + cfg.dataset + '.npz'
from src.h36m_dataset import Human36mDataset
dataset = Human36mDataset(dataset_path)
print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d
print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + cfg.dataset + '_' + cfg.keypoints + '.npz', allow_pickle=True)
keypoints_gt = np.load('data/data_2d_' + cfg.dataset + '_' + 'gt' + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()
keypoints_gt = keypoints_gt['positions_2d'].item()
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue
        for cam_idx in range(len(keypoints[subject][action])):
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            assert keypoints_gt[subject][action][cam_idx].shape[0] >= mocap_length
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
            if keypoints_gt[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints_gt[subject][action][cam_idx] = keypoints_gt[subject][action][cam_idx][:mocap_length]
        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
        assert len(keypoints_gt[subject][action]) == len(dataset[subject][action]['positions_3d'])
for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps
for subject in keypoints_gt.keys():
    for action in keypoints_gt[subject]:
        for cam_idx, kps_gt in enumerate(keypoints_gt[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps_gt[..., :2] = normalize_screen_coordinates(kps_gt[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints_gt[subject][action][cam_idx] = kps_gt
subjects_train = cfg.subjects_train
subjects_test = cfg.subjects_test
receptive_field = cfg.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
def fetch_actions(actions):
    out_poses_3d = []
    out_poses_2d = []
    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)):  # Iterate across cameras
            out_poses_2d.append(poses_2d[i])
        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)):  # Iterate across cameras
            out_poses_3d.append(poses_3d[i])
    stride = cfg.downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    return out_poses_3d, out_poses_2d
def prepare_actions_train():
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_train:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}
        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))
    return all_actions, all_actions_by_subject

_, all_actions_by_subject_train = prepare_actions_train()
S_list = ['S1','S5','S6', 'S7','S8']
Action_list = ['Directions', 'Photo', 'Discussion', 'Walking',
               'Purchases', 'Phoning', 'Eating', 'Sitting', 'Walking', 'WalkDog', 'Waiting',
               'Posing', 'Greeting', 'Smoking', 'SittingDown']
ref_3D = []
for i in range(16):
    SUBJECT = S_list[random.randint(0, len(S_list) - 1)]
    ACTION = Action_list[random.randint(0, len(Action_list) - 1)]
    poses_act, poses_2d_act = fetch_actions(all_actions_by_subject_train[SUBJECT][ACTION])
    camera_random = random.randint(0, 3)
    start = random.randint(0, poses_act[camera_random].shape[0] - 1 - 64)
    ref_3D_ = torch.from_numpy(poses_act[camera_random][start:start+64]).float()
    ref_3D.append(ref_3D_)
ref_3D = torch.cat(ref_3D, 0)
ref_3D[:, 0] = 0.0
model_pos = Model(cfg, seed=ref_3D)
if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
chk_filename = os.path.join(cfg.checkpoint)
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p

def evaluate_batch(test_generator, action=None, out=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_cps = 0
    with torch.no_grad():
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]
            ##### convert size
            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda(non_blocking=True)
                inputs_2d_flip = inputs_2d_flip.cuda(non_blocking=True)
                inputs_3d = inputs_3d.cuda(non_blocking=True)
            inputs_3d[:, :, 0] = 0
            num_samples = 1500
            if inputs_2d.shape[0] <= num_samples:
                predicted_3d_pos = model_pos(inputs_2d)
                predicted_3d_pos_flip = model_pos(inputs_2d_flip)
            else:
                predicted_3d_pos = []
                predicted_3d_pos_flip = []
                for kk in range(inputs_2d.shape[0] // num_samples + 1):
                    predicted_3d_pos.append(
                        model_pos(inputs_2d[kk * num_samples: min(kk * num_samples + num_samples, inputs_2d.shape[0]), ...]))
                    predicted_3d_pos_flip.append(
                        model_pos(inputs_2d_flip[kk * num_samples: min(kk * num_samples + num_samples, inputs_2d.shape[0]), ...]))
                predicted_3d_pos = torch.cat(predicted_3d_pos, dim=0)
                predicted_3d_pos_flip = torch.cat(predicted_3d_pos_flip, dim=0)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]
            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                          keepdim=True)
            error = mpjpe(predicted_3d_pos, inputs_3d)
            CPS = compute_CP_list(inputs_3d.squeeze(1).cpu().permute(0, 2, 1), predicted_3d_pos.squeeze(1).cpu().permute(0, 2, 1))
            CPS = CPS.mean()
            epoch_cps += inputs_3d.shape[0] * inputs_3d.shape[1] * CPS.item()
            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    ecps = epoch_cps / N
    if out:
        if action is None:
            print('----------')
        else:
            print('----' + action + '----')
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('CPS:', ecps)
        print('----------')
    return e1, e2, ecps

def prepare_actions():
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}
        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))
    return all_actions, all_actions_by_subject

def run_evaluation(actions, out=False):
    errors_p1 = []
    errors_p2 = []
    cpss = []
    for action_key in actions.keys():
        poses_act, poses_2d_act = fetch_actions(actions[action_key])
        gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                 pad=pad, causal_shift=False, augment=False,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                 joints_right=joints_right)
        e1, e2, ecps = evaluate_batch(gen, action_key, out=out)
        errors_p1.append(e1)
        errors_p2.append(e2)
        cpss.append(ecps)
    print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
    print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
    print('CPS:', np.mean(cpss))

print('Evaluating...')
all_actions, all_actions_by_subject = prepare_actions()
run_evaluation(all_actions, out=True)