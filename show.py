import matplotlib.pyplot as plt
import pylab as mpl
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np

print('Loading dataset...')
dataset_path = 'data/data_3d_' + 'h36m' + '.npz'
from src.h36m_dataset import Human36mDataset
dataset = Human36mDataset(dataset_path)
skeleton = dataset.skeleton()
parents = skeleton.parents()
cam = dataset.cameras()
azim = 0
radius = 1.7

def draw(pose, out='./show'):
    print(pose.shape)
    # pose: (F, 17, 3) numpy array   
    fig = plt.figure()
    for i in range(pose.shape[0]):
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(elev=90., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        #ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title('3D skeleton')  # , pad=35
        pos = pose[i]
        # pos = pos[:, (2, 1, 0)]
        pos = pos[:, (1, 0, 2)]
        pos[:, 2] = -1.0 * pos[:, 2]
        for j, j_parent in enumerate(parents):
            if j_parent == -1:
                continue
            col = 'red' if j in skeleton.joints_right() else 'black'
            ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 1], pos[j_parent, 1]],[pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)
        plt.savefig(os.path.join(out,'%06d.jpg'%i))
        fig.clear()