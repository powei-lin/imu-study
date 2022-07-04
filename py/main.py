import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from pytransform3d.transform_manager import TransformManager


def main():
    poses = {}
    with open("out.json") as ifile:
        poses = json.load(ifile)

    # load json and draw gt pose
    tm_gt = TransformManager()
    white_list = []
    total_pose_for_show = 250
    skip_num = len(poses) // total_pose_for_show

    for i, p in poses.items():
        i = int(i)
        qx, qy, qz, qw = p['qx'], p['qy'], p['qz'], p['qw']
        px, py, pz = p['px'], p['py'], p['pz']
        rmat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = rmat
        pose_mat[:3, 3] = [px, py, pz]

        if (i % skip_num == 0):
            tm_gt.add_transform("p" + str(i), "o", pose_mat.copy())
            white_list.append("p" + str(i))

    ll = 2
    fig = plt.figure(figsize=(10, 5))
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')

    tm_gt.plot_frames_in("p0",
                         s=ll / 10,
                         show_name=False,
                         whitelist=white_list,
                         ax=ax0)
    # ax = tm.plot_frames_in("o", s=ll / 10, show_name=False, whitelist=white_list)
    ax0.set_xlim((-ll, ll))
    ax0.set_ylim((-ll, ll))
    ax0.set_zlim((-ll, ll))
    ax0.view_init(elev=0, azim=0, vertical_axis='y')

    tm_gt.plot_frames_in("p0",
                         s=ll / 10,
                         show_name=False,
                         whitelist=white_list,
                         ax=ax1)
    ax1.set_xlim((-ll, ll))
    ax1.set_ylim((-ll, ll))
    ax1.set_zlim((-ll, ll))
    ax1.view_init(elev=0, azim=0, vertical_axis='y')
    plt.show()


if __name__ == "__main__":
    main()