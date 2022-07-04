import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from pytransform3d.transform_manager import TransformManager
import imu


def main():
    poses = {}
    with open("imu_data.json") as ifile:
        poses = json.load(ifile)

    # load json and draw gt pose
    tm_gt = TransformManager()
    tm_imu_integral = TransformManager()

    white_list = []
    total_pose_for_show = 200
    skip_num = len(poses) // total_pose_for_show

    g = np.array([0, 0, -9.81])
    prev_timestamp = 0
    prev_r_w_i = np.eye(3)
    prev_v_w_i = np.zeros((3, ))
    prev_p_w_i = np.zeros((3, ))
    curr_r_w_i = np.eye(3)
    curr_v_w_i = np.zeros((3, ))
    curr_p_w_i = np.zeros((3, ))

    for i, p in poses.items():
        i = int(i)

        # gt pose
        qx, qy, qz, qw = p['qx'], p['qy'], p['qz'], p['qw']
        px, py, pz = p['px'], p['py'], p['pz']
        ts_ns = p['time_stamp']
        a = np.array([p['ax'], p['ay'], p['az']])
        w = np.array([p['rx'], p['ry'], p['rz']])
        rmat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = rmat
        pose_mat[:3, 3] = [px, py, pz]

        # imu integration
        current_pose = np.eye(4)
        if prev_timestamp == 0:
            prev_r_w_i = imu.init_rotation(a, g)
            current_pose[:3, :3] = prev_r_w_i
            print(prev_r_w_i @ a)
        else:
            delta_t = (ts_ns - prev_timestamp) * 1e-9

            delta_t2 = delta_t * delta_t
            curr_r_w_i = prev_r_w_i @ Rotation.from_rotvec(
                w * delta_t).as_matrix()
            delta_v = (g + prev_r_w_i @ a) * delta_t
            curr_v_w_i = prev_v_w_i + delta_v
            curr_p_w_i = prev_p_w_i + prev_v_w_i * delta_t + 0.5 * g * delta_t2 + 0.5 * (
                prev_r_w_i @ a) * delta_t2
            current_pose[:3, :3] = curr_r_w_i
            current_pose[:3, 3] = curr_p_w_i
            prev_r_w_i = curr_r_w_i.copy()
            prev_v_w_i = curr_v_w_i.copy()
            prev_p_w_i = curr_p_w_i.copy()
            # print(curr_p_w_i)
        prev_timestamp = ts_ns

        if (i % skip_num == 0):
            pose_name = "p" + str(i)
            tm_gt.add_transform(pose_name, "o", pose_mat.copy())
            tm_imu_integral.add_transform(pose_name, "o", current_pose.copy())
            white_list.append(pose_name)

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

    tm_imu_integral.plot_frames_in("p0",
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