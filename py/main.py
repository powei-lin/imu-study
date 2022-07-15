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
    imu_freq = 500
    with open("imu_data" + str(imu_freq) + ".json") as ifile:
        poses = json.load(ifile)

    # load json and draw gt pose
    tms = [TransformManager() for _ in range(3)]

    white_list = []
    total_pose_for_show = 200
    skip_num = len(poses) // total_pose_for_show

    g = np.array([0, 0, -9.81])

    # normal integration
    prev_timestamp = 0
    prev_r_w_i = np.eye(3)
    prev_v_w_i = np.zeros((3,))
    prev_p_w_i = np.zeros((3,))
    curr_r_w_i = np.eye(3)
    curr_v_w_i = np.zeros((3,))
    curr_p_w_i = np.zeros((3,))

    # preintegration
    preintegral_timestamp_i = 0
    preintegral_ri = np.eye(3)
    preintegral_vi = np.zeros((3,))
    preintegral_pi = np.zeros((3,))
    preintegral_delta_r = np.eye(3)
    preintegral_delta_v = np.zeros((3,))
    preintegral_delta_p = np.zeros((3,))

    for i, p in poses.items():
        i = int(i)

        # gt pose
        qx, qy, qz, qw = p["qx"], p["qy"], p["qz"], p["qw"]
        px, py, pz = p["px"], p["py"], p["pz"]
        ts_ns = p["time_stamp"]
        a = np.array([p["ax"], p["ay"], p["az"]])
        w = np.array([p["rx"], p["ry"], p["rz"]])
        rmat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = rmat
        pose_mat[:3, 3] = [px, py, pz]

        # imu integration
        current_pose_integral = np.eye(4)
        current_pose_preintegral = np.eye(4)

        if prev_timestamp == 0:
            prev_r_w_i = imu.init_rotation(a, g)
            current_pose_integral[:3, :3] = prev_r_w_i
            # print(prev_r_w_i @ a)
            print(prev_r_w_i)

            current_pose_preintegral = current_pose_integral.copy()
            preintegral_timestamp_i = ts_ns
            preintegral_ri = current_pose_preintegral[:3, :3].copy()
            preintegral_pi = current_pose_preintegral[:3, 3].copy()
        else:
            delta_t = (ts_ns - prev_timestamp) * 1e-9
            delta_t2 = delta_t * delta_t

            # integrate
            curr_r_w_i = prev_r_w_i @ Rotation.from_rotvec(w * delta_t).as_matrix()
            delta_v = (g + prev_r_w_i @ a) * delta_t
            curr_v_w_i = prev_v_w_i + delta_v
            curr_p_w_i = (
                prev_p_w_i
                + prev_v_w_i * delta_t
                + 0.5 * g * delta_t2
                + 0.5 * (prev_r_w_i @ a) * delta_t2
            )
            current_pose_integral[:3, :3] = curr_r_w_i
            current_pose_integral[:3, 3] = curr_p_w_i

            prev_r_w_i = curr_r_w_i.copy()
            prev_v_w_i = curr_v_w_i.copy()
            prev_p_w_i = curr_p_w_i.copy()

            # preintegrate
            preintegral_delta_t = (ts_ns - preintegral_timestamp_i) * 1e-9
            preintegral_delta_v += preintegral_delta_r @ (a * delta_t)

            # TODO why is this different from the paper
            if preintegral_delta_t - delta_t > 0:
                preintegral_delta_p *= (preintegral_delta_t) / (
                    preintegral_delta_t - delta_t
                )
            preintegral_delta_p += 0.5 * (
                (preintegral_delta_r @ a) * delta_t * preintegral_delta_t
            )

            preintegral_delta_r = (
                preintegral_delta_r @ Rotation.from_rotvec(w * delta_t).as_matrix()
            )
            # print("pre d p", preintegral_delta_p)
        prev_timestamp = ts_ns

        if i % skip_num == 0:
            pose_name = "p" + str(i)
            white_list.append(pose_name)

            tms[0].add_transform(pose_name, "o", pose_mat.copy())
            tms[1].add_transform(pose_name, "o", current_pose_integral.copy())

            # add preintegral delta to current state
            # current_pose_preintegral = current_pose_integral.copy()
            if ts_ns != preintegral_timestamp_i:
                pre_delta_t = (ts_ns - preintegral_timestamp_i) * 1e-9
                pre_delta_t2 = pre_delta_t * pre_delta_t
                # print( 0.5 * g * pre_delta_t2 / (preintegral_ri @ preintegral_delta_p))
                # return
                preintegral_pi += (
                    preintegral_vi * pre_delta_t
                    + 0.5 * g * pre_delta_t2
                    + preintegral_ri @ preintegral_delta_p
                )
                preintegral_vi += g * pre_delta_t + preintegral_ri @ preintegral_delta_v
                preintegral_ri = preintegral_ri @ preintegral_delta_r
                current_pose_preintegral[:3, :3] = preintegral_ri.copy()
                current_pose_preintegral[:3, 3] = preintegral_pi.copy()

                # debug copy position from integration
                # print(current_pose_integral-current_pose_preintegral)
                # return
                # current_pose_preintegral[:3, 3] = current_pose_integral[:3, 3].copy()

            tms[2].add_transform(pose_name, "o", current_pose_preintegral.copy())
            # reset preintegral delta
            preintegral_timestamp_i = ts_ns
            preintegral_delta_r = np.eye(3)
            preintegral_delta_v = np.zeros((3,))
            preintegral_delta_p = np.zeros((3,))

    ll = 3
    fig = plt.figure(figsize=(16, 8))
    axs = [fig.add_subplot(131 + i, projection="3d") for i in range(3)]
    fig_names = ["ground truth pose", "imu integration pose", "preintegration"]

    for tm, ax, n in zip(tms, axs, fig_names):
        tm.plot_frames_in("p0", s=ll / 10, show_name=False, whitelist=white_list, ax=ax)
        # ax = tm.plot_frames_in("o", s=ll / 10, show_name=False, whitelist=white_list)
        ax.set_xlim((-ll, ll))
        ax.set_ylim((-ll, ll))
        ax.set_zlim((-ll, ll))
        ax.view_init(elev=0, azim=0, vertical_axis="y")
        ax.set_title(n)

    plt.show()


if __name__ == "__main__":
    main()
