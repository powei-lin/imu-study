from scipy.spatial.transform import Rotation
import numpy as np


def init_rotation(acc, g):
    v0 = acc
    v1 = -g
    unit_vector_1 = v0 / np.linalg.norm(v0)
    unit_vector_2 = v1 / np.linalg.norm(v1)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    rvec = np.cross(v0, v1)
    rvec = rvec / np.linalg.norm(rvec) * angle
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    return rmat
