import numpy as np

from poses.boundingboxes import find_people_bd


def isInputValid(out_colors, out_seg, points, points3D):

    if not (type(out_colors) is np.ndarray):
        raise Exception("out_color is wrong type: " + str(type(out_colors)))

    if not (type(out_seg) is np.ndarray):
        raise Exception("out_seg is wrong type: " + str(type(out_seg)))

    if not (type(points) is np.ndarray):
        raise Exception("points is wrong type: " + str(type(points)))

    n = out_colors.shape[0]

    if not (3 is  out_colors.shape[1]):
        raise Exception("out_color is wrong shape: " + str(type(out_colors.shape[1])))

    if not (n == len(out_seg)):
        print("--------------------------------------------")
        print(type(n))
        print(type(len(out_seg)))
        print(n)
        print("Out_seg Shape")
        print(len(out_seg))
        print("Color shape")
        print(out_colors.shape)
        print("Points shape")
        print(points.shape)
        raise Exception("out_seg is wrong shape: " + str((out_seg.shape[0])))

    if not (n == points.shape[1]):
        print("--------------------------------------------")
        print(type(n))
        print(type(len(out_seg)))
        print(n)
        print("Out_seg Shape")
        print(len(out_seg))
        print("Color shape")
        print(out_colors.shape)
        print("Points shape")
        print(points.shape)
        raise Exception("out_seg is wrong shape: " + str((out_seg.shape[0])))

        raise Exception("points is wrong shape: " + str((points.shape[1])))

    if not (3 is points.shape[0]):
        raise Exception("points is wrong shape: " + str((points.shape[0])))


def framepoints_to_collection(out_colors, out_seg, points, points3D, scale=5):

    isInputValid(out_colors, out_seg, points, points3D)

    label_hist = out_seg.reshape((-1,1))
    values = np.hstack((out_colors, label_hist))
    points = points.T * scale
    points = points.astype(np.int32)
    points3D_this = list(zip(points, values))
    for key, value in points3D_this:
        points3D[(key[0], key[1], key[2])].append(value)
    return points3D


def refine_voxel_values(points3D):
    # Input: points 3D: a dictionary with positions as key and an array of values as value.
    tensor = {}
    from scipy import stats
    for key, value in points3D.items():
        values_array = np.vstack(value)

        rgb = np.mean(values_array[:, 0:3], axis=0)

        label = stats.mode(values_array[:, 3])[0]

        tensor[key] = [rgb[0], rgb[1], rgb[2], label[0]]
    return tensor


def find_cars(K, P_inv, R, cars, constants, depth, disparity, frame, label):
    positions_p = np.where(label in constants.car_labels)
    clusters = find_people_bd(positions_p, constants)
    for vehicle in clusters:
        car_points = []
        for y in range(vehicle[0], vehicle[1]):
            for x in range(vehicle[2], vehicle[3]):
                if disparity[y, x] > 0:
                    u = np.array([x, y, 1])
                    p_c = np.linalg.solve(K, u)
                    z_v = depth[y, x] * 1.0 / p_c[2]
                    p_c = p_c * z_v
                    p_c = np.dot(P_inv, p_c)
                    p_c = np.append(p_c, 1)
                    p_v = np.dot(R, np.array(p_c))
                    car_points.append(p_v)

        car_points = np.array(car_points * 5)
        cars[frame].append([min(car_points[:, 0]), max(car_points[:, 0]), min(car_points[:, 1]), max(car_points[:, 1]),
                            min(car_points[:, 2]), max(car_points[:, 2])])
    return cars