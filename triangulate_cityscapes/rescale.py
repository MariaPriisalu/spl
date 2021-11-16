import numpy as np

from .movement import rotation


def rescale_cars(cars, range_reconstruction, scale_x, scale_y, scale_z):
    for vechicles in cars:
        for vehicle in vechicles:
            vehicle[0] = int((vehicle[0] - range_reconstruction[0, 0]) * scale_x)
            vehicle[1] = int((vehicle[1] - range_reconstruction[0, 0]) * scale_x)
            vehicle[2] = int((vehicle[2] - range_reconstruction[1, 0]) * scale_y)
            vehicle[3] = int((vehicle[3] - range_reconstruction[1, 0]) * scale_y)
            vehicle[4] = int((vehicle[4] - range_reconstruction[2, 0]) * scale_z)
            vehicle[5] = int((vehicle[5] - range_reconstruction[2, 0]) * scale_z)
            #vehicle[[0, 1, 2, 3, 4, 5]] = vehicle[[4, 5, 2, 3, 0, 1]]
    return cars


def rescale_pedestrians(people, range_reconstruction, scale_x, scale_y, scale_z):
    for pedestrians in people:
        for pedestrian in pedestrians:
            pedestrian[0] = ((pedestrian[0] - range_reconstruction[0, 0]) * scale_x).astype(int)
            pedestrian[1] = ((pedestrian[1] - range_reconstruction[1, 0]) * scale_y).astype(int)
            pedestrian[2] = ((pedestrian[2] - range_reconstruction[2, 0]) * scale_z).astype(int)
            pedestrian[[0, 1, 2]] = pedestrian[[2, 1, 0]]
    return people


def rescale3D(constants, range_reconstruction, points3D, points3D_RGB, label_hist):
    scale = 1.0  # 0.2
    if constants.same_coordinates:
        scale_x = (range_reconstruction[0, 1] - range_reconstruction[0, 0] + 1) / scale  # 127
        scale_y = (range_reconstruction[1, 1] - range_reconstruction[1, 0] + 1) / scale  # 255
        scale_z = (range_reconstruction[2, 1] - range_reconstruction[2, 0] + 1) / scale  # 32
        print(scale_x)
        print(scale_y)
        print(scale_z)
        print(len(points3D_RGB))
        print(len(label_hist))
        print(len(points3D))
        tensor = np.zeros(shape=(int(scale_z), int(scale_y), int(scale_x), 4))

        H, edges = np.histogramdd(points3D, bins=(scale_x * 5, scale_y * 5, scale_z * 5, 4))
        print(H)

        if constants.reconstruct_each_frame:
            voxel_color = {}
            voxel_label = {}
            for point in points3D:
                x = int((point[0] - range_reconstruction[0, 0]) / scale)
                y = int((point[1] - range_reconstruction[1, 0]) / scale)
                z = int((point[2] - range_reconstruction[2, 0]) / scale)
                point_new_coordinates = (x, y, z)
                if point_new_coordinates not in voxel_color:
                    voxel_color[point_new_coordinates] = []
                    voxel_label[point_new_coordinates] = {}  # create entry for new coordinate.
                voxel_color[point_new_coordinates] += points3D_RGB[point]
                # dictionary= label_hist[point]
                for label, count in label_hist[point].items():  # Go through all labels in label histogram
                    if label in voxel_label[point_new_coordinates]:
                        voxel_label[point_new_coordinates][label] += count
                    else:
                        voxel_label[point_new_coordinates][label] = count

            for pos in voxel_color:
                x = pos[0]
                y = pos[1]
                z = pos[2]

                colors = np.array(voxel_color[x, y, z])
                tensor[z, y, x, 0:3] = np.mean(colors)
                max_val = max(list(zip(list(voxel_label[pos].values()), list(voxel_label[pos].keys()))))
                tensor[z, y, x, 3] = max_val[1]
        else:
            points3D = np.array(points3D)
            origin = range_reconstruction[:, 0].reshape((1, 3))
            points3D = points3D - np.tile(origin, (points3D.shape[0], 1))
            points3D = np.multiply(points3D, (scale, scale, scale)).astype(np.int)
            points3D[:, [0, 1, 2]] = points3D[:, [2, 1, 0]]
            label_hist = (np.array(label_hist)).reshape((-1, 1))
            tmp = np.array(points3D_RBG)
            values = np.hstack((tmp, label_hist))
            points3D = list(zip(points3D, values))
            previous_pos = points3D[0][0]
            labels = []
            RGB = []
            for pair in points3D:
                if not sum(pair[0] - previous_pos) == 0:
                    RGB = np.array(RGB)
                    tensor[previous_pos[0], previous_pos[1], previous_pos[2], 0:3] = np.mean(np.copy(RGB), axis=0)
                    tensor[previous_pos[0], previous_pos[1], previous_pos[2], 3] = max(labels, key=labels.count)
                    RGB = []
                    labels = []
                    previous_pos = pair[0]
                labels.append(pair[1][3])
                RGB.append(np.array(pair[1][0:3], copy=True))
            tensor[previous_pos[0], previous_pos[1], previous_pos[2], 0:3] = np.mean(RGB)
            tensor[previous_pos[0], previous_pos[1], previous_pos[2], 3] = max(labels, key=labels.count)
        return tensor, scale_x, scale_y, scale_z


def points_to_same_coord(points, range_reconstruction, translation_x, translation_y, yaw_est):
    points = np.dot(rotation(yaw_est), np.array(points))
    points[0, :] -= translation_x
    points[1, :] -= translation_y
    range_reconstruction[0, 0] = np.min(points[0, :]) if range_reconstruction[0, 0] > np.min(points[0, :]) else \
        range_reconstruction[0, 0]  # min x
    range_reconstruction[0, 1] = np.max(points[0, :]) if range_reconstruction[0, 1] < np.max(points[0, :]) else \
        range_reconstruction[0, 1]  # max x
    range_reconstruction[1, 0] = np.min(points[1, :]) if range_reconstruction[1, 0] > np.min(points[1, :]) else \
        range_reconstruction[1, 0]  # min y
    range_reconstruction[1, 1] = np.max(points[1, :]) if range_reconstruction[1, 1] < np.max(points[1, :]) else \
        range_reconstruction[1, 1]  # max y
    range_reconstruction[2, 0] = np.min(points[2, :]) if range_reconstruction[2, 0] > np.min(points[2, :]) else \
        range_reconstruction[2, 0]  # min z
    range_reconstruction[2, 1] = np.max(points[2, :]) if range_reconstruction[2, 1] < np.max(points[2, :]) else \
        range_reconstruction[2, 1]  # max z
    return points, range_reconstruction