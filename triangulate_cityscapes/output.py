import numpy as np


def save_3DFile(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header    '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


def write_ply(filename, verts, colors, constants):
    # verts = verts.reshape(-1, 3)
    # colors = colors.reshape(-1, 3)

    verts = np.hstack([verts, colors])
    with open(filename, 'wb') as f:
        f.write((constants.ply_header % dict(vert_num=len(verts) + 1)).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def write_ply_dict(filename, dict3d, constants):
    with open(filename, 'wb') as f:
        f.write((constants.ply_header % dict(vert_num=len(dict3d) + 1)).encode('utf-8'))

        for point in dict3d:
            color = np.mean(dict3d[point], axis=0).astype(int)
            f.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + " " + str(color[0]) + " " + str(
                color[1]) + " " + str(color[2]) + "\n")

            # print color


def write_skeleton_ply(filename, joints, constants):
    limbs = np.matrix('1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14')
    colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0]
        , [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]]  # note BGR ...
    with open(filename, 'wb') as f:
        f.write((constants.ply_header_skeleton % dict(vert_num=len(joints) + 1)).encode('utf-8'))
        middle = (joints[8, :] + joints[11, :]) * 0.5
        np.savetxt(f, joints, fmt='%f %f %f 255 255 255 ')
        f.write((str(middle[0]) + " " + str(middle[1]) + " " + str(middle[2]) + " 255 255 255 ").encode('utf-8'))
        f.write(constants.skeleton_edges)


def write_skeleton_ply2(filename, joints, joints2, constants):
    print("------1: ")
    print(joints)
    print("------2:")
    print(joints2)
    print("------")
    limbs = np.matrix('1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14')
    colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0]
        , [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]]  # note BGR ...
    with open(filename, 'wb') as plyfile:
        plyfile.write((constants.ply_header_skeleton2 % dict(vert_num=len(joints) + 2 + len(joints2))).encode('utf-8'))
        middle = (joints[8, :] + joints[11, :]) * 0.5
        np.savetxt(plyfile, joints, fmt='%f %f %f 255 255 255 ')
        plyfile.write(
            (str(middle[0]) + " " + str(middle[1]) + " " + str(middle[2]) + " 255 255 255 \n").encode('utf-8'))

        middle = (joints2[8, :] + joints2[11, :]) * 0.5
        np.savetxt(plyfile, joints2, fmt='%f %f %f 255 255 255 ')
        plyfile.write((str(middle[0]) + " " + str(middle[1]) + " " + str(middle[2]) + " 255 255 255 ").encode('utf-8'))

        plyfile.write(constants.skeleton_edges2)