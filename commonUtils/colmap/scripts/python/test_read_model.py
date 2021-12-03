import numpy as np
from read_model import read_model


def compare_cameras(cameras1, cameras2):
    assert len(cameras1) == len(cameras2)
    for camera_id1 in cameras1:
        camera1 = cameras1[camera_id1]
        camera2 = cameras2[camera_id1]
        assert camera1.id == camera2.id
        assert camera1.width == camera2.width
        assert camera1.height == camera2.height
        assert np.allclose(camera1.params, camera2.params)


def compare_images(images1, images2):
    assert len(images1) == len(images2)
    for image_id1 in images1:
        image1 = images1[image_id1]
        image2 = images2[image_id1]
        assert image1.id == image2.id
        assert np.allclose(image1.qvec, image2.qvec)
        assert np.allclose(image1.tvec, image2.tvec)
        assert image1.camera_id == image2.camera_id
        assert image1.name == image2.name
        assert np.allclose(image1.xys, image2.xys)
        assert np.array_equal(image1.point3D_ids, image2.point3D_ids)


def compare_points(points3D1, points3D2):
    for point3D_id1 in points3D1:
        point3D1 = points3D1[point3D_id1]
        point3D2 = points3D2[point3D_id1]
        assert point3D1.id == point3D2.id
        assert np.allclose(point3D1.xyz, point3D1.xyz)
        assert np.array_equal(point3D1.rgb, point3D1.rgb)
        assert np.allclose(point3D1.error, point3D2.error)
        assert np.array_equal(point3D1.image_ids, point3D1.image_ids)
        assert np.array_equal(point3D1.point2D_idxs, point3D1.point2D_idxs)


def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python read_model.py "
              "path/to/model/folder/txt path/to/model/folder/bin")
        return

    print("Comparing text and binary models ...")

    path_to_model_txt_folder = sys.argv[1]
    path_to_model_bin_folder = sys.argv[2]
    cameras_txt, images_txt, points3D_txt = \
        read_model(path_to_model_txt_folder, ext=".txt")
    cameras_bin, images_bin, points3D_bin = \
        read_model(path_to_model_bin_folder, ext=".bin")
    compare_cameras(cameras_txt, cameras_bin)
    compare_images(images_txt, images_bin)
    compare_points(points3D_txt, points3D_bin)

    print("... text and binary models are equal.")


if __name__ == "__main__":
    main()
