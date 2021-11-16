import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
# Python Imaging Library imports
from PIL import Image
from PIL import ImageDraw


# Transform a rectangle from x,y offset and width,height to one with a certain angle
def get_rect(x, y, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def get_background():
    """Make an array for the demonstration."""
    X, Y = np.meshgrid(np.linspace(0, np.pi, 512), np.linspace(0, 2, 512))
    z = 0 + 0 * ((np.sin(X) + np.cos(Y)) ** 2 + 0.25)
    data = (255 * (z / z.max())).astype(int)
    return data


def get_polys():
    p1 = get_rect(x=120, y=200, width=100, height=40, angle=30.0)
    p2 = get_rect(x=320, y=100, width=100, height=200, angle=10.0)

    return p1, p2


def drawPolys(polygons, img):
    # Draw some polys on the image.
    draw = ImageDraw.Draw(img)

    for P in polygons:
        draw.polygon([tuple(pos) for pos in P], fill=1)


def getBordersImg(img):
    # Discover the borders of the objects in the image by XOR between an erosion with 2x2 structure AND original image
    binstruct = ndimage.generate_binary_structure(2, 2)
    eroded_img = ndimage.binary_erosion(img, binstruct)
    edges_img = (img > 0) ^ eroded_img
    return edges_img


# Given the pos of the car, unit velocity => occlusion img
def getOcclusionImg(bordersImg, car_pos, car_unit_vel):
    edge_points = np.nonzero(bordersImg)
    for pos in range(len(edge_points[0])):
        edge_point = np.array([edge_points[0][pos], edge_points[1][pos]], dtype=np.float)

        while True:
            # print(edge_point)
            edge_point += car_unit_vel
            roundedPoint_row = int(np.round(edge_point[0]))
            roundedPoint_col = int(np.round(edge_point[1]))

            # print(roundedPoint_row, roundedPoint_col)

            if roundedPoint_row > 0 and roundedPoint_col > 0 and \
                            roundedPoint_row < bordersImg.shape[0] and roundedPoint_col < bordersImg.shape[1]:
                if bordersImg[roundedPoint_row, roundedPoint_col] == 1:
                    break

                bordersImg[roundedPoint_row, roundedPoint_col] = 1
            else:
                break  # Outside of img

    return bordersImg


if __name__ == "__main__":
    data = get_background()

    # Convert the numpy array to an Image object.
    img = Image.fromarray(data.astype(np.uint8))

    polygons = get_polys()
    drawPolys(polygons, img)

    # Convert the Image data to a numpy array.
    base_img = np.asarray(img)

    print("check0")

    # Display the result using matplotlib.  (`img.show()` could also be used.)
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.imshow(base_img, cmap=plt.cm.gray)

    initial_img = getBordersImg(base_img)
    plt.subplot(132)
    plt.imshow(initial_img, cmap=plt.cm.gray)

    occlusionMap = getOcclusionImg(initial_img, np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    plt.subplot(133)
    plt.imshow(occlusionMap, cmap=plt.cm.gray)
    print("checkA")

    plt.show()