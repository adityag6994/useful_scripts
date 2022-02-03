# Aditya Gupta
# Impliment conv operation on image
# 1) ignore boundaries
# 2) add boundary condition (mirror)

import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors

cmap1 = ['#7fc97f', '#ffff99', '#386cb0', '#f0027f']
cmap = colors.ListedColormap(cmap1)


def on_boundary(i, j, filter_size, img_size):
    """
    check if coordinate is on boundary or not
    :param i:
    :param j:
    :param filter_shape:
    :return:
    """
    # corners
    if (i == 0 and j == 0) or (i == 0 and j == img_size[1] - 1) or (i == img_size[0] - 1 and j == 0) or (
            i == img_size[0] - 1 and j == img_size[1] - 1):
        return False

    # border except corner
    if i > (img_size[0] - (filter_size / 2) - 1) or (filter_size / 2) > i:
        return False

    if j > (img_size[1] - (filter_size / 2) - 1) or (filter_size / 2) > j:
        return False

    return True


def apply_filter_to_location(img, x, y, conv_filter, filter_size):
    """
    Apply conv filter at location img[i][j]
    :param img:
    :param x:
    :param y:
    :param conv_filter:
    :return:
    """
    val = 0
    for i in range(-int(filter_size / 2), int(filter_size / 2) + 1):
        for j in range(-int(filter_size / 2), int(filter_size / 2) + 1):
            val += img[x + i][y + j] * conv_filter[int(filter_size / 2) + i][int(filter_size / 2) + j]
            # val/=9
    return val


def update_value_reflect(x, img_size):
    """
    reflect update
    :param x:
    :param img_size:
    :return:
    """
    if x < 0:
        x = -x - 1
    elif x >= img_size[0]:
        x = 2 * img_size[0] - x - 1
    return x


def update_value_circular(x, img_size):
    """
    circular update
    :param x:
    :param img_size:
    :return:
    """
    if x < 0:
        x = img_size[0]+x
    elif x >= img_size[0]:
        x = x-img_size[0]
    return x


def apply_filter_to_boundary(img, x, y, conv_filter, filter_size, img_size, boundary_type="reflect"):
    """val[
    Apply filter to boundary location, using boundary folrmula, update the value based on type
    :param img:
    :param x:
    :param y:
    :param conv_filter:
    :param filter_size:
    :return:
    """
    val = 0
    for i in range(-int(filter_size / 2), int(filter_size / 2) + 1):
        for j in range(-int(filter_size / 2), int(filter_size / 2) + 1):
            try:
                # print("row : ", x,i, " | col : ", y,j)
                if boundary_type=="reflect":
                    row = update_value_reflect(x+i, img_size)
                    col = update_value_reflect(y+j, img_size)
                elif boundary_type=="circular":
                    row = update_value_circular(x + i, img_size)
                    col = update_value_circular(y + j, img_size)
                val += img[row][col] * conv_filter[int(filter_size / 2) + i][int(filter_size / 2) + j]
            except:
                print("row : ", x,i, " | col : ", y,j)
    return val


def apply_conv_circular(img, img_size, filter):
    """
    Apply conv to image with circular boundary conditions
    :param img_size: size of image
    :param img: input image
    :return: image after applying convolution operation
    """
    img_conv = copy.deepcopy(img)
    # img_conv.fill(0)
    conv_filter = filter
    filter_size = len(filter)
    for i, r in enumerate(img):
        for j, c in enumerate(img[i]):
            if on_boundary(i, j, filter_size, img_size):
                img_conv[i][j] = apply_filter_to_location(img, i, j, conv_filter, filter_size)
            else:
                img_conv[i][j] = apply_filter_to_boundary(img, i, j, conv_filter, filter_size,
                                                          img_size,"circular")
    return img_conv



def apply_conv_reflect(img, img_size, filter):
    """
    Apply conv to image with mirror boundary conditions
    :param img_size: size of image
    :param img: input image
    :return: image after applying convolution operation
    """
    img_conv = copy.deepcopy(img)
    # img_conv.fill(0)
    conv_filter = filter
    filter_size = len(filter)
    for i, r in enumerate(img):
        for j, c in enumerate(img[i]):
            if on_boundary(i, j, filter_size, img_size):
                img_conv[i][j] = apply_filter_to_location(img, i, j, conv_filter, filter_size)
            else:
                img_conv[i][j] = apply_filter_to_boundary(img, i, j, conv_filter, filter_size,
                                                          img_size,"reflect")
    return img_conv



def apply_conv(img, img_size, filter):
    """
    Apply conv to image with boundary set to 0
    :param img_size: size of image
    :param img: input image
    :return: image after applying convolution operation
    """
    img_conv = copy.deepcopy(img)
    conv_filter = filter
    filter_size = len(filter)
    for i, r in enumerate(img):
        for j, c in enumerate(img[i]):
            if on_boundary(i, j, filter_size, img_size):
                img_conv[i][j] = apply_filter_to_location(img, i, j, conv_filter, filter_size)
    return img_conv


def filter_selection(n):
    res=[]
    #################
    # 3 layer filter
    #################
    if n==3:
        conv_filter = [[1, 0, -1],
                       [1, 0, 1],
                       [1, 0, 1]]
        filter_size = 3
        res = conv_filter

    #################
    # 5 layer filter
    #################
    if n==5:
        conv_filter = [[1.5, 0, -1, 0, 1],
                       [1, 0, -1.5, 0, 1],
                       [1, 0, -1.5, 0, 1],
                       [1, 0, -1.5, 0, 1],
                       [1, 0, -1, 0, 1]]
        filter_size = 5
        res = conv_filter
    #################
    # 7 layer filter
    #################
    if n==7:
        conv_filter = [[1.5, 0, -1, 0, 1, 0, 1],
                       [1, 0, -1.5, 0, 1, 0, 1],
                       [1, 0, -1.5, 0, 1, 0, 1],
                       [1, 0, -1.5, 0, 1, 0, 1],
                       [1, 0, -1.5, 0, 1, 0, 1],
                       [1, 0, -1.5, 0, 1, 0, 1],
                       [1, 0, -1, 0, 1, 0, 1]]
        filter_size = 7
        res = conv_filter

    return res


def run():
    # img = np.ndarray((100, 100))
    # img.fill(100)
    img = cv2.imread("lines.png", cv2.COLOR_RGB2GRAY)
    img = img[0:10, 0:10, 0]
    img_size = img.shape
    img_conv1 = apply_conv(img, img_size, filter_selection(3))
    img_conv2 = apply_conv_reflect(img, img_size, filter_selection(3))
    img_conv3 = apply_conv_circular(img, img_size, filter_selection(3))
    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(1, 4, 1)
    plt.imshow(img, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("orignal")

    fig.add_subplot(1, 4, 2)
    plt.imshow(img_conv1, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("conv_wo_b")

    fig.add_subplot(1, 4, 3)
    plt.imshow(img_conv2, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("conv_w_reflect")

    fig.add_subplot(1, 4, 4)
    plt.imshow(img_conv3, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("conv_w_circular")

    plt.show()


# helpful is this script is used as module or run directly
if __name__ == "__main__":
    run()
