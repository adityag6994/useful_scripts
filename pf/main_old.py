import os
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import image
matplotlib.use( 'tkagg' )

import random
from math import log2
random.seed(1)

matplotlib.use('Agg')
print('||---------------------------------')
print("||- Particle Filter Template Matching")


class State:
    x = 0
    y = 0
    w = 0
    v_x = 0
    v_y = 0

    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def udpdateWeight(self, W):
        self.w = W

    def getLocation(self):
        return self.x, self.y


def generateSamples():
    # generate sample based on weights
    # if count > 0:
    # initialise samples equally across image
    x_s = random.sample(range(10, 600), N)
    y_s = random.sample(range(10, 450), N)

    for i in range(0, N):
        s = State(x_s[i], y_s[i])
        s.udpdateWeight(1.0/N)
        sampleList.append(s)


def kld(p, q):
    # find kl divergence between two histograms signatures
    epsilon = 0.00001
    p += epsilon
    q += epsilon
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def findCoherence(s, cur_frame):
    # for current location 's' find coherence
    # using template
    cur_w_s = sampleList[s].x
    cur_h_s = sampleList[s].y
    cur_w_e = sampleList[s].x + temp_w
    cur_h_e = sampleList[s].y + temp_h
    if cur_w_e > width:
        cur_w_e = width
    if cur_h_e > height:
        cur_h_e = height
    cur_hist = cv2.calcHist([cur_frame[cur_w_s:cur_w_e, cur_h_s:cur_h_e]], [0], None, [256], [0, 256])
    epslion = 0.000001
    # TODO : check, for edge-cases, hist output shouldn't be zero (?)
    cur_hist /= sum(cur_hist + epslion)
    cur_div = kld(cur_hist, temp_hist)
    liklihood = 1-np.exp(-cur_div)
    # print(cur_div, liklihood)
    # matplotlib.use('tkagg')
    # plt.plot(cur_hist,'r')
    # plt.plot(temp_hist)
    # plt.show()
    # plt.close()
    return liklihood, cur_div



def weightSamples(cur_frame):
    # weight each sample based on coherence using features
    total_liklihood = 0
    new_liklihood = 0
    for i in range(0, len(sampleList)):
        # print('sample : ', i)
        l, d = findCoherence(i, cur_frame)
        if l<0:
            l=0
        sampleList[i].udpdateWeight(l)
        total_liklihood += l
        # print(i,'=' ,sampleList[i].x, sampleList[i].y, ' liklihood - ', l, ' divergence - ' ,d)
    if total_liklihood == 0:
        print('No valid points, exiting!')
        exit()
    for i in range(0, len(sampleList)):
        sampleList[i].udpdateWeight(sampleList[i].w/total_liklihood)
        new_liklihood += sampleList[i].w
    # print('Total Liklihood : ', total_liklihood, ", New Liklihood : " , new_liklihood)
    # exit()


def predict():
    # predcit current location based on weights
    mean_x = 0
    mean_y = 0
    for k,i in enumerate(sampleList):
        mean_x += i.x*i.w
        mean_y += i.y*i.w
        # print(k,' = ',i.x,', ', i.y, ' , ', i.w)
    # mean_x /= N
    # mean_y /= N
    print('Count = ', count, ', current prediction : ', mean_x, mean_y)
    return mean_x, mean_y


def resample():
    # resample points proportional to their weights
    print('but how ?')
    # we want more sample if that entity has more weight


def plot_things():
    # plot particles and mean
    # put rectangle at mean
    matplotlib.use('tkagg')
    for i in sampleList:
        plt.plot(i.x, i.y, marker='v', color="white")
    plt.plot(cur_x, cur_y, marker='o', color="red")
    plt.imshow(frame)
    # plt.show()
    plt.savefig('./data/output/frame'+str(count)+'.jpg')


def applyMotionModel():
    # apply random movement motion model
    # print("Apply motion model")
    std_dev = 5
    for i in sampleList:
        cur_action = random.randint(-1, 0) #1 : add, 0:do nothing, -1:subtract
        new_x = int(i.x + cur_action * std_dev * random.random())
        new_y = int(i.y + cur_action * std_dev * random.random())
        i.x = min(width-25, new_x)
        i.y = min(height-25, new_y)


# . Template and Image to be tracked
obj_template = cv.imread('cloth_template.jpg', cv2.IMREAD_GRAYSCALE)
obj_video = cv2.VideoCapture('cloth_video.mp4')
width = 640
height = 480
template_width, template_height = obj_template.shape
count = 0
sampleList = []
# totalSamples = 10
temp_hist = cv2.calcHist([obj_template], [0], None, [256], [0, 256])
temp_hist /= sum(temp_hist)
temp_w, temp_h = obj_template.shape
N = 50
cur_x = 0
cur_y = 0
# INITIALISE PARTICLES
generateSamples()

while count < 300:
    ret, frame = obj_video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # current_frame_plot = image.imread(frame)
    if ret:
        name = './data/frame' + str(count) + '.jpg'
        # print('Creating...' + name)
        # cv2.imwrite(name, frame)
        #apply motion to model
        applyMotionModel()
        # generateSamples()
        # . Find matches in nearby areas us
        weightSamples(frame)
        # . Update new location, predict
        cur_x, cur_y = predict()
        # Resample
        # TODO
        resample()
        plot_things()
        # . Re-sample based on how much weight each sample has
    else:
        print('leaving out current count : ', count)
        break
    count += 1


# import cv2 as cv
# import numpy as np
# import copy
# from matplotlib import pyplot as plt
# img = cv.imread('messi5.jpg',0)
# img2 = img.copy()
# template = cv.imread('messi_face.jpg',0)
# w, h = template.shape[::-1]
# # All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF',
#            'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
#     # Apply template Matching
#     res = cv.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv.rectangle(img,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.savefig("as.png")
#
#     #--------------
#     # type('uint8')
#     # numpy.ndarray
