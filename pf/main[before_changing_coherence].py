import os
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import image

matplotlib.use('tkagg')

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

    ### Random Initialization
    # initialise samples equally across image
    x_s = random.sample(range(10, 590), N)
    y_s = random.sample(range(10, 430), N)

    for i in range(0, N):
        s = State(x_s[i], y_s[i])
        s.udpdateWeight(1.0 / N)
        sampleList.append(s)

    ### Uniform initilasition
    # TODO: too many particles, leaving for now



def kld(p, q):
    # find kl divergence between two histograms signatures
    epsilon = 0.00001
    p += epsilon
    q += epsilon
    return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))


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
    liklihood = 1 - np.exp(-cur_div)
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
        if l < 0:
            l = 0
        sampleList[i].udpdateWeight(l)
        total_liklihood += l
        # print(i,'=' ,sampleList[i].x, sampleList[i].y, ' liklihood - ', l, ' divergence - ' ,d)
    # if total_liklihood == 0:
    #     print('No valid points, exiting!')
    #     generateSamples()
    # exit()
    try:
        for i in range(0, len(sampleList)):
            sampleList[i].udpdateWeight(sampleList[i].w / total_liklihood)
            new_liklihood += sampleList[i].w
    except:
        print('No valid points, restart. Current total liklihood : ', total_liklihood)
        sampleList.clear()
        generateSamples()
    # print('Total Liklihood : ', total_liklihood, ", New Liklihood : " , new_liklihood)
    # exit()


def predict():
    # predcit current location based on weights
    mean_x = 0
    mean_y = 0
    for k, i in enumerate(sampleList):
        mean_x += i.x * i.w
        mean_y += i.y * i.w
        current_weight_list[k] = i.w
        # print(k,' = ',i.x,', ', i.y, ' , ', i.w)
    # mean_x /= N
    # mean_y /= N
    print('Count = ', count, ', current prediction : ', mean_x, mean_y)
    return mean_x, mean_y


def resample():
    # resample points proportional to their weights
    print('but how ?')
    # current_weight_list_norm = [float(i[0]) / sum(current_weight_list) for i in current_weight_list]
    current_weight_list_norm = []
    try:
        for i in current_weight_list:
            current_weight_list_norm.append(i[0])
    except:
        print('lets see whats the problem in wieghts')
        for i in current_weight_list:
            current_weight_list_norm.append(i)
    current_weight_list_norm = [float(i) / sum(current_weight_list_norm) for i in current_weight_list_norm]
    current_index = []
    for i in range(N):
        current_index.append(i)
    new_index = np.random.choice(a=current_index,
                                 size=N,
                                 replace=True,
                                 p=current_weight_list_norm)
    current_pts = []

    for i, j in enumerate(sampleList):
        current_pts.append([sampleList[i].x, sampleList[i].y])

    for i, j in enumerate(sampleList):
        sampleList[i].x = int(current_pts[new_index[i]][0])
        sampleList[i].y = int(current_pts[new_index[i]][1])
    return


def plot_things():
    # plot particles and mean
    # put rectangle at mean
    matplotlib.use('tkagg')
    for i in sampleList:
        plt.plot(i.x, i.y, marker='v', color="white")
        plt.annotate(str(v.w), xy=(v.x, v.y), xytext=(-7, 7), textcoords='offset points')
    plt.plot(cur_x, cur_y, marker='o', color="red")
    plt.imshow(frame)
    # plt.show()
    # plt.savefig('./data/output/frame' + str(count) + '.jpg')
    if count % 3 == 0:
        plt.clf()


def applyMotionModel():
    # apply random movement motion model
    # print("Apply motion model")
    std_dev = 10
    for i in sampleList:
        cur_action = random.randint(-1, 1)  # 1 : add, 0:do nothing, -1:subtract
        new_x = int(i.x + cur_action * std_dev * random.random())
        new_y = int(i.y + cur_action * std_dev * random.random())
        i.x = min(width - 25, new_x)
        i.x = max(25, new_x)
        i.y = min(height - 25, new_y)
        i.y = max(25, new_y)


def createList():
    sampleListCopy = []
    for k,i in enumerate(sampleList):
        # print(k,':',i.x, i.y, i.w)
        sampleListCopy.append([i.x,i.y,i.w])
    return sampleListCopy


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
N = 10
cur_x = 0
cur_y = 0
# INITIALISE PARTICLES
generateSamples()
current_weight_list = []
for i in range(N):
    current_weight_list.append(0)
current_index_range = []
s1 = createList()

while count < 300:
    ret, frame = obj_video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # current_frame_plot = image.imread(frame)
    if ret:
        name = './data/frame' + str(count) + '.jpg'
        # print('Creating...' + name)
        # cv2.imwrite(name, frame)
        # apply motion to model
        applyMotionModel()
        s2 = createList()
        # generateSamples()
        # . Find matches in nearby areas us
        weightSamples(frame)
        s3 = createList()
        # . Update new location, predict
        cur_x, cur_y = predict()
        s4 = createList()
        # Resample
        # TODO
        plot_things()
        resample()
        s5 = createList()
        # plot_things()
        # . Re-sample based on how much weight each sample has
    else:
        print('leaving out current count : ', count)
        break
    count += 1
