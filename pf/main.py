# 1) initial seed affected the results
# 2) in patch, (x,y) were interhchanged, that were affecting results
# 3) instead of taking particle as top-left point for
#    point in image, taking particle as middle point
# 4) trying different dynamical model value
# ===========================================================
# ===========================================================
# ===========================================================
import os
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import random
from math import log2

random.seed(10)

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
    ### Random Initialization
    # initialise samples equally across image
    x_s = random.sample(range(10, 590), N)
    y_s = random.sample(range(10, 430), N)

    for i in range(0, N):
        s = State(x_s[i], y_s[i])
        s.udpdateWeight(1.0 / N)
        sampleList.append(s)


def kld(p, q):
    # find kl divergence between two histograms signatures
    epsilon = 0.00001
    p = np.array(p)
    p += epsilon
    q = np.array(q)
    q += epsilon
    return sum(p[i] * log2(p[i] / q[i]) for i in range(len(p)))


def kld_example(P, Q):
    """ Calculates the Kullback-Lieber divergence
    according to the discrete definition:
    sum [P(i)*log[P(i)/Q(i)]]
    where P(i) and Q(i) are discrete probability
    distributions. In this case the one """

    """ Epsilon is used here to avoid conditional code for
    checking that neither P or Q is equal to 0. """
    epsilon = 0.00001

    # To avoid changing the color model, a copy is made
    temp_P = P+epsilon
    temp_Q = Q+epsilon

    divergence = np.sum(temp_P*np.log(temp_P/temp_Q))
    return divergence


def findHist(img):
    img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    epslion = 0.000001
    # TODO : check, for edge-cases, hist output shouldn't be zero (?)
    img_hist /= sum(img_hist + epslion)
    return img_hist


def findHistRGB(img, s):
    # plt.figure(0)
    matplotlib.use('tkagg')
    color = ('b', 'g', 'r')
    mean_hist = []
    plt.clf()
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        mean_hist.append(histr)
        # plt.plot(histr, color=col)
        # plt.xlim([0, 256])
    mean_hist_mean = []
    for i in range(len(mean_hist[0])):
        cur_avg = (mean_hist[0][i] + mean_hist[1][i] + mean_hist[2][i])/3.0
        mean_hist_mean.append(cur_avg)
    # plt.show()
    # plt.savefig('data/plots/'+str(s)+'.png')
    return mean_hist_mean


def findHistRGBexample(img, s):
    # plt.figure(0)
    matplotlib.use('tkagg')
    color = ('b', 'g', 'r')
    mean_hist = []
    plt.clf()
    num_bins = 32
    mask = None
    b_m = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    g_m = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
    r_m = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()
    color_patch = np.concatenate((b_m, g_m, r_m))

    # Normalize histogram values for the KL divergence computation
    color_patch = color_patch / np.sum(color_patch)
    return color_patch


def findCoherence(s, cur_frame):
    # for current location 's' find coherence
    # using template
    cur_w_s = max(5, sampleList[s].x - int(temp_w/2))
    cur_h_s = max(5, sampleList[s].y - int(temp_h/2))
    cur_w_e = min(width, sampleList[s].x + int(temp_w/2))
    cur_h_e = min(height, sampleList[s].y + int(temp_h/2))
    cur_hist = findHistRGBexample(cur_frame[cur_h_s:cur_h_e,
                                  cur_w_s:cur_w_e],
                                  s)
    cur_div = kld_example(temp_hist,
                          cur_hist)
    l = 1
    liklihood = np.exp(-cur_div*l)
    return liklihood, cur_div


def weightSamples(cur_frame):
    # weight each sample based on coherence using features
    total_liklihood = 0
    new_liklihood = 0
    for i in range(0, len(sampleList)):
        l, d = findCoherence(i, cur_frame)
        if l < 0:
            l = 0
        sampleList[i].udpdateWeight(l)
        total_liklihood += l
        # print('sample : ', i, sampleList[i].x, sampleList[i].y, sampleList[i].w)
    try:
        for i in range(0, len(sampleList)):
            sampleList[i].udpdateWeight(sampleList[i].w / total_liklihood)
            new_liklihood += sampleList[i].w
    except:
        print('No valid points, restart. Current total liklihood : ', total_liklihood)
        sampleList.clear()
        generateSamples()


def predict():
    # predcit current location based on weights
    mean_x = 0
    mean_y = 0
    for k, i in enumerate(sampleList):
        mean_x += i.x * i.w
        mean_y += i.y * i.w
        current_weight_list[k] = i.w
    return mean_x, mean_y


def resample():
    # resample points proportional to their weights
    current_weight_list_norm = []
    try:
        for i in current_weight_list:
            current_weight_list_norm.append(i[0])
    except:
        # print('lets see whats the problem in wieghts')
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
    plt.figure(1)
    plt.plot(cur_x, cur_y, marker='o', color="red")
    plt.imshow(frame)
    # plt.show()
    plt.savefig('./data/output_final_template_size/frame' + str(count) + '.png')
    if count % 3 == 0:
        plt.clf()


def applyMotionModel():
    # apply random movement motion model
    std_dev = 50
    for i in sampleList:
        cur_action = random.randint(-1, 1)  # 1 : add, 0:do nothing, -1:subtract
        new_x = int(i.x + cur_action * std_dev * random.random())
        new_y = int(i.y + cur_action * std_dev * random.random())
        i.x = min(width - 25, new_x)
        i.x = max(25, new_x)
        i.y = min(height - 25, new_y)
        i.y = max(25, new_y)


def createList(s=' - '):
    sampleListCopy = []
    for k,i in enumerate(sampleList):
        # print(s, k,':',i.x, i.y, i.w)
        sampleListCopy.append([i.x,i.y,i.w])
    return sampleListCopy


# . Template and Image to be tracked
# obj_template = cv.imread('cloth_template.jpg', cv2.IMREAD_GRAYSCALE)
# obj_template = cv.imread('cloth_template.jpg')
# obj_video = cv2.VideoCapture('cloth_video.mp4')

obj_template = cv.imread('cup_silver.jpg')
resize_template = True
if resize_template:
    scale_percent = 60  # percent of original size
    width = int(obj_template.shape[1] * scale_percent / 100)
    height = int(obj_template.shape[0] * scale_percent / 100)
    dim = (width, height)
    obj_template = cv2.resize(obj_template, dim, interpolation=cv2.INTER_AREA)

obj_video = cv2.VideoCapture('cup_silver.mp4')

width = 640
height = 480
template_width, template_height, channel = obj_template.shape
count = 0
sampleList = []
temp_hist = findHistRGBexample(obj_template,'t')  # Template Histogram
temp_h, temp_w, temp_d = obj_template.shape
N = 100  # Number of Particles
cur_x = 0
cur_y = 0
# INITIALISE PARTICLES
generateSamples()
current_weight_list = []
for i in range(N):
    current_weight_list.append(0)
current_index_range = []
s1 = createList()

obj_template_hsv = cv.cvtColor(obj_template, cv.COLOR_BGR2HSV)
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists
# Use the 0-th and 1-st channels
channels = [0, 1]
hist_base = cv.calcHist([obj_template_hsv], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

while True:
    if count%20==0:
        print("Count : ", count)
    ret, frame = obj_video.read()
    if ret:
        name = './data/frame' + str(count) + '.jpg'
        # apply motion to model
        applyMotionModel()
        # . Find matches in nearby areas us
        weightSamples(frame)
        # . Update new location, predict
        cur_x, cur_y = predict()
        # Resample
        # TODO
        plot_things()
        resample()
        plot_things()
    else:
        print('leaving out current count : ', count)
        break
    count += 1
