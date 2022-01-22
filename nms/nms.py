# jan 21, Aditya Gupta

import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

img = mpimg.imread('sample_2_object.png')
imgplot = plt.imshow(img)

fig, ax = plt.subplots()
ax.imshow(img)

eps = 0.00001
def iou(boxA, boxB):
    x_a = max(boxA[0], boxB[0])
    x_b = min(boxA[2], boxB[2])
    y_a = max(boxA[1], boxB[1])
    y_b = min(boxA[3], boxB[3])

    anb = max(0, x_b-x_a+1)*max(0, y_b-y_a+1)
    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    result = anb/float(area_B+area_A-anb + eps)

    return result

with open('predictions.list') as f:
    lines = f.readlines()

nms_th = 0.6

all_pred   = collections.defaultdict(list)
final_pred = collections.defaultdict(list)

for i in lines:
    all_pred[i.split()[0]].append([float(j) for j in i.split()[1:6]])

for i in all_pred.keys():
    print(i)
    for j in range(0, len(all_pred[i])):
        print('  ',j)
        x = all_pred[i][j][1]
        y = all_pred[i][j][2]
        w = all_pred[i][j][3]-all_pred[i][j][1]
        h = all_pred[i][j][4] - all_pred[i][j][2]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

for key in all_pred:
    all_pred[key].sort(reverse=True)
    print('Current Class : ', key)
    # for each class
    while len(all_pred[key]):
        # iterate until all boxes have been solved
        current_max_pred = all_pred[key][0]
        print('current_max_pred : ', current_max_pred)
        if len(all_pred[key])==1:
            final_pred[key].append(all_pred[key][0])
            all_pred[key].pop()
        else:
            to_remove = []
            for i in range(1, len(all_pred[key])):
                current_iou = iou(current_max_pred[1:], all_pred[key][i][1:])
                print(current_iou,current_max_pred[1:], all_pred[key][i][1:])
                if current_iou > nms_th:
                    to_remove.append(i)
            all_pred[key] = [i for j, i in enumerate(all_pred[key]) if j not in to_remove]
            final_pred[key].append(all_pred[key][0])
            all_pred[key].pop(0)

# print final predictions
for i in final_pred.keys():
    print(i)
    for j in range(0, len(final_pred[i])):
        print('  ',j)
        x = final_pred[i][j][1]
        y = final_pred[i][j][2]
        w = final_pred[i][j][3]-final_pred[i][j][1]
        h = final_pred[i][j][4] - final_pred[i][j][2]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
plt.show()

for i in final_pred.keys():
    print(i, final_pred[i])
print('Done!')