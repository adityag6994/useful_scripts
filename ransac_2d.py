# ransac fit for 2d line fit among points
# Advanatages of RanSAC
# - can work even in high degree of outliers
# - better than least square method (when we have outliers) since, LSM uses all the points to find the model
#  Drawbacks of ransac
# - does not garuntee succesful result, not good for high reliability scenarious
# - wont work if we dont have access to Random Number Generator
# - parameters have to be chosen application and situation wise
# - perform badly when number of inliers are less than 50%
# - can estimate only one model for dataset



# Create set of points
# for k interations:goo
#   Select 2 points at random which create line equation
#   Then calculate number of inlers which satisfy that as line
#   Then if number of inliers are greater than threshold, accept it
#   Not with new inliers re-calculate line (x - no)
#   With this line, calcilate inliners and error value
#   Save this line and min-error
# return line with minimum error

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

random.seed(1)
np.random.seed(1)

N = 100
n_outliers = 10
min_inliers = 7
min_distace = 0.4
min_error = 1000
best_line = []
max_outlier = 0
xPt = np.random.normal(4, 0.5, size=(N))
yPt = np.random.normal(4, 0.5, size=(N))
# xPt = random.sample(range(0, N), N)
# yPt = random.sample(range(0, N), N)

# k : probability of success
k = 0.99
# w : fraction of outliers
outlier_fraction = 0.1
num_trails = int(np.log(1-k)/np.log(1-np.power(outlier_fraction,2)))
# n : size of support
# 2D line  : 2 points
# 3D line  : 2 points
# 3D plane : 3 points

def dist(coefficients, i, data):
    dist = abs(-coefficients[0]*data[i][0] + data[i][1] - coefficients[1])/np.linalg.norm([-coefficients[0],1])
    return dist


X, y, coef = datasets.make_regression(
    n_samples=N,
    n_features=1,
    n_informative=1,
    noise=10,
    coef=True,
    random_state=0,
)


X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers,1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=(n_outliers))
# fig, ax = plt.subplots(figsize=(12, 12))

plt.scatter(X, y)

# data = np.asarray([xPt, yPt])
data = np.asarray([np.transpose(X)[0], y])
data = np.transpose(data)

for i in range(0, num_trails):
    plt.scatter(X, y)
    m, n = random.sample(range(0, N), 2)
    p1 = data[m]
    p2 = data[n]
    coefficients = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)
    plt.plot(data[m][0], data[m][1], marker="X", markersize=12, markeredgecolor="green")
    plt.plot(data[n][0], data[n][1], marker="X", markersize=12, markeredgecolor="green")
    x_values = [data[m][0], data[n][0]]
    y_values = [data[m][1], data[n][1]]
    current_inliers = 0
    inliers = []
    for pt in range(0, N):
        if pt != m and pt != n and dist(coefficients, pt, data) < min_distace:
            current_inliers += 1
            inliers.append((pt))
            plt.plot(data[pt][0], data[pt][1], marker="o", markersize=4, markeredgecolor="red")
    if current_inliers > min_inliers:
        error = 0
        for pt in inliers:
            error += dist(coefficients, pt, data)
        print(current_inliers, '-', error)
        if current_inliers > max_outlier:
            min_error = error
            best_line = [m, n]
            best_coeff = coefficients
            max_outlier = current_inliers
    else:
        print(current_inliers, '-', 0)
    plt.scatter(xPt, yPt)

    coefficients = np.polyfit(x_values, y_values, 1)
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(2, 5)
    y_axis = polynomial(x_axis)
    plt.plot(x_axis, y_axis,  linestyle="--")

print('Min_error : ', min_error, " | best_line : ", best_line, " | best_coef : ", best_coeff, " | inliers : ", max_outlier)
print('Done!')
