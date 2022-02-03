# Feb 2, Aditya Gupta
# difference in fully convolution vs fully connected

import torch
import numpy as np
import torch.nn as nn

EPS: float = 0.00001

torch.manual_seed(2)


def flatten_scratch(input1, weights):
    """
    impplimentation of fully connected layer
    :param input: using same input to torch fully connected layer
    :param weights: using same weights as torch layer
    :return: output
    """
    result = []
    for i in range(weights.weight.shape[0]):
        temp = weights.weight[i] * input1[0]
        result.append(sum(temp) + weights.bias[i])
    return result


def softmax_scratch(input1):
    """
    Returns softmax operation for input vector
    :param input1: input vector of values
    :return:returs softmax value of input vector
    """
    result = []
    # calculate denominator
    denm = EPS
    for i in input1:
        denm+=torch.exp(i)
    for i in input1:
        result.append(torch.exp(i)/denm)
    return result


def operations(input):
    ip = torch.from_numpy(input)
    ip = ip[None, :]
    fully_connected = True
    fully_convoluted = True
    ###################
    # fully connected #
    ###################
    if fully_connected:
        # for 2x2 input
        # 0 : 0.16149699687957764 | 1 : 0.8135430216789246 | 2 : 0.024959996342658997
        m = nn.Sequential(nn.Flatten())
        op1 = m(ip)
        # it will have 12*3 weights + 3 bias
        n = nn.Linear(12, 3)
        try:
            op2 = n(op1.float())
            op2_manual = flatten_scratch(op1.float(), n)
            p = nn.Softmax(dim=1)
            op3 = p(op2)
            op3_manual = softmax_scratch(op3)
            sum = 0
            for i in range(0, 3):
                sum += float(op3[0][i])
                print(i, float(op3[0][i]))
            print("sum : ", sum)
        except:
            print("Incompatible input for Fully Connected")
    print("=============================================")
    ####################
    # fully convoluted #
    ####################
    if fully_convoluted:
        # for 2x2 input
        # 0 : 0.6238686442375183 | 1 : 0.36447882652282715 | 2 : 0.011652548797428608
        ma = nn.Conv2d(3, 3, (2, 2))
        op2a = ma(ip.float())
        try:
            pa = nn.Softmax(dim=1)
            op3a = pa(op2a)
            suma = 0
            for i in range(0, 3):
                suma += float(op3a[0][i])
                print(i, float(op3a[0][i]))
            print("sum : ", suma)
        except:
            print("input size greater than 2x2, current output", op2a.shape)


def run():
    # NCHW : batch, channel, height, width
    input = np.asarray([[[1, 2], [3, 6]],
                        [[5, 6], [7, 6]],
                        [[9, 10], [11, 6]]]).astype(float)
    print("2x2 Input")
    operations(input)
    input = np.asarray([[[1,2,6],[3,4,6],[4,5,7]],
                        [[5,6,6],[7,8,6],[4,5,7]],
                        [[9,10,6],[11,12,6],[4,5,7]]]).astype(float)
    print("3x3 Input")
    operations(input)

if __name__ == "__main__":
    run()