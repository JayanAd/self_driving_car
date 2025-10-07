import cv2
import random
import numpy as np

xs = []
ys = []

train_batch_pointer = 0
valid_batch_pointer = 0

with open("data/driving_dataset/data.txt") as f:
    for line in f:
        xs.append("data/driving_dataset/" + line.split()[0])
        ys.append(float(line.split()[1]) * 3.14159265 / 180)  
        # we can use dict too
        
num_images = len(xs)

c = list(zip(xs, ys))

xs, ys = zip(*c)

train_xs = xs[:int(num_images * 0.8)]
train_ys = ys[:int(num_images * 0.8)]

val_xs = xs[-int(num_images * 0.2):]
val_ys = ys[-int(num_images * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        x_out.append(cv2.resize(cv2.imread(train_xs[(train_batch_pointer+i) % num_train_images])[-150:], (200, 66)) / 255.0)
        y_out.append([train_ys[(train_batch_pointer+i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global valid_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        x_out.append(cv2.resize(cv2.imread(val_xs[(valid_batch_pointer+i) % num_val_images])[-150:], (200, 66)) / 255.0)
        y_out.append([val_ys[(valid_batch_pointer+i) % num_val_images]])
    valid_batch_pointer += batch_size
    return x_out, y_out