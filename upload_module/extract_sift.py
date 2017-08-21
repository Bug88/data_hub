__author__ = 'liuzhen'

# extract sift features in given sift dir, this file is a kind of tools
import os
from os import listdir
from os.path import isfile, join
import cv2
import const_params

data = const_params.DATA_TRAIN
onlyfiles = [f for f in listdir(data) if isfile(join(data, f))]
post_script = '.sift'

sift = cv2.xfeatures2d.SIFT_create()

for f in onlyfiles:
    title, post = os.path.splitext(f)

    if post != '.jpg':
        continue

    img = cv2.imread(join(data, f), 0)
    kp, des = sift.detectAndCompute(img, None)

    if des is None:
        continue

    save_file_p = join(data, f)
    siftfile = open(save_file_p, 'w')
    des.tofile(siftfile)
    siftfile.close()


