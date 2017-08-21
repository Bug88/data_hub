__author__ = 'liuzhen'
# compute the fisher vector for the dataset images
import const_params
import os
import sys
sys.path.append(const_params.YAEL_LIB)
import numpy as np
import ynumpy
import pickle
import cv2
from sklearn import cluster

sub_group_num = const_params.IMG_GROUP_NUM
image_names = [filename
               for filename in os.listdir(const_params.DATA_TRAIN)
               if filename.endswith('.jpg')]

sift = cv2.xfeatures2d.SIFT_create()

# load images
image_descs = []
image_kps = []
image_labels = []

junk_img = []
valid_img = []

for img_name in image_names:
    im = cv2.imread(os.path.join(const_params.DATA_TRAIN, img_name), 0)
    kp, desc = sift.detectAndCompute(im, None)

    points = np.array([kp[idx].pt for idx in range(len(kp))])

    if points.shape[0] <= const_params.IMG_GROUP_NUM:
        junk_img.append(img_name)
        continue

    km = cluster.KMeans(n_clusters=sub_group_num, random_state=0).fit(points)

    valid_img.append(img_name)

    image_descs.append(desc)
    image_kps.append(kp)
    image_labels.append(km.labels_)

fp = open(const_params.GMM_PATH, 'r')
mean, pca_transform, gmm, egvec = pickle.load(fp)
fp.close()

# fisher vector
image_fvs = []
valid_image = []
c = 0
for cur_img in range(len(image_names)):
    if image_names[cur_img] in junk_img:
        continue
    image_desc = image_descs[c]
    image_label = image_labels[c]
    c = c + 1
    cur_fvs = []
    for c_lb in range(const_params.IMG_GROUP_NUM):
        cur_desc = image_desc[image_label == c_lb]
        cur_desc = np.dot(cur_desc - mean, pca_transform)
        fv = ynumpy.fisher(gmm, cur_desc, include = 'mu')
        fv = np.dot(egvec.T, fv)
        cur_fvs.append(fv)

    image_fvs.append(cur_fvs)
    valid_image.append(image_names[cur_img])

fp = open('./images_fvs.txt', 'w')
pickle.dump([image_fvs, valid_image], fp)
fp.close()

print(len(image_fvs[0]))
print(len(image_fvs))