__author__ = 'liuzhen'

# the important part is the similarity computation
import numpy as np
import pickle
import evaluation

# compute the similarity between images
img_fvs = []

# load data
fp = open('./images_fvs.txt', 'r')
img_fvs, valid_img = pickle.load(fp)
fp.close()

fp= open('./data_base.pkl', 'r')
all_imgs, query_imgs, labels = pickle.load(fp)
fp.close()

# prepare the query imgs
query_idx = []
for im in query_imgs:
    idx = valid_img.index(im)
    query_idx.append(idx)

img_num = len(img_fvs)

im_rank_list = [] # the dist rank list for each image

for c_1 in query_idx: # range(img_num)
    im_a = img_fvs[c_1]
    rank_list = [] # the dist rank list for im_a
    for c_2 in range(img_num):
        im_b = img_fvs[c_2]
        im_dist = 0 # image distance
        fv_num = len(im_a)
        for d_1 in range(fv_num):
            fv_a = im_a[d_1]
            dist_list = []
            for d_2 in range(fv_num):
                fv_b = im_b[d_2]
                dist = np.sqrt(np.sum(np.square(fv_a - fv_b))) # fisher vector distance
                dist_list.append(dist) # find the most similar fisher vector
            im_dist = im_dist + min(dist_list)
        rank_list.append(im_dist)

    im_rank_list.append(rank_list)

# find the image similarity rank list
im_sim_list = []

for c_1 in range(img_num):
    sim_list = range(img_num)
    dist_list = im_rank_list[c_1]
    sim_list = sorted(sim_list, key=lambda k:dist_list[k])

    sim_name_list = []
    for q_1 in range(img_num):
        sim_name_list.append(int(valid_img[sim_list[q_1]].split('.')[0]))

    im_sim_list.append(sim_name_list)

evaluation.evaluation(im_sim_list, valid_img)

print('finished!')