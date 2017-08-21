__author__ = 'liuzhen'

# prepare the database for testing developed algorithm
import const_params
import os
import pickle

gt = const_params.GROUND_TRUTH
dt = const_params.DATA_TRAIN

all_images = []

# collect all images to build image list
database_image_names = [filename.split('_')[0]
               for filename in os.listdir(const_params.DATA_TRAIN)
               if filename.endswith('.jpg')]

query_images = []
truth_response_images = []

labels = []

# process the ground truth file
f = open(gt, 'r')
for line in open(gt, 'r'):
    line = f.readline()
    img_titles = line.strip('\\\n')
    img_titles = img_titles.strip('}')
    img_titles = line.split(' ')

    cur_tt = []
    cur_tt.append(img_titles[0])
    query_images.append(img_titles[0])
    for tl in img_titles[2:]:
        truth_response_images.append(tl)
        cur_tt.append(tl)

    labels.append(cur_tt)
f.close()

# all images
all_images.extend(query_images)
all_images.extend(truth_response_images)
all_images.extend(database_image_names)

f = open('./data_base.pkl', 'w')
pickle.dump([all_images, query_images, labels], f)
f.close()