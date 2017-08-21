__author__ = 'liuzhen'
import const_params
import os
import sys
sys.path.append(const_params.YAEL_LIB)
import numpy as np
import ynumpy
import pickle

# scripts for the params training of fisher

image_names = [filename
               for filename in os.listdir(const_params.DATA_TRAIN)
               if filename.endswith('.sift')]
# load images
image_descs = []
for im in image_names:

    bFile = open(os.path.join(const_params.DATA_TRAIN, im), 'rb')
    siftFeats = np.fromfile(bFile, dtype=np.float32)
    bFile.close()
    desc = np.reshape(siftFeats, (-1, 128))

    image_descs.append(desc)

all_desc = np.vstack(image_descs)

k = const_params.CENT_NUM
num_sample = const_params.TRAIN_NUM

print('the number of centers are: {0}'.format(k))


# cov computing
sample_indices = np.random.choice(all_desc.shape[0], num_sample)
sample = all_desc[sample_indices]
mean = sample.mean(axis = 0)
sample = sample - mean
cov = np.dot(sample.T, sample)

# pca, 64-D
eigvals, eigvecs = np.linalg.eig(cov)
perm = eigvals.argsort()
pca_transform = eigvecs[:, perm[64:128]]
sample = np.dot(sample, pca_transform)

# gmm
gmm = ynumpy.gmm_learn(sample, k)

# fisher vector
image_fvs = []
for image_desc in image_descs:
   image_desc = np.dot(image_desc - mean, pca_transform)
   fv = ynumpy.fisher(gmm, image_desc, include = 'mu')
   image_fvs.append(fv)

image_fvs = np.vstack(image_fvs)

# gram method to perform pca

d, n = image_fvs.shape
mean_t = image_fvs.mean(axis=0)
X = image_fvs - mean_t

gramMatrix = np.dot(X, X.T)
egval, egvec = np.linalg.eig(gramMatrix)
egvec = egvec[:, :const_params.FISHER_DIM]
#X = np.dot(egvec.T, X)
egvec = np.dot(X.T, egvec)
X = np.dot(egvec.T, X.T)
X = X.T
# save the trained params, mean\pca_transform\gmm, fisher pca matrix
fp = open(const_params.GMM_PATH, 'w')
pickle.dump([mean, pca_transform, gmm, egvec], fp)
fp.close()
