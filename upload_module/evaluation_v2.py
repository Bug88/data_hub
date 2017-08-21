__author__ = 'liuzhen'

# evaluation version 2 for organized database, evaluate the recall @ topN

import const_params
import pickle

topN = const_params.TOPN

def evaluation(im_sim_list, valid_img):

    # load organized database images
    fp = open(const_params.DATABASE_PATH, 'r')
    all_imgs, query_imgs, __label__ = pickle.load(fp)
    fp.close()

    # get the query image
    im_list = []
    for i_q in im_sim_list:
        im_list.append(i_q[0])
    queryNum = len(im_list)

    acc = 0
    Total = 0

    # evaluate the search result, compute the average recall
    for i_q in range(queryNum):
        queryID = im_list[i_q]

        query_img = valid_img[queryID]
        idx = query_imgs.index(query_img)
        gt = __label__[idx]

        if queryID in im_list:
            idx = im_list.index(queryID)
        else:
            continue

        Total = Total + 1
        rank_list = im_sim_list[idx]
        c = 0
        # compute the recall rate
        for id in rank_list[:const_params.TOPN]:
            cur_img = valid_img[id]

            if cur_img in gt:
                c = c + 1

        acc = acc + c / float(len(gt))

    return acc / queryNum

    print('total valid img number: {0}, acc img number: {1}'.format(Total, acc))