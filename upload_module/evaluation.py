__author__ = 'liuzhen'

# evaluate the performance, recall @ topN
import pandas
import const_params

topN = const_params.TOPN

def evaluation(im_sim_list, valid_img):
    __label__ = pandas.read_csv(const_params.LABEL_PATH, sep=' ')

    #im_sim_list = []
    im_list = []
    # get the query image
    for i_q in im_sim_list:
        im_list.append(i_q[0])

    queryNum = __label__.shape[0]

    acc = 0
    Total = 0

    for i_q in range(queryNum):
        queryID = __label__.ix[i_q, 'query']

        if queryID in im_list:
            idx = im_list.index(queryID)
        else:
            continue

        Total = Total + 1
        rank_list = im_sim_list[idx]

        lab = __label__.ix[i_q, 'label']
        truth = __label__.ix[i_q, 'data']

        if lab == 1:
            if truth in rank_list[1:1+topN]:
                acc = acc + 1
        else:
            if truth not in rank_list[1:1+topN]:
                acc = acc + 1

    print('total valid img number: {0}, acc img number: {1}'.format(Total, acc))