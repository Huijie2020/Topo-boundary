import json
import random
#
def data_count():
    #with open('./data_split.json','r') as jf:
    with open('/mnt/git/Topo-boundary/segmentation_based_baselines/naive_baseline/dataset/data_split.json', 'r') as jf:
    # with open('/mnt/git/Topo-boundary/segmentation_based_baselines/naive_baseline/dataset/data_split.json','r') as jf:
        data = json.load(jf)
    # train_len = len(data['train'])
    train_sup_len = len(data['train_sup'])
    train_unsup_len = len(data['train_unsup'])
    # pos_unsup_len = len(data['pos_unsup'])
    # neg_unsup_len = len(data['neg_unsup'])
    valid_len = len(data['valid'])
    test_len = len(data['test'])
    # pretrain_len = len(data['pretrain'])

    # print("train_len", train_len)
    print("train_sup_len", train_sup_len)
    print("train_unsup_len", train_unsup_len)
    # print("pos_unsup_len", pos_unsup_len)
    # print("neg_unsup_len", neg_unsup_len)
    print("valid_len", valid_len)
    print("test_len", test_len)
    # print("pretrain_len", pretrain_len)

data_count()


# def data_count():
#     with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/growing/iter1_1600image/name_1600from1998.json', 'r') as jf:
#         data = json.load(jf)
#     train_sup_len = len(data['name_1600from1998'])
#
#     print("train_sup_len", train_sup_len)
#
#
# data_count()