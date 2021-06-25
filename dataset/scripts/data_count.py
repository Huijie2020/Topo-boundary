import json
import random

def data_count():
    #with open('./data_split.json','r') as jf:
    with open('/mnt/git/Topo-boundary/segmentation_based_baselines/naive_baseline/dataset/data_split.json', 'r') as jf:
    # with open('/mnt/git/Topo-boundary/segmentation_based_baselines/naive_baseline/dataset/data_split.json','r') as jf:
        data = json.load(jf)
    train_len = len(data['train'])
    valid_len = len(data['valid'])
    test_len = len(data['test'])
    pretrain_len = len(data['pretrain'])

    print("train_len", train_len)
    print("valid_len", valid_len)
    print("test_len", test_len)
    print("pretrain_len", pretrain_len)

data_count()
