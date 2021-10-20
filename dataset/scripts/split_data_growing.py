import json
import random

def split_dataset():
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_99unsup_100val_431test.json','r') as jf:
        data = json.load(jf)
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_99unsup_100val_431test.json','r') as jf:
        data_unsup = json.load(jf)
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/growing_contr/500_step/iter3/name_1500from1998.json','r') as jf: # for the first time self-training, use 'name_1600from1998.json'
        data_sup_add = json.load(jf)
    # with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/growing/iter2_1600image/name_904from1998.json','r') as jf:
    #     data_sup_add_last = json.load(jf)

    data_20 = data["train_sup"]
    data_1600 = data_sup_add["name_1500from1998"]
    train_sup_img = []
    for i in range(len(data_20)):
        train_sup_img.append(data_20[i])
    for i in range(len(data_1600)):
        train_sup_img.append(data_1600[i])
    # data_904 = data_sup_add_last["name_904from1998"]

    train_unsup_img = data_unsup["train_unsup"]
    valid_img = data_unsup["valid"]
    test_img = data_unsup['test']
    # pretrain_img = data_unsup['pretrain']

    train_sup_new = []
    for i in range(3):
        random.shuffle(train_sup_img)
        for i in range(len(train_sup_img)):
            train_sup_new.append(train_sup_img[i])
    data_20 = data["train_sup"]
    for i in range(len(data_20)):
        train_sup_new.append(data_20[i])
    train_sup_last = random.sample(data_1600, 1414)
    for i in range(len(train_sup_last)):
        train_sup_new.append(train_sup_last[i])


    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/growing_contr/500_step/iter3/data_split_sup1500_5994_unsup1998_growing_image_contr_iter3.json','w') as jf:
    # with open('./scripts/data_split_100val.json','w') as jf:
        json.dump({'train_sup':train_sup_new,
                    'train_unsup': train_unsup_img,
                    # 'valid':valid_img[:100],
                    # 'test':test_img[100:test_len],
                   'valid': valid_img,
                   'test': test_img,
                   'pretrain':[]
                    },jf)


split_dataset()
