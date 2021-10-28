import json
import random

def split_dataset():
    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/1_data_split/data_split_deepglobe_45sup_4451unsup_200val_1530test.json','r') as jf:
        data = json.load(jf)
    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/1_data_split/data_split_deepglobe_45sup_4451unsup_200val_1530test.json','r') as jf:
        data_unsup = json.load(jf)
    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/1_data_split/contrastive_3560/iter1_3560/name_3560from4451.json','r') as jf: # for the first time self-training, use 'name_1600from1998.json'
        data_sup_add = json.load(jf)
    # with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/1_data_split/contrastive_3560/iter2_3560/name_1597from4451.json','r') as jf: # for the first time self-training, use 'name_1600from1998.json'
    #     data_sup_last = json.load(jf)


    data_20 = data["train_sup"]
    data_1600 = data_sup_add["name_3560from4451"]
    # train_data_sup_last = data_sup_last['name_1597from4451']
    train_sup_img = []
    for i in range(len(data_20)):
        train_sup_img.append(data_20[i])
    for i in range(len(data_1600)):
        train_sup_img.append(data_1600[i])

    train_unsup_img = data_unsup["train_unsup"]
    valid_img = data_unsup["valid"]
    test_img = data_unsup['test']

    train_sup_new = []
    for i in range(2):
        random.shuffle(train_sup_img)
        for i in range(len(train_sup_img)):
            train_sup_new.append(train_sup_img[i])
    # data_20 = data["train_sup"]
    # for i in range(len(data_20)):
    #     train_sup_new.append(data_20[i])
    # train_sup_last = random.sample(data_1600, 1745)
    # for i in range(len(train_sup_last)):
    #     train_sup_new.append(train_sup_last[i])
    # for i in range(len(train_data_sup_last)):
    #     train_sup_new.append(train_data_sup_last[i])


    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/1_data_split/contrastive_3560/iter1_3560/data_split_deepglobe_sup3560_sup7210_unsup4551_growing_image_contr_iter1.json','w') as jf:
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
