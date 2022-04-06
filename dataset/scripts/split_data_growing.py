import json
import random

def split_dataset():
    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/5_data_split/data_split_deepglobe_225sup_4271unsup_200val_1530test.json','r') as jf:
        data = json.load(jf)
    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/5_data_split/contrstive_3417/iter2_3417/union_two_image_list_iter2.json','r') as jf: # for the first time self-training, use 'name_1600from1998.json'
        data_sup_add = json.load(jf)

    data_sup_original = data["train_sup"]
    data_sup_extra = data_sup_add["union_two_image_list"]
    train_sup_img = []
    for i in range(len(data_sup_original)):
        train_sup_img.append(data_sup_original[i])
    for i in range(len(data_sup_extra)):
        train_sup_img.append(data_sup_extra[i])

    train_unsup_img = data["train_unsup"]
    valid_img = data["valid"]
    test_img = data['test']

    train_sup_new = []
    for i in range(2):
        random.shuffle(train_sup_img)
        for i in range(len(train_sup_img)):
            train_sup_new.append(train_sup_img[i])
    data_sup_original = data["train_sup"]
    for i in range(len(data_sup_original)):
        train_sup_new.append(data_sup_original[i])
    # train_sup_last = random.sample(data_sup_extra, 1491)
    # for i in range(len(train_sup_last)):
    #     train_sup_new.append(train_sup_last[i])


    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/5_data_split/contrstive_3417/iter2_3417/data_split_deepglobe_0.05data_3712sup_225_3487_sup7649_unsup4271_growing_contr_iter2.json','w') as jf:
        json.dump({'train_sup':train_sup_new,
                    'train_unsup': train_unsup_img,
                   'valid': valid_img,
                   'test': test_img,
                   'pretrain':[]
                    },jf)

    print('\n train_sup length:', len(train_sup_new))
    print('\n train_unsup length:', len(train_unsup_img))
    print('\n valid length:', len(valid_img))
    print('\n test length:', len(test_img))


split_dataset()
