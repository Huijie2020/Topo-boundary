import json
import random

def split_dataset():
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/5_data_split/data_split_5data_100val_431test.json','r') as jf:
        data = json.load(jf)
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/5_data_split/data_split_1917test(0.05data).json','r') as jf:
        data_unsup = json.load(jf)
    train_len = round(len(data['train']))
    train_unsup_len = len(data_unsup['test'])
    valid_len = len(data['valid'])
    test_len = len(data['test'])
    pretrain_len = round(len(data['pretrain']))

    train_sup_img = data['train']
    train_unsup_img = data_unsup['test']
    valid_img = data['valid']
    test_img = data['test']
    pretrain_img = data['pretrain']

    #random.shuffle(train_img)
    #random.shuffle(pretrain_img)

   # with open('./data_split_previous.json','w') as jf:
   #     json.dump(data,jf)
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/5_data_split/data_split_5data_1917unsup_100val_431test.json','w') as jf:
    # with open('./scripts/data_split_100val.json','w') as jf:
        json.dump({'train_sup':train_sup_img,
                    'train_unsup': train_unsup_img,
                    # 'valid':valid_img[:100],
                    # 'test':test_img[100:test_len],
                   'valid': valid_img[:valid_len],
                   'test': test_img[:test_len],
                   'pretrain':[]
                    },jf)
        # json.dump({'train': [],
        #            'valid':[],
        #            'test': test_img[:200],
        #            'pretrain': []
        #            }, jf)
    # random.shuffle(valid_img)
    # with open('./scripts/data_split_w_val.json','w') as jf:
    #     json.dump({'train':train_img[:train_len],
    #                 'valid':valid_img[:valid_len_w],
    #                 'test':test_img[:test_len],
    #                 'pretrain':pretrain_img[:pretrain_len]
    #                 },jf)
    # print("train_sup_len", train_len)
    # print("train_unsup_len",train_unsup_len)
    # print('"valid_len', valid_len)
    # print("test_len", test_len)
    # print("pretrain_len", pretrain_len)

# split_dataset()

def split_data_deepglobe_full():
    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/fulll_semi_split/data_split_deepglobe_4496sup_4496unsup_200val_1530test.json','r') as jf:
        data = json.load(jf)
    jf.close()

    train_sup = data['train_sup'].copy()
    train_unsup = data['train_unsup'].copy()
    valid = data['valid'].copy()
    test = data['test'].copy()

    train_sup_new = []
    for i in range(2):
        random.shuffle(train_sup)
        for i in range(len(train_sup)):
            train_sup_new.append(train_sup[i])
    print('length of train_sup_new data {}'.format(len(train_sup_new)))
    print('length of train_sup data {}'.format(len(train_sup)))
    print('length of train_unsup data {}'.format(len(train_unsup)))
    print('length of valid data {}'.format(len(valid)))
    print('length of test data {}'.format(len(test)))

    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/fulll_semi_split/data_split_deepglobe_8992_2x4496sup_4496unsup_200val_1530test.json','w') as jf:
        json.dump({'train_sup': train_sup_new,
                   'train_unsup': train_unsup,
                   'valid': valid,
                   'test': test,
                   'pretrain': []
                    },jf)
    jf.close()

    with open('/mnt/git/Topo-boundary/conn_experiment/deepglobe_experiment/dataset_split/fulll_semi_split/data_split_deepglobe_4496test.json','w') as jf:
        json.dump({'train_sup': [],
                   'train_unsup': [],
                   'valid': [],
                   'test': train_sup,
                   'pretrain': []
                    },jf)
    jf.close()
# split_data_deepglobe_full()