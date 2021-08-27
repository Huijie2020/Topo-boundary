import json
import random

# # split according to unsup
# def split_dataset():
#     with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_99unsup_100val_431test.json','r') as jf:
#         data = json.load(jf)
#     with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1998test(0.01data).json','r') as jf:
#         data_unsup = json.load(jf)
#     train_sup_len = len(data['train_sup'])
#     train_unsup_len = len(data['train_unsup'])
#     valid_len = len(data['valid'])
#     test_len = len(data['test'])
#     pretrain_len = round(len(data['pretrain']))
#
#     train_sup_img = data['train_sup']
#
#     train_sup_last = random.sample(train_sup_img, 14)
#     train_sup_new = []
#     for i in range(299):
#         random.shuffle(train_sup_img)
#         # print(train_sup_img)
#         for i in range(len(train_sup_img)):
#             train_sup_new.append(train_sup_img[i])
#     for i in range(len(train_sup_last)):
#         train_sup_new.append(train_sup_last[i])
#     # random.shuffle(train_sup_new)
#     train_unsup_img = data['train_unsup']
#     valid_img = data['valid']
#     test_img = data['test']
#     pretrain_img = data['pretrain']
#
#     #random.shuffle(train_img)
#     #random.shuffle(pretrain_img)
#
#    # with open('./data_split_previous.json','w') as jf:
#    #     json.dump(data,jf)
#     with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_99unsup_100val_431test_5994sup.json','w') as jf:
#     # with open('./scripts/data_split_100val.json','w') as jf:
#         json.dump({'train_sup':train_sup_new[:len(train_sup_new)],
#                     'train_unsup': train_unsup_img[:train_unsup_len],
#                     # 'valid':valid_img[:100],
#                     # 'test':test_img[100:test_len],
#                    'valid': valid_img[:valid_len],
#                    'test': test_img[:test_len],
#                     'pretrain':pretrain_img[:pretrain_len]
#                     },jf)
#         # json.dump({'train': [],
#         #            'valid':[],
#         #            'test': test_img[:200],
#         #            'pretrain': []
#         #            }, jf)
#     # random.shuffle(valid_img)
#     # with open('./scripts/data_split_w_val.json','w') as jf:
#     #     json.dump({'train':train_img[:train_len],
#     #                 'valid':valid_img[:valid_len_w],
#     #                 'test':test_img[:test_len],
#     #                 'pretrain':pretrain_img[:pretrain_len]
#     #                 },jf)
#     print("train_sup_len", len(train_sup_new))
#     print("train_unsup_len",train_unsup_len)
#     print('"valid_len', valid_len)
#     print("test_len", test_len)
#     print("pretrain_len", pretrain_len)
#
# split_dataset()

# split according to sup
def split_dataset():
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_99unsup_100val_431test.json','r') as jf:
        data = json.load(jf)
    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/origianl_data_split/data_split_100data_100val_431test.json','r') as jf:
        data_2018 = json.load(jf)
    train_sup_len = len(data_2018['train'])
    # pos_unsup_len = len(data_addsup['pos_unsup'])
    # neg_unsup_len = len(data_addsup['neg_unsup'])
    train_unsup_len = len(data['train_unsup'])
    valid_len = len(data['valid'])
    test_len = len(data['test'])
    pretrain_len = round(len(data['pretrain']))

    train_sup_img = data_2018['train']
    # pos_unsup_img =data_addsup['pos_unsup']
    # neg_unsup_img = data_addsup['neg_unsup']
    # train_addsup_img = data_addsup['overlap_id']
    train_unsup_img = data['train_unsup']
    valid_img = data['valid']
    test_img = data['test']
    pretrain_img = data['pretrain']

    # train_totalsup_img = []
    # for i in range(train_sup_len):
    #     train_totalsup_img.append(train_sup_img[i])
    # for i in range(train_addsup_len):
    #     train_totalsup_img.append(train_addsup_img[i])
    #
    # print('train_totalsup_img length:', len(train_totalsup_img))

    train_sup_last = random.sample(train_sup_img, 1958)

    train_sup_new = []
    for i in range(2):
        random.shuffle(train_sup_img)
        for i in range(len(train_sup_img)):
            train_sup_new.append(train_sup_img[i])
    for i in range(len(train_sup_last)):
        train_sup_new.append(train_sup_last[i])

    with open('/mnt/git/Topo-boundary/conn_experiment/dataset_split/1_data_split/data_split_1data_pseudo1998_sup5994_unsup1998.json','w') as jf:
    # with open('./scripts/data_split_100val.json','w') as jf:
        json.dump({'train_sup':train_sup_new[:len(train_sup_new)],
                   'train_unsup': train_unsup_img[:len(train_unsup_img)],
                   # 'pos_unsup': pos_unsup_img[:len(pos_unsup_img)],
                   # 'neg_unsup': neg_unsup_img[:len(neg_unsup_img)],
                   'valid': valid_img[:valid_len],
                   'test': test_img[:test_len],
                   'pretrain':pretrain_img[:pretrain_len]
                    },jf)

    print("train_sup_len", len(train_sup_new))
    print("train_unsup",len(train_unsup_img))
    # print("pos_unsup_len",len(pos_unsup_img))
    # print("neg_unsup_len",len(neg_unsup_img))
    print('"valid_len', valid_len)
    print("test_len", test_len)
    print("pretrain_len", pretrain_len)

split_dataset()
