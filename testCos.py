import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import argparse
import torch.utils.data.sampler
import os
import glob
import random
import time
import torch.nn.functional as F
import feature_loader as feat_loader
from numpy.linalg import norm
from sklearn.preprocessing import normalize

def parse_feature(x,n_support):
    x = Variable(x.cuda())
    z_all = x
    z_support = z_all[:, :n_support]
    z_query = z_all[:, n_support:]

    return z_support, z_query
def cos_sim(features1,features2):
    norm1 = torch.norm(features1, dim=-1).reshape(features1.shape[0], 1)
    norm2 = torch.norm(features2, dim=-1).reshape(1, features2.shape[0])
    end_norm = torch.mm(norm1, norm2)
    cos = torch.mm(features1, features2.T) / end_norm
    return cos
def dis(features1, features2):
    return F.pairwise_distance(features1.unsqueeze(0), features2.unsqueeze(0), p=2)
def feature_evaluation(cl_data_file, n_way=5, n_support=5, n_query=15, adaptation=False):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))
    # If pooling, z_all can be evaluated by stages
    # z_all = torch.from_numpy(np.array(z_all))[:,:,:384] first stage / [:,:,384:768] second stage / [:,:,768:] third stage
    z_support, z_query = parse_feature(z_all, n_support)
    z_support = z_support.contiguous().view(n_way * n_support, -1)
    z_query = z_query.contiguous().view(n_way * n_query, -1)
    cos_score = cos_sim(z_support, z_query)
    pred = cos_score.cpu().numpy().argmax(axis=0)
    y = np.repeat(range(n_way), n_query)
    acc = (np.mean(pred//n_support == y) * 100)
    return acc

def testCos(args,server,epoch,pretrained_weights,file=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    n_query = 600 - args.num_shots
    acc_all1, acc_all2, acc_all3 = [], [], []
    few_shot_params = dict(n_way=args.num_ways, n_support=args.num_shots)
    # for each iteration
    iter_num = 10000
    print(iter_num)
    file_path = os.path.join(pretrained_weights,'{}_224_{}_{}.hdf5'.format(args.partition,epoch, args.checkpoint_key))
    if file is not None:
        file_path = file

    print('testfile:',file_path)
    cl_data_file = feat_loader.init_loader(file_path)
    acc_all1 = []
    print("evaluating over %d examples" % (n_query))
    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, n_query=n_query, adaptation=False,
                                 **few_shot_params)

        acc_all1.append(acc)
        if i % 1000 == 0:
            print("%d steps reached and the mean acc is %g " % (
                i, np.mean(np.array(acc_all1))))
    #         acc_all  = np.asarray(acc_all)
    acc_mean1 = np.mean(acc_all1)
    acc_std1 = np.std(acc_all1)
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num)))
    print(file_path)
    log_info = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    log_info += '\n%d Test Acc at %d= %4.2f%% +- %4.2f%% %s\n' % (
        epoch,iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num), args.checkpoint_key)
    # basz step method dataset partition
    with open(os.path.join(pretrained_weights,'{}_log_{}_{}.txt'.format(args.partition,server['dataset'],args.checkpoint_key)), 'a+') as f:
        f.write(log_info)



if __name__ == '__main__':
    # args.partition = 'test' if args.partition is None else args.partition
    # args.epoch = '30' if args.epoch is None else args.epoch
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--num_ways', default=5, type=int)
    parser.add_argument('--num_shots', default=5, type=int)
    parser.add_argument('--dataset', default='tiered', type=str)
    parser.add_argument('--seed', default=222, type=int)
    parser.add_argument('--partition', default='test', type=str)
    parser.add_argument('--pretrained_weights', default='/home/heyj/dino/checkpoint_tiered/checkpoint.pth', type=str,
                        help="Path to pretrained weights to evaluate.")
    args = parser.parse_args()
    pretrained_weights = '/home/heyj/dino/checkpoint_tiered/'
    testCos(args,66,1,pretrained_weights)
