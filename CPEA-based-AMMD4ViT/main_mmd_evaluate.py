import argparse
import os.path as osp

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cpea import CPEA
from mmd import AttentiveMMDPrompt
from models.backbones import BackBone
from dataloader.samplers import CategoriesSampler
from utils import pprint, ensure_path, Averager, count_acc, count_acc_mmd, compute_confidence_interval,triplet_loss
from tensorboardX import SummaryWriter

def seed_torch(seed=0): 
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--test_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lr_mul', type=float, default=1)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--model_type', type=str, default='small')
    parser.add_argument('--dataset', type=str, default='MiniImageNet')
    parser.add_argument('--init_weights', type=str, default='./initialization/miniimagenet/checkpoint1600.pth')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--exp', type=str, default='CPEA')
    parser.add_argument('--loss_gamma', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    pprint(vars(args))

    save_path = '-'.join([args.exp, args.dataset, args.model_type])
    args.save_path = osp.join('./results', save_path)
    ensure_path(args.save_path)

    if args.dataset == 'MiniImageNet':
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'tieredimagenet':
        from dataloader.tiered_imagenet import TieredImagenet as Dataset
    elif args.dataset == 'fc100':
        from dataloader.fc100 import FC100 as Dataset
    elif args.dataset == 'cifar_fs':
        from dataloader.cifarfs import CIFARFS as Dataset
    else:
        raise ValueError('Non-supported Dataset.')


    model = BackBone(args)
    dense_predict_network = AttentiveMMDPrompt(args)

    # Test Phase
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 2000, args.test_way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((2000,))


    test_dict=torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params']
    # test_dict=torch.load(osp.join(args.save_path, 'epoch-last' + '.pth'))['params']
    test_dict = {k.replace('module.', ''): v for k, v in test_dict.items()}
    model.load_state_dict(test_dict)
    model = model.cuda()
    model.eval()

    dense_predict_network.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '_dense_predict.pth'))['params'])
    # dense_predict_network.load_state_dict(torch.load(osp.join(args.save_path, 'epoch-last' + '_dense_predict.pth'))['params'])
    dense_predict_network = dense_predict_network.cuda()
    dense_predict_network.eval()

    ave_acc = Averager()
    label = torch.arange(args.test_way).repeat(args.query)

    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = args.test_way * args.shot
            data_shot, data_query = data[:k], data[k:]
            feat_shot, feat_query = model(data_shot, data_query)

            results = dense_predict_network(feat_query, feat_shot, args)  # Q x S
            # results = [torch.mean(idx, dim=0, keepdim=True) for idx in results]
            # results = torch.cat(results, dim=0)  # Q x S
            label = torch.arange(args.test_way).repeat(args.query).long().to('cuda')

            acc = count_acc_mmd(results.data, label)
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            print('batch {}: acc {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    m, pm = compute_confidence_interval(test_acc_record)
    print('Val Best Epoch {}, Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
