import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

import random
from modules.fsl_query import make_fsl 
from dataloader import make_dataloader
from engines.utils import mean_confidence_interval, AverageMeter, set_seed
from modules.layers.sync_batchnorm import convert_model

class Evaluator(object):
    def __init__(self, cfg, checkpoint_dir, num_gpus):

        self.n_way                 = cfg.n_way # 5
        self.k_shot                = cfg.k_shot # 5
        self.test_query_per_class   = cfg.test.query_per_class_per_episode  # 15

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.eval_epoch = osp.basename(checkpoint_dir)
        self.prediction_folder = osp.join(
            "./predictions/", osp.basename(checkpoint_dir[:checkpoint_dir.rfind("/")])
        )
        if not osp.exists(self.prediction_folder):
            os.mkdir(self.prediction_folder)

        self.prediction_dir = osp.join(
            self.prediction_folder,
            "predictions.txt"
            # osp.basename(checkpoint_dir).replace(".pth", ".txt")
        )

        self.checkpoint_dir = checkpoint_dir

        self.fsl = make_fsl(cfg)
        
        state_dict = torch.load(checkpoint_dir, map_location='cpu')
        state_dict["fsl"] = remove_module(state_dict["fsl"])
        msg = self.fsl.load_state_dict(state_dict["fsl"], strict=False)
        print(msg)
        self.fsl.encoder = nn.DataParallel(self.fsl.encoder, list(range(num_gpus)))
        self.fsl.encoder = convert_model(self.fsl.encoder)
       
        self.fsl.encoder = self.fsl.encoder.cuda()
        self.fsl.query = self.fsl.query.cuda()
        self.fsl.eval()

        self.test_episode = cfg.test.episode
        self.total_testtimes = cfg.test.total_testtimes

        self.cfg = cfg
    def run(self):
        with torch.no_grad():
            acc = self._run()
        return acc
    def _run(self):
        with open(self.prediction_dir, 'w') as f_txt:
            total_accuracies = 0.0
            total_h = 0.0
            print("evaluation epoch: ", self.eval_epoch, file=f_txt)
            set_seed(1)
            for epoch in range(self.total_testtimes):
                test_dataloader = make_dataloader(self.cfg, phase="test", batch_size=self.cfg.test.batch_size)
                tqdm_gen = tqdm(test_dataloader, ncols=80)
                accuracies = []
                acc = AverageMeter()
                for episode, (support_x, support_y, query_x, query_y) in enumerate(tqdm_gen):
                    support_x            = support_x.cuda()
                    support_y            = support_y.cuda()
                    query_x              = query_x.cuda()
                    query_y              = query_y.cuda()

                    rewards = self.fsl(support_x, support_y, query_x, query_y)
                    if isinstance(rewards, tuple):
                        rewards = rewards[0]

                    total_rewards = np.sum(rewards)
                    accuracy = total_rewards / (query_y.numel())
                    acc.update(accuracy, 1)
                    mesg = "Acc={:.4f}".format(acc.avg)
                    tqdm_gen.set_description(mesg)
                    accuracies.append(accuracy)

                test_accuracy, h = mean_confidence_interval(accuracies)
                print("test accuracy:",test_accuracy,"h:",h)
                print("test_accuracy:", test_accuracy, "h:", h, file=f_txt)
                total_accuracies += test_accuracy
                total_h += h
            print("aver_accuracy:", total_accuracies/self.total_testtimes, "h:", total_h/self.total_testtimes)
            print("aver_accuracy:", total_accuracies/self.total_testtimes, "h:", total_h/self.total_testtimes, file=f_txt)
            return test_accuracy

def remove_module(stat_dict):
    from collections import OrderedDict
    new_stat_dict = OrderedDict()
    for k, v in stat_dict.items():
        if 'module' in k:
            k=k.replace("module.", "")
        new_stat_dict[k] = v 
    return new_stat_dict