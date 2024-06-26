import re
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from time import gmtime, strftime
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from tensorboardX import SummaryWriter

import random
import tqdm
from modules.fsl_query import make_fsl
from dataloader import make_dataloader
from engines.utils import mean_confidence_interval, AverageMeter, set_seed
import pdb

class trainer(object):
    def __init__(self, cfg, checkpoint_dir, pretrained_dir=None):
        self.seed = cfg.seed
        set_seed(self.seed) # should set seed for training from scratch with Conv4 backbone

        self.n_way                 = cfg.n_way # 5
        self.k_shot                = cfg.k_shot # 5
        self.train_query_per_class = cfg.train.query_per_class_per_episode # 10
        self.val_query_per_class   = cfg.test.query_per_class_per_episode  # 15
        self.train_episode_per_epoch = cfg.train.episode_per_epoch
        self.prefix = osp.basename(checkpoint_dir)
        self.writer_dir = self._prepare_summary_snapshots(self.prefix, cfg)
        self.writer = SummaryWriter(self.writer_dir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.epochs = cfg.train.epochs
        self.query = cfg.model.query

        self.fsl = make_fsl(cfg).to(self.device)

        self.lr = cfg.train.learning_rate
        self.lr_decay = cfg.train.lr_decay
        self.lr_decay_epoch = cfg.train.lr_decay_epoch
        if cfg.train.optim == "Adam":
            # self.optim = Adam(self.fsl.parameters(),lr=cfg.train.learning_rate, betas=cfg.train.adam_betas)
            self.optim = Adam(self.fsl.parameters(),lr=cfg.train.learning_rate, weight_decay=0.001)
        elif cfg.train.optim == "SGD":
            self.optim = SGD(
                self.fsl.parameters(), 
                lr=cfg.train.learning_rate, 
                momentum=cfg.train.sgd_mom, 
                weight_decay=cfg.train.sgd_weight_decay,
                nesterov=True
            )
        else:
            raise NotImplementedError
        if pretrained_dir:
            # import ipdb
            # ipdb.set_trace()
            assert os.path.isdir(pretrained_dir)
            # pths = [osp.basename(f) for f in glob.glob(osp.join(pretrained_dir, "*.pth")) if "best" not in f]
            pths = [osp.basename(f) for ext in ["*.pth", "*.tar"] for f in glob.glob(osp.join(pretrained_dir, ext)) if "best" not in f]
        else:
            pths = []
        if pths:
            if "swin" in checkpoint_dir or "vit" in checkpoint_dir:
                pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
                pths = [p for p, e in zip(pths, pths_epoch) if e]
                pths_epoch = [int(e) for e in pths_epoch if e]
                self.train_start_epoch = max(pths_epoch)
                c = osp.join(pretrained_dir, pths[pths_epoch.index(self.train_start_epoch)])
                state_dict = torch.load(c)
                #pdb.set_trace()
                state_dict = state_dict['teacher']
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                state_dict = {k.replace("backbone.", "encoder."): v for k, v in state_dict.items()}
                # chkpt_state_dict = state_dict['teacher']
                self.fsl.load_state_dict(state_dict, strict=False)
                msg = self.fsl.load_state_dict(state_dict, strict=False)
                print(f'Info on loaded state dict and dropped head parameters: \n{msg}')
                print(
                    "Note: If unexpected_keys other than parameters relating to the discarded 'head' exist, go and check!")
                print("[*] Continue training from checkpoints: {}".format(c))
                lr_scheduler_last_epoch = self.train_start_epoch
            elif "deepemd_pretrain_model" in pretrained_dir:
                pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
                pths = [p for p, e in zip(pths, pths_epoch) if e]
                pths_epoch = [int(e) for e in pths_epoch if e]
                self.train_start_epoch = max(pths_epoch)
                c = osp.join(pretrained_dir, pths[pths_epoch.index(self.train_start_epoch)])
                state_dict = torch.load(c)
                state_dict = state_dict['params']
                state_dict = {"encoder." + key: value for key, value in state_dict.items()}
                self.fsl.load_state_dict(state_dict, strict=False)
                print("[*] Continue training from checkpoints: {}".format(c))
            elif "WRN" in checkpoint_dir:
                pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
                pths = [p for p, e in zip(pths, pths_epoch) if e]
                pths_epoch = [int(e) for e in pths_epoch if e]
                self.train_start_epoch = max(pths_epoch)
                c = osp.join(pretrained_dir, pths[pths_epoch.index(self.train_start_epoch)])
                state_dict = torch.load(c)
                state_dict = state_dict['state']
                state_dict = {k.replace("feature.", "encoder."): v for k, v in state_dict.items()}
                self.fsl.load_state_dict(state_dict, strict=False)
                print("[*] Continue training from checkpoints: {}".format(c))
            else:
                pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
                pths = [p for p, e in zip(pths, pths_epoch) if e]
                pths_epoch = [int(e) for e in pths_epoch if e]
                self.train_start_epoch = max(pths_epoch)
                c = osp.join(pretrained_dir, pths[pths_epoch.index(self.train_start_epoch)])
                state_dict = torch.load(c)
                self.fsl.load_state_dict(state_dict["fsl"], strict=False)
                print("[*] Continue training from checkpoints: {}".format(c))
            # if "optimizer" in state_dict and state_dict["optimizer"] is not None:
            #     self.optim.load_state_dict(state_dict["optimizer"])
        else:
            lr_scheduler_last_epoch = -1
        self.train_start_epoch = 0

        if cfg.train.lr_decay_milestones:
            self.lr_scheduler = MultiStepLR(self.optim, milestones=cfg.train.lr_decay_milestones,gamma=self.lr_decay)
        elif cfg.model.query == 'CPEA':
            self.lr_scheduler = StepLR(self.optim, step_size=4, gamma=0.5)
        else:
            self.lr_scheduler = StepLR(self.optim, step_size=self.lr_decay_epoch, gamma=self.lr_decay)

        self.snapshot_name = lambda prefix: \
            osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.pth".format(prefix, self.n_way, self.k_shot))
        self.snapshot_record = lambda prefix: \
            osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.txt".format(prefix, self.n_way, self.k_shot))
        self.cfg = cfg

    def _prepare_summary_snapshots(self, prefix, cfg):
        summary_prefix = osp.join(cfg.train.summary_snapshot_base, prefix)
        summary_dir = osp.join(summary_prefix, strftime("%Y-%m-%d-%H:%M", gmtime()))
        for d_ in [summary_prefix, summary_dir]:
            if not osp.exists(d_):
                os.mkdir(d_)
        return summary_dir

    def fix_bn(self):
        for module in self.fsl.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.SyncBatchNorm):
                module.eval()

    def validate(self, dataloader):
        accuracies = []
        tqdm_gen = tqdm.tqdm(dataloader, ncols=80)
        acc = AverageMeter()
        loss_meter = AverageMeter()
        for episode, (support_x, support_y, query_x, query_y) in enumerate(tqdm_gen):
            support_x            = support_x.to(self.device)
            support_y            = support_y.to(self.device)
            query_x              = query_x.to(self.device)
            query_y              = query_y.to(self.device)

            rewards = self.fsl(support_x, support_y, query_x, query_y)#,episode)
            if isinstance(rewards, tuple):
                rewards, losses = rewards
                loss_meter.update(losses.item(), len(query_x))
            total_rewards = np.sum(rewards)

            accuracy = total_rewards / (query_y.numel())

            acc.update(total_rewards / query_y.numel(), 1)
            mesg = "Val: acc={:.4f}".format(
                acc.avg
            )
            tqdm_gen.set_description(mesg)

            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        return test_accuracy, h, loss_meter

    def save_model(self, prefix, accuracy, h, epoch, final_epoch=False):
        filename = self.snapshot_name(prefix)
        recordname = self.snapshot_record(prefix)
        state = {
            'summary_dir': osp.basename(self.writer_dir),
            'episode': prefix,
            'fsl': self.fsl.state_dict(),
            'epoch': epoch,
            # "optimizer": None if not final_epoch else self.optim.state_dict()
        }
        with open(recordname, 'w') as f:
            f.write("prefix: {}\nepoch: {}\naccuracy: {}\nh: {}\n".format(prefix, epoch, accuracy, h)) 
        if int(re.search(r'([\d.]+)', torch.__version__).group(1).replace('.', '')) > 160:
            torch.save(state, filename, _use_new_zipfile_serialization=False) # compatible with early torch versions to load
        else:
            torch.save(state, filename)

    def train(self, dataloader, epoch):
        losses = AverageMeter()
        cls_losses = AverageMeter()
        tqdm_gen = tqdm.tqdm(dataloader, ncols=120)

        self.optim.zero_grad()
        for episode, (support_x, support_y, query_x, query_y) in enumerate(tqdm_gen):
            support_x            = support_x.to(self.device)
            support_y            = support_y.to(self.device)
            query_x              = query_x.to(self.device)
            query_y              = query_y.to(self.device)

            loss_ = self.fsl(support_x, support_y, query_x, query_y)#,episode)
            if isinstance(loss_, tuple):
                loss = loss_[0]
                cls_loss = loss_[1]
                cls_losses.update(cls_loss.item(), support_x.shape[0] * support_x.shape[1])
            else:
                loss = loss_
            self.optim.zero_grad()
            loss.backward()
            ##################CPEA################
            if self.query == 'CPEA':
                for name, param in self.fsl.named_parameters():
                    if param.requires_grad and name.startswith('query'):
                        param.grad *= 1
            ######################################
            self.optim.step()
            losses.update(loss.item(), len(query_x))
            mesg = "epoch {}, avg_loss={:.3f}, loss={:.3f}, cls_loss={:.3f}".format(
                epoch, 
                losses.avg,
                losses.val,
                cls_losses.avg
            )
            tqdm_gen.set_description(mesg)
        self.writer.add_scalar('loss_cls', cls_losses.avg, epoch)
        return losses.avg
    def run(self):
        print("[={}=]".format(self.prefix))
        best_accuracy = 0.0
        set_seed(self.seed)
        # import ipdb 
        # ipdb.set_trace()
        # 改改改
        val_dataloader = make_dataloader(self.cfg, phase="val", batch_size=self.cfg.test.batch_size)
        # val_dataloader = make_dataloader(self.cfg, phase="test", batch_size=self.cfg.test.batch_size)
        for epoch in range(self.train_start_epoch, self.epochs):
            train_dataloader = make_dataloader(
                self.cfg, phase="train", 
                batch_size=self.cfg.train.batch_size
            )
            loss_train = self.train(train_dataloader, epoch + 1)
            self.writer.add_scalar('loss_train', loss_train, epoch + 1)
            self.fsl.eval()

            with torch.no_grad():
                val_accuracy, h, val_loss_meter = self.validate(val_dataloader)
                if val_loss_meter.count > 0:
                    self.writer.add_scalar('loss_val', val_loss_meter.avg, epoch + 1)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model("best", val_accuracy, h, epoch + 1, True)
            if (epoch + 1) % 2 == 0:
                i = epoch + 1
                self.save_model(i, val_accuracy, h, epoch + 1, True)

            mesg = "\t Testing epoch {} validation accuracy: {:.4f}, h: {:.3f}".format(epoch + 1, val_accuracy, h)
            print(mesg)
            self.writer.add_scalar('acc_val', val_accuracy, epoch + 1)

            self.lr_scheduler.step()
            #self.save_model(epoch + 1, val_accuracy, h, epoch + 1, epoch == (self.epochs - 1))
            self.fsl.train()
            if self.cfg.train.fix_bn:
                self.fix_bn()

# import re
# import os
# import os.path as osp
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import glob
# from time import gmtime, strftime
# from torch.optim import Adam, SGD
# from torch.optim.lr_scheduler import StepLR, MultiStepLR
# import numpy as np
# from tensorboardX import SummaryWriter

# import random
# import tqdm
# from modules.fsl_query import make_fsl
# from dataloader import make_dataloader
# from engines.utils import mean_confidence_interval, AverageMeter, set_seed
# import pdb

# class trainer(object):
#     def __init__(self, cfg, checkpoint_dir, pretrained_dir=None):
#         self.seed = cfg.seed
#         set_seed(self.seed) # should set seed for training from scratch with Conv4 backbone

#         self.n_way                 = cfg.n_way # 5
#         self.k_shot                = cfg.k_shot # 5
#         self.train_query_per_class = cfg.train.query_per_class_per_episode # 10
#         self.val_query_per_class   = cfg.test.query_per_class_per_episode  # 15
#         self.train_episode_per_epoch = cfg.train.episode_per_epoch
#         self.prefix = osp.basename(checkpoint_dir)
#         self.writer_dir = self._prepare_summary_snapshots(self.prefix, cfg)
#         self.writer = SummaryWriter(self.writer_dir)

#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.checkpoint_dir = checkpoint_dir
#         self.epochs = cfg.train.epochs
#         self.query = cfg.model.query

#         self.fsl = make_fsl(cfg).to(self.device)

#         self.lr = cfg.train.learning_rate
#         self.lr_decay = cfg.train.lr_decay
#         self.lr_decay_epoch = cfg.train.lr_decay_epoch
#         self.optim = Adam(self.fsl.encoder.parameters(),lr=cfg.train.learning_rate, weight_decay=0.001)
#         self.optim_query = Adam(self.fsl.query.parameters(),lr=cfg.train.learning_rate * 100, weight_decay=0.001)


#         if pretrained_dir:
#             # import ipdb
#             # ipdb.set_trace()
#             assert os.path.isdir(pretrained_dir)
#             # pths = [osp.basename(f) for f in glob.glob(osp.join(pretrained_dir, "*.pth")) if "best" not in f]
#             pths = [osp.basename(f) for ext in ["*.pth", "*.tar"] for f in glob.glob(osp.join(pretrained_dir, ext)) if "best" not in f]
#         else:
#             pths = []
#         if pths:
#             if "swin" in checkpoint_dir or "vit" in checkpoint_dir:
#                 pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
#                 pths = [p for p, e in zip(pths, pths_epoch) if e]
#                 pths_epoch = [int(e) for e in pths_epoch if e]
#                 self.train_start_epoch = max(pths_epoch)
#                 c = osp.join(pretrained_dir, pths[pths_epoch.index(self.train_start_epoch)])
#                 state_dict = torch.load(c)
#                 #pdb.set_trace()
#                 state_dict = state_dict['teacher']
#                 state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#                 state_dict = {k.replace("backbone.", "encoder."): v for k, v in state_dict.items()}
#                 # chkpt_state_dict = state_dict['teacher']
#                 self.fsl.load_state_dict(state_dict, strict=False)
#                 msg = self.fsl.load_state_dict(state_dict, strict=False)
#                 print(f'Info on loaded state dict and dropped head parameters: \n{msg}')
#                 print(
#                     "Note: If unexpected_keys other than parameters relating to the discarded 'head' exist, go and check!")
#                 print("[*] Continue training from checkpoints: {}".format(c))
#                 lr_scheduler_last_epoch = self.train_start_epoch
#             elif "deepemd_pretrain_model" in pretrained_dir:
#                 pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
#                 pths = [p for p, e in zip(pths, pths_epoch) if e]
#                 pths_epoch = [int(e) for e in pths_epoch if e]
#                 self.train_start_epoch = max(pths_epoch)
#                 c = osp.join(pretrained_dir, pths[pths_epoch.index(self.train_start_epoch)])
#                 state_dict = torch.load(c)
#                 state_dict = state_dict['params']
#                 state_dict = {"encoder." + key: value for key, value in state_dict.items()}
#                 self.fsl.load_state_dict(state_dict, strict=False)
#                 print("[*] Continue training from checkpoints: {}".format(c))
#             elif "WRN" in checkpoint_dir:
#                 pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
#                 pths = [p for p, e in zip(pths, pths_epoch) if e]
#                 pths_epoch = [int(e) for e in pths_epoch if e]
#                 self.train_start_epoch = max(pths_epoch)
#                 c = osp.join(pretrained_dir, pths[pths_epoch.index(self.train_start_epoch)])
#                 state_dict = torch.load(c)
#                 state_dict = state_dict['state']
#                 state_dict = {k.replace("feature.", "encoder."): v for k, v in state_dict.items()}
#                 self.fsl.load_state_dict(state_dict, strict=False)
#                 print("[*] Continue training from checkpoints: {}".format(c))
#             else:
#                 pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
#                 pths = [p for p, e in zip(pths, pths_epoch) if e]
#                 pths_epoch = [int(e) for e in pths_epoch if e]
#                 self.train_start_epoch = max(pths_epoch)
#                 c = osp.join(pretrained_dir, pths[pths_epoch.index(self.train_start_epoch)])
#                 state_dict = torch.load(c)
#                 self.fsl.load_state_dict(state_dict["fsl"], strict=False)
#                 print("[*] Continue training from checkpoints: {}".format(c))
#             # if "optimizer" in state_dict and state_dict["optimizer"] is not None:
#             #     self.optim.load_state_dict(state_dict["optimizer"])
#         else:
#             lr_scheduler_last_epoch = -1
#         self.train_start_epoch = 0

#         # if cfg.train.lr_decay_milestones:
#         #     self.lr_scheduler = MultiStepLR(self.optim, milestones=cfg.train.lr_decay_milestones,gamma=self.lr_decay)
#         # else:
#         #     self.lr_scheduler = StepLR(self.optim, step_size=self.lr_decay_epoch, gamma=self.lr_decay)
#         self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.5)
#         self.query_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim_query, step_size=5, gamma=0.5)

#         self.snapshot_name = lambda prefix: \
#             osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.pth".format(prefix, self.n_way, self.k_shot))
#         self.snapshot_record = lambda prefix: \
#             osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.txt".format(prefix, self.n_way, self.k_shot))
#         self.cfg = cfg

#     def _prepare_summary_snapshots(self, prefix, cfg):
#         summary_prefix = osp.join(cfg.train.summary_snapshot_base, prefix)
#         summary_dir = osp.join(summary_prefix, strftime("%Y-%m-%d-%H:%M", gmtime()))
#         for d_ in [summary_prefix, summary_dir]:
#             if not osp.exists(d_):
#                 os.mkdir(d_)
#         return summary_dir

#     def fix_bn(self):
#         for module in self.fsl.modules():
#             if isinstance(module, torch.nn.modules.BatchNorm2d):
#                 module.eval()
#             if isinstance(module, torch.nn.modules.SyncBatchNorm):
#                 module.eval()

#     def validate(self, dataloader):
#         accuracies = []
#         tqdm_gen = tqdm.tqdm(dataloader, ncols=80)
#         acc = AverageMeter()
#         loss_meter = AverageMeter()
#         for episode, (support_x, support_y, query_x, query_y) in enumerate(tqdm_gen):
#             support_x            = support_x.to(self.device)
#             support_y            = support_y.to(self.device)
#             query_x              = query_x.to(self.device)
#             query_y              = query_y.to(self.device)

#             rewards = self.fsl(support_x, support_y, query_x, query_y)#,episode)
#             if isinstance(rewards, tuple):
#                 rewards, losses = rewards
#                 loss_meter.update(losses.item(), len(query_x))
#             total_rewards = np.sum(rewards)

#             accuracy = total_rewards / (query_y.numel())

#             acc.update(total_rewards / query_y.numel(), 1)
#             mesg = "Val: acc={:.4f}".format(
#                 acc.avg
#             )
#             tqdm_gen.set_description(mesg)

#             accuracies.append(accuracy)

#         test_accuracy, h = mean_confidence_interval(accuracies)
#         return test_accuracy, h, loss_meter

#     def save_model(self, prefix, accuracy, h, epoch, final_epoch=False):
#         filename = self.snapshot_name(prefix)
#         recordname = self.snapshot_record(prefix)
#         state = {
#             'summary_dir': osp.basename(self.writer_dir),
#             'episode': prefix,
#             'fsl': self.fsl.state_dict(),
#             'epoch': epoch,
#             # "optimizer": None if not final_epoch else self.optim.state_dict()
#         }
#         with open(recordname, 'w') as f:
#             f.write("prefix: {}\nepoch: {}\naccuracy: {}\nh: {}\n".format(prefix, epoch, accuracy, h)) 
#         if int(re.search(r'([\d.]+)', torch.__version__).group(1).replace('.', '')) > 160:
#             torch.save(state, filename, _use_new_zipfile_serialization=False) # compatible with early torch versions to load
#         else:
#             torch.save(state, filename)

#     def train(self, dataloader, epoch):
#         losses = AverageMeter()
#         cls_losses = AverageMeter()
#         tqdm_gen = tqdm.tqdm(dataloader, ncols=120)

#         self.optim.zero_grad()
#         for episode, (support_x, support_y, query_x, query_y) in enumerate(tqdm_gen):
#             support_x            = support_x.to(self.device)
#             support_y            = support_y.to(self.device)
#             query_x              = query_x.to(self.device)
#             query_y              = query_y.to(self.device)

#             loss_ = self.fsl(support_x, support_y, query_x, query_y)#,episode)
#             if isinstance(loss_, tuple):
#                 loss = loss_[0]
#                 cls_loss = loss_[1]
#                 cls_losses.update(cls_loss.item(), support_x.shape[0] * support_x.shape[1])
#             else:
#                 loss = loss_
#             self.optim.zero_grad()
#             loss.backward()
#             ##################CPEA################
#             # if self.query == 'CPEA':
#             #     for name, param in self.fsl.named_parameters():
#             #         if param.requires_grad and name.startswith('query'):
#             #             param.grad *= 100
#             ######################################
#             self.optim.step()
#             losses.update(loss.item(), len(query_x))
#             mesg = "epoch {}, avg_loss={:.3f}, loss={:.3f}, cls_loss={:.3f}".format(
#                 epoch, 
#                 losses.avg,
#                 losses.val,
#                 cls_losses.avg
#             )
#             tqdm_gen.set_description(mesg)
#         self.writer.add_scalar('loss_cls', cls_losses.avg, epoch)
#         return losses.avg
#     def run(self):
#         print("[={}=]".format(self.prefix))
#         best_accuracy = 0.0
#         set_seed(self.seed)
#         # import ipdb 
#         # ipdb.set_trace()
#         val_dataloader = make_dataloader(self.cfg, phase="val", batch_size=self.cfg.test.batch_size)
#         for epoch in range(self.train_start_epoch, self.epochs):
#             train_dataloader = make_dataloader(
#                 self.cfg, phase="train", 
#                 batch_size=self.cfg.train.batch_size
#             )
#             loss_train = self.train(train_dataloader, epoch + 1)
#             self.writer.add_scalar('loss_train', loss_train, epoch + 1)
#             self.fsl.eval()

#             with torch.no_grad():
#                 val_accuracy, h, val_loss_meter = self.validate(val_dataloader)
#                 if val_loss_meter.count > 0:
#                     self.writer.add_scalar('loss_val', val_loss_meter.avg, epoch + 1)
#             if val_accuracy > best_accuracy:
#                 best_accuracy = val_accuracy
#                 self.save_model("best", val_accuracy, h, epoch + 1, True)
#             if (epoch + 1) % 2 == 0:
#                 i = epoch + 1
#                 self.save_model(i, val_accuracy, h, epoch + 1, True)

#             mesg = "\t Testing epoch {} validation accuracy: {:.4f}, h: {:.3f}".format(epoch + 1, val_accuracy, h)
#             print(mesg)
#             self.writer.add_scalar('acc_val', val_accuracy, epoch + 1)

#             self.lr_scheduler.step()
#             self.query_lr_scheduler.step
#             #self.save_model(epoch + 1, val_accuracy, h, epoch + 1, epoch == (self.epochs - 1))
#             self.fsl.train()
#             if self.cfg.train.fix_bn:
#                 self.fix_bn()
