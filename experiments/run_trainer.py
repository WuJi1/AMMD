from doctest import FAIL_FAST
import os
import os.path as osp
import sys
import argparse

import shutil
from time import localtime, strftime
sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

from default.default import cfg

from engines.trainer import trainer as t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, dest='cfg',
                        default='/home/wuji/MMD-for-fewshot-AttentiveMMD/configs_arxived/configs_vit/miniImagenet/MMD_linear_triplet_N5k1_swin_tiny.yaml')
    #parser.add_argument('--cfg', type=str, dest='cfg', default='/home/wuji/MMD-for-fewshot-AttentiveMMD/configs_mmdsa/miniImagenet/mmd_linear_triplet_N5K1_R12_test.yaml')
    parser.add_argument("-d",'--device', type=str, dest='device', default='0')
    parser.add_argument('-p', '--checkpoint_dir', type=str, default='')
    parser.add_argument('-pt', '--pretrained_dir', type=str,
                        default='')
    parser.add_argument('-c', '--checkpoint_base', type=str, dest='checkpoint_base', default='./checkpoint')
    parser.add_argument('-s', '--snapshot_base', type=str, dest='snapshot_base', default='./snapshots')
    parser.add_argument('-e', '--eval_after_train', type=int, dest='eval_after_train', default=1)
    parser.add_argument('-m', '--masked_ratio', type=float, dest='masked_ratio', default=0.0)
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    pardir = osp.basename(osp.abspath(osp.join(args.cfg, osp.pardir)))
    if not args.checkpoint_dir:
        args.checkpoint_dir = pardir + "_" + osp.basename(args.cfg).replace(".yaml", "")
    
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.rest:
        cfg.merge_from_list(args.rest)
   
    # cfg.data.image_dir = osp.join(cfg.data.root, pardir)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    
    cfg.model.masked_ratio = args.masked_ratio
    dir_name = args.cfg.split('/')[1]
    checkpoint_dir = osp.join(args.checkpoint_base, dir_name, 'masked_ratio_{}'.format(cfg.model.masked_ratio))
    snapshot_dir = osp.join(args.snapshot_base, dir_name, 'masked_ratio_{}'.format(cfg.model.masked_ratio))
    for d in [checkpoint_dir, snapshot_dir]:
        if not osp.exists(d):
            os.makedirs(d)
    checkpoint_dir = osp.join(checkpoint_dir, args.checkpoint_dir)
    snapshot_dir = osp.join(snapshot_dir, args.checkpoint_dir)

    for d in [checkpoint_dir, snapshot_dir]:
        if not osp.exists(d):
            os.mkdir(d)
    

    print("[*] Source Image Path: {}".format(cfg.data.image_dir))
    print("[*] Target Checkpoint Path: {}".format(checkpoint_dir))
    # import ipdb 
    # ipdb.set_trace()
    if args.pretrained_dir:
        args.pretrained_dir = osp.join(args.pretrained_dir)
    trainer = t(cfg, checkpoint_dir, args.pretrained_dir)
    trainer.run()

    snapshot_dir = osp.join(snapshot_dir, strftime("%Y-%m-%d-%H:%M", localtime()))
    if not osp.exists(snapshot_dir):
        os.mkdir(snapshot_dir)

    shutil.copyfile(args.cfg, osp.join(snapshot_dir, osp.basename(args.cfg))) # , strftime("%Y-%m-%d-%H:%M", localtime())
    shutil.copyfile(trainer.snapshot_name("best"), osp.join(snapshot_dir, osp.basename(trainer.snapshot_name("best"))))
    shutil.copyfile(trainer.snapshot_record("best"), osp.join(snapshot_dir, osp.basename(trainer.snapshot_record("best"))))
    shutil.copytree(trainer.writer_dir, osp.join(snapshot_dir, osp.basename(trainer.writer_dir)))

    if args.eval_after_train:
        print("[*] Running Evaluations ...")
        from engines.evaluator import Evaluator as e
        evaluator = e(cfg, trainer.snapshot_name("best"))
        accuracy = evaluator.run()
        shutil.copyfile(evaluator.prediction_dir, osp.join(snapshot_dir, osp.basename(evaluator.prediction_dir)))
        shutil.move(snapshot_dir, snapshot_dir + "_{:.3f}".format(accuracy * 100))
        
if __name__ == "__main__":
    main()

