import os
import os.path as osp
import sys
import argparse

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

# from configs.miniimagenet_default import cfg
from default.default import cfg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', default='/home/wuji/MMD-for-fewshot-AttentiveMMD/configs_arxived/configs_vit/miniImagenet/proto_linear_triplet_N5k1_vit_s.yaml')
    parser.add_argument("-d",'--device', type=str, dest='device', default='1')
    parser.add_argument('-c', '--checkpoint', type=str, default='/media/omnisky/Disk2/wuji/checkpoints/vit_s/miniImagenet/checkpoint00.pth')
    parser.add_argument('-b', '--checkpoint_base', type=str, default='./checkpoint')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    pardir = osp.basename(osp.abspath(osp.join(args.cfg, osp.pardir)))
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.rest:
        cfg.merge_from_list(args.rest)

    if not args.checkpoint:
        args.checkpoint = pardir + "_" + osp.basename(args.cfg).replace(".yaml", "")
        args.checkpoint = osp.join(args.checkpoint_base, args.checkpoint, "ebest_{}way_{}shot.pth".format(cfg.n_way, cfg.k_shot))
        if not osp.exists(args.checkpoint):
            raise FileNotFoundError

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    print("[*] Source Image Path: {}".format(cfg.data.image_dir))
    print("[*] Source Checkpoint Path: {}".format(args.checkpoint))
    is_vit = 'vit' in args.cfg.split('/')[1]
    if not is_vit:
        from engines.evaluator import Evaluator as e
    else:
        from engines.evaluator_vit import Evaluator as e
    evaluator = e(cfg, args.checkpoint)
    evaluator.run()

if __name__ == "__main__":
    main()

