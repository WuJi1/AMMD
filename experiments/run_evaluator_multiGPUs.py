import os
import os.path as osp
import sys
import argparse

sys.path.append(osp.abspath(osp.join(osp.abspath(__file__), '..', '..')))

# from configs.miniimagenet_default import cfg
from default.default import cfg

from engines.evaluator_multiGPUs import Evaluator as e

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, dest='cfg', default='')
    parser.add_argument("-d",'--device', type=str, dest='device', default='0')
    parser.add_argument('-c', '--checkpoint', type=str, default='')
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
    if pardir == 'miniImagenet':
        assert cfg.data.mode == 'csv'

    num_gpus = set_gpu(args)

    print("[*] Source Image Path: {}".format(cfg.data.image_dir))
    print("[*] Source Checkpoint Path: {}".format(args.checkpoint))

    evaluator = e(cfg, args.checkpoint, num_gpus)
    evaluator.run()


def set_gpu(args):
    gpu_list = [int(x) for x in args.device.split(',')]
    print ('use gpu:',gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    return gpu_list.__len__()

if __name__ == "__main__":
    main()

