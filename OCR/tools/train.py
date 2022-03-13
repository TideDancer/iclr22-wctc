import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from vedastr.runners import TrainRunner
from vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('gpus', type=str, help='target gpus')
    parser.add_argument('--mask_ratio', type=float, default=0.0, help='mask ratio')
    parser.add_argument('--show', type=str, default=None, help='show')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    _, fullname = os.path.split(cfg_path)
    fname, ext = os.path.splitext(fullname)

    root_workdir = cfg.pop('root_workdir')
    workdir = os.path.join(root_workdir, fname)
    
    workdir = os.path.join(workdir, str(args.mask_ratio))
    os.makedirs(workdir, exist_ok=True)

    train_cfg = cfg['train']
    deploy_cfg = cfg['deploy']
    common_cfg = cfg['common']
    common_cfg['workdir'] = workdir
    deploy_cfg['gpu_id'] = args.gpus.replace(" ", "")

    runner = TrainRunner(train_cfg, deploy_cfg, common_cfg, mask_ratio=args.mask_ratio, show=args.show)
    runner()


if __name__ == '__main__':
    main()

