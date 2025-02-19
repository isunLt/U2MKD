import argparse
import sys

import random
import numpy as np

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import (InferenceRunner, MaxSaver, Saver)
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.spformer_trainer import NuScenesTrainer
from core.callbacks import MeanIoU
from core.models.utils import SparseSyncBatchNorm


def main() -> None:
    # dist.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--weight-path', metavar='FILE', default=None, help='weight to be load')
    parser.add_argument('--non-dist', action='store_false', help='set to disable distributed train')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.non_dist:
        dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2 ** 32 - 1)

    seed = configs.train.seed + dist.rank() * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = builder.make_dataset()
    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            # shuffle=False, )
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    model = builder.make_model().cuda()
    if args.non_dist:
        model = SparseSyncBatchNorm.convert_sync_batchnorm(model)
    print(model)
    if args.non_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.local_rank()])

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = NuScenesTrainer(model=model,
                              criterion=criterion,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              num_workers=configs.workers_per_gpu,
                              seed=seed,
                              weight_path=args.weight_path,
                              amp_enabled=configs.amp_enabled)
    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[
                      InferenceRunner(
                          dataflow[split],
                          callbacks=[MeanIoU(
                              name=f'iou/{split}/vox',
                              num_classes=configs.data.num_classes,
                              ignore_label=configs.data.ignore_label,
                              output_tensor='outputs_vox',
                              target_tensor='targets'
                          )])
                      for split in ['val']
                  ] + [
                      MaxSaver('iou/val/vox'),
                      # EpochSaver(),
                      Saver(max_to_keep=1),
                  ])


if __name__ == '__main__':
    main()
