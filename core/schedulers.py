from typing import Any, Callable

import numpy as np
import torchpack.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

__all__ = ['cosine_schedule_with_warmup']


def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
    """
    load checkpoint时要使lr保持稳定，必须保持batch_size, dist.size(),
    num_epochs, batch_size, dataset_size不变

    :param k: step数，最大值为iter_per_epoch*num_epoch
    :param num_epochs: configs.num_epochs预先设定的训练epoch数
    :param batch_size: configs.batch_size预先设定的batch size
    :param dataset_size: configs.data.training_size训练集大小
    :return:
    """
    batch_size *= dist.size()  # dist.size()表示训练使用的GPU个数
    # print("cosine_schedule_with_warmup: dist.size() =", dist.size())
    if dist.size() == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // dist.size()
    # print("cosine_schedule_with_warmup: warmup_iters =", warmup_iters)
    # print("cosine_schedule_with_warmup: k =", k)
    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        # print("cosine_schedule_with_warmup: iter_per_epoch =", iter_per_epoch)
        return 0.5 * (1 + np.cos(np.pi * (k - warmup_iters) /
                                 (num_epochs * iter_per_epoch)))


class LambdaStepLR(LambdaLR):

    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""

    def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
        super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1)) ** power, last_step)
