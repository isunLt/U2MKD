import numpy as np
import torch
import torchvision.transforms.functional
from torch import nn
import torch.nn.functional as F
import math
try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

__all__ = ['Lovasz_softmax', 'MixLovaszCrossEntropy']


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


# def flatten_probas(probas, labels, ignore=None):
#     """
#     Flattens predictions in the batch
#     """
#     if probas.dim() == 3:
#         # assumes output of a sigmoid layer
#         B, H, W = probas.size()
#         probas = probas.view(B, 1, H, W)
#     B, C, H, W = probas.size()
#     probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
#     labels = labels.view(-1)
#     if ignore is None:
#         return probas, labels
#     valid = (labels != ignore)
#     vprobas = probas[valid.nonzero().squeeze()]
#     vlabels = labels[valid]
#     return vprobas, vlabels

# class Lovasz_softmax(nn.Module):
#     def __init__(self, classes='present'):
#         super(Lovasz_softmax, self).__init__()
#         self.classes = classes
#
#     def forward(self, probas, labels):
#         return lovasz_softmax_flat(probas, labels, self.classes)
def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() != 2:
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', ignore_index=0):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, classes=self.classes, ignore=self.ignore_index)


class MixLovaszCrossEntropy(nn.Module):
    def __init__(self, weight=None, classes='present', ignore_index=255):
        super(MixLovaszCrossEntropy, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.lovasz = Lovasz_softmax(classes, ignore_index=ignore_index)
        if weight is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            assert isinstance(weight, torch.Tensor)
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, x, y):
        lovasz = self.lovasz(F.softmax(x, 1), y)
        ce = self.ce(x, y)
        return lovasz + ce


class MixLCLovaszCrossEntropy(nn.Module):
    def __init__(self, weight=None, classes='present', ignore_index=0):
        super(MixLCLovaszCrossEntropy, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.lovasz = Lovasz_softmax(classes, ignore_index=ignore_index)
        if weight is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            assert isinstance(weight, torch.Tensor)
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, out_dict: dict, y):
        x_vox = out_dict.get('x_vox')
        x_pix = out_dict.get('x_pix')
        ce_vox = self.ce(x_vox, y)
        lovz_vox = self.lovasz(F.softmax(x_vox, 1), y)
        ce_pix = self.ce(x_pix, y)
        lovz_pix = self.lovasz(F.softmax(x_pix, 1), y)
        return {
            'predict_vox': ce_vox,
            'lovz_vox': lovz_vox,
            'predict_pix': ce_pix,
            'lovz_pix': lovz_pix
        }


class DistillLovaszCrossEntropy(nn.Module):
    def __init__(self, weight=None, classes='present', ignore_index=0):
        super(DistillLovaszCrossEntropy, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.lovasz = Lovasz_softmax(classes, ignore_index=ignore_index)
        if weight is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            assert isinstance(weight, torch.Tensor)
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, out_dict: dict, y):
        x_vox = out_dict.get('x_vox')
        x_mix = out_dict.get('x_mix')
        fov_mask = out_dict.get('fov_mask')
        y_mix = y[fov_mask]
        ce_vox = self.ce(x_vox, y) + self.lovasz(F.softmax(x_vox, 1), y)
        ce_mix = self.ce(x_mix, y_mix) + self.lovasz(F.softmax(x_mix, 1), y_mix)
        distill_loss = self.kl_div(F.log_softmax(x_vox[fov_mask], dim=1), F.softmax(x_mix.detach(), dim=1))
        return {
            'predict_vox': ce_vox,
            'predict_mix': ce_mix,
            'distill_loss': distill_loss
        }





