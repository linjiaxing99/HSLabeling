import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt


def dice_coeff_new(inputs, target):
    # ONE HOT LABEL as input(bs, num_classes, m, n)

    smooth = 1.
    iflat = inputs.reshape(-1, 1)
    tflat = target.reshape(-1, 1)
    intersection = (iflat * tflat).sum()
    
    return  ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


class DiceLoss(_Loss):
    def forward(self, output, target, weights=None, mask=None):
        """
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
            """
        eps = 0.0001

        output = output.exp()
        encoded_target = output.detach() * 0
        if mask is not None:

            target = target.clone()
            # weight_map = torch.squeeze(mask[:,1:,:])
            target[mask < 1] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask < 1] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if mask is not None:
            denominator[mask < 1] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ignore_index = ignore_index
        self.cross_entropy_loss = CrossEntropyLoss2d(ignore_index=ignore_index)
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight, num_classes):
        # Step 1: softmax 输出
        input_soft = F.softmax(input, dim=1)
        prediction = input_soft.argmax(dim=1)

        mask = target != self.ignore_index  # 有效像素掩码
        target = target.type(torch.LongTensor).cuda()
        target_filtered = target.clone()    # 复制 target 避免修改原始数据
        target_filtered[~mask] = 0          # 将 ignore_index 处设置为 0
        target_filtered = target_filtered.to(torch.long).cuda()

        prediction_valid = prediction[mask]  # 过滤掉无效像素
        target_valid = target[mask]  # 过滤有效像素的 target
        acc = torch.mean((target_valid == prediction_valid).float())

        true_1_hot = torch.eye(num_classes, device=target.device)[target_filtered]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()  # [B, H, W] -> [B, C, H, W]
        true_1_hot = true_1_hot * mask.unsqueeze(1)  # 将无效像素置零

        y3 = dice_coeff_new(input_soft * mask.unsqueeze(1), true_1_hot)  # 确保只计算有效区域

        y2 = self.dice_loss(input_soft, target, weights=None, mask=mask)
        y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))

        eps = 1e-6
        y4 = -torch.log(input_soft + eps)  # 对 softmax 结果取对数
        y4 = torch.mul(y4, true_1_hot)  # 仅对有效像素处的类别计算
        y4 = torch.mean(y4, dim=(1, 2, 3))  # 平均损失

        y = y1 + y2  # Cross Entropy Loss + Dice Loss

        return y, y3, y4, acc, y2


# Loss functions
def loss_coteaching(output1, labels, w, loss_1, num_classes, forget_rate, ignore_index=255):
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember].cuda()

    # 使用 CombinedLoss 计算损失
    loss_func = CombinedLoss(ignore_index=ignore_index)
    loss_1_update = loss_func(output1[ind_1_update].float(), labels[ind_1_update].float(), w[ind_1_update].float(), num_classes)

    return torch.sum(loss_1_update[0]) / len(ind_1_update)


