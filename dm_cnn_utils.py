import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNxN(nn.Module):

    def __init__(
        self, k, n_in, n_out, stride=1, padding=0, bias=True, batchnorm=True,
        activate=True, track_running_stats=False
    ):
        super(ConvNxN, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=k, stride=stride, padding=padding, bias=bias)
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn1 = torch.nn.BatchNorm3d(
                n_out, track_running_stats=track_running_stats
            )
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
        self.activate = activate

    def forward(self, x):
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn1(x)
        if self.activate:
            x = F.relu(x)
        return x


def conv1x1(
    n_in, n_out, stride=1, bias=True, batchnorm=True, activate=True,
    track_running_stats=False
):
    return ConvNxN(
        1, n_in, n_out, stride=stride, padding=0, bias=bias,
        batchnorm=batchnorm, activate=activate,
        track_running_stats=track_running_stats
    )


def conv3x3(
    n_in, n_out, stride=1, bias=True, batchnorm=True, activate=True,
    track_running_stats=False
):
    return ConvNxN(
        3, n_in, n_out, stride=stride, padding=1, bias=bias,
        batchnorm=batchnorm, activate=activate,
        track_running_stats=track_running_stats
    )


def conv7x7(
    n_in, n_out, stride=1, bias=True, batchnorm=True, activate=True,
    track_running_stats=False
):
    return ConvNxN(
        7, n_in, n_out, stride=stride, padding=3, bias=bias,
        batchnorm=batchnorm, activate=activate,
        track_running_stats=track_running_stats
    )


def repeat_layers(layer, n):
    layers = [layer for i in range(n)]
    block = nn.Sequential(*layers)
    return block


class ResBlock(nn.Module):

    def __init__(self, n_in, n_out, stride=1, track_running_stats=False):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.conv1 = conv3x3(
            n_in, n_out, stride, bias=False,
            track_running_stats=track_running_stats
        )
        self.conv2 = conv3x3(
            n_out, n_out, stride=1, bias=False, activate=False,
            track_running_stats=track_running_stats
        )
        if self.stride > 1:
            self.conv3 = conv1x1(
                n_in, n_out, stride=self.stride, bias=False, activate=False,
                track_running_stats=track_running_stats
            )

    def forward(self, x):
        residual = x
        if self.stride > 1:
            residual = self.conv3(residual)
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = F.relu(x)
        return x


class BottleneckBlock(nn.Module):

    def __init__(self, n1, n2, stride=1):
        super(BottleneckBlock, self).__init__()
        self.stride = stride
        if self.stride > 1:
            self.conv1 = conv1x1(n1, n2, activate=True, stride=self.stride, bias=False, batchnorm=True)
        else:
            self.conv1 = conv1x1(n1, n2, activate=True, stride=1, bias=False, batchnorm=True)
        self.conv2 = conv3x3(n2, n2, activate=True, stride=1, bias=False, batchnorm=True)
        self.conv3 = conv1x1(n2, n2 * 4, activate=False, stride=1, bias=False, batchnorm=True)
        if self.stride > 1:
            self.conv_shortcut = conv1x1(n1, n2 * 4, activate=False, stride=self.stride, bias=False, batchnorm=True)
        else:
            self.conv_shortcut = conv1x1(n1, n2 * 4, activate=False, stride=1, bias=False, batchnorm=True)

    def forward(self, x):
        residual = x
        residual = self.conv_shortcut(residual)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        x = F.relu(x)
        return x


class FocalLoss(nn.Module):

    def __init__(self, weight, gamma=2, reduce=True, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduce = reduce
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none", ignore_index=ignore_index)
        self.mod = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def forward(self, scores, targets):
        ce_loss = self.ce(scores, targets)
        f_loss = (1 - torch.exp(-self.mod(scores, targets)))**self.gamma * ce_loss
        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss
