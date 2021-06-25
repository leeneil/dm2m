import torch
import torch.nn as nn

from dm_cnn_utils import *


class DeepMapNet(nn.Module):

    def __init__(
        self, n_in=1, n_out=176, n_filters=16, n_blocks=None, bottleneck=False,
        track_running_stats=False
    ):
        super(DeepMapNet, self).__init__()
        if bottleneck:
            block = BottleneckBlock
        else:
            block = ResBlock
        self.seen = 0
        if n_blocks is None:
            n_blocks = [3, 4, 6]
        self.in_layer = conv7x7(
            n_in, n_filters, stride=2, bias=False, batchnorm=True,
            activate=True, track_running_stats=track_running_stats
        )
        if bottleneck:
            self.out_layer = conv1x1(
                16 * n_filters, n_out, bias=True, batchnorm=False,
                activate=False, track_running_stats=track_running_stats
            )
            self.resnet = nn.Sequential(
                self.in_layer,
                block(
                    n1=n_filters, n2=n_filters, stride=1,
                    track_running_stats=track_running_stats
                ),
                repeat_layers(
                    block(
                        n1=4 * n_filters, n2=n_filters, stride=1,
                        track_running_stats=track_running_stats
                    ),
                    n_blocks[0]-1
                ),
                block(
                    n1=4*n_filters, n2=2 * n_filters, stride=2,
                    track_running_stats=track_running_stats
                ),
                repeat_layers(
                    block(
                        n1=8 * n_filters, n2=2 * n_filters, stride=1,
                        track_running_stats=track_running_stats
                    ),
                    n_blocks[1] - 1
                ),
                block(
                    n1=8 * n_filters, n2=4 * n_filters, stride=2,
                    track_running_stats=track_running_stats
                ),
                repeat_layers(
                    block(
                        n1=16 * n_filters, n2=4 * n_filters, stride=1,
                        track_running_stats=track_running_stats
                    ),
                    n_blocks[2] - 1
                ),
                self.out_layer
            )
        else:
            self.out_layer = conv1x1(
                4 * n_filters, n_out, bias=True, batchnorm=False, activate=False
            )
            self.resnet = nn.Sequential(
                self.in_layer,
                repeat_layers(
                    block(
                        n_in=n_filters, n_out=n_filters, stride=1,
                        track_running_stats=track_running_stats
                    ),
                    n_blocks[0]
                ),
                block(
                    n_in=n_filters, n_out=2 * n_filters, stride=2,
                    track_running_stats=track_running_stats
                ),
                repeat_layers(
                    block(
                        n_in=2 * n_filters, n_out=2 * n_filters, stride=1,
                        track_running_stats=track_running_stats
                    ),
                    n_blocks[1] - 1),
                block(
                    n_in=2 * n_filters, n_out=4 * n_filters, stride=2,
                    track_running_stats=track_running_stats
                ),
                repeat_layers(
                    block(
                        n_in=4 * n_filters, n_out=4 * n_filters, stride=1,
                        track_running_stats=track_running_stats
                    ),
                    n_blocks[2] - 1
                ),
                self.out_layer
            )

    def forward(self, x):
        x = self.resnet(x)
        return x
