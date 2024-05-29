import torch as t
from torch import nn
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear

class BlockGroup(nn.Module):
    def __init__(self, n_blocks, in_feats, out_feats, stride):
        super().__init__()
        self.blocks = Sequential(
            *[self._make_block(in_feats if i == 0 else out_feats, out_feats, stride if i == 0 else 1) for i in range(n_blocks)]
        )

    def _make_block(self, in_feats, out_feats, stride):
        return Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats),
        )

    def forward(self, x):
        return self.blocks(x)

class AveragePool(nn.Module):
    def forward(self, x):
        return nn.functional.avg_pool2d(x, x.shape[2:])

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ResNet(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        strides_per_group=[1, 2, 2, 2],
        n_classes=10,
    ):
        super().__init__()
        in_feats0 = 64

        self.in_layers = Sequential(
            Conv2d(3, in_feats0, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(in_feats0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        all_in_feats = [in_feats0] + out_features_per_group[:-1]
        self.residual_layers = Sequential(
            *(
                BlockGroup(*args)
                for args in zip(
                    n_blocks_per_group,
                    all_in_feats,
                    out_features_per_group,
                    strides_per_group,
                )
            )
        )
        self.out_layers = Sequential(
            AveragePool(),
            Flatten(),
            Linear(512, n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.in_layers(x)
        x = self.residual_layers(x)
        x = self.out_layers(x)
        return x