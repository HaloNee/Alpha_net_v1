import torch
from torch import nn

from torchinfo import summary

from self_defined_layers import *


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class AlphaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ts_corr = Lambda(ts_corr)
        self.ts_zscore = Lambda(ts_zscore)
        self.ts_mean = Lambda(ts_mean)
        self.ts_decaylinear = Lambda(ts_decaylinear)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 10), stride=(1,10))

        self.Flatten_1 = nn.Flatten(start_dim=1, end_dim=2)
        self.Flatten_2 = nn.Flatten(start_dim=1, end_dim=2)

        self.BN2d_1 = nn.BatchNorm2d(num_features=1, affine=True, track_running_stats=True)
        self.BN2d_4 = nn.BatchNorm2d(num_features=1, affine=True, track_running_stats=True)

        self.BN1d_1 = nn.BatchNorm1d(num_features=28, affine=True, track_running_stats=True)
        self.BN1d_4 = nn.BatchNorm1d(num_features=8, affine=True, track_running_stats=True)
        self.BN1d_6 = nn.BatchNorm1d(num_features=8, affine=True, track_running_stats=True)
        self.BN1d_9 = nn.BatchNorm1d(num_features=32, affine=True, track_running_stats=True)

        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.min_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.max_pool = nn.MaxPool1d(kernel_size=(3,), stride=(3,))
        self.avg_pool = nn.AvgPool1d(kernel_size=(3,), stride=(3,))
        self.min_pool = nn.MaxPool1d(kernel_size=(3,), stride=(3,))

        self.BN2d_max = nn.BatchNorm2d(num_features=1, affine=True, track_running_stats=True)
        self.BN2d_avg = nn.BatchNorm2d(num_features=1, affine=True, track_running_stats=True)
        self.BN2d_min = nn.BatchNorm2d(num_features=1, affine=True, track_running_stats=True)

        self.BN1d_max = nn.BatchNorm1d(num_features=76, affine=True, track_running_stats=True)
        self.BN1d_avg = nn.BatchNorm1d(num_features=76, affine=True, track_running_stats=True)
        self.BN1d_min = nn.BatchNorm1d(num_features=76, affine=True, track_running_stats=True)

        self.Flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.linear1 = nn.Linear(in_features=int((8 * 7 / 2 + 8 + 8 + 4 * 8) * 6 + (8 * 7 / 2 + 8 + 8 + 4 * 8) * 2), out_features=30)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(in_features=30, out_features=1)

        self.ts_corr_stack = nn.Sequential(
            nn.BatchNorm2d(num_features=1, affine=True, track_running_stats=True),
            nn.Flatten(start_dim=1, end_dim=2),
        )
        self.ts_zscore_stack = nn.Sequential(
            nn.BatchNorm2d(num_features=1, affine=True, track_running_stats=True),
            nn.Flatten(start_dim=1, end_dim=2),
        )

    def forward(self, xb):
        
        # # features catching layer
        xb_1 = self.Flatten_1(self.ts_corr(xb))
        xb_4 = self.Flatten_1(self.ts_zscore(xb))
        xb_6 = self.Flatten_1(self.ts_decaylinear(xb))
        xb_9 = self.Flatten_1(self.conv2d(xb))

        # normalization Layers
        # xb_1 = self.BN2d_1(self.ts_corr(xb))
        # xb_4 = self.BN2d_4(self.ts_zscore(xb))

        xb_1 = self.BN1d_1(xb_1)
        xb_4 = self.BN1d_4(xb_4)
        xb_6 = self.BN1d_6(xb_6)
        xb_9 = self.BN1d_9(xb_9)

        # xb = torch.cat([xb_1, xb_4], dim=2)
        xb = torch.cat([xb_1, xb_4, xb_6, xb_9], dim=1)

        # pooling and batch normalization layer
        # xb_max = self.BN2d_max(self.max_pool(xb))
        # xb_avg = self.BN2d_avg(self.avg_pool(xb))
        # xb_min = self.BN2d_min(-self.min_pool(-xb))

        xb_max = self.BN1d_max(self.max_pool(xb))
        xb_avg = self.BN1d_avg(self.avg_pool(xb))
        xb_min = self.BN1d_min(-self.min_pool(-xb))

        # flatten layer
        # xb = torch.cat([self.Flatten(xb), self.Flatten(xb_max),], dim=1)
        xb = torch.cat([self.Flatten_1(xb), self.Flatten_1(xb_max),], dim=1)

        xb = self.linear1(xb)
        xb = self.relu(xb)
        xb = self.linear2(xb)

        return xb.view(-1)

# """

if __name__ == "__main__":

    torch.random.manual_seed(7)
    input_sample = torch.rand(8, 1, 8, 60)
    print(input_sample.shape)
    alphanet = AlphaNet()

    x_1 = alphanet.BN1d_1(alphanet.Flatten_1(alphanet.ts_corr(input_sample)))
    x_4 = alphanet.BN1d_4(alphanet.Flatten_1(alphanet.ts_zscore(input_sample)))
    x_6 = alphanet.BN1d_6(alphanet.Flatten_1(alphanet.ts_decaylinear(input_sample)))
    x_9 = alphanet.BN1d_9(alphanet.Flatten_1(alphanet.conv2d(input_sample)))
    print(x_1.shape, x_4.shape, x_6.shape, x_9.shape)
    x = torch.cat([x_1, x_4, x_6, x_9], dim=1)
    x_max = alphanet.BN1d_max(alphanet.max_pool(x))
    x_avg = alphanet.BN1d_avg(alphanet.avg_pool(x))
    x_min = alphanet.BN1d_min(alphanet.avg_pool(x))
    x = torch.cat([alphanet.Flatten_1(x), alphanet.Flatten_1(x_max),], dim=1)
    print(x.shape)

    x = alphanet.linear1(x)
    print(x.shape)

    x = alphanet.linear2(x)
    print(x.shape)

    print("alphanet:", alphanet)

    for name, param in alphanet.named_parameters():
        print(f"Layer: {name} |Values: {param} | Size: {param.size()}\n")

    res = alphanet(input_sample)
    print(res)

    summary(alphanet, (128, 1, 8, 60))

# """
