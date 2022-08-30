import numpy as np

import torch
import torch.nn.functional as f
from audtorch.metrics.functional import pearsonr


# Hyper parameters
KERNEL_LENGTH_EXTRACT_LAYER = 10
STRIDE_LENGTH_EXTRACT_LAYER = 10
KERNEL_LENGTH_POOLING_LAYER = 3
STRIDE_LENGTH_POOLING_LAYER = 3


def get_corr(X, Y):
    corr = pearsonr(X, Y)
    return corr


def ts_corr_single(x: torch.Tensor, kernel_length, stride_length):
    assert x.dim() == 4
    medium = torch.concat(
        [f.unfold(x.float(), kernel_size=(2, kernel_length), stride=(
            1, stride_length), dilation=(i, 1)) for i in range(1, x.size()[2])], dim=2)
    medium = medium.permute(0, 2, 1).squeeze(0)
    rows = x.size()[2]
    rows = rows * (rows - 1) / 2
    result = torch.concat([get_corr(medium[i][:kernel_length], medium[i][kernel_length:]).unsqueeze(0)
                           for i in range(medium.size()[0])], 0)
    return result.view(1, 1, int(rows), -1)


def ts_corr(x, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim() == 4
    result = torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_corr_single(
            x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    result = torch.where(torch.isnan(result), torch.full_like(result, 0), result)

    return result


def get_cov(X, Y):
    assert len(X) == len(Y)
    cov = torch.sum((X - X.mean()) * (Y - Y.mean())) / len(X)
    return cov

def ts_cov_single(x, kernel_length, stride_length):
    assert x.dim() == 4
    medium = torch.concat([f.unfold(x.float(), kernel_size=(2, kernel_length), stride=(
        1, stride_length), dilation=(i, 1)) for i in range(1, x.size()[2])], dim=2)
    medium = medium.permute(0, 2, 1).squeeze(0)
    rows = x.size()[2]
    rows = rows * (rows - 1) / 2
    result = torch.concat([get_cov(medium[i][:kernel_length], medium[i]
                                   [kernel_length:]).unsqueeze(0) for i in range(medium.size()[0])], 0)
    return result.view(1, 1, int(rows), -1)


def ts_cov(x, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim() == 4
    result = torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_cov_single(
            x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    return result


def ts_stddev_single(x: torch.Tensor, kernel_length, stride_length):
    assert x.dim() == 4
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    medium = medium.permute(0, 2, 1).squeeze(0)
    result = torch.Tensor()
    for i in range(medium.size()[0]):
        result = torch.concat([result, medium[i].std(unbiased=False).unsqueeze(0)], dim=0)
    assert result.size()[0] == medium.size()[0]

    return result.view(1, 1, x.size()[2], -1)


def ts_stddev(x: torch.tensor, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim() == 4
    result = torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_stddev_single(x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    return result

def ts_zscore_single(x: torch.tensor, kernel_length, stride_length):
    assert x.dim() == 4
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    medium = medium.permute(0, 2, 1).squeeze(0)
    result = torch.Tensor()
    for i in range(medium.size()[0]):
        result = torch.concat([result, medium[i].mean() / (medium[i].std(unbiased=False)+1e-5).unsqueeze(0)], dim=0)
    assert result.size()[0] == medium.size()[0]
    
    result = torch.where(torch.isnan(result), torch.full_like(result, 0), result)
    return result.view(1, 1, x.size()[2], -1)


def ts_zscore(x: torch.Tensor, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim() == 4
    result = torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_zscore_single(x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)
    result = torch.where(torch.isnan(result), torch.full_like(result, 1), result)
    return result


def ts_decaylinear_single(x:torch.tensor, kernel_length, stride_length):
    assert x.dim() == 4
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    medium = medium.permute(0, 2, 1).squeeze(0)
    result = torch.Tensor()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weight = torch.arange(1, kernel_length + 1, 1).to(dev) / torch.sum(torch.arange(1, kernel_length + 1, 1).to(dev))
    for i in range(medium.size()[0]):
        result = torch.concat([result, torch.sum(torch.mul(medium[i], weight)).unsqueeze(0)], dim=0)
    assert result.size()[0] == medium.size()[0]

    return result.view(1, 1, x.size()[2], -1)



def ts_decaylinear(x:torch.Tensor, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim() == 4
    result = torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_decaylinear_single(x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    return result

def ts_min_single(x:torch.Tensor, kernel_length, stride_length):
    assert x.dim() == 4
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    medium = medium.permute(0, 2, 1).squeeze(0)
    result = torch.Tensor()
    for i in range(medium.size()[0]):
        result = torch.concat([result, torch.min(medium[i]).unsqueeze(0)], dim=0)
    assert result.size()[0] == medium.size()[0]

    return result.view(1, 1, x.size()[2], -1)

def ts_min(x:torch.Tensor, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim() == 4
    result= torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_min_single(x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    return result

def ts_max_single(x:torch.Tensor, kernel_length, stride_length):
    assert x.dim() == 4
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    medium = medium.permute(0, 2, 1).squeeze(0)
    result = torch.Tensor()
    for i in range(medium.size()[0]):
        result = torch.concat([result, torch.max(medium[i]).unsqueeze(0)], dim=0)
    assert result.size()[0] == medium.size()[0]

    return result.view(1, 1, x.size()[2], -1)


def ts_max(x:torch.Tensor, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim() == 4
    result= torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_max_single(x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    return result

def ts_sum_single(x:torch.Tensor, kernel_length, stride_length):
    assert x.dim() == 4
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    medium = medium.permute(0, 2, 1).squeeze(0)
    result = torch.Tensor()
    for i in range(medium.size()[0]):
        result = torch.concat([result, torch.sum(medium[i]).unsqueeze(0)], dim=0)
    assert result.size()[0] == medium.size()[0]

    return result.view(1, 1, x.size()[2], -1)


def ts_sum(x:torch.Tensor, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim() == 4
    result= torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_sum_single(x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    return result


def ts_mean_single(x:torch.Tensor, kernel_length, stride_length):
    assert x.dim() == 4
    medium = f.unfold(x.float(), kernel_size=(1, kernel_length), stride=(1, stride_length))
    medium = medium.permute(0, 2, 1).squeeze(0)
    result = torch.Tensor()
    for i in range(medium.size()[0]):
        result = torch.concat([result, torch.mean(medium[i]).unsqueeze(0)], dim=0)
    assert result.size()[0] == medium.size()[0]

    return result.view(1, 1, x.size()[2], -1)


def ts_mean(x:torch.Tensor, kernel_length=KERNEL_LENGTH_POOLING_LAYER, stride_length=STRIDE_LENGTH_POOLING_LAYER):
    assert x.dim() == 4
    result= torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_mean_single(x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    return result


def ts_vol_diff_single(x:torch.Tensor, kernel_length, stride_length):
    assert x.dim() == 4
    temp = x[:, :, 6, :]

def ts_vol_diff(x:torch.Tensor, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim() == 4
    result = torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_vol_diff_single(x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    return result


def ts_imbalance_single(X:torch.Tensor, kernel_length, stride_length):
    assert x.dim == 4


def ts_imbalance(x:torch.Tensor, kernel_length=KERNEL_LENGTH_EXTRACT_LAYER, stride_length=STRIDE_LENGTH_EXTRACT_LAYER):
    assert x.dim == 4
    result = torch.Tensor()
    for i in range(x.size()[0]):
        result = torch.concat([result, ts_imbalance_single(x[i].unsqueeze(0), kernel_length, stride_length)], dim=0)

    return result    



# just for test
if __name__ == '__main__':

    torch.random.manual_seed(7)
    sample_input = torch.arange(0, 360,).view(2, 1, 3, -1)
    sample_input = torch.rand(2, 1, 3, 60)
    sample_input = torch.ones(2, 1, 3, 60)

    sample_output = ts_cov(sample_input)
    print(f"sample_input {sample_output.shape} :\n", sample_output)

    x = sample_input[0, 0, 0, :10]
    y = sample_input[0, 0, 1, :10]
    x, y = x.numpy(), y.numpy()

    corr_output = ts_corr(sample_input)
    print(f"corr_output {corr_output.size()}:\n", corr_output)
    print(np.std(sample_input[1, 0, 2, 10:20].numpy()))

    
    std_output = ts_stddev(sample_input)
    print(f"std_output {std_output.size()}:\n", std_output)
    # print(np.corr(sample_input[1, 0, 2, 10:20].numpy()))

    zscore_output = ts_zscore(sample_input)
    print(f"zscore_output {zscore_output.size()}:\n", zscore_output)
    print(np.mean(sample_input[1, 0, 2, 10:20].numpy()) / (np.std(sample_input[1, 0, 2, 10:20].numpy()+1e-5)))

    decaylinear_output = ts_decaylinear(sample_input)
    print(f"decaylinear_output {decaylinear_output.size()}:\n", decaylinear_output)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    
    weight = torch.arange(1, 10 + 1, 1).to(dev) / torch.sum(torch.arange(1, 10 + 1, 1).to(dev))
    weight = weight.numpy()
    res = np.sum(y * weight)
    print(res)

    min_output = ts_min(sample_input)
    print(f"min_output {min_output.size()}:\n", min_output)
    res = np.min(y)
    print(res)

    max_output = ts_max(sample_input)
    print(f"max_output {max_output.size()}:\n", max_output)
    res = np.max(y)
    print(res)

    sum_output = ts_sum(sample_input)
    print(f"sum_output {sum_output.size()}:\n", sum_output)
    res = np.sum(y)
    print(res)

    mean_output = ts_mean(sample_input)
    print(f"mean_output {mean_output.size()}:\n", mean_output)
    res = np.mean(y)
    print(res)
