import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable


def get_features_vgg19(image, model_feature, layers=None):
    if layers is None:
        layers = {'2': 'conv1_2',
                  '7': 'conv2_2',
                  '16': 'conv3_4',
                  '25': 'conv4_4'
                 }
    features = {}
    x = image
    for idx, layer in enumerate(model_feature):
        x = layer(x)
        if str(idx) in layers:
            features[layers[str(idx)]] = x
    return features

def vgg_loss(outputs, label, vgg19, model_feature=[], device='cuda'):
    #global vgg19
    if not torch.is_tensor(outputs):
        out = torch.tensor(outputs)
    else:
        out = outputs.clone().detach()
    if not torch.is_tensor(label):
        lab = torch.tensor(label).detach()
    else:
        lab = label.clone()
    lab_max = torch.max(lab)
    out = out / lab_max
    lab = lab / lab_max
    if out.shape[1] == 1:
        out = out.repeat(1,3,1,1)
    if lab.shape[1] == 1:
        lab = lab.repeat(1,3,1,1)
    out = out.to(device)
    lab = lab.to(device)

    feature_out1 = 0.5*get_features_vgg19(out, vgg19, {'2': 'conv1_2'})['conv1_2']
    feature_out2 = 0.5*get_features_vgg19(out, vgg19, {'25': 'conv4_4'})['conv4_4']
    feature_lab1 = 0.5*get_features_vgg19(lab, vgg19, {'2': 'conv1_2'})['conv1_2']
    feature_lab2 = 0.5*get_features_vgg19(lab, vgg19, {'25': 'conv4_4'})['conv4_4']
    feature_loss = nn.MSELoss()(feature_out1, feature_lab1) + nn.MSELoss()(feature_out2, feature_lab2)
    return feature_loss

def l1_loss(inputs, targets):
    loss = nn.L1Loss()
    output = loss(inputs, targets)
    return output

def tv_loss(c):
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss


def tv_loss_norm(c):
    n = torch.numel(c)
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    loss = loss / n
    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim_loss(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)