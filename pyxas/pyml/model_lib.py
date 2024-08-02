import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class ResidualDenseBlock_3C(nn.Module):
    def __init__(self, nf=32, gc=16, bias=True, kernel_size=3):
        super(ResidualDenseBlock_3C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        #self.conv11 = nn.Conv2d(nf, int(gc/2), 3, stride=1, dilation=1, padding='same', bias=bias)
        self.conv1 = nn.Conv2d(nf, gc, kernel_size, stride=1, padding='same', bias=bias, padding_mode='replicate')
        self.conv2 = nn.Conv2d(nf + gc, gc, kernel_size, stride=1, padding='same', bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, kernel_size, 1, 'same', bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        s = {}
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        return x3 * 0.4 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, kernel_size=3):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_3C(nf, gc, kernel_size=kernel_size)
        self.RDB2 = ResidualDenseBlock_3C(nf, gc, kernel_size=kernel_size)
        self.RDB3 = ResidualDenseBlock_3C(nf, gc, kernel_size=kernel_size)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, padding_mode='zeros', kernel_size=3):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc, kernel_size=kernel_size)

        self.conv_first_1 = nn.Conv2d(in_nc, int(nf/2), kernel_size, 1, 1, padding_mode=padding_mode, bias=True)
        self.conv_first_2 = nn.Conv2d(in_nc, int(nf/2), kernel_size, stride=1, dilation=1, padding='same', padding_mode=padding_mode, bias=True)
        self.conv_first = nn.Conv2d(in_nc, nf, kernel_size, 1, 1, padding_mode=padding_mode, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size, 1, 1,  padding_mode=padding_mode, bias=True)
        #### upsampling
        #self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, kernel_size, 1, 1, padding_mode=padding_mode, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, kernel_size, 1, 1, padding_mode=padding_mode, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        #x1 = self.conv_first_1(x)
        #x2 = self.conv_first_2(x)
        #fea = torch.cat((x1, x2), 1)
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk # this is orignial one used
        
        #fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea))) # the following is added


        return out


class RRDBNet_padding_same(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, padding_mode='zeros', kernel_size=3):
        super(RRDBNet_padding_same, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc, kernel_size=kernel_size)

        self.conv_first_1 = nn.Conv2d(in_nc, int(nf / 2), kernel_size, 1, 'same', padding_mode=padding_mode, bias=True)
        self.conv_first_2 = nn.Conv2d(in_nc, int(nf / 2), kernel_size, stride=1, dilation=1, padding='same',
                                      padding_mode=padding_mode, bias=True)
        self.conv_first = nn.Conv2d(in_nc, nf, kernel_size, 1, 'same', padding_mode=padding_mode, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size, 1, 'same', padding_mode=padding_mode, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, kernel_size, 1, 'same', padding_mode=padding_mode, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, kernel_size, 1, 'same', padding_mode=padding_mode, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk  # this is orignial one used
        out = self.conv_last(self.lrelu(self.HRconv(fea)))  # the following is added

        return out

class RRDBNet_new(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet_new, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first_1 = nn.Conv2d(in_nc, int(nf/2), 3, 1, 1, padding_mode='replicate', bias=True)
        self.conv_first_2 = nn.Conv2d(in_nc, int(nf/2), 3, stride=1, dilation=1, padding='same', padding_mode='replicate', bias=True)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, padding_mode='replicate', bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, padding_mode='replicate', bias=True)
        #### upsampling
        #self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, padding_mode='replicate', bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, padding_mode='replicate', bias=True)
        self.conv_last2 = nn.Conv2d(out_nc, out_nc, 3, 1, 1, padding_mode='replicate', bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        #x1 = self.conv_first_1(x)
        #x2 = self.conv_first_2(x)
        #fea = torch.cat((x1, x2), 1)
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        # fea = fea + trunk # this is orignial one used
        fea = 0.1 * fea + 0.9 * trunk

        #fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea))) # the following is added


        return out



class ResidualDenseBlock_3C_d3(nn.Module):
    '''
    use dilation = 3
    '''
    def __init__(self, nf=32, gc=16, bias=True):
        super(ResidualDenseBlock_3C_d3, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        #self.conv11 = nn.Conv2d(nf, int(gc/2), 3, stride=1, dilation=1, padding='same', bias=bias)
        self.conv1 = nn.Conv2d(nf, gc, 3, stride=1, dilation=3, padding='same', bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, stride=1, dilation=3, padding='same', bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, 3, 1, dilation=3, padding='same', bias=bias)
        #self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        #self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        s = {}
        #x11 = self.lrelu(self.conv11(x))
        #x12 = self.lrelu(self.conv12(x))
        #x1 = torch.cat((x11, x12), 1)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        #x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        #x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x3 * 0.4 + x

class RRDB_d3(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB_d3, self).__init__()
        self.RDB1 = ResidualDenseBlock_3C_d3(nf, gc)
        self.RDB2 = ResidualDenseBlock_3C_d3(nf, gc)
        self.RDB3 = ResidualDenseBlock_3C_d3(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet_d3(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet_d3, self).__init__()
        RRDB_block_f = functools.partial(RRDB_d3, nf=nf, gc=gc)

        self.conv_first_1 = nn.Conv2d(in_nc, int(nf/2), 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(in_nc, int(nf/2), 3, stride=1, dilation=1, padding='same', bias=True)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        #self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(out_nc, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        #x1 = self.conv_first_1(x)
        #x2 = self.conv_first_2(x)
        #fea = torch.cat((x1, x2), 1)
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk # this is orignial one used
        
        #fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea))) # the following is added


        return out

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        #layers.append(nn.ReLU(inplace=True))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            #layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return x-out


def default_model():
    model = RRDBNet(1, 1, 16, 4, 32)
    return model

def default_model_path():
    ml_path = __file__
    ml_path = '/'.join(ml_path.split('/')[:-1])
    ml_path = ml_path + '/trained_model/pre_traind_model_xanes_denoise.pth'
    #print(ml_path)
    return ml_path

def load_default_model(device='cpu'):
    model = default_model()
    model_path = default_model_path()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    return model
