import pyxas
import skimage
import numpy as np
from skimage.transform import rescale
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import glob
from skimage import io
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d, UnivariateSpline
from scipy import ndimage
import tomopy
from skimage.transform import radon, iradon, iradon_sart
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt

tv_loss = pyxas.tv_loss

def plot_ref(*ref):
    n = len(ref)
    plt.figure()
    for i in range(n):
        r = ref[i]
        plt.plot(r[:, 0], r[:, 1], label=f'ref: {i}')
    plt.legend()

def shift_reference_energy(ref, offset):
    '''
    ref.shape = (n_eng, 2)
    '''
    ref_s = ref.copy()
    ref_s[:, 0] += offset
    return ref_s

def rot3D(img, rot_angle):
    img = np.array(img)
    s = img.shape
    if len(s) == 2:    # 2D image
        img_rot = ndimage.rotate(img, rot_angle, order=1, reshape=False)
    elif len(s) == 3:  # 3D image, rotating along axes=0
        img_rot = ndimage.rotate(img, rot_angle, axes=[1,2], order=1, reshape=False)
    elif len(s) == 4:  # a set of 3D image
        img_rot = np.zeros(img.shape)
        for i in range(s[0]):
            img_rot[i] = ndimage.rotate(img[i], rot_angle, axes=[1,2], order=1, reshape=False)
    else:
        raise ValueError('Error! Input image has dimension > 4')
    img_rot[img_rot < 0] = 0
    return img_rot

def show_tensor(data, clim=None):
    try:
        t = data.squeeze().cpu().numpy()
    except:
        t = data.squeeze()
    plt.figure()
    if clim is None:
        plt.imshow(t)
    else:
        try:
            plt.imshow(t, clim=clim)
        except:
            plt.imshow(t)
    plt.colorbar()

def shuffle_angle_spec(angle_l, eng, method, ref):

    '''
    method = 1: sequence 
    method = 2: equal space
    method = 3: shuffle
    '''

    n_eng = len(eng)
    n_angle = len(angle_l)
    n_ref = len(ref)

    eng_shuffle = np.zeros(n_angle)

    r = {}
    r_shuffle = {}
    for i in range(n_ref):
        r[i] = np.zeros(n_eng)
        r_shuffle[i] = []

    x = ref[0][:, 0]
    for i in range(n_eng):
        idx = pyxas.find_nearest(x, eng[i])
        for j in range(n_ref):
            r[j][i] = ref[j][idx, 1]

    # shuffle to get angle-energy
    n_div = n_angle // n_eng

    for j in range(n_ref):
        tmp = list(r[j]) * (n_div + 1)
        r_shuffle[j] = tmp[:n_angle]

    if method == 1: # in squence: e.g., [8.1, 8.1, 8.1, 8.2, 8.2, 8.2, 8.3, 8.3, 8.3]
        r_shuffle_copy = r_shuffle.copy()
        for j in range(n_ref):
            tmp = []
            for i in range(n_eng):
                t = np.where(r_shuffle_copy[j] == r_shuffle_copy[j][i])[0]
                tmp = tmp + [r_shuffle_copy[j][i]] * len(t)
            r_shuffle[j] = tmp

    if method == 2: # equal space: e.g., [8.1, 8.2, 8.3, 8.1, 8.2, 8.3 ...]
        r_shuffle = r_shuffle

    if method == 3: # random sequence
        for j in range(n_ref):
            tmp = list(r[j]) * (n_div + 1)
            r_shuffle[j] = tmp[:n_angle]

        idx = np.int16(np.arange(n_angle))
        np.random.shuffle(idx)
        for j in range(n_ref):
            old = r_shuffle[j].copy()
            for i in range(len(idx)):
                r_shuffle[j][i] = old[idx[i]]

    for j in range(n_ref):
        r_shuffle[j] = np.array(r_shuffle[j])

    for i in range(n_angle):
        idx = pyxas.find_nearest(ref[0][:, 1], r_shuffle[0][i])
        idy = pyxas.find_nearest(eng, ref[0][idx,0])
        eng_shuffle[i] = eng[idy]
    return r_shuffle, eng_shuffle



def shuffle_angle_spec_general(angle_l, eng, method, ref):

    '''
    method = 1: sequence 
    method = 2: equal space
    method = 3: shuffle
    '''
    r_shuffle, eng_shuffle = shuffle_angle_spec(angle_l, eng, method, ref)
    n_ref = len(r_shuffle)
    n_eng = len(eng_shuffle)

    r_shuffle_comb = np.zeros((n_ref, n_eng)) # (4, 60)
    for i in range(n_ref):
        r_shuffle_comb[i] = r_shuffle[i]
    r_shuffle_comb = r_shuffle_comb.T # convert to (60, 4)
    return r_shuffle_comb, eng_shuffle

def interp_single_spec(ref_spec, xanes_eng):
    f = InterpolatedUnivariateSpline(ref_spec[:, 0], ref_spec[:, 1], k=3)
    ref_spec_interp = f(xanes_eng)
    return ref_spec_interp

def line_interp(x, y, x_interp, k=1):
    f = InterpolatedUnivariateSpline(x, y, k=k)
    return f(x_interp)

def interp_ref_spec_general(eng, ref_comb):
    '''
    ref_comb = (ref1, ref2, ...)
    ref1 = (n_eng, 2)
    '''
    n_ref = len(ref_comb)
    n_eng = len(eng)
    ref_interp = np.array(np.zeros((n_eng, n_ref)))
    for i in range(n_ref):
        spec = interp_single_spec(ref_comb[i], eng)
        ref_interp[:, i] = spec
    return ref_interp


def unique_array(a):
    b = list(dict.fromkeys(a))
    return b

def get_bkg_avg(img, it=3):
    m = pyxas.otsu_mask(img.squeeze(), 3)
    m = 1 - m
    m = np.int16(m)
    struct = ndimage.generate_binary_structure(2, 1)
    struct1 = ndimage.iterate_structure(struct, it).astype(int)
    m = ndimage.binary_erosion(m, structure=struct1)
    bkg_sum = np.sum(img * m)
    bkg_avg = bkg_sum / np.sum(m)
    return bkg_avg


def rm_bkg_avg(img, it=3):
    img = img.squeeze()
    bkg_avg = get_bkg_avg(img, it)
    img_bkg_rm = img - bkg_avg
    return img_bkg_rm


def rm_bkg_avg_stack(img_stack, it=3):
    img_rm = img_stack.copy()
    for i in range(len(img_stack)):
        img_rm[i] = rm_bkg_avg(img_stack[i], it)
    return img_rm


def norm_intensity_spec_tomo(sino_Ni, eng, plot_fig=False):
    eng_u = np.sort(list(set(eng)))
    n = len(eng_u)
    sino_norm = np.zeros_like(sino_Ni)
    for i in range(n):
        idx = np.where(eng == eng_u[i])[0]
        sino_sub = sino_Ni[idx]
        sino_sub_sum = np.sum(sino_sub, axis=(1, 2))
        val_max = np.max(sino_sub_sum)
        n_sub = len(idx)
        for j in range(n_sub):
            sino_sub[j] = sino_sub[j] / sino_sub_sum[j] * val_max
        sino_norm[idx] = sino_sub
    if plot_fig:
        sum_sino = np.sum(sino_norm, axis=(1, 2))
        plt.figure()
        plt.plot(eng, sum_sino, '.')
    return sino_norm


def apply_shift(img_stack, r_shift, c_shift):
    n = len(img_stack)
    img2 = np.zeros(img_stack.shape)
    for i in trange(n):
        r = r_shift[i]
        c = c_shift[i]
        img2[i] = shift(img_stack[i], [r, c], mode='constant', cval=0)
    return img2


def circ_mask(img, axis, ratio=1, val=0):
    im = np.float32(img)
    s = im.shape
    if len(s) == 2:
        m = _get_mask(s[0], s[1], ratio)
        m_out = (1 - m) * val
        im_m = np.array(m, dtype=np.int) * im + m_out
    else:
        im = im.swapaxes(0, axis)
        dx, dy, dz = im.shape
        m = _get_mask(dy, dz, ratio)
        m_out = (1 - m) * val
        im_m = np.array(m, dtype=np.int16) * im + m_out
        im_m = im_m.swapaxes(0, axis)
    return im_m

"""
def tv_loss(c):
    n = torch.numel(c)
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    loss = loss / n
    return loss

def l1_loss(inputs, targets):
    loss = torch.nn.L1Loss()
    output = loss(inputs, targets)
    return output
"""

def poisson_likelihood_loss(inputs, targets):
    
    # loss = inputs - targets + targets * log(targets/inputs)
    n = torch.numel(inputs)
    id_inputs = inputs > 0
    id_targets = targets > 0
    idx = id_inputs * id_targets
    loss = inputs[idx] - targets[idx] + targets[idx] * torch.log(targets[idx]/inputs[idx])
    loss = torch.sum(loss) / n
    return loss
"""
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
"""

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def rot_img(img, theta, device='cpu', dtype=torch.float32):
    '''
    img.shape = (1, 1, 128, 128) or (n_ref, 1, 128, 128)
    '''
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(img.shape[0],1,1).to(device)
    grid = F.affine_grid(rot_mat, img.size(), align_corners=False).type(dtype).to(device)
    img_r = F.grid_sample(img, grid, align_corners=False)
    return img_r

def rot_img_general(img, theta, device='cpu', dtype=torch.float32):
    '''
    img.shape = (4, 1, 128, 128)
    '''
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(img.shape[0],1,1).to(device)
    grid = F.affine_grid(rot_mat, img.size(), align_corners=False).type(dtype).to(device)
    img_r = F.grid_sample(img, grid, align_corners=False)
    return img_r


def torch_sino(img2D, angle_l, device='cpu', w=[1], atten2D_at_angle=[1]):
    '''
    atten2D_at_angle: XRF attenuation coefficient at each angle, 
    shape = e.g., (n_angle, 1, 128, 128)
    '''
    n_angle = len(angle_l)
    theta_l = angle_l / 180 * np.pi
    w = torch.tensor(w)
    s0 = img2D.shape # (1, 1, 128, 128) 
    if not len(w) == n_angle:
        w = torch.ones((n_angle, s0[0])).to(device)
    if not len(atten2D_at_angle) == n_angle:
        atten2D_at_angle = torch.ones(n_angle)
    sino = torch.zeros((n_angle, *s0[:-2], s0[-1])).to(device) # (n_angle, n, 1, 128)
    for i in range(n_angle):    
        t = rot_img(img2D, theta_l[i], device)
        t = t * atten2D_at_angle[i]
        t_sum = torch.sum(t, axis=-2)
        sino[i] = t_sum * w[i]
    return sino


def torch_sino_general(img4d, angle_l,  w=[1], atten2D_at_angle=[1], device='cpu', cal_pix_atten=False):
    
    '''
    atten2D_at_angle: XRF attenuation coefficient at each angle, 
    shape = e.g., (n_angle, 1, 256, 256)
    '''
    s0 = img4d.shape # (4, 1, 256, 256)
    nr = s0[-2]  # num of rows
    n_ref = s0[0]
    n_angle = len(angle_l)
    theta_l = angle_l / 180 * np.pi
    w = torch.tensor(w)
    
    if w.shape != (n_angle, n_ref):
        w = torch.ones((n_angle, s0[0])).to(device) # (n_angle, 4)

    if not len(atten2D_at_angle) == n_angle:
        atten2D_at_angle = torch.ones(n_angle, 1, s0[-2], s0[-1])
    atten2D_at_angle = torch.tensor(atten2D_at_angle).to(device)
    if len(atten2D_at_angle.shape) == 3: # e.g., (60, 128, 128)
        atten2D_at_angle = atten2D_at_angle.unsqueeze(1) # (60, 1, 128, 128)

    sino = torch.zeros((n_angle, *s0[:-2], s0[-1])).to(device) # (n_angle, 4, 1, 256)

    a = torch.zeros(s0).to(device)
    a_sum_accum = torch.zeros(s0).to(device)

    if cal_pix_atten:
        l_atten = cal_pix_atten['l_atten'] # attenunation length
        pix = cal_pix_atten['pix'] # mass density

    for i in range(n_angle):    
        t = rot_img_general(img4d, theta_l[i], device) #(4, 1, 256, 256)
        if cal_pix_atten:
            for j in range(n_ref):
                a[j] = t[j] / l_atten * pix * w[i, j] # (4, 1, 256, 256),
                # 0.00127: (cm) gives absorbity = 1 --> attenunation length
                # 5e-6 (cm): pixel size in this case
            a_sum = torch.sum(a, axis=0, keepdim=True) # (1, 1, 256, 256)

            for k in range(nr-2, 0, -1): # sum in row
                a_sum_accum[:, :, k] = torch.sum(a_sum[:, :, k:nr], axis=-2, keepdim=True)
            at = torch.exp(-a_sum_accum)

        t = t * atten2D_at_angle[i]
        if cal_pix_atten:
            t = t * at
        t_sum = torch.sum(t, axis=-2) # (4, 1, 256)
        for j in range(n_ref):
            sino[i, j] = t_sum[j] * w[i, j]
    return sino



def torch_sino_xrf_atten(img4d, angle_l, f_scale, w=[1], atten2D_at_angle=[1], device='cpu'):
    '''
    w: weighted function, usually the refernece spectrum
    atten2D_at_angle: fluorescence attenuantion coefficient
    #rho_sli  (1, 1, 256, 256)
    '''

    s0 = img4d.shape  # (4, 1, 256, 256)
    n_ref = s0[0]
    n_angle = len(angle_l)
    theta_l = angle_l / 180 * np.pi
    theta_l = torch.tensor(theta_l)

    w = torch.tensor(w)
    if w.shape != (n_angle, n_ref):
        w = torch.ones((n_angle, s0[0])).to(device)  # (n_angle, 4)
    w = w.to(device)

    base = torch.ones(s0[1:]).to(device)
    for i in range(s0[2]):
        base[:, i] = s0[2] - i - 1

    sino_sum = torch.zeros((n_angle, 1, s0[1], s0[-1])).to(device) # (n_angle, 1, 256)

    img = img4d * f_scale
    for i in range(n_angle):
        t = rot_img_general(img, theta_l[i], device) #(4, 1, 256, 256)

        # xrf_pix: absorption cross-section at each pixel, need to multiply by (rho x pixel_size)
        for j in range(n_ref):
            t[j] = t[j] * w[i, j]
        xrf_pix = torch.sum(t, axis=0) #(1, 256, 256)

        # x_ray_atten: calculate incident beam intensity at pixel (row)
        x_ray_atten = torch.exp(-200*50e-6 * base) # (1, 256, 256), manually set to cs=200, rho=4.7
        # total attenuation:
        atten_2d = x_ray_atten * atten2D_at_angle[i]  # (1, 256, 256)

        # pixel emission =  proportional to pixel_absorption
        # xrf = I_incident * (1 - exp[-(a1*r1 + a2*r2)]) * cs_emission
        # since been normalized by cs_emission already, and small number of (a1*r1 + a2*r2)
        # xrf_pix ~ I_incident * (a1*r1 + a2*r2)


        xrf_intensity = xrf_pix * atten_2d

        sino_sum[i] = torch.sum(xrf_intensity, axis=1, keepdim=True)
    return sino_sum

    
def plot_loss(h_loss, start=0, plot_log=False, new_fig=False, multiply_rate=False):
    keys = list(h_loss.keys())
    if new_fig:
        plt.figure()
    for k in keys:
        rate = h_loss[k]['rate']
        v = np.array(h_loss[k]['value'][start:])
        if plot_log:
            v = np.log(v)
        if multiply_rate:
            v = v * rate
        plt.plot(v, label=f'{k}: rate={rate}')
    plt.legend()
    plt.title(f'rate multiplied: {multiply_rate}')


def plot_eng_select(angle_l, eng_shuffle):
    plt.figure()
    plt.scatter(angle_l, eng_shuffle)
    plt.xlabel('Angle (degree)')
    plt.ylabel('X-ray energy (keV)')
    plt.title(f'Energy points: {len(eng_shuffle)}')

def plot_res(angle_l, img1, img2, guess1, guess2, sino_out, sino_sum, loss_his, c_range=None, counts=None):
    if c_range is None:
        c_range = [0,1]
    if counts is None:
        counts = 'Noise free'
    plt.figure(figsize=(16, 10))
    plt.subplot(231)
    plt.imshow(guess1, clim=c_range)
    plt.colorbar()
    plt.subplot(232)
    plt.imshow(guess2, clim=c_range)
    plt.colorbar()
    plt.subplot(233)
    keys = list(loss_his.keys())
    for k in keys:
        plt.plot(np.log(loss_his[k]['value']), label=f'{k}: rate={loss_his[k]["rate"]}')
    #plt.yscale('log')    
    plt.xlabel('Iterations')

    plt.legend()
    plt.subplot(234)
    plt.imshow(img1.detach().cpu().numpy().squeeze(), clim=c_range)
    plt.colorbar()
    plt.subplot(235)
    plt.imshow(img2.detach().cpu().numpy().squeeze(), clim=c_range)
    plt.colorbar()
    plt.subplot(236)
    sino_dif = (sino_out - sino_sum.detach().cpu().numpy()).squeeze()
    plt.imshow(sino_dif)
    '''
    img_sum = (guess1.detach() + guess2.detach()).cpu().numpy().squeeze()
    img_dif = img_sum - img1_raw - img2_raw
    plt.imshow(img_dif)
    '''
    plt.colorbar()
    plt.suptitle(f'Noise counts = {counts}   total angles = {len(angle_l)}')


def plot_res_general(img4d, guess, sino_sum_general, sino_out, loss_his, c_range=None, counts=None):
    if c_range is None:
        c_range = [0, np.max(img4d)]
    if counts is None:
        counts = 'Noise free'

    n_angle = sino_out.shape[0]
    keys = list(loss_his.keys())
    fig0, ax0 = plt.subplots(2, 2, figsize=(12, 12))
    for k in keys:
        val = np.array(loss_his[k]['value'])
        rate = loss_his[k]["rate"]
        if loss_his[k]["rate"] > 0:            
            ax0[0, 0].plot(val*rate, label=f'{k}: rate={rate}')
            ax0[0, 0].set_yscale('log')  
    ax0[0, 0].legend()
    A = sino_sum_general.detach().cpu().numpy().squeeze()
    B = sino_out.squeeze()
    im = ax0[0, 1].imshow(B-A)
    ax0[0, 1].set_title(f'difference in sino')
    fig0.colorbar(im, ax=ax0[0, 1])
    im = ax0[1, 0].imshow(A)
    ax0[1, 0].set_title(f'Measured sino')
    fig0.colorbar(im, ax=ax0[1, 0])
    im = ax0[1, 1].imshow(B)
    ax0[1, 1].set_title(f'Reconstructed sino')
    fig0.colorbar(im, ax=ax0[1, 1])

    img4d = img4d.squeeze()
    guess = guess.squeeze()
    s = img4d.shape #(4, 256, 256) or (256, 256)
    if len(s) == 2:
        img4d = img4d[np.newaxis, :]
        guess = guess[np.newaxis, :]
    img_dif = guess - img4d 
    n = img4d.shape[0]
    if n >= 2:
        fig, ax = plt.subplots(3, n, figsize=(18, 12))
        for i in range(n):
            im0 = ax[0, i].imshow(guess[i], clim=c_range)
            ax[0, i].set_title(f'recon ref #{i+1}')
            ax[0, i].axis('off')
            fig.colorbar(im0, ax=ax[0, i])

            im1 = ax[1, i].imshow(img4d[i], clim=c_range)
            ax[1, i].set_title(f'Ground-Truth #{i+1}')
            ax[1, i].axis('off')
            fig.colorbar(im1, ax=ax[1, i])

            im2 = ax[2, i].imshow(img_dif[i])
            ax[2, i].set_title(f'Fit error #{i+1}')
            ax[2, i].axis('off')
            fig.colorbar(im2, ax=ax[2, i])
        fig.suptitle(f'Image-size = {s[-2], s[-1]},   max-counts={counts},   angles={n_angle}')
    elif n == 1:
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        im0 = ax[0].imshow(guess[0], clim=c_range)
        ax[0].set_title(f'recon')
        ax[0].axis('off')
        fig.colorbar(im0, ax=ax[0])

        im1 = ax[1].imshow(img4d[0], clim=c_range)
        ax[1].set_title(f'Ground-truth')
        ax[1].axis('off')
        fig.colorbar(im1, ax=ax[1])

        im2 = ax[2].imshow(img_dif[0])
        ax[2].set_title(f'Fit error')
        ax[2].axis('off')
        fig.colorbar(im2, ax=ax[2])
        fig.suptitle(f'Image-size = {s[-2], s[-1]},   max-counts={counts},   angles={n_angle}')


def plot_comparison(f_save_root):
    '''
    e.g,     
    f_save_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/loss_record/seq_select/iter_4000'
    '''
    fn_rec1 = np.sort(glob.glob(f_save_root + '/rec1*'))
    fn_rec2 = np.sort(glob.glob(f_save_root + '/rec2*'))
    fn_gt1 = f_save_root + '/img_gt_ref1.tiff'
    fn_gt2 = f_save_root + '/img_gt_ref2.tiff'

    img_gt = {}
    img_gt[0] = io.imread(fn_gt1)
    img_gt[1] = io.imread(fn_gt2)

    n = len(fn_rec1)

    img_dif_summary = {0: {}, 1:{}}
    img_summary = {0: {}, 1:{}}
    n_ang_summary = {}
    for i in range(n):
        img_rec = {}
        img_dif = {}
        img_rec[0] = io.imread(fn_rec1[i])
        img_rec[1] = io.imread(fn_rec2[i])
        img_dif[0] = img_rec[0] - img_gt[0]
        img_dif[1] = img_rec[1] - img_gt[1]
        img_dif_summary[0][i] = img_dif[0]
        img_dif_summary[1][i] = img_dif[1]
        img_summary[0][i] = img_rec[0]
        img_summary[1][i] = img_rec[1]
        
        n_ang = fn_rec1[i].split('/')[-1].split('.')[0]
        n_ang = n_ang.split('_')[-1]
        n_ang = int(n_ang)
        n_ang_summary[i] = n_ang
        plt.figure(figsize=(16, 8))
        for j in range(2):
            plt.subplot(2, 3, 1+j*3)
            plt.imshow(img_gt[j], clim=[0,1])
            plt.colorbar()
            plt.title(f'ref #{j+1}: ground-truth')
            plt.subplot(2, 3, 2+j*3)
            plt.imshow(img_rec[j], clim=[0,1])
            plt.colorbar()
            plt.title(f'ref #{j+1}: recovered')
            plt.subplot(2, 3, 3+j*3)
            plt.imshow(img_dif[j], clim=[-0.3, 0.3])
            plt.colorbar()
            plt.title(f'ref #{j+1}: difference')
        plt.suptitle(f'Number of angles = {n_ang}')
        fn_fig = f_save_root + f'/summary_angle_{n_ang}.png'
        plt.savefig(fn_fig)
        plt.pause(1)

    
    for k in range(2): # 2 references
        plt.figure(figsize=(20, 8))
        for i in range(2):
            for j in range(4):
                plt.subplot(2, 4, i*4+j+1)
                plt.imshow(img_dif_summary[k][i*4+j], clim=[-0.3, 0.3])
                plt.colorbar()
                plt.title(f'Angles = {n_ang_summary[i*4+j]}')
        fn_fig_dif = f_save_root + f'/Diff_ref{k+1}_summary.png'
        plt.suptitle(f'Difference for Ref #{k}')
        plt.pause(1)
        plt.savefig(fn_fig_dif)
            
    '''
    plot PSNR  
    '''    
    img_ang = []
    n = len(img_summary[0])
    img_ang = np.zeros(n)
    for i in range(n):
        img_ang[i] = n_ang_summary[i]
    img_psnr = {0: np.zeros(n), 1:np.zeros(n)}
    plt.figure()
    for k in range(2):
        for i in range(n):
            img_psnr[k][i] = psnr(img_gt[k], img_summary[k][i])
        plt.plot(img_ang, img_psnr[k], 'o-', label=f'ref #{k}')
    plt.legend()  
    plt.xlabel('Number of angles')
    plt.ylabel('PSNR')  
    fn_fig_psnr = f_save_root + f'/psnr_summary.png'
    plt.savefig(fn_fig_psnr)

    '''
    plot MSE  
    '''   
    n = len(img_summary[0])
    img_MSE = np.zeros([4, n]) # mse at iters: 500, 1000, 2000, 4000
    fn_json = glob.glob(f_save_root + '/*.json')[0]
    with open(fn_json, 'r') as f:
        loss_read = json.load(f)
    
    plt.figure()
    for j, v in enumerate([500, 1000, 2000, 4000]):
        for i in range(n):
            img_MSE[j, i] = loss_read[f'{i}']['mse']['value'][v-1]
        img_MSE[j] = img_MSE[j][::-1]
        plt.plot(img_ang, img_MSE[j], 'o-',label=f'iter = {v}')
    plt.legend()
    plt.xlabel('Number of angles')
    plt.ylabel('Mean square loss (MSE)')
    fn_fig_mse = f_save_root + f'/mse_summary.png'
    plt.savefig(fn_fig_mse)

    '''
    plot 2000 iterations to exam
    '''
    plt.figure();plt.plot(img_ang, img_MSE[2], 'go-',label=f'iter = {2000}')
    plt.xlabel('Angle (degree)')
    plt.ylabel('X-ray energy (keV)')
    plt.legend()


"""
def psnr(label, outputs, max_val=1):
    try:
        label = label.cpu().detach().numpy()
    except:
        pass
    try:
        outputs = outputs.cpu().detach().numpy()
    except:
        pass
    max_val = max(np.max(label), np.max(outputs))
    img_diff = (outputs - label) / max_val
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(1. / rmse)
        return PSNR
"""

class simpleDataset(Dataset):
    def __init__(self, blur_dir, gt_dir, length=None):
        super().__init__()
        self.fn_blur = np.sort(glob.glob(f'{blur_dir}/*'))
        self.fn_gt = np.sort(glob.glob(f'{gt_dir}/*'))
        if not length is None:
            self.fn_blur = self.fn_blur[:length]
            self.fn_gt = self.fn_gt[:length]

    def __len__(self):
        return len(self.fn_blur)

    def __getitem__(self, idx):
        img_blur = io.imread(self.fn_blur[idx]) # (8, 512, 512)
        img_gt = io.imread(self.fn_gt[idx]) # (8, 512, 512)

        img_blur = torch.tensor(img_blur, dtype=torch.float)        
        img_gt = torch.tensor(img_gt, dtype=torch.float)
        
        return img_blur, img_gt
    

def simple_train_valid_dataloader(blur_dir, gt_dir, num, split_ratio=0.8, batch_size=16):
    dataset = simpleDataset(blur_dir, gt_dir, num)
    n = len(dataset)
    n_train = int(split_ratio * n)
    n_valid = n - n_train
    train_ds, valid_ds = torch.utils.data.random_split(dataset, (n_train, n_valid))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    print(f'train dataset = {len(train_loader.dataset)}')
    print(f'valid dataset = {len(valid_loader.dataset)}')
    return train_loader, valid_loader


def check_model_fitting(model_den, dataloader, idx, device='cuda'):
    image_test = dataloader.dataset[idx][0][None]
    image_target = dataloader.dataset[idx][1][None]
    with torch.no_grad():
        img = image_test.to(device)
        img = img.unsqueeze(0)
        output = model_den(img)
    output = output.cpu().data.numpy().squeeze()
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(img.cpu().numpy().squeeze(), clim=[0,1])
    plt.subplot(132)
    plt.imshow(output, clim=[0,1])
    plt.subplot(133)
    plt.imshow(image_target.cpu().numpy().squeeze(), clim=[0,1])

def check_model_fitting_image(model_den, img, device='cuda'):
    s = img.shape
    if len(s) == 2:
        img = np.reshape(img, (1, 1, s[0], s[1]))
    img = torch.tensor(img).to(device)
    with torch.no_grad():
        output = model_den(img)
    output = output.cpu().data.numpy().squeeze()
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img.cpu().numpy().squeeze(), clim=[0,1])
    plt.subplot(122)
    plt.imshow(output, clim=[0,1])

def simple_train_denoise(model_den, dataloader, device, loss_r, lr=1e-4):
    global vgg19
    mse_criterion = MSELoss()

    keys = list(loss_r.keys())
    loss_value = {}
    running_psnr = 0.0
    running_loss = {}
    for k in keys:
        running_loss[k] = 0.0

    loss_summary = {}
    model_den.train()
    opt_den = optim.Adam(model_den.parameters(), lr=lr, betas=(0.5, 0.999))

    running_psnr = 0.0
    batch_size = dataloader.batch_size
    for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/batch_size)):
        image_data = data[0].to(device) # (16, 256, 256)
        label = data[1].to(device) # (16, 256, 256)

        s = image_data.size()
        s1 = (s[0], 1, s[1], s[2])
        image_data = image_data.reshape(s1)
        label = label.reshape(s1)

        img_out = model_den(image_data)
        loss_value['mse'] = mse_criterion(img_out, label)  
        loss_value['vgg'] = pyxas.vgg_loss(img_out, label, vgg19, device=device)

        total_loss = 0.0
        for k in keys:
            if loss_r[k] > 0:
                total_loss += loss_value[k] * loss_r[k]

        model_den.zero_grad()
        total_loss.backward()
        opt_den.step()

        batch_psnr = psnr(label, img_out)
        for k in keys:
            running_loss[k] += loss_value[k].item() * loss_r[k]
        running_psnr += batch_psnr
    for k in running_loss.keys():
        loss_summary[k] = running_loss[k] / len(dataloader.dataset)
    loss_summary['psnr'] = running_psnr/int(len(dataloader.dataset)/batch_size)
    return loss_summary


def test_model_denoise():
    device = 'cuda:2'
    global vgg19

    torch.manual_seed(0)
    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    vgg19.to(device).eval()

    loss_r = {}
    loss_r['vgg'] = 1           
    loss_r['mse'] = 1

    '''
    f_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/dataset'
    gt_dir = f_root + '/img_gt_256'
    blur_dir = f_root + '/img_noise_256'
    '''

    f_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/dataset1'
    gt_dir = f_root + '/img_gt_128'
    blur_dir = f_root + '/img_noise_128'

    train_loader, valid_loader = simple_train_valid_dataloader(blur_dir, gt_dir, 5000, 0.8, 16)
    #model_den = pyxas.RRDBNet(1, 1, 16, 4, 32).to(device)
    model_den = Unet().to(device)

    lr = 1e-5
    for epoch in range(0, 1000):
        print(f'epoch = {epoch}:')
        loss_summary = simple_train_denoise(model_den, train_loader, device, loss_r, lr)
        txt = f'mse_loss = {loss_summary["mse"]:1.5e}'
        txt += f'\nvgg_loss = {loss_summary["vgg"]:1.5e}'
        txt += f'\npsnr = {loss_summary["psnr"]}'
        print(txt)
    #f_model = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/model_saved'
    f_model = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/model_saved_128'
    fn_save_model = f_model + '/model_saved_unet.pth'
    torch.save(model_den.state_dict(), fn_save_model)


def load_sample_img_general(flag=3, sample=1):
    if sample == 1:
        fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/s1'
    else:
        fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/s2'
    files = np.sort(glob.glob(fn_root + '/*.tiff'))
    n_file = len(files)

    t = io.imread(files[0])
    s = t.shape

    img_n = np.zeros((n_file, *s))
    for i in range(n_file):
        img_n[i] = io.imread(files[i])

    if flag >= 3:
        max_val = np.max(img_n)
        img4d = img_n / max_val
        img4d = img4d[:, s[0]//2]
        
    elif flag <=2 and flag >=0:
        img4d = np.sum(img_n, axis=flag+1)
        max_val = np.max(np.sum(img4d, axis=0))
        img4d = img4d / max_val
    else:
        return None
    return img4d
    

def load_ref_general(sample=1):
    if sample == 1:
        fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/s1'
    else:
        fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/s2'
    ref_files = np.sort(glob.glob(fn_root + '/ref*.txt'))
    ref_comb = []
    n = len(ref_files)
    for i in range(n):
        t = np.loadtxt(ref_files[i])
        ref_comb.append(t)
    return ref_comb


def load_select_eng_general(sample=1, id_s=0):
    if sample == 1:
        fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/s1'
    else:
        fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/s2'
    eng_file = np.sort(glob.glob(fn_root + '/eng*.txt'))[0]
    eng = np.loadtxt(eng_file)[id_s:]
    return eng

def load_sample_img(flag=3, sample=1):

    if sample == 1:
        fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/s1'
    else:
        fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/s2'    
    fn1 = fn_root + '/img3D_1_ref1.tiff'
    fn2 = fn_root + '/img3D_1_ref2.tiff'
    img3D_ref1 = io.imread(fn1)
    img3D_ref2 = io.imread(fn2)

    if flag >= 3:
        img1_raw = img3D_ref1[128]
        img2_raw = img3D_ref2[128]
    elif flag <=2 and flag >=0:
        img1_raw = np.sum(img3D_ref1, axis=flag) 
        img2_raw = np.sum(img3D_ref2, axis=flag) 
        max_val = np.max(img1_raw+img2_raw)
        img1_raw = img1_raw / max_val
        img2_raw = img2_raw / max_val
    else:
        return None, None
    return img1_raw, img2_raw



def convert_to_cuda(img1_raw, img2_raw, device):
    s = img1_raw.shape
    img1 = img1_raw.reshape((1, 1, *s))
    img2 = img2_raw.reshape((1, 1, *s))
    img1 = torch.tensor(img1, dtype=torch.float32).to(device)
    img2 = torch.tensor(img2, dtype=torch.float32).to(device)
    return img1, img2


def convert_to_cuda_general(img4d, device):
    s = img4d.shape # e.g. (4, 128, 128): 4 components (references)
    img = img4d.reshape((s[0], 1, *s[1:])) # (1, 4, 128, 128)
    img_cuda = torch.tensor(img, dtype=torch.float32).to(device)
    return img_cuda


def simu_sino(img1, img2, r_shuffle, angle_l, counts=None, device='cuda:0'):
    
    r1_shuffle = r_shuffle[0]
    r2_shuffle = r_shuffle[1]
    sino1 = torch_sino(img1, angle_l, device, r1_shuffle)
    sino2 = torch_sino(img2, angle_l, device, r2_shuffle)
    s_sino = sino1.shape
    if counts is None:
        return sino1 + sino2
    noise1 = np.random.poisson(sino1.detach().cpu().numpy()*counts, s_sino) / counts
    noise2 = np.random.poisson(sino2.detach().cpu().numpy()*counts, s_sino) / counts
    sino1 = torch.tensor(noise1).to(device) 
    sino2 = torch.tensor(noise2).to(device) 
    return sino1+sino2

def simu_sino_general(img4d_cuda, r_shuffle_general, angle_l, counts=None, atten2D_at_angle=[1], device='cuda:0', cal_pix_atten=False):
    # img4d_cuda.shape = (n_ref, 1, 256, 256)

    sino_comb = torch_sino_general(img4d_cuda, angle_l, r_shuffle_general, atten2D_at_angle, device, cal_pix_atten)
    s_sino = sino_comb.shape # (n_eng, n_ref, 1, 256) = (60, 4, 1, 256)
    if counts is None:
        sino_sum = torch.sum(sino_comb, axis=1, keepdim=True) #(60, 1, 1, 256)
    else:
        sino_noise = np.random.poisson(sino_comb.detach().cpu().numpy()*counts, s_sino) / counts
        sino_noise = torch.tensor(sino_noise).to(device)
        #sino_noise = torch.poisson(sino_comb * counts) / counts
        sino_sum = torch.sum(sino_noise, axis=1, keepdim=True) #(60, 1, 1, 256)
    return sino_sum





def ml_xanes3D(sino_sum, r_shuffle, loss_r, n_epoch, angle_l, lr=0.1):
    '''
    sino.shape = (n_angle, 1, 1, 128)
    '''
    s = sino_sum.shape
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value':[], 'rate':loss_r[k]} 

    r1_shuffle = r_shuffle[0]
    r2_shuffle = r_shuffle[1]

    #guess_ini = np.zeros((1, 1, *s))
    guess_ini = np.zeros((1, 1, s[-1], s[-1]))
    guess1 = torch.tensor(guess_ini, dtype=torch.float32).to(device).requires_grad_(True)
    guess2 = torch.tensor(guess_ini, dtype=torch.float32).to(device).requires_grad_(True)

    sino_col = s[-1] // 2
    for epoch in trange(n_epoch):
        '''
        model_train_flag = 0
        if (epoch + 1) % 200 == 0:
            model_den.eval()
            #opt_den = optim.Adam(model_den.parameters(), lr=lr, betas=(0.5, 0.999))
            with torch.no_grad():
                guess1[guess1<0] = 0
                guess2[guess2<0] = 0       
                guess2.grad = None
                guess1.grad = None
            t1 = guess1.detach().clone()
            t2 = guess2.detach().clone()
            t1 = model_den(t1)
            t2 = model_den(t2)
            t1 = t1.detach()
            t2 = t2.detach()
            guess1 = t1.requires_grad_(True)
            guess2 = t2.requires_grad_(True)

            model_train_flag = 1
        '''
        sino_out1 = torch_sino(guess1, angle_l, device, r1_shuffle)
        sino_out2 = torch_sino(guess2, angle_l, device, r2_shuffle)
        sino_dif = sino_out1 + sino_out2 - sino_sum

        loss_val['mse'] = torch.square(sino_out1+sino_out2 - sino_sum).mean()    
        #loss_val['tv_sino'] = tv_loss(sino_dif[:,:,:, :sino_col]) + tv_loss(sino_dif[:,:,:, -sino_col:])
        loss_val['tv_sino'] = tv_loss(sino_dif)
        loss_val['tv_img'] = tv_loss(guess1) + tv_loss(guess2)
        loss_val['l1_sino'] = l1_loss(sino_sum, sino_out1+sino_out2)
        loss_val['likelihood_sino'] = poisson_likelihood_loss(sino_out1+sino_out2, sino_sum)
        
        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                loss = loss + loss_val[k] * loss_r[k]

        loss.backward()

        with torch.no_grad():
            guess1 -= lr * guess1.grad 
            guess2 -= lr * guess2.grad  
            if epoch > 10:
                guess1[guess1<0] = 0
                guess2[guess2<0] = 0       
            guess2.grad = None
            guess1.grad = None
    sino_out = sino_out1.detach() + sino_out2.detach()
    sino_out = sino_out.cpu().numpy()
    guess1 = guess1.detach().cpu().numpy().squeeze()
    guess2 = guess2.detach().cpu().numpy().squeeze()
    return loss_his, guess1, guess2, sino_out


def ml_tomo_general(guess, sino_sum_general, loss_r, n_epoch, angle_l, lr=0.1, threshold=0):
    '''
    sino_sum.shape = (n_angle, 1, 1, 128)
    '''
    s = sino_sum_general.shape
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value':[], 'rate':loss_r[k]} 

    if guess is None:
        guess = torch.zeros((1, 1, s[-1], s[-1]), dtype=torch.float32)
        guess = guess.to(device).requires_grad_(True) # (4, 1, 256, 256)
    else:
        if not torch.is_tensor(guess):
            guess = torch.tensor(guess, dtype=torch.float32)
        guess = guess.reshape(1, 1, s[-1], s[-1])
        guess = guess.to(device).requires_grad_(True)

    for epoch in trange(n_epoch):
        sino_out = simu_sino_general(guess, [1], angle_l, None, [1], device)
        sino_dif = sino_out - sino_sum_general

        loss_val['mse'] = torch.square(sino_dif).mean() 
        loss_val['tv_sino'] = tv_loss(sino_dif)
        loss_val['tv_img'] = tv_loss(guess)
        loss_val['l1_sino'] = l1_loss(sino_sum_general, sino_out)
        loss_val['likelihood_sino'] = poisson_likelihood_loss(sino_out, sino_sum_general)
        
        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                loss = loss + loss_val[k] * loss_r[k]
        loss.backward()

        with torch.no_grad():
            guess -= lr * guess.grad 
            if epoch > 50:
                guess[guess<threshold] = 0    
            guess.grad = None

    sino_out = sino_out.detach().cpu().numpy() 
    guess = guess.detach().cpu().numpy().squeeze()
    return loss_his, guess, sino_out


def ml_xanes3D_general(guess, sino_sum_general, r_shuffle_general, loss_r, n_epoch, angle_l, lr=0.1, row_range=None):
    '''
    sino_sum.shape = (n_angle, 1, 1, 128)
    r_shuffle.shape = (n_angle, n_ref)
    '''
    s = sino_sum_general.shape
    n_ref = r_shuffle_general.shape[-1]
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value':[], 'rate':loss_r[k]} 

    if guess is None:
        guess = torch.zeros((n_ref, 1, s[-1], s[-1]), dtype=torch.float32)
        guess = guess.to(device).requires_grad_(True) # (4, 1, 256, 256)
    else:
        if not torch.is_tensor(guess):
            guess = torch.tensor(guess, dtype=torch.float32)
        guess = guess.reshape(n_ref, 1, s[-1], s[-1])
        guess = guess.to(device).requires_grad_(True)

    for epoch in trange(n_epoch):

        sino_out = simu_sino_general(guess, r_shuffle_general, angle_l, None, [1], device)
        sino_dif = sino_out - sino_sum_general

        loss_val['mse'] = torch.square(sino_dif).mean() 
        if not row_range is None:
            loss_val['tv_sino'] = tv_loss(sino_dif[row_range[0]: row_range[1]])
        else:
            loss_val['tv_sino'] = tv_loss(sino_dif)
        loss_val['tv_img'] = tv_loss(guess)
        loss_val['l1_sino'] = l1_loss(sino_sum_general, sino_out)
        loss_val['likelihood_sino'] = poisson_likelihood_loss(sino_out, sino_sum_general)
        
        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                loss = loss + loss_val[k] * loss_r[k]
        loss.backward()

        with torch.no_grad():
            guess -= lr * guess.grad 
            if epoch > 10:
                guess[guess<0] = 0    
            guess.grad = None

    sino_out = sino_out.detach().cpu().numpy() 
    guess = guess.detach().cpu().numpy().squeeze()
    return loss_his, guess, sino_out


def ml_xanes3D_with_FL_correction(guess, sino_sum_general, r_shuffle_general,
                                    loss_r, n_epoch, angle_l, 
                                    lr=0.1, row_range=None, atten2D_at_angle=[1], device='cuda', cal_pix_atten=False):
    '''
    sino_sum.shape = (n_angle, 1, 1, 128)
    r_shuffle.shape = (n_angle, n_ref)
    '''
    guess_old = 0
    s = sino_sum_general.shape
    n_ref = r_shuffle_general.shape[-1]
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value':[], 'rate':loss_r[k]} 
    loss_his['img_diff'] = []
    if guess is None:
        guess = torch.zeros((n_ref, 1, s[-1], s[-1]), dtype=torch.float32)
        guess = guess.to(device).requires_grad_(True) # (4, 1, 256, 256)
    else:
        if not torch.is_tensor(guess):
            guess = torch.tensor(guess, dtype=torch.float32)
        guess = guess.reshape(n_ref, 1, s[-1], s[-1])
        guess = guess.to(device).requires_grad_(True)

    for epoch in trange(n_epoch):
        sino_out = simu_sino_general(guess, r_shuffle_general, angle_l, None, atten2D_at_angle, device, cal_pix_atten)
        sino_dif = sino_out - sino_sum_general

        loss_val['mse'] = torch.square(sino_dif).mean() 
        if not row_range is None:
            loss_val['tv_sino'] = tv_loss(sino_dif[row_range[0]: row_range[1]])
        else:
            loss_val['tv_sino'] = tv_loss(sino_dif)
        loss_val['tv_img'] = tv_loss(guess)
        loss_val['l1_sino'] = l1_loss(sino_sum_general, sino_out)
        loss_val['likelihood_sino'] = poisson_likelihood_loss(sino_out, sino_sum_general)
        
        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                if k == 'mse_fit':
                    if epoch < 100:
                        continue
                loss = loss + loss_val[k] * loss_r[k]
                    
        loss.backward()

        with torch.no_grad():
            guess -= lr * guess.grad 
            if epoch > 0:
                guess[guess<0] = 0  
            guess.grad = None

        guess_new = guess.detach().cpu().numpy().squeeze()
        loss_his['img_diff'].append(np.abs(np.mean(guess_new - guess_old)))
        guess_old = guess_new
    sino_out = sino_out.detach().cpu().numpy() 
    #guess = guess.detach().cpu().numpy().squeeze()
    return loss_his, guess_new, sino_out

"""

def ml_xanes3D_with_FL_correction_obsolete(guess, sino_sum_general, r_shuffle_general,
                                    spectrum_ref, eng, loss_r, n_epoch, angle_l, 
                                    lr=0.1, row_range=None, atten2D_at_angle=[1], device='cuda', cal_pix_atten=False):
    '''
    sino_sum.shape = (n_angle, 1, 1, 128)
    r_shuffle.shape = (n_angle, n_ref)
    '''

    s = sino_sum_general.shape
    n_ref = r_shuffle_general.shape[-1]
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value':[], 'rate':loss_r[k]} 

    if guess is None:
        guess = torch.zeros((n_ref, 1, s[-1], s[-1]), dtype=torch.float32)
        guess = guess.to(device).requires_grad_(True) # (4, 1, 256, 256)
    else:
        if not torch.is_tensor(guess):
            guess = torch.tensor(guess, dtype=torch.float32)
        guess = guess.reshape(n_ref, 1, s[-1], s[-1])
        guess = guess.to(device).requires_grad_(True)


    sino_sum = torch.mean(sino_sum_general, axis=-1, dtype=torch.float, keepdims=True) # e.g, (60, 1,1,1)
    #X, Y_fit = pyxas.fit_element_with_reference(eng, sino_sum, spectrum_ref, [], False, device)
    
    #X_old = torch.zeros((2,2)).to(device)
    #Y_fit_old = torch.zeros((1,2)).to(device)

    for epoch in trange(n_epoch):

        sino_out = simu_sino_general(guess, r_shuffle_general, angle_l, None, atten2D_at_angle, device, cal_pix_atten)
        sino_dif = sino_out - sino_sum_general

        sino_out_sum = torch.mean(sino_out, axis=-1, keepdims=True) # e.g, (60, 1,1,1)
        #X_out, Y_fit_out = pyxas.fit_element_with_reference(eng, sino_out_sum, spectrum_ref, [], False, device)
        #Y_fit_diff = Y_fit_out - Y_fit
        #X_diff = X_out - X
        loss_val['mse'] = torch.square(sino_dif).mean() 
        if not row_range is None:
            loss_val['tv_sino'] = tv_loss(sino_dif[row_range[0]: row_range[1]])
        else:
            loss_val['tv_sino'] = tv_loss(sino_dif)
        loss_val['tv_img'] = tv_loss(guess)
        loss_val['l1_sino'] = l1_loss(sino_sum_general, sino_out)
        loss_val['likelihood_sino'] = poisson_likelihood_loss(sino_out, sino_sum_general)
        
        #loss_val['mse_fit'] = torch.square(Y_fit_diff).mean() 
        #loss_val['mse_fit'] = torch.square(X_diff).mean()

        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                if k == 'mse_fit':
                    if epoch < 100:
                        continue
                loss = loss + loss_val[k] * loss_r[k]
                    
        loss.backward()

        with torch.no_grad():
            guess -= lr * guess.grad 
            if epoch > 0:
                guess[guess<0] = 0  
            guess.grad = None

    sino_out = sino_out.detach().cpu().numpy() 
    guess = guess.detach().cpu().numpy().squeeze()
    return loss_his, guess, sino_out
    
    
def ml_xrf_xanes3D_with_fitting_general(guess, sino_sli_cuda, ref_interp,
                                    spectrum_ref, eng, loss_r, n_epoch, angle_l, f_scale,
                                    lr=0.1, row_range=None, atten2D_at_angle=[1], device='cuda'):
    '''
    sino_sum.shape = (n_angle, 1, 1, 128)
    r_shuffle.shape = (n_angle, n_ref)
    rho_sli.shape = (1, 1, 128, 128)
    '''

    s = sino_sli_cuda.shape
    n_ref = ref_interp.shape[-1]
    keys = list(loss_r.keys())
    loss_his = {}
    loss_val = {}
    for k in keys:
        loss_his[k] = {'value': [], 'rate': loss_r[k]}

    if guess is None:
        guess = torch.zeros((n_ref, 1, s[-1], s[-1]), dtype=torch.float32)
        guess = guess.to(device).requires_grad_(True)  # (4, 1, 256, 256)
    else:
        if not torch.is_tensor(guess):
            guess = torch.tensor(guess, dtype=torch.float32)
        guess = guess.reshape(n_ref, 1, s[-1], s[-1])
        guess = guess.to(device).requires_grad_(True)

    sino_sum = torch.mean(sino_sli_cuda, axis=-1, dtype=torch.float, keepdims=True)  # e.g, (60, 1,1,1)
    #X, Y_fit = pyxas.fit_element_with_reference(eng, sino_sum, spectrum_ref, [], False, device)

    #X_old = torch.zeros((2, 2)).to(device)
    #Y_fit_old = torch.zeros((1, 2)).to(device)

    if not len(atten2D_at_angle) == n_angle:
        atten2D_at_angle = torch.ones(n_angle, 1, s0[-2], s0[-1])
    atten2D_at_angle = torch.tensor(atten2D_at_angle, dtype=torch.float).to(device)
    if len(atten2D_at_angle.shape) == 3:  # e.g., (60, 128, 128)
        atten2D_at_angle = atten2D_at_angle.unsqueeze(1)  # (60, 1, 128, 128)

    for epoch in trange(n_epoch):
        sino_out = torch_sino_xrf_atten(guess, angle_l, f_scale, ref_interp, atten2D_at_angle, device)
        sino_dif = sino_out - sino_sli_cuda

        sino_out_sum = torch.mean(sino_out, axis=-1, keepdims=True)  # e.g, (60, 1,1,1)
        #X_out, Y_fit_out = pyxas.fit_element_with_reference(eng, sino_out_sum, spectrum_ref, [], False, device)
        
        #X_diff = X_out - X
        loss_val['mse'] = torch.square(sino_dif).mean()
        if not row_range is None:
            loss_val['tv_sino'] = tv_loss(sino_dif[row_range[0]: row_range[1]])
        else:
            loss_val['tv_sino'] = tv_loss(sino_dif)
        loss_val['tv_img'] = tv_loss(guess)
        loss_val['l1_sino'] = l1_loss(sino_sli_cuda, sino_out)

        # loss_val['mse_fit'] = torch.square(Y_fit_diff).mean()
        #loss_val['mse_fit'] = torch.square(X_diff).mean()

        loss = 0.0
        for k in keys:
            loss_his[k]['value'].append(loss_val[k].item())
            if loss_r[k] > 0:
                if k == 'mse_fit':
                    if epoch < 100:
                        continue
                loss = loss + loss_val[k] * loss_r[k]

        loss.backward()

        with torch.no_grad():
            guess -= lr * guess.grad
            if epoch > 10:
                guess[guess < 0] = 0
            guess.grad = None

    sino_out = sino_out.detach().cpu().numpy()
    guess = guess.detach().cpu().numpy().squeeze()
    return loss_his, guess, sino_out

"""
############################################################
############# test script ##################################
############################################################

def test_number_of_angles():

    device = 'cuda:0'
    counts = 1000
    ang_interval = [1, 2, 3, 4, 5, 6, 10, 12]
    n_test = len(ang_interval)
    n_epoch = 4000
    eng = np.loadtxt(fn_eng)[1:]

    img1_raw, img2_raw = load_sample_img(flag=0)
    img1, img2 = convert_to_cuda(img1_raw, img2_raw, device)

    loss_record = {}

    #f_save_root = f'/home/mingyuan/Work/machine_learn/ml_xanes_tomo/loss_record/seq_select/iter_{n_epoch}'
    #f_save_root = f'/home/mingyuan/Work/machine_learn/ml_xanes_tomo/loss_record/random_select/iter_{n_epoch}'
    f_save_root = f'/home/mingyuan/Work/machine_learn/ml_xanes_tomo/loss_record/step_select/iter_{n_epoch}'
    f_save_raw1 = f_save_root + '/img_gt_ref1.tiff'
    f_save_raw2 = f_save_root + '/img_gt_ref2.tiff'
    io.imsave(f_save_raw1, img1.detach().cpu().numpy().squeeze())
    io.imsave(f_save_raw2, img2.detach().cpu().numpy().squeeze())
    try:
        os.makedirs(f_save_root)
    except:
        print(f'{f_save_root} exists')
    for i in trange(n_test):
        angle_l = np.arange(0, 180, ang_interval[i])
        
        r_shuffle, eng_shuffle = shuffle_angle_spec(angle_l, eng, 2, (ref1, ref2))
        plot_eng_select(angle_l, eng_shuffle)        
        plt.pause(1)
        fn_save_ang = f_save_root + f'/ang_energy_angle_{len(angle_l):03d}.png'
        plt.savefig(fn_save_ang)

        sino_sum = simu_sino(img1, img2, r_shuffle, angle_l, counts=counts, device=device)
        loss_his, guess1, guess2, sino_out = ml_xanes3D(sino_sum, r_shuffle, loss_r, n_epoch, angle_l, lr=0.2)
        loss_record[i] = loss_his
        
        fn_1 = f_save_root + f'/rec1_iter_{n_epoch}_angle_{len(angle_l):03d}.tiff'
        fn_2 = f_save_root + f'/rec2_iter_{n_epoch}_angle_{len(angle_l):03d}.tiff'
        io.imsave(fn_1, guess1)
        io.imsave(fn_2, guess2)

        fn_3 = f_save_root + f'/compare_iter_{n_epoch}_angle_{len(angle_l):03d}'
        plot_res(angle_l, img1, img2, guess1, guess2, sino_out, sino_sum, loss_his, [0,1], counts)
        plt.pause(1)
        plt.savefig(fn_3)

    fn_4 = f_save_root + f'/loss_record_angles_{n_epoch}_iters.json'
    with open(fn_4, 'w') as f:
        json.dump(loss_record, f, indent=4)


def test_recon_3D_xanes():
    device = 'cuda:1'
    counts = 1000
    n_epoch = 8000
    atten2D_at_angle = [1]
    loss_r = {}
    loss_r['mse'] = 1 # 1
    loss_r['tv_sino'] = 20 # 20
    loss_r['tv_img'] = 10 # 10
    loss_r['l1_sino'] = 0 #0
    loss_r['likelihood_sino'] = 0 # 0

    sample = 2 # 1: Ni,   2: Fe
    img4d = load_sample_img_general(flag=0, sample=sample)
    ref_comb = load_ref_general(sample=sample)
    eng = load_select_eng_general(sample=sample, id_s=1)
    img4d_cuda = convert_to_cuda_general(img4d, device) # (4, 1, 256, 256)
    angle_l = np.arange(0, 180, 3)
    
    r_shuffle_general, eng_shuffle = shuffle_angle_spec_general(angle_l, eng, 2, ref_comb)
    # r_shuffle.shape = (60, 4) = (n_eng, n_ref)
    #plot_eng_select(angle_l, eng_shuffle)  
    sino_sum_general = simu_sino_general(img4d_cuda, r_shuffle_general, angle_l, counts, atten2D_at_angle, device)
    
    guess = None    
    loss_his, guess, sino_out = ml_xanes3D_general(guess, sino_sum_general, r_shuffle_general, loss_r, n_epoch, angle_l, lr=0.2)
    plot_res_general(img4d, guess, sino_sum_general, sino_out, loss_his, c_range=None, counts=counts)


def test_recon_3D_tomo():
    from skimage.restoration import denoise_tv_chambolle
    device = 'cuda:1'
    counts = 1000
    n_epoch = 1000
    atten2D_at_angle = [1]
    loss_r = {}
    loss_r['mse'] = 1 # 1
    loss_r['tv_sino'] = 20 # 20
    loss_r['tv_img'] = 50 # 10
    loss_r['l1_sino'] = 0 #0
    loss_r['likelihood_sino'] = 0 # 0

    sample = 2 # 1: Ni,   2: Fe
    img4d = load_sample_img_general(flag=0, sample=sample)
    img2d = img4d[0:1] # (1, 256, 256)
    img2d_cuda = convert_to_cuda_general(img2d, device) # (1, 1, 256, 256)
    angle_l = np.arange(0, 180, 3)    
    sino_sum = simu_sino_general(img2d_cuda, [1], angle_l, counts, atten2D_at_angle, device)
    # sino_sum.shape = (n_angle, 1, 1, 256)
    guess = None    
    loss_his, guess, sino_out = ml_tomo_general(guess, sino_sum, loss_r, n_epoch, angle_l, lr=0.2)
    plot_res_general(img2d, guess, sino_sum, sino_out, loss_his, c_range=None, counts=counts)


def test_recon_3D_FXI():
    device = 'cuda:2'
    n_epoch = 500
    loss_r = {}
    loss_r['mse'] = 1 # 1
    loss_r['tv_sino'] = 20 # 20
    loss_r['tv_img'] = 100 # 10
    loss_r['l1_sino'] = 0 #0
    loss_r['likelihood_sino'] = 0 # 0


    fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/FXI'
    fn_prj = fn_root + '/prj_bin4_shift.tiff'
    fn_angle = fn_root + '/angle_list.txt'
    angle_l = np.loadtxt(fn_angle)[::1]
    sino = io.imread(fn_prj)[::1] # (30, 270, 316)
    sino_2d = sino[:, 125:126] # (ml_xanes3D_general30, 1, 316)
    sino_2d = sino_2d[:, np.newaxis] # (30, 1, 1, 316)
    sino_2d *= 100
    sino_sum = torch.tensor(sino_2d, dtype=torch.float).to(device)
    guess = None    
    loss_his, guess, sino_out = ml_tomo_general(guess, sino_sum, loss_r, n_epoch, angle_l, lr=0.2)
   
def test_recon_3D_HXN_chip():
    device = 'cuda:2'
    n_epoch = 200
    loss_r = {}
    loss_r['mse'] = 1 # 1
    loss_r['tv_sino'] = 5 # 20
    loss_r['tv_img'] = 5 # 10
    loss_r['l1_sino'] = 0 #0
    loss_r['likelihood_sino'] = 5 # 0
    fn_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo/sample_img/HXN/chip/Experimental_data'
    fn_prj = fn_root + '/Cu_example_22.tiff'
    fn_angle = fn_root + '/angle_22.txt'
    angle_l = np.loadtxt(fn_angle)[::3]
    sino = io.imread(fn_prj)[::3] # (30, 270, 316)

    s = sino.shape
    rec = np.zeros([s[1], s[2], s[2]])
    for i in trange(s[1]):
        sino_2d = sino[:, i:i+1] # (30, 1, 316)
        sino_2d = sino_2d[:, np.newaxis] # (30, 1, 1, 316)
        #sino_2d *= 50
        sino_sum = torch.tensor(sino_2d, dtype=torch.float).to(device)
        max_val = torch.max(sino_sum)
        sino_sum = sino_sum / max_val * 50
        guess = None    
        loss_his, guess, sino_out = ml_tomo_general(guess, sino_sum, loss_r, n_epoch, angle_l, lr=0.2)
        guess = guess / 50 * max_val.cpu().numpy()

        rec[i] = guess

    plot_res_general(guess, guess, sino_sum, sino_out, loss_his, c_range=None, counts=None)


############################################
## test script lagacy:
############################################
def test_phantom():
    device = 'cuda:0'
    angle_all = np.arange(0, 180, 3)
    # angle_l = angle_all[:40]
    angle_l = angle_all[:]
    theta_l = angle_l / 180 * np.pi
    n_theta = len(theta_l)

    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    vgg19.to(device).eval()

    img2D_raw = tomopy.shepp2d(128)[0] / 256.0  # shape=(128, 128)
    sino_radon = radon(img2D_raw, angle_l)
    img_iradon = iradon(sino_radon, theta=angle_l)
    s = img2D_raw.shape

    img2D = img2D_raw.reshape((1, 1, *s))
    img2D = torch.tensor(img2D, dtype=torch.float32).to(device)
    ang = 30
    # img_r = TF.rotate(img2D, ang, torchvision.transforms.InterpolationMode.BILINEAR) # (1, 1, 128, 128) # this does not work with autograd

    theta = ang / 180.0 * np.pi
    img_r = rot_img(img2D, theta, device)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img2D.cpu().numpy().squeeze())
    plt.subplot(122)
    plt.imshow(img_r.cpu().numpy().squeeze())

    ##
    sino = torch_sino(img2D, angle_l, device)
    s1 = sino.shape

    plt.figure()
    plt.imshow(sino.cpu().numpy().squeeze())

    mse_loss = torch.nn.MSELoss()
    # guess = np.random.random((1, 1, *s))

    mask = np.ones((1, *s))
    mask = pyxas.circ_mask(mask, 0)
    mask = mask[None, ...]
    guess_ini = np.ones((1, 1, *s))
    guess_ini = np.zeros((1, 1, *s))
    guess = guess_ini * mask
    guess = torch.tensor(guess, dtype=torch.float32).to(device).requires_grad_(True)
    m = torch.tensor(mask, dtype=torch.float32).to(device)

    lr = 1e0

    n_epoch = 500
    img_save = np.zeros((n_epoch, *s))
    loss_his = np.zeros(n_epoch)
    loss_mse_his = np.zeros(n_epoch)
    loss_tv_his = np.zeros(n_epoch)
    loss_l1_his = np.zeros(n_epoch)
    loss_vgg_his = np.zeros(n_epoch)

    for epoch in trange(n_epoch):
        sino_out = torch_sino(guess, angle_l, device)
        loss_mse = torch.square(sino - sino_out).mean()
        loss_tv = tv_loss(sino_out)
        loss_l1 = l1_loss(sino, sino_out)
        t1 = sino.reshape((s1[1], s1[2], s1[0], s1[3]))
        t2 = sino_out.reshape((s1[1], s1[2], s1[0], s1[3]))
        # loss_vgg = vgg_loss(t1, t2, vgg19, device=device)
        loss = loss_mse
        if epoch > 100:
            loss += loss_tv * 1e-8

        loss.backward()
        loss_his[epoch] = loss.item()
        loss_mse_his[epoch] = loss_mse.item()
        loss_tv_his[epoch] = loss_tv.item()
        loss_l1_his[epoch] = loss_l1.item()
        # loss_vgg_his[epoch] = loss_vgg.item()
        img_save[epoch] = guess.detach().cpu().numpy().squeeze()
        with torch.no_grad():
            guess -= lr * guess.grad
            guess[guess < 0] = 0
            guess.grad = None

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.imshow(guess.detach().cpu().numpy().squeeze(), clim=[0, 1])
    plt.colorbar()
    plt.subplot(122)
    plt.plot(np.log(loss_his), label='loss')
    plt.plot(np.log(loss_mse_his), label='loss_mse')
    plt.plot(np.log(loss_tv_his), label='loss_tv')
    plt.plot(np.log(loss_l1_his), label='loss_l1')
    # plt.plot(np.log(loss_vgg_his), label='loss_vgg')
    plt.legend()

    sino_all = torch_sino(img2D, angle_all, device)
    sino_guess_all = torch_sino(guess, angle_all, device)

    #####################################
    # use selected energy
    #####################################

    f_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo'
    fn_eng = f_root + '/eng_Ni_select.txt'
    fn_ref1 = f_root + '/ref_NiO.txt'
    fn_ref2 = f_root + '/ref_LiNiO2.txt'
    fn_img = f_root + '/img2D_phantom_128.tiff'

    img2D_raw = io.imread(fn_img)
    ref1 = np.loadtxt(fn_ref1)
    ref2 = np.loadtxt(fn_ref2)
    eng = np.loadtxt(fn_eng)[1:]

    n_eng = len(eng)
    angle_l = np.arange(0, 180, 4)
    n_angle = len(angle_l)

    r1 = np.zeros(n_eng)
    r2 = np.zeros(n_eng)
    x = ref1[:, 0]
    for i in range(n_eng):
        idx = pyxas.find_nearest(x, eng[i])
        r1[i] = ref1[idx, 1]
        r2[i] = ref2[idx, 1]

    c1 = 0.3  # ratio of ref1
    c2 = 0.7

    s = img2D_raw.shape
    img1_raw = img2D_raw * c1
    img2_raw = img2D_raw * c2
    img1 = img1_raw.reshape((1, 1, *s))
    img2 = img2_raw.reshape((1, 1, *s))

    img1 = torch.tensor(img1, dtype=torch.float32).to(device)
    img2 = torch.tensor(img2, dtype=torch.float32).to(device)

    # shuffle to get angle-energy
    r1_shuffle = []
    r2_shuffle = []
    n_div = n_angle // n_eng

    for i in range(n_eng - 1):
        r1_shuffle = r1_shuffle + [r1[i]] * n_div
        r2_shuffle = r2_shuffle + [r2[i]] * n_div
    r1_shuffle = r1_shuffle + [r1[i + 1]] * (n_angle - (i + 1) * n_div)
    r2_shuffle = r2_shuffle + [r2[i + 1]] * (n_angle - (i + 1) * n_div)

    r1_shuffle = np.array(r1_shuffle)
    r2_shuffle = np.array(r2_shuffle)
    idx = np.int16(np.arange(n_angle))
    np.random.shuffle(idx)
    r1_shuffle = r1_shuffle[idx]
    r2_shuffle = r2_shuffle[idx]

    sino1 = torch_sino(img1, angle_l, device, r1_shuffle)
    sino2 = torch_sino(img2, angle_l, device, r2_shuffle)

    s_sino = sino1.shape
    noise1 = np.random.poisson(1000, s_sino) / 1000
    noise2 = np.random.poisson(1000, s_sino) / 1000

    sino1 = sino1 * torch.tensor(noise1).to(device)
    sino2 = sino2 * torch.tensor(noise2).to(device)

    sino_sum = sino1 + sino2

    guess_ini = np.zeros((1, 1, *s))
    guess1 = torch.tensor(guess_ini, dtype=torch.float32).to(device).requires_grad_(True)
    guess2 = torch.tensor(guess_ini, dtype=torch.float32).to(device).requires_grad_(True)

    lr = 0.1
    n_epoch = 5000
    img_save = np.zeros((n_epoch, *s))
    loss_his = np.zeros(n_epoch)
    loss_mse_his = np.zeros(n_epoch)
    loss_tv_his = np.zeros(n_epoch)
    loss_l1_his = np.zeros(n_epoch)
    loss_vgg_his = np.zeros(n_epoch)

    for epoch in trange(n_epoch):
        sino_out1 = torch_sino(guess1, angle_l, device, r1_shuffle)
        sino_out2 = torch_sino(guess2, angle_l, device, r2_shuffle)
        loss_mse = torch.square(sino_out1 + sino_out2 - sino_sum).mean()
        loss_tv = (tv_loss(sino_out1 + sino_out2 - sino_sum)) / 128 / n_angle
        loss_l1 = l1_loss(sino_sum, sino_out1 + sino_out2)
        loss = loss_mse + loss_tv * 10  # + loss_l1 *0.1

        loss.backward()
        loss_his[epoch] = loss.item()
        loss_mse_his[epoch] = loss_mse.item()
        loss_tv_his[epoch] = loss_tv.item()
        loss_l1_his[epoch] = loss_l1.item()
        img_save[epoch] = guess2.detach().cpu().numpy().squeeze()
        with torch.no_grad():
            guess1 -= lr * guess1.grad
            guess2 -= lr * guess2.grad
            if epoch > 10:
                guess1[guess1 < 0] = 0
                guess2[guess2 < 0] = 0
            guess2.grad = None
            guess1.grad = None

    plot_res(angle_l, img1, img2, guess1, guess2, loss_his, c_range=None, counts=None)


## combine ML and tomo
#####################################
# use selected energy
#####################################
def test_phatom_xanes():
    f_root = '/home/mingyuan/Work/machine_learn/ml_xanes_tomo'
    fn_eng = f_root + '/eng_Ni_select.txt'
    fn_ref1 = f_root + '/ref_NiO.txt'
    fn_ref2 = f_root + '/ref_LiNiO2.txt'
    fn_img = f_root + '/img2D_phantom_128.tiff'

    img2D_raw = io.imread(fn_img)
    ref1 = np.loadtxt(fn_ref1)
    ref2 = np.loadtxt(fn_ref2)
    eng = np.loadtxt(fn_eng)[1:]  # [8.344 , 8.345 , 8.351 , 8.3575, 8.3675, 8.392 , 8.415 , 8.448 ]
    eng = np.array([8.34, 8.355, 8.37, 8.385, 8.4, 8.415, 8.43, 8.445])  # uniform selection
    c1 = 0.3  # ratio of ref1
    c2 = 0.7

    s = img2D_raw.shape
    img1_raw = img2D_raw * c1
    img2_raw = img2D_raw * c2
    img1 = img1_raw.reshape((1, 1, *s))
    img2 = img2_raw.reshape((1, 1, *s))

    img1 = torch.tensor(img1, dtype=torch.float32).to(device)
    img2 = torch.tensor(img2, dtype=torch.float32).to(device)

    angle_l = np.arange(0, 180, 6)

    n_eng = len(eng)
    n_angle = len(angle_l)

    r_shuffle, eng_shuffle = shuffle_angle_spec(angle_l, eng, 3, (ref1, ref2))
    '''
    plt.figure()
    plt.scatter(angle_l, eng_shuffle)
    plt.xlabel('Angle (degree)')
    plt.ylabel('X-ray energy (keV)')
    '''
    r1_shuffle = r_shuffle[0]
    r2_shuffle = r_shuffle[1]
    sino1 = torch_sino(img1, angle_l, device, r1_shuffle)
    sino2 = torch_sino(img2, angle_l, device, r2_shuffle)
    s_sino = sino1.shape
    counts = 2000
    noise1 = np.random.poisson(sino1.detach().cpu().numpy() * counts, s_sino) / counts
    noise2 = np.random.poisson(sino2.detach().cpu().numpy() * counts, s_sino) / counts
    sino1 = torch.tensor(noise1).to(device)
    sino2 = torch.tensor(noise2).to(device)

    sino_sum = sino1 + sino2

    lr = 0.1
    n_epoch = 1000
    loss_r = {}
    loss_r['mse'] = 1  # 1
    loss_r['tv_sino'] = 20  # 20
    loss_r['tv_img'] = 50  # 10
    loss_r['l1_sino'] = 0  # 0
    loss_r['likelihood_sino'] = 1  # 0

    sino1, sino2 = simu_sino(img1, img2, ref1, ref2, eng, angle_l, counts=None, device=device)
    loss_his, guess1, guess2 = ml_xanes3D(sino1, sino2, loss_r, n_epoch, angle_l, lr=0.1)
