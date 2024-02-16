from scipy.signal import medfilt2d
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import random
import math
import xraylib
from tqdm import trange
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io
from scipy.ndimage import gaussian_filter as gf
from .dataset_lib import *
from torch.utils.data import DataLoader, Dataset


def apply_model_to_stack(img_stack, model, device, n_iter=1, gaussian_filter=1):
    if torch.is_tensor(img_stack):
        img_stack = np.squeeze(img_stack.detach().cpu().numpy())
    s = img_stack.shape
    img_output = np.zeros(s)
    img_bkg = np.ones(s)
    if n_iter == 0:
        img_output = img_stack
    else:
        for i in trange(s[0]):
            img = img_stack[i]
            p = img.copy()
            for j in range(n_iter):
                p = check_image_fitting(model, p, device, 0)
                if gaussian_filter > 1:
                    p = gf(p, gaussian_filter)
            img_bkg[i] = p
            img_output[i] = img/img_bkg[i]
    return img_output, img_bkg


def check_image_fitting(model, image, device='cuda', plot_flag=0, clim=[0,1], ax='off', title='', figsize=(14, 8)):
    if ax == 'off':
        axes = 'off'
    else:
        axes = 'on'
    if len(image.shape) == 2:
        img = np.expand_dims(image, 0)
    else:
        img = image.copy()
    with torch.no_grad():
        img = torch.tensor(img, dtype=torch.float).to(device)
        img = img.unsqueeze(0)
        img_fit = model(img)
    img_fit = img_fit.cpu().data.numpy()
    if plot_flag:
        plt.figure(figsize=figsize)
        if clim is None:
            clim = [np.min(image), np.max(image)]
        plt.subplot(121);plt.imshow(image, clim=clim);plt.axis(axes);plt.title('raw image')
        plt.subplot(122);plt.imshow(np.squeeze(img_fit), clim=clim);plt.axis(axes);plt.title('recoverd image')
        plt.suptitle(title)
        plt.show()
    return np.squeeze(img_fit)


def check_validation_stack(model, dataloader, idx, idy=0, clim=None, device='cuda', plot_flag=1):
    image_test = dataloader.dataset[idx][0][idy][None]
    image_gt = dataloader.dataset[idx][1][idy][None]
    with torch.no_grad():
        img = image_test.to(device)
        img = img.unsqueeze(0)
        output = model(img)
    output = output.cpu().data.numpy()
    if clim is None:
        cmax, cmin = max(image_gt.flatten()), min(image_gt.flatten())
    else:
        cmin, cmax = np.min(clim), np.max(clim)

    image_test = np.squeeze(image_test.to('cpu').data.numpy())
    image_gt = np.squeeze(image_gt.to('cpu').data.numpy())
    output = np.squeeze(output)
    psnr_before_train = img_psnr(image_test, image_gt)
    psnr_after_train = img_psnr(output, image_gt)
    if plot_flag:
        plt.figure()
        plt.subplot(131);plt.imshow(image_test, clim=[cmin, cmax]);plt.title('initial')
        plt.subplot(132);plt.imshow(image_gt, clim=[cmin, cmax]);plt.title('ground truth')
        plt.subplot(133);plt.imshow(output, clim=[cmin, cmax]);plt.title('recover')
        plt.title(f'psnr: {psnr_before_train:.3f} --> {psnr_after_train:.3f}')
    return output, image_test, image_gt


def get_loss_r(mode='train'):
    '''
    mode = 'train' or 'production'
    '''
    loss_r = {}
    if 'train' in mode or 'T' in mode:        
        loss_r['vgg_identity'] = 1           # (model_outputs vs. label); "0" for "Production", "1" for trainning
        loss_r['vgg_fit'] = 1                # (fitted_image vs. label); "1" for both "trainning" and "production"
        loss_r['vgg_1st_last'] = 1         # (model_outputs[0] vs. model_outputs[-1]); "1e2" for both "trainning" and "production"

        loss_r['mse_identity_img'] = 1           # (model_outputs vs. label); "0" for "Production", "1" for trainning
        loss_r['mse_identity_bkg'] = 1
        loss_r['mse_identity_all'] = 1 
        loss_r['mse_fit_coef'] = 1           # (fit_coef_from_model_outputs vs. fit_coef_from_label); "1e8" for both "trainning" and "production"
        loss_r['mse_fit_self_consist'] = 1   # (fitting_output_from_model_output vs. model_outputs ); "1" for both "trainning" and "production"
      
        loss_r['bce_loss_gen'] = 1e-3          # cross-entropy loss from GAN
        loss_r['bce_loss_disc'] = 1
        loss_r['tv_fit_bkg'] = 0           # (total_variance_of: [fitted_bkg_outside_particle - label_outside_particle]); "1e-7" both "trainning" and "production"
        loss_r['l1_identity'] = 1  
    return loss_r


def img_psnr(img1, img2, d_range=None):
    
    if d_range is None:
        data_range = max(np.max(img1), np.max(img2))
    else:
        data_range = d_range
    #return psnr(img1.astype(np.float32), img2.astype(np.float32), data_range=data_range)
    return psnr(img1.astype(np.float32), img2.astype(np.float32))


def img_ssim(img_ref, img):
    
    return ssim(img_ref, img, data_range=img.max()-img.min())


def rm_noise(img, noise_level=0.001, filter_size=3, patten='abs'):
    '''
    patten: 
        'abs': take absolute difference
        'positive: only +error > "noise_level" will be filtered out
        'negative: only -error > "noise_level" will be filtered out
        
    '''
    img_s = medfilt2d(img, filter_size)
    id0 = img_s==0
    img_s[id0] = img[id0]
    if 'abs' in patten:
        img_diff = np.abs(img - img_s)/img_s
    if 'pos' in patten:
        img_diff = (img - img_s)/img_s
    if 'neg' in patten:
        img_diff = -(img - img_s)/img_s
    
    index = (img_diff > noise_level)
    img_m = img.copy()
    img_m[index] = img_s[index]
    return img_m


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


def plot_h_loss(h, method='log'):    
    n = len(h.keys())
    n_col = int(np.ceil(np.sqrt(n)))
    n_row = int(np.ceil(n/n_col))
    plt.figure(figsize=(16, 10))
    i = 1
    for k in h.keys():
        try:
            data = h[k]['value']
            rate = h[k]['rate']
        except:
            data = h[k]
            rate = [-1]
        plt.subplot(n_row, n_col, i)
        plt.plot(data, '.-')
        
        if method == 'log':
            plt.yscale('log')
        if rate[0] <1e3 and rate[0] > 1e-3:
            plt.title(f'{k} (r = {rate[0]:3.1f})')
        elif rate[0] == 0:
            plt.title(f'{k} (r = 0)')
        elif rate[0] == -1:
            plt.yscale('linear')
            plt.title('psnr')
        else:
            plt.title(f'{k} (r = {rate[0]:.1e})')
        i += 1
    plt.show()
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, hspace=0.4, wspace=0.45)


def extract_h_loss(h_loss, loss_summary, loss_r):
    txt = ''

    for k in h_loss.keys():
        if 'psnr' in k:
            continue
        value = loss_summary[k]
        prefix = 'Y' if loss_r[k] else 'N'
        rate = loss_r[k]
        
        h_loss[k]['value'].append(value)
        h_loss[k]['rate'].append(rate)
        txt = txt + f' ({prefix}) ({rate:.1e}) {k} = {value:.3e}\n'

    for k in loss_summary.keys():
        if 'psnr' in k:
            if not check_dic_key(h_loss, k):
                h_loss[k] = []
            h_loss[k].append(loss_summary[k])
            txt = txt + f'{k} = {loss_summary[k]:2.3f}\n'
    
    current_psnr = loss_summary["psnr"]
    return h_loss, txt, current_psnr


def check_dic_key(dic, key):
    if key in dic:
        return True
    else:
        return False


def show_tensor(img, idx=0):
    if len(img.shape) == 2:
        try:
            t = img.detach().cpu().numpy()
        except:
            t = img
    else:
        try:
            t = img[idx].detach().cpu().numpy()
        except:
            t = img[idx]
    t = np.squeeze(t)
    plt.figure()
    plt.imshow(t)


def load_json(fn_json):
    with open(fn_json) as js:
        t = json.load(js)
    return t


def plot_loss(h_loss):
    keys = list(h_loss.keys())
    n = len(keys)
    n_r = int(floor(np.sqrt(n)))
    n_c = int(ceil(n / n_r))
    fig, axes = plt.subplots(nrows=n_r, ncols=n_c)
    for r in range(n_r):
        for c in range(n_c):
            idx = r * n_r + c
            if idx >= n:
                break
            k = keys[idx]
            rate = np.array(h_loss[k]['rate'])
            rate[rate==0] = 1
            val = np.array(h_loss[k]['value'])
            val_scale = val / rate
            axes[r, c].plot(val_scale, '-', label=k)
            axes[r, c].legend()



def get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, num, transform_gt=None, transform_blur=None, split_ratio=0.8):
    dataset = xanesDataset(blur_dir, gt_dir, eng_dir, num, transform_gt, transform_blur)
    n = len(dataset)
    batch_size = 1     # for image_size (8, 512, 512)
    #split_ratio = 0.8
    n_train = int(split_ratio * n)
    n_valid = n - n_train

    #train_ds = torch.utils.data.Subset(dataset, range(n_train)) # read sequencially
    train_ds, valid_ds = torch.utils.data.random_split(dataset, (n_train, n_valid))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    print(f'train dataset = {len(train_loader.dataset)}')
    print(f'valid dataset = {len(valid_loader.dataset)}')
    return train_loader, valid_loader


def find_nearest(data, x):
    tmp = np.abs(data-x)
    return np.argmin(tmp)


def mkdirs(fn_folder):
    if not os.path.exists(fn_folder):
        os.makedirs(fn_folder)


def prepare_production_training_dataset(fn_img_xanes, elem, eng, eng_edge=[], num_img=50, f_norm=1.0, n_stack=16):
    try:
        xraylib.SymbolToAtomicNumber(elem)
    except:
        raise Exception(f'Unrecognized element symbol: {elem}')

    f_root = fn_img_xanes.split('.')[0] + '_train_production'
    mkdirs(f_root + '/img_gt_stack')
    mkdirs(f_root + '/img_blur_stack')
    mkdirs(f_root + '/img_eng_list')
    mkdirs(f_root + '/model_saved')
    mkdirs(f_root + '/model_output')

    img_xanes = io.imread(fn_img_xanes)
    img_xanes = img_xanes / f_norm
    n = len(eng)

    '''
    if not len(eng_edge) == 2:
        edge = xraylib.EdgeEnergy(xraylib.SymbolToAtomicNumber(elem), xraylib.K_SHELL)
        eng_edge = [edge-0.02, edge+0.1]
    '''
    if len(eng_edge) == 2:
        id1 = find_nearest(eng, eng_edge[0])
        id2 = find_nearest(eng, eng_edge[1])

        for i in trange(num_img):
            n1 = random.randint(5, 7)
            #n2 = 16 - n1
            n2 = n_stack - n1
            idx = np.sort(random.sample(range(id1), n1))
            idy = np.sort(random.sample(range(id2, n-1), n2))
            id_comb = list(idx) + list(idy)
            eng_list = eng[id_comb]
            img_list = img_xanes[id_comb]

            fsave_gt = f_root + f'/img_gt_stack/img_gt_stack_{i:04d}.tiff'
            fsave_bl = f_root + f'/img_blur_stack/img_blur_stack_{i:04d}.tiff'
            fsave_en = f_root + f'/img_eng_list/eng_list_{i:04d}_{elem}.txt'
            io.imsave(fsave_gt, img_list.astype(np.float32))
            io.imsave(fsave_bl, img_list.astype(np.float32))
            np.savetxt(fsave_en, eng_list)

    else: # using all energy, e.g., when reference spectra are available
        for i in trange(num_img):
            id_comb = np.sort(random.sample(range(n), min(n, n_stack)))
            eng_list = eng[id_comb]
            img_list = img_xanes[id_comb]
            fsave_gt = f_root + f'/img_gt_stack/img_gt_stack_{i:04d}.tiff'
            fsave_bl = f_root + f'/img_blur_stack/img_blur_stack_{i:04d}.tiff'
            fsave_en = f_root + f'/img_eng_list/eng_list_{i:04d}_{elem}.txt'
            io.imsave(fsave_gt, img_list.astype(np.float32))
            io.imsave(fsave_bl, img_list.astype(np.float32))
            np.savetxt(fsave_en, eng_list)