from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import h5py
import tomopy 
import os
import tomopy
import pyxas
from pyxas.misc import *
from scipy.signal import medfilt2d
from skimage import io
from PIL import Image


def find_nearest(data, value):
    data = np.array(data)
    return np.abs(data - value).argmin()


def find_rot(fn, thresh=0.05, method=1):
    from pystackreg import StackReg
    sr = StackReg(StackReg.TRANSLATION) 
    f = h5py.File(fn, 'r')
    ang = np.array(list(f['angle']))
    img_bkg = np.squeeze(np.array(f['img_bkg_avg']))
    if np.abs(ang[0]) < np.abs(ang[0]-90): # e.g, rotate from 0 - 180 deg
        tmp = np.abs(ang - ang[0] -180).argmin() 
    else: # e.g.,rotate from -90 - 90 deg
        tmp = np.abs(ang - np.abs(ang[0])).argmin() 
    img0 = np.array(list(f['img_tomo'][0]))       
    img180_raw = np.array(list(f['img_tomo'][tmp]))
    f.close()
    img0 = img0 / img_bkg
    img180_raw = img180_raw / img_bkg
    img180 = img180_raw[:,::-1] 
    s = np.squeeze(img0.shape)
    im1 = -np.log(img0)
    im2 = -np.log(img180)
    im1[np.isnan(im1)] = 0
    im2[np.isnan(im2)] = 0
    im1[im1 < thresh] = 0
    im2[im2 < thresh] = 0
    im1 = medfilt2d(im1,5)
    im2 = medfilt2d(im2, 5)
    im1_fft = np.fft.fft2(im1)
    im2_fft = np.fft.fft2(im2)
    results = dftregistration(im1_fft, im2_fft)
    row_shift = results[2]
    col_shift = results[3]
    rot_cen = s[1]/2 + col_shift/2 - 1 

    tmat = sr.register(im1, im2) 
    rshft = -tmat[1, 2]
    cshft = -tmat[0, 2]
    rot_cen0 = s[1]/2 + cshft/2 - 1

    print(f'rot_cen = {rot_cen} or {rot_cen0}')
    if method:
        return rot_cen
    else:
        return rot_cen0


def rotcen_test(fn, start=None, stop=None, steps=None, sli=0, block_list=[], return_flag=0, print_flag=1, bkg_level=0, txm_normed_flag=0, denoise_flag=0, fw_level=9, save_flag=0):  
    import tomopy 
    f = h5py.File(fn, 'r')
    tmp = np.array(f['img_tomo'][0])
    s = [1, tmp.shape[0], tmp.shape[1]]
    if denoise_flag:
        addition_slice = 100
    else:
        addition_slice = 0

    if sli == 0: sli = int(s[1]/2)
    sli_exp = [np.max([0, sli-addition_slice//2]), np.min([sli+addition_slice//2+1, s[1]])]

    theta = np.array(f['angle']) / 180.0 * np.pi
    
    img_tomo = np.array(f['img_tomo'][:, sli_exp[0]:sli_exp[1], :])     
    
    if txm_normed_flag:
        prj = img_tomo
    else:
        img_bkg = np.array(f['img_bkg_avg'][:, sli_exp[0]:sli_exp[1], :])
        img_dark = np.array(f['img_dark_avg'][:, sli_exp[0]:sli_exp[1], :])
        prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    f.close()
    
    prj = denoise(prj, denoise_flag)
    
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0    

    prj_norm -= bkg_level

    prj_norm = tomopy.prep.stripe.remove_stripe_fw(prj_norm,level=fw_level, wname='db5', sigma=1, pad=True)

    s = prj_norm.shape  
    if len(s) == 2:
        prj_norm = prj_norm.reshape(s[0], 1, s[1])
        s = prj_norm.shape    

    pos = find_nearest(theta, theta[0]+np.pi)
    block_list = list(block_list) + list(np.arange(pos+1, len(theta)))
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]
    if start==None or stop==None or steps==None:
        start = int(s[2]/2-50)
        stop = int(s[2]/2+50)
        steps = 26
    cen = np.linspace(start, stop, steps)          
    img = np.zeros([len(cen), s[2], s[2]])
    for i in range(len(cen)):
        if print_flag:
            print('{}: rotcen {}'.format(i+1, cen[i]))
        img[i] = tomopy.recon(prj_norm[:, addition_slice:addition_slice+1], theta, center=cen[i], algorithm='gridrec')    
    if save_flag:
        fout = 'center_test.h5'
        with h5py.File(fout, 'w') as hf:
            hf.create_dataset('img', data=img)
            hf.create_dataset('rot_cen', data=cen)
    img = tomopy.circ_mask(img, axis=0, ratio=0.8)
    tracker = image_scrubber(img)
    if return_flag:
        return img, cen

def img_variance(img):
    import tomopy
    s = img.shape
    variance = np.zeros(s[0])
    img = tomopy.circ_mask(img, axis=0, ratio=0.8)
    for i in range(s[0]):
        img[i] = medfilt2d(img[i], 5)
        img_ = img[i].flatten()
        t = img_>0
        img_ = img_[t]
        t = np.mean(img_)
        variance[i] = np.sqrt(np.sum(np.power(np.abs(img_ - t), 2))/len(img_-1))
    return variance


def recon(fn, rot_cen, sli=[], binning=None, zero_flag=0, block_list=[], bkg_level=0, txm_normed_flag=0, read_full_memory=0, denoise_flag=0, fw_level=9, ncore=None):
    '''
    reconstruct 3D tomography
    Inputs:
    --------  
    fn: string
        filename of scan, e.g. 'fly_scan_0001.h5'
    rot_cen: float
        rotation center
    sli: list
        a range of slice to recontruct, e.g. [100:300]
    bingning: int
        binning the reconstruted 3D tomographic image 
    zero_flag: bool 
        if 1: set negative pixel value to 0
        if 0: keep negative pixel value
    block_list: list
        a list of index for the projections that will not be considered in reconstruction
    denoise_flag: int
        0: no denoising on projection image
        1: wiener denoising
        2: gaussian denoising   
    '''
    
    from PIL import Image
    f = h5py.File(fn, 'r')
    tmp = np.array(f['img_tomo'][0])
    s = [1, tmp.shape[0], tmp.shape[1]]
    slice_info = ''
    bin_info = ''
    col_info = ''
    sli_step = 40
    if len(sli) == 0:
        sli = [0, s[1]]
        sli[1] = int((sli[1] - sli[0]) // sli_step * sli_step)
    elif len(sli) == 1 and sli[0] >=0 and sli[0] <= s[1]:
        sli = [sli[0], sli[0]+1]
        slice_info = '_slice_{}'.format(sli[0])
    elif len(sli) == 2 and sli[0] >=0 and sli[1] <= s[1]:
        sli[1] = int((sli[1] - sli[0]) // sli_step * sli_step) + sli[0]
        slice_info = '_slice_{}_{}'.format(sli[0], sli[1])
    else:
        print('non valid slice id, will take reconstruction for the whole object')  
          
    '''
    if len(col) == 0:
        col = [0, s[2]]
    elif len(col) == 1 and col[0] >=0 and col[0] <= s[2]:
        col = [col[0], col[0]+1]
        col_info = '_col_{}'.format(col[0])
    elif len(col) == 2 and col[0] >=0 and col[1] <= s[2]:
        col_info = '_col_{}_{}'.format(col[0], col[1])
    else:
        col = [0, s[2]]
        print('invalid col id, will take reconstruction for the whole object')
    '''
    #rot_cen = rot_cen - col[0] 
    scan_id = np.array(f['scan_id'])
    theta = np.array(f['angle']) / 180.0 * np.pi
    eng = np.array(f['X_eng'])

    pos = find_nearest(theta, theta[0]+np.pi)
    block_list = list(block_list) + list(np.arange(pos+1, len(theta)))
    allow_list = list(set(np.arange(len(theta))) - set(block_list))
    theta = theta[allow_list]
    tmp = np.squeeze(np.array(f['img_tomo'][0]))
    s = tmp.shape
    f.close()

    
    sli_total = np.arange(sli[0], sli[1])
    binning = binning if binning else 1
    bin_info = f'_bin_{binning}'
  
    n_steps = int(len(sli_total) / sli_step)
    rot_cen = rot_cen * 1.0 / binning 

    if read_full_memory:
        sli_step = sli[1] - sli[0]
        n_steps = 1

    # optional
    if denoise_flag:
        add_slice = min(sli_step // 2, 20)
    else:
        add_slice = 0

    try:
        rec = np.zeros([sli_step*n_steps // binning, s[1] // binning, s[1] // binning], dtype=np.float32)
    except:
        print('Cannot allocate memory')


    for i in range(n_steps):    
        if i == 0:
            sli_sub = [sli_total[0], sli_total[0]+sli_step] 
            current_sli = sli_sub
        elif i == n_steps-1:
            sli_sub = [i*sli_step+sli_total[0], len(sli_total)+sli[0]]
            current_sli = sli_sub
        else:
            sli_sub = [i*sli_step+sli_total[0], (i+1)*sli_step+sli_total[0]]
            current_sli = [sli_sub[0]-add_slice, sli_sub[1]+add_slice] 
        print(f'recon {i+1}/{n_steps}:    sli = [{sli_sub[0]}, {sli_sub[1]}] ... ')

        prj_norm = proj_normalize(fn, current_sli, txm_normed_flag, binning, allow_list, bkg_level, fw_level=fw_level, denoise_flag=denoise_flag)       
        
        if i!=0 and i!=n_steps-1:
            prj_norm = prj_norm[:, add_slice//binning:sli_step//binning+add_slice//binning]
        rec_sub = tomopy.recon(prj_norm, theta, center=rot_cen, algorithm='gridrec', ncore=ncore)
        rec[i*sli_step // binning : i*sli_step // binning + rec_sub.shape[0]] = rec_sub

    bin_info = f'_bin{int(binning)}'  
    fout = f'recon_scan_{str(scan_id)}{str(slice_info)}{str(bin_info)}'
    if zero_flag:
        rec[rec<0] = 0
    fout_h5 = f'{fout}.h5'
    with h5py.File(fout_h5, 'w') as hf:
        hf.create_dataset('img', data=np.array(rec, dtype=np.float32))
        hf.create_dataset('scan_id', data=scan_id)        
        hf.create_dataset('X_eng', data=eng)
        hf.create_dataset('rot_cen', data=rot_cen)
        hf.create_dataset('binning', data=binning)
    print(f'{fout} is saved.') 
    del rec
    #del img_tomo
    del prj_norm


def denoise(prj, denoise_flag):
    if denoise_flag == 1:  # Wiener denoise
        import skimage.restoration as skr
        ss = prj.shape
        psf = np.ones([2, 2])/(2 ** 2)
        reg = None
        balance = 0.3
        is_real = True
        clip = True
        for j in range(ss[0]):
            prj[j] = skr.wiener(prj[j], psf=psf, reg=reg, balance=balance, is_real=is_real, clip=clip)        
    elif denoise_flag == 2:  # Gaussian denoise
        from skimage.filters import gaussian as gf
        prj = gf(prj, [0, 1, 1])
    return prj


def proj_normalize(fn, sli, txm_normed_flag, binning, allow_list=[], bkg_level=0, fw_level=9, denoise_flag=0):
    f = h5py.File(fn, 'r')
    img_tomo = np.array(f['img_tomo'][:, sli[0]:sli[1], :])
    try:
        img_bkg = np.array(f['img_bkg_avg'][:, sli[0]:sli[1]])
    except:
        img_bkg = []
    try:
        img_dark = np.array(f['img_dark_avg'][:, sli[0]:sli[1]])
    except:
        img_dark = []
    if len(img_dark) == 0 or len(img_bkg) == 0 or txm_normed_flag == 1:
        prj = img_tomo
    else:
        prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    prj = denoise(prj, denoise_flag)
    s = prj.shape   
    prj = bin_ndarray(prj, (s[0], int(s[1]/binning), int(s[2]/binning)), 'mean')
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0    
    prj_norm = prj_norm[allow_list]       
    prj_norm = tomopy.prep.stripe.remove_stripe_fw(prj_norm,level=fw_level, wname='db5', sigma=1, pad=True)
    prj_norm -= bkg_level
    f.close()
    del img_tomo
    del img_bkg
    del img_dark
    del prj
    return prj_norm



def batch_recon(fs, cen_list, sli=[], block_list=[], binning=1, denoise_flag=0):
    num_file = len(fs)
    for i in range(num_file):
        rotcen = cen_list[i]
        fn = fs[i]
        print(f'recon {fn.split("/")[-1]} ...')
        recon(fn, rotcen, binning=binning, sli=sli, block_list=block_list, denoise_flag=denoise_flag)



def batch_find_rotcen(files, block_list, index=0):
    img = []
    r = []
    for i in range(len(files)):
        fn = files[i]
        r.append(find_rot(fn))
        print(f'#{i} {fn}: rotcen = {r[-1]}')
        tmp = recon(fn, r[-1], sli=[index], block_list=block_list, binning=None, tiff_flag=0, h5_flag=0, return_flag=1)
        img.append(np.squeeze(tmp))
    img = np.array(img, dtype=np.float32)
    with h5py.File('batch_rotcen.h5', 'w') as hf:
        hf.create_dataset('img', data = img)
        hf.create_dataset('rotcen', data = r)
    return img, r

