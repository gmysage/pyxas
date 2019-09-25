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


def find_rot(fn, thresh=0.05):
    
    f = h5py.File(fn, 'r')
    img_bkg = np.squeeze(np.array(f['img_bkg_avg']))
    img_dark = np.squeeze(np.array(f['img_dark_avg']))
    ang = np.array(list(f['angle']))
    
    tmp = np.abs(ang - ang[0] -180).argmin() 
    img0 = np.array(list(f['img_tomo'][0]))
    img180_raw = np.array(list(f['img_tomo'][tmp]))
    f.close()

    img0 = (img0 -img_dark) / (img_bkg - img_dark)
    img180_raw = (img180_raw - img_dark) / (img_bkg- img_dark)
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
    return rot_cen


def rotcen_test(fn, start=None, stop=None, steps=None, sli=0, block_list=[]):
    
    f = h5py.File(fn)
    tmp = np.array(f['img_bkg_avg'])
    s = tmp.shape
    if sli == 0: sli = int(s[1]/2)
    img_tomo = np.array(f['img_tomo'][:, sli, :])
    img_bkg = np.array(f['img_bkg_avg'][:, sli, :])
    img_dark = np.array(f['img_dark_avg'][:, sli, :])
    theta = np.array(f['angle']) / 180.0 * np.pi
    f.close()
    prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0
    s = prj_norm.shape
    prj_norm = prj_norm.reshape(s[0], 1, s[1])
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]
    if start==None or stop==None or steps==None:
        start = int(s[1]/2-50)
        stop = int(s[1]/2+50)
        steps = 31
    cen = np.linspace(start, stop, steps)
    img = np.zeros([len(cen), s[1], s[1]])
    for i in range(len(cen)):
        print('{}: rotcen {}'.format(i+1, cen[i]))
        img[i] = tomopy.recon(prj_norm, theta, center=cen[i], algorithm='gridrec')
    fout = 'center_test.h5'
    with h5py.File(fout, 'w') as hf:
        hf.create_dataset('img', data=img)
        hf.create_dataset('rot_cen', data=cen)



def recon_sub(img, theta, rot_cen, block_list=[], rm_stripe=False, stripe_remove_level=9):
    prj_norm = img
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]
    if rm_stripe:
        prj_norm = tomopy.prep.stripe.remove_stripe_fw(prj_norm, lepyxvel=rm_stripe_level, wname='db5', sigma=1, pad=True)
        #prj_norm = tomopy.prep.stripe.remove_all_stripe_tomo(prj_norm, 3, 81, 31)
    rec = tomopy.recon(prj_norm, theta, center=rot_cen, algorithm='gridrec')
    return rec




def recon(fn, rot_cen, sli=[], col=[], binning=None, zero_flag=0, tiff_flag=0, block_list=[], rm_stripe=True, stripe_remove_level=9):
    '''
    reconstruct 3D tomography
    Inputs:
    --------  
    fn: string
        filename of scan, e.g. 'fly_scan_0001.h5'
    rot_cen: float
        rotation center
    algorithm: string
        choose from 'gridrec' and 'mlem'
    sli: list
        a range of slice to recontruct, e.g. [100:300]
    num_iter: int
        iterations for 'mlem' algorithm
    bingning: int
        binning the reconstruted 3D tomographic image 
    zero_flag: bool 
        if 1: set negative pixel value to 0
        if 0: keep negative pixel value
        
    '''
    import tomopy
    from PIL import Image
    f = h5py.File(fn)
    tmp = np.array(f['img_bkg_avg'])
    s = tmp.shape
    slice_info = ''
    bin_info = ''
    col_info = ''


    if len(sli) == 0:
        sli = [0, s[1]]
    elif len(sli) == 1 and sli[0] >=0 and sli[0] <= s[1]:
        sli = [sli[0], sli[0]+1]
        slice_info = '_slice_{}_'.format(sli[0])
    elif len(sli) == 2 and sli[0] >=0 and sli[1] <= s[1]:
        slice_info = '_slice_{}_{}_'.format(sli[0], sli[1])
    else:
        print('non valid slice id, will take reconstruction for the whole object')
    
    if len(col) == 0:
        col = [0, s[2]]
    elif len(col) == 1 and col[0] >=0 and col[0] <= s[2]:
        col = [col[0], col[0]+1]
        col_info = '_col_{}_'.format(col[0])
    elif len(col) == 2 and col[0] >=0 and col[1] <= s[2]:
        col_info = 'col_{}_{}_'.format(col[0], col[1])
    else:
        col = [0, s[2]]
        print('invalid col id, will take reconstruction for the whole object')

    rot_cen = rot_cen - col[0]
    
    scan_id = np.array(f['scan_id'])
    img_tomo = np.array(f['img_tomo'][:, sli[0]:sli[1], :])
    img_tomo = np.array(img_tomo[:, :, col[0]:col[1]])
    img_bkg = np.array(f['img_bkg_avg'][:, sli[0]:sli[1], col[0]:col[1]])
    img_dark = np.array(f['img_dark_avg'][:, sli[0]:sli[1], col[0]:col[1]])
    theta = np.array(f['angle']) / 180.0 * np.pi
    pos_180 = pyxas.find_nearest(theta, theta[0]+np.pi)
    block_list = list(block_list) + list(np.arange(pos_180+1, len(theta)))

    eng = np.array(f['X_eng'])
    f.close() 

    s = img_tomo.shape
    if not binning == None:
        img_tomo = bin_ndarray(img_tomo, (s[0], int(s[1]/binning), int(s[2]/binning)), 'sum')
        img_bkg = bin_ndarray(img_bkg, (1, int(s[1]/binning), int(s[2]/binning)), 'sum')
        img_dark = bin_ndarray(img_dark, (1, int(s[1]/binning), int(s[2]/binning)), 'sum')
        rot_cen = rot_cen * 1.0 / binning 
        bin_info = 'bin{}'.format(int(binning))

    prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0   

    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]

    if rm_stripe:
        prj_norm = tomopy.prep.stripe.remove_stripe_fw(prj_norm,level=stripe_remove_level, wname='db5', sigma=1, pad=True)
    fout = 'recon_scan_' + str(scan_id) + str(slice_info) + str(col_info) + str(bin_info)

    if tiff_flag:
        cwd = os.getcwd()
        try:
            os.mkdir(cwd+f'/{fout}')
        except:
            print(cwd+f'/{fout} existed')
        for i in range(prj_norm.shape[1]):
            print(f'recon slice: {i:04d}/{prj_norm.shape[1]-1}')
            rec = tomopy.recon(prj_norm[:, i:i+1,:], theta, center=rot_cen, algorithm='gridrec')
            
            if zero_flag:
                rec[rec<0] = 0
            fout_tif = cwd + f'/{fout}' + f'/{i+sli[0]:04d}.tiff' 
            io.imsave(fout_tif, rec[0])
            #img = Image.fromarray(rec[0])
            #img.save(fout_tif)
    else:
        rec = tomopy.recon(prj_norm, theta, center=rot_cen, algorithm='gridrec')
        if zero_flag:
            rec[rec<0] = 0
        fout_h5 = fout +'.h5'
        with h5py.File(fout_h5, 'w') as hf:
            hf.create_dataset('img', data=rec)
            hf.create_dataset('scan_id', data=scan_id)        
            hf.create_dataset('X_eng', data=eng)
        print('{} is saved.'.format(fout)) 
    del rec
    del img_tomo
    del prj_norm




def batch_recon(file_path='.', file_prefix='fly', file_type='.h5', sli=[], col=[], block_list=[], binning=1, rm_stripe=True, stripe_remove_level=9):
    path = os.path.abspath(file_path)
    files = pyxas.retrieve_file_type(file_path, file_prefix, file_type)
    num_file = len(files)
    for i in range(num_file):
        fn = files[i].split('/')[-1]
        tmp = pyxas.get_img_from_hdf_file(fn, 'angle')
        angle = tmp['angle']
        pos_180 = pyxas.find_nearest(angle, angle[0]+180)
        block_list = list(block_list) + list(np.arange(pos_180+1, len(angle)))
        rotcen = pyxas.find_rot(fn)
        pyxas.recon(fn, rotcen, binning=binning, sli=sli, col=col, block_list=block_list, rm_stripe=rm_stripe, stripe_remove_level=stripe_remove_level)



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

