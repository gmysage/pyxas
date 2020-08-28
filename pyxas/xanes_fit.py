from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import os
import h5py
import pyxas
import scipy
import time
from scipy import ndimage
from skimage import io
from copy import deepcopy
from pyxas.align3D import *
from pyxas.xanes_util import *
from pyxas.misc import *
from pyxas.colormix import *


def load_xanes_ref_file(*args):
    num_ref = len(args)
    spectrum_ref ={}
    for i in range(num_ref):
        spectrum_ref[f'ref{i}'] = np.loadtxt(args[i])
    return spectrum_ref



def fit_2D_xanes(img_xanes, xanes_eng, spectrum_ref, fit_param):
   
    res = {}
    fit_eng = fit_param['fit_eng']
    norm_txm_flag = fit_param['norm_txm_flag']
    fit_post_edge_flag = fit_param['fit_post_edge_flag']
    try:
        fit_pre_edge_flag = fit_param['fit_pre_edge_flag']
    except:
        fit_pre_edge_flag = 1
    if fit_pre_edge_flag:
        pre_edge = fit_param['pre_edge']
    else:
        pre_edge = []
    if fit_post_edge_flag:
        post_edge = fit_param['post_edge']
    else:
        post_edge = []
    
    align_flag = fit_param['align_flag']
    roi_ratio = fit_param['roi_ratio'] 
    align_ref_index = fit_param['align_ref_index']
 
    fit_iter_flag = fit_param['fit_iter_flag']
    if fit_iter_flag:
        fit_iter_learning_rate = fit_param['fit_iter_learning_rate']
        fit_iter_num = int(fit_param['fit_iter_num'])
        fit_iter_bound = fit_param['fit_iter_bound']
        try:
            fit_iter_lambda = fit_param['fit_iter_lambda']
        except:
            fit_iter_lambda = 0

    regulation_flag = fit_param['regulation_flag']
    if regulation_flag:
        regulation_designed_max = fit_param['regulation_designed_max']
        regulation_gamma = fit_param['regulation_gamma']
    
    mask_xanes_flag = fit_param['mask_xanes_flag']  
    norm_edge_method = fit_param['norm_edge_method']
    n_comp = fit_param['n_comp'] if mask_xanes_flag else 1


    # optional: aligning xanes_image_stack
    img = img_xanes.copy()
    s = img.shape
    if roi_ratio >= 1:
        img_mask = None
    else:
        rs, re = 1-roi_ratio, roi_ratio
        img_mask = img_xanes[:, int(s[1]*rs):int(s[1]*re), int(s[2]*rs):int(s[2]*re)]
    if align_ref_index == -1:
        align_ref_index = img.shape[0] - 1
    if align_flag == 1: # using stackreg method
        img = pyxas.align_img_stack_stackreg(img_xanes, img_mask, select_image_index=align_ref_index)  
    elif align_flag == 2:
        img = pyxas.align_img_stack(img_xanes, img_mask, select_image_index=align_ref_index)
    else:
        img = img_xanes
    #tmp_mask = np.ones([img.shape[1], img.shape[2]])
    #mask_comp = {}
    #mask_comp['0'] = tmp_mask

    if norm_txm_flag:
        img = pyxas.norm_txm(img)
    # mask components using kmean_mask
    if mask_xanes_flag:
        n_comp = int(np.max([n_comp, 2]))
        img_xanes_norm_smooth = pyxas.medfilt(img, 3)
        mask_comp, _ = kmean_mask(img_xanes_norm_smooth[20:50], n_comp=n_comp)
    else:
        mask_comp = np.ones([1, img.shape[1], img.shape[2]])
        n_comp = 1
     #   if n_comp ==  2:
     #       tmp_mask = mask_comp[0]
     #   else:    
     #       tmp_mask = 1 - mask_comp[n_comp-1]

    # normalize edge
    img_xanes_norm, img_thickness, fit_eng_index = pyxas.fit_xanes2D_norm_edge(img, xanes_eng, pre_edge, post_edge, fit_eng, 0, fit_pre_edge_flag, fit_post_edge_flag, norm_method=norm_edge_method)

    # regularize xanes image
    if regulation_flag:
        img_xanes_norm = pyxas.normalize_2D_xanes_regulation(img_xanes_norm, xanes_eng, pre_edge, post_edge, regulation_designed_max, regulation_gamma)


    # fitting: non-iter
    xanes_2d_fit, xanes_2d_fit_offset, xanes_fit_cost = pyxas.fit_2D_xanes_non_iter(img_xanes_norm[fit_eng_index[0]:fit_eng_index[1]], xanes_eng[fit_eng_index[0]:fit_eng_index[1]], spectrum_ref)

    # fitting: iter
    if fit_iter_flag:
        xanes_2d_fit, xanes_2d_fit_offset, xanes_fit_cost = pyxas.fit_2D_xanes_iter(img_xanes_norm[fit_eng_index[0]:fit_eng_index[1]],
                                                                                    xanes_eng[fit_eng_index[0]:fit_eng_index[1]],
                                                                                    spectrum_ref, coef0=xanes_2d_fit, offset=xanes_2d_fit_offset,
                                                                                    learning_rate=fit_iter_learning_rate,
                                                                                    n_iter=fit_iter_num, bounds=fit_iter_bound,
                                                                                    fit_iter_lambda=fit_iter_lambda)

    # normalize fitting to [0, 1]
    xanes_2d_fit_norm = pyxas.rm_abnormal(xanes_2d_fit / np.sum(xanes_2d_fit, axis=0))
    xanes_2d_fit_norm[xanes_2d_fit_norm > 1] = 1
    xanes_2d_fit_norm[xanes_2d_fit_norm < 0] = 0
    

    res['xanes_fit_thickness'] = img_thickness
    res['xanes_2d_fit'] = xanes_2d_fit
    res['xanes_2d_fit_norm'] = xanes_2d_fit_norm
    res['xanes_fit_cost'] = xanes_fit_cost
    res['img_xanes_norm'] = img_xanes_norm
    res['xanes_2d_fit_offset'] = xanes_2d_fit_offset
    res['n_comp'] = n_comp 
    for i in range(n_comp):
        res[f'mask_{i}'] = mask_comp[i]
    return res



def save_xanes_fitting_image(res, file_save_path, fn, color='r,g,b'):
    file_save_colormix1 = f'{file_save_path}/colormix_concentration'
    file_save_colormix2 = f'{file_save_path}/colormix_ratio'
    file_save_fit_cost = f'{file_save_path}/fitting_cost'
    file_save_thickness = f'{file_save_path}/fitting_thickness'    
    file_save_mask = f'{file_save_path}/fitting_mask'
    file_save_fit = f'{file_save_path}/fitting_result'
    file_save_fit_norm = f'{file_save_path}/fitting_result_norm'
    
    create_directory(file_save_path)
    create_directory(file_save_colormix1)
    create_directory(file_save_colormix2)
    create_directory(file_save_fit_cost)
    create_directory(file_save_thickness)
    create_directory(file_save_mask)
    create_directory(file_save_fit)    
    create_directory(file_save_fit_norm)    
    create_directory(f'{file_save_mask}/mask')   
    for n in range(res['n_comp']):
        create_directory(f'{file_save_mask}/mask_{n}')

    fn = fn.split('/')[-1].split('.')[0]

    # save to jpg image
    tmp_concentration = res['xanes_2d_fit_norm'] * res['xanes_fit_thickness']
    tmp_ratio = res['xanes_2d_fit_norm']
    tmp_f_concentration = pyxas.medfilt(tmp_concentration, 3)
    tmp_f_ratio = pyxas.medfilt(tmp_ratio, 3)
    img_color_concentration = pyxas.colormix(tmp_concentration, color=color, clim=[0, np.max(tmp_f_concentration)])
    img_color_ratio = pyxas.colormix(tmp_ratio, color=color, clim=[0, 1])
    # mask1 = np.expand_dims(np.squeeze(res['mask_0']), axis=2)
    mask1 = np.expand_dims(np.squeeze(res['mask']), axis=2)
    mask1 = np.repeat(mask1, 3, axis=2)

    img_color_concentration *= mask1
    img_color_concentration[img_color_concentration<0] = 0
    if np.max(img_color_concentration) == 0:
        img_color_concentration[0,0,0] = 1
    fn_save = f'{file_save_colormix1}/colormix_concentration_{fn}.jpg'
    pyxas.toimage(img_color_concentration, cmin=0, cmax=1).save(fn_save)

    img_color_ratio *= mask1
    img_color_ratio[img_color_ratio<0] = 0
    if np.max(img_color_ratio) == 0:
        img_color_ratio[0,0,0] = 1
    fn_save = f'{file_save_colormix2}/colormix_ratio_{fn}.jpg'
    pyxas.toimage(img_color_ratio, cmin=0, cmax=1).save(fn_save)

    # save fitting cost
    fn_save = f'{file_save_fit_cost}/fitting_cost_{fn}.tiff'
    io.imsave(fn_save, np.array(res['xanes_fit_cost'], dtype=np.float32))
    # save fitting thickness
    fn_save = f'{file_save_thickness}/fitting_thickness_{fn}.tiff'
    io.imsave(fn_save, np.array(res['xanes_fit_thickness'], dtype=np.float32))
    # save fitting mask
    fn_save = f'{file_save_mask}/mask/mask_{fn}.tiff'
    io.imsave(fn_save, np.array(np.squeeze(res['mask']), dtype=np.int16))
    for n in range(res['n_comp']):
        fn_save = f'{file_save_mask}/mask_{n}/mask{0}_{fn}.tiff'
        io.imsave(fn_save, np.array(res[f'mask_{n}'], dtype=np.int16))
    # sae ftting results
    fn_save = f'{file_save_fit}/fit_{fn}.tiff'
    io.imsave(fn_save, np.array(res['xanes_2d_fit'], dtype=np.float32))
    fn_save = f'{file_save_fit_norm}/fit_norm_{fn}.tiff'
    io.imsave(fn_save, np.array(res['xanes_2d_fit_norm'], dtype=np.float32))





def fit_2D_xanes_file(file_path, file_prefix, file_type, fit_param, xanes_eng, spectrum_ref, file_range=[], save_hdf=1):
    '''
    batch processing xanes given xanes files
    '''
    time_start = time.time()
    n_comp = int(fit_param['n_comp'])

    file_save_path = f'{file_path}/fitted_xanes'
    pyxas.create_directory(file_save_path)

    files_scan = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)
    try:
        true_file_type = fit_param['file_type']
    except:
        true_file_type = 'tiff'
    if true_file_type == 'tiff' or true_file_type == 'tif':
        tmp = pyxas.get_img_from_tif_file(files_scan[0])
    elif true_file_type == 'h5' or files_scan[0].split('.')[-1] == 'h5':
        hdf_attr = fit_param['hdf_attr']
        tmp = pyxas.get_img_from_hdf_file(files_scan[0], hdf_attr)[hdf_attr]
    #tmp = pyxas.get_img_from_tif_file(files_scan[0])
    s = tmp.shape
    num_file = len(files_scan)
    num_channel = len(spectrum_ref)

    # initialization 
    thresh_thick = fit_param['fit_mask_thickness_threshold'] # thickness < thick_thresh will be 0
    thresh_cost = fit_param['fit_mask_cost_threshold'] # fit_error > cost_thresh will be 0
    fs, fe = 0, num_file
    if len(file_range):
        fs = int(np.max([fs, np.min(file_range)]))
        fe = int(np.min([fe, np.max(file_range)]))

    if save_hdf:
        thickness_3D = np.zeros([num_file, s[1], s[2]])   
        fitted_xanes_3D = np.zeros([num_channel, num_file, s[1], s[2]])
        fitted_xanes_3D_norm = fitted_xanes_3D.copy()
        mask_3D = np.ones([num_file, s[1], s[2]])     
        mask_3D_comp = np.ones([n_comp, num_file, s[1], s[2]])
        fitting_cost_3D = np.zeros([num_file, s[1], s[2]])
        fitting_offset_3D = np.zeros([num_file, s[1], s[2]])
    
    #### start file loop ####
    for i in range(num_file):
        if i < fs or i >= fe:
            print(f'skipping slice #{i+1}/{num_file}')
            continue
        time_s_sub = time.time()
        fn = files_scan[i].split('.')[0]
        print(f'fitting slice #{i+1}/{num_file} ...')
        try:
            true_file_type = fit_param['file_type']
        except:
            true_file_type = 'tiff'
        if true_file_type == 'tiff' or true_file_type == 'tif':
            img_xanes = pyxas.get_img_from_tif_file(files_scan[i])
        elif true_file_type == 'h5' or files_scan[0].split('.')[-1] == 'h5':
            hdf_attr = fit_param['hdf_attr']
            img_xanes = pyxas.get_img_from_hdf_file(files_scan[i], hdf_attr)[hdf_attr]
        #img_xanes = pyxas.get_img_from_tif_file(files_scan[i])
        res = pyxas.fit_2D_xanes(img_xanes, xanes_eng, spectrum_ref, fit_param)      

        ## mask
        mask = pyxas.fit_xanes2D_generate_mask(res['xanes_fit_thickness'], res['xanes_fit_cost'], thresh_cost, thresh_thick, num_iter=0)    
        res['mask'] = np.squeeze(mask)

        # save to jpg image
        try:
            color = fit_param['color']
        except:
            color = 'r, g, b'
        pyxas.save_xanes_fitting_image(res, file_save_path, files_scan[i], color)

        if save_hdf:
            mask_3D[i] = np.squeeze(mask)
            for n in range(res['n_comp']):        
                mask_3D_comp[n, i] = res[f'mask_{n}']
            thickness_3D[i] = np.squeeze(res['xanes_fit_thickness'])
            fitting_cost_3D[i] = np.squeeze(res['xanes_fit_cost'])
            fitting_offset_3D[i] = np.squeeze(res['xanes_2d_fit_offset'])
            for j in range(num_channel):
                fitted_xanes_3D[j, i] = np.squeeze(res['xanes_2d_fit'][j])  
                fitted_xanes_3D_norm[j, i] = np.squeeze(res['xanes_2d_fit_norm'][j])   
        print(f'time elapsed for #{i:4d}: {time.time()-time_start:6.1f}\n')
 
    ###### end of file loop ####
    print(f'time elapsed: {time.time()-time_start:6.1f}')
    
    if save_hdf:
        fn_save = f'{file_save_path}/fitted_3d_xanes.h5'
        print(f'saving fitted 3D xanes: {fn_save} ... ')
        with h5py.File(fn_save, 'w') as hf:
            hf.create_dataset('type', data = 'fit_3D_xanes')            
            hf.create_dataset('X_eng', data = xanes_eng)
            hf.create_dataset('pre_edge', data = fit_param['pre_edge'])
            hf.create_dataset('post_edge', data = fit_param['post_edge'])
            hf.create_dataset('unit', data = 'keV')
            hf.create_dataset('num_component', data = num_channel)
            if save_hdf:
                hf.create_dataset('xanes_fit_thickness', data=np.array(thickness_3D, dtype=np.float32))
                hf.create_dataset('xanes_fit_cost', data=np.array(fitting_cost_3D, dtype=np.float32))
                hf.create_dataset('xanes_fit_offset', data=np.array(fitting_offset_3D, dtype=np.float32))
                hf.create_dataset('mask', data=np.array(mask_3D, dtype=np.int8))
                for n in range(res['n_comp']):
                    hf.create_dataset(f'mask_{n}', data=np.array(mask_3D_comp[n], dtype=np.int8))
                for j in range(num_channel):
                    hf.create_dataset(f'xanes_fit_comp{j}', data=np.array(fitted_xanes_3D[j], dtype=np.float32))
                    hf.create_dataset(f'xanes_fit_comp{j}_norm', data=np.array(fitted_xanes_3D_norm[j], dtype=np.float32))
                    hf.create_dataset(f'ref{j}', data=np.array(fitted_xanes_3D[j], dtype=np.float32))



def fit_2D_xanes_file_mpi(file_path, file_prefix, file_type, fit_param, xanes_eng, spectrum_ref, file_range=[], save_hdf=0, num_cpu=8):
    '''
    batch processing xanes given xanes files
    '''
    from multiprocessing import Pool, cpu_count
    from functools import partial
    max_num_cpu = round(cpu_count() * 0.8)
    if num_cpu == 0 or num_cpu > max_num_cpu:
        num_cpu = max_num_cpu
    print(f'assembling slice using {num_cpu:2d} CPUs')

    time_start = time.time()
    file_save_path = f'{file_path}/fitted_xanes'
    create_directory(file_save_path)

    files_scan = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)
    try:
        true_file_type = fit_param['file_type']
    except:
        true_file_type = 'tiff'
    if true_file_type == 'tiff' or true_file_type == 'tif':
        tmp = pyxas.get_img_from_tif_file(files_scan[0])
    elif true_file_type == 'h5' or files_scan[0].split('.')[-1] == 'h5':
        hdf_attr = fit_param['hdf_attr']
        tmp = pyxas.get_img_from_hdf_file(files_scan[0], hdf_attr)[hdf_attr]

    s = tmp.shape
    num_file = len(files_scan)
    fs, fe = 0, num_file
    if len(file_range):
        fs = int(np.max([0, np.min(file_range)]))
        fe = int(np.min([np.max(file_range), num_file]))
    files_scan = files_scan[fs:fe]
    num_file = len(files_scan)
    print(f'processing {files_scan[0].split("/")[-1]}\n ... to {files_scan[-1].split("/")[-1]}')

    num_channel = len(spectrum_ref)
    try:
        n_comp = int(fit_param['n_comp']) if fit_param['mask_xanes_flag'] else 1 # num of mask
    except:
        n_comp = 1
    pool = Pool(num_cpu)
    res = pool.map(partial(pyxas.fit_2D_xanes_file_mpi_sub, fit_param=fit_param, xanes_eng=xanes_eng, spectrum_ref=spectrum_ref, file_save_path=file_save_path), files_scan)

    # assembling mpi fitting results and saving to hdf file
    if save_hdf:
        thickness_3D = np.zeros([num_file, s[1], s[2]])
        fitted_xanes_3D = np.zeros([num_channel, num_file, s[1], s[2]])
        fitted_xanes_3D_norm = fitted_xanes_3D.copy()
        fitting_cost_3D = np.zeros([num_file, s[1], s[2]])
        fitting_offset_3D = np.zeros([num_file, s[1], s[2]])
        thresh_thick = fit_param['fit_mask_thickness_threshold'] # thickness < thick_thresh will be 0
        thresh_cost = fit_param['fit_mask_cost_threshold'] # fit_error > cost_thresh will be 0
        mask_3D = np.ones([num_file, s[1], s[2]])
        if n_comp:
            mask_3D_comp = np.ones([n_comp, num_file, s[1], s[2]])

        for i in range(num_file):
            mask_3D[i] = np.squeeze(res[i]['mask'])
            thickness_3D[i] = np.squeeze(res[i]['xanes_fit_thickness'])
            fitting_cost_3D[i] = np.squeeze(res[i]['xanes_fit_cost'])
            fitting_offset_3D[i] = res[i]['xanes_2d_fit_offset']
            for j in range(num_channel):
                fitted_xanes_3D[j, i] = res[i][f'fitted_xanes_3D_ch{j}']
                fitted_xanes_3D_norm[j, i] = res[i][f'fitted_xanes_3D_ch{j}_norm']
            for n in range(n_comp):
                mask_3D_comp[n, i] = res[i][f'mask_{n}']


        fn_save = f'{file_save_path}/fitted_3d_xanes.h5'
        print(f'saving fitted 3D xanes: {fn_save} ... ')
        with h5py.File(fn_save, 'w') as hf:
            hf.create_dataset('type', data = 'fit_3D_xanes')
            hf.create_dataset('X_eng', data = xanes_eng)
            hf.create_dataset('pre_edge', data = fit_param['pre_edge'])
            hf.create_dataset('post_edge', data = fit_param['post_edge'])
            hf.create_dataset('unit', data = 'keV')
            hf.create_dataset('num_component', data = num_channel)
            if save_hdf:
                hf.create_dataset('xanes_fit_thickness', data=np.array(thickness_3D, dtype=np.float32))
                hf.create_dataset('xanes_fit_cost', data=np.array(fitting_cost_3D, dtype=np.float32))
                hf.create_dataset('xanes_fit_offset', data=np.array(fitting_offset_3D, dtype=np.float32))
                hf.create_dataset('mask', data=np.array(mask_3D, dtype=np.int8))
                for n in range(n_comp):
                    hf.create_dataset(f'mask_{n}', data=np.array(mask_3D_comp[n], dtype=np.int8))
                for j in range(num_channel):
                    hf.create_dataset(f'xanes_fit_comp{j}', data=np.array(fitted_xanes_3D[j], dtype=np.float32))
                    hf.create_dataset(f'xanes_fit_comp{j}_norm', data=np.array(fitted_xanes_3D_norm[j], dtype=np.float32))
                    hf.create_dataset(f'ref{j}', data=np.array(fitted_xanes_3D[j], dtype=np.float32))

    print(f'total time elapsed: {time.time()-time_start:6.1f}')





def fit_2D_xanes_file_mpi_sub(files_scan, fit_param, xanes_eng, spectrum_ref, file_save_path):

    res = {}
    time_start = time.time()
    files_scan_short = files_scan.split("/")[-1]
    print(f'fitting {files_scan_short} ...')
    num_channel = len(spectrum_ref)
    thresh_thick = fit_param['fit_mask_thickness_threshold'] # thickness < thick_thresh will be 0
    thresh_cost = fit_param['fit_mask_cost_threshold'] # fit_error > cost_thresh will be 0

    try:
        true_file_type = fit_param['file_type']
    except:
        true_file_type = 'tiff'
    if true_file_type == 'tiff' or true_file_type == 'tif':
        img_xanes = pyxas.get_img_from_tif_file(files_scan)
    elif true_file_type == 'h5':
        hdf_attr = fit_param['hdf_attr']
        img_xanes = pyxas.get_img_from_hdf_file(files_scan, hdf_attr)[hdf_attr]
    #img_xanes = pyxas.get_img_from_tif_file(files_scan)
    res = pyxas.fit_2D_xanes(img_xanes, xanes_eng, spectrum_ref, fit_param)
     
    mask = pyxas.fit_xanes2D_generate_mask(res['xanes_fit_thickness'], res['xanes_fit_cost'], thresh_cost, thresh_thick, num_iter=0)
    res['mask'] = mask
    try:
        color = fit_param['color']
    except:
        color = 'r, g, b'
    save_xanes_fitting_image(res, file_save_path, files_scan, color)
    
    for j in range(num_channel):
        res[f'fitted_xanes_3D_ch{j}'] = np.squeeze(res['xanes_2d_fit'][j])
        res[f'fitted_xanes_3D_ch{j}_norm'] = np.squeeze(res['xanes_2d_fit_norm'][j])
    print(f'{files_scan_short} taking time: {time.time() - time_start:05.1f}\n')
    return res


def fit_xanes2D_norm_edge(img_xanes, xanes_eng, pre_edge, post_edge, fit_eng=[], norm_txm_flag=1, fit_pre_edge_flag=1, fit_post_edge_flag=0, norm_method='new'):
    '''
    Normalize 2D xanes image with given fit_edge 

    Inputs:
    ---------
    img_xanes: 3D array
        
    xanes_eng: list

    fit_eng: 2-element list
        e.g., fit_edge = [8.3, 8.4] for Ni
        if [], will use all energy files

    norm_txm_flag: int  
        if 1: img_xanes = -np.log(img_xanes)
        if 0: do nothing

    Outputs:
    ---------
    img_xanes_norm: 3D array

    img_thickness: 3D array with shape of (0, row, col)

    fit_eng: 2-element int list
        contains the closest position of energy (fit_eng) in xanes_eng
    
    '''
    ## define fitting energy range
    if not len(fit_eng):
        fit_eng = [xanes_eng[0], xanes_eng[-1]]        
    fit_eng_s = pyxas.find_nearest(xanes_eng, fit_eng[0])
    fit_eng_e = pyxas.find_nearest(xanes_eng, fit_eng[1])
    ## normalize image
    if norm_txm_flag:
        img_xanes_norm = pyxas.norm_txm(img_xanes)
    else:
        img_xanes_norm = img_xanes.copy()


    if fit_pre_edge_flag:
        img_xanes_norm, img_thickness = pyxas.normalize_2D_xanes(img_xanes_norm, xanes_eng, pre_edge, post_edge, pre_edge_only_flag=1, method=norm_method)
        img_thickness_calulated_flag = 1
    else:
        img_thickness_calculated_flag = 0
        img_thickness = img_xanes_norm[-1] 

    if fit_post_edge_flag:
        img_xanes_norm = pyxas.normalize_2D_xanes_post_edge(img_xanes_norm, xanes_eng, post_edge)
            
        #img_xanes_norm, _ = pyxas.normalize_2D_xanes_old(img_xanes_norm, xanes_eng, pre_edge, post_edge) # should not update img_thickness
    fit_eng_index = [int(fit_eng_s), int(fit_eng_e)]  
    img_xanes_norm = pyxas.rm_abnormal(img_xanes_norm)
      
    return img_xanes_norm, img_thickness, fit_eng_index 


def fit_xanes2D_generate_mask(img_thickness, xanes_fit_cost, thresh_cost=0.1, thresh_thick=0, num_iter=2):
    mask1 = np.squeeze((xanes_fit_cost < thresh_cost).astype(int))
    mask2 = np.squeeze((img_thickness > thresh_thick).astype(int))
    mask = mask1 * mask2
    
    struct = ndimage.generate_binary_structure(2, 1)
    struct = ndimage.iterate_structure(struct, 2).astype(int)
    #mask,_ = pyxas.img_fill_holes(mask, struct=struct)
    mask = ndimage.binary_fill_holes(mask, structure=struct).astype(mask.dtype)
    mask = pyxas.medfilt(mask, 5)
    #mask = ndimage.binary_dilation(mask, structure=struct, iterations=num_iter).astype(mask.dtype)
    #mask = ndimage.binary_erosion(mask, structure=struct, iterations=num_iter).astype(mask.dtype)
    mask = mask.reshape([1, mask.shape[0], mask.shape[1]])
    return mask


def assemble_xanes_slice_from_tomo_mpi_sub(sli, file_path, files_recon, attr_img='img', attr_eng='X_eng', align_flag=0, align_ref_index=-1, align_roi_ratio=0.8, ali_sli=[], align_algorithm='stackreg', flag_save_2d_xanes=1, flag_mask=1):

    time_s = time.time()
    num_file = len(files_recon)
    f = h5py.File(files_recon[0], 'r')    
    num_slice = len(f[attr_img])
    s = np.array(list(f[attr_img][0])).shape
    f.close()
    img_xanes = np.zeros([num_file, s[0], s[1]])
    #mask3D = np.ones([num_slice, s[0], s[1]])
    xanes_eng = np.zeros(num_file)
    mask = np.ones([s[0], s[1]])
    res = {}    
    print(f'processing slice: {sli}')
    for j in range(num_file):
        f = h5py.File(files_recon[j], 'r')
        tmp = np.array(f[attr_img][sli])
        try:
            tmp_eng = np.array(f[attr_eng])
        except:
            tmp_eng = 0
        f.close()
        img_xanes[j] = tmp
        xanes_eng[j] = tmp_eng
    # align xanes stack
    if align_flag:
        s = img_xanes.shape          
        if len(ali_sli) == 0:
            ali_sli = [0, num_slice-1]        
        ali_sli = np.array(ali_sli)
        if sli >= np.min(ali_sli) and sli <= np.max(ali_sli):
            if align_roi_ratio >= 1:
                img_mask = img_xanes
            else:
                rs, re = 1-align_roi_ratio, align_roi_ratio
                img_mask = img_xanes[:, int(s[1]*rs):int(s[1]*re), int(s[2]*rs):int(s[2]*re)]
            if align_ref_index == -1:
                align_ref_index = img_xanes.shape[0] - 1
            if align_algorithm == 'stackreg':
                #print('hi')
                img_xanes = pyxas.align_img_stack_stackreg(img_xanes, img_mask, select_image_index=align_ref_index, print_flag=0) 
                #print('oh')
            else:
                img_xanes = pyxas.align_img_stack(img_xanes, img_mask, select_image_index=align_ref_index, print_flag=0) 
            # apply mask        
            if flag_mask:
                try:
                    mask = kmean_mask(img_xanes)
                except:
                    pass
    if flag_save_2d_xanes:
        io.imsave(f'{file_path}/xanes_assemble/xanes_2D_slice_{sli:03d}.tiff', np.array(img_xanes, dtype=np.float32))
        #if flag_mask:
            #io.imsave(f'{file_path}/xanes_assemble/mask_slice_{sli:03d}.tiff', np.array(mask, dtype=np.float32))
        print(f'xanes_2D_slice_{sli:03d}.tiff saved, using time: {time.time()-time_s:4.1f}sec\n')
    res['mask'] = mask
    return res


def assemble_xanes_slice_from_tomo_mpi(file_path='.', file_prefix='ali_recon', file_type='.h5', attr_img='img', attr_eng='X_eng', sli=[], align_flag=0, align_ref_index=-1, align_roi_ratio=0.8, ali_sli=[], align_algorithm='stackreg', flag_save_2d_xanes=1, flag_mask=1, num_cpu=0):

    # if flag_mask = 1, it will calculate the mask from img_xanes

    from multiprocessing import Pool, cpu_count
    from functools import partial
    max_num_cpu = round(cpu_count() * 0.8)
    if not num_cpu or num_cpu > max_num_cpu:
        num_cpu = round(cpu_count() * 0.8)
    print(f'assembling slice using {num_cpu:2d} CPUs')

    file_path = os.path.abspath(file_path)
    files_recon = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)
    num_file = len(files_recon)
        
    xanes_assemble_dir = f'{file_path}/xanes_assemble'
    if not os.path.exists(xanes_assemble_dir):
        os.makedirs(xanes_assemble_dir)
    f = h5py.File(files_recon[0], 'r')    
    num_slice = len(f['img'])
    if len(sli) == 0:
        sli=[0, num_slice]
    sli = np.arange(sli[0], sli[1])
    s = np.array(f['img'][0]).shape    
    f.close()

    mask_3D = np.ones([num_slice, s[0], s[1]])    
    time_s = time.time()
    pool = Pool(num_cpu)
    res = pool.map(partial(pyxas.assemble_xanes_slice_from_tomo_mpi_sub, file_path=file_path, files_recon=files_recon, attr_img=attr_img, attr_eng=attr_eng, align_flag=align_flag, align_ref_index=align_ref_index, align_roi_ratio=align_roi_ratio, ali_sli=ali_sli, align_algorithm=align_algorithm, flag_save_2d_xanes=flag_save_2d_xanes, flag_mask=flag_mask), sli)
    
    for i in range(len(files_recon)):
        mask_3D[i] = res[i]['mask']

    io.imsave(f'{xanes_assemble_dir}/mask3D.tiff', np.array(mask_3D, dtype=np.float32))
    print(f'total time for assembling xanes is: {time.time()-time_s:4.1f}')


def assemble_xanes_slice_from_tomo(file_path='.', file_prefix='ali_recon', file_type='.h5', attr_img='img', attr_eng='XEng', sli=[], align_flag=0, align_ref_index=-1, align_roi_ratio=0.8, ali_sli=[], align_algorithm='stackreg', flag_save_2d_xanes=1, flag_mask=0, return_flag=0):


#    Re-assemble the "ALIGNED" 3D xanes tomography into 2D xanes for each slices

    file_path = os.path.abspath(file_path)
    files_recon = pyxas.retrieve_file_type(file_path, file_prefix=file_prefix, file_type=file_type)
    num_file = len(files_recon)
    f = h5py.File(files_recon[0], 'r')    
    num_slice = len(f['img'])
    s = np.array(list(f['img'][0])).shape
    f.close()
    xanes_assemble_dir = f'{file_path}/xanes_assemble'
    if not os.path.exists(xanes_assemble_dir):
        os.makedirs(xanes_assemble_dir)    
    if len(sli) == 0:
        sli=[0, num_slice]
    sli = np.arange(sli[0], sli[1])
    img_xanes = np.zeros([num_file, s[0], s[1]])
    mask3D = np.ones([num_slice, s[0], s[1]])
    xanes_eng = np.zeros(num_file)
    for i in sli:
        print(f'processing slice: {i}/({sli[0]}: {sli[-1]})')
        
        for j in range(num_file):
            f = h5py.File(files_recon[j], 'r')
            tmp = np.array(f[attr_img][i])
            try:
                tmp_eng = np.array(f[attr_eng])
            except:
                tmp_eng = 0
            f.close()
            img_xanes[j] = tmp
            xanes_eng[j] = tmp_eng
        # align xanes stack
        if align_flag:            
            s = img_xanes.shape
            if len(ali_sli) == 0:
                ali_sli = [0, s[0]]
            ali_sli = np.array(ali_sli)
            if i >= np.min(ali_sli) and i <= np.max(ali_sli):
                if align_roi_ratio >= 1:
                    img_mask = None
                else:
                    rs, re = 1-align_roi_ratio, align_roi_ratio
                    img_mask = img_xanes[:, int(s[1]*rs):int(s[1]*re), int(s[2]*rs):int(s[2]*re)]
                if align_ref_index == -1:
                    align_ref_index = img_xanes.shape[0] - 1
                if align_algorithm == 'stackreg':
                    img_xanes = pyxas.align_img_stack_stackreg(img_xanes, img_mask, select_image_index=align_ref_index) 
                else:
                    img_xanes = pyxas.align_img_stack(img_xanes, img_mask, select_image_index=align_ref_index) 
                # apply mask        
                if flag_mask:
                    try:
                        mask = kmean_mask(img_xanes)
                    #img_xanes *= mask
                        mask3D[i] = mask
                    except:
                        pass
        if flag_save_2d_xanes:
            io.imsave(f'{file_path}/xanes_assemble/xanes_2D_slice_{i:03d}.tiff', np.array(img_xanes, dtype=np.float32))
    io.imsave(f'{file_path}/xanes_assemble/mask3D.tiff', np.array(mask3D, dtype=np.float32))
    if return_flag:
        return img_xanes, xanes_eng


