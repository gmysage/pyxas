#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
unicode_literals)
import pyxas
import os
import h5py
from pyxas.xanes_util import *
from pyxas.image_util import *
from pyxas.xanes_fit import *
from pyxas.tomo_recon import *
from skimage import io
from scipy import ndimage
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


  

def demo():

    fn_ref1 = '/home/mingyuan/Work/xanes_fit/Ni_ref/ref_NiO.txt'
    fn_ref2 = '/home/mingyuan/Work/xanes_fit/Ni_ref/ref_LiNiO2.txt'
    spectrum_ref = pyxas.load_xanes_ref_file(fn_ref1, fn_ref2)

    file_path='/home/mingyuan/Work/scan8538_8593/ali_recon/xanes_assemble'
    fn_eng = '/home/mingyuan/Work/scan8538_8593/eng_list.txt'
    file_start='xanes'
    file_type='.tiff'
    pre_edge=[8.2,8.3]
    post_edge=[8.4,9]
    fit_eng=[8.3,8.6]
    h5_attri_xanes='img_xanes'
    h5_attri_eng='XEng'
    norm_txm_flag=0
    regulation_param = {'flag': True, 'designed_max': 1.65, 'gamma':0.05}
    thresh_thick=5e-4 # thickness < thick_thresh will be 0
    thresh_cost=0.1 # fit_error > cost_thresh will be 0
    align_param = {'flag': True, 'align_ref': -1, 'roi_ratio': 0.6}

    pyxas.batch_xanes_2D(spectrum_ref, file_path, file_start, file_type, pre_edge, post_edge, fit_eng, fn_eng, h5_attri_xanes, h5_attri_eng, norm_txm_flag, thresh_thick, thresh_cost, regulation_param, align_param)




def demo1():

    file_path = '.'
    binning = 2    
    ref_index = -1
    ref_rot_cen = -1  # if -1: will find rotation center by pyxas.find_rot()
    block_list = []
    sli = []
    ratio = 0.8       # to generate circular mask in reconstruction
    file_prefix = 'fly'
    file_type = '.h5'
    # align tomo and reconstructing
    #pyxas.fit_xanes2D_align_tomo_proj(file_path, binning, ref_index, ref_rot_cen, block_list, sli ratio, file_prefix, file_type) # save aligned-recon in f'{file_path}/ali_recon'

    # retrieve single slice
    img_xanes, xanes_eng = pyxas.assemble_xanes_slice_from_tomo(file_path=f'{file_path}/ali_recon', file_prefix='ali_recon', file_type='.h5', sli=[300], return_flag=1)

    # fit 2D xanes
    fn_ref1 = '/home/mingyuan/Work/script_bank/xanes_fit/Ni_ref/ref_NiO.txt'
    fn_ref2 = '/home/mingyuan/Work/script_bank/xanes_fit/Ni_ref/ref_LiNiO2.txt'
    spectrum_ref = pyxas.load_xanes_ref_file(fn_ref1, fn_ref2)
    
    fit_param = {}
    fit_param['pre_edge'] = [8.2, 8.33]
    fit_param['post_edge'] = [8.4, 9]
    fit_param['fit_eng'] = [8.3, 8.6]
    fit_param['norm_txm_flag'] = False
    fit_param['fit_post_edge_flag'] = False

    fit_param['align_flag'] = False    
    fit_param['align_ref_index'] = -1
    fit_param['roi_ratio'] = 0.6 # only take effect when need to align image

    fit_param['fit_iter_flag'] = False
    fit_param['fit_iter_learning_rate'] = 0.005 # only take effect when fit_param['fit_iter_flag'] = True
    fit_param['fit_iter_num'] = 5 # only take effect when fit_param['fit_iter_flag'] = True
    fit_param['fit_iter_bound'] = [0, 1]

    
    fit_param['regulation_flag'] = False
    fit_param['regulation_designed_max'] = 1.6 # only take effect when fit_param['regulation_flag'] = True
    fit_param['regulation_gamma'] = 0.05 # only take effect when fit_param['regulation_flag'] = True

    img_thickness, xanes_2d_fit, xanes_fit_cost, img_xanes_norm = pyxas.fit_2D_xanes(img_xanes, xanes_eng, spectrum_ref, fit_param)

    # generate mask
    thresh_thick = 0.0012 # thickness < thick_thresh will be 0
    thresh_cost = 0.1 # fit_error > cost_thresh will be 0
    mask = pyxas.fit_xanes2D_generate_mask(img_thickness, xanes_fit_cost, thresh_cost, thresh_thick, num_iter=8)





                  







def demo4():
    '''
    reconstructing, aligning, fit_xanes
    '''
    file_path = '/NSLS2/xf18id1/users/2019Q1/SEONGMIN_Proposal_303787/NCA_48V_70cy_pos1'
    block_list = np.arange(50)
    num_file = len(file_path)
    files = batch_recon(file_path, file_prefix='fly', file_type='.h5', sli=[200, 1000], col=[], block_list=[], binning=1, rm_stripe=True, stripe_remove_level=9)

    pyxas.align_3D_tomo_file(file_path='.', ref_index=-1, binning=1, circle_mask_ratio=0.9, file_prefix='recon', file_type='.h5')

    pyxas.assemble_xanes_slice_from_tomo(file_path='.', file_prefix='ali_recon', file_type='.h5', attr_img='img', attr_eng='X_eng', sli=[], flag_save_2d_xanes=1, return_flag=0)

    # start to fit
    
    '''
    #file_path = '/media/mingyuan/MINGYUAN DATA/2019_TXM/scan14140_14230/recon/xanes_assemble'
    file_path = '/NSLS2/xf18id1/users/2019Q1/SEONGMIN_Proposal_303787/NCA_48V_70cy_pos1/xanes_assemble'
    file_prefix = 'xanes'
    file_type = '.tiff' 

    file_path = os.path.abspath(file_path)
    files_scan = pyxas.retrieve_file_type(file_path, file_prefix, file_type)

    fit_param = {}
    fit_param['pre_edge'] = [8.2, 8.33]
    fit_param['post_edge'] = [8.4, 9]
    fit_param['fit_eng'] = [8.3, 8.6]
    fit_param['norm_txm_flag'] = 0
    fit_param['fit_post_edge_flag'] = 0

    fit_param['align_flag'] = 1    
    fit_param['align_ref_index'] = -1
    fit_param['roi_ratio'] = 0.6 # only take effect when need to align image

    fit_param['fit_iter_flag'] = 1
    fit_param['fit_iter_learning_rate'] = 0.005 # only take effect when fit_param['fit_iter_flag'] = True
    fit_param['fit_iter_num'] = 5 # only take effect when fit_param['fit_iter_flag'] = True
    fit_param['fit_iter_bound'] = [0, 1]
    
    fit_param['regulation_flag'] = 0
    fit_param['regulation_designed_max'] = 1.6 # only take effect when fit_param['regulation_flag'] = True
    fit_param['regulation_gamma'] = 0.05 # only take effect when fit_param['regulation_flag'] = True

    fit_param['fit_mask_thickness_threshold'] = 0.0008
    fit_param['fit_mask_cost_threshold'] = 0.1
    '''
    #fn_ref1 = '/home/mingyuan/Work/script_bank/xanes_fit/Ni_ref/ref_NiO.txt'
    #fn_ref2 = '/home/mingyuan/Work/script_bank/xanes_fit/Ni_ref/ref_LiNiO2.txt'
    #fn_ref1 = '/NSLS2/xf18id1/users/2019Q1/SEONGMIN_Proposal_303787/ref/ref_Ni_II.txt'
    #fn_ref2 = '/NSLS2/xf18id1/users/2019Q1/SEONGMIN_Proposal_303787/ref/ref_Ni_III.txt'

<<<<<<< HEAD
    fn_ref1 = '/home/mingyuan/data1/Mingyuan/TXM_2019/Mingyuan_2019Q2/in_situ_NCA_20190823/Ni_ref/ref_NiO.txt'
    fn_ref2 = '/home/mingyuan/data1/Mingyuan/TXM_2019/Mingyuan_2019Q2/in_situ_NCA_20190823/Ni_ref/ref_LiNiO2.txt'

    #fn_ref1 = '/home/mingyuan/data1/Mingyuan/TXM_2019/Mingyuan_2019Q2/in_situ_NCA_20190823/ref_shift_to_FX_eng/ref_pristine_shifted_this_time.txt'
    #fn_ref2 = '/home/mingyuan/data1/Mingyuan/TXM_2019/Mingyuan_2019Q2/in_situ_NCA_20190823/ref_shift_to_FX_eng/ref_4.6V_shifted_this_time.txt'
=======
    fn_ref1 = '/home/mingyuan/data1/Mingyuan/TXM_2019/Mingyuan_2019Q2/in_situ_NCA_20190823/ref_shift_to_FX_eng/ref_pristine_shifted_this_time.txt'
    fn_ref2 = '/home/mingyuan/data1/Mingyuan/TXM_2019/Mingyuan_2019Q2/in_situ_NCA_20190823/ref_shift_to_FX_eng/ref_4.6V_shifted_this_time.txt'
>>>>>>> 48d2735dfb713dfe1979b2e28441a7b5290d7723

    spectrum_ref = pyxas.load_xanes_ref_file(fn_ref1, fn_ref2)
    #xanes_eng = np.loadtxt('/NSLS2/xf18id1/users/2019Q1/SEONGMIN_Proposal_303787/NC_48V_70cy_pos1/eng_list.txt')
    xanes_eng = np.loadtxt('/home/mingyuan/data1/Mingyuan/TXM_2019/Mingyuan_2019Q2/in_situ_NCA_20190823/s3_second_charge/pos1/rep1/eng_list.txt')
    #fit_param = pyxas.load_xanes_fit_param_file(fn='xanes_fit_param.csv', num_items=18)
    fit_param = pyxas.load_xanes_fit_param_file(fn='/home/mingyuan/data1/Mingyuan/TXM_2019/Mingyuan_2019Q2/in_situ_NCA_20190823/xanes_fit_param_20191008.csv', num_items=0)

    fn_xanes_assemble = f'{file_path}/xanes_assemble'
    file_prefix_xanes_assemble = 'xanes'
    file_type_xanes_assemble = '.tiff'

    pyxas.fit_2D_xanes_file(fn_xanes_assemble, file_prefix_xanes_assemble, file_type_xanes_assemble, fit_param, xanes_eng, spectrum_ref)








