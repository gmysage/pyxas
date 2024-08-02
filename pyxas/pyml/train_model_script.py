import shutil
import pyxas
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import json
from .model_lib import *
from .util import *
from .train_lib import *
from .fit_xanes import *
from skimage import io
from pyxas import kmean_mask, scale_img_xanes
from copy import deepcopy


def main_train_1_branch_bkg():
    device = torch.device('cuda:3')
    lr = 0.00001
    loss_r = {}
    loss_r['vgg_identity'] = 1           # (model_outputs vs. label); "0" for "Production", "1" for trainning
    loss_r['vgg_fit'] = 1e-1                # (fitted_image vs. label); "1" for both "trainning" and "production"
    #loss_r['vgg_1st_last'] = 0         # (model_outputs[0] vs. model_outputs[-1]); "1e2" for both "trainning" and "production"

    #loss_r['mse_identity_img'] = 1  
    loss_r['mse_identity_bkg'] = 1
    loss_r['mse_fit_coef'] = 1e10          # (fit_coef_from_model_outputs vs. fit_coef_from_label); "1e8" for both "trainning" and "production"
    loss_r['mse_fit_self_consist'] = 20   # (fitting_output_from_model_output vs. model_outputs ); "1" for both "trainning" and "production"
    loss_r['l1_identity'] = 0  

    global vgg19
    avgpool = nn.AvgPool2d(3, stride=1, padding=1)
    torch.manual_seed(0)    
    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    vgg19.to(device).eval()

    model_gen = pyxas.RRDBNet(1, 1, 16, 4, 32, 'zeros').to(device)
    #model_gen = RRDBNet_new(1, 1, 16, 4, 32).to(device)
    #initialize_weights(model_gen)

    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()

    opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(0.5, 0.999))

    h_loss_train = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value':[], 'rate':[]}
    best_psnr = 0
    epochs = 500
    n_train = 100

    f_root = '/data/xanes_bkg_denoise/IMG_256_stack/Co_thick' #'/data/xanes_bkg_denoise/IMG_256_stack/Co3'
    blur_dir = f_root + '/img_blur_stack'
    gt_dir = f_root + '/img_bkg_stack' #'/img_bkg_stack_gf'
    eng_dir = f_root + '/img_eng_list'
    trans_gt, trans_blur = None, None

    train_loader, valid_loader = pyxas.get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, trans_gt, trans_blur)

    best_psnr = 0
    #model_save_path2 = '/data/xanes_bkg_denoise/IMG_256_stack/Co3/Co_bkg_256_new.pth'
    #model_save_path2 = '/data/xanes_bkg_denoise/IMG_256_stack/Co_thin/Co_bkg.pth'
    model_save_path2 = f_root + '/Co_bkg_load_previous.pth'
   
    for epoch in range(500):
        loss_summary_train = pyxas.train_1_branch_bkg(model_gen, train_loader, loss_r, vgg19, device, lr=lr, train_fit=True)
        print(f'epoch #{epoch}')
        h_loss_train, txt_t, psnr_train = pyxas.extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        
        print(txt_t)
        print(f'train_psnr = {psnr_train:2.2f}')
        if psnr_train > best_psnr:
            torch.save(model_gen.state_dict(), model_save_path2)
            best_psnr = psnr_train
        #ftmp = f'/data/xanes_bkg_denoise/IMG_256_stack/Co3/model_tmp_new/tmp_{epoch:04d}.pth'
        #ftmp = f'/data/xanes_bkg_denoise/IMG_256_stack/Co_thin/model_saved/tmp_{epoch:04d}.pth'
        ftmp = f'/data/xanes_bkg_denoise/IMG_256_stack/Co_thin/model_saved_load_previous/tmp_{epoch:04d}.pth'
        torch.save(model_gen.state_dict(), ftmp)
        #with open('/data/xanes_bkg_denoise/IMG_256_stack/Co3/model_tmp_new/h_loss_Co3_bkg.json', 'w') as f:
        #with open('/data/xanes_bkg_denoise/IMG_256_stack/Co_thin/model_saved/h_loss_Co3_bkg.json', 'w') as f:
        with open('/data/xanes_bkg_denoise/IMG_256_stack/Co_thin/model_saved_load_previous/h_loss_Co3_bkg.json', 'w') as f:
            json.dump(h_loss_train, f)


def main_train_1_branch_bkg_with_gt_image():
    device = torch.device('cuda:3')
    lr = 0.0001
    loss_r = {}

    # loss_r['mse_identity_img'] = 1
    loss_r['mse_identity_bkg'] = 1
    loss_r['mse_fit_coef'] = 0  # (fit_coef_from_model_outputs vs. fit_coef_from_label); "1e8" for both "trainning" and "production"
    loss_r['mse_fit_self_consist'] = 0  # (fitting_output_from_model_output vs. model_outputs ); "1" for both "trainning" and "production"
    loss_r['l1_identity'] = 0

    #model_gen = pyxas.RRDBNet(1, 1, 16, 4, 32).to(device)
    model_gen = pyxas.RRDBNet_padding_same(1, 1, 16, 4, 32, 'zeros', 5).to(device)
    model_path = '/data/software/pyxas/pyxas/pyml/trained_model/tmp_1499.pth'
    model_gen.load_state_dict(torch.load(model_path))
    # initialize_weights(model_gen)

    h_loss_train = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value': [], 'rate': []}
    best_psnr = 0
    epochs = 500
    n_train = 100

    f_root = '/data/xanes_bkg_denoise/IMG_256_stack/Co_thick'  # '/data/xanes_bkg_denoise/IMG_256_stack/Co3'
    blur_dir = f_root + '/img_blur_stack'
    gt_bkg_dir = f_root + '/img_bkg_stack'  # '/img_bkg_stack_gf'
    gt_img_dir = f_root + '/img_gt_stack'
    eng_dir = f_root + '/img_eng_list'
    trans_gt, trans_blur = None, None

    train_loader, valid_loader = pyxas.get_train_valid_dataloader_new(blur_dir, gt_bkg_dir, gt_img_dir, eng_dir, n_train, trans_gt,
                                                                  trans_blur)

    best_psnr = 0
    # model_save_path2 = '/data/xanes_bkg_denoise/IMG_256_stack/Co3/Co_bkg_256_new.pth'
    # model_save_path2 = '/data/xanes_bkg_denoise/IMG_256_stack/Co_thin/Co_bkg.pth'
    model_save_path2 = f_root + '/Co_bkg_k5.pth'

    # fsave_loss = f_root +'/model_saved/h_loss_bkg.json'
    fsave_loss = f_root + '/model_saved_k5/h_loss_bkg.json'
    for epoch in range(10):
        loss_summary_train = pyxas.train_1_branch_bkg_with_gt_image(model_gen, train_loader, loss_r, device, lr=lr)
        print(f'\nepoch #{epoch}')
        h_loss_train, txt_t, psnr_train = pyxas.extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        print(txt_t)
        print(f'train_psnr = {psnr_train:2.2f}')
        if psnr_train > best_psnr:
            torch.save(model_gen.state_dict(), model_save_path2)
            best_psnr = psnr_train
        #ftmp = f_root + f'/model_saved/tmp_{epoch:04d}.pth'
        ftmp = f_root + f'/model_saved_k5/tmp_{epoch:04d}.pth'
        torch.save(model_gen.state_dict(), ftmp)
        with open(fsave_loss, 'w') as f:
            json.dump(h_loss_train, f)




def main_train_1_branch_production():
    device = torch.device('cuda:1')
    loss_r = {}
    #loss_r['vgg_1st_last'] = 0         # (model_outputs[0] vs. model_outputs[-1]); "1e2" for both "trainning" and "production"
    loss_r['mse_fit_img'] = 1           # (fit_coef_from_model_outputs vs. fit_coef_from_label); "1e8" for both "trainning" and "production"

    global vgg19
    avgpool = nn.AvgPool2d(3, stride=1, padding=1)
    torch.manual_seed(0)    
    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    vgg19.to(device).eval()

    model_raw = RRDBNet(1, 1, 16, 4, 32).to(device)
    model_prod = RRDBNet(1, 1, 16, 4, 32).to(device)
    model_bkg_load_path = '/data/xanes_bkg_denoise/IMG_256_stack/Co3/model_tmp/tmp_1499.pth'
    model_prod.load_state_dict(torch.load(model_bkg_load_path))
    model_raw.load_state_dict(torch.load(model_bkg_load_path))
    #initialize_weights(model_gen)
    mse_criterion = nn.MSELoss()

    lr = 0.0001
    opt_prod = optim.Adam(model_prod.parameters(), lr=lr)

    h_loss_train = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value':[], 'rate':''}

    epochs = 500
    

    f_root = '/data/xanes_bkg_denoise/test_experiment/sample8_N85'
    blur_dir = f_root + '/img_product'
    gt_dir = f_root + '/img_product'
    eng_dir = f_root + '/eng_list'
    trans_gt, trans_blur = None, None

    ############### calculate thickness 
    img_raw = io.imread(f_root + '/aligned_29874_256x256.tiff')
    mask, _ = kmean_mask(np.squeeze(img_raw), 2)
    mask = mask[1]
    img_all = img_raw[:, np.newaxis]
    img_all = torch.tensor(img_all).to(device)
    s = img_all.shape
    x_eng = np.loadtxt(f_root + '/exafs_Co.txt')
    x_eng = torch.tensor(x_eng).to(device)

    X, Y_fit =  fit_element_xraylib_barn('Co', x_eng, img_all, order=[-3, 0], rho=None, take_log=True, device=device)
    thickness = X[0].reshape((s[-2], s[-1])).detach()
    thickness = thickness * torch.tensor(mask).to(device)

    ##################### end thickness

    n_train = 200
    train_loader, valid_loader = get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, trans_gt, trans_blur)
   
    for epoch in range(100):
        loss_summary_train = train_1_branch_bkg_production(train_loader, loss_r, thickness, device)
        h_loss_train, txt_t, psnr_train = extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        print(f'epoch #{epoch}')
        print(txt_t)
        print(f'train_psnr = {psnr_train:2.2f}')
        a, b, c = check_validation_stack(model_prod, valid_loader, 3, device=device, plot_flag=0)
        fn_im_save = f'/data/xanes_bkg_denoise/test_experiment/sample8_N85/bkg_output/bkg_{epoch:04d}.tiff'
        io.imsave(fn_im_save, a.astype(np.float32))
        ftmp = f'/data/xanes_bkg_denoise/test_experiment/sample8_N85/model_tmp/tmp_{epoch:04d}.pth'
        torch.save(model_prod.state_dict(), ftmp)
        with open('/data/xanes_bkg_denoise/test_experiment/sample8_N85/model_tmp/h_loss_bkg_product.json', 'w') as f:
            json.dump(h_loss_train, f)
            
            
            
######################################################################################################################################            
            
def main_train_production_single_elem(f_root, img_raw, x_eng, elem, thickness_elem=None, ratio=0.99,
                                        mask=None, n_train=50, n_epoch=30, device='cuda:0', save_flag=True):
     
    #device = torch.device('cuda:2')
    loss_r = {}
    loss_r['mse_fit_img'] = 1           # (fit_coef_from_model_outputs vs. fit_coef_from_label); "1e8" for both "trainning" and "production"
    loss_r['tv_bkg'] = 6e-5

    model_prod = RRDBNet(1, 1, 16, 4, 32).to(device)
    model_bkg_load_path = '/data/xanes_bkg_denoise/IMG_256_stack/Co3/model_tmp/sorted/tmp_1499.pth'
    model_prod.load_state_dict(torch.load(model_bkg_load_path))
    mse_criterion = nn.MSELoss()

    lr = 0.00005
    opt_prod = optim.Adam(model_prod.parameters(), lr=lr)

    h_loss_train = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value':[], 'rate':''}

    #elem = 'Co'

    blur_dir = f_root + '/img_blur_stack'
    gt_dir = f_root + '/img_gt_stack'
    eng_dir = f_root + '/img_eng_list'
    trans_gt, trans_blur = None, None

    ############### calculate thickness 
    if mask is None:
        mask = 1

    img_all = img_raw[:, np.newaxis]
    img_all = torch.tensor(img_all).to(device)
    s = img_all.shape
    x_eng = torch.tensor(x_eng).to(device)
    if thickness_elem is None:
        thickness_elem = cal_thickness(elem, x_eng, img_all, order=[-3, 0], rho=None, take_log=True, device=device)
        thickness_elem = torch.tensor(thickness_elem * mask) 
    thickness = {}
    thickness[elem] = thickness_elem
    mask = torch.tensor(mask)
    ##################### end thickness

    #n_train = 50
    train_loader, valid_loader = get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, trans_gt, trans_blur)
   
    for epoch in range(n_epoch):
        #loss_summary_train = train_1_branch_bkg_production_NCM(train_loader, loss_r, thickness, device, ratio)
        loss_summary_train = train_1_branch_bkg_production(train_loader, loss_r, thickness, model_prod, device, ratio)
        
        if (epoch+1) % 5 == 0:
            print('\nupdate thickness\n')
            thickness_elem = update_thickness_elem(elem, img_all, x_eng, model_prod, device, n_iter=1, gaussian_filter=2)
            thickness[elem] = thickness_elem * mask
        img_output, img_bkg = apply_model_to_stack(img_raw, model_prod, device, 1, gaussian_filter=3)
        
        h_loss_train, txt_t, psnr_train = extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        print(f'epoch #{epoch}')
        print(txt_t)

        if save_flag:
            io.imsave(f_root + f'/model_output/output_{epoch:03d}.tiff', img_output.astype(np.float32))
            io.imsave(f_root + f'/model_output/bkg_{epoch:03d}.tiff', img_bkg.astype(np.float32))
            
            ftmp = f_root + f'/model_tmp/tmp_{epoch:04d}.pth'
            torch.save(model_prod.state_dict(), ftmp)
            with open(f_root + f'/model_tmp/h_loss.json', 'w') as f:
                json.dump(h_loss_train, f)


def mk_directory(fpath):
    if not os.path.exists(fpath):
        try:
            os.makedirs(fpath)
        except:
            shutil.rmtree(fpath)
            os.makedirs(fpath)


def main_train_xanes_bkg_production(f_root, img_raw, x_eng, elem, model_prod, loss_r=None, f_norm=1,
                                    blur_dir='', gt_dir='', eng_dir='',
                                    thickness_elem=None, ratio=0.99, lr=5e-5, thickness_update_rate=2,
                                    mask=None, n_train=50, n_epoch=30, device='cuda:0', save_flag=True):
    model_save_path = f_root + '/model_saved'
    img_save_path = f_root + '/model_output'
    mk_directory(model_save_path)
    mk_directory(img_save_path)
    if loss_r is None:
        loss_r = {}
        loss_r['mse_fit_img'] = 1
        loss_r['tv_bkg'] = 6e-5

    h_loss_train = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value': [], 'rate': []}

    if len(blur_dir):
        blur_dir = f_root + '/' + blur_dir
    else:
        blur_dir = f_root + '/img_blur_stack'
    if len(gt_dir):
        gt_dir = f_root + '/' + gt_dir
    else:
        gt_dir = f_root + '/img_gt_stack'
    if len(eng_dir):
        eng_dir = f_root + '/' + eng_dir
    else:
        eng_dir = f_root + '/img_eng_list'

    trans_gt, trans_blur = None, None

    #model_prod.load_state_dict(torch.load(model_load_path, map_location=device))

    if mask is None:
        mask = 1
    mask = torch.tensor(mask).to(device)
    ############### calculate thickness
    img_all = img_raw[:, np.newaxis]
    img_all = torch.tensor(img_all).to(device)
    s = img_all.shape
    x_eng = torch.tensor(x_eng).to(device)
    if thickness_elem is None:
        thickness_elem = cal_thickness(elem, x_eng, img_all/f_norm, order=[-3, 0], rho=None, take_log=True, device=device)

    thickness = {}
    thickness_elem = thickness_elem * mask
    thickness[elem] = thickness_elem
    thickness['mask'] = mask
    ##################### end thickness

    train_loader, valid_loader = get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, trans_gt, trans_blur)

    for epoch in range(n_epoch):
        loss_summary_train, model_prod = train_xanes_bkg_production(train_loader, loss_r, thickness, model_prod, lr, device, ratio)
        if (epoch + 1) % thickness_update_rate == 0:
            print('\nupdate thickness\n')
            thickness_elem = update_thickness_elem(elem, img_all/f_norm, x_eng, model_prod, device, n_iter=1, gaussian_filter=2)
            try:
                thickness[elem] = thickness_elem * mask
            except Exception as err:
                print(err)

        img_output, img_bkg = apply_model_to_stack(img_raw/f_norm, model_prod, device, 1, gaussian_filter=1)

        h_loss_train, txt_t, psnr_train = extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        print(f'epoch #{epoch}')
        print(txt_t)

        if save_flag:
            io.imsave(f_root + f'/model_output/output_{epoch:03d}.tiff', img_output.astype(np.float32))
            io.imsave(f_root + f'/model_output/bkg_{epoch:03d}.tiff', img_bkg.astype(np.float32))

            ftmp = f_root + f'/model_saved/m_prod_{epoch:04d}.pth'
            torch.save(model_prod.state_dict(), ftmp)
            with open(f_root + f'/model_saved/h_loss.json', 'w') as f:
                json.dump(h_loss_train, f)
    return h_loss_train


def main_train_xanes_bkg_production_with_reference(f_root, img_raw, x_eng, model_prod, spectrum_ref, loss_r=None,
                                                   f_norm=1, blur_dir='', gt_dir='', eng_dir='', lr=5e-5, ratio=0.99,
                                                   mask=None, n_train=50, n_epoch=30, device='cuda:0', save_flag=True):
    model_save_path = f_root + '/model_saved'
    img_save_path = f_root + '/model_output'
    mk_directory(model_save_path)
    mk_directory(img_save_path)

    if loss_r is None:
        loss_r = {}
        loss_r['mse_fit_img'] = 1
        loss_r['tv_bkg'] = 6e-5

    h_loss_train = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value': [], 'rate': ''}

    if len(blur_dir):
        blur_dir = f_root + '/' + blur_dir
    else:
        blur_dir = f_root + '/img_blur_stack'
    if len(gt_dir):
        gt_dir = f_root + '/' + gt_dir
    else:
        gt_dir = f_root + '/img_gt_stack'
    if len(eng_dir):
        eng_dir = f_root + '/' + eng_dir
    else:
        eng_dir = f_root + '/img_eng_list'

    trans_gt, trans_blur = None, None

    if mask is None:
        mask = 1

    mask = torch.tensor(mask)

    train_loader, valid_loader = get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, trans_gt, trans_blur)

    for epoch in range(n_epoch):
        loss_summary_train, model_prod = train_xanes_bkg_production_with_reference(train_loader, loss_r, model_prod, lr,
                                                                                   spectrum_ref, mask, ratio, device)
        img_output, img_bkg = apply_model_to_stack(img_raw/f_norm, model_prod, device, 1, gaussian_filter=1)

        h_loss_train, txt_t, psnr_train = extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        print(f'epoch #{epoch}')
        print(txt_t)

        if save_flag:
            io.imsave(f_root + f'/model_output/output_{epoch:03d}.tiff', img_output.astype(np.float32))
            io.imsave(f_root + f'/model_output/bkg_{epoch:03d}.tiff', img_bkg.astype(np.float32))

            ftmp = f_root + f'/model_saved/m_prod_{epoch:04d}.pth'
            torch.save(model_prod.state_dict(), ftmp)
            with open(f_root + f'/model_saved/h_loss.json', 'w') as f:
                json.dump(h_loss_train, f)
    plot_h_loss(h_loss_train)


def main_train_production_with_fitted_param(f_root, img_raw, x_eng, model_prod, spectrum_ref, loss_r=None,
                                        f_norm=1, blur_dir='', gt_dir='', eng_dir='', lr=5e-5, fit_update_rate=5,
                                        mask=None, n_train=50, n_epoch=30, device='cuda:0', save_flag=True):
    model_save_path = f_root + '/model_saved'
    img_save_path = f_root + '/model_output'
    mk_directory(model_save_path)
    mk_directory(img_save_path)

    if loss_r is None:
        loss_r = {}
        loss_r['mse_fit_img'] = 1
        loss_r['tv_bkg'] = 6e-5
        loss_r['ssim'] = 1e-4

    h_loss_train = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value': [], 'rate': ''}

    if len(blur_dir):
        blur_dir = f_root + '/' + blur_dir
    else:
        blur_dir = f_root + '/img_blur_stack'
    if len(gt_dir):
        gt_dir = f_root + '/' + gt_dir
    else:
        gt_dir = f_root + '/img_gt_stack'
    if len(eng_dir):
        eng_dir = f_root + '/' + eng_dir
    else:
        eng_dir = f_root + '/img_eng_list'

    trans_gt, trans_blur = None, None

    if mask is None:
        mask = 1

    mask = torch.tensor(mask)

    img_output, img_bkg = apply_model_to_stack(img_raw / f_norm, model_prod, device, 1, gaussian_filter=1)
    fit_param_X, _ = fit_element_with_reference(x_eng, img_output, spectrum_ref, [1, 0], True, device=device)

    train_loader, valid_loader = get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, trans_gt, trans_blur)

    for epoch in range(n_epoch):
        if (epoch+1) % fit_update_rate == 0:
            print('update fitting')
            fit_param_X, _ = fit_element_with_reference(x_eng, img_output, spectrum_ref,[1, 0], True, device=device)
        loss_summary, model_prod = train_production_with_fitted_param(train_loader, loss_r, model_prod, lr,
                                                                      spectrum_ref, fit_param_X, [1,0], mask, device)
        img_output, img_bkg = apply_model_to_stack(img_raw/f_norm, model_prod, device, 1, gaussian_filter=1)

        h_loss_train, txt_t, psnr_train = extract_h_loss(h_loss_train, loss_summary, loss_r)
        print(f'epoch #{epoch}')
        print(txt_t)

        if save_flag:
            io.imsave(f_root + f'/model_output/output_{epoch:03d}.tiff', img_output.astype(np.float32))
            io.imsave(f_root + f'/model_output/bkg_{epoch:03d}.tiff', img_bkg.astype(np.float32))

            ftmp = f_root + f'/model_saved/m_prod_{epoch:04d}.pth'
            torch.save(model_prod.state_dict(), ftmp)
            with open(f_root + f'/model_saved/h_loss.json', 'w') as f:
                json.dump(h_loss_train, f)
    plot_h_loss(h_loss_train)
    return h_loss_train


def ML_fit_xanes(img_xanes, param, x_eng, elem, eng_exclude=[], thickness_update_rate=5, fn_root='.', device='cuda'):
    try:
        if (not len(param)) or (not type(param) is dict):
            param = ML_xanes_default_param()
    except:
        param = ML_xanes_default_param()

    img_raw, f_scale, mask = scale_img_xanes(img_xanes)
    img_raw[img_raw < 0] = 0

    ## calculate scaled img_raw
    s = img_raw.shape
    img_raw_scale = np.ones(s)
    scale_forward = np.ones(s[0])
    mask_sum = np.sum(mask)
    for i in range(s[0]):
        scale_forward[i] = np.sum(img_raw[i] * mask) / mask_sum
        img_raw_scale[i] = img_raw[i] / scale_forward[i]

    loss_r = param['loss_r']
    n_train = param['n_train']
    n_epoch = param['n_epoch']
    lr = param['lr']
    order = param['order']

    h_loss_train = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value': [], 'rate': []}

    model_prod = DnCNN(channels=1, num_of_layers=17).to(device)
    opt_prod = optim.Adam(model_prod.parameters(), lr=lr)

    f_root = fn_root + '/tmp'
    mk_directory(f_root)
    fn_img_xanes = f_root + '/img_xanes.tiff'
    io.imsave(fn_img_xanes, img_raw)

    prepare_production_training_dataset(fn_img_xanes=fn_img_xanes,
                                              elem=elem,
                                              eng=x_eng,
                                              eng_edge=eng_exclude,
                                              num_img=n_train,
                                              f_norm=1.0,
                                              n_stack=16,
                                              f_root=f_root)

    blur_dir = f_root + '/img_blur_stack'
    gt_dir = f_root + '/img_gt_stack'
    eng_dir = f_root + '/img_eng_list'

    best_psnr = 0
    best_model = None

    mask = torch.tensor(mask, dtype=torch.float).to(device)
    # mask = None

    img_all = img_raw[:, np.newaxis]
    img_all = torch.tensor(img_all).to(device)
    img_output = img_raw.copy()

    thickness = {}
    thickness_elem = cal_thickness(elem, x_eng, img_all, order=order, rho=None, take_log=False, device=device)
    thickness_elem[thickness_elem < 0] = 0
    thickness[elem] = thickness_elem * mask

    train_loader, valid_loader = get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, None, None)
    for epoch in range(n_epoch):
        if epoch == n_epoch // 2:
            lr = lr / 2
        loss_summary_train, model_prod = train_xanes_3D_production(train_loader, loss_r, thickness, model_prod,
                                                                   opt_prod, lr, device, mask, order)
        if (epoch + 1) % thickness_update_rate == 0:
            print('\nupdate thickness\n')
            t = img_output[:, np.newaxis]
            t = torch.tensor(t).to(device)
            thickness_elem = cal_thickness(elem, x_eng, t, order=order, rho=None, take_log=False, device=device)
            thickness_elem[thickness_elem < 0] = 0
            thickness[elem] = thickness_elem * mask

        _, img_output = apply_model_to_stack(img_raw_scale, model_prod, device, 1, gaussian_filter=1)
        img_output[img_output < 0] = 0
        for i in range(s[0]):
            img_output[i] *= scale_forward[i]
        h_loss_train, txt_t, psnr_train = extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        if psnr_train > best_psnr:
            best_psnr = psnr_train
            best_img = img_output.copy()
            best_model = deepcopy(model_prod)

        print(f'epoch #{epoch}')
        print(txt_t)
        io.imsave(f_root + f'/model_output/output_{epoch:03d}.tiff', img_output.astype(np.float32))
        ftmp = f_root + f'/model_saved/m_prod_{epoch:04d}.pth'
        torch.save(model_prod.state_dict(), ftmp)
        with open(f_root + f'/model_saved/h_loss.json', 'w') as f:
            json.dump(h_loss_train, f)

    return best_img / f_scale, best_model, f_scale, h_loss_train

def example_train_prod():
    # on computer: office2
    f_root = '/data/quant_xanes/N77_NMC_comb/2023Q2/ML_train/train_Ni'
    img_raw = io.imread('Ni_pos1_crop_300eng.tiff')
    x_eng = np.loadtxt('Ni_comb.txt')
    elem = 'Ni'
    thickness_elem=None
    ratio=0.99
    device = 'cuda:0'
    mask=None
    n_train=50
    n_epoch=30
    save_flag=True
    main_train_production_single_elem(f_root, img_raw, x_eng, elem, thickness_elem, ratio,
                                        mask, n_train, n_epoch, device, save_flag)


def example2_train_prod():
    # on computer: office
    f_root = '/home/mingyuan/Work/pyxas/example/ml_xanes_denoise/train_Ni/train_production'
    img_raw = io.imread('/home/mingyuan/Work/pyxas/example/ml_xanes_denoise/train_Ni/Ni_pos1_crop_300eng.tiff')
    x_eng = np.loadtxt('/home/mingyuan/Work/pyxas/example/ml_xanes_denoise/train_Ni/Ni_comb.txt')
    elem = 'Ni'
    f_norm = 1
    gt_dir = 'img_gt_stack'
    blur_dir = 'img_blur_stack'
    eng_dir = 'img_eng_list'
    thickness_elem=None
    ratio=0.99
    device = 'cuda'
    mask=None
    loss_r = None
    n_train=50
    n_epoch=10
    save_flag=True
    model_prod = RRDBNet(1, 1, 16, 4, 32).to(device)
    model_load_path = '/home/mingyuan/Work/pyxas/example/ml_xanes_denoise/pre_trained_1499.pth'
    model_prod.load_state_dict(torch.load(model_load_path, map_location=device))
    lr = 5e-5
    thickness_update_rate = 2
    main_train_xanes_bkg_production(f_root, img_raw, x_eng, elem, model_prod, loss_r, f_norm,
                                    gt_dir, blur_dir, eng_dir,
                                    thickness_elem, ratio, lr, thickness_update_rate,
                                    mask, n_train, n_epoch, device, save_flag)


def xanes_3D_ml_denoise(img_xanes, x_eng, elem, eng_exclude, fn_root='',
                          thickness_update_rate=20,
                          n_epoch=20,
                          n_train=20,
                          lr=5e-4,
                          order=[-3, -2, -1, 0, 1],
                          loss_l1 = 0,
                          loss_tv = 1e-5,
                          loss_ssim = 0,
                          loss_mse_r2r = 1,
                          loss_mse_fit_img = 1e2,
                          save_flag=False,
                          fn_save='',
                          device='cuda'):

    '''
    xanes slice from 3D_xanes dataset
    it is after -log(), like XRF xanes
    '''

    #param = ML_xanes_default_param(n_epoch=n_epoch, n_train=n_train, lr=lr, order=order)
    param = ML_xanes_default_param(n_epoch, n_train, lr, order, loss_mse_r2r, loss_mse_fit_img, loss_tv, loss_ssim, loss_l1)
    if not len(fn_root):
        fn_root = '.'
    img = img_xanes.copy()

    img_ml, model_prod, _, h_loss_train = ML_fit_xanes(img,
                                         param,
                                         x_eng,
                                         elem,
                                         eng_exclude,
                                         thickness_update_rate,
                                         fn_root,
                                         device)
    if save_flag:
        fn_save_root = fn_root + '/ML'
        mk_directory(fn_save_root)
        if not len(fn_save):
            fn_save = 'ml_denoise.tiff'
        fn_save = fn_save_root + '/' + fn_save
        print(f'file saved to: {fn_save}')
        io.imsave(fn_save, img_ml)
    return img_ml, model_prod, h_loss_train