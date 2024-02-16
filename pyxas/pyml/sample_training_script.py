import pyxas
import shutil

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import json
from tqdm import tqdm, trange

def sample_train():
    device = torch.device('cuda:3')
    #lr = 0.001
    loss_r = {}
    loss_r['vgg_identity'] = 1           # (model_outputs vs. label); "0" for "Production", "1" for trainning
    loss_r['vgg_fit'] = 1e-1                # (fitted_image vs. label); "1" for both "trainning" and "production"
    loss_r['mse_identity_bkg'] = 1
    loss_r['mse_fit_coef'] = 0           # (fit_coef_from_model_outputs vs. fit_coef_from_label); "1e8" for both "trainning" and "production"
    loss_r['mse_fit_self_consist'] = 0   # (fitting_output_from_model_output vs. model_outputs ); "1" for both "trainning" and "production"
    loss_r['l1_identity'] = 0

    global vgg19

    torch.manual_seed(0)
    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    vgg19.to(device).eval()

    model_gen = pyxas.RRDBNet(1, 1, 16, 4, 32).to(device)

    '''
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(0.5, 0.999))
    '''
    h_loss_train = {}
    h_loss_valid = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value':[], 'rate':[]}
        h_loss_valid[k] = {'value': [], 'rate': []}

    n_train = 200

    f_root = '/data/xanes_bkg_denoise/IMG_256_stack/Co3'
    blur_dir = f_root + '/img_blur_stack'
    gt_dir = f_root + '/img_bkg_stack_gf'
    eng_dir = f_root + '/img_eng_list'
    trans_gt, trans_blur = None, None

    train_loader, valid_loader = pyxas.get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, trans_gt, trans_blur)

    # check: without fitting loss
    for epoch in range(400, 800):
        print(f'epoch = {epoch}:')
        loss_summary_train = pyxas.train_1_branch_bkg(model_gen, train_loader, loss_r, vgg19, device, train_fit=False)
        h_loss_train, txt_t, psnr_train = pyxas.extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        loss_summary_valid = pyxas.validate(model_gen, valid_loader, loss_r, vgg19, device)
        h_loss_valid, txt_v, psnr_valid = pyxas.extract_h_loss(h_loss_valid, loss_summary_valid, loss_r)
        print(txt_t)
        print(txt_v)

        fn_root = '/data/xanes_bkg_denoise/IMG_256_stack/re_train_20240213'
        model_folder = f'{fn_root}/model_without_fitting_loss'
        ftmp = f'{model_folder}/model_{epoch:04d}.pth'
        torch.save(model_gen.state_dict(), ftmp)
        with open(f'{model_folder}/h_loss_train.json', 'w') as f:
            json.dump(h_loss_train, f)
        with open(f'{model_folder}/h_loss_valid.json', 'w') as f:
            json.dump(h_loss_valid, f)




def sample_train_more_constraints():
    ####
    # load 100th model and add additional constraint

    device = torch.device('cuda:1')
    global vgg19

    torch.manual_seed(0)
    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    vgg19.to(device).eval()

    loss_r = {}
    loss_r['vgg_identity'] = 1  # (model_outputs vs. label); "0" for "Production", "1" for trainning
    loss_r['vgg_fit'] = 1e-1  # (fitted_image vs. label); "1" for both "trainning" and "production"
    loss_r['mse_identity_bkg'] = 1
    loss_r['mse_fit_coef'] = 1e10  # (fit_coef_from_model_outputs vs. fit_coef_from_label); "1e8" for both "trainning" and "production"
    loss_r['mse_fit_self_consist'] = 10  # (fitting_output_from_model_output vs. model_outputs ); "1" for both "trainning" and "production"
    loss_r['l1_identity'] = 0


    h_loss_train = {}
    h_loss_valid = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value':[], 'rate':[]}
        h_loss_valid[k] = {'value': [], 'rate': []}

    fn_root = '/data/xanes_bkg_denoise/IMG_256_stack/re_train_20240213'
    model_folder_exist = f'{fn_root}/model_without_fitting_loss'

    model_path = f'{model_folder_exist}/model_{20:04d}.pth'
    model_gen2 = pyxas.RRDBNet(1, 1, 16, 4, 32).to(device)
    model_gen2.load_state_dict(torch.load(model_path, map_location=device))


    n_train = 200

    f_root = '/data/xanes_bkg_denoise/IMG_256_stack/Co3'
    blur_dir = f_root + '/img_blur_stack'
    gt_dir = f_root + '/img_bkg_stack_gf'
    eng_dir = f_root + '/img_eng_list'
    trans_gt, trans_blur = None, None

    train_loader, valid_loader = pyxas.get_train_valid_dataloader(blur_dir, gt_dir, eng_dir, n_train, trans_gt, trans_blur)


    lr = 5e-4 # default 1e-4
    for epoch in range(20, 500):
        print(f'epoch = {epoch}:')
        loss_summary_train = pyxas.train_1_branch_bkg(model_gen2, train_loader, loss_r, vgg19, device, lr=lr, train_fit=True)
        h_loss_train, txt_t, psnr_train = pyxas.extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        loss_summary_valid = pyxas.validate(model_gen2, valid_loader, loss_r, vgg19, device)
        h_loss_valid, txt_v, psnr_valid = pyxas.extract_h_loss(h_loss_valid, loss_summary_valid, loss_r)
        print('training loss:')
        print(txt_t)
        print('validation loss:')
        print(txt_v)
        print('\n')

        fn_root = '/data/xanes_bkg_denoise/IMG_256_stack/re_train_20240213'
        #model_folder = f'{fn_root}/model_with_fitting_loss'
        model_folder = f'{fn_root}/model_with_fitting_loss_lr_5e-4'
        ftmp = f'{model_folder}/model_{epoch:04d}.pth'
        torch.save(model_gen2.state_dict(), ftmp)
        with open(f'{model_folder}/h_loss_train.json', 'w') as f:
            json.dump(h_loss_train, f)
        with open(f'{model_folder}/h_loss_valid.json', 'w') as f:
            json.dump(h_loss_valid, f)

    ###
    #model_folder = f'{fn_root}/model_without_fitting_loss'
    model_folder = f'{fn_root}/model_with_fitting_loss'
    fn_loss_train = f'{model_folder}/h_loss_train.json'
    fn_loss_valid = f'{model_folder}/h_loss_valid.json'
    loss_train = pyxas.load_json(fn_loss_train)
    loss_valid = pyxas.load_json(fn_loss_valid)
    pyxas.plot_h_loss(loss_train)
    pyxas.plot_h_loss(loss_valid)