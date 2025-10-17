import matplotlib.pyplot as plt

import pyxas
import shutil
import numpy as np
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
    from skimage import io
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

def compare_loss(id_type=0, plot_psnr=False):
    fn_root = '/data/xanes_bkg_denoise/IMG_256_stack/re_train_20240213'
    fn_loss = {}
    fn_loss[0] = f'{fn_root}/model_without_fitting_loss'
    fn_loss[1] = f'{fn_root}/model_with_fitting_loss'
    fn_loss[2] = f'{fn_root}/model_with_fitting_loss_lr_5e-4'

    loss_train = {}
    loss_valid = {}
    n = {}
    for i in range(3):
        loss_train[i] = pyxas.load_json(fn_loss[i] + '/h_loss_train.json')
        loss_valid[i] = pyxas.load_json(fn_loss[i] + '/h_loss_valid.json')
        n[i] = len(loss_train[i]['psnr'])

    keys = list(loss_train[1].keys())
    # note that: the first ten iterations of model2 and model3 are directly loaded from training of model1
    # append first 10 iters


    # keys = ['vgg_identity', 'vgg_fit', 'mse_identity_bkg', 'mse_fit_coef', 'mse_fit_self_consist', 'l1_identity', 'psnr']

    train_value = {}
    valid_value = {}
    for i in range(3):
        train_value[i] = {}
        valid_value[i] = {}
        for k in keys:
            if k == 'psnr':
                train_value[i][k] = loss_train[i][k]   # it is a list
                valid_value[i][k] = loss_valid[i][k]
            else:
                rate = loss_train[i][k]['rate'][0]
                if rate == 0: rate = 1
                tmp = np.array(loss_train[i][k]['value']) / rate
                train_value[i][k] = list(tmp)
                tmp1 = np.array(loss_valid[i][k]['value']) / rate
                valid_value[i][k] = list(tmp1)
    # append first 10 iters
    for k in keys:
        train_value[1][k] = train_value[0][k][:10] + train_value[1][k]
        train_value[2][k] = train_value[0][k][:10] + train_value[2][k]
        valid_value[1][k] = valid_value[0][k][:10] + valid_value[1][k]
        valid_value[2][k] = valid_value[0][k][:10] + valid_value[2][k]
    n = len(keys)
    #n_col = int(np.ceil(np.sqrt(n)))
    #n_row = int(np.ceil(n/n_col))
    # without "psnr", n_col = 2, n_row = 2
    n_col, n_row = 3, 2

    title = ['', '', '']
    title[0] = 'w/o fitting loss, lr=1e-4'
    title[1] = 'w/ fitting loss, lr=1e-4'
    title[2] = 'w/ fitting loss, lr=5e-4'
    if not plot_psnr:
        #id_type = 2
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 12})
        plt.suptitle(title[id_type])
        i = 1
        for k in keys:
            if k == 'psnr':
                i = i - 1
                continue
            plt.subplot(n_row, n_col, i)
            plt.plot(train_value[id_type][k], '-', label='train')
            plt.plot(valid_value[id_type][k], '-', label='valid', alpha=0.3)
            plt.legend()


            plt.yscale('log')
            rate = loss_train[id_type][k]['rate'][0]

            if (not rate == 0) and (rate < 1e-3 or rate > 1e3):
                rate = f'{rate:.1e}'
            plt.title(k + f'  (r = {rate})', fontsize=13)
            plt.xlabel('Epochs', fontsize=12)
            i = i + 1
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, hspace=0.3, wspace=0.25)

    if plot_psnr:
        # compare PSNR
        plt.figure()
        for i in range(3):
            if i == 1: alpha = 0.95
            else: alpha=0.8
            plt.plot(train_value[i]['psnr'], '-', label=title[i], alpha=alpha)
        plt.legend()
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('PSNR (train)', fontsize=12)

        plt.figure()
        for i in range(3):
            if i == 1: alpha = 0.5
            else: alpha=0.4
            plt.plot(valid_value[i]['psnr'], '-', label=title[i], alpha=alpha)
        plt.legend()
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('PSNR (valid)', fontsize=12)

        plt.figure(figsize=(16, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.plot(train_value[i]['psnr'], '-', label='train')
            plt.plot(valid_value[i]['psnr'], '-', label='valid', alpha=0.3)
            t = plt.axis()
            plt.axis([t[0]-20, t[1]+20, 16, 49])
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('PSNR', fontsize=12)
            plt.title(title[i])


def sample_train_pair():

    import pyxas
    import torch
    import torchvision
    import torch.nn as nn
    import torch.optim as optim
    import os
    import json
    from skimage import io


    device = torch.device('cuda:1')
    lr = 0.0001
    loss_r = {}

    loss_r['mse_identity_img'] = 1
    loss_r['mse_identity_bkg'] = 1
    loss_r['tv_bkg'] = 0
    loss_r['mse_self_consist'] = 1
    loss_r['ssim_img'] = 1
    loss_r['ssim_bkg'] = 1
    #loss_r['vgg_bkg'] = 1


    torch.manual_seed(0)
    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    vgg19.to(device).eval()


    f_root = '/data/xanes_bkg_denoise/IMG_256_pair/example1'
    img_gt_dir = f_root + '/img_gt_stack'
    blur_dir = f_root + '/img_blur_stack'
    bkg_dir = f_root + '/img_bkg_stack'


    model_gen = pyxas.RRDBNet(1, 1, 16, 4, 32).to(device)
    model_path = f_root + '/tmp_1499.pth'
    #model_path = f_root + '/model_saved/model_0030.pth'
    model_gen.load_state_dict(torch.load(model_path))

    #model_gen = pyxas.RRDBNet_padding_same(1, 1, 16, 3, 32, 'zeros', 5).to(device)

    h_loss_train = {}
    h_loss_valid = {}
    keys = list(loss_r.keys())
    for k in keys:
        h_loss_train[k] = {'value':[], 'rate':[]}
        h_loss_valid[k] = {'value': [], 'rate': []}

    n_train = 100



    train_loader, valid_loader = pyxas.get_train_valid_dataloader_pair(img_gt_dir, blur_dir, bkg_dir, n_train)

    # check: without fitting loss
    img_test = io.imread('img_blur_stack/img_blur_stack_0060.tiff')

    img_output = pyxas.check_model_output(model_gen, img_test, device)
    plt.figure()
    plt.subplot(221)
    plt.imshow(img_output[0]);plt.colorbar()

    plt.subplot(222)
    plt.imshow(img_output[1]);plt.colorbar()

    plt.subplot(223)
    plt.imshow(img_test[0]/img_output[0]);plt.colorbar()

    plt.subplot(224)
    plt.imshow(img_test[1] / img_output[1]);plt.colorbar()

    plt.pause(1)


    for epoch in range(20, 40):
        print(f'epoch = {epoch}:')
        loss_summary_train = pyxas.train_image_pair(model_gen, train_loader, loss_r, vgg19, device, lr)
        h_loss_train, txt_t, psnr_train = pyxas.extract_h_loss(h_loss_train, loss_summary_train, loss_r)
        print(txt_t)

        model_folder = f'{f_root}/model_saved'
        ftmp = f'{model_folder}/model_{epoch:04d}.pth'
        torch.save(model_gen.state_dict(), ftmp)
        with open(f'{model_folder}/h_loss_train.json', 'w') as f:
            json.dump(h_loss_train, f)

        img_output = pyxas.check_model_output(model_gen, img_test, device)
        if epoch % 5 == 0:
            plt.figure()
            plt.subplot(221)
            plt.imshow(img_output[0]);  plt.colorbar()

            plt.subplot(222)
            plt.imshow(img_output[1]);
            plt.colorbar()

            plt.subplot(223)
            plt.imshow(img_test[0] / img_output[0]);
            plt.colorbar()

            plt.subplot(224)
            plt.imshow(img_test[1] / img_output[1]);
            plt.colorbar()
            plt.suptitle(f'epoch = {epoch}')

            plt.pause(1)

