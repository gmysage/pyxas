import glob
import numpy as np
from skimage import io
from torch.utils.data import DataLoader, Dataset
import torch


class xanesDataset(Dataset):
    def __init__(self, blur_dir, gt_dir, eng_dir, length=None, transform_gt=None, transform_blur=None, gt_threshold=1):
        super().__init__()
        self.fn_blur = np.sort(glob.glob(f'{blur_dir}/*'))
        self.fn_gt = np.sort(glob.glob(f'{gt_dir}/*'))
        self.fn_eng = np.sort(glob.glob(f'{eng_dir}/*'))
        self.gt_threshold = gt_threshold

        self.transform_gt = transform_gt
        self.transform_blur = transform_blur
        if not length is None:
            self.fn_blur = self.fn_blur[:length]
            self.fn_gt = self.fn_gt[:length]
            self.fn_eng = self.fn_eng[:length]

    def __len__(self):
        return len(self.fn_blur)

    def __getitem__(self, idx):
        img_blur = io.imread(self.fn_blur[idx]) # (8, 512, 512)
        img_gt = io.imread(self.fn_gt[idx]) # (8, 512, 512)
        m = img_gt[img_gt < self.gt_threshold]
        scale = np.median(m)

        if self.transform_gt:
            img_gt = self.transform_gt(img_gt)        
        if self.transform_blur:
            img_blur = self.transform_blur(img_blur)

        img_blur = torch.tensor(img_blur, dtype=torch.float)        
        img_gt = torch.tensor(img_gt, dtype=torch.float)
        eng_list = np.loadtxt(self.fn_eng[idx])
        eng_list = torch.tensor(eng_list, dtype=torch.float)
        elem = self.fn_eng[idx].split('/')[-1].split('.')[0].split('_')[-1]
        
        return img_blur, img_gt, eng_list, elem



class xanesDataset_new(Dataset):
    def __init__(self, blur_dir, gt_bkg_dir, gt_img_dir, eng_dir, length=None, transform_gt=None, transform_blur=None, gt_threshold=1000):
        super().__init__()
        self.fn_blur = np.sort(glob.glob(f'{blur_dir}/*'))
        self.fn_gt_bkg = np.sort(glob.glob(f'{gt_bkg_dir}/*'))
        self.fn_gt_img = np.sort(glob.glob(f'{gt_img_dir}/*'))
        self.fn_eng = np.sort(glob.glob(f'{eng_dir}/*'))
        self.gt_threshold = gt_threshold

        self.transform_gt = transform_gt
        self.transform_blur = transform_blur
        if not length is None:
            self.fn_blur = self.fn_blur[:length]
            self.fn_gt_bkg = self.fn_gt_bkg[:length]
            self.fn_gt_img = self.fn_gt_img[:length]
            self.fn_eng = self.fn_eng[:length]

    def __len__(self):
        return len(self.fn_blur)

    def __getitem__(self, idx):
        blur_img = io.imread(self.fn_blur[idx]) # (8, 512, 512)
        gt_bkg = io.imread(self.fn_gt_bkg[idx]) # (8, 512, 512)
        gt_img = io.imread(self.fn_gt_img[idx]) # (8, 512, 512)
        m = gt_bkg[gt_bkg < self.gt_threshold]
        scale = np.median(m)

        if self.transform_gt:
            gt_img = self.transform_gt(gt_img) 
            gt_bkg = self.transform(gt_bkg)      
        if self.transform_blur:
            blur_img = self.transform_blur(blur_img)

        blur_img = torch.tensor(blur_img, dtype=torch.float)        
        gt_img = torch.tensor(gt_img, dtype=torch.float)
        gt_bkg = torch.tensor(gt_bkg, dtype=torch.float)
        eng_list = np.loadtxt(self.fn_eng[idx])
        eng_list = torch.tensor(eng_list, dtype=torch.float)
        elem = self.fn_eng[idx].split('/')[-1].split('.')[0].split('_')[-1]
        
        return blur_img, gt_bkg, gt_img, eng_list, elem


class Dataset_pair(Dataset):
    def __init__(self, img_gt_dir, blur_dir, bkg_dir, length=None):
        super().__init__()
        self.fn_img = np.sort(glob.glob(f'{img_gt_dir}/*'))
        self.fn_blur = np.sort(glob.glob(f'{blur_dir}/*'))
        self.fn_bkg = np.sort(glob.glob(f'{bkg_dir}/*'))
        if not length is None:
            self.fn_img = self.fn_img[:length]

    def __len__(self):
        return len(self.fn_img)

    def __getitem__(self, idx):
        img_gt = io.imread(self.fn_img[idx])  # (2, 512, 512)
        img_gt = torch.tensor(img_gt, dtype=torch.float)

        img_blur = io.imread(self.fn_blur[idx])
        img_blur = torch.tensor(img_blur, dtype=torch.float)

        img_bkg = io.imread(self.fn_bkg[idx])
        img_bkg = torch.tensor(img_bkg, dtype=torch.float)
        return img_gt, img_blur, img_bkg


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


def get_train_valid_dataloader_new(blur_dir, gt_bkg_dir, gt_img_dir, eng_dir, num, transform_gt=None, transform_blur=None, split_ratio=0.8):
    dataset = xanesDataset_new(blur_dir, gt_bkg_dir, gt_img_dir, eng_dir, num, transform_gt, transform_blur)
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


def get_train_valid_dataloader_pair(img_gt_dir, blur_dir, bkg_dir, length=None, split_ratio=0.8):
    dataset = Dataset_pair(img_gt_dir, blur_dir, bkg_dir, length)
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