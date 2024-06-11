from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import median_filter as mf
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.transform import rescale
from skimage.feature import peak_local_max
from scipy.ndimage import binary_fill_holes, binary_dilation, center_of_mass, generate_binary_structure

from pyxas.misc import bin_image
import numpy as np


def extract_mask(img, cen, ts=[200, 200, 200]):
    s = img.shape
    sli_s = max(int(cen[0] - ts[0] / 2), 0)
    #sli_e = min(int(cen[0] + ts[0] / 2), s[0])
    sli_e = min(sli_s + ts[0], s[0])

    row_s = max(int(cen[1] - ts[1] / 2), 0)
    #row_e = min(int(cen[1] + ts[1] / 2), s[1])
    row_e = min(row_s + ts[1], s[1])

    col_s = max(int(cen[2] - ts[2] / 2), 0)
    #col_e = min(int(cen[2] + ts[2] / 2), s[2])
    col_e = min(col_s + ts[2], s[2])

    l1 = sli_e - sli_s
    l2 = row_e - row_s
    l3 = col_e - col_s

    ss = (ts[0] - l1) // 2
    rs = (ts[1] - l2) // 2
    cs = (ts[2] - l3) // 2

    mask = np.zeros(ts)
    mask[ss:ss + l1, rs:rs + l2, cs:cs + l3] = img[sli_s:sli_e, row_s:row_e, col_s:col_e]
    return mask


def watershed_mask(img_raw, binning=2, gf_size=5, fs=15, min_distance=5, thresh=None, fill_hole=True):
    s = np.array(img_raw.shape)
    '''
    if len(s) == 3:
        img = img_raw[:s[0], :s[1], :s[2]]
    if len(s) == 2:
        img = img_raw[:s[0], :s[1]]
    '''
    if binning == 2:
        s = np.int16(s // 2 * 2)
        img = img_raw[:s[0], :s[1]]
        img = bin_image(img, 2)
    else:
        img = img_raw.copy()
    s = img.shape
    print('gaussian filtering...')
    #img_gf = gf(img, gf_size)
    img_gf = mf(img, gf_size)
    if thresh is None:
        thresh = threshold_otsu(img_gf)
    bw = img_gf.copy()
    bw[bw < thresh] = 0
    bw[bw >= thresh] = 1
    n = len(bw.shape)
    struct = generate_binary_structure(n, 1)
    if fill_hole:
        bw = binary_fill_holes(bw, structure=struct).astype(bw.dtype)
    print('cal. distance map...')
    distance = ndi.distance_transform_edt(bw)
    if len(s) == 3:
        coords = peak_local_max(distance, min_distance=min_distance, footprint=np.ones((fs, fs, fs)),
                                labels=np.int32(bw))
    elif len(s) == 2:
        coords = peak_local_max(distance, min_distance=min_distance, footprint=np.ones((fs, fs)), labels=np.int32(bw))
    else:
        return 0, 0
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=bw)
    if binning == 2:
        labels = rescale(labels, 2, order=0)
        bw = rescale(bw, 2, order=0)
    return labels, bw

def match_img_label(img, img_label, val, ms=200, dilation_iter=0):
    mask = img_label.copy()
    mask[mask == val] = 1000
    mask[mask < 1000] = 0
    mask = mask / 1000
    mask = binary_fill_holes(mask, structure=np.ones((3, 3, 3)))
    struct_dia = generate_binary_structure(3, 3)
    if dilation_iter > 0:
        mask = binary_dilation(mask, structure=struct_dia, iterations=dilation_iter)
    mask_area = mask * img
    cen = center_of_mass(mask_area)
    m = extract_mask(mask_area, cen, [ms, ms, ms])
    return m