import numpy as np
from scipy import ndimage
import scipy.fftpack as sf
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.transform import resize
from skimage.filters import threshold_otsu, threshold_yen
from multiprocessing import Pool, cpu_count
from tqdm import tqdm, trange
from functools import partial

def rm_abnormal(img):
    tmp = img.copy()
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    tmp[tmp < 0] = 0

    return tmp

def check_and_swap_img_axis(img):
    s = img.shape
    img_s = img.copy()
    if np.min(s) == s[-1] and len(s)==3:
        img_s = np.swapaxes(img_s, 1, 2)
        img_s = np.swapaxes(img_s, 0, 1)
    return img_s

def img_smooth(img, kernal_size, axis=0):
    s = img.shape
    if len(s) == 2:
        img_stack = img.reshape(1, s[0], s[1])
    else:
        img_stack = img.copy()

    if axis == 0:
        for i in range(img_stack.shape[0]):
            img_stack[i] = medfilt2d(img_stack[i], kernal_size)
    elif axis == 1:
        for i in range(img_stack.shape[1]):
            img_stack[:, i] = medfilt2d(img_stack[:,i], kernal_size)
    elif axis == 2:
        for i in range(img_stack.shape[2]):
            img_stack[:, :, i] = medfilt2d(img_stack[:,:, i], kernal_size)
    return img_stack


def otsu_mask(img, kernal_size, iters=1, bins=256, erosion_iter=0):
    img_s = img.copy()
    img_s[np.isnan(img_s)] = 0
    img_s[np.isinf(img_s)] = 0
    for i in range(iters):
        img_s = img_smooth(img_s, kernal_size)
    thresh = threshold_otsu(img_s, nbins=bins)
    mask = np.zeros(img_s.shape)
    #mask = np.float32(img_s > thresh)
    mask[img_s > thresh] = 1
    mask = np.squeeze(mask)
    if erosion_iter:
        struct = ndimage.generate_binary_structure(2, 1)
        struct1 = ndimage.iterate_structure(struct, 2).astype(int)
        mask = ndimage.binary_erosion(mask, structure=struct1).astype(mask.dtype)
    mask[:erosion_iter+1] = 1
    mask[-erosion_iter-1:] = 1
    mask[:, :erosion_iter+1] = 1
    mask[:, -erosion_iter-1:] = 1
    return mask

def otsu_mask_stack(img, kernal_size, iters=1, bins=256, erosion_iter=0):
    s = img.shape
    img_m = np.zeros(s)
    for i in trange(s[0]):
        img_m[i] = otsu_mask(img[i], kernal_size, iters, bins, erosion_iter)
    img_r = img * img_m
    return img_r
        


def rm_noise(img, noise_level=2e-3, filter_size=3):
    img_s = medfilt2d(img, filter_size)
    id0 = img_s==0
    img_s[id0] = img[id0]
    img_diff = np.abs(img - img_s)
    index = img_diff > noise_level
    img_m = img.copy()
    img_m[index] = img_s[index]
    return img_m


def rm_noise2(img, noise_level=0.02, filter_size=3):
    img_s = medfilt2d(img, filter_size)
    id0 = img_s==0
    img_s[id0] = img[id0]
    img_diff = (img - img_s) / img
    index = img_diff > noise_level
    img_m = img.copy()
    img_m[index] = img_s[index]
    return img_m
    
    
def img_denoise_bm3d(img, sigma=0.01):
    try:
        import bm3d
        s = img.shape
        if len(s) == 2:
            img_stack = img.reshape(1, s[0], s[1])
        else:
            img_stack = img.copy()
        img_d = img_stack.copy()
        n = img_stack.shape[0]
        for i in range(n):
            img_d[i] = bm3d.bm3d(img_stack[i], sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        return img_d
    except Exception as err:
        print(err)
        return img


def img_denoise_nl_single(img, patch_size=5, patch_distance=6):
    img_d = img.copy()
    patch_kw = dict(patch_size=patch_size,  # 5x5 patches
                    patch_distance=patch_distance,  # 13x13 search area
                    )
    sigma_est = np.mean(estimate_sigma(img_d))
    img_d = denoise_nl_means(img_d, h=1.2 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
    return img_d


def img_denoise_nl(img, patch_size=5, patch_distance=6):
    s = img.shape
    if len(s) == 2:
        img_stack = img.reshape(1, s[0], s[1])
    else:
        img_stack = img.copy()
    img_d = img_stack.copy()
    n = img_stack.shape[0]
    for i in range(n):
        img_d[i] = img_denoise_nl_single(img_stack[i], patch_size, patch_distance)
    return img_d


def img_denoise_nl_mpi(img, patch_size=5, patch_distance=6, n_cpu=8):
    max_cpu = round(cpu_count() * 0.8)
    n_cpu = min(n_cpu, max_cpu)
    n_cpu = max(n_cpu, 1)
    s = img.shape
    if len(s) == 2:
        img_stack = img.reshape(1, s[0], s[1])
    else:
        img_stack = img.copy()
    img_d = img_stack.copy()
    pool = Pool(n_cpu)
    res = []
    partial_func = partial(img_denoise_nl_single, patch_size=patch_size, patch_distance=patch_distance)
    for result in tqdm(pool.imap(func=partial_func, iterable=img), total=len(img)):
        res.append(result)
    pool.close()
    pool.join()
    img_d = np.array(res)
    return img_d


def img_fillhole(img, binary_threshold=0.5):
    img_b = img.copy()
    img_b[np.isnan(img_b)] = 0
    img_b[np.isinf(img_b)] = 0
    img_b[img_b > binary_threshold] = 1
    img_b[img_b < 1] = 0

    struct = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_fill_holes(img_b, structure=struct).astype(img.dtype)
    img_fillhole = img * mask
    return mask, img_fillhole



def img_dilation(img, binary_threshold=0.5, iterations=2):
    img_b = img.copy()
    img_b[np.isnan(img_b)] = 0
    img_b[np.isinf(img_b)] = 0
    img_b[img_b > binary_threshold] = 1
    img_b[img_b < 1] = 0

    struct = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_dilation(img_b, structure=struct, iterations=iterations).astype(img.dtype)
    img_dilated = img * mask
    return mask, img_dilated


def img_erosion(img, binary_threshold=0.5, iterations=2):
    img_b = img.copy()
    img_b[np.isnan(img_b)] = 0
    img_b[np.isinf(img_b)] = 0
    img_b[img_b > binary_threshold] = 1
    img_b[img_b < 1] = 0

    struct = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_erosion(img_b, structure=struct, iterations=iterations).astype(img.dtype)
    img_erosion = img * mask
    return mask, img_erosion






def _get_mask(dx, dy, ratio):
    """
    Calculate 2D boolean circular mask.

    Parameters
    ----------
    dx, dy : int
        Dimensions of the 2D mask.

    ratio : int
        Ratio of the circle's diameter in pixels to
        the smallest mask dimension.

    Returns
    -------
    ndarray
        2D boolean array.
    """
    rad1 = dx / 2.
    rad2 = dy / 2.
    if dx > dy:
        r2 = rad1 * rad1
    else:
        r2 = rad2 * rad2
    y, x = np.ogrid[0.5 - rad1:0.5 + rad1, 0.5 - rad2:0.5 + rad2]
    return x * x + y * y < ratio * ratio * r2

def circ_mask(img, axis, ratio=1, val=0):
    im = np.float32(img)
    s = im.shape
    if len(s) == 2:
        m = _get_mask(s[0], s[1], ratio)
        m_out = (1 - m) * val
        im_m = np.array(m, dtype=np.int) * im + m_out
    else:
        im = im.swapaxes(0, axis)
        dx, dy, dz = im.shape
        m = _get_mask(dy, dz, ratio)
        m_out = (1 - m) * val
        im_m = np.array(m, dtype=np.int16) * im + m_out
        im_m = im_m.swapaxes(0, axis)
    return im_m


def pad(img, thick, direction):

    """
    symmetrically padding the image with "0"

    Parameters:
    -----------
    img: 2d or 3d array
        2D or 3D images
    thick: int
        padding thickness for all directions
        if thick == odd, automatically increase it to thick+1
    direction: int
        0: padding in axes = 0 (2D or 3D image)
        1: padding in axes = 1 (2D or 3D image)
        2: padding in axes = 2 (3D image)

    Return:
    -------
    2d or 3d array

    """

    thick = np.int32(thick)
    if thick%2 == 1:
        thick = thick + 1
        print('Increasing padding thickness to: {}'.format(thick))

    img = np.array(img)
    s = np.array(img.shape)

    if thick == 0 or direction > 3 or s.size > 3:
        return img

    hf = np.int32(np.ceil(abs(thick)+1) / 2)  # half size of padding thickness
    if thick > 0:
        if s.size < 3:  # 2D image
            if direction == 0: # padding row
                pad_image = np.zeros([s[0]+thick, s[1]])
                pad_image[hf:(s[0]+hf), :] = img

            else:  # direction == 1, padding colume
                pad_image = np.zeros([s[0], s[1]+thick])
                pad_image[:, hf:(s[1]+hf)] = img

        else:  # s.size ==3, 3D image
            if direction == 0:  # padding slice
                pad_image = np.zeros([s[0]+thick, s[1], s[2]])
                pad_image[hf:(s[0]+hf), :, :] = img

            elif direction ==1:  # padding row
                pad_image = np.zeros([s[0], s[1]+thick, s[2]])
                pad_image[:, hf:(s[1]+hf), :] = img

            else:  # padding colume
                pad_image = np.zeros([s[0],s[1],s[2]+thick])
                pad_image[:, :, hf:(s[2]+hf)] = img

    else: # thick < 0: shrink the image
        if s.size < 3:  # 2D image
            if direction == 0:  # shrink row
                pad_image = img[hf:(s[0]-hf), :]

            else:  pad_image = img[:, hf:(s[1]-hf)]    # shrink colume

        else:  # s.size == 3, 3D image
            if direction == 0:  # shrink slice
                pad_image = img[hf:(s[0]-hf), :, :]

            elif direction == 1:  # shrink row
                pad_image = img[:, hf:(s[1]-hf),:]

            else:  # shrik colume
                pad_image = img[:, :, hf:(s[2]-hf)]
    return pad_image


def img_analysis(img, n_comp=2):
    from sklearn.cluster import KMeans
    img_reshape = img.transpose(1, 2, 0)
    s = img_reshape.shape
    img_reshape_flat = img_reshape.reshape(-1, s[2])

    result = {}
    kmeans = KMeans(n_clusters=n_comp, random_state=0).fit(img_reshape_flat)
    img_values = kmeans.cluster_centers_
    img_labels = kmeans.labels_
    img_compress = np.zeros([s[0] * s[1], s[2]])
    for i in range(s[2]):
        img_compress[:, i] = np.choose(img_labels, img_values[:, i])

    img_labels = img_labels.reshape(s[0], s[1])
    img_compress = img_compress.reshape(s)
    img_compress = img_compress.transpose(2, 0, 1)
    result['method'] = 'kmean'
    result['img_compress'] = img_compress
    result['img_labels'] = img_labels
    result['img_values'] = img_values
    return result


def kmean_mask(img, n_comp=2, index_select=-1):
    import numpy as np
    from scipy import ndimage
    flag_3d_image = 1
    s = img.shape
    if len(s) == 2:
        img3D = img.reshape([1, s[0], s[1]])
        s = img3D.shape
        flag_3d_image = 0
    else:
        img3D = img.copy()
    res = img_analysis(img3D, n_comp=n_comp)
    img_compress = res['img_compress']  # shape = (s[0], s[1], s[1]), e.g., (91, 750, 750)
    img_values = res['img_values']  # shape = (n_comp, s[0])      e.g., (2, 91)
    img_labels = res['img_labels']  # shape = (s[1], s[1])        e.g., (750, 750)
    mask_comp = np.zeros([n_comp, s[1], s[2]])
    try:
        img_labels_copy = img_labels.copy()
        val = img_values[:, index_select]
        val_sort = np.sort(val)[::-1]
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, 2).astype(int)
        for i in range(n_comp):
            id_mask = np.squeeze(np.where(val == val_sort[i]))
            mask = np.zeros(img_labels.shape)
            mask[img_labels_copy == id_mask] = 1
            img_labels[img_labels_copy == id_mask] = n_comp - i - 1
            #mask, _ = img_fill_holes(mask, struct=struct)
            mask_comp[i] = mask
    except:
        pass
    if flag_3d_image == 0:
        return mask_comp, img_labels, img_compress
    else:
        return mask_comp, img_labels


def kmean_mask_line_by_line_2D(img):
    s = img.shape
    img_m = np.zeros(s)
    for i in trange(s[0]):
        l = img[i]
        l = l.reshape([len(l), 1])
        mask_comp, img_labels, img_compress = kmean_mask(l, n_comp=2)
        img_m[i] = np.squeeze(mask_comp[0]) * img[i]
        
    return img_m
        
        
def bin_ndarray(ndarray, new_shape=None, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if new_shape == None:
        s = np.array(ndarray.shape)
        s1 = np.int32(s/2)
        new_shape = tuple(s1)
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def draw_circle(cen, r, theta=[0, 360.0]):
    th = np.linspace(theta[0]/180.*np.pi, theta[1]/180.*np.pi, 361)
    x = r * np.cos(th) + cen[0]
    y = r * np.sin(th) + cen[1]
    plt.plot(x,y,'r')

def get_circle_line_from_img(img, cen, r, pix_size=17.1, theta=[0, 360.0], f_out='circle_profile_with_fft.txt'):
    d_th = 1 / 10.0 / r
    th = np.arange(theta[0]/180.*np.pi, theta[1]/180.0*np.pi+d_th, d_th)
    num_data = len(th)
    x = r * np.sin(th) + cen[1]
    y = r * np.cos(th) + cen[0]

    x_int = np.int32(np.floor(x)); x_frac = x - x_int
    y_int = np.int32(np.floor(y)); y_frac = y - y_int


    data = []
    for i in range(num_data):
        t1 = img[x_int[i], y_int[i]] * (1 - x_frac[i]) * (1 - y_frac[i])
        t2 = img[x_int[i], y_int[i]+1] * (1 - x_frac[i]) * y_frac[i]
        t3 = img[x_int[i]+1, y_int[i]] * x_frac[i] * (1 - y_frac[i])
        t4 = img[x_int[i]+1, y_int[i]+1] * x_frac[i] * y_frac[i]
        t = t1 + t2 + t3 + t4
        data.append(t)

    line = th * r * pix_size

    plt.figure()
    plt.subplot(221)
    plt.imshow(img)
    draw_circle(cen, r, theta)

    plt.subplot(223);plt.plot(line, data)
    plt.title('line_profile: r={} pixels'.format(r))

    data_fft = np.fft.fftshift(np.fft.fft(data))
    fs = 1/(pix_size/10)
    f = fs/2 * np.linspace(-1, 1, len(data_fft))
    plt.subplot(224);plt.plot(f, np.abs(data_fft))
    plt.xlim([-0.04,0.04])
    plt.ylim([-10, 300])
    plt.title('fft of line_profile')

    # combine data to sigle variable and save it
    data_comb = np.zeros([len(data), 4])
    data_comb[:,0] = line
    data_comb[:,1] = data
    data_comb[:,2] = f
    data_comb[:,3] = np.abs(data_fft)

    np.savetxt(f_out, data_comb, fmt='%3.4e')
    return data_comb



class IndexTracker(object):
    def __init__(self, ax, X, cmap):
        self.ax = ax
        self._indx_txt = ax.set_title(' ', loc='center')
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[self.ind, :, :], cmap=cmap)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        #self.ax.set_ylabel('slice %s' % self.ind)
        self._indx_txt.set_text(f"frame {self.ind + 1} of {self.slices}")
        self.im.axes.figure.canvas.draw()


def image_movie(data, ax=None, cmap='gray'):
    # show a movie of image in python environment
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    tracker = IndexTracker(ax, data, cmap)
    # monkey patch the tracker onto the figure to keep it alive
    fig._tracker = tracker
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return tracker


def crop_scale_image(img_stack, output_size=(256, 256)):
    img = img_stack.copy()
    s = img.shape
    if len(s) == 2:
        img = np.expand_dims(img, axis=0)
    s = img.shape
    img_resize = resize(img, (s[0], output_size[0], output_size[1]))
    return np.squeeze(img_resize)


def scale_img_xanes(img_xanes):
    smart_mask, img_labels = kmean_mask(img_xanes, 2)
    mask = smart_mask[0]
    n = img_xanes.shape[0]
    img_mean = np.sum(img_xanes * mask, axis=0) / n
    img_sum = np.sum(img_mean)
    val = img_sum / np.sum(mask)

    f_scale = 1. / val * 0.8
    img_scale = img_xanes * f_scale
    return img_scale, f_scale, mask


from PIL import Image
_errstr = "Mode is unknown or incompatible with input array shape."


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.

    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image
###################################################################


def dftregistration(buf1ft,buf2ft,usfac=100):
   """
       # function [output Greg] = dftregistration(buf1ft,buf2ft,usfac);
       # Efficient subpixel image registration by crosscorrelation. This code
       # gives the same precision as the FFT upsampled cross correlation in a
       # small fraction of the computation time and with reduced memory
       # requirements. It obtains an initial estimate of the
crosscorrelation peak
       # by an FFT and then refines the shift estimation by upsampling the DFT
       # only in a small neighborhood of that estimate by means of a
       # matrix-multiply DFT. With this procedure all the image points
are used to
       # compute the upsampled crosscorrelation.
       # Manuel Guizar - Dec 13, 2007

       # Portions of this code were taken from code written by Ann M. Kowalczyk
       # and James R. Fienup.
       # J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued
       # object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458
       # (1990).

       # Citation for this algorithm:
       # Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
       # "Efficient subpixel image registration algorithms," Opt. Lett. 33,
       # 156-158 (2008).

       # Inputs
       # buf1ft    Fourier transform of reference image,
       #           DC in (1,1)   [DO NOT FFTSHIFT]
       # buf2ft    Fourier transform of image to register,
       #           DC in (1,1) [DO NOT FFTSHIFT]
       # usfac     Upsampling factor (integer). Images will be registered to
       #           within 1/usfac of a pixel. For example usfac = 20 means the
       #           images will be registered within 1/20 of a pixel.
(default = 1)

       # Outputs
       # output =  [error,diffphase,net_row_shift,net_col_shift]
       # error     Translation invariant normalized RMS error between f and g
       # diffphase     Global phase difference between the two images (should be
       #               zero if images are non-negative).
       # net_row_shift net_col_shift   Pixel shifts between images
       # Greg      (Optional) Fourier transform of registered version of buf2ft,
       #           the global phase difference is compensated for.
   """

   # Compute error for no pixel shift
   if usfac == 0:
       CCmax = np.sum(buf1ft*np.conj(buf2ft))
       rfzero = np.sum(abs(buf1ft)**2)
       rgzero = np.sum(abs(buf2ft)**2)
       error = 1.0 - CCmax*np.conj(CCmax)/(rgzero*rfzero)
       error = np.sqrt(np.abs(error))
       diffphase = np.arctan2(np.imag(CCmax),np.real(CCmax))
       return error, diffphase

   # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the
   # peak
   elif usfac == 1:
       ndim = np.shape(buf1ft)
       m = ndim[0]
       n = ndim[1]
       CC = sf.ifft2(buf1ft*np.conj(buf2ft))
       max1,loc1 = idxmax(CC)
       rloc = loc1[0]
       cloc = loc1[1]
       CCmax=CC[rloc,cloc]
       rfzero = np.sum(np.abs(buf1ft)**2)/(m*n)
       rgzero = np.sum(np.abs(buf2ft)**2)/(m*n)
       error = 1.0 - CCmax*np.conj(CCmax)/(rgzero*rfzero)
       error = np.sqrt(np.abs(error))
       diffphase=np.arctan2(np.imag(CCmax),np.real(CCmax))
       md2 = np.fix(m/2)
       nd2 = np.fix(n/2)
       if rloc > md2:
           row_shift = rloc - m
       else:
           row_shift = rloc

       if cloc > nd2:
           col_shift = cloc - n
       else:
           col_shift = cloc

       ndim = np.shape(buf2ft)
       nr = int(round(ndim[0]))
       nc = int(round(ndim[1]))
       Nr = sf.ifftshift(np.arange(-np.fix(1.*nr/2),np.ceil(1.*nr/2)))
       Nc = sf.ifftshift(np.arange(-np.fix(1.*nc/2),np.ceil(1.*nc/2)))
       Nc,Nr = np.meshgrid(Nc,Nr)
       Greg = buf2ft*np.exp(1j*2*np.pi*(-1.*row_shift*Nr/nr-1.*col_shift*Nc/nc))
       Greg = Greg*np.exp(1j*diffphase)
       image_reg = sf.ifft2(Greg) * np.sqrt(nr*nc)

       #return error,diffphase,row_shift,col_shift
       return error,diffphase,row_shift,col_shift, image_reg

   # Partial-pixel shift
   else:

       # First upsample by a factor of 2 to obtain initial estimate
       # Embed Fourier data in a 2x larger array
       ndim = np.shape(buf1ft)
       m = int(round(ndim[0]))
       n = int(round(ndim[1]))
       mlarge=m*2
       nlarge=n*2
       CC=np.zeros([mlarge,nlarge],dtype=np.complex128)

       CC[int(m-np.fix(m/2)):int(m+1+np.fix((m-1)/2)),int(n-np.fix(n/2)):int(n+1+np.fix((n-1)/2))] = (sf.fftshift(buf1ft)*np.conj(sf.fftshift(buf2ft)))[:,:]


       # Compute crosscorrelation and locate the peak
       CC = sf.ifft2(sf.ifftshift(CC)) # Calculate cross-correlation
       max1,loc1 = idxmax(np.abs(CC))

       rloc = int(round(loc1[0]))
       cloc = int(round(loc1[1]))
       CCmax = CC[rloc,cloc]

       # Obtain shift in original pixel grid from the position of the
       # crosscorrelation peak
       ndim = np.shape(CC)
       m = ndim[0]
       n = ndim[1]

       md2 = np.fix(m/2)
       nd2 = np.fix(n/2)
       if rloc > md2:
           row_shift = rloc - m
       else:
           row_shift = rloc

       if cloc > nd2:
           col_shift = cloc - n
       else:
           col_shift = cloc

       row_shift=row_shift/2
       col_shift=col_shift/2

       # If upsampling > 2, then refine estimate with matrix multiply DFT
       if usfac > 2:
           ### DFT computation ###
           # Initial shift estimate in upsampled grid
           row_shift = 1.*np.round(row_shift*usfac)/usfac;
           col_shift = 1.*np.round(col_shift*usfac)/usfac;
           dftshift = np.fix(np.ceil(usfac*1.5)/2); ## Center of output array at dftshift+1
           # Matrix multiply DFT around the current shift estimate
           CC = np.conj(dftups(buf2ft*np.conj(buf1ft),np.ceil(usfac*1.5),np.ceil(usfac*1.5),usfac,\
               dftshift-row_shift*usfac,dftshift-col_shift*usfac))/(md2*nd2*usfac**2)
           # Locate maximum and map back to original pixel grid
           max1,loc1 = idxmax(np.abs(CC))
           rloc = int(round(loc1[0]))
           cloc = int(round(loc1[1]))

           CCmax = CC[rloc,cloc]
           rg00 = dftups(buf1ft*np.conj(buf1ft),1,1,usfac)/(md2*nd2*usfac**2)
           rf00 = dftups(buf2ft*np.conj(buf2ft),1,1,usfac)/(md2*nd2*usfac**2)
           rloc = rloc - dftshift
           cloc = cloc - dftshift
           row_shift = 1.*row_shift + 1.*rloc/usfac
           col_shift = 1.*col_shift + 1.*cloc/usfac

       # If upsampling = 2, no additional pixel shift refinement
       else:
           rg00 = np.sum(buf1ft*np.conj(buf1ft))/m/n;
           rf00 = np.sum(buf2ft*np.conj(buf2ft))/m/n;

       error = 1.0 - CCmax*np.conj(CCmax)/(rg00*rf00);
       error = np.sqrt(np.abs(error));
       diffphase = np.arctan2(np.imag(CCmax),np.real(CCmax));
       # If its only one row or column the shift along that dimension has no
       # effect. We set to zero.
       if md2 == 1:
          row_shift = 0

       if nd2 == 1:
          col_shift = 0;

       # Compute registered version of buf2ft
       if usfac > 0:
          ndim = np.shape(buf2ft)
          nr = ndim[0]
          nc = ndim[1]
          Nr = sf.ifftshift(np.arange(-np.fix(1.*nr/2),np.ceil(1.*nr/2)))
          Nc = sf.ifftshift(np.arange(-np.fix(1.*nc/2),np.ceil(1.*nc/2)))
          Nc,Nr = np.meshgrid(Nc,Nr)
          Greg = buf2ft*np.exp(1j*2*np.pi*(-1.*row_shift*Nr/nr-1.*col_shift*Nc/nc))
          Greg = Greg*np.exp(1j*diffphase)
       #elif (nargout > 1) and (usfac == 0):
       if usfac == 0:
          Greg = np.dot(buf2ft,np.exp(1j*diffphase))

       #plt.figure(3)
       image_reg = sf.ifft2(Greg) * np.sqrt(nr*nc)
       #imgplot = plt.imshow(np.abs(image_reg))

       #a_ini = np.zeros((100,100))
       #a_ini[40:59,40:59] = 1.
       #a = a_ini * np.exp(1j*15.)
       #plt.figure(6)
       #imgplot = plt.imshow(np.abs(a))
       #plt.figure(3)
       #imgplot = plt.imshow(np.abs(a)-np.abs(image_reg))
       #plt.colorbar()

       # return error,diffphase,row_shift,col_shift,Greg
       return error,diffphase,row_shift,col_shift, image_reg


def dftups(inp,nor,noc,usfac=1,roff=0,coff=0):
   """
       # function out=dftups(in,nor,noc,usfac,roff,coff);
       # Upsampled DFT by matrix multiplies, can compute an upsampled
DFT in just
       # a small region.
       # usfac         Upsampling factor (default usfac = 1)
       # [nor,noc]     Number of pixels in the output upsampled DFT, in
       #               units of upsampled pixels (default = size(in))
       # roff, coff    Row and column offsets, allow to shift the
output array to
       #               a region of interest on the DFT (default = 0)
       # Recieves DC in upper left corner, image center must be in (1,1)
       # Manuel Guizar - Dec 13, 2007
       # Modified from dftus, by J.R. Fienup 7/31/06

       # This code is intended to provide the same result as if the following
       # operations were performed
       #   - Embed the array "in" in an array that is usfac times larger in each
       #     dimension. ifftshift to bring the center of the image to (1,1).
       #   - Take the FFT of the larger array
       #   - Extract an [nor, noc] region of the result. Starting with the
       #     [roff+1 coff+1] element.

       # It achieves this result by computing the DFT in the output
array without
       # the need to zeropad. Much faster and memory efficient than the
       # zero-padded FFT approach if [nor noc] are much smaller than
[nr*usfac nc*usfac]
   """

   ndim = np.shape(inp)
   nr = int(round(ndim[0]))
   nc = int(round(ndim[1]))
   noc = int(round(noc))
   nor = int(round(nor))

   # Compute kernels and obtain DFT by matrix products
   a = np.zeros([nc,1])
   a[:,0] = ((sf.ifftshift(np.arange(nc)))-np.floor(1.*nc/2))[:]
   b = np.zeros([1,noc])
   b[0,:] = (np.arange(noc)-coff)[:]
   kernc = np.exp((-1j*2*np.pi/(nc*usfac))*np.dot(a,b))
   nndim = kernc.shape
   #print nndim

   a = np.zeros([nor,1])
   a[:,0] = (np.arange(nor)-roff)[:]
   b = np.zeros([1,nr])
   b[0,:] = (sf.ifftshift(np.arange(nr))-np.floor(1.*nr/2))[:]
   kernr = np.exp((-1j*2*np.pi/(nr*usfac))*np.dot(a,b))
   nndim = kernr.shape
   #print nndim

   return np.dot(np.dot(kernr,inp),kernc)



def idxmax(data):
   ndim = np.shape(data)
   #maxd = np.max(data)
   maxd = np.max(np.abs(data))
   #t1 = plt.mlab.find(np.abs(data) == maxd)
   t1 = np.argmin(np.abs((np.abs(data) - maxd)))
   idx = np.zeros([len(ndim),])
   for ii in range(len(ndim)-1):
       t1,t2 = np.modf(1.*t1/np.prod(ndim[(ii+1):]))
       idx[ii] = t2
       t1 *= np.prod(ndim[(ii+1):])
   idx[np.size(ndim)-1] = t1

   return maxd,idx


def flip_conj(tmp):
    #ndims = np.shape(tmp)
    #nx = ndims[0]
    #ny = ndims[1]
    #nz = ndims[2]
    #tmp_twin = np.zeros([nx,ny,nz]).astype(complex)
    #for i in range(0,nx):
    #   for j in range(0,ny):
    #      for k in range(0,nz):
    #         i_tmp = nx - 1 - i
    #         j_tmp = ny - 1 - j
    #         k_tmp = nz - 1 - k
    #         tmp_twin[i,j,k] = tmp[i_tmp,j_tmp,k_tmp].conj()
    #return tmp_twin

    tmp_fft = sf.ifftshift(sf.ifftn(sf.fftshift(tmp)))
    return sf.ifftshift(sf.fftn(sf.fftshift(np.conj(tmp_fft))))

def check_conj(ref, tmp,threshold_flag, threshold,subpixel_flag):
    ndims = np.shape(ref)
    nx = ndims[0]
    ny = ndims[1]
    nz = ndims[2]

    if threshold_flag == 1:
       ref_tmp = np.zeros((nx,ny,nz))
       index = np.where(np.abs(ref) >= threshold*np.max(np.abs(ref)))
       ref_tmp[index] = 1.
       tmp_tmp = np.zeros((nx,ny,nz))
       index = np.where(np.abs(tmp) >= threshold*np.max(np.abs(tmp)))
       tmp_tmp[index] = 1.
       tmp_conj = flip_conj(tmp_tmp)
    else:
       ref_tmp = ref
       tmp_tmp = tmp
       tmp_conj = flip_conj(tmp)

    tmp_tmp = subpixel_align(ref_tmp,tmp_tmp,threshold_flag,threshold,subpixel_flag)
    tmp_conj = subpixel_align(ref_tmp,tmp_conj,threshold_flag,threshold,subpixel_flag)

    cc_1 = sf.ifftn(ref_tmp*np.conj(tmp_tmp))
    cc1 = np.max(cc_1.real)
    #cc1 = np.max(np.abs(cc_1))
    cc_2 = sf.ifftn(ref_tmp*np.conj(tmp_conj))
    cc2 = np.max(cc_2.real)
    #cc2 = np.max(np.abs(cc_2))
    print('{0}, {1}'.format(cc1, cc2))
    if cc1 > cc2:
        return 0
    else:
        return 1

def subpixel_align(ref,tmp,threshold_flag,threshold, subpixel_flag):
    ndims = np.shape(ref)
    if np.size(ndims) == 3:
       nx = ndims[0]
       ny = ndims[1]
       nz = ndims[2]

       if threshold_flag == 1:
          ref_tmp = np.zeros((nx,ny,nz))
          index = np.where(np.abs(ref) >= threshold*np.max(np.abs(ref)))
          ref_tmp[index] = 1.
          tmp_tmp = np.zeros((nx,ny,nz))
          index = np.where(np.abs(tmp) >= threshold*np.max(np.abs(tmp)))
          tmp_tmp[index] = 1.
          ref_fft = sf.ifftn(sf.fftshift(ref_tmp))
          tmp_fft = sf.ifftn(sf.fftshift(tmp_tmp))
          real_fft = sf.ifftn(sf.fftshift(tmp))
       else:
          ref_fft = sf.ifftn(sf.fftshift(ref))
          tmp_fft = sf.ifftn(sf.fftshift(tmp))

       nest = np.mgrid[0:nx,0:ny,0:nz]

       result = dftregistration(ref_fft[:,:,0],tmp_fft[:,:,0],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift_1 = cl
       y_shift_1 = r
       result = dftregistration(ref_fft[:,:,nz-1],tmp_fft[:,:,nz-1],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift_2 = cl
       y_shift_2 = r

       result = dftregistration(ref_fft[:,0,:],tmp_fft[:,0,:],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift_3 = cl
       z_shift_1 = r
       result = dftregistration(ref_fft[:,ny-1,:],tmp_fft[:,ny-1,:],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift_4 = cl
       z_shift_2 = r

       result = dftregistration(ref_fft[0,:,:],tmp_fft[0,:,:],usfac=100)
       e, p, cl, r, array_shift = result
       y_shift_3 = cl
       z_shift_3 = r
       result = dftregistration(ref_fft[nx-1,:,:],tmp_fft[nx-1,:,:],usfac=100)
       e, p, cl, r, array_shift = result
       y_shift_4 = cl
       z_shift_4 = r


       if subpixel_flag == 1:
          x_shift = (x_shift_1 + x_shift_2 + x_shift_3 + x_shift_4)/4.
          y_shift = (y_shift_1 + y_shift_2 + y_shift_3 + y_shift_4)/4.
          z_shift = (z_shift_1 + z_shift_2 + z_shift_3 + z_shift_4)/4.
       else:
          x_shift = np.floor((x_shift_1 + x_shift_2 + x_shift_3 + x_shift_4)/4.+0.5)
          y_shift = np.floor((y_shift_1 + y_shift_2 + y_shift_3 + y_shift_4)/4.+0.5)
          z_shift = np.floor((z_shift_1 + z_shift_2 + z_shift_3 + z_shift_4)/4.+0.5)

       print('x, y, z shift: {0}, {1}, {2}'.format(x_shift, y_shift, z_shift))

       if threshold_flag == 1:
          tmp_fft_new = sf.ifftshift(real_fft) * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:,:]-ny/2.)/(ny)-z_shift*(nest[2,:,:,:]-nz/2.)/(nz)))
       else:
          tmp_fft_new = sf.ifftshift(tmp_fft) * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:,:]-ny/2.)/(ny)-z_shift*(nest[2,:,:,:]-nz/2.)/(nz)))

    if np.size(ndims) == 2:
       nx = ndims[0]
       ny = ndims[1]

       if threshold_flag == 1:
          ref_tmp = np.zeros((nx,ny))
          index = np.where(np.abs(ref) >= threshold*np.max(np.abs(ref)))
          ref_tmp[index] = 1.
          tmp_tmp = np.zeros((nx,ny))
          index = np.where(np.abs(tmp) >= threshold*np.max(np.abs(tmp)))
          tmp_tmp[index] = 1.

          ref_fft = sf.ifftn(sf.fftshift(ref_tmp))
          mp_fft = sf.ifftn(sf.fftshift(tmp_tmp))
          real_fft = sf.ifftn(sf.fftshift(tmp))
       else:
          ref_fft = sf.ifftn(sf.fftshift(ref))
          tmp_fft = sf.ifftn(sf.fftshift(tmp))

       nest = np.mgrid[0:nx,0:ny]

       result = dftregistration(ref_fft[:,:],tmp_fft[:,:],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift = cl
       y_shift = r

       if subpixel_flag == 1:
          x_shift = x_shift
          y_shift = y_shift
       else:
          x_shift = np.floor(x_shift + 0.5)
          y_shift = np.floor(y_shift + 0.5)

       print ('x, y shift: {0}, {1}'.format(x_shift, y_shift))

       if threshold_flag == 1:
          tmp_fft_new = sf.ifftshift(real_fft) * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:]-ny/2.)/(ny)))
       else:
          tmp_fft_new = sf.ifftshift(tmp_fft) * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:]-ny/2.)/(ny)))

    return sf.ifftshift(sf.fftn(sf.fftshift(tmp_fft_new))),x_shift,y_shift


def remove_phase_ramp(tmp,threshold_flag, threshold,subpixel_flag):
   tmp_tmp,x_shift,y_shift = subpixel_align(sf.ifftshift(sf.ifftn(sf.fftshift(np.abs(tmp)))), sf.ifftshift(sf.ifftn(sf.fftshift(tmp))), threshold_flag, threshold,subpixel_flag)
   tmp_new = sf.ifftshift(sf.fftn(sf.fftshift(tmp_tmp)))
   phase_tmp = np.angle(tmp_new)
   ph_offset = np.mean(phase_tmp[np.where(np.abs(tmp) >= threshold)])
   phase_tmp = np.angle(tmp_new) - ph_offset
   return np.abs(tmp)*np.exp(1j*phase_tmp)

def pixel_shift(array,x_shift,y_shift,z_shift):
    nx,ny,nz = np.shape(array)
    tmp = sf.ifftshift(sf.ifftn(sf.fftshift(array)))
    nest = np.mgrid[0:nx,0:ny,0:nz]
    tmp = tmp * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:,:]-ny/2.)/(ny)-z_shift*(nest[2,:,:,:]-nz/2.)/(nz)))
    return sf.ifftshift(sf.fftn(sf.fftshift(tmp)))

def pixel_shift_2d(array,x_shift,y_shift):
    nx,ny = np.shape(array)
    tmp = sf.ifftshift(sf.ifftn(sf.fftshift(array)))
    nest = np.mgrid[0:nx,0:ny]
    tmp = tmp * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:]-ny/2.)/(ny)))
    return sf.ifftshift(sf.fftn(sf.fftshift(tmp)))

def rm_phase_ramp_manual_2d(array,x_shift,y_shift):
    nx,ny = np.shape(array)
    nest = np.mgrid[0:nx,0:ny]
    tmp = array * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:]-ny/2.)/(ny)))
    return tmp


def nurbs_surf_spline_fit(ctrlpts, delta=0.025, degree_u=3, degree_v=3, plot_flag=0):
    from geomdl import BSpline, utilities
    from geomdl.visualization import VisMPL
    from matplotlib import cm

    # Create a BSpline surface
    surf = BSpline.Surface()

    # Set degrees
    surf.degree_u = degree_u
    surf.degree_v = degree_v

    # Set control points
    surf.ctrlpts2d = ctrlpts

    # Set knot vectors
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)

    # Set evaluation delta
    surf.delta = delta

    # Evaluate surface points
    surf.evaluate()

    # Import and use Matplotlib's colormaps
    if plot_flag:
        # Plot the control points grid and the evaluated surface
        surf.vis = VisMPL.VisSurface()
        surf.render(colormap=cm.cool)

    evalpts = surf.evalpts
    return evalpts



def nurbs_surf_gen_ctrlpts(img, num_u, num_v):
    s = img.shape # e.g., (256, 256)
    cp = []
    y = np.linspace(0, s[0], num_v, endpoint=False)
    for i in range(num_v):
        cp_v = []
        idx_y = int(y[i])
        line_x = img[idx_y]
        idx_non_zero = np.squeeze(np.argwhere(line_x>0))
        line_x_non_zero = line_x[idx_non_zero]
        sx = len(line_x_non_zero)
        idx_x = np.int32(np.linspace(0, sx, num_u, endpoint=False))
        for j in range(num_u):
            idx_xx = idx_non_zero[idx_x[j]]
            cp_v.append([idx_y, idx_xx, img[idx_y, idx_xx]])
        cp.append(cp_v)
    return cp



def spline_img_filter(img, num_u=40, num_v=40, degree_u=3, degree_v=3, clim=None, plot_flag=0):
    ctrlpts = nurbs_surf_gen_ctrlpts(img, num_u, num_v)
    s = img.shape
    delta = 1 / np.max(s)
    evalpts = nurbs_surf_spline_fit(ctrlpts, delta, degree_u, degree_v, plot_flag=0)
    s1 = int(len(evalpts)**(0.5))
    max_size = int(np.max(s))
    img_s = np.array(evalpts).reshape([max_size, max_size, 3])[:,:,-1]
    img_resize = resize(img_s, (s[0], s[1]))
    if clim is None:
        clim = [np.min(img), np.max(img)]
    if plot_flag:
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.imshow(img, clim=clim)
        plt.subplot(122)
        plt.imshow(img_resize, clim=clim)
    return img_resize


def spline_img_filter_stack(img, num_u=40, num_v=40, degree_u=3, degree_v=3):
    s = img.shape
    img_r = np.zeros(s)
    for i in trange(s[0]):
        img_r[i] = spline_img_filter(img[i], num_u, num_v, degree_u, degree_v, plot_flag=0)
    return img_r
    

    
if (__name__ == '__main__'):
    pass
