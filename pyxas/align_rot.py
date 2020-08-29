import numpy as np
import matplotlib.pyplot as plt
import pyxas
from skimage import data
from skimage.feature import register_translation
from skimage.transform import warp_polar, rotate
from skimage.util import img_as_float
from scipy.ndimage import geometric_transform
'''
radius = 705
angle = 35
image = data.retina()
image = img_as_float(image)
rotated = rotate(image, angle)
image_polar = warp_polar(image, radius=radius, multichannel=True)
rotated_polar = warp_polar(rotated, radius=radius, multichannel=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].set_title("Original")
ax[0].imshow(image)
ax[1].set_title("Rotated")
ax[1].imshow(rotated)
ax[2].set_title("Polar-Transformed Original")
ax[2].imshow(image_polar)
ax[3].set_title("Polar-Transformed Rotated")
ax[3].imshow(rotated_polar)
plt.show()

shifts, error, phasediff = register_translation(image_polar, rotated_polar)
print("Expected value for counterclockwise rotation in degrees: "
      f"{angle}")
print("Recovered value for counterclockwise rotation: "
      f"{shifts[0]}")
'''
'''
from scipy.ndimage import shift
#radius = 735;img1 = image[:,:,0]
img1 = img2D;radius = min(img2D.shape)/2
#img1 = np.zeros([400,400])
#img1[100:300, 100:300] = img2D
#radius = 200
img2 = shift(img1, [5, 10])
img1_r = rotate(img1, 35)
img2_r = rotate(img2, 35)
shifts, error, phasediff = register_translation(img1, img2, upsample_factor=2)


img1_fft = np.fft.fftshift(np.fft.fft2(img1))
img1_r_fft = np.fft.fftshift(np.fft.fft2(img1_r))
img2_r_fft = np.fft.fftshift(np.fft.fft2(img2_r))

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].imshow(img1)
ax[1].imshow(img1_r)
ax[2].imshow(np.log(np.abs(img1_fft)))
ax[3].imshow(np.log(np.abs(img1_r_fft)))

img1_fft_polar = warp_polar(np.log(np.abs(img1_fft)), radius=radius, order=3)
img1_r_fft_polar = warp_polar(np.log(np.abs(img1_r_fft)), radius=radius, order=3)
img2_r_fft_polar = warp_polar(np.log(np.abs(img2_r_fft)), radius=radius)

shifts_fft, error, phasediff = register_translation(img1_fft_polar, img2_r_fft_polar, upsample_factor=2)
print(shifts_fft)
'''
#######################3


from numpy import sin, cos
from skimage.transform import warp
from mayavi.mlab import *
#img3D = tomopy.shepp3d(200)
from scipy import ndimage



def linear_polar_mapping_3D(polar_img_shape, center):
    '''
    output_coords = [r, theta, phi], lower-left is 0
    '''
    phi, theta, rad = np.mgrid[:polar_img_shape[0], :polar_img_shape[1], :polar_img_shape[2]]

    cc = rad * np.cos(phi) * np.cos(theta) + center[1]
    rr = rad * np.cos(phi) * np.sin(theta) + center[0]
    hh = rad * np.sin(phi) + center[2]
    coords = np.array([hh, cc, rr])
    return coords

def to_polar_3D(img3D, center=None, radius=None, output_shape=None,**kwargs):
    if center is None:    
        center = (np.array(img3D.shape)[:3] / 2) - 0.5
    if radius is None:
        h, r, c = np.array(img3D.shape)[:3] / 2
        radius = np.sqrt(h ** 2 + r ** 2 + c ** 2)
    if output_shape is None:
        height = 360 # phi
        col = int(np.ceil(radius)) # radius
        row = 360 # theta
        output_shape = (height, row, col)
    else:
        output_shape = safe_as_int(output_shape)
        height = output_shape[0]
        row = output_shape[1]
        col = output_shape[2]

    warp_args = {'center': center}
    warped = warp(img3D, linear_polar_mapping_3D(output_shape, center))
    return warped


def linear_polar_mapping_2D(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesion to polar coordinates

    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = ncols / radius``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.

    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords

def to_polar_2D(img, center=None, radius=None, output_shape=None,**kwargs):
    if center is None:    
        center = (np.array(img.shape)[:2] / 2) - 0.5
    if radius is None:
        w, h = np.array(img.shape)[:2] / 2
        radius = np.sqrt(w ** 2 + h ** 2)
    if output_shape is None:
        height = 360
        width = int(np.ceil(radius))
        output_shape = (height, width)
    else:
        output_shape = safe_as_int(output_shape)
        height = output_shape[0]
        width = output_shape[1]
    k_radius = width / radius
    k_angle = height / (2 * np.pi)
    warp_args = {'k_angle': k_angle, 'k_radius': k_radius, 'center': center}
    warped = warp(img, linear_polar_mapping_2D, map_args=warp_args,
                  output_shape=output_shape)
    return warped


def linear_cart_mapping_2D(coords, center):
    '''
    coords: [col, row]
    '''
    x = coords[:, 0] - center[0]
    y = coords[:, 1] - center[1]
    angle = np.arctan2(y, x) / np.pi * 180
    angle[angle<0] = angle[angle<0] + 360

    radius = np.sqrt(x**2 + y**2)
    coords = np.column_stack((radius, angle))     
    return coords


def to_linear_2D(img_p, center_offset=None, output_shape=None, **kwargs):
    s = img_p.shape
    if output_shape is None:
        w = floor(s[1]/np.sqrt(2))
        output_shape = (2*w, 2*w)  
    else:
        w = min(output_shape)/2  
    if center_offset is None:
        center = (w-0.5, w-0.5)
    else:
        center = (w-0.5+center_offset[0], w-0.5+center_offset[1])    
    warp_args = {'center':center}
    warped = warp(img_p, linear_cart_mapping_2D, map_args=warp_args,         output_shape=output_shape)
    return warped


def topolar(img, order=5):
    max_radius = 0.5*np.linalg.norm( img.shape )
    def transform(coords):
        theta = 2.0*np.pi*coords[0] / (img.shape[0])
        radius = max_radius * coords[1] / img.shape[1]
        i = 0.5*img.shape[1] - radius*np.sin(theta)
        j = radius*np.cos(theta) + 0.5*img.shape[0]
        return i,j
    polar = geometric_transform(img, transform, order=order,mode='nearest',prefilter=True)
    return polar


def tocart(img, order=5):
    max_radius = 0.5*np.linalg.norm( img.shape )
    def transform(coords):
        
        r = coords[1]
        theta = coords[0] / 360 * 2 * np.pi

        x = r * np.cos(theta) + 100
        y = r * np.sin(theta) + 100

        return (y, x)
    cart = geometric_transform(img, transform, order=order,mode='nearest',prefilter=True)
    return cart



def test_fun(r, theta,):
    x = r *  np.cos(theta)
    y = r *  np.sin(theta)
    #z = r * np.sin(phi)
    #return x, y, z
    return x, y



############## scipy ndimage.map_coordinates #############
from numpy import sin, cos
from skimage.transform import warp
from mayavi.mlab import *


#img3D = tomopy.shepp3d(200)
#img2D = img3D[100]
#img = np.repeat(img2D[np.newaxis, :,:], 200, axis=0)


def rotate_2D(img2D, theta, center=None):

    from numpy import sin, cos
    s = img2D.shape
    if center is None:
        center = np.array(s) / 2 - 0.5
    theta_r = -theta / 180 * np.pi
    m1 = [[cos(theta_r), -sin(theta_r)],
          [sin(theta_r),  cos(theta_r)],]    
    m1 = np.array(m1)
    y, x = np.mgrid[:s[0], :s[1]] # y has "0" at up-left corner
    ymax = np.max(y)
    y = ymax - y # change y =0 to low-left cornor
    x, y = x.flatten(), y.flatten()
    t = np.vstack((x-center[1], y-center[0]))
    r = m1 @ np.array(t)
    r[0] += center[1] # x --> column
    r[1] += center[0] # y --> row
    r[1] = ymax - r[1] # change back y = 0 to up-left corner
    coords = r[::-1]
    img2D_r = ndimage.map_coordinates(img2D, coords, order=1)
    img2D_r = img2D_r.reshape(img2D.shape)
    return img2D_r


def transform_3D(img3D, m, center=None, order=1):
    '''
    m is the transform matrix [3x3]
    img_transformed = m*img3D
    '''
    m_t = np.linalg.inv(m)
    s = img3D.shape
    if center is None:
        center = np.array(s) / 2 - 0.5
    z, y, x = np.mgrid[:s[0], :s[1], :s[2]]
    ymax, zmax = np.max(y), np.max(z)
    # y = ymax - y
    # z = zmax - z
    y = y[:,::-1]
    z = z[::-1]
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    t = np.vstack((x-center[2], y-center[1], z-center[0]))
    r = m_t @ np.array(t)
    r[0] += center[2] # x --> column
    r[1] += center[1] # y --> row
    r[2] += center[0] # z --> height
    r[1] = ymax - r[1]
    r[2] = zmax - r[2]
    coords = r[::-1]
    img3D_t = ndimage.map_coordinates(img3D, coords, order=order)
    img3D_t = img3D_t.reshape(img3D.shape)
    return img3D_t

    

def get_rotation_matrix(theta_x=0, theta_y=0, theta_z=0):
    t_x = theta_x / 180 * np.pi
    t_y = theta_y / 180 * np.pi
    t_z = theta_z / 180 * np.pi
    m_z = [[cos(t_z), -sin(t_z), 0],
          [sin(t_z),  cos(t_z), 0],
          [0, 0, 1]]    
    m_y = [[cos(t_y), 0, -sin(t_y)],
          [0, 1, 0],
          [sin(t_y), 0, cos(t_y)]]
    m_x = [[1, 0, 0],
          [0, cos(t_x), -sin(t_x)],
          [0, sin(t_x), cos(t_x)]]
    m = np.array(m_x) @ np.array(m_y) @ np.array(m_z)
    return m

def get_rotation_matrix1(theta_x=0, theta_y=0, theta_z=0):
    t_x = theta_x / 180 * np.pi
    t_y = theta_y / 180 * np.pi
    t_z = theta_z / 180 * np.pi
    m_z = [[cos(t_z), -sin(t_z), 0],
          [sin(t_z),  cos(t_z), 0],
          [0, 0, 1]]    
    m_y = [[cos(t_y), 0, -sin(t_y)],
          [0, 1, 0],
          [sin(t_y), 0, cos(t_y)]]
    m_x = [[1, 0, 0],
          [0, cos(t_x), -sin(t_x)],
          [0, sin(t_x), cos(t_x)]]
    m = np.array(m_z) @ np.array(m_y) @ np.array(m_x)
    return m


#mm = get_rotation_matrix1(theta_x, theta_y, theta_z)
#mm_inv = np.linalg.inv(mm)


def rotate_3D_new(img3D, theta_x=0, theta_y=0, theta_z=0, center=None):
    '''
    first rotate along x, then rotate along y, then rotate along z
    '''
    '''
    from numpy import sin, cos
    s = img3D.shape
    if center is None:
        center = np.array(s) / 2 - 0.5
    '''
    mm = get_rotation_matrix1(theta_x, theta_y, theta_z)
    img3D_r = transform_3D(img3D, mm, center)   
    '''
    m3 = get_rotation_matrix(0, 0, -theta_z)
    m2 = get_rotation_matrix(0, -theta_y, 0)
    m1 = get_rotation_matrix(-theta_x, 0, 0)
    m = m1 @ m2 @ m3
    '''
    '''
    m = mm.T
    z, y, x = np.mgrid[:s[0], :s[1], :s[2]]
    ymax, zmax = np.max(y), np.max(z)
    # y = ymax - y
    # z = zmax - z
    y = y[:,::-1]
    z = z[::-1]
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    t = np.vstack((x-center[2], y-center[1], z-center[0]))
    r = m @ np.array(t)
    r[0] += center[2] # x --> column
    r[1] += center[1] # y --> row
    r[2] += center[0] # z --> height
    r[1] = ymax - r[1]
    r[2] = zmax - r[2]
    coords = r[::-1]
    img3D_r = ndimage.map_coordinates(img3D, coords, order=1)
    img3D_r = img3D_r.reshape(img3D.shape)
    '''
    return img3D_r

def to_spherical(output_shape, center, radius_scale=1):
    s = output_shape
    phi, the, rad = np.mgrid[:s[0], :s[1], :s[2]]
    phi = phi / s[0] * 2 * np.pi
    the = the / s[1] * 2 * np.pi
    rad = rad / radius_scale
    h = rad * np.sin(phi) + center[0]
    r = rad * np.cos(phi) * np.sin(the) + center[1]
    c = rad * np.cos(phi) * np.cos(the) + center[2]
    h, r, c = h.reshape(-1), r.reshape(-1), c.reshape(-1)
    coords = np.vstack((h, r, c))
    return coords

def img_to_spherical(img3D, radius=None, radius_scale=1, phi_scale=1, theta_scale=1):
    '''
    return the image in spherical coordiante
    note that:
    img3D has a shape of (height, row, column)
    we define the theta as the in-plane angle, in which theta =0 means the points lie on the x-axis, equavelent to the column of the 3D image

    phi is defined as the angle away from the x-y plane. It is different from conventional definition. Here, phi=0 means points lie on the x-y plane, equavelent to row-column plane. 
    '''
    s = img3D.shape
    center = np.array(s) / 2 - 0.5
    #rad_size = int(np.sqrt(s[0]**2 + s[1]**2 + s[2]**2) / 2 * radius_scale)
    if radius is None:
        rad_size = int(min(s)/2)
    else:
        rad_size = radius
    theta_size = int(360 * theta_scale)
    phi_size = int(360 * phi_scale)
    output_shape = (phi_size, theta_size, rad_size)
    coords = to_spherical(output_shape, center, radius_scale)
    img_s = ndimage.map_coordinates(img3D, coords, order=1)
    img_s = img_s.reshape((phi_size, theta_size, rad_size))
    return img_s


def to_polar(output_shape, center, radius_scale=1):
    s = output_shape
    the, rad = np.mgrid[:s[0], :s[1]]
    the = the / s[0] * 2 * np.pi
    rad = rad / radius_scale
    r = rad * np.sin(the) + center[0]
    c = rad * np.cos(the) + center[1]
    r, c = r.reshape(-1), c.reshape(-1)
    coords = np.vstack((r, c))
    return coords

def img_to_polar(img2D, radius=None, radius_scale=1):
    s = img2D.shape
    center = np.array(s) / 2 - 0.5
    #rad_size = int(np.sqrt(s[0]**2 + s[1]**2 + s[2]**2) / 2 * radius_scale)
    if radius is None:
        rad_size = int(min(s)/2)
    else:
        rad_size = int(radius)
    theta_size = int(360)
    output_shape = (theta_size, rad_size)
    coords = to_polar(output_shape, center, radius_scale)
    img_s = ndimage.map_coordinates(img2D, coords, order=1)
    img_s = img_s.reshape((theta_size, rad_size))
    return img_s



def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
       # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

      # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
               for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        noisy = image + image * gauss
        return noisy


def find_rotation_angle_2D(img1, img2):
    radius = min(img1.shape)/2
    img1_fft = np.fft.fftshift(np.fft.fft2(img1))
    img2_fft = np.fft.fftshift(np.fft.fft2(img2))

    img1_fft_polar = img_to_polar(np.log(np.abs(img1_fft)), radius=radius)
    img2_fft_polar = img_to_polar(np.log(np.abs(img2_fft)), radius=radius)

    shifts_fft, error, phasediff = register_translation(img1_fft_polar, img2_fft_polar, upsample_factor=2)
    print(shifts_fft)
    return shifts_fft


def find_rotation_angle_3D(img3D, img3D_r, use_fft=False):
    if use_fft:    
        img1 = np.log(np.abs(np.fft.fftshift(np.fft.fftn(img3D))))
        img2 = np.log(np.abs(np.fft.fftshift(np.fft.fftn(img3D_r))))
    else:
        img1 = img3D
        img2 = img3D_r
    img1_s = img_to_spherical(img1, radius=None, radius_scale=1, phi_scale=1, theta_scale=1)
    img2_s = img_to_spherical(img2, radius=None, radius_scale=1, phi_scale=1, theta_scale=1)
    shifts_fft, error, phasediff = register_translation(img1_s, img2_s, upsample_factor=2)
    print(shifts_fft)
    return shifts_fft

#################################
########## PCA based ############
#################################
'''
# for 2D
y, x = (np.mgrid[:200, :200] -0)
x_mean, y_mean = np.mean(x), np.mean(y)
a, b = 10, 1
f = np.zeros([200, 200])
f[(((x-x_mean)/a)**2 + ((y-y_mean)/b)**2) <= 10**2] = 1
f = rotate_2D(f, 30)



M00 = np.sum(np.sum(f))
M10 = np.sum(x * f)
M01 = np.sum(y * f)
M20 = np.sum(x * x * f)
M02 = np.sum(y * y * f)

u00 = M00
u20 = M20 - x_mean * M10
u11 = M11 - x_mean * M01
u02 = M02 - y_mean * M01

u20_ = u20 / u00
u02_ = u02 / u00
u11_ = u11 / u00

cov2D = np.array([[u20_, u11_], 
                  [u11_, u02_]])

w, v = np.linalg.eig(cov2D)

2*u11_/(u20_-u02_)

f = scipy.ndimage.shift(f, [50,0])
x1 = x[f>0.5]
y1 = y[f>0.5]
f1 = f[f>0.5]
coords = np.vstack([x1,y1])
cov = np.cov(coords, f1)
evals, evecs = np.linalg.eig(cov)
sort_indices = np.argsort(evals)[::-1]
x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
x_v2, y_v2 = evecs[:, sort_indices[1]]
np.arctan2(y_v1, x_v1)/np.pi * 180
'''

def find_img_angle(img, thresh=None):
    s = img.shape
    y, x = np.mgrid[:s[0], :s[1]]
    y = np.flipud(y)
    x1, y1 = x.flatten(), y.flatten()
    coords = np.vstack([x1, y1])
    if not thresh is None:
        img[img<thresh] = 0
    aweights = img.flatten()    
    cov = np.cov(coords, aweights=aweights)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]
    vals = np.sort(evals)[::-1]
    vecs = np.zeros(evecs.shape)
    vecs[:, 0] = evecs[:, sort_indices[0]]
    vecs[:, 1] = evecs[:, sort_indices[1]]
    ang = np.arctan2(y_v1, x_v1)/np.pi * 180
    return vals, vecs, ang  

        
def align_2D_rotation(img_ref, img2, n=10, prec=0.1, angle_limit=None):
    '''
    n: num_iter
    prec: angle difference limit
    '''
    im1 = img_ref.copy()
    im2 = img2.copy()    
    rot_angle = 0
    for i in range(10):
        im1[im1<0] = 0
        im2[im2<0] = 0
        
        im2_, _, _ = pyxas.align_img(im1, im2)

        _, _, ang1 = pyxas.find_img_angle(im1)
        _, _, ang2 = pyxas.find_img_angle(im2)
        d_ang = np.abs(ang1 - ang2)
                
        img_ali1 = pyxas.rotate_2D(im2, d_ang)
        img_ali1, _, _ = pyxas.align_img(im1, img_ali1)
        sum1 = np.sum(im1 * img_ali1)

        img_ali2 = pyxas.rotate_2D(im2, -d_ang)
        img_ali2, _, _ = pyxas.align_img(im2, img_ali2)
        sum2 = np.sum(im1 * img_ali2)
        
        if sum1 > sum2:
            rot_angle += d_ang
            img_ali = img_ali1
        else:
            rot_angle += -d_ang
            img_ali = img_ali2
        if d_ang < 0.1:          
            break
        else:
            im2 = img_ali
        if not angle_limit is None:
            if np.abs(rot_angle) > angle_limit:
                img_ali = img2
                print('exceed limit, skip the alignment')
                break
    print(f'iter={i+1}  rotation angle = {rot_angle:2.1f}')
    return img_ali, rot_angle


def find_img_angle3D(img, thresh=None, sort=False):
    s = img.shape
    z, y, x = np.mgrid[:s[0], :s[1], :s[2]]
    y = y[:,::-1]
    z = z[::-1]
    x1, y1, z1 = x.flatten(), y.flatten(), z.flatten()
    coords = np.vstack([x1, y1, z1])
    if not thresh is None:
        img[img<thresh] = 0
    aweights = img.flatten() 
    aweights[aweights<0] = 0   
    cov = np.cov(coords, aweights=aweights)
    evals, evecs = np.linalg.eig(cov)
    if sort:
        sort_indices = np.argsort(evals)[::-1]
        vals = np.sort(evals)[::-1]
        vecs = np.zeros(evecs.shape)
        for i in range(3):
            vecs[:, i] = evecs[:, sort_indices[i]]
        return vals, vecs 
    else:
        return evals, evecs

def angle_along_axis(vec):
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    r_vec = np.linalg.norm(vec)
    ax = np.dot(vec, e1) / r_vec
    ay = np.dot(vec, e2) / r_vec
    az = np.dot(vec, e3) / r_vec
    theta_x = np.arccos(ax) * 180 / np.pi
    theta_y = np.arccos(ay) * 180 / np.pi
    theta_z = np.arccos(az) * 180 / np.pi
    return [theta_x, theta_y, theta_z]

def angle_between_vec(vec1, vec2):
    r1 = np.linalg.norm(vec1)
    r2 = np.linalg.norm(vec2)
    a =  np.dot(vec1, vec2) / (r1 * r2)
    theta = np.arccos(a) * 180 / np.pi
    return theta
    

def plot_line(p1, p2, ax=None, c='g'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    x = [p1[0], p2[0]]; y = [p1[1], p2[1]]; z = [p1[2], p2[2]]
    ax.plot(x, y, z, c)

def plot_box(ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    p1 = [-1,-1, 0]; p2 = [1,-1,0]; p3=[-1,1,0]; p4 = [1,1,0]
    p5 = [-1,-1, 1]; p6 = [1,-1,1]; p7=[-1,1,1]; p8 = [1,1,1]

    plot_line(p1, p2, ax); plot_line(p1, p3, ax)
    plot_line(p2, p4, ax); plot_line(p3, p4, ax)
    plot_line(p5, p6, ax); plot_line(p5, p7, ax) 
    plot_line(p6, p8, ax); plot_line(p7, p8, ax)
    plot_line(p1, p5, ax); plot_line(p2, p6, ax) 
    plot_line(p3, p7, ax); plot_line(p4, p8, ax)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

"""
def test3D():
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt
        z, y, x = (np.mgrid[:200, :200, :200])
        x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)
        a, b, c = 10, 4, 1
        img3D_new = np.zeros([200, 200, 200])
        img3D_new[(((x-x_mean)/a)**2 + ((y-y_mean)/b)**2 + ((z-z_mean)/c)**2) <= 10**2] = 1
        img3D_new[:100] = 0; img3D_new[:,:100] = 0; img3D_new[:,:,:100]=0



        img3D_old = tomopy.shepp3d(200)

        sort_flag = True
        img3D = img3D_new.copy()
        img2D = img3D[100]
        theta_x, theta_y, theta_z = 20, 60, 30
        img3D_r = rotate_3D_new(img3D, theta_x, theta_y, theta_z)
        img3D_r = scipy.ndimage.shift(img3D_r, [20,40, 0], order=1)

        img3D = move_3D_to_center(img3D, 1)
        img3D_r = move_3D_to_center(img3D_r, 1)

        img3D = noisy('poisson', img3D)
        img3D_r = noisy('poisson', img3D_r)

        m1 = get_rotation_matrix(theta_x, 0, 0)
        m2 = get_rotation_matrix(0, theta_y, 0)
        m3 = get_rotation_matrix(0, 0, theta_z)
        m = m3 @ m2 @ m1

        m_rot = get_rotation_matrix1(theta_x, theta_y, theta_z)

        vec_rot = m_rot @ vec
        '''
            for i in range(3):
            x = [0, vec_rot[0, i]]
            y = [0, vec_rot[1, i]]
            z = [0, vec_rot[2, i]]
            ax.scatter(x, y, z, c='c', marker='>')
            ax.plot(x, y, z,'c') 
            ax.text(x[1]+0.1, y[1]+0.1, z[1]+0.1, 'rot_'+str(chr(97+i))) 
        '''
    except:
        pass
"""


def align_3D_rotation_not_good(img_ref, img2, plot_flag=0):

    val, vec = find_img_angle3D(img_ref, sort=1)
    val_r, vec_r = find_img_angle3D(img2, sort=1)
            
    if plot_flag:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plot_box(ax)

        for i in range(3):
            x = [0, vec[0, i]]
            y = [0, vec[1, i]]
            z = [0, vec[2, i]]
            ax.scatter(x, y, z, c='r', marker='o')
            ax.plot(x, y, z,'r')
            ax.text(x[1], y[1], z[1], 'v1_'+str(chr(97+i)))
             
        for i in range(3):
            x = [0, vec_r[0, i]]
            y = [0, vec_r[1, i]]
            z = [0, vec_r[2, i]]
            ax.scatter(x, y, z, c='b', marker='+')
            ax.plot(x, y, z,'b') 
            ax.text(x[1]+0.05, y[1]+0.05, z[1]+0.05, 'v2_'+str(chr(97+i)))  

    v1, v2, v3 = vec_r[:,0], vec_r[:,1], vec_r[:,2]
    if sign(np.linalg.det(vec) * np.linalg.det(vec_r)) < 0:
        v1 = -v1
    mm = {}
    mm['0'] = np.array([v1, v2, v3]).T
    mm['1'] = np.array([v2, v3, v1]).T
    mm['2'] = np.array([v3, v1, v2]).T

    mm['3'] = np.array([-v1, v3, v2]).T
    mm['4'] = np.array([v3, v2, -v1]).T
    mm['5'] = np.array([v2, -v1, v3]).T
    mm['6'] = np.array([v3, -v2, v1]).T
    mm['7'] = np.array([-v2, v1, v3]).T
    mm['8'] = np.array([v1, v3, -v2]).T
    mm['9'] = np.array([v2, v1, -v3]).T
    mm['10'] = np.array([v1, -v3, v2]).T
    mm['11'] = np.array([-v3, v2, v1]).T

    mm['12'] = np.array([-v1, -v2, v3]).T
    mm['13'] = np.array([-v2, v3, -v1]).T
    mm['14'] = np.array([v3, -v1, -v2]).T
    mm['15'] = np.array([-v1, v2, -v3]).T
    mm['16'] = np.array([v2, -v3, -v1]).T
    mm['17'] = np.array([-v3, -v1, v2]).T

    mm['18'] = np.array([v1, -v2, -v3]).T
    mm['19'] = np.array([-v2, -v3, v1]).T
    mm['20'] = np.array([-v3, v1, -v2]).T

    mm['21'] = np.array([-v1, -v3, -v2]).T
    mm['22'] = np.array([-v3, -v2, -v1]).T
    mm['23'] = np.array([-v2, -v1, -v3]).T

    msum = np.zeros(len(mm))
    m_inv = np.linalg.inv(vec)
    
    for i in range(len(mm)):
        m_tmp = mm[f'{i}'] @ m_inv
        img_tmp = pyxas.transform_3D(img_ref, m_tmp, order=2)
        img_tmp, h, r, c = pyxas.align_img3D(img2, img_tmp)
        tsum = np.sum(img2 * img_tmp)
        msum[i] = tsum
        print(f'{i}: {tsum:6.2f}, h={h:3.1f}, r={r:3.1f}, c={c:3.1f}')

    idx = np.argmax(msum)
    m_tmp = mm[f'{int(idx)}'] @ m_inv
    m_t = np.linalg.inv(m_tmp)
    img3D_recover = pyxas.transform_3D(img2, m_t)
    img3D_recover, _, _, _ = pyxas.align_img3D(img_ref, img3D_recover)


######

def test3D_2():
    from scipy.ndimage.filters import gaussian_filter as gf

    fn = '/media/mingyuan/Seagate Backup Plus Drive/TXM_2019/Ruoqian/xanes_3D/recon_crop_39300.h5'
    f = h5py.File(fn,'r')
    img = np.array(f['img'])
    img_blur = gf(img, 5)
    img_blur[img_blur<0] = 0
    img_blur *= 1e5
    markers = np.zeros(img_blur.shape, dtype=np.uint)
    markers[img_blur<50] = 1
    markers[img_blur>50] = 2
    labels = random_walker(img, markers, beta=10, mode='cg_mg')
    ##########
    from scipy import ndimage as ndi
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max

    img_bw = np.zeros(img_blur.shape, dtype=np.uint)
    img_bw[img_blur>100] = 1
    distance = ndi.distance_transform_edt(img_bw)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3, 3)),labels=img_bw)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=img_bw, compactness=10)


    image_max = ndi.maximum_filter(distance, size=20, mode='constant')
    coordinates = peak_local_max(distance, min_distance=80)


    ##### kmean ##########

    from sklearn.cluster import KMeans
    s = img_bw.shape
    z, y, x = np.mgrid[:s[0], :s[1], :s[2]]
    xx = x[img_bw>0.5]
    yy = y[img_bw>0.5]
    zz = z[img_bw>0.5]
    coords = np.vstack((xx[::1000],yy[::1000],zz[::1000]))
    kmeans = KMeans(n_clusters = 3, random_state=0).fit(coords.T)
    img_values = kmeans.cluster_centers_
    img_labels = kmeans.labels_

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(xx[::10000], yy[::10000], zz[::10000])
    ax.set_xlabel('X axis') 
    ax.set_ylabel('Y axis') 
    ax.set_zlabel('Z axis')

    for i in range(len(img_values)):
        px, py, pz = img_values[i]
        ax.scatter3D(px, py, pz, 'ro')
        ax.text(px, py, pz, f'cen: #{i}')

    ################################

    img_bin = pyxas.bin_image(img, 1)
    img_bin = gf(img_bin, 5)
    pix = img_bin.flatten() * 1e5
    ts = time.time()
    kmeans = KMeans(n_clusters = 2, random_state=0, n_jobs=4).fit(pix.reshape(-1,1))
    te = time.time()
    print(f'taking: {te-ts:3.1f} sec')
    img_labels = kmeans.labels_
    img_labels.shape = img_bin.shape


##################
def rm_image_bkg(img3D, multiply_factor=1, n_jobs=4):
    from sklearn.cluster import KMeans
    from scipy.ndimage.filters import gaussian_filter as gf
    import time

    #img = gf(img3D * multiply_factor, 5)   
    img = img3D *multiply_factor
    pix = img.flatten()
    ts = time.time()
    kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=n_jobs).fit(pix.reshape(-1,1))
    te = time.time()
    print(f'taking: {te-ts:3.1f} sec')
    img_labels = kmeans.labels_
    img_labels.shape = img.shape
    img_values = kmeans.cluster_centers_ / multiply_factor
    if np.argmax(img_values) == 0:
        img_labels = 1 - img_labels
    res = {}
    res['img_labels'] = img_labels
    res['img'] = img3D * img_labels
    return res

def extract_particles_not_work(img_bw, n_comp=3):
    from sklearn.cluster import KMeans
    s = img_bw.shape
    z, y, x = np.mgrid[:s[0], :s[1], :s[2]]
    idx = img_bw > 0.5
    idx_flat = idx.flatten()
    xx = x[idx]
    yy = y[idx]
    zz = z[idx]
    coords = np.vstack((xx,yy,zz))
    kmeans = KMeans(n_clusters = n_comp, random_state=0).fit(coords.T)
    img_labels = kmeans.labels_
    img_values = kmeans.cluster_centers_ 

    particle = {}
    img_p = np.zeros(s).flatten()
    for i in range(n_comp):
        id1 = (np.abs(img_labels-i)<0.1)
        tmp = img_p[idx_flat].copy()
        tmp[id1] = i+1
        img_p[idx_flat] = tmp
    img_p.shape = s

def extract_particles(img_bw, n_particles=3):
    from sklearn.cluster import KMeans
    from skimage.measure import label, regionprops
    s = img_bw.shape
    img_label = label(img_bw)
    p_area = []
    for region in regionprops(img_label):
        p_area.append(region.area)
    idx = np.argsort(p_area)[::-1]
    p_mask= {}
    for i in range(n_particles):
        p_mask[f'{i}'] = regionprops(img_label)[idx[i]]
    return p_mask




def batch_test():
    from scipy.ndimage import generate_binary_structure 
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.filters import gaussian_filter as gf
    from scipy.ndimage.filters import median_filter as mf
    from skimage import io
    n = 3
    fn_root = '/media/mingyuan/Seagate Backup Plus Drive/TXM_2019/Ruoqian/xanes_3D'
    file_scan = pyxas.retrieve_file_type(fn_root, 'recon', 'h5')
    for i in range(len(file_scan)):
        print(f'processing {i+1} / {len(file_scan)} ...')
        ts = time.time()
        scan_id = file_scan[i].split('.')[0][-5:]
        fn = file_scan[i]
        img = pyxas.get_img_from_hdf_file(fn, 'img')['img']
        img_blur = gf(img, 5)
        img_blur[img_blur < 0.0005] = 0
        #res = rm_image_bkg(img_blur, multiply_factor=1, n_jobs=4)
        #mask = extract_particles(res['img_labels'], n_particles=n)
        t=np.array(img_blur>0, dtype=np.int8) 
        mask = extract_particles(t, n_particles=n)
        fn1 = fn_root + f'/1/{scan_id}'
        fn2 = fn_root + f'/2/{scan_id}'
        fn3 = fn_root + f'/3/{scan_id}'
        coord = {}
        coord['0'], coord['1'], coord['2'] = [], [], []
        for j in range(n):
            t = mask[f'{j}'].image 
            s1 = t.shape
            t1 = np.zeros([500, 500])
            t1[:s1[1], :s1[2]] = t[len(t)//2]
            io.imsave(f'{fn_root}/{j}/{scan_id}.tiff',np.array(t1,dtype=np.float32))
            coord[f'{j}'].append(list(mask[f'{j}'].bbox))
        te = time.time()
        print(f'#{i+1} / {len(file_scan)} takes {te-ts:4.1f} sec')        
                




        














