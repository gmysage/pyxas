import numpy as np


def colormix(img, color='r, g, b', scale=[1], clim=[0,1], plot_flag=0):
    s = img.shape
    assert len(s) >= 2, 'colormix: need 3D image to convert rgb'
    num_channel = s[0]   

    color = color.replace(' ','')
    color = color.replace(';', ',')
    color = color.split(',') 
    if len(color) < num_channel:
        color = ['r', 'g', 'b', 'c', 'p', 'y']
    color = color[:num_channel]
    color_vec = convert_rgb_vector(color)

    if len(scale) == 1 or len(scale) != num_channel:
        scale = np.ones(len(s))

    img_color = img.copy()    
    img_color = (img_color - clim[0]) / (clim[1] - clim[0])
    for i in range(num_channel):
         img_color[i] *= scale[i]

    img_color = convert_rgb_img(img_color, color_vec)
    if plot_flag:
        plt.figure()
        plt.imshow(img_color)
    return img_color


def convert_rgb_vector(color):
    n = len(color)
    vec = np.zeros([n, 3])
    for i in range(n):
        if color[i] == 'r': vec[i] = [1, 0, 0]
        if color[i] == 'g': vec[i] = [0, 1, 0] 
        if color[i] == 'b': vec[i] = [0, 0, 1]
        if color[i] == 'c': vec[i] = [0, 1, 1] 
        if color[i] == 'p': vec[i] = [1, 0, 1] 
        if color[i] == 'y': vec[i] = [1, 1, 0]
    return vec 


def convert_rgb_img(img, color_vec):
    s = img.shape
    assert len(s) >= 2, 'need 3D image to convert rgb'
    img_color = np.zeros([s[1], s[2], 3])
    cR, cG, cB = 0, 0, 0
    for i in range(s[0]):
        cR += img[i] * color_vec[i][0]
        cG += img[i] * color_vec[i][1]
        cB += img[i] * color_vec[i][2]
    img_color[:, :, 0] = cR
    img_color[:, :, 1] = cG
    img_color[:, :, 2] = cB
    return img_color








