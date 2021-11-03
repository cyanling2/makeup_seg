import numpy as np
import colorsys

def BGR_to_CMYK(bgr_img):
    rows, cols ,channels = bgr_img.shape
    R_norm = bgr_img[:,:,2] / 255
    G_norm = bgr_img[:,:,1] / 255
    B_norm = bgr_img[:,:,0] / 255
    ones = np.ones([rows, cols])
    K = ones - np.maximum(R_norm, G_norm, B_norm) + 0.00001
    C = (ones - R_norm - K) / (ones - K)
    M = (ones - G_norm - K) / (ones - K)
    Y = (ones - B_norm - K) / (ones - K)
    cmyk_img = np.zeros([rows, cols, 4])
    cmyk_img[:,:,0] = C
    cmyk_img[:,:,1] = M
    cmyk_img[:,:,2] = Y
    cmyk_img[:,:,3] = K
    return cmyk_img

def BGR_to_HSV(bgr_img):
    rows, cols ,channels = bgr_img.shape
    R_norm = bgr_img[:,:,2] / 255
    G_norm = bgr_img[:,:,1] / 255
    B_norm = bgr_img[:,:,0] / 255
    hsv_img = np.zeros(bgr_img.shape)
    for i in range(rows):
        for j in range(cols):
            hsv_img[i,j] = colorsys.rgb_to_hsv(bgr_img[i,j,2]/255,
                                                bgr_img[i,j,1]/255,
                                                bgr_img[i,j,0]/255)
    return hsv_img
