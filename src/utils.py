import numpy as np
from keras.preprocessing.image import random_rotation, random_shift, \
    random_shear, random_zoom,random_channel_shift

def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    #assert (imgs.shape[3] == 3)  # check the channel is 3
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = (imgs[i] - imgs_mean) / imgs_std
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i])))
    return imgs_normalized

def argumentation(imgs):
    print(np.shape(imgs))
    r_imgs = [
        random_rotation(imgs, 30, row_axis=1, col_axis=2, channel_axis=3, fill_mode='nearest') * 255 for _ in
        range(5)]
    s_imgs = [
    random_shear(imgs, intensity=0.4, row_axis=1, col_axis=2, channel_axis=3, fill_mode='nearest') * 255 for _ in range(5)]
    sh_imgs =  [
    random_shift(imgs, wrg=0.1, hrg=0.3, row_axis=1, col_axis=2, channel_axis=3, fill_mode='nearest') * 255  for _ in range(5)]
    z_imgs = [
    random_zoom(imgs, zoom_range=(1.5, 0.7), row_axis=1, col_axis=2, channel_axis=3, fill_mode='nearest') * 255  for _ in range(5)]
    imgs = np.append(imgs, r_imgs)
    imgs = np.append(imgs, s_imgs)
    imgs = np.append(imgs, sh_imgs)
    imgs = np.append(imgs, z_imgs)
    return imgs
