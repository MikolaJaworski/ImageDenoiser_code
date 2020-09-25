from __future__ import print_function, division
import os
import skimage
import cv2
from skimage.util import random_noise
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore")

'''
images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print('Working with {} files from: {}'.format(len(images), path))
'''

def show_image(image):
    plt.imshow(image)
    plt.show()

# show_image(io.imread(os.path.join(path, images[0])), images[0])

def target_image(args):
    images = [io.imread(os.path.join(args.data_path, i)) for i in os.listdir(args.data_path)]
    for im in tqdm(range(len(images))):
        norm_img = cv2.normalize(images[im], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        nois_img = random_noise(norm_img, mode='gaussian')
        f_in_name = '{}_{}.npy'.format(im, 'input')
        f_target_name = '{}_{}.npy'.format(im, 'target')
        np.save(os.path.join(args.save_path, f_in_name), nois_img)
        np.save(os.path.join(args.save_path, f_target_name), norm_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train_images/')
    parser.add_argument('--save_path', type=str, default='./TrainingDataset/')
    args = parser.parse_args()
    target_image(args)
