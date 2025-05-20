# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.fftpack import dct, idct

# %% DCT and IDCT
class DCT_IDCT:
    def __init__(self, block_size=8):
        self.block_size = block_size

    def dct2_blockwise(self, img): # DCT2 with block size 8x8
        h, w = img.shape
        padded_h = (h + self.block_size - 1) // self.block_size * self.block_size
        padded_w = (w + self.block_size - 1) // self.block_size * self.block_size
        padded_img = np.pad(img, ((0, padded_h - h), (0, padded_w - w)), mode='edge')
        dct_img = np.zeros_like(padded_img, dtype=np.float32)
        for i in range(0, padded_h, self.block_size):
            for j in range(0, padded_w, self.block_size):
                block = padded_img[i:i+self.block_size, j:j+self.block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_block[4:, :] = 0
                dct_block[:, 4:] = 0
                dct_img[i:i+self.block_size, j:j+self.block_size] = dct_block
        return dct_img, h, w

    def idct2_blockwise(self, dct_img, orig_h, orig_w): # IDCT2 with block size 8x8
        recon_img = np.zeros_like(dct_img, dtype=np.float32)
        for i in range(0, dct_img.shape[0], self.block_size):
            for j in range(0, dct_img.shape[1], self.block_size):
                block = dct_img[i:i+self.block_size, j:j+self.block_size]
                idct_block = idct(idct(block.T, norm='ortho').T, norm='ortho')
                recon_img[i:i+self.block_size, j:j+self.block_size] = idct_block
        recon_img = recon_img[:orig_h, :orig_w]
        return recon_img

# %%
