import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os

def trans_in_fda(src_path,tar_path,save_path,L=0,theta=1,lamda=None):
    """
    The Package Utils funtion to use the Fourier transform DA by amplitude
    """
    src_img=Image.open(src_path).convert('RGB')
    tar_img=Image.open(tar_path).convert('RGB')
    src_img=np.asarray(src_img,np.float32).transpose((2,0,1))
    tar_img=np.asarray(tar_img,np.float32).transpose((2,0,1))
    src_fft=np.fft.fft2(src_img)
    tar_fft=np.fft.fft2(tar_img)
    if lamda is None:
        new_fft=trans_in_fourier_by_amplitude(src_fft,tar_fft,L=L,lamda=theta)
    else:
        new_fft=trans_in_lambda(src_fft,tar_fft,lamda)
    new_img=np.real(np.fft.ifft2(new_fft)).transpose((1,2,0))
    imin = new_img.min()
    imax = new_img.max()
    image = (255 * (new_img - imin) / (imax - imin)).astype("uint8")
    image = Image.fromarray(image)
    image.save(save_path)
    return 0




def trans_in_lambda(src,tar,lamda):
    """
    Use the lamda-matrix to compute the mix transform
    """
    src=np.fft.fftshift(src)
    tar=np.fft.fftshift(tar)
    src_amp,src_pha=np.abs(src),np.angle(src)
    tar_amp,tar_pha=np.abs(tar),np.angle(tar)
    src_amp=np.multiply(lamda,tar_amp)+np.multiply((1-lamda),src_amp)
    new_src_fft=np.fft.ifftshift(src_amp*np.exp(1j*src_pha))
    # new_src_fft=src_amp*np.exp(1j*src_pha)
    return  new_src_fft


def trans_in_fourier_by_amplitude(src,tar,anchor=None,L=0,lamda=1):
    """
    This compute the amplitude exchange,

    anchor: the center of change rectangle
    L: the ratio of change area 
    """
    src=np.fft.fftshift(src)
    tar=np.fft.fftshift(tar)
    src_amp,src_pha=np.abs(src),np.angle(src)
    tar_amp,tar_pha=np.abs(tar),np.angle(tar)
    if anchor is None:
        # default is the center point
        anchor=(src.shape[1]//2,src.shape[2]//2)
    if L == 0:
        # Not explicit the ratio, then whole exchange
        L=0.5
    h,w=np.floor(src.shape[1]*L).astype(int),np.floor(src.shape[2]*L).astype(int)
    a=max(anchor[0]-h,0)
    b=min(anchor[0]+h,src_amp.shape[1])
    c=max(anchor[1]-w,0)
    d=min(anchor[1]+w,src_amp.shape[1])
    # print(f'The transfer ares size is {b-a,d-c}')
    src_amp[:,a:b,c:d]=lamda*tar_amp[:,a:b,c:d]+(1-lamda)*src_amp[:,a:b,c:d]
    new_src_fft=np.fft.ifftshift(src_amp*np.exp(1j*src_pha))
    # new_src_fft=src_amp*np.exp(1j*src_pha)
    return  new_src_fft


def trans_in_fourier_by_phase(src,tar,anchor=None,L=0):
    """
    This compute the phase exchange,

    anchor: the center of change rectangle
    L: the ratio of change area 
    """
    src=np.fft.fftshift(src)
    tar=np.fft.fftshift(tar)
    src_amp,src_pha=np.abs(src),np.angle(src)
    tar_amp,tar_pha=np.abs(tar),np.angle(tar)
    if anchor is None:
        # default is the center point
        anchor=(src.shape[1]//2,src.shape[2]//2)
    if L == 0:
        # Not explicit the ratio, then whole exchange
        L=0.5
    h,w=np.floor(src.shape[1]*L).astype(int),np.floor(src.shape[2]*L).astype(int)
    a=max(anchor[0]-h,0)
    b=min(anchor[0]+h,src_amp.shape[1])
    c=max(anchor[1]-w,0)
    d=min(anchor[1]+w,src_amp.shape[1])
    src_pha[:,a:b,c:d]=tar_pha[:,a:b,c:d]
    new_src_fft=np.fft.ifftshift(src_amp*np.exp(1j*src_pha))
    return  new_src_fft

def traditional_trans_in_spatial(src_img,tar_img):
    src_mean,src_std=src_img.mean(),src_img.std()
    tar_mean,tar_std=tar_img.mean(),src_img.std()
    # return np.uint8(((src_img-src_mean)/src_std))
    return np.uint8(((src_img-src_mean)/src_std)*tar_std+tar_mean)

if __name__ =='__main__':
    pass