import numpy as np
from torch.utils.data.dataset import Dataset
#np.set_printoptions(threshold=np.nan)
#import pandas as pd
import torch
import math
# Create the training dataset for pytorch
# Deals with one image at a time


# Clip to patch for its center pixel
def img_to_patches_to_tensor(index, list_id, image, patch_size):
    id = list_id[index]
    i = int(id/(len(image[0][0])-(patch_size-1)))
    j = id % (len(image[0][0])-(patch_size-1))
    patch = image[:, i:(i+patch_size), j:(j+patch_size)]
    return patch


# Trnsfrom numpy array to tensor
def toTensor(pic):
    if isinstance(pic, np.ndarray):
        pic = pic.astype(float)
        img = torch.from_numpy(pic).float()
        return img


# Class of the dataset
class ImageDataset(Dataset):

    def __init__(self, imgarray,patchsize):
        self.patch_size = patchsize
        self.image=imgarray
        self.patches=self.gridwise_sample(self.image,self.patch_size)


    def gridwise_sample(self,imgarray, patchsize):
        patchsamples1=[]
 
        nbands, nrows, ncols = imgarray.shape
        #print(imgarray.shape)
        patchsamples = np.zeros(shape=(nbands, patchsize, patchsize, 0),
									dtype=imgarray.dtype)
        for i in range(int(nrows/patchsize)):
            for j in range(int(ncols/patchsize)):
                tocat = imgarray[:,i*patchsize:(i+1)*patchsize,j*patchsize:(j+1)*patchsize]

                tocat=np.float32(tocat)

                val=tocat[tocat>=0]#[tocat!=0]
                if ((len(val)>4000)and np.all(tocat!=math.nan)):

                    patchsamples1.append(toTensor(tocat))
            tocat=None 

        return patchsamples1

    def __getitem__(self, index):
        # get patch for a pixel
        patches1=self.patches[index]
        return patches1
		
    def __len__(self):
        return len(self.patches)

class ImageDataset_test(Dataset):

    def __init__(self, imgarray,patchsize):
        self.patch_size = patchsize
        self.image=imgarray
        self.patches=self.gridwise_sample(self.image,self.patch_size)


    def gridwise_sample(self,imgarray, patchsize):
        patchsamples1=[]
 
        nbands, nrows, ncols = imgarray.shape
        #print(imgarray.shape)
        patchsamples = np.zeros(shape=(nbands, patchsize, patchsize, 0),
									dtype=imgarray.dtype)
        for i in range(int(nrows/patchsize)):
            for j in range(int(ncols/patchsize)):
                tocat = imgarray[:,i*patchsize:(i+1)*patchsize,
									 j*patchsize:(j+1)*patchsize]
				#print(tocat.shape)
				#tocat=tocat.permute(3,1,2,0)
                tocat=np.float32(tocat)
				#tocat = np.expand_dims(tocat, axis=3)
				#patchsamples = np.concatenate((patchsamples, tocat),
												  #axis=3)
                #print(tocat.shape)
                #if ((np.all(tocat)!=0) and (np.all(tocat) != math.nan)):
                
                patchsamples1.append(toTensor(tocat))


        return patchsamples1

    def __getitem__(self, index):
        # get patch for a pixel
        patches1=self.patches[index]
        return patches1
		
    def __len__(self):
        return len(self.patches)
