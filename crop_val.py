


import os 
import cv2 
from tqdm import tqdm
import numpy as np
#from scipy.misc import imsave
import glob
from skimage import io
import sys
np.set_printoptions(threshold=sys.maxsize)
import pickle




folders = 'processed'

try:
    os.makedirs('processed_crop')
except:
    pass


top = left = 512
bottom = right = -1

for folder in tqdm(next(os.walk(folders))[1]):
    #print(folder)
    for pickleFileName in tqdm(next(os.walk(os.path.join(folders, folder)))[2]):
        if pickleFileName.endswith('Y.p'):    
            #print(pickleFileName)
            fullAddr = os.path.join(folders, folder, pickleFileName)
            #print(fullAddr)
            mask = pickle.load(open(fullAddr, 'rb'))
            
            x,y = np.where(mask!=0)
            top_margin, bottom_margin = min(x),max(x)
            left_margin, right_margin = min(y),max(y)

            top = min(top, top_margin)
            left = min(left, left_margin)
            bottom = max(bottom, bottom_margin)
            right = max(right, right_margin)

print(str(top)+ " "+ str(bottom))
print(str(left)+ " "+ str(right))

top -= 20
bottom += 20
left -= 20
right += 20

print(str(top)+ " "+ str(bottom))
print(str(left)+ " "+ str(right))
