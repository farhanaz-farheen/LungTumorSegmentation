


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

'''
for folder in tqdm(next(os.walk(folders))[1]):
    #print(folder)
    try:
        os.makedirs(os.path.join('processed_crop', folder))
    except:
        pass
    for pickleFileName in tqdm(next(os.walk(os.path.join(folders, folder)))[2]):
        fullAddr = os.path.join(folders, folder, pickleFileName)
        img = pickle.load(open(fullAddr, 'rb'))
        cropped_img = img[top:bottom, left:right]
        pickle.dump(cropped_img, open(os.path.join('processed_crop', folder, pickleFileName),'wb'))




if False:
    for image_path in tqdm(glob.glob("masks/*.p")):
        #print(image_path)
        image = pickle.load(open(image_path, 'rb'))
        image_path = image_path[6:]
        
        
        
        
        
        x,y = np.where(image!=0)
        top_margin, bottom_margin = min(x),max(x)
        left_margin, right_margin = min(y),max(y)
        
        
        lr_centre = int((left_margin+right_margin)/2)
        tb_centre = int((top_margin+bottom_margin)/2)

        if (tb_centre-32)<0:
            top_margin = 0
            bottom_margin = 63
        elif (tb_centre+31)>127:
            top_margin = 128-64
            bottom_margin = 127
        else:
            top_margin = tb_centre - 32
            bottom_margin = tb_centre + 31

        if (lr_centre-32)<0:
            left_margin = 0
            right_margin = 63
        elif (lr_centre+31)>127:
            left_margin = 128-64
            right_margin = 127
        else:
            left_margin = lr_centre - 32
            right_margin = lr_centre + 31

        '''
        
        top_margin = max(0, tb_centre - 35)
        bottom_margin = min(127, tb_centre + 34)
        left_margin = max(0, lr_centre - 35)
        right_margin = min(127, lr_centre + 34)
        '''
        
        '''
        new_image = image[top_margin:(bottom_margin+1), left_margin:(right_margin+1)]
        
        imsave("eroded_masks/" + image_path+"_cropped.png", new_image)
        '''

        fp.write(str(top_margin) + " " + str(left_margin))

        fp2.write(str(lr_centre) + " " + str(tb_centre))

        original_image = pickle.load(open("images/" + image_path, 'rb'))
        #print(original_image.shape)
        new_image = original_image[top_margin:(bottom_margin+1), left_margin:(right_margin+1)]
        
        #new_image = new_image * 255.0
        new_image = new_image.astype(np.uint8)
        new_image = cv2.resize(new_image, (128,128), interpolation=cv2.INTER_CUBIC)
        #imsave(crop_dir + "/images/" + image_path, new_image)
        cv2.imwrite(crop_dir + "/images/" + image_path[:-2] + ".png", new_image)

        original_mask = pickle.load(open("masks/" + image_path, 'rb'))
        #original_mask = io.imread("masks/" + image_path)
        #print(original_mask.shape)
        new_image = original_mask[top_margin:(bottom_margin+1), left_margin:(right_margin+1)]
        #new_image = new_image * 255.0
        new_image = new_image.astype(np.uint8)
        new_image = cv2.resize(new_image, (128,128), interpolation=cv2.INTER_CUBIC)
        new_image = (new_image >= 0.5)*255
        new_image = new_image.astype(np.uint8)
        #imsave(crop_dir + "/masks/" + image_path, new_image)
        cv2.imwrite(crop_dir + "/masks/" + image_path[:-2] + ".png", new_image)    
        

    fp.close()
    fp2.close()

'''
