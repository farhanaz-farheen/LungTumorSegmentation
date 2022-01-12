import os 
import cv2 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import pywt



def wavelet_selection():

    try:
        os.makedirs('filtered_wavelets_png')
        os.makedirs('filtered_wavelets_png/images')
        os.makedirs('filtered_wavelets_png/masks')
        os.makedirs('filtered_wavelets')
        os.makedirs('filtered_wavelets/images')
        os.makedirs('filtered_wavelets/masks')
    except:
        pass 



    
    folders = next(os.walk('./processed_crop'))[1]
    folders.sort()

    for folder in tqdm(folders):

        files = next(os.walk('./processed_crop/'+folder))[2]

        tot = int(len(files)/2)

        #files.sort()
        cnt = 0

        #for eachFile in files:
        for i in range(tot):
            #if eachFile.endswith('X.p'):
        
            img1 = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'X.p','rb'))
            #img1 = cv2.imread('./training_data/images/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
            
            img2, (h,v,d) = pywt.dwt2(img1, 'db1')
            img3, (h,v,d) = pywt.dwt2(img2, 'db1')

            '''plt.subplot(1,3,1), plt.imshow(img1)
            plt.subplot(1,3,2), plt.imshow(img2)
            plt.subplot(1,3,3), plt.imshow(img3)
            plt.show()'''

            mask = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'Y.p','rb'))
            
            #mask = cv2.imread('./training_data/masks/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)                 

            #kernel = np.ones((5,5),np.uint8)
            #mask = cv2.dilate(mask,kernel,iterations = 2)         
            print("datatype of mask: ",mask.dtype)
            img1 = cv2.resize(img1,(128, 128), interpolation = cv2.INTER_CUBIC)
            img2 = cv2.resize(img2,(128, 128), interpolation = cv2.INTER_CUBIC)
            img3 = cv2.resize(img3,(128, 128), interpolation = cv2.INTER_CUBIC)
            mask = cv2.resize(mask,(128, 128), interpolation = cv2.INTER_CUBIC)
            
            mask = (mask>0)
            mask = mask*255
            mask = mask.astype(np.uint8)
        
            img = np.zeros((128,128,3))
            img[:,:,0] = img1
            img[:,:,1] = img2 / 2
            img[:,:,2] = img3 / 4

            img = img.astype(np.uint8)

            cnt+=1

            cv2.imwrite('filtered_wavelets_png/images/'+folder+'-'+str(cnt)+'.png',img)
            cv2.imwrite('filtered_wavelets_png/masks/'+folder+'-'+str(cnt)+'.png',mask)
            try:

                pickle.dump(img,open(os.path.join('filtered_wavelets/images/',folder + str(cnt) + '.p'),'wb'))
                pickle.dump(mask, open(os.path.join('filtered_wavelets/masks/', folder + str(cnt) + '.p'), 'wb'))

            except:
                pass



def main():

    wavelet_selection()

if __name__ == '__main__':
    main()


    

