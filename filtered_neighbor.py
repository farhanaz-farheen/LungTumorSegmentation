import os 
import cv2 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import matplotlib.pyplot as plt 


def neighbor_selection():

    

    try:
        os.makedirs('filtered_neighbors_png')
        os.makedirs('filtered_neighbors_png/images')
        os.makedirs('filtered_neighbors_png/masks')
        os.makedirs('filtered_neighbors')
        os.makedirs('filtered_neighbors/images')
        os.makedirs('filtered_neighbors/masks')
    except:
        pass 

    folders = next(os.walk('./processed_crop'))[1]
    folders.sort()

    for folder in tqdm(folders):

        files = next(os.walk('./processed_crop/'+folder))[2]

        tot = int(len(files)/2)

        #files.sort()
        #print("sorted list of " + str(folder) + "-" + str(files))

        cnt = 0

        for i in range(tot):

            if(i==0):
                img1 = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'X.p','rb'))
                img2 = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'X.p','rb'))
                img3 = pickle.load(open('./processed_crop/' + folder + '/' + str(i+1) + 'X.p','rb'))
                #img1 = cv2.imread('./training_data/images/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
                #img2 = cv2.imread('./training_data/images/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
                #img3 = cv2.imread('./training_data/images/'+folder+'/'+str(i+1)+'.png', cv2.IMREAD_GRAYSCALE)

                mask = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'Y.p','rb'))
                '''
                plt.figure(figsize=(20,10))
                plt.subplot(1,4,1)
                plt.imshow(img1)
                plt.title('Img 1')
                plt.subplot(1,4,2)
                plt.imshow(img2)
                plt.title('img 2')
                plt.subplot(1,4,3)
                plt.imshow(img3)
                plt.title('img 3')
                plt.subplot(1,4,4)
                plt.imshow(mask)
                plt.title('mask')
                plt.show()
                '''

                #mask = cv2.imread('./training_data/masks/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE) 

                
                img1 = cv2.resize(img1,(128, 128), interpolation = cv2.INTER_CUBIC)
                img2 = cv2.resize(img2,(128, 128), interpolation = cv2.INTER_CUBIC)
                img3 = cv2.resize(img3,(128, 128), interpolation = cv2.INTER_CUBIC)
                mask = cv2.resize(mask,(128, 128), interpolation = cv2.INTER_CUBIC)
                
                mask = (mask>0)
                mask = mask*255
                mask = mask.astype(np.uint8)

            elif(i==(tot-1)):

                img1 = pickle.load(open('./processed_crop/' + folder + '/' + str(i-1) + 'X.p','rb'))
                img2 = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'X.p','rb'))
                img3 = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'X.p','rb'))

                #img1 = cv2.imread('./training_data/images/'+folder+'/'+str(i-1)+'.png', cv2.IMREAD_GRAYSCALE)
                #img2 = cv2.imread('./training_data/images/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
                #img3 = cv2.imread('./training_data/images/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)

                #mask = cv2.imread('./training_data/masks/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE) 
                mask = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'Y.p','rb')) 
                '''
                plt.figure(figsize=(20,10))
                plt.subplot(1,4,1)
                plt.imshow(img1)
                plt.title('Img 1')
                plt.subplot(1,4,2)
                plt.imshow(img2)
                plt.title('img 2')
                plt.subplot(1,4,3)
                plt.imshow(img3)
                plt.title('img 3')
                plt.subplot(1,4,4)
                plt.imshow(mask)
                plt.title('mask')
                '''

                img1 = cv2.resize(img1,(128, 128), interpolation = cv2.INTER_CUBIC)
                img2 = cv2.resize(img2,(128, 128), interpolation = cv2.INTER_CUBIC)
                img3 = cv2.resize(img3,(128, 128), interpolation = cv2.INTER_CUBIC)
                mask = cv2.resize(mask,(128, 128), interpolation = cv2.INTER_CUBIC)
                
                mask = (mask>0)
                mask = mask*255
                mask = mask.astype(np.uint8)

            else:
                #print(str(i)+ "total: " + str(tot))
                img1 = pickle.load(open('./processed_crop/' + folder + '/' + str(i-1) + 'X.p','rb'))
                img2 = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'X.p','rb'))
                img3 = pickle.load(open('./processed_crop/' + folder + '/' + str(i+1) + 'X.p','rb'))

                #img1 = cv2.imread('./training_data/images/'+folder+'/'+str(i-1)+'.png', cv2.IMREAD_GRAYSCALE)
                #img2 = cv2.imread('./training_data/images/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
                #img3 = cv2.imread('./training_data/images/'+folder+'/'+str(i+1)+'.png', cv2.IMREAD_GRAYSCALE)

                #mask = cv2.imread('./training_data/masks/'+folder+'/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE) 

                mask = pickle.load(open('./processed_crop/' + folder + '/' + str(i) + 'Y.p','rb')) 
                
                '''
                if folder.endswith('69'):
                    plt.figure(figsize=(20,10))
                    plt.subplot(1,4,1)
                    plt.imshow(img1)
                    plt.title(files[2*cnt - 2])
                    plt.subplot(1,4,2)
                    plt.imshow(img2)
                    plt.title(files[2*cnt])
                    plt.subplot(1,4,3)
                    plt.imshow(img3)
                    plt.title(files[2*cnt + 2])
                    plt.subplot(1,4,4)
                    plt.imshow(mask)
                    plt.title(files[2*cnt + 1])
                    plt.show()
                '''

                
                img1 = cv2.resize(img1,(128, 128), interpolation = cv2.INTER_CUBIC)
                img2 = cv2.resize(img2,(128, 128), interpolation = cv2.INTER_CUBIC)
                img3 = cv2.resize(img3,(128, 128), interpolation = cv2.INTER_CUBIC)
                mask = cv2.resize(mask,(128, 128), interpolation = cv2.INTER_CUBIC)
                
                mask = (mask>0)
                mask = mask*255
                mask = mask.astype(np.uint8)
            
            cnt+=1

            img = np.zeros((128,128,3))
            img[:,:,0] = img1
            img[:,:,1] = img2
            img[:,:,2] = img3

            cv2.imwrite('filtered_neighbors_png/images/'+folder+'-'+str(cnt)+'.png',img)
            cv2.imwrite('filtered_neighbors_png/masks/'+folder+'-'+str(cnt)+'.png',mask)

            try:

                pickle.dump(img,open(os.path.join('filtered_neighbors/images/',folder + str(cnt) + '.p'),'wb'))
                pickle.dump(mask, open(os.path.join('filtered_neighbors/masks/', folder + str(cnt) + '.p'), 'wb'))

            except:
                pass



def main():

    neighbor_selection()

if __name__ == '__main__':
    main()


    

