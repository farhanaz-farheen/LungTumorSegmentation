
import numpy as np
import matplotlib.pyplot as plt 
import h5py
import datetime
import os 
#from sklearn.metrics import classification_report

#from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Flatten, Reshape, BatchNormalization
from keras.models import Model, model_from_json
from keras.optimizers import Adam, SGD
from losses import dice_coef_loss
from metrics import dice_coef , jacard
import pickle
import numpy as np 
import os 
import time
from tqdm import tqdm
import glob
from tensorflow import set_random_seed
import cv2


globalX = []
globalY = []
imgNames = []




def loadModel(fold, modelName):
    
    json_file = open('models/'+str(fold)+'/'+modelName+'/'+'modelP.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    model = model_from_json(loaded_model_json)
    
    model.load_weights('models/'+str(fold)+'/'+modelName+'/'+'modelW.h5')

    sgd = SGD(lr=0.005, momentum=0.8, decay=0.01/50, nesterov=True)


    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01/150, amsgrad=False)
    #test 2
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001/150, amsgrad=False)
    
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])


    #model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])

    return model




def evaluateModel(model,modelName,fold,numRotations,numImages,wav_path,nei_path,maskPath,batchSize):


    try:
        os.makedirs('valid_res')
    except:
        pass

    try:
        os.makedirs('valid_res/'+str(fold))
    except:
        pass

    try:
        os.makedirs('valid_res/'+str(fold)+'/'+modelName)
    except:
        pass

    

    #global globalX
    #global globalY
    #X_test = globalX
    #Y_test = globalY


    #print("X_test 0? ",len(X_test))

    value = evaluateFold(model,modelName,fold,batchSize,numImages,numRotations,wav_path,nei_path,maskPath)

    fp = open('valid_res/'+str(fold)+'_dice.txt','a')
    fp.write(str(value[1]))
    fp.close()

    '''
    fp = open('valid_res/'+str(fold)+'/'+modelName+'/log.txt','w')
    fp.write(str(value[1]))
    fp.close()
    '''
    
    #print('Jacard : '+str(value[0]))
    print('Dice : '+str(value[1]))
    #print(value[2])

    fp = open('models/'+str(fold)+'/'+modelName+'/log.txt','a')
    fp.write(str(value)+'\n')
    fp.close()

    fp = open('models/'+str(fold)+'/'+modelName+'/best.txt','r')
    best = fp.read()
    fp.close()

    if(value[1]>float(best)):
        print('***********************************************')
        print('Dice improve from '+str(best)+' to '+str(value[1]))
        print('***********************************************')
        '''
        fp = open('models/'+str(fold)+'/'+modelName+'/best.txt','w')
        fp.write(str(value[1]))
        fp.close()

        saveModel(model,modelName,fold,'best')
        '''

    print(datetime.datetime.time(datetime.datetime.now()))



def TTA(model,batchSize,wav_path,nei_path,maskPath,numRotations,numImages):

    global globalX
    global globalY


    try:
        os.makedirs('tempData')
    except:
        pass


    X = []
    Y = []
    YP = []
    
    tempArrxn = []
    tempArrxw = []
    tempArry = []
    count = 0

    Xnew = []
    angArr = []
    fp = open('imagenames.txt','a')

    for imgId in tqdm(glob.glob(wav_path + '/*.p')):
        tempLen = len(wav_path) + 1
        imgId = imgId[tempLen:]

        #print(imgId)
        imgNames.append(imgId)
        
        fp.write(str(imgId)+'\n')


        #xw1 = cv2.imread(wav_path+'/'+imgId, cv2.IMREAD_COLOR)
        #xn1 = cv2.imread(nei_path+'/'+imgId, cv2.IMREAD_COLOR)
        #y1 = cv2.imread(maskPath+'/'+imgId,cv2.IMREAD_GRAYSCALE)
        xw1 = pickle.load(open(wav_path+'/'+imgId,'rb'))
        xn1 = pickle.load(open(nei_path+'/'+imgId,'rb'))
        y1 = pickle.load(open(maskPath+'/'+imgId,'rb'))

        
        y2 = y1.reshape(y1.shape[0], y1.shape[1], 1)

        
        y2 = y2 / 255
        y2 = np.round(y2,0)               
        

        Y.append(y2)



        x_org = np.zeros((128,128,5))
        x_org[:,:,0] = xn1[:,:,0]
        x_org[:,:,1] = xw1[:,:,0]
        x_org[:,:,2] = xw1[:,:,1]
        x_org[:,:,3] = xw1[:,:,2]            
        x_org[:,:,4] = xn1[:,:,2]
        


        x_org = x_org / 255

              
        
        X.append(x_org)




        tempArrxw.append(xw1)
        tempArrxn.append(xn1)
        tempArry.append(y1)

        count += 1

        if (count%numImages) == 0:
            for j in range(numImages):
                angArr.append(0)
                xw = tempArrxw[j]
                xn = tempArrxn[j]
                y = tempArry[j]

                #x = cv2.resize(x ,(width,height), interpolation=cv2.INTER_CUBIC)
                #y = cv2.resize(y ,(width,height), interpolation=cv2.INTER_CUBIC)

                x = np.zeros((128,128,5))
                x[:,:,0] = xn[:,:,0]
                x[:,:,1] = xw[:,:,0]
                x[:,:,2] = xw[:,:,1]
                x[:,:,3] = xw[:,:,2]            
                x[:,:,4] = xn[:,:,2]
                
                y = y.reshape(y.shape[0], y.shape[1], 1)

                x = x / 255
                y = y / 255
                y = np.round(y,0)               
                
                #X.append(x)
                #Y.append(y)

                Xnew.append(x)

                    
                for i in range(numRotations):
                    #ang = np.random.rand()*40.0 - 20.0
                    ang = (i+1)*(360.0/numRotations)
                    angArr.append(ang)


                    xw = tempArrxw[j]
                    xn = tempArrxn[j]
                    y = tempArry[j]

                    #plt.subplot(1,2,1)
                    #plt.imshow(xn)
                    
                    rows,cols = y.shape

                    #print('ROCOL HERE HEHE: ',rows,' ',cols)

                    M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)

                    xw = cv2.warpAffine(xw,M,(cols,rows))
                    xn = cv2.warpAffine(xn,M,(cols,rows))
                    
                    y = cv2.warpAffine(y,M,(cols,rows))

                    #plt.subplot(1,2,2)
                    #plt.imshow(xn)
                    #plt.show()

                    x = np.zeros((128,128,5))
                    x[:,:,0] = xn[:,:,0]
                    x[:,:,1] = xw[:,:,0]
                    x[:,:,2] = xw[:,:,1]
                    x[:,:,3] = xw[:,:,2]            
                    x[:,:,4] = xn[:,:,2]
                
                    y = y.reshape(y.shape[0], y.shape[1], 1)

                    x = x / 255
                    y = y / 255
                    y = np.round(y,0) 

                    Xnew.append(x)  

                    #X.append(x)
                    #Y.append(y)

            #now Xnew has (1+numrotations)*numimages elements
            Xnew = np.array(Xnew)
            yp = model.predict(x=Xnew, batch_size=batchSize, verbose=1)
            yp = np.array(yp[0])
            '''
            print('here\n', yp[0])
            
            #print('shape of yp -> ', yp.shape)
            print('len of yp: ', len(yp))
            break
            '''
            
            for k in range(len(yp)):

                ang = (-1)*angArr[k]

                rows = yp.shape[1]
                cols = yp.shape[2]
                #rows,cols = yp.shape

                M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1) 
                #print('rocol here -> ', rows,' ', cols)

                tmp = yp[k][:,:,0]
                
                tmp = cv2.warpAffine(tmp,M,(cols,rows))
                yp[k] = tmp.reshape(yp.shape[1], yp.shape[2], 1)

                

            for idx in range(numImages):
                start = idx*(1+numRotations)
                subyp = yp[start:(start+(1+numRotations))]
                
                subyp = np.array(subyp)
                '''
                print('something\n', subyp.shape)
                while (True):
                    x = 1
                '''
                y_img = np.average(subyp, axis=0)
                YP.append(y_img)
                






            tempArrxw = []
            tempArrxn = []
            tempArry = []
            Xnew = []

    print("size: ",len(imgNames))
    
    fp.close()


    Y = np.array(Y)
    globalY = Y

    #print(Y)
    YP = np.array(YP)
    print('before return : ', YP.shape)

    X = np.array(X)
    globalX = X

    print("size of y and global y are: ",Y.shape," ",globalY.shape)
    print("size of x and global x are: ",X.shape," ",globalX.shape)

    return YP

                




    




def evaluateFold(model,modelName,fold,batchSize,numImages,numRotations,wav_path,nei_path,maskPath):

    global globalX
    global globalY
    try:
        os.makedirs('image_results')
    except:
        pass 
    

    
    X = globalX
    Y = globalY


    yp = TTA(model,batchSize,wav_path,nei_path,maskPath,numRotations,numImages)
    
    #Y = yp
    X = globalX
    Y = globalY
    #REMOVE this line from others
    #yp = np.round(yp,0)
    yp = (yp > 0.4) * 1.0
    print('after TTA: ',yp.shape)

    

    for i in tqdm(range(len(yp))):

        img = np.zeros([128,128])

        img[:,:] = yp[i].reshape(128,128)
        img = img * 255

        nameImage = imgNames[i]

     







    for i in tqdm(range(len(yp))):
    #for i in tqdm(range(0)):
        #if (np.sum(Y[i])!=0 and np.sum(yp[i])!=0):
        plt.subplot(1,3,1)
        plt.imshow(X[i][:,:,1].reshape(X[i].shape[0],X[i].shape[1]))
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y[i].reshape(Y[i].shape[0],Y[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
        plt.title('Prediction')

        #intersection = yp[i].ravel() * Y[i].ravel()
        #union = yp[i].ravel() + Y[i].ravel() - intersection

        #print(set((yp[i].ravel())), set((Y[i].ravel())),set(union),set(intersection))

        #jacard = (np.sum(intersection)/np.sum(union))  
        #plt.suptitle('Jacard '+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard))
        #plt.show()
        #plt.savefig('image_results/'+str(i)+'.png',format='png')
        plt.close()

        lung_img = X[i][:,:,1].reshape(X[i].shape[0],X[i].shape[1])
        lung_img = lung_img * 255
        grnd_truth_img = Y[i].reshape(Y[i].shape[0],Y[i].shape[1])
        grnd_truth_img = grnd_truth_img * 255
        predicted_img = yp[i].reshape(yp[i].shape[0],yp[i].shape[1])
        predicted_img = predicted_img * 255

        
        try:
            os.makedirs('image_results/'+str(i))
        except:
            pass

        cv2.imwrite('image_results/' + str(i) + '/' + 'Input_' + str(i) + '.png', lung_img)

        cv2.imwrite('image_results/' + str(i) + '/' + 'GroundTruth_' + str(i) + '.png', grnd_truth_img)

        cv2.imwrite('image_results/' + str(i) + '/' + 'Prediction_' + str(i) + '.png', predicted_img)
        

    
    


    X = None 

    #jacard = 0
    dice = 0
    
    yp_2arr = []
    y2_arr = []
    fncnt = 0
    tpcnt = 0
    tpden = 0
    fpcnt = 0
    tncnt = 0
    
    cnt = 0
    for i in range(len(yp)):
        yp_2 = yp[i].ravel()
        y2 = Y[i].ravel()
        intersection = yp_2 * y2
        #union = yp_2 + y2 - intersection
        '''
        print(np.sum(intersection))
        print(np.sum(yp_2))
        print(np.sum(y2))
        break
        '''
        temp = 0
        #print(np.sum(intersection),np.sum(union), jacard)
        #input('')
        #jacard += (np.sum(intersection)/np.sum(union))
        # 
        # best one  
        #temp = ((2. * np.sum(intersection) ) + 0.001) / (np.sum(yp_2) + np.sum(y2) + 0.001)
        if(np.sum(y2)==0 and np.sum(yp_2)==0):
            dice += 1
            cnt += 1
            temp = 1
            tncnt += 1
        elif(np.sum(y2)==0 and np.sum(yp_2)!=0):
            cnt += 1
            fpcnt += 1
            temp = 0
        else:
            cnt += 1
            temp = ((2. * np.sum(intersection) + 0.001)) / (np.sum(yp_2) + np.sum(y2) + 0.001)
            #temp = ((2. * np.sum(intersection) / (np.sum(yp_2) + np.sum(y2)))
            dice += temp
            #print('Dice tmp: ', temp)

        if(np.sum(y2)!=0 and np.sum(yp_2)==0):
            fncnt += 1
        if(np.sum(y2)!=0 and np.sum(yp_2)!=0):
            tpcnt += 1
        if(np.sum(y2)!=0):
            tpden += 1
        fp = open('valid_res/'+str(fold)+'_log.txt','a+')
        fp.write(str(temp) + '\n')
        fp.close()

        '''
        if(np.sum(yp_2)==0):
            dummy = 1
        else:
            cnt += 1
            temp = ((2. * np.sum(intersection) + 0.0001)) / (np.sum(yp_2) + np.sum(y2) + 0.0001)
            #temp = ((2. * np.sum(intersection) / (np.sum(yp_2) + np.sum(y2)))
            dice += temp
            #print('Dice tmp: ', temp)
            fp = open('valid_res/'+str(fold)+'_log.txt','a+')
            fp.write(str(temp) + '\n')
            fp.close()
        '''
    

        #yp_2arr.extend(yp_2)
        #y2_arr.extend(y2)

   

    dice /= len(Y)
    #dice /= cnt
    print("dice: " + str(dice))
    tprate = tpcnt/tpden
    fnrate = fncnt/(fncnt + tpcnt)
    print("false negative rate: " + str(fnrate))

    print("true positive rate: " + str(tprate))

    fprate = fpcnt/(fpcnt + tncnt)
    print("false positive rate: " + str(fprate))
    print("fncount - " + str(fncnt) + " tpcount - " + str(tpcnt) + " fpcount - " + str(fpcnt) + " tncount - " + str(tncnt))
    #clf_rprt = classification_report(y2_arr,yp_2arr)


    return [0 , dice, 0]





def foldData(wav_path, nei_path,maskPath,width=128, height=128):

    global globalX
    global globalY


    try:
        os.makedirs('tempData')
    except:
        pass
    
    #for fold in tqdm(range(1,6)):

    #fp = open('data/'+str(fold)+'.txt','r')

    #ids = fp.read().split('\n')[:-1] 
    fold = 4

    X = []
    Y = []

    for imgId in tqdm(glob.glob(wav_path + '/*.p')):
        tempLen = len(wav_path) + 1
        imgId = imgId[tempLen:]

        #print(imgId)
        
        '''
        xw = cv2.imread(wav_path+'/'+imgId, cv2.IMREAD_COLOR)
        xn = cv2.imread(nei_path+'/'+imgId, cv2.IMREAD_COLOR)
        y = cv2.imread(maskPath+'/'+imgId,cv2.IMREAD_GRAYSCALE)
        '''
        xw = pickle.load(open(wav_path+'/'+imgId,'rb'))
        xn = pickle.load(open(nei_path+'/'+imgId,'rb'))
        y = pickle.load(open(maskPath+'/'+imgId,'rb'))


        #x = cv2.resize(x ,(width,height), interpolation=cv2.INTER_CUBIC)
        #y = cv2.resize(y ,(width,height), interpolation=cv2.INTER_CUBIC)

        x = np.zeros((128,128,5))
        x[:,:,0] = xn[:,:,0]
        x[:,:,1] = xw[:,:,0]
        x[:,:,2] = xw[:,:,1]
        x[:,:,3] = xw[:,:,2]            
        x[:,:,4] = xn[:,:,2]
        
        y = y.reshape(y.shape[0], y.shape[1], 1)

        x = x / 255
        y = y / 255
        y = np.round(y,0)               
        
        X.append(x)
        Y.append(y)


        '''
        if(np.random.rand()<0.5):

            direc = np.random.choice([0,1,-1])

            xw = cv2.imread(wav_path+'/'+imgId, cv2.IMREAD_COLOR)
            xn = cv2.imread(nei_path+'/'+imgId, cv2.IMREAD_COLOR)
            y = cv2.imread(maskPath+'/'+imgId,cv2.IMREAD_GRAYSCALE)

            #plt.subplot(1,2,1)
            #plt.imshow(xn)

            xn = cv2.flip( xn, direc )
            xw = cv2.flip( xw, direc )
            y = cv2.flip( y, direc )

            #plt.subplot(1,2,2)
            #plt.imshow(xn)
            #plt.show()
        
            x = np.zeros((128,128,5))
            x[:,:,0] = xn[:,:,0]
            x[:,:,1] = xw[:,:,0]
            x[:,:,2] = xw[:,:,1]
            x[:,:,3] = xw[:,:,2]            
            x[:,:,4] = xn[:,:,2]
        
            y = y.reshape(y.shape[0], y.shape[1], 1)

            x = x / 255
            y = y / 255
            y = np.round(y,0)


            X.append(x)
            Y.append(y)
        '''

        #else:   

              
        for i in range(6):
            ang = np.random.rand()*40.0 - 20.0
            '''
            xw = cv2.imread(wav_path+'/'+imgId, cv2.IMREAD_COLOR)
            xn = cv2.imread(nei_path+'/'+imgId, cv2.IMREAD_COLOR)
            y = cv2.imread(maskPath+'/'+imgId,cv2.IMREAD_GRAYSCALE)
            '''
            xw = pickle.load(open(wav_path+'/'+imgId,'rb'))
            xn = pickle.load(open(nei_path+'/'+imgId,'rb'))
            y = pickle.load(open(maskPath+'/'+imgId,'rb'))


            #plt.subplot(1,2,1)
            #plt.imshow(xn)
            
            rows,cols = y.shape

            M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)

            xw = cv2.warpAffine(xw,M,(cols,rows))
            xn = cv2.warpAffine(xn,M,(cols,rows))
            
            y = cv2.warpAffine(y,M,(cols,rows))

            #plt.subplot(1,2,2)
            #plt.imshow(xn)
            #plt.show()

            x = np.zeros((128,128,5))
            x[:,:,0] = xn[:,:,0]
            x[:,:,1] = xw[:,:,0]
            x[:,:,2] = xw[:,:,1]
            x[:,:,3] = xw[:,:,2]            
            x[:,:,4] = xn[:,:,2]
        
            y = y.reshape(y.shape[0], y.shape[1], 1)

            x = x / 255
            y = y / 255
            y = np.round(y,0)   

            X.append(x)
            Y.append(y)
    
    

    print("globalx 0 BEFORE NP? ",len(globalX))

    X = np.array(X)
    Y = np.array(Y)
    print("x 0? ",len(X))
    #X = np.array(X)
    #Y = np.array(Y)

    globalX = X
    globalY = Y

    tqdm.write(str(len(X))+' '+str(len(Y)) ) 

    hfx = h5py.File('tempData/X'+str(fold)+'.h5', 'w')
    hfy = h5py.File('tempData/Y'+str(fold)+'.h5', 'w')

    hfx.create_dataset('X', data=X)
    hfy.create_dataset('Y', data=Y)

    hfx.close()
    hfy.close()





def main():



    np.random.seed(3)

    wav_path = './filtered_wavelets_validation/images'
    nei_path = './filtered_neighbors_validation/images'
    mask_path = './filtered_wavelets_validation/masks'

    #foldData(wav_path, nei_path, mask_path)
    #packData()


    batchSize =25
    nDivisions = 1
    #modelName = 'unetWaveletNeighborNorm'
    #modelName = 'MultiResUnetWaveletNeighbor'
    modelName = 'DeepSupervisionModel'
    fold = 1

    numRotations = 20
    numImages = 5
    
    model = loadModel(fold,modelName)
    model = evaluateModel(model,modelName,fold,numRotations,numImages,wav_path,nei_path,mask_path,batchSize)

    '''
    for fold in range(1,6):

        np.random.seed(fold)
        set_random_seed(fold)
        model = loadModel(fold,modelName)
        model = evaluateModel(model,modelName,fold,batchSize=batchSize)
        model = None
        time.sleep(30)
    '''
    


if __name__ == '__main__':
    main()
