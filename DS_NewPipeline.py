#from helper_functions import *
from metrics import *
from models import *
import pickle
import os
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt 
import h5py
import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Flatten, Reshape, BatchNormalization
from keras.models import Model, model_from_json
from keras.optimizers import Adam, SGD
import pickle
import time
import cv2 
from tensorflow import set_random_seed
from losses import dice_coef_loss, balanced_cross_entropy

from keras import backend as K 


def saveModel(model, modelName, fold, best =''):

    model_json = model.to_json()

    try:
        os.makedirs('models')
    except:
        pass

    try:
        os.makedirs('models/'+str(fold))
    except:
        pass

    try:
        os.makedirs('models/'+str(fold)+'/'+modelName)
    except:
        pass

    fp = open('models/'+str(fold)+'/'+modelName+'/'+best+'modelP.json','w')
    fp.write(model_json)
    model.save_weights('models/'+str(fold)+'/'+modelName+'/'+best+'modelW.h5')




def createModel(fold, modelName):

    #model = UNet(height=128, width=128)
    model = MultiResUNet(128,128)
    #model = UNetDS32(128)
    #sgd = SGD(lr=0.5, momentum=0.8, decay=0.5/150, nesterov=True)
    #test a
    #sgd = SGD(lr=0.01, momentum=0.5, decay=0.01/150, nesterov=True)
    #test b
    #sgd = SGD(lr=0.01, momentum=0.5, decay=0.1/150, nesterov=True)

    #test 1
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01/150, amsgrad=False)
    #test 2 - training best
    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01/150, amsgrad=False)
    #test 2
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001/150, amsgrad=False)
    #model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_squared_error'], loss_weights=[1., 0.9, 0.8, 0.7, 0.6])
    #model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_squared_error'], loss_weights=[1.0, 0.05, 0.0025, 0.000125, 0.00000625])
    #model.compile(loss=dice_coef_loss,optimizer='adam',metrics=[dice_coef], loss_weights=[1.0, 0.8, 0.6, 0.4, 0.2])
    #model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])
    #model.compile(loss='binary_crossentropy',optimizer=adam,metrics=[dice_coef], loss_weights=[1.0, 0.8, 0.6, 0.4, 0.2])
    model.compile(loss=balanced_cross_entropy(0.95),optimizer=adam,metrics=[dice_coef], loss_weights=[1.0, 0.8, 0.6, 0.4, 0.2])
    
    #model.compile(loss='binary_crossentropy',optimizer=adam,metrics=[dice_coef], loss_weights=[1.0, 0.7,0.5,0.3,0.1])
    
    saveModel(model, modelName, fold)

    saveModel(model, modelName, fold)
    fp = open('models/'+str(fold)+'/'+modelName+'/log.txt','w')
    fp.close()
    fp = open('models/'+str(fold)+'/'+modelName+'/best.txt','w')
    fp.write('-1.0')
    fp.close()
    return model



def evaluateFold(model,modelName,fold,X,Y,batchSize,epo):

    try:
        os.makedirs('results')
    except:
        pass 
    
    try:
        os.makedirs('results/'+str(fold))
    except:
        pass

    try:
        os.makedirs('results/'+str(fold)+'/'+str(modelName))
    except:
        pass

    print("EPO : ", epo)

    yp1 = model.predict(x=X, batch_size=batchSize, verbose=1)
    yp1 = np.array(yp1[0])
    print("shapw of yp1!!!!!: ",yp1.shape)
    print("\n\n\n\n\n")
    print("Predict max: ", np.amax(np.array(yp1[:, :, :, 0])))
    print("\n\n\n\n\n")
    
    yp = yp1
    print(yp.shape)
    yp = np.round(yp,0)
    Y = np.array(Y)
    Y = np.round(Y,0)

    '''
    if epo==5:
        for i in range(len(yp)):

            plt.figure(figsize=(20,10))
            plt.subplot(1,3,1)
            plt.imshow(X[i][:,:,1].reshape(128,128))
            plt.title('Input')
            plt.subplot(1,3,2)
            plt.imshow(Y[i].reshape(Y[i].shape[0],Y[i].shape[1]))
            plt.title('Ground Truth')
            plt.subplot(1,3,3)
            plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
            plt.title('Prediction')

            intersection = yp[i].ravel() * Y[i].ravel()
            union = yp[i].ravel() + Y[i].ravel() - intersection

            #print(set((yp[i].ravel())), set((Y[i].ravel())),set(union),set(intersection))

            jacard = (np.sum(intersection)/np.sum(union))  
            plt.suptitle('Jacard '+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard))
            #plt.show()
            plt.savefig('results/'+str(fold)+'/'+modelName+'/'+str(i)+'.png',format='png') 
            plt.close()
    '''


    X = None 

    jacard = 0
    dice = 0
    
    yp_2arr = []
    y2_arr = []

    cnt = 0

    for i in range(len(Y)):
        yp_2 = yp[i].ravel()
        y2 = Y[i].ravel()
        intersection = yp_2 * y2
        #union = yp_2 + y2 - intersection

        #print(np.sum(intersection),np.sum(union), jacard)
        #input('')
        #jacard += (np.sum(intersection)/np.sum(union))  
        #print(np.sum(yp_2))
        #print(np.sum(y2))
        if(np.sum(yp_2)==0 and np.sum(y2)==0):
            #pass
            dice += 1.0
            cnt+=1
        else:
            dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))
            cnt+=1

        #yp_2arr.extend(yp_2)
        #y2_arr.extend(y2)

   
    dice /= cnt
    #clf_rprt = classification_report(y2_arr,yp_2arr)


    return [0 , dice, 0]


def evaluateModel(model,modelName,fold,X_test, Y_test,batchSize,epo):

    value = evaluateFold(model,modelName,fold,X_test, Y_test,batchSize,epo)
    
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
        fp = open('models/'+str(fold)+'/'+modelName+'/best.txt','w')
        fp.write(str(value[1]))
        fp.close()

        saveModel(model,modelName,fold,'best')

    print(datetime.datetime.time(datetime.datetime.now()))


def trainModel(model,modelName,fold,batchSize,epochss): 

    model = trainFold(model,modelName,fold,batchSize,epochss)

    saveModel(model, modelName,fold)

    return model

def trainFold(model,modelName,fold,batchSize,epochss):

    hfx = h5py.File('tempData/fold_X'+str(fold)+'.h5', 'r')
    hfy = h5py.File('tempData/fold_Y'+str(fold)+'.h5', 'r')

    X = hfx.get('X')
    Y = hfy.get('Y')

    X = np.array(X)
    Y = np.array(Y)
    print("\n\n\n\n\n")
    print("Image Sample: ", np.amax(np.array(X[0, :, :, 0])), np.amax(np.array(Y[0, :, :, 0])))
    print("\n\n\n\n\n")
    
    X = X*255
    X = X.astype(np.uint8)
    Y = Y*255
    Y = Y.astype(np.uint8)

    print("Shape of x: ",X.shape)
    print("Dtype of x: ",X.dtype)

    Ylv1 = []
    Ylv2 = []
    Ylv3 = []
    Ylv4 = []
    print("length of x is: ",len(X))

    for i in range(len(X)):
        #temp1 = cv2.resize(X[i] ,(int(X[i].shape[1]/2),int(X[i].shape[2]/2)), interpolation=cv2.INTER_CUBIC)
        #Xlv1.append(temp1)
        temp1 = cv2.resize(Y[i] ,(int(Y.shape[1]/2),int(Y.shape[2]/2)), interpolation=cv2.INTER_CUBIC)
        Ylv1.append(temp1)
        #Xlv2 = cv2.resize(Xlv1 ,(Xlv1.shape[1]/2,Xlv1.shape[2]/2), interpolation=cv2.INTER_CUBIC)
        temp2 = cv2.resize(Ylv1[i] ,(int(Y.shape[1]/4),int(Y.shape[2]/4)), interpolation=cv2.INTER_CUBIC)
        Ylv2.append(temp2)
        #Xlv3 = cv2.resize(Xlv2 ,(Xlv2.shape[1]/2,Xlv2.shape[2]/2), interpolation=cv2.INTER_CUBIC)
        temp3 = cv2.resize(Ylv2[i] ,(int(Y.shape[1]/8),int(Y.shape[2]/8)), interpolation=cv2.INTER_CUBIC)
        Ylv3.append(temp3)
        #Xlv4 = cv2.resize(Xlv3 ,(Xlv3.shape[1]/2,Xlv3.shape[2]/2), interpolation=cv2.INTER_CUBIC)
        temp4 = cv2.resize(Ylv3[i] ,(int(Y.shape[1]/16),int(Y.shape[2]/16)), interpolation=cv2.INTER_CUBIC)
        Ylv4.append(temp4)


    Ylv1 = np.array(Ylv1)
    Ylv2 = np.array(Ylv2)
    Ylv3 = np.array(Ylv3)
    Ylv4 = np.array(Ylv4)
    print("shape of ylevel1: ",Ylv1.shape)
    #Xlv1 = cv2.resize(X ,(X.shape[1]/2,X.shape[2]/2), interpolation=cv2.INTER_CUBIC)

    print("omg it works!!!!!")
    
    X = X/255
    #Xlv1 = Xlv1/255
    #Xlv2 = Xlv2/255
    #Xlv3 = Xlv3/255
    #Xlv4 = Xlv4/255

    Y = Y/255
    Y = np.round(Y,0)
    Ylv1 = Ylv1/255
    Ylv1 = np.round(Ylv1,0)
    Ylv2 = Ylv2/255
    Ylv2 = np.round(Ylv2,0)
    Ylv3 = Ylv3/255
    Ylv3 = np.round(Ylv3,0)
    Ylv4 = Ylv4/255
    Ylv4 = np.round(Ylv4,0)

    hfx.close()
    hfy.close()

    hfx = None
    hfy = None


    hfx = h5py.File('tempData/X'+str(fold)+'.h5', 'r')
    hfy = h5py.File('tempData/Y'+str(fold)+'.h5', 'r')

    X_test = hfx.get('X')
    Y_test = hfy.get('Y')

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    '''
    X_testlv1 = cv2.resize(X_test ,(X_test.shape[0]/2,X_test.shape[1]/2), interpolation=cv2.INTER_CUBIC)
    Y_testlv1 = cv2.resize(Y_test ,(Y_test.shape[0]/2,Y_test.shape[1]/2), interpolation=cv2.INTER_CUBIC)

    X_testlv2 = cv2.resize(X_testlv1 ,(X_testlv1.shape[0]/2,X_testlv1.shape[1]/2), interpolation=cv2.INTER_CUBIC)
    Y_testlv2 = cv2.resize(Y_testlv1 ,(Y_testlv1.shape[0]/2,Y_testlv1.shape[1]/2), interpolation=cv2.INTER_CUBIC)

    X_testlv3 = cv2.resize(X_testlv2 ,(X_testlv2.shape[0]/2,X_testlv2.shape[1]/2), interpolation=cv2.INTER_CUBIC)
    Y_testlv3 = cv2.resize(Y_testlv2 ,(Y_testlv2.shape[0]/2,Y_testlv2.shape[1]/2), interpolation=cv2.INTER_CUBIC)

    X_testlv4 = cv2.resize(X_testlv3 ,(X_testlv3.shape[0]/2,X_testlv3.shape[1]/2), interpolation=cv2.INTER_CUBIC)
    Y_testlv4 = cv2.resize(Y_testlv3 ,(Y_testlv3.shape[0]/2,Y_testlv3.shape[1]/2), interpolation=cv2.INTER_CUBIC)
    '''
    print("\n\n\n\n\n")
    print("Image Sample test: ", np.amax(np.array(X_test[0, :, :, 0])), np.amax(np.array(Y_test[0, :, :, 0])))
    print("\n\n\n\n\n")
    

    hfx.close()
    hfy.close()

    hfx = None
    hfy = None

    for epoch in range(1):

        for epo in (range(epochss)):
            
            print(epo)

            #model.fit(x=X, y=Y, batch_size=batchSize, epochs=1, verbose=1)     
            history1 = model.fit(X,{'out': Y, 'level1': Ylv1.reshape(Ylv1.shape[0], Ylv1.shape[1], Ylv1.shape[2], 1), 'level2':Ylv2.reshape(Ylv2.shape[0], Ylv2.shape[1], Ylv2.shape[2], 1), 'level3':Ylv3.reshape(Ylv3.shape[0], Ylv3.shape[1], Ylv3.shape[2], 1) , 'level4':Ylv4.reshape(Ylv4.shape[0], Ylv4.shape[1], Ylv4.shape[2], 1)},epochs=1,batch_size=batchSize, verbose=1)
            #fip = open("logtrain.txt","a+")
            #fip.write("\n" + str(history1))
            #fip.close()
   
            evaluateModel(model,modelName,fold,X_test, Y_test,batchSize, epo)


    return model 





def main():



    batchSize =25
    nDivisions = 1
    modelName = 'DeepSupervisionModel'

    
    for fold in range(3,6):

        np.random.seed(fold)
        set_random_seed(fold)

        model = createModel(fold,modelName)
        #model = createModel(fold,modelName)
        model = trainModel(model,modelName,fold,batchSize=batchSize,epochss=150)
        model = None
        time.sleep(30)
        
if __name__=='__main__':
    main()