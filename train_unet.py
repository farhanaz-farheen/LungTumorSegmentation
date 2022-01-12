import numpy as np
import matplotlib.pyplot as plt 
import h5py
import datetime

import LungNet as LN
import os 
from sklearn.metrics import classification_report

try:
    os.makedirs('results/')
except:
    pass

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


def evaluateModel(model,modelName,fold,X_test, Y_test,batchSize):

    value = evaluateFold(model,modelName,fold,X_test, Y_test,batchSize)
    
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


def evaluateFold(model,modelName,fold,X,Y,batchSize):

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


    yp = model.predict(x=X, batch_size=batchSize, verbose=1)

    yp = np.round(yp,0)

    for i in range(0):

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

        plt.savefig('results/'+str(fold)+'/'+modelName+'/'+str(i)+'.png',format='png')
        plt.close()

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



def predictFold(model,fold,modelName,cnt=10,batchSize=16):

    fp = open('data/'+str(fold)+'.txt','r')

    ids = fp.read().split('.jpg\n')[:-1]

    #print(ids)
    #print(len(ids))

    imagePath = '../ISIC_2017_smallData'
    maskPath = '../ISIC_2017_smallMask'

    st = np.random.randint(0,len(ids)-1-cnt)

    ids = ids[st:st+cnt]

    (X,Y) = loadData(imagePath, maskPath, ids)



    yP = model.predict(x=X, batch_size=batchSize, verbose=1)

    for i in range(X.shape[0]):

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X[i])
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y[i].reshape(Y[i].shape[0],Y[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yP[i].reshape(yP[i].shape[0],yP[i].shape[1]))
        plt.title('Prediction')

        plt.savefig('../results/'+modelName+str(i)+'.png',format='png')



import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import h5py



import numpy as np
import matplotlib.pyplot as plt 
import h5py
import datetime
import os 
from sklearn.metrics import classification_report

try:
    os.makedirs('results/')
except:
    pass

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


def evaluateModel(model,modelName,fold,X_test, Y_test,batchSize):

    value = evaluateFold(model,modelName,fold,X_test, Y_test,batchSize)
    
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



def trainFold(model,modelName,fold,batchSize,epochss):

    hfx = h5py.File('tempData/fold_X'+str(fold)+'.h5', 'r')
    hfy = h5py.File('tempData/fold_Y'+str(fold)+'.h5', 'r')

    X = hfx.get('X')
    Y = hfy.get('Y')

    X = np.array(X)
    Y = np.array(Y)

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

    hfx.close()
    hfy.close()

    hfx = None
    hfy = None

    sample_weights = np.zeros((128*128, 4))
    sample_weights[:, 0] += 1
    sample_weights[:, 1] += 100

    for epoch in range(1):

        for epo in (range(epochss)):
            
            print(epo)

            model.fit(x=X, y=Y, batch_size=batchSize, epochs=1, verbose=1)     

            evaluateModel(model,modelName,fold,X_test, Y_test,batchSize)


    return model 


def trainAllButFold(model, modelName, fold, batchSize):


    for i in tqdm(range(1,11)):

        if(i==fold):
            continue

        else:
            model = trainFold(model,modelName,i,fold,batchSize)
            

    return model 


from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Flatten, Reshape, BatchNormalization
from keras.models import Model, model_from_json
from keras.optimizers import Adam, SGD
import pickle
import numpy as np 
import os 
import time
from tensorflow import set_random_seed

from keras import backend as K 

def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def jacard_loss ( y_true, y_pred):
    return - jacard(y_true,y_pred)

def UNet(height,width):

    inputs = Input((height, width, 5))
    inputs_norm = BatchNormalization(axis=3, scale=False)(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs_norm)
    conv1 = BatchNormalization(axis=3, scale=False)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis=3, scale=False)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BatchNormalization(axis=3, scale=False)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization(axis=3, scale=False)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=3, scale=False)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization(axis=3, scale=False)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization(axis=3, scale=False)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=3, scale=False)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization(axis=3, scale=False)(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization(axis=3, scale=False)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis=3, scale=False)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization(axis=3, scale=False)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization(axis=3, scale=False)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=3, scale=False)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    up6 = BatchNormalization(axis=3, scale=False)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization(axis=3, scale=False)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=3, scale=False)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    up7 = BatchNormalization(axis=3, scale=False)(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization(axis=3, scale=False)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=3, scale=False)(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    up8 = BatchNormalization(axis=3, scale=False)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization(axis=3, scale=False)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=3, scale=False)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    up9 = BatchNormalization(axis=3, scale=False)(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization(axis=3, scale=False)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization(axis=3, scale=False)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    

    model = Model(inputs=[inputs], outputs=[conv10])

    

    return model

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

def loadModel(fold, modelName):
    
    json_file = open('models/'+str(fold)+'/'+modelName+'/'+'modelP.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    model = model_from_json(loaded_model_json)
    
    model.load_weights('models/'+str(fold)+'/'+modelName+'/'+'modelW.h5')

    sgd = SGD(lr=0.005, momentum=0.8, decay=0.01/50, nesterov=True)
    
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])

    return model

def createModel(fold, modelName):

    model = UNet(height=128, width=128)
    #sgd = SGD(lr=0.5, momentum=0.8, decay=0.5/150, nesterov=True)
    #test a
    #sgd = SGD(lr=0.01, momentum=0.5, decay=0.01/150, nesterov=True)
    #test b
    #sgd = SGD(lr=0.01, momentum=0.5, decay=0.1/150, nesterov=True)
    #test c
    sgd = SGD(lr=0.1, momentum=0.5, decay=0.1/150, nesterov=True)
    
    #test 1
    #adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1/150, amsgrad=False)
    #test 2 - training best
    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01/150, amsgrad=False)
    #test 2
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001/150, amsgrad=False)
    
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])
    saveModel(model, modelName, fold)

    saveModel(model, modelName, fold)
    fp = open('models/'+str(fold)+'/'+modelName+'/log.txt','w')
    fp.close()
    fp = open('models/'+str(fold)+'/'+modelName+'/best.txt','w')
    fp.write('-1.0')
    fp.close()
    return model

def trainModel(model,modelName,fold,batchSize,epochss): 

    model = trainFold(model,modelName,fold,batchSize,epochss)

    saveModel(model, modelName,fold)

    return model

def main():

    batchSize =25
    nDivisions = 1
    modelName = 'unetWaveletNeighborNorm'

    
    for fold in range(1,6):

        np.random.seed(fold)
        set_random_seed(fold)
        model = createModel(fold,modelName)
        model = trainModel(model,modelName,fold,batchSize=batchSize,epochss=50)
        model = None
        time.sleep(30)

    while(False):
        
        for fold in range(1,6):

            np.random.seed(fold)
            set_random_seed(fold)
            model = loadModel(fold,modelName)
            model = trainModel(model,modelName,fold,batchSize=batchSize,epochss=50)
            model = None
            time.sleep(300)

        time.sleep(300)

if __name__=='__main__':
    main()