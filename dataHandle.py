import os 
from sklearn.model_selection import KFold
import cv2
import pickle
import numpy as np 
import gc
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt 
import h5py
from PIL import Image
from shutil import copyfile


def packData():

	for fold in tqdm(range(1,6)):

		X = []
		Y = []

		for i in tqdm(range(1,6)):

			if(fold==i):
				continue

			else:
				hfx = h5py.File('tempData/X'+str(i)+'.h5', 'r')
				hfy = h5py.File('tempData/Y'+str(i)+'.h5', 'r')

				x = hfx.get('X')
				y = hfy.get('Y')

				x = np.array(x)
				y = np.array(y)

				hfx.close()
				hfy.close()

				for x_ in x:

					X.append(x_)

				for y_ in y:

					Y.append(y_)

		x = None
		y = None
		X = np.array(X)
		Y = np.array(Y)

		hfx = h5py.File('tempData/fold_X'+str(fold)+'.h5', 'w')
		hfy = h5py.File('tempData/fold_Y'+str(fold)+'.h5', 'w')

		hfx.create_dataset('X', data=X)
		hfy.create_dataset('Y', data=Y)

		hfx.close()
		hfy.close()

def foldData(wav_path, nei_path,maskPath,width=128, height=128):

	try:
		os.makedirs('tempData')
	except:
		pass
	
	for fold in tqdm(range(1,6)):

		fp = open('data/'+str(fold)+'.txt','r')

		ids = fp.read().split('\n')[:-1] 
		

		X = []
		Y = []

		for imgId in tqdm(ids):
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

			if(np.random.rand()<0.5):

				direc = np.random.choice([0,1,-1])
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

			else:			

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

		X = np.array(X)
		Y = np.array(Y)

		tqdm.write(str(len(X))+' '+str(len(Y)) ) 

		hfx = h5py.File('tempData/X'+str(fold)+'.h5', 'w')
		hfy = h5py.File('tempData/Y'+str(fold)+'.h5', 'w')

		hfx.create_dataset('X', data=X)
		hfy.create_dataset('Y', data=Y)

		hfx.close()
		hfy.close()

	
def partitionDataset(path):

	try:

		os.mkdir('data')

	except:

		pass 

	files = next(os.walk(path))[2]

	fileFolds = KFold(n_splits=5,shuffle=True,random_state=3)

	dirName = 1

	for i in fileFolds.split(files):
		
		fileNames = ''

		for index in i[1]:

			fileNames += files[index] + '\n'

		fp = open('data/'+str(dirName)+'.txt','w')
		fp.write(fileNames)
		fp.close()

		dirName += 1

def rename_data_files():

	imgFolders = next(os.walk('./vip_cup_data/images'))[1]

	for imgFolder in imgFolders:

		imgFiles = next(os.walk(os.path.join('vip_cup_data','images',imgFolder)))[2]
		

		for imgFile in imgFiles :

			copyfile(os.path.join('vip_cup_data','images',imgFolder,imgFile), os.path.join('training_data','images',imgFolder+'_'+imgFile))

	imgFolders = next(os.walk('./vip_cup_data/masks'))[1]

	for imgFolder in imgFolders:

		imgFiles = next(os.walk(os.path.join('vip_cup_data','masks',imgFolder)))[2]
		

		for imgFile in imgFiles :

			copyfile(os.path.join('vip_cup_data','masks',imgFolder,imgFile), os.path.join('training_data','masks',imgFolder+'_'+imgFile))

		

def main():
	
	np.random.seed(3)

	wav_path = './filtered_wavelets/images'
	nei_path = './filtered_neighbors/images'
	mask_path = './filtered_wavelets/masks'
	
	
	partitionDataset(wav_path)
	foldData(wav_path, nei_path, mask_path)
	packData()
 

if __name__ == '__main__':
	main()







