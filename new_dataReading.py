import pydicom
from dicom_contour import contour
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt 
from math import ceil
import shutil
import numpy as np
import pickle
from tqdm import tqdm
import cv2

TUMOR_MIN = -1020
TUMOR_MAX = 3500


globalminSIEMENS = 99999
globalmaxSIEMENS = -99999

globalminCMS = 99999
globalmaxCMS = -99999


def readAll(path):

    global globalmaxSIEMENS
    global globalminSIEMENS

    global globalmaxCMS
    global globalminCMS

    folders = (next(os.walk(path)))[1]
    folders.sort()
    

    for folder in tqdm(folders,total=len(folders)):
        tqdm.write(folder)

        failed = [139,212]   ,# 252,281,284, 295,296,297,298,299,300,303,305,306,307,308,309,311]

        '''if(int(folder[-3:]) <=212):
            continue
        '''

        #first step get globals first below

        
        if(int(folder[-3:])==139 or int(folder[-3:])==212):
            continue

        elif(int(folder[-3:]) in [43,45]):
            getglobal_type1_New(os.path.join(path,folder),folder)

        elif(int(folder[-3:])<=80):
            
            getglobal_type1(os.path.join(path,folder),folder)

        elif(int(folder[-3:])>81):

            getglobal_type2(os.path.join(path,folder),folder)

    print("globalmaxSIEMENS - ",globalmaxSIEMENS)
    print("globalmaxCMS - ",globalmaxCMS)
    print("globalminSIEMENS - ",globalminSIEMENS)
    print("globalminCMS - ",globalminCMS)


       
        
    for folder in tqdm(folders,total=len(folders)):
        tqdm.write(folder)

        failed = [139,212]   ,# 252,281,284, 295,296,297,298,299,300,303,305,306,307,308,309,311]

        '''if(int(folder[-3:]) <=212):
            continue
        '''

        #first step get globals first below




        #second step - savefiles here below

        
        if(int(folder[-3:])==139 or int(folder[-3:])==212):
            continue

        elif(int(folder[-3:]) in [43,45]):
            readDir_type1_New(os.path.join(path,folder),folder)

        elif(int(folder[-3:])<=80):
            
            readDir_type1(os.path.join(path,folder),folder)

        elif(int(folder[-3:])>81):

            readDir_type2(os.path.join(path,folder),folder)

       





def getglobal_type2(path,folderName):
    global globalmaxSIEMENS
    global globalminSIEMENS

    global globalmaxCMS
    global globalminCMS

    def readImages(dcmPath):

        dcmMapper = {}

        for dcmFile in tqdm(next(os.walk(dcmPath))[2]): 

            dcm = pydicom.dcmread(os.path.join(dcmPath,dcmFile))

            try:
                z_val = dcm.ImagePositionPatient[2]
                dcmMapper[z_val] = dcmFile
            except:
                pass

        return dcmMapper    

    def readAnnotation(path):


        annotation_mapper = {}
        annotation_mapper_r = {}

        d = pydicom.dcmread(path)

        dr =  d.ROIContourSequence

        index = None 

        for i in range(len(d.StructureSetROISequence)):

            if(d.StructureSetROISequence[i].ROIName=="GTV-1" or d.StructureSetROISequence[i].ROIName=="GTV1" or 
                d.StructureSetROISequence[i].ROIName=="gtv-1" or d.StructureSetROISequence[i].ROIName=="gtv1" ):

                index = i

        drc = dr[index].ContourSequence

        for i in range(len(drc)):

            drc_ = drc[i]
            z_value = drc_.ContourData[2]

            drcc_ = drc_.ContourImageSequence
            SOP_ID = drcc_[0].ReferencedSOPInstanceUID

            annotation_mapper[z_value] = SOP_ID
            annotation_mapper_r[SOP_ID] = z_value

            

        return annotation_mapper,annotation_mapper_r, index
        

    def renameDcmFiles(path, rename_mapper):
        
        
        for dcmFile in tqdm(rename_mapper): 

            dcm = pydicom.dcmread(os.path.join(path,dcmFile))

            #for i in dir(dcm):
                #print(i)
            #print(dcm.SOPClassUID)
            
            #print(dcm.PixelSpacing)
         

            try:
                shutil.copyfile(os.path.join(path,dcmFile), os.path.join(path, rename_mapper[dcmFile] +'.dcm'))
            except Exception as e:
                pass
                    #print('**************************************')
                    #print(e)
                    #print('**************************************')
                

    '''
    try:
        os.makedirs(os.path.join('training_data_new','images',folderName))        
    except:
        pass

    try:
        os.makedirs(os.path.join('training_data_new','masks',folderName))
    except:
        pass

    try:
        os.makedirs(os.path.join('processed',folderName))
    except:
        pass
    '''


    X = []
    Y = []

    twoFolders = next(os.walk(path))[1]

    dir1 = next(os.walk( os.path.join(path,twoFolders[0])))[1]
    dir2 = next(os.walk( os.path.join(path,twoFolders[1])))[1]

    if(dir1[0]=='DICOM'):
        contourFile = os.path.join(path,twoFolders[1], dir2[0], next(os.walk( os.path.join(path,twoFolders[1], dir2[0])))[2][0])
        dcmPath = os.path.join(path,twoFolders[0], dir1[0])

    else:
        contourFile = os.path.join(path,twoFolders[0], dir1[0], next(os.walk( os.path.join(path,twoFolders[0], dir1[0])))[2][0])
        dcmPath = os.path.join(path,twoFolders[1], dir2[0])

    shutil.copyfile(contourFile, os.path.join(dcmPath ,'contour_copy.dcm'))

    dcmMapper = readImages(dcmPath)
    annotation_mapper, annotation_mapper_r, index = readAnnotation(contourFile)

    sorted_keys = {}

    z_vals = []

    for z_val in annotation_mapper:

        z_vals.append(float(z_val))

    z_vals.sort()


    for z_i in range(len(z_vals)):

        sorted_keys[str(z_vals[z_i])] = z_i

    '''print(dcmMapper)
    print('---------------------------------------------')
    print(annotation_mapper)
    print('---------------------------------------------')
    print(annotation_mapper_r)
    print('---------------------------------------------')
    print(sorted_keys)
    print('---------------------------------------------')'''

    rename_mapper = {} 

    for z_val in annotation_mapper:
        
        try:
            rename_mapper[dcmMapper[z_val]] = annotation_mapper[z_val]
        except:
            pass


    renameDcmFiles(dcmPath, rename_mapper)
    #readAnnotation(newPath+'/'+annotationFolder+'/')
    #print(contour.get_contour_file(newPath+'/'+dcmFolder+'/'))


    contour_data = pydicom.dcmread(os.path.join(dcmPath,'contour_copy.dcm'))
    
    contour_arrays = contour.cfile2pixels(file='contour_copy.dcm', path=dcmPath, ROIContourSeq=index)

    for i in tqdm(range(len(contour_arrays))):
        first_image, first_contour, img_id = contour_arrays[i]

        first_contour = contour.fill_contour(first_contour)



        minn = np.min(first_image)
        #first_image = first_image - minn 
        maxx = np.max(first_image)
        #first_image = first_image / maxx


        pFile = pydicom.dcmread(dcmPath + '/' + img_id +'.dcm')
        device = pFile.Manufacturer

        if device == 'SIEMENS':
            if minn < globalminSIEMENS:
                globalminSIEMENS = minn
            if maxx > globalmaxSIEMENS:
                globalmaxSIEMENS = maxx
        else:
            if minn < globalminCMS:
                globalminCMS = minn
            if maxx > globalmaxCMS:
                globalmaxCMS = maxx

        
        #first_image *= 255
        #first_image = first_image.astype(np.uint8)

        #first_contour *= 255
        #first_contour = first_contour.astype(np.uint8)

        #plt.subplot(1,3,1)
        #plt.imshow(first_image,cmap='gray')
        #plt.title('Slice')
        #plt.subplot(1,3,2)
        #plt.imshow(first_contour,cmap='gray')
        #plt.title('Tumor Mask')
        #plt.xlabel(path)
        #plt.subplot(1,3,3)
        #plt.imshow(first_contour*first_image,cmap='gray')
        #plt.title('Segmented Tumor')
        #plt.show()

      
    
        #cv2.imwrite(os.path.join('training_data_new','images',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_image)
        #cv2.imwrite(os.path.join('training_data_new','masks',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_contour)

            




        

def getglobal_type1_New(path,folderName):
    global globalmaxSIEMENS
    global globalminSIEMENS

    global globalmaxCMS
    global globalminCMS

    def readAnnotation(path):

        annotation_mapper = {}
        annotation_mapper_r = {}
    
        annotationFile = next(os.walk(path))[2][0]

        d = pydicom.dcmread(path+annotationFile)
        
        dr =  d.ROIContourSequence

        index = 0

        drc = dr[index].ContourSequence

        for i in range(len(drc)):

            drc_ = drc[i]
            z_value = drc_.ContourData[2]

            drcc_ = drc_.ContourImageSequence
            SOP_ID = drcc_[0].ReferencedSOPInstanceUID

            annotation_mapper[z_value] = SOP_ID
            annotation_mapper_r[SOP_ID] = z_value

        return annotation_mapper,annotation_mapper_r
    
    def renameDcmFiles(path):

        dcmMapper = {}
        
        for dcmFile in tqdm(next(os.walk(path))[2]): 

            dcm = pydicom.dcmread(os.path.join(path,dcmFile))

            #for i in dir(dcm):
                #print(i)
            #print(dcm.SOPClassUID)

            try:
                z_val = dcm.ImagePositionPatient[2]
                dcmMapper[z_val] = dcmFile
            except:
                pass
            
            #print(dcm.PixelSpacing)
            try:
                uid = dcm.SOPInstanceUID
                img = dcm.pixel_array


                try:
                    shutil.copyfile(os.path.join(path,dcmFile), os.path.join(path, dcm.SOPInstanceUID +'.dcm'))
                except Exception as e:
                    pass

            except:
                pass


        return dcmMapper
    '''
    try:
        os.makedirs(os.path.join('training_data_new','images',folderName))        
    except:
        pass

    try:
        os.makedirs(os.path.join('training_data_new','masks',folderName))
    except:
        pass
    '''


    X = []
    Y = []

    nextStep = next(os.walk(path))[1]

    newPath = os.path.join(path , nextStep[0])
    
    folders = next(os.walk(newPath))[1]

    if(len(next(os.walk(os.path.join(newPath,folders[0])))[2])==1):

        annotationFolder = folders[0]
        dcmFolder = folders[1]

    else: 

        annotationFolder = folders[1]
        dcmFolder = folders[0]

    contourFile = os.path.join(newPath,annotationFolder, list(next(os.walk(newPath+'/'+annotationFolder+'/'))[2])[0])
    dcmPath = os.path.join(newPath,dcmFolder)


    shutil.copyfile(contourFile, os.path.join(newPath,dcmFolder ,'contour_copy.dcm'))
    dcmMapper = renameDcmFiles(os.path.join(newPath,dcmFolder))
    annotation_mapper, annotation_mapper_r = readAnnotation(newPath+'/'+annotationFolder+'/')
    #print(contour.get_contour_file(newPath+'/'+dcmFolder+'/'))

    sorted_keys = {}

    z_vals = []

    for z_val in annotation_mapper:

        z_vals.append(float(z_val))

    z_vals.sort()


    for z_i in range(len(z_vals)):

        sorted_keys[str(z_vals[z_i])] = z_i

    '''print(dcmMapper)
    print('---------------------------------------------')
    print(annotation_mapper)
    print('---------------------------------------------')
    print(annotation_mapper_r)
    print('---------------------------------------------')
    print(sorted_keys)
    print('---------------------------------------------')'''



    contour_data = pydicom.dcmread(os.path.join(newPath,dcmFolder,'contour_copy.dcm'))
    
    contour_arrays = contour.cfile2pixels(file='contour_copy.dcm', path=dcmPath, ROIContourSeq=0)

    for i in tqdm(range(len(contour_arrays))):
        first_image, first_contour, img_id = contour_arrays[i]

        #plt.subplot(1,2,1)
        #plt.imshow(first_image)

        first_contour = contour.fill_contour(first_contour)

        #plt.subplot(1,2,2)
        #plt.imshow(first_image)
        #plt.show()

        minn = np.min(first_image)
        #first_image = first_image - minn 
        maxx = np.max(first_image)
        #first_image = first_image / maxx



        pFile = pydicom.dcmread(dcmPath + '/' + img_id + '.dcm')
        device = pFile.Manufacturer

        if device == 'SIEMENS':
            if minn < globalminSIEMENS:
                globalminSIEMENS = minn
            if maxx > globalmaxSIEMENS:
                globalmaxSIEMENS = maxx
        else:
            if minn < globalminCMS:
                globalminCMS = minn
            if maxx > globalmaxCMS:
                globalmaxCMS = maxx

        #first_image *= 255
        #first_image = first_image.astype(np.uint8)

        #first_contour *= 255
        #first_contour = first_contour.astype(np.uint8)

        #plt.subplot(1,3,1)
        #plt.imshow(first_image,cmap='gray')
        #plt.title('Slice')
        #plt.subplot(1,3,2)
        #plt.imshow(first_contour,cmap='gray')
        #plt.title('Tumor Mask')
        #plt.xlabel(path)
        #plt.subplot(1,3,3)
        #plt.imshow(first_contour*first_image,cmap='gray')
        #plt.title('Segmented Tumor')
        #plt.show()
      

        #cv2.imwrite(os.path.join('training_data','images',folderName, img_id +'.png'), first_image)
        #cv2.imwrite(os.path.join('training_data','masks',folderName, img_id +'.png'), first_contour)

        #cv2.imwrite(os.path.join('training_data_new','images',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_image)
        #cv2.imwrite(os.path.join('training_data_new','masks',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_contour)

            





       

def getglobal_type1(path,folderName):
    global globalmaxSIEMENS
    global globalminSIEMENS

    global globalmaxCMS
    global globalminCMS

    def readImages(dcmPath):

        dcmMapper = {}

        for dcmFile in tqdm(next(os.walk(dcmPath))[2]): 

            dcm = pydicom.dcmread(os.path.join(dcmPath,dcmFile))

            try:
                z_val = dcm.ImagePositionPatient[2]
                dcmMapper[z_val] = dcmFile
            except:
                pass

        return dcmMapper    

    def readAnnotation(path):


        annotation_mapper = {}
        annotation_mapper_r = {}

        d = pydicom.dcmread(path)

        dr =  d.ROIContourSequence

        index = None 

        for i in range(len(d.StructureSetROISequence)):

            if(d.StructureSetROISequence[i].ROIName=="GTV-1" or d.StructureSetROISequence[i].ROIName=="GTV1" or 
                d.StructureSetROISequence[i].ROIName=="gtv-1" or d.StructureSetROISequence[i].ROIName=="gtv1" ):

                index = i

        drc = dr[index].ContourSequence

        for i in range(len(drc)):

            drc_ = drc[i]
            z_value = drc_.ContourData[2]

            drcc_ = drc_.ContourImageSequence
            SOP_ID = drcc_[0].ReferencedSOPInstanceUID

            annotation_mapper[z_value] = SOP_ID
            annotation_mapper_r[SOP_ID] = z_value

            

        return annotation_mapper,annotation_mapper_r, index
        

    def renameDcmFiles(path, rename_mapper):
        
        
        for dcmFile in tqdm(rename_mapper): 

            dcm = pydicom.dcmread(os.path.join(path,dcmFile))

            #for i in dir(dcm):
                #print(i)
            #print(dcm.SOPClassUID)
            
            #print(dcm.PixelSpacing)
         

            try:
                shutil.copyfile(os.path.join(path,dcmFile), os.path.join(path, rename_mapper[dcmFile] +'.dcm'))
            except Exception as e:
                pass
                    #print('**************************************')
                    #print(e)
                    #print('**************************************')
                

    '''
    try:
        os.makedirs(os.path.join('training_data_new','images',folderName))        
    except:
        pass

    try:
        os.makedirs(os.path.join('training_data_new','masks',folderName))
    except:
        pass

    try:
        os.makedirs(os.path.join('processed',folderName))
    except:
        pass
    '''


    X = []
    Y = []



    nextStep = next(os.walk(path))[1]

    newPath = os.path.join(path , nextStep[0])
    
    folders = next(os.walk(newPath))[1]

    if(len(next(os.walk(os.path.join(newPath,folders[0])))[2])==1):

        annotationFolder = folders[0]
        dcmFolder = folders[1]

    else: 

        annotationFolder = folders[1]
        dcmFolder = folders[0]

    contourFile = os.path.join(newPath,annotationFolder, list(next(os.walk(newPath+'/'+annotationFolder+'/'))[2])[0])
    dcmPath = os.path.join(newPath,dcmFolder)

    '''
    oneFolder = next(os.walk(path))[1]

    twoFolders = next(os.walk(os.path.join(path, oneFolder[0]))[1]

    dir1 = next(os.walk( os.path.join(path,twoFolders[0])))[1]
    dir2 = next(os.walk( os.path.join(path,twoFolders[1])))[1]

    if(dir1[0]=='DICOM'):
        contourFile = os.path.join(path,twoFolders[1], dir2[0], next(os.walk( os.path.join(path,twoFolders[1], dir2[0])))[2][0])
        dcmPath = os.path.join(path,twoFolders[0], dir1[0])

    else:
        contourFile = os.path.join(path,twoFolders[0], dir1[0], next(os.walk( os.path.join(path,twoFolders[0], dir1[0])))[2][0])
        dcmPath = os.path.join(path,twoFolders[1], dir2[0])
    '''

    shutil.copyfile(contourFile, os.path.join(dcmPath ,'contour_copy.dcm'))

    dcmMapper = readImages(dcmPath)
    annotation_mapper, annotation_mapper_r, index = readAnnotation(contourFile)

    sorted_keys = {}

    z_vals = []

    for z_val in annotation_mapper:

        z_vals.append(float(z_val))

    z_vals.sort()


    for z_i in range(len(z_vals)):

        sorted_keys[str(z_vals[z_i])] = z_i

    '''print(dcmMapper)
    print('---------------------------------------------')
    print(annotation_mapper)
    print('---------------------------------------------')
    print(annotation_mapper_r)
    print('---------------------------------------------')
    print(sorted_keys)
    print('---------------------------------------------')'''

    rename_mapper = {} 

    for z_val in annotation_mapper:
        
        try:
            rename_mapper[dcmMapper[z_val]] = annotation_mapper[z_val]
        except:
            pass


    renameDcmFiles(dcmPath, rename_mapper)
    #readAnnotation(newPath+'/'+annotationFolder+'/')
    #print(contour.get_contour_file(newPath+'/'+dcmFolder+'/'))


    contour_data = pydicom.dcmread(os.path.join(dcmPath,'contour_copy.dcm'))
    
    contour_arrays = contour.cfile2pixels(file='contour_copy.dcm', path=dcmPath, ROIContourSeq=index)

    for i in tqdm(range(len(contour_arrays))):
        first_image, first_contour, img_id = contour_arrays[i]

        first_contour = contour.fill_contour(first_contour)



        minn = np.min(first_image)
        #first_image = first_image - minn 
        maxx = np.max(first_image)
        #first_image = first_image / maxx


        pFile = pydicom.dcmread(dcmPath + '/' + img_id +'.dcm')
        device = pFile.Manufacturer

        if device == 'SIEMENS':
            if minn < globalminSIEMENS:
                globalminSIEMENS = minn
            if maxx > globalmaxSIEMENS:
                globalmaxSIEMENS = maxx
        else:
            if minn < globalminCMS:
                globalminCMS = minn
            if maxx > globalmaxCMS:
                globalmaxCMS = maxx

        
        #first_image *= 255
        #first_image = first_image.astype(np.uint8)

        #first_contour *= 255
        #first_contour = first_contour.astype(np.uint8)

        #plt.subplot(1,3,1)
        #plt.imshow(first_image,cmap='gray')
        #plt.title('Slice')
        #plt.subplot(1,3,2)
        #plt.imshow(first_contour,cmap='gray')
        #plt.title('Tumor Mask')
        #plt.xlabel(path)
        #plt.subplot(1,3,3)
        #plt.imshow(first_contour*first_image,cmap='gray')
        #plt.title('Segmented Tumor')
        #plt.show()

      
    
        #cv2.imwrite(os.path.join('training_data_new','images',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_image)
        #cv2.imwrite(os.path.join('training_data_new','masks',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_contour)

            








def readDir_type1(path,folderName):
    global globalmaxSIEMENS
    global globalminSIEMENS

    global globalmaxCMS
    global globalminCMS

    def readImages(dcmPath):

        dcmMapper = {}

        for dcmFile in tqdm(next(os.walk(dcmPath))[2]): 

            dcm = pydicom.dcmread(os.path.join(dcmPath,dcmFile))

            try:
                z_val = dcm.ImagePositionPatient[2]
                dcmMapper[z_val] = dcmFile
            except:
                pass

        return dcmMapper    

    def readAnnotation(path):


        annotation_mapper = {}
        annotation_mapper_r = {}

        d = pydicom.dcmread(path)

        dr =  d.ROIContourSequence

        index = None 

        for i in range(len(d.StructureSetROISequence)):

            if(d.StructureSetROISequence[i].ROIName=="GTV-1" or d.StructureSetROISequence[i].ROIName=="GTV1" or 
                d.StructureSetROISequence[i].ROIName=="gtv-1" or d.StructureSetROISequence[i].ROIName=="gtv1" ):

                index = i

        drc = dr[index].ContourSequence

        for i in range(len(drc)):

            drc_ = drc[i]
            z_value = drc_.ContourData[2]

            drcc_ = drc_.ContourImageSequence
            SOP_ID = drcc_[0].ReferencedSOPInstanceUID

            annotation_mapper[z_value] = SOP_ID
            annotation_mapper_r[SOP_ID] = z_value

            

        return annotation_mapper,annotation_mapper_r, index
        

    def renameDcmFiles(path, rename_mapper):
        
        
        for dcmFile in tqdm(rename_mapper): 

            dcm = pydicom.dcmread(os.path.join(path,dcmFile))

            #for i in dir(dcm):
                #print(i)
            #print(dcm.SOPClassUID)
            
            #print(dcm.PixelSpacing)
         

            try:
                shutil.copyfile(os.path.join(path,dcmFile), os.path.join(path, rename_mapper[dcmFile] +'.dcm'))
            except Exception as e:
                pass
                    #print('**************************************')
                    #print(e)
                    #print('**************************************')
                


    try:
        os.makedirs(os.path.join('training_data_new','images',folderName))        
    except:
        pass

    try:
        os.makedirs(os.path.join('training_data_new','masks',folderName))
    except:
        pass
    try:
        os.makedirs(os.path.join('processed',folderName))
    except:
        pass


    X = []
    Y = []



    nextStep = next(os.walk(path))[1]

    newPath = os.path.join(path , nextStep[0])
    
    folders = next(os.walk(newPath))[1]

    if(len(next(os.walk(os.path.join(newPath,folders[0])))[2])==1):

        annotationFolder = folders[0]
        dcmFolder = folders[1]

    else: 

        annotationFolder = folders[1]
        dcmFolder = folders[0]

    contourFile = os.path.join(newPath,annotationFolder, list(next(os.walk(newPath+'/'+annotationFolder+'/'))[2])[0])
    dcmPath = os.path.join(newPath,dcmFolder)

    
    '''
    twoFolders = next(os.walk(path))[1]

    dir1 = next(os.walk( os.path.join(path,twoFolders[0])))[1]
    dir2 = next(os.walk( os.path.join(path,twoFolders[1])))[1]

    if(dir1[0]=='DICOM'):
        contourFile = os.path.join(path,twoFolders[1], dir2[0], next(os.walk( os.path.join(path,twoFolders[1], dir2[0])))[2][0])
        dcmPath = os.path.join(path,twoFolders[0], dir1[0])

    else:
        contourFile = os.path.join(path,twoFolders[0], dir1[0], next(os.walk( os.path.join(path,twoFolders[0], dir1[0])))[2][0])
        dcmPath = os.path.join(path,twoFolders[1], dir2[0])
    '''

    shutil.copyfile(contourFile, os.path.join(dcmPath ,'contour_copy.dcm'))

    dcmMapper = readImages(dcmPath)
    annotation_mapper, annotation_mapper_r, index = readAnnotation(contourFile)

    sorted_keys = {}

    z_vals = []

    for z_val in annotation_mapper:

        z_vals.append(float(z_val))

    z_vals.sort()


    for z_i in range(len(z_vals)):

        sorted_keys[str(z_vals[z_i])] = z_i

    '''print(dcmMapper)
    print('---------------------------------------------')
    print(annotation_mapper)
    print('---------------------------------------------')
    print(annotation_mapper_r)
    print('---------------------------------------------')
    print(sorted_keys)
    print('---------------------------------------------')'''

    rename_mapper = {} 

    for z_val in annotation_mapper:
        
        try:
            rename_mapper[dcmMapper[z_val]] = annotation_mapper[z_val]
        except:
            pass


    renameDcmFiles(dcmPath, rename_mapper)
    #readAnnotation(newPath+'/'+annotationFolder+'/')
    #print(contour.get_contour_file(newPath+'/'+dcmFolder+'/'))


    contour_data = pydicom.dcmread(os.path.join(dcmPath,'contour_copy.dcm'))
    
    contour_arrays = contour.cfile2pixels(file='contour_copy.dcm', path=dcmPath, ROIContourSeq=index)

    for i in tqdm(range(len(contour_arrays))):
        first_image, first_contour, img_id = contour_arrays[i]

        first_contour = contour.fill_contour(first_contour)


       

        pFile = pydicom.dcmread(dcmPath + '/' + img_id + '.dcm')
        device = pFile.Manufacturer

        if device == 'SIEMENS':
            minn = globalminSIEMENS
            maxx = globalmaxSIEMENS
        else:
            minn = globalminCMS
            maxx = globalmaxCMS



        #minn = np.min(first_image)
        first_image = first_image - minn 
        #maxx = np.max(first_image)
        first_image = first_image / maxx
        first_image_png = first_image * 255
        first_image_png = first_image_png.astype(np.uint8)

        first_contour_png = first_contour * 255
        first_contour_png = first_contour_png.astype(np.uint8)

        #plt.subplot(1,3,1)
        #plt.imshow(first_image,cmap='gray')
        #plt.title('Slice')
        #plt.subplot(1,3,2)
        #plt.imshow(first_contour,cmap='gray')
        #plt.title('Tumor Mask')
        #plt.xlabel(path)
        #plt.subplot(1,3,3)
        #plt.imshow(first_contour*first_image,cmap='gray')
        #plt.title('Segmented Tumor')
        #plt.show()

      
    
        cv2.imwrite(os.path.join('training_data_new','images',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_image_png)
        cv2.imwrite(os.path.join('training_data_new','masks',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_contour_png)

            
        try:

            pickle.dump(first_image_png,open(os.path.join('processed',folderName,  str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'X.p'),'wb'))
            pickle.dump(first_contour_png, open(os.path.join('processed', folderName ,  str(sorted_keys[str(float(annotation_mapper_r[img_id]))])+ 'Y.p'), 'wb'))

        except:
            pass



def readDir_type1_New(path,folderName):
    global globalmaxSIEMENS
    global globalminSIEMENS

    global globalmaxCMS
    global globalminCMS

    def readAnnotation(path):

        annotation_mapper = {}
        annotation_mapper_r = {}
    
        annotationFile = next(os.walk(path))[2][0]

        d = pydicom.dcmread(path+annotationFile)
        
        dr =  d.ROIContourSequence

        index = 0

        drc = dr[index].ContourSequence

        for i in range(len(drc)):

            drc_ = drc[i]
            z_value = drc_.ContourData[2]

            drcc_ = drc_.ContourImageSequence
            SOP_ID = drcc_[0].ReferencedSOPInstanceUID

            annotation_mapper[z_value] = SOP_ID
            annotation_mapper_r[SOP_ID] = z_value

        return annotation_mapper,annotation_mapper_r
    
    def renameDcmFiles(path):

        dcmMapper = {}
        
        for dcmFile in tqdm(next(os.walk(path))[2]): 

            dcm = pydicom.dcmread(os.path.join(path,dcmFile))

            #for i in dir(dcm):
                #print(i)
            #print(dcm.SOPClassUID)

            try:
                z_val = dcm.ImagePositionPatient[2]
                dcmMapper[z_val] = dcmFile
            except:
                pass
            
            #print(dcm.PixelSpacing)
            try:
                uid = dcm.SOPInstanceUID
                img = dcm.pixel_array


                try:
                    shutil.copyfile(os.path.join(path,dcmFile), os.path.join(path, dcm.SOPInstanceUID +'.dcm'))
                except Exception as e:
                    pass

            except:
                pass


        return dcmMapper

    try:
        os.makedirs(os.path.join('training_data_new','images',folderName))        
    except:
        pass

    try:
        os.makedirs(os.path.join('training_data_new','masks',folderName))
    except:
        pass
    try:
        os.makedirs(os.path.join('processed',folderName))
    except:
        pass


    X = []
    Y = []

    nextStep = next(os.walk(path))[1]

    newPath = os.path.join(path , nextStep[0])
    
    folders = next(os.walk(newPath))[1]

    if(len(next(os.walk(os.path.join(newPath,folders[0])))[2])==1):

        annotationFolder = folders[0]
        dcmFolder = folders[1]

    else: 

        annotationFolder = folders[1]
        dcmFolder = folders[0]

    contourFile = os.path.join(newPath,annotationFolder, list(next(os.walk(newPath+'/'+annotationFolder+'/'))[2])[0])
    dcmPath = os.path.join(newPath,dcmFolder)


    shutil.copyfile(contourFile, os.path.join(newPath,dcmFolder ,'contour_copy.dcm'))
    dcmMapper = renameDcmFiles(os.path.join(newPath,dcmFolder))
    annotation_mapper, annotation_mapper_r = readAnnotation(newPath+'/'+annotationFolder+'/')
    #print(contour.get_contour_file(newPath+'/'+dcmFolder+'/'))

    sorted_keys = {}

    z_vals = []

    for z_val in annotation_mapper:

        z_vals.append(float(z_val))

    z_vals.sort()


    for z_i in range(len(z_vals)):

        sorted_keys[str(z_vals[z_i])] = z_i

    '''print(dcmMapper)
    print('---------------------------------------------')
    print(annotation_mapper)
    print('---------------------------------------------')
    print(annotation_mapper_r)
    print('---------------------------------------------')
    print(sorted_keys)
    print('---------------------------------------------')'''



    contour_data = pydicom.dcmread(os.path.join(newPath,dcmFolder,'contour_copy.dcm'))
    
    contour_arrays = contour.cfile2pixels(file='contour_copy.dcm', path=dcmPath, ROIContourSeq=0)

    for i in tqdm(range(len(contour_arrays))):
        first_image, first_contour, img_id = contour_arrays[i]

        #plt.subplot(1,2,1)
        #plt.imshow(first_image)

        first_contour = contour.fill_contour(first_contour)



        #plt.subplot(1,2,2)
        #plt.imshow(first_image)
        #plt.show()
        


        pFile = pydicom.dcmread(dcmPath + '/' + img_id + '.dcm')
        device = pFile.Manufacturer

        if device == 'SIEMENS':
            minn = globalminSIEMENS
            maxx = globalmaxSIEMENS
        else:
            minn = globalminCMS
            maxx = globalmaxCMS


        #minn = np.min(first_image)
        first_image = first_image - minn 
        #maxx = np.max(first_image)
        first_image = first_image / maxx
        first_image_png = first_image * 255
        first_image_png = first_image_png.astype(np.uint8)

        first_contour_png = first_contour * 255
        first_contour_png = first_contour_png.astype(np.uint8)

        #plt.subplot(1,3,1)
        #plt.imshow(first_image,cmap='gray')
        #plt.title('Slice')
        #plt.subplot(1,3,2)
        #plt.imshow(first_contour,cmap='gray')
        #plt.title('Tumor Mask')
        #plt.xlabel(path)
        #plt.subplot(1,3,3)
        #plt.imshow(first_contour*first_image,cmap='gray')
        #plt.title('Segmented Tumor')
        #plt.show()
      

        #cv2.imwrite(os.path.join('training_data','images',folderName, img_id +'.png'), first_image)
        #cv2.imwrite(os.path.join('training_data','masks',folderName, img_id +'.png'), first_contour)

        cv2.imwrite(os.path.join('training_data_new','images',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_image_png)
        cv2.imwrite(os.path.join('training_data_new','masks',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_contour_png)

        try:

            pickle.dump(first_image_png,open(os.path.join('processed',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))])  +'X.p'),'wb'))
            pickle.dump(first_contour_png, open(os.path.join('processed', folderName , str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) + 'Y.p'), 'wb'))

        except:
            pass
            


def readDir_type2(path,folderName):
    global globalmaxSIEMENS
    global globalminSIEMENS

    global globalmaxCMS
    global globalminCMS

    def readImages(dcmPath):

        dcmMapper = {}

        for dcmFile in tqdm(next(os.walk(dcmPath))[2]): 

            dcm = pydicom.dcmread(os.path.join(dcmPath,dcmFile))

            try:
                z_val = dcm.ImagePositionPatient[2]
                dcmMapper[z_val] = dcmFile
            except:
                pass

        return dcmMapper    

    def readAnnotation(path):


        annotation_mapper = {}
        annotation_mapper_r = {}

        d = pydicom.dcmread(path)

        dr =  d.ROIContourSequence

        index = None 

        for i in range(len(d.StructureSetROISequence)):

            if(d.StructureSetROISequence[i].ROIName=="GTV-1" or d.StructureSetROISequence[i].ROIName=="GTV1" or 
                d.StructureSetROISequence[i].ROIName=="gtv-1" or d.StructureSetROISequence[i].ROIName=="gtv1" ):

                index = i

        drc = dr[index].ContourSequence

        for i in range(len(drc)):

            drc_ = drc[i]
            z_value = drc_.ContourData[2]

            drcc_ = drc_.ContourImageSequence
            SOP_ID = drcc_[0].ReferencedSOPInstanceUID

            annotation_mapper[z_value] = SOP_ID
            annotation_mapper_r[SOP_ID] = z_value

            

        return annotation_mapper,annotation_mapper_r, index
        

    def renameDcmFiles(path, rename_mapper):
        
        
        for dcmFile in tqdm(rename_mapper): 

            dcm = pydicom.dcmread(os.path.join(path,dcmFile))

            #for i in dir(dcm):
                #print(i)
            #print(dcm.SOPClassUID)
            
            #print(dcm.PixelSpacing)
         

            try:
                shutil.copyfile(os.path.join(path,dcmFile), os.path.join(path, rename_mapper[dcmFile] +'.dcm'))
            except Exception as e:
                pass
                    #print('**************************************')
                    #print(e)
                    #print('**************************************')
                


    try:
        os.makedirs(os.path.join('training_data_new','images',folderName))        
    except:
        pass

    try:
        os.makedirs(os.path.join('training_data_new','masks',folderName))
    except:
        pass
    try:
        os.makedirs(os.path.join('processed',folderName))
    except:
        pass


    X = []
    Y = []

    twoFolders = next(os.walk(path))[1]

    dir1 = next(os.walk( os.path.join(path,twoFolders[0])))[1]
    dir2 = next(os.walk( os.path.join(path,twoFolders[1])))[1]

    if(dir1[0]=='DICOM'):
        contourFile = os.path.join(path,twoFolders[1], dir2[0], next(os.walk( os.path.join(path,twoFolders[1], dir2[0])))[2][0])
        dcmPath = os.path.join(path,twoFolders[0], dir1[0])

    else:
        contourFile = os.path.join(path,twoFolders[0], dir1[0], next(os.walk( os.path.join(path,twoFolders[0], dir1[0])))[2][0])
        dcmPath = os.path.join(path,twoFolders[1], dir2[0])

    shutil.copyfile(contourFile, os.path.join(dcmPath ,'contour_copy.dcm'))

    dcmMapper = readImages(dcmPath)
    annotation_mapper, annotation_mapper_r, index = readAnnotation(contourFile)

    sorted_keys = {}

    z_vals = []

    for z_val in annotation_mapper:

        z_vals.append(float(z_val))

    z_vals.sort()


    for z_i in range(len(z_vals)):

        sorted_keys[str(z_vals[z_i])] = z_i

    '''print(dcmMapper)
    print('---------------------------------------------')
    print(annotation_mapper)
    print('---------------------------------------------')
    print(annotation_mapper_r)
    print('---------------------------------------------')
    print(sorted_keys)
    print('---------------------------------------------')'''

    rename_mapper = {} 

    for z_val in annotation_mapper:
        
        try:
            rename_mapper[dcmMapper[z_val]] = annotation_mapper[z_val]
        except:
            pass


    renameDcmFiles(dcmPath, rename_mapper)
    #readAnnotation(newPath+'/'+annotationFolder+'/')
    #print(contour.get_contour_file(newPath+'/'+dcmFolder+'/'))


    contour_data = pydicom.dcmread(os.path.join(dcmPath,'contour_copy.dcm'))
    
    contour_arrays = contour.cfile2pixels(file='contour_copy.dcm', path=dcmPath, ROIContourSeq=index)

    for i in tqdm(range(len(contour_arrays))):
        first_image, first_contour, img_id = contour_arrays[i]

        first_contour = contour.fill_contour(first_contour)


       

        pFile = pydicom.dcmread(dcmPath + '/' + img_id + '.dcm')
        device = pFile.Manufacturer

        if device == 'SIEMENS':
            minn = globalminSIEMENS
            maxx = globalmaxSIEMENS
        else:
            minn = globalminCMS
            maxx = globalmaxCMS



        #minn = np.min(first_image)
        first_image = first_image - minn 
        #maxx = np.max(first_image)
        first_image = first_image / maxx
        first_image_png = first_image * 255
        first_image_png = first_image_png.astype(np.uint8)

        first_contour_png = first_contour * 255
        first_contour_png = first_contour_png.astype(np.uint8)

        #plt.subplot(1,3,1)
        #plt.imshow(first_image,cmap='gray')
        #plt.title('Slice')
        #plt.subplot(1,3,2)
        #plt.imshow(first_contour,cmap='gray')
        #plt.title('Tumor Mask')
        #plt.xlabel(path)
        #plt.subplot(1,3,3)
        #plt.imshow(first_contour*first_image,cmap='gray')
        #plt.title('Segmented Tumor')
        #plt.show()

      
    
        cv2.imwrite(os.path.join('training_data_new','images',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_image_png)
        cv2.imwrite(os.path.join('training_data_new','masks',folderName, str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'.png'), first_contour_png)

            
        try:

            pickle.dump(first_image_png,open(os.path.join('processed',folderName,  str(sorted_keys[str(float(annotation_mapper_r[img_id]))]) +'X.p'),'wb'))
            pickle.dump(first_contour_png, open(os.path.join('processed', folderName ,  str(sorted_keys[str(float(annotation_mapper_r[img_id]))])+ 'Y.p'), 'wb'))

        except:
            pass



def main():

    #readAll('./VIP_CUP_DATA')
    readAll('VIP_CUP_DATA')

if __name__ == '__main__':
    main()

