# LungTumorSegmentation

For preprocessing, use the following files - new_dataReading.py, crop64.py, crop_val.py, filtered_wavelet.py, filtered_neighbor.py

Before training, run dataHandle.py
For training unet and multiresunet, use train_unet.py and train_multiresunet.py files respectively.
For training deep supervision, change the function call to the appropriate model (UNet or MultiResUNet) in DS_NewPipeline.py

For testing, use testfinal_images.py
