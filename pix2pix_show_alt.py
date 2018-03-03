#https://github.com/tommyfms2/pix2pix-keras-byt
#http://toxweblog.toxbe.com/2017/12/24/keras-%e3%81%a7-pix2pix-%e3%82%92%e5%ae%9f%e8%a3%85/
#tommyfms2/pix2pix-keras-byt より


import os
import argparse

import numpy as np

import h5py
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from keras.preprocessing.image import load_img, img_to_array

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD

import models

def my_normalization(X):
    return X / 127.5 - 1

def my_inverse_normalization(X):
    return (X + 1.) / 2.

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def plot_generated_batch(X_proc, X_raw, generator_model, batch_size, suffix):
    X_gen = generator_model.predict(X_raw)
    X_raw = my_inverse_normalization(X_raw)
    X_proc = my_inverse_normalization(X_proc)  #超えた絵合わせのため、解画像
    X_gen = my_inverse_normalization(X_gen)

    Xs = to3d(X_raw[:10])
    Xg = to3d(X_gen[:10])
    Xr = to3d(X_proc[:10])
    Xs = np.concatenate(Xs, axis=1)
    Xg = np.concatenate(Xg, axis=1)
    Xr = np.concatenate(Xr, axis=1)
    #XX = np.concatenate((Xs,Xg,Xr), axis=0)
    XX = np.concatenate((Xs,Xg), axis=0)
    return XX

def my_load_data_test(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_val = hf["val_data_gen"][:].astype(np.float32)
        X_full_val = my_normalization(X_full_val)
        X_sketch_val = hf["val_data_raw"][:].astype(np.float32)
        X_sketch_val = my_normalization(X_sketch_val)
        return X_full_val, X_sketch_val

def my_train(args):
    # create output finder
    
    if not os.path.exists(os.path.expanduser(args.datasetpath1)):
        os.mkdir(findername)
    
    # create figures
    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    # load test data
    procImage_val0, rawImage_val0 = my_load_data_test(args.datasetpath0)
    procImage_val1, rawImage_val1 = my_load_data_test(args.datasetpath1)
    procImage_val2, rawImage_val2 = my_load_data_test(args.datasetpath2)
    procImage_val3, rawImage_val3 = my_load_data_test(args.datasetpath3)
    procImage_val4, rawImage_val4 = my_load_data_test(args.datasetpath4)
    procImage_val5, rawImage_val5 = my_load_data_test(args.datasetpath5)

    
    print('procImage_val : ', procImage_val0.shape)
    print('rawImage_val : ', rawImage_val0.shape)

    img_shape = rawImage_val0.shape[-3:]   #rawImage.shape[-3:]
    print('img_shape : ', img_shape)

    disc_img_shape = (args.patch_size, args.patch_size, procImage_val0.shape[-1])
    print('disc_img_shape : ', disc_img_shape)

    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_model = models.my_load_generator(img_shape, disc_img_shape)
    generator_model.load_weights('params_generator6u_pix_epoch_8000.hdf5')  #160-akb kao
    generator_model.compile(loss='mae', optimizer=opt_discriminator)
    generator_model.trainable = False
    XX=[]
    imgs = []
    YY=[]
    for j in range(0,1):
        idx = [0+10*j,1+10*j,2+10*j,3+10*j,4+10*j,5+10*j,6+10*j,7+10*j,8+10*j,9+10*j]  #np.random.choice(procImage_val.shape[0], args.batch_size)
        print("j=",j,"idx=",idx)
        X_gen_target, X_gen = procImage_val0[idx], rawImage_val0[idx]
        XX=plot_generated_batch(X_gen_target, X_gen, generator_model, args.batch_size, "validation"+str(0)+".png")
        imgarray = img_to_array(XX)
        imgs.append(imgarray)
        X_gen_target, X_gen = procImage_val1[idx], rawImage_val1[idx]
        XX=plot_generated_batch(X_gen_target, X_gen, generator_model, args.batch_size, "validation"+str(0)+".png")
        imgarray = img_to_array(XX)
        imgs.append(imgarray)
        X_gen_target, X_gen = procImage_val2[idx], rawImage_val2[idx]
        XX=plot_generated_batch(X_gen_target, X_gen, generator_model, args.batch_size, "validation"+str(0)+".png")
        imgarray = img_to_array(XX)
        imgs.append(imgarray)
        X_gen_target, X_gen = procImage_val3[idx], rawImage_val3[idx]
        XX=plot_generated_batch(X_gen_target, X_gen, generator_model, args.batch_size, "validation"+str(0)+".png")
        imgarray = img_to_array(XX)
        imgs.append(imgarray)
        X_gen_target, X_gen = procImage_val4[idx], rawImage_val4[idx]
        XX=plot_generated_batch(X_gen_target, X_gen, generator_model, args.batch_size, "validation"+str(0)+".png")
        imgarray = img_to_array(XX)
        imgs.append(imgarray)
        X_gen_target, X_gen = procImage_val5[idx], rawImage_val5[idx]
        XX=plot_generated_batch(X_gen_target, X_gen, generator_model, args.batch_size, "validation"+str(0)+".png")
        imgarray = img_to_array(XX)
        imgs.append(imgarray)
        
        #plt.imshow(imgs[j])
        #plt.axis('off')
        #plt.savefig("./figures/current_batch_"+"validation"+str(j)+".png")
        
    YY = np.concatenate((imgs[0],imgs[1],imgs[2],imgs[3],imgs[4],imgs[5]), axis=0)    
    plt.imshow(YY)
    plt.axis('off')
    plt.savefig("./figures/current_batch_"+"mayu2kitaRaw_test1"+".png")
    plt.clf()
    plt.close()
    """
    YY = np.concatenate((imgs[5],imgs[6],imgs[7],imgs[8],imgs[9]), axis=0)    
    plt.imshow(YY)
    plt.axis('off')
    plt.savefig("./figures/current_batch_"+"mayu2kitaRaw_test2"+".png")
    plt.clf()
    plt.close()
    """
def main():
    parser = argparse.ArgumentParser(description='Train Font GAN')
    #parser.add_argument('--datasetpath0', '-d_train', type=str, default ="dataset1train.hdf5")  #,  "required=True)
    parser.add_argument('--datasetpath5', '-d_test0', type=str, default ="sugao2egaoRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath4', '-d_test1', type=str, default ="egao2ikariRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath3', '-d_test2', type=str, default ="ikari2kuyasiRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath2', '-d_test3', type=str, default ="kuyasi2nakiRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath1', '-d_test4', type=str, default ="naki2henRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath0', '-d_test5', type=str, default ="hen2sugaoRaw_test.hdf5")   #, required=True)
    parser.add_argument('--patch_size', '-p', type=int, default=64)
    parser.add_argument('--batch_size', '-b', type=int, default=5)
    parser.add_argument('--epoch','-e', type=int, default=200)
    args = parser.parse_args()

    K.set_image_data_format("channels_last")

    my_train(args)


if __name__=='__main__':
    main()

