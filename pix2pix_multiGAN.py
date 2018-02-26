#https://github.com/tommyfms2/pix2pix-keras-byt
#http://toxweblog.toxbe.com/2017/12/24/keras-%e3%81%a7-pix2pix-%e3%82%92%e5%ae%9f%e8%a3%85/
#tommyfms2/pix2pix-keras-byt より
#20data 
"""
j 2000, Epoch1 2000/2001, Time: 5786.568003177643
20/20 [==============================] - 1s - D logloss: 0.5624 - G tot: 4.0277 - G L1: 0.2641 - G logloss: 1.3863
20/20 [==============================] - 3s - D logloss: 0.5624 - G tot: 5.1342 - G L1: 0.3748 - G logloss: 1.3863
j 2001, Epoch1 2001/2001, Time: 5792.787416219711
"""

import os
import argparse

import numpy as np

import h5py
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD

import models

def my_normalization(X):
    return X / 127.5 - 1
def my_inverse_normalization(X):
    return (X + 1.) / 2.

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def plot_generated_batch(X_proc, X_raw, generator_model, batch_size, suffix):
    X_gen = generator_model.predict(X_raw)
    X_raw = my_inverse_normalization(X_raw)
    X_proc = my_inverse_normalization(X_proc)
    X_gen = my_inverse_normalization(X_gen)

    Xs = to3d(X_raw[:5])
    Xg = to3d(X_gen[:5])
    Xr = to3d(X_proc[:5])
    Xs = np.concatenate(Xs, axis=1)
    Xg = np.concatenate(Xg, axis=1)
    Xr = np.concatenate(Xr, axis=1)
    XX = np.concatenate((Xs,Xg,Xr), axis=0)

    plt.imshow(XX)
    plt.axis('off')
    plt.savefig("./figures/current_batch_"+suffix+".png")
    plt.clf()
    plt.close()

# tmp load data gray to color
def my_load_data(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_train = hf["train_data_gen"][:].astype(np.float32)
        X_full_train = my_normalization(X_full_train)
        X_sketch_train = hf["train_data_raw"][:].astype(np.float32)
        X_sketch_train = my_normalization(X_sketch_train)
        X_full_val = hf["val_data_gen"][:].astype(np.float32)
        X_full_val = my_normalization(X_full_val)
        X_sketch_val = hf["val_data_raw"][:].astype(np.float32)
        X_sketch_val = my_normalization(X_sketch_val)
        return X_full_train, X_sketch_train, X_full_val, X_sketch_val
    
# tmp load data gray to color
# for train & test data exactly select each other
def my_load_data_train(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_train = hf["train_data_gen"][:].astype(np.float32)
        X_full_train = my_normalization(X_full_train)
        X_sketch_train = hf["train_data_raw"][:].astype(np.float32)
        X_sketch_train = my_normalization(X_sketch_train)
        return X_full_train, X_sketch_train
    
def my_load_data_test(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_val = hf["val_data_gen"][:].astype(np.float32)
        X_full_val = my_normalization(X_full_val)
        X_sketch_val = hf["val_data_raw"][:].astype(np.float32)
        X_sketch_val = my_normalization(X_sketch_val)
        return X_full_val, X_sketch_val
    
def extract_patches(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1] // patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2] // patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(procImage, rawImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        # produce an output
        X_disc = generator_model.predict(rawImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1
    else:
        X_disc = procImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc


def my_train(args):
    # create output finder
    
    if not os.path.exists(os.path.expanduser(args.datasetpath00)):
        os.mkdir(findername)
    
    # create figures
    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    # load data
    procImage, rawImage = my_load_data_train(args.datasetpath00)
    procImage_val, rawImage_val = my_load_data_test(args.datasetpath01)
    print('procImage.shape : ', procImage.shape)
    print('rawImage.shape : ', rawImage.shape)
    print('procImage_val : ', procImage_val.shape)
    print('rawImage_val : ', rawImage_val.shape)

    procImage2, rawImage2 = my_load_data_train(args.datasetpath10)
    procImage_val2, rawImage_val2 = my_load_data_test(args.datasetpath11)
    print('procImage.shape : ', procImage2.shape)
    print('rawImage.shape : ', rawImage2.shape)
    print('procImage_val : ', procImage_val2.shape)
    print('rawImage_val : ', rawImage_val2.shape)

    procImage1, rawImage1 = my_load_data_train(args.datasetpath20)
    procImage_val1, rawImage_val1 = my_load_data_test(args.datasetpath21)
    print('procImage.shape : ', procImage1.shape)
    print('rawImage.shape : ', rawImage1.shape)
    print('procImage_val : ', procImage_val1.shape)
    print('rawImage_val : ', rawImage_val1.shape)

    img_shape = rawImage.shape[-3:]
    print('img_shape : ', img_shape)
    patch_num = (img_shape[0] // args.patch_size) * (img_shape[1] // args.patch_size)
    disc_img_shape = (args.patch_size, args.patch_size, procImage.shape[-1])
    print('disc_img_shape : ', disc_img_shape)

    # train
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_model = models.my_load_generator(img_shape, disc_img_shape)
    #generator_model.load_weights('params_generator1_pix_epoch_2000.hdf5')
    # load discriminator model
    discriminator_model = models.my_load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num)
    #discriminator_model.load_weights('params_discriminator1_pix_epoch_2000.hdf5')
    generator_model.compile(loss='mae', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = models.my_load_DCGAN(generator_model, discriminator_model, img_shape, args.patch_size)

    loss = [l1_loss, 'binary_crossentropy']
    loss_weights = [1E1, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    # start training
    j=0
    print('start training')
    starttime = time.time()
    perm = np.random.permutation(rawImage.shape[0])
    
    X_procImage = procImage[perm]
    X_rawImage  = rawImage[perm]
    X_procImageIter = [X_procImage[i:i+args.batch_size] for i in range(0, rawImage.shape[0], args.batch_size)]
    X_rawImageIter  = [X_rawImage[i:i+args.batch_size] for i in range(0, rawImage.shape[0], args.batch_size)]
    
    X_procImage2 = procImage2[perm]
    X_rawImage2  = rawImage2[perm]
    X_procImageIter2 = [X_procImage2[i:i+args.batch_size] for i in range(0, rawImage2.shape[0], args.batch_size)]
    X_rawImageIter2  = [X_rawImage2[i:i+args.batch_size] for i in range(0, rawImage2.shape[0], args.batch_size)]
    
    X_procImage1 = procImage1[perm]
    X_rawImage1  = rawImage1[perm]
    X_procImageIter1 = [X_procImage1[i:i+args.batch_size] for i in range(0, rawImage1.shape[0], args.batch_size)]
    X_rawImageIter1  = [X_rawImage1[i:i+args.batch_size] for i in range(0, rawImage1.shape[0], args.batch_size)]
     
        
    for e in range(args.epoch):
        b_it = 0
        progbar = generic_utils.Progbar(len(X_procImageIter)*args.batch_size)
        for (X_proc_batch, X_raw_batch) in zip(X_procImageIter, X_rawImageIter):
            b_it += 1
            X_disc, y_disc = get_disc_batch(X_proc_batch, X_raw_batch, generator_model, b_it, args.patch_size)
            raw_disc, _ = get_disc_batch(X_raw_batch, X_raw_batch, generator_model, 1, args.patch_size)
            x_disc = X_disc + raw_disc
            # update the discriminator
            disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)

            # create a batch to feed the generator model
            idx = np.random.choice(procImage.shape[0], args.batch_size)
            X_gen_target, X_gen = procImage[idx], rawImage[idx]
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            progbar.add(args.batch_size, values=[
                ("D logloss", disc_loss),
                ("G tot", gen_loss[0]),
                ("G L1", gen_loss[1]),
                ("G logloss", gen_loss[2])
            ])

        b_it2 = 0
        progbar2 = generic_utils.Progbar(len(X_procImageIter2)*args.batch_size)
        for (X_proc_batch2, X_raw_batch2) in zip(X_procImageIter2, X_rawImageIter2):
            b_it += 1
            X_disc2, y_disc2 = get_disc_batch(X_proc_batch2, X_raw_batch2, generator_model, b_it2, args.patch_size)
            raw_disc2, _ = get_disc_batch(X_raw_batch2, X_raw_batch2, generator_model, 1, args.patch_size)
            x_disc2 = X_disc2 + raw_disc2
            # update the discriminator
            disc_loss2 = discriminator_model.train_on_batch(x_disc2, y_disc2)

            # create a batch to feed the generator model
            idx = np.random.choice(procImage2.shape[0], args.batch_size)
            X_gen_target2, X_gen2 = procImage2[idx], rawImage2[idx]
            y_gen2 = np.zeros((X_gen2.shape[0], 2), dtype=np.uint8)
            y_gen2[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss2 = DCGAN_model.train_on_batch(X_gen2, [X_gen_target2, y_gen2])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            progbar2.add(args.batch_size, values=[
                ("D logloss", disc_loss2),
                ("G tot", gen_loss2[0]),
                ("G L1", gen_loss2[1]),
                ("G logloss", gen_loss2[2])
            ])

        b_it1 = 0
        progbar1 = generic_utils.Progbar(len(X_procImageIter1)*args.batch_size)
        for (X_proc_batch1, X_raw_batch1) in zip(X_procImageIter1, X_rawImageIter1):
            b_it1 += 1
            X_disc1, y_disc1 = get_disc_batch(X_proc_batch1, X_raw_batch1, generator_model, b_it1, args.patch_size)
            raw_disc1, _ = get_disc_batch(X_raw_batch1, X_raw_batch1, generator_model, 1, args.patch_size)
            x_disc1 = X_disc1 + raw_disc1
            # update the discriminator
            disc_loss1 = discriminator_model.train_on_batch(x_disc1, y_disc1)

            # create a batch to feed the generator model
            idx = np.random.choice(procImage1.shape[0], args.batch_size)
            X_gen_target1, X_gen1 = procImage1[idx], rawImage1[idx]
            y_gen1 = np.zeros((X_gen1.shape[0], 2), dtype=np.uint8)
            y_gen1[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss1 = DCGAN_model.train_on_batch(X_gen1, [X_gen_target1, y_gen1])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            progbar1.add(args.batch_size, values=[
                ("D logloss", disc_loss1),
                ("G tot", gen_loss1[0]),
                ("G L1", gen_loss1[1]),
                ("G logloss", gen_loss1[2])
            ])
            
            # save images for visualization　file名に通し番号を記載して残す
            #if b_it % (procImage.shape[0]//args.batch_size//2) == 0:
            if j % 100==0:
                plot_generated_batch(X_proc_batch, X_raw_batch, generator_model, args.batch_size, "training" +str(j))
                idx = np.random.choice(procImage_val.shape[0], args.batch_size)
                X_gen_target, X_gen = procImage_val[idx], rawImage_val[idx]
                plot_generated_batch(X_gen_target, X_gen, generator_model, args.batch_size, "validation"+str(j))
 
                plot_generated_batch(X_proc_batch2, X_raw_batch2, generator_model, args.batch_size, "training2_" +str(j))
                idx = np.random.choice(procImage_val2.shape[0], args.batch_size)
                X_gen_target2, X_gen2 = procImage_val2[idx], rawImage_val2[idx]
                plot_generated_batch(X_gen_target2, X_gen2, generator_model, args.batch_size, "validation2_"+str(j))
 
                plot_generated_batch(X_proc_batch1, X_raw_batch1, generator_model, args.batch_size, "training1_" +str(j))
                idx = np.random.choice(procImage_val1.shape[0], args.batch_size)
                X_gen_target1, X_gen1 = procImage_val1[idx], rawImage_val1[idx]
                plot_generated_batch(X_gen_target1, X_gen1, generator_model, args.batch_size, "validation1_"+str(j))
                    
            else:
                continue
            #else:
                #continue
            
        j += 1
        print("")
        print('j %d, Epoch1 %s/%s, Time: %s' % (j, e + 1, args.epoch, time.time() - starttime))
        if j % 100==0:
            generator_model.save_weights('params_generator1_pix_epoch_{0:03d}.hdf5'.format(j), True)
            discriminator_model.save_weights('params_discriminator1_pix_epoch_{0:03d}.hdf5'.format(j), True)
        else:
            continue        
        


def main():
    parser = argparse.ArgumentParser(description='Train Font GAN')
    parser.add_argument('--datasetpath00', '-d_train0', type=str, default ="neko2airpRaw_train.hdf5")  #,  "required=True)
    parser.add_argument('--datasetpath10', '-d_train2', type=str, default ="airp2boyRaw_train.hdf5")  #,  "required=True)
    parser.add_argument('--datasetpath20', '-d_train1', type=str, default ="neko2boyRaw_train.hdf5")  #,  "required=True)
    parser.add_argument('--datasetpath01', '-d_test0', type=str, default ="neko2airpRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath11', '-d_test2', type=str, default ="airp2boyRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath21', '-d_test1', type=str, default ="neko2boyRaw_test.hdf5")   #, required=True)
    parser.add_argument('--patch_size', '-p', type=int, default=64)
    parser.add_argument('--batch_size', '-b', type=int, default=5)
    parser.add_argument('--epoch','-e', type=int, default=8001)
    args = parser.parse_args()

    K.set_image_data_format("channels_last")

    my_train(args)


if __name__=='__main__':
    main()

