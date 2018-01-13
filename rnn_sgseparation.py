# -*- coding: utf-8 -*-
"""
   This code classifies stars and galaxies in the S-PLUS survey, using the
   Recurrent Neural Networks (RNN/LSTM). This algorithm is trained and validated 
   using S-PLUS observations in combination with overlapped SDSS/S82 classification. 

                                              mvcduarte - 12/2017
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
import matplotlib.pyplot as plt
import starlight_tools as stl

######################################################################

def load_train_test_validation_samples(infile, mag_min_max, n_bands, n_morph, flag_morph):

    """
      This routine loads the training, test or validation samples
      with the option of including or not the morpholofical part. 

                      mvcduarte - 14/11/2017
    """

    if flag_morph == 1: # morph
        columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
        type_variable = [0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    else: # only mags (not columns 15 and 16)
        columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18]
        type_variable = [0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

    print(len(columns), len(type_variable))

    data = stl.read_ascii_table(infile, '#', columns, type_variable)  
    data = np.array(data)

    #print mag_min_max

    mag_ref = np.array(data[9], dtype = float)
    idx_sample = np.where((mag_ref >= mag_min_max[0]) & (mag_ref <= mag_min_max[1]))[0]
    ngal = len(mag_ref[idx_sample])
    print('ngal=', ngal)

    aid_initial = np.array(data[0])[idx_sample]

    Y_initial = np.array(data[1], dtype = int)[idx_sample]
    if flag_morph == 1:
        X_initial = np.zeros(ngal * (n_bands + n_morph)).reshape(ngal, (n_bands + n_morph))
        for i in range((n_bands + n_morph)):
            X_initial[:, i] = np.array(data[2+i])[idx_sample]
    else:
        X_initial = np.zeros(ngal * (n_bands)).reshape(ngal, (n_bands))
        for i in range((n_bands)):           
            X_initial[:, i] = np.array(data[2+i])[idx_sample]

    if flag_morph == 1: # morph
        field_initial = np.array(data[17], dtype = str)
    else:
        field_initial = np.array(data[14], dtype = str)

    return X_initial, Y_initial,field_initial,  aid_initial

def slice_matrix_SPLUS_bands(X, Y, idx_bands):

    """
        This routine slices the X and Y arrays in order to select
        only objects with a set of measured S-PLUS mags.  

                      mvcduarte - 02/12/2017
    """
    #print np.shape(X)
    idx_sample = np.zeros(len(X[:, 0]))
    for j in range(len(idx_bands)): # each band which should be "ok"

        # indexing values of the "idx_bands[j]" mags which are missed (99.)
        idx = np.where(X[:, idx_bands[j]] == 99.)[0]
        # Flag them as 1.
        idx_sample[idx] = 1. # flag objects with missing band as 1.

    # Select objects which are not affected by missing values for this set of bands (idx_bands) 
    idx_sample = [(idx_sample == 0.)]

    # Output them into new arrays
    X_out = X[:, idx_bands][idx_sample].copy()
    Y_out = Y[idx_sample].copy()

    return X_out, Y_out

def make_matrix_Y(Y_matrix):
    """
       It makes the label Y matrix in a proper shape to RNN
       It returns a 2-element array (eg [0, 1]) that indicate the class.

    """
    Yout = np.zeros(2 * len(Y_matrix)).reshape(len(Y_matrix), 2)
    idx_star = np.where(Y_matrix == 6)[0]
    idx_galaxy = np.where(Y_matrix == 3)[0]

    Yout[idx_star, 0] = 1. 
    Yout[idx_galaxy, 1] = 1. 
    
    return Yout

def check_performance(Y_predicted, Y_test):
    """
        Calculate the performance of the RNN by checking the 
        fraction of correct classification.
    """
    n = 0.
    for i in range(len(Y_predicted)):
        if np.argmin(1. - Y_predicted[i], axis = 0) == np.argmin(1. - Y_test[i], axis = 0):
            n += 1.

    return n / float(len(Y_predicted))

def main():

    # Use morphology?

    flag_morph = 1 #(0/1) = (YES/NO)

    # TRAINING and TEST samples

    str_date = '08012018'
    #path_samples = '/home/mvcduarte/Dropbox/TMP/samples_sg_separation/'
    path_samples = '/Users/marcusduarte/Dropbox/TMP/samples_sg_separation/'

    infile_training = 'match_SPLUS_SDSS_S82_phot.cat_PSF_max2.4.v2_' + str_date + '_training'
    infile_test = 'match_SPLUS_SDSS_S82_phot.cat_PSF_max2.4.v2_' + str_date + '_test'

    # Number of SPLUS bands

    n_bands = 12

    # Number of morphological parameters

    n_morph = 3

    # Magnitude range (r-band)

    mag_min_max = [13., 21.]

    # Indexing S-PLUS bands

    idx_broad_bands = [5, 7, 9, 11] # g r i z
    idx_narrow_bands = [0, 1, 2, 3, 4, 6, 8, 10] # uJAVA + narrow bands
    if flag_morph == 1:
        idx_all_bands = np.arange(15) # 12 SPLUS filters + morph
        n_morph = 3
    else:
        idx_all_bands = np.arange(12) # 12 SPLUS filters
        n_morph = 0

    # Number of epochs for the RNN

    n_epoch = 20 # (>200)

    # Loading the TRAINING and VALIDATION samples

    X_training_initial, Y_training_initial, field_initial, aid_training_initial = \
    load_train_test_validation_samples(path_samples + infile_training, mag_min_max, n_bands, n_morph, flag_morph)

    X_test_initial, Y_test_initial, field_initial, aid_test_initial = \
    load_train_test_validation_samples(path_samples + infile_test, mag_min_max, n_bands, n_morph, flag_morph)

    # Slice the INITIAL samples in order to get a subsamples with all features  

    X_training, Y_training = slice_matrix_SPLUS_bands(X_training_initial, Y_training_initial, idx_all_bands)
    X_test, Y_test = slice_matrix_SPLUS_bands(X_test_initial, Y_test_initial, idx_all_bands)

    # Reshape TRAINING and TEST samples for LSTM

    X_training = np.reshape(X_training, [-1, 1, len(idx_all_bands)])
    X_test = np.reshape(X_test, [-1, 1, len(idx_all_bands)])

    Y_test = make_matrix_Y(Y_test)
    Y_training = make_matrix_Y(Y_training)

    print(np.shape(X_training), np.shape(X_test))
    print(np.shape(Y_training), np.shape(Y_test))

    # Mounting the RNN

    net = tflearn.input_data(shape=[None, 1, len(idx_all_bands)]) # Define the input format of objects (images 28x28) 
    net = tflearn.lstm(net, n_units=128, return_seq=True)# 128 is number of ....?? 
    net = tflearn.lstm(net, n_units=128)
    net = tflearn.fully_connected(net, len(Y_test[0,:]), activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy') # Define a regression
    model = tflearn.DNN(net, tensorboard_verbose=2)

    # Fitting...

    print('Fitting RNN...')
    print(np.shape(X_training), np.shape(Y_training))
    model.fit(X_training, Y_training, n_epoch = n_epoch, validation_set=0.2, show_metric=True,
              snapshot_step=100, run_id='RNN_SG_SEPARATION')

    # Save the trained model
    model.save('model_sg_separation.tflearn')

    # Prediction of the VALIDATION sample

    Y_predicted = model.predict(X_test)
    
    performance = check_performance(Y_predicted, Y_test)

    print('Final Performance=', performance)

    return

if __name__ == '__main__':

    main()
