### NEURAL NETWORK ###
# Source: Nitin Kamra, for USC Fall 2016 CSCI567 HW4
# Modified and used by Sourya Dey with permission
# Uses Deep Learning library Keras 1.1.1 <https://keras.io/>

import make_dataset as md
import numpy as np

from keras.layers import Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

def genmodel(num_units, actfn='relu', reg_coeff=0.0, last_act='softmax'):
    '''
    Generate a neural network model of approporiate architecture
    Glorot normal initialization used for all layers
        num_units: architecture of network in the format [n1, n2, ... , nL]
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeff: L2-regularization coefficient
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
    Returns model: Keras sequential model with appropriate fully-connected architecture
    '''
    model = Sequential()
    for i in range(1, len(num_units)):
        if i == 1 and i < len(num_units) - 1: #Input layer
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=actfn, 
                            W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == 1 and i == len(num_units) - 1: #Input layer, network has only 1 layer
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=last_act, 
                            W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i < len(num_units) - 1: #Hidden layer
            model.add(Dense(output_dim=num_units[i], activation=actfn, 
                            W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == len(num_units) - 1: #Output layer
            model.add(Dense(output_dim=num_units[i], activation=last_act, 
                            W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
    return model


def testmodels(xtr,ytr,xva,yva,xte,yte, archs=[md.NUM_FEATURES,1], actfn='relu', last_act='relu', reg_coeffs=[0.0], 
               num_epoch=50, batch_size=100, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0], 
               sgd_Nesterov=False, EStop=False, verbose=0):
    '''
    Train and test neural network architectures with varying parameters
        xtr, ytr, xva, yva, xte, yte: (Training, validation and test) (features and prices)
        archs: List of architectures
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeffs: Lsit of L2-regularization coefficients
        num_epoch: number of iterations for SGD
        batch_size: batch size for gradient descent
        sgd_lr: Learning rate for SGD
        sgd_decays: List of decay parameters for the learning rate
        sgd_moms: List of momentum coefficients, works only if sgd_Nesterov = True
        sgd_Nesterov: Boolean variable to use/not use momentum
        EStop: Boolean variable to use/not use early stopping
        verbose: 0 or 1 to determine whether keras gives out training and test progress report
    '''
    best_mse = np.inf
    best_config = []
    call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')
    for arch in archs:
        for reg_coeff in reg_coeffs:
            for sgd_decay in sgd_decays:
                for sgd_mom in sgd_moms:
                    model = genmodel(num_units=arch, actfn=actfn, reg_coeff=reg_coeff, last_act=last_act) # Generate Model
                    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_mom, nesterov=sgd_Nesterov)
                    model.compile(loss='mean_squared_error', optimizer=sgd) # Compile Model
                    # Train Model
                    if EStop:
                        model.fit(xtr,ytr, validation_data=(xva,yva), nb_epoch=num_epoch, 
                                  batch_size=batch_size, verbose=verbose, callbacks=[call_ES])
                    else:
                        model.fit(xtr,ytr, validation_data=(xva,yva), nb_epoch=num_epoch, 
                                  batch_size=batch_size, verbose=verbose)
                    # Evaluate Model
                    score = model.evaluate(xte,yte, batch_size=batch_size, verbose=verbose)
                    if score < best_mse:
                        best_mse = score
                        best_config = [arch, reg_coeff, sgd_decay, sgd_mom, actfn, best_mse]
                        best_model = model
                    print 'MSE for architecture = {0}, lambda = {1}, decay = {2}, momentum = {3}, actfn = {4}: {5}'.format(arch, reg_coeff, sgd_decay, sgd_mom, actfn, score)
    print 'Best Config: architecture = {0}, lambda = {1}, decay = {2}, momentum = {3}, actfn = {4}, best_mse = {5}'.format(best_config[0], best_config[1], best_config[2], best_config[3], best_config[4], best_config[5])
    return best_model

#%% MAIN
features,prices = md.gen_data()
features,prices = md.shuffle_data(features,prices)
features = md.normalize(features)
xtr,ytr,xva,yva,xte,yte = md.split_data(features,prices)
del (features,prices)
model = testmodels(xtr,ytr,xva,yva,xte,yte,
           archs=[[md.NUM_FEATURES,100,100,1]],
           sgd_lr=0.001, verbose=1)
