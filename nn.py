### NEURAL NETWORK ###
# Source: Nitin Kamra, for USC Fall 2016 CSCI567 HW4
# Modified and used by Sourya Dey with permission
# Uses Deep Learning library Keras 1.1.1 <https://keras.io/>

import make_dataset as md
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

def opt_price_hist(prices,bins):
    ''' Try to get optimum distribution of prices by forming suitable bins '''
    hist = np.histogram(prices,bins=bins)[0]
    plt.stem(bins[:-1],hist) #Should be as horizontally even as possible
    plt.show()
    binfreq = np.transpose(np.asarray((bins[:-1],hist))) #Frequency in each bin should be as equal as possible
    print binfreq


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


def testmodels(xtr,ytr,xte,yte, archs=[], actfn='relu', last_act='softmax', reg_coeffs=[0.0], 
               num_epoch=50, batch_size=100, sgd_lr=1e-5, sgd_decays=[0.0], sgd_moms=[0.0], 
               sgd_Nesterov=False, EStop=False, verbose=0):
    '''
    Train and test neural network architectures with varying parameters
        xtr, ytr, xte, yte: (Training and test) (features and prices)
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
    best_acc = 0
    best_config = []
    call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')
    for arch in archs:
        for reg_coeff in reg_coeffs:
            for sgd_decay in sgd_decays:
                for sgd_mom in sgd_moms:
                    model = genmodel(num_units=arch, actfn=actfn, reg_coeff=reg_coeff, last_act=last_act)
                    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_mom, nesterov=sgd_Nesterov)
                    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                    # Train Model
                    if EStop:
                        model.fit(xtr,ytr, nb_epoch=num_epoch, batch_size=batch_size, verbose=verbose, 
                                  callbacks=[call_ES], validation_split=0.17, shuffle=True)
                    else:
                        model.fit(xtr,ytr, nb_epoch=num_epoch, batch_size=batch_size, verbose=verbose)
                    # Evaluate Models
                    score = model.evaluate(xte,yte, batch_size=batch_size, verbose=verbose)
                    if score[1] > best_acc:
                        best_acc = score[1]
                        best_config = [arch, reg_coeff, sgd_decay, sgd_mom, actfn, best_acc]
                        best_model = model
                    print 'Score for architecture = {0}, lambda = {1}, decay = {2}, momentum = {3}, actfn = {4}: {5}'.format(arch, reg_coeff, sgd_decay, sgd_mom, actfn, score[1])
    print 'Best Config: architecture = {0}, lambda = {1}, decay = {2}, momentum = {3}, actfn = {4}, best_acc = {5}'.format(best_config[0], best_config[1], best_config[2], best_config[3], best_config[4], best_config[5])
    return best_model

#%% Data preprocessing
features,prices = md.gen_data()
features,prices = md.shuffle_data(features,prices)
features = md.normalize(features)
#bins = [int(min(prices)),170,190,210,230,250,275,300,315,350,375,400,425,450,475,
#        500,525,550,575,600,625,650,700,720,740,780,810,840,870,900,
#        940,980,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2250,2500,2750,
#        3000,3300,3600,3900,4200,4500,5000,5500,6000,7000,8500,10000,16000,22000,30000,int(max(prices))+1]
bins = [int(min(prices)),200,250,300,350,400,450,500,
        550,600,675,750,825,900,1000,
        1100,1200,1300,1500,1700,1900,2200,2700,
        3200,3800,4500,5500,7000,10000,20000,int(max(prices))+1]
#opt_price_hist(prices,bins) #Comment out this line when running NN
cat_prices = md.categorical_prices(prices,bins)
xtr,ytr,xte,yte = md.split_data(features,cat_prices)
del (features,prices)

#%% Run Neural Network
model = testmodels(xtr,ytr,xte,yte,
           archs=[[len(xtr[1]),1000,len(ytr[1])]],
           num_epoch=100, batch_size=10, reg_coeffs=[1e-5], sgd_lr=0.0001, verbose=1)

#%% Do specific input tests
num = 10
print model.predict_classes(xte[:num],verbose=0)
print np.argmax(yte[:num],axis=1)
