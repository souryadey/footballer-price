### NEURAL NETWORK ###
# Source: Nitin Kamra, for USC Fall 2016 CSCI567 HW4
# Modified and used by Sourya Dey with permission
# Uses Deep Learning library Keras 1.1.1 <https://keras.io/>

#%% Imports and constants
import numpy as np
np.set_printoptions(threshold=np.inf) #View full arrays in console
import matplotlib.pyplot as plt
import os
from pprint import pprint

from keras.layers import Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import load_model

import make_dataset as md
NUM_TEST = 2500
NUM_TRAIN = md.NUM_TOTAL - NUM_TEST #12840
#NUM_VAL = 2000


#%% Data preprocessing
def normalize(features):
    ''' Normalize features by converting to N(0,1)'''
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    features = (features-mu)/sigma
    return features

def analyze_prices(prices):
    ''' Return dictionary showing number of times each price occurs '''
    pdict = {}
    for p in prices:
        if p in pdict.keys(): pdict[p] += 1
        else: pdict[p] = 1
    return pdict

def opt_price_hist(prices,bins):
    ''' Try to get optimum distribution of prices by forming suitable bins '''
    if bins[0]!=min(prices): bins.insert(0,min(prices))
    if bins[-1]<=max(prices): bins.append(max(prices)+1) #so that max values in prices have their own bin and don't fall in previous bin
    hist = np.histogram(prices,bins=bins)[0]
    plt.stem(bins[:-1],hist) #Should be as horizontally even as possible
    plt.show()
    binfreq = np.transpose(np.asarray((bins[:-1],hist))) #Frequency in each bin should be as equal as possible
    print binfreq

def categorical_prices(prices,bins):
    ''' Main function: Split prices into one-hot based on intervals in bins
        Side function: Create new price array which rounds outliers based on intervals in bins
        bins: A list with the starting points of each interval and ending point of the last interval.
            Must be in ascending order
        Eg: If bins = [100,200,300,401], then there are 3 bins - [100,200), [200,300) and [300,401)
            Then a price of 243 would show as [0,1,0]
        Returns cat_prices of size (len(prices),len(bins)-1)
        Also returns rounded_prices: the new price array. This is used for histogram plotting
    '''
    cat_prices = np.zeros((len(prices),len(bins)))
    rounded_prices = np.zeros_like(prices)
    for p in xrange(len(prices)):
        if prices[p] <= bins[0]:
            cat_prices[p][0] = 1. #round up lowest prices to min threshold price
            rounded_prices[p] = bins[0]
        elif prices[p] >= bins[-1]:
            cat_prices[p][len(bins)-1] = 1. #round down highest prices to max threshold price 
            rounded_prices[p] = bins[-1]
        else:
            for b in xrange(len(bins)-2,0,-1):
                if prices[p]>=bins[b]:
                    cat_prices[p][b] = 1.
                    rounded_prices[p] = bins[b]
                    break
    return (cat_prices,rounded_prices)

def shuffle_data(features,prices,rounded_prices):
    ''' Shuffle features '''
    np.random.seed(0) #To maintain consistency across runs
    perm = np.random.permutation(md.NUM_TOTAL)
    temp_features = np.zeros_like(features)
    temp_prices = np.zeros_like(prices)
    temp_rounded_prices = np.zeros_like(rounded_prices)
    for p in xrange(len(perm)):
        temp_features[p][:] = features[perm[p]][:]
        temp_prices[p][:] = prices[perm[p]][:]
        temp_rounded_prices[p] = rounded_prices[perm[p]]
    return (temp_features,temp_prices,temp_rounded_prices)

def split_data(features,prices):
    ''' Separate into training and test '''
    xtr = features[:NUM_TRAIN][:]
    ytr = prices[:NUM_TRAIN][:]
    #xva = features[NUM_TRAIN:NUM_TRAIN+NUM_VAL][:]
    #yva = prices[NUM_TRAIN:NUM_TRAIN+NUM_VAL][:]
    xte = features[NUM_TRAIN:md.NUM_TOTAL][:]
    yte = prices[NUM_TRAIN:md.NUM_TOTAL][:]
    return (xtr,ytr,xte,yte)

features,prices = md.gen_data()
features = normalize(features)
prices_dict = analyze_prices(prices)
#pprint(prices_dict)
prices_prebins = sorted(prices_dict.keys())
prices_bins = [prices_prebins[prices_prebins.index(43):prices_prebins.index(12400)],[13000,14000,15100,16200,18400,20500,22100,24800,28600,36700]]
prices_bins = [item for sublist in prices_bins for item in sublist] #flatten list
cat_prices,rounded_prices = categorical_prices(prices,prices_bins)
features,cat_prices,rounded_prices = shuffle_data(features,cat_prices,rounded_prices)
xtr,ytr,xte,yte = split_data(features,cat_prices)
nin = len(xtr[1])
nout = len(ytr[1])
#del (features,cat_prices)

#bins = [int(min(prices)),170,190,210,230,250,275,300,315,350,375,400,425,450,475,
#        500,525,550,575,600,625,650,700,720,740,780,810,840,870,900,
#        940,980,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2250,2500,2750,
#        3000,3300,3600,3900,4200,4500,5000,5500,6000,7000,8500,10000,16000,22000,30000,int(max(prices))+1]
#bins = [int(min(prices)),200,250,300,350,400,450,500,
#        550,600,675,750,825,900,1000,
#        1100,1200,1300,1500,1700,1900,2200,2700,
#        3200,3800,4500,5500,7000,10000,20000,int(max(prices))+1]
#opt_price_hist(rounded_prices,prices_bins) #Comment out this line when running NN


#%% Neural network
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


def neighbor(y_true, y_pred, n=2): ##### NOT WORKING #####
    ''' Trying to do what neighbor_accuracy does later '''
    return K.cast(K.lesser_equal(K.abs(K.argmax(y_pred,axis=-1) - K.argmax(y_true,axis=-1)), n), K.floatx())

def testmodels(xtr,ytr,xte,yte, num_epoch=50, batch_size=20, actfn='relu', last_act='softmax',
               EStop=True, verbose=1, archs=[[2000, 1500, 500]], reg_coeffs=[5e-4],
               sgd_lrs=[0.01], sgd_decays=[0.001], sgd_moms=[0.99], sgd_Nesterov=True,
               results_file='results.txt'):
    '''
    Train and test neural network architectures with varying parameters
        xtr, ytr, xte, yte: (Training and test) (features and prices)
        archs: List of architectures. ONLY ENTER hidden layer sizes
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeffs: List of L2-regularization coefficients
        num_epoch: number of iterations for SGD
        batch_size: batch size for gradient descent
        sgd_lr: Learning rate for SGD
        sgd_decays: List of decay parameters for the learning rate
        sgd_moms: List of momentum coefficients, works only if sgd_Nesterov = True
        sgd_Nesterov: Boolean variable to use/not use momentum
        EStop: Boolean variable to use/not use early stopping
        verbose: 0 or 1 to determine whether keras gives out training and test progress report
    '''
    f = open(os.path.dirname(os.path.realpath(__file__))+'/result_files/'+results_file,'wb')
    best_acc = 0
    best_config = []
    best_model = None
#    best_mse = np.inf
#    best_config_mse = []
#    best_model_mse = None
    call_ES = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
    for arch in archs:
        arch.insert(0,nin)
        arch.append(nout)
        for reg_coeff in reg_coeffs:
            for sgd_lr in sgd_lrs:
                for sgd_decay in sgd_decays:
                    for sgd_mom in sgd_moms:
                        print 'Starting architecture = {0}, lambda = {1}, eta = {2}, decay = {3}, momentum = {4}, actfn = {5}'.format(arch, reg_coeff, sgd_lr, sgd_decay, sgd_mom, actfn)
                        model = genmodel(num_units=arch, actfn=actfn, reg_coeff=reg_coeff, last_act=last_act)
                        sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_mom, nesterov=sgd_Nesterov)
                        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                        # Train Model
                        if EStop:
                            model.fit(xtr,ytr, nb_epoch=num_epoch, batch_size=batch_size, verbose=verbose, 
                                      callbacks=[call_ES], validation_split=0.15, shuffle=True)
                        else:
                            model.fit(xtr,ytr, nb_epoch=num_epoch, batch_size=batch_size, verbose=verbose)
                        # Evaluate Models
                        score = model.evaluate(xte,yte, batch_size=batch_size, verbose=verbose)
                        if score[1] > best_acc:
                            best_acc = score[1]
                            best_config = [arch, reg_coeff, sgd_lr, sgd_decay, sgd_mom, actfn, best_acc]
                            best_model = model
#                        if score[2] < best_mse:
#                            best_mse = score[2]
#                            best_config_mse = [arch, reg_coeff, sgd_lr, sgd_decay, sgd_mom, actfn, best_mse]
#                            best_model_mse = model
                        result = 'Score for architecture = {0}, lambda = {1}, eta = {2}, decay = {3}, momentum = {4}, actfn = {5}: Acc = {6}%\n'.format(arch, reg_coeff, sgd_lr, sgd_decay, sgd_mom, actfn, score[1]*100)
                        print result
                        f.write(result)
    final_result_acc = 'Best Config: architecture = {0}, lambda = {1}, eta = {2}, decay = {3}, momentum = {4}, actfn = {5}, best_acc = {6}%\n'.format(best_config[0], best_config[1], best_config[2], best_config[3], best_config[4], best_config[5], best_config[6]*100)
    print final_result_acc
    f.write(final_result_acc)
#    final_result_mse = 'Best Config MSE: architecture = {0}, lambda = {1}, eta = {2}, decay = {3}, momentum = {4}, actfn = {5}, best_mse = {6}\n'.format(best_config_mse[0], best_config_mse[1], best_config_mse[2], best_config_mse[3], best_config_mse[4], best_config_mse[5], best_config_mse[6])
#    print final_result_mse
#    f.write(final_result_mse)
    f.close()
    return best_model


def neighbor_accuracy(model,xte,yte, neighbor_range=2, num=NUM_TEST):
    ''' Returns percentage of correct = (predicted label is +/- n from accurate label)
        Eg: If n=2, it's essentially top5 because predicted label can be accurate label -2, -1, +0, +1, +2
        For some reason this fails if num=1, i.e. single cases can't be tested '''
    y_pred = model.predict_classes(xte[:num],verbose=0)
    y_true = np.argmax(yte[:num],axis=1)
    acc = [np.abs(y_pred[i]-y_true[i])<=neighbor_range for i in xrange(num)]
    return 100.0*acc.count(True)/num

def price_error(model,xte,rounded_prices,num=NUM_TEST):
    ''' Returns absolute error between predicted price and actual price, percentage absolute error, and their averages
        Pass the entire rounded_prices into this, it will automatically extract what's required
    '''
    rounded_prices = rounded_prices[NUM_TRAIN:NUM_TRAIN+num]
    y_pred = model.predict_classes(xte[:num],verbose=0)
    pred_prices = [prices_bins[i] for i in y_pred]
    error = [np.abs(pred_prices[i]-rounded_prices[i]) for i in xrange(num)]
    pc_error = [100.0*error[i]/rounded_prices[i] for i in xrange(num)]
    avg_error = np.mean(error)
    avg_pc_error = np.mean(pc_error)
    return (error,pc_error,avg_error,avg_pc_error)


#%% Trial
#model = testmodels(xtr,ytr,xte,yte, num_epoch=2,
#                             archs=[[300]],
#                             results_file = 'trial.txt')

#%% Vary batch sizes only
#model,model_mse = testmodels(xtr,ytr,xte,yte, batch_size=1, 
#                             archs=[[nin,1000,nout]],
#                             results_file = 'batch_size.txt')

#%% Vary architectures only
#archs = [[nin,a,nout] for a in xrange(100,5001,100)]
#model,model_mse = testmodels(xtr,ytr,xte,yte, 
#                             archs=archs,
#                             results_file = 'archs_1hiddenlayer.txt')

#%% Vary activation functions over architectures
#Possiblities are [relu,soft], [sigm,soft], [tanh,soft], [relu,sigm], [sigm,sigm], [tanh,sigm]
#[relu,soft] is default. Here I'm trying the next 3
#archs = [[nin,a,nout] for a in xrange(500,4501,1000)]
#model = testmodels(xtr,ytr,xte,yte, actfn='sigmoid',
#                   archs=archs,
#                   results_file = 'act_sigmoid.txt')
#model = testmodels(xtr,ytr,xte,yte, actfn='tanh',
#                   archs=archs,
#                   results_file = 'act_tanh.txt')
#model = testmodels(xtr,ytr,xte,yte, last_act='sigmoid',
#                   archs=archs,
#                   results_file = 'lastact_sigmoid.txt')

#%% Vary number of hidden layers in architecture only
#archs = [[nin,2000,3000,nout],[nin,2000,2000,nout],[nin,2000,1500,nout],[nin,2000,1000,nout],[nin,2000,500,nout],[nin,2000,nout,nout]]
#model = testmodels(xtr,ytr,xte,yte,
#                   archs=archs,
#                   results_file = 'archs_2hiddenlayers.txt')
#model = testmodels(xtr,ytr,xte,yte,
#                   archs=[[nin,2200,1500,nout]],
#                   results_file = 'particular_2hiddenlayer.txt')

#%% Vary eta and archs
#sgd_lrs = [1e-6,1e-5,5e-5,5e-4,1e-3,1e-2]
#archs = [[nin,900,nout],[nin,1700,nout],[nin,2100,nout],[nin,2900,nout],[nin,3900,nout],[nin,4600,nout]]
#model = testmodels(xtr,ytr,xte,yte,
#                   archs=archs,
#                   sgd_lrs=sgd_lrs,
#                   results_file = 'etas_1hiddenlayer.txt')
#model = testmodels(xtr,ytr,xte,yte,
#                   archs=[[nin,3900,1500,nout]],
#                   sgd_lrs=sgd_lrs,
#                   results_file = 'etas_particular_2hiddenlayer.txt')

#%% Vary eta and archs (>1 hidden layer)
#model = testmodels(xtr,ytr,xte,yte,
#                   archs=[[2100]],
#                   sgd_lrs=np.arange(0.02,0.1,0.01),
#                   results_file = 'dump.txt')
#archs = [[2000,500],[2000,1000],[2000,1500],[2000,2000],[2000,2500],[2000,1500,500],[2000,1500,1000],[2000,1500,1500],[2000,1500,2000],[2000,1500,2500]]
#sgd_lrs = [0.008,0.01,0.033,0.067,0.1,0.133,0.167,0.2,0.25,0.3,0.35,0.4]
#model = testmodels(xtr,ytr,xte,yte,
#                   archs=archs,
#                   sgd_lrs=sgd_lrs,
#                   results_file = 'etas_archs_manyhiddenlayers.txt')
#model = testmodels(xtr,ytr,xte,yte,
#                   results_file = 'reg_1pm3_lessbatchsize.txt')

#%% Final
model = testmodels(xtr,ytr,xte,yte, num_epoch=100,
                   results_file = 'final_epoch100_batch20.txt')

           
#%% Post-processing
model.save(os.path.dirname(os.path.realpath(__file__))+'/model_files/final_epoch100_batch20.h5')
#del model
#model = load_model(os.path.dirname(os.path.realpath(__file__))+'/model_files/trial.h5')
top5acc = neighbor_accuracy(model,xte,yte)
error,pc_error,avg_error,avg_pc_error = price_error(model,xte,rounded_prices)

