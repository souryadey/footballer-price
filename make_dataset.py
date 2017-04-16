### GENERATE DATASETS ###
# Uses manually prepared txt files containing HTML source code for FIFA 2017 player stats or features
# Goalkeepers are excluded
# <http://sofifa.com/players?gender=0&pn%5B0%5D=27&pn%5B1%5D=25&pn%5B2%5D=23&pn%5B3%5D=22&pn%5B4%5D=21&pn%5B5%5D=20&pn%5B6%5D=18&pn%5B7%5D=16&pn%5B8%5D=14&pn%5B9%5D=12&pn%5B10%5D=10&pn%5B11%5D=8&pn%5B12%5D=7&pn%5B13%5D=5&pn%5B14%5D=3&pn%5B15%5D=2&col=vl&sort=desc&offset=0>
# To get following pages, add 100 to offset
# Features are arranged as:
    ## Acceleration, Aggression, Agility, Balance, Ball Control, Composure, Crossing, Curve, DEF, DRI,
    ## Dribbling, Finishing, Free Kick Accuracy, Heading Accuracy, Interceptions, Jumping, Long Passing,
    ## Long Shots, Marking, OVA, PAC, PAS, Penalties, PHY, Positioning, POT, Reactions, SHO, Short Passing,
    ## Shot Power, Sliding Tackle, Sprint Speed, Stamina, Standing Tackle, Strength, Vision, Volleys
# Above 37 are all out of 100. Next is age, and last 3 are stars out of 5
    ## Age, International Reputation, Skill Moves, Weak Foot
# Finally is Value, i.e. player price, which is the output variable
# Must login using Sourya's account credentials to get stats automatically arranged like this

#%% Imports
import numpy as np
np.set_printoptions(threshold=np.inf) #View full arrays in console
import os

#%% Constants
NUM_TRAIN = 10000
NUM_VAL = 2000
NUM_TEST = 2000
NUM_TOTAL = NUM_TRAIN + NUM_VAL + NUM_TEST
NUM_FEATURES = 41
NUM_100SCALEFEATURES = 37
NUM_FILES = NUM_TOTAL/100
#NUM_FILES = 1 #initial test case

#%% Generate X and Y datasets - features and prices
def gen_data():
    features = np.zeros((NUM_TOTAL,NUM_FEATURES))
    prices = np.zeros(NUM_TOTAL)
    playercounter = 0 #never resets
    for filecounter in xrange(1,NUM_FILES+1): #process 1 file = 100 players
        f = open(os.path.dirname(os.path.realpath(__file__))+'/data_files/p{0}.txt'.format(filecounter),'rb')
        lines = f.readlines()
        linecounter = 0 #resets for every new file
        while linecounter != len(lines):
            if lines[linecounter].strip()[:11] == '<span class': #start processing 1 player
                featurecounter = 0 #resets for every new player
                while featurecounter<NUM_100SCALEFEATURES: #process all 'out of 100' features
                    line = lines[linecounter]
                    line = line.strip()
                    val = line[24:26]
                    if val[1] == '<': val = val[0] #in case value is single digit
                    features[playercounter,featurecounter] = float(val)
                    linecounter += 2 #next feature is always 2 lines away
                    featurecounter += 1
                #end of 37 'out of 100' features, now process age and 3 starred 'out of 5' features
                features[playercounter,featurecounter] = float(lines[linecounter].strip()[:2]) #age
                linecounter += 2
                featurecounter += 1
                features[playercounter,featurecounter] = float(lines[linecounter].strip()[0]) #international reputation
                linecounter += 2
                featurecounter += 1
                features[playercounter,featurecounter] = float(lines[linecounter].strip()[0]) #skill moves
                linecounter += 2
                featurecounter += 1
                features[playercounter,featurecounter] = float(lines[linecounter].strip()[0]) #weak foot
                linecounter += 2
                price = '' #all PRICES IN USDx1000
                for char in lines[linecounter].strip()[2:]: #price line
                    if char=='M': #million
                        price = float(price)/100
                        break                    
                    elif char=='K': #thousand
                        price = float(price)/100000 
                        break
                    price += char
                prices[playercounter] = price
                playercounter += 1
            linecounter += 1
        f.close()
        print '{0} players done'.format(playercounter) #track progress
    return (features,prices)

#%% Shuffle features
def shuffle_data(features,prices):
    perm = np.random.permutation(NUM_TOTAL)
    temp_features = np.zeros_like(features)
    for p in xrange(len(perm)):
        temp_features[p][:] = features[perm[p]][:]
    features = temp_features
    del temp_features
    prices = prices[perm]
    return (features,prices)

#%% Normalize features
def normalize(features):
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    features = (features-mu)/sigma
    return features

#%% Separate into training, validation, test
def split_data(features,prices):
    xtr = features[:NUM_TRAIN][:]
    ytr = prices[:NUM_TRAIN]
    xva = features[NUM_TRAIN:NUM_TRAIN+NUM_VAL][:]
    yva = prices[NUM_TRAIN:NUM_TRAIN+NUM_VAL]
    xte = features[NUM_TRAIN+NUM_VAL:NUM_TOTAL][:]
    yte = prices[NUM_TRAIN+NUM_VAL:NUM_TOTAL]
    return (xtr,ytr,xva,yva,xte,yte)


#%% Extra code: Trying to read webpage directly
#import requests
#from bs4 import BeautifulSoup
##import urllib2
##user_agent = 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_4; en-US) AppleWebKit/534.3 (KHTML, like Gecko) Chrome/6.0.472.63 Safari/534.3'
##headers = { 'User-Agent' : user_agent }
#url = "http://sofifa.com/players?gender=0&pn%5B0%5D=27&pn%5B1%5D=25&pn%5B2%5D=23&pn%5B3%5D=22&pn%5B4%5D=21&pn%5B5%5D=20&pn%5B6%5D=18&pn%5B7%5D=16&pn%5B8%5D=14&pn%5B9%5D=12&pn%5B10%5D=10&pn%5B11%5D=8&pn%5B12%5D=7&pn%5B13%5D=5&pn%5B14%5D=3&pn%5B15%5D=2&col=vl&sort=desc&offset=0"
##req = urllib2.Request(url, None, headers)
##response = urllib2.urlopen(req)
#r = requests.get(url)
#soup = BeautifulSoup(r.text)
#f = open('short_p1.txt','wb')
##f.write(response.read())
#f.write(r.content)
##response.close()
#r.close()
#f.close()
