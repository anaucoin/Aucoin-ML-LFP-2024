import pandas as pd
import torch
import torchvision
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy.random as rand
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
from scipy import signal
import csv
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data as dta 
from sklearn import svm
import sys 

import gc
gc.collect()

# +
import argparse
parser = argparse.ArgumentParser(description='LFPnet GED training')
parser.add_argument('--save', action='store_true')
parser.add_argument('--nosave', dest='save', action='store_false')
parser.set_defaults(save=True)

parser.add_argument('--null', dest = 'null', action='store_true')
parser.set_defaults(null=False)

parser.add_argument('--raw', action='store_true')
parser.add_argument('--rel', dest='raw', action='store_false')
parser.set_defaults(raw=True)

parser.add_argument('--d','--day', metavar='d', default=0, type=int, help='date of experiment')
parser.add_argument('--r', '--region',metavar='region', default='amygdala', type=str, help='region (i.e. amygdala vs cortex)')
parser.add_argument('--w', metavar='w', default=5, type=int, help='Morlet wavelet transform main frequency')
parser.add_argument('--epochs',metavar='epochs', default=60, type=int, help='number of total epochs to run')
parser.add_argument('--lr', default=.0001, type=float, metavar='lr',
                                        help='learning rate')
parser.add_argument('--bs', default=5, type=int, metavar='batchsize', help='training batch size')
parser.add_argument('--nwin', default=256, type=int, metavar='nwin', help='number of windows for spectrogram')
parser.add_argument('--divfs', default=10, type=int, metavar='divfs',
                                        help='scaling number to determine total frequencies in spectrogram')
parser.add_argument('--nuclei', default='B', type=str, metavar='nuclei',
                                        help='define which nuclei to use')
parser.add_argument('--csv', default='singletrain.csv', type=str, metavar='csvname',
                                        help='output csv file name')
parser.add_argument('--scaler', default='minmax', type=str, metavar='scaler',
                                        help='choice of minmax or stand. scaler')
parser.add_argument('--mach', default='hpc', type=str, metavar='mach',
                                        help='identifies which machine code is running on')
parser.add_argument('--monkey', default='Aster', type=str, metavar='monkey',
                                        help='identifies which monkey')
parser.add_argument('--seg', default='base', type=str, metavar='seg', help='segment of lfp (e.g. base, stim, full)')


# -

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# +
def getdatasplits(splits,perms, puffind, touchind, data):
    '''
        Randomly split data into train and test set. This ensures validation and test sets have equal rep. 
        however, need bootstrapping for training set to have same amount of data?
    '''
    numtouch = len(touchind)
    numpuff = len(puffind)
    numtest = int(numpuff*splits[2])
    numval = int(numpuff*splits[1])
    numtrain_puff = numpuff - numtest - numval
    numtrain_touch = numtouch - numtest - numval
    
    wid = np.shape(data)[1]
    leng = np.shape(data)[2]
    
    permpuffind = puffind[perms[0]]
    permtouchind = touchind[perms[1]]
    
    trainset = torch.randn(((numtrain_puff + numtrain_touch),1,wid, leng))
    valset = torch.randn((2*numval,1,wid, leng))
    testset = torch.randn((2*numtest,1,wid, leng))
    
    trainset[:numtrain_puff, 0, :,:] = torch.from_numpy(data[permpuffind[:numtrain_puff],:,:])
    trainset[numtrain_puff:,0,:,:] = torch.from_numpy(data[permtouchind[:numtrain_touch],:,:])
    valset[:numval, 0,:,:] = torch.from_numpy(data[permpuffind[numtrain_puff:numtrain_puff+numval],:,:])
    valset[numval:, 0,:,:] = torch.from_numpy(data[permtouchind[numtrain_touch:numtrain_touch+numval],:,:])
    testset[:numtest,0,:,:] = torch.from_numpy(data[permpuffind[numtrain_puff+numval:],:,:])
    testset[numtest:,0,:,:] = torch.from_numpy(data[permtouchind[numtrain_touch+numval:],:,:])
    
    traintags = np.ones(numtrain_puff + numtrain_touch)
    traintags[:numtrain_puff] = 0*traintags[:numtrain_puff]
    
    valtags = np.ones(2*numval)
    valtags[:numval] = 0*valtags[:numval]
    
    testtags = np.ones(2*numtest)
    testtags[:numtest] = 0*testtags[:numtest]
    
    
    traintags = torch.LongTensor(traintags)
    valtags = torch.LongTensor(valtags)
    testtags = torch.LongTensor(testtags)
    
    return trainset, valset, testset, traintags, valtags, testtags

def get_anatomy_cs(day_code, region, nuclei):
    header = [None]*65
    header[0] = 'dates'
    for i in range(1,65):
        header[i] = int(i-1)
    filename = f'{region}_anatomy.csv'
    df = pd.read_csv(filename,header=None, skiprows=1, names=header)
    return np.where(df.iloc[day_code, 1:].values == nuclei)[0]


# -

import time
startTime = time.time()

if __name__ ==  '__main__':


    use_cuda = torch.cuda.is_available()
    print('CUDA is available') if use_cuda else print('CUDA is unavailable')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_cuda == True: 
        args = parser.parse_args()
    else: 
        args = Namespace(d=0, r='amygdala',w = 25,epochs = 30,lr = 0.0001, 
                         bs = 30, nwin = 128, divfs = 20, nuclei='C', monkey = 'Aster', seg= 'base',
                         csv = 'svm.csv', scaler='minmax', null=True, save = False, raw = True)

    # set manual random seed (useful for debugging/analysis during training)
    now = time.time()
    torch.manual_seed(now)
    print(now)

    if args.monkey == 'Aster':
        dates = ['060619','061019','061319','061819','062019','062419','070819','071019','071219','071619','071819','080619'];
    if args.monkey == 'Cotton':
        dates = ['010520', '010720', '021220', '021620', '022120', '022820', '030620', '031020', '121619']
    if args.monkey == "Saguaro":
        dates = ['42221', '50521', '51021', '51221', '51421', '51721', '51921']
    num_days = len(dates)

    sessdate = dates[args.d]

    # thelio pwd: /u1/aucoin/extradrive1/aucoin/python_lfpnet/
    #if use_cuda:
    #    filename = f'/u1/aucoin/extradrive1/aucoin/python_lfpnet/cleanbase_{sessdate}_{args.r}.npz'
    #else:
    filename = f'clean{args.seg}_{sessdate}_{args.r}.npz'

    with np.load(filename, allow_pickle=True) as alldata:
        olddata = alldata['data']
        df = pd.DataFrame(alldata['df'], columns = ['modality', 'loc', 'stimstart', 'stimend','is BadT', 'is Block end', 'is long trial','is wild trial'])
        contactinds = alldata['cs']
        badtrialinds = alldata['badtrialinds']
        buffer = alldata['buffer']

    if args.r == '3b':
       # contactinds = cortexcs
        contactinds = contactinds
    else:
        # contactinds = amycs
        contactinds = contactinds - contactinds[0] 

    dt = .001
    if args.seg == 'base':
        t = np.arange((-buffer - np.shape(olddata)[2])/1000, -(buffer/1000), dt)[0:olddata.shape[2]]
    elif args.seg == 'stim':
        t = np.arange(-buffer, np.shape(olddata)[2] - buffer)

    #collecting the right cells 
    data = np.take(olddata, contactinds, axis=1)
    
    # meanzero data 
    data = data - np.mean(data, axis=2)[:,:,None]
    
    #make lfp relative to contact-average 
    if not args.raw:
        data = data - np.mean(data, axis=1)[:,None]

    if args.monkey == 'Aster':
        nucinds = get_anatomy_cs(args.d, args.r, args.nuclei)
    if args.monkey == 'Cotton': 
        nucinds = get_anatomy_cs(args.d + 12, args.r, args.nuclei)
    if args.monkey == 'Saguaro': 
        nucinds = get_anatomy_cs(args.d + 21, args.r, args.nuclei)

    data = np.take(olddata, nucinds, axis=1)

    # meanzero data 
    data = data - np.mean(data, axis=2)[:,:,None]

    #make lfp relative to contact-average 
    if not args.raw:
        data = data - np.mean(data, axis=1)[:,None]

    #reshape data to be trial X time 
    modeldata = np.reshape(data, (data.shape[0]*data.shape[1], -1))

    if args.null: 
        #randomly shuffle the labels (keeps same shuffling for each contact tho)
        if args.seg == 'stim':
            tags = np.copy(df['modality'])[np.repeat(np.array([i for i in range(len(df)) if i not in badtrialinds]), len(nucinds), axis=0)]
            tags = tags[list(rand.permutation(len(tags)))]
        else:
            tags = np.copy(df['modality'])[np.repeat(np.array([i for i in range(len(df)-1) if i not in badtrialinds]), len(nucinds), axis=0)]
            tags = tags[list(rand.permutation(len(tags)))]
    else: 
        if args.seg == 'stim':
            tags = df['modality'][np.repeat(np.array([i for i in range(len(df)) if i not in badtrialinds]), len(nucinds), axis=0)]
        else:
            tags = df['modality'][np.repeat(np.array([i for i in range(len(df)-1) if i not in badtrialinds]), len(nucinds), axis=0)]
            
    puffind = np.where(tags == 'puff')
    touchind = np.where(tags == 'touch')
    numpuff = np.size(puffind)
    numtouch = np.size(touchind)

    fs = 1/dt
    freq = np.linspace(1,fs/args.divfs, args.nwin)
    widths = args.w*fs / (2*freq*np.pi)

    print('Computing spectrograms...')
    numtrials = modeldata.shape[0]
    lent = len(t)
    specdata = np.empty((numtrials, args.nwin,lent), dtype= np.float64)
    for i in range(numtrials):
        specdata[i,:] = abs(signal.cwt(modeldata[i,:], signal.morlet2, widths, w=args.w))

    # set up data for training 
    splits = [.8, .1, .1]
    perms =  (list(rand.permutation(numpuff)), list(rand.permutation(numtouch)))

    print('Splitting data...')
    trainset, valset, testset, traintags, valtags, testtags = getdatasplits(splits,perms,puffind[0], touchind[0], specdata)

    numtouchtrain = int(sum(traintags))
    numpufftrain = int(len(traintags)- numtouchtrain)

    diff = numpufftrain-numtouchtrain
    inds = np.where(traintags ==1)[0]

    tocat = trainset[np.mod(np.random.permutation(int(np.ceil((diff/numtouchtrain))*numtouchtrain)), numtouchtrain)[:diff],:,:,:]

    trainset = torch.cat((trainset, tocat))
    traintags = torch.cat((traintags, torch.LongTensor(np.ones(diff))))

    if args.scaler == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        trainset[:,0,:,:] = torch.from_numpy(scaler.fit_transform(trainset[:,0,:,:].reshape(-1, trainset[:,0,:,:].shape[-1])).reshape(trainset[:,0,:,:].shape))
        valset[:,0,:,:] = torch.from_numpy(scaler.transform(valset[:,0,:,:].reshape(-1, valset[:,0,:,:].shape[-1])).reshape(valset[:,0,:,:].shape))
        testset[:,0,:,:] = torch.from_numpy(scaler.transform(testset[:,0,:,:].reshape(-1, testset[:,0,:,:].shape[-1])).reshape(testset[:,0,:,:].shape))
    elif args.scaler == 'stand':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        trainset[:,0,:,:] = torch.from_numpy(scaler.fit_transform(trainset[:,0,:,:].reshape(-1, trainset[:,0,:,:].shape[-1])).reshape(trainset[:,0,:,:].shape))
        valset[:,0,:,:] = torch.from_numpy(scaler.transform(valset[:,0,:,:].reshape(-1, valset[:,0,:,:].shape[-1])).reshape(valset[:,0,:,:].shape))
        testset[:,0,:,:] = torch.from_numpy(scaler.transform(testset[:,0,:,:].reshape(-1, testset[:,0,:,:].shape[-1])).reshape(testset[:,0,:,:].shape))
    else: 
        print('Did not scale data.')

    trainset = trainset.detach().numpy()
    traintags = traintags.detach().numpy()

    flattrain = trainset.reshape((trainset.shape[0], -1))

    print(f'{flattrain.shape} -- {traintags.shape}')

    Cs = [10**i for i in np.arange(-6,4,1, dtype=float)]
    for C in Cs:
        clf = svm.LinearSVC(max_iter=1000, dual = True, C=C)

        clf.fit(flattrain, traintags)

        dur = time.time() - now
        testset = testset.detach().numpy().reshape((testset.shape[0], -1))
        testtags = testtags.detach().numpy()

        tperm = rand.permutation(len(testtags))

        pred = clf.predict(testset[tperm])

        acc = sum(pred == testtags[tperm])/len(testtags)

        pinds = np.where(testtags == 0)[0]
        tinds = np.where(testtags == 1)[0]

        predpuff = clf.predict(testset[pinds])
        predtouch = clf.predict(testset[tinds])

        puffacc = 1 - predpuff.sum()/len(pinds)
        touchacc = predtouch.sum()/len(tinds)

        with open(args.csv, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([sessdate, args.d, args.r, args.raw, args.nuclei, nucinds, args.scaler, args.w, 
                             args.nwin,args.divfs, (freq[0],freq[-1]),
                             100*acc, 100*touchacc, 100*puffacc, C])



