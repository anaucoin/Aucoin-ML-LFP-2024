# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: lfpnet
#     language: python
#     name: lfpnet
# ---

import torch
import torchvision
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import numpy.random as rand
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
from scipy import signal
import csv
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset

import gc
gc.collect()

# +
import argparse
parser = argparse.ArgumentParser(description='LFPnet GED training')
parser.add_argument('--save', action='store_true')
parser.add_argument('--nosave', dest='save', action='store_false')
parser.set_defaults(save=True)
parser.add_argument('--stand', action='store_true')
parser.add_argument('--instand', dest='stand', action='store_false')
parser.set_defaults(stand=False)


parser.add_argument('--d','--day', metavar='d', default=2, type=int, help='date of experiment')
parser.add_argument('--r', '--region',metavar='region', default='amygdala', type=str, help='region (i.e. amygdala vs cortex)')
parser.add_argument('--bm','--basemodality', default='touch', type=str, metavar='basemodality', help='modality of baseline')
parser.add_argument('--seg', default='base', type=str, metavar='seg', help='segment of lfp (e.g. base, stim, full)')
parser.add_argument('--w', metavar='w', default=5, type=int, help='Morlet wavelet transform main frequency')
parser.add_argument('--epochs',metavar='epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--lr', default=.001, type=float, metavar='lr',
                                        help='learning rate')
parser.add_argument('--bs', default=15, type=int, metavar='batchsize', help='training batch size')
parser.add_argument('--k', default=5, type=int, metavar='k', help='number of k-folds for validation')
parser.add_argument('--nwin', default=256, type=int, metavar='nwin', help='number of windows for spectrogram')
parser.add_argument('--divfs', default=10, type=int, metavar='divfs',
                                        help='scaling number to determine total frequencies in spectrogram')
parser.add_argument('--m', default='ged', type=str, metavar='model',
                                        help='sets the reduced order model type')
parser.add_argument('--csv', default='lfpnetlog', type=str, metavar='csvname',
                                        help='output csv file name')


# -

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# +
#some useful functions
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        # print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()
    
def getfinsize(wid, leng, constride, poolstride):
    finwid = wid 
    finleng = leng
    confact = 2*(2*constride) # 2*constride decrease per layer * 2 layers per block 
    for i in range(3): #total of 3 blocks
        finwid = int((finwid-confact)/poolstride)
        finleng = int((finleng-confact)/poolstride)
    return finwid*finleng

def getdatasplits(splits, perms, puff, touch):
    numtouch = np.shape(touch)[0]
    numtrain = int(numtouch*splits[0])
    numtest = numtouch - numtrain
    
    wid = np.shape(puff)[1]
    leng = np.shape(puff)[2]
    

    permpuff = puff[perms[0], :,:]
    permtouch = touch[perms[1], :,:]
    
    trainset = torch.randn((2*numtrain,1,wid, leng))
    testset = torch.randn((2*numtest,1,wid, leng))
    
    trainset[:numtrain, 0, :,:] = torch.from_numpy(permpuff[:numtrain,:,:])
    trainset[numtrain:,0,:,:] = torch.from_numpy(permtouch[:numtrain,:,:])
    testset[:numtest,0,:,:] = torch.from_numpy(permpuff[numtrain:numtouch,:,:])
    testset[numtest:,0,:,:] = torch.from_numpy(permtouch[numtrain:numtouch,:,:])
    
    traintags = np.ones(2*numtrain)
    traintags[:numtrain] = 0*traintags[:numtrain]
    
    testtags = np.ones(2*numtest)
    testtags[:numtest] = 0*testtags[:numtest]
    
    traintags = torch.LongTensor(traintags)
    testtags = torch.LongTensor(testtags)
    
    return trainset, testset, traintags, testtags


# -

class LFPnet(nn.Module):
    def __init__(self, finsize = 28*59, constride = 1, poolstride = 2,**kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, 3, stride=constride) # layer 1 
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24,48, 3, stride=constride) # layer 2 
        self.maxpool1 = nn.MaxPool2d((2,2),stride=poolstride) #layer 3 
        self.bn2 = nn.BatchNorm2d(48)
        
        self.conv3 = nn.Conv2d(48, 48, 3, stride=constride) # layer 4 
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 24, 3, stride=constride) # layer 5 
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=poolstride) #layer 6 
        self.bn4 = nn.BatchNorm2d(24)
        
        self.conv5 = nn.Conv2d(24, 24, 3, stride=constride) # layer 7 
        self.bn5 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24, 48, 3, stride=constride) # layer 8 
        self.maxpool3 = nn.MaxPool2d((2, 2), stride=poolstride) #layer 9 
        self.bn6 = nn.BatchNorm2d(48)
        

        self.fc1 = nn.Linear(finsize*48, 64) #layer 10 
        self.drop = nn.Dropout(p=0.5)                             
        self.fc2 = nn.Linear(64, 2) #layer 11
    
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.maxpool1(F.relu(self.conv2(x)))
        x = self.bn2(x)
        
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.maxpool2(F.relu(self.conv4(x)))
        x = self.bn4(x)
                                     
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.maxpool3(F.relu(self.conv6(x)))
        x = self.bn6(x)
     
        x = torch.flatten(x,1) #flatten all except batch dim                            
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        return x

import time
startTime = time.time()

if __name__ ==  '__main__':

    use_cuda = torch.cuda.is_available()
    print('CUDA is available') if use_cuda else print('CUDA is unavailable')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_cuda: 
        args = parser.parse_args()
    else: 
        args = Namespace(d=4, r='amygdala', bm = 'touch',seg='base', w = 5, epochs = 1,lr = 0.001,k=2, bs = 15, nwin = 256, divfs = 50, m= 'ged',csv = 'lfpnetlog.csv', stand = False, save = False)

    # set manual random seed (useful for debugging/analysis during training)
    #torch.manual_seed(42)

    dates = ['060619','061019','061319','061819','062019','062419','070819','071019','071219','071619','071819','080619'];
    num_days = len(dates)

    sessdate = dates[args.d]

    # thelio pwd: /u1/aucoin/extradrive1/aucoin/python_lfpnet/
    if use_cuda:
        filename = '/u1/aucoin/extradrive1/aucoin/python_lfpnet/cleanAster_' + sessdate +'_fulllfp.npz'
    else:
        filename = 'cleanAster_' + sessdate +'_fulllfp.npz'

    with np.load(filename, allow_pickle=True) as alldata:
        olddata = alldata['data']
        df = pd.DataFrame(alldata['df'], columns = ['modality', 'loc', 'stimstart', 'stimend'])

    tags = df['modality']

    if use_cuda:
        if args.r == 'amygdala':
            sfilename = '/u1/aucoin/extradrive1/aucoin/python_lfpnet/RMC_pearsonrged_maps.npz'
        else:
            sfilename = '/u1/aucoin/extradrive1/aucoin/python_lfpnet/RMC_pearsonrged_maps_3b.npz' 
    else:
        if args.r == 'amygdala':
            sfilename = 'RMC_pearsonrged_maps.npz'
        else:
            sfilename = 'RMC_pearsonrged_maps_3b.npz' 

    with np.load(sfilename, allow_pickle=True) as alldata:
        puffmaps = alldata['puffmaps']
        touchmaps = alldata['touchmaps']

    if args.bm == 'touch':
        gedfilt = touchmaps[args.d]
    else:
        gedfilt = puffmaps[args.d]

    oldn = np.shape(olddata)[2]
    oldt, dt = np.linspace(-1,1,oldn, retstep = True)

    if args.seg == 'full':
        bind = range(len(oldt))
    elif args.seg == 'base':
        bind = range(200,701) 
    else:
        bind = range(1000,len(oldt))

    t = oldt[bind]

    #collecting the right cells 
    numchan = np.shape(gedfilt)[0]
    if args.r == 'amygdala':
        data = olddata[:,-numchan:,bind]
    else: 
        data = olddata[:,:numchan:,bind]

    # meanzero data 
    data = data - np.mean(data, axis=2)[:,:,None]

    gedtimeseries = np.dot(gedfilt,data[0,:,:])

    numtrials = np.shape(data)[0]
    lent = len(t)
    geddata = np.empty((numtrials,lent), dtype= np.float64)

    for i in range(numtrials):
        geddata[i,:] = np.dot(gedfilt,data[i,:,:])

    fs = 1/dt
    freq = np.linspace(1,fs/args.divfs, args.nwin)
    widths = args.w*fs / (2*freq*np.pi)

    specdata = np.empty((numtrials, args.nwin,lent), dtype= np.float64)
    for i in range(numtrials):
        specdata[i,:] = abs(signal.cwt(geddata[i,:], signal.morlet2, widths, w=args.w))
    if args.stand:
        specdata = (specdata - specdata.min())/(specdata.max() - specdata.min())

    puffind = np.where(tags == 'puff')
    touchind = np.where(tags == 'touch')
    numpuff = np.size(puffind)
    numtouch = np.size(touchind)
    diff = numpuff-numtouch
    
    #sample touch to have same number of trials as puff 
    puff = specdata[puffind[0],:]
    touch = np.vstack((specdata[touchind[0],:], specdata[np.random.permutation(numtouch)[:diff],:]))
    numtouch = numpuff

    if not args.stand:
        puff = (puff - puff.min())/(puff.max() - puff.min())
        touch = (touch - touch.min())/(touch.max() - touch.min())

    # set up data for training 
    splits = [.8, .2]
    perms =  (rand.permutation(numpuff), rand.permutation(numtouch))

    trainset, testset, traintags, testtags = getdatasplits(splits, perms, puff, touch)

    wid = np.shape(trainset)[2]
    leng = np.shape(trainset)[3]
    constride = 1
    poolstride = 2

    finsize = getfinsize(wid, leng, constride, poolstride)

    traindata = torch.utils.data.TensorDataset(trainset, traintags)
    testdata = torch.utils.data.TensorDataset(testset, testtags)
    dataset = ConcatDataset([traindata, testdata])

    # set up configuration for k-fold validation 
    results = {}
    class_results = {} 
    kfold = KFold(n_splits=args.k, shuffle = True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Starting FOLD {fold}\n')
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.bs, sampler = train_subsampler)
        
        testloader = torch.utils.data.DataLoader(
        dataset, batch_size=25, sampler=test_subsampler)
    
        net = LFPnet(finsize = finsize, constride = constride, poolstride = poolstride)
        net.apply(reset_weights)
        # move network to GPU 
        net = net.to(device)
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
        for e in range(args.epochs): 
            # Print epoch
            print(f'Starting epoch {e+1}')
            train_loss = 0.0
            net.train() #let the model know we are in training mode 
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # move inputs and labels to gpu 
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + compute loss + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                train_loss += loss.item()
                if i % 10 == 9:
                   # print('Loss after mini-batch %5d: %.3f' %
                          #(i + 1, train_loss / 5 ))
                    train_loss = 0.0

            print('Training finished. Saving model and starting testing...')
            if args.save:
                    # Saving the model
                    PATH = f'{sessdate}/{args.r[:2]}{args.seg}{sessdate}{args.m}_{args.bm[0]}{args.bs}_{args.lr}_{args.divfs}_{args.nwin}fold_{fold}.pth'
                    torch.save(net.state_dict(), PATH)

            correct, total = 0, 0
            classes = ('puff', 'touch')
            # prepare to count predictions for each class
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            net.eval() 

            with torch.no_grad():
                for i, data in enumerate(testloader, 0):
                    inputs, labels = data 

                    # move inputs and labels to gpu 
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # collect the correct predictions for each class
                    for label, prediction in zip(labels, predicted):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1

            # Print accuracy
            #print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                #print(f'Accuracy for class: {classname:5s} is {accuracy:.2f} %')
            #print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
            class_results[fold] = {'touch acc' : 100 * float(correct_pred['touch']) / total_pred['touch'],
                                   'puff acc': 100 * float(correct_pred['puff']) / total_pred['puff']}

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {args.k} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')
    puffsum = 0.0
    touchsum = 0.0
    for folds, class_pred in class_results.items():
        touchsum += class_pred['touch acc']
        puffsum += class_pred['puff acc']
    touchacc = touchsum/args.k
    puffacc = puffsum/args.k

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    if args.save:
        with open(args.csv, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([sessdate, args.d, args.r, args.m, args.bm,
                             args.seg,args.stand, args.w, args.nwin,
                             args.divfs, (freq[0],freq[-1]),args.lr, 
                             args.bs, args.k, args.epochs, sum/len(results.items()),
                             results, class_results,  PATH ])
