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

import gc
gc.collect()

# +
import argparse
parser = argparse.ArgumentParser(description='LFPnet GED training')
parser.add_argument('--save', action='store_true')
parser.add_argument('--nosave', dest='save', action='store_false')
parser.set_defaults(save=True)

parser.add_argument('--d','--day', metavar='d', default=2, type=int, help='date of experiment')
parser.add_argument('--r', '--region',metavar='region', default='amygdala', type=str, help='region (i.e. amygdala vs cortex)')
parser.add_argument('--bm','--basemodality', default='touch', type=str, metavar='basemodality', help='modality of baseline')
parser.add_argument('--w', metavar='w', default=5, type=int, help='Morlet wavelet transform main frequency')
parser.add_argument('--epochs',metavar='epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--bs', default=10, type=int, metavar='batchsize', help='training batch size')
parser.add_argument('--nwin', default=256, type=int, metavar='nwin', help='number of windows for spectrogram')
parser.add_argument('--divfs', default=10, type=int, metavar='divfs',
                                        help='scaling number to determine total frequencies in spectrogram')
parser.add_argument('--m', default='ged', type=str, metavar='model',
                                        help='sets the reduced order model type')


# -

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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


# +
#some useful functions
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
    numval = int(numtouch*splits[1])
    numtest = numtouch - numtrain - numval
    
    wid = np.shape(puff)[1]
    leng = np.shape(puff)[2]
    

    permpuff = puff[perms[0], :,:]
    permtouch = touch[perms[1], :,:]
    
    trainset = torch.randn((2*numtrain,1,wid, leng))
    valset = torch.randn((2*numval,1,wid, leng))
    testset = torch.randn((2*numtest,1,wid, leng))
    
    trainset[:numtrain, 0, :,:] = torch.from_numpy(permpuff[:numtrain,:,:])
    trainset[numtrain:,0,:,:] = torch.from_numpy(permtouch[:numtrain,:,:])
    valset[:numval, 0,:,:] = torch.from_numpy(permpuff[numtrain:numtrain+numval,:,:])
    valset[numval:, 0,:,:] = torch.from_numpy(permtouch[numtrain:numtrain+numval,:,:])
    testset[:numtest,0,:,:] = torch.from_numpy(permpuff[numtrain+numval:numtouch,:,:])
    testset[numtest:,0,:,:] = torch.from_numpy(permtouch[numtrain+numval:numtouch,:,:])
    
    traintags = np.ones(2*numtrain)
    traintags[:numtrain] = 0*traintags[:numtrain]
    
    valtags = np.ones(2*numval)
    valtags[:numval] = 0*valtags[:numval]
    
    testtags = np.ones(2*numtest)
    testtags[:numtest] = 0*testtags[:numtest]
    
    
    traintags = torch.LongTensor(traintags)
    valtags = torch.LongTensor(valtags)
    testtags = torch.LongTensor(testtags)
    
    return trainset, valset, testset, traintags, valtags, testtags
# -

import time
startTime = time.time()

if __name__ ==  '__main__':

    use_cuda = torch.cuda.is_available()
    print('CUDA is available') if use_cuda else print('CUDA is unavailable')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_cuda: 
        args = parser.parse_args()
    else: 
        args = Namespace(d=5, r='3b', bm = 'touch', w = 5, epochs = 15, bs = 30, nwin = 256, divfs = 50, m= 'ged', save = False)

    dates = ['060619','061019','061319','061819','062019','062419','070819','071019','071219','071619','071819','080619'];
    num_days = len(dates)
    train_acc = np.empty(num_days)
    test_acc = np.empty(num_days)

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

    bind = range(200,701)
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

    puffind = np.where(tags == 'puff')
    touchind = np.where(tags == 'touch')
    numpuff = np.size(puffind)
    numtouch = np.size(touchind)

    puff = np.empty((numpuff, args.nwin,lent), dtype= np.float64)
    touch = np.empty((numtouch, args.nwin,lent), dtype = np.float64)

    for i in range(numpuff):
        sig = geddata[puffind[0][i],:]
        puff[i,:,:] = abs(signal.cwt(sig, signal.morlet2, widths, w=args.w))

    for i in range(numtouch):
        sig = geddata[touchind[0][i],:]
        touch[i,:,:] = abs(signal.cwt(sig, signal.morlet2, widths, w=args.w))

    mn = np.concatenate((puff,touch), axis=0).mean()
    std = np.concatenate((puff,touch), axis=0).std()

    puff = (puff-mn)/std
    touch = (touch-mn)/std

    # set up data for training 
    splits = [.8, .1, .1]
    perms =  (rand.permutation(numpuff), rand.permutation(numtouch))

    trainset, valset, testset, traintags, valtags, testtags = getdatasplits(splits, perms, puff, touch)

    ptrain = rand.permutation(np.shape(traintags)[0])
    pval = rand.permutation(np.shape(valtags)[0])
    ptest = rand.permutation(np.shape(testtags)[0])
    
    trainset = trainset[ptrain,:,:,:]
    traintags = traintags[ptrain]
    valset = valset[pval, :, :,:]
    valtags = valtags[pval]
    testset = testset[ptest,:,:,:]
    testtags = testtags[ptest]

    wid = np.shape(trainset)[2]
    leng = np.shape(trainset)[3]
    constride = 1
    poolstride = 2

    finsize = getfinsize(wid, leng, constride, poolstride)

    net = LFPnet(finsize = finsize, constride = constride, poolstride = poolstride)

    # move network to GPU 
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    traindata = torch.utils.data.TensorDataset(trainset, traintags)
    valdata = torch.utils.data.TensorDataset(valset, valtags)
    testdata = torch.utils.data.TensorDataset(testset, testtags)

    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=args.bs, shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=20, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=20, shuffle=False, num_workers=1)

    epochs = args.epochs
    min_valid_loss = np.inf
    
    for e in range(args.epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        net.train() #let the model know we are in training mode 
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # move inputs and labels to gpu 
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()

        valid_loss = 0.0 
        net.eval() #let model know we're in evaluation mode
        for i, data in enumerate(valloader, 0):
            inputs, labels = data 
            
            # move inputs and labels to gpu 
            inputs, labels = inputs.to(device), labels.to(device)
            target = net(inputs)
            loss = criterion(target, labels)
            valid_loss += loss.item()
                
                
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(valloader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            
            # Saving State Dict
            #torch.save(net.state_dict(), 'saved_model.pth')

    print('Finished Training')

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    if args.save:
        PATH = sessdate + '/' + sessdate+args.m +'_df'+ str(args.divfs) + 'divfs' + str(args.divfs)+ 'nwin' + str(args.nwin)+ '.pth'
        torch.save(net.state_dict(), PATH)
    
        sfilename = sessdate + '/' + sessdate+args.m +'_df'+ str(args.divfs) + 'divfs' + str(args.divfs)+ 'nwin' + str(args.nwin)+'.npz'
        np.savez(sfilename, nepochs = args.epochs,perms=perms, splits = splits, nwin=args.nwin, w = args.w, t = t,batchsize=args.bs, allow_pickle=True)


    correct = 0
    total = 0

    classes = ('puff', 'touch')
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total : .2f} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.2f} %')

    if args.save:
        with open('lfpnetlog.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([sessdate, args.d, args.r, args.m, args.bm, args.w, args.nwin,args.divfs, (freq[0],freq[-1]), args.bs, args.epochs,100 * correct // total,100*float(correct_pred['touch']) / total_pred['touch'],100* float(correct_pred['puff']) / total_pred['puff'],  PATH ])
