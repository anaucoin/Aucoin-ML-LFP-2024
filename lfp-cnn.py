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

import gc
gc.collect()

# +
import argparse
parser = argparse.ArgumentParser(description='LFP-CNN training')
parser.add_argument('--save', action='store_true')
parser.add_argument('--nosave', dest='save', action='store_false')
parser.set_defaults(save=True)

parser.add_argument('--null', dest = 'null', action='store_true')
parser.set_defaults(null=False)

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
parser.add_argument('--monkey', default='Aster', type=str, metavar='monkey',
                                        help='identifies which monkey')
parser.add_argument('--seg', default='base', type=str, metavar='seg', help='segment of lfp (e.g. base, stim, full)')


# -

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# +
class LFPnet(nn.Module):
    def __init__(self, finsize = 28*59, constride = 1, poolstride = 2,**kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, 3, stride=constride) # layer 1 
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24,48, 3, stride=constride) # layer 2 
        self.maxpool1 = nn.MaxPool2d((2,2),stride=poolstride) #layer 3 
        self.bn2 = nn.BatchNorm2d(48)
        
#         self.conv3 = nn.Conv2d(48, 48, 3, stride=constride) # layer 4 
#         self.bn3 = nn.BatchNorm2d(48)
#         self.conv4 = nn.Conv2d(48, 24, 3, stride=constride) # layer 5 
#         self.maxpool2 = nn.MaxPool2d((2, 2), stride=poolstride) #layer 6 
#         self.bn4 = nn.BatchNorm2d(24)
        
#         self.conv5 = nn.Conv2d(24, 24, 3, stride=constride) # layer 7 
#         self.bn5 = nn.BatchNorm2d(24)
#         self.conv6 = nn.Conv2d(24, 48, 3, stride=constride) # layer 8 
#         self.maxpool3 = nn.MaxPool2d((2, 2), stride=poolstride) #layer 9 
#         self.bn6 = nn.BatchNorm2d(48)
        

        self.fc1 = nn.Linear(finsize*48, 64) #layer 10 
        self.drop = nn.Dropout(p=0.5)                             
        self.fc2 = nn.Linear(64, 2) #layer 11
    
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.maxpool1(F.relu(self.conv2(x)))
        x = self.bn2(x)
        
#         x = self.bn3(F.relu(self.conv3(x)))
#         x = self.maxpool2(F.relu(self.conv4(x)))
#         x = self.bn4(x)
                                     
#         x = self.bn5(F.relu(self.conv5(x)))
#         x = self.maxpool3(F.relu(self.conv6(x)))
#         x = self.bn6(x)
     
        x = torch.flatten(x,1) #flatten all except batch dim                            
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        # add linear layer regression to fix 
        
        return x


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
        
def getfinsize(wid, leng, constride, poolstride, nlayers):
    '''
      Determine fully connected layer size based on 
      model configuration. Note: This assumes network has specific layer blocks
      (e.g. CONV2D, BatchNorm, CONV2D,MaxPool, BatchNorm)
    '''
    finwid = wid 
    finleng = leng
    confact = 2*(2*constride) # 2*constride decrease per layer * 2 layers per block 
    for i in range(nlayers): #total of blocks
        finwid = int((finwid-confact)/poolstride)
        finleng = int((finleng-confact)/poolstride)
    return finwid*finleng

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
        args = Namespace(d=3, r='3b',w = 25,epochs = 30,lr = 0.0001, 
                         bs = 30, nwin = 128, divfs = 20, nuclei='3B', monkey = 'Aster', seg= 'stim',
                         csv = 'single.csv', scaler='stand', null=True, save = False, raw = True)

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

    filename = f'clean{args.seg}_{sessdate}_{args.r}.npz'

    with np.load(filename, allow_pickle=True) as alldata:
        olddata = alldata['data']
        df = pd.DataFrame(alldata['df'], columns = ['modality', 'loc', 'stimstart', 'stimend','is BadT', 'is Block end', 'is long trial','is wild trial'])
        contactinds = alldata['cs']
        badtrialinds = alldata['badtrialinds']
        buffer = alldata['buffer']

    dt = .001
    if args.seg == 'base':
        t = np.arange((-buffer - np.shape(olddata)[2])/1000, -(buffer/1000), dt)[0:olddata.shape[2]]
    elif args.seg == 'stim':
        t = np.arange(-buffer, np.shape(olddata)[2] - buffer)

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

    wid = np.shape(trainset)[2]
    leng = np.shape(trainset)[3]
    constride = 1
    poolstride = 2

    nlayers = 1
    finsize = getfinsize(wid, leng, constride, poolstride, nlayers)

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

    traindata = torch.utils.data.TensorDataset(trainset, traintags)
    valdata = torch.utils.data.TensorDataset(valset, valtags)
    testdata = torch.utils.data.TensorDataset(testset, testtags)

    net = LFPnet(finsize = finsize, constride = constride, poolstride = poolstride)

    # move network to GPU 
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=args.bs, shuffle=True, num_workers=1)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=args.bs, shuffle=True, num_workers=1)

    if args.save:
        sfilename = 'savestats/' + args.r[:2]+args.seg+ sessdate+str(args.nuclei) +'_' +str(args.bs) +'_' + str(args.lr) + '_'+ str(args.divfs) + '_' + str(args.nwin)+ '.npz'
        np.savez_compressed(sfilename, trainset = trainset.numpy(), traintags = traintags.numpy(), testset = testset.numpy(), testtags = testtags.numpy(), valset = valset.numpy(), valtags = valtags.numpy(), allow_pickle=True)
    print("Saving data splits and starting training.")

    trainhistory = []
    valhistory = []
    
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
            
        trainhistory.append(train_loss)

        valid_loss = 0.0 
        net.eval() #let model know we're in evaluation mode
        for i, data in enumerate(valloader, 0):
            inputs, labels = data 
            
            # move inputs and labels to gpu 
            inputs, labels = inputs.to(device), labels.to(device)
            target = net(inputs)
            loss = criterion(target, labels)
            valid_loss += loss.item()
            
        valhistory.append(valid_loss)      
                
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(valloader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            
            #Saving State Dict
            savePATH = 'savestats/' + args.r[:2]+args.seg+ sessdate+str(args.nuclei) +'_' +str(args.bs) +'_' + str(args.lr) + '_'+ str(args.divfs) + '_' + str(args.nwin)+ '.pth'
            torch.save(net.state_dict(), savePATH)

    print('Finished Training')

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

    gc.collect()

    print('Running best model on test data...')

    model = LFPnet(finsize = finsize, constride = constride, poolstride = poolstride)
    model.load_state_dict(torch.load(savePATH))
    model.to(device)

    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=args.bs, shuffle=True, num_workers=1)
    
    correct = 0
    total = 0

    classes = ('puff', 'touch')
    # since we're not training, we don't need to calculate the gradients for our outputs
    model.eval()
    for data in testloader:
        images, labels = data

        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total : .2f} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    model.eval()
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
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

    history = {'train': trainhistory, 'val': valhistory}

    with open(args.csv, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([sessdate, args.d, args.r, args.seg, args.raw, args.nuclei, nucinds, args.scaler, args.w, 
                         args.nwin,args.divfs, (freq[0],freq[-1]),
                         args.lr, args.bs, args.epochs,
                         100 * correct // total,100*float(correct_pred['touch']) / total_pred['touch'],100* float(correct_pred['puff']) / total_pred['puff'], history, savePATH,executionTime])

    if device == 'cpu':
        plt.plot(np.array(trainhistory)/len(trainloader), label='train')
        plt.plot(np.array(valhistory)/len(valloader), label = 'val')
        plt.legend()