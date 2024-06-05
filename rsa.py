import numpy as np
import pandas as pd
import scipy.io as io
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp

r = 'amygdala'
monkeys = ['Aster', 'Saguaro', 'Cotton']
allpuff = [] 
alltouch = []
PUFF = []
TOUCH = []
for monid in range(3):
    monkey = monkeys[monid]
    if monkey == 'Aster':
        dates = ['060619','061019','061319','061819','062019','062419','070819','071019','071219','071619','071819','080619'];
    if monkey == 'Cotton':
        dates = ['010520', '010720', '021220', '021620', '022120', '022820', '030620', '031020', '121619']
    if monkey == "Saguaro":
        dates = ['42221', '50521', '51021', '51221', '51421', '51721', '51921']
    num_days = len(dates)
    
    ap = []
    at = []
    meanpuff = []
    meantouch = []
    for d in range(num_days):
        sessdate = dates[d]
        print(f'{monkey} {sessdate}...')
        filename = f'cleanbase_{sessdate}_{r}.npz'

        with np.load(filename, allow_pickle=True) as alldata:
            df = pd.DataFrame(alldata['df'], columns = ['modality', 'loc', 'stimstart', 'stimend','is BadT', 'is Block end', 'is long trial','is wild trial'])
            buffer = alldata['buffer']

        rootPATH = '/Users/alexaaucoin/Documents/Arizona/Research/neuralgc/tactiledata/'

        if monkey == 'Saguaro':
            beatfile = rootPATH + f'{monkey}_0{sessdate}/BPM_0{sessdate}.mat'
            timesfile = rootPATH + f'{monkey}_0{sessdate}/beatTs_0{sessdate}.mat'
        else: 
            beatfile = rootPATH + f'{monkey}_{sessdate}/BPM_{sessdate}.mat'
            timesfile = rootPATH + f'{monkey}_{sessdate}/BeatTimes_{sessdate}.mat'

        beats = io.loadmat(beatfile, simplify_cells=True)['BPM']
        beattimes = io.loadmat(timesfile, simplify_cells=True)['beatTs']

        badbeatfile = 'badHRtimes.mat'

        beatsess = io.loadmat(badbeatfile,simplify_cells=True)['badBeats']['date']
        badtimes = io.loadmat(badbeatfile,simplify_cells=True)['badBeats']['badtimes']

        if monkey == 'Saguaro':
            ind = np.where(beatsess == '0' + sessdate)[0][0]
        else:
            ind = np.where(beatsess == sessdate)[0][0]

        badtimes = badtimes[ind] - 1 #matlab indexing

        cntrl = np.where(beattimes*1000 >= df.stimend.max())[0][0]
        
        mask=np.full(len(beattimes),False,dtype=bool)
        if len(badtimes) > 0:
            mask[badtimes] = [True]*len(badtimes)

        beatTs = np.array([int(1000*i) for i in beattimes[:cntrl][~mask[:cntrl]]])
        beats = beats[:cntrl][~mask[:cntrl]]

        fn = sp.interpolate.Akima1DInterpolator(beatTs,beats)

        tt = np.arange(beatTs.min(), beatTs.max(),1)

        #grab true start and end times 
        logname = "Tactile session info.xlsx"

        sessinfo = pd.read_excel(logname, skiprows=2, header=0)

        sessinfo = sessinfo[(sessinfo.monkey == monkey) & (sessinfo["'date'"] == int(sessdate))]

        impkeys = ['monkey', "'date'","'touch start 1'", "'touch end 1'", "'touch start 2'", "'touch end 2'","'touch start 3'","'touch end 3'",
                  "'puff start 1'", "'puff end 1'", "'puff start 2'", "'puff end 2'","'puff start 3'","'puff end 3'","'puff start 4'","'puff end 4'"]

        strts = []
        nds = [] 
        mods = []
        for s,e in zip(sessinfo[impkeys].dropna(axis=1).keys()[2::2], sessinfo[impkeys].dropna(axis=1).keys()[3::2]):
            mod = s[1:6]
            strts.append(int(sessinfo[s].values[0]*1000))
            nds.append(int(sessinfo[e].values[0]*1000))
            mods.append(mod)
        strts = np.array(strts)
        nds = np.array(nds)

        lbls = np.array(['neither']*len(tt), dtype=str)

        for block in range(len(strts)):
            inds = np.where((tt>=strts[block]) & (tt < nds[block]))[0]
            lbls[inds] = [mods[block]]*len(inds)

        cmap = {'neither' : 'grey', 'puff ' : 'darkblue', 'puff baseline' : 'lightblue', 'touch' : 'purple', 'touch baseline': 'mediumpurple'}

        labelmap = [cmap[lbl] for lbl in lbls]

        t = tt
        ft = fn(tt)
        beatlabels = lbls
        
        lenwin = 60000 #(~ 60s or 60000 ms )
        slide = 3000 #(~ 30s or 30000 ms )
        #half-bandwidth and time half bandwidth
        hbw = .07 #as prescribed by Anne in "From sensing to feeling..."
        TW = lenwin*.001*hbw #(lenwin in seconds * half-bandwidth)

        TW = max(int(TW), 1)

        # N slepian tapers = floor(2*TW) - 1
        tapers = sp.signal.windows.dpss(60000,TW,Kmax=7)

        f, t1, S1 = sp.signal.spectrogram(ft, fs=1000, window=tapers[0], nperseg=60000, noverlap=lenwin-slide)
        f, t2, S2 = sp.signal.spectrogram(ft, fs=1000, window=tapers[1], nperseg=60000, noverlap=lenwin-slide)
        f, t3, S3 = sp.signal.spectrogram(ft, fs=1000, window=tapers[2], nperseg=60000, noverlap=lenwin-slide)
        f, t4, S4 = sp.signal.spectrogram(ft, fs=1000, window=tapers[3], nperseg=60000, noverlap=lenwin-slide)
        f, t5, S5 = sp.signal.spectrogram(ft, fs=1000, window=tapers[4], nperseg=60000, noverlap=lenwin-slide)
        f, t6, S6 = sp.signal.spectrogram(ft, fs=1000, window=tapers[5], nperseg=60000, noverlap=lenwin-slide)
        f, t7, S7 = sp.signal.spectrogram(ft, fs=1000, window=tapers[6], nperseg=60000, noverlap=lenwin-slide)
        start_i = np.where(f >= .1)[0][0]
        end_i = np.where(f >=.6)[0][0] + 1
        meantapers = (S1 + S2 + S3 + S4 + S5 + S6 + S7)/7
        F = meantapers
        
        LF = [.1, .6]
        inds = np.where((f>=LF[0]) & (f<=LF[1]))[0]
        dx = np.diff(f)[0]
        normF = np.empty(F.shape)

        for i in range(F.shape[1]):
            A = dx*F[inds,i].sum()
            normbin = F[:,i]/A
            normF[:, i] = normbin
            
        RSA_strength = np.empty(normF.shape[1])
        for ii in range(normF.shape[1]):        
            thisbin = normF[start_i:end_i,ii]
            peaks, properties = sp.signal.find_peaks(thisbin, height = 0, width = 0, prominence=0)
            sorti = properties["prominences"].argsort()
            if len(peaks) > 0:
                peaki = peaks[sorti[-1]]
                L =properties["left_ips"][sorti[-1]]
                R =properties["right_ips"][sorti[-1]]
                halfW = int((R-L)/2)
                #store results 
                if (len(thisbin[peaki - halfW:peaki + halfW]) !=0):
                    RSA_strength[ii] = thisbin[peaki - halfW:peaki + halfW].mean()
                else:
                    RSA_strength[ii] = thisbin[int(len(thisbin)/2) - 6: int(len(thisbin)/2) + 6].mean()
            else: 
                RSA_strength[ii] = thisbin[int(len(thisbin)/2) - 6: int(len(thisbin)/2) + 6].mean()
        
        normRSA = (RSA_strength - np.median(RSA_strength))/(np.median(RSA_strength))

        RSA_puff = normRSA[np.where((beatlabels[lenwin:len(t):slide] == 'puff '))[0]]
        RSA_touch = normRSA[np.where((beatlabels[lenwin:len(t):slide]== 'touch'))[0]]

        meanpuff.append(RSA_puff.mean())
        meantouch.append(RSA_touch.mean())
        ap.append([RSA_puff])
        at.append([RSA_touch])
    allpuff.append(ap)
    alltouch.append(at)
    PUFF.append(meanpuff)
    TOUCH.append(meantouch)

# ### Plot results

# +
mpl.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1,3, figsize=(8,4), sharey='row',gridspec_kw = {'wspace':.04, 'hspace':.1})
for monid in range(3):
    monkey = monkeys[monid]
    if monkey == 'Aster':
        dates = ['060619','061019','061319','061819','062019','062419','070819','071019','071219','071619','071819','080619'];
    if monkey == 'Cotton':
        dates = ['010520', '010720', '021220', '021620', '022120', '022820', '030620', '031020', '121619']
    if monkey == "Saguaro":
        dates = ['42221', '50521', '51021', '51221', '51421', '51721', '51921']
    num_days = len(dates)

    for day in range(num_days): 
        color = 'red' if ((PUFF[monid][day] > TOUCH[monid][day]) & (day < 2)) else 'black'
        #color = 'black'
        ax.flat[monid].plot([0, 1], [PUFF[monid][day], TOUCH[monid][day]], color, linewidth=.8)
    ax.flat[monid].set_title(f'monkey {monkey[0]}')
    ax.flat[monid].set_ylim([-.6,1.5])
    ax.flat[monid].set_xlim([-.4,1.4])
    ax.flat[monid].set_xticks([0,1],['Airflow', 'Touch'])
    ax.flat[monid].spines[['right', 'top']].set_visible(False)

ax.flat[0].set_ylabel('mean RSA')
# -

# # T-test with alternative hypothesis that puff < touch

from scipy.stats import ttest_ind
# ## T-test on means across all sessions

T = 0
tsig = 0
alpha = .05
for m in range(3): 
    res = ttest_ind(PUFF[m], TOUCH[m],equal_var=False, alternative='less')
    issig = '***' if res.pvalue < alpha else ''
    print(f'{monkeys[m]} - F:{res.statistic}, p: {res.pvalue}{issig}')


