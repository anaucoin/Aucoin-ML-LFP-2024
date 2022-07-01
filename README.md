# LFPnet
This project contains a mix of Python and Julia scripts for neural recording analysis using principal component analysis (PCA), generalized eigendecomposition (GED) and LFPNet, a deep convolutional neural network designed to classify time-frequency spectrograms of local field potential (LFP) recordings.

Details on LFPNet can be found in Golshan, Hebb, and  Mahoor. LFP-Net: A deep learning framework to recognize human behavioral activities using brain STN-LFP signals. Journal of Neuroscience Methods (2020) https://doi.org/10.1016/j.jneumeth.2020.108621. 

## To run lfpnet.py
Make sure you are in a folder with the lfpnet.py script and data in the .npz format.
The script can be run from command line with the default parameters by calling the script name:

>  python lfpnet.py 

There are many custom flags available for user-specified inputs. The following flags are currently supported:\
 --d or --day: Specifies which experimental day to use. This should be an integer between 0-11.\
 --r or --region: Specifies to use data from either "amygdala" or "3b".\
 --m: Specifies which model to use for dimensionality reduction. Currently only ged is supported. PCA support will follow. \
 --bm or --basemodality: Specifies the basemodality for GED analysis. Either "puff" or "touch".\
 --w: Specifies the central frequency for Morlet Wavelet Transform(MVT, used to create the input spectrograms)\
 --nwin: Specifies the number of windows (or bins) for the MVT \
 --divfs: Specifies the largest frequency to consider. Frequency = linspace(fs/adivfs, nwin). Fs is the frames per second, usually 1000. \
 --epochs: Specifies the number of training epochs.\
 --bs: Specifies the training batch-size.\
 --nosave: This will tell the script NOT to save the final model or variables. \
 --save : This will save the state of the trained model to a .pth file and the necessary variables to a .npz file. This feature will also prompt the results to be written into the lfpnetlog.csv file. The file should have headers: 
['date', 'day_code', 'region', 'model', 'basemodality','w', 'nbins','divfs', 'frequency_bands', 'batch_size', 'epochs', 'accuracy', 'touch accuracy', 'puff accuracy', 'filename']

## Some useful bash scripts 
It is nice to be able to run many cases of the model at one time. To do so, one can use the following bash scripts: \

1. To loop over all experimental days, using both puff and touch as the GED baseline: \
```text
#!/usr/bin/env bash
#This script runs multiple instances of lfpnet with varying parameters. Useful for hyperparameter tuning, and multiday performance analysis
cd path/to/files
for day in {0..11}
do
    echo Starting Day "$day", Touch
    nice python lfpnet.py --d \$day > multiday.txt
    echo $(tail -n -3 multiday.txt)
    echo Puff
    nice python lfpnet.py --d "$day" --bm puff > multiday.txt
    echo $(tail -n -3 multiday.txt)
done
```


One might also want to parameter tune by trying multiple instances of say, the trianing batch size: 
```text
#!/usr/bin/env bash
# This script runs multiple instances of lfpnet with varying parameters. Useful for hyperparameter tuning, and multiday performance analysis

batchsize=(1 2 5 10 15 20 25 30 35)
for bs in ${batchsize[@]}
do
        echo Starting training for size "$bs"
        nice python lfpnet.py --d 5 --nosave --bs "$bs" > batchloop.txt
        echo $(head -n 1 batchloop.txt)
        echo $(tail -n -3 batchloop.txt)
done
```


More info coming soon... 
