# Introduction
This project contains Python scripts for reproducing the results in "Detection of latent brain states from baseline neural activity in the amygdala" (Aucoin et. al. 2024)

Details of CNN architecture were inspired Golshan, Hebb, and  Mahoor. LFP-Net: A deep learning framework to recognize human behavioral activities using brain STN-LFP signals. Journal of Neuroscience Methods (2020) https://doi.org/10.1016/j.jneumeth.2020.108621. 

## Machine Learning Methods
To run analysis using the CNN, SVM, and Linear SVM methods, run lfp-cnn.py, lfp-svm.py and lfp-svm-linear.py respectively. Make sure you are in a folder with the .py script, the amygdala_anatomy.csv file and the relevant data in the .npz format.
The scripts can be run from command line with the default parameters by calling the desired script name:

NOTE: Currenlty, these flags are only supported when use_cuda is available. To change the inputs on a local machine when cuda is unavaible, one can update the initialization of the Namespace object (args). 

There are many custom flags available for user-specified inputs. The following flags are currently supported:\
 --monkey: Specifies the name of the subject to analyze.\
 --d or --day: Specifies the index of desired experimental day (found in dates array)\
 --r or --region: Specifies to use data from either "amygdala" or "3b".\
 --seg: Specifies the segment of lfp to use (e.g. base, stim, full).\
 --w: Specifies the central frequency for Morlet Wavelet Transform(MVT, used to create the input spectrograms)\
 --nwin: Specifies the number of windows (or bins) for the MVT \
 --divfs: Specifies the largest frequency to consider. Frequency = linspace(fs/adivfs, nwin). Fs is the frames per second, usually 1000. \
 --epochs: Specifies the number of training epochs.\
 --lr: Specifies the learning rate.\
 --bs: Specifies the training batch-size.\
 --scaler: Specifies the scaler used for normalizing the data between [0,1] (minmax or stand).\
 --null: An optional flag to compute the null distribution through randomly shuffling labels in the trainign data.\
 --rel: An optional flag to use relative timeseries signal (i.e. subtract the average lfp signal across contacts from all contacts. This is NOT the default behavior).\
 --nuclei: Specifies which nuclei to analyze ('C' : central, 'B': basal, 'AB': accessory basal, 'L': lateral).\
 --csv: Specifies the name of the .csv file to write output summaries.\
 --nosave: This will tell the script NOT to save the final model or variables. \
 --save : This will save the state of the trained model to a .pth file and the necessary variables to a .npz file. 


## Some useful bash scripts 
Below is an example bash script to run the code. You may modify the script name and add/remove flags to fit your needs.
```text
#!/usr/bin/env bash
cd paht/to/files
nuclei=('C' 'B' 'AB' 'L')
csvname="lfp-cnn.csv"
script="lfp-cnn.py"
for nuc in ${nuclei[@]}
do
        for day in {0..11}
        do
        echo Training Day $day and nuclei $nuc
                python $script --monkey Aster --scaler minmax --day $day --nuclei $nuc --nwin 160 --divfs 20 --csv $csvname
                python $script --monkey Cotton --scaler minmax --day $day --nuclei $nuc --nwin 160 --divfs 20 --csv $csvname
                python $script --monkey Saguaro --scaler minmax --day $day --nuclei $nuc --nwin 160 --divfs 20 --csv $csvname
        done
done
```

## RSA.py
This script computes the RSA strength for each session and computes the one-sided t-test to compare airflow vs. grooming. Simply call this script by name to run. 

