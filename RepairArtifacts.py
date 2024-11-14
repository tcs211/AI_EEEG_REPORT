############################################
# This file is used to repair artifacts in EEG data
# The function getFeatures is used to extract features from EEG data
# The function findArtifacts is used to repair artifacts in EEG data
#  Using the HBOS algorithm to detect artifacts in EEG data
#  Using neighboring channels to repair artifacts
#  The function returns the repaired EEG data, the ratio of bad channels, 
#  and the number of bad channels in each epoch
############################################
import eegFeatureExtract as eeg
import numpy as np
import mne
from tqdm import tqdm
from pyod.models import hbos 

import matplotlib.pyplot as plt
adjacencyMatrix = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
])
electrodesOrder = ['Fp1','F7','F3','T3','C3','T5','P3','O1','Fz','Cz','Pz',
                   'Fp2','F8','F4','T4','C4','T6','P4','O2']

def getFeatures(eegData, fs):
    #Shannon Entropy
    ShannonRes = eeg.shannonEntropy(eegData, bin_min=-200, bin_max=200, binWidth=2)
    # print ('ShannonRes.shape)',ShannonRes.shape)
    # Subband Information Quantity
    # delta (1–4 Hz)
    eegData_delta = eeg.filt_data(eegData, 1, 4, fs)
    ShannonRes_delta = eeg.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2)
    # print ('ShannonRes_delta',ShannonRes_delta.shape)
    # theta (4–8 Hz)
    eegData_theta = eeg.filt_data(eegData, 4, 8, fs)
    ShannonRes_theta = eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2)
    # print ('ShannonRes_theta',ShannonRes_theta.shape)
    # alpha (8–13 Hz)
    eegData_alpha = eeg.filt_data(eegData, 8, 13, fs)
    ShannonRes_alpha = eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2)
    # print ('ShannonRes_alpha',ShannonRes_alpha.shape)
    # beta (13–30 Hz)
    eegData_beta = eeg.filt_data(eegData, 13, 30, fs)
    ShannonRes_beta = eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2)
    # print ('ShannonRes_beta',ShannonRes_beta.shape)
    # gamma (30–100 Hz)
    # eegData_gamma = eeg.filt_data(eegData, 30, 100, fs)
    # ShannonRes_gamma = eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2)

   # print ('ShannonRes_delta',ShannonRes_delta.shape, 'ShannonRes_theta',ShannonRes_theta.shape, 'ShannonRes_alpha',ShannonRes_alpha.shape, 'ShannonRes_beta',ShannonRes_beta.shape)

    # Cepstrum Coefficients (n=2)
    CepstrumRes = eeg.mfcc(eegData, fs, order=2)
    # print ('CepstrumRes',CepstrumRes.shape)

    # Lyapunov Exponent
    LyapunovRes = eeg.lyapunov(eegData)
    # print ('LyapunovRes.shape)',LyapunovRes.shape)

    # Hjorth Mobility
    # Hjorth Complexity
    HjorthMob, HjorthComp = eeg.hjorthParameters(eegData)
    # print ('HjorthMob.shape, HjorthComp.shape)',HjorthMob.shape, HjorthComp.shape)

    # False Nearest Neighbor
    FalseNnRes = eeg.falseNearestNeighbor(eegData)
    # print ('FalseNnRes',FalseNnRes.shape)

    ###################################
    # Median Frequency
    medianFreqRes = eeg.medianFreq(eegData,fs)
    # print ('medianFreqRes.shape)',  medianFreqRes.shape)

    # δ band Power    
    bandPwr_delta = eeg.bandPower(eegData, 1, 4, fs)
    # print ('bandPwr_delta',bandPwr_delta.shape)
    # bandUpDelta = eeg.bandPower(eegData, 2, 4, fs)
    # θ band Power
    bandPwr_theta = eeg.bandPower(eegData, 4, 8, fs)
    # print ('bandPwr_theta',bandPwr_theta.shape)
    # α band Power
    bandPwr_alpha = eeg.bandPower(eegData, 8, 13, fs)
    # print ('bandPwr_alpha',bandPwr_alpha.shape)
    # β band Power
    bandPwr_beta = eeg.bandPower(eegData, 13, 30, fs)
    # print ('bandPwr_beta',bandPwr_beta.shape)
    # γ band Power
    # bandPwr_gamma = eeg.bandPower(eegData, 30, 100, fs)

    # print ('bandPwr_delta',bandPwr_delta.shape, 'bandPwr_theta',bandPwr_theta.shape, 'bandPwr_alpha',bandPwr_alpha.shape, 'bandPwr_beta',bandPwr_beta.shape)

    # Standard Deviation
    std_res = eeg.eegStd(eegData)
    # print ('std_res',std_res.shape)
    
    # Regularity (burst-suppression)
    regularity_res = eeg.eegRegularity(eegData,fs)
    # print ('regularity_res',regularity_res.shape)

    # Voltage < 5μ
    volt05_res = eeg.eegVoltage(eegData,voltage=5)
    # print ('volt05_res',volt05_res.shape)
    # Voltage < 10μ
    volt10_res = eeg.eegVoltage(eegData,voltage=10)
    # print ('volt10_res',volt10_res.shape)
    # Voltage < 20μ
    volt20_res = eeg.eegVoltage(eegData,voltage=20)
    # print ('volt20_res',volt20_res.shape)

    # print ('volt05_res',volt05_res.shape, 'volt10_res',volt10_res.shape, 'volt20_res',volt20_res.shape)

    # Diffuse Slowing
    df_res = eeg.diffuseSlowing(eegData,fs)
    # print ('df_res',df_res.shape)
    # print (df_res.shape)

    # Spikes
    minNumSamples = int(70*fs/1000)
    spikeNum_res = eeg.spikeNum(eegData,minNumSamples)
    # print ('spikeNum_res',spikeNum_res.shape)

    # Delta burst after Spike
    deltaBurst_res = eeg.burstAfterSpike(eegData,eegData_delta,minNumSamples=7,stdAway = 3)
    # print ('deltaBurst_res',deltaBurst_res.shape)
    # Sharp spike
    sharpSpike_res = eeg.shortSpikeNum(eegData,minNumSamples)
    # print ('sharpSpike_res',sharpSpike_res.shape)

    # Number of Bursts
    numBursts_res = eeg.numBursts(eegData,fs)
    # print ('numBursts_res',numBursts_res.shape)

    # Burst length μ and σ
    burstLenMean_res,burstLenStd_res = eeg.burstLengthStats(eegData,fs)
    # print ('burstLenMean_res',burstLenMean_res.shape, 'burstLenStd_res',burstLenStd_res.shape)
    # Burst Band Power for δ
    burstBandPwrAlpha = eeg.burstBandPowers(eegData, 0.5, 4, fs)
    # print ('burstBandPwrAlpha',burstBandPwrAlpha.shape)

    # Number of Suppressions
    numSupps_res = eeg.numSuppressions(eegData,fs)
    # print ('numSupps_res',numSupps_res.shape)
    # Suppression length μ and σ
    suppLenMean_res,suppLenStd_res = eeg.suppressionLengthStats(eegData,fs)
    # print ('suppLenMean_res',suppLenMean_res.shape, 'suppLenStd_res',suppLenStd_res.shape)
    feature_list = []
    # feature_list.append(armaRes[:,:,0])
    # feature_list.append(armaRes[:,:,1])
    # feature_list.append(tsalisRes)
    feature_list.append(CepstrumRes[:,:,0]) #0
    feature_list.append(CepstrumRes[:,:,1])
    feature_list.append(LyapunovRes)
    feature_list.append(HjorthMob)
    feature_list.append(HjorthComp)
    feature_list.append(FalseNnRes)
    feature_list.append(medianFreqRes) 
    feature_list.append(ShannonRes)
    feature_list.append(ShannonRes_delta) #8
    feature_list.append(ShannonRes_theta)
    feature_list.append(ShannonRes_alpha)
    feature_list.append(ShannonRes_beta)
    feature_list.append(bandPwr_delta) #12
    feature_list.append(bandPwr_theta) #13
    feature_list.append(bandPwr_alpha) #14
    feature_list.append(bandPwr_beta)
    feature_list.append(std_res)
    feature_list.append(regularity_res)
    feature_list.append(volt05_res)
    feature_list.append(volt10_res)
    feature_list.append(volt20_res)
    feature_list.append(df_res)
    feature_list.append(spikeNum_res)
    feature_list.append(deltaBurst_res)
    feature_list.append(sharpSpike_res)
    feature_list.append(numBursts_res)
    feature_list.append(numSupps_res)
    feature_list.append(suppLenMean_res)
    feature_list.append(suppLenStd_res)
    feature_list.append(burstBandPwrAlpha)
    feature_list.append(burstLenMean_res)
    feature_list.append(burstLenStd_res)

    features=np.array(feature_list)
    features=features.transpose(2,1,0)
    features=np.nan_to_num(features)
    return features
def findArtifacts(epochs, feature_arr, showRepairs=False):    

    colors=[]

    data=epochs.copy().get_data(copy=True, verbose=False)
    alphaRatio=feature_arr[:,:,14]/feature_arr[:,:,12]

    # median
    median=np.median(alphaRatio)
    mean=np.mean(alphaRatio)
    cutoff=(0.9*median+1.1*mean)/2
    if showRepairs:
        plt.hist(alphaRatio.copy().flatten(), bins=10000, color='c', alpha=0.7, rwidth=0.85)     
        plt.axvline(x=median, color='r', linestyle='--')
        plt.axvline(x=(cutoff), color='g', linestyle='--')
        plt.axvline(x=mean, color='b', linestyle='--')
        legend = ['median', 'Cutoff', 'mean']
        plt.legend(legend)
        plt.xlim(0, mean*2)
        plt.show()
        plt.close()

    for x in tqdm(range(feature_arr.shape[0])):
        feature=feature_arr[x]
        # epoch level
        color=[]
        for i in range(len(feature)):
            # channel level
            fetureOfChn=[feature[i]]            
            # Filter out channels with alphaRatio > median, consider them as normal
            # changed to alphaRatio>(0.9*median+1.1*mean)/2, considered as normal
            if alphaRatio[x][i]>cutoff:
                color.append('gray')
                continue

            # Find the features of adjacent channels
            adjacencyFeature=feature[adjacencyMatrix[i]==1]
            
            adj_data=data[x][adjacencyMatrix[i]==1]
            if adj_data.shape[0]<3:
                color.append('gray')
                continue
            
            # Take the features of the previous and next two epochs
            if x>0 and x<feature_arr.shape[0]-1:
                d=[data[x-1][i]]
                adj_data=np.append(adj_data,d,axis=0)
                
                d=[data[x+1][i]]
                adj_data=np.append(adj_data,d,axis=0)
            # index 0 epoch, take the next two epochs' features
            elif x==0:                
                d=[data[x+1][i]]
                adj_data=np.append(adj_data,d,axis=0)
                d=[data[x+2][i]]
                adj_data=np.append(adj_data,d,axis=0)
            # last epoch, take the previous two epochs' features
            elif x==feature_arr.shape[0]-1:
                d=[data[x-1][i]]
                adj_data=np.append(adj_data,d,axis=0)
                d=[data[x-2][i]]
                adj_data=np.append(adj_data,d,axis=0)

            fetureOfChn.extend(adjacencyFeature)  
            adj_data=data[x][adjacencyMatrix[i]==1]
            fetureOfChn=np.array(fetureOfChn)

            if fetureOfChn.shape[0]>3:                
                clf = hbos.HBOS( alpha=0.01, tol=0.5, contamination=.1)
                try:
                    clf.fit(fetureOfChn)
                except:
                    color.append('grey')
                    continue
                scores=clf.predict(fetureOfChn)           
                                    
                # if the score of the target channel is 1, and all the scores of the adjacent channels are 0, it is considered abnormal
                if scores[0]==1:     
                    color.append('red')            

                    adj_data=adj_data[scores[1:]==0].mean(axis=0)
                    data[x][i]=adj_data
                else:
                    color.append('gray')
                    
            else:
                color.append('gray')
        colors.append(color)

    epoch_gen=epochs.copy()
    # set epoch data with new data
    epoch_gen._data=data

    if showRepairs:
        mne.viz.set_browser_backend('qt')    
        epochs.plot(scalings=dict(eeg=70e-6), n_epochs=25, epoch_colors=colors, block=True, title='original')
        epoch_gen.plot(scalings=dict(eeg=70e-6), n_epochs=25, epoch_colors=colors, block=True, title='generated')
    
    colors=np.array(colors)
    
    bad_ratio={}
    ch_names=epochs.ch_names
    for i in range(colors.shape[1]):
        ch_colors=colors[:,i]
        
        r=round(np.count_nonzero(ch_colors=='red')/ch_colors.shape[0],2)
        bad_ratio[ch_names[i]]=r

    badChnsNum=[]
    for  c in colors:
        num=np.count_nonzero(c=='red')
        badChnsNum.append(num)

    return epoch_gen, bad_ratio, badChnsNum