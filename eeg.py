############################################
# This file is used to process EEG data
# File format: .edf, .vhdr, .mat, .fif
# Channels must have: ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 
# 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'C3', 'C4', 'Cz', 
# 'Fz', 'Pz', 'T3', 'T4']
# and extract alpha epochs with or without repair artifacts
############################################
import os
import mne
import pandas as pd
import numpy as np
from scipy.integrate import simpson
import RepairArtifacts as repair
from pymatreader import read_mat

mne.viz.set_browser_backend('qt')

class eegProcess:
    def __init__(self, eegFile, filePath, 
                 useRepair=True,
                 epoch_length=4,  removeEpochsRationThreshold=0.3, 
                 dropEpochSD=2.2, tmax=None, tmin=None,
                 renameChannels=True,
                 unit_uV=True):
        self.unit_uV = unit_uV
        self.eegFile = eegFile
        self.isFifFile = '.fif' in eegFile
        self.epoch_length = epoch_length
        self.removeEpochsRationThreshold=removeEpochsRationThreshold
        self.useRepair=useRepair
        if not filePath.endswith('/') or not filePath.endswith('\\'):
            self.filePath = filePath + '/' if '/' in filePath else filePath + '\\'
        self.eegFullName = self.filePath + self.eegFile
        self.dropEpochSD=dropEpochSD
        self.tmax = tmax
        self.tmin = tmin
        self.renameChannels = renameChannels
        print ('eegFullName: ', self.eegFullName)

    def getRawData(self):
        events = []
        eegFullName = self.eegFullName

        if self.isFifFile:
            raw = mne.io.read_raw_fif(eegFullName, preload=True, verbose=False)
            annot = raw.annotations
            # print (len(annot))
            for i in range(len(annot)):
                # print(i, annot.description[i], annot.duration[i], annot.onset[i])
                # find Eyes Open or Eyes Closed
                if annot.description[i] in ['Eyes Open', 'Eyes Closed']:
                    # get start and end timee
                    events.append(annot.onset[i])

        elif 'vhdr' in eegFullName:
            raw = mne.io.read_raw_brainvision(eegFullName, preload=True, verbose=False)
            ch_names = raw.ch_names
            ch_names = [ch.replace('T7', 'T3').replace('T8', 'T4').replace('P7', 'T5').replace('P8', 'T6').replace('POz', 'Pz') for ch in ch_names]
            raw.rename_channels(dict(zip(raw.ch_names, ch_names)))
        # mat file
        elif 'mat' in eegFullName:
            print ('mat file: ', eegFullName)
            data = read_mat(eegFullName)
            info=mne.create_info(ch_names=['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1',
                               'C1','C3','C5','T3','TP7','CP5','CP3','CP1','P1','P3','P5',
                               'T5','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz',
                               'Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6',
                               'FC4','FC2','FCz','Cz','C2','C4','C6','T4','TP8','CP6','CP4',
                               'CP2','P2','P4','P6','T6','P10','PO8','PO4','O2','EOG1','EOG2','EOG3','Trigger'],
                                 sfreq='256', ch_types=['eeg']*64+['eog']*3+['stim'])
            raw=mne.io.RawArray(data['dataRest']/1e9,info)
            print (raw.info['sfreq'])
            # raw.plot(block=True)

        else:
            if self.unit_uV:
                raw = mne.io.read_raw_edf(eegFullName, preload=True, units='uV', verbose=False)
            else: #TUH EEG data, get only 600s
                raw = mne.io.read_raw_edf(eegFullName, preload=True, verbose=False)
                if self.tmax is None and self.tmin is None:                    
                    raw=raw.crop(0, 600)
                print ('raw length: ', raw.times[-1])
            txtfilename = eegFullName.replace('.edf', '.txt')
            # print ('txtfilename: ', txtfilename)
            if os.path.isfile(txtfilename):
                # read events from txt file
                event = pd.read_csv(txtfilename, sep='\t', header=0)
                # print(event)

                # create mne annotations
                onset = event['sample'].values
                duration = 0 #data['duration'].values
                description = event['description'].values
                annot = mne.Annotations(onset, duration, description)

                raw.set_annotations(annot)
                # read event file
                events=self.readEyeEvents()
        # if tmax or tmin is not None, crop the raw data
        if self.tmax is not None or self.tmin is not None:
            tmin = 0 if self.tmin is None else self.tmin
            tmax = raw.times[-1] if self.tmax is None else self.tmax
            raw = raw.crop(tmin, tmax)
            print ('crop raw data from %s to %s' % (tmin, tmax))
        # print ('events: ', events)        
        chnsToDrop = ['EKG', 'Photic', 'A1', 'A2']
        # rename channels only alpha numeric, first character uppercase, others lowercase
        print ('raw channels: ', raw.ch_names)
        # replace('EEG', '').replace('REF', '')
        if self.renameChannels:
            raw.rename_channels(lambda x: ''.join([i for i in x.replace('EEG', '').replace('REF', '') if i.isalnum()]
                                              ).capitalize())
        print ('raw channels: ', raw.ch_names)
        # rename alternative channels to 10-20 system
        alternatChIn10_10=['T7', 'T8', 'P7', 'P8', 'POz']
        mappedChIn10_20=['T3', 'T4', 'T5', 'T6', 'Pz']

        # rename alternative channels to 10-20 system if exist
        for i in range(len(alternatChIn10_10)):
            if alternatChIn10_10[i] in raw.ch_names and not mappedChIn10_20[i] in raw.ch_names:
                raw.rename_channels({alternatChIn10_10[i]: mappedChIn10_20[i]})
                

        for ch in chnsToDrop:
            if ch in raw.ch_names:
                raw.drop_channels([ch])
        chnsMustHave = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'C3', 'C4', 'Cz', 'Fz', 'Pz', 'T3', 'T4']
        
        # remove channels not in chnsMustHave
        for ch in raw.ch_names:
            if ch not in chnsMustHave:
                # print ('Drop channel: ', ch)
                try :
                    raw.drop_channels([ch])
                except:
                    print ('Error: drop channel %s failed!' % ch)
        for ch in chnsMustHave:
            if ch not in raw.ch_names:
                print ('Error: Channel %s is missing!' % ch)
                return None, None
          
            
        sampling_rate = int(raw.info['sfreq'])
        if not sampling_rate==125:
            raw.resample(125)
            sampling_rate = 125
        self.sampling_rate = sampling_rate
        
        raw.set_montage('standard_1020')
        raw.filter(1, 60, verbose=False)
        sphere = mne.make_sphere_model("auto", "auto", raw.info, verbose=False)
        src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=15.0, verbose=False)
        forward = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere, verbose=False)
        raw.set_eeg_reference('REST', forward=forward, verbose=False)    
        
        self.raw = raw
        self.events = events
        return raw, events 

    
    def readEyeEvents(self):
        # event filename=edf_file_name.split('.')[0]+'.txt'
        edf_file=self.eegFullName
        # print ('edf_file: ', edf_file)
        event_file=edf_file.replace('.edf', '.txt')
        # print ('event_file: ', event_file)
        if os.path.isfile(event_file):

            events=[]
            with open(event_file, 'r') as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip().split('\t')
                    if len(line)>2 and (line[2]=='Eyes Open' or line[2]=='Eyes Closed'):
                        events.append(float(line[0]))
            # print ('events: ', events)
            return events
        else:
            return []
        
    def extractAlphaEpochs(self):
        raw=self.raw
        raw_copy = raw.copy()    
        epoch_length=self.epoch_length
        events = mne.make_fixed_length_events(raw_copy, duration=epoch_length)
        epochs = mne.Epochs(raw_copy, events, tmin=0, tmax=epoch_length, baseline=None, verbose=False, preload=True)
        epoch_data=epochs.get_data(copy=True, units='uV')
        bandWidth=np.round(1/epoch_length, 2)
        
        psds=epochs.copy().compute_psd(method='multitaper', fmin=1, fmax=30, n_jobs=4, bandwidth=bandWidth)
        
        ch_names=epochs.ch_names
        epochs_num=psds.shape[0]
        
        antChs=['Fp1', 'Fp2', 'F7', 'F8', 'F3',  'F4']
        posChs=['T5', 'T6', 'P3', 'P4', 'O1', 'O2']

        # 計算每個epochs的平均power
        def calcPowerSum(epoch_psds, fmin, fmax, pick_chs, bandWidth):
            epoch_psds=epoch_psds.copy().pick(pick_chs).get_data(fmin=fmin, fmax=fmax)
            epoch_psds=epoch_psds*1e12
            
            psdsSum= simpson(epoch_psds, dx=bandWidth)
            return psdsSum
        print ('All epochs_num: ', epochs_num)

        # 計算各頻帶不同分布的總和
        sumArray=[[], [], [], [], []]
        bandRange=[(8, 13), (8, 13), (13, 30), (8, 13), (1, 8)]
        ch_names=[antChs, posChs, ch_names, ch_names, ch_names]
        psds_alpha_ant_post_ratio=[]
        for i in range(epochs_num):
            for j in range(len(sumArray)):
                sumArray[j].append(calcPowerSum(psds[i], bandRange[j][0], bandRange[j][1], ch_names[j], bandWidth))
            psds_alpha_ant_post_ratio.append(np.mean(sumArray[0][i])/np.mean(sumArray[1][i]))


        for i in range(len(sumArray)):
            sumArray[i]=np.array(sumArray[i])
        _, _, alpha_sum, beta_sum, slow_sum=sumArray
        psds_alpha_ant_post_ratio=np.array(psds_alpha_ant_post_ratio)

        mean_ap_ratio=np.mean(psds_alpha_ant_post_ratio)
        sd_ap_ratio=np.std(psds_alpha_ant_post_ratio)

        beta_ratio=beta_sum/alpha_sum
        sd_beta_ratio=np.std(beta_ratio)

        slow_ratio=slow_sum/alpha_sum
        sd_slow_ratio=np.std(slow_ratio)

        mean_beta_ratio=np.mean(beta_ratio)
        mean_slow_ratio=np.mean(slow_ratio)

        bad_epochs=[]    
        
        # 各epoch的最大值
        maxValue=np.max(np.max(np.abs(epoch_data), axis=2), axis=1)

        totalEpochs=epoch_data.shape[0]

        badRatio=1
        # get bad epochs
        threshold=self.dropEpochSD
        absolute_threshold=150
        while badRatio>0.8 and threshold<5:

            bad_epochs=[i for i in range(totalEpochs) 
                        if #[i] > threshold or
                        np.max(beta_ratio[i])>mean_beta_ratio+threshold*sd_beta_ratio or
                        np.max(slow_ratio[i])>mean_slow_ratio+threshold*sd_slow_ratio or 
                        psds_alpha_ant_post_ratio[i]>mean_ap_ratio+threshold*sd_ap_ratio or
                        maxValue[i]>absolute_threshold
            ]
            badRatio=len(bad_epochs)/totalEpochs
            print ('threshold: ', threshold, 'absolute_threshold: ', absolute_threshold, 'badRatio: ', badRatio)
            
            threshold+=0.2
            absolute_threshold+=2
        
        # 讀取眼動資料
        eyeEvents=self.events
        eyeEvents=np.array(eyeEvents)
        # 前後各0.5秒都納入
        eventsFront=eyeEvents-0.5 
        eventsEnd=eyeEvents+0.5
        eyeEvents=np.concatenate((eventsFront, eventsEnd, eyeEvents))

        eyeEvents=[v//2 for v in eyeEvents ]
        # remove <0 and > totaalEpochs
        eyeEvents=[v for v in eyeEvents if v>=0 and v<totalEpochs]
        # print ('eventsPosition: ', eyeEvents)
        # concatenate bad_epochs and eyeEvents
        bad_epochs=np.concatenate((bad_epochs, eyeEvents))
        

        # remove duplicate
        bad_epochs=list(set(bad_epochs))

        badRatio=len(bad_epochs)/totalEpochs*100
        # decimal 2
        badRatio=round(badRatio, 2)
        print ('badRatio: ', badRatio)

        if totalEpochs-len(bad_epochs)>3:

            epochs=epochs.drop(bad_epochs,  verbose=False)
            bad_epochs=np.int32(bad_epochs)
            # print ('bad_epochs: ', bad_epochs)
            # drop ap_ratio by bad_epochs
            psds_alpha_ant_post_ratio=np.delete(psds_alpha_ant_post_ratio, bad_epochs, axis=0)


            # order epochs by alpha_ap_ratio
            order=np.argsort(psds_alpha_ant_post_ratio)

            # print ('order: ', order)
            epochs=epochs[order]




        if self.useRepair:            
        
            # 使用unsupervised classification重建eeg data
            epochs_repair=epochs.copy().reorder_channels(repair.electrodesOrder)
            eegData=epochs_repair.get_data(copy=True, units='uV')
            eegData = eegData.transpose(1,2,0)
            features=np.array(repair.getFeatures(eegData, self.sampling_rate))        
            epochs_repair, repair_ratio, badChnNum=repair.findArtifacts(epochs, features)
            # print ('bad_ratio: ', repair_ratio)

            bad_channels=[ch_name for ch_name in repair_ratio if repair_ratio[ch_name]>=self.removeEpochsRationThreshold]
            print ('bad_channels: ', bad_channels)
        else:
            epochs_repair=epochs
            bad_channels=[]

        return epochs_repair, badRatio, bad_channels
