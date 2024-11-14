##############################################
#   The main file for the EEG report generation task.
#   This file contains the main class CreateReport, which is used to 
#   generate the EEG report based on the EEG data provided.
#   The class CreateReport has the following methods:
#   - __init__(): Initializes the CreateReport class.
#   - process(): Processes the EEG data and generates the raw data and results.
#   - getMeanAmplitudes(): Calculates the mean amplitudes of the EEG data.
#   - getFeatures(): Extracts features from the EEG data.
#   - checkFollowOrder(): Checks if the patient is able to follow commands.
#   - slow_score(): Calculates the slow wave scores for the EEG data.
#   - evaluate_alpha_amp(): Evaluates the alpha amplitude of the EEG data.
#   - symmetric_frequency_of_background(): Calculates the symmetric frequency 
#       of the background EEG data.
#   - focal_slow_conclusion(): Generates the conclusion for focal slow waves.
#   - eeg_quality_conclusion(): Generates the conclusion for EEG quality.
#   - background_conclusion(): Generates the conclusion for the background EEG data.
#   - genFinalResults(): Generates the final results of the EEG report.
#   - AI_generate(): Generates AI text based on the EEG data.
#   - AI_Text_generate(): Generates AI text for the EEG report.
#   The CreateReport class takes the following parameters:
#   - fileName: The name of the EEG file.
#   - filePath: The path to the EEG file.
#   - GOOGLE_API_KEY: The Google API key for AI text generation.
#   - dest_pdfPath: The destination path for the PDF report.
#   - autogenerate: A boolean value indicating whether to automatically generate the report.
#   - outputPdf: A boolean value indicating whether to output the report as a PDF.
#   - aiReport: A boolean value indicating whether to generate an AI report.
#   - reportLang: The language of the report.
#   - model_folder: The folder containing the AI models.
#   - model_names: The names of the AI models.
#   - useRepair: A boolean value indicating whether to use repair.
#   - removeEpochsRationThreshold: The threshold for removing epochs.
#   - dropEpochSD: The standard deviation for dropping epochs.
#   The CreateReport class returns the
#   raw data and results of the EEG data.
#   The CreateReport class also generates the EEG report
#   and AI text based on the EEG data.
##############################################

import numpy as np
import os
from scipy.integrate import simpson
from eeg import eegProcess
from models import predict
import google.generativeai as genai
import markdown
from bs4 import BeautifulSoup
import time
from createPDF import writePDF
import re
import prompt as pmt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class CreateReport:
    benchmark = {
        'DBS': 0, # DBS
        'FSlow': 0, # focal slow        
    }

    alphaEpochs=None
    def __init__(self, fileName, filePath, GOOGLE_API_KEY='', dest_pdfPath='./',  
                 autogenerate=True, outputPdf=False, aiReport=False, reportLang='English',
                 model_folder='./models/',
                 model_names=['CNN', 'GoogleNet','ResNet'], useRepair=True, unit_uV=True,
                 removeEpochsRationThreshold=0.3, dropEpochSD=2.2,
                 tmax=None, tmin=None, renameChannels=True 
                ): 
        self.benchmark = {
            'DBS': 0, # DBS
            'FSlow': 0, # focal slow        
        }
        # if path not end with '\', or '/', add it
        if filePath[-1] not in ['\\', '/']:
            if '/' in filePath:
                filePath+='/'
            else:
                filePath += '\\'
        self.unit_uV=unit_uV
        self.eegFullName=filePath+fileName
        self.dest_pdfPath=dest_pdfPath
        self.model_folder=model_folder
        self.fileName=fileName
        self.filePath=filePath
        self.removeEpochsRationThreshold=removeEpochsRationThreshold
        self.model_names=model_names
        self.useRepair=useRepair
        self.results=None
        self.GOOGLE_API_KEY=GOOGLE_API_KEY
        self.outputPdf=outputPdf
        self.dropEpochSD=dropEpochSD
        self.aiReport=aiReport
        self.reportLang=reportLang
        self.tmax=tmax
        self.tmin=tmin
        self.renameChannels=renameChannels
        
        if os.path.isfile(self.eegFullName):            
            print ('File exists: ', self.eegFullName)
            if autogenerate:
                raw, _=self.process()
                self.raw=raw
                print ('Run AI_Text_generate() to generate the report')
                
            else:
                self.raw=None
                self.results=None
                print ("""Please call process() to create the raw data and results""")
        else:
            print('File not exists: ', self.eegFullName)
    
    
    def getMeanAmplitudes(self,epochs):
        bandNames=['alpha', 'beta', 'theta', 'delta']
        bandFreqs=[[8, 13], [13, 30], [4, 8], [1, 4]]
        epochs_crop=epochs.copy().crop(tmax=0.5)
        results={}
        for i in range(4):
            band=bandNames[i]
            low, high=bandFreqs[i]
            band_epo=epochs_crop.copy().filter(low, high, verbose=False)
            data=band_epo.get_data(units='uV')
            amp=np.max(data, axis=2)-np.min(data, axis=2)
            # TOP 2% quantile
            amp=np.quantile(amp.reshape(-1), 0.98)
            results[band]=round(amp, 2)
        return results


    ################### get features ###################
    def getFeatures(self, epochs):
        
        chn_order=['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 
                   'C3', 'C4', 'T3', 'T4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2']
        # if epochs don't have the chanel in chn_order, it will raise error
        if not set(chn_order).issubset(set(epochs.ch_names)):
            print ('Error: channels not match', 'Must have channels: ', chn_order)
            return None, None
        epochs.reorder_channels(chn_order)      

        epoch_for_psd=epochs.copy()        
        sfr=epochs.info['sfreq']
        epochLength=epochs._data.shape[2]
        
        # calculate bandwidth and psds
        bandWidth=np.round(sfr/epochLength,2)
        psds=epoch_for_psd.compute_psd(method='multitaper', fmin=1.5, fmax=30, bandwidth=bandWidth, n_jobs=4, verbose=False)
        
        rightPreds, leftPreds=predict(self.model_folder, self.model_names).ensemble(psds.copy().average())
        print ('rightPreds: ', rightPreds, 'leftPreds: ', leftPreds)

        bandPsds=[[], [], []]
        bandRange=[(1.5, 4), (4, 8), (8, 13)]
        for i in range(3):
            band=bandRange[i]
            bandPsds[i]=psds.copy().average().get_data(fmin=band[0], fmax=band[1])*1e12# convert V^2/Hz to uV^2/Hz
        psdsDelta, psdsTheta, psdsAlpha=bandPsds


        ch_names=epochs.ch_names 
        
        def pltDiff(psds):
            # L_R_diff odd channel - even channel
            L_R_diff=[]
            L_R_Channels=[]
            for i in range(0, len(ch_names), 2):
                # get psd1: left channel, psd2: right channel
                psd1=psds[i, :]
                psd2=psds[i+1, :]
                # flatten psd1, psd2 and sum
                psd1=simpson(psd1.reshape(-1), dx=bandWidth)
                psd2=simpson(psd2.reshape(-1), dx=bandWidth)
                
                diff=round(2*(psd1-psd2)/(psd1+psd2)  ,3)
                L_R_diff.append(diff)
                L_R_Channels.append(ch_names[i]+'-'+ch_names[i+1])
            return L_R_diff, L_R_Channels        

        diff1, _ = pltDiff(psdsDelta)
        diff2, _ = pltDiff(psdsTheta)
        diff3, alphaDiffChannels = pltDiff(psdsAlpha)        

        right_chs=['Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
        left_chs=['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5']

        def calcPowerSum(psds, fmin, fmax, pick_chs, bandWidth):
            psds=psds.copy().pick(pick_chs).average().get_data(fmin=fmin, fmax=fmax)*1e12
            psds= np.average( simpson(psds, dx=bandWidth))
            return psds

        # slow /all ratio
        bandRange=[(1.5, 8), (8, 13), (13, 30), (1.5, 30)]

        # slow, beta, all power sum, right/left/all
        pSums=[]
        for i in range(len(bandRange)):
            band=bandRange[i]
            for c in [right_chs, left_chs, ch_names]:
                pSum=calcPowerSum(psds, band[0], band[1], c, bandWidth)
                pSums.append(pSum)

        right_slow, left_slow, total_slow=pSums[0:3]
        right_alpha, left_alpha, total_alpha=pSums[3:6]
        right_beta, left_beta, total_beta=pSums[6:9]
        right_all, left_all, total_all=pSums[9:12]
        
        # slow ratio
        right_slow_ratio=round(100*right_slow/right_all, 2)
        left_slow_ratio=round(100*left_slow/left_all, 2)
        all_slow_ratio=round(100*total_slow/total_all, 2)

        # beta ratio
        right_beta_ratio=round(100*right_beta/right_all, 2)
        left_beta_ratio=round(100*left_beta/left_all, 2)
        all_beta_ratio=round(100*total_beta/total_all, 2)

        # beta/alpha ratio
        right_beta_alpha_ratio=round(100*right_beta/right_alpha, 2)
        left_beta_alpha_ratio=round(100*left_beta/left_alpha, 2)
        all_beta_alpha_ratio=round(100*total_beta/total_alpha, 2)
        
        # AP difference
        antChs=['Fp1', 'Fp2', 'F7', 'F8', 'F3',  'F4']
        right_antChs=['Fp2', 'F4', 'F8']
        left_antChs=['Fp1', 'F3', 'F7']
        posChs=['T5', 'T6', 'P3', 'P4', 'O1', 'O2']
        right_posChs=['T6', 'P4', 'O2']
        left_posChs=['T5', 'P3', 'O1']

        APPsds=[]

        for c in [antChs, posChs, right_antChs, right_posChs, left_antChs, left_posChs]:
            pSum=calcPowerSum(psds, 8, 13, c, bandWidth)
            APPsds.append(pSum)
        antPsds,posPsds=APPsds[0:2]
        right_antPsds,right_posPsds=APPsds[2:4]
        left_antPsds,left_posPsds=APPsds[4:6]

        AP_difference=100*antPsds/(antPsds+posPsds)
        AP_difference=round(AP_difference, 2)

        right_AP_difference=100*right_antPsds/(right_antPsds+right_posPsds)
        right_AP_difference=round(right_AP_difference, 2)

        left_AP_difference=100*left_antPsds/(left_antPsds+left_posPsds)
        left_AP_difference=round(left_AP_difference, 2)


        # log        
        results={
            'AP_difference': AP_difference,
            'right_AP_difference': right_AP_difference,
            'left_AP_difference': left_AP_difference,
            'left_backgroud_frequency': leftPreds,
            'right_backgroud_frequency': rightPreds,
            'right_slow_ratio': right_slow_ratio,
            'left_slow_ratio': left_slow_ratio,
            'total_slow_ratio': all_slow_ratio,
            'right_beta_ratio': right_beta_ratio,
            'left_beta_ratio': left_beta_ratio,
            'total_beta_ratio': all_beta_ratio,
            'right_beta_alpha_ratio': right_beta_alpha_ratio,
            'left_beta_alpha_ratio': left_beta_alpha_ratio,
            'total_beta_alpha_ratio': all_beta_alpha_ratio,
            'LR_delta_ratio': diff1,
            'LR_theta_ratio': diff2,
            'LR_alpha_ratio': diff3,
            'alphaDiffChannels': alphaDiffChannels,
            'DiffTwoFold': "1",
            "followOrder": ''
        }
        
        return  results, psds.average()


    def checkFollowOrder(self):
        # event filename=edf_file_name.split('.')[0]+'.txt'
        edf_file=self.eegFullName
        # print ('edf_file: ', edf_file)
        event_file=edf_file.replace('.edf', '.txt')
        # print ('event_file: ', event_file)
        followOrder="True"
        if os.path.isfile(event_file):
            
            with open(event_file, 'r') as f:
                lines=f.readlines()
                for line in lines:
                    if "Unable to Follow Commands" in line:
                        followOrder="False"
                        break
        return followOrder

    def process(self):
        eegWork=eegProcess(self.fileName, self.filePath, useRepair=self.useRepair, 
                           dropEpochSD=self.dropEpochSD, unit_uV=self.unit_uV, 
                           tmax=self.tmax, tmin=self.tmin, renameChannels=self.renameChannels)
        raw, events=eegWork.getRawData()
        sample_rate=int(raw.info['sfreq'])
        self.sample_rate=sample_rate
        epochs,bad_ratio, bad_channels=eegWork.extractAlphaEpochs()
        results, psds=self.getFeatures(epochs)
        self.alphaEpochs=epochs
        rawData=raw.copy().get_data(units='uV')
        amplitude=self.getMeanAmplitudes(epochs)
        results['removeEpochsRatio']=bad_ratio
        results['bad_channels']=bad_channels
        results['amplitudes']=amplitude
        self.results=results
        self.genFinalResults()

        if self.outputPdf:  
            ai_report_text=None
            if self.aiReport:          
                ai_report_text=self.AI_Text_generate()
            writePDF(self.fileName, rawData, epochs, psds, results,self.dest_pdfPath, sample_rate, raw.ch_names, ai_report_text)

        return raw, results
    


    optionBackground = [
        'Normal background frequency',
        'Diffuse background slowing',
    ]

    optionAmplitude = [
        'low (<10 mV)',
        'medium (10–50 mV)',
        'high (>50 mV)'
    ]

    frequencySym = [
        'symmetric',
        'lower in right',
        'lower in left',
        'borderline lower in right',
        'borderline lower in left'
    ]

    amplitudeSym = [
        'symmetric',
        'lower in right',
        'lower in left',
        'borderline lower in right',
        'borderline lower in left'
    ]


    def slow_score(self):
        left_right_delta_ratio = self.results['LR_delta_ratio']
        left_right_theta_ratio = self.results['LR_theta_ratio']
        left_right_alpha_ratio = self.results['LR_alpha_ratio']
        bad_channels = self.results['bad_channels']
        # channels ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'C3', 'C4', 'T3', 'T4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2']
        right_channels = ['Fp2', 'F8', 'F4', 'C4', 'T4', 'T6', 'P4', 'O2']
        right_posterior_channels = ['T6', 'P4', 'O2']
        left_channels = ['Fp1', 'F7', 'F3', 'C3', 'T3', 'T5', 'P3', 'O1']
        left_posterior_channels = ['T5', 'P3', 'O1']
        abnormal_threshold = 0.5

        right_slow_channels = []
        left_slow_channels = []

        for i in range(len(left_right_delta_ratio)):
            theta_ratio = left_right_theta_ratio[i]
            delta_ratio = left_right_delta_ratio[i]
            alpha_ratio = left_right_alpha_ratio[i]
            
            leftCh=left_channels[i]
            rightCh=right_channels[i]
            if delta_ratio >= abnormal_threshold or theta_ratio >= abnormal_threshold:
                if leftCh not in left_slow_channels \
                    and not (leftCh in left_posterior_channels and max(delta_ratio, theta_ratio, alpha_ratio) == alpha_ratio) \
                    and leftCh not in bad_channels:
                    left_slow_channels.append(left_channels[i])
            elif delta_ratio <= -abnormal_threshold or theta_ratio <= -abnormal_threshold:
                if rightCh not in right_slow_channels \
                    and not (rightCh in right_posterior_channels and min(delta_ratio, theta_ratio, alpha_ratio) == alpha_ratio) \
                    and rightCh not in bad_channels:
                    right_slow_channels.append(right_channels[i])

        # print('left_slow_channels', left_slow_channels, 'right_slow_channels', right_slow_channels)


        left_adjacents = {
            'Fp1': ['F7', 'F3'],
            'F7': ['Fp1', 'F3', 'T3'],
            'F3': ['Fp1', 'F7', 'C3'],
            'C3': ['F3', 'T3', 'P3'],
            'T3': ['F3', 'C3', 'T5'],
            'T5': ['F7', 'T3', 'P3'],
            'P3': ['C3', 'T5', 'O1'],
            'O1': ['P3', 'T5']
        }

        right_adjacents = {
            'Fp2': ['F8', 'F4'],
            'F8': ['Fp2', 'F4', 'T4'],
            'F4': ['Fp2', 'F8', 'C4'],
            'C4': ['F4', 'T4', 'P4'],
            'T4': ['F4', 'C4', 'T6'],
            'T6': ['F8', 'T4', 'P4'],
            'P4': ['C4', 'T6', 'O2'],
            'O2': ['P4', 'T6']
        }

        slow_scores =[0,0]
        slow_channels = [[],[]]
        for i in range(2):
            s_chs= left_slow_channels if i == 0 else right_slow_channels
            for ch in s_chs:
                adj_of_ch = left_adjacents[ch] if i == 0 else right_adjacents[ch]
                # has_neighbour = False
                for adj_ch in adj_of_ch:
                    if adj_ch in s_chs:
                        # has_neighbour = True
                        slow_channels[i].append(ch)
                        all_chs = left_channels if i == 0 else right_channels
                        index=all_chs.index(ch) 

                        for r in [abs(left_right_delta_ratio[index]), abs(left_right_theta_ratio[index])]:
                            if r >= abnormal_threshold:
                                slow_scores[i] += round(r, 3)
                                break
                                
                        break

        return slow_scores, slow_channels


    def evaluate_alpha_amp(self):

        left_right_alpha_ratio = self.results['LR_alpha_ratio']
        bad_channels = self.results['bad_channels']
        val = left_right_alpha_ratio
        left_lower = []
        right_slow=self.right_slow_channels
        right_lower = [] 
        left_slow=self.left_slow_channels

        left_score = 0
        right_score = 0
        right_channels = ['Fp2', 'F8', 'F4', 'C4', 'T4', 'T6', 'P4', 'O2']
        left_channels = ['Fp1', 'F7', 'F3', 'C3', 'T3', 'T5', 'P3', 'O1']
        abnormal_threshold = 0.5
        bad_channels = bad_channels

        for i in range(1,len(val)): 

            ch_r = right_channels[i]
            ch_l = left_channels[i]
            if val[i] >= abnormal_threshold:
                if ch_l not in left_lower and ch_l not in bad_channels and ch_l not in left_slow:
                    right_lower.append(ch_r)
                    right_score += abs(val[i])
            elif val[i] <= -abnormal_threshold:
                if ch_r not in right_lower and ch_r not in bad_channels and ch_r not in right_slow:
                    left_lower.append(ch_l)
                    left_score += abs(val[i])

        self.results['left_lower_alpha_channels'] = left_lower
        self.results['right_lower_alpha_channels'] = right_lower
        self.results['right_lower_alpha_score'] = right_score
        self.results['left_lower_alpha_score'] = left_score

        value = ''
        lowAlphaChannels = []
        if right_score > left_score and right_score >1.6:
            #right abnormally 
            value = self.amplitudeSym[1]
            lowAlphaChannels = right_lower
            
        elif left_score > right_score and left_score >1.6:
            value = self.amplitudeSym[2]
            lowAlphaChannels = left_lower
        else:
            value = self.amplitudeSym[0]

        # print('amplitude_symmetry', value)
        bg_amplitude_symmetry = value
        
        return bg_amplitude_symmetry, lowAlphaChannels

    def symmetric_frequency_of_background(self):
        left_freq = self.results['left_backgroud_frequency']
        right_freq = self.results['right_backgroud_frequency']
        value = ''
        # >0.5 Hz difference
        # print('left_freq', left_freq, 'right_freq', right_freq)

        if left_freq - right_freq >= 1:
            value = self.frequencySym[1]
        elif right_freq - left_freq >=1:
            value = self.frequencySym[2]
        else:
            value = self.frequencySym[0]

        # print('symmetric_frequency_of_background', value)
        return value
    def focal_slow_conclusion(self):
        slow_scores, slow_channels=self.slow_score()
        self.left_slow_channels=slow_channels[0]
        self.right_slow_channels=slow_channels[1]
        self.results['left_slow_channels']=slow_channels[0]
        self.results['right_slow_channels']=slow_channels[1]
        self.results['right_slow_score']=slow_scores[1]
        self.results['left_slow_score']=slow_scores[0]
        # print ('slow_scores: ', slow_scores)
        # print ('slow_channels: ', slow_channels)
        str=''
        for i in range(2):
            if slow_scores[i] >= 2.4:
                self.benchmark['FSlow'] = 1
                str='Focal abnormality: higher slow wave power in  {} when compare left and right channels'.format(', '.join(slow_channels[i]))
                break
        
        return str

    def eeg_quality_conclusion(self):
        results=self.results
        str=''
        bad_ratio = results['removeEpochsRatio']
        
        if bad_ratio <= 50:
            str='Good'
        elif bad_ratio <= 75:
            str='Fair'
        elif results['removeEpochsRatio'] > 75:
            str='Poor'
        return str

    def background_conclusion(self, lowerAlphaChannels):
        results=self.results
        right_background_frequency = results['right_backgroud_frequency']
        left_background_frequency = results['left_backgroud_frequency']
        right_slow_ratio = results['right_slow_ratio']
        left_slow_ratio = results['left_slow_ratio']
        total_slow_ratio = results['total_slow_ratio']
        right_beta_ratio = results['right_beta_ratio']
        left_beta_ratio = results['left_beta_ratio']
        total_beta_ratio = results['total_beta_ratio']

        alpha_amplitude = results['amplitudes']['alpha']
        ap_gradient = results['AP_difference']
        max_freq = max(right_background_frequency, left_background_frequency)
        min_freq = min(right_background_frequency, left_background_frequency)
        
        # mild diffuse background slowing
        if  (max_freq <7.5) or (max_freq < 8 and total_slow_ratio >50 ):
            results['bg_active'] = self.optionBackground[1]
            self.benchmark['DBS'] = 1
            # self.benchmark['AI4'] = 0
        
        # elif  min(right_slow_ratio, left_slow_ratio) >=60 :             
            
        #     results['bg_active'] = self.optionBackground[1]
        #     self.benchmark['DBS'] = 1
            # self.benchmark['AI4'] = 0
        else:
            results['bg_active'] = self.optionBackground[0]
            self.benchmark['DBS'] = 0

        if alpha_amplitude < 10:
            results['bg_amp'] = self.optionAmplitude[0]
        elif alpha_amplitude <= 50:
            results['bg_amp'] = self.optionAmplitude[1]
        else:
            results['bg_amp'] = self.optionAmplitude[2]

        if results['bg_amp_sym'] != self.amplitudeSym[0] or  results['bg_freq'] != self.frequencySym[0]:
            self.benchmark['FSlow'] = 1        

        self.results=results
        str_abnormal_bg = ''
        if results['bg_active'] != self.optionBackground[0] :
            str_abnormal_bg = f'Abnormal, diffuse bacground slowing;'           
        
        if results['focal_abnormality']!= '' or results['bg_amp_sym']!=self.amplitudeSym[0] or results['bg_freq']!=self.frequencySym[0]:
            str_abnormal_bg+= 'Focal slow wave or asymmetric abnormality detected:'
            str_lower_alpha = ', '.join(lowerAlphaChannels)
            if results['focal_abnormality']!= '':
                str_abnormal_bg+=results['focal_abnormality']
            if results['bg_amp_sym']!=self.amplitudeSym[0]:
                str_abnormal_bg+=f';Lower alpha amplitude in {str_lower_alpha} channels'
            if results['bg_freq']!=self.frequencySym[0]:
                str_abnormal_bg+=f';Lower dominant frequency in {results["bg_freq"]} channels'    

        if str_abnormal_bg=='':
            str_abnormal_bg='Normal background activity'    

        return str_abnormal_bg

    def genFinalResults(self):
        results=self.results
            # example of results

        finalResults={'EEG_quality':'',
                    'bad_channels':'',
                    'backgroundFrequency':'',
                    'bg_active':'',
                    'bg_amp':'',
                    'bg_amp_sym':'',
                    'bg_freq':'',
                    'abnormalFindings':[''],
                }
        finalResults['bad_channels']=results['bad_channels']
        finalResults['backgroundFrequency'] = 'Right: ' + str(results['right_backgroud_frequency']) + ' Hz, Left: ' + str(results['left_backgroud_frequency']) + ' Hz'


        focal_slow=self.focal_slow_conclusion()
        # print ('focal_slow: ', focal_slow)
        # if not focal_slow=='':
        finalResults['abnormalFindings'].append(focal_slow)
        results['focal_abnormality']=focal_slow


            
        bg_amplitude_symmetry, lowerAlphaChannels=self.evaluate_alpha_amp()
        results['bg_amp_sym']=bg_amplitude_symmetry
        finalResults['bg_amp_sym']=bg_amplitude_symmetry
        # print ('alpha_amp: ', bg_amplitude_symmetry, lowerAlphaChannels)
        symmetric_background= self.symmetric_frequency_of_background()
        results['bg_freq']=symmetric_background
        finalResults['bg_freq']=symmetric_background
        # print ('backgroundFrequency: ', symmetric_background, lowerAlphaChannels)

        eeg_quality=self.eeg_quality_conclusion()
        # print ('eeg_quality: ', eeg_quality)
        finalResults['EEG_quality']=eeg_quality

        bg_conclusion=self.background_conclusion(lowerAlphaChannels)
        # print ('conclusion: ', bg_conclusion)
        if bg_conclusion != 'Normal background activity':
            finalResults['abnormalFindings'].append(bg_conclusion)
        
        finalResults['bg_active']=results['bg_active']
        finalResults['bg_amp']=results['bg_amp']
            
        self.finalResults=finalResults
        self.results=results
        return finalResults
    
    def AI_generate(self, message, token=None, model_name='gemini-1.5-pro'):
        genai.configure(api_key=self.GOOGLE_API_KEY)
        limitWords=''
        if token:
            limitWords=' Output in {} words'.format(token)
        else:
            generation_config =None
            
        model = genai.GenerativeModel(model_name)
        i=1
        
        while i<4:
            print ('Attempt: {}, AI is generating the content...'.format(i))
            try:
                response = model.generate_content([message,limitWords])
                # print(response)
                text=response.text
                html=markdown.markdown(text)
                soup = BeautifulSoup(html, features='html.parser')
                return soup.get_text()
                break
            except Exception as e:
                i+=1
                print(e)
                time.sleep(20)
                continue

        return ''


    def AI_Text_generate(self, reportLang=None, prompt=0, token=None, model= 'gemini-1.5-pro-latest'):
        if not self.finalResults:
            self.genFinalResults()
        finalResults=self.finalResults
        if not reportLang:
            reportLang=self.reportLang

        if prompt==0:
            message=pmt.reportPrompt(finalResults, reportLang, promptLength='long')
    
        elif prompt==1:
            message=pmt.reportPrompt(finalResults, reportLang, promptLength='medium')

        elif prompt==2:
            message=pmt.reportPrompt(finalResults, reportLang, promptLength='short')

        text=self.AI_generate(message, token, model)
        # remove blank lines
        text = os.linesep.join([s for s in text.splitlines() if s])
        # use regexp to insert  \n before === if line start with ===
        text = re.sub(r'(?m)^(===)', r'\n\1', text)
        return text
