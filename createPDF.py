############################################
#  This module is used to create a PDF file with EEG analysis results
#  The PDF file contains figures of EEG epochs, PSDs, topomaps, spectrogram, 
#   and left/right power ratio
#  The EEG analysis results are also included in the PDF file
#  The PDF file is saved in the destination folder
#  The EEG jpg files are deleted after the PDF file is created
#  If ai_report_text is not None, the AI report by LLMs is included in the PDF file
############################################
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os
import mne

mne.viz.set_browser_backend('matplotlib')

class writePDF:    
    def __init__(self, filename, rawData, epochs, psds, results, dest_folder, sr, chNames, ai_report_text=None, deleteJpg=False):
        self.fileName=filename
        
        # print ('eegFullNmae: ', self.eegFullNmae)
        if dest_folder[-1] not in ['\\', '/']:
            if '/' in dest_folder:
                dest_folder+='/'
            else:
                dest_folder += '\\'

        #create a folder if not exist
        if not os.path.exists(dest_folder):
            try:
                os.makedirs(dest_folder)
            except Exception as e:
                print ('Error: ', e)
        self.dest_folder=dest_folder  
        self.results=results

        self.drawEpochs(epochs)
        self.drawPsds(psds)
        self.drawLeftRightDiff(results['LR_alpha_ratio'], results['LR_theta_ratio'], results['LR_delta_ratio'], results['alphaDiffChannels'])
        self.drawFreqPower(psds)
        self.plotTopMaps(epochs)
        self.plotSpectrogram(rawData, sr, chNames)
        self.ai_report_text=ai_report_text
        self.deleteJpg=deleteJpg
        
        self.savePDF(results)  
        # delete eeg jpg
        if deleteJpg:
            for jpg in ['eeg0.jpg','eeg1.jpg', 'eeg2.jpg', 'eeg3.jpg', 'eeg4.jpg', 'eeg5.jpg']:
                jpgFile= self.dest_folder+jpg
                if os.path.exists(jpgFile):
                    try: 
                        os.remove(jpgFile)
                    except Exception as e:
                        print ('Error: ', e)
                        continue


    def savePDF(self, results):     
        print ('savePDF')
        removeEpochsRatio=results['removeEpochsRatio']
        AP_difference=results['AP_difference']
        right_AP_difference=results['right_AP_difference']
        left_AP_difference=results['left_AP_difference']
        left_backgroud_frequency=results['left_backgroud_frequency']
        right_backgroud_frequency=results['right_backgroud_frequency']
        right_slow_ratio=results['right_slow_ratio']
        left_slow_ratio=results['left_slow_ratio']
        total_slow_ratio=results['total_slow_ratio']
        left_beta_ratio=results['left_beta_ratio']
        right_beta_ratio=results['right_beta_ratio']
        total_beta_ratio=results['total_beta_ratio']
        bad_channels=results['bad_channels']
        amplitudes=results['amplitudes']

        # create PDF with figList and removeEpochsRatio, slowRatio
        # create a pdf file
        # import FPDF class
        from fpdf import FPDF
        # 頁面大小
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        # set margins
        pdf.set_margins(left=8, top=10, right=8)
        
        line_height = 7.5
        fontName='Arial Unicode MS'
        pdf.add_font(fontName, '', './arialuni.ttf', uni=True)
    
        
        pdf.set_font(fontName, size=18)
        pdf.add_page(orientation = 'P')
        # set page horizontal    
        # eeg5.jpg
        pdf.cell(196, line_height, text='Power Spectrum Topomap', ln=1, align='C')
        # pdf.set_font(fontName, size=12)
        # pdf.cell(196, line_height-3, text=self.fileName, ln=1, align='C' )
        jpgFile = self.dest_folder+'eeg0.jpg'
        pdf.image(jpgFile, x=0, w=196)
        
        # 左右channel差異
        jpgFile=self.dest_folder+'eeg1.jpg'
        pdf.image(jpgFile, x=0, w=196)       
        
        # O1, O2 主頻率
        jpgFile=self.dest_folder+'eeg2.jpg'
        pdf.image(jpgFile, x=0, w=196)

        
        if self.ai_report_text:
            # write ai report to a page, ai_report_text is a multi-line string            
            pdf.add_page()
            pdf.set_font(fontName, size=18)
            pdf.cell(196, line_height, text='EEG AI Analysis', ln=1, align='C')
            # draw a line 
            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.5)
            pdf.line(10, 20, 200, 20)
            pdf.set_font(fontName, size=13)
            pdf.multi_cell(196, line_height-1, self.ai_report_text, 0, 'L')

        else:
            # 第一頁
            pdf.set_font(fontName, size=18)
            pdf.add_page()
            # 標題
            pdf.cell(200, 16, text='Computer EEG anlysis - '+self.fileName, ln=1, align='C',border=0)

            # 文字大小
            pdf.set_font(fontName, size=13)
            pdf.cell(200, 8, text='(Informal Report)', ln=1, align='C',border=0)

            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' Drop epochs ratio', ln=0, align='L', border=1)
            #epochs比例
            color= (0, 0, 0)
            if removeEpochsRatio>=75:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(46, line_height, text=' '+str(removeEpochsRatio)+'%', ln=0, align='L', border=1)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(24, line_height, text=' ', ln=1, align='L', border=1)

            # 振幅
            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' Amplitudes(μV)', ln=0, align='L', border=1)
            # text='α: '+str(round(amplitudes[0], 1))+' β: '+str(round(amplitudes[1], 1))+' θ: '+str(round(amplitudes[2], 1))+' δ: '+str(round(amplitudes[3], 1))
            pdf.cell(17, line_height, text='α: '+str(round(amplitudes['alpha'], 1)), ln=0, align='L', border=1)
            pdf.cell(17, line_height, text='β: '+str(round(amplitudes['beta'], 1)), ln=0, align='L', border=1)
            pdf.cell(17, line_height, text='θ: '+str(round(amplitudes['theta'], 1)), ln=0, align='L', border=1)
            pdf.cell(19, line_height, text='δ: '+str(round(amplitudes['delta'], 1)), ln=1, align='L', border=1)


            
            # 慢波比例
            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' Slow wave ratio', ln=0, align='L', border=1)
            pdf.cell(23, line_height, text=' Right', ln=0, align='L', border=1)
            pdf.cell(23, line_height, text=' Left', ln=0, align='L', border=1)
            pdf.cell(24, line_height, text=' Total', ln=1, align='L', border=1)

            #  左右、總慢波比例
            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' <60', ln=0, align='R', border=1)
            color= (0, 0, 0)
            if right_slow_ratio>=60:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(23, line_height, text=' '+str(right_slow_ratio)+'%', ln=0, align='L', border=1)
            color= (0, 0, 0)
            if left_slow_ratio>=60:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(23, line_height, text=' '+str(left_slow_ratio)+'%', ln=0, align='L', border=1)
            color= (0, 0, 0)
            if total_slow_ratio>=60:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(24, line_height, text=' '+str(total_slow_ratio)+'%', ln=1, align='L', border=1)
            pdf.set_text_color(0, 0, 0)

            # 快波比例
            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' Beta wave ratio', ln=0, align='L', border=1)
            pdf.cell(23, line_height, text=' Right', ln=0, align='L', border=1)
            pdf.cell(23, line_height, text=' Left', ln=0, align='L', border=1)
            pdf.cell(24, line_height, text=' Total', ln=1, align='L', border=1)

            #  左右、
            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' <30', ln=0, align='R', border=1)
            color= (0, 0, 0)
            if right_beta_ratio>=30:
                color=(255, 0, 0)
            pdf.set_text_color(*color)

            pdf.cell(23, line_height, text=' '+str(right_beta_ratio)+'%', ln=0, align='L', border=1)
            color= (0, 0, 0)
            if left_beta_ratio>=30:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(23, line_height, text=' '+str(left_beta_ratio)+'%', ln=0, align='L', border=1)
            color= (0, 0, 0)
            if total_beta_ratio>=30:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(24, line_height, text=' '+str(total_beta_ratio)+'%', ln=1, align='L', border=1)
            pdf.set_text_color(0, 0, 0)

            # 前後gradient
            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' AP gradient', ln=0, align='L', border=1)
            pdf.cell(23, line_height, text=' Right', ln=0, align='L', border=1)
            pdf.cell(23, line_height, text=' Left', ln=0, align='L', border=1)
            pdf.cell(24, line_height, text=' Total', ln=1, align='L', border=1)

            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' <40', ln=0, align='R', border=1)
            color= (0, 0, 0)
            if right_AP_difference>=40:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(23, line_height, text=' '+str(right_AP_difference)+'%', ln=0, align='L', border=1)
            pdf.set_text_color(0, 0, 0)
            color= (0, 0, 0)
            if left_AP_difference>=40:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(23, line_height, text=' '+str(left_AP_difference)+'%', ln=0, align='L', border=1)
            color= (0, 0, 0)
            if AP_difference>=40:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(24, line_height, text=' '+str(AP_difference)+'%', ln=1, align='L', border=1)
            pdf.set_text_color(0, 0, 0)

            # O1, O2 主頻率
            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' Background peak', ln=0, align='L', border=1)
            color= (0, 0, 0)
            if left_backgroud_frequency<8:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(23, line_height, text=' '+str(right_backgroud_frequency), ln=0, align='L', border=1)
            color= (0, 0, 0)
            if right_backgroud_frequency<8:
                color=(255, 0, 0)
            pdf.cell(23, line_height, text=' '+str(left_backgroud_frequency), ln=0, align='L', border=1)
            pdf.set_text_color((0, 0, 0))
            pdf.cell(24, line_height, text=' >=8', ln=1, align='L', border=1)

            # O1, O2 主頻率差異
            fDiff=left_backgroud_frequency-right_backgroud_frequency
            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' Background difference', ln=0, align='L', border=1)
            color= (0, 0, 0)
            if abs(fDiff)>0.5:
                color=(255, 0, 0)
            pdf.set_text_color(*color)
            pdf.cell(46, line_height, text=' '+str(round(abs(fDiff), 2)), ln=0, align='L', border=1)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(24, line_height, text='<=0.5', ln=1, align='L', border=1)

            # 左右channel差異，列出異常channel
            # for ab in abnormals:
            #     st=','.join(abnormals[ab]) 
            #     # pdf.cell(200, line_height, text=ab+': '+st, ln=1, align='C')
            #     pdf.cell(30, line_height, text='', ln=0, align='L')
            #     pdf.cell(70, line_height, text=' '+ab, ln=0, align='L', border=1)
            #     pdf.set_text_color(255, 0, 0)
            #     pdf.cell(70, line_height, text=' '+st, ln=1, align='L', border=1)
            #     pdf.set_text_color(0, 0, 0)

            
            # bad channels
            pdf.cell(30, line_height, text='', ln=0, align='L')
            pdf.cell(70, line_height, text=' Bad electrodes', ln=0, align='L', border=1)
            pdf.set_text_color(255, 0, 0)
            pdf.cell(70, line_height, text=' '+','.join(bad_channels), ln=1, align='L', border=1)   
            pdf.set_text_color(0, 0, 0)


            # 新增頁面
            pdf.add_page(orientation = 'L')
            # set page horizontal    
            # eeg5.jpg
            pdf.cell(280, line_height, text='EEG, 4s/epochs', ln=0, align='C')
            jpgFile = self.dest_folder+'eeg5.jpg'
            pdf.image(jpgFile, x=10, y=20, w=280)
            
            # 第二頁
            pdf.add_page()
            pdf.set_font(fontName, size=18)
            # 標題
            pdf.cell(200, 20, text='Topography and Power Spectrum', ln=0, align='C')
            jpgFile=self.dest_folder+'eeg3.jpg'
            pdf.image(jpgFile, w=200, x=10, y=25)

            # 第三頁
            pdf.add_page()
            pdf.set_font(fontName, size=18)
            # 標題
            pdf.cell(200, 20, text='Spectrogram', ln=0, align='C')
            jpgFile=self.dest_folder+'eeg4.jpg'
            pdf.image(jpgFile, w=200, x=10, y=25)
        outFile=self.dest_folder+self.fileName.split('.')[0]+'.pdf'
        pdf.output(outFile)
        print ('Successfully generate pdf file: ', outFile)
        # open pdf file
        # os.system('start '+ outFile)
        
    
    def drawPsds(self, psds):   
        picks=['T5', 'T6', 'O1', 'O2', 'P3', 'P4']
        powers, freqs=psds.copy().get_data(picks=picks,  return_freqs=True, fmin=1, fmax=25)
        powers=powers*1e12
        # print(freqs)
        # print ('p.shape: ', powers.shape, 'f.shape: ', freqs.shape) #p.shape:  (6, 18) f.shape:  (18,)
        powers=10*np.log10(powers)
        
        plt.figure(figsize=(12, 3))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power [dB/Hz]')
        plt.title('Posterior Dominant Frequency')
        legends=[]
            
        for i in range(powers.shape[0]):
            ch_name=picks[i]
            psd=powers[i, :]
            max=np.max(psd)
            # peaks, _ = find_peaks(psd, height=0.3*max, prominence=0.3)
            # plt.plot(freqs[peaks], psd[peaks], "x" if i%2==0 else "o")
            # legends.append(ch_name+' peaks')
            # for peak in peaks:
                
            #     plt.text(freqs[peak], psd[peak], picks[i], fontsize=12)

            # print ('ch_name: ', ch_name)
            if ch_name == 'P3':
                color='blue'
            elif ch_name == 'P4':
                color='green'
            elif i%2==0:
                color='black'
            else:
                color='red'

            lineW=0.6

            if ch_name in ['O1', 'O2']:
                plt.plot(freqs, powers[i], label=ch_name, color=color, linewidth=lineW, alpha=1)
            elif ch_name in ['P3', 'P4']:
                plt.plot(freqs,  powers[i], label=ch_name, alpha=0.7, color=color, linewidth=lineW)
            else:
                plt.plot(freqs,  powers[i], label=ch_name, linestyle='--', alpha=0.5, color=color, linewidth=lineW)
            legends.append(ch_name)
        # draw x grid
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.xticks(np.arange(3, 25,1))
        plt.title('Posterior power sepectrum densiy', fontsize=18)
            
        # plt.show()
        plt.legend(legends)
        jpgFile=self.dest_folder+'eeg2.jpg'
        plt.savefig(jpgFile, dpi=300)
        plt.close()
    

    def drawEpochs(self, epochs):
        # 畫出epochs的圖
            epochs_sub=epochs.copy()
            # color list [black, black, darkred,
            ch_num=len(epochs_sub.ch_names)
            # colorList=['black' for i in range(ch_num) if i%2==0 else 'darkred']
            # colorList=['black', 'black', 'darkred', 'darkred'.....
            colorList=['black' if i%2==0 else 'darkred' for i in range(ch_num)]
            epoch_num=len (epochs_sub)
            colorList=[colorList]*epoch_num
                        
            fig=epochs_sub.plot(scalings={'eeg': 60e-6, 'misc':50e-6, 'seeg':50e-6}, block=False, n_epochs=6, show=False,
                        overview_mode='hidden', show_scrollbars=False, epoch_colors=colorList)
            
            # fig.grab().save('eeg5.jpg')
            jpgFile=self.dest_folder+'eeg5.jpg'
            fig.savefig(jpgFile, dpi=300)
            plt.close()


    def drawLeftRightDiff(self, diffAlpha, diffTheta, diffDelta, chNames):
        fig, ax = plt.subplots(figsize=(12,3))

        # range -1-1
        ax.set_ylim([-1, 1])

        # plot bar in separate positions
        
        aplha=0.9
        ax.bar(np.arange(len(diffDelta)), diffDelta, width=0.2, label='1-4Hz', alpha=aplha, color='salmon')
        ax.bar(np.arange(len(diffTheta)) + 0.2, diffTheta, width=0.2, label='4-8Hz', alpha=aplha, color='peachpuff')
        ax.bar(np.arange(len(diffAlpha)) + 0.4, diffAlpha, width=0.2, label='8-12Hz', alpha=aplha, color='mediumseagreen')


        ax.set_xticks(np.arange(len(diffDelta)) + 0.2)
        ax.set_xticklabels(chNames)

        # plot y=0 green, y=0.5 red y=-0.5 red
        ax.axhline(y=0, color='green', linestyle='--')
        ax.axhline(y=0.5, color='red', linestyle='--')
        ax.axhline(y=-0.5, color='red', linestyle='--')

        ax.legend()
        plt.title('Left/Right power ratio, left is positive', fontsize=18)

        # Append the figure to the list
        jpgFile=self.dest_folder+'eeg1.jpg'
        plt.savefig(jpgFile, dpi=300)
        plt.close()

    def drawFreqPower(self, psds):
        # Create a 20x25 figure
        fig = plt.figure(figsize=(20, 25))
        gs = fig.add_gridspec(4, 4)    

        # psds.plot_topomap(
        #     bands = [(1,4,'Delta'), (4,8,'Theta'), (8,13,'Alpha'), (13,30,'Beta')], show=True, 
        #     normalize=True , axes=[fig.add_subplot(gs[0, i]) for i in range(4) ], cmap='coolwarm',)
        # psds:  <Power Spectrum (from Epochs, multitaper method) | 277 epochs × 21 channels × 15 freqs, 1.0-8.0 Hz>

        
        # psds=raw_epochs.compute_psd(fmin=1, fmax=30, verbose=False, method='welch', n_fft=64, n_overlap=16, n_per_seg=64)
        def plot_psds(L1, L2, axes):
            dataL1=psds.copy().pick(L1).get_data()*1e12
            dataL2=psds.copy().pick(L2).get_data()*1e12
            freqs=psds.freqs

            axes.plot(freqs, 10*np.log10(dataL1.mean(axis=0)), color='maroon', alpha=0.7)
            axes.plot(freqs, 10*np.log10(dataL2.mean(axis=0)), color='midnightblue', alpha=0.7)
            # plot vertical grid
            axes.grid(axis='x', linestyle='--', alpha=0.5)
            

            # psds.copy().pick(chs).plot( show=True, axes=axes, color='maroon', spatial_colors=False,)
            # psds.copy().pick(L2).plot( show=True,  axes=axes, color='midnightblue', spatial_colors=False, )
            axes.set_title('Power Spectrum (PSD){}'.format(' - '.join(L1+L2)))
            axes.set_xticks(range(0, 30, 1))  # Set x-axis ticks at 1 Hz intervals
            # legend concat L1 and L2, L1 
            axes.legend(['{}'.format(ch_name) for ch_name in L1+L2], labelcolor=['maroon' if ch_name in L1 else 'midnightblue' for ch_name in L1+L2])


        #plot in axes row 1 , 
        plot_psds(['O1'], ['O2'], fig.add_subplot(gs[0, 0:2]))
        plot_psds(['T5'], ['T6'], fig.add_subplot(gs[0, 2:4]))
        plot_psds(['T3'], ['T4'], fig.add_subplot(gs[1,0:2]))
        plot_psds(['P3'], ['P4'], fig.add_subplot(gs[1,2:4]))
        plot_psds(['C3'], ['C4'], fig.add_subplot(gs[2,0:2]))
        plot_psds(['F7'], ['F8'], fig.add_subplot(gs[2,2:4]))
        plot_psds(['F3'], ['F4'], fig.add_subplot(gs[3,0:2]))
        plot_psds(['Fp1'], ['Fp2'], fig.add_subplot(gs[3,2:4]))
        jpgFile=self.dest_folder+'eeg3.jpg'
        plt.savefig(jpgFile, dpi=72)
        plt.close()

    
    def plotSpectrogram(self, data, sr, picks_chs):
        f, t, Sxx = signal.spectrogram(data, fs=sr, nperseg=sr*2, noverlap=sr, nfft=sr*4)
        freq_range_mask = (f >= 1) & (f <= 30)
        f = f[freq_range_mask]
        Sxx = Sxx[:, freq_range_mask, :]

        # subplot 1 row 2 column, figsize=(20, 5)    
        plt.figure(figsize=(20, 30))
        rows=len(picks_chs)//2+1  
        
        for idx in range(len(picks_chs)):
            plt.subplot(rows, 2, idx+1)       

            plt.pcolormesh(t, f, 10 * np.log10(Sxx[idx]), cmap='coolwarm', vmin=-25, vmax=25)
            ch_name=picks_chs[idx]
            plt.ylabel(ch_name+'-Hz')
            plt.xlabel('Time [sec]')
            # plt pink line at y=8
            plt.axhline(y=8, color='darkred', linestyle='--')
            plt.colorbar(label='Power [dB/Hz]')
        jpgFile=self.dest_folder+'eeg4.jpg'
        plt.savefig(jpgFile, dpi=72)
        plt.close()

    def plotTopMaps(self, epochs):
        
        # Define your power spectrum data and channel names
        # Replace this with your actual power spectrum array and channel names

        # plot 4 columns
        fig, ax = plt.subplots(3,4, figsize=(12,9))
        # grid line
        fig.subplots_adjust( hspace=0.5)
        fv = 0.2
        L = 1*fv
        y = np.linspace(-L/2, L/2, 200)
        for i in range(4):
            for j in range(3):
                x=plot_egg_contour( ax[j,i], y, L, 0.9*fv, 0.005*fv, 0.77*fv, show=False)
                
        # y= np.concatenate((y, y))
        # x= np.concatenate((-x, x))
            


        axes=[ax[0,0], ax[0,1], ax[0,2], ax[0,3]]

        power_spectrum=epochs.copy().compute_psd(method='welch', fmin=1, fmax=30, n_jobs=4)
        # outlines outlines‘head’ | dict | None
        # The outlines to be drawn. If ‘head’, the default head scheme will be drawn. If dict, each key refers to a tuple of x and y positions, the values in ‘mask_pos’ will serve as image mask. Alternatively, a matplotlib patch object can be passed for advanced masking options, either directly or as a function that returns patches (required for multi-axis plots). If None, nothing will be drawn. Defaults to ‘head’.
        # custom outline
        outlines_dict = dict(
                        head=([],[]),
                        mask_pos=(x, y),
                        clip_radius=[0.09,0.1]

                    )
        power_spectrum.plot_topomap( normalize=True, bands = [(1,4,'Delta'), (4,8,'Theta'), (8,13,'Alpha'), (13,30,'Beta')], border=0,
                axes=axes,  contours=2, cmap=('turbo',True), outlines=outlines_dict, vlim=(0,1), sensors=False, colorbar=False)

        axes=[ax[1,0], ax[1,1], ax[1,2], ax[1,3]]
        power_spectrum.plot_topomap( normalize=False, bands = [(1,4,'Delta'), (4,8,'Theta'), (8,13,'Alpha'), (13,30,'Beta')], 
                dB=True, axes=axes,  contours=1, cmap='turbo', outlines=outlines_dict, sensors=False, vlim='joint', colorbar=True)


        axes=[ax[2,0], ax[2,1], ax[2,2], ax[2,3]]
        power_spectrum.plot_topomap( normalize=False, bands = [(1,4,'Delta'), (4,8,'Theta'), (8,13,'Alpha'), (13,30,'Beta')],  
            dB=True, axes=axes,  contours=1, cmap='turbo', outlines=outlines_dict, sensors=False, colorbar=False)
        rowTitle=[r'Normalized PSD (μV${^2}$/Hz)', 'Absolute SPD(dB)', 'Band absolute SPD(dB)']
        for i in range(3):

            for j in range(4):
                # draw border
                if j==0:
                    ax[i,j].text(-max(x), 1.4*max(y), rowTitle[i], color='black', fontsize=16)
                ax[i,j].axis('off')
                ax[i,j].plot(x, y, 'k', alpha=0.9, linewidth=0.5)
                ax[i,j].plot(-x, y, 'k', alpha=0.9, linewidth=0.5)
                # write text "Left" in the left lower corner
                ax[i,j].text(-max(x), min(y), 'Left O', color='black')  

        # ax[0,0].text(3*max(x), 1.8*max(y),'Power spectrum density', fontsize=20, color='black')
        jpgFile=self.dest_folder+'eeg0.jpg'
        fig.savefig(jpgFile, dpi=300)
        plt.close()
        


# set font chinese
plt.rc('font', family='Arial Unicode MS')
def yegg(x, L, B, w, D):
    """
    The "universal" formula for an egg, from Narushin et al., "Egg and math:
    introducing a universal formula for egg shape", *Ann. N.Y. Acad. Sci.*,
    **1505**, 169 (2021).
    x should vary between -L/2 and L/2 where L is the length of the egg; B
    is the maximum breadth of the egg; w is the distance between two vertical
    lines corresponding to the maximum breadth and y-axis (with the origin
    taken to be at the centre of the egg); D is the egg diameter at the point
    a distance L/4 from the pointed end.

    """

    fac1 = np.sqrt(5.5*L**2 + 11*L*w + 4*w**2)
    fac2 = np.sqrt(L**2 + 2*w*L + 4*w**2)
    fac3 = np.sqrt(3)*B*L
    fac4 = L**2 + 8*w*x + 4*w**2
    return (B/2) * np.sqrt((L**2 -4*x**2) / fac4) * (
        1 - (fac1 * (fac3 - 2*D*fac2) / (fac3 * (fac1 - 2*fac2)))
     * (1 - np.sqrt(L*fac4 / (2*(L - 2*w)*x**2
                    + (L**2 + 8*L*w - 4*w**2)*x + 2*L*w**2 + L**2*w + L**3))))

def plot_egg_contour(axes, y, L, B, w, D, show=True):

    x = yegg(y, L, B, w, D)
    if show:
        axes.plot(x, y, 'k')
        axes.plot(-x, y, 'k')
        axes.axis('equal')
    return x
    # plt.axis('off')






