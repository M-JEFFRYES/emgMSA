import json
import numpy as np
import scipy.signal as signal
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


class ProccessChannel:

    def __init__(self, dsFrequency=100, f_emg = 1000, fc_high = 40, fc_low = 8):
        self.f_emg = f_emg
        self.fc_high = fc_high
        fCritHP = self.fc_high/(0.5*self.f_emg)
        self.aHP, self.bHP = signal.butter(4, fCritHP, 'high', analog=True)
        self.fc_low = fc_low
        fCritLP = fc_low/(0.5*self.f_emg)
        self.aLP, self.bLP = signal.butter(4, fCritLP, 'low', analog=True)
        self.dsFrequency = dsFrequency

    def run(self, emgChannel):
        channel = np.array(emgChannel)
        hpFiltered = self.highPassFilter(channel)
        rectified = self.rectify(hpFiltered)
        lpFiltered = self.lowPassFilter(rectified)
        downsampled = self.downsample(lpFiltered)
        normalised = self.nomalise(downsampled)
        processedChannel = normalised
        return processedChannel

    def highPassFilter(self, channel):
        hpFiltered = signal.filtfilt(self.bHP, self.aHP, channel)
        return hpFiltered

    def rectify(self, hpFiltered):
        rectified = np.sqrt(np.square(hpFiltered))
        return rectified
    
    def lowPassFilter(self, rectified):
        lpFiltered = signal.filtfilt(self.bLP, self.aLP, rectified)
        return lpFiltered

    def downsample(self, lpFiltered):
        dstotal = int((len(lpFiltered)/self.f_emg)*self.dsFrequency)
        downsampled = signal.resample(lpFiltered, dstotal)
        for i, val in enumerate(downsampled):
            if val<0:
                downsampled[i] = 0
        return downsampled
    
    def nomalise(self, downsampled):
        normalised = downsampled/max(downsampled)
        return normalised

class ProcessInputData:
    """This class loads raw EMG data, processes the channels and prepares data ready for muscle synergy analysis."""

    def __init__(self, emgData, emgFrequency=1000, dsFrequency=100): 
        self.dsFrequency=dsFrequency
        self.emgFrequency=emgFrequency
        self.emgData = emgData

        # Self preprocess EMG data
        self.processInputData()

        # Get labels and data array for the model
        self.nmfDataReady()
        return
    
    def processInputData(self):
        # create channel processor
        emgProcessor = ProccessChannel(dsFrequency=self.dsFrequency)

        self.inputData = {}
        for key, value in self.emgData.items():
            self.inputData[key] = emgProcessor.run(value)
        return
    
    def nmfDataReady(self):
        labels =[]
        data = []
        for key, value in self.inputData.items():
            labels.append(key)
            data.append(value)
        self.modelLabels = labels
        self.modelData = np.array(data)
        return
    
    def compareRawProcessedEMG(self):
        
        xraw = np.linspace(0, len(self.emgData[self.modelLabels[0]])+1, num=len(self.emgData[self.modelLabels[0]]))/self.emgFrequency
        xlin = np.linspace(0, len(self.inputData[self.modelLabels[0]])+1, num=len(self.inputData[self.modelLabels[0]]))/self.dsFrequency 
        xticks = np.linspace(0, max(xlin), num=6)
        xlabs = np.linspace(0, 100, num=6)

        for i, key in enumerate(self.modelLabels):

            rawEmg = self.emgData[key]
            linEnv = self.inputData[key]

            fig, ax = plt.subplots()
            ax.plot(xraw,rawEmg, color="blue")
            ax.set_ylim([-0.55, 0.55])
            ax.set_xlabel('Gait Cycle (%)')
            ax.set_ylabel('Raw EMG Amplitude (V)', color="blue" , fontsize=11)
            ax.set_title(f'Raw vs. processed {key} channel', fontsize=14)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabs)

            ax2=ax.twinx()
            ax2.plot(xlin, linEnv, color="red")
            ax2.set_ylabel("Processed EMG",color="red",fontsize=11)
            ax2.set_ylim([-1.1,1.1])
            plt.show()
        return

class RunModel:
    def __init__(self, processedData, noSynergies):
        self.noSynergies = noSynergies
        self.emgData = processedData

        self.nmf()

        self.calculateTVAF()
        return

    def nmf(self):
        """ Creates and NMF model. Outputs the weightings (W), activations (H) and the approximation (M). """ 
        self.model = NMF(
            init='random',
            n_components=self.noSynergies,
             random_state=0)
        # Learn NMF model for input data
        self.W = self.model.fit_transform(self.emgData)

        # Pull H component
        self.H = self.model.components_

        # M is a Reconstruct data using W and H
        self.M = self.W.dot(self.H)
        return
    
    def calculateTVAF(self):

        approx = self.M.flatten()
        actual = self.emgData.flatten()

        ss_error = np.sum(np.square(actual-approx))
        ss_actual = np.sum(np.square(actual))

        self.TVAF = (1-(ss_error/ss_actual))*100
        return

    def calculateWALK_DMC(self, referenceTVAF1avg: float, referenceTVAF1stdev: float):

        self.walkDMC = 100+10*((referenceTVAF1avg-self.TVAF)/referenceTVAF1stdev)
        return

class MSAtrial(ProcessInputData):
    def __init__(self, emgData, dsFrequency=100, refTVAFavg=None, refTVAFstdev=None):
        
        self.nCrit = 90
        self.rawEmgData = emgData
        # Process raw emg data
        ProcessInputData.__init__(self, self.rawEmgData, dsFrequency=dsFrequency)

        # Calculate tVAF1 (and walk-DMC if refs provided)
        self.TVAFS =[]
        self.calculateSingleSynergy(refTVAFavg, refTVAFstdev)

        # Calculate tVAF for a range of msyn pairs
        self.calculateMultipleSynergies()

        # Interpolate and calulate the N90
        self.findN90()
        return

    def calculateSingleSynergy(self, refTVAFavg, refTVAFstdev):
        msyn1 = RunModel(self.modelData, 1)
        if (refTVAFavg!=None) and (refTVAFstdev!=None):  
            self.walkDMC = msyn1.calculateWALK_DMC(refTVAFavg, refTVAFstdev)
        else:
            pass
        self.TVAFS.append(msyn1.TVAF)
        return

    def calculateMultipleSynergies(self):

        for i in range(2,4):
            msyns = RunModel(self.modelData, i)
            self.TVAFS.append(msyns.TVAF)
        
        i = i+1
        while(np.mean(self.TVAFS[-3:])<99.5) or (i<10):
            msyns = RunModel(self.modelData, i)
            self.TVAFS.append(msyns.TVAF)
            i =i+1
            if i>49:
                break
        return



        self.findN90()

        return
        
    def findN90(self):
        if self.TVAFS[0]<self.nCrit:
            for i, value in enumerate(self.TVAFS):
                if (value>self.nCrit):
                    deltaY = (self.nCrit-self.TVAFS[i-1])

                    gradient = (value-self.TVAFS[i-1])

                    deltaX = deltaY/gradient
                    
                    self.N90 = deltaX+i
                    break
        else:
            print("Single synergy is above the threshold")
            self.N90=1
        return

    def plotN90(self):

        x = np.arange(1, len(self.TVAFS)+1, 1)
        plt.figure()
        plt.plot(x, self.TVAFS)
        plt.plot([self.N90],[self.nCrit], 'r*', markersize=10)
        plt.axhline(90, color='gray', linestyle='--')
        plt.axvline(self.N90, color='gray', linestyle='--')

        plt.title("n_components comparison")
        plt.xlabel("Number of features")
        plt.ylabel("tVAF (%)")
        plt.show()
        return

    def compareInputApprox(self):

        noPairs = int(self.N90)
        msyn = RunModel(self.modelData, noPairs)


        xinput = np.linspace(0, len(self.modelData[0,:])+1, num=len(self.modelData[0,:]))/self.dsFrequency
        xticks = np.linspace(0, max(xinput), num=6)
        xlabs = np.linspace(0, 100, num=6)

        for i, key in enumerate(self.modelLabels):

            inputEmg = self.modelData[i]
            approxEmg = msyn.M[i]

            fig, ax = plt.subplots()
            ax.plot(xinput,inputEmg, color="blue")
            ax.plot(xinput,approxEmg, color="red")
            ax.set_ylim([-0.1, 1.1])
            ax.set_xlabel('Gait Cycle (%)')
            ax.set_ylabel('EMG Amplitude', fontsize=11)
            ax.set_title(f'Input EMG vs. Approx. EMG {key} channel', fontsize=14)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabs)
            plt.show()
        return



# ####################################

# from ezc3d import c3d
# from gaittrial import TrialData


# # pth = "F:/msc_data/C3D_FILES_REF/SUB352_1_2.c3d"
# # pth = "F:/msc_data/C3D_FILES_REF/SUB284_3_1.c3d"


# pth= "F:\msc_data\C3D_FILES_REF\MSC_NORM_2.c3d"
# trialc3d = c3d(pth)
# tr = TrialData(trialc3d)
# rawemg = tr.emgLeft
# rawemg = {}
# for key, value in tr.emgLeft.items():

#     if ("LRF" in key) or ("LMH" in key) or ("LTA" in key) or ("LMG" in key) or ("LSOL" in key):
#         pass
#     else:
#         rawemg[key] = value




# pth= "F:\msc_data\C3D_FILES_REF\MSC_NORM_5.c3d"
# trialc3d = c3d(pth)
# tr = TrialData(trialc3d)
# rawemg = tr.emgLeft
# rawemg = {}
# for key, value in tr.emgLeft.items():

#     if ("RSOL" in key) or ("RMH" in key) or ("RRF" in key):
#         pass
#     else:
#         rawemg[key] = value



# p = ProcessInputData(rawemg)


# p.compareRawProcessedEMG()

# msa = MSAtrial(rawemg)

# msa.compareRawProcessedEMG()
# msa.compareInputApprox()

# msa.N90

# msa.plotN90()



# ########




# # pth = 'F:/msc_data/C3D_FILES_REF/SUB014_1_2.c3d'
# # pth = 'F:/msc_data/C3D_FILES_REF/SUB014_1_3.c3d'
# # pth = 'F:/msc_data/C3D_FILES_REF/SUB014_1_1.c3d'
# # pth = 'F:/msc_data/C3D_FILES_REF/SUB014_1_4.c3d'



# pth = "F:/msc_data/C3D_FILES_REF/SUB284_3_1.c3d"

# trialc3d = c3d(pth)

# analogchannels = np.transpose(np.array([trialc3d['parameters']['ANALOG']['LABELS']['value'], trialc3d['parameters']['ANALOG']['DESCRIPTIONS']['value']]))
# analogdata = trialc3d['data']['analogs'][0]


# for i, chn in enumerate(analogchannels):
#     plt.figure()
#     plt.title(f'raw data {chn[0]}')
#     plt.plot(analogdata[i])
#     plt.show()


# full = tr.emgLeft
# l ={}
# r ={}
# for key,value in full.items():
#     if key[0] == 'L':
#         l[key] = value
#     else:
#         r[key] = value




# for key, value in full.items():

#     plt.figure()
#     plt.title(f'selected data {key}')
#     plt.plot(value)
#     plt.show()


# msa = MSAtrial(full)

# msaL = MSAtrial(l).plotN90()
# msaR = MSAtrial(r).plotN90()

# del full['LTA']
# del full['LMG']


# for key, value in tr.emgLeft.items():
#     plt.figure()
#     plt.title(key)
#     plt.plot(value)
#     plt.show()


