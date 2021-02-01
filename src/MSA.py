import json
import numpy as np
import scipy.signal as signal


from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

class ProccessChannel:

    def __init__(self,  f_emg = 1000, fc_high = 40, fc_low = 8):
        self.f_emg = f_emg
        self.fc_high = fc_high
        fCritHP = self.fc_high/(0.5*self.f_emg)
        self.aHP, self.bHP = signal.butter(4, fCritHP, 'high', analog=True)
        self.fc_low = fc_low
        fCritLP = fc_low/(0.5*self.f_emg)
        self.aLP, self.bLP = signal.butter(4, fCritLP, 'low', analog=True)

    def go(self, emgChannel):
        channel = np.array(emgChannel)
        hpFiltered = self.highPassFilter(channel)
        rectified = self.rectify(hpFiltered)
        lpFiltered = self.lowPassFilter(rectified)
        downsampled = self.downsample(lpFiltered)
        normalised = self.nomalise(downsampled)
        processedChannel = normalised
        
        
        plt.figure(figsize=(10,10))
        plt.subplot(6,1,1)
        plt.plot(channel)

        plt.subplot(6,1,2)
        plt.plot(hpFiltered)

        plt.subplot(6,1,3)
        plt.plot(rectified)

        plt.subplot(6,1,4)
        plt.plot(lpFiltered)

        plt.subplot(6,1,5)
        plt.plot(downsampled)

        plt.subplot(6,1,6)
        plt.plot(normalised)

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
        dstotal = int((len(lpFiltered)/self.f_emg)*100)
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

    def __init__(self, rawEMG, emgFrequency=1000): 
        self.emgFrequency=emgFrequency

        self.rawData = rawEMG

        self.processEMG()

        # need to add function to select channels
        self.nmfReady()
        return
    
    def processEMG(self):
        emgProcessor = ProccessChannel()
        self.processedData ={}
        for key, value in self.rawData.items():
            self.processedData[key] = emgProcessor.go(value)
        return

    def nmfReady(self):
        labels =[]
        data = []
        for key, value in self.processedData.items():
            labels.append(key)
            data.append(value)
        self.modelLabels = labels
        self.modelData = np.array(data)
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


class MSAtrial:
    def __init__(self, rawEMG, refTVAFavg=None, refTVAFstdev=None):

        self.processedEMG = ProcessInputData(rawEMG).modelData

        self.data = self.processedEMG
        self.nCrit = 90

        self.TVAFS =[]

        self.singleSnergy(refTVAFavg, refTVAFstdev)

        self.multipleSynergies()

        self.findN90()

        return
        
    def singleSnergy(self, refTVAFavg, refTVAFstdev):
        msyn1 = RunModel(self.processedEMG, 1)
        if (refTVAFavg!=None) and (refTVAFstdev!=None):  
            self.walkDMC = msyn1.calculateWALK_DMC(refTVAFavg, refTVAFstdev)
        else:
            pass
        
        self.TVAFS.append(msyn1.TVAF)
        return

    def multipleSynergies(self):

        for i in range(2,4):
            msyns = RunModel(self.processedEMG, i)
            self.TVAFS.append(msyns.TVAF)
        
        i = i+1
        while(np.mean(self.TVAFS[-3:])<99.5):
            msyns = RunModel(self.processedEMG, i)
            self.TVAFS.append(msyns.TVAF)
            i =i+1
            if i>49:
                break
        
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

# ##################################
# emgpath = "../tests/exampledata/t1_EMG_Left.json"
# with open(emgpath, 'rb') as f:
#     rawEMG = json.load(f)

# goodCHN = ['RMG',"RTA", "RMH","RRF", 'LMG',"LTA", "LMH","LRF"]

# EMG ={}
# for c in goodCHN:
#     EMG[c] = rawEMG[c]

# # test chn process
# keys = list(EMG.keys())

# CHN = np.array(EMG[keys[0]])

# q = ProccessChannel()
# em = q.go(CHN)

# # test process input
# tr = ProcessInputData(EMG).modelData
# tr.shape


# ###########################################
# ################################################
# emgpath = "../tests/exampledata/t1_EMG_Left.json"
# with open(emgpath, 'rb') as f:
#     rawEMG = json.load(f)

# tr = ProcessInputData(rawEMG).modelData
# tr.shape


# nmfModel = RunModel(tr, 1)

# ####################################

# ################################################
# emgpath = "../tests/exampledata/t1_EMG_Left.json"
# with open(emgpath, 'rb') as f:
#     rawEMG = json.load(f)

# processedEMG = ProcessInputData(rawEMG).modelData

# trial = MSAtrial(rawEMG)
# trial = MSAtrial(tr)
# trial.N90
# trial.plotN90()

# nmfModel = RunModel(processedEMG, 1)

# ####################################

