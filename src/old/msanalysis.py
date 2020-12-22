import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json

def run_NMF(data, features):
    """ Creates and NMF model. Outputs the weightings (W), activations (H) and the approximation (M). """ 

    model = NMF(
        init='random',
        n_components=features,
        random_state=0
    )
    # Learn NMF model for input data
    W = model.fit_transform(data)

    # Pull H component
    H = model.components_

    # M is a Reconstruct data using W and H
    M = W.dot(H)
    return W, H, M

def calculate_tVAF(inputdata, M):
    """
    Calculates the percentage of the signal represented by the NMF ouput
    """

    actual = inputdata.flatten()
    approx = M.flatten()

    SQerror = 0
    SQemg = 0
    for i in range(len(approx)):
        SQerror += ((actual[i]-approx[i])**2)
        SQemg += (actual[i])**2
    
    tVAF = (1-(SQerror/SQemg))*100
    return tVAF

def find_N90(features_tvaf):
    """"Find the N90 score using the list of tVAF scores for varying amounts of features no=i+1.
    Set threshold to 90 (some practice datasets have tVAF of >90)"""

    threshold = 90

    # Check for non-convergence
    if features_tvaf[-1] <threshold:
        return print("Threshold ({}) not met".format(threshold))

    # Find no of features that passes the threshold
    for i in range(len(features_tvaf)):
        if features_tvaf[i] > threshold:
            break
        else:
            pass
    
    m = (features_tvaf[i]-features_tvaf[i-1])/1 # interpolation - gradient

    dist = features_tvaf[i]-threshold  # dist to 90 (change in y)
    
    x = dist/m # (change in x)

    N90 = (i-x)+1 # n90 is i - the x dist from the threshold (+1 for zero index)
    return N90

####################################

def bland_altman_plot(data1, data2):
    #data1     = np.asarray(data1)
    #data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    plt.figure()
    plt.title("Signal vs Approximation Bland-Altman Graph")
    plt.scatter(mean, diff)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel("Signal")
    plt.ylabel("NMF Approximation")
    plt.show()
    return

####################################

class MuscleSynergyAnalysis:

    def __init__(self, MSAdata, channels_used, plot_n90=False, plot_WH=False, plot_MSA_info=False):

        self.inputdata = MSAdata
        self.chns = channels_used

        # Calculate the tVAF1
        self.single_synergy_analysis()

        # Find the N90 value
        self.assess_synergy_tvaf()

        if plot_n90==True:
            # Plot the features vs tVAF relationship
            self.plot_features_tvaf()
        
        if plot_WH==True:
            #Plot the weighting and activation patterns for N90 - ish
            self.plot_weighings_activations()

        if plot_MSA_info==True:
            self.compare_WH_input()
            self.plot_input_approx()
            self.plot_synergy_contribution()

    def single_synergy_analysis(self):
        """
        Performs NMF for a single feature, returning the total variation accounted for by a single synergy.
        """
        W, H, M = run_NMF(self.inputdata, 1)

        self.tVAF1 = calculate_tVAF(self.inputdata, M) 
        return
    
    def assess_synergy_tvaf(self):
        """
        Performs NMF for a a range of feature, returning a list of values of the total variation accounted for by that number of synergies.
        """
        max_features = 25

        self.features_tVAF = []

        for i in range(1, max_features):
            W,H,M = run_NMF(self.inputdata, i)
            self.features_tVAF.append(calculate_tVAF(self.inputdata, M))
        
        if self.features_tVAF[0] <90:
            self.N90 = find_N90(self.features_tVAF)
        else:
            print("tVAF1 greater than 90%")
            self.N90 = None
        return
    
    def plot_features_tvaf(self):
        """ Plot the features vs tVAF. """
    
        plt.figure()
        plt.plot(self.features_tVAF)
        plt.axhline(90,           color='gray', linestyle='--')
        plt.title("n_components comparison")
        plt.xlabel("Number of features")
        plt.ylabel("tVAF (%)")
        
        plt.show()
        return

    def calculate_walk_DMC(self, REF_tVAF1_AVG, REF_tVAF1_SD):
        """ Calculates the walk-DMC score using the average and standard deviation of the tVAF1 values for the reference group
        and the tVAF1 of the subject."""

        self.walkDMC = 100+10*((REF_tVAF1_AVG-self.tVAF1)/REF_tVAF1_SD)
        return

    def plot_weighings_activations(self):

        W, H, M = run_NMF(self.inputdata, int(np.round(self.N90,0)))

        # Prepare Activation weightings data for plotting
        cols = ["Muscle", "Synergy", "Weighting"]
        syn_labs = ["SYN_{}".format(x+1) for x in range(len(W[0,:]))]

        w = []
        for i in range(len(W[:,0])):
            for j in range(len(W[0,:])):
                w.append([self.chns[i], syn_labs[j], W[i,j]])
            
        dfW = pd.DataFrame(data=w, columns=cols)

        # Prepare activation patters
        h = H.transpose()

        dfH = pd.DataFrame(columns=syn_labs, data=h)

        # Plot activations and patterns
        plt.title("Synergy Acivation Weightings") 
        axW = sns.barplot(x="Muscle", y="Weighting", hue="Synergy", data=dfW)
        plt.show()

        plt.title("Synergy Acivation Patterns")
        axH = sns.lineplot(data=dfH)
        plt.xlabel("Sample")
        plt.ylabel("Pattern Scaling Value")
        plt.show()
        return

    def compare_WH_input(self):

        self._W, self._H, self._M = run_NMF(self.inputdata, int(np.round(self.N90,0)))

        cols = ["Muscle", "Input", "Model"]

        bl_at = [self.inputdata.flatten(), self._M.flatten()]
        bl_at = np.array(bl_at)

        plt.figure()
        plt.title("Input vs Approximation")
        plt.scatter(x=bl_at[0], y=bl_at[1])
        plt.xlabel("Input")
        plt.ylabel("Approximation")
        plt.show()
        return
    
    def plot_input_approx(self):
        for i in range(len(self._M)):

            plt.figure()
            plt.title(self.chns[i])
            signal = plt.plot(self.inputdata[i], 'r', label="Input Signal")
            approx = plt.plot(self._M[i], 'g', label="NMF Approximation")
            plt.xlabel("Sample")
            plt.ylabel("Signal Amplitude")
            plt.legend()
            plt.show()
        
        bland_altman_plot(self.inputdata.flatten(), self._M.flatten())

        return
    
    def plot_synergy_contribution(self):
        for i in range(len(self._W)):  
            plt.figure()
            plt.title(self.chns[i])
            plt.plot(self.inputdata[i], 'g')
            plt.plot(self._M[i], 'r')
            for j in range(len(self._W[0])):
                plt.plot(self._W[i,j]*self._H[j],'--')
            plt.xlabel("Sample")
            plt.ylabel("Signal Amplitude")
            plt.show()
        return


#chns = ['LRF','LMH','LTA','LMG','RRF','RMH','RTA','RMG']
#chns_l = ['LRF','LMH','LTA','LMG']
#chns_r = ['RRF','RMH','RTA','RMG']

#x = MuscleSynergyAnalysis(abs(emg_b), chns, plot_n90=True)

#x = MuscleSynergyAnalysis(abs(emg_b), chns, plot_MSA_info=True, plot_WH=True)


