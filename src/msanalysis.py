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


class MuscleSynergyAnalysis:

    def __init__(self, MSAdata, plot_n90=False, plot_WH=False, plot_relat=False):

        self.inputdata = MSAdata

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

        if plot_relat==True:
            self.compare_WH_input()

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
                w.append([chns[i], syn_labs[j], W[i,j]])
            
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
        plt.show()
        return

    def compare_WH_input(self):

        W, H, M = run_NMF(self.inputdata, int(np.round(self.N90,0)))

        cols = ["Muscle", "Input", "Model"]

        bl_at = [self.inputdata.flatten(), M.flatten()]
        bl_at = np.array(bl_at)

        plt.figure()
        plt.title("Input vs Approximation")
        plt.scatter(x=bl_at[0], y=bl_at[1])
        plt.xlabel("Input")
        plt.ylabel("Approximation")
        plt.show()
        return









