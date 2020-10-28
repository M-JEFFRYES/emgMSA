import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def envelope_EMG(array):
    """ #### Needs improvement"""
    for i in range(len(array)):
        array[i] = np.sqrt(array[i]**2)
    
    env= []
    for i in range(2, len(array)-2):
        env.append((np.sum(array[i-2:i+2]))/5)
    return env

def get_input_dataset(df, slice_start, slice_end):
    """Pull data from a dataframe and produce a 2D array of data ready for the model. """

    data = df.iloc[slice_start:slice_end,:]
    input_data = []
    for col in data.columns:
        print(data[col])
        input_data.append(envelope_EMG(list(data[col])))

    input_data = np.array(input_data)
    return input_data

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

def assess_NMF(inp_data, W,H,M):
    """ #### Needs checking."""

    approx = M.flatten()
    EMG = inp_data.flatten()

    RMS = 0
    for i in range(len(EMG)):
        RMS+=(EMG[i]-approx[i])**2
    RMS = np.sqrt(RMS/len(approx))

    print("RMS: {}".format(RMS))
    return RMS, EMG

def calculate_tVAF(actual, M):

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

    threshold = 98

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


def plot_tVAF(features_tvaf):
    """ Plot the features vs tVAF. """
    
    plt.figure()
    plt.plot(features_tvaf)
    plt.title("n_components comparison")
    plt.xlabel("Number of features")
    plt.ylabel("tVAF (%)")
    
    plt.show()
    return


def investigate_EMG_dataset(data, max_features, plot_features=False):
    
    # 1d array of the actual EMG data
    EMG = data.flatten()

    # NMF run for one feature
    W,H,M = run_NMF(data, 1)

    # calculate the tVAF for a single synergy
    tVAF1 = calculate_tVAF(EMG, M)
    
    # Plot tVAF for increasing no of features
    features_tvaf =[]
    for i in range(1, 50):#max_features):
        W,H,M = run_NMF(data, i)
        features_tvaf.append(calculate_tVAF(EMG, M))
    
    # Find N90
    N90 = find_N90(features_tvaf)

    # Plot tVAF
    if plot_features == True:
        plot_tVAF(features_tvaf)

    analysis_data = {
        'tVAF1': tVAF1,
        'features_tVAF': features_tvaf,
        'N90': N90
    }
    return analysis_data


def calculate_walk_DMC(REF_tVAF1_AVG, SUBJ_tVAF1, REF_tVAF1_SD):
    """ Calculates the walk-DMC score using the average and standard deviation of the tVAF1 values for the reference group
    and the tVAF1 of the subject."""

    walkDMC = 100+10((REF_tVAF1_AVG-SUBJ_tVAF1)/REF_tVAF1_SD)
    return walkDMC


def process(fname):
    # Open a df
    data = pd.read_csv(fname)
    #data.info()

    # get input data array
    inp_data = get_input_dataset(data, 700, 1100)

    analysis_data = investigate_EMG_dataset(inp_data, 30)
    
    return analysis_data


def plot_features_comparison(df):

    
    sns.set_style("whitegrid")
    sns.set_context('talk')

    plt.figure("test plot")
    plt.title("aaa")

    #sns.set(rc={'figure.figsize':(12,10)})
    ax = sns.lineplot(data=df,legend="full", style="choice")
    #plt.scatter(x=x_[:,0], y=x_[:,1], marker="None", linewidths=1.5)
    #plt.scatter(x=y_[:,0], y=y_[:,1], marker="None", linewidths=1.5)
    ax.set_ylim(min(df.min().values)-5,105)
    ax.set_xlabel("Number of Synergies")
    ax.set_ylabel("tVAF (%)")
    return

def plot_n90_intersect(n90):
    threshold = 98

    x_ = np.array([[0,threshold],[n90,threshold]])
    y_ = np.array([[n90,0],[n90,threshold]])
    return x_, y_





def repeat_models(fname, repeats):

    data = pd.read_csv(fname)

    # get input data array
    inp_data = get_input_dataset(data, 700, 1100)

    # run first model
    analysis_data = investigate_EMG_dataset(inp_data, 30)

    #
    N90S = analysis_data['features_tVAF']
    N90S = np.array(N90S)
    N90S = np.transpose(N90S)

    repeat_data = pd.DataFrame(data=N90S, columns=["ITER_1"])

    for i in range(repeats-1):
        analysis_data = investigate_EMG_dataset(inp_data, 30)
        N90S = analysis_data['features_tVAF']
        N90S = np.array(N90S)
        N90S = np.transpose(N90S)
        repeat_data["ITER_{}".format(i+1)] = N90S

    plot_features_comparison(repeat_data)
    return


def compare_files(fnames):
    plt.figure()
    plt.title("Compare files")
    for fname in fnames:
        data = process(fname)
        plt.plot(data['features_tvaf'])
    plt.xlabel("Features")
    plt.ylabel("tVAF (%)")
    plt.show()#
    return

# Data path
fname = ['raw_emg_data_unprocessed\\index_finger_motion_raw.csv',
'raw_emg_data_unprocessed\\index_finger_motion_raw.csv',
'raw_emg_data_unprocessed\\little_finger_motion_raw.csv',
'raw_emg_data_unprocessed\\middle_finger_motion_raw.csv']

# process data
a1 = process(fname[0])
a2 = process(fname[1])
a3 = process(fname[2])
a4 = process(fname[3])

# print n90
print(a1['N90'])
print(a2['N90'])
print(a3['N90'])
print(a4['N90'])

# repeat model
repeat_models(fname[0], 5)

compare_files(fnames)
x_, y_ =  plot_n90_intersect(a1['N90'])

