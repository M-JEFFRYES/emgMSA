from c3d_extract_data import c3dExtract
from gpscalculator import GPSData
from msanalysis import MuscleSynergyAnalysis

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

#######################################
############ functions

def batch_extract_c3d_data(paths, outputdir):
    a=0
    for i, path in enumerate(paths):
        try:
            data = c3dExtract(path, "SUB_{}".format(i))

            try:
                data.export_MSA_data(outputdir, cycle='Left')
                
            except:
                a+=1
                print("Unable to save EMG data from {}".format(path))
            
            try:
                data.export_GPS_data(outputdir)
            except:
                a+=1
                print("Unable to save GPS data from {}".format(path))
        except:
            a+=1
            print("Cant open {}". format(path))
        
        print("file {} of {} checked".format(i+1, len(paths)))
    print("Couldn't check {} files".format(a))
    return

def slice_MSA_data(msa_dict, chns):

    data = []

    for key in chns:
        data.append(msa_dict[key])
    data = np.array(data)
    return data

def collect_MSA_data(msa_dict):
    chns_b = ['LRF','LMH','LTA','LMG','RRF','RMH','RTA','RMG']
    chns_l = ['LRF','LMH','LTA','LMG']
    chns_r = ['RRF','RMH','RTA','RMG']

    data_b = slice_MSA_data(msa_dict, chns_b)
    
    data_l = slice_MSA_data(msa_dict, chns_l)
    data_r = slice_MSA_data(msa_dict, chns_r)

    return data_b, data_l, data_r

def open_GPS_MSA(gps_path, msa_path):

    name = msa_path.split("_")[0]
    
    GPS = GPSData(gps_path, "C:\\Development_projects\\EXAMPLE_FILES\\GPS\\reference")

    gps = GPS.GPS_SCORE['GPS']
    gps_l = GPS.GPS_SCORE['GPS Left']
    gps_r = GPS.GPS_SCORE['GPS Right']

    with open(msa_path, 'rb') as f:
        msa_dict = json.load(f)

    emg_b, emg_l, emg_r = collect_MSA_data(msa_dict)

    msa_b =  MuscleSynergyAnalysis(abs(emg_b))#, plot_n90=True, plot_WH=True, plot_relat=True)
    print("{} bilateral calculated".format(name))
    msa_l =  MuscleSynergyAnalysis(abs(emg_l))#, plot_n90=True, plot_WH=True, plot_relat=True)
    print("{} left calculated".format(name))
    msa_r =  MuscleSynergyAnalysis(abs(emg_r))#, plot_n90=True, plot_WH=True, plot_relat=True)
    print("{} right calculated".format(name))
    data = [name, gps, gps_l, gps_r, msa_b.N90, msa_l.N90, msa_r.N90]
    return data

def batch_MSA_GPS(msa_paths, gps_paths):

    results = []
    for i in range(len(gps_paths)):

        try:
            results.append(open_GPS_MSA(gps_paths[i], msa_paths[i]))
        except:
            print(gps_paths[i], msa_paths[i])

        print("{} of {}".format(i+1, len(gps_paths)))

    return results


def plotting_results(results):

    cols= ["Subject", "GPS", "GPS Left", "GPS Right", "N90", "N90 L", "N90 R"]

    ress = np.array(results)

    df = pd.DataFrame(data=results, columns=cols)

    plt.title("Bilaterally assessed")
    ax1 = sns.scatterplot(x="GPS", y="N90", data=df[(df['GPS']<30) & (df['N90']>1.0001) ])
    plt.show()

    plt.title("Unilaterally assessed")
    ax2 = sns.scatterplot(x="GPS Left", y="N90 L", data=df[(df['GPS Left']<30) & (df['N90 L']>1.0001) ])
    ax = sns.scatterplot(x="GPS Right", y="N90 R", data=df[(df['GPS Right']<30) & (df['N90 R']>1.0001) ])
    plt.show()

    crit = 2.5

    plt.title("One side significantly worse (GPS+{}) Right side affected".format(crit))
    ax2 = sns.scatterplot(x="GPS Left", y="N90 L", data=df[(df['GPS Left']<30) & (df['N90 L']>1.0001) & (df['GPS Right']>df['GPS Left']+crit)])
    axx = sns.scatterplot(x="GPS Right", y="N90 R", data=df[(df['GPS Left']<30) & (df['N90 L']>1.0001) & (df['GPS Right']>df['GPS Left']+crit)])
    plt.show()

    plt.title("One side significantly worse (GPS+{}), Left side affected".format(crit))
    ax2 = sns.scatterplot(x="GPS Left", y="N90 L", data=df[(df['GPS Left']<30) & (df['N90 L']>1.0001) & (df['GPS Left']>df['GPS Right']+crit)])
    axx = sns.scatterplot(x="GPS Right", y="N90 R", data=df[(df['GPS Left']<30) & (df['N90 L']>1.0001) & (df['GPS Left']>df['GPS Right']+crit)])
    plt.show()

    plt.title("Comparison of N90 Sides")
    ax = sns.scatterplot(x="N90 L", y="N90 R", data=df)#df[(df['GPS Left']<30) & (df['N90 L']>1.0001) & (df['GPS Left']>df['GPS Right']+crit)])
    plt.show()

    plt.title("Comparison of GPS Sides")
    ax = sns.scatterplot(x="GPS Left", y="GPS Right", data=df[(df['GPS Left']<30) & (df['GPS Right']<30)])
    plt.show()
    return


###########################################

#
c3ddir = "F:\\Mike\\Initial_Analysis\\C3D_files"
extractdir = "F:\\Mike\\Initial_Analysis\\extracted_data"

# collect paths
paths = []
for pth in os.listdir(c3ddir):
    paths.append("{}\\{}".format(c3ddir, pth))

# Batch extract
batch_extract_c3d_data(paths, extractdir)

# collect gps and msa paths
gps_paths = []
msa_paths = []

for path in os.listdir(extractdir):
    if "GPS" in path:
        gps_paths.append("{}\\{}".format(extractdir, path))
    if "MSA" in path:
        msa_paths.append("{}\\{}".format(extractdir, path))

## Channel lists
chns = ['LRF','LMH','LTA','LMG','RRF','RMH','RTA','RMG']
chns_l = ['LRF','LMH','LTA','LMG']
chns_r = ['RRF','RMH','RTA','RMG']


#a = open_GPS_MSA(gps_paths[1], msa_paths[1])
# batch GPS and MSA
results = batch_MSA_GPS(msa_paths, gps_paths)

# Plot results
plotting_results(results)
######################################
