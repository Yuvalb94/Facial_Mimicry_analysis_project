import matplotlib.pyplot as plt
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.stats import zscore
import os

def ArrayReverse(arr):
    return arr[::-1]

def TakeOneChannel(sig, channels):
    if channels == 1:
        return sig.reshape(-1,1)
    return sig[0][:,0].reshape(-1,1)

def envelope(sig,a=0,b=0):
    return np.abs(signal.hilbert(sig))

def moving_average(values, n) :
    return np.average(sliding_window_view(values, window_shape = n), axis=(1,1))


def ComputeCrossCorrToEnvelope(sig1, sig2):
    sig1 = zscore(sig1)
    sig2 = zscore(sig2)
    
    corr = signal.correlate(sig1,sig2,mode ="full", method="fft");
    return np.max(corr)/len(sig1)

def plot(y_val, X, totalSize, index, title, color, ticks): #, maximun_val):
    plot = plt.subplot2grid(totalSize, index, rowspan=1)
    plot.plot(X,y_val, color = color)
    plot.set_title(title)
    for tick in ticks:
        plot.axvline(tick, linestyle='--')
    plt.gca().axes.get_xaxis().set_visible(False)


def plotChunkData(X, ticks):
    y_axis = X
    total_size = (2,1)
    #maximun_val = np.max(y_axis) # much better
    for i in range(0,2):
        plot(y_axis[i], range(0,len(y_axis[i])) , total_size, (i, 0), f"participant {i}", 'r', ticks[i])

    plt.show()

def window_rms_single(a, window_size = int(0.25*4000)):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))


def getPairwiseZip(arr):
    return list(zip(arr[0::2], arr[1::2]))

def loadJson(filename):
    data = None
    ticks = []
    titles = []
    with open(filename, "r") as f:
        data = json.load(f)
    for item in data:
        ticks.append(item["start"]/80)
        ticks.append(item["end"]/80)
        titles.append(item["key"])
    return ticks, titles

def downsample(data, dsType='mean', sampleSize=80):
    s = 0
    dSample = []
    print("original array size:", len(data), "\n")
    if dsType == 'mean':
        while s < len(data) - (sampleSize - 2):
            dSample.append(np.mean(data[s:s+80]))
            s = s + sampleSize
    print("downsampled array size:", len(dSample))
    return dSample


origin = r'C:\Users\yuval\OneDrive\Desktop\לימודים\שנה ג\סדנת מחקר\משימה שנייה\data_for_analysis'
os.chdir(origin)
A=0
B=1
freq = 4000
titles = []
data = []
ticks = []
folder = "04122022_1545"
stateToAnalyze = "smile"
svmType = "svr" # could be either "svc" for classification ( {0,1} ) or svr for regresssion [-1,1]
path = fr"{folder}\{stateToAnalyze}\{svmType}"
# A/B.List is JSON file that has {objects}. in each object there are keys for events and their values.
# Load A/B.list writes the keys in titles, and the start/end times from all values to ticksA/B.
# in the end we get ticksA/B to be a list of event times (start/end)
ticksA, titles = loadJson(fr"{path}\A.list")
ticks.append(ticksA)
ticksB, titles = loadJson(f"{path}\B.list")
ticks.append(ticksB)

# data reads the smiles data with values (for svc 1 or 0, and for rvc value from -1 to 1)  that indicate the amount of 'smile' in each sample.
# we get 2 numpy arrays of data in data. one for A and one for B. each one holds large number of values
# then, window_rms_single is used on both A and B values and applies convolution to their rms. results in data with 1000 values less because this is the window defined in the function.
data.append(downsample(np.loadtxt(f"{path}\A.csv", delimiter=",")))
data.append(downsample(np.loadtxt(f"{path}\B.csv", delimiter=",")))
# print(f"tick len: {[len(x) for x in ticks]}")

# data = [window_rms_single(x) for x in data]

# ticksPairWise holds 2 lists, one for each subject(A/B).
# each list containing pairs of subsequent values (ticks) - each pair is start and end time for each event.
# total of 25 pairs X 2 sets.
ticksPairWise = [getPairwiseZip(x) for x in ticks]

# for i in range(0, len(ticksPairWise[0])):
#     A_start = ticksPairWise[A][i][0]
#     A_end = ticksPairWise[A][i][1]
#     B_start = ticksPairWise[B][i][0]
#     B_end = ticksPairWise[B][i][1]
#     corr = ComputeCrossCorrToEnvelope(data[A][A_start:A_end],data[B][B_start:B_end])
#     print(f"{titles[i]} : {corr}")

# computerCrossCorrelate checks the correlation between subjects A and B for every event.


plotChunkData(data, ticks)
