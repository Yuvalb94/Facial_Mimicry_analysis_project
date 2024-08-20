import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore


def loadJson(filename):
    data = None
    ticks = []
    titles = []
    with open(filename, "r") as f:
        data = json.load(f)
    for item in data:
        ticks.append(item["start"])
        ticks.append(item["end"])
        titles.append(item["key"])
    return ticks, titles


def loadJsonByKey(filename, key='all'):
    '''

    :param filename:
    :param key: keys can be from list ['all', 'reading', 'listening']
    :return:
    '''
    data = None
    ticks = []
    titles = []
    choices = []
    storyNumbers = []
    with open(filename, "r") as f:
        data = json.load(f)
    for item in data:
        if key=='listening':
            if ".ogg" in item["key"]:
                ticks.append(item["start"])
                ticks.append(item["end"])
                titles.append(item["key"])
                # choices.append(item[f"choice{subject}"])
        elif key=='reading':
            print("read JSON by reading not yet developed")
        else:
            print("read JSON all keys not yet developed")
    return ticks, titles

def downsample(data, dsType='mean', sampleSize=80):
    s = 0
    dSample = []
    # print("original array size:", len(data))
    if dsType == 'mean':
        while s < len(data) - (sampleSize - 2):
            dSample.append(np.mean(data[s:s+80]))
            s = s + sampleSize
    elif dsType == 'rms':
        print("rms downsampling not yet developed.")
    # print("downsampled array size:", len(dSample))
    return dSample

def normCorrelate(sig1, sig2):
    corr_val = 0
    if len(sig1) != len(sig2):
        print("datasets not the same length.")
        return False
    norm = (np.sqrt(np.sum(np.power(sig1, 2)))) * (np.sqrt(np.sum(np.power(sig2, 2))))
    corr_val = sum(sig1[x] * sig2[x] for x in range(len(sig1))) / norm
    # corr_val = sum(abs(sig1[x] * sig2[x]) for x in range(len(sig1))) / norm
    return corr_val


def getLeadersData(dsA, dsB, ppw, lagPixels, s_a=0, threshold = 0.4, RT = 0):
    '''
    this function cross-correlates signal A (reference) with signal B and calculates for each window in the reference signal the lag of the best-matched window in B.
    It then decides according to that which subject led the way and adds 1 to their score (lead_a/b).
    In addition, the CC value is normalized and
    :param dsA: downsampled data A
    :param dsB: downsampled data B
    :param ppw: pixels per window
    :param lagPixels: the step taken between each window in B
    :param s_a: start index for A (reference signal).
    :param threshold: the Cross-Correlation values threshold. any value below the threshold will not count as significant.
    :param RT: Response time in pixels
    :return:
    [0] - lead_a = number of times A was leading
    [1] - lead_b = number of times B was leading
    [2] - normCC_a = sum of CC values while A led, normalized
    [3] - normCC_b = sum of CC values while B led, normalized
    '''
    lead_a = 0
    lead_b = 0
    CC_a = []
    CC_b = []
    ties = 0
    corr_thrd = threshold #correlation value threshold
    # corr_allValues = []
    # thresholds = []
    # thrd_sampleSizes = []
    # s_a = 0  # ppw # start index
    while s_a < len(dsA) - (ppw - 1):
        corr_vec = []
        corr_val = 0
        sig1 = dsA[s_a:s_a + ppw]
        # e = s_a + ppw  # end index
        max_corr_val = 0
        max_corr_ind = 0
        # print("window A:", sig1)
        # while lag_ind[i] < ppw and s_b
        for i in range(-ppw, ppw + 1, lagPixels):
            s_b = s_a + i
            if s_b + ppw > len(dsB) or s_b < 0:
                continue
            sig2 = dsB[s_b:s_b + ppw]
            # print("\t window B:", sig2)
            corr_val = normCorrelate(sig1, sig2)
            if corr_val < corr_thrd:
                continue
            corr_vec.append(corr_val)
            if corr_val > max_corr_val:
                max_corr_val = corr_val
                max_corr_ind = i
            # corr_allValues.append(corr_val) # accumulate CC values to determine threshold later.
        if max_corr_ind < (-1)*RT_p:
            lead_b += 1
            CC_b.append(max_corr_val)
        elif max_corr_ind > RT_p:
            lead_a += 1
            CC_a.append(max_corr_val)
        else:
            ties += 1
        s_a += ppw


        # #determine significant threshold value for current window in signal A:
        # CC_values = np.array(corr_vec)
        # alpha = 0.05
        # # Calculate the threshold
        # thresholds.append(np.percentile(CC_values, (1 - alpha) * 100))
        # thrd_sampleSizes.append(len(CC_values))

    #calculate standartized Leader Scores and Cross-Correlation Scores:
    numWindows = lead_a + lead_b + ties
    # lead_a = lead_a / numWindows
    # lead_b = lead_b / numWindows
    # normCC_a = sum(CC_a) / numWindows
    # normCC_b = sum(CC_b) / numWindows

    ##print checks:
    # print(f"number_windows = {numWindows}, number of max index vals = {len(max_index_all)}")
    # print(f"lead_a:{lead_a}, lead_b:{lead_b}, ties:{ties}\nCC_a:{normCC_a}, CC_b:{normCC_b}")

    ##return options:
    # return [lead_a, lead_b, normCC_a, normCC_b, thresholds, thrd_sampleSizes, corr_allValues]
    # return [lead_a, lead_b, normCC_a, normCC_b]
    return [lead_a, lead_b, sum(CC_a), sum(CC_b), numWindows]



if __name__=='__main__':
    # origin = r'C:\Users\yuval\OneDrive\Desktop' #'C:\Yuval'
    # os.chdir(origin)
    A = 0
    B = 1
    origFreq = 4000
    dsBy = 10
    dsFreq = origFreq / dsBy  # = pixels per second
    lagPixels = 1  # number of pixels gap from first pixel of window X to first pixel of window (X+1) in signal B
    windowSizeInMs = 1000
    ppw = int(dsFreq * windowSizeInMs / 1000)  # window size in pixels
    corr_thrd = 0.6 #threshold values for significant correlation between two signals
    RT_ms = 125 # Response time for Smiling in milliseconds
    RT_p = dsFreq * (RT_ms / windowSizeInMs) # Response time in pixels
    print("window size in pixels:", ppw)
    print("Response time in pixels:", RT_p)

    trials = []
    data = [[],[]]
    ticks = []
    folder = 'data'
    path_to_data = r'Z:\Backup\Liron\ForYuval\data'
    experiments = os.listdir(path_to_data)
    print("experiments:\n", experiments)
    stateToAnalyze = "smile"
    svmType = "svr"  # could be either "svc" for classification ( {0,1} ) or svr for regresssion [-1,1]

    res_filename = r'female_particiapnts_analyzed_withChoice_16_4.csv' #name of csv with questionnaire results

    exp_names = [] # these 3 lists will be used to add the final data to the results dataframe
    lead_score = []
    normCC_score = []


    threshold_test_vals = []
    num_samples = []
    all_CC_Vals = []
    for exp in experiments:
        path = fr"{path_to_data}\{exp}\{stateToAnalyze}\{svmType}"
        # A/B.List is JSON file that has {objects}. in each object there are keys for events and their values.
        # Load A/B.list writes the keys in titles, and the start/end times from all values to ticksA/B.
        # in the end we get ticksA/B to be a list of event times (start/end)
        ticksA, trials = loadJsonByKey(fr"{path}\A.list", key='listening')
        ticks.append(ticksA)
        ticksB, trials = loadJsonByKey(fr"{path}\B.list", key='listening')
        ticks.append(ticksB)

        # print(f"trial: \t list A: \t list B:")
        # j = 0
        # for i in range(len(ticks[0])):
        #     print(f"{trials[j]} \t {ticks[0][i]} \t {ticks[1][i]}")
        #     if i%2==1: # ticksA/B is twice as long as trials, duplicat each trial name to adjust
        #         j += 1

        # data[0] = np.loadtxt(f"{path}\A.csv", delimiter=",")
        # data[1] = np.loadtxt(f"{path}\B.csv", delimiter=",")
        data[0] = zscore(np.loadtxt(f"{path}\A.csv", delimiter=","))
        data[1] = zscore(np.loadtxt(f"{path}\B.csv", delimiter=","))
        exp_lead_a = [[], []] # [0]-lead_a, [1]- CC_A
        exp_lead_b = [[], []]
        threshold_test_vals = []
        num_samples = []
        num_windows = 0
        for i in range(0, len(trials), 2):
            dsA = downsample(data[0][ticks[0][i]:ticks[0][i+1]], sampleSize=dsBy)
            dsB = downsample(data[1][ticks[1][i]:ticks[1][i+1]], sampleSize=dsBy)

            res = getLeadersData(dsA, dsB, ppw, lagPixels, s_a=0, threshold = corr_thrd, RT = RT_p)
            exp_lead_a[0].append(res[0]) #lead_a for current trial
            exp_lead_b[0].append(res[1]) #lead_b for current trial
            exp_lead_a[1].append(res[2]) #sum of CC values where a was leading in current trial
            exp_lead_b[1].append(res[3]) # sum of CC values for b
            num_windows = num_windows + res[4]
            # threshold_test_vals = threshold_test_vals + res[4]
            # num_samples = num_samples + res[5]
            # all_CC_Vals = all_CC_Vals + res[6]

            #figure of all correlation values in this trial
            # plt.subplot(1, i + 1, i + 1)
            # lines = 12
            # for i in range(lines):
                # plt.plot(np.arange(0, len(res[4][i])), res[4][i])
            # print("corr_data size:", len(res[4]))
            # for i in range(len(res[4][i]))
            # plt.plot()
            # plt.show()

        # print(f"for session {exp}, through four trials, lead score A is: {sum(exp_lead_a[0])}, and lead score for B is: {sum(exp_lead_b[0])}.")
        # print(f"for session {exp}, through four trials, CC score A is: {sum(exp_lead_a[1])}, and CC score B is: {sum(exp_lead_b[1])}")
        # lead_a_list.append(exp_lead_a)
        # lead_b_list.append(exp_lead_b)
        lead_score.append(sum(exp_lead_a[0]) / num_windows)
        lead_score.append(sum(exp_lead_b[0]) / num_windows)
        # normCC_a_list.append(np.mean(exp_lead_a))
        # normCC_b_list.append(np.mean(exp_lead_b))
        normCC_score.append(sum(exp_lead_a[1]) / num_windows)
        normCC_score.append(sum(exp_lead_b[1]) / num_windows)
        exp_names.append(f"{exp}_A")
        exp_names.append(f"{exp}_B")

        print(f"Finished calculating {num_windows} correlations in experiment {exp}")

    results = pd.DataFrame()
    results["session"] = exp_names
    results["lead_score"] = lead_score
    results["normCC_score"] = normCC_score
    print(results)
    # print(f"mean threshold over {len(threshold_test_vals)} tests: {np.mean(threshold_test_vals)}")
    # print("mean sample size:", np.mean(num_samples))
    results.to_csv('CC_results_RT_thrd.csv', index=False)
    # results = pd.read_csv(res_filename)
    # results["lead_score"] = [0]*len(results["session"])
    # for i, exp in enumerate(exp_names):
    #     results["session"==f"{exp}_A"]
    #

