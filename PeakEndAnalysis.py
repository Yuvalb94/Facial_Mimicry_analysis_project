import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import os
import fnmatch
from other_functions import getPairwiseZip
from other_functions import window_rms_single
from other_functions import spliceListOfStringsByChar
from other_functions import writeDataFrameToCsv

class storyData:
    def __init__(self) -> None:
        self.storyNumber = 0
        # self.experiment  = ""
        self.baseline = 0
        self.peak = 0
        self.end = 0
        self.post = 0
        self.diff = 0
        self.isChoice = bool
        self.isReading = bool

    def toList(self):
        attributes = []
        for attribute in self.__dict__.items():
            attributes.append(attribute)
        keys, values = zip(*attributes)
        values = list(values)
        return values

def loadStoriesFromJson(filename):
    data = None
    if "A.list" in filename:
        subject = 'A'
    elif "B.list" in filename:
        subject = 'B'
    ticks = []
    titles = []
    choices = []
    storyNumbers = [] 
    with open(filename, "r") as f:
        data = json.load(f)
    for item in data:
        if ".ogg" in item["key"] or ".png" in item["key"]:
            ticks.append(item["start"])
            ticks.append(item["end"])
            titles.append(item["key"])
            choices.append(item[f"choice{subject}"])
            if item["story1"] not in storyNumbers:
                storyNumbers.append(item["story1"])
            else:
                storyNumbers.append(item["story2"])
            storyOrder = spliceListOfStringsByChar(storyNumbers, '.')
 
    return ticks, titles, choices, storyOrder
       


def getStoryPeakEndValues(path_to_main_folder, sampling_freq=4000):
    storiesData = dict()
    experiments = fnmatch.filter(os.listdir(), '*2022_*')
    for exp in experiments:
        path = fr"{exp}\smile\svr"
        path = os.path.join(path_to_main_folder, exp, "smile", "svr")
        print(f"begining analysis from experiment {exp} , \t")
        for subject in ['A', 'B']:
            storyTicks, titles, choices, storyOrder = loadStoriesFromJson(fr"{path}\{subject}.list")
            ticks = (getPairwiseZip(storyTicks))
            # storyOrders = [spliceListOfStringsByChar(x, '.') for x in storyOrders]
            data = window_rms_single(np.loadtxt(fr"{path}\{subject}.csv", delimiter=","))
            previous_story = ""
            for i, storyTimes in enumerate(ticks):
                start_time = storyTimes[0]
                end_time = storyTimes[1]
                current_story = f"{exp}_{storyOrder[i]}.{subject}"
                storiesData[current_story] = storyData()
                storiesData[current_story].storyNumber = storyOrder[i]
                storiesData[current_story].baseline = np.mean(data[start_time:end_time])
                storiesData[current_story].peak = np.max(data[start_time:end_time])

                end_period = (end_time - sampling_freq, end_time) # optimal margin is 1 sec = 4000 frames 
                storiesData[current_story].end = np.mean(data[int(end_period[0]):int(end_period[1])])

                post_period = (end_time, end_time + 2*sampling_freq) # 4 seconds * (sampling_freq = 4000) frames per second
                storiesData[current_story].post = np.mean(data[int(post_period[0]):int(post_period[1])])
                                
                # for every second iteration = story , fill in the diffs of both stories in this bulk (1stPeak - 2ndPeak)
                # since the data for the second story is not as approachable in the previous, odd iteration
                if i%2 == 1:
                    storiesData[previous_story].diff = storiesData[previous_story].peak - storiesData[current_story].peak
                    storiesData[current_story].diff = storiesData[current_story].peak - storiesData[previous_story].peak
                              

                if (i%2 == 0 and choices[i] == 1) or (i%2 == 1 and choices[i] == 2):
                    storiesData[current_story].isChoice = True
                elif (i%2 == 0 and choices[i] == 2) or (i%2 == 1 and choices[i] == 1):
                    storiesData[current_story].isChoice = False
                    
                if (subject == 'A') and (i in [0, 1, 4, 5]) or ((subject == 'B') and (i in [2, 3, 6, 7])):
                    storiesData[current_story].isReading = True
                else:
                    storiesData[current_story].isReading = False

                
                previous_story = current_story # this keeps the key of this iteration for the calculation of diff in the even iterations.
            
          
                
        print(f"finished analysis from all subjects in experiment {exp}. ")
    total_data = []
    for story in storiesData:
        print(storiesData[story].toList())
        total_data.append(storiesData[story].toList())
    stories_df = pd.DataFrame(total_data)
    stories_df.columns = ['storyNumber','baseline', 'peak', 'end', 'post', 'diff', 'isChoice', 'isReading']
    stories_df.insert(0, 'storyID', list(storiesData.keys()))
  
    return stories_df



if __name__=='__main__':
    path_to_data = r"C:\Users\yuval\OneDrive\Desktop\לימודים\שנה ג\סדנת מחקר\משימה שנייה\data_for_analysis"
    os.chdir(path_to_data)
    sampling_freq = 4000
    # stories = getStoryPeakEndValues(path_to_data, sampling_freq)
    path = r'C:\Users\yuval\OneDrive\Desktop\לימודים\שנה ג\סדנת מחקר\משימה שנייה\data_for_analysis\04122022_1545\smile\svr'
    # writeDataFrameToCsv(stories, filename = 'peakEndAnalysis')
    data = np.loadtxt(fr"{path}\{'A'}.csv", delimiter=",")
    data2 = window_rms_single(data)
    # print(data[10000:10100])
    # print(np.mean(data), np.mean(data2))
    # print(data2[10000:10100])
    # print(stories.to_string())
    for i in range(10000,10100):
        print(round(data[i],3), round(data2[i],3))
    