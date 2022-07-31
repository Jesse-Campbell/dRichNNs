import uproot
import os
import numpy as np
import pandas as pd

directory = os.path.expanduser("~/Desktop/rich-detector-analysis/data/eventData")
MINIMUM_HITS = 10


def isEnough(data):
    counter = 0
    for line in data:
        for number in line:
            counter += number
    if counter > MINIMUM_HITS:
        return True
    return False



for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)) and filename.endswith('.root') and filename.startswith("single"):
        file = uproot.open(os.path.join(directory, filename))
        tree = file['events']
        branches = tree.arrays()
        name = filename.split('.')
        yList = branches['DRICHHits.position.y'].tolist()
        xList = branches['DRICHHits.position.x'].tolist()
        for event in range(len(yList)):
            data = np.zeros((500, 500), dtype=int)
            for photon in range(len(yList[event])):
                yUpdate = int(np.around(yList[event][photon], 0)) + 250
                xUpdate = int(np.around(xList[event][photon], 0)) - 1225
                if 0 <= yUpdate < 500 and 0 <= xUpdate < 500:
                    data[yUpdate][xUpdate] = int(1)
            if isEnough(data):
                DF = pd.DataFrame(data)
                DF.to_csv(os.path.expanduser("~/Desktop/rich-detector-analysis/data/eventData/mapData/"
                                             + name[1] + " " + name[0] + "/" + filename + " " + str(event) + ".csv"),
                          header=False, index=False)
