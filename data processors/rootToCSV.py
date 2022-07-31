import uproot
import os
import numpy as np
import csv

directory = os.path.expanduser("~/Desktop/rich-detector-analysis/data/eventData")
f = open(os.path.expanduser("~/Desktop/rich-detector-analysis/data/eventData/attributes.csv"), 'w')
writer = csv.writer(f)
fields = ['particle'] + ['pixel ' + str(i) for i in range(1, 250001)]
writer.writerow(fields)

for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)) and filename.endswith('.root'):
        file = uproot.open(os.path.join(directory, filename))
        tree = file['events']
        branches = tree.arrays()
        name = filename.split('.')
        data = np.zeros(250000)
        yList = branches['DRICHHits.position.y'].tolist()
        xList = branches['DRICHHits.position.x'].tolist()
        for event in range(len(yList)):
            for photon in range(len(yList[event])):
                yUpdate = int(np.around(yList[i][j]), 0)) + 250
                xUpdate = int(np.around(xList[i][j]), 0)) - 1225
                if 0 <= yUpdate < 500 and 0 <= xUpdate < 500:
                    data[500 * yUpdate + xUpdate + 1] = 1
            newRow = np.concatenate([[name[1]], data])
            writer.writerow(newRow)
