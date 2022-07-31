import uproot
import os
import pickle
CATEGORIES = ["e- single", "kaon+ single", "pi+ single", "proton single"]
directory = os.path.expanduser("~/Desktop/rich-detector-analysis/data/eventData")
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)) and filename.endswith('.root') and \
            filename.startswith('single'):
        file = uproot.open(os.path.join(directory, filename))
        tree = file['events']
        branches = tree.arrays()
        name = filename.split('.')
        y_list = branches['DRICHHits.position.y'].tolist()
        x_list = branches['DRICHHits.position.x'].tolist()
        z_list = branches['DRICHHits.position.z'].tolist()
        for i in range(len(y_list)):
            l = list(map(list, zip(x_list[i], y_list[i])))
            pickle_out = open(os.path.join(directory, 'featureData/' + name[1] + ' ' + name[0] + '/'
                                           + name[1] + '.' + name[2] + '.' + str(i) + '.pickle'), 'wb')
            pickle.dump(l, pickle_out)
            pickle_out.close()

