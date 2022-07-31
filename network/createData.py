import numpy as np
import os
import pickle
import uproot

particles = ['e-', 'kaon+', 'pi+', 'proton']  # labels
folders = ['e- single', 'kaon+ single', 'pi+ single', 'proton single']


def load_data():
    output = []
    directory = os.path.expanduser("~/Desktop/rich-detector-analysis/data/eventData")
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and filename.endswith('.root') and \
                filename.startswith('single'):
            file = uproot.open(os.path.join(directory, filename))
            tree = file['events']
            branches = tree.arrays()
            y_list = branches['DRICHHits.position.y'].tolist()
            x_list = branches['DRICHHits.position.x'].tolist()
            z_list = branches['DRICHHits.position.z'].tolist()
            path_length = branches['DRICHHits.pathLength'].tolist()
            time = branches['DRICHHits.time'].tolist()
            for i in range(len(y_list)):
                output.append([filename + str(i), x_list[i], y_list[i], z_list[i], path_length[i], time[i]])
    return output


def create_feature_data():
    directory = os.path.expanduser("~/Desktop/rich-detector-analysis/data/graphData")
    hits = load_data()
    for i in range(len(hits)):
        name, x_list, y_list, z_list, path_length, time = hits[i][0], normalize(hits[i][1]), normalize(hits[i][2]), \
                                                          normalize(hits[i][3]), normalize(hits[i][4]), \
                                                          normalize(hits[i][5])
        if x_list == None:
            continue
        n = len(x_list)  # number of nodes
        x = np.zeros((n, 5), dtype=float)  # node attributes
        for i in range(len(x_list)):
            x[i] = (x_list[i], y_list[i], z_list[i], path_length[i], time[i])
        y = np.zeros(4)
        index = -1
        split_name = name.split('.')
        for j in range(len(particles)):
            if split_name[1] == particles[j]:
                y[j] = 1
                index = j
        np.savez(os.path.join(directory, folders[index], name), x=x, y=y)


def normalize(hits):
    if len(hits) == 0:
        return
    output = []
    min = np.amin(hits)
    max = np.amax(hits)
    for i in range(len(hits)):
        output.append((hits[i] - min) / (max - min + 1e-3))
    return output


create_feature_data()
