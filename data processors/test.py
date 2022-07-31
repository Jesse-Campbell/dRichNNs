import pickle
pickle_in = open('positions/e-.50GeV.30.pickle', "rb")
loaded_list = pickle.load(pickle_in)
print(loaded_list)