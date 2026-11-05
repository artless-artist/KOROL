import pickle, numpy as np
demos = pickle.load(open("./Door/Data/Testing.pickle", "rb"))
n = sum(len(d["actions"]) for d in demos)          # 6729
pickle.dump([{i: np.zeros(8)} for i in range(n)],
            open("./Door/feature_door_demo_full.pickle", "wb"))
print("dummy feature:", n)