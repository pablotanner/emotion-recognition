import glob

import numpy as np

annotations = glob.glob("../data/annotations/*_exp.npy")


expressions = {}

for annotation in annotations:
    exp = np.load(annotation, allow_pickle=True).flatten()[0]
    if exp not in expressions:
        expressions[exp] = 0
    expressions[exp] += 1




