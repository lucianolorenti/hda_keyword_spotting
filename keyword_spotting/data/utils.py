import numpy as np
from pathlib import Path

def distribution_labels(labels, dataset):
    distri = zip(labels,np.zeros(len(labels)))
    distri = dict(distri)
    for x in dataset:
        folder = Path(x).parent.stem
        distri[folder]+=1
    return distri