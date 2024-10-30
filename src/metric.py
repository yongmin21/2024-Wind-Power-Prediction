import numpy as np

def NMAE(true, pred, capacity=20700):
    mae = np.mean(np.abs(true - pred))
    score = mae / capacity #np.mean(np.abs(true)) - original nmae
    return score * 100