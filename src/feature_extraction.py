import numpy as np
from tqdm import tqdm

def extract_features(data):

    # remove label and id columns
    X = data.drop(["target", "id"], axis=1)

    # label
    y = data["target"]

    features = []

    for row in tqdm(X.values, desc="Extracting features"):

        mean = np.mean(row)
        std = np.std(row)
        maximum = np.max(row)
        minimum = np.min(row)

        features.append([mean, std, maximum, minimum])

    features = np.array(features)

    return features, y