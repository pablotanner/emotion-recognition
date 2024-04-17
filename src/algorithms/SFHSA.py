import random
from itertools import combinations

import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mutual_info_score

MAX_ITER = 10

def cosine_similarity(v1, v2):
    """
    Function to calculate the cosine similarity between two vectors.

    :param v1: first vector
    :param v2: second vector
    :return: cosine similarity value
    """
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def cosine_similarity_range(v1, v2, epsilon):
    """
    Function to check if the cosine similarity between two vectors is within a certain range.

    :param v1: first vector
    :param v2: second vector
    :param epsilon: range value
    :return: boolean value
    """
    return -epsilon <= cosine_similarity(v1, v2) <= epsilon




def calculate_mRMR(features, target):
    """
    Calculate mRMR score for given features and target.
    """
    relevance = np.mean([mutual_info_score(target, features[:, i]) for i in range(features.shape[1])])
    redundancy = np.mean([mutual_info_score(features[:, i], features[:, j]) for i, j in combinations(range(features.shape[1]), 2)])
    return relevance - redundancy

def find_worst_subset(HM, features, target):
    """
    Find the worst feature subset based on mRMR score.
    """
    return np.argmin([calculate_mRMR(features[:, subset], target) for subset in HM])


def sfhs(features, y, hms=15, hmcr=0.8, par=0.5):
    """
    Function for the supervised filter harmony search (SFHS) algorithm (SFHSA),
    used for feature selection in facial emotion recognition.


    :param features: original features array
    :param y: labels array
    :param hms: harmony memory size
    :param hmcr: harmony memory considering rate
    :param par: pitch adjusting rate
    :return: reduced feature subset array
    """
    # Initialize harmony memory with random subsets of features
    hm = [np.random.choice(features.shape[1], size=features.shape[1] // 2, replace=False) for _ in range(hms)]

    t = 0
    while t < MAX_ITER:
        i = 0
        while i < hms:
            new_subset = list(hm[i])
            j = 0
            while j < len(new_subset):
                p1 = random.uniform(0, 1)
                if p1 < hmcr:
                    fj = new_subset[j]
                    p2 = random.uniform(0, 1)
                    if p2 < par:
                        epsilon = random.uniform(-1, 1)
                        candidates = [idx for idx in new_subset if
                                      cosine_similarity(features[:, fj], features[:, idx]) < epsilon]
                        if candidates:
                            fk = random.choice(candidates)
                            new_subset[j] = fk
                else:
                    fr = random.choice(range(features.shape[1]))
                    new_subset[j] = fr
                j += 1

            # Evaluate new subset
            new_score = calculate_mRMR(features[:, new_subset], y)
            worst_index = find_worst_subset(hm, features, y)
            worst_score = calculate_mRMR(features[:, hm[worst_index]], y)

            if new_score > worst_score:
                hm[worst_index] = new_subset

            i += 1
        t += 1

    # Return the best feature subset
    best_index = np.argmax([calculate_mRMR(features[:, subset], y) for subset in hm])
    return hm[best_index]