import glob

import numpy as np

annotations = glob.glob("../../data/annotations/*_exp.npy")

images = glob.glob("../../data/images/*.jpg")


def get_expression_count():
    expressions = {}

    for annotation in annotations:
        exp = np.load(annotation, allow_pickle=True).flatten()[0]
        if exp not in expressions:
            expressions[exp] = 0
        expressions[exp] += 1

    return expressions


if __name__ == "__main__":
    count = get_expression_count()