import numpy as np
from matplotlib import pyplot as plt

from ConstrainedKMeans import ConstrainedKMeans as CKM

def run_test(n_points):
    ckm = CKM(n_clusters=10)

    # Generate random dataset
    # For visualization purposes, initialize 2d data
    x = np.random.random((n_points, 2))
    # Generate random labels
    init_labels = np.random.randint(0, 10, n_points)
    # Generate 0s with probability 0.2
    # these shall mask the "known" labels
    can_change = np.random.binomial(2, 0.7, n_points)

    labels = ckm.fit_predict(x, can_change, init_labels)

    plt.scatter(x[:, 0], x[:, 1], c=labels)
    plt.show()

if __name__ == '__main__':
    run_test(1000)
