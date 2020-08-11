<h2>Constrained KMeans</h2>
Modified version of KMeans algorithm that takes into account
partial information about the data.

Given a partial list of known labels `init_labels`, Constrained KMeans
finds a cluster configuration that complies with `init_labels`.
`init_labels` is the same length as x.shape[0], which is why
a second array `can_change` masks out which labels should be
marked as known and which labels can change.
Formally, the output of the algorithm is an array `labels` such that
`np.all((labels[can_change == 0] == init_labels[can_change == 0]))` is `True`.

Can be installed via (requires Python>=3.7)
```bash
pip install ConstrainedKMeans
```

Example basic usage:
```python
import numpy as np
from matplotlib import pyplot as plt

from ConstrainedKMeans import ConstrainedKMeans as CKM

def run_test():
    # Generate random dataset consisting of 5 gaussian centers
    ppt = 100
    free = 1000
    x = []
    init_labels = []
    can_change = []
    var = np.eye(2)
    for i, mu in enumerate([[1, 1], [8, 8], [1, 8], [8, 1], [4, 4]]):
        x.append(np.random.multivariate_normal(mu, var, ppt))
        init_labels.append(np.full(ppt, i))
        can_change.append(np.zeros(ppt))

    # Add some free points (shown in yellow in the plot)
    x.append(np.random.multivariate_normal([4, 4], np.eye(2) * 8, free))
    init_labels.append(np.full(free, 5))
    can_change.append(np.zeros(free) + 1)

    x = np.vstack(x)
    init_labels = np.hstack(init_labels)
    can_change = np.hstack(can_change)

    fig, axes = plt.subplots(1, 2, sharey=True, squeeze=False)

    # Plot before
    axes[0, 0].scatter(x[:, 0], x[:, 1], c=init_labels, s=10)
    axes[0, 0].set_title("Before clustering (yellow points are free)")

    # Fit labels
    ckm = CKM(n_clusters=5)
    ckm.fit(x, can_change, init_labels)
    labels = ckm.get_labels()

    # Plot after
    axes[0, 1].scatter(x[:, 0], x[:, 1], c=labels, s=10)
    axes[0, 1].set_title("After clustering")
    plt.show()

if __name__ == '__main__':
    run_test()

```

<img src="https://github.com/ferrocactus/ConstrainedKMeans/blob/master/images/example.png" style="zoom:72%;" />
