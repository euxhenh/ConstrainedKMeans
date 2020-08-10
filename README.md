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

```

<img src="https://github.com/ferrocactus/ConstrainedKMeans/blob/master/images/example.png" style="zoom:72%;" />
