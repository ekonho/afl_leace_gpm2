"""Algorithm registry for federated learning methods.

Each algorithm exports a function `{alg}_alg(args, n_rounds, nets, global_model,
party_list_rounds, net_dataidx_map, train_local_dls, test_dl, traindata_cls_counts,
device, logger)` that runs the full federated learning loop.

To add a new algorithm:
    1. Create algorithms/your_alg.py with a `your_alg_alg()` function
    2. Import and expose it here
"""

from algorithms.dualortho import dualortho_alg
from algorithms.client import local_train_net

__all__ = ["dualortho_alg", "local_train_net"]
