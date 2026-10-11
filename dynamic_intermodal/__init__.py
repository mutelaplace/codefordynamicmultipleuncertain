"""Dynamic optimisation of intermodal transport plans under multi-source uncertainty."""

import os

# The linear algebra here is on small matrices; multi-threaded OpenBLAS is orders of magnitude
# slower for them on some machines. Must be set before numpy is imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
