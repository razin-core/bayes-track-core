# -*- coding: utf-8 -*-
"""
Neutral mathematical and runtime configuration for bayes-track-core.
Enforces single-thread BLAS/OMP quotas for deterministic low-latency ARM64 VPS execution.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Echo suppression & novelty decay constants for multi-source evidence fusion
ECHO_SUPPRESSION_WINDOW_SEC = 180
ECHO_NOVELTY_DISCOUNT = 0.20
MAHALANOBIS_GATE_CHI2_99 = 9.21
DAILY_WEIGHT_DECAY_LAMBDA = 0.98
