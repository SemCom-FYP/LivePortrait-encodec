# coding: utf-8
"""
Neural web conferencing over aiortc.

Instead of shipping pixels, every participant transmits:

  * video  — a 140-byte LivePortrait motion vector per frame (implicit keypoint
             deltas), re-rendered locally on each receiver against a source
             portrait that was exchanged once at call setup.
  * audio  — EnCodec residual-VQ codes, bit-packed at 10 bits/entry.

Both ride custom SCTP DataChannels, so no standard RTP codec is involved.
"""

import os as _os

# onnxruntime (face detection / landmarks) and torch each ship their own Intel
# OpenMP runtime; on Windows the second one to load aborts the process. This has
# to be set before either import, which is why it lives in the package __init__.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
