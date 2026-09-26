"""`python -m exp ...`, for anywhere the console script is not on PATH.

A Kaggle kernel and a fresh container are both such places, and a run that
cannot start is a worse failure than one that is slow.
"""
from . import main

raise SystemExit(main())
