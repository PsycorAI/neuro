"""Thin wrapper around train_gpu.py so processes can be targeted with
`pkill -f train_5090_gpu.py` without affecting the 5080's training process."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from train_gpu import main

if __name__ == "__main__":
    main()
