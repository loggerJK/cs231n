import os
from sys import path

cifar10_dir = os.path.abspath("./datasets/cifar-10-batches-py")

with open(cifar10_dir, "rb") as f:
    pr = f.readlines
    print(pr)
