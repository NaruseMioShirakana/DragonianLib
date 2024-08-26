import torch
import time
for i in range(10):
    beg = time.time()
    a = torch.ones(114,514,1919)
    b = torch.ones_like(a)
    c = a * b
    print (time.time() - beg)