import torch
import time
for i in range(10):
    a = torch.randn(1,768,10000)
    b = torch.randn(1,768,10000)
    beg = time.time()
    c = (a + a)
    print(time.time() - beg)