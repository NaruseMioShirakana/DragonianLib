import torch
import time

for i in range(20):
    begin = time.time()
    tensora = torch.ones(11451400*8)
    print(time.time() - begin)
    print(tensora)


print(
    torch.sqrt(
        torch.nn.functional.conv1d(
            torch.FloatTensor([[[1., 2., 3., 4., 5.]]]) ** 2,
            torch.ones(1, 1, 2)
        ) + 1e-8
    )
)
print(
    torch.nn.functional.conv1d(
        torch.FloatTensor([[[1., 2., 3., 4., 5.]]]),
        torch.FloatTensor([[[0.3, 0.7]]])
    )
)