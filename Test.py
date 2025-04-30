import torch


class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.register_buffer('weight', torch.randn(1, 1, dtype=torch.complex64))

    def forward(self, x):
        return x * self.weight
    


torch.onnx.export(Test(), torch.randn(1, 1, dtype=torch.complex64), opset_version=17, dynamic_axes={'input': {0: 'batch_size'}})