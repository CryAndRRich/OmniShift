import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class BinarizeWeight(autograd.Function):
    @staticmethod
    def forward(ctx, w):
        out = w.sign()
        return torch.where(out == 0, torch.ones_like(out), out)

    @staticmethod
    def backward(ctx, grad):
        return grad

class BinarizeInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = x.sign()
        return torch.where(out == 0, torch.ones_like(out), out)

    @staticmethod
    def backward(ctx, grad):
        return grad

class XNORConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        alpha = self.weight.abs().mean(dim=[1, 2, 3], keepdim=True)
        W_b = BinarizeWeight.apply(self.weight)

        K = x.abs().mean(dim=1, keepdim=True)
        H = BinarizeInput.apply(x)

        out = F.conv2d(H, W_b, None, self.stride, self.padding)
        out = out * alpha.view(1, -1, 1, 1)

        kH, kW = self.kernel_size
        K_pooled = F.avg_pool2d(K, kernel_size=(kH, kW),
                                 stride=self.stride, padding=self.padding)
        out = out * K_pooled

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, binary_W=True, binary_A=True")