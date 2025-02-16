import torch

class SquaredExponential:
    def __init__(self,l,amplitude):
        self.l = l
        self.amplitude = amplitude

    def __call__(self,x1,x2,return_amplitude_root=True):
        if x1.dim()==1:
            x1 = x1.unsqueeze(-1)
        if x2.dim()==1:
            x2 = x2.unsqueeze(-1)
        D = torch.cdist(x1,x2)
        if return_amplitude_root:
            p = 0.5
        else:
            p = 1.0
        return self.amplitude**p*torch.exp(-D**2/(2*self.l**2))
