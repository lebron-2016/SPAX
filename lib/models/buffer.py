import torch.nn as nn
import torch

class Buffer(nn.Module):
     def __init__(self,):
        
        self.re1 = torch.tensor(0.)
        self.re2 = torch.tensor(10.)

        super(Buffer, self).__init__()
     