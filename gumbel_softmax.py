import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=5):
    y      = gumbel_softmax_sample(logits, temperature)
    
    shape  = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

# Alternative
# def gumbel_softmax2(logits, temperature=5, eps=1e-20):
#     shape  = logits.shape
    
#     # Gumbel-softmax sampling
#     U = torch.cuda.FloatTensor(shape).uniform_()
#     U = - Variable(torch.log(-torch.log(U + eps) + eps))
#     y = F.softmax((logits + U) / temperature, dim=-1)
    
#     # One-hot encoding
#     y_hard = torch.zeros_like(y)
#     y_hard = y_hard.view(-1, shape[-1])
#     y_hard.scatter_(1, y.argmax(dim=-1).view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
    
#     # Hard on forward; soft on backward
#     return (y_hard - y).detach() + y
