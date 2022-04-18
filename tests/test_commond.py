import torch
from torch.nn import functional as F

loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

# target = torch.zeros((16,1,32,32))
pred = torch.randn((1,1,32,32))
pred = torch.nn.Sigmoid()(pred)

pred[pred >=0.5] = 1
pred[pred < 0.5] = 0
pred = torch.tensor(pred,dtype=torch.uint8)
# l = F.binary_cross_entropy_with_logits(pred,target)
print(pred<<6)