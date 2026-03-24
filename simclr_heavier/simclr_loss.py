import torch
import torch.nn as nn
import torch.nn.functional as F

#stands for normalized temperature-scaled cross entropy loss
class NTXentLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        
        similarity_mat = torch.mm(z, z.T) / self.temp  
        identity_matrix_mask = torch.eye(2*batch_size,dtype=torch.bool,device=z.device)
        similarity_mat.masked_fill_(identity_matrix_mask, float('-inf'))

        counterpart_idx = torch.cat([
            torch.arange(batch_size, 2*batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ]) 

        loss = F.cross_entropy(similarity_mat, counterpart_idx)
        return loss