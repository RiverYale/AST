import torch
from torch import nn
import numpy as np

# this is equivalent to the loss function in CVMNet with alpha=10, here we simplify it with cosine similarity
class SoftTripletBiLoss(nn.Module):
    def __init__(self, margin=None, alpha=15, **kwargs):
        super(SoftTripletBiLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, inputs_q, inputs_k, labels):
        loss_1, mean_pos_sim_1, mean_neg_sim_1 = self.single_forward(inputs_q, inputs_k, labels)
        loss_2, mean_pos_sim_2, mean_neg_sim_2 = self.single_forward(inputs_k, inputs_q, labels)
        return (loss_1+loss_2)*0.5, (mean_pos_sim_1+mean_pos_sim_2)*0.5, (mean_neg_sim_1+mean_neg_sim_2)*0.5

    def single_forward(self, inputs_q, inputs_k, labels):
        n = inputs_q.size(0)

        normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)
        # Compute similarity matrix
        sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())

        # Compute batch loss
        pos_mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        neg_mask = ~pos_mask
        loss_batch = 0
        for i in range(n):
            pos_sim = torch.masked_select(sim_mat[i], pos_mask[i])
            neg_sim = torch.masked_select(sim_mat[i], neg_mask[i])
            pos_n = len(pos_sim)
            neg_n = len(neg_sim)
            pos_sim_ = pos_sim.unsqueeze(dim=1).expand(pos_n, neg_n)
            neg_sim_ = neg_sim.unsqueeze(dim=0).expand(pos_n, neg_n)
            loss_i = torch.log(1 + torch.exp((neg_sim_ - pos_sim_) * self.alpha))
            if torch.isnan(loss_i).any() or torch.isinf(loss_i).any():
                print(inputs_q, inputs_k)
                raise Exception
            loss_batch += loss_i.mean()

        if torch.isnan(loss_batch).any():
            print(inputs_q, inputs_k)
            raise Exception

        loss = loss_batch / n

        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()
        return loss, mean_pos_sim, mean_neg_sim
