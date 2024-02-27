import torch
import numpy as np
import pdb

# def direction(feature, pixel_xy):
#     # pdb.set_trace()    
#     B = feature.shape[0]
#     weight_ = feature[:, 0:1, :, :]
#     pixel_vec = feature[:, 1:, :, :] + pixel_xy.unsqueeze(0)
#     weight = torch.nn.functional.softmax(weight_.reshape([B, -1]), dim=-1).reshape([B, 1, feature.shape[-2], feature.shape[-1]])
#     mean_vec = (weight * pixel_vec).sum((-1, -2), keepdim=True) # B,2
#     # pdb.set_trace()
#     var = (((pixel_vec - mean_vec) ** 2) * weight).sum((-1, -2), keepdim=True).mean()
#     return mean_vec, var

# def my_loss(direction, label_cuda):
#     direction_loss = ((direction - label_cuda) ** 2).mean()
#     # var_loss = variance
#     return direction_loss


def my_loss(feature, label_cuda, pixel_xy):
    B = feature.shape[0]
    weight_ = feature[:, 0:1, :, :]
    pixel_vec = feature[:, 1:, :, :] + pixel_xy.unsqueeze(0)
    weight = torch.nn.functional.softmax(weight_.reshape([B, -1]), dim=-1).reshape([B, 1, feature.shape[-2], feature.shape[-1]])
    mean_vec = (weight * pixel_vec).sum((-1, -2), keepdim=True) # B,2
    # pdb.set_trace()
    var = (((pixel_vec - mean_vec) ** 2) * weight).sum((-1, -2)).mean(-1)
    if label_cuda is None:
        direction_loss = None
    else:
        direction_loss = ((mean_vec.squeeze(-1).squeeze(-1) - label_cuda) ** 2).mean(-1)
    # pdb.set_trace()
    return mean_vec.squeeze(-1).squeeze(-1), var, direction_loss