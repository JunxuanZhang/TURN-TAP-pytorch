import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import numpy as np


class GroupNormalize(object):
    def __init__(self, p=2):
        self.p = p

    def __call__(self, feat_group):
        if isinstance(feat_group, np.ndarray):
            feat_group = torch.from_numpy(feat_group).contiguous()

        assert len(feat_group.size()) == 1, 'the size of feats is {}, ' \
                                            'but expected {}'.format(len(feat_group.size()), 2)

        return F.normalize(feat_group, p=2, dim=0)


class TURN(torch.nn.Module):
    def __init__(self, tr_batch_size, ts_batch_size, lambda_reg,
                 unit_feature_dim, middle_layer_dim=1000,
                 dropout=0.5, num_class=4, norm_p=2):
        super(TURN, self).__init__()

        self.tr_batch_size = tr_batch_size
        self.ts_batch_size = ts_batch_size
        self.lambda_reg = lambda_reg
        self.unit_feature_dim = unit_feature_dim
        self.input_feature_dim = unit_feature_dim * 3
        self.middle_layer_dim = middle_layer_dim
        self.dropout = dropout
        self.num_class = num_class
        self.norm_p = norm_p

        print(("""
               Initializing TURN ...
               
               Configurations of TURN:
               training batch size:        {}
               testing batch size:         {}
               lambda for regression:      {}
               unit feature size:          {}
               input feature size:         {}
               middle_layer_dim:           {}
               dropout_ratio:              {}
              """.format(tr_batch_size, ts_batch_size, lambda_reg, unit_feature_dim,
                         self.input_feature_dim, middle_layer_dim, dropout)))

        self._prepare_turn_model()

    def _prepare_turn_model(self):
        self.fc_layer = nn.Linear(self.input_feature_dim, self.middle_layer_dim)

        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        self.output_layer = nn.Linear(self.middle_layer_dim, self.num_class)

        nn.init.normal(self.fc_layer.weight.data, 0, 0.001)
        nn.init.constant(self.fc_layer.bias.data, 0)
        nn.init.normal(self.output_layer.weight.data, 0, 0.001)
        nn.init.constant(self.output_layer.bias.data, 0)

    def forward(self, inputdata):
        out = self.fc_layer(inputdata)
        out = F.relu(out, inplace=True)
        if self.training:
            out = self.dropout_layer(out)
        out = self.output_layer(out)

        return out

    def get_optim_policies(self):
        weights = list()
        bias = list()

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                weights.append(ps[0])
                if len(ps) == 2:
                    bias.append(ps[1])
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': weights, 'lr_mult': 1, 'weight_decay_mult': 1,
             'name': "weight"},
            {'params': bias, 'lr_mult': 2, 'weight_decay_mult': 0,
             'name': "bias"},
        ]

    def data_preparation(self):
        return torchvision.transforms.Compose([GroupNormalize(self.norm_p), ])
