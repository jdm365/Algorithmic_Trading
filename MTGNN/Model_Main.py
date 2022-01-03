import torch as T
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import time
import datetime


class Linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(Linear, self).__init__()
        self._mlp = T.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameteres():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)

    def forward(self, X):
        return self._mlp(X)


class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdepth, dropout, alpha):
        super(MixProp, self).__init__()
        self._mlp = Linear((gdepth + 1) * c_in, c_out)
        self._gdepth = gdepth
        self._dropout = dropout
        self._alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameteres():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)

    def forward(self, X, A):
        A = A + T.eye(A.size(0)).to(X.device)
        d = A.sum(1)
        H = X
        H_0 = X
        A = A / d.view(-1, 1)
        for _ in range(self._gdepth):
            H = self._alpha * X + (1- self._alpha) * T.einsum(
                'ncwl,vw->ncvl', (H, A)
            )
            H_0 = T.cat((H_0, H), dim=1)
        H_0 = self._mlp(H_0)
        return H_0


class DilatedInception(nn.Module):
    def __init__(self, c_in, c_out, kernel_set, dilation_factor):
        super(DilatedInception, self).__init()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out / len(self._kernel_set))
        for kernel in self._kernel_set:
            self._time_conv.append(
                nn.Conv2d(c_in, c_out, kernel_size=(1, kernel), dilation=(1, dilation_factor))
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameteres():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)

    def forward(self, X_in):
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3):]
        X = T.cat(X, dim=1)
        return X


class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha, xd=None):
        super(GraphConstructor, self).__init__()
        if xd is not None:
            self._static_feature_dim = xd
            self._linear1 = nn.Linear(xd, dim)
            self._linear2 = nn.Linear(xd, dim)
        else:
            self._embedding1 = nn.Embedding(nnodes, dim)
            self._embedding2 = nn.Embedding(nnodes, dim)
            self._linear1 = nn.Linear(dim, dim)
            self._linear2 = nn.Linear(dim, dim)
        self._k = k
        self._alpha = alpha

    def _reset_parameters(self):
        for param in self.parameteres():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)

    def forward(self, idx, FE=None):
        if FE is None:
            nodevec1 = self._embedding1(idx)
            nodevec2 = self._embedding2(idx)
        else:
            assert FE.shape[1] == self._static_feature_dim
            nodevec1 = FE[idx, :]
            nodevec2 = nodevec1
        
        nodevec1 = T.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = T.tanh(self._alpha * self._linear2(nodevec2))

        a = T.mm(nodevec1, nodevec2.transpose(1, 0)) - T.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        A = F.relu(T.tanh(self._alpha * a))
        mask = T.zeros(idx.size(0), idx.size(0)).to(A.device)
        mask.fill_(float('0'))
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask
        return A


class LayerNormalization(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNormalization, self).__init__()
        self._normalized_shape = normalized_shape
        self._eps = eps
        self._elementwise_affine = elementwise_affine
        if self._elementwise_affine:
            self._weight = nn.Parameter(T.Tensor(*normalized_shape))
            self._bias = nn.Parameter(T.Tensor(*normalized_shape))
        else:
            self.register_parameter('_weight', None)
            self.register_parameter('_bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._elementwise_affine:
            init.ones_(self._weight)
            init.zeros_(self._bias)

    def forward(self, X, idx):
        if self._elementwise_Affine:
            return F.layer_norm(
                X,
                tuple(X.shape[1:]),
                self._weight[:, idx, :],
                self._bias[:, idx, :],
                self._eps
            )
        else:
            return F.layer_norm(
                X, tuple(X.shape[1:]), self._weight, self._bias, self._eps
            )


class MTGNNLayer(nn.Module):
    def __init__(
        self,
        dilation_exponential,
        rf_size_i,
        kernel_size,
        j,
        residual_channels,
        conv_channels,
        skip_channels,
        kernel_set,
        new_dilation,
        layer_norm_affine,
        gcn_true,
        seq_length,
        receptive_field,
        dropout,
        gcn_depth,
        num_nodes,
        propalpha
    ):
        super(MTGNNLayer, self).__init__()
        self._dropout= dropout
        self._gcn_true = gcn_true

        if dilation_exponential > 1:
            rf_size_j = int(rf_size_i + (kernel_size - 1) 
                * (dilation_exponential ** j - 1 ) / (dilation_exponential - 1))
        else:
            rf_size_j = rf_size_i + j * (kernel_size - 1)

        self._filter_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation
        )

        self._gate_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation
        )

        self._residual_conv = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=residual_channels,
            kernel_size=(1, 1)
        )

        if seq_length > receptive_field:
            self._skip_conv =nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, seq_length - rf_size_j + 1)
            )
        else:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, receptive_field - rf_size_j + 1)
            )

        if gcn_true:
            self._mixprop_conv1 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )
            self._mixprop_conv2 = MixProp(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha
            )
        
        if seq_length > receptive_field:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, receptive_field - rf_size_j + 1),
                elementwise_affine=layer_norm_affine
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameteres():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)

    def forward(self ,X, X_skip, A_tilde, idx, training):
        X_residual = X
        X_filter = self._filter_conv(X)
        X_filter = T.tanh(X_filter)
        X_gate = self._gate_conv(X)
        X_gate = T.sigmoid(X_gate)
        X = X_filter * X_gate
        X = F.dropout(X, self._dropout, training=training)
        X_skip = self._skip_conv(X) + X_skip
        if self._gcn_true:
            X = self._mixprop_conv1(X, A_tilde) + self._mixprop_conv2(
                X, A_tilde.transpose(1, 0)
            )
        else:
            X = self._residual_conv(X)

        X = X + X_residual[:, :, :, -X.size(3):]
        X = self._normalization(X, idx)
        return X, X_skip
