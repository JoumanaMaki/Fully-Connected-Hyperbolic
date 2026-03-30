"""Graph decoders."""
import torch
import manifolds
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


class LorentzMLRDecoder(Decoder):
    """
    MLR decoder using signed geodesic distances to class hyperplanes.

    Computes (1/sqrt(c)) * asinh(sqrt(c) * <x, V>_L) for each class hyperplane,
    producing logits directly from hyperbolic geometry. Only works with Hyperboloid.
    """

    def __init__(self, c, args):
        super(LorentzMLRDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes

        in_spatial = self.input_dim - 1

        # Hyperplane parameters: direction (weight-normalized) + offset
        self.v = nn.Parameter(torch.Tensor(in_spatial, self.output_dim))
        self.g = nn.Parameter(torch.Tensor(self.output_dim))
        self.a = nn.Parameter(torch.Tensor(1, self.output_dim))

        self.reset_parameters()
        self.decode_adj = False

    def reset_parameters(self):
        init.kaiming_normal_(self.v)
        std = (5.0 / (self.input_dim - 1)) ** 0.5
        self.g.data.fill_(std)
        self.a.data.fill_(0.0)

    def get_U(self):
        v_norm = self.v.norm(dim=0, keepdim=True).clamp(min=1e-8)
        g_pos = F.softplus(self.g)
        return g_pos.unsqueeze(0) * self.v / v_norm

    def create_spacelike_vector(self, c):
        K = 1.0 / c if isinstance(c, torch.Tensor) else 1.0 / c
        sqrt_K = K.sqrt() if isinstance(K, torch.Tensor) else K ** 0.5
        U = self.get_U()
        U_norm = U.norm(dim=0, keepdim=True).clamp(min=1e-10)
        arg = (sqrt_K * self.a / U_norm).clamp(-100, 100)
        time = -U_norm * torch.sinh(arg)
        space = torch.cosh(arg) * U
        return torch.cat([time, space], dim=0)

    def decode(self, x, adj):
        c = self.c
        K = 1.0 / c if isinstance(c, torch.Tensor) else 1.0 / c
        sqrt_K = K.sqrt() if isinstance(K, torch.Tensor) else K ** 0.5
        V = self.create_spacelike_vector(c)
        # Signed distance as logits: sqrt(K) * asinh(<x, V>_L / sqrt(K))
        return sqrt_K * torch.asinh((x @ V) / sqrt_K)

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
                self.input_dim, self.output_dim, self.c
        )


def _resolve_decoder(model_name, args):
    """Resolve decoder class, respecting --decoder-variant for hyperbolic models."""
    base = model2decoder[model_name]
    variant = getattr(args, 'decoder_variant', 'standard')
    if variant == 'mlr' and model_name in ('HGCN', 'HNN'):
        return LorentzMLRDecoder
    return base


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}

