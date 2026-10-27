import torch
import torch.nn.functional as F
from torch import nn


class KANLayer(nn.Module):
    # Each input-output edge is phi(x) = w_b * SiLU(x) + w_s * sum_i c_i B_i(x), with cubic B-splines on
    # `grid` uniform intervals over [low, high] (Liu et al., ICLR 2025). Outside that range only the SiLU term varies.
    # monotone: optional sequence with one entry per input, +1 (non-decreasing), -1 (non-increasing) or 0 (free).
    # A constrained edge uses ordered spline coefficients and a linear base term with a sign-fixed weight, so phi is
    # monotone on [low, high]; inputs to a constrained layer must already be clipped to that range.
    def __init__(self, inputs, outputs, grid=5, order=3, low=-3.0, high=3.0, monotone=None):
        super().__init__()
        self.register_buffer('monotone', torch.zeros(inputs) if monotone is None else torch.tensor(monotone, dtype=torch.float32))
        step = (high - low) / grid
        knots = torch.arange(-order, grid + order + 1, dtype=torch.float32) * step + low
        self.register_buffer('knots', knots.expand(inputs, -1).contiguous())
        self.order = order
        self.base_weight = nn.Parameter(torch.empty(outputs, inputs))
        self.spline_weight = nn.Parameter(torch.randn(outputs, inputs, grid + order) * 0.1)
        self.spline_scale = nn.Parameter(torch.ones(outputs, inputs))
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)

    def bases(self, x):
        # Cox-de Boor recursion; returns (batch, inputs, grid + order).
        knots, x = self.knots, x.unsqueeze(-1)
        values = ((x >= knots[:, :-1]) & (x < knots[:, 1:])).to(x.dtype)
        for degree in range(1, self.order + 1):
            values = ((x - knots[:, :-(degree + 1)]) / (knots[:, degree:-1] - knots[:, :-(degree + 1)]) * values[..., :-1]
                      + (knots[:, degree + 1:] - x) / (knots[:, degree + 1:] - knots[:, 1:-degree]) * values[..., 1:])
        return values

    def coefficients(self):
        free = self.spline_weight * self.spline_scale.unsqueeze(-1)
        if not self.monotone.any():
            return free
        first = self.spline_weight[..., :1]
        ordered = torch.cat([first, first + torch.cumsum(F.softplus(self.spline_weight[..., 1:]), dim=-1)], dim=-1)
        sign = self.monotone.view(1, -1, 1)
        return torch.where(sign != 0, ordered * sign, free)

    def forward(self, x):
        constrained = self.monotone != 0
        activation = torch.where(constrained, x, F.silu(x))
        weight = torch.where(constrained, F.softplus(self.base_weight) * self.monotone, self.base_weight)
        return F.linear(activation, weight) + torch.einsum('bik,oik->bo', self.bases(x), self.coefficients())


class KAN(nn.Module):
    def __init__(self, widths, grid=5, monotone=None):
        super().__init__()
        # Monotone constraints are only defined for single-layer KANs, where each input enters through one edge function.
        assert monotone is None or len(widths) == 2
        self.layers = nn.ModuleList(KANLayer(inputs, outputs, grid, monotone=monotone) for inputs, outputs in zip(widths[:-1], widths[1:]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def spline_penalty(self):
        return torch.cat([layer.coefficients().abs().flatten() for layer in self.layers]).mean()

    def smoothness_penalty(self):
        # Squared second differences of the spline coefficients (P-spline style roughness).
        return torch.cat([(c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]).pow(2).flatten() for c in (layer.coefficients() for layer in self.layers)]).mean()
