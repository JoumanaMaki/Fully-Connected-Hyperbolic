"""Hyperboloid manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature.

    All standard methods (proj, expmap, logmap, etc.) use the original HGCN
    formulas which factor singularities as f(x) * direction/norm, giving
    natural gradient cancellation that works with learnable curvature.

    Extra methods (k, normL, projection_space_orthogonal) are added for
    HypLinearOurs compatibility.
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6
        self._k = torch.tensor(1.0)

    # ---- Curvature access (for HypLinearOurs) ----

    def k(self):
        """Return curvature parameter.

        When the model sets manifold.c (e.g. a learnable nn.Parameter),
        returns that so HypLinearOurs stays consistent with the pipeline.
        Falls back to the stored default.
        """
        return getattr(self, 'c', self._k)

    # ---- Extra methods for HypLinearOurs ----

    def normL(self, x, metric=None):
        """Lorentz norm: sqrt(-x0^2 + x1^2 + ... + xn^2)."""
        if metric is None:
            metric = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        metric[0] = -1
        return (x * x * metric).sum(dim=-1, keepdim=True).sqrt()

    def projection_space_orthogonal(self, x, manifold_dim=-1):
        """Prepend time component computed from space components."""
        k = self.k()
        if isinstance(k, torch.Tensor):
            k = k.to(x.device)
        return torch.cat(
            [torch.sqrt(1 / k + x.pow(2).sum(dim=manifold_dim, keepdim=True)), x],
            dim=manifold_dim
        )

    def _inner(self, u, v, keepdim=False, dim=-1):
        """Minkowski inner product: -u0*v0 + u1*v1 + ... + un*vn."""
        d = u.size(dim) - 1
        uv = u * v
        if not keepdim:
            return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
                dim, 1, d
            ).sum(dim=dim, keepdim=False)
        else:
            return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
                dim=dim, keepdim=True
            )

    # ---- Original HGCN methods ----

    def _safe_K(self, c):
        if isinstance(c, torch.Tensor):
            c = c.clamp(min=1e-6)
        return 1. / c

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        # Detach c: arcosh backward has a singularity at theta=1 (coincident
        # points) that does not cancel for c's gradient.
        c_det = c.detach() if isinstance(c, torch.Tensor) else c
        K = self._safe_K(c_det)
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = self._safe_K(c)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        K = self._safe_K(c)
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap(self, u, x, c):
        K = self._safe_K(c)
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = self._safe_K(c)
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = self._safe_K(c)
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        # Detach c: arcosh backward singularity doesn't cancel for c's gradient.
        c_det = c.detach() if isinstance(c, torch.Tensor) else c
        K = self._safe_K(c_det)
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        K = self._safe_K(c)
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = self._safe_K(c)
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)
