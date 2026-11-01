"""
smnlmm2: block Gibbs sampler for Bayesian linear mixed models with
  * several variance components (u split into K components with scales lambda_k),
  * scale-mixture random effects, possibly with an unknown tail parameter
    (finite grid prior; the tail parameter is drawn jointly with the weights),
  * scale-mixture errors (weights omega_j), possibly with unknown tail parameter.

Model
    y_j | beta,u,lam_e,omega ~ N(x_j'beta + z_j'u, (lam_e omega_j)^{-1}),  omega_j iid ~ G
    u_i | lam, eta          ~ N(0, (lam_{k(i)} eta_i)^{-1}),              eta_i iid ~ H_{k(i)}
    beta ~ N(mu, Sigma),  lam_e ~ Gamma(a_e,b_e),  lam_k ~ Gamma(a_k,b_k)
Sampler (three blocks):  lambda | (eta,omega,theta)  ->  (eta,omega[,d,nu]) | (lambda,theta)
                          ->  theta | (lambda,eta,omega)
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass, field
import numpy as np
from scipy import special
from smnlmm import Mixing, Normal, StudentT, Slash, ContNormal, Laplace


class GridMixing:
    """Mixing family with unknown tail parameter on a finite grid (prior weights).
    family in {'t','slash'}.  The pair (param, weights) is drawn jointly given s."""
    def __init__(self, family, values, prior=None):
        self.family = family
        self.values = np.asarray(values, float)
        p = np.ones(len(values)) if prior is None else np.asarray(prior, float)
        self.logprior = np.log(p / p.sum())
        self.make = (lambda v: StudentT(v)) if family == 't' else (lambda v: Slash(v))

    def label(self):
        return f"{self.family}(grid {self.values.min():g}-{self.values.max():g})"

    def logZ(self, v, s):
        """log of int eta^{1/2} exp(-s eta) h_v(eta) d eta (vectorised in s)."""
        s = np.asarray(s, float)
        if self.family == 't':
            a = v / 2.0
            return a * math.log(a) - special.gammaln(a) + special.gammaln(a + 0.5) - (a + 0.5) * np.log(a + s)
        a = v + 0.5
        out = np.full_like(s, math.log(v) - math.log(a))
        big = s > 1e-10
        sb = s[big]
        out[big] = math.log(v) + special.gammaln(a) + np.log(special.gammainc(a, sb)) - a * np.log(sb)
        return out

    def sample_block(self, rng, s):
        lp = np.array([self.logprior[k] + self.logZ(v, s).sum() for k, v in enumerate(self.values)])
        p = np.exp(lp - lp.max()); p /= p.sum()
        k = rng.choice(len(self.values), p=p)
        v = self.values[k]
        return v, self.make(v).sample_tilted(rng, s)

    def sample_prior(self, rng, size):
        v = self.values[rng.choice(len(self.values), p=np.exp(self.logprior))]
        return v, self.make(v).sample_prior(rng, size)


def _draw_weights(mix, rng, s):
    if isinstance(mix, GridMixing):
        return mix.sample_block(rng, s)
    return np.nan, mix.sample_tilted(rng, s)


@dataclass
class Priors2:
    mu_beta: np.ndarray
    Sigma_beta: np.ndarray
    a_e: float = 1.0
    b_e: float = 0.1
    a_u: np.ndarray | float = 1.0      # per component
    b_u: np.ndarray | float = 0.1


@dataclass
class Fit2:
    beta: np.ndarray
    lam_e: np.ndarray
    lam_u: np.ndarray        # (n, K)
    tail_u: np.ndarray       # (n, K) grid parameter per component (nan if fixed)
    tail_e: np.ndarray       # (n,)
    u_mean: np.ndarray
    eta_mean: np.ndarray
    omega_mean: np.ndarray
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def gibbs2(y, X, Z, comp, re_mix, priors: Priors2, err_mix=None, n_iter=20000, burn=2000,
           seed=1, init=None, keep=None):
    """comp: int labels (length q) of the variance component of each u_i;
    re_mix: list (length K) of Mixing/GridMixing; err_mix: Mixing/GridMixing or None."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    N, p = X.shape; q = Z.shape[1]
    comp = np.asarray(comp, int); K = comp.max() + 1
    qk = np.bincount(comp, minlength=K)
    a_u = np.broadcast_to(np.asarray(priors.a_u, float), (K,)); b_u = np.broadcast_to(np.asarray(priors.b_u, float), (K,))
    Sb_inv = np.linalg.inv(priors.Sigma_beta); Sb_mu = Sb_inv @ priors.mu_beta
    W = np.hstack([X, Z])
    one_per_row = np.all((Z != 0).sum(1) <= 1)
    zcol = np.argmax(Z != 0, axis=1); zval = Z[np.arange(N), zcol]
    if init is None:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]; u = np.zeros(q); eta = np.ones(q); omega = np.ones(N)
    else:
        beta, u, eta, omega = (np.array(v, float) for v in init)
    nk = n_iter - burn
    out = dict(beta=np.empty((nk, p)), lam_e=np.empty(nk), lam_u=np.empty((nk, K)), tail_u=np.full((nk, K), np.nan),
               tail_e=np.full(nk, np.nan))
    keepd = {k: [] for k in (keep or [])}
    u_s = np.zeros(q); eta_s = np.zeros(q); om_s = np.zeros(N)
    for it in range(n_iter):
        r = y - X @ beta - Z @ u
        lam_e = rng.gamma(priors.a_e + N / 2.0, 1.0 / (priors.b_e + 0.5 * np.sum(omega * r * r)))
        ssq = np.bincount(comp, weights=eta * u * u, minlength=K)
        lam_u = rng.gamma(a_u + qk / 2.0, 1.0 / (b_u + 0.5 * ssq))
        lam_i = lam_u[comp]
        # joint block: (eta [, tail params]) and (omega [, nu]) given (lambda, theta)
        tails = np.full(K, np.nan)
        for k in range(K):
            m = comp == k
            tails[k], eta[m] = _draw_weights(re_mix[k], rng, 0.5 * lam_i[m] * u[m] ** 2)
        nu = np.nan
        if err_mix is not None:
            nu, omega = _draw_weights(err_mix, rng, 0.5 * lam_e * r * r)
        # theta | lambda, eta, omega
        lw = lam_e * omega
        if one_per_row:
            Xw = X * lw[:, None]
            A = X.T @ Xw + Sb_inv
            B = np.zeros((p, q)); np.add.at(B.T, zcol, (Xw * zval[:, None]))
            c = np.bincount(zcol, weights=lw * zval ** 2, minlength=q) + lam_i * eta
            b1 = Xw.T @ y + Sb_mu
            b2 = np.bincount(zcol, weights=lw * zval * y, minlength=q)
            S = A - (B / c) @ B.T
            L = np.linalg.cholesky(S)
            mbeta = np.linalg.solve(S, b1 - B @ (b2 / c))
            beta = mbeta + np.linalg.solve(L.T, rng.standard_normal(p))
            u = (b2 - B.T @ beta) / c + rng.standard_normal(q) / np.sqrt(c)
        else:
            if err_mix is None:
                if '_WtW' not in locals():
                    _WtW = W.T @ W; _Wty = W.T @ y
                P = lam_e * _WtW
                bv = lam_e * _Wty
            else:
                Ww = W * lw[:, None]
                P = W.T @ Ww
                bv = Ww.T @ y
            P[:p, :p] += Sb_inv
            P[p:, p:] += np.diag(lam_i * eta)
            bv = bv.copy(); bv[:p] += Sb_mu
            L = np.linalg.cholesky(P)
            th = np.linalg.solve(P, bv) + np.linalg.solve(L.T, rng.standard_normal(p + q))
            beta, u = th[:p], th[p:]
        if it >= burn:
            k = it - burn
            out['beta'][k] = beta; out['lam_e'][k] = lam_e; out['lam_u'][k] = lam_u
            out['tail_u'][k] = tails; out['tail_e'][k] = nu
            u_s += u; eta_s += eta; om_s += omega
            for key in keepd:
                keepd[key].append(dict(u=u, eta=eta, omega=omega)[key].copy())
    f = Fit2(out['beta'], out['lam_e'], out['lam_u'], out['tail_u'], out['tail_e'], u_s / nk, eta_s / nk, om_s / nk,
             time.time() - t0)
    f.extra = {k: np.array(v) for k, v in keepd.items()}
    return f
