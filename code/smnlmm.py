"""
smnlmm: block Gibbs sampler for Bayesian linear mixed models whose random
effects are scale mixtures of normals (SMN).

Model
-----
    Y | beta, u, lam_e        ~ N_N(X beta + Z u, lam_e^{-1} I)
    beta                      ~ N_p(mu_beta, Sigma_beta)
    lam_e ~ Gamma(a_e, b_e),  lam_u ~ Gamma(a_u, b_u)          (rate parametrisation)
    u_i | lam_u, eta_i        ~ N(0, (lam_u eta_i)^{-1}),  eta_i iid ~ H

The sampler cycles  lambda -> eta -> theta=(beta,u)  exactly as analysed in the
paper.  The eta-step draws from the exponentially tilted law
    H_s(d eta) propto eta^{1/2} exp(-s eta) H(d eta),   s = lam_u u_i^2 / 2.

Mixing families implemented (parametrised by the precision multiplier eta):
    Normal()              eta = 1
    StudentT(d)           eta ~ Gamma(d/2, rate d/2)
    Slash(nu)             eta ~ Beta(nu, 1)
    ContNormal(nu, gam)   eta = gam w.p. nu, 1 w.p. 1-nu   (0 < gam < 1)
    Laplace()             1/eta ~ Exp(rate 1/2)  (u | lam_u is Laplace)
    GIG(rho, chi, psi)    eta ~ GIG(rho, chi, psi), chi, psi > 0  (normal-GIG; since eta is a precision,
                          the NIG law is eta ~ GIG(1/2, psi, chi) and the hyperbolic law eta ~ GIG(-1, psi, chi))
    VarianceGamma(al, be) eta ~ IG(al, be) = GIG(-al, 2 be, 0)  (Laplace is al=1, be=1/2)
GIG(rho, chi, psi) has density proportional to x^{rho-1} exp{-(chi/x + psi x)/2}.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import special, stats


# ----------------------------------------------------------------------------
# Mixing families
# ----------------------------------------------------------------------------
class Mixing:
    name = "base"

    def sample_prior(self, rng, size):
        raise NotImplementedError

    def sample_tilted(self, rng, s):
        """Draw eta_i ~ H_{s_i} independently for the array s >= 0."""
        raise NotImplementedError

    def quadrature(self, n=80):
        """Nodes and weights with E_H[g(eta)] ~= sum_k w_k g(eta_k)."""
        raise NotImplementedError

    def label(self):
        return self.name


def _loggrid_quadrature(logdens, lo=-30.0, hi=12.0, n=600):
    """Trapezoid rule in t = log(eta): E_H[g] ~= sum_k w_k g(eta_k)."""
    t = np.linspace(lo, hi, n)
    eta = np.exp(t)
    lw = logdens(eta) + t
    w = np.exp(lw - lw.max())
    w[0] *= 0.5
    w[-1] *= 0.5
    w /= w.sum()
    return eta, w


class Normal(Mixing):
    name = "Normal"

    def sample_prior(self, rng, size):
        return np.ones(size)

    def sample_tilted(self, rng, s):
        return np.ones_like(s)

    def quadrature(self, n=1):
        return np.array([1.0]), np.array([1.0])


@dataclass
class StudentT(Mixing):
    d: float
    name: str = field(default="t", init=False)

    def label(self):
        return f"t({self.d:g})"

    def sample_prior(self, rng, size):
        return rng.gamma(self.d / 2.0, 2.0 / self.d, size)

    def sample_tilted(self, rng, s):
        shape = (self.d + 1.0) / 2.0
        rate = self.d / 2.0 + s
        return rng.gamma(shape, 1.0 / rate)

    def logdens(self, eta):
        a = self.d / 2.0
        return a * math.log(a) - special.gammaln(a) + (a - 1.0) * np.log(eta) - a * eta

    def quadrature(self, n=1500):
        lo = -40.0 / min(self.d / 2.0, 1.0)
        return _loggrid_quadrature(self.logdens, lo=lo, hi=12.0, n=n)


@dataclass
class Slash(Mixing):
    nu: float
    name: str = field(default="slash", init=False)

    def label(self):
        return f"slash({self.nu:g})"

    def sample_prior(self, rng, size):
        return rng.beta(self.nu, 1.0, size)

    def sample_tilted(self, rng, s):
        # density propto eta^{nu-1/2} exp(-s eta) on (0,1): Gamma(nu+1/2, s) truncated to (0,1)
        a = self.nu + 0.5
        s = np.asarray(s, dtype=float)
        out = np.empty_like(s)
        small = s < 1e-8
        if np.any(small):  # limit s -> 0 is Beta(nu+1/2, 1)
            out[small] = rng.beta(a, 1.0, small.sum())
        big = ~small
        if np.any(big):
            sb = s[big]
            upper = special.gammainc(a, sb)            # P(G <= 1) for G ~ Gamma(a, rate sb)
            v = rng.uniform(size=sb.size) * upper
            x = special.gammaincinv(a, v) / sb
            out[big] = np.clip(x, 1e-300, 1.0)
        return out

    def logdens(self, eta):
        return np.where(eta <= 1.0, math.log(self.nu) + (self.nu - 1.0) * np.log(eta), -np.inf)

    def quadrature(self, n=200):
        # Gauss-Jacobi rule for the weight eta^{nu-1/2} on (0,1), returned as weights for
        # E_H[g] = nu * int_0^1 eta^{nu-1/2} {g(eta) eta^{-1/2}} d eta.  The subject-level
        # likelihood behaves like eta^{1/2} near zero, so the bracketed function is smooth.
        b = self.nu - 0.5
        x, w = special.roots_jacobi(n, 0.0, b)
        eta = (1.0 + x) / 2.0
        return eta, w * self.nu * 2.0 ** (-(b + 1.0)) / np.sqrt(eta)


@dataclass
class ContNormal(Mixing):
    nu: float
    gam: float
    name: str = field(default="CN", init=False)

    def label(self):
        return f"CN({self.nu:g},{self.gam:g})"

    def sample_prior(self, rng, size):
        return np.where(rng.uniform(size=size) < self.nu, self.gam, 1.0)

    def sample_tilted(self, rng, s):
        s = np.asarray(s, dtype=float)
        lw1 = math.log(self.nu) + 0.5 * math.log(self.gam) - s * self.gam
        lw2 = math.log1p(-self.nu) - s
        p1 = 1.0 / (1.0 + np.exp(lw2 - lw1))
        return np.where(rng.uniform(size=s.size) < p1, self.gam, 1.0)

    def quadrature(self, n=2):
        return np.array([self.gam, 1.0]), np.array([self.nu, 1.0 - self.nu])


class Laplace(Mixing):
    """1/eta ~ Exp(rate 1/2), so u | lam_u ~ Laplace with scale lam_u^{-1/2}."""
    name = "Laplace"

    def sample_prior(self, rng, size):
        return 1.0 / rng.exponential(2.0, size)

    def sample_tilted(self, rng, s):
        # eta ~ GIG(p=-1/2, chi=1, psi=2s) = inverse Gaussian(mean 1/sqrt(2s), shape 1)
        s = np.maximum(np.asarray(s, dtype=float), 1e-300)
        return rng.wald(1.0 / np.sqrt(2.0 * s), 1.0)

    def logdens(self, eta):
        return -math.log(2.0) - 2.0 * np.log(eta) - 0.5 / eta

    def quadrature(self, n=600):
        return _loggrid_quadrature(self.logdens, lo=-12.0, hi=30.0, n=n)


def _rgig_std(rng, lam, om):
    """Draws from the density proportional to x^{lam-1} exp{-om (x + 1/x)/2}, lam >= 0, om > 0
    (arrays of equal shape), by the rejection algorithm of Devroye (2014, Statistics and
    Computing 24, 239-246), vectorised over the elements."""
    lam = np.asarray(lam, float); om = np.asarray(om, float)
    out = np.empty(lam.shape)
    todo = np.arange(lam.size)
    lam_f, om_f = lam.ravel(), om.ravel()
    out_f = out.ravel()
    while todo.size:
        L, O = lam_f[todo], om_f[todo]
        al = np.sqrt(O * O + L * L) - L
        psi = lambda x: -al * (np.cosh(x) - 1.0) - L * (np.expm1(x) - x)
        dpsi = lambda x: -al * np.sinh(x) - L * np.expm1(x)
        m1 = -psi(np.ones_like(L))
        with np.errstate(divide="ignore"):
            t = np.where(m1 > 2.0, np.sqrt(2.0 / (al + L)), np.where(m1 < 0.5, np.log(4.0 / (al + 2.0 * L)), 1.0))
            m2 = -psi(-np.ones_like(L))
            s2 = np.minimum(np.where(L > 0, 1.0 / np.where(L > 0, L, 1.0), np.inf),
                            np.log1p(1.0 / al + np.sqrt(1.0 / al ** 2 + 2.0 / al)))
            s = np.where(m2 > 2.0, np.sqrt(4.0 / (al * np.cosh(1.0) + L)), np.where(m2 < 0.5, s2, 1.0))
        eta, zeta = -psi(t), -dpsi(t)
        theta, xi = -psi(-s), dpsi(-s)
        p, r = 1.0 / xi, 1.0 / zeta
        td, sd = t - r * eta, s - p * theta
        qq = td + sd
        U, V, W = rng.uniform(size=(3, todo.size))
        tot = p + qq + r
        X = np.where(U < qq / tot, -sd + qq * V,
                     np.where(U < (qq + r) / tot, td - r * np.log(V), -sd + p * np.log(V)))
        chi = np.where(X > td, np.exp(-eta - zeta * (X - t)),
                       np.where(X < -sd, np.exp(-theta + xi * (X + s)), 1.0))
        acc = W * chi <= np.exp(psi(X))
        idx = todo[acc]
        Lacc, Oacc = L[acc], O[acc]
        out_f[idx] = (Lacc / Oacc + np.sqrt(1.0 + (Lacc / Oacc) ** 2)) * np.exp(X[acc])
        todo = todo[~acc]
    return out_f.reshape(lam.shape)


def _rgig(rng, rho, chi, psi):
    """Draws from GIG(rho, chi, psi), density proportional to x^{rho-1} exp{-(chi/x + psi x)/2},
    with chi, psi > 0 (vectorised in chi and psi).  Uses X ~ GIG(-rho, psi, chi) => 1/X ~ GIG(rho, chi, psi)."""
    chi, psi = np.broadcast_arrays(np.asarray(chi, float), np.asarray(psi, float))
    om = np.sqrt(chi * psi)
    y = _rgig_std(rng, np.full(om.shape, abs(rho)), om)
    if rho < 0:
        y = 1.0 / y
    return np.sqrt(chi / psi) * y


@dataclass
class GIG(Mixing):
    """eta ~ GIG(rho, chi, psi) with chi, psi > 0 (normal-GIG random effects)."""
    rho: float
    chi: float
    psi: float
    name: str = field(default="GIG", init=False)

    def label(self):
        return f"GIG({self.rho:g},{self.chi:g},{self.psi:g})"

    def sample_prior(self, rng, size):
        return _rgig(rng, self.rho, np.full(size, self.chi), np.full(size, self.psi))

    def sample_tilted(self, rng, s):
        s = np.asarray(s, dtype=float)
        return _rgig(rng, self.rho + 0.5, np.full(s.shape, self.chi), self.psi + 2.0 * s)

    def logdens(self, eta):
        w = math.sqrt(self.chi * self.psi)
        lc = 0.5 * self.rho * math.log(self.psi / self.chi) - math.log(2.0) - (math.log(special.kve(self.rho, w)) - w)
        return lc + (self.rho - 1.0) * np.log(eta) - 0.5 * (self.chi / eta + self.psi * eta)

    def quadrature(self, n=1500):
        return _loggrid_quadrature(self.logdens, lo=-40.0, hi=40.0, n=n)


@dataclass
class VarianceGamma(Mixing):
    """eta ~ IG(al, be) = GIG(-al, 2 be, 0): u | lam_u is variance-gamma (Laplace for al=1, be=1/2)."""
    al: float
    be: float
    name: str = field(default="VG", init=False)

    def label(self):
        return f"VG({self.al:g},{self.be:g})"

    def sample_prior(self, rng, size):
        return self.be / rng.gamma(self.al, 1.0, size)

    def sample_tilted(self, rng, s):
        s = np.maximum(np.asarray(s, dtype=float), 1e-300)
        return _rgig(rng, 0.5 - self.al, np.full(s.shape, 2.0 * self.be), 2.0 * s)

    def logdens(self, eta):
        return self.al * math.log(self.be) - special.gammaln(self.al) - (self.al + 1.0) * np.log(eta) - self.be / eta

    def quadrature(self, n=1500):
        return _loggrid_quadrature(self.logdens, lo=-40.0, hi=60.0, n=n)


# ----------------------------------------------------------------------------
# Priors and data containers
# ----------------------------------------------------------------------------
@dataclass
class Priors:
    mu_beta: np.ndarray
    Sigma_beta: np.ndarray
    a_e: float = 1.0
    b_e: float = 0.01
    a_u: float = 1.0
    b_u: float = 0.01


@dataclass
class Fit:
    family: str
    beta: np.ndarray        # (n_keep, p)
    lam_e: np.ndarray       # (n_keep,)
    lam_u: np.ndarray       # (n_keep,)
    u_mean: np.ndarray      # posterior mean of u
    eta_mean: np.ndarray    # posterior mean of eta
    eta_low: np.ndarray     # P(eta_i < 1 | y), outlier indicator
    u_draws: np.ndarray | None = None
    eta_draws: np.ndarray | None = None
    seconds: float = 0.0
    scale_groups: np.ndarray | None = None


# ----------------------------------------------------------------------------
# Gibbs sampler
# ----------------------------------------------------------------------------
def gibbs(y, X, Z, mixing: Mixing, priors: Priors, n_iter=20000, burn=2000,
          seed=1, init=None, keep_u=False, keep_eta=False, scale_groups=None):
    """Run the lambda -> eta -> theta block Gibbs sampler.

    If Z^T Z is diagonal (e.g. random intercepts) the theta-draw uses the exact
    Schur-complement factorisation; otherwise a dense Cholesky factorisation.
    If scale_groups (integer labels of length q) is given, u_i has its own scale
    precision lam_u[g(i)] with independent Gamma(a_u, b_u) priors.
    """
    import time

    t0 = time.time()
    rng = np.random.default_rng(seed)
    N, p = X.shape
    q = Z.shape[1]
    Sb_inv = np.linalg.inv(priors.Sigma_beta)
    Sb_inv_mu = Sb_inv @ priors.mu_beta
    XtX, XtZ, ZtZ = X.T @ X, X.T @ Z, Z.T @ Z
    Xty, Zty = X.T @ y, Z.T @ y
    diagZ = np.allclose(ZtZ, np.diag(np.diag(ZtZ)))
    nZ = np.diag(ZtZ).copy()
    W = np.hstack([X, Z])
    WtW = W.T @ W

    # initial state (eta, theta)
    if init is None:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        u = np.zeros(q)
        eta = np.ones(q)
    else:
        beta, u, eta = (np.array(v, dtype=float) for v in init)

    sg = None if scale_groups is None else np.asarray(scale_groups, dtype=int)
    G = 1 if sg is None else sg.max() + 1
    qg = None if sg is None else np.bincount(sg, minlength=G)
    n_keep = n_iter - burn
    out_beta = np.empty((n_keep, p))
    out_le = np.empty(n_keep)
    out_lu = np.empty(n_keep) if sg is None else np.empty((n_keep, G))
    u_sum = np.zeros(q)
    eta_sum = np.zeros(q)
    eta_low = np.zeros(q)
    u_draws = np.empty((n_keep, q)) if keep_u else None
    eta_draws = np.empty((n_keep, q)) if keep_eta else None

    for it in range(n_iter):
        # ---- lambda | eta, theta -------------------------------------------
        resid = y - X @ beta - Z @ u
        lam_e = rng.gamma(priors.a_e + N / 2.0, 1.0 / (priors.b_e + 0.5 * resid @ resid))
        if sg is None:
            lam_u = rng.gamma(priors.a_u + q / 2.0, 1.0 / (priors.b_u + 0.5 * np.sum(eta * u * u)))
            lam_ui = lam_u
        else:
            ssq = np.bincount(sg, weights=eta * u * u, minlength=G)
            lam_u = rng.gamma(priors.a_u + qg / 2.0, 1.0 / (priors.b_u + 0.5 * ssq))
            lam_ui = lam_u[sg]
        # ---- eta | lambda, theta ---------------------------------------------
        eta = mixing.sample_tilted(rng, 0.5 * lam_ui * u * u)
        # ---- theta | lambda, eta ---------------------------------------------
        if diagZ:
            A = lam_e * XtX + Sb_inv
            B = lam_e * XtZ
            c = lam_e * nZ + lam_ui * eta
            b1 = lam_e * Xty + Sb_inv_mu
            b2 = lam_e * Zty
            S = A - (B / c) @ B.T
            L = np.linalg.cholesky(S)
            m = np.linalg.solve(S, b1 - B @ (b2 / c))
            beta = m + np.linalg.solve(L.T, rng.standard_normal(p))
            u = (b2 - B.T @ beta) / c + rng.standard_normal(q) / np.sqrt(c)
        else:
            P = lam_e * WtW
            P[:p, :p] += Sb_inv
            P[p:, p:] += np.diag(lam_ui * eta)
            bvec = lam_e * np.concatenate([Xty, Zty])
            bvec[:p] += Sb_inv_mu
            L = np.linalg.cholesky(P)
            m = np.linalg.solve(P, bvec)
            th = m + np.linalg.solve(L.T, rng.standard_normal(p + q))
            beta, u = th[:p], th[p:]

        if it >= burn:
            k = it - burn
            out_beta[k] = beta
            out_le[k] = lam_e
            out_lu[k] = lam_u
            u_sum += u
            eta_sum += eta
            eta_low += eta < 1.0
            if keep_u:
                u_draws[k] = u
            if keep_eta:
                eta_draws[k] = eta

    return Fit(mixing.label(), out_beta, out_le, out_lu, u_sum / n_keep,
               eta_sum / n_keep, eta_low / n_keep, u_draws, eta_draws,
               time.time() - t0, sg)


# ----------------------------------------------------------------------------
# Subject-level marginal likelihood (random-intercept models), WAIC and LPML
# ----------------------------------------------------------------------------
def subject_loglik(fit: Fit, y, X, groups, mixing: Mixing, thin=10, n_quad=None, chunk=50):
    """log p(y_i | beta, lam_e, lam_u) integrating u_i and eta_i out, for each
    retained draw (rows) and subject (columns).  Random-intercept models only."""
    idx = np.arange(0, fit.beta.shape[0], thin)
    nodes, weights = mixing.quadrature() if n_quad is None else mixing.quadrature(n_quad)
    with np.errstate(divide="ignore"):
        logw = np.log(weights)
    labels, inv = np.unique(groups, return_inverse=True)
    G = labels.size
    n = np.bincount(inv, minlength=G).astype(float)[None, :, None]
    out = np.empty((idx.size, G))
    for start in range(0, idx.size, chunk):
        sl = idx[start:start + chunk]
        beta, le, lu = fit.beta[sl], fit.lam_e[sl], fit.lam_u[sl]
        R = y[None, :] - beta @ X.T
        rr = np.stack([np.bincount(inv, weights=r * r, minlength=G) for r in R])[:, :, None]
        rs = np.stack([np.bincount(inv, weights=r, minlength=G) for r in R])[:, :, None]
        sig2 = (1.0 / le)[:, None, None]
        if lu.ndim == 2:   # group-specific scales; subject j <-> random effect j
            lu = lu[:, fit.scale_groups[labels]][:, :, None]
        else:
            lu = lu[:, None, None]
        tau2 = 1.0 / (lu * nodes[None, None, :])
        denom = sig2 + n * tau2
        quad = (rr - tau2 * rs ** 2 / denom) / sig2
        logdet = (n - 1.0) * np.log(sig2) + np.log(denom)
        ll_k = -0.5 * (n * math.log(2 * math.pi) + logdet + quad)
        out[start:start + sl.size] = special.logsumexp(ll_k + logw[None, None, :], axis=2)
    return out


def waic_lpml(ll):
    """WAIC (deviance scale) and LPML from a (draws x subjects) log-lik matrix."""
    S = ll.shape[0]
    lppd = special.logsumexp(ll, axis=0) - math.log(S)
    p_waic = ll.var(axis=0, ddof=1)
    waic = -2.0 * (lppd.sum() - p_waic.sum())
    log_cpo = -(special.logsumexp(-ll, axis=0) - math.log(S))
    return dict(waic=waic, p_waic=p_waic.sum(), lpml=log_cpo.sum(), log_cpo=log_cpo)


# ----------------------------------------------------------------------------
# Output analysis: batch means, multivariate ESS, minimum ESS
# ----------------------------------------------------------------------------
BATCH_EXPONENT = 0.6   # b_n = floor(n^0.6): strong consistency for geometrically ergodic chains with
                       # all moments requires b_n = n^nu with nu > 1/2 (Banerjee and Vats, 2024, Section 3.1)


def batch_means_cov(x, b=None):
    """Multivariate batch-means estimate of the asymptotic covariance Sigma (batch size n^0.6 by default)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] == 1:
        x = x.T
    n = x.shape[0]
    b = int(np.floor(n ** BATCH_EXPONENT)) if b is None else b
    a = n // b
    xb = x[: a * b].reshape(a, b, -1).mean(axis=1)
    mu = x[: a * b].mean(axis=0)
    d = xb - mu
    return b * (d.T @ d) / (a - 1)


def mcse(x, b=None):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] == 1:
        x = x.T
    return np.sqrt(np.diag(batch_means_cov(x, b)) / x.shape[0])


def multi_ess(x, b=None):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] == 1:
        x = x.T
    n, p = x.shape
    lam = np.cov(x, rowvar=False)
    sig = batch_means_cov(x, b)
    sign1, ld1 = np.linalg.slogdet(np.atleast_2d(lam))
    sign2, ld2 = np.linalg.slogdet(np.atleast_2d(sig))
    return n * math.exp((ld1 - ld2) / p)


def min_ess(p, alpha=0.05, eps=0.05):
    """Minimum ESS of Vats, Flegal and Jones (2019) for relative precision eps."""
    crit = stats.chi2.ppf(1 - alpha, p)
    logv = (2.0 / p) * math.log(2.0) + math.log(math.pi) - (2.0 / p) * (math.log(p) + special.gammaln(p / 2.0))
    return math.exp(logv) * crit / eps ** 2


def summarize(fit: Fit):
    """Monitored vector g of the paper: (beta, log sigma_e, log sigma_u), with
    sigma_e = lam_e^{-1/2} and sigma_u = lam_u^{-1/2} (one column per group if
    scale_groups was used)."""
    return np.column_stack([fit.beta, -0.5 * np.log(fit.lam_e), -0.5 * np.log(fit.lam_u)])
