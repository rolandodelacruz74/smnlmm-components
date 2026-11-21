"""Tests for GridMixing: normalising constants against quadrature and the joint draw of (d, eta) given s."""
import numpy as np, math
from scipy import integrate, stats
from smnlmm import StudentT, Slash
from smnlmm2 import GridMixing
rng = np.random.default_rng(5)
worst = 0.0
for fam, vals in [('t', [0.5, 1, 2.5, 4, 10, 50]), ('slash', [0.3, 0.8, 1.5, 3, 6])]:
    g = GridMixing(fam, vals)
    for v in vals:
        h = (lambda e, v=v: stats.gamma.pdf(e, v/2, scale=2/v)) if fam == 't' else (lambda e, v=v: v*e**(v-1))
        hi = np.inf if fam == 't' else 1.0
        for s in [1e-12, 1e-3, 0.3, 2.0, 25.0]:
            num = integrate.quad(lambda e: e**0.5*math.exp(-s*e)*h(e), 0, hi, limit=400, epsabs=0, epsrel=1e-12)[0]
            err = abs(g.logZ(v, np.array([s]))[0] - math.log(num))
            worst = max(worst, err)
print(f"logZ: max abs error {worst:.2e}")
assert worst < 1e-7
# joint draw: p(d | s) and E[eta | s] under the joint law
for fam, vals in [('t', [1, 3, 8]), ('slash', [0.5, 1.5, 4])]:
    g = GridMixing(fam, vals, prior=[0.2, 0.5, 0.3])
    s = np.array([0.05, 0.4, 1.3, 3.0])
    lp = np.array([g.logprior[k] + g.logZ(v, s).sum() for k, v in enumerate(vals)])
    pex = np.exp(lp - lp.max()); pex /= pex.sum()
    R = 40000
    D = np.empty(R); E = np.empty((R, len(s)))
    for r in range(R):
        D[r], E[r] = g.sample_block(rng, s)
    phat = np.array([(D == v).mean() for v in vals])
    z = (phat - pex) / np.sqrt(pex*(1-pex)/R)
    # E[eta_1 | s] = sum_d p(d|s) E_d,s[eta_1]
    m_ex = sum(pex[k]*integrate.quad(lambda e, v=v: e**1.5*math.exp(-s[0]*e)*((stats.gamma.pdf(e, v/2, scale=2/v)) if fam=='t' else v*e**(v-1)), 0, np.inf if fam=='t' else 1)[0]
               / math.exp(g.logZ(v, s[:1])[0]) for k, v in enumerate(vals))
    zm = (E[:, 0].mean() - m_ex)/(E[:, 0].std()/math.sqrt(R))
    print(fam, "p(d|s) exact", np.round(pex, 4), "empirical", np.round(phat, 4), "z", np.round(z, 2), "| E eta_1 z", round(zm, 2))
    assert np.all(np.abs(z) < 4) and abs(zm) < 4
print("test_grid: OK")
