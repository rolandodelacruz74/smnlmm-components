"""Block (shared-weight) condition for multivariate Student-t random intercepts and slopes with known shape Psi,
beta-HCG design: s_b(c) = n^{-c} sum_{b} ||Pi_{b b'}||_F^{2c}, R^{(n)}(c) = Gamma(c+n/2)Gamma((d+n)/2-c)/(Gamma(n/2)Gamma((d+n)/2)).
Condition: R^{(2)}(c) max_b s_b(c) < 1. Reports the smallest d on a fine grid for the best c."""
import numpy as np
from scipy import special, linalg
from data import load_bhcg
y,X,Z1,inv,gs,_=load_bhcg(); t=X[:,1]; S=Z1.shape[1]; n=2
def cond(Psi):
    Pm=linalg.sqrtm(np.linalg.inv(Psi)).real          # u_tilde = Psi^{1/2} b  =>  Z_tilde = Z (I kron Psi^{-1/2})
    Zs=np.zeros((len(y),2*S))
    for i in range(S):
        m=inv==i; Zi=np.column_stack([np.ones(m.sum()),t[m]]); Zs[np.ix_(m,[2*i,2*i+1])]=Zi@Pm
    U,sv,Vt=np.linalg.svd(Zs,full_matrices=True); r=(sv>1e-9*sv[0]).sum(); N0=Vt[r:].T; Pi=N0@N0.T
    F=(Pi**2).reshape(S,2,S,2).sum(axis=(1,3))         # F[b,b'] = ||Pi_{bb'}||_F^2
    return r,F
def thresh(F,cs=np.linspace(0.02,1,50),ds=np.arange(0.05,6,0.01)):
    best=(np.inf,None)
    for c in cs:
        s=(n**-c*(F**c).sum(0)).max()
        for d in ds:
            if c<(d+n)/2:
                R=np.exp(special.gammaln(c+n/2)+special.gammaln((d+n)/2-c)-special.gammaln(n/2)-special.gammaln((d+n)/2))
                if R*s<1: 
                    if d<best[0]: best=(d,c,s)
                    break
    return best
for name,Psi in [('Psi=I',np.eye(2)),('Psi=diag(1,27.4)',np.diag([1,27.4]))]:
    r,F=cond(Psi); s1=(F.sum(0)/2).max()
    print(name,'rank',r,'max s_b(1)',round(s1,3),'threshold d (best c)',thresh(F))
