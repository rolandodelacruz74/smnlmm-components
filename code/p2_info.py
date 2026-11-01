"""Exploration of the scale and of the tail parameter for shared-weight random intercepts and slopes, as the information
per subject varies. Model as in p2_shared.py (m=60 subjects, d uniform on {0.5,1,2,4,8,16}); the number of subjects with a
single observation is n1 in {0, 15, 30} and the true degrees of freedom are d0 in {1, 4}; three data sets per configuration,
two chains of 55,000 iterations (5,000 burn-in) per data set. Reports, averaged over data sets, the ESS per iteration of
beta_1, beta_2, log sigma_e, log sigma_u and of the indicator 1{d <= 1}, the mESS per iteration of g, the largest split
R-hat, Pr(d <= 1 | y) and its Monte Carlo standard error.
Usage: python3 p2_info.py N1 D0"""
import sys, json, numpy as np
from scipy import special
from smnlmm import batch_means_cov, multi_ess, min_ess
D=np.array([0.5,1,2,4,8,16]); n0=2; m=60
a_e,b_e,a_u,b_u=1.0,0.1,1.0,0.1
def make(n1,d0,seed):
    rng=np.random.default_rng(seed)
    ni=np.r_[np.ones(n1,int),rng.integers(2,7,size=m-n1)]; subj=np.repeat(np.arange(m),ni); N=subj.size
    t=rng.uniform(0,1,size=N); X=np.column_stack([np.ones(N),t])
    Z=np.zeros((N,2*m)); Z[np.arange(N),2*subj]=1; Z[np.arange(N),2*subj+1]=t
    eta=rng.gamma(d0/2,2/d0,size=m); v=(rng.normal(size=(m,2))*0.5/np.sqrt(eta)[:,None]).ravel()
    y=X@np.array([1.0,0.5])+Z@v+0.3*rng.normal(size=N)
    return y,X,Z
def logZ(dd,s): return (dd/2)*np.log(dd/2)-special.gammaln(dd/2)+special.gammaln((dd+n0)/2)-((dd+n0)/2)*np.log(dd/2+s)
def run(y,X,Z,seed,n_iter,burn):
    N,p=X.shape; q=Z.shape[1]; W=np.hstack([X,Z]); WtW=W.T@W; Wty=W.T@y; Sb=np.eye(p)/1000
    rg=np.random.default_rng(seed); beta=rg.normal(scale=2,size=p); v=rg.normal(scale=2,size=q); eta=rg.gamma(1,1,size=m)
    G=[]; Dd=[]
    for it in range(n_iter):
        res=y-X@beta-Z@v; le=rg.gamma(a_e+N/2,1/(b_e+res@res/2))
        ss=(v.reshape(m,2)**2).sum(1); lu=rg.gamma(a_u+q/2,1/(b_u+(eta*ss).sum()/2)); s=lu*ss/2
        lp=np.array([logZ(dd,s).sum() for dd in D]); pr=np.exp(lp-lp.max()); pr/=pr.sum(); dd=D[rg.choice(D.size,p=pr)]
        eta=rg.gamma((dd+n0)/2,1/(dd/2+s))
        P=le*WtW; P[:p,:p]+=Sb; P[p:,p:]+=np.diag(lu*np.repeat(eta,2))
        L=np.linalg.cholesky(P); mu=np.linalg.solve(L.T,np.linalg.solve(L,le*Wty)); th=mu+np.linalg.solve(L.T,rg.standard_normal(p+q))
        beta,v=th[:p],th[p:]
        if it>=burn: G.append([beta[0],beta[1],-0.5*np.log(le),-0.5*np.log(lu)]); Dd.append(dd)
    return np.array(G),np.array(Dd)
if __name__=='__main__':
    n1=int(sys.argv[1]); d0=float(sys.argv[2]); n_iter=55000; burn=5000; rows=[]
    for rep in range(3):
        y,X,Z=make(n1,d0,1000*n1+100*int(d0)+rep); r=int(np.linalg.matrix_rank(Z))
        out=[run(y,X,Z,7000+10*rep+c,n_iter,burn) for c in range(2)]
        G=np.array([o[0] for o in out]); I=np.array([(o[1]<=1).astype(float) for o in out]); n=G.shape[1]
        C=np.concatenate([np.array_split(x,2) for x in G]); C=np.stack(C); W=C.var(1,ddof=1).mean(0); B=C.shape[1]*C.mean(1).var(0,ddof=1)
        rhat=np.sqrt(((C.shape[1]-1)/C.shape[1]*W+B/C.shape[1])/W)
        ess=np.mean([G[c].var(0,ddof=1)/np.diag(batch_means_cov(G[c])) for c in range(2)],0)
        vI=np.mean([batch_means_cov(I[c][:,None])[0,0] for c in range(2)]); essI=I.var()/vI if vI>0 else float('nan')
        rows.append(dict(rank=r,q=Z.shape[1],ess=ess.tolist(),essI=float(essI),mess=float(np.mean([multi_ess(G[c])/n for c in range(2)])),
                         rhat=float(rhat.max()),p=float(I.mean()),mcse=float(np.sqrt(vI/(2*n)))))
        print(n1,d0,rep,rows[-1],flush=True)
    json.dump(dict(n1=n1,d0=d0,rows=rows),open(f'../results/p2_info_{n1}_{d0:g}.json','w'))
