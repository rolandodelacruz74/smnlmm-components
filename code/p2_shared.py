"""Shared-weight random intercepts and slopes (Corollary on multivariate scale mixtures): y_ij = b0 + b1 t_ij + v_i0 + v_i1 t_ij + e_ij,
v_i | lam, eta_i, d ~ N_2(0, (lam eta_i)^{-1} I), eta_i ~ Gamma(d/2, d/2), d uniform on a finite grid including d < 1.
Some subjects have a single observation, so Z is rank deficient. Exact three-block Gibbs sampler:
lambda | (eta, theta) -> (d, eta) | (lambda, theta) -> theta | (lambda, eta).
Reports split R-hat, ESS per iteration, mESS of g = (b0, b1, log sigma_e, log sigma_u), the posterior of d and Monte Carlo
standard errors of Pr(d = 0.5 | y), from four chains with dispersed starting values."""
import sys, json, math, numpy as np
from scipy import special
from smnlmm import batch_means_cov, multi_ess, min_ess
D=np.array([0.5,1,2,4,8,16]); n0=2
m=60; rng=np.random.default_rng(11)
ni=np.r_[np.ones(15,int),rng.integers(2,7,size=m-15)]; subj=np.repeat(np.arange(m),ni); N=subj.size
t=rng.uniform(0,1,size=N); X=np.column_stack([np.ones(N),t]); p=2
Z=np.zeros((N,2*m)); Z[np.arange(N),2*subj]=1; Z[np.arange(N),2*subj+1]=t; q=2*m
d_true=1.0; beta0=np.array([1.0,0.5]); sig_e=0.3; sig_u=0.5
eta0=rng.gamma(d_true/2,2/d_true,size=m); v0=(rng.normal(size=(m,2))*sig_u/np.sqrt(eta0)[:,None]).ravel()
y=X@beta0+Z@v0+sig_e*rng.normal(size=N)
r=np.linalg.matrix_rank(Z)
a_e,b_e,a_u,b_u=1.0,0.1,1.0,0.1; Sb_inv=np.eye(p)/1000
W=np.hstack([X,Z]); WtW=W.T@W; Wty=W.T@y
def logZ(dd,s):  # log int eta^{n0/2} e^{-s eta} Ga(d/2,d/2)(d eta)
    return (dd/2)*np.log(dd/2)-special.gammaln(dd/2)+special.gammaln((dd+n0)/2)-((dd+n0)/2)*np.log(dd/2+s)
def run(seed,n_iter,burn,init):
    rg=np.random.default_rng(seed); beta,v,eta=init; out=[]; dd_out=[]
    for it in range(n_iter):
        res=y-X@beta-Z@v
        le=rg.gamma(a_e+N/2,1/(b_e+res@res/2))
        V=v.reshape(m,2); ss=(V**2).sum(1)
        lu=rg.gamma(a_u+q/2,1/(b_u+(eta*ss).sum()/2))
        s=lu*ss/2
        lp=np.array([logZ(dd,s).sum() for dd in D]); pr=np.exp(lp-lp.max()); pr/=pr.sum()
        dd=D[rg.choice(D.size,p=pr)]
        eta=rg.gamma((dd+n0)/2,1/(dd/2+s))
        P=le*WtW; P[:p,:p]+=Sb_inv; P[p:,p:]+=np.diag(lu*np.repeat(eta,2))
        L=np.linalg.cholesky(P); mu=np.linalg.solve(L.T,np.linalg.solve(L,le*Wty))
        th=mu+np.linalg.solve(L.T,rg.standard_normal(p+q)); beta,v=th[:p],th[p:]
        if it>=burn: out.append([beta[0],beta[1],-0.5*np.log(le),-0.5*np.log(lu)]); dd_out.append(dd)
    return np.array(out),np.array(dd_out)
if __name__=='__main__':
    n_iter=int(sys.argv[1]) if len(sys.argv)>1 else 55000; burn=5000
    G=[];Dd=[]
    for c in range(4):
        rg=np.random.default_rng(500+c)
        init=(rg.normal(scale=2,size=p),rg.normal(scale=2,size=q),rg.gamma(1,1,size=m))
        g,dd=run(900+c,n_iter,burn,init); G.append(g); Dd.append(dd); print('chain',c,'done',flush=True)
    G=np.array(G); Dd=np.array(Dd); n=G.shape[1]
    C=np.concatenate([np.array_split(x,2) for x in G]); L=min(len(x) for x in C); C=np.stack([x[:L] for x in C])
    Wv=C.var(1,ddof=1).mean(0); B=L*C.mean(1).var(0,ddof=1); rhat=np.sqrt(((L-1)/L*Wv+B/L)/Wv)
    ess=np.mean([np.var(G[c],0,ddof=1)/np.diag(batch_means_cov(G[c])) for c in range(4)],0)
    mess=[multi_ess(G[c])/n for c in range(4)]
    I=(Dd==0.5).astype(float); pI=I.mean(); vI=np.mean([batch_means_cov(I[c][:,None])[0,0] for c in range(4)])
    out=dict(N=int(N),q=q,rank=int(r),n_single=15,d_true=d_true,grid=D.tolist(),rhat=rhat.tolist(),ess=ess.tolist(),mess=mess,
             minESS=min_ess(4),iters_needed=float(min_ess(4)/np.mean(mess)),
             d_post={str(x):float((Dd==x).mean()) for x in D},p_d_eq05=float(pI),mcse_p_d_eq05=float(np.sqrt(vI/(4*n))),
             ess_ind=float(I.var()/vI) if vI>0 else None,
             post_mean=G.reshape(-1,4).mean(0).tolist(),mcse=np.sqrt(np.mean([np.diag(batch_means_cov(G[c])) for c in range(4)],0)/(4*n)).tolist())
    print(json.dumps(out,indent=1)); json.dump(out,open('../results/p2_shared.json','w'))
