"""Prior sensitivity for the slope scales in M1: Ga(1, b) on the precisions, b in {0.1, 0.01, 0.001}."""
import sys, json, numpy as np, math
from scipy import special
import p2_app as A
from smnlmm import waic_lpml
from smnlmm2 import Priors2, gibbs2
out={}
for bk in [0.1,0.01,0.001]:
    Z,comp,re=A.setup('M1'); K=4; q=Z.shape[1]
    pr=Priors2(np.zeros(5),1000*np.eye(5),1.0,0.1,np.ones(K),np.array([0.1,0.1,bk,bk]))
    f=gibbs2(A.y,A.X,Z,comp,re,pr,n_iter=15000,burn=3000,seed=901)
    LL=np.array([A.loglik_draw('M1',f.beta[j],f.lam_e[j],f.lam_u[j],{}) for j in range(0,len(f.lam_e),6)])
    ic=waic_lpml(LL); sd=(f.lam_u**-0.5)
    out[bk]=dict(waic=ic['waic'],lpml=ic['lpml'],sd_med=np.median(sd,0).tolist(),sd_q=np.quantile(sd,[.025,.975],axis=0).tolist())
    print(bk,round(ic['waic'],2),np.round(np.median(sd,0),3),np.round(np.quantile(sd,.025,axis=0),3),flush=True)
json.dump({str(k):v for k,v in out.items()},open('../results/p2_prior_sens.json','w'))
