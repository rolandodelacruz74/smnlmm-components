"""Paper 2 application (beta-HCG, random intercepts and slopes): 4 chains per model, output analysis
with batch size n^0.6 and split R-hat, subject-level WAIC/LPML with quadrature over the latent weights."""
import json, sys, math, time, numpy as np
from scipy import special
from data import load_bhcg
from smnlmm import Normal, StudentT, mcse, multi_ess, min_ess, waic_lpml
from smnlmm2 import GridMixing, Priors2, gibbs2
y,X,Z1,inv,gs,d=load_bhcg(); t=X[:,1]; grp=gs.astype(int); N=len(y); S=Z1.shape[1]
Dgrid=[2,2.5,3,4,5,6,8,10,15,20,30,50,100]
def setup(m):
    if m in ('M1','M2'):
        Z=np.hstack([Z1,Z1*t[:,None]]); comp=np.r_[grp,2+grp]
        re=[Normal()]*4 if m=='M1' else [GridMixing('t',Dgrid) for _ in range(4)]
    else:
        Z=Z1; comp=grp; re=[Normal()]*2 if m=='M0' else [GridMixing('t',Dgrid) for _ in range(2)]
    return Z,comp,re
# --- quadrature for Gamma(a,a) weights, absorbing eta^{1/2}: E[g] = sum w_k g(x_k)
def gamma_nodes(dd,n=40):
    a=dd/2.0; x,w=special.roots_genlaguerre(n,a-0.5)     # weight x^{a-1/2} e^{-x}
    eta=x/a; w=w/np.sqrt(x)*np.exp(-special.gammaln(a))  # back to x^{a-1}e^{-x}/Gamma(a)
    return eta,w
NODES={dd:gamma_nodes(dd) for dd in Dgrid}
# subject summaries
def subj_stats(B):
    R=y-X@B
    rr=np.bincount(inv,R*R,minlength=S); r1=np.bincount(inv,R,minlength=S); rt=np.bincount(inv,R*t,minlength=S)
    return rr,r1,rt
n_i=np.bincount(inv,minlength=S).astype(float); st=np.bincount(inv,t,minlength=S); stt=np.bincount(inv,t*t,minlength=S)
def loglik_draw(m,B,le,lu,dk):
    rr,r1,rt=subj_stats(B); s2=1/le
    if m in ('M0','M3'):
        lam=lu[grp]
        if m=='M0': e=np.ones((1,)); w=np.ones(1); E=np.tile(e,(S,1)); Wt=np.tile(w,(S,1))
        else:
            E=np.stack([NODES[dk[g]][0] for g in grp]); Wt=np.stack([NODES[dk[g]][1] for g in grp])
        tau2=1/(lam[:,None]*E); den=s2+n_i[:,None]*tau2
        quad=(rr[:,None]-tau2*r1[:,None]**2/den)/s2
        ll=-0.5*(n_i[:,None]*math.log(2*math.pi)+(n_i[:,None]-1)*math.log(s2)+np.log(den)+quad)
        return special.logsumexp(ll,b=Wt,axis=1)
    # intercept + slope: components (int,g)=g, (slope,g)=2+g
    l1=lu[grp]; l2=lu[2+grp]
    if m=='M1':
        E1=np.ones((S,1)); W1=np.ones((S,1)); E2=E1; W2=W1
    else:
        E1=np.stack([NODES[dk[g]][0] for g in grp]); W1=np.stack([NODES[dk[g]][1] for g in grp])
        E2=np.stack([NODES[dk[2+g]][0] for g in grp]); W2=np.stack([NODES[dk[2+g]][1] for g in grp])
    a=1/(l1[:,None,None]*E1[:,:,None]); b=1/(l2[:,None,None]*E2[:,None,:])     # variances of intercept, slope
    # M = D^{-1} + Z'Z/s2 ;  D = diag(a,b)
    m11=1/a+n_i[:,None,None]/s2; m22=1/b+stt[:,None,None]/s2; m12=st[:,None,None]/s2
    det=m11*m22-m12**2
    v1=r1[:,None,None]/s2; v2=rt[:,None,None]/s2
    qf=(m22*v1*v1-2*m12*v1*v2+m11*v2*v2)/det
    logdetV=n_i[:,None,None]*math.log(s2)+np.log(a)+np.log(b)+np.log(det)
    ll=-0.5*(n_i[:,None,None]*math.log(2*math.pi)+logdetV+rr[:,None,None]/s2-qf)
    Wt=W1[:,:,None]*W2[:,None,:]
    return special.logsumexp(ll.reshape(S,-1),b=Wt.reshape(S,-1),axis=1)
def split_rhat(chains):
    C=[c for ch in chains for c in np.array_split(ch,2)]; n=min(len(c) for c in C); C=np.stack([c[:n] for c in C])
    m=C.mean(1); W=C.var(1,ddof=1).mean(0); B=n*m.var(0,ddof=1); return np.sqrt(((n-1)/n*W+B/n)/W)
if __name__=='__main__':
    m=sys.argv[1]; NIT=int(sys.argv[2]); BURN=5000; SCALED=len(sys.argv)>3 and sys.argv[3]=='scaled'
    BK=0.1 if (len(sys.argv)<=3 or SCALED) else float(sys.argv[3]); TAG='_scaled' if SCALED else ('' if BK==0.1 else f'_b{BK:g}')
    Z,comp,re=setup(m); K=comp.max()+1; q=Z.shape[1]
    bk=BK*np.ones(K)
    if SCALED and K==4: bk[2:]=BK/np.mean(t**2)   # slope precisions: Ga(1,0.1) on the precision of t_rms * slope
    pr=Priors2(np.zeros(5),1000*np.eye(5),1.0,BK,np.ones(K),bk)
    G=[]; fits=[]
    for c in range(4):
        rng=np.random.default_rng(700+c)
        init=(np.linalg.lstsq(X,y,rcond=None)[0]+rng.normal(scale=0.5,size=5),rng.normal(scale=0.5,size=q),np.exp(rng.normal(size=q)),np.ones(N))
        f=gibbs2(y,X,Z,comp,re,pr,n_iter=NIT,burn=BURN,seed=800+c,init=init); fits.append(f)
        G.append(np.column_stack([f.beta,-0.5*np.log(f.lam_e),-0.5*np.log(f.lam_u)]))
        print(m,'chain',c,round(f.seconds),'s',flush=True)
    np.savez_compressed(f'../results/p2_{m}{TAG}_draws.npz',beta=np.stack([f.beta[::10] for f in fits]),lam_e=np.stack([f.lam_e[::10] for f in fits]),
        lam_u=np.stack([f.lam_u[::10] for f in fits]),tail_u=np.stack([f.tail_u[::10] for f in fits]),
        mess_n=[multi_ess(gg)/len(gg) for gg in G],mcse_max=max(mcse(gg).max() for gg in G),rhat=split_rhat(G))
    g0=G[0]; n0=len(g0); p=g0.shape[1]
    Gall=np.concatenate(G); beta=np.concatenate([f.beta for f in fits]); b4=beta[:,3]
    thin=max(1,len(Gall)//2000); LL=[]
    for f in fits:
        for j in range(0,len(f.lam_e),thin*1):
            pass
    idx=[(ci,j) for ci,f in enumerate(fits) for j in range(0,len(f.lam_e),thin)]
    t0=time.time()
    for ci,j in idx:
        f=fits[ci]; dk={k:(f.tail_u[j,k] if not np.isnan(f.tail_u[j,k]) else None) for k in range(K)}
        LL.append(loglik_draw(m,f.beta[j],f.lam_e[j],f.lam_u[j],dk))
    LL=np.array(LL); ic=waic_lpml(LL); S_=LL.shape[0]
    lppd=special.logsumexp(LL,axis=0)-math.log(S_); pw=LL.var(axis=0,ddof=1)
    res=dict(model=m,b_prior=bk.tolist(),nit=NIT,burn=BURN,waic=ic['waic'],p_waic=ic['p_waic'],lpml=ic['lpml'],elpd_i=(lppd-pw).tolist(),
             mess_n=[multi_ess(gg)/len(gg) for gg in G],mcse_max=float(max(mcse(gg).max() for gg in G)),rhat=split_rhat(G).tolist(),
             minESS=min_ess(p),b4=float(b4.mean()),b4_ci=np.quantile(b4,[.025,.975]).tolist(),p_b4=float((b4>0).mean()),
             diff4=[float((beta[:,3]+4*beta[:,4]).mean())]+np.quantile(beta[:,3]+4*beta[:,4],[.025,.975]).tolist(),
             sig_u=np.mean([(f.lam_u**-0.5).mean(0) for f in fits],0).tolist(),seconds=[f.seconds for f in fits])
    if m in ('M2','M3'):
        T=np.concatenate([f.tail_u for f in fits]); res['d_post']={str(k):{str(v):float((T[:,k]==v).mean()) for v in Dgrid} for k in range(K)}
    json.dump(res,open(f'../results/p2_{m}{TAG}.json','w'))
    print(m,'WAIC',round(ic['waic'],2),'LPML',round(ic['lpml'],2),'mESS/n',np.round(res['mess_n'],2),'maxRhat',round(max(res['rhat']),4),'b4',round(res['b4'],3),'ll s',round(time.time()-t0),flush=True)
