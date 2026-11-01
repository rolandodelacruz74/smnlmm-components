import sys, numpy as np
from smnlmm import *; from smnlmm2 import *
case=sys.argv[1]; M=int(sys.argv[2]); rng=np.random.default_rng(int(sys.argv[3]) if len(sys.argv)>3 else 11)
N=9; g=np.array([0,0,0,1,1,2,2,3,3]); t=np.linspace(0.2,1.5,N)
X=np.column_stack([np.ones(N),t])
if case=='multi':      # intercepts + slopes for 4 subjects; subject-level design rank deficient? make one subject single obs
    g=np.array([0,0,0,1,1,2,2,3,3]); Z1=np.eye(4)[g]; Z2=Z1*t[:,None]; Z=np.hstack([Z1,Z2]); comp=np.r_[np.zeros(4,int),np.ones(4,int)]
    Z=Z.copy(); Z[:,3]=0; Z[7,3]=1; Z[7,7]=t[7]; Z[8,3]=0; Z[8,7]=0   # subject 3: single observation -> rank deficient
    re=[GridMixing('t',[2.5,4,8]),StudentT(3)]; err=None
elif case=='err':
    Z=np.eye(4)[g]; comp=np.zeros(4,int); re=[StudentT(3)]; err=ContNormal(0.2,0.1)
elif case=='sl08':
    Z=np.eye(4)[g]; comp=np.zeros(4,int); re=[Slash(0.8)]; err=None
elif case=='gridlight':
    Z=np.eye(4)[g]; comp=np.zeros(4,int); re=[GridMixing('slash',[1.5,3,6])]; err=GridMixing('t',[2.5,5,10])
elif case=='grid':
    Z=np.eye(4)[g]; comp=np.zeros(4,int); re=[GridMixing('slash',[0.8,1.5,3])]; err=GridMixing('t',[2.5,5,10])
K=comp.max()+1; q=Z.shape[1]
pr=Priors2(np.zeros(2),np.eye(2),3.0,2.0,3.0,2.0)
def draw_prior():
    b=rng.normal(size=2); le=rng.gamma(3,1/2); lu=rng.gamma(3,1/2,size=K)
    eta=np.empty(q); tails=[]
    for k in range(K):
        m=comp==k; mix=re[k]
        if isinstance(mix,GridMixing): v,e=mix.sample_prior(rng,m.sum()); tails.append(v)
        else: e=mix.sample_prior(rng,m.sum()); tails.append(np.nan)
        eta[m]=e
    u=rng.normal(size=q)/np.sqrt(lu[comp]*eta)
    if err is None: om=np.ones(N); nu=np.nan
    elif isinstance(err,GridMixing): nu,om=err.sample_prior(rng,N)
    else: om=err.sample_prior(rng,N); nu=np.nan
    return b,u,eta,om,le,lu,tails,nu
def ydraw(b,u,le,om): return X@b+Z@u+rng.normal(size=N)/np.sqrt(le*om)
def stats_(b,u,eta,om,le,lu,tails,nu):
    return [b[0],b[1],np.abs(u[0])**.5,np.abs(u[-1])**.5,le,lu[0],lu[-1],np.log(eta[0]),np.log(om[0]),
            tails[0] if not np.isnan(tails[0]) else 0., nu if not np.isnan(nu) else 0.]
F=np.array([stats_(*draw_prior()) for _ in range(M)])
st=draw_prior(); b,u,eta,om=st[0],st[1],st[2],st[3]; y=ydraw(b,u,st[4],om)
S=[]
for k in range(M):
    f=gibbs2(y,X,Z,comp,re,pr,err_mix=err,n_iter=1,burn=0,seed=int(rng.integers(1<<31)),init=(b,u,eta,om),keep=['u','eta','omega'])
    b=f.beta[-1]; u=f.extra['u'][-1]; eta=f.extra['eta'][-1]; om=f.extra['omega'][-1]
    le=f.lam_e[-1]; lu=f.lam_u[-1]; tails=list(f.tail_u[-1]); nu=f.tail_e[-1]
    y=ydraw(b,u,le,om); S.append(stats_(b,u,eta,om,le,lu,tails,nu))
S=np.array(S)
names=['b0','b1','|u0|^.5','|ulast|^.5','lam_e','lam_u0','lam_uK','log eta0','log om0','tail_u0','nu']
bad=0
for j,n in enumerate(names):
    a,fw=S[:,j],F[:,j]
    if np.allclose(a,a[0]) and np.allclose(fw,fw[0]): continue
    se=np.sqrt(mcse(a)[0]**2+fw.var()/fw.size); z=(a.mean()-fw.mean())/se; bad+=abs(z)>4
    print(f"{case:5s} {n:10s} gibbs={a.mean():8.4f} fwd={fw.mean():8.4f} z={z:6.2f}")
print(case,"FLAGS",bad)
