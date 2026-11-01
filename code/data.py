import numpy as np, pandas as pd, os
HERE=os.path.dirname(os.path.abspath(__file__))
def load_bhcg():
    d=pd.read_csv(os.path.join(HERE,'..','data','bHCG.txt'),sep=r'\s+')
    y=d.Y.to_numpy(float); t=d.Time7.to_numpy(float); g=d.Grupo.to_numpy(float)
    X=np.column_stack([np.ones_like(t),t,t**2,g,g*t])
    labels,inv=np.unique(d.Sujeto.to_numpy(),return_inverse=True)
    Z=np.eye(labels.size)[inv]
    grp_subject=d.groupby('Sujeto').Grupo.first().to_numpy()
    return y,X,Z,inv,grp_subject,d
