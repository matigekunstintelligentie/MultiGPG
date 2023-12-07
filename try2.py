from pygpg.sk import GPGRegressor as GPGR
import sympy as sp

hyper_params = [
    { # 2
     'd' : (4,), 'optimiser_choice' : ("lm", "bfgs",),
    },
    { # 2
     'd' : (5,), 'optimiser_choice' : ("lm", "bfgs",),
    },
    { # 1
     'd' : (6,), 'optimiser_choice' : ("lm","bfgs",),
    },
    
]

est = GPGR(t=600, g=-1, e=499500, tour=4, d=4, use_optim=True, optimiser_choice="lm",
        disable_ims=True, pop=1024, feat_sel=-1, no_univ_exc_leaves_fos=False, bs_opt=256,
        no_large_fos=False, 
        finetune=True,
        ff="lsmse",
        use_ftol=False,
        tol=1e-9,
        use_mse_opt=False,
        bs=2048,
        cmp=0.,
        log=False,
        fset='+,-,*,/', rci=0.0,
        random_state=1,
        verbose=True
        )

est = GPGR(t=120, g=-1, e=-1, tour=4, d=4,  use_optim=True, optimiser_choice="lm",
        disable_ims=True, pop=1024, nolink=False, feat_sel=-1,
        no_large_fos=False, no_univ_exc_leaves_fos=False,
        finetune=True, bs_opt=256,
        bs=2048, random_accept_p=0.,
        verbose=False, opt_per_gen=1, use_clip=False,
        fset='+,-,*,/,sin,cos,log,exp,sqrt', cmp=0.0, rci=0.0, use_ftol=False, tol=1e-9, use_mse_opt=False, log=False, reinject_elite=True, ff="lsmse", use_local_search=False,
        optimise_after=False, add_addition_multiplication=False, add_any=False, use_max_range=False, equal_p_coeffs=True,

        random_state=1)  

def complexity(est):
  m = est.model
  c = 0
  for _ in sp.preorder_traversal(m):
    c+=1
  return c

def model(est):
    return str(est.model)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#df = pd.read_csv("../pmlb/datasets/618_fri_c3_1000_50/618_fri_c3_1000_50.tsv.gz", compression='gzip', sep='\t')
df = pd.read_csv("dataset/boston.csv")
X = df.drop(columns=['target']).to_numpy()
y = df['target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

est.fit(X_train, y_train)
p = np.mean((est.predict(X_train)-y_train)**2)
print(p)
