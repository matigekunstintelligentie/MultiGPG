from pymgpg.sk import MGPGRegressor as MGPGR
import sympy as sp

hyper_params = [
    { # 2
     'd' : (4,),
    },
]

est = MGPGR(t=2*60*60, g=100, e=-1, max_non_improve=-1, tour=4, d=4,use_optim=True, pop=100, bs_opt=256, bs=2048,
verbose=True, opt_per_gen=1, use_clip=False,fset='+,-,*,/,sin,cos,log,sqrt',cmp=1.,use_mse_opt=False,log=False,ff='lsmse',use_max_range=True,equal_p_coeffs=True,MO_mode=True,use_adf=True,use_aro=False,
n_clusters=5,max_coeffs=-1,discount_size=False,random_state=0,nr_multi_trees=4,balanced=False,donor_fraction=1.,accept_diversity=True,k2=True)

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
