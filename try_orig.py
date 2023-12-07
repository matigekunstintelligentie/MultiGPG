import numpy as np
from pygpg.sk import GPGRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize, least_squares

np.random.seed(42)


X = np.random.randn(24, 3)*10

def grav_law(X):
  return 6.67 * X[:,0]*X[:,1]/(np.square(X[:,2]))

y = grav_law(X)

#from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split

#X, y = load_boston(return_X_y=True)


"""
Load 618_fri_c3_1000_50
"""

hyper_params = [
    { # 1
     'd' : (3,), 'rci' : (0.0,),
    },
    { # 2
     'd' : (4,), 'rci' : (0.0, 0.1),
    },
    { # 2
     'd' : (5,), 'rci' : (0.0, 0.1,),
    },
    { # 1
     'd' : (6,), 'rci' : (0.1,),  'no_univ_exc_leaves_fos' : (True,),
    },
]

# load poker
import pandas as pd
#df = pd.read_csv("../pmlb/datasets/618_fri_c3_1000_50/618_fri_c3_1000_50.tsv.gz", compression='gzip', sep='\t')
df = pd.read_csv("dataset/boston.csv")
X = df.drop(columns=['target']).to_numpy()
y = df['target'].to_numpy()


from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import KFold
#
experiments = [(True,"bfgs","auto","bfgsgtol",1,False,1e-6,False),(True,"lm",256,"lmgtol",1,False,1e-6,False),(False,"-",256,"no_opt",1,False,1e-6,False),(True,"lm",256,"lmftol",1,True,1e-6,False),(True,"lm","auto","lmgtolauto",1,False,1e-6,False)]

def mse(x, y):
    return np.mean((y-x)**2)

best_train_r2 = []
best_test_r2 = []
for i in range(10):

    median_train_r2 = []
    median_train_mse = []
    median_test_r2 = []
    median_test_mse = []
    for exp_idx, experiment in enumerate(experiments):
        g = GPGRegressor(t=-1, g=100, e=-1, tour=4, d=5,  use_optim=experiment[0], optimiser_choice=experiment[1],
                disable_ims=True, pop=1024, nolink=False, feat_sel=-1,
                no_large_fos=False, no_univ_exc_leaves_fos=False,
                finetune=False, bs_opt=experiment[2],
                verbose=True, csv_file="./results/{}.csv".format(experiment[3]), opt_per_gen=experiment[4],
                fset='+,-,*,/,log,sqrt,sin,cos,exp,pow', cmp=0., rci=0.0, ff="lsmse", reinject_elite=True, use_ftol=experiment[5], tol=experiment[6], use_mse_opt=experiment[7], log=True, clip=False,
                random_state=i+1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i+1)

        use_ss=True


        if use_ss:
            s = SS()
            X_train = s.fit_transform(X_train)
            X_test = s.transform(X_test)
            y_train = s.fit_transform(y_train.reshape((-1,1)))
            y_test = s.transform(y_test.reshape((-1,1)))

            y_test = np.float32(y_test).squeeze(axis=1)
            y_train = np.float32(y_train).squeeze(axis=1)
        else:
            y_test = np.float32(y_test)
            y_train = np.float32(y_train)

        X_test = np.float32(X_test)
        X_train = np.float32(X_train)


        # x_0 = X_train[:,0]
        # x_1 = X_train[:,1]
        # x_2 = X_train[:,2]
        # x_3 = X_train[:,3]
        # x_4 = X_train[:,4]
        # x_5 = X_train[:,5]
        # x_6 = X_train[:,6]
        # x_7 = X_train[:,7]
        # x_8 = X_train[:,8]
        # x_9 = X_train[:,9]
        # x_10 = X_train[:,10]
        # x_11 = X_train[:,11]
        # x_12 = X_train[:,12]


        # origx = [0.694364]
        # x = origx

        # formula = lambda x: (0.128210+(-0.299964*(((x_12+(np.cos(x_5)+x_12))-(((x_12*x[0])**2)-((x_12*x_9)+(x_5*x_9))))-x_5)))
        # def rosen_bfgs(x):
        #     return ((y_train - formula(x))**2).mean()

        # print("Before bfgs", rosen_bfgs(x))
        
        # res = minimize(rosen_bfgs,x, method='BFGS', options={'maxiter':100, 'gtol':1e-30, 'disp': True})
        
        # x = res.x
        # print(x)
        # print("After bfgs", rosen_bfgs(x))
        
        # x = origx

        # def rosen(x):
        #     return y_train - formula(x)

        
        # res = least_squares(rosen,x,method="lm")
        
        # x = res.x
        # print(x)
        # print("After lm", rosen_bfgs(x))
        
        # quit()


        

        
        
        print('fit')
        g.fit(X_train, y_train)
        print('pred')
        p = g.predict(X_test)
        

        print(g.txt_models)
        #quit()
        #models = ['-0.709673+-0.887527*min(x_12-min(x_5,min(x_9+6.167003,x_0)**sin(4.327713+x_5)),sin(min(x_9,cos(cos(-2.138806))*min(x_3,x_1)+x_12)))', '-0.000000+-0.450763*x_12-x_5', '-0.299741+-0.602655*min(x_12,x_10-x_5)', '-0.708280+-0.883989*min(x_12-min(x_5,cos(x_9-2.318122)**x_0),sin(min(x_9,cos(cos(-2.128374))*min(x_3,x_1)+x_12)))', '-0.707625+-0.886489*min(x_12-min(x_5,cos(x_9-2.545202)**ln(x_0+x_5)),sin(min(x_9,cos(cos(-2.144274))*min(x_3,x_1)+x_12)))', '-0.699040+-0.882697*min(x_12-min(x_5,min(x_9,x_10)**x_0),sin(min(x_9,cos(cos(x_2))*min(x_3,x_1)+x_12)))', '-0.633359+-0.849541*min(x_12-min(x_5,min(x_9,x_10)**x_0),sin(min(x_9,cos(cos(x_2))*sin(x_3)+x_12)))', '-0.637645+-0.622647*min(x_12-x_5-sin(3.867158),x_0)', '0.000000+-0.755934*x_12', '-0.769718+-0.691176*min(x_12-x_5-sin(3.865925),sin(min(x_9,x_0)))', '-0.758235+-0.710615*min(x_12-x_5-sin(3.772117),sin(min(min(x_6,sqrt(x_4)),x_0)))', '-0.672821+-0.869987*min(x_12-min(x_5,3.114263**sin(cos(x_0))),min(min(sin(x_6),x_0),x_12))', '-0.112117+-1.163603*sin(x_12)', '-0.640133+-0.857191*min(x_12-min(x_5,min(x_9,x_10)**x_0),sin(min(sin(x_6),x_0)))', '-0.708961+-0.744989*min(x_12-x_5-sin(3.549821),sin(min(sin(x_6),x_0)))']
        #g._pick_best_model(X, y, models)
        #pow,min,max,abs,exp,1/,¬
