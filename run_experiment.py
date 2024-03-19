import numpy as np
from pygpg.sk import GPGRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import argparse
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import KFold
import pandas as pd

def str2bool(v):
    """
    Converts argparse string into Boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--optimize', type=str2bool)
parser.add_argument('--dir')
parser.add_argument('--csv_name')
parser.add_argument('--batch_size')
parser.add_argument('--seed', type=int)
parser.add_argument('--every_n_steps')
parser.add_argument('--clip', type=str2bool)
parser.add_argument('--coeff_p')
parser.add_argument('--dataset')
parser.add_argument('--log', type=str2bool)
parser.add_argument('--contains_train', type=str2bool)
parser.add_argument('--t', type=int)
parser.add_argument('--g', type=int)
parser.add_argument("--use_mse_opt", type=str2bool)
parser.add_argument("--ff", type=str)
parser.add_argument('--depth', type=int)
parser.add_argument("--ss", type=str2bool)
parser.add_argument("--use_ftol", type=str2bool)
parser.add_argument("--use_max_range", type=str2bool)
parser.add_argument("--equal_p_coeffs", type=str2bool)
parser.add_argument("--tour", type=int)
parser.add_argument("--fset", type=str)
parser.add_argument('--popsize', type=int)

parser.add_argument('--nr_multi_trees', type=int)
parser.add_argument('--n_clusters', type=int)
parser.add_argument('--max_coeffs', type=int)
parser.add_argument('--MO_mode', type=str2bool)
parser.add_argument('--use_adf', type=str2bool)
parser.add_argument('--use_aro', type=str2bool)
parser.add_argument('--verbose', type=str2bool)
parser.add_argument('--discount_size', type=str2bool)
parser.add_argument('--balanced', type=str2bool)
parser.add_argument('--full_donor', type=str2bool)

args = parser.parse_args()

print(args.csv_name, args.seed, args.dataset)

df = pd.read_csv("dataset/{}.csv".format(args.dataset))
X = df.drop(columns=['target']).to_numpy()
y = df['target'].to_numpy()


experiment = [args]

# Seed 0 does not work in GPG Regressor
args.seed = args.seed + 1
np.random.seed(args.seed)

def inject_string(b):
    if b:
        return "reinject_elite"
    else:
        return "noinject_elite"

g = GPGRegressor(t=args.t, g=args.g, tour=args.tour, d=args.depth,  use_optim=args.optimize,
        pop=args.popsize,
        bs_opt=args.batch_size,
        bs=2048,
        verbose=args.verbose, csv_file="{}/{}_{}_{}.csv".format(args.dir, args.seed, args.csv_name, args.dataset), opt_per_gen=args.every_n_steps, use_clip=args.clip,
        fset=args.fset, cmp=args.coeff_p, use_ftol=args.use_ftol, tol=1e-9, use_mse_opt=args.use_mse_opt, log=args.log, ff=args.ff,
         use_max_range=args.use_max_range, equal_p_coeffs=args.equal_p_coeffs,
        MO_mode=args.MO_mode, use_adf=args.use_adf, use_aro=args.use_aro, n_clusters=args.n_clusters, max_coeffs=args.max_coeffs, discount_size=args.discount_size,
        random_state=args.seed, nr_multi_trees=args.nr_multi_trees, balanced=args.balanced, full_donor=args.full_donor)

#+,-,*,/,¬,log,pow,max,min,abs,exp,sqrt,sin,cos
if not args.contains_train:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=args.seed)

    if args.ss:
        s = SS()
        X_train = s.fit_transform(X_train)
        X_val = s.transform(X_val)
        y_train = s.fit_transform(y_train.reshape((-1,1)))
        y_val = s.transform(y_val.reshape((-1,1)))
else:
    if args.ss:
        s = SS()
        X_train = s.fit_transform(X)
        y_train = s.fit_transform(y.reshape((-1,1)))
    else:
        X_train = X
        y_train = y

# new with val in there
g.fit_val(X_train, y_train, X_val, y_val)


print("done", args.csv_name, args.seed - 1, args.dataset)
