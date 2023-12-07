import glob
import os
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor
from pygpg.sk import GPGRegressor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess

class fancydict:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


def run_gp_gomea(args, X_train, y_train, X_val, y_val):
    print(args)
    g = GPGRegressor(t=args.time, g=-1, e=-1, tour=-1, d=4, use_optim=False, optimiser_choice="LM",
        disable_ims=True, pop=1024, nolink=False, feat_sel=-1,
        no_large_fos=False, no_univ_exc_leaves_fos=False,
        finetune=False, bs_opt="auto",
        bs=2048, random_accept_p=0.0,
        verbose=False, csv_file=args.results_path + "/stats.csv", opt_per_gen=1, use_clip=False,
        fset="+,-,*,/,sin,cos,exp,log,sqrt", cmp=0.0, rci=0.0, use_ftol=False, tol=1e-9, use_mse_opt=False, log=True, reinject_elite=False, ff="lsmse", use_local_search=False,
        optimise_after=False, add_addition_multiplication=False, add_any=False, use_max_range=True, equal_p_coeffs=False,

        random_state=args.repeat) 
    
    g.fit_val(X_train, y_train, X_val, y_val)
    

def all_feynmann(repeats):
    for filename in glob.glob("dataset/feynmann_all/*.tsv"):
        df = pd.read_csv(filename).iloc[:5000]
        X = df.drop(columns=['target']).to_numpy()
        y = df['target'].to_numpy()

        name = os.path.splitext(os.path.basename(filename))[0]

        for repeat in range(repeats):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=repeats+1)

            odir = os.path.join("results", name, f"{repeat:05d}")
            os.makedirs(odir, exist_ok=True)
    
            params = fancydict(
                time=900,
                value_to_reach=1e-6,
                generations=-1,
                evaluations=-1,
                seed=-1,
                prob="symbreg",
                functions="+_-_*_/_sin_cos_exp_log_sqrt",
                erc=True,
                gomea=True,
                gomfos="LT",
                inittype="RHH",
                initmaxtreeheight=4,
                syntuniqinit=10000,
                popsize=1024,
                results_path=odir,
                parallel=1, # no parallelism
                verbose=False,
                repeat=repeat
            )

            yield params, X_train, y_train, X_test, y_test

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("cmd", choices=["run", "status", "postprocess"])
    parser.add_argument("-r", "--repeats", type=int, default=10, dest="repeats")
    parser.add_argument("-j", "--mp", type=int, default=1, dest="max_parallel", help="Maximum concurrency")
    
    args = parser.parse_args()

    # nohup python feynmann.py run -r 10 -j 65 &
    if args.cmd == "run":
        with ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
            scheduled = [ pool.submit(run_gp_gomea, *experiment) \
                for experiment in all_feynmann(args.repeats) ]
            total, completed = len(scheduled), 0
            for future in as_completed(scheduled):
                e = future.exception()
                if e is not None:
                    print(e)
                else:
                    completed += 1
                    print(f"{completed / total * 100: .2f}% done ({completed} of {total})")
        
    elif args.cmd == "status":
        total, completed = 0, 0
        for run_dir in glob.glob(f"results/*/*"):
            total += 1

            results_file = os.path.join(run_dir, "result.txt")
            if os.path.exists(results_file) and os.path.isfile(results_file):
                completed += 1
        print(f"{completed / total * 100: .2f}% done ({completed} of {total})")

    elif args.cmd == "postprocess":
        # .csv files contain generation, time_seconds, evaluations, mse_train, r2_train, mse_test, r2_test for each generation
        # problem, run and version are added later
        runs = []
        for run_dir in glob.glob(f"results/*/*"):
            _, problem, run_str = run_dir.split(os.path.sep)

            results_file = os.path.join(run_dir, "result.txt")
            if not (os.path.exists(results_file) and os.path.isfile(results_file)):
                print(f"Missing run {run_str} for problem {problem}!")
                continue

            stats = os.path.join(run_dir, "stats.csv")
            run_df = pd.read_csv(stats)
            run_df["problem"] = problem
            run_df["run"] = int(run_str)
            run_df["version"] = "GP-GOMEA"
            runs.append(run_df)
        if len(runs) == 0:
            print("No data to postprocess")
            sys.exit(1)
        df = pd.concat(runs, ignore_index=True)
        df.to_csv("feynmann_GP-GOMEA.zip", index=False, compression=dict(
            method="zip",
            archive_name="feynmann_GP-GOMEA.csv"
        ))