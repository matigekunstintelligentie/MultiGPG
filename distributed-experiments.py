import subprocess
from multiprocessing.pool import ThreadPool, Pool
import os

class experiment:
    def __init__(self):
        self.experiment_dict = {}

    def check_dict(self):
        for idx, (key, value) in enumerate(self.experiment_dict.items()):
            if value is None:
                raise Exception("Key {} is None".format(key))

    def construct_string(self):
        stri = "python run_experiment.py"
        for idx, (key, value) in enumerate(self.experiment_dict.items()):
            stri += " --{} {}".format(key, value)
        return stri

    def __str__(self):
        stri = ""
        for idx, (key, value) in enumerate(self.experiment_dict.items()):
            if idx != len(self.experiment_dict.items()):
                stri += "{}: {}\n".format(key, value)
            else:
                stri += "{}: {}".format(key, value)

        return stri

    def __repr__(self):
        stri = ""
        for idx, (key, value) in enumerate(self.experiment_dict.items()):
            if idx != len(self.experiment_dict.items()):
                stri += "{}: {}\n".format(key, value)
            else:
                stri += "{}: {}".format(key, value)

        return stri


class gpgomea_experiment(experiment):
    def __init__(self, experiment_dict):
        super(gpgomea_experiment, self).__init__()

        self.experiment_dict = {
            "optimize": None,
            "csv_name": None,
            "dir": None,
            "batch_size": None,
            "seed": None,
            "every_n_steps": 1,
            "clip": False,
            "coeff_p": None,
            "dataset": None,
            "depth": None,
            "log": None,
            "contains_train": None,
            "t": None,
            "g": None,
            "use_mse_opt": None,
            "ff": None,
            "ss": None,
            "use_ftol": None,
            "equal_p_coeffs": True,
            "tour": 4,
            "fset": None,
            "popsize": None,
            "nr_multi_trees": None,
            "max_coeffs": None,
            "use_adf": None,
            "use_aro": None,
            "MO_mode": None,
            "n_clusters": None,
            "verbose": None,
            "use_max_range": True,
            "discount_size": None
        }

        for key, value in experiment_dict.items():
            self.experiment_dict[key] = value



def run(experiment):
    experiment.check_dict()
    print(experiment.construct_string())
    subprocess.run(experiment.construct_string(), shell=True)

verbose = True
n_processes = 1
duration = 60
popsize = 2048

experiments = []

fset = '+,-,*,/,sin,cos,log,sqrt'
extra_fset = '+,-,*,/,sin,cos,log,sqrt,max,min,exp,**2,1/'
arithmetic_fset = '+,-,*,/'

result_dir = "./results/multi_trees"
datasets = ["dowchemical","tower", "air", "concrete", "bike"]

for i in range(1):
    for dataset in datasets:
        contains_train = "_train" in dataset

        directory = result_dir
        isExist = os.path.exists(directory)
        if not isExist:
           os.makedirs(directory)

        for exp in [
            #gpgomea_experiment({"csv_name":"tree_8", "depth": 8, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 7, "max_coeffs": 0, "nr_multi_trees": 1,  "t": duration, "g":-1, "use_adf":True, "use_aro": True, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False}),
            #gpgomea_experiment({"csv_name":"tree_42", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 7, "max_coeffs": 0, "nr_multi_trees": 2,  "t": duration, "g":-1, "use_adf":True, "use_aro": True, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False}),
            #
            #gpgomea_experiment({"csv_name":"SO", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": False, "popsize":popsize, "n_clusters": 7, "max_coeffs": 0, "nr_multi_trees": 4,  "t": duration, "g":-1, "use_adf":True, "use_aro": True, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False}),
            #gpgomea_experiment({"csv_name":"MO", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 7, "max_coeffs": 0, "nr_multi_trees": 4,  "t": duration, "g":-1, "use_adf":True, "use_aro": True, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False}),
            gpgomea_experiment({"csv_name":"discount", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 7, "max_coeffs": 0, "nr_multi_trees": 4,  "t": duration, "g":-1, "use_adf":True, "use_aro": True, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":True}),
            #gpgomea_experiment({"csv_name":"MO_nocluster", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 1, "max_coeffs": 0, "nr_multi_trees": 4,  "t": duration, "g":-1, "use_adf":True, "use_aro": True, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False}),
            #
            #gpgomea_experiment({"csv_name":"MO_noadf", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 7, "max_coeffs": 0, "nr_multi_trees": 4,  "t": duration, "g":-1, "use_adf":False, "use_aro": True, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False}),
            #gpgomea_experiment({"csv_name":"MO_noaro", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 7, "max_coeffs": 0, "nr_multi_trees": 4,  "t": duration, "g":-1, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False}),
        ]:

            experiments.append(exp)


p = Pool(n_processes)
p.map(run, experiments)

