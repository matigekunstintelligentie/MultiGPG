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
            "log_pop": None,
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
            "discount_size": None,
            "balanced": False,
            "donor_fraction": 2.,
            "log_pop": False,
            "accept_diversity": True,
            "k2": False
        }

        for key, value in experiment_dict.items():
            self.experiment_dict[key] = value



def run(experiment):
    experiment.check_dict()
    print(experiment.construct_string())
    subprocess.run(experiment.construct_string(), shell=True)

log_pop = False
verbose = False
n_processes = 30
duration = 3600*2
generations = -1
popsize = 1024

experiments = []

fset = '+,-,*,/,sin,cos,log,sqrt'
extra_fset = '+,-,*,/,sin,cos,log,sqrt,max,min,exp,**2,1/'
arithmetic_fset = '+,-,*,/'

sin_fset = "+,sin"

result_dir = "./results/all"
datasets = ["dowchemical","tower", "air", "concrete", "bike", "synthetic_dataset"]

for i in range(10):
    for dataset in datasets:
        contains_train = "_train" in dataset

        directory = result_dir
        isExist = os.path.exists(directory)
        if not isExist:
           os.makedirs(directory)

        directory_pop = result_dir + "/pop"
        isExist = os.path.exists(directory_pop)
        if not isExist:
            os.makedirs(directory_pop)

        for exp in [
            # Why clustering is necessary
            gpgomea_experiment({"csv_name":"MO", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":log_pop}),
            gpgomea_experiment({"csv_name":"MO_nocluster", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 1, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":log_pop}),

            gpgomea_experiment({"csv_name":"SO", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": False, "popsize":popsize, "n_clusters": 1, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":log_pop}),

            # Normal experiments
            gpgomea_experiment({"csv_name":"MO_equalclustersize", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":log_pop}),
            gpgomea_experiment({"csv_name":"MO_balanced", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":True,"log_pop":log_pop}),
            gpgomea_experiment({"csv_name":"MO_k2", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"k2":True,"log_pop":log_pop}),
            gpgomea_experiment({"csv_name":"MO_frac1", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":log_pop,"donor_fraction":1.}),

            # equal size
            gpgomea_experiment({"csv_name":"MO_equalclustersize_frac1", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":log_pop,"donor_fraction":1.}),
            gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":True,"log_pop":log_pop}),
            gpgomea_experiment({"csv_name":"MO_equalclustersize_k2", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"k2":True,"log_pop":log_pop}),

            gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced_frac1", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":True,"log_pop":log_pop,"donor_fraction":1.}),
            gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"k2":True,"log_pop":log_pop,"donor_fraction":1.}),

            gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_noadf", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":False, "use_aro": True, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":True,"k2":False,"log_pop":log_pop,"donor_fraction":2.}),
            gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced_discount", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":True,"balanced"True,"k2":False,"log_pop":log_pop,"donor_fraction":2.}),
            
        ]:

            experiments.append(exp)

p = Pool(n_processes)
p.map(run, experiments)

# n_processes = 15
#
# result_dir = "./results/all"
# for i in range(30):
#     for dataset in ["synthetic_dataset"]:
#         contains_train = "_train" in dataset
#         directory = result_dir
#         isExist = os.path.exists(directory)
#         if not isExist:
#             os.makedirs(directory)
#         for exp in [
#             gpgomea_experiment({"csv_name":"tree_42", "depth":  4, "dir": directory, "batch_size": 256, "ff": "mse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": 0, "nr_multi_trees": 2,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": sin_fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"k2":True,"log_pop":log_pop,"donor_fraction":1.}),
#             gpgomea_experiment({"csv_name":"tree_44", "depth":  4, "dir": directory, "batch_size": 256, "ff": "mse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": 0, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": sin_fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"k2":True,"log_pop":log_pop,"donor_fraction":1.}),
#             gpgomea_experiment({"csv_name":"tree_7", "depth":  7, "dir": directory, "batch_size": 256, "ff": "mse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": 0, "nr_multi_trees": 1,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": sin_fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"k2":True,"log_pop":log_pop,"donor_fraction":1.}),
#         ]:
#             experiments.append(exp)
# p = Pool(n_processes)
# p.map(run, experiments)











# verbose = False
# n_processes = 12
# duration = -1
# generations = 100
# popsize = 2048
#
# experiments = []
#
# fset = '+,-,*,/,sin,cos,log,sqrt'
# extra_fset = '+,-,*,/,sin,cos,log,sqrt,max,min,exp,**2,1/'
# arithmetic_fset = '+,-,*,/'
#
# sin_fset = "+,sin"
#
# result_dir = "./results/test"
# datasets = ["tower"]
#
# for i in range(2):
#     for dataset in datasets:
#         contains_train = "_train" in dataset
#
#         directory = result_dir
#         isExist = os.path.exists(directory)
#         if not isExist:
#             os.makedirs(directory)
#
#         directory_pop = result_dir + "/pop"
#         isExist = os.path.exists(directory_pop)
#         if not isExist:
#             os.makedirs(directory_pop)
#
#         for exp in [
#             gpgomea_experiment({"csv_name":"SO", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": False, "popsize":popsize, "n_clusters": 1, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":True}),
#             # gpgomea_experiment({"csv_name":"MO", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":True}),
#             gpgomea_experiment({"csv_name":"MO_k2", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":True,"k2":True}),
#
#             #
#             # gpgomea_experiment({"csv_name":"MO_balanced", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":True,"log_pop":True}),
#             # gpgomea_experiment({"csv_name":"MO_equalclustersize", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":True}),
#             # gpgomea_experiment({"csv_name":"MO_equalclustersize_frac1", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"log_pop":True,"donor_fraction":1.}),
#             # gpgomea_experiment({"csv_name":"MO_equalclustersize_discount", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":True,"balanced":False,"log_pop":True}),
#             #
#             gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":True,"log_pop":True}),
#             gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced_frac1", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":True,"log_pop":True,"donor_fraction":1.}),
#             # gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced_discount", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":True,"balanced":True,"log_pop":True}),
#             #
#
#             gpgomea_experiment({"csv_name":"MO_equalclustersize_k2", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"k2":True,"log_pop":True}),
#             gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False,"k2":True,"log_pop":True,"donor_fraction":1.}),
#             gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_discount", "depth":  4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize*5, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":True,"balanced":False,"k2":True,"log_pop":True}),
#
#             #gpgomea_experiment({"csv_name":"discount", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":True,"balanced":False}),
#             #gpgomea_experiment({"csv_name":"MO_nocluster", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 1, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False}),
#
#             #gpgomea_experiment({"csv_name":"MO_noadf", "depth": 4, "dir": directory, "batch_size": 256, "ff": "lsmse", "seed": i, "coeff_p": 1., "MO_mode": True, "popsize":popsize, "n_clusters": 5, "max_coeffs": -1, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":False, "use_aro": True, "dataset": dataset, "fset": fset, "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": True, "discount_size":False,"balanced":False}),
#         ]:
#
#             experiments.append(exp)
#
# p = Pool(n_processes)
# p.map(run, experiments)
