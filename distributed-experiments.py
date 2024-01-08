import subprocess
from multiprocessing.pool import ThreadPool, Pool

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
            "optimizer": None,
            "every_n_steps": None,
            "clip": None,
            "coeff_p": None,
            "dataset": None,
            "depth": None,
            "log": None,
            "contains_train": None,
            "t": None,
            "g": None,
            "reinject_elite": None,
            "use_local_search": None,
            "optimise_after": None,
            "max_size": None,
            "use_mse_opt": None,
            "ff": None,
            "ss": None,
            "add_addition_multiplication": None,
            "use_ftol": None,
            "equal_p_coeffs": None,
            "random_accept_p": None,
            "tour": None,
            "fset": None,
            "popsize": None
        }

        if experiment_dict is not None:
            self.experiment_dict = experiment_dict



def run(experiment):
    experiment.check_dict()
    subprocess.run(experiment.construct_string(), shell=True)


n_processes = 15
duration = 100

experiments = []

import os

fset = '+,-,*,/,sin,cos,log,sqrt'
extra_fset = '+,-,*,/,sin,cos,log,sqrt,max,min,exp,**2,1/'
arithmetic_fset = '+,-,*,/'

result_dir = lambda depth:"./results/multi_trees"
datasets = ["boston"] 


for i in range(10):
    for dataset in datasets:
        contains_train = "_train" in dataset
        for depth in [4]:
            directory = result_dir(depth)
            isExist = os.path.exists(directory)
            if not isExist:
               os.makedirs(directory)
            
            reinject = True
            for exp in [
                gpgomea_experiment({"popsize": 1024, "nr_multi_trees": 1, "optimize": "False", "dir":directory, "csv_name": "mt_1","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                gpgomea_experiment({"popsize": 1024, "nr_multi_trees": 2, "optimize": "False", "dir":directory, "csv_name": "mt_2","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                gpgomea_experiment({"popsize": 1024, "nr_multi_trees": 3, "optimize": "False", "dir":directory, "csv_name": "mt_2","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                gpgomea_experiment({"popsize": 1024, "nr_multi_trees": 4, "optimize": "False", "dir":directory, "csv_name": "mt_2","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),

                gpgomea_experiment({"popsize": 1024, "nr_multi_trees": 1, "optimize": "False", "dir":directory, "csv_name": "mt_1_depth5","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":5, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                gpgomea_experiment({"popsize": 1024, "nr_multi_trees": 2, "optimize": "False", "dir":directory, "csv_name": "mt_2_depth3","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":3, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
            ]:
                    experiments.append(exp)


p = Pool(n_processes)
p.map(run, experiments)

