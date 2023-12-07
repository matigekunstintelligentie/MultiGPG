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


n_processes = 1
duration = 120

experiments = []

import os

fset = '+,-,*,/,sin,cos,log,sqrt'
extra_fset = '+,-,*,/,sin,cos,log,sqrt,max,min,exp,**2,1/'
arithmetic_fset = '+,-,*,/'

result_dir = lambda depth:"./results/optimisation_fair_bfgs_{}".format(depth)
datasets = ["boston"] 


for i in range(1):
    for dataset in datasets:
        contains_train = "_train" in dataset
        for depth in [4]:
            directory = result_dir(depth)
            isExist = os.path.exists(directory)
            if not isExist:
               os.makedirs(directory)
            
            reinject = True
            for exp in [
                gpgomea_experiment({"popsize": 1024, "optimize": "False", "dir":directory, "csv_name": "no_opt_ff","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),                
                # gpgomea_experiment({"popsize": 1024, "optimize": "False", "dir":directory, "csv_name": "coeffmut","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 1.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "bfgs","batch_size": "auto","seed": i,"optimizer": "bfgs","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "bfgs_coeffmut","batch_size": "auto","seed": i,"optimizer": "bfgs","every_n_steps": 1,"clip": "False","coeff_p": 1.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_coeffmut","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 1.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),

                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "bfgs_ftol","batch_size": "auto","seed": i,"optimizer": "bfgs","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": True, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_ftol","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": True, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "bfgs_b256","batch_size": 256,"seed": i ,"optimizer": "bfgs","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                #gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_ff","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "bfgs_b32","batch_size": 32,"seed": i,"optimizer": "bfgs","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b32","batch_size": 32,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "bfgs_clip","batch_size": "auto","seed": i,"optimizer": "bfgs","every_n_steps": 1,"clip": "True","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
                # gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_clip","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "True","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),


#                 # Best settings test
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_4steps","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 4,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_8steps","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 8,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_random_accept","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.05,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_coeffmut","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 1.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),               
#                 # The role of forced improvements
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_noforced","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":-1, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
# 
#                 #LS experiments
#                 gpgomea_experiment({"popsize": 1024, "optimize": "False", "dir":directory, "csv_name": "no_opt_ls","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),                
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_ls","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_ls_noopt","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": True, "ff": "lsmse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
# 
#                 gpgomea_experiment({"popsize": 1024, "optimize": "False", "dir":directory, "csv_name": "no_opt_add","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": True, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_add","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": True, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "False", "dir":directory, "csv_name": "no_opt_addany","batch_size": "auto","seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": True, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_addany","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": True, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 
#                 #ss experiments
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_maxrange","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": True, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_z","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": True, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_maxrange_z","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": True, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": True, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_z_ls","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": True, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_maxrange_z_ls","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "lsmse", "ss": True, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": True, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#               
#                 # Depth experiments
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_depth3","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":3, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_depth5","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":5, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_depth6","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":6, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                 
#                 # Fset experiments
#                  gpgomea_experiment({"popsize": 4096, "optimize": "True", "dir":directory, "csv_name": "lm_b256_largepop","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":fset}),
#                  gpgomea_experiment({"popsize": 4096, "optimize": "True", "dir":directory, "csv_name": "lm_b256_afset_largepop","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":arithmetic_fset}),
#                  gpgomea_experiment({"popsize": 4096, "optimize": "True", "dir":directory, "csv_name": "lm_b256_extrafset_largepop","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":extrafset}),
#                 gpgomea_experiment({"popsize": 1024, "optimize": "True", "dir":directory, "csv_name": "lm_b256_extrafset","batch_size": 256,"seed": i,"optimizer": "lm","every_n_steps": 1,"clip": "False","coeff_p": 0.,"dataset":dataset, "tour":4, "depth":depth, "contains_train": contains_train, "log":True, 't':duration, 'g':-1, 'reinject_elite': reinject, "use_local_search": False, "optimise_after": False, "use_mse_opt": False, "ff": "mse", "ss": False, "add_addition_multiplication": False, "add_any": False, "use_ftol": False, "use_max_range": False, "equal_p_coeffs": True, "random_accept_p": 0.0,"fset":extrafset}),
# =============================================================================
# =============================================================================
                ]:
                    experiments.append(exp)


p = Pool(n_processes)
p.map(run, experiments)

