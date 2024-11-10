import subprocess
from multiprocessing.pool import ThreadPool, Pool
import os
import random

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

    def construct_command(self):
        """
        Construct a command that will be passed to subprocess.run.
        Returns a list of command parts instead of a shell string.
        """
        # Assuming your command is constructed from the dictionary in a non-shell format
        command = [
            'python', 'run_experiment.py'
        ]

        for idx, (key, value) in enumerate(self.experiment_dict.items()):
            command.append("--{}".format(key))
            command.append("{}".format(value))

        return command


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
            "k2": False,
            "nr_objs": None,
            "remove_duplicates": None,
            "replacement_strategy": None,
            "use_GA": False,
            "use_GP": False,
            "drift": False,
            "koza": False,
            "change_second_obj": "size",
            "full_mode":False,
            "log_front": True
        }

        for key, value in experiment_dict.items():
            self.experiment_dict[key] = value


skip_duplicates = True

def run(experiment):
    experiment.check_dict()
    print(experiment.construct_string())

    filename = f'{experiment.experiment_dict["dir"]}/{int(experiment.experiment_dict["seed"])+1}_{experiment.experiment_dict["csv_name"]}_{experiment.experiment_dict["dataset"]}.csv'
    if os.path.isfile(filename) and skip_duplicates:
        print("Skipping", filename)

    else:
        try:
            # command = experiment.construct_command()
            # # subprocess.run takes a list of command and args when shell=False
            # subprocess.run(command, shell=False, check=True)
            command = experiment.construct_string()
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Experiment failed with error: {e}")


log_pop = False

verbose = False
n_processes = 30
duration = 3600*3
generations = -1
popsize = 4096
nr_objs = 2


fset = '+,-,*,/,sin,cos,log,sqrt,abs'
fset_nthg = '+,-,*,/,sin,cos,log,sqrt,abs,nthg'
extra_fset = '+,-,*,/,sin,cos,log,sqrt,max,min,exp,**2,1/,abs'
arithmetic_fset = '+,-,*,/'

sin_fset = "+,sin"


result_dir = "./results/prelim"


experiments = []
datasets = ["dowchemical","tower", "air", "concrete", "bike", "synthetic_1", "synthetic_2", "synthetic_3", "synthetic_4", "synthetic_5"]


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

        ff = "lsmse"
        max_coeffs = -1
        coeff_p = 1.
        if "synthetic" in dataset:
            ff = "mse"
            max_coeffs = 0
            coeff_p = 0.

            for exp in [

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_discounted", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":True, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),
                ]:
                    #pass
                    experiments.append(exp)

        else:
            for exp in [

                gpgomea_experiment({"csv_name":"SO", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_nocluster", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"SO_equalclustersize", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize*5,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"SO_equalclustersize_mutate", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize*5,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"SO_mutate", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_balanced", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":True,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_k2", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":True,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":True,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_k2_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_balanced_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":True,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_mutate", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),



                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_mutate_maxerror", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "maxerror"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_maxerror", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "maxerror"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_mutate_kommenda", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "kommenda"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_kommenda", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "kommenda"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_mutate_plus", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "plus"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_plus", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "plus"}),

                gpgomea_experiment({"csv_name":"SO_maxerror", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False, "change_second_obj": "maxerror"}),

                gpgomea_experiment({"csv_name":"SO_kommenda", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False, "change_second_obj": "kommenda"}),

                gpgomea_experiment({"csv_name":"SO_plus", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False, "change_second_obj": "plus"}),
                #Missing: k2+equal+frac1, balanced_equal_frac1 -> use best of these with: init, duplicates and also SO (5)
                #Best with Different objs, both SO and MO (2)
                #Total: (7)


            ]:
                #pass
                experiments.append(exp)







print("NR EXPERIMENTS", len(experiments))
# random.shuffle(experiments)


p = Pool(processes=n_processes)
p.map(run, experiments)


log_pop = True

for i in [30]:
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

        ff = "lsmse"
        max_coeffs = -1
        coeff_p = 1.
        if "synthetic" in dataset:
            ff = "mse"
            max_coeffs = 0
            coeff_p = 0.

            for exp in [

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_discounted", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":True, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),
            ]:
                #pass
                experiments.append(exp)

        else:
            for exp in [

                gpgomea_experiment({"csv_name":"SO", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_nocluster", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"SO_equalclustersize", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize*5,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"SO_equalclustersize_mutate", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize*5,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"SO_mutate", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_balanced", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":True,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_k2", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":True,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_balanced_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":True,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_k2_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_balanced_frac1", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":True,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_mutate", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False}),



                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_mutate_maxerror", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "maxerror"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_maxerror", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "maxerror"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_mutate_kommenda", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "kommenda"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_kommenda", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "kommenda"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_mutate_plus", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":True, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "plus"}),

                gpgomea_experiment({"csv_name":"MO_equalclustersize_k2_frac1_plus", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": True, "popsize":popsize*5,
                                    "n_clusters": 5, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":True,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"mutate", "drift": True,
                                    "donor_fraction":1., "full_mode":False, "change_second_obj": "plus"}),

                gpgomea_experiment({"csv_name":"SO_maxerror", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False, "change_second_obj": "maxerror"}),

                gpgomea_experiment({"csv_name":"SO_kommenda", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False, "change_second_obj": "kommenda"}),

                gpgomea_experiment({"csv_name":"SO_plus", "depth":4 , "dir": directory, "batch_size": 256, "ff": ff, "seed": i, "coeff_p": coeff_p, "MO_mode": False, "popsize":popsize,
                                    "n_clusters": 1, "max_coeffs": max_coeffs, "nr_multi_trees": 4,  "t": duration, "g":generations, "use_adf":True, "use_aro": False, "dataset": dataset, "fset": fset,
                                    "log": True, "verbose": verbose, "contains_train": contains_train, "use_mse_opt": False, "ss": False, "use_ftol": False, "optimize": False,
                                    "discount_size":False, "k2":False,"balanced":False,"log_pop":log_pop,"nr_objs": 2, "remove_duplicates":False, "replacement_strategy":"sample", "drift": True,
                                    "donor_fraction":2., "full_mode":False, "change_second_obj": "plus"}),

                #Missing: k2+equal+frac1, balanced_equal_frac1 -> use best of these with: init, duplicates and also SO (5)
                #Best with Different objs, both SO and MO (2)
                #Total: (7)


            ]:
                #pass
                experiments.append(exp)

print("NR EXPERIMENTS", len(experiments))
# random.shuffle(experiments)


p = Pool(processes=n_processes)
p.map(run, experiments)




