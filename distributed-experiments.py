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
            "optimizer": "lm",
            "every_n_steps": 1,
            "clip": False,
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
            "popsize": None,
            "nr_multi_trees": None,
            "max_coeffs": None

        }

        if experiment_dict is not None:
            self.experiment_dict = experiment_dict



def run(experiment):
    experiment.check_dict()
    subprocess.run(experiment.construct_string(), shell=True)


n_processes = 1
duration = 100

experiments = []



fset = '+,-,*,/,sin,cos,log,sqrt'
extra_fset = '+,-,*,/,sin,cos,log,sqrt,max,min,exp,**2,1/'
arithmetic_fset = '+,-,*,/'

result_dir = lambda depth:"./results/multi_trees"
datasets = ["dowchemical","tower", "air", "concrete", "bike"]


for i in range(5):
    for dataset in datasets:
        contains_train = "_train" in dataset

        directory = result_dir
        isExist = os.path.exists(directory)
        if not isExist:
           os.makedirs(directory)

        for exp in [

        ]:

        experiments.append(exp)


p = Pool(n_processes)
p.map(run, experiments)

