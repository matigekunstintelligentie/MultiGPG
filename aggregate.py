import pandas as pd
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
import numpy as np
from numpy.polynomial import Polynomial
import seaborn as sns
from pymoo.indicators.hv import HV

def non_dom(x,y):
    nondom_list_x = []
    nondom_list_y = []
    for el in sorted(zip(x,y)):
        nondom = False
        for el2 in zip(x,y):
            if el[0]<el2[0] and el[1]>el2[1]:
                nondom = True
                break

        if not nondom:
            nondom_list_x.append(el[0])
            nondom_list_y.append(el[1])
    
    return nondom_list_x, nondom_list_y         

def calc_hv(dataset_filename_fronts, key1, key2):
    hvs = 0
    count = 0
    for el in dataset_filename_fronts[key1][key2]:
        x = el[0]
        y = el[1]

        x_array = 1. - np.array(x)
        y_array = np.array(y)/100.

        # Stack the arrays horizontally
        result = np.column_stack((x_array, y_array))

        ref_point = np.array([1., 1.])

        ind = HV(ref_point=ref_point)

        hvs += ind(result)
        count += 1

    return hvs/count

plt.style.use('seaborn')

dataset_filename_fronts = defaultdict(lambda: defaultdict(list))

max_gen = 90

for dataset in ["dowchemical","tower", "air", "concrete", "bike", "synthetic_dataset"]:

    d = defaultdict(lambda: defaultdict(list))
    count = defaultdict(int)
    for filename in glob.glob("./results/multi_trees/*.csv"):
        nr = filename.split("/")[-1].split("_")[0]
        d_key = "_".join(filename.split("/")[-1].split("_")[1:]).replace(dataset,"").replace(".csv","")[:-1]

        if(dataset in filename):


            scatter_x = []
            scatter_y = []

            df = pd.read_csv(filename, sep="\t", header=None)
            gens = len(df.iloc[-1][14].split(","))

            mg = -1
            if max_gen is not None and len(df.iloc[-1][13].split(";")) >= max_gen:
                mg = max_gen - 1

            for el in df.iloc[-1][13].split(";")[mg].split("],"):
                rep = el.replace("[","").replace("{","").split(",")
                scatter_x.append(1. - float(rep[0])/float(df.iloc[-1][6]))
                scatter_y.append(float(rep[2]))



            dataset_filename_fronts[dataset][d_key].append((scatter_x,scatter_y))

            d[d_key][0].extend(scatter_x)
            d[d_key][1].extend(scatter_y)
            d[d_key][2].append(gens)
            count[d_key] += 1

    for el in [['tree_42', 'tree_7'], ['MO_equalclustersize', 'SO', 'MO'], ['MO', 'discount'], ['MO', 'MO_nocluster'], ['MO_noadf','MO']]:
        fig = plt.figure()
        if max_gen is not None:
            plt.title("Dataset: {} - max gen: {}".format(dataset.capitalize(), max_gen))
        else:
            plt.title("Dataset: {}".format(dataset.capitalize()))
        for key in d.keys():
            if key in el:
                x = d[key][0]
                y = d[key][1]

                gens = np.mean(d[key][2])

                hvs = calc_hv(dataset_filename_fronts, dataset, key)

                coefficients = np.polyfit(x, y, deg=2)  # Adjust the degree of polynomial as needed
                poly = np.poly1d(coefficients)

                # Generate points for the fitted curve
                x_fit = np.linspace(min(x), max(x), 100)
                y_fit = poly(x_fit)

                # Plot scatter plot and line plot
                color = sns.color_palette()[int(el.index(key))]  # Get color from tab10 colormap
                plt.scatter(x, y, alpha=0.6, s=10, label=key + " Average HV={}, Average gens={}".format(hvs, gens), c=color)
                plt.plot(x_fit, y_fit, c=color)
                nondom_list_x, nondom_list_y = non_dom(x,y)
                plt.plot(nondom_list_x, nondom_list_y, linestyle='--', c=color)


        plt.xlabel(r'$r^2$')
        plt.ylabel('Model size')
        # plt.yscale('log', base=5)
        plt.legend()
        fig.set_size_inches(32, 18)

        if max_gen is None:
            plt.savefig("./results/plots/{}.pdf".format(dataset + "".join(el)), dpi=300, bbox_inches='tight')
        else:
            plt.savefig("./results/plots/{}_{}gen.pdf".format(dataset + "".join(el), max_gen), dpi=300, bbox_inches='tight')


#
# for key1 in dataset_filename_fronts.keys():
#     for key2 in dataset_filename_fronts[key1].keys():
#
#         hvs = 0
#         count = 0
#         for el in dataset_filename_fronts[key1][key2]:
#             x = el[0]
#             y = el[1]
#
#             x_array = 1. - np.array(x)
#             y_array = np.array(y)
#
#             # Stack the arrays horizontally
#             result = np.column_stack((x_array, y_array))
#
#             ref_point = np.array([2., 1000.])
#
#             ind = HV(ref_point=ref_point)
#
#             hvs += ind(result)
#             count += 1
#         print(key1, key2, hvs/count)
#
#     # log average HV
#     # ref point should be max*10
#     # ref_point = np.array([4., 4.])
#     #
#     # ind = HV(ref_point=ref_point)
#     # print("HV", ind(np.array([[1., 3.],[2,1]])))
#     # quit()
#
# # -----------------------------------------------------------------------------------

# for dataset in ["synthetic_dataset"]:
#
#     for filename in glob.glob("./results/multi_trees/*.csv"):
#         nr = filename.split("/")[-1].split("_")[0]
#         d_key = "_".join(filename.split("/")[-1].split("_")[1:]).replace(dataset,"").replace(".csv","")[:-1]
#
#         if(dataset in filename):
#             df = pd.read_csv(filename, sep="\t", header=None)
#
#             MO_archive_sols_only = []
#             MO_archive = []
#
#             for idx, col in enumerate(df.iloc[-1][13].split(";")):
#                 for el in col.split("],"):
#                     rep = el.replace("[","").replace("{","").split(",")
#
#                     transform_rep = 1. - float(rep[0]) / float(df.iloc[-1][6])
#
#                     if ("{:.2f}".format(transform_rep), rep[2]) not in MO_archive_sols_only:
#                         MO_archive_sols_only.append(("{:.2f}".format(transform_rep), float(rep[2])))
#                         MO_archive.append((transform_rep, float(rep[2]), idx))
#
#             sol_idxs = []
#             for sol_idx, sol in enumerate(MO_archive):
#                 dom = False
#                 for sol_comp in MO_archive:
#                     if (sol_comp[0]-sol[0]) > 1e-3 and sol[1]>sol_comp[1]:
#                         dom = True
#                         break
#
#                 if dom:
#                     sol_idxs.append(sol_idx)
#
#             for idx in sorted(list(set(sol_idxs)), reverse=True):
#                 del MO_archive_sols_only[idx]
#                 del MO_archive[idx]
#
#
#             plt.figure()
#             plt.title("Dataset: {}, filename: {}".format(dataset.capitalize(), d_key))
#             xs = []
#             ys = []
#             cs = []
#             for x,y,color in MO_archive:
#                 xs.append(x)
#                 ys.append(y)
#                 cs.append(color)
#
#             plt.scatter(xs, ys, alpha=0.5, s=30, c=cs)
#             plt.colorbar(label="Generation added")
#
#             plt.xlabel(r'$r^2$')
#             plt.ylabel('Model size')
#             plt.show()


