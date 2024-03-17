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


dataset_filename_fronts = defaultdict(lambda: defaultdict(list))

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

def calc_hv(dataset_filename_fronts, key1, key2, x_index, max_size):
    hvs = 0
    count = 0
    for el in dataset_filename_fronts[key1][key2]:
        x = el[x_index]
        y = el[1]

        x_array = 1. - np.array(x)
        y_array = np.array(y)/max_size

        # Stack the arrays horizontally
        result = np.column_stack((x_array, y_array))

        ref_point = np.array([1., 1.])

        ind = HV(ref_point=ref_point)

        hvs += ind(result)
        count += 1

    return hvs/count

def make_plots(d, x_index, appendix):
    for el in [['tree_42', 'tree_7'], ['MO_equalclustersize', 'SO', 'MO'], ['MO', 'discount'], ['MO', 'MO_nocluster'], ['MO_noadf','MO']]:
        fig = plt.figure()
        plt.title("Dataset: {}".format(dataset.capitalize()))
        markers = ['o', 'x', '^']
        x = []
        for key in d.keys():
            if key in el:
                x = d[key][x_index]
                y = d[key][1]

                gens = np.mean(d[key][2])

                hvs = calc_hv(dataset_filename_fronts, dataset, key, x_index, max_size)

                coefficients = np.polyfit(x, y, deg=2)  # Adjust the degree of polynomial as needed
                poly = np.poly1d(coefficients)

                # Generate points for the fitted curve
                x_fit = np.linspace(min(x), max(x), 1000)
                y_fit = poly(x_fit)

                # Plot scatter plot and line plot
                color = sns.color_palette()[int(el.index(key))]  # Get color from tab10 colormap
                marker = markers[int(el.index(key))]
                plt.scatter(x, y, alpha=0.6, s=25, label=key + " Average HV={0:.3f}, Average gens={1:.1f}".format(hvs, gens), c=color, marker=marker)
                plt.plot(x_fit, y_fit, c=color)
                nondom_list_x, nondom_list_y = non_dom(x,y)
                plt.plot(nondom_list_x, nondom_list_y, linestyle='--', c=color)


        plt.xlabel(r'$r^2$')
        plt.ylabel('Model size')
        # plt.yscale('log', base=5)
        plt.legend()
        fig.set_size_inches(32, 18)
        if len(x)!=0:
            plt.savefig("./results/plots/{}_{}.pdf".format(dataset + "".join(el), appendix), dpi=300, bbox_inches='tight')    

plt.style.use('seaborn')


max_size = 0
for dataset in ["dowchemical","tower", "air", "concrete", "bike", "synthetic_dataset"]:

    d = defaultdict(lambda: defaultdict(list))
    for filename in glob.glob("./results/multi_trees/*.csv"):
        nr = filename.split("/")[-1].split("_")[0]
        d_key = "_".join(filename.split("/")[-1].split("_")[1:]).replace(dataset,"").replace(".csv","")[:-1]

        if(dataset in filename):


            scatter_x = []
            scatter_y = []

            scatter_x_val = []

            df = pd.read_csv(filename, sep="\t", header=None)
            gens = len(df.iloc[0][14].split(","))


            for el in df.iloc[0][13].split(";")[-1].split("],"):
                rep = el.replace("[","").replace("{","").split(",")
                scatter_x.append(1. - float(rep[0])/float(df.iloc[0][6]))
                scatter_y.append(float(rep[2]))
                scatter_x_val.append(1. - float(rep[0])/float(df.iloc[0][7]))

                if float(rep[2])>max_size:
                    max_size = float(rep[2])




            dataset_filename_fronts[dataset][d_key].append((scatter_x,scatter_y,gens,scatter_x_val))

            d[d_key][0].extend(scatter_x)
            d[d_key][1].extend(scatter_y)
            d[d_key][2].append(gens)
            d[d_key][3].extend(scatter_x_val)


    make_plots(d, x_index=0, appendix="train")        
    make_plots(d, x_index=3, appendix="val")  
