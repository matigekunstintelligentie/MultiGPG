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
import os
import scikit_posthocs as sp

plt.style.use('seaborn')

max_gen = None
max_size = 10

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
    hvs = []
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

        hvs.append( ind(result))
    

    return hvs

def statistics(hv_list, el, dataset, appendix):
    try:
        print("#"*20, dataset, appendix, "#"*20)
        print(sorted(list(hv_list.keys())))
        stri = ""
        for x in ["{0:.2f} $\\pm$ {1:.2f}".format(np.mean(hv_list[k]), np.std(hv_list[k])) for k in sorted(list(hv_list.keys()))]:
            stri += x + " & "
        print(stri)
        x = sp.posthoc_wilcoxon(hv_list, p_adjust="Holm")
        print(x)
        print("#"*20)
        return x
    except:
        pass


def make_plots(d, folder, x_index, appendix):
    
    for el in [
    ['SO','MO'],
    ['MO_equalclustersize', 'MO_balanced', 'MO_k2', 'MO_frac1'], 
    ['MO_equalclustersize_frac1', 'MO_equalclustersize_balanced', 'MO_equalclustersize_k2'], 
    ['MO_equalclustersize_balanced_frac1','MO_equalclustersize_k2_frac1','SO']]:
        fig = plt.figure()
        plt.title("Dataset: {}".format(dataset.capitalize()))
        markers = ['o', 'x', '^','s']
        x = []

        hv_list = defaultdict(list)

        for key in d.keys():
            if key in el:
                x = d[key][x_index]
                y = d[key][1]

                gens = np.min(d[key][2])

                hvs = calc_hv(dataset_filename_fronts, dataset, key, x_index, max_size)

                hv_list[key].append(hvs)

                coefficients = np.polyfit(x, y, deg=5)  # Adjust the degree of polynomial as needed
                poly = np.poly1d(coefficients)

                # Generate points for the fitted curve
                x_fit = np.linspace(min(x), max(x), 1000)
                y_fit = poly(x_fit)


                # Plot scatter plot and line plot
                color = sns.color_palette()[int(el.index(key))]  # Get color from tab10 colormap
                marker = markers[int(el.index(key))]
                plt.scatter(x, y, alpha=0.5, s=25, label=key + " Average HV={0:.3f}, Average gens={1:.1f}".format(np.mean(hvs), gens), color=color, marker=marker)
                plt.plot(x_fit, y_fit, c=color, alpha=0.5)

                # Commented out, because extreme slow
                # print("nondom")
                # nondom_list_x, nondom_list_y = non_dom(x,y)
                # plt.plot(nondom_list_x, nondom_list_y, linestyle='--', c=color,alpha=0.5)
                # print("done")

        statistics(hv_list, el, dataset, appendix)      

        if(len(x)>0):
            plt.xlim(0.5,None)
            plt.xlabel(r'$r^2$')
            plt.ylabel('Model size')
            # plt.yscale('log', base=5)
            plt.legend()
            fig.set_size_inches(32, 18)



            directory = "./results/plots/" + folder
            isExist = os.path.exists(directory)
            if not isExist:
                os.makedirs(directory)

            if max_gen is None:
                plt.savefig(directory + "/{}_{}.pdf".format(dataset + "".join(el), appendix), dpi=300, bbox_inches='tight')
            else:
                plt.savefig(directory + "/{}_{}gen_{}.pdf".format(dataset + "".join(el), max_gen, appendix), dpi=300, bbox_inches='tight')
        plt.close()




#

for dataset in ["air", "bike", "concrete","dowchemical","tower", "synthetic_dataset"]:


    d = defaultdict(lambda: defaultdict(list))

    folder = "MOSO"
    dir = "./results/" + folder
    for filename in glob.glob(dir + "/*.csv"):
        nr = filename.split("/")[-1].split("_")[0]
        d_key = "_".join(filename.split("/")[-1].split("_")[1:]).replace(dataset,"").replace(".csv","")[:-1]

        if(dataset in filename):
            try:

                scatter_x = []
                scatter_y = []

                scatter_x_val = []

                #, error_bad_lines=False
                df = pd.read_csv(filename, sep="\t", header=None, nrows=max_gen)


                gens = len(df.iloc[0][8].split(","))


                mg = -1
                if max_gen is not None and len(df.iloc[0][13].split(";")) >= max_gen:
                    mg = max_gen - 1

                for el in df.iloc[0][13].split(";")[mg].split("],"):
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
                d[d_key][4].append(df.iloc[0][1])
            except:
                pass

    # print("DATASET", dataset) 
    # for d_key in sorted(list(d.keys()), key=lambda d_key: np.mean(d[d_key][4])):
    #     print(d_key)
    #     print(np.mean(d[d_key][4]), len(d[d_key][4]))


    print("-"*10,"Train","-"*10)
    make_plots(d, folder, x_index=0, appendix="train")
    # print("-"*10,"Validation","-"*10)
    # make_plots(d, folder, x_index=3, appendix="val")



