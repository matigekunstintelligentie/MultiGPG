import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
import glob
import traceback

plt.style.use('seaborn')
import os

def read_tsv_file(file_path):
    try:
        # Read the TSV file into a pandas DataFrame
        df = pd.read_csv(file_path, header=None, sep='\t')
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def process_row(row, max_mse, max_len, max_complexity, df_var, data, update_max=False, grid=False):
    coordinates = []
    # Split each long line by ';' to get individual coordinate strings
    coordinate_strings = row[0].split(';')
    for coord_str in coordinate_strings:
        # Split each coordinate string by ',' to get x, y, z values
        if(grid):
            x, y, z, z1, z2, z3, _ = coord_str.replace("inf","-1").split(',')
            coordinates.append((float(z), float(y), float(z), float(z1), float(z2), float(z3)))
        else:
            x, y, z, z1, z2, z3 = coord_str.replace("inf","-1").split(',')
            coordinates.append((1. - float(x)/float(df_var), float(y), float(z), int(z1), int(z2), int(z3)))

        if update_max:
            if 1. - float(x)/float(df_var)<max_mse and int(float(x)) !=-1:
                max_mse = 1. - float(x)/float(df_var)
            if float(y)>max_len and int(float(x))!=-1:
                max_len = float(y)
            if float(z)>max_complexity and int(float(x))!=-1:
                max_complexity = float(z)

    data.append(coordinates)
    return max_mse, max_len, max_complexity

def process_rows(df, df_var):
    max_len = 0
    max_mse = 0
    max_complexity = 0
    processed_data = []
    processed_cluster_data = []
    processed_donor_data = []
    processed_front_data = []
    processed_minmax = []
    for index, row in df.iterrows():
        if index%5==0:
            max_mse, max_len, max_complexity = process_row(row, max_mse, max_len, max_complexity, df_var, processed_data, update_max=True)
        if index%5==1:
            process_row(row, max_mse, max_len, max_complexity, df_var, processed_cluster_data, update_max=False)
        if index%5==2:
            process_row(row, max_mse, max_len, max_complexity, df_var, processed_donor_data, update_max=False)
        if index%5==3:
            process_row(row, max_mse, max_len, max_complexity, df_var, processed_front_data, update_max=False)
        if index%5==4:
            process_row(row, max_mse, max_len, max_complexity, df_var, processed_minmax, update_max=False, grid=True)

    return processed_data, processed_cluster_data, processed_donor_data, processed_front_data, processed_minmax, max_len, max_mse, max_complexity


def update(frame):
    obj_names = {0:"MSE", 1:"expression size", 2:"complexity", 4:"MO", -1:"None"}
    ax.clear()

    best_mse_ind = (0,999999,999999,-1)
    best_size_ind = (0,999999,999999,-1)
    best_complexity_ind = (0,999999,999999,-1)

    row = data[frame]



    # min_0, min_1, max_0, max_1, num_box = minmax[frame][0]
    # linspace_0 = np.linspace(min_0,max_0,num_box)
    #
    # for vline in linspace_0:
    #     plt.axvline(x = vline, color = 'b', linestyle='dashed', alpha=0.5)
    #
    # linspace_1 = np.linspace(min_1,max_1,num_box)
    #
    # for hline in linspace_1:
    #     plt.axhline(y = hline, color = 'b', linestyle='dashed', alpha=0.5)

    sorted_by_z = [[],[],[],[],[],[],[]]

    for coord in row:


      x = float(str(coord[0]).replace("-1.0", str(max_mse)))  
      y = float(str(coord[1]).replace("-1.0", str(max_len)))
      z = float(str(coord[2]).replace("-1.0", str(max_complexity)))
      sorted_by_z[coord[3]].append((x, y, z, coord[4], coord[5]))


    x_idx = 0
    y_idx = 1

    for z_val, row in enumerate(sorted_by_z):
      if(len(row)>0):
          x_vals = [coord[x_idx] for coord in row]
          y_vals = [coord[y_idx] for coord in row]


          ax.scatter(x_vals, y_vals, label=f'Cluster id={z_val} Objective={obj_names[int(row[0][3])]} Cluster size={row[0][4]}', alpha=0.5)

          for coord in row:
              if coord[0]>best_mse_ind[0]:
                best_mse_ind = (coord[0],coord[1],coord[2],z_val,row[0][3],row[0][4])
              if coord[1]<=best_size_ind[1]:
                best_size_ind = (coord[0],coord[1],coord[2],z_val,row[0][3],row[0][4])
              if coord[2]<best_complexity_ind[2]:
                  best_complexity_ind = (coord[0],coord[1],coord[2],z_val,row[0][3],row[0][4])

    # ax.set_ylim(0,200)
    # ax.set_xlim(0,4000)

    print(best_mse_ind)
    ax.scatter(best_mse_ind[x_idx],best_mse_ind[y_idx],marker='x', label=f'Cluster id={best_mse_ind[4]}, Objective={obj_names[int(best_mse_ind[4])]}')
    ax.scatter(best_size_ind[x_idx],best_size_ind[y_idx],marker='x', label=f'Cluster id={best_size_ind[4]}, Objective={obj_names[int(best_size_ind[4])]}')
    ax.scatter(best_complexity_ind[x_idx],best_complexity_ind[y_idx],marker='x', label=f'Cluster id={best_complexity_ind[4]}, Objective={obj_names[int(best_complexity_ind[4])]}')



    idx_to_label = {0:r"$R^2$", 1:"Model size", 2:"Complexity"}

    ax.set_xlabel(idx_to_label[x_idx])
    ax.set_ylabel(idx_to_label[y_idx])
    ax.set_title(f'Generation {frame+1}')
    ax.legend()


def update2(frame):
    obj_names = {0:"MSE", 1:"size", 3:"MO", -1:"None"}
    ax.clear()

    colors = ['b','g','r','c','m','y','k']

    row = data[frame]

    sorted_by_z = [[],[],[],[],[],[],[]]

    for coord in row:
      x = float(str(coord[0]).replace("-1.0", str(0.)))  
      y = float(str(coord[1]).replace("-1.0", str(max_len)))  
      sorted_by_z[coord[2]].append((x, y, coord[3], coord[4]))

    for z_val, row in enumerate(sorted_by_z):
      if(len(row)>0):
          x_vals = [coord[0] for coord in row]
          y_vals = [coord[1] for coord in row]
          ax.scatter(x_vals, y_vals, label=f'Cluster id={z_val} objective={obj_names[int(row[0][2])]} Cluster size={row[0][3]}', alpha=0.5, marker="x", color=[colors[z_val]])

    row = data2[frame]
    sorted_by_z = [[],[],[],[],[],[],[]]

    for coord in row:
      x = float(str(coord[0]).replace("-1.0", str(max_mse)))  
      y = float(str(coord[1]).replace("-1.0", str(max_len)))  
      sorted_by_z[coord[2]].append((x, y, coord[3], coord[4]))

    for z_val, row in enumerate(sorted_by_z):
      if(len(row)>0):
          x_vals = [coord[0] for coord in row]
          y_vals = [coord[1] for coord in row]
          if(z_val == 2):
            ax.scatter(x_vals, y_vals, label=f'Cluster \# {z_val} objective={obj_names[int(row[0][2])]} Cluster size={row[0][3]}', alpha=0.2, marker="o", color=[colors[z_val]])      
    
    
    
    ax.set_xlabel('MSE')
    ax.set_ylabel('Model size')
    ax.set_title(f'Generation {frame+1}')
    ax.legend()    


def make_frames(data, folder, title):
    global ax
    fig, ax = plt.subplots()
    

    ani = FuncAnimation(fig, update, frames=50, interval=1000)

    directory = f'./frames/{folder}'
    isExist = os.path.exists(directory)
    if not isExist:
        os.makedirs(directory)

    for i in range(len(data)):
        update(i)
        plt.savefig(f'{directory}/{title}_{i:04d}.png',dpi=600,bbox_inches="tight")

def make_frames2(data, data2, folder, title):
    global ax
    fig, ax = plt.subplots()

    ani = FuncAnimation(fig, update2, frames=len(data), interval=1000)

    directory = f'./frames/{folder}'
    isExist = os.path.exists(directory)
    if not isExist:
        os.makedirs(directory)

    for i in range(len(data)):
        update2(i)
        plt.savefig(f'{directory}/{title}_{i:04d}.png',dpi=600,bbox_inches="tight")


# for filename in ["./results/cmp/pop/31_MO_concrete.csv", "./results/cmp/pop/31_MO_balanced_concrete.csv", "./results/cmp/pop/31_SO_concrete.csv", "./results/cmp/pop/31_MO_k2_concrete.csv", "./results/cmp/31_MO_equalclustersize_k2_concrete.csv"]:
#     folder = filename.split("/")[-1][:-4]
#     print(folder)
#     df_var = pd.read_csv(filename.replace("pop/",""),header=None, sep="\t").iloc[-1][7]
#     df = read_tsv_file(filename)


folder = "balanced"
df = read_tsv_file("balanced.csv")
df_var = 9.2*9.2



try:
    processed_data, processed_cluster_data, processed_donor_data, processed_front_data, processed_minmax, max_len, max_mse, max_complexity = process_rows(df, df_var)

    data = processed_front_data[:50]
    minmax = processed_minmax
    make_frames(data, folder, "front")

    data = processed_cluster_data[:50]
    make_frames(data, folder, "pop")
#
# data = processed_cluster_data
# data2 = processed_donor_data
# make_frames2(data, data2, folder, "donors")
except Exception as e:
    print(e)
    print(traceback.format_exc())
    pass


