import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
import glob

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


def process_row(row, max_mse, max_len, data, update_max=False, grid=False):
    coordinates = []
    # Split each long line by ';' to get individual coordinate strings
    coordinate_strings = row[0].split(';')
    for coord_str in coordinate_strings:
        # Split each coordinate string by ',' to get x, y, z values
        x, y, z, z1, z2 = coord_str.replace("inf","-1").split(',')
        if(grid):
            coordinates.append((float(x), float(y), float(z), float(z1), int(z2)))
        else:
            coordinates.append((float(x), float(y), int(z), int(z1), int(z2)))

        if update_max:
            if float(x)>max_mse and int(float(x)) !=-1:
                max_mse = float(x)
            if float(y)>max_len and int(float(x))!=-1:
                max_len = float(y)
    data.append(coordinates)
    return max_mse, max_len

def process_rows(df):
    max_len = 0
    max_mse = 0
    processed_data = []
    processed_cluster_data = []
    processed_donor_data = []
    processed_front_data = []
    processed_minmax = []
    for index, row in df.iterrows():
        if index%5==0:
            max_mse, max_len = process_row(row, max_mse, max_len, processed_data, update_max=True)
        if index%5==1:
            process_row(row, max_mse, max_len, processed_cluster_data, update_max=False)
        if index%5==2:
            process_row(row, max_mse, max_len, processed_donor_data, update_max=False)
        if index%5==3:
            process_row(row, max_mse, max_len, processed_front_data, update_max=False)
        if index%5==4:
            process_row(row, max_mse, max_len, processed_minmax, update_max=False, grid=True)

    return processed_data, processed_cluster_data, processed_donor_data, processed_front_data, processed_minmax, max_len, max_mse


def update(frame):
    ax.clear()

    best_mse_ind = (999999.,999999,-1)
    best_size_ind = (999999.,999999,-1)

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
      sorted_by_z[coord[2]].append((x, y, coord[3], coord[4]))

    for z_val, row in enumerate(sorted_by_z):
      if(len(row)>0):
          x_vals = [coord[0] for coord in row]
          y_vals = [coord[1] for coord in row]
          ax.scatter(x_vals, y_vals, label=f'Z={z_val} obj={row[0][2]} nr={row[0][3]}', alpha=0.5)

          for coord in row:
              if coord[0]<best_mse_ind[0]:
                best_mse_ind = (coord[0],coord[1],z_val,row[0][2],row[0][3])
              if coord[1]<best_size_ind[1]:
                best_size_ind = (coord[0],coord[1],z_val,row[0][2],row[0][3])

    # ax.set_ylim(0,200)
    # ax.set_xlim(0,4000)
    ax.scatter(best_mse_ind[0],best_mse_ind[1],marker='x', label=f'Z={best_mse_ind[2]}, obj={best_mse_ind[3]}')
    ax.scatter(best_size_ind[0],best_size_ind[1],marker='x', label=f'Z={best_size_ind[2]}, obj={best_size_ind[3]}')
    #ax.scatter(xs,ys)
    ax.set_xlabel('MSE')
    ax.set_ylabel('Model size')
    ax.set_title(f'Generation {frame+1} best mse {best_mse_ind[0]}')
    ax.legend()


def update2(frame):
    ax.clear()

    colors = ['b','g','r','c','m','y','k']

    row = data[frame]

    sorted_by_z = [[],[],[],[],[],[],[]]

    for coord in row:
      x = float(str(coord[0]).replace("-1.0", str(max_mse)))  
      y = float(str(coord[1]).replace("-1.0", str(max_len)))  
      sorted_by_z[coord[2]].append((x, y, coord[3], coord[4]))

    for z_val, row in enumerate(sorted_by_z):
      if(len(row)>0):
          x_vals = [coord[0] for coord in row]
          y_vals = [coord[1] for coord in row]
          ax.scatter(x_vals, y_vals, label=f'Z={z_val} obj={row[0][2]} nr={row[0][3]}', alpha=0.5, marker="x", color=[colors[z_val]])

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
            ax.scatter(x_vals, y_vals, label=f'Z={z_val} obj={row[0][2]} nr={row[0][3]}', alpha=0.2, marker="o", color=[colors[z_val]])      
    
    ax.set_ylim(0,200)
    ax.set_xlim(0,4000)
    ax.set_xlabel('MSE')
    ax.set_ylabel('Model size')
    ax.set_title(f'Generation {frame+1}')
    ax.legend()    


def make_frames(data, folder, title):
    global ax
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    ani = FuncAnimation(fig, update, frames=len(data), interval=1000)

    directory = f'./frames/{folder}'
    isExist = os.path.exists(directory)
    if not isExist:
        os.makedirs(directory)

    for i in range(len(data)):
        update(i)
        plt.savefig(f'{directory}/{title}_{i:04d}.png')

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
        plt.savefig(f'{directory}/{title}_{i:04d}.png')


#for filename in glob.glob("./results/test/*.csv"):
#folder = filename.split("/")[-1]
folder = "balanced"
# tsv_file_path =  f'./results/pop/{folder}'


df = read_tsv_file("balanced.csv")
try:
    processed_data, processed_cluster_data, processed_donor_data, processed_front_data, processed_minmax, max_len, max_mse = process_rows(df)

    data = processed_front_data
    minmax = processed_minmax
    make_frames(data, folder, "front")

    # data = processed_cluster_data
    # make_frames(data, folder, "pop")
    #
    # data = processed_cluster_data
    # data2 = processed_donor_data
    # make_frames2(data, data2, folder, "donors")
except:
    print(tsv_file_path)
    pass    


