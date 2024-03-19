import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation



def read_tsv_file(file_path):
    try:
        # Read the TSV file into a pandas DataFrame
        df = pd.read_csv(file_path, header=None, sep='\t')
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def process_rows(df):

    max_len = 0
    max_mse =0
    processed_data = []
    for index, row in df.iterrows():
        coordinates = []
        # Split each long line by ';' to get individual coordinate strings
        coordinate_strings = row[0].split(';')
        for coord_str in coordinate_strings:
            # Split each coordinate string by ',' to get x, y, z values
            x, y, z, z1, z2 = coord_str.replace("inf","200").split(',')
            coordinates.append((float(x), float(y), int(z), int(z1), int(z2)))


            if float(x)>max_mse:
            	max_mse = float(x)
            if float(y)>max_len:
            	max_len = float(y)
        processed_data.append(coordinates)
    return processed_data, max_len, max_mse



# def update(frame):
#     ax.clear()
#     ax.set_xlim(15, 50)  # Set fixed x-axis limits
#     ax.set_ylim(0, 32)  
#     row = processed_data[frame]
#     sorted_by_z = [[],[],[],[],[],[],[]]

#     for coord in row:
#     	sorted_by_z[coord[2]].append((coord[0], coord[1]))

#     for z_val, row in enumerate(sorted_by_z):
# 	    x_vals = [coord[0] for coord in row]
# 	    y_vals = [coord[1] for coord in row]

# 	    ax.scatter(x_vals, y_vals, label=f'Z={z_val}')
#     ax.set_xlabel('MSE')
#     ax.set_ylabel('Model size')
#     ax.set_title(f'Generation {frame+1}')
#     ax.legend()

def update(frame):
    ax.clear()
    # ax.set_xlim(0, max_mse*1.5)  # Set fixed x-axis limits
    # ax.set_ylim(0, max_len*1.5)

    # row = processed_data[frame]
    #
    # xs = []
    # ys = []
    # for coord in row:
    #     xs.append(coord[0])
    #     ys.append(coord[1])

    best_mse_ind = (999999.,999999,-1)
    best_size_ind = (999999.,999999,-1)



    row = processed_data[frame]
    sorted_by_z = [[],[],[],[],[],[],[]]

    for coord in row:
      sorted_by_z[coord[2]].append((coord[0], coord[1], coord[3], coord[4]))

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

    ax.scatter(best_mse_ind[0],best_mse_ind[1],marker='x', label=f'Z={best_mse_ind[2]}, obj={best_mse_ind[3]}')
    ax.scatter(best_size_ind[0],best_size_ind[1],marker='x', label=f'Z={best_size_ind[2]}, obj={best_size_ind[3]}')
    #ax.scatter(xs,ys)
    ax.set_xlabel('MSE')
    ax.set_ylabel('Model size')
    ax.set_title(f'Generation {frame+1}')
    ax.legend()



tsv_file_path = "./MOMT.csv"
df = read_tsv_file(tsv_file_path)
processed_data, max_len, max_mse = process_rows(df)
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=len(processed_data), interval=1000)

for i in range(len(processed_data)):
    update(i)
    plt.savefig(f'frame_{i:04d}.png')

