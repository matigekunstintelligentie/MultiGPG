import numpy as np
import csv
import math

np.random.seed(42)

# Function to generate first n primes
def generate_primes(n):
    primes = []
    X = 0
    i = 2
    flag = False
    while(X < n):
        flag = True
        for j in range(2, math.floor(math.sqrt(i)) + 1):
            if (i%j == 0):
                flag = False
                break
        if(flag):
            primes.append(i)
            X+=1
        i+=1
    return primes

def make_csv(dataset, title):
    # Write dataset to CSV file
    with open(f'./dataset/{title}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['x_' + str(i) for i in range(len(dataset[0])-1)] + ['target']
        writer.writerow(header)
        
        # Write data
        for row in dataset:
            writer.writerow(row)

    # Write dataset to CSV file without header for calls without lib
    with open(f'./dataset/{title}_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write data
        for row in dataset:
            writer.writerow(row)

def check_nans_infs(dataset):
    if(np.isnan(dataset).any()):
        raise Exception("Nan found in dataset")
    if(np.isinf(dataset).any()):
        raise Exception("Inf found in dataset")

# Function to generate dataset: summation of 8 sin(2pi + xi), where xi are the 8 prime numbers
def generate_dataset_1(num_rows, num_cols, primes):
    dataset = np.zeros((num_rows, num_cols + 2))  # +1 for the target column + 1 for fixed column
    prime_idx = 0
    prime = primes[prime_idx]
    dataset[:,0] = prime * np.random.rand(num_rows)
    for i in range(num_rows):
        stri = ""
        for j in range(1,num_cols+1):
            prime_idx = j % len(primes)  # Wrap around primes list if necessary
            prime = primes[prime_idx]
            dataset[i, j] = prime * np.random.rand()  # Sample from distribution

            stri = stri + "np.sin(" +  str(dataset[i, j]) + "*" + str(dataset[i,0])  + ")"
            if(j!=num_cols):
                stri += "+"


        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)


    return dataset

# Function to generate dataset: summation of 7 sin(2pi + 5) + sin(7*11)
def generate_dataset_2(num_rows, primes):
    dataset = np.zeros((num_rows, 5))  # +1 for the target column + 1 for fixed column
    prime_idx = 0
    prime = primes[prime_idx]
    dataset[:,0] = prime * np.random.rand(num_rows)

    prime_idx = 1
    prime = primes[prime_idx]
    dataset[:,1] = prime * np.random.rand(num_rows)

    prime_idx = 2
    prime = primes[prime_idx]
    dataset[:,2] = prime * np.random.rand(num_rows)

    prime_idx = 3
    prime = primes[prime_idx]
    dataset[:,3] = prime * np.random.rand(num_rows)

    for i in range(num_rows):
        stri = ""
        for j in range(1,num_cols):

            stri = stri + "np.sin(" +  str(dataset[i, 0]) + "*" + str(dataset[i, 1]) + ")"

            stri += "+"

        stri += stri + "np.sin(" +  str(dataset[i, 2]) + "*" + str(dataset[i, 3]) + ")"

        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)

    return dataset

# Function to generate dataset: summation of 4 sqrt(abs(sin(2/pi +xi)))
def generate_dataset_3(num_rows, primes):
    dataset = np.zeros((num_rows, 6))  # +1 for the target column + 1 for fixed column
    prime_idx = 0
    prime = primes[prime_idx]
    dataset[:,0] = prime * np.random.rand(num_rows)

    for i in range(num_rows):
        stri = ""
        for j in range(1,5):
            prime_idx = j % len(primes)  # Wrap around primes list if necessary
            prime = primes[prime_idx]
            dataset[i, j] = prime * np.random.rand()  # Sample from distribution

            stri = stri + "np.sqrt(abs(np.sin(" +  str(dataset[i, j]) + "*" + str(dataset[i, 0]) + ")))"
            if(j!=4):
                stri += "+"

        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)

    return dataset

# Function to generate dataset: summation of 4 sqrt(abs(sin(2/pi +xi)))
def generate_dataset_4(num_rows, primes):
    dataset = np.zeros((num_rows, 9))  # +1 for the target column
    dataset[:,0:8] = np.random.randn(num_rows, 8)
    for i in range(num_rows):

        stri = "np.sin(" + "np.cos(" + str(dataset[i, 0]) + " * " + str(dataset[i, 1]) + ")" + " + " + "np.cos(" + str(dataset[i, 2]) + " * " + str(dataset[i, 3]) + ")" + ") + "
        stri += "np.sin(" + "np.cos(" + str(dataset[i, 4]) + " * " + str(dataset[i, 5]) + ")" + " + " + "np.cos(" + str(dataset[i, 6]) + " * " + str(dataset[i, 7]) + ")" + ") + "
        stri += "np.cos(" + "np.sin(" + str(dataset[i, 0]) + " + " + str(dataset[i, 1]) + ")" + " * " + "np.sin(" + str(dataset[i, 2]) + " + " + str(dataset[i, 3]) + ")" + ") + "
        stri += "np.cos(" + "np.sin(" + str(dataset[i, 4]) + " + " + str(dataset[i, 5]) + ")" + " * " + "np.sin(" + str(dataset[i, 6]) + " + " + str(dataset[i, 7]) + ")" + ")"

       
        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)

    return dataset

# Function to generate dataset: summation of 4 sqrt(abs(sin(2/pi +xi)))
def generate_dataset_5(num_rows, primes):
    dataset = np.zeros((num_rows, 4))  # +1 for the target column
    r = np.random.randn(num_rows, 3)
    for i in range(num_rows):
        dataset[i, 0:3] = r[i,:]
        stri = "np.cos(" + str(dataset[i, 0]) + "* np.sin(" + str(dataset[i, 1]) + "+" + str(dataset[i, 2]) + ")) + "
        stri += "np.cos(" + str(dataset[i, 0]) + "* np.sin(" + str(dataset[i, 2]) + "+" + str(dataset[i, 1]) + ")) + "
        stri += "np.cos(" + str(dataset[i, 1]) + "* np.sin(" + str(dataset[i, 0]) + "+" + str(dataset[i, 2]) + ")) + "
        stri += "np.cos(" + str(dataset[i, 1]) + "* np.sin(" + str(dataset[i, 2]) + "+" + str(dataset[i, 0]) + "))"


        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)

    return dataset

# Define parameters
num_rows = 1000
num_cols = 8

# Generate prime numbers
primes = generate_primes(10)

# Generate dataset 1
dataset = generate_dataset_1(num_rows, num_cols, primes)
check_nans_infs(dataset)
make_csv(dataset, "synthetic_1")

# Generate dataset 2
dataset = generate_dataset_2(num_rows, primes)
check_nans_infs(dataset)
make_csv(dataset, "synthetic_2")

# Generate dataset 3
dataset = generate_dataset_3(num_rows, primes)
check_nans_infs(dataset)
make_csv(dataset, "synthetic_3")

# Generate dataset 4
dataset = generate_dataset_4(num_rows, primes)
check_nans_infs(dataset)
make_csv(dataset, "synthetic_4")

# Generate dataset 5
dataset = generate_dataset_5(num_rows, primes)
check_nans_infs(dataset)
make_csv(dataset, "synthetic_5")
