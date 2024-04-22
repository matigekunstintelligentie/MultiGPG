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



# Function to generate dataset: summation of 8 sin(2pi + xi), where xi are the 8 prime numbers
def generate_dataset_1(num_rows, num_cols, primes):
    dataset = np.zeros((num_rows, num_cols + 2))  # +1 for the target column + 1 for fixed column
    dataset[:, 0] = np.rand(num_rows, 1)
    for i in range(num_rows):
        stri = ""
        for j in range(1,num_cols+1):
            prime_idx = j % len(primes)  # Wrap around primes list if necessary
            prime = primes[prime_idx]
            dataset[i, j] = prime * np.random.rand()  # Sample from distribution

            stri = stri + "np.sin(" +  str(dataset[i, j]) + "+" + str(dataset[i, 0]) + ")"
            if(j!=num_cols):
                stri += "+"
        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)

    return dataset

# Function to generate dataset: summation of 7 sin(2pi + 5) + sin(7*11)
def generate_dataset_2(num_rows, primes):
    dataset = np.zeros((num_rows, 3 + 2))  # +1 for the target column + 1 for fixed column
    dataset[:, 0] = np.rand(num_rows, 1)

    prime_idx = 3
    prime = primes[prime_idx]
    dataset[:, 1] = prime * np.random.rand(num_rows)

    prime_idx = 4
    prime = primes[prime_idx]
    dataset[:, 2] = prime * np.random.rand(num_rows)

    prime_idx = 5
    prime = primes[prime_idx]
    dataset[:, 2] = prime * np.random.rand(num_rows)

    for i in range(num_rows):
        stri = ""
        for j in range(1,num_cols):

            stri = stri + "np.sin(" +  str(dataset[i, 1]) + "+" + str(dataset[i, 0]) + ")"

            stri += "+"

        str += stri + "np.sin(" +  str(dataset[i, 2]) + "+" + str(dataset[i, 3]) + ")"

        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)

    return dataset

# Function to generate dataset: summation of 4 sqrt(abs(sin(2/pi +xi)))
def generate_dataset_3(num_rows, primes):
    dataset = np.zeros((num_rows, 4 + 2))  # +1 for the target column + 1 for fixed column
    dataset[:, 0] = np.rand(num_rows, 1)
    for i in range(num_rows):
        stri = ""
        for j in range(1,5):
            prime_idx = j % len(primes)  # Wrap around primes list if necessary
            prime = primes[prime_idx]
            dataset[i, j] = prime * np.random.rand()  # Sample from distribution

            stri = stri + "np.sqrt(abs(np.sin(" +  str(dataset[i, j]) + "+" + str(dataset[i, 0]) + ")))"
            if(j!=4):
                stri += "+"
        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)

    return dataset

# Function to generate dataset: summation of 4 sqrt(abs(sin(2/pi +xi)))
def generate_dataset_4(num_rows, primes):
    dataset = np.zeros((num_rows, 4 + 2))  # +1 for the target column + 1 for fixed column
    dataset[:,0] = np.rand(num_rows, 1)
    for i in range(num_rows):
        stri = ""
        for j in range(1,5):
            prime_idx = j % len(primes)  # Wrap around primes list if necessary
            prime = primes[prime_idx]
            dataset[i, j] = prime * np.random.rand()  # Sample from distribution

            stri = stri + "np.sqrt(abs(np.sin(" +  str(dataset[i, j]) + "+" + str(dataset[i, 0]) + ")))"
            if(j!=4):
                stri += "+"


        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)

    return dataset

# Define parameters
num_rows = 1000
num_cols = 8  # Change this to the desired number of columns

# Generate prime numbers
primes = generate_primes(num_cols)

# Generate dataset
dataset = generate_dataset(num_rows, num_cols, primes)

