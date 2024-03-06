import numpy as np
import csv
import math
 
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

# Function to generate dataset
def generate_dataset(num_rows, num_cols, primes):
    dataset = np.zeros((num_rows, num_cols + 1))  # +1 for the target column
    for i in range(num_rows):
        stri = ""
        for j in range(num_cols):
            prime_idx = j % len(primes)  # Wrap around primes list if necessary
            prime = primes[prime_idx]
            dataset[i, j] = prime * np.random.rand()  # Sample from distribution
            stri = stri + "np.sin(" +  str(dataset[i, j])
            if(j!=num_cols-1):
                stri += "+"

        stri += ")"*num_cols
        # Target column calculation (example: sum of x variables)
        dataset[i, -1] = eval(stri)
    return dataset

# Define parameters
num_rows = 1000
num_cols = 6  # Change this to the desired number of columns

# Generate prime numbers
primes = generate_primes(num_cols)

# Generate dataset
dataset = generate_dataset(num_rows, num_cols, primes)

# Write dataset to CSV file
with open('./dataset/synthetic_dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    header = ['x_' + str(i) for i in range(num_cols)] + ['target']
    #writer.writerow(header)
    
    # Write data
    for row in dataset:
        writer.writerow(row)
