
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import os
import sys

#########################################################
#Setting up the path for interpreters with relative paths
#########################################################

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
grand_parent_directory = os.path.dirname(parent_directory)

# Add the grad parent directory to the system path
sys.path.append(grand_parent_directory)
# Change the current working directory to the parent directory
os.chdir(grand_parent_directory)

###########################################
# Generation of a sinousoidal wave with noise
###########################################
rd.seed(42)

#---------- Wave frequency
f_1 = 2
f_2 = 11
f_3 = 7

#---------- Wave Amplitude
A_1 = 3
A_2 = 0.5
A_3 = 1

A_noise = 0.5

t_array = np.arange(0,1,0.01)
y_base = np.array([A_1 * np.sin(f_1*np.pi*t) + A_2 *np.sin(f_2*np.pi*t) +  A_3 * np.sin(f_3*np.pi*t)  for t in t_array])
y_w_noise = y_base + np.array([ A_noise * rd.random() for t in t_array ])

plt.plot(t_array, y_base)
plt.scatter(t_array, y_w_noise,color = 'red')
plt.grid()

print(f'---------------------------------------------')
print(f'Sinuosoidal wave with noise dataframe successfully generated')
print(f' -> {y_w_noise.shape[0]} data points.')
print(f'')

df_sinuosoidal = pd.DataFrame({'x': t_array, 'y': y_w_noise})
df_sinuosoidal.to_csv('support files/sample database/sin_wave_with_noise.csv', index=False) 

###########################################
# Generation of a sample classification database
###########################################
from sklearn.datasets import make_classification

# Generate synthetic classification data
# Generated via a normal distribution centered at multi-dimensional data points

X, y = make_classification(n_samples=500, n_features=4, n_informative=4, n_redundant=0, n_classes=3, random_state=42)

# Convert to DataFrame for easier viewing
df_classifier = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
df_classifier['target'] = y
df_classifier.to_csv('support files/sample database/classifier_cluster_sample_data.csv', index=False)

# Display first few rows

print(f'---------------------------------------------')
print(f'Sample classification data successfully generated')
print(f' -> {X.shape[1]} features')
print(f' -> {X.shape[0]} datapoints')
print(f' -> label 0: {(y == 0).sum()}')
print(f' -> label 1: {(y == 1).sum()}')
print(f' -> label 2: {(y == 2).sum()}')
print(f'\n')
print(df_classifier.head())
print(f'\n')

plt.show()