import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

DATA = 'data/'
ODE = 'ode/'
PDE = 'pde/'

data = np.loadtxt(DATA + ODE + 'infliximab.csv', skiprows=1, delimiter=',')

plt.figure(figsize=(10,6))
plt.plot(data[:,0], data[:,1], label='L')
plt.show()
plt.figure(figsize=(10,6))
plt.plot(data[:,0], data[:,2], label='r')
plt.show()
plt.figure(figsize=(10,6))
plt.plot(data[:,0], data[:,3], label='A')
plt.figure(figsize=(10,6))
plt.show()
plt.figure(figsize=(10,6))
plt.plot(data[:,0], data[:,4], label='C')
plt.show()
