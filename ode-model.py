import numpy as np
import pandas as pd
import pysindy as ps

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, UnivariateSpline

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.base import BaseEstimator

from matplotlib import pyplot as plt

DIR = 'data/pde/'

def read_file(path):
    data = np.loadtxt(DIR+path, delimiter=',')
    data = data[np.argsort(data[:, 0])]
    t_unique, indices = np.unique(data[:, 0], return_index=True)
    x_unique = data[indices, 1]
    
    return t_unique, x_unique

t17_scale = 1e-3
t_t17, t17 = read_file('t17.csv')
t_t17[-1] = 100

m_scale = 1e-1
t_m, m = read_file('m.csv')
t_m[-1] = 100

f_scale = 1e-2
t_f, f = read_file('f.csv')
t_f[-1] = 100

i17_scale = 1e-12
t_i17, i17 = read_file('i17.csv')
t_i17[-1] = 100

i23_scale = 1e-8
t_i23, i23 = read_file('i23.csv')
t_i23[-1] = 100

ta_scale = 1e-11
t_ta, ta = read_file('ta.csv')
t_ta[-1] = 100

i6_scale = 1e-9
t_i6, i6 = read_file('i6.csv')
t_i6[-1] = 100

s_scale = 1e-12
t_s, s = read_file('s.csv')
t_s[-1] = 100

g_scale = 1e-11
t_g, g = read_file('g.csv')
t_g[-1] = 100

p_scale = 1e-9
t_p, p = read_file('p.csv')
t_p[-1] = 100

q_scale = 1e-5
t_q, q = read_file('q.csv')
t_q[-1] = 100

qr_scale = 1e-6
t_qr, qr = read_file('qr.csv')
t_qr[-1] = 100

tspan = (0, 100)

t_data = [t_t17, t_m, t_f, t_i17, t_i23, t_ta, t_i6, t_s, t_g, t_p, t_q, t_qr ]
x_data = [t17, m, f, i17, i23, ta, i6, s, g, p, q, qr]
x_scale = [t17_scale, m_scale, f_scale, i17_scale, i23_scale, ta_scale, i6_scale, s_scale, g_scale, p_scale, q_scale, qr_scale]
x_names = ['t17', 'm', 't', 'i17', 'i23', 'ta', 'i6', 's', 'g', 'p', 'q', 'qr']

t_global = np.linspace(*tspan, 100)
X = np.array([ interp1d(t_data[i], x_data[i], kind=1)(t_global) for i in range(12) ]).T # rows are the cell types
dt = t_global[1] - t_global[0]

diff_method = ps.FiniteDifference(order=2, drop_endpoints=True) # ps.SmoothedFiniteDifference(smoother_kws={'window_length':10, 'polyorder':3 })
feat_lib = ps.PolynomialLibrary(degree=2)
optimizer = ps.STLSQ(threshold=0.005)
model = ps.SINDy(
    differentiation_method=diff_method,
    feature_library=feat_lib,
    optimizer=optimizer
    )
model.fit(X, t=dt)
model.print()
print(model.predict(X[0].reshape(1, -1)))
print(X[0])

def rhs(t, z):
    return model.predict(z.reshape(1, -1))[0]
t = np.linspace(*tspan, 200)
sol = solve_ivp(rhs, tspan, X[0], t_eval=t)

print(sol.t[:10])

fig, (ax1, ax2) = plt.subplots(1, 2)
for i in range(12):
    ax1.plot(sol.t, sol.y[i])
    ax2.plot(t_global, X[:, i])
    pass
plt.show()
print(x_scale)
