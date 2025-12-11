import numpy as np
import pandas as pd
import pysindy as ps

from deeptime.sindy import SINDy as SI
from deeptime.sindy import STLSQ

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, UnivariateSpline

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
t_global = np.linspace(*tspan, 100)

t_data = [t_t17, t_m, t_f, t_i17, t_i23, t_ta, t_i6, t_s, t_g, t_p, t_q, t_qr ]
x_data = [t17, m, f, i17, i23, ta, i6, s, g, p, q, qr]
x_scale = [t17_scale, m_scale, f_scale, i17_scale, i23_scale, ta_scale, i6_scale, s_scale, g_scale, p_scale, q_scale, qr_scale]
x_names = ['T17', 'M', 'T', 'I17', 'I23', 'Ta', 'I6', 'S', 'G', 'P', 'Q', 'Qr']

X = np.array([ interp1d(t_data[i], x_data[i], kind=1)(t_global) for i in range(12) ]).T # rows are the cell types
dt = t_global[1] - t_global[0]


class SINDyCV(SI):
    def fit(self, X, y=None, **kwargs):
        # fit normally
        return super().fit(X, t=kwargs.get("t", None))
    
    def score(self, X, y=None):
        # simulate the model from initial condition
        sim = self.fetch_model().simulate(X[0], t=np.linspace(0, dt*(len(X)-1), len(X)))
        # use r2_score over all variables
        return r2_score(X, sim)
    pass

feat_lib = PolynomialFeatures(degree=1)
optimizer = STLSQ(threshold=0.00001)

model = SINDyCV(
    library=feat_lib,
    optimizer=optimizer,
    input_features=x_names
    )

param_grid = {
    "optimizer__threshold": [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-6],
    "library__degree": [0, 2, 3, 4],
}
search = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=5))

search.fit(X)
print(search.best_params_)

best = search.best_estimator_
func = best.fetch_model()
func.print()
sim = func.simulate(X[0], t=t_global)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
for i in range(12):
    ax1.plot(t_global, sim[:, i], label=x_names[i])
    ax2.plot(t_global, X[:, i], label=x_names[i])
    pass
ax1.set_title("Regressed Model Simulation")
ax2.set_title("Interpolated Raw Data")
fig.suptitle("Scaled Concentrations vs Time in Days (Quadratic Model)") # Could be linear model
fig.supylabel("Scaled Concentrations (g/cm^3)")
fig.supxlabel("Time in Days")
fig.legend()
plt.savefig('true_vs_pred_quad.png') # could be linear.png
plt.show()
print(x_scale)

print("\n=== Loss Between Model Simulation and Raw Data ===\n")
losses = {}

for i, name in enumerate(x_names):
    y_true = X[:, i]        # shape (100,)
    y_pred = sim[:, i]      # shape (100,)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)

    losses[name] = dict(mse=mse, mae=mae, r2=r2)

    print(f"{name}:")
    print(f"    MSE = {mse:.6e}")
    print(f"    MAE = {mae:.6e}")
    print(f"    R²  = {r2:.6f}")
    print()

# Optional: summary table
loss_df = pd.DataFrame(losses).T
print(loss_df)
