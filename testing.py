import numpy as np
import pandas as pd
import pysindy as ps 
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class SINDyWrapper(BaseEstimator):
    def __init__(self, 
                 differentiation_method=None, 
                 feature_library=None, 
                 optimizer=None,
                 dt=None):
        """
        Wrapper around pysindy.SINDy for sklearn compatibility.
        """
        self.differentiation_method = differentiation_method
        self.feature_library = feature_library
        self.optimizer = optimizer
        self.dt = dt
        
        # Build the internal SINDy model
        self._build_model()
        
    def _build_model(self):
        self.model_ = ps.SINDy(
            differentiation_method=self.differentiation_method,
            feature_library=self.feature_library,
            optimizer=self.optimizer
        )
    
    def fit(self, X, y=None):
        # Fit the SINDy model with the stored dt
        self.model_.fit(X, t=self.dt)
        return self
    
    def predict(self, X):
        return self.model_.predict(X)
    
    def score(self, X, y=None):
        """
        Negative mean squared error on derivatives for GridSearchCV scoring.
        """
        # Compute numerical derivatives
        dXdt_true = ps.SmoothedFiniteDifference()._differentiate(X, t=self.dt)
        dXdt_pred = self.model_.predict(X)
        # Higher is better for sklearn, so return negative mse
        return -np.mean((dXdt_true - dXdt_pred)**2)
    
    def get_params(self, deep=True):
        return {
            "differentiation_method": self.differentiation_method,
            "feature_library": self.feature_library,
            "optimizer": self.optimizer,
            "dt": self.dt
        }
    
    def set_params(self, **params):
        """
        Update parameters for GridSearchCV. Only supports
        updating optimizer threshold and library degree safely.
        """
        for key, value in params.items():
            # Update nested library degree safely
            if key.startswith("feature_library__"):
                attr = key.split("__")[1]
                setattr(self.feature_library, attr, value)
            elif key.startswith("optimizer__"):
                attr = key.split("__")[1]
                setattr(self.optimizer, attr, value)
            else:
                setattr(self, key, value)
        # Rebuild SINDy with updated objects
        self._build_model()
        return self

# np.random.seed(43)
# 
# --- Generate synthetic data ---
tspan = (0, 10)
t_x = np.linspace(*tspan, 200)
t_y = np.linspace(*tspan, 130)
x = np.exp(t_x)
y = (1/2) * np.exp(t_y) - (3/2) * np.exp(-t_y)

t = np.linspace(*tspan, 300)
x_inter = interp1d(t_x, x, kind='cubic')(t)
y_inter = interp1d(t_y, y, kind='cubic')(t)
dt = t[1] - t[0]

noise_level = 0.05  # 5% noise
x_noisy = x_inter * (1 + noise_level * np.random.randn(len(x_inter)))
y_noisy = y_inter * (1 + noise_level * np.random.randn(len(y_inter)))

X = np.vstack([x_noisy, y_noisy]).T
print(X.shape)

differentiation_method = ps.SmoothedFiniteDifference(
    smoother_kws={'window_length': 15, 'polyorder': 3}
)
feature_library = ps.PolynomialLibrary(degree=3)
optimizer = ps.STLSQ(threshold=0.1)

param_grid = {
    "optimizer__threshold": [0.005, 0.01, 0.05, 0.1],
    "feature_library__degree": [1, 2, 3],
    "differentiation_method__smoother_kws": [
        {'window_length': 10, 'polyorder': 2},
        {'window_length': 15, 'polyorder': 3},
        {'window_length': 20, 'polyorder': 3},
    ]
}

tscv = TimeSeriesSplit(n_splits=5)
search = GridSearchCV(model, param_grid, cv=tscv)

base = SINDyWrapper(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    dt=dt
)

search = GridSearchCV(base, param_grid, cv=tscv)
search.fit(X)
print(search.best_params_)
best = search.best_estimator_.model_
best.print()
print(search.best_estimator_.score(X))

model = ps.SINDy(
    differentiation_method=ps.TotalVariationRegularizedDerivative(kind='small', alpha=1e-3, order=2),
    feature_library=ps.PolynomialLibrary(degree=1),
    optimizer=ps.STLSQ(threshold=0.005))
model.fit(X, t=dt)
model.print()
    

def rhs(t, z):
    return model.predict(z.reshape(1, -1))[0]
z0 = X[0]
t_eval = np.linspace(*tspan, 200)
sol = solve_ivp(rhs, tspan, z0, t_eval=t_eval)
plt.plot(t, x_noisy, 'b', label='x_true')
plt.plot(t, y_noisy, 'r', label='y_true')
plt.plot(sol.t, sol.y[0], 'b-', label='x_pred')
plt.plot(sol.t, sol.y[1], 'r-', label='y_pred')
plt.legend()
plt.show()

dXdt_noisy = ps.SmoothedFiniteDifference()._differentiate(X, t=dt)

# Predicted derivatives from true ODE
def true_derivatives(X):
    x = X[:,0]
    y = X[:,1]
    dxdt = x
    dydt = x - y
    return np.column_stack([dxdt, dydt])

dXdt_pred = true_derivatives(X)

score = -np.mean((dXdt_noisy - dXdt_pred)**2)
print("Derivative-based score of true ODE on noisy data:", score)
