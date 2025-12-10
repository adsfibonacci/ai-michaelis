import numpy as np
import pysindy as ps
from pysr import PySRRegressor
from matplotlib import pyplot as plt

# --- Generate synthetic data ---
t = np.sort(np.random.rand(200))  # random time samples
x = 3 * np.exp(-2*t)              # linear dynamics for x
y = 0.5 * np.exp(t)               # linear dynamics for y
z = np.sin(t) + 0.1*x*y           # add nonlinear interaction for demo
X = np.stack((x, y, z), axis=-1)

# --- Step 1: Fit SINDy (sparse discovery) ---
diff_method = ps.SmoothedFiniteDifference()
model = ps.SINDy(
    differentiation_method=diff_method,
    feature_library=ps.PolynomialLibrary(degree=3),
    optimizer=ps.STLSQ(threshold=0.05)  # threshold controls sparsity
)
model.fit(X, t=t, feature_names=["x", "y", "z"])
print("SINDy model:")
model.print()

# --- Step 2: Estimate derivatives using SINDy differentiation method ---
dx_dt = model.differentiation_method._differentiate(X, t)

# --- Step 3: Fit PySR symbolic regression for each variable ---
symbolic_models = {}
for i, var in enumerate(["x", "y", "z"]):
    pysr_model = PySRRegressor(
        niterations=200,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log"],
        populations=40,
        weight_randomize=0.1,
        model_selection="best"             # pick the simplest model with good fit
    )
    pysr_model.fit(X, dx_dt[:, i])
    symbolic_models[var] = pysr_model
    pass
print(f"\nSymbolic regression for d{var}/dt:")
print(pysr_model)
