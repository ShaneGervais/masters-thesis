import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{physics}"
})


# Function definition
def weak_value_real(theta, V):
    return np.cos(theta)**2 + 2 * V * np.sin(theta) * np.cos(theta)

# Domain
theta = np.linspace(0, 2 * np.pi, 500)

# Visibility values
V_values = [1, 0.5, 0.1, 0.01]
colors = ['blue', 'orange', 'green', 'red']
labels = [f'Visibilité = {V}' for V in V_values]

# Create figure
plt.figure(figsize=(8, 6))
for V, color, label in zip(V_values, colors, labels):
    plt.plot(theta, weak_value_real(theta, V), label=label, color=color)

# Bold text label but normal math formatting for symbols and units
plt.xlabel(r"\textbf{État d’entrée } $|\psi(\theta)\rangle$ (degré)", fontsize=14)
plt.ylabel(r"\textbf{Valeur faible } $\mathcal{R}(\expval{\hat{\pi}}_W)$ (u.a.)", fontsize=14)

plt.legend()
plt.grid(True)
plt.tight_layout()

# Save high-res image
plt.savefig("real_part_visibility.png", dpi=300)

plt.show()
