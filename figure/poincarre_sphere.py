import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Define axes limits (from -1 to 1 for all Stokes parameters)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Set axis labels with LaTeX formatting and ensure S_3 is properly positioned
ax.set_xlabel(r'$\mathbf{S_1}$', fontsize=18, fontweight='bold', labelpad=15)
ax.set_ylabel(r'$\mathbf{S_2}$', fontsize=18, fontweight='bold', labelpad=15)
ax.set_zlabel(r'$\mathbf{S_3}$', fontsize=18, fontweight='bold', labelpad=15)

# Generate smooth sphere surface
u = np.linspace(0, 2 * np.pi, 150)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Apply soft aesthetic color with transparency
ax.plot_surface(x, y, z, color='lightpink', alpha=0.4, edgecolor='none')

# Draw thick coordinate axes with labeled ticks
axis_linewidth = 4  # Make axes very clear
ax.plot([-1, 1], [0, 0], [0, 0], 'k-', lw=axis_linewidth)  # S1 axis
ax.plot([0, 0], [-1, 1], [0, 0], 'k-', lw=axis_linewidth)  # S2 axis
ax.plot([0, 0], [0, 0], [-1, 1], 'k-', lw=axis_linewidth)  # S3 axis

# Set tick values for better readability
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_zticks([-1, -0.5, 0, 0.5, 1])

# Ensure S_3 is properly visible by adjusting view angle
ax.view_init(elev=20, azim=30)

# Adjust aspect ratio to make the sphere appear correctly
ax.set_box_aspect([1,1,1])

# Basis states at extreme points, with more spacing for better readability
label_offset = 0.3  # Increased offset to prevent overlap
points = {
    r'$\mathbf{|H\rangle}$': (1 + label_offset, 0, 0),
    r'$\mathbf{|V\rangle}$': (-1 - label_offset, 0, 0),
    r'$\mathbf{|D\rangle}$': (0, 1 + label_offset, 0),
    r'$\mathbf{|A\rangle}$': (0, -1 - label_offset, 0),
    r'$\mathbf{|R\rangle}$': (0, 0, 1 + label_offset),
    r'$\mathbf{|L\rangle}$': (0, 0, -1 - label_offset),
}

# Plot labels with high contrast and increased spacing
for label, (px, py, pz) in points.items():
    ax.text(px, py, pz, label, fontsize=20, fontweight='bold', color='black', ha='center', va='center')

# Add a circular blue path demonstrating the path of a turning half-wave plate
num_points = 100
angles = np.linspace(0, 2 * np.pi, num_points)
circle_x = np.sin(angles)  # Keep the path in the S2-S3 plane
circle_y = np.cos(angles)
circle_z = np.zeros_like(angles)
ax.plot(circle_x, circle_y, circle_z, 'b-', linewidth=3)

# Add an arrow indicating direction
arrow_idx = num_points // 4  # Place the arrow at 1/4th of the path
ax.quiver(circle_x[arrow_idx], circle_y[arrow_idx], circle_z[arrow_idx],
          circle_x[arrow_idx + 1] - circle_x[arrow_idx],
          circle_y[arrow_idx + 1] - circle_y[arrow_idx],
          circle_z[arrow_idx + 1] - circle_z[arrow_idx],
          color='blue', linewidth=3, arrow_length_ratio=0.2)
arrow_idx = num_points // 4  # Place the arrow at 1/4th of the path






# Save figure with 600 dpi resolution
plt.savefig("poincare_sphere_HDVA.png", dpi=300)

# Show figure
plt.show()
