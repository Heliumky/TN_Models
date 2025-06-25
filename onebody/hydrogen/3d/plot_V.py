import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Function to calculate the potential V(x)
def func(x, cutoff):
    return np.where(
        np.abs(x) < cutoff, 
        -1 / (0.5 * cutoff),  # Approximation for small r
        -1 / np.abs(x)        # Standard form for larger r
    )

# Initial parameters
shift_init = 0
N_init = 4  # Exponent: number of points will be 2^N
cutoff_init = 0.1

# Create the figure and initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

# Generate initial data
num_points = 2**N_init
x = np.linspace(shift_init, -shift_init, num_points)
dx = np.abs(x[1] - x[0])
line, = ax.plot(x, func(x, cutoff=dx), 'o', markersize=5, label='V(x)')

# Customize the plot
ax.set_xlabel('x')
ax.set_ylabel('V(x)')
ax.set_title(f"Potential V(x) with shift={shift_init} and N=2^{N_init} ({num_points} points)")
ax.legend()
ax.grid(True)

# Add sliders for shift and N
ax_shift = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_N = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_shift = Slider(ax_shift, 'Shift', -20, 20, valinit=shift_init, valstep=1)
slider_N = Slider(ax_N, 'N (Exponent)', 1, 20, valinit=N_init, valstep=1)

# Update function for sliders
def update(val):
    shift = slider_shift.val
    N = int(slider_N.val)  # Ensure N is an integer
    num_points = 2**N
    x = np.linspace(shift, -shift, num_points)
    dx = np.abs(x[1] - x[0])
    line.set_xdata(x)
    line.set_ydata(func(x, cutoff=dx))
    ax.set_title(f"Potential V(x) with shift={shift} and N=2^{N} ({num_points} points)")
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_shift.on_changed(update)
slider_N.on_changed(update)

plt.show()

