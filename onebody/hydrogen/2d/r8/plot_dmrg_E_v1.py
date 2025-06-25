import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import plotsetting as ps

# Load data

N = 8

E0 = np.loadtxt(f"2d_terr0_N={N}.dat", dtype=np.complex128)
E1 = np.loadtxt(f"2d_terr1_N={N}.dat", dtype=np.complex128)


# Create figure
fig, ax = plt.subplots()

# Main plot settings
ax.relim()
ax.autoscale_view()
#ax.tick_params(axis='both', direction='in', length=6, width=0.7, grid_color='black', grid_alpha=0.5)
ax.set_ylim(-2.1,0)
#ax.set_xlim(0,8*1e3)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))      
ax.set_xlabel(r'$\mathrm{Step}$')
ax.set_ylabel(r'$E$')

# Main plot curves
ax.plot(range(len(E0[0])), np.real(E0[1]), color='red',ls = "solid", label=r"GS decay")
ax.axhline(y=-2, color='red', linestyle='--', linewidth=1.5, label=r"Exact GSE=-2")
# Adjusted legend position (upper left to avoid overlapping)
#ax.legend(bbox_to_anchor=(0.6, 1), fontsize=14)

# Add text (b)
ps.text(ax, x=0.1, y=0.9, t="(b)", fontsize=22)

# Create inset plot
ax_inset = ps.new_panel(fig, left=0.45, bottom=0.4, width=0.52, height=0.52)
#ax_inset.tick_params(axis='both', direction='in', length=4, width=0.5)

# Ensure more x-ticks in the inset
ax_inset.yaxis.set_major_locator(MaxNLocator(integer=True))
ax_inset.xaxis.set_major_locator(MaxNLocator(nbins=5)) 
ax_inset.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_inset.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#ax_inset.set_ylim(-0.23,-0.12)
#ax_inset.set_xlim(0,8*1e3)
# Inset plot curves
ax_inset.plot(range(len(E1[0])), np.real(E1[1]), color='black',ls = "solid", label=r"FS decay")
ax_inset.axhline(y=-0.2222, color='black', linestyle='--', linewidth=1.5, label=r"$Exact FSE-2/9$")
# Inset labels
#ax_inset.set_xlabel(r'$\mathrm{Step}$', fontsize=22)
#ax_inset.set_ylabel(r'$E$', fontsize=22)

# Adjusted legend position (outside, top right)
#ax_inset.legend(loc="upper right", fontsize=8, frameon=False)

# Add text (c)
#ps.text(ax_inset, x=0.1, y=0.9, t="(c)", fontsize=22)

# Apply plot settings
ps.set(ax)
ps.set(ax_inset, fontsize=22)

# Save and show
plt.savefig("DMRGE_step.pdf", transparent=False)
plt.show()
