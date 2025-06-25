import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import plotsetting as ps

# Load data
Nmin = 7
Nmax = 7
Nstate = 1
parity = 0
Rad = 4.0
dtype = np.float64

data_arr = [
    [
        np.loadtxt(f"3d_terr{i}_N={N}_par={parity}.dat", dtype=dtype)
        for N in range(Nmin, Nmax + 1)
    ]
    for i in range(Nstate)
]

# 假設 exact 是你提供的 dict，例如：
'''
exact = {
    0: [-1.12239818, -0.22764680, -0.18946912],
    1: [-1.42007881, -0.22233476, -0.20311525],
    2: [-1.64473609, -0.22119796, -0.21221143],
    3: [-1.79558341, -0.21930881, -0.21958344],
}
'''

def sub_label(index):
    return f"({chr(ord('a') + index)})"

#def plot_energy(ax, data, Nsite, Nstate, exact, sub_lab):
def plot_energy(ax, data, Nsite, Nstate, sub_lab):
    label = ['GS'] + ['FS'] * (Nstate - 1)
    colors = ps.get_default_colors()
    #ax.set_ylim(exact[Nsite][0]-0.001, exact[Nsite][0]+0.1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlabel(r'$\mathrm{Step}$')
    ax.set_ylabel(r'$E$')
    ax.set_yscale('log')

    ax.plot(range(len(data[0][Nsite][2])), np.real(data[0][Nsite][2]) - np.real(data[0][Nsite][2][-1]),
            color=colors[0], ls="solid", label=label[0])
    #ax.axhline(y=exact[Nsite][0], color='black', linestyle='--', linewidth=1.5)

    ps.text(ax, x=0.1, y=0.9, t=sub_lab, fontsize=22)
    ps.set(ax)

    # Inset: 繪製其他態
    #ax_inset = ax.inset_axes([0.4, 0.4, 0.58, 0.58])
    #ax_inset.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax_inset.xaxis.set_major_locator(MaxNLocator(nbins=5))
    #ax_inset.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    #ax_inset.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    #for i in range(1, Nstate):
    #    ax_inset.plot(range(len(data[i][Nsite][2])), np.real(data[i][Nsite][2]),
    #                  color=colors[i], ls="solid", label=label[i])
        #ax_inset.set_ylim(-0.23,-0.07)
    #ax_inset.axhline(y=exact[Nsite][1], color='black', linestyle='--', linewidth=1.5)
    #ax_inset.axhline(y=exact[Nsite][2], color='gray', linestyle='--', linewidth=1.5)
    #ps.set(ax_inset, fontsize=16)

# Plotting
fig = plt.figure(figsize=(20, int(20/(Nmax - Nmin + 1))))
for i in range(Nmax - Nmin + 1):
    ax1 = plt.subplot(1, Nmax-Nmin +1, i + 1)
    plot_energy(ax1, data_arr, i, Nstate, sub_label(i))
    #plot_energy(ax1, data_arr, i, Nstate, exact, sub_label(i))

plt.tight_layout()
plt.savefig("2d_dmrg_E.pdf", format='pdf')
plt.show()
