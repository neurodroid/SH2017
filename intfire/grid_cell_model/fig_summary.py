from stfio import plot as stfio_plot
from scipy.io import loadmat
import matplotlib.pyplot as plt

nonmda = loadmat("output_000nmda_nonmda50_3.00gext_10000.0placeT/job0000_dump000_results.mat")
nmdaconst = loadmat("output_001nmda_nonmda50_2.30gext_10000.0placeT/job0000_dump000_results.mat")
nmda = loadmat("output_001nmda_-50.00nmda50_2.30gext_10000.0placeT/job0000_dump000_results.mat")

fig = plt.figure(figsize=(4,12))
ax1 = stfio_plot.StandardAxis(fig, 211, hasx=False, hasy=True)
ax2 = stfio_plot.StandardAxis(fig, 212, hasx=False, hasy=True)
xposs = [1,2,3]
cols = ['k', 'r', 'r']
yposs = [nonmda['vel_error'], nmdaconst['vel_error'], nmda['vel_error']]
for xpos, col, ypos in zip(xposs, cols, yposs):
    ax1.bar(xpos, ypos,
            edgecolor='none', color=col)

yposs = [nonmda['gridness'], nmdaconst['gridness'], nmda['gridness']]
for xpos, col, ypos in zip(xposs, cols, yposs):
    ax2.bar(xpos, ypos,
            edgecolor='none', color=col)

plt.savefig('fig_summary.pdf')
plt.show()
