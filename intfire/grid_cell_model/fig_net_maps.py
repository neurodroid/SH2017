#
#   fig_spike_maps.py
#
#   Spike and rate maps of grid cells. Data analysis
#
#       Copyright (C) 2012  Lukas Solanka <l.solanka@sms.ed.ac.uk>
#       
#       This program is free software: you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation, either version 3 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function

import sys
import os

import numpy as np

from scipy.io           import loadmat
from scipy.optimize     import brent
from scipy.io           import savemat
from matplotlib.pyplot  import *
from numpy.fft          import fft2

from grid_cell_analysis import *

cwd = os.path.expanduser("~/Dropbox/grid/repo")
sys.path.append(cwd)
sys.path.append(cwd + "/../../Documents/eval/invivo/tools")

import spectral
from stfio import plot as stfio_plot

jobRange = [0, 0]
dumpNum = 0

jobN = jobRange[1] - jobRange[0] + 1

rcParams['font.size'] = 14


arenaDiam = 180.0     # cm
h = 3.0

# Neuron to extract spikes from
neuronNum = 128
spikeType = 'excitatory'


parameters = {}
parameters['time']              = 599.9e3      # ms
parameters['placeT']            = 10e3       # ms, orig: 1e3
parameters['gext_e_const']      = 3.0        # nS
parameters['NMDA_amount_ext']   = 0.00       # %
parameters['NMDA_half']         = -999.0      # mV

if parameters['placeT'] != 1e3:
    placeTString = "_{0}placeT".format(parameters['placeT'])
else:
    placeTString = ""

if parameters['NMDA_half'] >= -500.0:
    nmda50String = "_%.2fnmda50" % parameters['NMDA_half']
else:
    nmda50String = "_nonmda50"

parameters['output_dir'] = "output_%03dnmda%s_%.2fgext%s/" % (
    parameters['NMDA_amount_ext'], nmda50String, parameters['gext_e_const'], placeTString)

dirName = parameters['output_dir']
fileNamePrefix = ''
fileNameTemp = "{0}/{1}job{2:04}_dump{3:03}"

gridnessScores = []

def unwrap_net(net_1d, net_size):
    vel = np.diff(net_1d)
    vel[np.abs(vel)>net_size*0.3] = 0
    return np.concatenate([[0], np.cumsum(vel)])

def fit_tracks(net_xy, pos_xy):
    net_dxy = np.diff(net_xy, axis=1)
    pos_dxy = np.diff(pos_xy, axis=1)
    min_dscale = lambda x: np.linalg.norm(x*net_dxy - pos_dxy, axis=0)
    min_dscale_sum = lambda x: np.sum(np.linalg.norm(x*net_dxy - pos_dxy, axis=0))
    min_x = brent(min_dscale_sum, brack=(0, 20.0))
    sum_v = min_dscale(min_x)
    return min_x, sum_v

for job_it in range(jobN):
    jobNum = job_it + jobRange[0]
    print('jobNum: ' + str(jobNum))

    fileName = fileNameTemp
    fileName = fileName.format(dirName, fileNamePrefix, jobNum, dumpNum)
    try:
        data = loadmat(fileName +  '_output.mat')
    except:
        print("warning: could not open: " + fileName)
        continue

    pos_x           = data['pos_x'].ravel()
    pos_y           = data['pos_y'].ravel()
    rat_dt          = data['dt'][0][0]
    velocityStart   = data['velocityStart'][0][0]
    if spikeType == 'excitatory':
        spikeTimes  = data['spikeCell_e'].ravel()
    if spikeType == 'inhibitory':
        spikeTimes  = data['spikeCell_i'].ravel()

    gridSep         = data['options']['gridSep'][0][0][0][0]
    corr_cutRmin    = gridSep / 2

    net_t           = data['Fe_times'][0]
    net_dt          = net_t[1]-net_t[0]
    net_x           = -(unwrap_net(data['Fe_path'][:,1], data['Ne_xy'][0][1]))[net_t > velocityStart*1e-3]
    net_y           = -(unwrap_net(data['Fe_path'][:,0], data['Ne_xy'][0][0]))[net_t > velocityStart*1e-3]

    net_x = spectral.lowpass(stfio_plot.Timeseries(np.array(net_x, dtype=np.float), net_dt), 0.01/net_dt).data
    net_y = spectral.lowpass(stfio_plot.Timeseries(np.array(net_y, dtype=np.float), net_dt), 0.01/net_dt).data
    ipos_end = (net_t[-1]-velocityStart*1e-3)/rat_dt
    assert(np.allclose(np.diff(net_t), net_dt))
    sub_sample = (net_t[1]-net_t[0])/rat_dt
    assert(int(sub_sample)-sub_sample < 1e-6)
    sub_sample = int(sub_sample)
    min_x, sum_v = fit_tracks(np.array([net_x, net_y]),
                              np.array([pos_x[:ipos_end:sub_sample],
                                        pos_y[:ipos_end:sub_sample]]))
    print("Velocity error: ", (sum_v**2).mean())
    res_dict = {}
    res_dict['vel_error'] = (sum_v**2).mean()
    
    figure(figsize=(8,8))
    plot(min_x*(net_x-net_x[0]), min_x*(net_y-net_y[0]))
    # plot(net_x-net_x[0], net_y-net_y[0])
    plot(pos_x[:ipos_end]-pos_x[0], pos_y[:ipos_end]-pos_y[0])
    axis('equal')
    # axis('off')
    savefig(fileName + '_paths' + '.pdf')
    
    spikes = spikeTimes[neuronNum] - velocityStart*1e-3
    spikes = np.delete(spikes, np.nonzero(spikes < 0)[0])

    figure()
    plotSpikes2D(spikes, pos_x, pos_y, rat_dt)
    savefig(fileName + '_spikePlot_' + spikeType + '.pdf')

    figure()
    rateMap, xedges, yedges = SNSpatialRate2D(spikes, pos_x, pos_y, rat_dt, arenaDiam, h)
    X, Y = np.meshgrid(xedges, yedges)
    pcolormesh(X, Y, rateMap)
    colorbar()
    axis('equal')
    axis('off')
    savefig(fileName + '_rateMap_' + spikeType + '.pdf')

    
    figure()
    FT_size = 256
    Fs = 1.0/(h/100.0) # units: 1/m
    rateMap_pad = np.ndarray((FT_size, FT_size))
    rateMap_pad[:, :] = 0
    rateMap_pad[0:rateMap.shape[0], 0:rateMap.shape[0]] = rateMap - np.mean(rateMap.flatten())
    FT = fft2(rateMap_pad)
    fxy = np.linspace(-1.0, 1.0, FT_size)
    fxy_igor = Fs/2.0*np.linspace(-1.0, 1.0, FT_size+1)
    FX, FY = np.meshgrid(fxy, fxy)
    FX *= Fs/2.0
    FY *= Fs/2.0
    PSD_centered = np.abs(np.fft.fftshift(FT))**2
    pcolormesh(FX, FY, PSD_centered)
    #axis('equal')
    xlim([-10, 10])
    ylim([-10, 10])
    savefig(fileName + '_fft2' + spikeType + '.pdf')


    figure()
    corr, xedges_corr, yedges_corr = SNAutoCorr(rateMap, arenaDiam, h)
    X, Y = np.meshgrid(xedges_corr, yedges_corr)
    pcolormesh(X, Y, corr)
    axis('equal')
    axis('off')
    savefig(fileName + '_rateCorr_' + spikeType + '.pdf')


    figure()
    G, crossCorr, angles = cellGridnessScore(rateMap, arenaDiam, h, corr_cutRmin)
    gridnessScores.append(G)
    plot(angles, crossCorr)
    xlabel('Angle (deg.)')
    ylabel('Corr. coefficient')
    savefig(fileName + '_gridnessCorr_' + spikeType + '.pdf')

    show()
    close('all')

print("Gridness scores:")
print(gridnessScores)

res_dict['gridness'] = gridnessScores
savemat(fileName +  '_results.mat', res_dict)
