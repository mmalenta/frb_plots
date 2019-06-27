import matplotlib
matplotlib.use('Agg')

import glob
import json
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import struct
import threading
import time
import sys

from astropy.time import Time

# This class does the actual plotting
class Plotter:
    
    def __init__(self, timing, verbose, outdir = './'):
        # General managment
        self._plotdir = outdir
        self._timing = timing
        self._verbose = verbose

        # Filterbank file parameters
        self._nchans = 4096
        self._tsamp = 306.24299e-06 # in s
        self._ftop = 1712.104492
        self._fband = -0.208984
        self._fbottom = self._ftop + self._nchans * self._fband

        # Extracted candidate plot parameters
        self._timeavg = 16
        self._freqavg = 16
        self._outdir = outdir
        self._time_width = 8 # How many time samples we want to have across the averaged profile

        # .spccl plots parameters
        # Plot last 120 seconds of data
        self._spccl_refresh_s = 30.0
        self._spccl_length_s = 120.0
        self._spccl_length_mjd = self._spccl_length_s / 86400.0
        self._spccl_pad = 2.5 / 86400.0
        self._beam_colours = ['firebrick', 'deepskyblue', 'darkorange', 'limegreen', 'purple', 'darkgrey']

        # Dedispersion parameters
        self._disp_const = 4.15e+03 * (1.0 / (self._fbottom * self._fbottom) - 1.0 / (self._ftop * self._ftop)) # in s per unit DM
        self._cand_pad_s = 0.5
        self._dedisp_pad_s = self._cand_pad_s / 2.0
        self._dedisp_bands = int(self._nchans / self._freqavg)

        if self._verbose:
            print("Starting the plotter up")
    
    def Dedisperse(self, inputdata, dm, outbands):
    
        dm_start = time.time()

        perband = int(self._nchans / outbands)
        print(perband)
        padding = 0
        sampout = 0
        
        lastbandtop = self._ftop + (outbands - 1) * perband * self._fband
        lastbandbottom = lastbandtop + perband * self._fband

        print(lastbandtop)
        print(lastbandbottom)

        largestsampdelay = int(np.ceil(4.15e+03 * dm * (1.0 / (lastbandbottom * lastbandbottom) - 1.0 / (lastbandtop * lastbandtop)) / self._tsamp))
        #largestsampdelay = int(np.ceil(largestdelay / self._tsamp))
        
        if (outbands == 1):
            padding = np.floor(self._dedisp_pad_s / self._tsamp)
            sampout = int(2 * padding)
        else:
            padding = 0
            sampout = int(inputdata.shape[1] - largestsampdelay)

        print(sampout)
        
        print(largestsampdelay)
        print(sampout)
        print(inputdata.shape)

        if ((int(padding) + largestsampdelay + sampout) > inputdata.shape[1]):
            print("We got something wrong: need more samples than there is in the data")
            padding = 0
            sampout =  int(inputdata.shape[1] - largestsampdelay)

        dedispersed = np.zeros((outbands, sampout))

        for band in np.arange(outbands):
            bandtop = self._ftop + band * perband * self._fband
            for chan in np.arange(perband):
                chanfreq = bandtop + chan * self._fband
                delay = int(np.ceil(4.15e+03 * dm * (1.0 / (chanfreq * chanfreq) - 1.0 / (bandtop * bandtop)) / self._tsamp))
                dedispersed[band, :] = np.add(dedispersed[band, :], inputdata[chan + band * perband, int(padding) + delay : int(padding) + delay + sampout])
        
            '''
            if ((int(padding) + fullsampdelay + sampout) < inputdata.shape[1]):
                
                for chan in np.arange(perband):
                    chanfreq = self._ftop + chan * self._fband
                    delay = int(np.round(4.15e+03 * dm * (1.0 / (chanfreq * chanfreq) - 1.0 / (self._ftop * self._ftop)) / self._tsamp))
                    dedispersed[band, :] = np.add(dedispersed[band, :], inputdata[chan, int(padding) + delay : int(padding) + delay + sampout])
            '''     

        #else:
        #    print("We got something wrong: need more samples than there is in the data")


        dm_end = time.time()
        
        if (self._verbose):
            print("Dedispersion took %.2fs" % (dm_end - dm_start))

        return dedispersed       

    def PlotSpcclCands(self, candidates):

        last_mjd = candidates['MJD'].values[-1]
        first_mjd = last_mjd - self._spccl_length_mjd

        candidates_to_plot = candidates.loc[(candidates['MJD'] >= first_mjd) & (candidates['MJD'] <= last_mjd)]

        cmap = matplotlib.colors.ListedColormap(self._beam_colours)
        bound = [0, 1, 2, 3 , 4, 5, 6]
        
        norm = matplotlib.colors.BoundaryNorm(bound, cmap.N, clip=True)
    
        print("Plotting the full candidates")
        
        figspccl = plt.figure(figsize=(15,10))
        axspccl = figspccl.gca()
        axspccl.ticklabel_format(useOffset=False)
        candsc = axspccl.scatter(x=candidates_to_plot['MJD'], y=candidates_to_plot['DM'] + 1, s=candidates_to_plot['SNR'], c=candidates_to_plot['Beam'], norm=norm, cmap=cmap)
        axspccl.set_xlim(first_mjd - self._spccl_pad, last_mjd + self._spccl_pad)
        axspccl.set_xlabel('MJD', fontsize=14, weight='bold')
        axspccl.set_ylabel('DM', fontsize=14, weight='bold')
        axspccl.set_yscale('log')
        sccbar = figspccl.colorbar(candsc, ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        sccbar.set_label('Beam', fontsize=14, weight='bold')
        sccbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '5'])
        figspccl.savefig(self._plotdir + '/full_spccl_candidates.png')
        plt.close(figspccl)

        time.sleep(self._spccl_refresh_s)
    
    def PlotExtractedCand(self, beam_dir, filename, headsize, nchans, properties, ibeam=0, nodebeam=0):
        
        mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # How many time samples we have across the pulse width
        pulse_samples = properties['Width'] * 1e-03 / self._tsamp

        if (self._verbose):
            print("%d samples across pulse of %fms" % (pulse_samples, properties['Width']))
        
        # Don't average if we have less than 16 samples
        if pulse_samples < 16:
            self._timeavg = 1
        # Otherwise average
        else:
            self._timeavg = int(np.floor(pulse_samples / self._time_width))
        
        
        # Read original data
        fildata = np.reshape(np.fromfile(beam_dir + filename, dtype='B')[headsize:], (-1, nchans)).T
        print("Read %d time samples" % (fildata.shape[1]))
        fildata = fildata * mask[:, np.newaxis]
        # Time average the data
        
        # Frequency average the data
        frequencies = int(np.floor(fildata.shape[0] / self._freqavg) * self._freqavg)
        #filfreqavg = fildata[:frequencies, :timesamples].reshape(-1, self._freqavg, timesamples).sum(axis=1)
        filfreqavg = self.Dedisperse(fildata, properties['DM'], self._dedisp_bands)

        # Both averages
        timesamples = int(np.floor(filfreqavg.shape[1] / self._timeavg) * self._timeavg)                
        filbothavg = filfreqavg[:, :timesamples].reshape(filfreqavg.shape[0], (int)(timesamples / self._timeavg), self._timeavg).sum(axis=2) / self._timeavg / self._freqavg
        
        #notzero = (filbothavg[:,0])[np.where(filbothavg[:,0]!=0)[0]]
        
        filband = np.mean(filbothavg, axis=1)
        filbothavg = filbothavg - filband[:, np.newaxis]

        datamean = np.mean(filbothavg[:, 0])
        datastd = np.std(filbothavg[:, 0])        

        ctop = int(np.ceil(datamean + 1.25 * datastd))
        cbottom = int(np.floor(datamean - 0.5 * datastd))
        
        fmt = lambda x: "{:.2f}".format(x)
        
        # Prepare the frequency ticks
        avg_freq_pos = np.linspace(0, int(nchans / self._freqavg), num=9)
        avg_freq_pos[-1] = avg_freq_pos[-1] - 1       
        avg_freq_label = self._ftop + avg_freq_pos * self._fband * self._freqavg
        avg_freq_label_str = [fmt(label) for label in avg_freq_label]
        
        # Prepare the time ticks
        avg_time_pos = np.linspace(0, (int)(timesamples / self._timeavg), num=5)
        avg_time_label = avg_time_pos * self._tsamp * self._timeavg
        avg_time_label_str = [fmt(label) for label in avg_time_label]
        
        cmap = 'binary'
        
        fil_fig, fil_axis = plt.subplots(2, 1, figsize=(11.69,8.27), dpi=75)
        fil_fig.tight_layout(h_pad=3.25, rect=[0, 0.03, 1, 0.95])
        
        axboth = fil_axis[0]
        axboth.imshow(filbothavg, interpolation='none', vmin=cbottom, vmax=ctop, aspect='auto', cmap=cmap)
        axboth.set_title('Time (' + str(self._timeavg) + '), freq (' + str(self._freqavg) + ') avg', fontsize=11)
        axboth.set_xlabel('Time [s]', fontsize=10)
        axboth.set_ylabel('Frequency [MHz]', fontsize=10)
        axboth.set_xticks(avg_time_pos)
        axboth.set_xticklabels(avg_time_label_str, fontsize=8)
        axboth.set_yticks(avg_freq_pos)
        axboth.set_yticklabels(avg_freq_label_str, fontsize=8)        

        # Dedisperse the original filterbank data
        dedispersed = self.Dedisperse(fildata, properties['DM'], 1)
        
        # Average the dedispersed time series
        dedisp_avg_time = int(np.floor(dedispersed.shape[1] / self._timeavg) * self._timeavg)
        print(dedispersed.shape)
        print(dedisp_avg_time)
        dedisp_avg = dedispersed[0, :dedisp_avg_time].reshape(1, int(dedisp_avg_time / self._timeavg), self._timeavg).sum(axis=2)

        dedisp_time_pos = np.linspace(0, int(dedisp_avg_time / self._timeavg), num=9)
        print(dedisp_time_pos)
        dedisp_time_label = dedisp_time_pos * self._tsamp * self._timeavg + self._dedisp_pad_s
        dedisp_time_label_str = [fmt(label) for label in dedisp_time_label]
        
        axdedisp = fil_axis[1]
        #axdedisp.plot(dedispersed[0, :], linewidth=0.4, color='black')
        axdedisp.plot(dedisp_avg[0, :], linewidth=0.4, color='black')
        fmtdm = "{:.2f}".format(properties['DM'])
        axdedisp.set_title('Dedispersed time series, DM ' + fmtdm)
        axdedisp.set_xticks(dedisp_time_pos)
        axdedisp.set_xticklabels(dedisp_time_label_str)
        axdedisp.set_xlabel('Time [s]')
        axdedisp.set_ylabel('Power [arbitrary units]')
        
        if (np.sum(dedispersed) == 0):
            axdedisp.text(0.5, 0.6, 'Not dedispersed - please report!', fontsize=14, weight='bold', color='firebrick',  horizontalalignment='center', verticalalignment='center', transform=axdedisp.transAxes)
        
        plotdir = self._outdir + '/beam0' + str(nodebeam) + '/Plots/'
        
        fil_fig.savefig(plotdir + str(properties['MJD']) + '_DM_' + fmtdm + '_beam_' + str(ibeam) + '.jpg', bbox_inches = 'tight', quality=75)
        
        plt.close(fil_fig)

    # This might do something in the future
    def PlotDist(self, filename, selected):
        print("Empty")

# This is an overarching class that watches the directories and detects any changes
class Watcher:
    
    def __init__(self, indir, eventsfile, timing=True, verbose=False):
        self._events_file = eventsfile
        self._directory = indir
        self._timing = timing
        self._verbose = verbose
        self._watching = True
        self._nchans = 4096
        self._headsize = 136
        self._nbeams = 6
        self._header_names = ['MJD', 'DM', 'Width', 'SNR']
        self._start_time = time.time()
        self._mjd_pad = 1.0 / 86400.0
        self._beam_info = pd.DataFrame()
        
        self._plot_length = 120.0 # how many seconds of data to plot
        
        if self._verbose:
            print("Starting the watcher up")
            print("Watching directory", self._directory)
            if self._timing:
                print("Enabling timing")
        
        self._plotter = Plotter(self._timing, self._verbose, self._directory)
    
    def GetNewSpcclFiles(self):
        if self._verbose:
            print("Watching for .spccl files")
    
        old_files = []
        total_cands = np.zeros(6)
        new_cands = np.zeros(6)
        
        full_cands = pd.DataFrame()
        all_beams = False
    
        while self._watching:
            
            for ibeam in np.arange(self._nbeams):
                beam_dir = self._directory + '/beam0' + str(ibeam) + '/'
                
                if os.path.isdir(beam_dir):

                    cand_file = glob.glob(beam_dir + '/*.spccl')
                    print(cand_file)
                    if (len(cand_file) != 0):
                        cand_file = cand_file[0]
                        if (self._verbose):
                            print("Found candidate file for beam %d" % (ibeam))
                            print(cand_file)

                        skipcands = int(1 + total_cands[ibeam])
                        print(skipcands)
                        beam_cands = pd.read_csv(cand_file, sep='\s+', names=self._header_names, skiprows=skipcands)
                        beam_cands['Beam'] = ibeam
                        new_cands[ibeam] = beam_cands.shape[0]
                        total_cands[ibeam] = total_cands[ibeam] + new_cands[ibeam]
                        
                        full_cands = full_cands.append(beam_cands)
                        
                        if (self._verbose):
                            print("Found %d new candidates for beam %d" % (new_cands[ibeam], ibeam))
                            print("Total of %d candidates for beam %d" % (total_cands[ibeam], ibeam))
                else:
                    if (self._verbose):
                        print("No directory %s" % (beam_dir))
            self._plotter.PlotSpcclCands(full_cands)
            
    
    def GetNewFilFiles(self):
        if self._verbose:
            print("Watching for .fil files")
        
        fil_latest = np.zeros(6)
        fil_latest_mjd = np.zeros(6)
        new_fil_files = []
        
        
        while self._watching:
            
            start_plot = time.time()
            print (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            
            for ibeam in np.arange(self._nbeams):
                
                beam_dir = self._directory + '/beam0' + str(ibeam) + '/'
                new_fil_files = []

                if os.path.isdir(beam_dir):

                    full_beam = self._beam_info['beam'].values[ibeam]
                    beam_ra = self._beam_info['ra'].values[ibeam]
                    beam_dec = self._beam_info['dec'].values[ibeam]
                    
                    fil_files = os.scandir(beam_dir)
                    for ff in fil_files:
                        if ((ff.name.endswith('fil')) & (ff.stat().st_mtime > fil_latest[ibeam])):
                            new_fil_files.append([ff.name, ff.stat().st_mtime])

                    new_len = len(new_fil_files)
                    print (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print("Found %d new filterbank files for beam %d" % (new_len, ibeam))

                    if new_len > 0:

                        # Will hopefully get .spccl written
                        fil_latest[ibeam] = max(new_fil_files, key = lambda nf: nf[1])[1]

                        latest_fil_mjd = 0.0
                        for new_ff in new_fil_files:
                            with open(beam_dir + new_ff[0], mode='rb') as file:
                                # Unused, but vital to skipping first 114 bytes
                                skip_head = file.read(114)
                                mjdtime = struct.unpack('d', file.read(8))[0]
                                if mjdtime > latest_fil_mjd:
                                    latest_fil_mjd = mjdtime

                        print("Latest .fil file MJD: %.10f" % (latest_fil_mjd))

                        cand_dir = self._directory + '/beam0' + str(ibeam) + '/'
                        cand_file = glob.glob(cand_dir + '/*.spccl')

                        while (len(cand_file) == 0):
                            if self._verbose:
                                print("No .spccl file for beam %d yet..." % (ibeam))
                            time.sleep(0.125)
                            cand_file = glob.glob(cand_dir + '/*.spccl')


                        beam_cands = pd.read_csv(cand_file[0], sep='\s+', names=self._header_names, skiprows=1)
                        latest_cand_mjd = beam_cands.tail(1)['MJD'].values[0]
                        print("Latest candidate MJD: %.10f" % (latest_cand_mjd))

                        while latest_cand_mjd < latest_fil_mjd:
                            time.sleep(0.1)
                            beam_cands = pd.read_csv(cand_file[0], sep='\s+', names=self._header_names, skiprows=1)
                            latest_cand_mjd = beam_cands.tail(1)['MJD'].values[0]
                            if self._verbose:
                                print("Waiting for an updated .spccl file for beam %d..." % (ibeam))
                                print("Latest candidate MJD: %.10f" % (latest_cand_mjd))

                        mjd_pad = 1.0 / 86400.0
                        
                        #print(fil_latest)
                        for new_ff in new_fil_files:
                            print("Finding a match for file %s" % (new_ff[0]))
                            
                            with open(beam_dir + new_ff[0], mode='rb') as file:
                                # Unused, but vital to skipping first 114 bytes
                                skip_head = file.read(114)
                                mjdtime = struct.unpack('d', file.read(8))[0]
                            
                            filsplit = new_ff[0][:-4].split('_')
                            filtime = filsplit[0] + '-' + filsplit[1] + '-' + filsplit[2] + 'T' + filsplit[3]
                            aptime = Time(filtime, format='isot', scale='utc')
                            mjdtimeutc = aptime.mjd                        
                            print("UTC: %s -> MJD: %.8f" % (new_ff[0][:-4], mjdtime))
                            print("%.10f -> %.10f, %.10f" % (mjdtime, mjdtime + 2 * mjd_pad, mjdtimeutc))
                            
                            selected = (beam_cands.loc[(beam_cands['MJD'] >= mjdtime) & (beam_cands['MJD'] <= mjdtime + 2 * mjd_pad)]).reset_index()
                            #self._plotter.PlotDist(new_ff[0], selected)
                            if (selected.shape[0] > 0):
                                
                                if self._verbose:
                                    print("Found %d matching candidates" % (selected.shape[0]))

                                highest_snr = selected.iloc[selected['SNR'].idxmax()]
                                selected['Beam'] = full_beam
                                selected['RA'] = beam_ra
                                selected['Dec'] = beam_dec
                                selected['File'] = new_ff[0]

                                fmtdm = "{:.2f}".format(highest_snr['DM'])

                                selected['Plot'] = str(highest_snr['MJD']) + '_DM_' + fmtdm + '_beam_' + str(full_beam) + '.jpg'
                                highest_snr = selected.iloc[selected['SNR'].idxmax()]

                                self._plotter.PlotExtractedCand(beam_dir, new_ff[0], self._headsize, self._nchans, highest_snr, full_beam, ibeam)

                                with open(beam_dir + 'Plots/used_candidates.spccl.extra.full' , 'a') as f:
                                    selected.to_csv(f, sep='\t', header=False, float_format="%.4f", index=False, index_label=False)

                                with open(beam_dir + 'Plots/used_candidates.spccl.extra' , 'a') as f:
                                    f.write("%d\t%.10f\t%.4f\t%.4f\t%.2f\t%d\t%s\t%s\t%s\t%s\n" % (0, highest_snr['MJD'], highest_snr['DM'], highest_snr['Width'], highest_snr['SNR'], highest_snr['Beam'], highest_snr['RA'], highest_snr['Dec'], highest_snr['File'], highest_snr['Plot']))
                                
                            else:
                                print("Something went wrong - did not find matching candidates")
                                
                else:
                    if (self._verbose):
                        print("No directory %s" % (beam_dir))
                    
            end_plot = time.time()
            
            if (self._verbose):
                print("Took %.2fs to plot" % (end_plot - start_plot))
                       
            time.sleep(5)
    
    def GetLogs(self, logfile):
        with open((self._directory + logfile)) as f:
            lines = f.readlines()
            start_event = lines[0]
            end_event = lines[-1]

        log_info = pd.DataFrame(columns=['beam', 'ra', 'dec'])
        for beam in json.loads(start_event)['beams']:
            log_info.loc[len(log_info)] = ({'beam':int(beam['fbfuse_id'].split('bf')[-1]),
                                    'ra':beam['ra_hms'],
                                    'dec':beam['dec_dms']})
        log_start_utc = json.loads(start_event)['utc']
        log_end_utc = json.loads(end_event)['utc']
        self._beam_info = log_info
    
    def Watch(self):
        print("I am watching")
        
        if (self._verbose):
            print("Creating plots output directory")
            
            for ibeam in np.arange(6):
                beam_dir = self._directory + '/beam0' + str(ibeam) + '/'
                if os.path.isdir(beam_dir):
                    try:
                        os.mkdir(self._directory + '/beam0' + str(ibeam) + '/Plots')
                    except FileExistsError:
                        if (self._verbose):
                            print("Directory already exists")
                else:
                    if (self._verbose):
                        print("No directory %s" % (beam_dir))
            
        if (self._verbose):
            print("Parsing log files")
            self.GetLogs(self._events_file)
        
        #spccl_thread = threading.Thread(target=self.GetNewSpcclFiles)
        #spccl_thread.start()
        fil_thread = threading.Thread(target=self.GetNewFilFiles)
        fil_thread.start()
        fil_thread.join()
        #spccl_thread.join()


def main():
    
    basedir = sys.argv[1]
    events_file = 'pipeline_events.log'

    #utc_dirs = []

    # for inbase in os.listdir(basedir):
    #     full_inbase = os.path.join(basedir, inbase)
    #     if os.path.isdir(full_inbase):
    #         utc_dirs.append(full_inbase)
    

    # latest_utc_dir = max(utc_dirs, key=os.path.getmtime)

    plotdir = basedir + '/'

    #print(utc_dirs)
    #print(latest_utc_dir)

    print("Will use directory %s" % (plotdir))
    watcher = Watcher(plotdir, events_file, True, True)
    watcher.Watch()

if __name__ == "__main__":
    main()
