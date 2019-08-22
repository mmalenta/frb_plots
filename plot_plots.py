import matplotlib
matplotlib.use('Agg')

import argparse
import glob
import json
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import struct
import sys
import threading
import time

from astropy.time import Time

# This class does the actual plotting
class Plotter:
    
    def __init__(self, timing, verbose,  maskfile, outdir = './', single_pass=False):
        # General managment
        self._plotdir = outdir
        self._mask_file = maskfile
        self._single_pass = single_pass
        self._timing = timing
        self._verbose = verbose

        # Filterbank file parameters
        self._nchans = 0
        self._tsamp = 0.0
        self._ftop = 0.0
        self._fband = 0.0
        self._fbottom = 0.0

        # Extracted candidate plot parameters
        self._timeavg = 16
        self._freqavg = 64
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
        self._disp_const = 0.0
        self._filfile_pad_s = 1.0
        self._filfile_pad_mjd = self._filfile_pad_s / 86400.0
        self._plot_pad_s = self._filfile_pad_s / 2.0
        self._dedisp_bands = 0

        if self._verbose:
            print("Starting the plotter up")
    
    def Dedisperse(self, inputdata, filmjd, properties, outbands):
    
        dm_start = time.time()
        dm = properties['DM']
        candmjd = properties['MJD']
        perband = int(self._nchans / outbands)
        original_data_length = inputdata.shape[1]
        filfile_padding_samples = int(np.floor(self._filfile_pad_s / self._tsamp))
        plot_padding_samples = int(np.floor(self._plot_pad_s / self._tsamp))
        cand_samples_from_start = int(np.ceil((candmjd - filmjd) * 86400.0 / self._tsamp))

        output_samples = 0
        plot_skip_samples = 0
        start_padding_added = 0

        last_band_top = self._ftop + (outbands - 1) * perband * self._fband
        last_band_bottom = last_band_top + perband * self._fband
        full_delay_samples = int(np.ceil(self._disp_const * dm / self._tsamp))
        last_band_delay_samples = int(np.ceil(4.15e+03 * dm * (1.0 / (last_band_bottom * last_band_bottom) - 1.0 / (last_band_top * last_band_top)) / self._tsamp))

        # There are 4(5) cases in general (if everything is extracted correctly):
        ## 1. Not enough padding at the start of the file (candidate was early enough in the file that there is no full padding)
        ## 2. Enough padding on both sides
        ## 3. Not enough samples at the end of the file (these can really be combined into one check)
        ### a. Just part of the padding missing
        ### b. Whole padding and part of the actual candidate data missing
        ## 4. Not enough padding to cover the sweep in the last band
        # If we have case 2 - we are good
        # Cases 1 and 3 can appear together with how the candidates are extracted at the moment
        # Case 4 definitely appears in case 3, but can also appear in case 2
        
        # We have case 1 - add extra 0 padding at the start
        if ((candmjd - self._filfile_pad_mjd) < filmjd):
            actualpad = (candmjd - filmjd)
            zero_padding_samples = int(np.ceil((self._filfile_pad_mjd - actualpad) * 86400.0 / self._tsamp))
            start_padding_added = zero_padding_samples
            inputdata = np.append(np.zeros((self._nchans, zero_padding_samples)), inputdata, axis=1)
            if (self._verbose):
                print("Not enough data at the start. Padding with %d extra samples" % (zero_padding_samples))

        # We have case 3 - add extra 0 padding at the end
        if ((start_padding_added + cand_samples_from_start + full_delay_samples + filfile_padding_samples) > inputdata.shape[1]):
            zero_padding_samples = (start_padding_added + cand_samples_from_start + full_delay_samples + filfile_padding_samples) - inputdata.shape[1]
            inputdata = np.append(inputdata, np.zeros((self._nchans, zero_padding_samples)), axis=1)

            # Add extra full Cheetah padding
            #zero_padding_samples = int(np.ceil(self._cand_pad_s / self._tsamp))
            #inputdata = np.append(inputdata, np.zeros((self._nchans, zero_padding_samples)), axis=1)            
            if (self._verbose):
                print("Not enough data at the end. Padding with %d extra samples" % (zero_padding_samples))

        # How many samples to skip from start of the data block
        plot_skip_samples = cand_samples_from_start - plot_padding_samples + start_padding_added

        if (outbands == 1):
            # Padding on both sides
            # Currently ignores the pulse width
            output_samples = int(np.ceil(2 * plot_padding_samples))
        else:
            # Padding on both sides + extra DM sweep
            output_samples = int(np.ceil(2 * plot_padding_samples)) + full_delay_samples - last_band_delay_samples
            # We have case 4 - add extra 0 padding at the end
            # We only have to worry about the extra delay when we do a subband dedispersion
            if (last_band_delay_samples > filfile_padding_samples):
                zero_padding_samples = int(last_band_delay_samples - filfile_padding_samples)
                inputdata = np.append(inputdata, np.zeros((self._nchans, zero_padding_samples)), axis=1)
                if (self._verbose):
                    print("Adding extra zero padding of %d time samples to account for last band dispersion" % (zero_padding_samples))
        
        if (self._verbose):
            print("Candidate plotting:")
            if (outbands == 1):
                print("\tFully dedispersed time series")
            else:
                print("\tSubband dedispersed")

            print("\tInput data length (original): %d" % (original_data_length))
            print("\tInput data length (with all padding included): %d" % (inputdata.shape[1]))
            print("\tOutput plot samples: %d" % (output_samples))
            print("\tDM sweep samples: %d" % (full_delay_samples))
            print("\tPadding at the start: %d" % (start_padding_added))
            print("\tSamples skipped at the start: %d" % (plot_skip_samples))


        dedispersed = np.zeros((outbands, output_samples))

        for band in np.arange(outbands):
            bandtop = self._ftop + band * perband * self._fband
            for chan in np.arange(perband):
                chanfreq = bandtop + chan * self._fband
                delay = int(np.ceil(4.15e+03 * dm * (1.0 / (chanfreq * chanfreq) - 1.0 / (bandtop * bandtop)) / self._tsamp))
                dedispersed[band, :] = np.add(dedispersed[band, :], inputdata[chan + band * perband, int(plot_skip_samples) + delay : int(plot_skip_samples) + delay + output_samples])

        dm_end = time.time()
        
        if (self._verbose):
            print("Dedispersion took %.2fs" % (dm_end - dm_start))

        return dedispersed, start_padding_added       

    def PlotSpcclCands(self, candidates):

        last_mjd = candidates['MJD'].values[-1]
        first_mjd = last_mjd - self._spccl_length_mjd

        candidates_to_plot = candidates.loc[(candidates['MJD'] >= first_mjd) & (candidates['MJD'] <= last_mjd)]

        cmap = matplotlib.colors.ListedColormap(self._beam_colours)
        bound = [0, 1, 2, 3 , 4, 5, 6]
        
        norm = matplotlib.colors.BoundaryNorm(bound, cmap.N, clip=True)
    
        print("Plotting the full candidates")
        
        figspccl = plt.figure(figsize=(10.24, 7.68), frameon=False, dpi=10)
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
        figspccl.savefig(self._plotdir + '/full_spccl_candidates.jpg')
        plt.close(figspccl)

        time.sleep(self._spccl_refresh_s)
    
    def PlotExtractedCand(self, beam_dir, filename, headsize, nchans, ftop, fband, tsamp, properties, filmjd, ibeam=0, nodebeam=0):
        
        # Update the filterbank file parameters
        self._nchans = nchans
        self._tsamp = tsamp
        self._ftop = ftop
        self._fband = fband # A negative value
        self._fbottom = self._ftop + self._nchans * self._fband

        # Update the dedispersion parameters
        self._disp_const = 4.15e+03 * (1.0 / (self._fbottom * self._fbottom) - 1.0 / (self._ftop * self._ftop)) # in s per unit DM
        self._dedisp_bands = int(self._nchans / self._freqavg)

        if (self._verbose):
            print("Filerbank file parameters:")
            print("\t# channels: %d" % (self._nchans))
            print("\tSampling time: %.8f" % (self._tsamp))
            print("\tTop frequency: %.8f" % (self._ftop))
            print("\tChannel bandwidth: %.8f" % (self._fband))
            print("\tBottom frequency: %.8f" % (self._fbottom))

        mask = np.loadtxt(self._mask_file)

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
        read_start = time.time()
        fildata = np.reshape(np.fromfile(os.path.join(beam_dir, filename), dtype='B')[headsize:], (-1, nchans)).T
        read_end = time.time()

        print("Read %d time samples in %.4fs" % (fildata.shape[1], (read_end - read_start)))
        samples_read = fildata.shape[1]
        fildata = fildata * mask[:, np.newaxis]
        filband = np.mean(fildata[:, 128:], axis=1)
        fildata = fildata - filband[:, np.newaxis]
        
        # Frequency average the data (i.e. subband dedisperse)
        filfreqavg, skip_padding = self.Dedisperse(fildata, filmjd, properties, self._dedisp_bands)

        # Time average the data
        timesamples = int(np.floor(filfreqavg.shape[1] / self._timeavg) * self._timeavg)                
        filbothavg = filfreqavg[:, :timesamples].reshape(filfreqavg.shape[0], (int)(timesamples / self._timeavg), self._timeavg).sum(axis=2) / self._timeavg / self._freqavg
        
        # We are no longer dealing with original samples when the data is averaged
        skip_padding_time_avg = int(np.floor(skip_padding / self._timeavg))
        samples_read_time_avg = int(np.floor(samples_read / self._timeavg))
 
        datamean = np.mean(filbothavg[:, skip_padding_time_avg : (skip_padding_time_avg + samples_read_time_avg)])
        datastd = np.std(filbothavg[:, skip_padding_time_avg : (skip_padding_time_avg + samples_read_time_avg)])        

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
        avg_time_label = avg_time_pos * self._tsamp * self._timeavg + skip_padding * self._tsamp + ((properties['MJD'] - filmjd) * 86400 - self._plot_pad_s)
        avg_time_label_str = [fmt(label) for label in avg_time_label]
        
        cmap = 'binary'
        
        fil_fig, fil_axis = plt.subplots(2, 1, figsize=(10.24, 7.68), frameon=False, dpi=100)
        fil_fig.tight_layout(h_pad=3.25, rect=[0, 0.03, 1, 0.95])
        
        dedispchans = filbothavg.shape[0]
        delays = np.zeros(dedispchans)

        for ichan in np.arange(dedispchans):
            chanfreq = self._ftop + ichan * self._fband * self._freqavg
            delays[ichan] = int(np.ceil(4.15e+03 * properties['DM'] * (1.0 / (chanfreq * chanfreq) - 1.0 / (self._ftop * self._ftop)) / self._tsamp) / self._timeavg) + 0.95 * self._plot_pad_s / (self._timeavg * self._tsamp)

        axboth = fil_axis[0]
        axboth.imshow(filbothavg, interpolation='none', vmin=cbottom, vmax=ctop, aspect='auto', cmap=cmap)
        axboth.plot(delays, np.arange(dedispchans), linewidth=0.5, color='deepskyblue')
        axboth.set_title('Time (' + str(self._timeavg) + '), freq (' + str(self._freqavg) + ') avg', fontsize=11)
        axboth.set_xlabel('Time [s]', fontsize=10)
        axboth.set_ylabel('Frequency [MHz]', fontsize=10)
        axboth.set_xticks(avg_time_pos)
        axboth.set_xticklabels(avg_time_label_str, fontsize=8)
        axboth.set_yticks(avg_freq_pos)
        axboth.set_yticklabels(avg_freq_label_str, fontsize=8)        

        # Fully dedisperse the original filterbank data
        dedispersed, skip_padding = self.Dedisperse(fildata, filmjd, properties, 1)
        dedispersed = dedispersed / dedispersed.shape[1]
        # Average the dedispersed time series
        dedisp_avg_time = int(np.floor(dedispersed.shape[1] / self._timeavg) * self._timeavg)
        dedisp_avg = dedispersed[0, :dedisp_avg_time].reshape(1, int(dedisp_avg_time / self._timeavg), self._timeavg).sum(axis=2) / self._timeavg

        dedisp_time_pos = np.linspace(0, int(dedisp_avg_time / self._timeavg), num=9)
        dedisp_time_label = dedisp_time_pos * self._tsamp * self._timeavg + self._plot_pad_s + (properties['MJD'] - filmjd) * 86400.0
        dedisp_time_label = dedisp_time_pos * self._tsamp * self._timeavg + skip_padding * self._tsamp + ((properties['MJD'] - filmjd) * 86400 - self._plot_pad_s)        
        dedisp_time_label_str = [fmt(label) for label in dedisp_time_label]
        
        axdedisp = fil_axis[1]
        axdedisp.plot(dedisp_avg[0, :], linewidth=0.4, color='black')
        fmtdm = "{:.2f}".format(properties['DM'])
        axdedisp.axvline(int(dedisp_avg_time / self._timeavg / 2), color='deepskyblue', linewidth=0.5)
        axdedisp.set_ylim()
        axdedisp.set_title('Dedispersed time series, DM ' + fmtdm)
        axdedisp.set_xticks(dedisp_time_pos)
        axdedisp.set_xticklabels(dedisp_time_label_str)
        axdedisp.set_xlabel('Time [s]')
        axdedisp.set_ylabel('Power [arbitrary units]')
        
        if (np.sum(dedispersed) == 0):
            axdedisp.text(0.5, 0.6, 'Not dedispersed properly - please report!', fontsize=14, weight='bold', color='firebrick',  horizontalalignment='center', verticalalignment='center', transform=axdedisp.transAxes)
        
        if (self._single_pass):
            plotdir = os.path.join(self._outdir, 'beam0' + str(nodebeam), 'Plots_single')
        else:
            plotdir = os.path.join(self._outdir, 'beam0' + str(nodebeam), 'Plots')

        save_start = time.time()
        fil_fig.savefig(os.path.join(plotdir, str(properties['MJD']) + '_DM_' + fmtdm + '_beam_' + str(ibeam) + '.jpg'), bbox_inches = 'tight', quality=95)
        plt.close(fil_fig)
        save_end = time.time()

        print("Saved figure for beam %d in %.4fs" % (ibeam, (save_end - save_start)))

    # This might do something in the future
    def PlotDist(self, filename, selected):
        print("Empty")

# This is an overarching class that watches the directories and detects any changes
class Watcher:
    
    def __init__(self, data_dir, events_file, mask_file, timing=False, verbose=False, single_pass=False):
        self._events_file = events_file
        self._directory = data_dir
        self._mask_file = mask_file
        self._single_pass = single_pass
        self._timing = timing
        self._verbose = verbose
        self._watching = True
        self._spccl_wait = 5.0 # How long to wait (in seconds) for missing/empty .spccl files
        self._beam_skip = False
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
        
        self._plotter = Plotter(self._timing, self._verbose, self._mask_file, self._directory, self._single_pass)
    
    def GetHeaderValue(self, file, key, type):
        to_read = len(key)
        step_back = -1 * (to_read - 1)

        while(True):
            read_key = str(file.read(to_read).decode('iso-8859-1'))
            if (read_key == key):

                if (type == "int"):
                    value, = struct.unpack('i', file.read(4))

                if (type == "double"):
                    value, = struct.unpack('d', file.read(8))

                if (type == "str"):
                    to_read, = struct.unpack('i', file.read(4))
                    value = str(file.read(to_read).decode('iso-8859-1'))

                file.seek(0)
                break
            file.seek(step_back, 1)

        return value

    def GetNewSpcclFiles(self):
        if self._verbose:
            print("Watching for .spccl files")
    
        total_cands = np.zeros(6)
        new_cands = np.zeros(6)
        full_cands = pd.DataFrame()
    
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
        new_fil_files = []
        
        waited = 0.0

        while self._watching:
            
            start_plot = time.time()
            print (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            
            for ibeam in np.arange(self._nbeams):
                
                beam_dir = os.path.join(self._directory, 'beam0' + str(ibeam))
                new_fil_files = []

                if os.path.isdir(beam_dir):

                    find_start = time.time()

                    full_beam = self._beam_info['beam'].values[ibeam]
                    beam_ra = self._beam_info['ra'].values[ibeam]
                    beam_dec = self._beam_info['dec'].values[ibeam]
                    
                    fil_files = os.scandir(beam_dir)
                    for ff in fil_files:
                        if ((ff.name.endswith('fil')) & (ff.stat().st_mtime > fil_latest[ibeam])):
                            new_fil_files.append([ff.name, ff.stat().st_mtime])

                    new_len = len(new_fil_files)

                    find_end = time.time()

                    print (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print("Found %d new filterbank files for beam %d in %.4fs" % (new_len, ibeam, find_end - find_start))

                    if new_len > 0:

                        # Will hopefully get .spccl written
                        fil_latest[ibeam] = max(new_fil_files, key = lambda nf: nf[1])[1]

                        latest_fil_mjd = 0.0
                        for new_ff in new_fil_files:
                            with open(os.path.join(beam_dir, new_ff[0]), mode='rb') as file:
                                mjdtime = self.GetHeaderValue(file, "tstart", "double")
                                if mjdtime > latest_fil_mjd:
                                    latest_fil_mjd = mjdtime

                        print("Latest .fil file MJD: %.10f" % (latest_fil_mjd))

                        cand_file = glob.glob(beam_dir + '/*.spccl')
                        # Wait until we get the .spccl file - it should be saved at some point
                        waited = 0.0
                        if ( not self._single_pass):
                            while ( (len(cand_file) == 0) and (waited < self._spccl_wait) ):
                                if self._verbose:
                                    print("No .spccl file for beam %d yet..." % (ibeam))

                                time.sleep(0.1)
                                cand_file = glob.glob(beam_dir + '/*.spccl')
                                waited = waited + 0.1

                            if (waited >= self._spccl_wait):
                                if (self._verbose):
                                    print("WARNING: no walid .spccl file for beam %d after 5.0s" % (ibeam))
                                fil_latest[ibeam] = 0
                                continue

                        # Bail out - this should not happen, as we should always have .spccl file when extracted .fil files are found
                        else:
                            if (len(cand_file) == 0):
                                print("ERROR: did not find an .spccl file")
                                # Continue to the next beam
                                continue

                        waited = 0.0
                        # At this stage we can be sure there is an .spccl file for a given beam
                        beam_cands = pd.read_csv(cand_file[0], sep='\s+', names=self._header_names, skiprows=1)
                        # This should not happen at all in during proper operations
                        if (beam_cands.size == 0):
                            
                            if ( not self._single_pass):
                                while( (beam_cands.size == 0) and (waited < self._spccl_wait / 2.0) ):
                                    if self._verbose:
                                        print("No filled .spccl file for beam %d yet..." % (ibeam))
                                    time.sleep(0.1)
                                    beam_cands = pd.read_csv(cand_file[0], sep='\s+', names=self._header_names, skiprows=1)
                                    waited = waited + 0.1
                        
                                if (waited >= self._spccl_wait / 2.0):
                                    if (self._verbose):
                                        print("WARNING: empty .spccl file for beam %d after 5.0s" % (ibeam))
                                    fil_latest[ibeam] = 0
                                    continue

                            else:
                                print("ERROR: found an empty .spccl file %s" % (cand_file[0]))
                                # Continue to the next beam and hope for the best next time
                                continue

                        latest_cand_mjd = beam_cands.tail(1)['MJD'].values[0]
                        print("Latest candidate MJD: %.10f" % (latest_cand_mjd))
                        #print(fil_latest)

                        waited = 0.0
                        # Don't wait for an updated .spccl file in single-pass mode - we work with what we have
                        if ( not self._single_pass):
                            while ( (latest_cand_mjd < latest_fil_mjd) and (waited < self._spccl_wait / 2.0)):
                                time.sleep(0.1)
                                beam_cands = pd.read_csv(cand_file[0], sep='\s+', names=self._header_names, skiprows=1)
                                latest_cand_mjd = beam_cands.tail(1)['MJD'].values[0]
                                waited = waited + 0.1
                                if self._verbose:
                                    print("Waiting for an updated .spccl file for beam %d..." % (ibeam))
                                    print("Latest candidate MJD: %.10f" % (latest_cand_mjd))

                            if (waited >= self._spccl_wait / 2.0):
                                if (self._verbose):
                                    print("WARNiNG: no up-to-date candidates in the .spccl file for beam %d..." % (ibeam))
                                fil_latest[ibeam] = 0
                                continue

                        if (self._single_pass):
                            extra_file = os.path.join(beam_dir, 'Plots_single/used_candidates.spccl.extra')
                            extra_full_file = os.path.join(beam_dir, 'Plots_single/used_candidates.spccl.extra.full')
                        else:
                            extra_file = os.path.join(beam_dir, 'Plots/used_candidates.spccl.extra')
                            extra_full_file = os.path.join(beam_dir, 'Plots/used_candidates.spccl.extra.full')

                        # At this stage we can be sure there are valid candidates for a given beam
                        for new_ff in new_fil_files:
                            print("Finding a match for file %s" % (new_ff[0]))
                            
                            with open(os.path.join(beam_dir, new_ff[0]), mode='rb') as file:
                                nchans = self.GetHeaderValue(file, "nchans", "int")
                                ftop = self.GetHeaderValue(file, "fch1", "double")
                                fband = -1.0 * np.abs(self.GetHeaderValue(file, "foff", "double")) # Make sure bandwidth is negative
                                tsamp = self.GetHeaderValue(file, "tsamp", "double")
                                mjdtime = self.GetHeaderValue(file, "tstart", "double")
                            
                            selected = (beam_cands.loc[(beam_cands['MJD'] >= mjdtime) & (beam_cands['MJD'] <= mjdtime + 2 * self._mjd_pad)]).reset_index()

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

                                plot_start = time.time()
                                self._plotter.PlotExtractedCand(beam_dir, new_ff[0], self._headsize, nchans, ftop, fband, tsamp, highest_snr, mjdtime, full_beam, ibeam)
                                plot_end = time.time()

                                print("Plotting took %.2fs for beam %d" % (plot_end - plot_start, ibeam))

                                with open(extra_full_file, 'a') as f:
                                    selected.to_csv(f, sep='\t', header=False, float_format="%.4f", index=False, index_label=False)

                                with open(extra_file, 'a') as f:
                                    f.write("%d\t%.10f\t%.4f\t%.4f\t%.2f\t%d\t%s\t%s\t%s\t%s\n" % (0, highest_snr['MJD'], highest_snr['DM'], highest_snr['Width'], highest_snr['SNR'], highest_snr['Beam'], highest_snr['RA'], highest_snr['Dec'], highest_snr['File'], highest_snr['Plot']))

                                print("\n\n")

                            else:
                                print("Something went wrong - did not find matching candidates")
                            
                else:
                    if (self._verbose):
                        print("No directory %s" % (beam_dir))

            end_plot = time.time()
            
            if (self._verbose):
                print("Took %.2fs to plot" % (end_plot - start_plot))

            if (self._single_pass):
                self._watching = False
            else: 
                time.sleep(5)
    
    def GetLogs(self, logfile):
        with open(logfile) as f:
            lines = f.readlines()
            startevent = lines[0]
            endevent = lines[-1]

        log_info = pd.DataFrame(columns=['beam', 'ra', 'dec'])
        for beam in json.loads(startevent)['beams']:
            log_info.loc[len(log_info)] = ({'beam':int(beam['fbfuse_id'].split('bf')[-1]),
                                    'ra':beam['ra_hms'],
                                    'dec':beam['dec_dms']})
        
        # Currently not used, but might become useful in the future, so don't remove
        log_start_utc = json.loads(startevent)['utc']
        log_end_utc = json.loads(endevent)['utc']
        self._beam_info = log_info
    
    def Watch(self):
        print("I am watching")
        
        if (self._verbose):
            print("Creating plots output directory")
            
        for ibeam in np.arange(6):
            beamdir = os.path.join(self._directory, 'beam0' + str(ibeam))
            if os.path.isdir(beamdir):

                try:
                    if (self._single_pass):
                        os.mkdir(os.path.join(beamdir, 'Plots_single'))
                    else:
                        os.mkdir(os.path.join(beamdir, 'Plots'))

                except FileExistsError:
                    if (self._verbose):
                        print("Directory already exists")
            else:
                if (self._verbose):
                    print("No directory %s" % (beamdir))
                    # Need to quit here if we can't find beam directories

        if (self._verbose):
            print("Parsing log files")
            self.GetLogs(self._events_file)
        
        #spccl_thread = threading.Thread(target=self.GetNewSpcclFiles)
        #spccl_thread.start()
        filthread = threading.Thread(target=self.GetNewFilFiles)
        filthread.start()
        filthread.join()
        #spccl_thread.join()

def main():
    
    parser = argparse.ArgumentParser(description="Generating candidate plots for MeerTRAP",
                                        usage="%(prog)s <options>",
                                        epilog="For any bugs, please start an issue at https://gitlab.com/MeerTRAP/frb_plots")
    parser.add_argument("-v", "--verbose", help="Enable verbose mode", action="store_true")
    parser.add_argument("-t", "--timing", help="Print the timing information", action="store_true")
    parser.add_argument("-s", "--single", help="Single pass through the data (watch the directory by default)", action="store_true")
    parser.add_argument("-m", "--mask", help="Channel mask file", type=str)
    parser.add_argument("-d", "--directory", help="Base data directory", required=True, type=str)
    parser.add_argument("-e", "--events", help="Pipeline events log file", type=str)

    arguments = parser.parse_args()

    verbose = arguments.verbose
    basedir = os.path.abspath(arguments.directory)
    maskfile = arguments.mask
    eventsfile = arguments.events

    if (maskfile == None):
        if (verbose):
            print("Will not mask any channels")

    if (verbose):
        print("Will use directory %s" % (basedir))

    if (eventsfile == None):
        eventsfile = 'pipeline_events.log'
            
    eventsfile = os.path.abspath(eventsfile)

    if (verbose):
        print("Will use events file %s" % (eventsfile))
    
    watcher = Watcher(basedir, eventsfile, maskfile, arguments.timing, verbose, arguments.single)
    watcher.Watch()

if __name__ == "__main__":
    main()
