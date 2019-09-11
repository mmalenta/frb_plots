import matplotlib
matplotlib.use('Agg')

import argparse
import glob
import json
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import struct
import sys
import threading
import time

# This class does the actual plotting
class Plotter:
    
    def __init__(self, config):
        # General managment
        self._mask_file = config['mask_file']
        self._verbose = config['verbose']
        self._single_pass = config['single_pass']
        self._timing = config['verbose']

        # Filterbank file parameters
        self._nchans = 0
        self._tsamp = 0.0
        self._ftop = 0.0
        self._fband = 0.0
        self._fbottom = 0.0

        # Extracted candidate plot parameters
        self._timeavg = 16
        self._freqavg = 64
        self._outdir = config['base_dir']
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
        # This should really be an option rather than a hardcoded value
        self._filfile_pad_s = 1.0
        self._filfile_pad_mjd = self._filfile_pad_s / 86400.0
        self._plot_pad_s = 0.075 #self._filfile_pad_s / 2.0
        self._dedisp_bands = 0

    def Dedisperse(self, inputdata, filmjd, properties, outbands):
    
        dm_start = time.time()
        dm = properties['DM']
        candmjd = properties['MJD']
        perband = int(self._nchans / outbands)
        original_data_length = inputdata.shape[1]
        filfile_padding_samples = int(np.floor(self._filfile_pad_s / self._tsamp))
        plot_padding_samples = int(np.floor(self._plot_pad_s / self._tsamp))
        cand_samples_from_start = int(np.ceil((candmjd - filmjd) * 86400.0 / self._tsamp))

        output_samples_sub = 0
        output_samples_full = 0
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
        
        # TODO: Need to create a single, larger array, which includes all the padding and insert the data into that
        # That should be faster than appending the zeros array at the start and the end
        zero_padding_samples_start = 0
        zero_padding_samples_end = 0
        input_file_samples = inputdata.shape[1]
        total_data_samples = 0

        # We have case 1 - add extra 0 padding at the start
        if ((candmjd - self._filfile_pad_mjd) < filmjd):
            actualpad = (candmjd - filmjd)
            zero_padding_samples_start = int(np.ceil((self._filfile_pad_mjd - actualpad) * 86400.0 / self._tsamp))
            start_padding_added = zero_padding_samples_start
            if (self._verbose):
                print("Not enough data at the start. Padding with %d extra samples" % (zero_padding_samples_start))

        # We have case 3 - add extra 0 padding at the end
        if ((start_padding_added + cand_samples_from_start + full_delay_samples + filfile_padding_samples) > inputdata.shape[1]):
            zero_padding_samples_end = (start_padding_added + cand_samples_from_start + full_delay_samples + filfile_padding_samples) - inputdata.shape[1]
            if (self._verbose):
                print("Not enough data at the end. Padding with %d extra samples" % (zero_padding_samples_end))

        # How many samples to skip from start of the data block
        plot_skip_samples = cand_samples_from_start - plot_padding_samples + start_padding_added

        # We have case 4 - add extra 0 padding at the end
        # We only have to worry about the extra delay when we do a subband dedispersion
        if (last_band_delay_samples > filfile_padding_samples):
            zero_padding_samples_end = zero_padding_samples_end + int(last_band_delay_samples - filfile_padding_samples)
            if (self._verbose):
                print("Adding extra zero padding of %d time samples to account for last band dispersion" % (zero_padding_samples_end))
        
        total_data_samples = zero_padding_samples_start + zero_padding_samples_end + input_file_samples

        full_data = np.zeros((self._nchans, total_data_samples))
        full_data[:, zero_padding_samples_start : zero_padding_samples_start + input_file_samples] = inputdata

        # Full dedispersion: padding on both sides
        # Currently ignores the pulse width
        output_samples_full = int(np.ceil(2 * plot_padding_samples))
        # Subband dedispersion: padding on both sides + extra DM sweep
        output_samples_sub = int(np.ceil(2 * plot_padding_samples)) + full_delay_samples - last_band_delay_samples

        if (self._verbose):
            print("Candidate plotting:")

            print("\tInput data length (original): %d" % (original_data_length))
            print("\tInput data length (with all padding included): %d" % (inputdata.shape[1]))
            print("\tNumber of dedispersed subbands: %d" % (outbands))
            print("\tOutput plot samples: %d (subband), %d (full)" % (output_samples_sub, output_samples_full))
            print("\tDM sweep samples: %d" % (full_delay_samples))
            print("\tPadding at the start: %d" % (start_padding_added))
            print("\tSamples skipped at the start: %d" % (plot_skip_samples))

        dedispersed_sub = np.zeros((outbands, output_samples_sub))
        dedispersed_not_sum = np.zeros((outbands, output_samples_full))
        dedispersed_full = np.zeros((1, output_samples_full))

        for band in np.arange(outbands):
            bandtop = self._ftop + band * perband * self._fband
            for chan in np.arange(perband):
                chanfreq = bandtop + chan * self._fband
                delay_sub = int(np.ceil(4.15e+03 * dm * (1.0 / (chanfreq * chanfreq) - 1.0 / (bandtop * bandtop)) / self._tsamp))
                dedispersed_sub[band, :] = np.add(dedispersed_sub[band, :], full_data[chan + band * perband, int(plot_skip_samples) + delay_sub : int(plot_skip_samples) + delay_sub + output_samples_sub])
                delay_full = int(np.ceil(4.15e+03 * dm * (1.0 / (chanfreq * chanfreq) - 1.0 / (self._ftop * self._ftop)) / self._tsamp))
                dedisp_chunk = full_data[chan + band * perband, int(plot_skip_samples) + delay_full : int(plot_skip_samples) + delay_full + output_samples_full]
                dedispersed_full[0, :] = np.add(dedispersed_full[0, :], dedisp_chunk)
                dedispersed_not_sum[band, :] = np.add(dedispersed_not_sum[band, :], dedisp_chunk)

        dm_end = time.time()
        
        if (self._verbose):
            print("Dedispersion took %.2fs" % (dm_end - dm_start))

        return dedispersed_sub, dedispersed_not_sum, dedispersed_full, start_padding_added

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

        # There must be a better way of doing it once only
        if (not self._mask_file == None):
            mask = np.loadtxt(self._mask_file)
        else:
            mask = np.ones(nchans)

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
        
        self._tsamp = self._tsamp * self._timeavg
        # Read original data
        fildata = np.reshape(np.fromfile(os.path.join(beam_dir, filename), dtype='B')[headsize:], (-1, nchans)).T
        if (self._verbose):
            print("Read %d time samples" % (fildata.shape[1]))
        fildata = fildata * mask[:, np.newaxis]
        filband = np.mean(fildata[:, 128:], axis=1)
        fildata = fildata - filband[:, np.newaxis]

        # Time average the original data
        timesamples = int(np.floor(fildata.shape[1] / self._timeavg) * self._timeavg) 
        time_avg_data = fildata[:, :timesamples].reshape(fildata.shape[0], (int)(timesamples / self._timeavg), self._timeavg).sum(axis=2) / self._timeavg / self._freqavg

        # Frequency average the data (i.e. subband dedisperse)
        dedisp_sub, dedisp_not_sum, dedisp_full, skip_padding = self.Dedisperse(time_avg_data, filmjd, properties, self._dedisp_bands)
        dedisp_full = dedisp_full / dedisp_full.shape[1]

        datamean = np.mean(dedisp_sub[:, skip_padding : (skip_padding + timesamples)])
        datastd = np.std(dedisp_sub[:, skip_padding : (skip_padding + timesamples)])        
        ctop = int(np.ceil(datamean + 1.25 * datastd))
        cbottom = int(np.floor(datamean - 0.50 * datastd))

        fmt = lambda x: "{:.2f}".format(x)
        
        # Prepare the frequency ticks
        avg_freq_pos = np.linspace(0, int(nchans / self._freqavg), num=9)
        avg_freq_pos[-1] = avg_freq_pos[-1] - 1       
        avg_freq_label = self._ftop + avg_freq_pos * self._fband * self._freqavg
        avg_freq_label_str = [fmt(label) for label in avg_freq_label]
        
        # Prepare the time ticks
        avg_time_pos = np.linspace(0, dedisp_sub.shape[1], num=5)
        avg_time_label = avg_time_pos * self._tsamp + skip_padding * self._tsamp + ((properties['MJD'] - filmjd) * 86400 - self._plot_pad_s)
        avg_time_label_str = [fmt(label) for label in avg_time_label]
        
        cmap = 'binary'

        fil_fig = plt.figure(figsize=(10.24, 7.68), frameon=False, dpi=100)
        fil_fig.tight_layout(h_pad=3.25, rect=[0, 0.03, 1, 0.95])

        plot_area = gs.GridSpec(2, 1)
        top_area = gs.GridSpecFromSubplotSpec(1, 5, subplot_spec=plot_area[0])
        bottom_area = gs.GridSpecFromSubplotSpec(1, 2, subplot_spec=plot_area[1])

        ax_spectrum = plt.Subplot(fil_fig, top_area[0, :-1])
        fil_fig.add_subplot(ax_spectrum)

        ax_band = plt.Subplot(fil_fig, top_area[0, -1])
        fil_fig.add_subplot(ax_band)

        ax_dedisp = plt.Subplot(fil_fig, bottom_area[0])
        fil_fig.add_subplot(ax_dedisp)

        ax_time = plt.Subplot(fil_fig, bottom_area[1])
        fil_fig.add_subplot(ax_time)

        dedispchans = dedisp_sub.shape[0]
        delays = np.zeros(dedispchans)

        for ichan in np.arange(dedispchans):
            chanfreq = self._ftop + ichan * self._fband * self._freqavg
            delays[ichan] = int(np.ceil(4.15e+03 * properties['DM'] * (1.0 / (chanfreq * chanfreq) - 1.0 / (self._ftop * self._ftop)) / self._tsamp)) + 0.95 * self._plot_pad_s / (self._tsamp)


        datamean = np.mean(dedisp_sub[:, skip_padding : (skip_padding + timesamples)])
        datastd = np.std(dedisp_sub[:, skip_padding : (skip_padding + timesamples)])        
        ctop = int(np.ceil(datamean + 1.75 * datastd))
        cbottom = int(np.floor(datamean - 0.50 * datastd))

        ax_spectrum.imshow(dedisp_sub, interpolation='none', vmin=cbottom, vmax=ctop, aspect='auto', cmap=cmap)
        ax_spectrum.plot(delays, np.arange(dedispchans), linewidth=0.5, color='deepskyblue')
        ax_spectrum.set_title('Time (' + str(self._timeavg) + '), freq (' + str(self._freqavg) + ') avg', fontsize=11)
        ax_spectrum.set_xlabel('Time [s]', fontsize=10)
        ax_spectrum.set_ylabel('Frequency [MHz]', fontsize=10)
        ax_spectrum.set_xticks(avg_time_pos)
        ax_spectrum.set_xticklabels(avg_time_label_str, fontsize=8)
        ax_spectrum.set_yticks(avg_freq_pos)
        ax_spectrum.set_yticklabels(avg_freq_label_str, fontsize=8)        

        sub_spectrum = np.sum(dedisp_sub, axis=1)

        ax_band.plot(sub_spectrum, np.arange(sub_spectrum.shape[0]), color='black', linewidth=0.5)
        ax_band.invert_yaxis()

        dedisp_time_pos = np.linspace(0, dedisp_full.shape[1], num=9)
        dedisp_time_label = dedisp_time_pos * self._tsamp + self._plot_pad_s + (properties['MJD'] - filmjd) * 86400.0
        dedisp_time_label = dedisp_time_pos * self._tsamp + skip_padding * self._tsamp + ((properties['MJD'] - filmjd) * 86400 - self._plot_pad_s)        
        dedisp_time_label_str = [fmt(label) for label in dedisp_time_label]
        
        datamean = np.mean(dedisp_not_sum[:, :])
        datastd = np.std(dedisp_not_sum[:, :])        
        ctop = int(np.ceil(datamean + 1.25 * datastd))
        cbottom = int(np.floor(datamean - 0.50 * datastd))

        print(dedisp_not_sum[:, 0])

        #datamean = np.mean(dedisp_not_sum[:, :])
        #datastd = np.std(dedisp_not_sum[:, :])
        #ctop = int(np.ceil(datamean + 1.25 * datastd))
        #cbottom = int(np.floor(datamean - 0.50 * datastd))
        ax_dedisp.imshow(dedisp_not_sum, interpolation='none', vmin=cbottom, vmax=ctop, aspect='auto', cmap=cmap)

        ax_time.plot(dedisp_full[0, :], linewidth=0.4, color='black')
        fmtdm = "{:.2f}".format(properties['DM'])
        ax_time.axvline(int(dedisp_full.shape[1] / 2), color='deepskyblue', linewidth=0.5)
        ax_time.set_ylim()
        ax_time.set_title('Dedispersed time series, DM ' + fmtdm)
        ax_time.set_xticks(dedisp_time_pos)
        ax_time.set_xticklabels(dedisp_time_label_str)
        ax_time.set_xlabel('Time [s]')
        ax_time.set_ylabel('Power [arbitrary units]')
        
        if (np.sum(dedisp_full) == 0):
            ax_time.text(0.5, 0.6, 'Not dedispersed properly - please report!', fontsize=14, weight='bold', color='firebrick',  horizontalalignment='center', verticalalignment='center', transform=ax_time.transAxes)
        
        if (self._single_pass):
            plotdir = os.path.join(self._outdir, 'beam0' + str(nodebeam), 'Plots_single')
        else:
            plotdir = os.path.join(self._outdir, 'beam0' + str(nodebeam), 'Plots')

        save_start = time.time()
        fil_fig.savefig(os.path.join(plotdir, str(properties['MJD']) + '_DM_' + fmtdm + '_beam_' + str(ibeam) + '.jpg'), bbox_inches = 'tight', quality=95)
        plt.close(fil_fig)
        save_end = time.time()
        if (self._verbose):
            print("Saved figure for beam %d in %.4fs" % (ibeam, (save_end - save_start)))

# This is an overarching class that watches the directories and detects any changes
class Watcher:
    
    def __init__(self, config):
        self._events_file = config['events_file']
        self._directory = config['base_dir']
        self._mask_file = config['mask_file']
        self._single_pass = config['single_pass']
        self._timing = config['timing']
        self._verbose = config['verbose']
        self._number_beams = config['number_beams']
        self._number_processes = config['number_proc']
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
        

        if (self._verbose):
            print("Creating plots output directory for %d beams" % (self._number_beams))

        for ibeam in np.arange(self._number_beams):
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

    def GetNewFilFiles(self, procid, config):
    
        verbose = config['verbose']
        beams = config['beams']
        single_pass = config['single_pass']
        beam_info = config['beam_info']
        spccl_wait = config['spccl_wait']
        header_names = config['header_names']
        fil_header_size = config['fil_header_size']

        if verbose:
            print("%s: Process %d watching for .fil files in beams %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), procid, beams))

        plotter = Plotter(config)

        watching = True

        fil_latest = np.zeros(6)
        new_fil_files = []
        
        waited = 0.0

        print ()

        while(watching):

            start_plot = time.time()

            for ibeam in beams:

                beam_dir = os.path.join(config['base_dir'], 'beam0' + str(ibeam))
                new_fil_files = []

                if os.path.isdir(beam_dir):
                    full_beam = beam_info['beam'].values[ibeam]
                    beam_ra = beam_info['ra'].values[ibeam]
                    beam_dec = beam_info['dec'].values[ibeam]

                fil_files = os.scandir(beam_dir)
                for ff in fil_files:
                    if ((ff.name.endswith('fil')) & (ff.stat().st_mtime > fil_latest[ibeam])):
                        new_fil_files.append([ff.name, ff.stat().st_mtime])

                new_len = len(new_fil_files)
                if (verbose):
                    print("%s, beam %d: Found %d new filterbank files" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam, new_len))    

                if (new_len > 0):
                    # Will hopefully get .spccl written
                    fil_latest[ibeam] = max(new_fil_files, key = lambda nf: nf[1])[1]

                    latest_fil_mjd = 0.0
                    for new_ff in new_fil_files:
                        with open(os.path.join(beam_dir, new_ff[0]), mode='rb') as file:
                            mjdtime = self.GetHeaderValue(file, "tstart", "double")
                            if mjdtime > latest_fil_mjd:
                                latest_fil_mjd = mjdtime
                    
                    if (verbose):
                        print("%s, beam %d: Latest .fil file MJD %.10f" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam, latest_fil_mjd))

                    cand_file = glob.glob(beam_dir + '/*.spccl')
                    # Wait until we get the .spccl file - it should be saved at some point
                    if ( not single_pass):
                        while ( (len(cand_file) == 0) and (waited < spccl_wait) ):
                            if (verbose):
                                print("%s, beam %d: No .spccl file found yet..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam))

                            time.sleep(0.1)
                            cand_file = glob.glob(beam_dir + '/*.spccl')
                            waited = waited + 0.1

                        if (waited >= spccl_wait):
                            if (verbose):
                                print("%s, beam %d: WARNING: no walid .spccl file after 5.0s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam))
                            fil_latest[ibeam] = 0
                            continue
                    # Bail out - this should not happen in singe pass, as we should always have .spccl file when extracted .fil files are found
                    else:
                        if (len(cand_file) == 0):
                            print("%s, beam %d: ERROR: did not find an .spccl file" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam))
                            # Continue to the next beam
                            continue

                    waited = 0.0
                    # At this stage we can be sure there is an .spccl file for a given beam
                    beam_cands = pd.read_csv(cand_file[0], sep='\s+', names=header_names, skiprows=1)
                    # This should not happen at all in during proper operations
                    if (beam_cands.size == 0):
                        if ( not single_pass):
                            while( (beam_cands.size == 0) and (waited < spccl_wait / 2.0) ):
                                if (verbose):
                                    print("No filled .spccl file for beam %d yet..." % (ibeam))
                                time.sleep(0.1)
                                beam_cands = pd.read_csv(cand_file[0], sep='\s+', names=header_names, skiprows=1)
                                waited = waited + 0.1
                    
                            if (waited >= spccl_wait / 2.0):
                                if (verbose):
                                    print("%s, beam %d: WARNING: empty .spccl file after 5.0s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam))
                                fil_latest[ibeam] = 0
                                continue

                        else:
                            print("%s, beam %d: ERROR: found an empty .spccl file %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam, cand_file[0]))
                            # Continue to the next beam and hope for the best next time
                            continue
                        
                    latest_cand_mjd = beam_cands.tail(1)['MJD'].values[0]
                    if (verbose):
                        print("%s, beam %d: Latest candidate MJD: %.10f" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam, latest_cand_mjd))

                    waited = 0.0
                    # Don't wait for an updated .spccl file in single-pass mode - we work with what we have
                    if ( not single_pass):
                        while ( (latest_cand_mjd < latest_fil_mjd) and (waited < spccl_wait / 2.0)):
                            time.sleep(0.1)
                            beam_cands = pd.read_csv(cand_file[0], sep='\s+', names=header_names, skiprows=1)
                            latest_cand_mjd = beam_cands.tail(1)['MJD'].values[0]
                            waited = waited + 0.1
                            if (verbose):
                                print("%s, beam %d: Waiting for an updated .spccl file..." % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam))
                                print("%s, beam %d: Latest candidate MJD: %.10f" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam, latest_cand_mjd))

                        if (waited >= spccl_wait / 2.0):
                            if (verbose):
                                print("%s, beam %d: WARNING: no up-to-date candidates in the .spccl file." % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam))
                            fil_latest[ibeam] = 0
                            continue

                    if (single_pass):
                        extra_file = os.path.join(beam_dir, 'Plots_single/used_candidates.spccl.extra')
                        extra_full_file = os.path.join(beam_dir, 'Plots_single/used_candidates.spccl.extra.full')
                    else:
                        extra_file = os.path.join(beam_dir, 'Plots/used_candidates.spccl.extra')
                        extra_full_file = os.path.join(beam_dir, 'Plots/used_candidates.spccl.extra.full')

                    # At this stage we can be sure there are valid candidates for a given beam
                    for new_ff in new_fil_files:
                        if (verbose):
                            print("%s, beam %d: Finding a match for file %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam, new_ff[0]))
                        
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
                            plotter.PlotExtractedCand(beam_dir, new_ff[0], fil_header_size, nchans, ftop, fband, tsamp, highest_snr, mjdtime, full_beam, ibeam)
                            plot_end = time.time()
                            if (self._verbose):
                                print("Plotting took %.2fs for beam %d" % (plot_end - plot_start, ibeam))

                            with open(extra_full_file, 'a') as f:
                                selected.to_csv(f, sep='\t', header=False, float_format="%.4f", index=False, index_label=False)

                            with open(extra_file, 'a') as f:
                                f.write("%d\t%.10f\t%.4f\t%.4f\t%.2f\t%d\t%s\t%s\t%s\t%s\n" % (0, highest_snr['MJD'], highest_snr['DM'], highest_snr['Width'], highest_snr['SNR'], highest_snr['Beam'], highest_snr['RA'], highest_snr['Dec'], highest_snr['File'], highest_snr['Plot']))
                            
                            if (verbose):
                                print("\n\n")

                        else:
                            print("%s, beam %d: ERROR: did not find matching candidates" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ibeam))


            if (single_pass):
                watching = False
            else:
                time.sleep(5)

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

        process_pool = []

        if (self._verbose):
            print("Launching the processes")

        if (self._number_processes > self._number_beams):
            if (self._verbose):
                print("WARNING: More processes than beams requested; reducing to the number of beams")
            self._number_processes = self._number_beams

        rem_beam = 0

        if ( (self._number_beams % self._number_processes) == 0):
            beams_per_process = int(self._number_beams / self._number_processes)
        else:
            beams_per_process = int(np.floor(self._number_beams / self._number_processes))
            rem_beam = self._number_beams - beams_per_process * (self._number_processes)

        configuration = {'verbose': self._verbose,
                        'single_pass': self._single_pass,
                        'base_dir': self._directory,
                        'beam_info': self._beam_info,
                        'spcc_wait': self._spccl_wait,
                        'header_names': self._header_names,
                        'fil_header_size': self._headsize,
                        'spccl_wait': self._spccl_wait,
                        'mask_file': self._mask_file}

        for iproc in np.arange(self._number_processes):
            if ( iproc < (self._number_processes - 1)):
                configuration['beams'] = np.arange(beams_per_process) + iproc * beams_per_process
            else:
                configuration['beams'] = np.arange(beams_per_process + rem_beam) + iproc * beams_per_process

            process = multiprocessing.Process(target=self.GetNewFilFiles, args=(iproc, configuration))
            process_pool.append(process)
            process.start()
            del configuration['beams']

        
    

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
    parser.add_argument("-n", "--nbeams", help="Number of beams to watch", required=False, type=int, default=6)
    parser.add_argument("-p", "--nproc", help="Number of plotting processes to run", required=False, type=int, default=1)

    arguments = parser.parse_args()

    verbose = arguments.verbose
    base_dir = os.path.abspath(arguments.directory)
    mask_file = arguments.mask
    events_file = arguments.events
    number_beams = arguments.nbeams
    number_proc = arguments.nproc

    if (number_beams == 0):
        print("ERROR: I need one or more beams to watch")
        sys.exit(1)

    if (number_proc == 0):
        print("ERROR: I need one or process(es) to run")
        sys.exit(1)

    if (mask_file == None):
        if (verbose):
            print("Will not mask any channels")
    else:
        mask_file = os.path.abspath(mask_file)
        if (not os.path.isfile(mask_file)):
            print("ERROR: No mask file %s" % (mask_file))
            sys.exit(1)

    if (verbose):
        print("Will use directory %s" % (base_dir))

    if (events_file == None):
        events_file = 'pipeline_events.log'
            
    events_file = os.path.abspath(events_file)

    if (not os.path.isfile(events_file)):
        print("ERROR: No events file %s" % (events_file))
        sys.exit(1)


    if (verbose):
        print("Will use events file %s" % (events_file))
    
    configuration = {'verbose': verbose,
                    'timing': arguments.timing,
                    'single_pass': arguments.single,
                    'base_dir': base_dir,
                    'mask_file': mask_file,
                    'events_file': events_file,
                    'number_beams': number_beams,
                    'number_proc': number_proc}
    
    watcher = Watcher(configuration)
    watcher.Watch()

if __name__ == "__main__":
    main()