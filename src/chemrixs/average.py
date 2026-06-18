
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from functools import cached_property
from pathlib import Path
from chemrixs.utils import *
from chemrixs.process import Reduced

class Average():
    """
    A class to average over several runs. 

    Takes runnumber as input, if these runs have been processed before
    they will simply be averaged, otherwise the 'Reduced' class is called 
    to process these runs first.

    Parameters
    ----------
    runs : list
        containing the integers of the runs to average

    proc_path : str or Path
        The filename structure and location of the processed runs. 
        Should be complete by simply adding the '{run:04d}.h5'

    avg : str
        String determining the way of averaging: 

    output_path : str or Path
        The filename for the output of averaged data.
    
    
    raw_path : str or Path
        The filename structure for the data to analyse.
        Should be complete by simply adding the '{run:04d}.h5'

    bgpath : str or Path
        The filename for a darkscan that can be used for background subtraction.

    fyaml : str or Path
        yaml file with settings for processing of unprocessed runs.

    bgyaml : str or Path
        yaml file with settings for processing the BG data, needed for unprocessed runs.

    save : bool
        Boolean determining if the processed data should be saved where processing necessary.

    scantype : str
        Optional, determining the type of scan for processing. If not given, this will be
        determined by the data structure.

    norm : bool
        Boolean determining if data should be normalised by I0 or not when processing.
        
    """

#####ZY_edits - 051726 - add despeckle option to handle hot pixels.
    def __init__(self, runs: list | int, proc_path: str | Path, avg: str, output_path: str | Path,
                    raw_path: str | Path, bgpath: str | Path, fyaml: str | Path, bgyaml: str | Path, 
                    save: bool = True, scantype: str = '',norm: bool = True, emi_calib: bool = False,
                    despeckle: bool = True):
        
        self.runs = runs
        self.proc_path = proc_path
        self.avg = avg
        self.output_path = output_path

        try:
            with open(fyaml, 'r') as file:
                 self.yaml = yaml.safe_load(file)
        except FileNotFoundError as fe: 
            raise FileNotFoundError('Config yaml file not found - check filename') from fe


        if len(raw_path) is not None:
            self.raw_path = raw_path
            self.bgpath = bgpath
            self.fyaml = fyaml
            self.bgyaml = bgyaml
            self.scantype = scantype
            self.norm = norm
            self.despeckle = despeckle

        self.check_reduce()

        if 'on' in list(self.average.keys())[0]:
            self.laser = True
        elif 'off' in list(self.average.keys())[0]:
            self.laser = True
        else:
            self.laser = False
        if self.despeckle:
            self.apply_despeckle()
        self.average
        self.get_PFYs()

        if emi_calib:
            self.plot_svls2D_ET()

        if save:
            self.save_avg()


    def check_reduce(self):
        for run in self.runs:
            proc_path = self.proc_path + f'{run:04d}.h5'
            if os.path.isfile(proc_path) == False:
                fname = self.raw_path + f'{run:04d}.h5'
                tmp = Reduced(fname,self.bgpath,self.fyaml,self.bgyaml, 
                        save=True,scantype=self.scantype,norm=self.norm)
                self.scantype = tmp.data.scantype

        
        return
    
    @cached_property
    def average(self):
        try:
            avg = avg_data_count(self.runs, self.proc_path)
        except:
            print('not count averaged')
            avg = avg_data(self.runs, self.proc_path)

        #####ZY_edits - 060126 - use "sum over sum - sos"
        if 'axis_svls_on_sumSV' in avg and 'axis_svls_on_sumI0' in avg:
            avg['axis_svls_on_norm']  = avg['axis_svls_on_sumSV']  / avg['axis_svls_on_sumI0'][:, None]
            avg['axis_svls_off_norm'] = avg['axis_svls_off_sumSV'] / avg['axis_svls_off_sumI0'][:, None]
            avg['PFY_on_sos']  = avg['axis_svls_on_sumSV'].sum(axis=1)  / avg['axis_svls_on_sumI0']
            avg['PFY_off_sos'] = avg['axis_svls_off_sumSV'].sum(axis=1) / avg['axis_svls_off_sumI0']
        ##### ZY_edits - 061726 - add mono+TT 3d
        if 'axis_svls_on_ET_sumSV' in avg and 'axis_svls_on_ET_sumI0' in avg:
            with np.errstate(invalid='ignore', divide='ignore'):
                avg['axis_svls_on_ET'] = (avg['axis_svls_on_ET_sumSV']
                                          / avg['axis_svls_on_ET_sumI0'][:, :, None])
        return avg
    
    def get_PFYs(self):
        if self.laser:
            self.average['PFY_on_mean'],self.average['PFY_on_std'] = get_PFY(self.average['axis_svls_on_mean'],self.average['axis_svls_on_std'])
            self.average['PFY_off_mean'],self.average['PFY_off_std'] = get_PFY(self.average['axis_svls_off_mean'],self.average['axis_svls_off_std'])
        else:
            self.average['PFY_mean'],self.average['PFY_std'] = get_PFY(self.average['axis_svls_mean'],self.average['axis_svls_std'])
    
    def get_emi(self):
        if self.laser:
            px = np.arange(self.average['axis_svls_off_mean'].shape[1])
        else:
            px = np.arange(self.average['axis_svls_mean'].shape[1])

        for i in np.arange(len(self.yaml['emi_calib'])):
            emi_config = np.asarray(self.yaml['emi_calib'][i])
            if np.logical_and(np.all(self.runs>emi_config[0]),np.all(self.runs<emi_config[1])):
                emi_calib = emi_config[2:4]
        emi = emi_calib[0]*px+emi_calib[1]
        return emi

#####ZY_edits - 051726 - add despeckle method
    def apply_despeckle(self, kernel_size=3, n_sigma=10, verbose=True):
        """
        Replace anomaly pixels in all 2D maps inside self.average.

        For each pixel, compute median + MAD over a kernel_size x kernel_size
        neighborhood (excluding the center). If |val - local_median| exceeds
        n_sigma * local_MAD, the pixel is flagged and replaced with the
        local median.

        """
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')
        half = kernel_size // 2
        offsets = [(di, dj) for di in range(-half, half + 1)
                           for dj in range(-half, half + 1)
                           if not (di == 0 and dj == 0)]

        mean_keys = [k for k in ['axis_svls_off_mean', 'axis_svls_on_mean',
                                  'axis_svls_mean'] if k in self.average]

        self.despeckle_masks = {}
        total_flagged = 0
        for mkey in mean_keys:
            Z = np.asarray(self.average[mkey]).astype(float)
            ny, nx = Z.shape

            # build a (n_neighbors, ny, nx) stack with NaN padding at edges
            nbr_stack = np.full((len(offsets), ny, nx), np.nan)
            for k, (di, dj) in enumerate(offsets):
                src_i0 = max(0, -di); src_i1 = ny - max(0, di)
                src_j0 = max(0, -dj); src_j1 = nx - max(0, dj)
                dst_i0 = max(0, di);  dst_i1 = ny - max(0, -di)
                dst_j0 = max(0, dj);  dst_j1 = nx - max(0, -dj)
                nbr_stack[k, dst_i0:dst_i1, dst_j0:dst_j1] = \
                    Z[src_i0:src_i1, src_j0:src_j1]

            local_median = np.nanmedian(nbr_stack, axis=0)
            local_mad    = np.nanmedian(np.abs(nbr_stack - local_median[None]), axis=0)
            local_sigma  = 1.4826 * local_mad

            # avoid divide-by-zero where local MAD is 0 (flat regions)
            floor = np.nanmedian(local_sigma[local_sigma > 0])
            if not np.isfinite(floor):
                floor = np.nanmedian(np.abs(Z)) * 1e-3
            local_sigma = np.where(local_sigma > 0, local_sigma, floor)

            deviation = np.abs(Z - local_median)
            mask = (deviation > n_sigma * local_sigma) & ~np.isnan(Z)

            # replace flagged pixels in the mean map with local median
            Z_clean = Z.copy()
            Z_clean[mask] = local_median[mask]
            self.average[mkey] = Z_clean
            self.despeckle_masks[mkey] = mask

            # propagate mask to companion std/sum arrays: set them to NaN
            laser_tag = mkey.replace('_mean', '')
            for suffix in ('_std', '_sum'):
                ckey = laser_tag + suffix
                if ckey in self.average:
                    arr = np.asarray(self.average[ckey]).astype(float).copy()
                    arr[mask] = np.nan
                    self.average[ckey] = arr

            total_flagged += mask.sum()
            if verbose:
                print(f'  despeckle [{mkey}]: kernel={kernel_size}, '
                      f'n_sigma={n_sigma} -> flagged {mask.sum()} pixels '
                      f'({100*mask.sum()/mask.size:.3f}%)')

        if verbose:
            print(f'  despeckle TOTAL: {total_flagged} pixels across '
                  f'{len(mean_keys)} map(s)')


    def plot_svls2D(self, calibrated=True,savefig=False,transparent=True,figsize=(12,8),scale=1):


        if self.laser:
            if calibrated == True:
                emi = self.get_emi()
                self.average['E_emi'] = emi
            else:
                print('emission is not calibrated')
                emi = np.arange(self.average['axis_svls_off_mean'].shape[1])

            ddatmax = np.nanmax(self.average['axis_svls_on_mean'].T-self.average['axis_svls_off_mean'].T)
    
            fig,ax = plt.subplots(1,3,sharex=True, sharey=True,figsize=figsize)
            ax[0].pcolor(self.average['scanvar_on'],emi,(self.average['axis_svls_off_mean']).T,cmap = 'Reds',
                        vmin=0,vmax=np.nanmax(self.average['axis_svls_off_mean'])/scale)
            ax[1].pcolor(self.average['scanvar_on'],emi,(self.average['axis_svls_on_mean']).T,cmap = 'Reds',
                        vmin=0,vmax=np.nanmax(self.average['axis_svls_on_mean'])/scale)
            ax[2].pcolor(self.average['scanvar_on'],emi,(self.average['axis_svls_on_mean']).T-(self.average['axis_svls_off_mean']).T,cmap = 'bwr',
                        vmin=-ddatmax,vmax=ddatmax)

            ax[0].set_xlabel('inc. energy (eV)')
            ax[1].set_xlabel('inc. energy (eV)')
            ax[2].set_xlabel('inc. energy (eV)')

            if calibrated == True:
                ax[0].set_ylabel('emission (eV)')
            else:
                ax[0].set_ylabel('emission (pixel)')

            ax[0].set_title('laser off')
            ax[1].set_title('laser on')
            ax[2].set_title('difference')

            ax[0].set_xlim([np.nanmin(self.average['scanvar_on']),np.nanmax(self.average['scanvar_on'])])
            ax[0].set_title(f'Runs {self.runs[0]} to {self.runs[-1]}')
        else:
            if calibrated == True:
                emi = self.get_emi()
                self.average['E_emi'] = emi
            else:
                print('emission is not calibrated')
                emi = np.arange(self.average['axis_svls_mean'].shape[1])

            datmax = np.nanmax(self.average['axis_svls_mean'].T)
    
            fig,ax = plt.subplots(1,1,sharex=True, sharey=True,figsize=figsize)
            ax.pcolor(self.average['scanvar'],emi,(self.average['axis_svls_mean']).T,cmap = 'Reds',
                        vmin=0,vmax=np.nanmax(self.average['axis_svls_mean'])/scale)

            ax.set_xlabel('inc. energy (eV)')

            if calibrated == True:
                ax.set_ylabel('emission (pixel)')
            else:
                ax.set_ylabel('emission (pixel)')

            ax.set_xlim([np.nanmin(self.average['scanvar']),np.nanmax(self.average['scanvar'])])
            ax.set_title(f'Runs {self.runs[0]} to {self.runs[-1]}')

        if savefig:
            fig.savefig(f'figs/SVLS2D_{self.runs[0]}_{self.runs[-1]}.png',transparent=transparent,
                        dpi=200, bbox_inches='tight')

        # return fig

    def plot_svls1D(self, savefig=False, transparent=True, figsize=(12,8), plot_err=True, mode='mean'):
        """Plotting binned and averaged SVLS detector, collapsed on scanvar axis."""
        # NEW: mode='mean' or 'sos'
        key_pfy  = 'sos' if mode == 'sos' else 'mean'
        plot_std = plot_err and (mode == 'mean')          # std bands not defined in sos mode

        if self.laser == True:
            PFY_on  = self.average[f'PFY_on_{key_pfy}']
            PFY_off = self.average[f'PFY_off_{key_pfy}']

            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)
            ax[0].plot(self.average['scanvar_off'], PFY_off, color='tab:blue')
            ax[0].plot(self.average['scanvar_on'],  PFY_on,  color='tab:orange')
            ax[1].plot(self.average['scanvar_on'],  PFY_on - PFY_off, color='tab:blue')
            ax[1].plot(self.average['scanvar_on'],  np.zeros(len(self.average['scanvar_on'])), '--k')

            if plot_std:
                PFY_on_std  = self.average['PFY_on_std']
                PFY_off_std = self.average['PFY_off_std']
                ax[0].fill_between(self.average['scanvar_off'], PFY_off - PFY_off_std, PFY_off + PFY_off_std,
                                   alpha=0.2, color='tab:blue')
                ax[0].fill_between(self.average['scanvar_off'], PFY_on  - PFY_on_std,  PFY_on  + PFY_on_std,
                                   alpha=0.2, color='tab:orange')
                dPFYerr = np.sqrt(PFY_on_std**2 + PFY_off_std**2)
                ax[1].fill_between(self.average['scanvar_off'],
                                   (PFY_on - PFY_off) - dPFYerr,
                                   (PFY_on - PFY_off) + dPFYerr,
                                   alpha=0.2, color='tab:blue')
                ax[2].plot(self.average['scanvar_on'],  PFY_on_std,  color='tab:orange')
                ax[2].plot(self.average['scanvar_off'], PFY_off_std, color='tab:blue')
                ax[2].plot(self.average['scanvar_on'],  dPFYerr,     color='tab:green')
                ax[2].set_ylabel('error')

            if self.scantype == 'mono_fly':
                for a in ax: a.set_xlabel('inc. energy (eV)')
            elif self.scantype == 'delay_fly':
                for a in ax: a.set_xlabel('delay (s)')

            ax[0].set_xlim([np.nanmin(self.average['scanvar_on']), np.nanmax(self.average['scanvar_on'])])
            ax[0].set_title(f'Runs {self.runs[0]} to {self.runs[-1]}   [{mode}]')
        else:
            PFY = self.average[f'PFY_{key_pfy}'] if mode == 'sos' else self.average['PFY_mean']
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(self.average['scanvar'], PFY)
            if plot_std:
                ax.fill_between(self.average['scanvar'],
                                PFY - self.average['PFY_std'],
                                PFY + self.average['PFY_std'],
                                alpha=0.2)
            if self.scantype == 'mono_fly':
                ax.set_xlabel('inc. energy (eV)')
            elif self.scantype == 'delay_fly':
                ax.set_xlabel('delay (s)')
            ax.set_xlim([np.nanmin(self.average['scanvar']), np.nanmax(self.average['scanvar'])])
            ax.set_title(f'Runs {self.runs[0]} to {self.runs[-1]}   [{mode}]')

        if savefig:
            fig.savefig(f'figs/SVLS1D_{self.runs[0]}_{self.runs[-1]}_{mode}.png',
                        transparent=transparent, dpi=200, bbox_inches='tight')
       
    def plot_svls2D_ET(self, savefig=False, transparent=True, figsize=(12,8), scale=1, ETstep=0.2, mode='mean'):
        # NEW: mode='mean' or 'sos'
        key_2d = 'norm' if mode == 'sos' else 'mean'

        try:
            self.average['E_emi'] = self.get_emi()
        except:
            print('emission is not calibrated, cannot plot energy transfer')

        if self.laser:
            on_data  = self.average[f'axis_svls_on_{key_2d}']
            off_data = self.average[f'axis_svls_off_{key_2d}']
            if mode == 'mean':
                on_std  = self.average['axis_svls_on_std']
                off_std = self.average['axis_svls_on_std']   # preserving existing behavior (on_std passed for off too)
            else:
                on_std  = np.zeros_like(on_data)
                off_std = np.zeros_like(off_data)

            mono_on,  E_trans_on,  data_trans_on,  std_trans_on  = emi2ET(self.average['scanvar_on'],  self.average['E_emi'], on_data,  on_std,  ETstep)
            mono_off, E_trans_off, data_trans_off, std_trans_off = emi2ET(self.average['scanvar_off'], self.average['E_emi'], off_data, off_std, ETstep)

            self.average['mono_on']        = mono_on
            self.average['mono_off']       = mono_off
            self.average['E_trans_on']     = E_trans_on
            self.average['E_trans_off']    = E_trans_off
            self.average['data_trans_on']  = data_trans_on
            self.average['data_trans_off'] = data_trans_off
            self.average['std_trans_on']   = std_trans_on
            self.average['std_trans_off']  = std_trans_off

            ddatmax = np.nanmax(on_data.T - off_data.T)

            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)
            ax[0].pcolor(mono_on,  E_trans_on,  data_trans_on.T,  cmap='Reds',
                         vmin=0, vmax=np.nanmax(data_trans_on)/scale, shading='auto')
            ax[1].pcolor(mono_off, E_trans_off, data_trans_off.T, cmap='Reds',
                         vmin=0, vmax=np.nanmax(data_trans_off)/scale, shading='auto')
            ax[2].pcolor(mono_on,  E_trans_on, (data_trans_on - data_trans_off).T, cmap='bwr',
                         vmin=-ddatmax, vmax=ddatmax, shading='auto')

            ax[0].set_xlabel('inc. energy (eV)')
            ax[1].set_xlabel('inc. energy (eV)')
            ax[2].set_xlabel('inc. energy (eV)')
            ax[0].set_ylabel('energy transfer (eV)')
            ax[0].set_title('laser off')
            ax[1].set_title('laser on')
            ax[2].set_title('difference')
            ax[0].set_xlim([np.nanmin(self.average['scanvar_on']), np.nanmax(self.average['scanvar_on'])])
            ax[0].set_title(f'Runs {self.runs[0]} to {self.runs[-1]}   [{mode}]')
        else:
            data = self.average[f'axis_svls_{key_2d}'] if mode == 'sos' else self.average['axis_svls_mean']
            std  = np.zeros_like(data) if mode == 'sos' else self.average['axis_svls_std']
            mono, E_trans, data_trans, std_trans = emi2ET(self.average['scanvar'], self.average['E_emi'], data, std, ETstep)

            self.average['mono']       = mono
            self.average['E_trans']    = E_trans
            self.average['data_trans'] = data_trans
            self.average['std_trans']  = std_trans

            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=figsize)
            ax.pcolor(mono, E_trans, data_trans.T, cmap='Reds',
                      vmin=0, vmax=np.nanmax(data_trans)/scale, shading='auto')
            ax.set_xlabel('inc. energy (eV)')
            ax.set_ylabel('energy transfer (eV)')
            ax.set_xlim([np.nanmin(self.average['scanvar']), np.nanmax(self.average['scanvar'])])
            ax.set_title(f'Runs {self.runs[0]} to {self.runs[-1]}   [{mode}]')

        if savefig:
            fig.savefig(f'figs/SVLS2D_ET_{self.runs[0]}_{self.runs[-1]}_{mode}.png',
                        transparent=transparent, dpi=200, bbox_inches='tight')
            

    def elastic_calibrate_from_two_points(self,
        mono,
        rixs_map_full,
        p1, p2,
        width_pixels=20,
        plot_on=True
        ):
        """
        Calibrate using a parallelogram ROI defined by two points and a fixed width.

        Parameters
        ----------
        mono : (nx,) array
            X-axis values (energy). Assumed monotonic; used on FIRST axis of rixs_map_full.
        rixs_map_full : (nx, ny) array
            2D map with first index along mono (x), second index = pixel (y).
        p1, p2 : tuple
            Two points defining the *center line* of the ROI, as (x_value_in_mono_units, y_pixel_index).
            Example: p1 = (x1, y1_pixel), p2 = (x2, y2_pixel).
        width_pixels : int or float
            Total vertical thickness of the parallelogram in pixels (constant in y).
            Top/Bottom are ±width/2 around the center line y(x) in pixel space.
        plot_on : bool
            If True, show a diagnostic plot.

        Returns
        -------
        calibrated_axis : (ny,) array
            mono ≈ a*pixel + b evaluated over all pixels.
        fit : (a, b)
            Linear fit parameters such that mono ≈ a*pixel + b.
        details : dict
            Extra info: sampled (x_mono, y_pix) of maxima, indices, etc.
        """
        mono = np.asarray(mono)
        Z = np.asarray(rixs_map_full)
        nx, ny = Z.shape

        # --- Convert the two points to (ix, iy) using mono -> x index
        x1, y1 = p1
        x2, y2 = p2
        ind_x1 = find_nearest(mono, x1)
        ind_x2 = find_nearest(mono, x2)
        px_y1 = int(round(y1))
        px_y2 = int(round(y2))
        # --- Center line
        m =(px_y2-px_y1)/(mono[ind_x2]-mono[ind_x1])
        c = px_y1-m*mono[ind_x1]

        half_w = 0.5 * float(width_pixels)

        # --- Determine x-span (in indices) to scan
        ix_start = min(ind_x1, ind_x2)
        ix_end   = max(ind_x1, ind_x2)

        x_samples = []
        ypix_at_max = []
        used_ix = []

        # --- Scan each x index within the span; search along y within top/bottom bounds
        for i in range(ix_start, ix_end + 1):
            x_here = mono[i]
            y_center = m * x_here + c
            y_lo = int(np.floor(y_center - half_w))
            y_hi = int(np.ceil (y_center + half_w))
            if y_lo > y_hi:
                y_lo, y_hi = y_hi, y_lo
            y_lo = max(0, min(ny - 1, y_lo))
            y_hi = max(0, min(ny - 1, y_hi))
            if y_hi < y_lo:
                continue

            col_slice = Z[i, y_lo:y_hi + 1]
            if col_slice.size == 0 or np.all(np.isnan(col_slice)):
                continue

            rel = int(np.nanargmax(col_slice))
            iy_max = y_lo + rel

            x_samples.append(x_here)
            ypix_at_max.append(iy_max)
            used_ix.append(i)

        x_samples = np.asarray(x_samples)
        ypix_at_max = np.asarray(ypix_at_max)

        if x_samples.size < 2:
            raise RuntimeError("Not enough maxima found within the parallelogram to fit a line.")

        # --- Fit mono ≈ a * pixel + b  (same calibration as your original)
        a, b = np.polyfit(ypix_at_max, x_samples, 1)
        calibrated_axis = np.polyval([a, b], np.arange(ny))

        # --- Optional diagnostic plot
        if plot_on:
            plt.figure(figsize=(6, 4))
            # Display Z with axes: x = mono (horizontal), y = pixel (vertical)
            plt.pcolormesh(mono,range(0,Z.shape[1]),Z.T,cmap='Reds',vmin=0,vmax=np.max(Z))
            plt.colorbar(label='Intensity')

            # Draw the two points
            plt.scatter([mono[ind_x1], mono[ind_x2]], [px_y1, px_y2],
                        s=60, c='white', edgecolor='k')

            # Draw center and top/bottom edges
            x_line = np.linspace(mono[ix_start], mono[ix_end], 200)
            y_center = m * x_line + c
            y_top = y_center + half_w
            y_bot = y_center - half_w
            plt.plot(x_line, y_center, 'w--', lw=1.5, color='k',label='initial guess')
            plt.plot(x_line, y_top,    'w-',  lw=1.0, alpha=0.8,color='k')
            plt.plot(x_line, y_bot,    'w-',  lw=1.0, alpha=0.8,color='k')

            # Scatter maxima used for the fit
            plt.scatter(x_samples, ypix_at_max, s=20, c='yellow', edgecolor='k')

            # Plot fitted calibration line (x vs pixel)
            ypix = np.arange(ny)
            plt.plot(np.polyval([a, b], ypix), ypix, 'r-', lw=2,
                     label=f'fit: mono = {a:.6g} * pixel + {b:.6g}')
            plt.xlim(mono[0],mono[-1])
            plt.xlabel('Energy (mono)')
            plt.ylabel('Pixel')
            plt.title(f'Elastic Calibration via Parallelogram (width={width_pixels} px)')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

        details = {
            "x_samples": x_samples,
            "ypix_at_max": ypix_at_max,
            "used_x_indices": np.array(used_ix, dtype=int),
            "center_line_m_c": (m, c),
            "width_pixels": width_pixels,
        }
        return calibrated_axis, (a, b), details
    
    def emi_calibration(self, p1, p2, width_pixels = 10, plot_on = True, use = 'on'):
        if use == 'on':
            mono = self.average['scanvar_on']
            rixs = self.average['axis_svls_on_mean']
        elif use == 'off':
            mono = self.average['scanvar_off']
            rixs = self.average['axis_svls_off_mean']
        else:
            mono = self.average['scanvar']
            rixs = self.average['axis_svls_mean']

        calibrated_axis, (a, b), details = self.elastic_calibrate_from_two_points(
                                    mono,
                                    rixs,
                                    p1, p2,
                                    width_pixels=width_pixels,
                                    plot_on=plot_on
                                 )
        self.average['E_emi'] = calibrated_axis
        return calibrated_axis, (a, b), details

    def save_avg(self):
        runs = self.runs
        output = h5py.File(f'./avg/Run{runs[0]:04d}to{runs[-1]:04d}.h5','w')
        fname = f'./avg/Run{runs[0]:04d}to{runs[-1]:04d}.h5'
        print(f'saving data in {fname}')
        keys = self.average.keys()
        print(keys)

        for key in keys:
            output.create_dataset(key,dtype='f',data=self.average[key])

        output.close()

    
    def plot_CIE(self, inc_energy, width = 2, savefig=False,transparent=True,figsize=(12,8),xlim =[]):
        '''
        Plot Constant incident energy cuts for given energies
        
        Parameters
        ----------
        inc_energy : float
        center energy at which the CIE will be taken

        width: integer
        number of pixels 'left and right' of center energy which will be used for CIE sum

        savefig : boolean
        save figure or not

        transparent : boolean
        background for saved figure transparent or not

        figsize : array 
        size of figure

        xlim : array
        limits of x-axis (energy transfer axis) to be plotted

        '''
        if self.laser:
            CIE_on,CIE_on_std = get_CIE(inc_energy,width,self.average['mono_on'],self.average['data_trans_on'],self.average['std_trans_on'])
            CIE_off,CIE_off_std = get_CIE(inc_energy,width,self.average['mono_off'],self.average['data_trans_off'],self.average['std_trans_off'])
            dCIEerr = 1/2*np.sqrt(CIE_on_std**2+CIE_off_std**2)
            if xlim==[]:
                xlim = [np.nanmin(self.average['E_trans_on']),np.nanmax(self.average['E_trans_on'])]
            # inds = np.argmin(np.abs(np.asarray(self.average['mono_off'])-inc_energy))
            # self.average['CIE_off'] = np.nansum(np.asarray(self.average['data_trans_off'])[inds-width:inds+width,:],axis=0)
            # inds = np.argmin(np.abs(np.asarray(self.average['mono_on'])-inc_energy))
            # self.average['CIE_on'] = np.nansum(np.asarray(self.average['data_trans_on'])[inds-width:inds+width,:],axis=0)
            fig,ax=plt.subplots(1,2,sharex=True,figsize=figsize)
            ax[0].plot(self.average['E_trans_off'], CIE_off,label='laser off',color='tab:blue')
            ax[0].fill_between(self.average['E_trans_off'], CIE_off-CIE_off_std, CIE_off+CIE_off_std,color='tab:blue',alpha=0.2)
            ax[0].plot(self.average['E_trans_on'], CIE_on,label='laser on',color='tab:orange')
            ax[0].fill_between(self.average['E_trans_off'], CIE_on-CIE_on_std, CIE_on+CIE_on_std,color='tab:orange',alpha=0.2)
            ax[0].legend()
            ax[0].set_xlabel('energy transfer (eV)')
            ax[0].set_ylabel('intensity (arb. u.)')
            ax[1].plot(self.average['E_trans_off'], CIE_on-CIE_off,label='laser off',color='tab:blue')
            ax[1].plot(self.average['E_trans_off'],np.zeros(len(self.average['E_trans_off'])),'--k')
            ax[1].fill_between(self.average['E_trans_off'], CIE_on-CIE_off-dCIEerr, CIE_on-CIE_off+dCIEerr,color='tab:blue',alpha=0.2)
            ax[1].set_xlabel('energy transfer (eV)')
            ax[0].set_xlim(xlim)
        else:
            if xlim==[]:
                xlim = [np.nanmin(self.average['data_trans']),np.nanmax(self.average['data_trans'])]
            inds = np.argmin(np.abs(np.asarray(self.average['mono'])-inc_energy))
            self.average['CIE'] = np.nansum(np.asarray(self.average['data_trans'])[inds-width:inds+width,:],axis=0)
            fig,ax=plt.subplots(1,1)
            ax.plot(self.average['E_trans'], self.average['CIE'])
            ax.set_xlabel('energy transfer (eV)')
            ax.set_ylabel('intensity (arb. u.)')
            ax.set_xlim(xlim)
        if savefig:
            fig.savefig(f'figs/SVLS_CIE_{self.runs[0]}_{self.runs[-1]}.png',transparent=transparent,
                        dpi=200, bbox_inches='tight')
       

# cutE = 528.5
# inds = np.argmin(np.abs(np.asarray(OK_mono['mono_off'])-cutE))
# cutE = 528.3
# ind_comp = np.argmin(np.abs(np.asarray(dat_o_es['arr_0'])-(cutE-8)))
# f = 1e4

# ax[0].plot(np.asarray(OK_mono['E_trans_off']), np.nansum(np.asarray(OK_mono['data_trans_off'])[inds-2:inds+2,:],axis=0),label='laser off, 522.8eV')
# ax[1].plot(np.asarray(OK_mono['E_trans_off']), np.nansum(np.asarray(OK_mono['data_trans_on'])[inds-2:inds+2,:],axis=0),label='laser on')

    ##### plotting method for 3D data
    def plot_mono_at_delay(self, tau, emi_lo=None, emi_hi=None, diff=False, figsize=(8,4)):
            """Mono spectrum (PFY vs energy) at the nearest absolute-time bin to `tau` (s)."""
            ET = self.average['axis_svls_on_ET']                      # (nT, nE, emission)
            tb, eb = self.yaml['time_bins'], self.yaml['bins']
            tcen = 0.5*(np.linspace(float(tb[1][0]),float(tb[1][1]),int(tb[1][2]))[:-1]
                        + np.linspace(float(tb[1][0]),float(tb[1][1]),int(tb[1][2]))[1:])
            ecen = 0.5*(np.linspace(float(eb[1][0]),float(eb[1][1]),int(eb[1][2]))[:-1]
                        + np.linspace(float(eb[1][0]),float(eb[1][1]),int(eb[1][2]))[1:])
            it = int(np.argmin(np.abs(tcen - tau)))
            sl = slice(None) if emi_lo is None else slice(emi_lo, emi_hi+1)

            pfy_on = np.nansum(ET[it][:, sl], axis=1)
            plt.figure(figsize=figsize)
            if diff:
                pfy_off = np.nansum(self.average['axis_svls_off_norm'][:, sl], axis=1)
                plt.plot(ecen, pfy_on - pfy_off, 'k'); plt.ylabel('on − off PFY (SoS)')
            else:
                plt.plot(ecen, pfy_on); plt.ylabel('PFY (SoS)')
            plt.xlabel('inc. energy (eV)')
            plt.title(f'mono spectrum at τ ≈ {tcen[it]*1e15:.0f} fs')
            plt.tight_layout(); plt.show()
            return ecen, pfy_on