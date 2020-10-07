import numpy as np
import matplotlib.pyplot as plt
import IPython
import argparse
from astropy import units as u

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, help='birefringence angle in DEGREE',
                    default=0)
parser.add_argument('--Nsim', type=int, help='number of simulations',
                    default=1)
parser.add_argument('--Frequency', type=int, help='Frequency in GHz')
# parser.add_argument('--path_to_FitAlpha', type=str,
#                     help='Path to folder storing angle estimation results for nsim',
#                     default='./')

args = parser.parse_args()

nsim = args.Nsim
angle = args.alpha
freq = args.Frequency
# path = args.path_to_FitAlpha
path = '/global/homes/j/jost/these/spectra_based_analysis/results_and_data/noCMB_' +\
    str(freq)+'GHz_'+str(nsim)+'sim_alpha' +\
    str(angle).replace('.', 'p')+'deg/'

cmb = np.load(path+'fit_alpha_cmb.npy')
cmb_dust = np.load(path+'fit_alpha_cmb_dust.npy')
cmb_sync = np.load(path+'fit_alpha_cmb_sync.npy')
cmb_dust_sync = np.load(path+'fit_alpha_cmb_dust_sync.npy')
# ell = np.load('effective_ells.npy')
# noise = np.load('Cl_noise.npy')


# i = 3  # spectra index
# index_list = [0, 1, 3]
# spectra = {0: 'EE', 1: 'EB', 2: 'EB', 3: 'BB'}
plt.close()

# for i in index_list:
# plt.plot(ell, noise, 'noise')

# plt.hist(cmb, label='CMB + noise', alpha=0.3)
plt.hist(cmb_dust, label='dust', alpha=0.3, histtype='step')
plt.hist(cmb_sync, label='sync', alpha=0.3, histtype='step')
plt.hist(cmb_dust_sync, label='dust + sync', histtype='step',
         color='red')
plt.legend()


plt.xlabel('radian', fontsize=22)
plt.title(r'Distribution angle estimation over {} simulations at {}GHz $\alpha$='.format(
    nsim, freq)+str((angle*u.deg).to(u.rad)))
plt.tight_layout()
plt.savefig(path+'sep_hist_'+str(nsim)+'sim_'+str(freq) +
            'GHz_alpha'+str(angle).replace('.', 'p')+'deg.png')
plt.close()
