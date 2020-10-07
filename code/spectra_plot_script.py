import numpy as np
import matplotlib.pyplot as plt
import IPython
import argparse
# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, help='birefringence angle in DEGREE',
                    default=0)
parser.add_argument('--Nsim', type=int, help='number of simulations',
                    default=1)
parser.add_argument('--Frequency', type=int, help='Frequency in GHz')
# parser.add_argument('--path_to_Cells', type=str,
#                     help='Path to folder storing the spectra',
#                     default='./')

args = parser.parse_args()

nsim = args.Nsim
angle = args.alpha
freq = args.Frequency
# path = args.path_to_Cells

path = '/global/homes/j/jost/these/spectra_based_analysis/results_and_data/noCMB_' +\
    str(freq)+'GHz_'+str(nsim)+'sim_alpha' +\
    str(angle).replace('.', 'p')+'deg/'

cmb = np.load(path+'Cl_cmb.npy')
dust = np.load(path+'Cl_dust.npy')
sync = np.load(path+'Cl_sync.npy')
cmb_dust_sync = np.load(path+'Cl_cmb_dust_sync.npy')
ell = np.load(path+'effective_ells.npy')
noise = np.load(path+'Cl_noise.npy')


index_list = [0, 1, 3]
spectra = {0: 'EE', 1: 'EB', 2: 'EB', 3: 'BB'}
for i in index_list:
    plt.plot(ell, noise[0, i], label='noise')
    plt.plot(ell, cmb[0, i], label='CMB + noise')
    plt.plot(ell, dust[0, i], label='Dust')
    plt.plot(ell, sync[0, i], label='Sync')
    plt.plot(ell, cmb_dust_sync[0, i], label='CMB + noise + dust + sync')
    plt.legend()
    plt.xscale('log')
    if i == 0 or i == 3:
        plt.yscale('log')
    plt.title('Spectra at '+str(freq)+r'GHz, $\alpha$='+str(angle)+'deg')
    plt.xlabel(r'$\ell$', fontsize=22)
    plt.ylabel(r'$C_{\ell}^{'+spectra[i]+'}$', fontsize=22)
    plt.tight_layout()
    plt.savefig(path+'Cell'+spectra[i]+'_'+str(freq) +
                'GHz_alpha'+str(angle).replace('.', 'p')+'deg.png')
    plt.close()
# IPython.embed()
