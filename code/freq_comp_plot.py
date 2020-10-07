import bjlib.class_faraday as cf
import numpy as np
from astropy import units as u
import bjlib.cl_lib as cl_lib
import matplotlib.pyplot as plt
import IPython


def main():
    # nsim = 192
    # rotation_angle = (angle_arg * u.deg).to(u.rad)
    nside = 512
    r = 0.0
    lmin = 30
    lmax = 500
    nlb = 10
    # f_sky_SAT = 0.1

    custom_bins = True

    b = cl_lib.binning_definition(nside, lmin=lmin, lmax=lmax,
                                  nlb=nlb, custom_bins=custom_bins)

    path_noCMB27 = '/global/homes/j/jost/these/spectra_based_analysis/results_and_data/noCMB_27GHz_16sim_alpha0p0deg/'
    path_noCMB280 = '/global/homes/j/jost/these/spectra_based_analysis/results_and_data/noCMB_280GHz_16sim_alpha0p0deg/'

    # path = '/global/homes/j/jost/these/spectra_based_analysis/results_and_data/27GHz_192sim_alpha0p0deg/'
    # total_fit = np.load(path+'fit_alpha_cmb_dust_sync.npy')
    # std_totalfit = np.std(total_fit)

    fit27_dust = np.load(path_noCMB27 + 'fit_alpha_cmb_dust_EB.npy')
    fit280_dust = np.load(path_noCMB280 + 'fit_alpha_cmb_dust_EB.npy')
    print('fit27_dust = ', fit27_dust)
    print('fit280_dust = ', fit280_dust)
    fit27_dust_all = np.load(path_noCMB27 + 'fit_alpha_cmb_dust.npy')
    print('fit27_dust_all = ', fit27_dust_all)

    fit27_sync = np.load(path_noCMB27 + 'fit_alpha_cmb_sync_EB.npy')
    fit280_sync = np.load(path_noCMB280 + 'fit_alpha_cmb_sync_EB.npy')
    print('fit27_sync = ', fit27_sync)
    print('fit280_sync = ', fit280_sync)

    exit()
    angle27_dust = np.mean(fit27_dust)
    angle280_dust = np.mean(fit280_dust)

    angle27_sync = np.mean(fit27_sync)
    angle280_sync = np.mean(fit280_sync)

    dust_min = min(min(fit27_dust), min(fit280_dust))[0]
    sync_min = min(min(fit27_sync), min(fit280_sync))[0]

    dust_max = max(max(fit27_dust), max(fit280_dust))[0]
    sync_max = max(max(fit27_sync), max(fit280_sync))[0]

    # print('angle27_dust - angle280_dust = ', angle27_dust - angle280_dust)
    # print('angle27_sync - angle280_sync = ', angle27_sync - angle280_sync)
    # print('')
    print('angle27_dust = ', angle27_dust)
    print('angle280_dust = ', angle280_dust)

    print('angle27_sync = ', angle27_sync)
    print('angle280_sync = ', angle280_sync)

    Cl_dust27 = np.load(path_noCMB27+'Cl_dust.npy')[0]
    Cl_dust280 = np.load(path_noCMB280+'Cl_dust.npy')[0]

    # Cl_cmbdust27 = np.load(path_noCMB27+'Cl_cmb_dust.npy')[0]
    # print('Cl_dust27 - Cl_cmbdust27 = ', Cl_dust27 - Cl_cmbdust27)

    Cl_sync27 = np.load(path_noCMB27+'Cl_sync.npy')[0]
    Cl_sync280 = np.load(path_noCMB280+'Cl_sync.npy')[0]

    ell = np.load(path_noCMB27+'effective_ells.npy')

    fit_spectra_dust = cf.power_spectra_operation(r=r, rotation_angle=angle27_dust * u.rad,
                                                  powers_name='total')
    fit_spectra_dust.get_spectra()
    fit_spectra_dust.spectra_rotation()
    fit_spectra_dust27_binned = b.bin_cell(fit_spectra_dust.cl_rot.spectra[:3*nside].T).T

    fit_spectra_dust.spectra_rotation(dust_min * u.rad)
    fit_spectra_dust_min_binned = b.bin_cell(fit_spectra_dust.cl_rot.spectra[:3*nside].T).T

    fit_spectra_dust.spectra_rotation(dust_max * u.rad)
    fit_spectra_dust_max_binned = b.bin_cell(fit_spectra_dust.cl_rot.spectra[:3*nside].T).T

    fit_spectra_dust.spectra_rotation(angle280_dust * u.rad)
    fit_spectra_dust280_binned = b.bin_cell(fit_spectra_dust.cl_rot.spectra[:3*nside].T).T

    index_list = [0, 1, 3]
    camb_index = [1, 4, 2]

    spectra = {0: 'EE', 1: 'EB', 2: 'EB', 3: 'BB'}
    j = 0
    for i in index_list:
        print('i =', i, ':: j =', j, ':: camb_index[j] =',
              camb_index[j], ':: spectra[i]=', spectra[i])
        plt.plot(ell, Cl_dust27[i], label='dust 27GHz')
        plt.plot(ell, Cl_dust280[i], label='dust 280GHz')
        plt.plot(ell, fit_spectra_dust27_binned.T[camb_index[j]],
                 label='CAMB*birefringence({:.3f}) fit 27GHz'.format(angle27_dust*u.rad))
        plt.plot(ell, fit_spectra_dust280_binned.T[camb_index[j]],
                 label='CAMB*birefringence({:.3f}) fit 280GHz'.format(angle280_dust*u.rad))
        plt.fill_between(b.get_effective_ells(), fit_spectra_dust_min_binned.T[camb_index[j]],
                         fit_spectra_dust_max_binned.T[camb_index[j]], color='red', alpha=0.2,
                         label='from {:.3f} to {:.3f}'.format(dust_min * u.rad, dust_max * u.rad))
        # plt.plot(ell, fit_spectra_dust_min_binned.T[camb_index[j]],
        #          label='0.rad')
        # plt.plot(ell, fit_spectra_dust_max_binned.T[camb_index[j]],
        #          label='0.1 rad')
        plt.legend()
        plt.xscale('log')
        # if i == 0 or i == 3:
        plt.yscale('symlog', linthreshy=5e-10)
        plt.title(r'Dust spectra at 27 & 280GHz, $\alpha$=0deg')
        plt.xlabel(r'$\ell$', fontsize=22)
        plt.ylabel(r'$C_{\ell}^{'+spectra[i]+'}$', fontsize=22)
        plt.tight_layout()
        plt.savefig(path_noCMB27+'Cell'+spectra[i]+'_27vs280GHz_Dust'
                    '_alpha0p0deg_EBonlyinfit.png')
        plt.close()
        j += 1
    # IPython.embed()
    fit_spectra_sync = cf.power_spectra_operation(r=r, rotation_angle=angle27_sync * u.rad,
                                                  powers_name='total')
    fit_spectra_sync.get_spectra()
    fit_spectra_sync.spectra_rotation()
    fit_spectra_sync27_binned = b.bin_cell(fit_spectra_sync.cl_rot.spectra[:3*nside].T).T

    # fit_spectra_sync.spectra_rotation(-angle27_sync * u.rad)
    # fit_spectra_sync_abs_binned = b.bin_cell(fit_spectra_dust.cl_rot.spectra[:3*nside].T).T

    fit_spectra_sync.spectra_rotation(sync_min * u.rad)
    fit_spectra_sync_min_binned = b.bin_cell(fit_spectra_sync.cl_rot.spectra[:3*nside].T).T

    fit_spectra_sync.spectra_rotation(sync_max * u.rad)
    fit_spectra_sync_max_binned = b.bin_cell(fit_spectra_sync.cl_rot.spectra[:3*nside].T).T

    fit_spectra_sync.spectra_rotation(angle280_sync * u.rad)
    fit_spectra_sync280_binned = b.bin_cell(fit_spectra_sync.cl_rot.spectra[:3*nside].T).T

    index_list = [0, 1, 3]
    camb_index = [1, 2, 4]
    spectra = {0: 'EE', 1: 'EB', 2: 'EB', 3: 'BB'}
    j = 0
    for i in index_list:
        plt.plot(ell, Cl_sync27[i], label='Sync 27GHz')
        plt.plot(ell, Cl_sync280[i], label='Sync 280GHz')
        # plt.plot(ell, fit_spectra_sync_binned.T[camb_index[j]],
        #          label='CAMB*birefringence({})'.format(angle27_sync*u.rad))
        # plt.fill_between(b.get_effective_ells(), fit_spectra_sync_min_binned.T[camb_index[j]],
        #                  fit_spectra_sync_max_binned.T[camb_index[j]], color='red', alpha=0.2,
        #                  label='from 0.rad to 0.1rad')
        plt.plot(ell, fit_spectra_sync27_binned.T[camb_index[j]],
                 label='CAMB*birefringence({:.3f}) fit 27GHz'.format(angle27_sync*u.rad))
        plt.plot(ell, fit_spectra_sync280_binned.T[camb_index[j]],
                 label='CAMB*birefringence({:.3f}) fit 280GHz'.format(angle280_sync*u.rad))
        plt.fill_between(b.get_effective_ells(), fit_spectra_sync_min_binned.T[camb_index[j]],
                         fit_spectra_sync_max_binned.T[camb_index[j]], color='red', alpha=0.2,
                         label='from {:.3f} to {:.3f}'.format(sync_min * u.rad, sync_max * u.rad))

        plt.legend()
        plt.xscale('log')
        # if i == 0 or i == 3:
        plt.yscale('symlog', linthreshy=5e-10)
        plt.title(r'Synchrotron spectra at 27 & 280GHz, $\alpha$=0deg')
        plt.xlabel(r'$\ell$', fontsize=22)
        plt.ylabel(r'$C_{\ell}^{'+spectra[i]+'}$', fontsize=22)
        plt.tight_layout()
        plt.savefig(path_noCMB27+'Cell'+spectra[i]+'_27vs280GHz_Synchrotron'
                    '_alpha0p0deg_EBonlyinfit.png')
        plt.close()
        j += 1


if __name__ == "__main__":
    main()
