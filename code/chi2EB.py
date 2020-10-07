from scipy.optimize import minimize
# import bjlib.likelihood_SO as l_SO
import bjlib.class_faraday as cf
import numpy as np
from astropy import units as u
import bjlib.cl_lib as cl_lib
# import argparse
import IPython
import matplotlib.pyplot as plt


def chi2(angle, model, data, nside, bin):
    angle = angle * u.rad
    model.spectra_rotation(angle)
    model.l_min_instru = 0
    model.l_max_instru = 3*nside
    model.get_noise()
    model.get_instrument_spectra()
    model_spectra = bin.bin_cell(model.instru_spectra.spectra[:3*nside].T).T
    chi2 = np.abs(np.sum(model_spectra.T[4] - data))
    return chi2


def main():
    angle_arg = 0.0
    freq_arg = 280
    nsim = 192
    rotation_angle = (angle_arg * u.deg).to(u.rad)
    nside = 512
    r = 0.0
    # root = 0
    lmin = 30
    lmax = 500
    nlb = 10
    # pysm_model = 'c1s0d0'
    # f_sky_SAT = 0.1
    custom_bins = True

    b = cl_lib.binning_definition(nside, lmin=lmin, lmax=lmax,
                                  nlb=nlb, custom_bins=custom_bins)

    rotation_angle_model_init = rotation_angle  # 0*u.rad
    spectra_cl = cf.power_spectra_operation(r=r,
                                            rotation_angle=rotation_angle_model_init,
                                            powers_name='total')
    spectra_cl.get_spectra()
    spectra_cl.spectra_rotation()

    spectra_cl.l_min_instru = 0
    spectra_cl.l_max_instru = 3*nside
    spectra_cl.get_noise()
    spectra_cl.get_instrument_spectra()

    path = '/global/homes/j/jost/these/spectra_based_analysis/results_and_data/' +\
        str(freq_arg)+'GHz_'+str(nsim)+'sim_alpha' +\
        str(angle_arg).replace('.', 'p')+'deg/'
    data_spectrum = np.load(path+'Cl_cmb_dust_sync.npy')[0][1]
    # IPython.embed()
    angle_init = 0
    minimisation = minimize(chi2, angle_init, (spectra_cl, data_spectrum, nside, b))
    print(minimisation)

    ell = np.load(path+'effective_ells.npy')
    spectra_cl.spectra_rotation(minimisation.x * u.rad)
    spectra_cl.l_min_instru = 0
    spectra_cl.l_max_instru = 3*nside
    spectra_cl.get_noise()
    spectra_cl.get_instrument_spectra()
    model_fit = b.bin_cell(spectra_cl.instru_spectra.spectra[:3*nside].T).T

    plt.plot(ell, data_spectrum, label='CMB + noise + dust + sync')
    plt.plot(ell, model_fit.T[4],
             label=r'Model minimisation $\alpha$={}'.format(minimisation.x * u.rad))
    plt.legend()
    plt.xscale('log')
    plt.title('Spectra at '+str(freq_arg)+r'GHz, $\alpha$=' +
              str(angle_arg)+'deg')
    plt.xlabel(r'$\ell$', fontsize=22)
    plt.ylabel(r'$C_{\ell}^{EB}$', fontsize=22)
    plt.tight_layout()
    plt.savefig(path+'ModelfitVSobsCellEB_'+str(freq_arg) +
                'GHz_alpha'+str(angle_arg).replace('.', 'p')+'deg.png')
    plt.close()


if __name__ == "__main__":
    main()
