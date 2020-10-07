# import bjlib.likelihood_SO as l_SO
import bjlib.class_faraday as cf
import numpy as np
# import copy
from astropy import units as u
# from mpi4py import MPI
import bjlib.cl_lib as cl_lib
import argparse
import matplotlib.pyplot as plt
import IPython


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, help='birefringence angle in DEGREE',
                        default=0)
    parser.add_argument('--Frequency', type=int, help='Frequency in GHz',
                        default=93)

    args = parser.parse_args()

    angle_arg = args.alpha
    freq_arg = args.Frequency
    # angle_arg = 0.0
    # freq_arg = 280
    nsim = 192
    rotation_angle = (angle_arg * u.deg).to(u.rad)
    nside = 512
    r = 0.0
    lmin = 30
    lmax = 500
    nlb = 10
    f_sky_SAT = 0.1

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

    path_noCMB = '/global/homes/j/jost/these/spectra_based_analysis/results_and_data/noCMB_' +\
        str(freq_arg)+'GHz_16sim_alpha' +\
        str(angle_arg).replace('.', 'p')+'deg/'

    data_cl = np.load(path+'Cl_cmb_dust_sync.npy')
    data_cl_cmb_nsim = np.load(path+'Cl_cmb_dust_sync_nsim.npy')
    data_cl_nsim = np.load(path+'Cl_cmb_nsim.npy')

    data_cl_cmb_nsim_mean = np.mean(data_cl_cmb_nsim, axis=0)
    data_cl_nsim_mean = np.mean(data_cl_nsim, axis=0)

    f = 0
    data_matrix = cf.power_spectra_obj(np.array(
        [[data_cl[f, 0], data_cl[f, 1]],
         [data_cl[f, 1], data_cl[f, 3]]]).T,
        b.get_effective_ells())

    data_matrix_EB = cf.power_spectra_obj(np.array(
        data_cl[f, 1]).T,
        b.get_effective_ells())

    data_matrix_cmb_mean = cf.power_spectra_obj(np.array(
        [[data_cl_cmb_nsim_mean[f, 0], data_cl_cmb_nsim_mean[f, 1]],
         [data_cl_cmb_nsim_mean[f, 1], data_cl_cmb_nsim_mean[f, 3]]]).T,
        b.get_effective_ells())

    data_matrix_cmb_mean_EB = cf.power_spectra_obj(np.array(
        data_cl_cmb_nsim_mean[f, 1]).T,
        b.get_effective_ells())

    data_matrix_mean = cf.power_spectra_obj(np.array(
        [[data_cl_nsim_mean[f, 0], data_cl_nsim_mean[f, 1]],
         [data_cl_nsim_mean[f, 1], data_cl_nsim_mean[f, 3]]]).T,
        b.get_effective_ells())

    data_matrix_mean_EB = cf.power_spectra_obj(np.array(
        data_cl_nsim_mean[f, 1]).T,
        b.get_effective_ells())

    total_fit = np.load(path+'fit_alpha_cmb_dust_sync.npy')
    no_CMB_dust = np.load(path_noCMB+'fit_alpha_cmb_dust.npy')
    no_CMB_sync = np.load(path_noCMB+'fit_alpha_cmb_sync.npy')
    fit_CMB = np.load(path_noCMB+'fit_alpha_cmb.npy')

    angle_totalfit = np.mean(total_fit)
    angle_noCMB_sync = np.mean(no_CMB_sync)
    angle_noCMB_dust = np.mean(no_CMB_dust)
    angle_CMB = np.mean(fit_CMB)
    std_totalfit = np.std(total_fit)
    std_noCMB_sync = np.std(no_CMB_sync)
    std_noCMB_dust = np.std(no_CMB_dust)
    std_CMB = np.std(fit_CMB)

    print('total fit = ', angle_totalfit, '+-', std_totalfit)
    print('no CMB sync = ', angle_noCMB_sync, '+-', std_noCMB_sync)
    print('no CMB dust = ', angle_noCMB_dust, '+-', std_noCMB_dust)
    l_input_angle = cf.likelihood_for_hessian_a(rotation_angle.value, spectra_cl,
                                                data_matrix, b, nside, f_sky_SAT,
                                                spectra_used='all')
    l_total_fit = cf.likelihood_for_hessian_a(angle_totalfit, spectra_cl,
                                              data_matrix, b, nside, f_sky_SAT,
                                              spectra_used='all')
    l_noCMB_sync = cf.likelihood_for_hessian_a(angle_noCMB_sync, spectra_cl,
                                               data_matrix, b, nside, f_sky_SAT,
                                               spectra_used='all')
    l_noCMB_dust = cf.likelihood_for_hessian_a(angle_noCMB_dust, spectra_cl,
                                               data_matrix, b, nside, f_sky_SAT,
                                               spectra_used='all')

    print('l_input =', l_input_angle)
    print('l_total_fit =', l_total_fit)
    print('l_noCMB_sync =', l_noCMB_sync)
    print('l_noCMB_dust =', l_noCMB_dust)

    if l_noCMB_dust < l_total_fit:
        print('likelihood for angle_dust is smaller than angle_total')

    if l_noCMB_sync < l_total_fit:
        print('likelihood for angle_sync is smaller than angle_total')

    if l_noCMB_dust < l_input_angle:
        print('likelihood for angle_dust is smaller than angle_input')

    if l_noCMB_sync < l_input_angle:
        print('likelihood for angle_sync is smaller than angle_input')

    if l_total_fit < l_input_angle:
        print('likelihood for angle_total is smaller than angle_input')

    min_angle = rotation_angle.value - 4*std_totalfit
    max_angle = rotation_angle.value + 2*std_totalfit
    steps = 500
    alpha_grid = np.arange(min_angle, max_angle, (max_angle - min_angle)/steps)
    cl_sync = np.load(path_noCMB + 'Cl_cmb_sync.npy')
    data_matrix_sync = cf.power_spectra_obj(np.array(
        [[cl_sync[f, 0], cl_sync[f, 1]],
         [cl_sync[f, 1], cl_sync[f, 3]]]).T,
        b.get_effective_ells())

    data_matrix_sync_EB = cf.power_spectra_obj(np.array(
        cl_sync[f, 1]).T,
        b.get_effective_ells())

    sync_grid, sync_logL = cl_lib.L_gridding(spectra_cl, data_matrix_sync_EB, alpha_grid,
                                             b, nside, f_sky_SAT, spectra_used='EB',
                                             return_m2logL=True)
    # IPython.embed()

    cl_dust = np.load(path_noCMB + 'Cl_cmb_dust.npy')
    data_matrix_dust = cf.power_spectra_obj(np.array(
        [[cl_dust[f, 0], cl_dust[f, 1]],
         [cl_dust[f, 1], cl_dust[f, 3]]]).T,
        b.get_effective_ells())

    data_matrix_dust_EB = cf.power_spectra_obj(np.array(
        cl_dust[f, 1]).T,
        b.get_effective_ells())

    dust_grid, dust_logL = cl_lib.L_gridding(spectra_cl, data_matrix_dust_EB, alpha_grid,
                                             b, nside, f_sky_SAT, spectra_used='EB',
                                             return_m2logL=True)

    cmb_grid, cmb_logL = cl_lib.L_gridding(spectra_cl, data_matrix_EB, alpha_grid,
                                           b, nside, f_sky_SAT, spectra_used='EB',
                                           return_m2logL=True)

    cmb_mean_grid, cmb_mean_logL = cl_lib.L_gridding(spectra_cl, data_matrix_cmb_mean_EB, alpha_grid,
                                                     b, nside, f_sky_SAT, spectra_used='EB',
                                                     return_m2logL=True)

    all_mean_grid, all_mean_logL = cl_lib.L_gridding(spectra_cl, data_matrix_mean_EB, alpha_grid,
                                                     b, nside, f_sky_SAT, spectra_used='EB',
                                                     return_m2logL=True)
    # plt.close()
    # plt.plot(alpha_grid, sync_logL, label='-2logL synchrotron only in data')
    # plt.plot(alpha_grid, dust_logL, label='-2logL dust only in data')
    # plt.plot(alpha_grid, cmb_logL, label='-2logL cmb+noise+fg in data')
    # plt.xlabel('angle in rad')
    # plt.title('-2logL at {} GHz'.format(freq_arg))
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(path_noCMB+'m2logL_grid_all_{}GHz_alpha{}deg_.png'.format(freq_arg, angle_arg))
    #
    # plt.close()
    # plt.plot(alpha_grid, dust_logL, label='-2logL dust only in data')
    # plt.xlabel('angle in rad')
    # plt.title('-2logL at {} GHz'.format(freq_arg))
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(path_noCMB+'m2logL_grid_dust_{}GHz_alpha{}deg_.png'.format(freq_arg, angle_arg))
    #
    # plt.close()
    # plt.plot(alpha_grid, sync_logL, label='-2logL synchrotron only in data')
    # plt.xlabel('angle in rad')
    # plt.title('-2logL at {} GHz'.format(freq_arg))
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(path_noCMB+'m2logL_grid_sync_{}GHz_alpha{}deg_.png'.format(freq_arg, angle_arg))
    # # IPython.embed()
    #
    # plt.close()
    # plt.plot(alpha_grid, cmb_logL, label='-2logL cmb+noise+fg in data')
    # plt.xlabel('angle in rad')
    # plt.title('-2logL at {} GHz'.format(freq_arg))
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(path_noCMB+'m2logL_grid_cmb_{}GHz_alpha{}deg_.png'.format(freq_arg, angle_arg))
    # IPython.embed()
    # exit()
    plt.close()
    plt.plot(alpha_grid, cmb_mean_grid, label='likelihood CMB+noise only in data')
    # plt.errorbar(angle_CMB, 1, xerr=std_CMB, capsize=8,
    #              label='mean CMB+noise fit with std over {}sim'.format(nsim), fmt='o')

    plt.plot(alpha_grid, all_mean_grid, label='likelihood CMB+noise+fg in data')
    # plt.errorbar(angle_totalfit, 1, xerr=std_totalfit, capsize=8,
    #              label='mean fit total with std over {}sim'.format(nsim), fmt='o')
    # plt.vlines(angle_noCMB_dust, 0, 1, label='mean fit dust only')

    # plt.xlabel('angle in rad')
    # plt.title('Likelihood at {} GHz'.format(freq_arg))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(
    #     path_noCMB+'likelihood_grid_CMBvsCMBpFG_{}GHz_alpha{}deg_zoomerror.png'.format(freq_arg, angle_arg))
    #
    # plt.close()
    #
    # exit()

    plt.plot(alpha_grid, sync_grid, label='likelihood synchrotron only in data')
    # plt.errorbar(angle_noCMB_sync, 1, xerr=std_noCMB_sync, capsize=8,
    #              label='mean fit synchrotron only with std over 16sim', fmt='o')

    plt.plot(alpha_grid, dust_grid, label='likelihood dust only in data')
    # plt.errorbar(angle_noCMB_dust, 1, xerr=std_noCMB_dust, capsize=8,
    #              label='mean fit dust only with std over 16sim', fmt='o')

    plt.plot(alpha_grid, cmb_grid, label='likelihood cmb+fg in data')
    # plt.errorbar(angle_totalfit, 1, xerr=std_totalfit, capsize=8,
    #              label='mean fit total with std over {}sim'.format(nsim), fmt='o')
    # plt.vlines(angle_noCMB_dust, 0, 1, label='mean fit dust only')

    plt.xlabel('angle in rad')
    plt.title('Likelihood at {} GHz'.format(freq_arg))
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        path_noCMB+'likelihood_grid_all_EB_{}GHz_alpha{}deg_zoomerror.png'.format(freq_arg, angle_arg))

    print('std grid =', np.std(cmb_grid))
    print('std cmb fit =', std_totalfit)
    print('std grid - std cmb fit =', np.std(cmb_grid) - std_totalfit)


if __name__ == "__main__":
    main()
