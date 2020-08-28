# import numdifftools as nd
# from scipy.linalg import logm
import IPython
# import sys
# sys.path.insert(1, '/home/baptiste/Documents/stage_npac')
import bjlib.lib_project as lib
# import plot_project as plotpro
# import math
from fgbuster import visualization as visu

# from astropy import constants as c
from astropy import units as u
import matplotlib.pyplot as plt
# import healpy as hp
import numpy as np
# import V3calc as V3
# import copy
import time
import bjlib.class_faraday as cf


def main():

    r_data = 0.0
    rotation = 0.00 * u.rad
    noise_model = '00'

    """====================FISHER COMPUTATION================================="""

    test = cf.power_spectra_operation(r=r_data, rotation_angle=rotation, l_max=700)
    # Bfield_strength=45 * 10**-13 * u.T)
    test.get_spectra()
    test.spectra_rotation()
    test.get_noise(noise_model)
    test.get_instrument_spectra()
    noisy_spectra = test.instru_spectra.spectra
    # noisy_spectra = test.cl_rot.spectra

    if np.shape(noisy_spectra)[1] == 4:
        # TODO: use diagonal instead of hardcoded ell
        print('olaa')
        cov_matrix = cf.power_spectra_obj(np.array(
            [[noisy_spectra[:, 1], np.zeros(270)],
             [np.zeros(270), noisy_spectra[:, 2]]]).T, test.instru_spectra.ell)
    else:
        print('halloo')
        cov_matrix = cf.power_spectra_obj(np.array(
            [[noisy_spectra[:, 1], noisy_spectra[:, 4]],
             [noisy_spectra[:, 4], noisy_spectra[:, 2]]]).T, test.instru_spectra.ell)
    deriv1 = cf.power_spectra_obj(lib.cl_rotation_derivative(
        test.spectra.spectra, rotation), test.spectra.ell)
    deriv_matrix1 = cf.power_spectra_obj(np.array(
        [[deriv1.spectra[:, 1], deriv1.spectra[:, 4]],
         [deriv1.spectra[:, 4], deriv1.spectra[:, 2]]]).T, deriv1.ell)

    pw_r1 = cf.power_spectra_operation(r=1, rotation_angle=rotation,
                                       l_max=700, powers_name='unlensed_total')
    pw_r1.get_spectra()

    deriv_matrix2 = cf.power_spectra_obj(lib.get_dr_cov_bir_EB(
        pw_r1.spectra.spectra, rotation).T, pw_r1.spectra.ell)

    fishaa = cf.fisher_pws(cov_matrix, deriv_matrix1, 0.1)
    fishrr = cf.fisher_pws(cov_matrix, deriv_matrix2, 0.1)
    fishar = cf.fisher_pws(cov_matrix, deriv_matrix1, 0.1, deriv2=deriv_matrix2)
    fishra = cf.fisher_pws(cov_matrix, deriv_matrix2, 0.1, deriv2=deriv_matrix1)
    # IPython.embed()
    # cov_matrixBB = power_spectra_obj(cov_matrix.spectra[:, 1, 1], cov_matrix.ell)
    # deriv_matrixBB = power_spectra_obj(deriv_matrix2.spectra[:, 1, 1], deriv_matrix2.ell)
    # fishrrBB = fisher_pws(cov_matrixBB, deriv_matrixBB, 0.1)

    lensed_scalar = cf.power_spectra_operation(r=1, l_max=700, rotation_angle=rotation,
                                               powers_name='lensed_scalar')
    lensed_scalar.get_spectra()
    data2 = cf.power_spectra_operation(r=r_data, l_max=700, rotation_angle=rotation)
    data2.get_spectra()
    data2.spectra.spectra[:, 2] = r_data*pw_r1.spectra.spectra[:, 2] + \
        lensed_scalar.spectra.spectra[:, 2]
    data2.get_noise(noise_model)
    data2.spectra_rotation()
    data2.get_instrument_spectra()
    cov_matrixBB = cf.power_spectra_obj(
        data2.instru_spectra.spectra[:, 2], data2.instru_spectra.ell)
    deriv_matrixBB = cf.power_spectra_obj(deriv_matrix2.spectra[:, 1, 1], deriv_matrix2.ell)
    fishrrBB = cf.fisher_pws(cov_matrixBB, deriv_matrixBB, 0.1)

    fisher_matrix = np.array([[fishaa, fishar], [fishar, fishrr]])
    sigma_sq_matrix = np.linalg.inv(fisher_matrix)

    # IPython.embed()

    visu.corner_norm([rotation.value, r_data], sigma_sq_matrix, labels=[r'$\alpha$', r'$r$'])
    plt.show()

    IPython.embed()

    """========================= LIKELIHOOD ON ALPHA ========================="""

    data = cf.power_spectra_operation(r=r_data, l_max=700, rotation_angle=rotation)
    data.get_spectra()
    data.get_noise(noise_model)
    data.spectra_rotation()
    data.get_instrument_spectra()
    data_spectra = data.instru_spectra.spectra
    data_matrix = cf.power_spectra_obj(
        np.array([[data_spectra[:, 1], data_spectra[:, 4]],
                  [data_spectra[:, 4], data_spectra[:, 2]]]).T,
        data.instru_spectra.ell)

    model = cf.power_spectra_operation(r=r_data, l_max=700, rotation_angle=0.0*u.rad)
    model.get_spectra()
    model.get_noise(noise_model)
    model.spectra_rotation()
    model.get_instrument_spectra()

    min_angle = rotation.value - 5*(1/np.sqrt(fishaa))
    max_angle = rotation.value + 5*(1/np.sqrt(fishaa))
    nstep_angle = 100
    angle_grid = np.arange(min_angle, max_angle,
                           (max_angle - min_angle)/nstep_angle)*u.radian
    idx1 = (np.abs(angle_grid.value - rotation.value)).argmin()
    print('angle_grid check ', angle_grid[idx1])
    likelihood_values = []

    for angle in angle_grid:
        model.spectra_rotation(angle)
        model.get_instrument_spectra()
        model_spectra = model.instru_spectra.spectra

        model_matrix = cf.power_spectra_obj(np.array(
            [[model_spectra[:, 1], model_spectra[:, 4]],
             [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)

        likelihood_val = cf.likelihood_pws(model_matrix, data_matrix, 0.1)
        likelihood_values.append(likelihood_val)

    """======================= PLOT LIKELIHOOD ON ALPHA ======================="""

    likelihood_norm = np.array(likelihood_values) - min(likelihood_values)
    plt.plot(angle_grid, np.exp(-likelihood_norm),
             label='Likelihood -min(likelihood)')

    fisher_gaussian = np.exp(-(angle_grid.value - rotation.value)**2 / (2*sigma_sq_matrix[0, 0]))
    # fisher_gaussian_norm = fisher_gaussian - min(fisher_gaussian)
    plt.plot(angle_grid, fisher_gaussian, label='fisher gaussian')

    plt.xlabel(r'$\alpha$ in radian')
    plt.ylabel('Likelihood')
    plt.title(r'Likelihood on $\alpha$ with true $\alpha$={}'.format(rotation))
    plt.legend()
    plt.show()

    print('gradient fisher = ', np.gradient(np.gradient(fisher_gaussian))[idx1])
    print('gradient likelihood = ', np.gradient(np.gradient(np.exp(-likelihood_norm)))[idx1])

    """================= GRIDDING LIKELIHOOD ON ALPHA AND R ================="""

    likelihood_values_ar = []
    likelihood_values_ar_2D = []

    model = cf.power_spectra_operation(r=0, l_max=700, rotation_angle=rotation)
    model.get_spectra()
    model.get_noise(noise_model)
    model.spectra_rotation()
    model.get_instrument_spectra()
    min_r = r_data-5*(1/np.sqrt(fishrr))
    max_r = r_data+5*(1/np.sqrt(fishrr))
    nstep_r = 100
    r_grid = np.arange(min_r, max_r, (max_r - min_r)/nstep_r)

    idx2 = (np.abs(r_grid - r_data)).argmin()
    print('r_grid check', r_grid[idx2])

    # angle_grid = np.array([(0.33*u.deg).to(u.rad).value])*u.rad
    # nstep_angle = 1
    # idx1 = 0
    # lensed_scalar = power_spectra_operation(r=1, l_max=700, rotation_angle=rotation,
    #                                         powers_name='lensed_scalar')
    # lensed_scalar.get_spectra()

    # lensed_scalar = lensed_scalar_['lensed_scalar'][]
    # data2 = power_spectra_operation(r=r_data, l_max=700, rotation_angle=rotation)
    # data2.get_spectra()
    # data2.spectra.spectra[:, 2] = r_data*pw_r1.spectra.spectra[:, 2] + \
    #     lensed_scalar.spectra.spectra[:, 2]
    #
    # data2.get_noise(noise_model)
    # data2.spectra_rotation()
    # data2.get_instrument_spectra()
    # data_spectra = data2.instru_spectra.spectra
    # data_matrix = power_spectra_obj(np.array(
    #     [[data_spectra[:, 2]]]).T, model.instru_spectra.ell)
    # data_matrix = power_spectra_obj(
    #     np.array([r_data*pw_r1.spectra.spectra[:, 2] + lensed_scalar.spectra.spectra[:, 2]
    #               ]).T,
    #     pw_r1.spectra.ell)

    # model_r07 = power_spectra_operation(r=0.07, l_max=700, rotation_angle=rotation)
    # model_r07.get_spectra()
    # model_r07.get_noise(noise_model)
    # model_r07.spectra_rotation()
    # model_r07.get_instrument_spectra()

    start = time.time()
    model = cf.power_spectra_operation(l_max=700, rotation_angle=rotation)
    model.get_spectra(r=r_data)
    model.get_noise(noise_model)
    for r in r_grid:
        model.get_spectra(r=r)

        # modelBB = r * pw_r1.spectra.spectra[:, 2] + \
        #     lensed_scalar.spectra.spectra[:, 2]
        # model.spectra.spectra[:, 2] = modelBB

        # likelihood_values_ar_2D.append([])
        for angle in angle_grid:
            # model_r07.spectra_rotation(angle)
            # model_r07.get_instrument_spectra()

            model.spectra_rotation(angle)
            model.get_instrument_spectra()
            # model.spectra.spectra[:, 1] = model_r07.spectra.spectra[:, 1]

            model_spectra = model.instru_spectra.spectra
            model_matrix = cf.power_spectra_obj(np.array(
                [[model_spectra[:, 1], model_spectra[:, 4]],
                 [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)
            # model_matrix = power_spectra_obj(np.array(
            #     [[model_r07.instru_spectra.spectra[:, 1], model_spectra[:, 4]],
            #      [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)

            # model_matrix = power_spectra_obj(np.array(
            #     [[model_spectra[:, 2]]]).T, model.instru_spectra.ell)

            likelihood_value = cf.likelihood_pws(model_matrix, data_matrix, 0.1)
            likelihood_values_ar.append(likelihood_value)
            # likelihood_values_ar_2D[-1].append(likelihood_value)

    print('time gridding r =', time.time() - start)

    angle_array, r_array = np.meshgrid(angle_grid, r_grid, indexing='xy')
    likelihood_mesh = np.reshape(likelihood_values_ar, (-1, nstep_angle))
    # plt.plot(r_grid, likelihood_mesh)

    """==================== PLOT LIKELIHOOD ON R AND ALPHA ===================="""

    # HEEERE FOR LIKELIHOOD
    likelihood_norm = likelihood_mesh[:, idx1] - min(likelihood_mesh[:, idx1])
    plt.plot(r_grid, np.exp(-likelihood_norm), label='likelihood-min(likelihood)')

    fisher_gaussian_r = np.exp(-((r_grid - r_data)**2) / (2*sigma_sq_matrix[1, 1]))
    # fisher_gaussian_norm_r = fisher_gaussian_r - min(fisher_gaussian_r)
    plt.plot(r_grid, fisher_gaussian_r, label='fisher gaussian')

    # fisher_gaussian_rbb = np.exp(-((r_grid - r_data)**2) / (2/fishrrBB))
    # fisher_gaussian_norm_r = fisher_gaussian_r - min(fisher_gaussian_r)
    # plt.plot(r_grid, fisher_gaussian_rbb, label='fisher gaussian BB')

    plt.xlabel('r')
    plt.ylabel('likelihood')
    plt.legend()
    plt.title('likelihood on r with true r={}'.format(r_data))
    plt.show()
    print('gradient fisher = ', np.gradient(np.gradient(fisher_gaussian_r))[idx2])
    print('gradient likelihood = ', np.gradient(np.gradient(np.exp(-likelihood_norm)))[idx2])
    IPython.embed()
    # plt.contour(r_array, angle_array, likelihood_mesh)

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    mu = np.array([rotation.value, r_data])
    pos = np.empty(angle_array.shape + (2,))
    pos[:, :, 0] = angle_array
    pos[:, :, 1] = r_array

    fisher_2D_gaussian = multivariate_gaussian(pos, mu, sigma_sq_matrix)

    # plt.contourf(angle_array, r_array, np.exp(-(likelihood_mesh-np.min(likelihood_mesh))))
    # plt.colorbar()
    # plt.contour(angle_array, r_array, fisher_2D_gaussian, colors='r', linestyles='--')
    #
    # plt.ylabel(r'$r$')
    # plt.xlabel(r'miscalibration angle $\alpha$ in radian')
    # plt.title(r'Joint likelihood for estimation of miscalibration angle $\alpha$ and $r$. True value are $r=${}, $\alpha={}$'.format(
    #     r_data, rotation))
    # # IPython.embed()
    # plt.show()

    # plt.pcolormesh(angle_array.value, r_array,
    #                np.exp(-(likelihood_mesh-np.min(likelihood_mesh))), vmin=0, vmax=1)
    #
    # plt.colorbar()
    # plt.clim(0,1)
    # plt.contour(angle_array, r_array, fisher_2D_gaussian, colors='r',linestyles = '--')

    # plt.ylabel(r'$r$')
    # plt.xlabel(r'miscalibration angle $\alpha$ in radian')
    # plt.title(r'Joint likelihood for estimation of miscalibration angle $\alpha$ and $r$. True value are $r=${}, $\alpha={}$'.format(
    #     r_data, rotation))
    # # IPython.embed()
    # plt.show()
    plt.rc('font', size=22)
    fig, ax = plt.subplots()
    levels = np.arange(0, 1+1/8, 1/8)
    cs = ax.contourf(angle_array, r_array,
                     np.exp(-(likelihood_mesh-np.min(likelihood_mesh))), levels=levels)
    cs2 = ax.contour(angle_array, r_array, fisher_2D_gaussian /
                     np.max(fisher_2D_gaussian), levels=cs.levels, colors='r', linestyles='--')
    cbar = fig.colorbar(cs)
    cbar.add_lines(cs2)
    cbar.ax.set_xlabel(r'$\mathcal{L}$')
    plt.ylabel(r'$r$')
    plt.xlabel(r'miscalibration angle $\alpha$ in radian')
    plt.title(r'Joint likelihood on $r$ and $\alpha$ with $r_{input} =$'+'{},'.format(r_data)+r' $\alpha_{input}=$'+'{}'.format(
        rotation))
    h1, _ = cs2.legend_elements()
    ax.legend([h1[0]], ["Fisher prediction"])

    plt.show()

    # def likelihood_for_hessian_a(angle, model, data_matrix):
    #     angle = angle * u.rad
    #     # r = param_array[1]
    #
    #     # model.get_spectra(r=r)
    #     model.spectra_rotation(angle)
    #     model.get_instrument_spectra()
    #     model_spectra = model.instru_spectra.spectra
    #     model_matrix = power_spectra_obj(np.array(
    #         [[model_spectra[:, 1], model_spectra[:, 4]],
    #          [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)
    #     likelihood_value = likelihood_pws(
    #         model_matrix, data_matrix, 0.1)
    #     return likelihood_value

    def likelihood_for_hessian_r(r, model, data_matrix):
        # angle = angle * u.rad

        model.get_spectra(r=r)
        model.spectra_rotation()
        model.get_instrument_spectra()
        model_spectra = model.instru_spectra.spectra
        model_matrix = cf.power_spectra_obj(np.array(
            [[model_spectra[:, 1], model_spectra[:, 4]],
             [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)
        likelihood_value = cf.likelihood_pws(
            model_matrix, data_matrix, 0.1)
        return likelihood_value

    def likelihood_for_hessian(param_array, model, data_matrix):
        angle = param_array[0] * u.rad
        r = param_array[1]

        model.get_spectra(r=r)
        model.spectra_rotation(angle)
        model.get_instrument_spectra()
        model_spectra = model.instru_spectra.spectra
        model_matrix = cf.power_spectra_obj(np.array(
            [[model_spectra[:, 1], model_spectra[:, 4]],
             [model_spectra[:, 4], model_spectra[:, 2]]]).T, model.instru_spectra.ell)
        likelihood_value = cf.likelihood_pws(
            model_matrix, data_matrix, 0.1)
        return likelihood_value

    IPython.embed()

    """=================================purgatory=============================="""
    # test.get_frequencies()
    # test.get_faraday_angles()
    # # test.split_spectra()
    # test.get_faraday_spectra()
    # freq1 = test.faraday_angles
    # cl_farad_norm1 = lib.get_normalised_cl(test.cl_faraday)
    # dico = {'faraday': cl_farad_norm1}
    #
    # test.spectra_rotation(min(test.faraday_angles)*u.rad)
    # test.cl_rot.normalisation = 1
    # cl_min = test.cl_rot.spectra
    #
    # test.spectra_rotation(max(test.faraday_angles)*u.rad)
    # test.cl_rot.normalisation = 1
    # cl_max = test.cl_rot.spectra
    #
    # dico['min'] = cl_min
    # dico['max'] = cl_max
    #
    #
    # dico['normal'] = lib.get_normalised_cl(test.spectra.spectra)

    # hessian = nd.Hessian(likelihood_for_hessian)
    # h = hessian([rotation, r_data], model, data_matrix)[0, 0]
    exit()


if __name__ == "__main__":
    main()
