import bjlib.likelihood_SO as l_SO
import bjlib.class_faraday as cf
import numpy as np
import copy
from astropy import units as u
from mpi4py import MPI
import bjlib.cl_lib as cl_lib
import argparse


def main():
    comm = MPI.COMM_WORLD
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    nsim = comm.Get_size()
    print(mpi_rank, nsim)

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, help='birefringence angle in DEGREE',
                        default=0)
    parser.add_argument('--Frequency', type=int, help='Frequency in GHz',
                        default=93)

    args = parser.parse_args()

    angle_arg = args.alpha
    freq_arg = args.Frequency
    save_path = '/global/homes/j/jost/these/spectra_based_analysis/results_and_data/noCMB_' +\
                str(freq_arg)+'GHz_'+str(nsim)+'sim_alpha' +\
                str(angle_arg).replace('.', 'p')+'deg/'

    rotation_angle = (angle_arg * u.deg).to(u.rad)

    # no_inh = False
    # ny_lf = 1
    nside = 512
    r = 0.0
    root = 0
    lmin = 30
    lmax = 500
    nlb = 10
    pysm_model = 'c1s0d0'
    f_sky_SAT = 0.1
    custom_bins = True
    # aposize = 8.0
    # apotype = 'C1'
    # purify_b = True

    # sensitivity = 0
    # knee_mode = 0

    # BBPipe_path = '/global/homes/j/jost/BBPipe'
    # norm_hits_map_path = BBPipe_path + '/test_mapbased_param/norm_nHits_SA_35FOV_G_nside512.fits'

    sky_map = l_SO.sky_map(nside=nside, sky_model=pysm_model)
    sky_map.get_pysm_sky()
    sky_map.get_frequency()
    # frequencies2use = [93]
    frequencies2use = [freq_arg]

    frequencies_index = []
    for f in frequencies2use:
        frequencies_index.append(sky_map.frequencies.tolist().index(f))

    # nhits, noise_maps_, nlev = mknm.get_noise_sim(
        # sensitivity=sensitivity, knee_mode=knee_mode,
        # ny_lf=ny_lf, nside_out=nside,
        # norm_hits_map=hp.read_map(norm_hits_map_path), no_inh=no_inh)
    # del noise_maps_, nlev
    # mask_ = hp.read_map(BBPipe_path +
    # "/test_mapbased_param/norm_nHits_SA_35FOV_G_nside512_binary.fits")
    # mask = hp.ud_grade(mask_, nside)
    # mask[np.where(nhits < 1e-6)[0]] = 0.0

    '''***********************NAMASTER INITIALISATION***********************'''

    print('initializing Namaster ...')
    # wsp = nmt.NmtWorkspace()
    b = cl_lib.binning_definition(nside, lmin=lmin, lmax=lmax,
                                  nlb=nlb, custom_bins=custom_bins)

    # del mask_
    # mask_apo = nmt.mask_apodization(
    # mask, aposize, apotype=apotype)
    # nh = hp.smoothing(nhits, fwhm=1*np.pi/180.0, verbose=False)
    # nh /= nh.max()
    # mask_apo *= nh

    # cltt, clee, clbb, clte = hp.read_cl(BBPipe_path +
    # "/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits")[:, :4000]
    # mp_t_sim, mp_q_sim, mp_u_sim = hp.synfast(
    # [cltt, clee, clbb, clte], nside=nside, new=True, verbose=False)
    # f2y0 = cl_lib.get_field(mp_q_sim, mp_u_sim, mask_apo)
    # wsp.compute_coupling_matrix(f2y0, f2y0, b)

    """________________________test nsim likelihood________________________"""

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

    # data_model = spectra_cl.instru_spectra.spectra
    # data_model = b.bin_cell(copy.deepcopy(spectra_cl.instru_spectra.spectra[:3*nside]).T).T
    # data_model_nobin = copy.deepcopy(spectra_cl.instru_spectra.spectra[:3*nside].T)
    # np.save(save_path + 'data_model_noise.npy', data_model)
    # np.save(save_path + 'data_model_nobin_noise.npy', data_model_nobin)

    Cl_cmb_nsim = None
    Cl_cmb_dust_nsim = None
    Cl_cmb_sync_nsim = None
    Cl_cmb_dust_sync_nsim = None
    if comm.rank == 0:
        Cl_cmb_nsim = np.load(save_path + 'Cl_cmb_nsim.npy')
        Cl_cmb_dust_nsim = np.load(save_path + 'Cl_cmb_dust_nsim.npy')
        Cl_cmb_sync_nsim = np.load(save_path + 'Cl_cmb_sync_nsim.npy')
        Cl_cmb_dust_sync_nsim = np.load(save_path + 'Cl_cmb_dust_sync_nsim.npy')
        # remove EE and BB :

        Cl_cmb_nsim[0, 0] = np.zeros(b.get_effective_ells().shape[0])
        Cl_cmb_nsim[0, -1] = np.zeros(b.get_effective_ells().shape[0])

        Cl_cmb_dust_nsim[0, 0] = np.zeros(b.get_effective_ells().shape[0])
        Cl_cmb_dust_nsim[0, -1] = np.zeros(b.get_effective_ells().shape[0])

        Cl_cmb_sync_nsim[0, 0] = np.zeros(b.get_effective_ells().shape[0])
        Cl_cmb_sync_nsim[0, -1] = np.zeros(b.get_effective_ells().shape[0])

        Cl_cmb_dust_sync_nsim[0, 0] = np.zeros(b.get_effective_ells().shape[0])
        Cl_cmb_dust_sync_nsim[0, -1] = np.zeros(b.get_effective_ells().shape[0])

    Cl_cmb = np.empty([1, 4, b.get_effective_ells().shape[0]])
    Cl_cmb_dust = np.empty([1, 4, b.get_effective_ells().shape[0]])
    Cl_cmb_sync = np.empty([1, 4, b.get_effective_ells().shape[0]])
    Cl_cmb_dust_sync = np.empty([1, 4, b.get_effective_ells().shape[0]])

    comm.Scatter(Cl_cmb_nsim, Cl_cmb, root)
    comm.Scatter(Cl_cmb_dust_nsim, Cl_cmb_dust, root)
    comm.Scatter(Cl_cmb_sync_nsim, Cl_cmb_sync, root)
    comm.Scatter(Cl_cmb_dust_sync_nsim, Cl_cmb_dust_sync, root)

    if comm.rank == 0:
        del Cl_cmb_nsim
        del Cl_cmb_dust_nsim
        del Cl_cmb_sync_nsim
        del Cl_cmb_dust_sync_nsim

    fit_alpha_cmb, H_cmb = cl_lib.min_and_error_nsim_freq_mpi(
        spectra_cl, Cl_cmb, nsim, frequencies_index, b, nside, f_sky_SAT,
        spectra_used='all', spectra_indexation='NaMaster',
        minimisation_init=0.001,
        compute_error68=False, step_size=1e-5, output_grid=False)
    fit_alpha_cmb_dust, H_cmb_dust = cl_lib.min_and_error_nsim_freq_mpi(
        spectra_cl, Cl_cmb_dust, nsim, frequencies_index, b, nside, f_sky_SAT,
        spectra_used='all', spectra_indexation='NaMaster',
        minimisation_init=0.001,
        compute_error68=False, step_size=1e-5, output_grid=False)
    fit_alpha_cmb_sync, H_cmb_sync = cl_lib.min_and_error_nsim_freq_mpi(
        spectra_cl, Cl_cmb_sync, nsim, frequencies_index, b, nside, f_sky_SAT,
        spectra_used='all', spectra_indexation='NaMaster',
        minimisation_init=0.001,
        compute_error68=False, step_size=1e-5, output_grid=False)
    fit_alpha_cmb_dust_sync, H_cmb_dust_sync = cl_lib.min_and_error_nsim_freq_mpi(
        spectra_cl, Cl_cmb_dust_sync, nsim, frequencies_index, b, nside, f_sky_SAT,
        spectra_used='all', spectra_indexation='NaMaster',
        minimisation_init=0.001,
        compute_error68=False, step_size=1e-5, output_grid=False)

    fit_alpha_cmb_nsim = None
    H_cmb_nsim = None
    fit_alpha_cmb_dust_nsim = None
    H_cmb_dust_nsim = None
    fit_alpha_cmb_sync_nsim = None
    H_cmb_sync_nsim = None
    fit_alpha_cmb_dust_sync_nsim = None
    H_cmb_dust_sync_nsim = None

    if comm.rank == 0:
        fit_alpha_cmb_nsim = np.empty([nsim, fit_alpha_cmb.shape[0]])
        H_cmb_nsim = np.empty([nsim, H_cmb.shape[0]])

        fit_alpha_cmb_dust_nsim = np.empty([nsim, fit_alpha_cmb_dust.shape[0]])
        H_cmb_dust_nsim = np.empty([nsim, H_cmb_dust.shape[0]])

        fit_alpha_cmb_sync_nsim = np.empty([nsim, fit_alpha_cmb_sync.shape[0]])
        H_cmb_sync_nsim = np.empty([nsim, H_cmb_sync.shape[0]])

        fit_alpha_cmb_dust_sync_nsim = np.empty([nsim, fit_alpha_cmb_dust_sync.shape[0]])
        H_cmb_dust_sync_nsim = np.empty([nsim, H_cmb_dust_sync.shape[0]])

    comm.Gather(fit_alpha_cmb, fit_alpha_cmb_nsim, root)
    comm.Gather(H_cmb, H_cmb_nsim, root)

    comm.Gather(fit_alpha_cmb_dust, fit_alpha_cmb_dust_nsim, root)
    comm.Gather(H_cmb_dust, H_cmb_dust_nsim, root)

    comm.Gather(fit_alpha_cmb_sync, fit_alpha_cmb_sync_nsim, root)
    comm.Gather(H_cmb_sync, H_cmb_sync_nsim, root)

    comm.Gather(fit_alpha_cmb_dust_sync, fit_alpha_cmb_dust_sync_nsim, root)
    comm.Gather(H_cmb_dust_sync, H_cmb_dust_sync_nsim, root)

    if comm.rank == 0:

        np.save(save_path + 'fit_alpha_cmb_EB.npy', fit_alpha_cmb_nsim)
        np.save(save_path + 'H_cmb_EB.npy', H_cmb_nsim)

        np.save(save_path + 'fit_alpha_cmb_dust_EB.npy', fit_alpha_cmb_dust_nsim)
        np.save(save_path + 'H_cmb_dust_EB.npy', H_cmb_dust_nsim)

        np.save(save_path + 'fit_alpha_cmb_sync_EB.npy', fit_alpha_cmb_sync_nsim)
        np.save(save_path + 'H_cmb_sync_EB.npy', H_cmb_sync_nsim)

        np.save(save_path + 'fit_alpha_cmb_dust_sync_EB.npy', fit_alpha_cmb_dust_sync_nsim)
        np.save(save_path + 'H_cmb_dust_sync_EB.npy', H_cmb_dust_sync_nsim)
    exit()


if __name__ == "__main__":
    main()
