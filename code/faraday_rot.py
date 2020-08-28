import IPython
import bjlib.lib_project as lib
import bjlib.plot_project as plotpro
import math
from astropy import constants as c
from astropy import units as u
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from bjlib.class_faraday import power_spectra_operation as power_spectra


# import scipy

# from fgbuster.observation_helpers import get_instrument, get_sky
# import pysm

def get_spectra_diff(spectra1, spectra2):

    if np.shape(spectra1)[1] != np.shape(spectra1)[1] == 6:
        print('WARNING: in spectra_diff() different number of spectra between \
               1 & 2')

    spectra1T = spectra1.T
    spectra2T = spectra2.T

    spect_diff = np.array([spectra1T[0] - spectra2T[0]])
    spect_diff = np.append(spect_diff, [spectra1T[1] - spectra2T[1]], axis=0)
    spect_diff = np.append(spect_diff, [spectra1T[2] - spectra2T[2]], axis=0)
    spect_diff = np.append(spect_diff, [spectra1T[3] - spectra2T[3]], axis=0)
    if np.shape(spectra1)[1] == 6 & np.shape(spectra2)[1] == 6:
        spect_diff = np.append(spect_diff, [spectra1T[4] - spectra2T[4]],
                               axis=0)
        spect_diff = np.append(spect_diff, [spectra1T[5] - spectra2T[5]],
                               axis=0)

    spect_diff = spect_diff.T
    return spect_diff


def get_map_camb(nside, l_max=5000, raw_cl=True, lens_potential=False,
                 ratio=0.07, spectra_name='total', cls=None):
    if cls.all(None):
        print('bla')
        pars, results, powers = lib.get_basics(
            l_max=5000, raw_cl=True, lens_potential=False, ratio=ratio)
    # TODO: get ell from nside
        print('shape powers=', np.shape(powers[spectra_name]))
        map = hp.synfast(powers[spectra_name].T, nside, new=True)

    else:
        print('blou')
        map = hp.synfast(cls.T, nside, new=True)

    return map


def ellmax2nside(ellmax):
    try:
        n = 2**math.ceil(math.log2((ellmax+1)/3))
    except ValueError:
        print('\nIn ellmax2nside() wrong ellmax e.g. negativ log\n\
NSIDE SET TO 512 BY DEFAULT\n')
        n = 512
    return n


def nside2ellmax(nside):  # TODO: finish, see healpix doc
    return nside*3 - 1


# def map_rotation(map, rotation_angle):
#     """Rotates the Q,U maps given a certain angle. Handles multiple frequecies
#     and frequency dependent rotation angle.
#     Parameters
#     ----------
#     map : numpy array, [frequency, I/Q/U, pixels] OR [I/Q/U, pixels]
#         sky map typically given by fgbuster/pysm. Contains temperature and
#         polarisation maps. Can contain frequency maps also (optionnal)
#     rotation_angle : float or numpy array
#         Angle by which to rotate the map. Might be an array with same length as
#         number of frequencies if frequency dependent rotation.
#
#     Returns
#     -------
#     numpy array
#         new map array of same shape as argument but with now mixed Q & U
#         component according to the rotation.
#
#     """
#     len_map = len(np.shape(map))
#     frequency_dependent = isinstance(rotation_angle, np.ndarray)
#
#     if 1-frequency_dependent & len_map == 3:  # TODO: why ??
#
#         rotation_angle = np.ones(len(map[:, 0, 0])) * rotation_angle
#     if len_map == 3:
#
#         map_rotated = np.empty(np.shape(map))
#
#         for i in range(len(map[:, 0, 0])):
#             Qrot = np.cos(2*rotation_angle[i])*map[i, 1, :] + \
#                 np.sin(2 * rotation_angle[i])*map[i, 2, :]
#             Urot = - np.sin(2*rotation_angle[i])*map[i, 1, :] + \
#                 np.cos(2 * rotation_angle[i])*map[i, 2, :]
#             map_rotated[i, 0] = map[i, 0]
#             map_rotated[i, 1] = Qrot
#             map_rotated[i, 2] = Urot
#
#     else:
#         print('WARNING: in map_rotation() only one frequency map given')
#         map_rotated = np.empty((3))
#         Qrot = np.cos(2*rotation_angle)*map[1, :] - \
#             np.sin(2 * rotation_angle)*map[2, :]
#         Urot = np.sin(2*rotation_angle)*map[1, :] + \
#             np.cos(2 * rotation_angle)*map[2, :]
#         map_rotated[0] = map[i, 0]
#         map_rotated[1] = Qrot
#         map_rotated[2] = Urot
#
#     return map_rotated

def map_rotation(map, rotation_angle):
    """Rotates the Q,U maps given a certain angle. Handles multiple frequecies
    and frequency dependent rotation angle.
    Parameters
    ----------
    map : numpy array, [frequency, I/Q/U, pixels] OR [I/Q/U, pixels]
        sky map typically given by fgbuster/pysm. Contains temperature and
        polarisation maps. Can contain frequency maps also (optionnal)
    rotation_angle : float or numpy array
        Angle by which to rotate the map. Might be an array with same length as
        number of frequencies if frequency dependent rotation.

    Returns
    -------
    numpy array
        new map array of same shape as argument but with now mixed Q & U
        component according to the rotation.

    """
    len_map = len(np.shape(map))
    frequency_dependent = isinstance(rotation_angle, np.ndarray)

    if 1-frequency_dependent and len_map == 3:  # TODO: why ?
        print('Check map rotation !!!')
        rotation_angle = np.ones(len(map[:, 0, 0])) * rotation_angle
    if len_map == 3:

        map_rotated = np.empty(np.shape(map))
        print('HELLO')
        for i in range(len(map[:, 0, 0])):
            Qrot = np.cos(2*rotation_angle[i])*map[i, 1, :] - \
                np.sin(2 * rotation_angle[i])*map[i, 2, :]
            Urot = np.sin(2*rotation_angle[i])*map[i, 1, :] + \
                np.cos(2 * rotation_angle[i])*map[i, 2, :]
            map_rotated[i, 0] = map[i, 0]
            map_rotated[i, 1] = Qrot
            map_rotated[i, 2] = Urot

    else:
        print('OLA')
        map_rotated = np.empty(np.shape(map))

        Qrot = np.cos(2*rotation_angle)*map[1, :] - \
            np.sin(2 * rotation_angle)*map[2, :]
        Urot = np.sin(2*rotation_angle)*map[1, :] + \
            np.cos(2 * rotation_angle)*map[2, :]
        for i in range(len(map[0][:])):
            map_rotated[0, i] = map[0, i]
        map_rotated[1] = Qrot
        map_rotated[2] = Urot

    return map_rotated


def plot_faraday_angle(frequency, min_field=10**-13 * u.T, max_field=1 * u.T,
                       step=10**4):
    grid = np.linspace(min_field, max_field, step)
    # in SI units
    fd_cst = (c.e.si**3) / (8 * (np.pi**2) * c.eps0 * (c.m_e**2) * (c.c**3))

    e_density_cmb = 10**8 / u.m**3
    traveled_length_at_recom = (120 * 10**3 * u.year).to(u.s) * \
        c.c  # length of photon path in the B field
    fd_fct = (fd_cst * e_density_cmb * traveled_length_at_recom *
              (c.c / frequency)**2).decompose()

    # * u.rad #TODO check rad or degree
    # fd_angle = fd_angle.decompose()

    plt.plot(grid, fd_fct * grid % (2*np.pi))
    plt.title('faraday angle at {}'.format(frequency))

    return 0


# nside = 1024
r = 0.07
nu = np.array([40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9,
               140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1]) * u.GHz
# nu = np.array([78.0, 88.5, 100.0, 118.9, 140.0, 166.0, 195.0, 234.9]) * u.GHz
ellmax_asked = 300
nside = ellmax2nside(ellmax_asked)
l_max = nside2ellmax(nside)
# nsidetest = ellmax2nside(2000)
print('nside=', nside)
print('ellmax', l_max)


"""------------------------faraday angle computation------------------------"""
Bfield_strength = 45 * 10**-13 * u.T  # = 100 nG #10**-13 * u.T #*nu * nu
# in SI units
fd_cst = (c.e.si**3) / (8 * (np.pi**2) * c.eps0 * (c.m_e**2) * (c.c**3))
# Bfield_strength = 10**7 * u.T #10**-13 * u.T #*nu * nu
# e_density_cmb = 10**12 / u.m**3 # in number / m^3
e_density_cmb = 10**8 / u.m**3
traveled_length_at_recom = (100 * 10**3 * u.year).to(u.s) * \
    c.c  # length of photon path in the B field
# 100 k années de recombination

fd_fct = fd_cst * Bfield_strength * e_density_cmb * traveled_length_at_recom
fd_angle = fd_fct * (c.c / nu)**2  # * u.rad #TODO check rad or degree
# print('fd_angle',fd_angle.decompose())
fd_angle = fd_angle.decompose()
# u.add_enabled_equivalencies(u.dimensionless_angles())
# makes radian dimensionless to allow remainder operation
fd_angle = np.array(fd_angle * u.rad)
fd_angle_reduced = np.array((fd_angle % (2*np.pi)) * u.rad)
print('fd_angle = ', fd_angle, '\n')
print('fd_angle_reduced = ', fd_angle_reduced, '\n')

min_angle = fd_angle_reduced.min()
max_angle = fd_angle_reduced.max()
print('min angle = ', min_angle * u.rad, ' max angle = ', max_angle*u.rad)

faraday_factor = 1 * Bfield_strength / nu / nu * u.deg

plot_faraday_angle(40.0*u.GHz, min_field=0 * u.T, max_field=10**-11 * u.T,
                   step=10**6)
plt.xlabel('B field in Tesla')
plt.ylabel('faraday rotation angle')
plt.show()

# exit()
# for nu in [1, 2, 3]:
#     print(nu)
"""----------------------------camb map creation----------------------------"""
# l_max = rotated_spectra.T.shape[1]
raw_cl = True
# nside = ellmax2nside(1000)
# ellmax_test = nside2ellmax(nside)
pars, results, powers = lib.get_basics(l_max, raw_cl, ratio=r)
# WARNING:lmax-1?
total_power_camb = lib.get_normalised_cl(powers['total'])  # [:l_max+1]

sky_camb = get_map_camb(nside, cls=powers['total'][:l_max+1])
print('sky camb shape:', np.shape(sky_camb))

"""-------------------------camb faraday rotation--------------------------"""

map_camb = np.array([sky_camb/len(nu) for i in range(len(nu))])
print('shape map_camb', np.shape(map_camb))
camb_rotated = np.sum(map_rotation(map_camb, fd_angle), 0)
camb_reduced_rot = np.sum(map_rotation(map_camb, fd_angle_reduced), 0)


# sky_camb_spectrum_1 = hp.anafast(sky_camb, iter=100)
# sky_camb_spectrum = lib.get_normalised_cl(sky_camb_spectrum_1.T)

# camb_rot_spectrum_un = hp.anafast(camb_rotated, iter=100)
# camb_rot_spectrum = lib.get_normalised_cl(camb_rot_spectrum_un.T)

camb_redu_spec_un = hp.anafast(camb_reduced_rot, iter=100)
camb_redu_spec = lib.get_normalised_cl(camb_redu_spec_un.T)

"""---------------------------birefringence spectra-------------------------"""
print('min_angle=', min_angle)

print('np.array([min_angle,max_angle]=',
      np.array([min_angle, max_angle])*u.rad)

minmax_angles = np.array([min_angle, max_angle])*u.rad

birefringence_spectra = lib.get_spectra_dict(
    total_power_camb, minmax_angles, include_unchanged=False)


"""------------------------------sky simulation-----------------------------"""

# sky = pysm.Sky(get_sky(nside, 'c1'))
# conversion_coeffs = pysm.convert_units('K_RJ','K_CMB',nu)
# cmb = np.array([sky.cmb(nu)[i,:,:]*
# conversion_coeffs[i] for i in range(len(nu))])
# print('////shape cmb////',np.shape(cmb))


"""-----------------------polarization angle rotation-----------------------"""

# cmb_rotated = map_rotation(cmb,fd_angle)
# print('shape cmb_rotated =',np.shape(cmb_rotated))

# rotation_conversion_ = map_rotation(sky.cmb(nu),fd_angle)
# rotation_conversion = np.array([rotation_conversion_[i,:,:]*
#                       conversion_coeffs[i] for i in range(len(nu))])
#
# print('sum diff T=',sum(sum(cmb_rotated[:,0,:] -rotation_conversion[:,0,:])))
# print('sum diff Q=',sum(sum(cmb_rotated[:,1,:] -rotation_conversion[:,1,:])))
# print('sum diff U=',sum(sum(cmb_rotated[:,2,:] -rotation_conversion[:,2,:])))


"""------------------spectra computation and normalisation------------------"""

# cmb_rotated_map = np.sum(cmb_rotated,0) / len(nu)
# cmb_sum = np.sum(cmb,0) / len(nu)


# cmb_spectrum_all_freq_rotated = hp.anafast( cmb_rotated_map , iter = 100)
# rotated_spectra = lib.get_normalised_cl(cmb_spectrum_all_freq_rotated.T)
# print('shape rotated=',np.shape(rotated_spectra))

# cmb_spectrum_all_freq = hp.anafast( cmb_sum , iter = 100)
# cmb_all_freq_normal = lib.get_normalised_cl(cmb_spectrum_all_freq.T)


# """------------------------camb faraday rotation--------------------------"""
# map_camb = np.array([sky_camb/len(nu) for i in range(len(nu)) ])
# print('shape map_camb',np.shape(map_camb))
# camb_rotated = np.sum(map_rotation(map_camb,fd_angle),0)
# camb_rot_spectrum_un = hp.anafast(camb_rotated, iter=100)
# camb_rot_spectrum = lib.get_normalised_cl(camb_rot_spectrum_un.T)
#
#
# sky_camb_spectrum_1 = hp.anafast(sky_camb, iter = 100)
# sky_camb_spectrum = lib.get_normalised_cl(sky_camb_spectrum_1.T)

dict_plot = {'camb reduced rot': camb_redu_spec, 'camb': total_power_camb}
# 'camb_sky': sky_camb_spectrum,
# 'camb rotated': camb_rot_spectrum
dict_plot.update(birefringence_spectra)

test = power_spectra(r=r)
test.get_spectra()
test.get_frequencies()
test.get_faraday_angles()
# test.split_spectra()
test.get_faraday_spectra()
dict_plot['faraday_class'] = lib.get_normalised_cl(test.cl_faraday)[:l_max+1]

IPython.embed()

lw_list = [1.5, 3, 1.5, 1.5]
plotpro.spectra(dict_plot, lw_list=lw_list)
plt.show()
# plt.savefig('faraday_comp_{}_passiv.png'.format(nside))


# spect_diff = get_spectra_diff(camb_rot_spectrum, camb_redu_spec)


print('TT error rad =', np.sum(spect_diff[:, 0]))
print('EE error rad =', np.sum(spect_diff[:, 1]))
print('BB error rad =', np.sum(spect_diff[:, 2]))
print('TE error rad =', np.sum(spect_diff[:, 3]))
print('EB error rad =', np.sum(spect_diff[:, 4]))
print('TB error rad =', np.sum(spect_diff[:, 5]))


# plotpro.spectra( {'camb diff':spect_diff} )
# plt.savefig('faraday_comp_{}_raddiff_passiv.png'.format(nside))
# plt.show()
print('END')
