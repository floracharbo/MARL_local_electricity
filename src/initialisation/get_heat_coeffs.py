"""
author: Flora Charbonnier.

get_heat_coeffs.py computes the heating coefficients from
the ISO simple hourly model given the input data
"""

import numpy as np


def _get_building_characteristics(heat):
    """
     From heat inputs, obtain intermediate building characteristic.

    required to compute the heating coefficients

    input:
    heat:
        geometric, coefficients, u values, etc

    output:
    A:
        dictionary of areas needed for computing coefficients [m2]
        - A['m'] effective mass area
        - A['tot'] area of all surfaces facing the building zone
    H:
        dictionary of transmission heat transfer coefficientw [W/K]
    psi:
        dictionary of heat flow rates [W]
        IS02007 13790:2007(E) for definitions
    """
    A = {}  # dictionary of areas needed for computing coefficients [m2]
    H = {}  # dictionary of transmission heat transfer coefficients [W/K]
    psi = {}  # dictionary of heat flow rates [W]

    # Areas [m2]
    A['1d'] = heat['hd'] * heat['wd']  # of a single door
    A['f'] = heat['L2']  # conditioned floor area (m^2) (ORG. PARAM) also sg
    A['roof'] = A['f'] * heat['roof']  # roof
    # all surfaces facing the building zone 7.2.2.2
    A['tot'] = heat['Lambda_at'] * A['f']

    # compute coefficients
    kAm = heat['kAms'][heat['classbuild']]
    kCm = heat['kCms'][heat['classbuild']]

    # SAP 2012 sect 3.2. eq (3)
    Uwd_eff = 1 / (np.divide(1, heat['Uwd']) + 0.04)

    # building geometry
    # Area of a single window (m^2)
    A['1wd'] = heat['hwd'] * heat['wwd']
    # Total door area (North, East, South, West)
    A['d'] = sum(heat['nd']) * A['1d']
    # Max number of windows allowed for AopeAf_combined
    nwd_max = round((heat['AopeAf_combined'] * A['f'] - A['d']) / A['1wd'])

    # number windows north, east, south, west
    nwd = [round(heat['fracwd'][i] * nwd_max) for i in range(4)]
    # Total window area (North, East, South, West)
    A['wd'] = sum(nwd) * A['1wd']
    # Wall area (excluding windows and doors)
    A['w'] = 4 * heat['L'] * heat['h'] - A['wd'] - A['d']
    # Conditioned floor area (m^2) (ORG. PARAM) also sg
    A['f'] = heat['L'] ** 2
    # [m2] table 12 effective mass area
    A['m'] = kAm * A['f']
    # Conditioned air volume (m^3) (ORG. PARAM)
    Vr = A['f'] * heat['h']

    # ignore internal heat flow rate from lighting, hot/mains water
    # sewage, HVAC, processes, goods
    # heat capacity and transfer coefficients
    # ISO J/K table 12 12.3.1. internal heat capacity of the building zone
    Cm = kCm * A['f']
    # ISO G.11. internal heat flow rate from occupants
    psi['intOC'] = 1.5 * A['f']
    # ISO table G.10. internal heat flow rate from appliances
    psi['intA'] = 2 * A['f']
    # ISO from 10.2 heat flow rate from internal heat sources W,
    # derived by dividing Qsol(MJ)/0.036
    psi['int'] = psi['intOC'] + psi['intA']
    # ISO C.2
    psi['ia'] = 0.5 * psi['int']

    # Transmission heat transfer coefficient (W/K) =
    # fabric heat loss p 181 SAP 2012
    # ISO 2007 Htw = transfer for walls
    H['tw'] = A['w'] * heat['Uw'][heat['Uvalues']]
    # % (R1) Transmission heat transfer coefficient:
    # Windows and doors (W/K) H_tr_w in matlab
    H['twd'] = (A['wd'] + A['d']) * Uwd_eff[heat['Uvalues']]
    # transmission roof
    H['tr'] = A['roof'] * heat['Ur'][heat['Uvalues']]
    # fg1 = 1.45 # EN 12831 D.4.3 correction factor heat
    # losses through the ground
    # fg2 =  0.5 # temperature reduction factor taking into
    # account the difference between annual
    # mean external temperature and external design temperature
    # - arbitrarily do like matlab
    # GW = 1 # EN 12831 D.4.3 correction factor heat losses through the ground
    # ground
    H['tf'] = A['f'] * (1 - heat['kf_party']) * heat['Ug'][heat['Uvalues']]
    # ISO 2007 7.2.2.2. coupling conductance W 7.2.2. IS02007 13790:2007(E)
    H['is'] = heat['his'] * A['tot']
    # ISO 2007 from 12.2.2. [W/K] coupling conductance between nodes m and s
    H['ms'] = heat['hms'] * A['m']
    # 7.2.2. transmission heat transfer coefficient for opaque elements
    H['op'] = H['tw'] + H['tr'] + H['tf']
    # ISO 2007 W/k from 12.2.2
    H['em'] = 1 / (1 / H['op'] - 1 / H['ms'])

    # alternative to Hve based on BS12831
    # BS12831 hygiene minimum air flow rate fo a heated space # eq 7.2.1
    Vmin = heat['nmin'] * Vr
    # eq 7.2.2 infiltration through buioding envelope eq (17) m3/h
    Vinf = 2 * Vr * heat['n50'] * heat['e'] * heat['epsilon']
    # air flow rate of heated space (i) in cubic metres per second (m3/s)
    # without ventilation system eq 14 p 26 BS12831
    V = max(Vinf, Vmin)
    # equation 13, p 25 BS12831 design ventilation heat loss coefficient
    H['ve'] = 0.34 * V

    # rearrange equations
    H[1] = 1 / (1 / H['ve'] + 1 / H['is'])  # ISO 2007
    H[2] = H[1] + H['twd']  # ISO 2007
    H[3] = 1 / (1 / H[2] + 1 / H['ms'])  # ISO 2007

    return A, H, psi, Cm


def _get_required_temperatures(heat, syst):
    day_T_req = np.ones((syst['n_homes'], syst['H'])) * heat['Ts']
    if len(np.shape(heat['hrs_c'][0])) == 1:
        # if hours comfort only specified once -> same for all
        heat['hrs_c'] = [heat['hrs_c'] for _ in range(syst['n_homes'])]
    for home in range(syst['n_homes']):
        for interval in heat['hrs_c'][home]:
            day_T_req[home][
                interval[0] * syst['n_int_per_hr']: interval[1] * syst['n_int_per_hr']
            ] = [heat['Tc'] for _ in range((interval[1] - interval[0]) * syst['n_int_per_hr'])]

    heat['T_req'] = day_T_req
    for day in range(syst['D'] - 1):
        heat['T_req'] = np.concatenate((heat['T_req'], day_T_req), axis=1)

    heat['T_req'] = np.concatenate((heat['T_req'], day_T_req[:, 0:2]), axis=1)

    heat['T_UB'] = heat['T_req'] + heat['dT']
    for home in range(syst['n_homes']):
        # allow for heating one hour before when specified
        # temperature increases
        for t in range(syst['N']):
            for dt in range(1, 6):
                if t < syst['N'] - 1 - dt \
                        and heat['T_UB'][home][t + dt] > heat['T_UB'][home][t]:
                    heat['T_UB'][home][t] = heat['T_UB'][home][t + dt]

    heat['T_LB'] = heat['T_req'] - heat['dT']

    for e in ['T_req', 'T_LB', 'T_UB']:
        heat[e + 'P'] = [heat[e][0] for _ in range(syst['n_homesP'])]

    return heat


def get_heat_coeffs(heat, syst, paths):
    """
    Compute heating coefficients from ISO simple hourly model.

    inputs
    heat:
        the heating input data (e.g. building fabric and geometry, etc)
    ntw:
        the network input data
        (here, the number of homes 'n_homes' and 'n_homesP' (passive) is relevant)
    syst:
        the system input data
        (here, the number of time steps 'N' is relevant)

    output
     heat:
        updated heating parameters including the coefficients
        to describe the heating behaviour of the buildings
    """
    # boolean for whether comfort temperature is required
    heat = _get_required_temperatures(heat, syst)

    heat['T_out_all'] = np.load(paths['open_inputs'] / paths['temp_file'])

    tau = 60 * 60 * 24 / syst['N']  # time step in seconds
    A, H, psi, Cm = _get_building_characteristics(heat)

    # me rearranging
    a = Cm / tau + 1 / 2 * (H[3] + H['em'])
    b = 1 - A['m'] / A['tot'] - H['twd'] / (9.1 * A['tot'])
    c = b * psi['int'] / 2
    d = A['m'] * psi['int'] / (2 * A['tot']) \
        + H[3] / H[2] * (c + (H[1] * psi['ia'] / H['ve']))
    e = H['em'] + H[3] / H[2] * (H['twd'] + H[1])

    a_t = d / a
    b_t = (Cm / tau - 1 / 2 * (H[3] + H['em'])) / a
    c_t = e / a
    d_t = (H[3] / H[2] * b + A['m'] / A['tot']) / a * heat['COP']
    e_t = H[3] * H[1] / (H[2] * H['ve'] * a)

    f = H['ms'] + H['twd'] + H[1]
    g = 1 / f * (H['ms'] * a_t / 2 + c + H[1] / H['ve'] * psi['ia'])
    h = H['ms'] / (2 * f) * (1 + b_t)
    i = 1 / f * (H['ms'] * c_t / 2 + H['twd'] + H[1])
    j = 1 / f * (H['ms'] * d_t / 2 + b)
    k = 1 / f * (H['ms'] * e_t / 2 + H[1] / H['ve'])

    a_t_air = (H['is'] * g + psi['ia']) / (H['is'] + H['ve'])
    b_t_air = (H['is'] * h) / (H['is'] + H['ve'])
    c_t_air = (H['is'] * i + H['ve']) / (H['is'] + H['ve'])
    d_t_air = (H['is'] * j) / (H['is'] + H['ve']) * heat['COP']
    e_t_air = (1 + H['is'] * k) / (H['is'] + H['ve'])

    t_coeff_0 = np.reshape([a_t, b_t, c_t, d_t, e_t], (1, 5))
    t_air_coeff_0 = np.reshape([a_t_air, b_t_air, c_t_air, d_t_air, e_t_air], (1, 5))
    for passive_ext in ["", "P"]:
        for label, value in zip(['T_coeff', 'T_air_coeff'], [t_coeff_0, t_air_coeff_0]):
            heat[label + passive_ext] = np.repeat(
                value, repeats=syst["n_homes" + passive_ext], axis=0
            )

    return heat
