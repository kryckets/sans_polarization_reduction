import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import os
from scipy import ndimage

from .reduction_functions import _sans_get_by_filenumber, _sans_sample_base_name_descrip

def _he3_pol_at_given_time(entry_time, HE3_Cell_Summary):
    """Compute the 3He cell polarization and transmissions at a given time.

    Uses the decay parameters of the most recently inserted cell relative to
    ``entry_time`` to evaluate the atomic polarization, the resulting neutron
    polarization, the unpolarized 3He transmission, and the spin-state
    transmissions T_MAJ and T_MIN.

    Parameters
    ----------
    entry_time : float
        Required. Time (hours) at which to evaluate the cell, on the same
        clock as the keys of ``HE3_Cell_Summary``.
    HE3_Cell_Summary : dict
        Required. Mapping ``insert_time -> {'Atomic_P0', 'Gamma(hours)',
        'Mu', 'Te', ...}`` describing each 3He cell load.

    Returns
    -------
    NeutronPol : float
        Neutron polarization ``tanh(Mu * AtomicPol)``.
    UnpolHE3Trans : float
        Transmission of unpolarized neutrons through the cell.
    T_MAJ : float
        Transmission of the majority (aligned) spin state.
    T_MIN : float
        Transmission of the minority (anti-aligned) spin state.
    """
    counter = 0
    for time in HE3_Cell_Summary:
        if counter == 0:
            holder_time = time
            counter += 1
        if entry_time >= time:
            holder_time = time
        if entry_time < time:
            break
    delta_time = entry_time - holder_time     
    P0 = HE3_Cell_Summary[holder_time]['Atomic_P0']
    gamma = HE3_Cell_Summary[holder_time]['Gamma(hours)']
    Mu = HE3_Cell_Summary[holder_time]['Mu']
    Te = HE3_Cell_Summary[holder_time]['Te']
    AtomicPol = P0 * np.exp(-delta_time / gamma)
    NeutronPol = np.tanh(Mu * AtomicPol)
    UnpolHE3Trans = Te * np.exp(-Mu)*np.cosh(Mu * AtomicPol)
    T_MAJ = Te * np.exp(-Mu*(1.0 - AtomicPol))
    T_MIN = Te * np.exp(-Mu*(1.0 + AtomicPol))
        
    return NeutronPol, UnpolHE3Trans, T_MAJ, T_MIN

def _sans_record_data_processing(save_path, Plex_Name = 'not used', Scatt = {}, BlockBeam = {}, Trans = {}, Pol_Trans = {}, HE3_Cell_Summary = {}, YesNoManualHe3Entry = False, Contents = 'not used'):
    """Write a human-readable summary of the data reduction inputs.

    Produces ``DataReductionSummary.txt`` inside ``save_path`` recording the
    sample/config catalog, transmission files, block-beam files, polarized-
    transmission results, and 3He cell parameters used by the reduction.

    Parameters
    ----------
    save_path : str
        Required. Output directory in which ``DataReductionSummary.txt``
        will be written.
    Plex_Name : str, optional
        Name (or path) of the plex/efficiency file (default ``'not used'``).
    Scatt : dict, optional
        Scattering catalog keyed by sample name (default ``{}``).
    BlockBeam : dict, optional
        Block-beam catalog keyed by configuration (default ``{}``).
    Trans : dict, optional
        Transmission catalog keyed by sample name (default ``{}``).
    Pol_Trans : dict, optional
        Polarized-transmission catalog keyed by sample name (default ``{}``).
    HE3_Cell_Summary : dict, optional
        3He cell summary keyed by insertion time (default ``{}``).
    YesNoManualHe3Entry : bool or int, optional
        If truthy, additionally write the manually supplied ``New_HE3_Files``,
        ``MuValues``, and ``TeValues`` (default ``False``).
    Contents : str, optional
        Verbatim user-input block to embed at the top of the summary
        (default ``'not used'``).

    Returns
    -------
    None
    """

    file_path = os.path.join(save_path, "DataReductionSummary.txt")
    file1 = open(file_path,"w+")
    file1.write("Record of Data Reduction \n")
    file1.write("****************************************** \n")
    file1.write("****************************************** \n")
    file1.write("User-defined Inputs: \n")
    file1.write('\n')
    file1.write(Contents)
    file1.write("****************************************** \n")
    file1.write("****************************************** \n")
    file1.write('\n')

    if YesNoManualHe3Entry >= 1:
        file1.write("New_HE3_Files = ")
        for x in New_HE3_Files:
            file1.write(str(x) + ' ')
        file1.write('\n')
        file1.write("MuValues = ")
        for x in MuValues:
            file1.write(str(x) + ' ')
        file1.write('\n')
        file1.write("TeValues = ")
        for x in TeValues:
            file1.write(str(x) + ' ')
        file1.write('\n')
    file1.write('\n')
    file1.write('Plex file is ' + str(Plex_Name) + '\n')
    file1.write('\n')
    file1.write('Detector shadowing is automatically corrected for. \n')
        
    for Sample in Scatt:
        file1.write(str(Sample) +  '(' +  str(Scatt[Sample]['Intent']) + ') \n')
        for Config in Scatt[Sample]['Config(s)']:
            file1.write(' Config:' + str(Config) + '\n')
            if Config in BlockBeam:
                str1 = str(BlockBeam[Config]['Scatt']['File'])
                str2 = str(BlockBeam[Config]['Trans']['File'])
                str3 = '  Block Beam: '
                file1.write(str3)
                if str(BlockBeam[Config]['Scatt']['File']).find('NA') == -1 and str(BlockBeam[Config]['Trans']['File']).find('NA') == -1:
                    file1.write(str1)
                    file1.write(' (Scatt) and (Trans) ')
                    file1.write(str2)
                    file1.write('\n')
                elif str(BlockBeam[Config]['Scatt']['File']).find('NA') == -1 and str(BlockBeam[Config]['Trans']['File']).find('NA') != -1:
                    file1.write(str1)
                    file1.write('\n')
                elif str(BlockBeam[Config]['Scatt']['File']).find('NA') != -1 and str(BlockBeam[Config]['Trans']['File']).find('NA') == -1:
                    file1.write(str2)
                    file1.write('\n')
            else:
                str4 = '      ' + 'Block Beam Scatt, Trans files are not available \n'
                file1.write(str4)
            TransUnpol = str(Trans[Sample]['Config(s)'][Config]['Unpol_Files'])
            if TransUnpol.find('N') != -1:
                TransUnpol = 'NA'
            TransPol = str(Trans[Sample]['Config(s)'][Config]['U_Files'])
            if TransPol.find('N') != -1:
                TransPol = 'NA'
            #file1.write('  Unpol, Pol scaling trans: ' + TransUnpol + ' , ' + TransPol + '\n')
            file1.write('  Unpol scaling trans: ' + TransUnpol + '\n')
            file1.write('  Pol scaling trans: ' + TransPol + '\n')
            file1.write('  Unpolarized Scatt ' + str(Scatt[Sample]['Config(s)'][Config]['Unpol']) + '\n')
            file1.write('  Up Scatt ' + str(Scatt[Sample]['Config(s)'][Config]['U']) + '\n')
            file1.write('  Down Scatt ' + str(Scatt[Sample]['Config(s)'][Config]['D']) + '\n')
            file1.write('  Up-Up Scatt ' + str(Scatt[Sample]['Config(s)'][Config]['UU']) + '\n')
            file1.write('  Up-Down Scatt ' + str(Scatt[Sample]['Config(s)'][Config]['UD']) + '\n')
            file1.write('  Down-Down Scatt ' + str(Scatt[Sample]['Config(s)'][Config]['DD']) + '\n')
            file1.write('  Down-Up Scatt '+ str(Scatt[Sample]['Config(s)'][Config]['DU']) + '\n')
        if Sample in Pol_Trans:
            if 'P_SM' in Pol_Trans[Sample] and str(Pol_Trans[Sample]['P_SM']).find('NA') == -1:
                file1.write(' Full Polarization Results: \n')
                pol_num = int(Pol_Trans[Sample]['P_SM']*10000)/10000
                file1.write(' P_SM  x Depol: ' + str(pol_num) + '\n')
                file1.write(' UU Trans ' + str(Pol_Trans[Sample]['T_UU']['File']) + '\n')
                file1.write(' DU Trans ' + str(Pol_Trans[Sample]['T_DU']['File']) + '\n')
                file1.write(' DD Trans ' + str(Pol_Trans[Sample]['T_DD']['File']) + '\n')
                file1.write(' UD Trans ' + str(Pol_Trans[Sample]['T_UD']['File']) + '\n')
                file1.write(' SM Trans ' + str(Pol_Trans[Sample]['T_SM']['File']) + '\n')
        file1.write(' \n')

    for entry in HE3_Cell_Summary:
        file1.write('3He Cell: ' + str(HE3_Cell_Summary[entry]['Name']) + '\n')
        file1.write('Lifetime (hours): ' + str(HE3_Cell_Summary[entry]['Gamma(hours)']) + ' +/- ' + str(HE3_Cell_Summary[entry]['Gamma_Unc']) + '\n')
        file1.write('Atomic P_0: ' + str(HE3_Cell_Summary[entry]['Atomic_P0']) + ' +/- ' + str(HE3_Cell_Summary[entry]['Atomic_P0_Unc']) + '\n')
        file1.write('Neutron P_0: ' + str(HE3_Cell_Summary[entry]['Neutron_P0']) + ' +/- ' + str(HE3_Cell_Summary[entry]['Neutron_P0_Unc']) + '\n')
        file1.write('\n')
    file1.close()

    return

def _he3_evaluation(He3Only_Check, HE3_TransCatalog):
    """Print a per-cell listing of 3He transmission measurements.

    Useful for the helium team to inspect transmission files without
    running a full reduction. Does nothing unless ``He3Only_Check`` is true.

    Parameters
    ----------
    He3Only_Check : bool
        Required. If true, print the catalog; otherwise return immediately.
    HE3_TransCatalog : dict
        Required. Per-cell mapping with keys ``'Cell_name'``,
        ``'HE3_OUT_file'``, ``'HE3_IN_file'``, ``'Transmission'``,
        ``'Elasped_time'``, and ``'Config'``, each holding a list of values.

    Returns
    -------
    None
    """

    if He3Only_Check:
        for entry in HE3_TransCatalog:
            num = len(HE3_TransCatalog[entry]['HE3_IN_file'])
            print('List of transmission files per cell:')
            print("Cell_Name, He_Out, He_In, Transmission, Elapsed_Time (hours), Config")
            for holder in range(0,num):
                print(HE3_TransCatalog[entry]['Cell_name'][holder], HE3_TransCatalog[entry]['HE3_OUT_file'][holder],
                      HE3_TransCatalog[entry]['HE3_IN_file'][holder], HE3_TransCatalog[entry]['Transmission'][holder],
                      HE3_TransCatalog[entry]['Elasped_time'][holder], HE3_TransCatalog[entry]['Config'][holder])
    return

def _solid_angle_all_detectors(Detector_Panels, Instrument, input_path, representative_filenumber, Config):
    """Compute the per-pixel solid angle for every relevant detector panel.

    Reads the geometry of ``representative_filenumber`` and returns the
    product of the angular pixel sizes for each panel; the high-resolution
    back detector ``'B'`` is included when ``Config`` contains ``'CvB'``.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names (e.g. ``['FR','FL','MR','ML',...]``).
    Instrument : str
        Required. Instrument identifier; must contain ``'VSANS'`` or
        ``'NG7SANS'``.
    input_path : str
        Required. Directory containing the raw NeXus files.
    representative_filenumber : int
        Required. Run number used to read panel geometry.
    Config : str
        Required. Configuration label; if it contains ``'CvB'`` the back
        detector is included.

    Returns
    -------
    Solid_Angle : dict[str, float]
        Per-pixel solid angle (steradians) keyed by detector short name.
    """

    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')
    
    Solid_Angle = {}
    f = _sans_get_by_filenumber(Instrument, input_path, representative_filenumber)
    if f is not None:
        for dshort in relevant_detectors:
            if 'VSANS' in Instrument:
                detector_distance = f['entry/instrument/detector_{ds}/distance'.format(ds=dshort)][0]
                x_pixel_size = f['entry/instrument/detector_{ds}/x_pixel_size'.format(ds=dshort)][0]/10.0
                y_pixel_size = f['entry/instrument/detector_{ds}/y_pixel_size'.format(ds=dshort)][0]/10.0
            elif 'NG7SANS' in Instrument:
                detector_distance = f['entry/instrument/detector/distance'][0]
                #detector_distance = int(f['entry/DAS_logs/detectorPosition/desiredSoftPosition'][0]) #in cm
                x_pixel_size = f['entry/instrument/detector/x_pixel_size'][0]/10.0
                y_pixel_size = f['entry/instrument/detector/y_pixel_size'][0]/10.0
            if dshort == 'MT' or dshort == 'MB' or dshort == 'FT' or dshort == 'FB':
                setback = f['entry/instrument/detector_{ds}/setback'.format(ds=dshort)][0]
            else:
                setback = 0
            realDistZ = detector_distance + setback
            theta_x_step = x_pixel_size / realDistZ
            theta_y_step = y_pixel_size / realDistZ
            Solid_Angle[dshort] = theta_x_step * theta_y_step
        f.close()

    return Solid_Angle

def _all_sans_blocked_beam_counts_per_second_list_of_files(Detector_Panels, Instrument, input_path, filelist, Config, examplefilenumber):
    """Sum a list of block-beam files and return per-pixel counts/second.

    Accumulates counts and live time across ``filelist``, then divides to
    obtain a counts-per-second map and Poisson uncertainty for each panel.
    If ``filelist`` is empty or contains no usable files, arrays of zeros
    matching ``examplefilenumber`` are returned.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names to process.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    input_path : str
        Required. Directory containing raw NeXus files.
    filelist : Sequence[int]
        Required. Block-beam run numbers to sum (may be empty).
    Config : str
        Required. Configuration label; ``'CvB'`` triggers inclusion of the
        back detector.
    examplefilenumber : int
        Required. Fallback run number used to size the zero arrays when no
        block-beam files are usable.

    Returns
    -------
    BB_CountsPerSecond : dict[str, np.ndarray]
        2-D counts-per-second array keyed by panel.
    BB_Unc : dict[str, np.ndarray]
        Matching per-pixel uncertainty (sqrt(counts)/seconds).
    """

    BB_Counts = {}
    BB_Unc = {}
    BB_Seconds = {}
    BB_CountsPerSecond = {}
    
    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')

    item_counter = 0
    for item in filelist:
        filename = input_path + "sans" + str(item) + ".nxs.ngv"
        f = _sans_get_by_filenumber(Instrument, input_path, item)
        if f is not None:            
            Count_time = f['entry/collection_time'][0]
            if Count_time > 0:
                for dshort in relevant_detectors:
                    if 'VSANS' in Instrument:
                        bb_data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                        unc = np.sqrt(np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)]))
                    elif 'NG7SANS' in Instrument:
                        bb_data = np.array(f['entry/instrument/detector/data'])
                        unc = np.sqrt(np.array(f['entry/instrument/detector/data']))
                
                    if item_counter < 1:
                        BB_Counts[dshort] = bb_data
                        BB_Seconds[dshort] = Count_time
                    else:
                        BB_Counts[dshort] = BB_Counts[dshort] + bb_data
                        BB_Seconds[dshort] = BB_Seconds[dshort] + Count_time
                    BB_CountsPerSecond[dshort] = BB_Counts[dshort]/BB_Seconds[dshort]
                    BB_Unc[dshort] = np.sqrt(BB_Counts[dshort])/BB_Seconds[dshort]
                item_counter += 1
            f.close()

    if len(BB_CountsPerSecond) < 1:
        f = _sans_get_by_filenumber(Instrument, input_path, examplefilenumber)
        if f is not None:
            for dshort in relevant_detectors:
                if 'VSANS' in Instrument:
                    data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                elif 'NG7SANS' in Instrument:
                    data = np.array(f['entry/instrument/detector/data'])
                BB_CountsPerSecond[dshort] = np.zeros_like(data)
                BB_Unc[dshort] = np.zeros_like(data)
            f.close()

    return BB_CountsPerSecond, BB_Unc #returns empty list or 2D, detector-panel arrays

def _q_calculation_all_detectors(Detector_Panels, Instrument, SampleApertureInMM, SampleDescriptionKeywordsToExclude, input_path, Calc_Q_From_Trans, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, representative_filenumber, Config, MiddlePixelBorderHorizontal, MiddlePixelBorderVertical, AlignDet_Trans):
    """Build per-pixel Q maps and a shadow mask for every relevant panel.

    Computes pixel-level ``Qx``, ``Qy``, ``Qz``, ``|Q|``, parallel and
    perpendicular Q uncertainties (including gravity correction), the in-plane
    azimuthal angle map, and a beam-line shadow mask (beam stop and detector
    overlaps) for the configuration of ``representative_filenumber``.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    SampleApertureInMM : bool
        Required. If true, sample aperture values read from the NeXus file
        are converted from mm to cm.
    SampleDescriptionKeywordsToExclude : list[str] or None
        Required. Substrings used by ``_sans_sample_base_name_descrip`` to
        strip noise from the sample description.
    input_path : str
        Required. Directory containing raw NeXus files.
    Calc_Q_From_Trans : bool
        Required. If true, override the recorded beam center with one
        derived from aligned transmission files via
        :func:`_assign_beam_center_for_scatt_file`.
    HighResMinX, HighResMaxX, HighResMinY, HighResMaxY : int
        Required. Pixel-index bounds defining the high-resolution back
        detector subset.
    ConvertHighResToSubset : bool
        Required. If true, crop back-detector Q arrays to the
        ``HighRes*`` bounds.
    representative_filenumber : int
        Required. Run number whose geometry is read.
    Config : str
        Required. Configuration label; ``'CvB'`` adds detector ``'B'``.
    MiddlePixelBorderHorizontal, MiddlePixelBorderVertical : int
        Required. Width (in pixels) of the masked border on the middle
        carriage detectors.
    AlignDet_Trans : dict
        Required. Aligned-transmission catalog used by
        :func:`_assign_beam_center_for_scatt_file`.

    Returns
    -------
    Qx, Qy, Qz, Q_total : dict[str, np.ndarray]
        Per-panel components and magnitude of Q in inverse Angstroms.
    Q_perp_unc, Q_parl_unc : dict[str, np.ndarray]
        Per-pixel perpendicular and parallel Q uncertainties.
    InPlaneAngleMap : dict[str, np.ndarray]
        Per-pixel azimuthal angle in degrees (``[-180, 180]``).
    dimXX, dimYY : dict[str, int]
        Per-panel detector dimensions (after any high-res cropping).
    Shadow_Mask : dict[str, np.ndarray]
        Per-panel mask (0 inside the shadow, ~1.2 outside).
    """

    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')

    Q_total = {}
    deltaQ = {}
    Qx = {}
    Qy = {}
    Qz = {}
    Q_perp_unc = {}
    Q_parl_unc = {}
    InPlaneAngleMap = {}
    TwoThetaAngleMap = {}
    twotheta_x = {}
    twotheta_y = {}
    twotheta_xmin = {}
    twotheta_xmax = {}
    twotheta_ymin = {}
    twotheta_ymax = {}
    dimXX = {}
    dimYY = {}

    Sample_Base, Sample_Name, Descrip, Listed_Config, Desired_Temp, Voltage = _sans_sample_base_name_descrip(Instrument, SampleDescriptionKeywordsToExclude, input_path, representative_filenumber)

    f = _sans_get_by_filenumber(Instrument, input_path, representative_filenumber)
    if f is not None:
        for dshort in relevant_detectors:
            if 'VSANS' in Instrument:
                data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                Wavelength = f['entry/instrument/beam/monochromator/wavelength'][0] # Angstroms
                Wavelength_spread = f['entry/instrument/beam/monochromator/wavelength_spread'][0] # fraction of Wavelength (dL/L)
                dimX = f['entry/instrument/detector_{ds}/pixel_num_x'.format(ds=dshort)][0]
                dimY = f['entry/instrument/detector_{ds}/pixel_num_y'.format(ds=dshort)][0]
                dimXX[dshort] = f['entry/instrument/detector_{ds}/pixel_num_x'.format(ds=dshort)][0]
                dimYY[dshort] = f['entry/instrument/detector_{ds}/pixel_num_y'.format(ds=dshort)][0]
                beam_center_x = f['entry/instrument/detector_{ds}/beam_center_x'.format(ds=dshort)][0]
                beam_center_y = f['entry/instrument/detector_{ds}/beam_center_y'.format(ds=dshort)][0]
                if Calc_Q_From_Trans:
                    X_FR, Y_FR, X_MR, Y_MR = _assign_beam_center_for_scatt_file(Instrument, input_path, Sample_Name, Config, AlignDet_Trans)
                    x_ctr_offset = 0.0
                    y_ctr_offset = 0.0
                    beam_center_x_infile = beam_center_x
                    beam_center_y_infile = beam_center_y
                    if dshort == 'FR' or dshort == 'FL' or dshort == 'FT' or dshort == 'FB':
                        if 'NA' not in str(X_FR) and 'NA' not in str(Y_FR):
                            if dshort == 'FR': #FR x_ctr_offset: 0.0, y_ctr_offset: 0.0, panel_gap: 0.35, setback: 0.0
                                x_ctr_offset = 0.0
                                y_ctr_offset = 0.0
                            elif dshort == 'FL': #FL x_ctr_offset: 0.13, y_ctr_offset: 0.35, panel_gap: 0.35, setback: 0.0
                                x_ctr_offset = 0.13
                                y_ctr_offset = 0.35
                            elif dshort == 'FT': #FT x_ctr_offset: 1.59, y_ctr_offset: 0.09, panel_gap: 0.33, setback: 41.0
                                x_ctr_offset = 1.59
                                y_ctr_offset = 0.09
                            elif dshort == 'FB': #FB x_ctr_offset: 0.95, y_ctr_offset: 0.77, panel_gap: 0.33, setback: 41.0
                                x_ctr_offset = 0.95
                                y_ctr_offset = 0.77
                            beam_center_x = X_FR + x_ctr_offset
                            beam_center_y = Y_FR + y_ctr_offset
                    if dshort == 'MR' or dshort == 'ML' or dshort == 'MT' or dshort == 'MB':
                        if 'NA' not in str(X_MR) and 'NA' not in str(Y_MR):
                            if dshort == 'MR': #MR x_ctr_offset: 0.0, y_ctr_offset: 0.0, panel_gap: 0.59, setback: 0.0
                                x_ctr_offset = 0.0
                                y_ctr_offset = 0.0
                            elif dshort == 'ML':#ML x_ctr_offset: 0.26, y_ctr_offset: -0.16, panel_gap: 0.59, setback: 0.0
                                x_ctr_offset = 0.26
                                y_ctr_offset = -0.16
                            elif dshort == 'MT':#MT x_ctr_offset:  -0.28, y_ctr_offset: 0.60, panel_gap: 1.83, setback: 41.0
                                x_ctr_offset = -0.28
                                y_ctr_offset = 0.60
                            elif dshort == 'MB':#MB x_ctr_offset:  -0.89, y_ctr_offset: 0.96, panel_gap: 1.83, setback: 41.0
                                x_ctr_offset = -0.89
                                y_ctr_offset = 0.96
                            beam_center_x = X_MR + x_ctr_offset
                            beam_center_y = Y_MR + y_ctr_offset
                    beamstop_diameter = f['/entry/DAS_logs/C2BeamStop/diameter'][0]/10.0 #beam stop in cm; sits right in front of middle detector?
                detector_distance = f['entry/instrument/detector_{ds}/distance'.format(ds=dshort)][0]
                x_pixel_size = f['entry/instrument/detector_{ds}/x_pixel_size'.format(ds=dshort)][0]/10.0
                y_pixel_size = f['entry/instrument/detector_{ds}/y_pixel_size'.format(ds=dshort)][0]/10.0
                if dshort != 'B':
                    panel_gap = f['entry/instrument/detector_{ds}/panel_gap'.format(ds=dshort)][0]/10.0
                    coeffs = f['entry/instrument/detector_{ds}/spatial_calibration'.format(ds=dshort)][0][0]/10.0
                SampleApInternal = f['/entry/DAS_logs/geometry/internalSampleApertureHeight'][0] #internal sample aperture in cm
                SampleApShape = f['/entry/DAS_logs/geometry/externalSampleApertureShape'][0]
                if SampleApShape == 'CIRCLE':
                    SampleApExternal = f['/entry/DAS_logs/geometry/externalSampleAperture'][0] # in cm
                else:
                    SampleApExternal = f['/entry/DAS_logs/geometry/externalSampleApertureHeight'][0] #external sample aperture in cm
                if SampleApertureInMM:
                    SampleApExternal *= 0.1 # convert mm to cm
                SampleApOffset = f['/entry/instrument/sample_aperture_2/distance'][0] # distance from sample to external sample aperture, in cm
                SourceAp = f['/entry/DAS_logs/geometry/sourceApertureHeight'][0] #source aperture in cm, assumes circular aperture(?) #0.75, 1.5, or 3 for guides; otherwise 6 cm for >= 1 guides
                FrontDetToGateValve = f['/entry/DAS_logs/carriage/frontTrans'][0] #400
                MiddleDetToGateValve = f['/entry/DAS_logs/carriage/middleTrans'][0] #1650
                RearDetToGateValve = f['/entry/DAS_logs/carriage/rearTrans'][0]
                FrontDetToSample = f['/entry/DAS_logs/geometry/sampleToFrontLeftDetector'][0] #491.4
                MiddleDetToSample = f['/entry/DAS_logs/geometry/sampleToMiddleLeftDetector'][0] #1741.4
                RearDetToSample = f['/entry/DAS_logs/geometry/sampleToRearDetector'][0]
                SampleToSourceAp = f['/entry/DAS_logs/geometry/sourceApertureToSample'][0] #1490.6; "Calculated distance between sample and source aperture" in cm
                    
                if dshort == 'MT' or dshort == 'MB' or dshort == 'FT' or dshort == 'FB':
                    setback = f['entry/instrument/detector_{ds}/setback'.format(ds=dshort)][0]
                    vertical_offset = f['entry/instrument/detector_{ds}/vertical_offset'.format(ds=dshort)][0]
                    lateral_offset = 0
                else:
                    setback = 0
                    vertical_offset = 0
                    lateral_offset = f['entry/instrument/detector_{ds}/lateral_offset'.format(ds=dshort)][0]

            elif 'NG7SANS' in Instrument:
                data = np.array(f['entry/instrument/detector/data'])
                Wavelength = f['entry/instrument/monochromator/wavelength'][0]
                Wavelength_spread = f['entry/instrument/monochromator/wavelength_error'][0]
                dimX = 128 
                dimY = 128
                dimXX[dshort] = dimX
                dimYY[dshort] = dimY
                beam_center_x = f['entry/instrument/detector/beam_center_x'][0]
                beam_center_y = f['entry/instrument/detector/beam_center_y'][0]
                beamstop_diameter = f['/entry/DAS_logs/beamStop/size'][0] #says beam stop in inches, but seems to be in cm
                beamstop_to_detector = f['/entry/DAS_logs/beamStop/detectorDistance'][0] #in cm
                detector_distance = f['entry/instrument/detector/distance'][0] #in cm
                x_pixel_size = f['entry/instrument/detector/x_pixel_size'][0]/10.0
                y_pixel_size = f['entry/instrument/detector/y_pixel_size'][0]/10.0
                lateral_offset = f['entry/DAS_logs/areaDetector/offset'][0] #in cm?

                SampleApShape = f['/entry/DAS_logs/geometry/externalSampleApertureShape'][0]
                if SampleApShape == 'CIRCLE':
                    SampleApExternal = f['/entry/DAS_logs/geometry/externalSampleAperture'][0] # in cm
                else:
                    SampleApExternal = f['/entry/DAS_logs/geometry/externalSampleApertureHeight'][0] #external sample aperture in cm
                if SampleApertureInMM:
                    SampleApExternal *= 0.1 # convert mm to cm
                SampleApOffset = f['/entry/instrument/sample_aperture/distance'][0] # distance from sample to external sample aperture, in cm
                SourceAp = str(f['/entry/DAS_logs/geometry/sourceAperture'][0])
                SourceAp = SourceAp[2:-4]
                SourceAp = float(SourceAp)/10.0
                SampleToSourceAp = f['/entry/DAS_logs/geometry/sourceApertureToSample'][0] #1490.6; "Calculated distance between sample and source aperture" in cm
                setback = 0

            realDistZ = detector_distance + setback

            if 'NG7SANS' in Instrument:
                realDistX =  x_pixel_size*(1.0) + lateral_offset
                realDistY =  y_pixel_size*(1.0)
            elif dshort == 'B' and 'VSANS' in Instrument:
                realDistX =  x_pixel_size*(0.5)
                realDistY =  y_pixel_size*(0.5)
            elif 'VSANS' in Instrument:
                position_key = dshort[1]
                if position_key == 'T':
                    realDistX =  coeffs
                    realDistY =  0.5 * y_pixel_size + vertical_offset + panel_gap/2.0
                elif position_key == 'B':
                    realDistX =  coeffs
                    realDistY =  vertical_offset - (dimY - 0.5)*y_pixel_size - panel_gap/2.0
                elif position_key == 'L':
                    realDistX =  lateral_offset - (dimX - 0.5)*x_pixel_size - panel_gap/2.0
                    realDistY =  coeffs
                elif position_key == 'R':
                    realDistX =  x_pixel_size*(0.5) + lateral_offset + panel_gap/2.0
                    realDistY =  coeffs

            X, Y = np.indices(data.shape)
            if 'NG7SANS' in Instrument:
                x0_pos =  realDistX - beam_center_x*x_pixel_size + (X)*x_pixel_size 
                y0_pos =  realDistY - beam_center_y*y_pixel_size + (Y)*y_pixel_size
                x_min =  realDistX - beam_center_x*x_pixel_size - x_pixel_size 
                y_min =  realDistY - beam_center_y*y_pixel_size - y_pixel_size
                x_max =  realDistX - beam_center_x*x_pixel_size + (dimX)*x_pixel_size 
                y_max =  realDistY - beam_center_y*y_pixel_size + (dimY)*y_pixel_size
            elif dshort == 'B' and 'VSANS' in Instrument:
                x0_pos =  realDistX - beam_center_x*x_pixel_size + (X)*x_pixel_size 
                y0_pos =  realDistY - beam_center_y*y_pixel_size + (Y)*y_pixel_size
                x_min =  realDistX - beam_center_x*x_pixel_size - x_pixel_size 
                y_min =  realDistY - beam_center_y*y_pixel_size - y_pixel_size
                x_max =  realDistX - beam_center_x*x_pixel_size + (dimX)*x_pixel_size 
                y_max =  realDistY - beam_center_y*y_pixel_size + (dimY)*y_pixel_size
            elif 'VSANS' in Instrument:
                x0_pos =  realDistX - beam_center_x + (X)*x_pixel_size 
                y0_pos =  realDistY - beam_center_y + (Y)*y_pixel_size
                x_min =  realDistX - beam_center_x - (1.0)*x_pixel_size
                y_min =  realDistY - beam_center_y - (1.0)*y_pixel_size
                x_max =  realDistX - beam_center_x + (dimX)*x_pixel_size
                y_max =  realDistY - beam_center_y + (dimY)*y_pixel_size
                
            if ConvertHighResToSubset and dshort == 'B':
                dimXX[dshort] = int(HighResMaxX - HighResMinX + 1)
                dimYY[dshort] = int(HighResMaxY - HighResMinY + 1)
                x0_pos = x0_pos[HighResMinX:HighResMaxX+1,HighResMinY:HighResMaxY+1]
                y0_pos = y0_pos[HighResMinX:HighResMaxX+1,HighResMinY:HighResMaxY+1]
                x_min =  realDistX - beam_center_x*x_pixel_size + HighResMinX*x_pixel_size 
                y_min =  realDistY - beam_center_y*y_pixel_size + HighResMinY*y_pixel_size
                x_max =  realDistX - beam_center_x*x_pixel_size + HighResMaxX*x_pixel_size 
                y_max =  realDistY - beam_center_y*y_pixel_size + HighResMaxY*y_pixel_size
                
            InPlane0_pos = np.sqrt(x0_pos**2 + y0_pos**2)
            twotheta = np.arctan2(InPlane0_pos,realDistZ)
            twotheta_x[dshort] = np.arctan2(x0_pos,realDistZ)
            twotheta_y[dshort] = np.arctan2(y0_pos,realDistZ)
            twotheta_xmin[dshort] = np.arctan2(x_min,realDistZ)
            twotheta_xmax[dshort] = np.arctan2(x_max,realDistZ)
            twotheta_ymin[dshort] = np.arctan2(y_min,realDistZ)
            twotheta_ymax[dshort] = np.arctan2(y_max,realDistZ)

            g = 981.0 #in cm/s^2
            m_div_h = 252.77 #in s cm^-2
            acc = 3.956e5 # velocity [cm/s] of 1 A neutron
            L2 = realDistZ
            L1 = SampleToSourceAp
            Pix = 0.82
            R1 = SourceAp * 0.5 #source aperture diameter, to radius in cm
            R2 = SampleApExternal * 0.5 #sample aperture diameter, to radius in cm
            Inv_LPrime = 1.0/L1 + 1.0/L2
            k = 2*np.pi/Wavelength
            YG_d = -0.5*g*L2*(L1+L2)*(Wavelength/acc)**2
            phi = np.mod(np.arctan2(y0_pos + 2.0*YG_d,x0_pos), 2.0*np.pi) # constrain to [0, 2pi]
            Sigma_D_Perp = np.abs(np.sin(phi)*x_pixel_size) + np.abs(np.cos(phi)*y_pixel_size)
            Sigma_D_Parl = np.abs(np.cos(phi)*x_pixel_size) + np.abs(np.sin(phi)*y_pixel_size)
            SigmaQPerpSqr = (k*k/12.0)*(3*np.power(R1/L1,2) + 3.0*np.power(R2*Inv_LPrime,2)+ np.power(Sigma_D_Perp/L2,2))
            SigmaQParlSqr = (k*k/12.0)*(3*np.power(R1/L1,2) + 3.0*np.power(R2*Inv_LPrime,2)+ np.power(Sigma_D_Parl/L2,2))
            R = np.sqrt(np.power(x0_pos,2)+np.power(y0_pos + 2.0*YG_d,2))
            Q0 = k*R/L2
            '''
            #If no gravity correction:
            #SigmaQParlSqr = SigmaQParlSqr + np.power(Q0,2)*np.power(Wavelength_spread/np.sqrt(6.0),2)
            #Else, if adding gravity correction:
            '''
            
            A = 0.5*g*L2*(L1+L2)*np.power(m_div_h , 2) # in units 1/cm
            WL = Wavelength*1E-8 # in cm

            SigmaQParlSqr = SigmaQParlSqr + np.power(Wavelength_spread*k/(L2),2)*(R*R - 4*R*A*np.sin(phi)*WL*WL + 4*A*A*np.power(WL,4))/6.0 #gravity correction makes vary little difference for wavelength spread < 20%
            '''VSANS IGOR 2D ASCII delta_Q seems to be way off the mark, but this 2D calculaation matches the VSANS circular average closely when pixels are converted to circular average...'''
            

            Q_total[dshort] = (4.0*np.pi/Wavelength)*np.sin(twotheta/2.0)
            QQ_total = (4.0*np.pi/Wavelength)*np.sin(twotheta/2.0)
            Qx[dshort] = QQ_total*np.cos(twotheta/2.0)*np.cos(phi)
            Qy[dshort] = QQ_total*np.cos(twotheta/2.0)*np.sin(phi)
            Qz[dshort] = QQ_total*np.sin(twotheta/2.0)     
            Q_perp_unc[dshort] = np.ones_like(Q_total[dshort])*np.sqrt(SigmaQPerpSqr)
            Q_parl_unc[dshort] = np.sqrt(SigmaQParlSqr)
            Phi_deg = phi*180.0/np.pi
            TwoTheta_deg = twotheta*180.0/np.pi
            InPlaneAngleMap[dshort] = Phi_deg
            TwoThetaAngleMap[dshort] = TwoTheta_deg
            '''#returns values between -180.0 degrees and +180.0 degrees'''

    Shadow_Mask = {}
    for dshort in relevant_detectors:
        Shadow = 1.2*np.ones_like(Qx[dshort])
        x,y = np.shape(Shadow)
        if 'NG7SANS' in Instrument:
            #Note: Should add R_Shadow to VSANS Shadow Mask, too! (Kludge)
            R_shadow = (beamstop_diameter/2.0 + SampleApExternal/2.0)*detector_distance/(detector_distance - beamstop_to_detector)
            for i in range(0,2):#NG7 Change, was range(0,MiddlePixelBorderHorizontal)
                Shadow[:,i] = np.zeros(x)
                Shadow[:,y-i-1] = np.zeros(x)
            for j in range(0,2): #NG7 Change, was range(0,MiddlePixelBorderVertical)
                Shadow[j] = np.zeros(y)
                Shadow[x-j-1] = np.zeros(y)
            Shadow[R <= R_shadow] = 0.0 #NG7 Change
            Shadow_Mask[dshort] = Shadow
        elif 'VSANS' in Instrument: 
            if dshort == 'ML' or dshort == 'MR' or dshort == 'MT' or dshort == 'MB':
                for i in range(0,MiddlePixelBorderHorizontal):
                    Shadow[:,i] = np.zeros(x)
                    Shadow[:,y-i-1] = np.zeros(x)
                for j in range(0,MiddlePixelBorderVertical):
                    if dshort != 'MR':
                        Shadow[j] = np.zeros(y)
                    if dshort != 'ML':
                        Shadow[x-j-1] = np.zeros(y)
                        
            if dshort == 'FT' or dshort == 'FB':
                Shadow[twotheta_x[dshort] <= twotheta_xmax['FL']] = 0.0
                Shadow[twotheta_x[dshort] >= twotheta_xmin['FR']] = 0.0
                
            if dshort == "ML" or dshort == "MR":
                Shadow[twotheta_x[dshort] >= twotheta_xmin['FR']] = 0.0
                Shadow[twotheta_x[dshort] <= twotheta_xmax['FL']] = 0.0
                Shadow[twotheta_y[dshort] >= twotheta_ymin['FT']] = 0.0
                Shadow[twotheta_y[dshort] <= twotheta_ymax['FB']] = 0.0
                
            if dshort == "MT" or dshort == "MB":
                Shadow[twotheta_x[dshort] <= twotheta_xmax['FL']] = 0.0
                Shadow[twotheta_x[dshort] >= twotheta_xmin['FR']] = 0.0
                Shadow[twotheta_y[dshort] >= twotheta_ymin['FT']] = 0.0
                Shadow[twotheta_y[dshort] <= twotheta_ymax['FB']] = 0.0
                Shadow[twotheta_x[dshort] <= twotheta_xmax['ML']] = 0.0
                Shadow[twotheta_x[dshort] >= twotheta_xmin['MR']] = 0.0       
            if dshort == "B":
                Shadow[twotheta_x[dshort] <= twotheta_xmax['FL']] = 0.0
                Shadow[twotheta_x[dshort] >= twotheta_xmin['FR']] = 0.0
                Shadow[twotheta_y[dshort] >= twotheta_ymin['FT']] = 0.0
                Shadow[twotheta_y[dshort] <= twotheta_ymax['FB']] = 0.0
                Shadow[twotheta_x[dshort] <= twotheta_xmax['ML']] = 0.0
                Shadow[twotheta_x[dshort] >= twotheta_xmin['MR']] = 0.0
                Shadow[twotheta_y[dshort] >= twotheta_ymin['MT']] = 0.0
                Shadow[twotheta_y[dshort] <= twotheta_ymax['MB']] = 0.0

            Shadow_Mask[dshort] = Shadow
        f.close()

    return Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, InPlaneAngleMap, dimXX, dimYY, Shadow_Mask

def _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, PrimaryAngle, AngleWidth, BothSides):
    """Build a per-panel azimuthal sector mask.

    Returns 1.0 where pixels lie within ``AngleWidth`` of ``PrimaryAngle``
    (and, if ``BothSides``, of ``PrimaryAngle + 180``), 0.0 elsewhere.
    A Si-mirror specific cutout around +90° is applied when ``SiMirror``
    contains ``'IN'``.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    SiMirror : str
        Required. Si-mirror state; ``'IN'`` triggers an extra ±60° cutout
        centered on +90°.
    Config : str
        Required. Configuration label; ``'CvB'`` adds detector ``'B'``.
    InPlaneAngleMap : dict[str, np.ndarray]
        Required. Per-panel in-plane angle map in degrees.
    PrimaryAngle : float
        Required. Center of the sector in degrees.
    AngleWidth : float
        Required. Half-width of the sector in degrees.
    BothSides : int or bool
        Required. If >= 1, also include the diametrically opposite sector.

    Returns
    -------
    SectorMask : dict[str, np.ndarray]
        Per-panel binary sector mask.
    """

    SectorMask = {}
    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')
    for dshort in relevant_detectors:
        Angles = InPlaneAngleMap[dshort]
        SM = np.zeros_like(Angles)

        if PrimaryAngle > 180.0:
            PrimaryAngle = PrimaryAngle - 360.0
        if PrimaryAngle < -180.0:
            PrimaryAngle = PrimaryAngle + 360.0
        SM[np.absolute(Angles - PrimaryAngle) <= AngleWidth] = 1.0
        SM[np.absolute(Angles + 360 - PrimaryAngle) <= AngleWidth] = 1.0
        SM[np.absolute(Angles - 360 - PrimaryAngle) <= AngleWidth] = 1.0
        if BothSides >= 1:
            SecondaryAngle = PrimaryAngle + 180
            if SecondaryAngle > 180.0:
                SecondaryAngle = SecondaryAngle - 360.0
            if SecondaryAngle < -180.0:
                SecondaryAngle = SecondaryAngle + 360.0
            SM[np.absolute(Angles - SecondaryAngle) <= AngleWidth] = 1.0
            SM[np.absolute(Angles + 360 - SecondaryAngle) <= AngleWidth] = 1.0
            SM[np.absolute(Angles - 360 - SecondaryAngle) <= AngleWidth] = 1.0

        if 'IN' in SiMirror:
            SM[np.absolute(Angles - 90.0) <= 60.0] = 0.0
            
        SectorMask[dshort] = SM
        
    return SectorMask

def _min_max_q(Detector_Panels, Instrument, Absolute_Q_min, Absolute_Q_max, Q_total, Config, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY):
    """Choose Q binning limits and bin count for the current configuration.

    Selects ``Q_min`` and ``Q_max`` as the tighter of user-supplied absolute
    limits and the limits actually realized by the detector panels. ``Q_bins``
    scales with the spanned fraction and is increased when the high-resolution
    back detector is included via ``Config`` containing ``'CvB'``.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'`` (selects the panel layout
        used to compute Q extents).
    Absolute_Q_min, Absolute_Q_max : float
        Required. User-imposed absolute Q limits in inverse Angstroms.
    Q_total : dict[str, np.ndarray]
        Required. Per-panel |Q| maps from :func:`_q_calculation_all_detectors`.
    Config : str
        Required. Configuration label.
    HighResMinX, HighResMaxX, HighResMinY, HighResMaxY : int
        Required. High-resolution back-detector pixel bounds (used only to
        size the extra bin allocation).

    Returns
    -------
    Q_min : float
    Q_max : float
    Q_bins : int
    """

    if 'NG7SANS' in Instrument:
        ds = Detector_Panels[0]
        Q_minCalc = np.amin(Q_total[ds])
        Q_maxneg = np.amin(Q_total[ds])
        Q_maxA = np.sqrt(np.power(Q_maxneg,2))
        Q_maxB = np.amax(Q_total[ds])
        Q_maxCalc = np.maximum(Q_maxA, Q_maxB)

    elif 'VSANS' in Instrument:
        MinQ1 = np.amin(Q_total['MR'])
        MinQ2 = np.amin(Q_total['ML'])
        MinQ3 = np.amin(Q_total['MT'])
        MinQ4 = np.amin(Q_total['MB'])
        MinQs = np.array([MinQ1, MinQ2, MinQ3, MinQ4])
        MinQ_Middle = np.amin(MinQs)
        MaxQ1 = np.amax(Q_total['FR'])
        MaxQ2 = np.amax(Q_total['FL'])
        MaxQ3 = np.amax(Q_total['FT'])
        MaxQ4 = np.amax(Q_total['FB'])
        MaxQs = np.array([MaxQ1, MaxQ2, MaxQ3, MaxQ4])
        MaxQ_Front = np.amax(MaxQs)
        Q_minCalc = MinQ_Middle 
        Q_maxCalc = MaxQ_Front
        
    Q_min = np.maximum(Absolute_Q_min, Q_minCalc)
    Q_max = np.minimum(Absolute_Q_max, Q_maxCalc)
    Q_bins = int(150*(Q_max - Q_min)/(Q_maxCalc - Q_minCalc))
    

    if str(Config).find('CvB') != -1:
        HR_Q_min = np.amin(Q_total['B'])
        Q_min_HR = np.maximum(HR_Q_min, Absolute_Q_min)
        HR_bins = int(np.sqrt(np.power((HighResMaxX - HighResMinX + 1)/2, 2) + np.power((HighResMaxY - HighResMinY + 1)/2, 2)))

        Q_min = Q_min_HR
        Q_bins = 4*(Q_bins + HR_bins)
    
    return Q_min, Q_max, Q_bins

def _assign_beam_center_for_scatt_file(Instrument, input_path, Sample_Name, Config, AlignTrans):
    """Look up the FR and MR beam-center coordinates for a sample/config.

    Selects an aligned-transmission file (polarized preferred, unpolarized
    fallback) for the FR and MR panels and computes the beam center via
    :func:`_get_beam_center`. Returns ``'NA'`` for any axis that cannot be
    resolved.

    Parameters
    ----------
    Instrument : str
        Required. Instrument identifier.
    input_path : str
        Required. Directory containing raw NeXus files.
    Sample_Name : str
        Required. Sample key used to index ``AlignTrans``.
    Config : str
        Required. Configuration key used to index ``AlignTrans[Sample_Name]``.
    AlignTrans : dict
        Required. Aligned-transmission catalog with FR/MR Pol/Unpol file
        lists under each sample/config.

    Returns
    -------
    X_FR, Y_FR : float or 'NA'
        Beam-center coordinates on the FR panel.
    X_MR, Y_MR : float or 'NA'
        Beam-center coordinates on the MR panel.
    """


    FR_filenumber = 0
    MR_filenumber = 0
    X_FR = 'NA'
    Y_FR = 'NA'
    X_MR = 'NA'
    Y_MR = 'NA'

    if Sample_Name in AlignTrans:
        if Config in AlignTrans[Sample_Name]['Config(s)']:
            if 'NA' not in AlignTrans[Sample_Name]['Config(s)'][Config]['FR_Pol_Files']:
                FR_filenumber = AlignTrans[Sample_Name]['Config(s)'][Config]['FR_Pol_Files'][0]
            elif 'NA' not in AlignTrans[Sample_Name]['Config(s)'][Config]['FR_Unpol_Files']:
                FR_filenumber = AlignTrans[Sample_Name]['Config(s)'][Config]['FR_Unpol_Files'][0]
            if 'NA' not in AlignTrans[Sample_Name]['Config(s)'][Config]['MR_Pol_Files']:
                MR_filenumber = AlignTrans[Sample_Name]['Config(s)'][Config]['MR_Pol_Files'][0]
            elif 'NA' not in AlignTrans[Sample_Name]['Config(s)'][Config]['MR_Unpol_Files']:
                MR_filenumber = AlignTrans[Sample_Name]['Config(s)'][Config]['MR_Unpol_Files'][0]

        trans_max_width_pixels = 10
        if FR_filenumber != 0:
            X_FR, Y_FR = _get_beam_center(Instrument, input_path, FR_filenumber, 'FR', trans_max_width_pixels)
        if MR_filenumber != 0:
            X_MR, Y_MR = _get_beam_center(Instrument, input_path, MR_filenumber, 'MR', trans_max_width_pixels)

    return X_FR, Y_FR, X_MR, Y_MR

def _abs_scale(Detector_Panels, Instrument, YesNoManualHe3Entry, input_path, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, HighResGain, ScattType, Sample, Config, BlockBeam_per_second, Solid_Angle, Plex, Scatt, Trans):
    """Absolute-scale and block-beam-subtract the scattering for one cross-section.

    For each run in ``Scatt[Sample]['Config(s)'][Config][ScattType]`` the
    per-panel data is block-beam corrected, divided by the plex, solid
    angle, monitor counts, transmission, and (for polarized cross-sections)
    the 3He glass transmission, then accumulated.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    YesNoManualHe3Entry : bool
        Required. If true, use manually supplied ``TeValues[0]`` instead of
        the NeXus glass transmission entry.
    input_path : str
        Required. Directory containing raw NeXus files.
    HighResMinX, HighResMaxX, HighResMinY, HighResMaxY : int
        Required. High-resolution back-detector pixel bounds.
    ConvertHighResToSubset : bool
        Required. If true, crop the back detector to the high-res bounds.
    HighResGain : float
        Required. Gain factor applied to the back-detector counts.
    ScattType : {'UU','DU','DD','UD','U','D','Unpol'}
        Required. Which scattering cross-section to scale.
    Sample : str
        Required. Sample name.
    Config : str
        Required. Configuration label.
    BlockBeam_per_second : dict[str, np.ndarray]
        Required. Block-beam counts/second per panel.
    Solid_Angle : dict[str, float]
        Required. Per-panel pixel solid angles.
    Plex : dict[str, np.ndarray]
        Required. Per-panel plex / detector efficiency arrays.
    Scatt : dict
        Required. Scattering catalog.
    Trans : dict
        Required. Transmission catalog supplying ``U_Trans_Cts`` or
        ``Unpol_Trans_Cts``.

    Returns
    -------
    Scaled_Data : dict[str, np.ndarray] or 'NA'
        Absolute-scaled intensity per panel; ``'NA'`` if the inputs flag
        missing data.
    UncScaled_Data : dict[str, np.ndarray] or 'NA'
        Matching per-panel uncertainty.
    """

    Scaled_Data = {}
    UncScaled_Data = {}
    masks = {}
    BB = {}

    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')

    if Sample in Scatt:
        if Config in Scatt[Sample]['Config(s)']:
            Number_Files = 1.0*len(Scatt[Sample]['Config(s)'][Config][ScattType])
            if ScattType == 'UU' or ScattType == 'DU'  or ScattType == 'DD'  or ScattType == 'UD' or ScattType == 'U' or ScattType == 'D':
                TransType = 'U_Trans_Cts'
                TransTypeAlt = 'Unpol_Trans_Cts'
            elif ScattType == 'Unpol':
                TransType = 'Unpol_Trans_Cts'
                TransTypeAlt = 'U_Trans_Cts'
            else:
                print('There is a problem with the Scatting Type requested in the Absobulte Scaling Function')
                
            ABS_Scale = 1.0
            if Sample in Trans and str(Scatt[Sample]['Config(s)'][Config][ScattType]).find('NA') == -1:
                if Sample in Trans:
                    if Config in Trans[Sample]['Config(s)']:
                        if TransType in Trans[Sample]['Config(s)'][Config] and str(Trans[Sample]['Config(s)'][Config][TransType]).find("NA") == -1 :
                            ABS_Scale = np.average(np.array(Trans[Sample]['Config(s)'][Config][TransType]))
                        elif TransTypeAlt in Trans[Sample]['Config(s)'][Config] and str(Trans[Sample]['Config(s)'][Config]).find("NA") == -1 :
                            ABS_Scale = np.average(np.array(Trans[Sample]['Config(s)'][Config][TransTypeAlt]))
                            
                
            for dshort in relevant_detectors:
                Holder =  np.array(BlockBeam_per_second[dshort])

                masks[dshort] = np.ones_like(Holder)
                Sum = np.sum(Holder[masks[dshort] > 0])
                Pixels = np.sum(masks[dshort])
                Unc = np.sqrt(Sum)/Pixels
                Ave = np.average(Holder[masks[dshort] > 0])
                BB[dshort] = Ave
                if ConvertHighResToSubset and dshort == 'B':
                    bb_holder = Holder[HighResMinX:HighResMaxX+1,HighResMinY:HighResMaxY+1]
                    bb_sum = np.sum(bb_holder)
                    bb_ave = np.average(Holder[HighResMinX:HighResMaxX+1,HighResMinY:HighResMaxY+1])
                    BB[dshort] = (bb_holder)/HighResGain # Better to subtract BB pixel-by-pixel than average for HighRes detector


            He3Glass_Trans = 1.0
            filecounter = 0
            if str(Scatt[Sample]['Config(s)'][Config][ScattType]).find('NA') != -1:
                Scaled_Data = 'NA'
                UncScaled_Data = 'NA'
            else:
                for filenumber in Scatt[Sample]['Config(s)'][Config][ScattType]:
                    filecounter += 1
                    f = _sans_get_by_filenumber(Instrument, input_path, filenumber)
                    if f is not None:
                        MonCounts = f['entry/control/monitor_counts'][0]
                        Count_time = f['entry/collection_time'][0]
                        He3Glass_Trans = 1.0
                        if ScattType == 'UU' or ScattType == 'DU'  or ScattType == 'DD'  or ScattType == 'UD':
                            if not YesNoManualHe3Entry:
                                He3Glass_Trans = f['/entry/DAS_logs/backPolarization/glassTransmission'][0]
                            else:
                                He3Glass_Trans = TeValues[0]
                        for dshort in relevant_detectors:
                            if 'NG7SANS' in Instrument:
                                data = np.array(f['entry/instrument/detector/data'])
                                unc = np.array(f['entry/instrument/detector/data'])
                            elif 'VSANS' in Instrument:    
                                data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                                unc = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                            if ConvertHighResToSubset and dshort == 'B':
                                data_holder = data/HighResGain
                                data = data_holder[HighResMinX:HighResMaxX+1,HighResMinY:HighResMaxY+1]
                                unc = data_holder[HighResMinX:HighResMaxX+1,HighResMinY:HighResMaxY+1]
                            data = (data - Count_time*BB[dshort])/(Number_Files*Plex[dshort]*Solid_Angle[dshort])
                            if filecounter < 2:
                                Scaled_Data[dshort] = ((1E8/MonCounts)/(ABS_Scale*He3Glass_Trans))*data
                                UncScaled_Data[dshort] = unc
                            else:
                                Scaled_Data[dshort] += ((1E8/MonCounts)/(ABS_Scale*He3Glass_Trans))*data
                                UncScaled_Data[dshort] += unc
                        f.close()
                for dshort in relevant_detectors:
                    UncScaled_Data[dshort] = np.sqrt(UncScaled_Data[dshort])*((1E8/MonCounts)/(ABS_Scale*He3Glass_Trans))/(Number_Files*Plex[dshort]*Solid_Angle[dshort])
        else:
            Scaled_Data = 'NA'
            UncScaled_Data = 'NA'

                
    return Scaled_Data, UncScaled_Data

def _sans_make_slices_and_save_ascii(YesNoShowPlots, Detector_Panels, Instrument, SampleApertureInMM, SampleDescriptionKeywordsToExclude, UsePolCorr, YesNoManualHe3Entry, input_path, save_path, He3CorrectionType, YesNo_2DFilesPerDetector, YesNo_2DCombinedFiles, Absolute_Q_min, Absolute_Q_max, AverageQRanges, Calc_Q_From_Trans, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, HighResGain, HE3_Cell_Summary, Plex, Truest_PSM, Minimum_PSM, AlignDet_Trans, He3Only_Check, ScattCatalog, BlockBeamCatalog, Configs, Sample_Names, TransCatalog, Pol_TransCatalog, MiddlePixelBorderHorizontal, MiddlePixelBorderVertical, SectorCutAngles, Slices,  YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax):
    """Drive the per-sample, per-configuration scaling, pol-correction, and slicing.

    For every configuration in ``Configs`` and every sample in ``Sample_Names``
    this scales the four polarized cross-sections, half-pol pair, and unpol
    runs (via :func:`_abs_scale`); applies the full polarization correction
    (via :func:`_all_sans_pol_corr_scatt_files`); optionally writes 2-D
    ASCII files (per-panel and/or combined); and produces 1-D slices (circ /
    horz / vert / diag) for full-pol, half-pol, and unpol samples.

    Parameters
    ----------
    YesNoShowPlots : bool
        Required. If true, generated matplotlib figures are also displayed.
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    SampleApertureInMM : bool
        Required. See :func:`_q_calculation_all_detectors`.
    SampleDescriptionKeywordsToExclude : list[str] or None
        Required. Keywords stripped from sample descriptions; ``None`` is
        treated as an empty list.
    UsePolCorr : bool or int
        Required. If truthy, apply the full polarization-correction matrix
        inversion; otherwise apply only the He3-efficiency correction.
    YesNoManualHe3Entry : bool
        Required. Use manually supplied 3He values rather than NeXus entries.
    input_path, save_path : str
        Required. Input file directory and output directory.
    He3CorrectionType : {0, 1, 2}
        Required. Selects the polarization-efficiency matrix form (see
        :func:`_all_sans_pol_corr_scatt_files`).
    YesNo_2DFilesPerDetector, YesNo_2DCombinedFiles : bool
        Required. Toggles for writing per-panel and combined 2-D ASCII files.
    Absolute_Q_min, Absolute_Q_max : float
        Required. User-imposed absolute Q limits.
    AverageQRanges : bool
        Required. If true, overlapping carriage Q bins are averaged rather
        than trimmed.
    Calc_Q_From_Trans : bool
        Required. If true, refine the beam center from transmission files.
    HighResMinX, HighResMaxX, HighResMinY, HighResMaxY : int
        Required. High-resolution back-detector bounds.
    ConvertHighResToSubset : bool
        Required. Crop back detector to high-res bounds when true.
    HighResGain : float
        Required. Gain factor for the back detector.
    HE3_Cell_Summary : dict
        Required. 3He cell parameter summary.
    Plex : dict[str, np.ndarray]
        Required. Per-panel plex arrays.
    Truest_PSM, Minimum_PSM : float
        Required. Best-known and floor values for the supermirror polarization.
    AlignDet_Trans : dict
        Required. Aligned-transmission catalog.
    He3Only_Check : bool
        Required. If true, skip the full reduction.
    ScattCatalog, BlockBeamCatalog, TransCatalog, Pol_TransCatalog : dict
        Required. The four catalogs driving the reduction.
    Configs : dict[str, int]
        Required. Mapping of configuration label to representative file
        number; a value of 0 skips that config.
    Sample_Names : Iterable[str]
        Required. Sample keys to process.
    MiddlePixelBorderHorizontal, MiddlePixelBorderVertical : int
        Required. Border widths masked on middle-carriage detectors.
    SectorCutAngles : float
        Required. Sector half-width (degrees) for Horz/Vert/Diag slices.
    Slices : Iterable[str]
        Required. Slice keys to compute (``'Circ'``/``'Horz'``/``'Vert'``/``'Diag'``).
    YesNoSetPlotXRange, YesNoSetPlotYRange : bool
        Required. Toggle manual plot axis limits.
    PlotXmin, PlotXmax, PlotYmin, PlotYmax : float
        Required. Manual plot axis limits (used only when toggled on).

    Returns
    -------
    AllFullPolSlices : dict
        ``{Config: {Sample: {slice_key: {...}}}}`` of full-pol slices.
    AllHalfPolSlices : dict
        Same shape, for half-pol slices.
    AllUnpolSlices : dict
        Same shape, for unpolarized slices.
    """

    if SampleDescriptionKeywordsToExclude == None:
        SampleDescriptionKeywordsToExclude = []
    if not He3Only_Check:
        QValues_All = {}
        AllFullPolSlices = {}
        AllHalfPolSlices = {}
        AllUnpolSlices = {}
        for Config in Configs:
            representative_filenumber = Configs[Config]
            if representative_filenumber != 0:
                Solid_Angle = _solid_angle_all_detectors(Detector_Panels, Instrument, input_path, representative_filenumber, Config)
                BBList = [0]
                if Config in BlockBeamCatalog:
                    if 'NA' not in BlockBeamCatalog[Config]['Trans']['File']:
                        BBList = BlockBeamCatalog[Config]['Trans']['File']
                    elif 'NA' not in BlockBeamCatalog[Config]['Scatt']['File']:
                        BBList = BlockBeamCatalog[Config]['Scatt']['File']
                BB_per_second, BBUnc_per_second = _all_sans_blocked_beam_counts_per_second_list_of_files(Detector_Panels, Instrument, input_path, BBList, Config, representative_filenumber)
                Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, InPlaneAngleMap, dimXX, dimYY, Shadow_Mask = _q_calculation_all_detectors(Detector_Panels, Instrument, SampleApertureInMM, SampleDescriptionKeywordsToExclude, input_path, Calc_Q_From_Trans, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, representative_filenumber, Config, MiddlePixelBorderHorizontal, MiddlePixelBorderVertical, AlignDet_Trans)
                QValues_All = {'QX':Qx,'QY':Qy,'QZ':Qz,'Q_total':Q_total,'Q_perp_unc':Q_perp_unc,'Q_parl_unc':Q_parl_unc}
                Q_min, Q_max, Q_bins = _min_max_q(Detector_Panels, Instrument, Absolute_Q_min, Absolute_Q_max, Q_total, Config, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY)
                            
                relevant_detectors = list(Detector_Panels)
                if str(Config).find('CvB') != -1:
                    relevant_detectors.append('B')
                                

                FullPolSampleSlices = {}
                HalfPolSampleSlices = {}
                UnpolSampleSlices = {}
                for Sample in Sample_Names:
                    if Sample in ScattCatalog:
                        _assign_beam_center_for_scatt_file(Instrument, input_path, Sample, Config, AlignDet_Trans)
                                                    
                        if str(ScattCatalog[Sample]['Intent']).find('Sample') != -1 or str(ScattCatalog[Sample]['Intent']).find('Empty') != -1:

                            UUScaledData, UUScaledData_Unc = _abs_scale(Detector_Panels, Instrument, YesNoManualHe3Entry, input_path, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, HighResGain, 'UU', Sample, Config, BB_per_second, Solid_Angle, Plex, ScattCatalog, TransCatalog)
                            DUScaledData, DUScaledData_Unc = _abs_scale(Detector_Panels, Instrument, YesNoManualHe3Entry, input_path, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, HighResGain, 'DU', Sample, Config, BB_per_second, Solid_Angle, Plex, ScattCatalog, TransCatalog)
                            DDScaledData, DDScaledData_Unc = _abs_scale(Detector_Panels, Instrument, YesNoManualHe3Entry, input_path, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, HighResGain, 'DD', Sample, Config, BB_per_second, Solid_Angle, Plex, ScattCatalog, TransCatalog)
                            UDScaledData, UDScaledData_Unc = _abs_scale(Detector_Panels, Instrument, YesNoManualHe3Entry, input_path, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, HighResGain, 'UD', Sample, Config, BB_per_second, Solid_Angle, Plex, ScattCatalog, TransCatalog)
                            FullPolGo = 0
                            if 'NA' not in UUScaledData and 'NA' not in DUScaledData and 'NA' not in DDScaledData and 'NA' not in UDScaledData:

                                
                                representative_filenumber = ScattCatalog[Sample]['Config(s)'][Config]['UU'][0]
                                Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, InPlaneAngleMap, dimXX, dimYY, Shadow_Mask = _q_calculation_all_detectors(Detector_Panels, Instrument, SampleApertureInMM, SampleDescriptionKeywordsToExclude, input_path, Calc_Q_From_Trans, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, representative_filenumber, Config, MiddlePixelBorderHorizontal, MiddlePixelBorderVertical, AlignDet_Trans)
                                QValues_All = {'QX':Qx,'QY':Qy,'QZ':Qz,'Q_total':Q_total,'Q_perp_unc':Q_perp_unc,'Q_parl_unc':Q_parl_unc}
                                FullPolGo, UnpolEquiv, PolCorrUU, PolCorrDU, PolCorrDD, PolCorrUD, UnpolEquiv_Unc, PolCorrUU_Unc, PolCorrDU_Unc, PolCorrDD_Unc, PolCorrUD_Unc = _all_sans_pol_corr_scatt_files(Detector_Panels, Instrument, UsePolCorr, input_path, He3CorrectionType, Truest_PSM, Minimum_PSM, dimXX, dimYY, Sample, Config, ScattCatalog, TransCatalog, Pol_TransCatalog, UUScaledData, DUScaledData, DDScaledData, UDScaledData, UUScaledData_Unc, DUScaledData_Unc, DDScaledData_Unc, UDScaledData_Unc, HE3_Cell_Summary)

                                if YesNo_2DCombinedFiles:
                                    if FullPolGo >= 2:
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'PolCorrUU', Sample, Config, PolCorrUU, PolCorrUU_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'PolCorrDU', Sample, Config, PolCorrDU, PolCorrDU_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'PolCorrDD', Sample, Config, PolCorrDD, PolCorrDD_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'PolCorrUD', Sample, Config, PolCorrUD, PolCorrUD_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'PolCorrSumAllCS', Sample, Config, UnpolEquiv, UnpolEquiv_Unc, QValues_All, Shadow_Mask)
                                    elif FullPolGo >= 1 and FullPolGo < 2:
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'He3CorrUU', Sample, Config, PolCorrUU, PolCorrUU_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'He3CorrDU', Sample, Config, PolCorrDU, PolCorrDU_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'He3CorrDD', Sample, Config, PolCorrDD, PolCorrDD_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'He3CorrUD', Sample, Config, PolCorrUD, PolCorrUD_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'He3CorrSumAllCS', Sample, Config, UnpolEquiv, UnpolEquiv_Unc, QValues_All, Shadow_Mask)
                                    else:
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'NotCorrUU', Sample, Config, UUScaledData, UUScaledData_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'NotCorrDU', Sample, Config, DUScaledData, DUScaledData_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'NotCorrDD', Sample, Config, DDScaledData, DDScaledData_Unc, QValues_All, Shadow_Mask)
                                        _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'NotCorrUD', Sample, Config, UDScaledData, UDScaledData_Unc, QValues_All, Shadow_Mask)

                                if str(ScattCatalog[Sample]['Intent']).find('Sample') != -1:
                                    SiMirror = ScattCatalog[Sample]['Config(s)'][Config]['SiMirror']
                                    FullPolSampleSlices[Sample] = _sans_full_pol_slices(YesNoShowPlots, save_path, Detector_Panels, SiMirror, Slices, SectorCutAngles, AverageQRanges, FullPolGo, Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, PolCorrUU, PolCorrUU_Unc, PolCorrDU, PolCorrDU_Unc, PolCorrDD, PolCorrDD_Unc, PolCorrUD, PolCorrUD_Unc, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax)
                                if str(ScattCatalog[Sample]['Intent']).find('Empty') != -1:
                                    SiMirror = ScattCatalog[Sample]['Config(s)'][Config]['SiMirror']
                                    FullPolSampleSlices['Empty'] = _sans_full_pol_slices(YesNoShowPlots, save_path, Detector_Panels, SiMirror, Slices, SectorCutAngles, AverageQRanges, FullPolGo, Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, PolCorrUU, PolCorrUU_Unc, PolCorrDU, PolCorrDU_Unc, PolCorrDD, PolCorrDD_Unc, PolCorrUD, PolCorrUD_Unc, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax)
                            
                            UScaledData, UScaledData_Unc = _abs_scale(Detector_Panels, Instrument, YesNoManualHe3Entry, input_path, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, HighResGain, 'U', Sample, Config, BB_per_second, Solid_Angle, Plex, ScattCatalog, TransCatalog)
                            DScaledData, DScaledData_Unc = _abs_scale(Detector_Panels, Instrument, YesNoManualHe3Entry, input_path, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, HighResGain, 'D', Sample, Config, BB_per_second, Solid_Angle, Plex, ScattCatalog, TransCatalog)
                            if 'NA' not in UScaledData and 'NA' not in DScaledData:
                                if YesNo_2DCombinedFiles:
                                    representative_filenumber = ScattCatalog[Sample]['Config(s)'][Config]['U'][0]
                                    Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, InPlaneAngleMap, dimXX, dimYY, Shadow_Mask = _q_calculation_all_detectors(Detector_Panels, Instrument, SampleApertureInMM,SampleDescriptionKeywordsToExclude, input_path, Calc_Q_From_Trans, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, representative_filenumber, Config, MiddlePixelBorderHorizontal, MiddlePixelBorderVertical, AlignDet_Trans)
                                    QValues_All = {'QX':Qx,'QY':Qy,'QZ':Qz,'Q_total':Q_total,'Q_perp_unc':Q_perp_unc,'Q_parl_unc':Q_parl_unc}
                                    _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'U', Sample, Config, UScaledData, UScaledData_Unc, QValues_All, Shadow_Mask)
                                    _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'D', Sample, Config, DScaledData, DScaledData_Unc, QValues_All, Shadow_Mask)
                                    _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'DMinusU', Sample, Config, DiffData, DiffData_Unc, QValues_All, Shadow_Mask)
                                    _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'DPlusU', Sample, Config, SumData, SumData_Unc, QValues_All, Shadow_Mask)
                                if str(ScattCatalog[Sample]['Intent']).find('Sample') != -1:
                                    SiMirror = ScattCatalog[Sample]['Config(s)'][Config]['SiMirror']
                                    HalfPolSampleSlices[Sample] = _sans_half_pol_slices(Detector_Panels, SiMirror, Slices, SectorCutAngles, AverageQRanges, 'HalfPol', Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, UScaledData, UScaledData_Unc, DScaledData, DScaledData_Unc)
                                if str(ScattCatalog[Sample]['Intent']).find('Empty') != -1:
                                    SiMirror = ScattCatalog[Sample]['Config(s)'][Config]['SiMirror']
                                    HalfPolSampleSlices['Empty'] = _sans_half_pol_slices(Detector_Panels, SiMirror, Slices, SectorCutAngles, AverageQRanges, 'HalfPol', Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, UScaledData, UScaledData_Unc, DScaledData, DScaledData_Unc)

                            UnpolScaledData, UnpolScaledData_Unc = _abs_scale(Detector_Panels, Instrument, YesNoManualHe3Entry, input_path, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, HighResGain, 'Unpol', Sample, Config, BB_per_second, Solid_Angle, Plex, ScattCatalog, TransCatalog)
                            if 'NA' not in UnpolScaledData:
                                if YesNo_2DCombinedFiles:
                                    representative_filenumber = ScattCatalog[Sample]['Config(s)'][Config]['Unpol'][0]
                                    Qx, Qy, Qz, Q_total, Q_perp_unc, Q_parl_unc, InPlaneAngleMap, dimXX, dimYY, Shadow_Mask = _q_calculation_all_detectors(Detector_Panels, Instrument, SampleApertureInMM,SampleDescriptionKeywordsToExclude, input_path, Calc_Q_From_Trans, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, ConvertHighResToSubset, representative_filenumber, Config, MiddlePixelBorderHorizontal, MiddlePixelBorderVertical, AlignDet_Trans)
                                    QValues_All = {'QX':Qx,'QY':Qy,'QZ':Qz,'Q_total':Q_total,'Q_perp_unc':Q_perp_unc,'Q_parl_unc':Q_parl_unc}
                                    _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, 'Unpol', Sample, Config, UnpolScaledData, UnpolScaledData_Unc, QValues_All, Shadow_Mask)
                                if str(ScattCatalog[Sample]['Intent']).find('Sample') != -1:
                                    SiMirror = ScattCatalog[Sample]['Config(s)'][Config]['SiMirror']
                                    UnpolSampleSlices[Sample] = _sans_unpol_slices(Detector_Panels, SiMirror, Slices, SectorCutAngles, AverageQRanges, 'Unpol', Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, UnpolScaledData, UnpolScaledData_Unc)
                                if str(ScattCatalog[Sample]['Intent']).find('Empty') != -1:
                                    SiMirror = ScattCatalog[Sample]['Config(s)'][Config]['SiMirror']
                                    UnpolSampleSlices['Empty'] = _sans_unpol_slices(Detector_Panels, SiMirror, Slices, SectorCutAngles, AverageQRanges, 'Unpol', Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, UnpolScaledData, UnpolScaledData_Unc)

                AllFullPolSlices[Config] = FullPolSampleSlices
                AllHalfPolSlices[Config] = HalfPolSampleSlices
                AllUnpolSlices[Config] = UnpolSampleSlices

    return AllFullPolSlices, AllHalfPolSlices, AllUnpolSlices


def _sans_full_pol_slices(YesNoShowPlots, save_path, Detector_Panels, SiMirror, Slices, SectorCutAngles, AverageQRanges, PolCorrDegree, Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, PolCorrUU, PolCorrUU_Unc, PolCorrDU, PolCorrDU_Unc, PolCorrDD, PolCorrDD_Unc, PolCorrUD, PolCorrUD_Unc, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax):
    """Compute 1-D slices of the four polarized cross-sections for one sample.

    Builds Circ/Horz/Vert/Diag sector masks, calls
    :func:`_two_dim_to_one_dim` on each polarized cross-section, then writes
    a combined ``SliceFullPol_*.txt`` file and a four-cross-section PNG per
    slice. The ``PolCorrDegree`` flag controls the file-name tag
    (``PolCorr`` / ``He3Corr`` / ``NotCorr``).

    Parameters
    ----------
    YesNoShowPlots : bool
        Required. Display generated plots in addition to saving them.
    save_path : str
        Required. Output directory.
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    SiMirror : str
        Required. Si-mirror state — passed to the sector-mask builder.
    Slices : Iterable[str]
        Required. Subset of ``{'Circ','Horz','Vert','Diag'}`` to compute.
    SectorCutAngles : float
        Required. Sector half-width (degrees).
    AverageQRanges : bool
        Required. If true, average overlapping carriage bins; auto-disabled
        when ``Config`` contains ``'CvB'``.
    PolCorrDegree : int
        Required. Correction tier: ``>=2`` PolCorr, ``1`` He3Corr, else uncorrected.
    Sample, Config : str
        Required. Sample and configuration labels (used for file naming).
    InPlaneAngleMap : dict[str, np.ndarray]
        Required. Per-panel azimuthal angle map.
    Q_min, Q_max : float
    Q_bins : int
        Required. Q binning grid.
    QValues_All : dict
        Required. Per-panel Q grids from :func:`_q_calculation_all_detectors`.
    Shadow_Mask : dict[str, np.ndarray]
        Required. Per-panel shadow masks.
    PolCorrUU, PolCorrUU_Unc, PolCorrDU, PolCorrDU_Unc, PolCorrDD, PolCorrDD_Unc, PolCorrUD, PolCorrUD_Unc : dict[str, np.ndarray]
        Required. Polarization-corrected intensities and uncertainties for
        the four cross-sections.
    YesNoSetPlotXRange, YesNoSetPlotYRange : bool
        Required. Toggle manual plot axis limits.
    PlotXmin, PlotXmax, PlotYmin, PlotYmax : float
        Required. Manual axis limits (used when toggled on).

    Returns
    -------
    ReturnSlices : dict
        Mapping ``slice_key -> {'PolType': str, 'UU','DU','DD','UD': dict}``
        where each cross-section dict is the output of
        :func:`_two_dim_to_one_dim`.
    """

    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')
        AverageQRanges = False

    Corr = "PolCorr"
    if PolCorrDegree >= 2:
        Corr = "PolCorr"
    elif PolCorrDegree >= 1:
        Corr = "He3Corr"
    else:
        Corr = "NotCorr"

    PlotYesNo = 0
    BothSides = 1
    HorzMask = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 0, SectorCutAngles, BothSides)
    VertMask = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 90, SectorCutAngles, BothSides)
    CircMask = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 0, 180, BothSides)
    DiagMaskA = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 45, SectorCutAngles, BothSides)
    DiagMaskB = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, -45, SectorCutAngles, BothSides)
    DiagMask = {}
    for dshort in relevant_detectors:
        DiagMask[dshort] = DiagMaskA[dshort] + DiagMaskB[dshort]

    ReturnSlices = {}
    
    for slices in Slices:
        if slices == "Circ":
            slice_key = "CircAve"
            local_mask = CircMask
        elif slices == "Vert":
            slice_key = "Vert"+str(SectorCutAngles)
            local_mask = VertMask
        elif slices == "Horz":
            slice_key = "Horz"+str(SectorCutAngles)
            local_mask = HorzMask
        elif slices == "Diag":
            slice_key = "Diag"+str(SectorCutAngles)
            local_mask = DiagMask

        UU = _two_dim_to_one_dim(Detector_Panels, slice_key, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, local_mask, PolCorrUU, PolCorrUU_Unc, Sample, Config, PlotYesNo, AverageQRanges)
        DU = _two_dim_to_one_dim(Detector_Panels, slice_key, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, local_mask, PolCorrDU, PolCorrDU_Unc, Sample, Config, PlotYesNo, AverageQRanges)
        DD = _two_dim_to_one_dim(Detector_Panels, slice_key, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, local_mask, PolCorrDD, PolCorrDD_Unc, Sample, Config, PlotYesNo, AverageQRanges)
        UD = _two_dim_to_one_dim(Detector_Panels, slice_key, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, local_mask, PolCorrUD, PolCorrUD_Unc, Sample, Config, PlotYesNo, AverageQRanges)

        file_name = 'SliceFullPol_{samp},{cf}_{corr}{slice_key}.txt'.format(samp=Sample, cf=Config, corr=Corr, slice_key=slice_key) 
        _save_text_data_four_cross_sections(save_path, file_name, slice_key, Sample, Config, UU, DU, DD, UD)
        '''saves data as SliceFullPol_{samp},{cf}_{corr}{slice_key}.txt'''

        _plot_four_cross_sections(save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, '{corr}'.format(corr = Corr), slice_key, Sample, Config, UU, DU, DD, UD)
        '''saves data as SliceFullPol_{samp},{cf}_{corr}{slice_key}.png'''

        ReturnSlices[slice_key] = {'PolType' : Corr, 'UU' : UU, 'DU' : DU, 'DD' : DD, 'UD' : UD}

    return ReturnSlices

def _sans_half_pol_slices(Detector_Panels, SiMirror, Slices, SectorCutAngles, AverageQRanges, PolType, Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, U, U_Unc, D, D_Unc):
    """Compute 1-D slices for half-polarized (U and D) scattering.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    SiMirror : str
        Required. Si-mirror state.
    Slices : Iterable[str]
        Required. Subset of ``{'Circ','Horz','Vert','Diag'}``.
    SectorCutAngles : float
        Required. Sector half-width (degrees).
    AverageQRanges : bool
        Required. Average overlapping carriage bins; auto-disabled when
        ``Config`` contains ``'CvB'``.
    PolType : str
        Required. Tag stored on each returned slice (e.g. ``'HalfPol'``).
    Sample, Config : str
        Required. Labels used for file naming.
    InPlaneAngleMap : dict[str, np.ndarray]
        Required. Per-panel azimuthal angle map.
    Q_min, Q_max : float
    Q_bins : int
        Required. Q binning grid.
    QValues_All : dict
        Required. Per-panel Q grids.
    Shadow_Mask : dict[str, np.ndarray]
        Required. Per-panel shadow masks.
    U, U_Unc, D, D_Unc : dict[str, np.ndarray]
        Required. Absolute-scaled U and D intensities with uncertainties.

    Returns
    -------
    ReturnSlices : dict
        ``slice_key -> {'PolType', 'U', 'D'}`` with each cut a dict from
        :func:`_two_dim_to_one_dim`.
    """

    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')
        AverageQRanges = False

    PlotYesNo = 0
    BothSides = 1
    HorzMask = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 0, SectorCutAngles, BothSides)
    VertMask = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 90, SectorCutAngles, BothSides)
    CircMask = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 0, 180, BothSides)
    DiagMaskA = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 45, SectorCutAngles, BothSides)
    DiagMaskB = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, -45, SectorCutAngles, BothSides)
    DiagMask = {}
    for dshort in relevant_detectors:
        DiagMask[dshort] = DiagMaskA[dshort] + DiagMaskB[dshort]

    ReturnSlices = {}
    
    for slices in Slices:
        if slices == "Circ":
            slice_key = "CircAve"
            local_mask = CircMask
        elif slices == "Vert":
            slice_key = "Vert"+str(SectorCutAngles)
            local_mask = VertMask
        elif slices == "Horz":
            slice_key = "Horz"+str(SectorCutAngles)
            local_mask = HorzMask
        elif slices == "Diag":
            slice_key = "Diag"+str(SectorCutAngles)
            local_mask = DiagMask

        UCut = _two_dim_to_one_dim(Detector_Panels, slice_key, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, local_mask, U, U_Unc, Sample, Config, PlotYesNo, AverageQRanges)
        DCut = _two_dim_to_one_dim(Detector_Panels, slice_key, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, local_mask, D, D_Unc, Sample, Config, PlotYesNo, AverageQRanges)

        ReturnSlices[slice_key] = {'PolType' : PolType, 'U' : UCut, 'D' : DCut}

    return ReturnSlices

def _sans_unpol_slices(Detector_Panels, SiMirror, Slices, SectorCutAngles, AverageQRanges, PolType, Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, Unpol, Unpol_Unc):
    """Compute 1-D slices for unpolarized scattering.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    SiMirror : str
        Required. Si-mirror state.
    Slices : Iterable[str]
        Required. Subset of ``{'Circ','Horz','Vert','Diag'}``.
    SectorCutAngles : float
        Required. Sector half-width (degrees).
    AverageQRanges : bool
        Required. Average overlapping carriage bins; auto-disabled when
        ``Config`` contains ``'CvB'``.
    PolType : str
        Required. Tag stored on each returned slice (e.g. ``'Unpol'``).
    Sample, Config : str
        Required. Labels.
    InPlaneAngleMap : dict[str, np.ndarray]
        Required. Per-panel azimuthal angle map.
    Q_min, Q_max : float
    Q_bins : int
        Required. Q binning grid.
    QValues_All : dict
        Required. Per-panel Q grids.
    Shadow_Mask : dict[str, np.ndarray]
        Required. Per-panel shadow masks.
    Unpol, Unpol_Unc : dict[str, np.ndarray]
        Required. Absolute-scaled unpol intensity and uncertainty.

    Returns
    -------
    ReturnSlices : dict
        ``slice_key -> {'PolType', 'Unpol'}``.
    """

    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')
        AverageQRanges = False

    PlotYesNo = 0
    BothSides = 1
    HorzMask = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 0, SectorCutAngles, BothSides)
    VertMask = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 90, SectorCutAngles, BothSides)
    CircMask = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 0, 180, BothSides)
    DiagMaskA = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, 45, SectorCutAngles, BothSides)
    DiagMaskB = _sector_mask_all_detectors(Detector_Panels, SiMirror, Config, InPlaneAngleMap, -45, SectorCutAngles, BothSides)
    DiagMask = {}
    for dshort in relevant_detectors:
        DiagMask[dshort] = DiagMaskA[dshort] + DiagMaskB[dshort]

    ReturnSlices = {}
    
    for slices in Slices:
        if slices == "Circ":
            slice_key = "CircAve"
            local_mask = CircMask
        elif slices == "Vert":
            slice_key = "Vert"+str(SectorCutAngles)
            local_mask = VertMask
        elif slices == "Horz":
            slice_key = "Horz"+str(SectorCutAngles)
            local_mask = HorzMask
        elif slices == "Diag":
            slice_key = "Diag"+str(SectorCutAngles)
            local_mask = DiagMask

        UnpolCut = _two_dim_to_one_dim(Detector_Panels, slice_key, Q_min, Q_max, Q_bins, QValues_All, Shadow_Mask, local_mask, Unpol, Unpol_Unc, Sample, Config, PlotYesNo, AverageQRanges)
        
        ReturnSlices[slice_key] = {'PolType' : PolType, 'Unpol' : UnpolCut}

    return ReturnSlices

def _sans_save_slices_and_results(StructurallyIsotropic, Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, UseMTCirc, He3Only_Check, Configs, Sample_Names, ScattCatalog, AllFullPolSlices, AllHalfPolSlices, AllUnpolSlices):
    """Per sample/config, finalize slices, do MT subtraction, save plots and data.

    Iterates over configurations and samples and calls the three processors
    :func:`_sans_process_full_pol_slices`, :func:`_sans_process_half_pol_slices`,
    and :func:`_sans_process_unpol_slices`. Empty samples are also processed
    when ``AutoSubtractEmpty`` is false.

    Parameters
    ----------
    StructurallyIsotropic : bool
        Required. Use the horizontal sum as the denominator in the magnetic
        decomposition when true; otherwise use the vertical sum.
    Slices : Iterable[str]
        Required. Slice keys.
    SectorCutAngles : float
        Required. Sector half-width (degrees).
    save_path : str
        Required. Output directory.
    YesNoShowPlots : bool
        Required. Display plots in addition to saving them.
    YesNoSetPlotXRange, YesNoSetPlotYRange : bool
        Required. Toggle manual plot axis limits.
    PlotXmin, PlotXmax, PlotYmin, PlotYmax : float
        Required. Manual axis limits.
    AutoSubtractEmpty : bool
        Required. Subtract the empty-cell scattering inside the processors.
    UseMTCirc : bool
        Required. Use the empty-cell circular slice for MT subtraction in
        Horz/Vert/Diag cuts.
    He3Only_Check : bool
        Required. If true, skip processing.
    Configs : dict[str, int]
        Required. Configuration -> representative file number map.
    Sample_Names : Iterable[str]
        Required.
    ScattCatalog : dict
        Required. Scattering catalog.
    AllFullPolSlices, AllHalfPolSlices, AllUnpolSlices : dict
        Required. Slice data from :func:`_sans_make_slices_and_save_ascii`.

    Returns
    -------
    AllFullPolResults : dict
    AllHalfPolResults : dict
    AllUnpolResults : dict
        Each keyed by ``Config -> Sample -> results dict``.
    """
    AllFullPolResults = {}
    AllHalfPolResults = {}
    AllUnpolResults = {}
    if not He3Only_Check:
        for Config in Configs:
            representative_filenumber = Configs[Config]
            if representative_filenumber != 0:
                FullPolResults = {}
                HalfPolResults = {}
                UnpolResults = {}
                for Sample in Sample_Names:
                    if Sample in ScattCatalog:                
                        if str(ScattCatalog[Sample]['Intent']).find('Sample') != -1:
                            if Sample in AllFullPolSlices[Config]:
                                FullPolResults[Sample] = _sans_process_full_pol_slices(StructurallyIsotropic, Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, Config, AllFullPolSlices[Config], Sample)
                            if Sample in AllHalfPolSlices[Config]:
                                HalfPolResults[Sample] = _sans_process_half_pol_slices(StructurallyIsotropic, Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, UseMTCirc, Config, AllHalfPolSlices[Config], Sample)
                            if Sample in AllUnpolSlices[Config]:
                                UnpolResults[Sample] = _sans_process_unpol_slices(Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, UseMTCirc, Config, AllUnpolSlices[Config], Sample)
                if not AutoSubtractEmpty:
                    if 'Empty' in AllFullPolSlices[Config]:
                        FullPolResults['Empty'] = _sans_process_full_pol_slices(StructurallyIsotropic, Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, Config, AllFullPolSlices[Config], 'Empty')
                    if 'Empty' in AllHalfPolSlices[Config]:
                        HalfPolResults['Empty'] = _sans_process_half_pol_slices(StructurallyIsotropic, Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, UseMTCirc, Config, AllHalfPolSlices[Config], 'Empty')
                    if 'Empty' in AllUnpolSlices[Config]:
                        UnpolResults['Empty'] = _sans_process_unpol_slices(Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, UseMTCirc, Config, AllUnpolSlices[Config], 'Empty')

                AllFullPolResults[Config] = FullPolResults
                AllHalfPolResults[Config] = HalfPolResults
                AllUnpolResults[Config] = UnpolResults
        
    return AllFullPolResults, AllHalfPolResults, AllUnpolResults

def _match_q_pa_data_sets(A, B, Type):
    """Trim two slice dicts so they share a common Q grid.

    Removes Q points present in one set but not the other, in place on the
    intensity / uncertainty / Q-metadata arrays appropriate to ``Type``.

    Parameters
    ----------
    A, B : dict
        Required. Slice dicts (must contain ``'Q'``, ``'Q_Mean'``, ``'Q_Unc'``,
        ``'Shadow'`` plus the intensity columns selected by ``Type``).
    Type : {0, 1, 2}
        Required. 0 = Unpol, 1 = Half Pol (U/D), 2 = Full Pol (UU/DU/DD/UD).

    Returns
    -------
    Horz_Data : dict
        ``A`` reduced to the shared Q grid.
    Vert_Data : dict
        ``B`` reduced to the shared Q grid.
    """

    Horz_Data = A
    Vert_Data = B
    
    for entry in Horz_Data['Q']:
        if entry not in Vert_Data['Q']:
            result = np.where(Horz_Data['Q'] == entry)
            Horz_Data['Q'] = np.delete(Horz_Data['Q'], result)
            Horz_Data['Q_Mean'] = np.delete(Horz_Data['Q_Mean'], result)
            Horz_Data['Q_Unc'] = np.delete(Horz_Data['Q_Unc'], result)
            Horz_Data['Shadow'] = np.delete(Horz_Data['Shadow'], result)
            if Type == 0:
                Horz_Data['Unpol'] = np.delete(Horz_Data['Unpol'], result)
                Horz_Data['Unpol_Unc'] = np.delete(Horz_Data['Unpol_Unc'], result)
            if Type == 1:
                Horz_Data['U'] = np.delete(Horz_Data['U'], result)
                Horz_Data['U_Unc'] = np.delete(Horz_Data['U_Unc'], result)
                Horz_Data['D'] = np.delete(Horz_Data['D'], result)
                Horz_Data['D_Unc'] = np.delete(Horz_Data['D_Unc'], result)
            if Type == 2:
                Horz_Data['UU'] = np.delete(Horz_Data['UU'], result)
                Horz_Data['UU_Unc'] = np.delete(Horz_Data['UU_Unc'], result)
                Horz_Data['DU'] = np.delete(Horz_Data['DU'], result)
                Horz_Data['DU_Unc'] = np.delete(Horz_Data['DU_Unc'], result)
                Horz_Data['DD'] = np.delete(Horz_Data['DD'], result)
                Horz_Data['DD_Unc'] = np.delete(Horz_Data['DD_Unc'], result)
                Horz_Data['UD'] = np.delete(Horz_Data['UD'], result)
                Horz_Data['UD_Unc'] = np.delete(Horz_Data['UD_Unc'], result)
    for entry in Vert_Data['Q']:
        if entry not in Horz_Data['Q']:
            result = np.where(Vert_Data['Q'] == entry)
            Vert_Data['Q'] = np.delete(Vert_Data['Q'], result)
            Vert_Data['Q_Mean'] = np.delete(Vert_Data['Q_Mean'], result)
            Vert_Data['Q_Unc'] = np.delete(Vert_Data['Q_Unc'], result)
            Vert_Data['Shadow'] = np.delete(Vert_Data['Shadow'], result)
            if Type == 0:
                Vert_Data['Unpol'] = np.delete(Vert_Data['Unpol'], result)
                Vert_Data['Unpol_Unc'] = np.delete(Vert_Data['Unpol_Unc'], result)
            if Type == 1:
                Vert_Data['U'] = np.delete(Vert_Data['U'], result)
                Vert_Data['U_Unc'] = np.delete(Vert_Data['U_Unc'], result)
                Vert_Data['D'] = np.delete(Vert_Data['D'], result)
                Vert_Data['D_Unc'] = np.delete(Vert_Data['D_Unc'], result)
            if Type == 2:
                Vert_Data['UU'] = np.delete(Vert_Data['UU'], result)
                Vert_Data['UU_Unc'] = np.delete(Vert_Data['UU_Unc'], result)
                Vert_Data['DU'] = np.delete(Vert_Data['DU'], result)
                Vert_Data['DU_Unc'] = np.delete(Vert_Data['DU_Unc'], result)
                Vert_Data['DD'] = np.delete(Vert_Data['DD'], result)
                Vert_Data['DD_Unc'] = np.delete(Vert_Data['DD_Unc'], result)
                Vert_Data['UD'] = np.delete(Vert_Data['UD'], result)
                Vert_Data['UD_Unc'] = np.delete(Vert_Data['UD_Unc'], result)

    return Horz_Data, Vert_Data

def _subtract_pa_data_sets(A, B, Type):
    """Subtract slice dict ``B`` from ``A`` and combine uncertainties in quadrature.

    Assumes ``A`` and ``B`` already share a common Q grid (typically after
    :func:`_match_q_pa_data_sets`).

    Parameters
    ----------
    A, B : dict
        Required. Matched slice dicts.
    Type : {0, 1, 2}
        Required. 0 = Unpol, 1 = Half Pol, 2 = Full Pol — selects which
        intensity channels to subtract.

    Returns
    -------
    C : dict
        ``A - B`` for the relevant intensity columns; Q metadata copied
        from ``A``.
    """

    C = {}
    C['Q'] = A['Q']
    C['Q_Mean'] = A['Q_Mean']
    C['Q_Unc'] = A['Q_Unc']
    C['Shadow'] = A['Shadow']
    if Type == 0:
        C['Unpol'] = A['Unpol'] - B['Unpol']
        C['Unpol_Unc'] = np.sqrt(np.power(A['Unpol_Unc'],2) + np.power(B['Unpol_Unc'],2))
    if Type == 1:
        C['U'] = A['U'] - B['U']
        C['U_Unc'] = np.sqrt(np.power(A['U_Unc'],2) + np.power(B['U_Unc'],2))
        C['D'] = A['D'] - B['D']
        C['D_Unc'] = np.sqrt(np.power(A['D_Unc'],2) + np.power(B['D_Unc'],2))
    if Type == 2:
        C['UU'] = A['UU'] - B['UU']
        C['UU_Unc'] = np.sqrt(np.power(A['UU_Unc'],2) + np.power(B['UU_Unc'],2))
        C['DD'] = A['DD'] - B['DD']
        C['DD_Unc'] = np.sqrt(np.power(A['DD_Unc'],2) + np.power(B['DD_Unc'],2))
        C['UD'] = A['UD'] - B['UD']
        C['UD_Unc'] = np.sqrt(np.power(A['UD_Unc'],2) + np.power(B['UD_Unc'],2))
        C['DU'] = A['DU'] - B['DU']
        C['DU_Unc'] = np.sqrt(np.power(A['DU_Unc'],2) + np.power(B['DU_Unc'],2))
    
    return C

def _two_dim_to_one_dim(Detector_Panels, Key, Q_min, Q_max, Q_bins, QGridPerDetector, generalmask, sectormask, PolCorr_AllDetectors, Unc_PolCorr_AllDetectors, ID, Config, PlotYesNo, AverageQRanges):
    """Radially bin masked 2-D intensity into a 1-D Q profile.

    For each panel the intensity inside ``generalmask * sectormask`` is
    histogrammed on a uniform ``[Q_min, Q_max]`` grid with ``Q_bins`` bins.
    Front/middle/back carriages are accumulated separately, then either
    trimmed of overlap (``AverageQRanges=False``) or averaged
    (``AverageQRanges=True``).

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Key : str
        Required. Slice tag (e.g. ``'CircAve'``) used in titles and file
        names when plotting.
    Q_min, Q_max : float
    Q_bins : int
        Required. Q-binning grid.
    QGridPerDetector : dict
        Required. Per-panel ``Q_total`` / ``Q_parl_unc`` maps (output of
        :func:`_q_calculation_all_detectors`).
    generalmask : dict[str, np.ndarray]
        Required. Per-panel general mask (typically the shadow mask).
    sectormask : dict[str, np.ndarray]
        Required. Per-panel sector mask from
        :func:`_sector_mask_all_detectors`.
    PolCorr_AllDetectors : dict[str, np.ndarray]
        Required. Per-panel intensities to bin.
    Unc_PolCorr_AllDetectors : dict[str, np.ndarray]
        Required. Matching per-pixel uncertainties.
    ID, Config : str
        Required. Used in plot titles and file names.
    PlotYesNo : bool or int
        Required. Render and save a per-call diagnostic plot when nonzero.
    AverageQRanges : bool
        Required. Average overlapping carriage bins when true.

    Returns
    -------
    Output : dict
        Keys ``'Q'``, ``'Shadow'``, ``'I'``, ``'I_Unc'``, ``'Q_Mean'``,
        ``'MeanQ_Unc'``, ``'Pixels'``, ``'Q_Uncertainty'``.
    """

    Q_step = (Q_max - Q_min) / Q_bins
    Q_Values = np.linspace(Q_min, Q_max, Q_bins, endpoint=True) + Q_step/2

    masks = {}
    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')
    for dshort in relevant_detectors:
        masks[dshort] = generalmask[dshort]*sectormask[dshort]

    Histograms = {} # store results by carriage_key
    zeros_like_Q = np.zeros_like(Q_Values)
    carriage_keys = ["F", "M", "B"]
    result_keys = ["I", "I_Unc", "Q_Mean", "MeanQ_Unc", "Pixels", "Sigma_UU"]
    for carriage_key in carriage_keys:
        Histograms[carriage_key] = {}
        for k in result_keys:
            Histograms[carriage_key][k] = zeros_like_Q.copy()

    Q_lookups = {}
    Exp_bins = np.linspace(Q_min, Q_max + Q_step, Q_bins + 1, endpoint=True)
    for dshort in relevant_detectors:
        carriage_key = dshort[0]
        CurrentHistogram = Histograms[carriage_key]
        Q_tot = QGridPerDetector['Q_total'][dshort][:][:]
        UU = PolCorr_AllDetectors[dshort][:][:]
        UU_Unc = Unc_PolCorr_AllDetectors[dshort][:][:]

        Q_lookup = np.searchsorted(Exp_bins, Q_tot[masks[dshort] > 0], side="right") - 1
        Q_lookups[dshort] = Q_lookup
        countsUU, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=UU[masks[dshort] > 0])
        UncUU, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.power(UU_Unc[masks[dshort] > 0],2))        
        MeanQSum, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=Q_tot[masks[dshort] > 0])
        pixels, _ = np.histogram(Q_tot[masks[dshort] > 0], bins=Exp_bins, weights=np.ones_like(UU)[masks[dshort] > 0])

        CurrentHistogram["I"] += countsUU
        CurrentHistogram["I_Unc"] += UncUU
        CurrentHistogram["Q_Mean"] += MeanQSum
        CurrentHistogram["Pixels"] += pixels

    CombinedPixels = sum(Histograms[k]["Pixels"] for k in carriage_keys)
    nonzero_combined_mask = (CombinedPixels > 0) #True False map

    for carriage_key in carriage_keys:
        CurrentHistogram = Histograms[carriage_key]
        nonzero_mask = CurrentHistogram["Pixels"] > 0
        CurrentHistogram["nonzero_mask"] = nonzero_mask
        CurrentHistogram["Q"] = Q_Values[nonzero_mask]
        CurrentHistogram["Sigma_UU"][nonzero_mask] = np.sqrt(CurrentHistogram["I_Unc"][nonzero_mask]) / CurrentHistogram["Pixels"][nonzero_mask]

    # now that we have MeanQ, we can calculate sigmaQ statistically:
    for dshort in relevant_detectors:
        carriage_key = dshort[0]
        CurrentHistogram = Histograms[carriage_key]
        nonzero_mask = CurrentHistogram["nonzero_mask"]
        # Only the projection of dQ(single_pixel) parallel to Q is important in this calculation:
        Q_unc = (QGridPerDetector['Q_parl_unc'][dshort][:][:])[masks[dshort] > 0]
        # This is the Q value for the pixel
        Q_tot = (QGridPerDetector['Q_total'][dshort][:][:])[masks[dshort] > 0]
        # Get the lookup table for which bin we're in:
        Q_lookup = Q_lookups[dshort]
        Q_lookup_mask = np.logical_and((Q_lookup < Q_bins), (Q_lookup >= 0))
        # Get the MeanQ for that bin:
        MeanQ = CurrentHistogram["Q_Mean"].copy()
        MeanQ[nonzero_mask] /= CurrentHistogram["Pixels"][nonzero_mask]
        Q_mean_center = MeanQ[Q_lookup[Q_lookup_mask]]
        Q_pixel_center = Q_tot[Q_lookup_mask]
        Q_pixel_uncertainty = Q_unc[Q_lookup_mask]
        Q_var_contrib = (Q_mean_center - Q_pixel_center)**2 + (Q_unc[Q_lookup_mask])**2 
        Q_var, _ = np.histogram(Q_tot[Q_lookup_mask], bins=Exp_bins, weights=Q_var_contrib)
        CurrentHistogram["MeanQ_Unc"][nonzero_mask] += Q_var[nonzero_mask]

    ErrorBarsYesNo = 0
    if PlotYesNo == 1:
        fig = plt.figure()
        if ErrorBarsYesNo == 1:
            ax = plt.axes()
            ax.set_xscale("log")
            ax.set_yscale("log")
            hist = Histograms["F"]
            ax.errorbar(hist["Q"], hist["I"][hist["nonzero_mask"]], yerr=hist["Sigma_UU"][hist["nonzero_mask"]], fmt = 'b*', label='Front')
            hist = Histograms["M"]
            ax.errorbar(hist["Q"], hist["I"][hist["nonzero_mask"]], yerr=hist["Sigma_UU"][hist["nonzero_mask"]], fmt = 'g*', label='Middle')
            if str(Config).find('CvB') != -1:
                hist = Histograms["B"]
                ax.errorbar(hist["Q"], hist["I"][hist["nonzero_mask"]], yerr=hist["Sigma_UU"][hist["nonzero_mask"]], fmt = 'r*', label='HighRes')
        else:
            hist = Histograms["F"]
            plt.loglog(hist["Q"], hist["I"][hist["nonzero_mask"]], 'b*', label='Front')
            hist = Histograms["M"]
            plt.loglog(hist["Q"], hist["I"][hist["nonzero_mask"]], 'g*', label='Middle')
            if str(Config).find('CvB') != -1:
                hist = Histograms["B"]
                plt.loglog(hist["Q"], hist["I"][hist["nonzero_mask"]], 'r*', label='High Res')
                
        plt.xlabel('Q')
        plt.ylabel('Intensity')
        plt.title('{keyword}_{idnum},{cf}'.format(keyword=Key, idnum=ID, cf = Config))
        plt.legend()
        fig.savefig('{keyword}_{idnum},CF{cf}.png'.format(keyword=Key, idnum=ID, cf = Config))
        #plt.show()
        
    Q_Common = Q_Values[nonzero_combined_mask]
    Output = {        
        "Q": Q_Values,
        "Shadow": np.ones_like(Q_Values),
    }
    if not AverageQRanges:
        '''Remove points overlapping in Q space before joining'''
        final_masks = {
            "B": np.logical_and(Histograms["B"]["nonzero_mask"], np.logical_not(Histograms["M"]["nonzero_mask"])),
            "M": np.logical_and(Histograms["M"]["nonzero_mask"], np.logical_not(Histograms["F"]["nonzero_mask"])),
            "F": Histograms["F"]["nonzero_mask"]
        }
        # overlaps = sum([final_masks["B"].astype("float"), final_masks["M"].astype("float"), final_masks["F"].astype("float")])
        
        for k in ["I", "I_Unc", "Q_Mean", "MeanQ_Unc", "Pixels"]:
            Output[k] = zeros_like_Q.copy()
            for carriage_key in carriage_keys:
                mask = final_masks[carriage_key]
                Output[k][mask] += Histograms[carriage_key][k][mask] # / overlaps[mask]

        Output["I_Unc"] = np.sqrt(Output["I_Unc"])
        Output["I_Unc"][nonzero_combined_mask] /= Output["Pixels"][nonzero_combined_mask]
        Output["I"][nonzero_combined_mask] /= Output["Pixels"][nonzero_combined_mask]
        Output["Q_Mean"][nonzero_combined_mask] /= Output["Pixels"][nonzero_combined_mask]

        # This is correct: with no overlap, there is no averaging of uncertainties  
        Output["Q_Uncertainty"] = np.sqrt(Output["MeanQ_Unc"])
        Output["Q_Uncertainty"][nonzero_combined_mask] /= Output["Pixels"][nonzero_combined_mask]

        # e.g.:        
        # Q_Mean = zeros_like_Q.copy()
        # Q_Mean[back_mask] = Histograms["B"]["Q_Mean"][back_mask]
        # Q_Mean[middle_mask] = Histograms["M"]["Q_Mean"][middle_mask]
        # Q_Mean[front_mask] = Histograms["F"]["Q_Mean"][front_mask]
    else:
        # add all points for all carriages:
        final_masks = {
            "B": nonzero_combined_mask,
            "M": nonzero_combined_mask,
            "F": nonzero_combined_mask
        }
        # overlaps = sum([final_masks["B"].astype("float"), final_masks["M"].astype("float"), final_masks["F"].astype("float")])

        for k in ["I", "I_Unc", "Q_Mean", "MeanQ_Unc", "Pixels"]:
            Output[k] = zeros_like_Q.copy()
            for carriage_key in carriage_keys:
                mask = final_masks[carriage_key]
                Output[k][mask] += Histograms[carriage_key][k][mask] # / overlaps[mask]

        Output["I_Unc"] = np.sqrt(Output["I_Unc"])
        Output["I_Unc"][nonzero_combined_mask] /= Output["Pixels"][nonzero_combined_mask]
        Output["I"][nonzero_combined_mask] /= Output["Pixels"][nonzero_combined_mask]
        Output["Q_Mean"][nonzero_combined_mask] /= Output["Pixels"][nonzero_combined_mask]
  
        # Q_uncertainty is not quite the average of the Q_uncertaintes from all carriages(it should be weighted).
        Output["Q_Uncertainty"] = np.sqrt(Output["MeanQ_Unc"])
        Output["Q_Uncertainty"][nonzero_combined_mask] /= CombinedPixels[nonzero_combined_mask]
    
    return Output

def _ascii_like_output(Detector_Panels, save_path, YesNo_2DFilesPerDetector, Type, ID, Config, Data_AllDetectors, Unc_Data_AllDetectors, QGridPerDetector, GeneralMask):
    """Write 2-D pixel-level (Qx, Qy, I, ...) ASCII files in SasView format.

    For each panel, flattens the masked (Qx, Qy, Qz, I, dI, dQparl, dQperp,
    shadow) arrays. Optionally writes one file per panel (when
    ``YesNo_2DFilesPerDetector`` is true) and always writes a combined
    file that concatenates all panels.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    save_path : str
        Required. Output directory.
    YesNo_2DFilesPerDetector : bool
        Required. If true, also emit one ``.DAT`` per panel.
    Type : str
        Required. File-name tag (e.g. ``'PolCorrUU'``, ``'NotCorrDD'``).
    ID : str
        Required. Sample identifier in the file name.
    Config : str
        Required. Configuration label.
    Data_AllDetectors : dict[str, np.ndarray] or 'NA'
        Required. Per-panel intensities; nothing is written if either
        input contains ``'NA'``.
    Unc_Data_AllDetectors : dict[str, np.ndarray] or 'NA'
        Required. Matching per-panel uncertainties.
    QGridPerDetector : dict
        Required. Per-panel ``QX/QY/QZ/Q_perp_unc/Q_parl_unc/Q_total`` arrays.
    GeneralMask : dict[str, np.ndarray]
        Required. Per-panel mask selecting pixels to export.

    Returns
    -------
    None
    """

    relevant_detectors = list(Detector_Panels)
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')

    if 'NA' not in Data_AllDetectors and 'NA' not in Unc_Data_AllDetectors:

        for dshort in relevant_detectors:

            Mask = np.array(GeneralMask[dshort])
            mini_mask = Mask > 0

            Q_tot = QGridPerDetector['Q_total'][dshort][:][:]
            Q_unc = np.sqrt(np.power(QGridPerDetector['Q_perp_unc'][dshort][:][:],2) + np.power(QGridPerDetector['Q_parl_unc'][dshort][:][:],2))

            QQX = QGridPerDetector['QX'][dshort][:][:]
            QQX = QQX[mini_mask,...]
            QQX = QQX.T
            QXData = QQX.flatten()
            QQY = QGridPerDetector['QY'][dshort][:][:]
            QQY = QQY[mini_mask,...]
            QQY = QQY.T
            QYData = QQY.flatten()
            QQZ = QGridPerDetector['QZ'][dshort][:][:]
            QQZ = QQZ[mini_mask,...]
            QQZ = QQZ.T
            QZData = QQZ.flatten()
            QPP = QGridPerDetector['Q_perp_unc'][dshort][:][:]
            QPP = QPP[mini_mask,...]
            QPP = QPP.T
            QPerpUnc = QPP.flatten()
            QPR = QGridPerDetector['Q_parl_unc'][dshort][:][:]
            QPR = QPR[mini_mask,...]
            QPR = QPR.T
            QParlUnc = QPR.flatten()
            Shadow = np.ones_like(Q_tot)
            Shadow = Shadow[mini_mask,...]
            Shadow = Shadow.T
            ShadowHolder = Shadow.flatten()

            Intensity = Data_AllDetectors[dshort]
            Intensity = Intensity[mini_mask,...]
            Intensity = Intensity.T
            Int = Intensity.flatten()
            Intensity = Intensity.flatten()
            IntensityUnc = Unc_Data_AllDetectors[dshort]
            IntensityUnc = IntensityUnc[mini_mask,...]
            IntensityUnc = IntensityUnc.T
            DeltaInt = IntensityUnc.flatten()
            IntensityUnc = IntensityUnc.flatten()
            if YesNo_2DFilesPerDetector:
                ASCII_like = np.array([QXData, QYData, Int, DeltaInt, QZData, QParlUnc, QPerpUnc, ShadowHolder])
                ASCII_like = ASCII_like.T
                file_name = '{TP}Scatt_{Samp}_{CF}_{det}.DAT'.format(TP=Type, Samp=ID, CF=Config, det=dshort)
                file_path = os.path.join(save_path, file_name) 
                np.savetxt(file_path, ASCII_like, delimiter = ' ', comments = '', header = 'ASCII data created Mon, Jan 13, 2020 2:39:54 PM')
           

            if dshort == relevant_detectors[0]:
                Int_Combined = Intensity
                DeltaInt_Combined = IntensityUnc
                QXData_Combined = QXData
                QYData_Combined = QYData
                QZData_Combined = QZData
                QPP_Combined = QPP
                QPerpUnc_Combined = QPerpUnc
                QPR_Combined = QPR
                QParlUnc_Combined = QParlUnc
                Shadow_Combined = ShadowHolder
            else:
                Int_Combined = np.concatenate((Int_Combined, Intensity), axis=0)
                DeltaInt_Combined = np.concatenate((DeltaInt_Combined, IntensityUnc), axis=0)
                QXData_Combined = np.concatenate((QXData_Combined, QXData), axis=0)
                QYData_Combined = np.concatenate((QYData_Combined, QYData), axis=0)
                QZData_Combined = np.concatenate((QZData_Combined, QZData), axis=0)
                QPP_Combined = np.concatenate((QPP_Combined, QPP), axis=0)
                QPerpUnc_Combined = np.concatenate((QPerpUnc_Combined, QPerpUnc), axis=0)
                QPR_Combined = np.concatenate((QPR_Combined, QPR), axis=0)
                QParlUnc_Combined = np.concatenate((QParlUnc_Combined, QParlUnc), axis=0)
                Shadow_Combined = np.concatenate((Shadow_Combined, ShadowHolder), axis=0)
   
        ASCII_Combined = np.array([QXData_Combined, QYData_Combined, Int_Combined, DeltaInt_Combined, QZData_Combined, QParlUnc_Combined, QPerpUnc_Combined, Shadow_Combined])
        ASCII_Combined = ASCII_Combined.T
        file_name = '{TP}Scatt_{Samp}_{CF}.DAT'.format(TP=Type, Samp=ID, CF=Config)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, ASCII_Combined, delimiter = ' ', comments = '', header = 'ASCII data created Mon, Jan 13, 2020 2:39:54 PM')

    return

def _save_text_data(save_path, Type, Slice, Sample, Config, DataMatrix):
    """Write a single-cross-section 1-D slice to ``Slice{Type}_*.txt``.

    Parameters
    ----------
    save_path : str
        Required. Output directory.
    Type : str
        Required. File-name tag (e.g. ``'Unpol'``, ``'PolCorrUU'``).
    Slice : str
        Required. Slice label (``'CircAve'``, ``'Horz15'``, ...).
    Sample, Config : str
        Required. Labels used in the file name.
    DataMatrix : dict
        Required. Output of :func:`_two_dim_to_one_dim` with keys ``'Q'``,
        ``'I'``, ``'I_Unc'``, ``'Q_Mean'``, ``'Q_Uncertainty'``.

    Returns
    -------
    None
    """

    Q = DataMatrix['Q']
    Int = DataMatrix['I']
    IntUnc = DataMatrix['I_Unc']
    Q_mean = DataMatrix['Q_Mean']
    Q_Unc = DataMatrix['Q_Uncertainty']
    Shadow = np.ones_like(Q)
    text_output = np.array([Q, Int, IntUnc, Q_Unc, Q_mean, Shadow])
    #text_output = np.array([Q, Int, IntUnc, Q_mean, Q_Unc, Shadow])
    text_output = text_output.T
    file_name = 'Slice{key}_{samp},{cf}_{cut}.txt'.format(key=Type, samp=Sample, cf = Config, cut = Slice)
    file_path = os.path.join(save_path, file_name)
    np.savetxt(file_path, text_output, delimiter = ' ', comments = '', header= 'Q, I, DelI, Q_Unc, Q_mean, Shadow', fmt='%1.4e')
  
    return

def _save_text_data_unpol(save_path, Sub, Slice, Sample, Config, DataMatrix):
    """Write an unpolarized 1-D slice to ``SliceUnpol_*.txt``.

    Parameters
    ----------
    save_path : str
        Required. Output directory.
    Sub : str
        Required. Empty-subtraction tag appended to the file name (e.g.
        ``''`` or ``',SubMT'``).
    Slice : str
        Required. Slice label.
    Sample, Config : str
        Required. Labels used in the file name.
    DataMatrix : dict
        Required. Unpol slice dict with keys ``'Q'``, ``'Unpol'``,
        ``'Unpol_Unc'``, ``'Q_Mean'``, ``'Q_Unc'``, ``'Shadow'``.

    Returns
    -------
    None
    """

    Q = DataMatrix['Q']
    Int = DataMatrix['Unpol']
    IntUnc = DataMatrix['Unpol_Unc']
    Q_mean = DataMatrix['Q_Mean']
    Q_Unc = DataMatrix['Q_Unc']
    Shadow = DataMatrix['Shadow']
    text_output = np.array([Q, Int, IntUnc, Q_Unc, Q_mean, Shadow])
    #text_output = np.array([Q, Int, IntUnc, Q_mean, Q_Unc, Shadow])
    text_output = text_output.T
    file_name = 'SliceUnpol_{samp},{cf}_{cut}{key}.txt'.format(samp=Sample, cf = Config, cut = Slice, key = Sub)
    file_path = os.path.join(save_path, file_name)
    np.savetxt(file_path, text_output, delimiter = ' ', comments = '', header= 'Q, I, DelI, Q_Unc, Q_mean, Shadow', fmt='%1.4e')
  
    return


def _save_text_data_four_cross_sections(save_path, Type, Slice, Sample, Config, UUMatrix, DUMatrix, DDMatrix, UDMatrix):
    """Write the four full-pol cross-sections to a single ``SliceFullPol_*.txt``.

    Parameters
    ----------
    save_path : str
        Required. Output directory.
    Type : str
        Required. File-name tag (e.g. ``'PolCorr'``, ``'He3Corr'``,
        ``'NotCorr'``).
    Slice : str
        Required. Slice label.
    Sample, Config : str
        Required. Labels used in the file name.
    UUMatrix, DUMatrix, DDMatrix, UDMatrix : dict
        Required. Per-cross-section output of :func:`_two_dim_to_one_dim`.
        ``UUMatrix`` supplies the shared Q grid and Q metadata.

    Returns
    -------
    None
    """

    Q = UUMatrix['Q']
    UU = UUMatrix['I']
    UU_Unc = UUMatrix['I_Unc']
    DU = DUMatrix['I']
    DU_Unc = DUMatrix['I_Unc']
    DD = DDMatrix['I']
    DD_Unc = DDMatrix['I_Unc']
    UD = UDMatrix['I']
    UD_Unc = UDMatrix['I_Unc']
    Q_mean = UUMatrix['Q_Mean']
    Q_Unc = UUMatrix['Q_Uncertainty']
    Shadow = np.ones_like(Q)
    text_output = np.array([Q, UU, UU_Unc, DU, DU_Unc, DD, DD_Unc, UD, UD_Unc, Q_Unc, Q_mean, Shadow])
    text_output = text_output.T
    file_name = 'SliceFullPol_{samp},{cf}_{key}{cut}.txt'.format(samp=Sample, cf = Config, key = Type, cut = Slice)
    file_path = os.path.join(save_path, file_name)
    np.savetxt(file_path, text_output,
               delimiter = ' ', comments = '', header= 'Q, UU, DelUU, DU, DelDU, DD, DelDD, UD, DelUD, Q_Unc, Q_mean, Shadow', fmt='%1.4e')
  
    return

def _save_text_data_four_combined_cross_sections(save_path,  Type, Slice, Sub, Sample, Config, Matrix):
    """Write four already-combined cross-sections to ``SliceFullPol_*.txt``.

    Unlike :func:`_save_text_data_four_cross_sections` this expects a single
    matched dict (e.g. after MT subtraction) holding all four cross-sections.

    Parameters
    ----------
    save_path : str
        Required. Output directory.
    Type : str
        Required. File-name tag.
    Slice : str
        Required. Slice label.
    Sub : str
        Required. Empty-subtraction tag (``''`` or ``',SubMT'``).
    Sample, Config : str
        Required. Labels.
    Matrix : dict
        Required. Must contain ``'Q'``, ``'UU'``, ``'UU_Unc'``, ``'DU'``,
        ``'DU_Unc'``, ``'DD'``, ``'DD_Unc'``, ``'UD'``, ``'UD_Unc'``,
        ``'Q_Mean'``, ``'Q_Unc'``, ``'Shadow'``.

    Returns
    -------
    None
    """

    Q = Matrix['Q']
    UU = Matrix['UU']
    UU_Unc = Matrix['UU_Unc']
    DU = Matrix['DU']
    DU_Unc = Matrix['DU_Unc']
    DD = Matrix['DD']
    DD_Unc = Matrix['DD_Unc']
    UD = Matrix['UD']
    UD_Unc = Matrix['UD_Unc']
    Q_mean = Matrix['Q_Mean']
    Q_Unc = Matrix['Q_Unc']
    Shadow = Matrix['Shadow']
    text_output = np.array([Q, UU, UU_Unc, DU, DU_Unc, DD, DD_Unc, UD, UD_Unc, Q_Unc, Q_mean, Shadow])
    text_output = text_output.T
    file_name = 'SliceFullPol_{samp},{cf}_{key}{cut}{sub}.txt'.format(samp=Sample, cf = Config, key = Type, cut = Slice, sub = Sub)
    file_path = os.path.join(save_path, file_name)
    np.savetxt(file_path, text_output,
               delimiter = ' ', comments = '', header= 'Q, UU, DelUU, DU, DelDU, DD, DelDD, UD, DelUD, Q_Unc, Q_mean, Shadow', fmt='%1.4e')
  
    return

def _plot_four_cross_sections(save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, Type, Slice, Sample, Config, UU, DU, DD, UD):
    """Save a log-log plot of the four full-pol cross-sections.

    Parameters
    ----------
    save_path : str
        Required. Output directory.
    YesNoShowPlots : bool
        Required. Display the plot in addition to saving it.
    YesNoSetPlotXRange, YesNoSetPlotYRange : bool
        Required. Toggle manual axis limits.
    PlotXmin, PlotXmax, PlotYmin, PlotYmax : float
        Required. Manual axis limits (used only when toggled on).
    Type : str
        Required. File-name tag for the saved PNG.
    Slice : str
        Required. Slice label.
    Sample, Config : str
        Required. Labels.
    UU, DU, DD, UD : dict
        Required. Per-cross-section slice dicts with ``'Q'``, ``'I'``,
        ``'I_Unc'`` keys.

    Returns
    -------
    None
    """

    fig = plt.figure()
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    if YesNoSetPlotYRange:
        ax.set_ylim(bottom = PlotYmin, top = PlotYmax)
    if YesNoSetPlotXRange:
        ax.set_xlim(left = PlotXmin, right = PlotXmax)
    ax.errorbar(UU['Q'], UU['I'], yerr=UU['I_Unc'], fmt = 'b*', label='UU')
    ax.errorbar(DU['Q'], DU['I'], yerr=DU['I_Unc'], fmt = 'g*', label='DU')
    ax.errorbar(DD['Q'], DD['I'], yerr=DD['I_Unc'], fmt = 'r*', label='DD')
    ax.errorbar(UD['Q'], UD['I'], yerr=UD['I_Unc'], fmt = 'm*', label='UD')
    plt.xlabel('Q (inverse angstroms)')
    plt.ylabel('Intensity')
    plt.title('{slice_type}_{idnum},{cf}'.format(slice_type = Slice, idnum=Sample, cf = Config))
    plt.legend()
    file_name = 'SliceFullPol_{samp},{cf}_{corr}{slice_type}.png'.format(samp=Sample, cf = Config, corr = Type, slice_type = Slice)
    file_path = os.path.join(save_path, file_name)
    fig.savefig(file_path)
    if YesNoShowPlots:
        plt.show()
    plt.close()

    return

def _plot_four_combined_cross_sections(save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, Type, Slice, Sub, Sample, Config, Matrix):
    """Save a log-log plot of four cross-sections stored in a single matched dict.

    Counterpart of :func:`_plot_four_cross_sections` for a dict already
    holding ``UU``/``DU``/``DD``/``UD`` columns (typically after MT
    subtraction).

    Parameters
    ----------
    save_path : str
        Required. Output directory.
    YesNoShowPlots : bool
        Required. Display the plot in addition to saving it.
    YesNoSetPlotXRange, YesNoSetPlotYRange : bool
        Required. Toggle manual axis limits.
    PlotXmin, PlotXmax, PlotYmin, PlotYmax : float
        Required. Manual axis limits.
    Type : str
        Required. File-name tag.
    Slice : str
        Required. Slice label.
    Sub : str
        Required. Empty-subtraction tag (``''`` or ``',SubMT'``).
    Sample, Config : str
        Required. Labels.
    Matrix : dict
        Required. Combined cross-section dict with ``'Q'``, ``'UU'``,
        ``'UU_Unc'``, ``'DU'``, ``'DU_Unc'``, ``'DD'``, ``'DD_Unc'``,
        ``'UD'``, ``'UD_Unc'``.

    Returns
    -------
    None
    """

    fig = plt.figure()
    ax = plt.axes()
    ax.set_xscale("log")
    ax.set_yscale("log")
    if YesNoSetPlotYRange:
        ax.set_ylim(bottom = PlotYmin, top = PlotYmax)
    if YesNoSetPlotXRange:
        ax.set_xlim(left = PlotXmin, right = PlotXmax)
    ax.errorbar(Matrix['Q'], Matrix['UU'], yerr=Matrix['UU_Unc'], fmt = 'b*', label='UU')
    ax.errorbar(Matrix['Q'], Matrix['DU'], yerr=Matrix['DU_Unc'], fmt = 'g*', label='DU')
    ax.errorbar(Matrix['Q'], Matrix['DD'], yerr=Matrix['DD_Unc'], fmt = 'r*', label='DD')
    ax.errorbar(Matrix['Q'], Matrix['UD'], yerr=Matrix['UD_Unc'], fmt = 'm*', label='UD')
    plt.xlabel('Q (inverse angstroms)')
    plt.ylabel('Intensity')
    plt.title('{slice_type}_{idnum},{cf}'.format(slice_type = Slice, idnum=Sample, cf = Config))
    plt.legend()
    file_name = 'SliceFullPol_{samp},{cf}_{corr}{slice_type}{sub}.png'.format(samp=Sample, cf = Config, corr = Type, slice_type = Slice, sub = Sub)
    file_path = os.path.join(save_path, file_name) 
    fig.savefig(file_path)
    if YesNoShowPlots:
        plt.show()
    plt.close()

    return

def _get_beam_center(Instrument, input_path, filenumber, dshort, trans_max_width_pixels):
    """Find the beam-center coordinates from a transmission run.

    For NG7SANS, returns the centers stored in the NeXus file directly.
    For VSANS, computes the center of mass of the requested panel, refines
    it on a small subregion (clamped to ``trans_max_width_pixels`` away
    from each edge), and converts pixel coordinates to centimeters using
    the panel's spatial calibration.

    Parameters
    ----------
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    input_path : str
        Required. Directory containing raw NeXus files.
    filenumber : int
        Required. Transmission run number to read.
    dshort : str
        Required. Short panel name on which to find the center (VSANS).
    trans_max_width_pixels : int
        Required. Maximum half-width (pixels) of the subregion used for
        the center-of-mass refinement.

    Returns
    -------
    middle_bc_x : float
        Beam-center X in cm (VSANS) or pixels (NG7SANS).
    middle_bc_y : float
        Beam-center Y in cm (VSANS) or pixels (NG7SANS).
    """


    f = _sans_get_by_filenumber(Instrument, input_path, filenumber)
    middle_bc_x = 0
    middle_bc_y = 0
    if 'NG7SANS' in Instrument: #Will add functionality to NG7SANS
        beam_center_x = f['entry/instrument/detector/beam_center_x'][0] #NG7 Change
        beam_center_y = f['entry/instrument/detector/beam_center_y'][0]
        middle_bc_x = beam_center_x
        middle_bc_y = beam_center_y
    elif 'VSANS' in Instrument:
        data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
        beam_center_x = f['entry/instrument/detector_{ds}/beam_center_x'.format(ds=dshort)][0]
        beam_center_y = f['entry/instrument/detector_{ds}/beam_center_y'.format(ds=dshort)][0]
        x_width, y_width = np.shape(data)
        x_cen, y_cen = ndimage.measurements.center_of_mass(data)
        lateral_width_left = int(x_cen) - 0
        lateral_width_right = int(x_width) - int(x_cen) - 1
        lateral_width_possible = np.minimum(lateral_width_left, lateral_width_right)
        lateral_width = np.minimum(lateral_width_possible, trans_max_width_pixels)
        
        vertical_width_bottom = int(y_cen) - 0
        vertical_width_top = int(y_width) - int(y_cen) - 1
        vertical_width_possible = np.minimum(vertical_width_bottom, vertical_width_top)
        vertical_width = np.minimum(vertical_width_possible, trans_max_width_pixels)

        y_int = int(y_cen)
        x_min = int(x_cen) - lateral_width
        x_max = int(x_cen) + lateral_width
        y_min = int(y_cen) - vertical_width
        y_max = int(y_cen) + vertical_width  
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max > int(x_width) - 1:
            x_max = int(x_width) - 1
        if y_max > int(y_width) - 1:
            y_max = int(y_width) - 1 
        data_subset = data[x_min:x_max,y_min:y_max]
        x_cen, y_cen = ndimage.measurements.center_of_mass(data_subset)
        x_cen = x_cen + x_min
        y_cen = y_cen + y_min
        dimX = f['entry/instrument/detector_{ds}/pixel_num_x'.format(ds=dshort)][0]
        dimY = f['entry/instrument/detector_{ds}/pixel_num_y'.format(ds=dshort)][0]
        lateral_offset = f['entry/instrument/detector_{ds}/lateral_offset'.format(ds=dshort)][0]
        panel_gap = f['entry/instrument/detector_{ds}/panel_gap'.format(ds=dshort)][0]/10.0
        tube_width = f['entry/instrument/detector_{ds}/tube_width'.format(ds=dshort)][0]/10.0
        sc1 = f['entry/instrument/detector_{ds}/spatial_calibration'.format(ds=dshort)][0][0]/10.0
        sc2 = f['entry/instrument/detector_{ds}/spatial_calibration'.format(ds=dshort)][1][0]/10.0
        sc3 = f['entry/instrument/detector_{ds}/spatial_calibration'.format(ds=dshort)][2][0]/10.0
        middle_bc_x = lateral_offset +(x_cen + 0.5)*tube_width + panel_gap/2.0
        middle_bc_y = sc1 + sc2*y_cen + sc3*y_cen*y_cen
        middle_bc_x = int(middle_bc_x*100.0)/100
        middle_bc_y = int(middle_bc_y*100.0)/100

    if f is not None:
        f.close()
    return middle_bc_x, middle_bc_y

def _all_sans_pol_corr_scatt_files(Detector_Panels, Instrument, UsePolCorr, input_path, He3CorrectionType, BestPSM, Minimum_PSM, dimXX, dimYY, Sample, Config, Scatt, Trans, Pol_Trans, UUScaledData, DUScaledData, DDScaledData, UDScaledData, UUScaledData_Unc, DUScaledData_Unc, DDScaledData_Unc, UDScaledData_Unc, HE3_Cell_Summary):
    """Apply the full-polarization correction (or He3-only correction) per panel.

    Builds the time-averaged 4x4 polarization-efficiency matrix from each
    cross-section's flipper times and 3He decay, inverts it, and applies
    it to the four scaled cross-sections. Three matrix variants are
    available via ``He3CorrectionType``. If ``UsePolCorr`` is false, only
    the diagonal He3-efficiency correction is applied. The high-resolution
    back detector is handled separately when ``Config`` contains ``'CvB'``.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'`` (selects flat-array sizing).
    UsePolCorr : bool or int
        Required. If truthy, apply the full inversion; otherwise apply
        only the He3-efficiency diagonal correction.
    input_path : str
        Required. Directory containing raw NeXus files.
    He3CorrectionType : {0, 1, 2}
        Required. Selects the polarization-efficiency model:
        0 ``X`` depol before sample, ``Y`` depol after = 1;
        1 default (``X = Y``);
        2 ``Y`` after, ``X`` before = 1.
    BestPSM : float
        Required. Best-known supermirror polarization.
    Minimum_PSM : float
        Required. Floor on the measured supermirror polarization.
    dimXX, dimYY : dict[str, int]
        Required. Per-panel detector dimensions (output of
        :func:`_q_calculation_all_detectors`).
    Sample, Config : str
        Required. Sample and configuration labels.
    Scatt : dict
        Required. Scattering catalog (used to read file lists and times).
    Trans : dict
        Required. Transmission catalog (used to check presence).
    Pol_Trans : dict
        Required. Polarized-transmission catalog supplying ``P_SM`` / ``P_F``.
    UUScaledData, DUScaledData, DDScaledData, UDScaledData : dict[str, np.ndarray]
        Required. Absolute-scaled per-cross-section intensities.
    UUScaledData_Unc, DUScaledData_Unc, DDScaledData_Unc, UDScaledData_Unc : dict[str, np.ndarray]
        Required. Matching uncertainties.
    HE3_Cell_Summary : dict
        Required. 3He cell parameter summary.

    Returns
    -------
    Have_FullPol : int
        0 = no correction applied, 1 = He3-only correction, 2 = full
        polarization correction.
    PolCorr_ALLCS : dict[str, np.ndarray]
        Per-panel sum of UU + DD.
    PolCorr_UU, PolCorr_DU, PolCorr_DD, PolCorr_UD : dict[str, np.ndarray]
        Per-panel corrected cross-sections.
    PolCorr_ALLCS_Unc, PolCorr_UU_Unc, PolCorr_DU_Unc, PolCorr_DD_Unc, PolCorr_UD_Unc : dict[str, np.ndarray]
        Per-panel uncertainties.
    """

    if 'VSANS' in Instrument:
        Scaled_Data = np.zeros((8,4,6144))
        UncScaled_Data = np.zeros((8,4,6144))
    elif 'NG7SANS' in Instrument:
        Scaled_Data = np.zeros((8,4,16384))
        UncScaled_Data = np.zeros((8,4,16384))

    relevant_detectors = list(Detector_Panels)

    Det_counter = 0
    for dshort in relevant_detectors:
        UUD = np.array(UUScaledData[dshort])
        Scaled_Data[Det_counter][0][:] += UUD.flatten()
        
        DUD = np.array(DUScaledData[dshort])
        Scaled_Data[Det_counter][1][:] += DUD.flatten()

        DDD = np.array(DDScaledData[dshort])
        Scaled_Data[Det_counter][2][:] += DDD.flatten()

        UDD = np.array(UDScaledData[dshort])
        Scaled_Data[Det_counter][3][:] += UDD.flatten()

        UUD_Unc = np.array(UUScaledData_Unc[dshort])
        UncScaled_Data[Det_counter][0][:] += UUD_Unc.flatten()
        
        DUD_Unc = np.array(DUScaledData_Unc[dshort])
        UncScaled_Data[Det_counter][1][:] += DUD_Unc.flatten()

        DDD_Unc = np.array(DDScaledData_Unc[dshort])
        UncScaled_Data[Det_counter][2][:] += DDD_Unc.flatten()

        UDD_Unc = np.array(UDScaledData_Unc[dshort])
        UncScaled_Data[Det_counter][3][:] += UDD_Unc.flatten()

        Det_counter += 1

    '''#Full-Pol Reduction:'''
    PolCorr_ALLCS = {}
    PolCorr_UU = {}
    PolCorr_DU = {}
    PolCorr_DD = {}
    PolCorr_UD = {}
    PolCorr_ALLCS_Unc = {}
    PolCorr_UU_Unc = {}
    PolCorr_DU_Unc = {}
    PolCorr_DD_Unc = {}
    PolCorr_UD_Unc = {}

    Pol_Efficiency = np.zeros((4,4))
    Pol_Efficiency_V2 = np.zeros((4,4))
    Pol_Efficiency_V3 = np.zeros((4,4))
    HE3_Efficiency = np.zeros((4,4))
    PolCorr_AllDetectors = {}
    HE3Corr_AllDetectors = {}
    Uncertainty_PolCorr_AllDetectors = {}
    Have_FullPol = 0
    if Sample in Trans and str(Scatt[Sample]['Config(s)'][Config]['UU']).find('NA') == -1 and str(Scatt[Sample]['Config(s)'][Config]['DU']).find('NA') == -1 and str(Scatt[Sample]['Config(s)'][Config]['DD']).find('NA') == -1 and str(Scatt[Sample]['Config(s)'][Config]['UD']).find('NA') == -1:
        Have_FullPol = 1

        if Sample in Pol_Trans: 
            PSM = Pol_Trans[Sample]['P_SM']
            PF = Pol_Trans[Sample]['P_F']
            if UsePolCorr >= 1:
                Have_FullPol = 2
        else:
            PF = 1.0
            PSM = 1.0
        '''#Calculating an average block beam counts per pixel and time (seems to work better than a pixel-by-pixel subtraction,
        at least for shorter count times)'''

        Number_UU = 1.0*len(Scatt[Sample]['Config(s)'][Config]["UU"])
        Number_DU = 1.0*len(Scatt[Sample]['Config(s)'][Config]["DU"])
        Number_DD = 1.0*len(Scatt[Sample]['Config(s)'][Config]["DD"])
        Number_UD = 1.0*len(Scatt[Sample]['Config(s)'][Config]["UD"])      
            
        Scatt_Type = ["UU", "DU", "DD", "UD"]
        for type in Scatt_Type:
            type_time = type + "_Time"
            filenumber_counter = 0
            for filenumber in Scatt[Sample]['Config(s)'][Config][type]:
                #Check this correction
                f = _sans_get_by_filenumber(Instrument, input_path, filenumber)
                if f is not None:
                    entry = Scatt[Sample]['Config(s)'][Config][type_time][filenumber_counter]
                    NP, UT, T_MAJ, T_MIN = _he3_pol_at_given_time(entry, HE3_Cell_Summary)
                    C = NP
                    S = BestPSM
                    if PSM < Minimum_PSM:
                        PSM = Minimum_PSM
                    '''#0.9985 is the highest I've recently gotten at 5.5 Ang from EuSe 60 nm 0.95 V and 2.0 K'''
                    X = np.sqrt(PSM/S)
                    Y = X
                    SX = PSM
                    SY = PSM
                    if type == "UU":
                        CrossSection_Index = 0
                        UT = UT / Number_UU
                        Pol_Efficiency[CrossSection_Index][:] += [(C*(S*X*Y + Y) + S*X + 1)*UT, (C*(-S*X*Y + Y) - S*X + 1)*UT, (C*(S*X*Y - Y) - S*X + 1)*UT, (C*(-S*X*Y - Y) + S*X + 1)*UT]
                        Pol_Efficiency_V2[CrossSection_Index][:] += [(C*(SX + 1) + SX + 1)*UT, (C*(-SX + 1) - SX + 1)*UT, (C*(SX - 1) - SX + 1)*UT, (C*(-SX - 1) + SX + 1)*UT]
                        Pol_Efficiency_V3[CrossSection_Index][:] += [(C*(SY + Y) + S + 1)*UT, (C*(-SY + Y) - S + 1)*UT, (C*(SY - Y) - S + 1)*UT, (C*(-SY - Y) + S + 1)*UT]
                        HE3_Efficiency[CrossSection_Index][:] += [ UT, 0.0, 0.0, 0.0]
                    elif type == "DU":
                        CrossSection_Index = 1
                        UT = UT / Number_DU
                        Pol_Efficiency[CrossSection_Index][:] += [(C*(-S*X*Y + Y) - S*X + 1)*UT, (C*(S*X*Y + Y) + S*X + 1)*UT, (C*(-S*X*Y - Y) + S*X + 1)*UT, (C*(S*X*Y - Y) - S*X + 1)*UT]
                        Pol_Efficiency_V2[CrossSection_Index][:] += [(C*(-SX + 1) - SX + 1)*UT, (C*(SX + 1) + SX + 1)*UT, (C*(-SX - 1) + SX + 1)*UT, (C*(SX - 1) - SX + 1)*UT]
                        Pol_Efficiency_V3[CrossSection_Index][:] += [(C*(-SY + Y) - S + 1)*UT, (C*(SY + Y) + S + 1)*UT, (C*(-SY - Y) + S + 1)*UT, (C*(SY - Y) - S + 1)*UT]
                        HE3_Efficiency[CrossSection_Index][:] += [ 0.0, UT, 0.0, 0.0]
                    elif type == "DD":
                        CrossSection_Index = 2
                        UT = UT / Number_DD
                        Pol_Efficiency[CrossSection_Index][:] += [(C*(S*X*Y - Y) - S*X + 1)*UT, (C*(-S*X*Y - Y) + S*X + 1)*UT, (C*(S*X*Y + Y) + S*X + 1)*UT, (C*(-S*X*Y + Y) - S*X + 1)*UT]
                        Pol_Efficiency_V2[CrossSection_Index][:] += [(C*(SX - 1) - SX + 1)*UT, (C*(-SX - 1) + SX + 1)*UT, (C*(SX + 1) + SX + 1)*UT, (C*(-SX + 1) - SX + 1)*UT]
                        Pol_Efficiency_V3[CrossSection_Index][:] += [(C*(SY - Y) - S + 1)*UT, (C*(-SY - Y) + S + 1)*UT, (C*(SY + Y) + S + 1)*UT, (C*(-SY + Y) - S + 1)*UT]
                        HE3_Efficiency[CrossSection_Index][:] += [ 0.0, 0.0, UT, 0.0]
                    elif type == "UD":
                        CrossSection_Index = 3
                        UT = UT / Number_UD
                        Pol_Efficiency[CrossSection_Index][:] += [(C*(-S*X*Y - Y) + S*X + 1)*UT, (C*(S*X*Y - Y) - S*X + 1)*UT, (C*(-S*X*Y + Y) - S*X + 1)*UT, (C*(S*X*Y + Y) + S*X + 1)*UT]
                        Pol_Efficiency_V2[CrossSection_Index][:] += [(C*(-SX - 1) + SX + 1)*UT, (C*(SX - 1) - SX + 1)*UT, (C*(-SX + 1) - SX + 1)*UT, (C*(SX + 1) + SX + 1)*UT]
                        Pol_Efficiency_V3[CrossSection_Index][:] += [(C*(-SY - Y) + S + 1)*UT, (C*(SY - Y) - S + 1)*UT, (C*(-SY + Y) - S + 1)*UT, (C*(SY + Y) + S + 1)*UT]
                        HE3_Efficiency[CrossSection_Index][:] += [ 0.0, 0.0, 0.0, UT]
                    f.close()

        Prefactor = inv(Pol_Efficiency) #default for UsePolCorr == 1 and He3CorrectionType == 1
        if UsePolCorr and He3CorrectionType == 0: #old way with X depol before sample and Y depol after sample = 1
            Prefactor = inv(Pol_Efficiency_V2)
        if UsePolCorr and He3CorrectionType == 2: #Y depol after sample and X depol before sample = 1
            Prefactor = inv(Pol_Efficiency_V3)
        if not UsePolCorr:
            Prefactor = inv(4.0*HE3_Efficiency)
            
        if str(Config).find('CvB') != -1:
            HRX = int(dimXX['B'])
            HRY = int(dimYY['B'])
            highrespixels = HRX*HRY
            RearScaled_Data = np.zeros((4, highrespixels))
            UncRearScaled_Data = np.zeros((4, highrespixels))
            
            UUR = np.array(UUScaledData['B'])
            RearScaled_Data[0][:] += UUR.flatten()
            DUR = np.array(DUScaledData['B'])
            RearScaled_Data[1][:] += DUR.flatten()
            DDR = np.array(DDScaledData['B'])
            RearScaled_Data[2][:] += DDR.flatten()
            UDR = np.array(UDScaledData['B'])
            RearScaled_Data[3][:] += UDR.flatten()
            UncRearScaled_Data[0][:] = RearScaled_Data[0][:]
            UncRearScaled_Data[1][:] = RearScaled_Data[1][:]
            UncRearScaled_Data[2][:] = RearScaled_Data[2][:]
            UncRearScaled_Data[3][:] = RearScaled_Data[3][:]

            BackPolCorr = np.dot(Prefactor, RearScaled_Data)
            BackUncertainty_PolCorr = UncRearScaled_Data
            
            PolCorr_UU['B'] = BackPolCorr[0][:][:].reshape((HRX, HRY))
            PolCorr_UU_Unc['B'] = BackUncertainty_PolCorr[0][:][:].reshape((HRX, HRY))
            PolCorr_DU['B'] = BackPolCorr[1][:][:].reshape((HRX, HRY))
            PolCorr_DU_Unc['B'] = BackUncertainty_PolCorr[1][:][:].reshape((HRX, HRY))
            PolCorr_DD['B'] = BackPolCorr[2][:][:].reshape((HRX, HRY))
            PolCorr_DD_Unc['B'] = BackUncertainty_PolCorr[2][:][:].reshape((HRX, HRY))
            PolCorr_UD['B'] = BackPolCorr[3][:][:].reshape((HRX, HRY))
            PolCorr_UD_Unc['B'] = BackUncertainty_PolCorr[3][:][:].reshape((HRX, HRY))

        
        Det_Index = 0
        for dshort in relevant_detectors:
            UncData_Per_Detector = UncScaled_Data[Det_Index][:][:]
            Data_Per_Detector = Scaled_Data[Det_Index][:][:]
            
            PolCorr_Data = np.dot(2.0*Prefactor, Data_Per_Detector)

            PolCorr_AllDetectors[dshort] = PolCorr_Data
            Uncertainty_PolCorr_AllDetectors[dshort] = UncData_Per_Detector
            Det_Index += 1

            dimX = dimXX[dshort]
            dimY = dimYY[dshort]
            PolCorr_UU[dshort] = PolCorr_AllDetectors[dshort][0][:][:].reshape((dimX, dimY))
            PolCorr_DU[dshort] = PolCorr_AllDetectors[dshort][1][:][:].reshape((dimX, dimY))
            PolCorr_DD[dshort] = PolCorr_AllDetectors[dshort][2][:][:].reshape((dimX, dimY))
            PolCorr_UD[dshort] = PolCorr_AllDetectors[dshort][3][:][:].reshape((dimX, dimY))
            PolCorr_ALLCS[dshort] = PolCorr_UU[dshort] + PolCorr_DD[dshort]

            PolCorr_UU_Unc[dshort] = Uncertainty_PolCorr_AllDetectors[dshort][0][:][:].reshape((dimX, dimY))
            PolCorr_DU_Unc[dshort] = Uncertainty_PolCorr_AllDetectors[dshort][1][:][:].reshape((dimX, dimY))
            PolCorr_DD_Unc[dshort] = Uncertainty_PolCorr_AllDetectors[dshort][2][:][:].reshape((dimX, dimY))
            PolCorr_UD_Unc[dshort] = Uncertainty_PolCorr_AllDetectors[dshort][3][:][:].reshape((dimX, dimY))
            PolCorr_ALLCS_Unc[dshort] = np.sqrt(np.power(PolCorr_UU_Unc[dshort],2) + np.power(PolCorr_UD_Unc[dshort],2) + np.power(PolCorr_DD_Unc[dshort],2) + np.power(PolCorr_UD_Unc[dshort],2))
            #PolCorr_NSF_Sum_Unc[dshort] = np.sqrt(np.power(PolCorr_UU_Unc[dshort],2) + np.power(PolCorr_DD_Unc[dshort],2))
            #PolCorr_NSF_Diff_Unc[dshort] = PolCorr_NSF_Sum_Unc[dshort]
            #PolCorr_SF_Sum_Unc[dshort] = np.sqrt(np.power(PolCorr_UD_Unc[dshort],2) + np.power(PolCorr_UD_Unc[dshort],2))

    return Have_FullPol, PolCorr_ALLCS, PolCorr_UU, PolCorr_DU, PolCorr_DD, PolCorr_UD, PolCorr_ALLCS_Unc, PolCorr_UU_Unc, PolCorr_DU_Unc, PolCorr_DD_Unc, PolCorr_UD_Unc

def _sans_process_full_pol_slices(StructurallyIsotropic, Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, Config, PolSampleSlices, Sample):
    """Combine four full-pol cross-section slices into structural/magnetic results.

    For each sector slice the four cross-sections are optionally MT-subtracted,
    saved as text and PNG, and the Horz/Vert pair is used to derive
    ``M_Perp``, ``M_Parl_NSF``, and the non-spin-flip structural sum, which
    are then plotted and written as ``ResultsFullPol_*`` / ``PlotFullPol*``.

    Parameters
    ----------
    StructurallyIsotropic : bool
        Required. Use the horizontal NSF sum as the denominator in the
        ``M_Parl_NSF`` decomposition when true; otherwise the vertical.
    Slices : Iterable[str]
        Required. Slice keys.
    SectorCutAngles : float
        Required. Sector half-width (degrees).
    save_path : str
        Required. Output directory.
    YesNoShowPlots : bool
        Required. Display plots in addition to saving.
    YesNoSetPlotXRange, YesNoSetPlotYRange : bool
        Required. Toggle manual plot axis limits.
    PlotXmin, PlotXmax, PlotYmin, PlotYmax : float
        Required. Manual axis limits.
    AutoSubtractEmpty : bool or int
        Required. Subtract the empty-cell scattering when an ``'Empty'``
        entry is present in ``PolSampleSlices``.
    Config : str
        Required. Configuration label.
    PolSampleSlices : dict
        Required. Mapping ``Sample -> slice_key -> {cross_section -> dict}``
        for the current configuration; ``'Empty'`` is consulted for MT
        subtraction.
    Sample : str
        Required. Sample key to process.

    Returns
    -------
    Results : dict
        Per-slice circ/horz/vert sums plus ``M_Perp``, ``M_Parl_NSF``,
        and their uncertainties (keys depend on which slices were
        computed).
    """

    Sub = ""
    HaveMT = 0
    Data_Cuts = {}
    MT_Cuts = {}
    for sector_cut in Slices:
        Data_Cuts[sector_cut] = {'Q' : 'NA', 'UU': 'NA', 'UD': 'NA', 'DD': 'NA', 'DU': 'NA', 'Q_Mean': 'NA', 'Q_Unc': 'NA', 'Shadow': 'NA'}
        MT_Cuts[sector_cut] = {'Q' : 'NA', 'UU': 'NA', 'UD': 'NA', 'DD': 'NA', 'DU': 'NA', 'Q_Mean': 'NA', 'Q_Unc': 'NA', 'Shadow': 'NA'}
        
        if sector_cut == "Circ":
            slice_details = "CircAve"
        else:
            slice_details = sector_cut + str(SectorCutAngles)
            
        PolType = PolSampleSlices[Sample][slice_details]['PolType']
        
        for entry in PolSampleSlices[Sample][slice_details]:
            if entry == 'UU':
                Data_Cuts[sector_cut]['Q'] = PolSampleSlices[Sample][slice_details][entry]['Q']
                Data_Cuts[sector_cut]['UU'] = PolSampleSlices[Sample][slice_details][entry]['I']
                Data_Cuts[sector_cut]['UU_Unc'] = PolSampleSlices[Sample][slice_details][entry]['I_Unc']
                Data_Cuts[sector_cut]['Q_Mean'] = PolSampleSlices[Sample][slice_details][entry]['Q_Mean']
                Data_Cuts[sector_cut]['Q_Unc'] = PolSampleSlices[Sample][slice_details][entry]['Q_Uncertainty']
                Data_Cuts[sector_cut]['Shadow'] = PolSampleSlices[Sample][slice_details][entry]['Shadow']
            elif entry == 'DU' or entry == 'DD'or entry == 'UD':
                Data_Cuts[sector_cut][entry] = PolSampleSlices[Sample][slice_details][entry]['I']
                Data_Cuts[sector_cut][entry+"_Unc"] = PolSampleSlices[Sample][slice_details][entry]['I_Unc']
                    
        if 'Empty' in PolSampleSlices and AutoSubtractEmpty == 1:
            if PolType in PolSampleSlices['Empty'][slice_details]['PolType']:
                Sub = ",SubMT"
                HaveMT = 1
                for entry in PolSampleSlices['Empty'][slice_details]:
                    if entry == 'UU':
                        MT_Cuts[sector_cut]['Q'] = PolSampleSlices['Empty'][slice_details][entry]['Q']
                        MT_Cuts[sector_cut]['UU'] = PolSampleSlices['Empty'][slice_details][entry]['I']
                        MT_Cuts[sector_cut]['UU_Unc'] = PolSampleSlices['Empty'][slice_details][entry]['I_Unc']
                        MT_Cuts[sector_cut]['Q_Mean'] = PolSampleSlices['Empty'][slice_details][entry]['Q_Mean']
                        MT_Cuts[sector_cut]['Q_Unc'] = PolSampleSlices['Empty'][slice_details][entry]['Q_Uncertainty']
                        MT_Cuts[sector_cut]['Shadow'] = PolSampleSlices['Empty'][slice_details][entry]['Shadow']
                    elif entry == 'DU' or entry == 'DD'or entry == 'UD':
                        MT_Cuts[sector_cut][entry] = PolSampleSlices['Empty'][slice_details][entry]['I']
                        MT_Cuts[sector_cut][entry+"_Unc"] = PolSampleSlices['Empty'][slice_details][entry]['I_Unc']
                DataMatch, MTMatch = _match_q_pa_data_sets(Data_Cuts[sector_cut], MT_Cuts[sector_cut], 2)
                Data_Cuts[sector_cut] = _subtract_pa_data_sets(DataMatch, MTMatch, 2)

        _save_text_data_four_combined_cross_sections(save_path,  '{corr}'.format(corr = PolType), slice_details, Sub, Sample, Config, Data_Cuts[sector_cut])
        _plot_four_combined_cross_sections(save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, '{corr}'.format(corr = PolType), slice_details, Sub, Sample, Config, Data_Cuts[sector_cut])

    AngleA = (45 - SectorCutAngles)*3.141593/180.0
    AngleB = (45 + SectorCutAngles)*3.141593/180.0
    DiagSinSinCosCosFactor = ((AngleB - AngleA)/8.0 + (np.sin(4.0*AngleA) - np.sin(4.0*AngleB))/32.0)/(AngleB - AngleA)
    DiagCosCosCosCosFactor = (3.0*(AngleB - AngleA)/8.0 + (np.sin(2.0*AngleB) - np.sin(2.0*AngleA))/4.0 + (np.sin(4.0*AngleB) - np.sin(4.0*AngleA))/32.0)/(AngleB - AngleA)
    AngleA = (0 - SectorCutAngles)*3.141593/180.0
    AngleB = (0 + SectorCutAngles)*3.141593/180.0
    HorzCosCosCosCosFactor = (3.0*(AngleB - AngleA)/8.0 + (np.sin(2.0*AngleB) - np.sin(2.0*AngleA))/4.0 + (np.sin(4.0*AngleB) - np.sin(4.0*AngleA))/32.0)/(AngleB - AngleA)
    AngleA = (90 - SectorCutAngles)*3.141593/180.0
    AngleB = (90 + SectorCutAngles)*3.141593/180.0
    VertSinSinFactor = ((AngleB - AngleA)/2.0 - (np.sin(2.0*AngleB) - np.sin(2.0*AngleA))/4.0)/(AngleB - AngleA)
    

    if "Horz" in Slices and "Vert" in Slices:
        HorzMatch, VertMatch = _match_q_pa_data_sets(Data_Cuts["Horz"], Data_Cuts["Vert"], 2)
        for entry in Data_Cuts["Horz"]:
            Data_Cuts["Horz"][entry] = HorzMatch[entry]
        for entry in Data_Cuts["Vert"]:
            Data_Cuts["Vert"][entry] = VertMatch[entry]

        if "Diag" in Slices:
            DiagMatch, VertMatch = _match_q_pa_data_sets(Data_Cuts["Diag"], Data_Cuts["Vert"], 2)
            for entry in Data_Cuts["Diag"]:
                Data_Cuts["Diag"][entry] = DiagMatch[entry]
            for entry in Data_Cuts["Vert"]:
                Data_Cuts["Vert"][entry] = VertMatch[entry]

            DiagMatch, HorzMatch = _match_q_pa_data_sets(Data_Cuts["Diag"], Data_Cuts["Horz"], 2)
            for entry in Data_Cuts["Diag"]:
                Data_Cuts["Diag"][entry] = DiagMatch[entry]
            for entry in Data_Cuts["Horz"]:
                Data_Cuts["Horz"][entry] = HorzMatch[entry]

        Horz_NSFSum = (Data_Cuts["Horz"]['DD'] + Data_Cuts["Horz"]['UU'])/2.0
        Horz_NSFSum_Unc = np.sqrt(np.power(Data_Cuts["Horz"]['DD_Unc'],2) + np.power(Data_Cuts["Horz"]['UU_Unc'],2))/2.0
        

        Vert_NSFSum = (Data_Cuts["Vert"]['UU'] + Data_Cuts["Vert"]['DD'])/2.0
        Vert_NSFSum_Unc = np.sqrt(np.power(Data_Cuts["Vert"]['UU_Unc'],2) + np.power(Data_Cuts["Vert"]['DD_Unc'],2))

        M_Parl_Sub = (Vert_NSFSum - Horz_NSFSum)
        M_Parl_Sub_Unc = Horz_NSFSum/20.0


        M_Perp = ((Data_Cuts["Horz"]['DU'] + Data_Cuts["Horz"]['UD'] + Data_Cuts["Vert"]['DU'] + Data_Cuts["Vert"]['UD']))/(2.0 + HorzCosCosCosCosFactor)
        M_Perp_Unc = (np.sqrt(np.power(Data_Cuts["Horz"]['DU_Unc'],2) + np.power(Data_Cuts["Horz"]['UD_Unc'],2) + np.power(Data_Cuts["Vert"]['DU_Unc'],2) + np.power(Data_Cuts["Vert"]['UD_Unc'],2)))/(2.0 + HorzCosCosCosCosFactor)
        
        Diff = Data_Cuts["Vert"]['DD'] - Data_Cuts["Vert"]['UU']
        Diff_Unc = np.sqrt(np.power(Data_Cuts["Vert"]['DD_Unc'],2) + np.power(Data_Cuts["Vert"]['UU_Unc'],2))
        Num = np.power((Diff),2)
        Num_Unc = np.sqrt(2.0)*Diff*Diff_Unc
        if StructurallyIsotropic:
            Denom = (8.0*(Data_Cuts["Horz"]['DD'] + Data_Cuts["Horz"]['UU']))
            Denom_Unc = np.sqrt(np.power(Data_Cuts["Horz"]['DD_Unc'],2) + np.power(Data_Cuts["Horz"]['UU_Unc'],2))
        else:
            Denom = (8.0*(Data_Cuts["Vert"]['DD'] + Data_Cuts["Vert"]['UU']))
            Denom_Unc = np.sqrt(np.power(Data_Cuts["Vert"]['DD_Unc'],2) + np.power(Data_Cuts["Vert"]['UU_Unc'],2))
        MParl_Mask = np.ones_like(Denom)
        MParl_Mask[Denom <= 0] = 0
        Num[Denom <= 0] = 1
        Num_Unc[Denom <= 0] = 1
        Denom_Unc[Denom <= 0] = 1
        Denom[Denom <= 0] = 1
        MParl_Mask[Num <= 0] = 0
        Denom_Unc[Num <= 0] = 1
        Denom[Num <= 0] = 1
        Num_Unc[Num <= 0] = 1
        Num[Num <= 0] = 1
        if Sample != 'Empty':            
            M_Parl_NSF = MParl_Mask*(Num / Denom)/VertSinSinFactor
            M_Parl_NSF_Unc = MParl_Mask*(M_Parl_NSF * np.sqrt(np.power(Num_Unc,2)/np.power(Num,2) + np.power(Denom_Unc,2)/np.power(Denom,2)))
        else:
            M_Parl_NSF = np.zeros_like(Num)
            M_Parl_NSF_Unc = np.zeros_like(Num)
            
        if "Diag" in Slices:
            M_Parl_SF = ((Data_Cuts["Diag"]['UD'] + Data_Cuts["Diag"]['DU']) - (1.0 + HorzCosCosCosCosFactor)*M_Perp )/(2.0*DiagSinSinCosCosFactor)
            M_Parl_SF_Unc = M_Parl_SF/10
            #(np.sqrt(M_Perp_Unc,2) + np.power(Data_Cuts["Diag"]['DU_Unc'],2) + np.power(Data_Cuts["Diag"]['UD_Unc'],2))/(2.0*DiagSinSinCosCosFactor)

        Factor = 1.0 #Factor = 29.0 for Fe3O4 NPs

        Width = str(SectorCutAngles) + "Deg"
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        if YesNoSetPlotYRange:
            ax.set_ylim(bottom = PlotYmin, top = PlotYmax)
        if YesNoSetPlotXRange:
            ax.set_xlim(left = PlotXmin, right = PlotXmax)
        ax.errorbar(Data_Cuts["Horz"]['Q'], Factor*M_Perp, yerr=Factor*M_Perp_Unc, fmt = 'r*', label='Sum(M_Perp^2), spin-flip')
        if Sample != 'Empty':
            ax.errorbar(Data_Cuts["Vert"]['Q'], Factor*M_Parl_NSF, yerr=Factor*M_Parl_NSF_Unc, fmt = 'g*', label='(Sum M_Parl)^2, non spin-flip')
            #ax.errorbar(Data_Cuts["Horz"]['Q'], Factor*M_Parl_Sub, yerr=Factor*M_Parl_Sub_Unc, fmt = 'y*', label='Sum(M_Parl^2), non spin-flip sub')
            #if "Diag" in Slices:
                #ax.errorbar(Data_Cuts["Diag"]['Q'], Factor*M_Parl_SF, yerr=Factor*M_Parl_SF_Unc, fmt = 'm*', label='Sum(M_Parl)^2, spin-flip')
        ax.errorbar(Data_Cuts["Horz"]['Q'], Factor*Horz_NSFSum, yerr=Factor*Horz_NSFSum_Unc, fmt = 'b*', label='Sum(N^2), non spin-flip')
        ax.errorbar(Data_Cuts["Vert"]['Q'], Factor*Vert_NSFSum, yerr=Factor*Vert_NSFSum_Unc, fmt = 'c*', label='Vert-only Sum(N^2), non spin-flip')
        plt.xlabel('Q (inverse angstroms)')
        plt.ylabel('Intensity')
        plt.title('Full-Pol Magnetic and Structural Scattering of {samp}'.format(samp=Sample))
        plt.legend()
        file_name = 'ResultsFullPol_{samp},{cf}_{key}{width}{sub}.png'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        fig.savefig(file_path, dpi=300)
        if YesNoShowPlots:
            plt.show()
        plt.close()

        Q = Data_Cuts["Horz"]['Q']
        Q_mean = Data_Cuts["Horz"]['Q_Mean']
        Q_Unc = Data_Cuts["Horz"]['Q_Unc']
        Shadow = Data_Cuts["Horz"]['Shadow']

        text_output = np.array([Q, Horz_NSFSum, Horz_NSFSum_Unc, M_Perp, M_Perp_Unc, M_Parl_NSF, M_Parl_NSF_Unc, Q_Unc, Q_mean, Shadow])
        text_output = text_output.T
        file_name = 'ResultsFullPol_{samp},{cf}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, text_output,
               delimiter = ' ', comments = '', header= 'Q, Struc, DelStruc, M_Perp, DelM_Perp, M_Parl_NSF, DelM_Parl_NSF, Q_Unc, Q_mean, Shadow', fmt='%1.4e')

        text_output2 = np.array([Q, M_Parl_NSF, M_Parl_NSF_Unc, Q_Unc, Q_mean, Shadow])
        text_output2 = text_output2.T
        file_name = 'PlotFullPolMparl_{samp},{cf}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, text_output2,
               delimiter = ' ', comments = '', header= 'Q, M_Parl_NSF, DelM_Parl_NSF, Q_Unc, Q_mean, Shadow', fmt='%1.4e')
        text_output3 = np.array([Q, M_Perp, M_Perp_Unc, Q_Unc, Q_mean, Shadow])
        text_output3 = text_output3.T
        file_name = 'PlotFullPolMperp_{samp},{cf}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, text_output3,
               delimiter = ' ', comments = '', fmt='%1.4e')
        text_output4 = np.array([Q, Horz_NSFSum, Horz_NSFSum_Unc, Q_Unc, Q_mean, Shadow])
        text_output4 = text_output4.T
        file_name = 'PlotFullPolStruc_{samp},{cf}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, text_output4,
               delimiter = ' ', comments = '', header= 'Q, Struc, DelStruc, Q_Unc, Q_mean, Shadow', fmt='%1.4e')
        
    Results = {}

    if 'Circ' in Slices:
        Circ_Sum = Data_Cuts["Circ"]['UU'] + Data_Cuts["Circ"]['DU'] + Data_Cuts["Circ"]['DD'] + Data_Cuts["Circ"]['UD']
        Circ_Sum_Unc = np.sqrt(np.power(Data_Cuts["Circ"]['UU_Unc'],2) + np.power(Data_Cuts["Circ"]['DU_Unc'],2) + np.power(Data_Cuts["Circ"]['DD_Unc'],2) + np.power(Data_Cuts["Circ"]['UD_Unc'],2))
        Results['QCirc'] = Data_Cuts["Circ"]['Q']
        Results['CircSum'] = Circ_Sum
        Results['CircSum_Unc'] = Circ_Sum_Unc
    if 'Horz' in Slices:
        Horz_Sum = Data_Cuts["Horz"]['UU'] + Data_Cuts["Horz"]['DU'] + Data_Cuts["Horz"]['DD'] + Data_Cuts["Horz"]['UD']
        Horz_Sum_Unc = np.sqrt(np.power(Data_Cuts["Horz"]['UU_Unc'],2) + np.power(Data_Cuts["Horz"]['DU_Unc'],2) + np.power(Data_Cuts["Horz"]['DD_Unc'],2) + np.power(Data_Cuts["Horz"]['UD_Unc'],2))
        Horz_NSFSum = Data_Cuts["Horz"]['UU'] + Data_Cuts["Horz"]['DD']
        Horz_NSFSum_Unc = np.sqrt(np.power(Data_Cuts["Horz"]['UU_Unc'],2) + np.power(Data_Cuts["Horz"]['DD_Unc'],2))
        Results['QHorz'] = Data_Cuts["Horz"]['Q']
        Results['HorzSum'] = Data_Cuts["Horz"]['Q']
        Results['HorzSum_Unc'] = Horz_Sum_Unc
        Results['HorzNSFSum'] = Horz_NSFSum
        Results['HorzNSFSum_Unc'] = Horz_NSFSum_Unc
    if 'Horz' in Slices and 'Vert' in Slices:
        Results['QHorzVert'] = Data_Cuts["Vert"]['Q']   
        Results['M_Perp'] = M_Perp
        Results['M_Perp_Unc'] = M_Perp_Unc
        Results['M_Parl_NSF'] = M_Parl_NSF
        Results['M_Parl_NSF_Unc'] = M_Parl_NSF_Unc
        Results['VertNSFSum'] = Vert_NSFSum
        Results['VertNSFSum_Unc'] = Vert_NSFSum_Unc
    
    return Results

def _sans_process_half_pol_slices(StructurallyIsotropic, Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, UseMTCirc, Config, PolSampleSlices, Sample):
    """Combine half-pol (U/D) slices into structural and magnetic results.

    For each requested slice, U and D are optionally MT-subtracted (with
    the option to use the empty-cell circular slice as the MT source). When
    both ``Horz`` and ``Vert`` slices exist, computes ``M_Parl`` via both
    subtraction and division forms plus the structural sum, then saves
    ``ResultsHalfPol_*`` and ``PlotHalfPol*`` text and PNG files.

    Parameters
    ----------
    StructurallyIsotropic : bool
        Required. Use horizontal sum as the denominator in the ``M_Parl``
        decomposition when true; vertical otherwise.
    Slices : Iterable[str]
        Required. Slice keys.
    SectorCutAngles : float
        Required. Sector half-width (degrees).
    save_path : str
        Required. Output directory.
    YesNoShowPlots : bool
        Required. Display plots in addition to saving.
    YesNoSetPlotXRange, YesNoSetPlotYRange : bool
        Required. Toggle manual plot axis limits.
    PlotXmin, PlotXmax, PlotYmin, PlotYmax : float
        Required. Manual axis limits.
    AutoSubtractEmpty : bool or int
        Required. Subtract empty-cell scattering.
    UseMTCirc : bool or int
        Required. Use the empty-cell circular slice as the MT source for
        Horz/Vert subtractions when both are available.
    Config : str
        Required. Configuration label.
    PolSampleSlices : dict
        Required. Per-sample half-pol slice dict for this configuration.
    Sample : str
        Required. Sample key.

    Returns
    -------
    Results : dict
        ``QCirc`` / ``CircSum`` / ``CircSum_Unc`` when circular data is
        present; empty otherwise.
    """

    Sub = ""

    Vert_Data = {}
    Horz_Data = {}
    Circ_Data = {}
    MT = {}
    MTCirc = {}
    HaveMTCirc = 0
    HaveVertData = 0
    HaveHorzData = 0
    HaveCircData = 0

    if "Circ" in Slices:
        slice_details = "CircAve"
        PolType = PolSampleSlices[Sample][slice_details]['PolType']
        HaveCircData = 1
        Circ_Data['Q'] = PolSampleSlices[Sample][slice_details]['U']['Q']
        Circ_Data['U'] = PolSampleSlices[Sample][slice_details]['U']['I']
        Circ_Data['U_Unc'] = PolSampleSlices[Sample][slice_details]['U']['I_Unc']
        Circ_Data['D'] = PolSampleSlices[Sample][slice_details]['D']['I']
        Circ_Data['D_Unc'] = PolSampleSlices[Sample][slice_details]['D']['I_Unc']
        Circ_Data['Q_Mean'] = PolSampleSlices[Sample][slice_details]['U']['Q_Mean']
        Circ_Data['Q_Unc'] = PolSampleSlices[Sample][slice_details]['U']['Q_Uncertainty']
        Circ_Data['Shadow'] = PolSampleSlices[Sample][slice_details]['U']['Shadow']

        if 'Empty' in PolSampleSlices and AutoSubtractEmpty == 1:
            if PolType in PolSampleSlices['Empty'][slice_details]['PolType']:
                HaveMTCirc = 1
                Sub = ",SubMT"
                MT['Q'] = PolSampleSlices['Empty'][slice_details]['U']['Q']
                MT['U'] = PolSampleSlices['Empty'][slice_details]['U']['I']
                MT['U_Unc'] = PolSampleSlices['Empty'][slice_details]['U']['I_Unc']
                MT['D'] = PolSampleSlices['Empty'][slice_details]['D']['I']
                MT['D_Unc'] = PolSampleSlices['Empty'][slice_details]['D']['I_Unc']
                MT['Q_Mean'] = PolSampleSlices['Empty'][slice_details]['U']['Q_Mean']
                MT['Q_Unc'] = PolSampleSlices['Empty'][slice_details]['U']['Q_Uncertainty']
                MT['Shadow'] = PolSampleSlices['Empty'][slice_details]['U']['Shadow']
                MTCirc = MT

                CircMatch, MTMatch = _match_q_pa_data_sets(Circ_Data, MT, 1)
                Circ_Data = _subtract_pa_data_sets(CircMatch, MTMatch, 1)

        #_save_text_data_four_combined_cross_sections(save_path,  '{corr}'.format(corr = PolType), slice_details, Sub, Sample, Config, Circ_Data)
        '''saves data as SliceFullPol_{samp},{cf}_{corr}{slice_key}.txt'''
        #_plot_four_combined_cross_sections(save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, '{corr}'.format(corr = PolType), slice_details, Sub, Sample, Config, Circ_Data)
        '''saves data as SliceFullPol_{samp},{cf}_{corr}{slice_key}.png'''
        Circ_Sum = (Circ_Data['U'] + Circ_Data['D'])/2.0
        Circ_Sum_Unc = (np.sqrt(np.power(Circ_Data['U_Unc'],2) + np.power(Circ_Data['D_Unc'],2)))/2.0

    if "Horz" in Slices:
        slice_details = "Horz"+str(SectorCutAngles)
        PolType = PolSampleSlices[Sample][slice_details]['PolType']
        HaveHorzData = 1
        Horz_Data['Q'] = PolSampleSlices[Sample][slice_details]['U']['Q']
        Horz_Data['U'] = PolSampleSlices[Sample][slice_details]['U']['I']
        Horz_Data['U_Unc'] = PolSampleSlices[Sample][slice_details]['U']['I_Unc']
        Horz_Data['D'] = PolSampleSlices[Sample][slice_details]['D']['I']
        Horz_Data['D_Unc'] = PolSampleSlices[Sample][slice_details]['D']['I_Unc']
        Horz_Data['Q_Mean'] = PolSampleSlices[Sample][slice_details]['U']['Q_Mean']
        Horz_Data['Q_Unc'] = PolSampleSlices[Sample][slice_details]['U']['Q_Uncertainty']
        Horz_Data['Shadow'] = PolSampleSlices[Sample][slice_details]['U']['Shadow']

        if 'Empty' in PolSampleSlices and AutoSubtractEmpty == 1:
            if PolType in PolSampleSlices['Empty'][slice_details]['PolType']:
                Sub = ",SubMT"
                MT['Q'] = PolSampleSlices['Empty'][slice_details]['U']['Q']
                MT['U'] = PolSampleSlices['Empty'][slice_details]['U']['I']
                MT['U_Unc'] = PolSampleSlices['Empty'][slice_details]['U']['I_Unc']
                MT['D'] = PolSampleSlices['Empty'][slice_details]['D']['I']
                MT['D_Unc'] = PolSampleSlices['Empty'][slice_details]['D']['I_Unc']
                MT['Q_Mean'] = PolSampleSlices['Empty'][slice_details]['U']['Q_Mean']
                MT['Q_Unc'] = PolSampleSlices['Empty'][slice_details]['U']['Q_Uncertainty']
                MT['Shadow'] = PolSampleSlices['Empty'][slice_details]['U']['Shadow']

                if UseMTCirc == 1 and HaveMTCirc == 1:
                    HorzMatch, MTMatch = _match_q_pa_data_sets(Horz_Data, MTCirc, 1)
                else:
                    HorzMatch, MTMatch = _match_q_pa_data_sets(Horz_Data, MT, 1)
                Horz_Data = _subtract_pa_data_sets(HorzMatch, MTMatch, 1)

        #_save_text_data_four_combined_cross_sections(save_path,  '{corr}'.format(corr = PolType), slice_details, Sub, Sample, Config, Horz_Data)
        '''saves data as SliceFullPol_{samp},{cf}_{corr}{slice_key}.txt'''
        #_plot_four_combined_cross_sections(save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, '{corr}'.format(corr = PolType), slice_details, Sub, Sample, Config, Horz_Data)
        '''saves data as SliceFullPol_{samp},{cf}_{corr}{slice_key}.png'''

    if "Vert" in Slices:
        slice_details = "Vert"+str(SectorCutAngles)
        PolType = PolSampleSlices[Sample][slice_details]['PolType']
        HaveVertData = 1
        Vert_Data['Q'] = PolSampleSlices[Sample][slice_details]['U']['Q']
        Vert_Data['U'] = PolSampleSlices[Sample][slice_details]['U']['I']
        Vert_Data['U_Unc'] = PolSampleSlices[Sample][slice_details]['U']['I_Unc']
        Vert_Data['D'] = PolSampleSlices[Sample][slice_details]['D']['I']
        Vert_Data['D_Unc'] = PolSampleSlices[Sample][slice_details]['D']['I_Unc']
        Vert_Data['Q_Mean'] = PolSampleSlices[Sample][slice_details]['U']['Q_Mean']
        Vert_Data['Q_Unc'] = PolSampleSlices[Sample][slice_details]['U']['Q_Uncertainty']
        Vert_Data['Shadow'] = PolSampleSlices[Sample][slice_details]['U']['Shadow']

        if 'Empty' in PolSampleSlices and AutoSubtractEmpty == 1:
            if PolType in PolSampleSlices['Empty'][slice_details]['PolType']:
                Sub = ",SubMT"
                MT['Q'] = PolSampleSlices['Empty'][slice_details]['U']['Q']
                MT['U'] = PolSampleSlices['Empty'][slice_details]['U']['I']
                MT['U_Unc'] = PolSampleSlices['Empty'][slice_details]['U']['I_Unc']
                MT['D'] = PolSampleSlices['Empty'][slice_details]['D']['I']
                MT['D_Unc'] = PolSampleSlices['Empty'][slice_details]['D']['I_Unc']
                MT['Q_Mean'] = PolSampleSlices['Empty'][slice_details]['U']['Q_Mean']
                MT['Q_Unc'] = PolSampleSlices['Empty'][slice_details]['U']['Q_Uncertainty']
                MT['Shadow'] = PolSampleSlices['Empty'][slice_details]['U']['Shadow']

                if UseMTCirc == 1 and HaveMTCirc == 1:
                    VertMatch, MTMatch = _match_q_pa_data_sets(Vert_Data, MTCirc, 1)
                else:
                    VertMatch, MTMatch = _match_q_pa_data_sets(Vert_Data, MT, 1)
                Vert_Data = _subtract_pa_data_sets(VertMatch, MTMatch, 1)

        #_save_text_data_four_combined_cross_sections(save_path,  '{corr}'.format(corr = PolType), slice_details, Sub, Sample, Config, Vert_Data)
        '''saves data as SliceFullPol_{samp},{cf}_{corr}{slice_key}.txt'''
        #_plot_four_combined_cross_sections(save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, '{corr}'.format(corr = PolType), slice_details, Sub, Sample, Config, Vert_Data)
        '''saves data as SliceFullPol_{samp},{cf}_{corr}{slice_key}.png'''

    if HaveHorzData == 1 and HaveVertData == 1:

        HorzMatch, VertMatch = _match_q_pa_data_sets(Horz_Data, Vert_Data, 1)
        for entry in Horz_Data:
            Horz_Data[entry] = HorzMatch[entry]
        for entry in Vert_Data:
            Vert_Data[entry] = VertMatch[entry]

        M_Parl_Sub = (Vert_Data['D'] + Vert_Data['U'] - (Horz_Data['D'] + Horz_Data['U']) )/4.0
        M_Parl_Sub_Unc = np.sqrt(np.power(Vert_Data['D_Unc'],2) + np.power(Vert_Data['U_Unc'],2) + np.power(Horz_Data['D_Unc'],2) + np.power(Horz_Data['U_Unc'],2))/4.0

        Struc = (Horz_Data['D'] + Horz_Data['U'])/2.0
        Struc_Unc = np.sqrt(np.power(Horz_Data['D_Unc'],2) + np.power(Horz_Data['U_Unc'],2))/2.0

        Diff = Vert_Data['D'] - Vert_Data['U']
        Diff_Unc = np.sqrt(np.power(Vert_Data['D_Unc'],2) + np.power(Vert_Data['U_Unc'],2))
        Num = np.power((Diff),2)
        Num_Unc = np.sqrt(2.0)*Diff*Diff_Unc

        if StructurallyIsotropic:
            Denom = (8.0*(Horz_Data['D'] + Horz_Data['U']))
            Denom_Unc = np.sqrt(np.power(Horz_Data['D_Unc'],2) + np.power(Horz_Data['U_Unc'],2))
        else:
            Denom = (8.0*(Vert_Data['D'] + Vert_Data['U']))
            Denom_Unc = np.sqrt(np.power(Vert_Data['D_Unc'],2) + np.power(Vert_Data['U_Unc'],2))

        MParl_Mask = np.ones_like(Denom)
        MParl_Mask[Denom <= 0] = 0
        Num[Denom <= 0] = 1
        Num_Unc[Denom <= 0] = 1
        Denom_Unc[Denom <= 0] = 1
        Denom[Denom <= 0] = 1
        
        MParl_Mask[Num <= 0] = 0
        Denom_Unc[Num <= 0] = 1
        Denom[Num <= 0] = 1
        Num_Unc[Num <= 0] = 1
        Num[Num <= 0] = 1
        
        if Sample != 'Empty':
            M_Parl_Div = MParl_Mask*(Num / Denom)
            M_Parl_Div_Unc = MParl_Mask*M_Parl_Div * np.sqrt( np.power(Num_Unc,2)/np.power(Num,2) + np.power(Denom_Unc,2)/np.power(Denom,2))
        else:
            M_Parl_Div = np.zeros_like(Num)
            M_Parl_Div_Unc = np.zeros_like(Num)

        Width = str(SectorCutAngles) + "Deg"
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        if YesNoSetPlotYRange:
            ax.set_ylim(bottom = PlotYmin, top = PlotYmax)
        if YesNoSetPlotXRange:
            ax.set_xlim(left = PlotXmin, right = PlotXmax)
        ax.errorbar(Horz_Data['Q'], M_Parl_Sub, yerr=M_Parl_Sub_Unc, fmt = 'b*', label='M_Parl (subtraction)')
        if Sample != 'Empty':
            ax.errorbar(Horz_Data['Q'], M_Parl_Div, yerr=M_Parl_Div_Unc, fmt = 'g*', label='M_Parl (*dividion)')
        ax.errorbar(Horz_Data['Q'], Struc, yerr=Struc_Unc, fmt = 'r*', label='Structural (horizontal)')
        plt.xlabel('Q (inverse angstroms)')
        plt.ylabel('Intensity')
        plt.title('Half-Pol Magnetic and Structural Scattering of {samp}'.format(samp=Sample))
        plt.legend()
        file_name = 'ResultsHalfPol_{samp},{cf}_{key}{width}{sub}.png'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        fig.savefig(file_path)
        if YesNoShowPlots:
            plt.show()
        plt.close()              

        Q = Horz_Data['Q']
        Q_mean = Horz_Data['Q_Mean']
        Q_Unc = Horz_Data['Q_Unc']
        Shadow = Horz_Data['Shadow']
        text_output = np.array([Q, Struc, Struc_Unc, M_Parl_Div, M_Parl_Div_Unc, M_Parl_Sub, M_Parl_Sub_Unc, Q_Unc, Q_mean, Shadow])
        text_output = text_output.T
        file_name = 'ResultsHalfPol_{samp},{cf}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)  
        np.savetxt(file_path, text_output,
        delimiter = ' ', comments = '', header= 'Q, Struc, DelStruc, M_Parl_Div, DelM_Parl_Div, M_Parl_Sub, DelM_Parl_Sub, Q_Unc, Q_mean, Shadow', fmt='%1.4e')

        text_output2 = np.array([Q, M_Parl_Div, M_Parl_Div_Unc, Q_Unc, Q_mean, Shadow])
        text_output2 = text_output2.T
        file_name = 'PlotHalfPolMparlDiv_{samp},{cf}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, text_output2,
        delimiter = ' ', comments = '', header= 'Q, M_Parl_Div, DelM_Parl_Div, Q_Unc, Q_mean, Shadow', fmt='%1.4e')
        
        text_output3 = np.array([Q, M_Parl_Sub, M_Parl_Sub_Unc, Q_Unc, Q_mean, Shadow])
        text_output3 = text_output3.T
        file_name = 'PlotHalfPolMparlSub_{samp},{cf}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, text_output3,
        delimiter = ' ', comments = '', fmt='%1.4e')

        text_output4 = np.array([Q, Struc, Struc_Unc, Q_Unc, Q_mean, Shadow])
        text_output4 = text_output4.T
        file_name = 'PlotHalfPolStruc_{samp},{cf}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, text_output4,
        delimiter = ' ', comments = '', header= 'Q, M_Parl_Sub, DelM_Parl_Sub, Q_Unc, Q_mean, Shadow', fmt='%1.4e')

    Results = {}
    if 'Circ' in Slices:
        Results['QCirc'] = Circ_Data['Q']
        Results['CircSum'] = Circ_Sum
        Results['CircSum_Unc'] = Circ_Sum_Unc
    
    return Results

def _sans_process_unpol_slices(Slices, SectorCutAngles, save_path, YesNoShowPlots, YesNoSetPlotXRange, YesNoSetPlotYRange, PlotXmin, PlotXmax, PlotYmin, PlotYmax, AutoSubtractEmpty, UseMTCirc, Config, PolSampleSlices, Sample):
    """Combine unpolarized slices and save structural / M_Parl_Sub results.

    For each requested slice, applies optional empty-cell subtraction
    (optionally using the empty circular slice as the MT source) and saves
    a per-slice ``SliceUnpol_*`` file. When both Horz and Vert slices exist,
    computes the simple difference (Vert - Horz) as ``M_Parl_Sub`` and the
    horizontal sum as the structural channel, saving ``ResultsUnpol_*``
    and ``PlotUnpol*`` text/PNG files.

    Parameters
    ----------
    Slices : Iterable[str]
        Required. Slice keys.
    SectorCutAngles : float
        Required. Sector half-width (degrees).
    save_path : str
        Required. Output directory.
    YesNoShowPlots : bool
        Required. Display plots in addition to saving.
    YesNoSetPlotXRange, YesNoSetPlotYRange : bool
        Required. Toggle manual plot axis limits.
    PlotXmin, PlotXmax, PlotYmin, PlotYmax : float
        Required. Manual axis limits.
    AutoSubtractEmpty : bool or int
        Required. Subtract empty-cell scattering.
    UseMTCirc : bool or int
        Required. Use the empty-cell circular slice as the MT source for
        Horz/Vert subtractions.
    Config : str
        Required. Configuration label.
    PolSampleSlices : dict
        Required. Per-sample unpol slice dict for this configuration.
    Sample : str
        Required. Sample key.

    Returns
    -------
    Results : dict
        Combination of ``QCirc/CircSum/CircSum_Unc``, ``QHorz/HorzSlice/
        HorzSlice_Unc``, and ``QVert/VertSlice/VertSlice_Unc`` for whichever
        slices were computed.
    """

    Sub = ""

    Vert_Data = {}
    Diag_Data = {}
    Horz_Data = {}
    Circ_Data = {}
    MT = {}
    MTCirc = {}
    HaveMTCirc = 0
    HaveVertData = 0
    HaveHorzData = 0
    HaveCircData = 0

    if "Circ" in Slices:
        slice_details = "CircAve"
        PolType = PolSampleSlices[Sample][slice_details]['PolType']
        HaveCircData = 1
        Circ_Data['Q'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q']
        Circ_Data['Unpol'] = PolSampleSlices[Sample][slice_details]['Unpol']['I']
        Circ_Data['Unpol_Unc'] = PolSampleSlices[Sample][slice_details]['Unpol']['I_Unc']
        Circ_Data['Q_Mean'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q_Mean']
        Circ_Data['Q_Unc'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q_Uncertainty']
        Circ_Data['Shadow'] = PolSampleSlices[Sample][slice_details]['Unpol']['Shadow']

        if 'Empty' in PolSampleSlices and AutoSubtractEmpty == 1:
            if PolType in PolSampleSlices['Empty'][slice_details]['PolType']:
                HaveMTCirc = 1
                Sub = ",SubMT"
                MT['Q'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Q']
                MT['Unpol'] = PolSampleSlices['Empty'][slice_details]['Unpol']['I']
                MT['Unpol_Unc'] = PolSampleSlices['Empty'][slice_details]['Unpol']['I_Unc']
                MT['Q_Mean'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Q_Mean']
                MT['Q_Unc'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Q_Uncertainty']
                MT['Shadow'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Shadow']
                MTCirc = MT

                CircMatch, MTMatch = _match_q_pa_data_sets(Circ_Data, MT, 0)
                Circ_Data = _subtract_pa_data_sets(CircMatch, MTMatch, 0)

        _save_text_data_unpol(save_path, Sub, slice_details, Sample, Config, Circ_Data)
        
        Circ_Sum = Circ_Data['Unpol']
        Circ_Sum_Unc = Circ_Data['Unpol_Unc']

    if "Horz" in Slices:
        slice_details = "Horz"+str(SectorCutAngles)
        PolType = PolSampleSlices[Sample][slice_details]['PolType']
        HaveHorzData = 1
        Horz_Data['Q'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q']
        Horz_Data['Unpol'] = PolSampleSlices[Sample][slice_details]['Unpol']['I']
        Horz_Data['Unpol_Unc'] = PolSampleSlices[Sample][slice_details]['Unpol']['I_Unc']
        Horz_Data['Q_Mean'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q_Mean']
        Horz_Data['Q_Unc'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q_Uncertainty']
        Horz_Data['Shadow'] = PolSampleSlices[Sample][slice_details]['Unpol']['Shadow']

        #if "Circ" in Slices:#Fix for Q
            #Horz_Data, CircMatchHorz_Data = _match_q_pa_data_sets(Horz_Data, Circ_Data, 0)
            #Horz_Data['Q_Unc'] = CircMatchHorz_Data['Q_Unc']
            #Horz_Data['Q_Mean'] = CircMatchHorz_Data['Q_Mean']

        if 'Empty' in PolSampleSlices and AutoSubtractEmpty == 1:
            if PolType in PolSampleSlices['Empty'][slice_details]['PolType']:
                Sub = ",SubMT"
                MT['Q'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Q']
                MT['Unpol'] = PolSampleSlices['Empty'][slice_details]['Unpol']['I']
                MT['Unpol_Unc'] = PolSampleSlices['Empty'][slice_details]['Unpol']['I_Unc']
                MT['Q_Mean'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Q_Mean']
                MT['Q_Unc'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Q_Uncertainty']
                MT['Shadow'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Shadow']

                if UseMTCirc == 1 and HaveMTCirc == 1:
                    HorzMatch, MTMatch = _match_q_pa_data_sets(Horz_Data, MTCirc, 0)
                else:
                    HorzMatch, MTMatch = _match_q_pa_data_sets(Horz_Data, MT, 0)
                Horz_Data = _subtract_pa_data_sets(HorzMatch, MTMatch, 0)

        _save_text_data_unpol(save_path, Sub, slice_details, Sample, Config, Horz_Data)

    if "Diag" in Slices:
        slice_details = "Diag"+str(SectorCutAngles)
        PolType = PolSampleSlices[Sample][slice_details]['PolType']
        Diag_Data['Q'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q']
        Diag_Data['Unpol'] = PolSampleSlices[Sample][slice_details]['Unpol']['I']
        Diag_Data['Unpol_Unc'] = PolSampleSlices[Sample][slice_details]['Unpol']['I_Unc']
        Diag_Data['Q_Mean'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q_Mean']
        Diag_Data['Q_Unc'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q_Uncertainty']
        Diag_Data['Shadow'] = PolSampleSlices[Sample][slice_details]['Unpol']['Shadow']

        #if "Circ" in Slices:#Fix for Q
            #Diag_Data, CircMatchDiag_Data = _match_q_pa_data_sets(Diag_Data, Circ_Data, 0)
            #Diag_Data['Q_Unc'] = CircMatchDiag_Data['Q_Unc']
            #Diag_Data['Q_Mean'] = CircMatchDiag_Data['Q_Mean']

    if "Vert" in Slices:
        slice_details = "Vert"+str(SectorCutAngles)
        PolType = PolSampleSlices[Sample][slice_details]['PolType']
        HaveVertData = 1
        Vert_Data['Q'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q']
        Vert_Data['Unpol'] = PolSampleSlices[Sample][slice_details]['Unpol']['I']
        Vert_Data['Unpol_Unc'] = PolSampleSlices[Sample][slice_details]['Unpol']['I_Unc']
        Vert_Data['Q_Mean'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q_Mean']
        Vert_Data['Q_Unc'] = PolSampleSlices[Sample][slice_details]['Unpol']['Q_Uncertainty']
        Vert_Data['Shadow'] = PolSampleSlices[Sample][slice_details]['Unpol']['Shadow']

        #if "Circ" in Slices:#Fix for Q
            #Vert_Data, CircMatchVert_Data = _match_q_pa_data_sets(Vert_Data, Circ_Data, 0)
            #Vert_Data['Q_Unc'] = CircMatchVert_Data['Q_Unc']
            #Vert_Data['Q_Mean'] = CircMatchVert_Data['Q_Mean']

        if 'Empty' in PolSampleSlices and AutoSubtractEmpty == 1:
            if PolType in PolSampleSlices['Empty'][slice_details]['PolType']:
                Sub = ",SubMT"
                MT['Q'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Q']
                MT['Unpol'] = PolSampleSlices['Empty'][slice_details]['Unpol']['I']
                MT['Unpol_Unc'] = PolSampleSlices['Empty'][slice_details]['Unpol']['I_Unc']
                MT['Q_Mean'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Q_Mean']
                MT['Q_Unc'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Q_Uncertainty']
                MT['Shadow'] = PolSampleSlices['Empty'][slice_details]['Unpol']['Shadow']

                if UseMTCirc == 1 and HaveMTCirc == 1:
                    VertMatch, MTMatch = _match_q_pa_data_sets(Vert_Data, MTCirc, 0)
                else:
                    VertMatch, MTMatch = _match_q_pa_data_sets(Vert_Data, MT, 0)
                Vert_Data = _subtract_pa_data_sets(VertMatch, MTMatch, 0)

        _save_text_data_unpol(save_path, Sub, slice_details, Sample, Config, Vert_Data)

    if HaveHorzData == 1 and HaveVertData == 1:

        HorzMatch, VertMatch = _match_q_pa_data_sets(Horz_Data, Vert_Data, 0)
        for entry in Horz_Data:
            Horz_Data[entry] = HorzMatch[entry]
        for entry in Vert_Data:
            Vert_Data[entry] = VertMatch[entry]

        M_Parl_Sub = Vert_Data['Unpol'] - Horz_Data['Unpol']
        M_Parl_Sub_Unc = np.sqrt(np.power(Vert_Data['Unpol_Unc'],2) + np.power(Horz_Data['Unpol_Unc'],2))

        Struc = Horz_Data['Unpol']
        Struc_Unc = Horz_Data['Unpol_Unc']
        #Kludge Dec2020
        StrucV = Vert_Data['Unpol']
        StrucV_Unc = Vert_Data['Unpol_Unc']
        StrucD = Diag_Data['Unpol']
        StrucD_Unc = Diag_Data['Unpol_Unc']

        Width = str(SectorCutAngles) + "Deg"
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xscale("log")
        ax.set_yscale("log")
        if YesNoSetPlotYRange:
            ax.set_ylim(bottom = PlotYmin, top = PlotYmax)
        if YesNoSetPlotXRange:
            ax.set_xlim(left = PlotXmin, right = PlotXmax)
        #if Sample != 'Empty':
            #ax.errorbar(Horz_Data['Q'], M_Parl_Sub, yerr=M_Parl_Sub_Unc, fmt = 'b*', label='M_Parl (subtraction)')
        #ax.errorbar(Diag_Data['Q'], StrucD, yerr=StrucD_Unc, fmt = 'b*', label='Structural (diagonal)')
        ax.errorbar(Horz_Data['Q'], Struc, yerr=Struc_Unc, fmt = 'r*', label='Structural (horizontal)')
        ax.errorbar(Vert_Data['Q'], StrucV, yerr=StrucV_Unc, fmt = 'g*', label='Structural (vertical)')
        plt.xlabel('Q (inverse angstroms)')
        plt.ylabel('Intensity')
        plt.title('Unpol Magnetic and Structural Scattering of {samp}'.format(samp=Sample))
        plt.legend()
        file_name = 'ResultsUnpol_{samp},{cf}_{key}{width}{sub}.png'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)  
        fig.savefig(file_path)
        if YesNoShowPlots:
            plt.show()
        plt.close()              

        Q = Horz_Data['Q']
        Q_mean = Horz_Data['Q_Mean']
        Q_Unc = Horz_Data['Q_Unc']
        Shadow = Horz_Data['Shadow']
        text_output = np.array([Q, Struc, Struc_Unc, M_Parl_Sub, M_Parl_Sub_Unc, Q_Unc, Q_mean, Shadow])
        text_output = text_output.T
        file_name = 'ResultsUnpol_{samp},{cf}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)  
        np.savetxt(file_path, text_output,
               delimiter = ' ', comments = '', header= 'Q, Struc, DelStruc, M_Parl_Sub, DelM_Parl_Sub, Q_Unc, Q_mean, Shadow', fmt='%1.4e')

        text_output3 = np.array([Q, M_Parl_Sub, M_Parl_Sub_Unc, Q_Unc, Q_mean, Shadow])
        text_output3 = text_output3.T
        file_name = 'PlotUnpolMparlSub_{samp}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, text_output3,
        delimiter = ' ', comments = '', fmt='%1.4e')

        text_output4 = np.array([Q, Struc, Struc_Unc, Q_Unc, Q_mean, Shadow])
        text_output4 = text_output4.T
        file_name = 'PlotUnpolStruc_{samp}_{key}{width}{sub}.txt'.format(samp=Sample, cf = Config, key = PolType, width = Width, sub = Sub)
        file_path = os.path.join(save_path, file_name)
        np.savetxt(file_path, text_output4,
        delimiter = ' ', comments = '', header= 'Q, Struc, DelStruc, Q_Unc, Q_mean, Shadow', fmt='%1.4e')


    Results = {}
    if 'Circ' in Slices:
        Results['QCirc'] = Circ_Data['Q']
        Results['CircSum'] = Circ_Sum
        Results['CircSum_Unc'] = Circ_Sum_Unc
    if 'Horz' in Slices:
        Results['QHorz'] = Horz_Data['Q']
        Results['HorzSlice'] = Horz_Data['Unpol']
        Results['HorzSlice_Unc'] = Horz_Data['Unpol_Unc']
    if 'Vert' in Slices:
        Results['QVert'] = Vert_Data['Q']
        Results['VertSlice'] = Vert_Data['Unpol']
        Results['VertSlice_Unc'] = Vert_Data['Unpol_Unc']
    
    return Results

def _annular_average(Detector_Panels, Instrument, save_path, Sample, Config, InPlaneAngleMap, Q_min, Q_max, Q_total, GeneralMask, ScaledData, ScaledData_Unc):
    """Plot intensity vs. azimuthal angle, averaged over an annular Q band.

    Steps through 72 sectors covering [0, 360) degrees, computes the
    per-sector ratio of summed intensity to summed pixels inside the
    annular band ``[Q_min, Q_max]``, and saves a PNG titled
    ``AnnularAverage_{Sample},{Config}.png``.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str
        Required. Instrument identifier (kept for API parity).
    save_path : str
        Required. Output directory.
    Sample, Config : str
        Required. Labels.
    InPlaneAngleMap : dict[str, np.ndarray]
        Required. Per-panel azimuthal angle map.
    Q_min, Q_max : float
        Required. Q-band bounds in inverse Angstroms.
    Q_total : dict[str, np.ndarray]
        Required. Per-panel |Q| maps.
    GeneralMask : dict[str, np.ndarray]
        Required. Per-panel general mask.
    ScaledData : dict[str, np.ndarray]
        Required. Per-panel absolute-scaled intensity.
    ScaledData_Unc : dict[str, np.ndarray]
        Required. Per-panel uncertainties (kept for API parity; not used).

    Returns
    -------
    None
    """

    relevant_detectors = list(Detector_Panels)
    AverageQRanges = 1
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')
        AverageQRanges = False

    Q_Mask = {}
    for dshort in relevant_detectors:
        QBorder = np.ones_like(Q_total[dshort])
        QBorder[Q_total[dshort] < Q_min] = 0.0
        QBorder[Q_total[dshort] > Q_max] = 0.0
        Q_Mask[dshort] = QBorder

    
    Counts = -101
    Deg = -101
    BothSides = 0
    PlotYesNo = 0
    for x in range(0, 72):
        degree = x*5
        Sector_Mask = _sector_mask_all_detectors(Detector_Panels, 'Unknown', Config, InPlaneAngleMap, degree, 2.5, BothSides)

        summed_pixels = 0
        summed_intensity = 0
        for dshort in relevant_detectors:
            pixel_counts = Sector_Mask[dshort]*Q_Mask[dshort]*GeneralMask[dshort]
            intensity_counts = pixel_counts*ScaledData[dshort]
            summed_pixels = summed_pixels + np.sum(pixel_counts)
            summed_intensity = summed_intensity + np.sum(intensity_counts)
        ratio = summed_intensity/summed_pixels
        if Counts == -101 and summed_pixels > 0:
            Counts = [ratio]
            Deg = [degree]
        elif summed_pixels > 0:
            Counts.append(ratio)
            Deg.append(degree)

    xdata = np.array(Deg)
    ydata = np.array(Counts)
    fig = plt.figure()
    plt.plot(xdata, ydata, 'b*-', label='annular_average')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Summed Counts')
    plt.title('Annular Average_{qmin}to{qmax}invang'.format(qmin = Q_min, qmax = Q_max))
    plt.legend()
    file_name = 'AnnularAverage_{samp},{cf}.png'.format(samp=Sample, cf = Config)   
    file_path = os.path.join(save_path, file_name)
    fig.savefig(file_path)
    #plt.show()
    plt.close()
      
    return

    
def _sans_categorize_samples_and_bases(He3Only_Check, Configs, Sample_Bases, Sample_Names, ScattCatalog, AllFullPolSlices,AllHalfPolSlices, AllUnpolSlices):
    """Group samples by base name and polarization mode for comparison plots.

    For each configuration, builds three ``{base_name -> [samples]}`` maps
    listing samples whose name contains the base name, that are marked as
    ``'Sample'`` in ``ScattCatalog``, and for which slice data exists in
    the corresponding mode.

    Parameters
    ----------
    He3Only_Check : bool
        Required. If true, all returned maps are empty.
    Configs : dict[str, int]
        Required. Configuration label -> representative file number.
    Sample_Bases : Iterable[str]
        Required. Base substrings used to group samples.
    Sample_Names : Iterable[str]
        Required. Sample keys.
    ScattCatalog : dict
        Required. Scattering catalog (consulted for the ``'Intent'`` field).
    AllFullPolSlices, AllHalfPolSlices, AllUnpolSlices : dict
        Required. ``Config -> Sample -> slices`` mappings.

    Returns
    -------
    All_FullPol_BaseToSampleMap : dict
    All_HalfPol_BaseToSampleMap : dict
    All_Unpol_BaseToSampleMap : dict
        Each ``Config -> Base -> [Sample, ...]``.
    """

    All_FullPol_BaseToSampleMap = {}
    All_HalfPol_BaseToSampleMap = {}
    All_Unpol_BaseToSampleMap = {}
    if not He3Only_Check:
        for Config in Configs:
            representative_filenumber = Configs[Config]
            if representative_filenumber != 0:
                FullPol_BaseToSampleMap = {}
                HalfPol_BaseToSampleMap = {}
                Unpol_BaseToSampleMap = {}

                for Base in Sample_Bases:
                    for Sample in Sample_Names:
                        if str(Sample).find(Base) != -1:
                            if Sample in ScattCatalog:                
                                if str(ScattCatalog[Sample]['Intent']).find('Sample') != -1:
                                    if Sample in AllFullPolSlices[Config]:
                                        if Base not in FullPol_BaseToSampleMap:
                                            FullPol_BaseToSampleMap[Base] = [Sample]
                                        else:
                                            FullPol_BaseToSampleMap[Base].append(Sample)
                for Base in Sample_Bases:
                    for Sample in Sample_Names:
                        if str(Sample).find(Base) != -1:
                            if Sample in ScattCatalog:                
                                if str(ScattCatalog[Sample]['Intent']).find('Sample') != -1:
                                    if Sample in AllHalfPolSlices[Config]:
                                        if Base not in HalfPol_BaseToSampleMap:
                                            HalfPol_BaseToSampleMap[Base] = [Sample]
                                        else:
                                            HalfPol_BaseToSampleMap[Base].append(Sample)
                for Base in Sample_Bases:
                    for Sample in Sample_Names:
                        if str(Sample).find(Base) != -1:
                            if Sample in ScattCatalog:                
                                if str(ScattCatalog[Sample]['Intent']).find('Sample') != -1:
                                    if Sample in AllUnpolSlices[Config]:
                                        if Base not in Unpol_BaseToSampleMap:
                                            Unpol_BaseToSampleMap[Base] = [Sample]
                                        else:
                                            Unpol_BaseToSampleMap[Base].append(Sample)
                                            
            All_FullPol_BaseToSampleMap[Config] = FullPol_BaseToSampleMap
            All_HalfPol_BaseToSampleMap[Config] = HalfPol_BaseToSampleMap
            All_Unpol_BaseToSampleMap[Config] = Unpol_BaseToSampleMap

    return All_FullPol_BaseToSampleMap, All_HalfPol_BaseToSampleMap, All_Unpol_BaseToSampleMap

def _sans_save_comparative_plots(Slices, ScattCatalog, save_path, FullPol_BaseToSampleMap, HalfPol_BaseToSampleMap, Unpol_BaseToSampleMap, AllFullPolSlices, AllHalfPolSlices, AllUnpolSlices, AllFullPolResults, AllHalfPolResults, AllUnpolResults, Configs, He3Only_Check, CompareUnpolCirc, CompareHalfPolSumCirc, CompareFullPolSumCirc, CompareFullPolStruc, CompareFullPolMagnetism):
    """Emit per-base comparison plots/text across samples sharing a base name.

    Dispatches several calls into :func:`_sans_comparison_plots_and_text`
    covering full-pol circular sum, full-pol horizontal NSF sum, half-pol
    circular sum, unpol circular, and full-pol M_Perp / M_Parl_NSF, gated
    on the corresponding ``Compare*`` toggles.

    Parameters
    ----------
    Slices : Iterable[str]
        Required. Slice keys actually computed.
    ScattCatalog : dict
        Required. Scattering catalog (passed through for ``'Intent'`` checks).
    save_path : str
        Required. Output directory.
    FullPol_BaseToSampleMap, HalfPol_BaseToSampleMap, Unpol_BaseToSampleMap : dict
        Required. ``Config -> Base -> [Sample, ...]`` maps from
        :func:`_sans_categorize_samples_and_bases`.
    AllFullPolSlices, AllHalfPolSlices, AllUnpolSlices : dict
        Required. ``Config -> Sample -> slices`` mappings.
    AllFullPolResults, AllHalfPolResults, AllUnpolResults : dict
        Required. ``Config -> Sample -> results`` mappings produced by the
        slice processors.
    Configs : dict[str, int]
        Required. Configuration label -> representative file number.
    He3Only_Check : bool
        Required. If true, skip all plotting.
    CompareUnpolCirc, CompareHalfPolSumCirc, CompareFullPolSumCirc, CompareFullPolStruc, CompareFullPolMagnetism : bool
        Required. Per-comparison toggles.

    Returns
    -------
    None
    """

    if not He3Only_Check:
        for Config in Configs:
            representative_filenumber = Configs[Config]
            if representative_filenumber != 0:

                CompareVariable = CompareFullPolSumCirc
                CutVariable = 'Circ'
                FullCutName = 'FullPolSumCirc'
                BaseMap = FullPol_BaseToSampleMap[Config]
                if Config in AllFullPolSlices and Config in AllFullPolResults:
                    SampleSlices = AllFullPolSlices[Config]
                    ResultsArray = AllFullPolResults[Config]
                    QName = 'QCirc'
                    IName = 'CircSum'
                    UncName = 'CircSum_Unc'
                    _sans_comparison_plots_and_text(save_path, Slices, ScattCatalog, Config, CompareVariable, CutVariable, FullCutName, BaseMap, SampleSlices, ResultsArray, QName, IName, UncName)

                CompareVariable = CompareFullPolStruc
                CutVariable = 'Horz'
                FullCutName = 'FullPolNSFHorzSum'
                BaseMap = FullPol_BaseToSampleMap[Config]
                if Config in AllFullPolResults and Config in AllFullPolSlices:
                    SampleSlices = AllFullPolSlices[Config]
                    ResultsArray = AllFullPolResults[Config]
                    QName = 'QHorz'
                    IName = 'HorzNSFSum'
                    UncName = 'HorzNSFSum_Unc'
                    _sans_comparison_plots_and_text(save_path, Slices, ScattCatalog, Config, CompareVariable, CutVariable, FullCutName, BaseMap, SampleSlices, ResultsArray, QName, IName, UncName)


                CompareVariable = CompareHalfPolSumCirc
                CutVariable = 'Circ'
                FullCutName = 'HalfPolPolSumCirc'
                BaseMap = HalfPol_BaseToSampleMap
                if Config in AllHalfPolSlices and Config in AllHalfPolResults:
                    SampleSlices = AllHalfPolSlices[Config]
                    ResultsArray = AllHalfPolResults[Config]
                    QName = 'QCirc'
                    IName = 'CircSum'
                    UncName = 'CircSum_Unc'
                    _sans_comparison_plots_and_text(save_path, Slices, ScattCatalog, Config, CompareVariable, CutVariable, FullCutName, BaseMap, SampleSlices, ResultsArray, QName, IName, UncName)


                CompareVariable = CompareUnpolCirc
                CutVariable = 'Circ'
                FullCutName = 'UnpolCirc'
                BaseMap = Unpol_BaseToSampleMap
                if Config in AllUnpolSlices and Config in AllUnpolResults:
                    SampleSlices = AllUnpolSlices[Config]
                    ResultsArray = AllUnpolResults[Config]
                    QName = 'QCirc'
                    IName = 'CircSum'
                    UncName = 'CircSum_Unc'
                    _sans_comparison_plots_and_text(save_path, Slices, ScattCatalog, Config, CompareVariable, CutVariable, FullCutName, BaseMap, SampleSlices, ResultsArray, QName, IName, UncName)

                CompareVariable = CompareFullPolMagnetism
                CutVariable = 'Horz'
                FullCutName = 'FullPolM_Perp'
                BaseMap = FullPol_BaseToSampleMap
                if Config in AllFullPolSlices and Config in AllFullPolResults:
                    SampleSlices = AllFullPolSlices[Config]
                    ResultsArray = AllFullPolResults[Config]
                    QName = 'QHorz'
                    IName = 'M_Perp'
                    UncName = 'M_Perp_Unc'
                    _sans_comparison_plots_and_text(save_path, Slices, ScattCatalog, Config, CompareVariable, CutVariable, FullCutName, BaseMap, SampleSlices, ResultsArray, QName, IName, UncName)
              
                CompareVariable = CompareFullPolMagnetism
                CutVariable = 'Horz'
                FullCutName = 'FullPolM_Parl_NSF'
                BaseMap = FullPol_BaseToSampleMap
                if Config in AllFullPolSlices and Config in AllFullPolResults:
                    SampleSlices = AllFullPolSlices[Config]
                    ResultsArray = AllFullPolResults[Config]
                    QName = 'QHorz'
                    IName = 'M_Parl_NSF'
                    UncName = 'M_Parl_NSF_Unc'
                    _sans_comparison_plots_and_text(save_path, Slices, ScattCatalog, Config, CompareVariable, CutVariable, FullCutName, BaseMap, SampleSlices, ResultsArray, QName, IName, UncName)
                    
    return

def _sans_comparison_plots_and_text(save_path, Slices, ScattCatalog, Config, CompareVariable, CutVariable, FullCutName, BaseMap, SampleSlices, ResultsArray, QName, IName, UncName):
    """Overlay one quantity for all samples sharing a base name, save PNG + TXT.

    For each base with at least two qualifying samples, this loops over
    samples, plots ``ResultsArray[Sample][IName]`` vs ``ResultsArray[Sample][QName]``
    with ``UncName`` error bars, and writes a stacked ASCII table.
    No-op if ``CompareVariable`` is not truthy or ``CutVariable`` is not
    in ``Slices``.

    Parameters
    ----------
    save_path : str
        Required. Output directory.
    Slices : Iterable[str]
        Required. Slice keys present in the run.
    ScattCatalog : dict
        Required. Scattering catalog (consulted for ``'Intent'``).
    Config : str
        Required. Configuration label.
    CompareVariable : bool or int
        Required. Master toggle for this comparison.
    CutVariable : str
        Required. Slice key that must be present in ``Slices`` for the
        comparison to run (e.g. ``'Circ'``).
    FullCutName : str
        Required. Label used in titles and file names (e.g.
        ``'FullPolSumCirc'``).
    BaseMap : dict
        Required. ``Base -> [Sample, ...]`` map.
    SampleSlices : dict
        Required. ``Sample -> slices`` map (used as a presence check).
    ResultsArray : dict
        Required. ``Sample -> {QName, IName, UncName, ...}`` results dict.
    QName, IName, UncName : str
        Required. Keys in each per-sample results dict that supply the Q
        array, intensity, and uncertainty respectively.

    Returns
    -------
    None
    """

    plot_symbols = ['b*', 'r*', 'g*', 'c*','m*', 'y*', 'k*','b-', 'r-', 'g-', 'c-','m-', 'y-', 'k-']
    symbol_max = 14        
    if CompareVariable == 1 and CutVariable in Slices:
        for Base in BaseMap:
            symbol_counter = 0
            HaveData = 0
            text_output = {''}
            descrip = ''
            Q_previous = np.array([0, 0, 0])
            counter = 0
            fig = plt.figure()
            ax = plt.axes()
            ax.set_xscale("log")
            ax.set_yscale("log")
            for Sample in BaseMap[Base]:
                if Sample in ScattCatalog:                
                    if 'Sample' in str(ScattCatalog[Sample]['Intent']):
                        if Sample in SampleSlices:
                            HaveData += 1
                            Q = np.array(ResultsArray[Sample][QName])
                            I = np.array(ResultsArray[Sample][IName])
                            DI = np.array(ResultsArray[Sample][UncName])
                            if counter == 0:
                                text_output = [Q]
                                Q_previous = Q
                                text_output.append(I)
                                text_output.append(DI)
                                descrip = descrip + 'Q' + ' '
                            else:
                                if np.array_equal(Q, Q_previous) == False:
                                    text_output.append(Q)
                                text_output.append(I)
                                text_output.append(DI)
                            Condition = Sample
                            Base2 = Base + "_"
                            Condition = Condition.replace(Base2, '')
                            Condition = Condition.replace('_naK', '')
                            if np.array_equal(Q, Q_previous):
                                descrip = descrip + Condition + ' ' + 'Unc' + ' '
                            else:
                                descrip = descrip + 'Q' + ' ' + Condition + ' ' + 'Unc' + ' '
                            counter += 1
                            Q_previous = Q
                            symbol = str(plot_symbols[symbol_counter])
                            ax.errorbar(Q, I, yerr=DI, fmt = symbol, label='{name}, {samp}'.format(name = FullCutName, samp=Sample))
                            symbol_counter += 1
                            if symbol_counter >= symbol_max:
                                symbol_counter = 0
            if HaveData >= 2:
                text_outputII = np.array(text_output)
                text_outputII = text_outputII.T
                file_name = 'Compare_{base},{cf}_{name}.txt'.format(base = Base, cf=Config, name = FullCutName)
                file_path = os.path.join(save_path, file_name)
                np.savetxt(file_path, text_outputII, delimiter = ' ', comments = '', header =  descrip, fmt='%1.4e')

                plt.xlabel('Q')
                plt.ylabel('Intensity')
                plt.title('{name} for_{cf}'.format(name = FullCutName, cf = Config))
                plt.legend()
                file_name = 'Compare_{base},{cf}_{name}.png'.format(base = Base, cf=Config, name = FullCutName)
                file_path = os.path.join(save_path, file_name)
                fig.savefig(file_path)
                #plt.show()
                plt.close()
            else:
                plt.close()
    return

def polarization_correction_pipeline(
    Detector_Panels,
    Instrument,
    SampleDescriptionKeywordsToExclude,
    UsePolCorr,
    YesNoManualHe3Entry,
    input_path,
    save_path,
    HighResMinX,
    HighResMaxX,
    HighResMinY,                       
    HighResMaxY, 
    HighResGain,
    Plex,
    HE3_Cell_Summary,
    Slices,
    Truest_PSM,
    ScattCatalog,
    BlockBeamCatalog,
    Configs,
    Sample_Names,
    Sample_Bases,
    TransCatalog,
    Pol_TransCatalog,
    AlignDet_Trans,
    SectorCutAngles = 15.0, #Default is typically 10.0 to 20.0 (degrees)
    He3CorrectionType = 1, #0 for chi, 1 for chi = upsilon (only active if YesNoManualHe3Entry = True), 2 for upsilon
    YesNoShowPlots = False, #False = No and simply saves plots; True = yes and displays plots when code is run
    StructurallyIsotropic = False,
    Minimum_PSM = 0.01,
    Calc_Q_From_Trans = True, #Default is True for yes; False for no
    AverageQRanges = False, #False for no; True for yes
    CompareUnpolCirc = True,
    CompareHalfPolSumCirc = True,
    CompareFullPolSumCirc = True,
    CompareFullPolStruc = True,
    CompareFullPolMagnetism = True,
    YesNo_2DCombinedFiles = False, #Default is False (no), True = yes which can be read using SasView
    YesNo_2DFilesPerDetector = False, #Default is False (no), True = yes; Note all detectors will be summed after beamline masking applied and can be read by SasView 4.2.2 (and higher?)
    MiddlePixelBorderHorizontal = 4, #Default = 4
    MiddlePixelBorderVertical = 4, #Default = 4
    SampleApertureInMM = True, #Override in case sample aperture entered in cm rather than mm
    UseMTCirc = True, #Default is True for yes, False for no (which instead subtracts sector-by-sector MT from data)
    He3Only_Check = False, #Default False = No (runs full reduction), True = Yes (for helium team's use)
    Absolute_Q_min = 0.005, #Default 0; Will take the maximum of Q_min_Calc from all detectors and this value
    Absolute_Q_max = 0.12,
    YesNoSetPlotXRange = False, #Default is False (no), True = yes
    YesNoSetPlotYRange = False, #Default is False (no), True = yes
    PlotXmin = 0.015, #Only used if YesNoSetPlotXRange = True
    PlotXmax = 0.115, #Only used if YesNoSetPlotXRange = True
    PlotYmin = 1E-4, #Only used if YesNoSetPlotYRange = True
    PlotYmax = 1, #Only used if YesNoSetPlotYRange = True
    AutoSubtractEmpty = True,
    ConvertHighResToSubset = True):
    """Run the end-to-end polarization reduction pipeline.

    Composes :func:`_sans_make_slices_and_save_ascii`,
    :func:`_sans_save_slices_and_results`,
    :func:`_sans_categorize_samples_and_bases`, and
    :func:`_sans_save_comparative_plots` to take raw catalogs through
    absolute scaling, polarization correction, 1-D slicing, optional MT
    subtraction, and cross-sample comparison plots.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names (e.g. ``['FR','FL','MR','ML',...]``).
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    SampleDescriptionKeywordsToExclude : list[str] or None
        Required. Keywords stripped from sample descriptions; ``None`` is
        treated as an empty list.
    UsePolCorr : bool or int
        Required. If truthy, apply the full polarization-correction matrix
        inversion; otherwise apply only the He3-efficiency correction.
    YesNoManualHe3Entry : bool
        Required. Use manually supplied 3He values rather than NeXus entries.
    input_path : str
        Required. Directory containing raw NeXus files.
    save_path : str
        Required. Output directory for ASCII and PNG files.
    HighResMinX, HighResMaxX, HighResMinY, HighResMaxY : int
        Required. High-resolution back-detector pixel bounds.
    HighResGain : float
        Required. Gain factor for the high-resolution back detector.
    Plex : dict[str, np.ndarray]
        Required. Per-panel plex / efficiency arrays.
    HE3_Cell_Summary : dict
        Required. 3He cell parameter summary.
    Slices : Iterable[str]
        Required. Slice keys to compute
        (``'Circ'``/``'Horz'``/``'Vert'``/``'Diag'``).
    Truest_PSM : float
        Required. Best-known supermirror polarization.
    ScattCatalog, BlockBeamCatalog, TransCatalog, Pol_TransCatalog : dict
        Required. The four catalogs driving the reduction.
    Configs : dict[str, int]
        Required. Configuration label -> representative file number; a
        value of 0 skips that configuration.
    Sample_Names : Iterable[str]
        Required. Sample keys to process.
    Sample_Bases : Iterable[str]
        Required. Base substrings used for cross-sample comparison plots.
    AlignDet_Trans : dict
        Required. Aligned-transmission catalog.
    SectorCutAngles : float, optional
        Sector half-width in degrees (default 15.0).
    He3CorrectionType : {0, 1, 2}, optional
        Selects the polarization-efficiency matrix form (default 1 = chi
        equals upsilon).
    YesNoShowPlots : bool, optional
        If true, display generated plots in addition to saving them
        (default ``False``).
    StructurallyIsotropic : bool, optional
        Use horizontal NSF sum as the denominator in the magnetic
        decomposition when true; vertical otherwise (default ``False``).
    Minimum_PSM : float, optional
        Floor applied to measured supermirror polarization (default 0.01).
    Calc_Q_From_Trans : bool, optional
        Refine beam center using transmission files (default ``True``).
    AverageQRanges : bool, optional
        Average overlapping carriage Q bins instead of trimming
        (default ``False``).
    CompareUnpolCirc, CompareHalfPolSumCirc, CompareFullPolSumCirc, CompareFullPolStruc, CompareFullPolMagnetism : bool, optional
        Toggles for per-base comparison plots (each default ``True``).
    YesNo_2DCombinedFiles : bool, optional
        Write combined 2-D ASCII files for SasView (default ``False``).
    YesNo_2DFilesPerDetector : bool, optional
        Write one 2-D ASCII file per detector panel (default ``False``).
    MiddlePixelBorderHorizontal, MiddlePixelBorderVertical : int, optional
        Width (in pixels) of the masked border on middle-carriage detectors
        (each default 4).
    SampleApertureInMM : bool, optional
        Convert sample-aperture values from mm to cm when true
        (default ``True``).
    UseMTCirc : bool, optional
        Use the empty-cell circular slice as the MT source for Horz/Vert
        subtraction (default ``True``).
    He3Only_Check : bool, optional
        Run only the 3He transmission summary, skip full reduction
        (default ``False``).
    Absolute_Q_min, Absolute_Q_max : float, optional
        Hard Q-range cap in inverse Angstroms
        (defaults 0.005 and 0.12).
    YesNoSetPlotXRange, YesNoSetPlotYRange : bool, optional
        Toggle manual plot axis limits (each default ``False``).
    PlotXmin, PlotXmax, PlotYmin, PlotYmax : float, optional
        Manual axis limits used only when toggled on
        (defaults 0.015, 0.115, 1e-4, 1).
    AutoSubtractEmpty : bool, optional
        Subtract the empty-cell scattering when an ``'Empty'`` entry is
        available (default ``True``).
    ConvertHighResToSubset : bool, optional
        Crop the back detector to the high-res pixel bounds
        (default ``True``).

    Returns
    -------
    None
    """


    #This is where the polarization correction is applied,
    # and where the final reduced slices are made and saved as ASCII files for plotting in 
    # Origin or other software and for reading into SasView.
    (AllFullPolSlices, AllHalfPolSlices, AllUnpolSlices) = _sans_make_slices_and_save_ascii(
        YesNoShowPlots = YesNoShowPlots,
        Detector_Panels = Detector_Panels, 
        Instrument = Instrument, 
        SampleApertureInMM = SampleApertureInMM,
        SampleDescriptionKeywordsToExclude = SampleDescriptionKeywordsToExclude, 
        UsePolCorr = UsePolCorr, 
        YesNoManualHe3Entry = YesNoManualHe3Entry, 
        input_path = input_path, 
        save_path = save_path, 
        He3CorrectionType = He3CorrectionType, 
        YesNo_2DFilesPerDetector = YesNo_2DFilesPerDetector, 
        YesNo_2DCombinedFiles = YesNo_2DCombinedFiles, 
        Absolute_Q_min = Absolute_Q_min, 
        Absolute_Q_max = Absolute_Q_max, 
        AverageQRanges = AverageQRanges, 
        Calc_Q_From_Trans = Calc_Q_From_Trans, 
        HighResMinX = HighResMinX, 
        HighResMaxX = HighResMaxX, 
        HighResMinY = HighResMinY, 
        HighResMaxY = HighResMaxY, 
        ConvertHighResToSubset = ConvertHighResToSubset, 
        HighResGain = HighResGain, 
        HE3_Cell_Summary = HE3_Cell_Summary, 
        Plex = Plex, 
        Truest_PSM = Truest_PSM, 
        Minimum_PSM = Minimum_PSM, 
        AlignDet_Trans = AlignDet_Trans,
        He3Only_Check = He3Only_Check, 
        ScattCatalog = ScattCatalog, 
        BlockBeamCatalog = BlockBeamCatalog, 
        Configs = Configs, 
        Sample_Names = Sample_Names, 
        TransCatalog = TransCatalog, 
        Pol_TransCatalog = Pol_TransCatalog, 
        MiddlePixelBorderHorizontal = MiddlePixelBorderHorizontal, 
        MiddlePixelBorderVertical = MiddlePixelBorderVertical, 
        SectorCutAngles = SectorCutAngles, 
        Slices = Slices,
        YesNoSetPlotXRange = YesNoSetPlotXRange,
        YesNoSetPlotYRange = YesNoSetPlotYRange, 
        PlotXmin = PlotXmin, 
        PlotXmax = PlotXmax, 
        PlotYmin = PlotYmin, 
        PlotYmax = PlotYmax)
    
    #This is where the slices and completed and saved.
    (AllFullPolResults, 
    AllHalfPolResults, 
    AllUnpolResults) = _sans_save_slices_and_results(
        StructurallyIsotropic = StructurallyIsotropic,
        Slices = Slices, 
        SectorCutAngles = SectorCutAngles, 
        save_path = save_path, 
        YesNoShowPlots = YesNoShowPlots, 
        YesNoSetPlotXRange = YesNoSetPlotXRange, 
        YesNoSetPlotYRange = YesNoSetPlotYRange, 
        PlotXmin = PlotXmin, 
        PlotXmax = PlotXmax, 
        PlotYmin = PlotYmin, 
        PlotYmax = PlotYmax, 
        AutoSubtractEmpty = AutoSubtractEmpty, 
        UseMTCirc = UseMTCirc, 
        He3Only_Check = He3Only_Check, 
        Configs = Configs, 
        Sample_Names = Sample_Names, 
        ScattCatalog = ScattCatalog, 
        AllFullPolSlices = AllFullPolSlices, 
        AllHalfPolSlices = AllHalfPolSlices, 
        AllUnpolSlices = AllUnpolSlices
        )
    
    #Categorize samples and bases for comparisons.
    (FullPol_BaseToSampleMap, 
    HalfPol_BaseToSampleMap, 
    Unpol_BaseToSampleMap) = _sans_categorize_samples_and_bases(He3Only_Check, 
                                    Configs = Configs, 
                                    Sample_Bases = Sample_Bases, 
                                    Sample_Names = Sample_Names, 
                                    ScattCatalog = ScattCatalog, 
                                    AllFullPolSlices = AllFullPolSlices, 
                                    AllHalfPolSlices = AllHalfPolSlices, 
                                    AllUnpolSlices = AllUnpolSlices)

    #Create comparative plots of the categorized samples.
    _sans_save_comparative_plots(Slices = Slices,
                               ScattCatalog = ScattCatalog,
                               save_path = save_path,
                               FullPol_BaseToSampleMap = FullPol_BaseToSampleMap, 
                               HalfPol_BaseToSampleMap = HalfPol_BaseToSampleMap, 
                               Unpol_BaseToSampleMap = Unpol_BaseToSampleMap, 
                               AllFullPolSlices = AllFullPolSlices, 
                               AllHalfPolSlices = AllHalfPolSlices, 
                               AllUnpolSlices = AllUnpolSlices, 
                               AllFullPolResults = AllFullPolResults, 
                               AllHalfPolResults = AllHalfPolResults, 
                               AllUnpolResults = AllUnpolResults, 
                               Configs = Configs, 
                               He3Only_Check = He3Only_Check, 
                               CompareUnpolCirc = CompareUnpolCirc, 
                               CompareHalfPolSumCirc = CompareHalfPolSumCirc, 
                               CompareFullPolSumCirc = CompareFullPolSumCirc, 
                               CompareFullPolStruc = CompareFullPolStruc, 
                               CompareFullPolMagnetism = CompareFullPolMagnetism)


    
    return




