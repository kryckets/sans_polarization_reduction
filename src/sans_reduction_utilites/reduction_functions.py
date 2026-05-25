import os

import numpy as np
import h5py
import dateutil
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json


def _sans_instrument_selection(Instrument = 'VSANS'):
    """Resolve the detector-panel layout and default slice keys for an instrument.

    Parameters
    ----------
    Instrument : str, optional
        Instrument identifier; must contain ``'VSANS'`` or ``'NG7SANS'``
        (default ``'VSANS'``).

    Returns
    -------
    Detector_Panels : list[str]
        Short panel names for the instrument (e.g. ``['MT','MB','MR',...]``
        for VSANS, ``['Full_Panel']`` for NG7SANS).
    TransPanel : str
        Short name of the panel used to integrate transmissions
        (``'MR'`` for VSANS, ``'Full_Panel'`` for NG7SANS).
    Slices : list[str]
        Default slice keys (``['Vert','Horz','Diag','Circ']``).
    """

    Slices = ["Vert", "Horz", "Diag", "Circ"]
    if 'VSANS' in Instrument:
        Detector_Panels = ["MT", "MB", "MR", "ML", "FT", "FB", "FR", "FL"]
        TransPanel = 'MR'
    elif 'NG7SANS' in Instrument:
        Detector_Panels = ["Full_Panel"]
        TransPanel = 'Full_Panel'
    else:
        Detector_Panels = []
        TransPanel = 'NA'
        print('No instrument assigned!')

    return Detector_Panels, TransPanel, Slices

def _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, filenumber):
    """Open the NeXus file for ``filenumber`` and return an h5py handle.

    Returns ``None`` if the file does not exist. The caller is responsible
    for closing the returned handle (or using it inside a ``with``-block).

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names (kept for API parity; not used here).
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``; selects the file extension
        (``.nxs.ngv`` vs ``.nxs.ng7``).
    input_path : str
        Required. Directory containing the raw NeXus files.
    filenumber : int
        Required. Run number used to build the file name.

    Returns
    -------
    h5py.File or None
    """
    if 'VSANS' in Instrument:
        filename = "sans" + str(filenumber) + ".nxs.ngv"
    elif 'NG7SANS' in Instrument:
        filename = "sans" + str(filenumber) + ".nxs.ng7"
    fullpath = os.path.join(input_path, filename)
    if os.path.isfile(fullpath):
        return h5py.File(fullpath, 'r')
    else:
        return None


def _sans_sample_base_name_descrip(Detector_Panels, Instrument, SampleDescriptionKeywordsToExclude, input_path, filenumber):
    """Derive sample names and metadata from the NeXus file description string.

    Reads the raw sample description, strips out polarization tags
    (``T_UU``, ``S_DU``, etc.), the configuration string, the temperature
    and voltage substrings, and any user-supplied keywords, and returns
    both the bare base name and a name decorated with measured voltage
    and/or temperature.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Forwarded to :func:`_sans_get_by_filenumber`.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    SampleDescriptionKeywordsToExclude : Iterable[str]
        Required. Substrings to strip from the description.
    input_path : str
        Required. Directory containing raw NeXus files.
    filenumber : int
        Required. Run number to read.

    Returns
    -------
    Sample_Base : str
        Bare sample name with all decorations removed.
    Sample_Name : str
        Sample name decorated with voltage and/or temperature when those
        DAS log entries are present.
    Descrip : str
        Cleaned-up raw description string.
    Listed_Config : str
        Configuration label from ``DAS_logs/configuration/key``.
    Desired_Temp : str
        Desired primary-node temperature (default ``'295.0'``).
    Voltage : str
        Adam4021 voltage if present (default ``'na'``).
    """

    record_temp = 0
    record_adam4021 = 0
  
    f = _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, filenumber)
    if f is not None:
        Descrip = 'Instrument not assigned'
        if 'VSANS' in Instrument:
            Descrip = str(f['entry/sample/description'][0])
        elif 'NG7SANS' in Instrument:
            Descrip = str(f['/entry/DAS_logs/sample/description'][()])
        Descrip = Descrip[2:]
        Descrip = Descrip[:-1]
        Descrip = Descrip.replace("'", '')
        Descrip = Descrip.replace("|", '')
        
        Listed_Config = str(f['entry/DAS_logs/configuration/key'][0])
        Listed_Config = Listed_Config[2:]
        Listed_Config = Listed_Config[:-1]
        
        Sample_Name = Descrip.replace(Listed_Config, '')
        Not_Sample = ['T_UU', 'T_DU', 'T_DD', 'T_UD', 'T_SM', 'T_NP', 'HeIN', 'HeOUT', 'S_UU', 'S_DU', 'S_DD', 'S_UD', 'S_NP', 'S_HeU', 'S_HeD', 'S_SMU', 'S_SMD']
        for i in Not_Sample:
            Sample_Name = Sample_Name.replace(i, '')
        Desired_Temp = '295.0'
        if "temp" in f['entry/DAS_logs/']:
            Desired_Temp = str(f['entry/DAS_logs/temp/desiredPrimaryNode'][(0)])
            record_temp = 1    
        Voltage = 'na'
        if "adam4021" in f['entry/DAS_logs/']:
            Voltage = str(f['entry/DAS_logs/adam4021/voltage'][(0)])
            record_adam4021 = 1

        for keyword in SampleDescriptionKeywordsToExclude:
            Sample_Name = Sample_Name.replace(keyword, '')
            
        DT5 = Desired_Temp + " K,"
        DT4 = Desired_Temp + " K"
        DT3 = Desired_Temp + "K,"
        DT2 = Desired_Temp + "K"
        DT1 = Desired_Temp
        V5 = Voltage + " V,"
        V4 = Voltage + " V"
        V3 = Voltage + "V,"
        V2 = Voltage + "V"
        V1 = Voltage
        TempShort = Desired_Temp
        TempShort = TempShort.replace('.0', '')
        VShort = Voltage
        VShort = VShort.replace('.0', '')
        DT6 = TempShort + " K,"
        DT7 = TempShort + " K"
        V6 = VShort + " V,"
        V7 = VShort + " V"
        Z = "/"
        Not_Sample = [DT5, DT4, DT3, DT2, DT1, DT6, DT7, V5, V4, V3, V2, V1, V6, V7, Z]
        for i in Not_Sample:
            Sample_Name = Sample_Name.replace(i, '')
        Sample_Name = Sample_Name.replace(' ', '')
        Sample_Base = Sample_Name
        if record_adam4021 == 0 and record_temp == 0:
            Sample_Name = Sample_Name
        elif record_adam4021 == 1 and record_temp == 0:
            Sample_Name = Sample_Name + '_' + str(Voltage) + 'V'
        elif record_adam4021 == 0 and record_temp == 1:
            Sample_Name = Sample_Name + '_' + str(Desired_Temp) + 'K'
        else:
            Sample_Name = Sample_Name + '_' + str(Voltage) + 'V_' + str(Desired_Temp) + 'K'
        f.close()

    return Sample_Base, Sample_Name, Descrip, Listed_Config, Desired_Temp, Voltage

def _sans_purpose_intent_polarization_solenoid(Detector_Panels, Instrument, UsePolCorr, input_path, filenumber):
    """Classify a run by purpose, intent, polarization state, and solenoid state.

    For VSANS, reads dedicated DAS log entries; for NG7SANS, falls back to
    keyword sniffing on the sample description. The combined front + back
    flipper directions are mapped to one of ``'UNPOL'``, ``'Front_U'``,
    ``'Front_D'``, ``'Back_U'``, ``'Back_D'``, ``'UU'``, ``'DU'``, ``'DD'``,
    ``'UD'``. When ``UsePolCorr`` is true and the description ends in
    a polarized tag (``S_UU``, ``T_DU``, ...), that tag overrides the
    state.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Forwarded to :func:`_sans_get_by_filenumber`.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    UsePolCorr : bool or int
        Required. If truthy, allow the description tag to override the
        polarization state.
    input_path : str
        Required. Directory containing raw NeXus files.
    filenumber : int
        Required. Run number to read.

    Returns
    -------
    SiMirror : str
        ``'IN'``/``'OUT'`` or ``'UNKNOWN'``.
    Purpose : str
        One of ``'SCATTERING'``, ``'TRANSMISSION'``, ``'HE3'``,
        ``'Empty'``, ``'Blocked'``, or ``'UNKNOWN'``.
    Intent : str
        ``'Sample'``, ``'Empty'``, ``'Blocked'``, ``'Open'``, or
        ``'UNKNOWN'`` (with ``' Beam'`` stripped).
    PolarizationState : str
        Polarization tag as described above.
    FrontPolDirection : str
        Raw front flipper direction (``'UP'``/``'DOWN'``/``'UNPOLARIZED'``).
    BackPolDirection : str
        Raw back flipper direction.
    SolenoidPosition : str
        ``'IN'`` if any back-pol direction is set, else ``'OUT'``.
    """

    SiMirror = 'UNKNOWN'
    Purpose = 'UNKNOWN'
    Intent = 'UNKNOWN'
    FrontPolDirection = 'UNKNOWN'
    BackPolDirection = 'UNKNOWN'
    SolenoidPosition = 'UNKNOWN'
    PolarizationState = 'UNKNOWN'  
    f = _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, filenumber)
    if f is not None:
        if 'VSANS' in Instrument:
            SiMirror = str(f['entry/DAS_logs/siMirror/siMirror'][()]) #SCATTERING, TRANSMISSION, HE3
            SiMirror = SiMirror[3:-2]
            Purpose = str(f['entry/reduction/file_purpose'][()]) #SCATTERING, TRANSMISSION, HE3
            Purpose = Purpose[3:-2]
            Intent = str(f['entry/reduction/intent'][()]) #Sample, Empty, Blocked Beam, Open Beam (will manually remove the 'Beam')
            Intent = Intent[3:-2]
            Intent = Intent.replace(' Beam', '')
        else: #if NG7SANS (assuming older format)
            SiMirror = "OUT"
            Descrip = str(f['/entry/DAS_logs/sample/description'][()])
            if 'HeOUT' in Descrip or 'HeIN' in Descrip:
                Purpose = 'HE3'
            elif 'TRANS' in Descrip or 'Trans' in Descrip:
                Purpose = 'TRANSMISSION'
            elif 'SCATT' in Descrip or 'Scatt' in Descrip:
                Purpose = 'SCATTERING'
            Intent = "Sample"
            if 'Empty' in Descrip or 'EMPTY' in Descrip:
                Purpose = 'Empty'
            if 'BLOCK' in Descrip or 'Block' in Descrip:
                Purpose = 'Blocked'
                
        if "frontPolarization" in f['entry/DAS_logs/']:
            FrontPolDirection = str(f['entry/DAS_logs/frontPolarization/direction'][()])
            FrontPolDirection = FrontPolDirection[3:-2]
        else:
            FrontPolDirection = 'UNPOLARIZED'
        if "backPolarization" in f['entry/DAS_logs/']:
            BackPolDirection = str(f['entry/DAS_logs/backPolarization/direction'][()])
            BackPolDirection = BackPolDirection[3:-2]
        else:
            BackPolDirection = 'UNPOLARIZED'
        if 'UP' in BackPolDirection or 'DOWN' in BackPolDirection:
            SolenoidPosition = 'IN'
        else:
            SolenoidPosition = 'OUT'
        if 'UNPOL' in FrontPolDirection and 'UNPOL' in BackPolDirection:
            PolarizationState = 'UNPOL'
        elif 'UP' in FrontPolDirection and 'UNPOL' in BackPolDirection:
            PolarizationState = 'Front_U'
        elif 'DOWN' in FrontPolDirection and 'UNPOL' in BackPolDirection:
            PolarizationState = 'Front_D'
        elif 'UNPOL' in FrontPolDirection and 'UP' in BackPolDirection:
            PolarizationState = 'Back_U'
        elif 'UNPOL' in FrontPolDirection and 'DOWN' in BackPolDirection:
            PolarizationState = 'Back_D'
        elif 'UP' in FrontPolDirection and 'UP' in BackPolDirection:
            PolarizationState = 'UU'
        elif 'DOWN' in FrontPolDirection and 'DOWN' in BackPolDirection:
            PolarizationState = 'DD'
        elif 'UP' in FrontPolDirection and 'DOWN' in BackPolDirection:
            PolarizationState = 'UD'
        elif 'DOWN' in FrontPolDirection and 'UP' in BackPolDirection:
            PolarizationState = 'DU'

        if UsePolCorr:
            Type = str(f['entry/sample/description'][()])
            if Type[-6:-2] == 'S_UU' or Type[-6:-2] == 'T_UU':
                PolarizationState = 'UU'
            elif Type[-6:-2] == 'S_DU' or Type[-6:-2] == 'T_DU':
                PolarizationState = 'DU'
            elif Type[-6:-2] == 'S_DD' or Type[-6:-2] == 'T_DD':
                PolarizationState = 'DD'
            elif Type[-6:-2] == 'S_UD' or Type[-6:-2] == 'T_UD':
                PolarizationState = 'UD'

        f.close()

    return SiMirror, Purpose, Intent, PolarizationState, FrontPolDirection, BackPolDirection, SolenoidPosition


def _sans_config_id(Detector_Panels, Instrument, input_path, filenumber):
    """Build a short configuration identifier from instrument geometry.

    For VSANS, encodes guides, front/middle carriage distances and
    wavelength (e.g. ``'5Gd450cmF1650cmM6Ang'``); ``'CvB'`` (converging
    beam) is substituted for the guide count when the guide log contains
    ``'CONV'``. For NG7SANS, encodes guides, detector distance and
    wavelength.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Forwarded to :func:`_sans_get_by_filenumber`.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    input_path : str
        Required. Directory containing raw NeXus files.
    filenumber : int
        Required. Run number to read.

    Returns
    -------
    Configuration_ID : str
        Short configuration label (``'UNKNOWN'`` if the wavelength entry
        is not finite or the file is missing).
    """

    Configuration_ID = 'UNKNOWN'
    f = _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, filenumber)
    if f is not None:
        if np.isfinite(f['entry/DAS_logs/wavelength/wavelength'][0]):
            WV = str(f['entry/DAS_logs/wavelength/wavelength'][0])
            Wavelength = WV[:3]
            GuideHolder = f['entry/DAS_logs/guide/guide'][0]
            if 'VSANS' in Instrument:
                Desired_FrontCarriage_Distance = int(f['entry/DAS_logs/carriage1Trans/desiredSoftPosition'][0]) #in cm
                Desired_MiddleCarriage_Distance = int(f['entry/DAS_logs/carriage2Trans/desiredSoftPosition'][0]) #in cm
                if "CONV" in str(GuideHolder):
                    Guides =  "CvB"
                else:
                    GuideNum = int(f['entry/DAS_logs/guide/guide'][0])
                    Guides = str(GuideNum) 
                Configuration_ID = str(Guides) + "Gd" + str(Desired_FrontCarriage_Distance) + "cmF" + str(Desired_MiddleCarriage_Distance) + "cmM" + str(Wavelength) + "Ang"
            elif 'NG7SANS' in Instrument:
                GuideNum = int(f['entry/DAS_logs/guide/guide'][0])
                Guides = str(GuideNum) 
                Desired_Distance = int(f['entry/DAS_logs/detectorPosition/desiredSoftPosition'][0]) #NG7 Change, in cm
                Configuration_ID = str(Guides) + "Gd" + str(Desired_Distance) + "cm" + str(Wavelength) + "Ang"
        f.close()

    return Configuration_ID

def _sans_sort_data_automatic(Detector_Panels, input_path, Instrument='VSANS', UsePolCorr=True, SampleDescriptionKeywordsToExclude=None, TransPanel=None, YesNoManualHe3Entry=False, New_HE3_Files=None, MuValues=None, TeValues=None, Excluded_Filenumbers=None, Min_Filenumber=0, Max_Filenumber=1000000, Min_Scatt_Filenumber=0, Max_Scatt_Filenumber=1000000, Min_Trans_Filenumber=0, Max_Trans_Filenumber=1000000, ReAssignBlockBeamIntent=None, ReAssignEmptyIntent=None, ReAssignOpenIntent=None, ReAssignSampleIntent=None, YesNoRenameEmpties=True):
    """Walk ``input_path`` and build catalogs of every relevant SANS run.

    For each file in range, classifies it via :func:`_sans_config_id` and
    :func:`_sans_purpose_intent_polarization_solenoid` and accumulates it
    into one of the per-purpose catalogs (block-beam, scatt, trans,
    pol-trans, align-det-trans, 3He-trans).

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names for the instrument.
    input_path : str
        Required. Directory containing raw NeXus files.
    Instrument : str, optional
        ``'VSANS'`` or ``'NG7SANS'`` (default ``'VSANS'``).
    UsePolCorr : bool, optional
        Forwarded to :func:`sans_purpose_intent_polarization_solenoid`
        (default ``True``).
    SampleDescriptionKeywordsToExclude : list[str] or None, optional
        Keywords stripped from sample descriptions (default ``None`` ->
        empty list).
    TransPanel : str or None, optional
        Transmission panel; derived from ``Instrument`` if ``None``
        (default ``None``).
    YesNoManualHe3Entry : bool, optional
        If true, take cell parameters from ``New_HE3_Files``/``MuValues``/
        ``TeValues`` rather than NeXus entries (default ``False``).
    New_HE3_Files : list[int] or None, optional
        Manually flagged 3He cell-load files (default ``None`` -> empty).
    MuValues : list[float] or None, optional
        Manual ``Mu`` values, one per cell load (default ``None`` -> empty).
    TeValues : list[float] or None, optional
        Manual ``Te`` values, one per cell load (default ``None`` -> empty).
    Excluded_Filenumbers : list[int] or None, optional
        Files to skip entirely (default ``None`` -> empty).
    Min_Filenumber, Max_Filenumber : int, optional
        Global file-number bounds (defaults 0 and 1000000).
    Min_Scatt_Filenumber, Max_Scatt_Filenumber : int, optional
        Bounds restricting which scattering runs are kept (defaults 0 and
        1000000).
    Min_Trans_Filenumber, Max_Trans_Filenumber : int, optional
        Bounds restricting which transmission runs are kept (defaults 0
        and 1000000).
    ReAssignBlockBeamIntent, ReAssignEmptyIntent, ReAssignOpenIntent, ReAssignSampleIntent : list[int] or None, optional
        Override the auto-detected ``Intent`` for the listed file numbers
        (each default ``None`` -> empty).
    YesNoRenameEmpties : bool, optional
        Rename any ``Intent == 'Empty'`` runs to ``Sample_Name = 'Empty'``
        (default ``True``).

    Returns
    -------
    Sample_Names : list[str]
    Sample_Bases : list[str]
    Configs : dict[str, int]
        ``Config -> representative file number``.
    BlockBeam : dict
    Scatt : dict
    Trans : dict
    Pol_Trans : dict
    AlignDet_Trans : dict
    HE3_Trans : dict
    start_number : int
        First file number that passed all filters.
    FileNumberList : list[int]
        All file numbers ingested (always starts with 0).
    """
    
    # Set defaults for None parameters from module-level values
    if SampleDescriptionKeywordsToExclude is None:
        SampleDescriptionKeywordsToExclude = []
    if New_HE3_Files is None:
        New_HE3_Files = []
    if MuValues is None:
        MuValues = []
    if TeValues is None:
        TeValues = []
    if Excluded_Filenumbers is None:
        Excluded_Filenumbers = []
    if ReAssignBlockBeamIntent is None:
        ReAssignBlockBeamIntent = []
    if ReAssignEmptyIntent is None:
        ReAssignEmptyIntent = []
    if ReAssignOpenIntent is None:
        ReAssignOpenIntent = []
    if ReAssignSampleIntent is None:
        ReAssignSampleIntent = []
    if TransPanel is None:
        # Derive TransPanel from Instrument
        if 'VSANS' in Instrument:
            TransPanel = 'MR'
        elif 'NG7SANS' in Instrument:
            TransPanel = 'Full_Panel'
    
    Sample_Names = {}
    Sample_Bases = {}
    Configs = {}
    BlockBeam = {}
    Scatt = {}
    Trans = {}
    Pol_Trans = {}
    AlignDet_Trans = {}
    HE3_Trans = {}
    start_number = 0
    FileNumberList = [0]
    Configs = {}

    UU_filenumber = -10
    DU_filenumber = -10
    DD_filenumber = -10
    UD_filenumber = -10
    filenames = '0'
    CellIdentifier = 0
    HE3OUT_filenumber = -10

    if 'VSANS' in Instrument:
        filelist = [fn for fn in os.listdir(input_path) if fn.endswith(".nxs.ngv")] #or filenames = [fn for fn in os.listdir("./") if os.path.isfile(fn)]
    elif 'NG7SANS' in Instrument:
        filelist = [fn for fn in os.listdir(input_path) if fn.endswith(".nxs.ng7")] #or filenames = [fn for fn in os.listdir("./") if os.path.isfile(fn)]
    filelist.sort()
    if len(filelist) >= 1:
        for filename in filelist:
            filenumber = int(filename[4:9])
            if filenumber >= Min_Filenumber and filenumber <= Max_Filenumber and filenumber not in Excluded_Filenumbers:
                if start_number == 0:
                    start_number = filenumber
                f = _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, filenumber)
                if f is not None:
                    Sample_Base, Sample_Name, Descrip, ListedConfig, Temp, Voltage = _sans_sample_base_name_descrip(Detector_Panels, Instrument, SampleDescriptionKeywordsToExclude, input_path, filenumber)
                    SiMirror, Purpose, Intent, PolarizationState, FrontPolDirection, BackPolDirection, SolenoidPosition = _sans_purpose_intent_polarization_solenoid(Detector_Panels, Instrument, UsePolCorr, input_path, filenumber)

                    if filenumber in ReAssignBlockBeamIntent:
                        Intent = 'Block'
                    if filenumber in  ReAssignEmptyIntent:
                        Intent = 'Empty'
                    if filenumber in ReAssignOpenIntent:
                        Intent = 'Open'
                    if filenumber in ReAssignSampleIntent:
                        Intent = 'Sample'
                    if YesNoRenameEmpties and 'Empty' in Intent:
                        Sample_Base = 'Empty'
                        Sample_Name = 'Empty'
                    
                    Config = _sans_config_id(Detector_Panels, Instrument, input_path, filenumber)
                    Count_time = f['entry/collection_time'][0]
                    End_time = dateutil.parser.parse(f['entry/end_time'][0])
                    TimeOfMeasurement = (End_time.timestamp() - Count_time/2)/3600.0 #in hours
                    if filenumber not in Excluded_Filenumbers and 'UNKNOWN' not in Config and Count_time > 29: #and str(Descrip).find("Align") == -1 and str(Descrip).find("align") == -1:
                        #print('Reading:', filenumber, ' ', Sample_Base, Descrip)
                        FileNumberList.append(filenumber)
                        if Config not in Configs and 'SCATT' in Purpose and 'Block' not in Intent:
                            Configs[Config] = filenumber

                        if 'Block' in Intent:
                            if Config not in BlockBeam:
                                BlockBeam[Config] = {'Scatt':{'File' : 'NA'}, 'Trans':{'File' : 'NA', 'CountsPerSecond' : 'NA'}, 'ExampleFile' : filenumber}
                            Trans_Counts = f['entry/instrument/detector_{ds}/integrated_count'.format(ds=TransPanel)][0]
                            if 'TRANS' in Purpose or 'HE3' in Purpose:
                                if 'NA' in BlockBeam[Config]['Trans']['File']:
                                    BlockBeam[Config]['Trans']['File'] = [filenumber]
                                    BlockBeam[Config]['Trans']['CountsPerSecond'] = [Trans_Counts/Count_time]
                                else:
                                    BlockBeam[Config]['Trans']['File'].append(filenumber)
                                    BlockBeam[Config]['Trans']['CountsPerSecond'].append(Trans_Counts/Count_time)
                            elif 'SCATT' in Purpose:
                                if 'NA' in BlockBeam[Config]['Scatt']['File']:
                                    BlockBeam[Config]['Scatt']['File'] = [filenumber]
                                else:
                                    BlockBeam[Config]['Scatt']['File'].append(filenumber)


                        elif 'SCATT' in Purpose and filenumber >= Min_Scatt_Filenumber and filenumber <= Max_Scatt_Filenumber:
                            if len(Sample_Names) < 1:
                                Sample_Names = [Sample_Name]
                            else:
                                if Sample_Name not in Sample_Names:
                                    Sample_Names.append(Sample_Name)
                            if len(Sample_Bases) < 1:
                                Sample_Bases = [Sample_Base]
                            else:
                                if Sample_Base not in Sample_Bases:
                                    Sample_Bases.append(Sample_Base)

                                
                            if Sample_Name not in Scatt:
                                Scatt[Sample_Name] = {'Temp' : Temp, 'Intent': Intent, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'SiMirror' : 'OUT', 'Unpol': 'NA', 'U' : 'NA', 'D' : 'NA','UU' : 'NA', 'DU' : 'NA', 'DD' : 'NA', 'UD' : 'NA', 'UU_Time' : 'NA', 'DU_Time' : 'NA', 'DD_Time' : 'NA', 'UD_Time' : 'NA'}}}
                            if Config not in Scatt[Sample_Name]['Config(s)']:
                                Scatt[Sample_Name]['Config(s)'][Config] = {'SiMirror' : 'OUT', 'Unpol': 'NA', 'U' : 'NA', 'D' : 'NA','UU' : 'NA', 'DU' : 'NA', 'DD' : 'NA', 'UD' : 'NA', 'UU_Time' : 'NA', 'DU_Time' : 'NA', 'DD_Time' : 'NA', 'UD_Time' : 'NA'}

                            if SiMirror != 'OUT':
                                Scatt[Sample_Name]['Config(s)'][Config]['SiMirror'] = 'IN'
                        
                            if 'UNPOL' in PolarizationState:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['Unpol']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['Unpol'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['Unpol'].append(filenumber)
                            if 'Front_U' in PolarizationState:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['U']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['U'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['U'].append(filenumber)
                            if 'Front_D' in PolarizationState:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['D']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['D'] = [filenumber]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['D'].append(filenumber)
                            if 'UU' in PolarizationState:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['UU']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['UU'] = [filenumber]
                                    Scatt[Sample_Name]['Config(s)'][Config]['UU_Time'] = [TimeOfMeasurement]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['UU'].append(filenumber)
                                    Scatt[Sample_Name]['Config(s)'][Config]['UU_Time'].append(TimeOfMeasurement)
                            if 'DU' in PolarizationState:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['DU']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['DU'] = [filenumber]
                                    Scatt[Sample_Name]['Config(s)'][Config]['DU_Time'] = [TimeOfMeasurement]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['DU'].append(filenumber)
                                    Scatt[Sample_Name]['Config(s)'][Config]['DU_Time'].append(TimeOfMeasurement)
                            if 'DD' in PolarizationState:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['DD']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['DD'] = [filenumber]
                                    Scatt[Sample_Name]['Config(s)'][Config]['DD_Time'] = [TimeOfMeasurement]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['DD'].append(filenumber)
                                    Scatt[Sample_Name]['Config(s)'][Config]['DD_Time'].append(TimeOfMeasurement)
                            if 'UD' in PolarizationState:
                                if 'NA' in Scatt[Sample_Name]['Config(s)'][Config]['UD']:
                                    Scatt[Sample_Name]['Config(s)'][Config]['UD'] = [filenumber]
                                    Scatt[Sample_Name]['Config(s)'][Config]['UD_Time'] = [TimeOfMeasurement]
                                else:
                                    Scatt[Sample_Name]['Config(s)'][Config]['UD'].append(filenumber)
                                    Scatt[Sample_Name]['Config(s)'][Config]['UD_Time'].append(TimeOfMeasurement)

                        elif 'TRANS' in Purpose and 'FR' in ListedConfig and filenumber >= Min_Trans_Filenumber and filenumber <= Max_Trans_Filenumber:
                            if Sample_Name not in AlignDet_Trans:
                                AlignDet_Trans[Sample_Name] = {'Temp' : Temp, 'Intent': Intent, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}}}
                            if Config not in AlignDet_Trans[Sample_Name]['Config(s)']:
                                AlignDet_Trans[Sample_Name]['Config(s)'][Config] = {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}
                            if 'UNPOL' in PolarizationState:
                                if 'NA' in AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Unpol_Files']:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Unpol_Files'] = [filenumber]
                                else:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Unpol_Files'].append(filenumber)
                            if 'Front_U' in PolarizationState or 'Front_D' in PolarizationState or 'UU' in PolarizationState or 'DD' in PolarizationState:
                                if 'NA' in AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Pol_Files']:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Pol_Files'] = [filenumber]
                                else:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Pol_Files'].append(filenumber)

                        elif 'TRANS' in Purpose and 'NG7SANS' in Instrument and filenumber >= Min_Trans_Filenumber and filenumber <= Max_Trans_Filenumber:
                            if Sample_Name not in AlignDet_Trans:
                                AlignDet_Trans[Sample_Name] = {'Temp' : Temp, 'Intent': Intent, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}}}
                            if Config not in AlignDet_Trans[Sample_Name]['Config(s)']:
                                AlignDet_Trans[Sample_Name]['Config(s)'][Config] = {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}
                            if 'UNPOL' in PolarizationState:
                                if 'NA' in AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Unpol_Files']:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Unpol_Files'] = [filenumber]
                                else:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Unpol_Files'].append(filenumber)
                            if 'Front_U' in PolarizationState or 'Front_D' in PolarizationState or 'UU' in PolarizationState or 'DD' in PolarizationState:
                                if 'NA' in AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Pol_Files']:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Pol_Files'] = [filenumber]
                                else:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['FR_Pol_Files'].append(filenumber)

                        if 'TRANS' in Purpose and 'FR' not in ListedConfig and filenumber >= Min_Trans_Filenumber and filenumber <= Max_Trans_Filenumber:
                            if Sample_Name not in AlignDet_Trans:
                                AlignDet_Trans[Sample_Name] = {'Temp' : Temp, 'Intent': Intent, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}}}
                            if Config not in AlignDet_Trans[Sample_Name]['Config(s)']:
                                AlignDet_Trans[Sample_Name]['Config(s)'][Config] = {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}
                            if 'UNPOL' in PolarizationState:
                                if 'NA' in AlignDet_Trans[Sample_Name]['Config(s)'][Config]['MR_Unpol_Files']:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['MR_Unpol_Files'] = [filenumber]
                                else:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['MR_Unpol_Files'].append(filenumber)
                            if 'Front_U' in PolarizationState or 'Front_D' in PolarizationState or 'UU' in PolarizationState or 'DD' in PolarizationState:
                                if 'NA' in AlignDet_Trans[Sample_Name]['Config(s)'][Config]['MR_Pol_Files']:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['MR_Pol_Files'] = [filenumber]
                                else:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config]['MR_Pol_Files'].append(filenumber)
                                    
                            if Sample_Name not in Trans:
                                Trans[Sample_Name] = {'Intent': Intent, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files' : 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}}}
                            if Config not in Trans[Sample_Name]['Config(s)']:
                                Trans[Sample_Name]['Config(s)'][Config] = {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files': 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}
                            if Sample_Name not in Pol_Trans:
                                Pol_Trans[Sample_Name] = {'T_UU' : {'File' : 'NA'},
                                                              'T_DU' : {'File' : 'NA'},
                                                              'T_DD' : {'File' : 'NA'},
                                                              'T_UD' : {'File' : 'NA'},
                                                              'T_SM' : {'File' : 'NA'},
                                                              'Config' : 'NA', 'Temp' : Temp, 'Voltage': Voltage, 'Sample_Base': Sample_Base}
                            if 'UNPOL' in PolarizationState:
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files']:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files'].append(filenumber)
                            if 'Front_U' in PolarizationState:
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['U_Files']:
                                    Trans[Sample_Name]['Config(s)'][Config]['U_Files'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['U_Files'].append(filenumber)
                            if 'Front_D' in PolarizationState:
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['D_Files']:
                                    Trans[Sample_Name]['Config(s)'][Config]['D_Files'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['D_Files'].append(filenumber)
                            if 'UU' in PolarizationState:
                                UU_filenumber = filenumber
                                UU_Time = TimeOfMeasurement
                            if 'DU' in PolarizationState:
                                DU_filenumber = filenumber
                                DU_Time = TimeOfMeasurement
                            if 'DD' in PolarizationState:
                                DD_filenumber = filenumber
                                DD_Time = TimeOfMeasurement
                            if 'UD' in PolarizationState:
                                UD_filenumber = filenumber
                                UD_Time = TimeOfMeasurement
                            if 'Front_U' in PolarizationState:
                                SM_filenumber = filenumber
                                if SM_filenumber - UU_filenumber == 4:
                                    if 'NA' in Pol_Trans[Sample_Name]['T_UU']['File']:
                                        Pol_Trans[Sample_Name]['T_UU']['File'] = [UU_filenumber]
                                        Pol_Trans[Sample_Name]['T_UU']['Meas_Time'] = [UU_Time]
                                        Pol_Trans[Sample_Name]['T_DU']['File'] = [DU_filenumber]
                                        Pol_Trans[Sample_Name]['T_DU']['Meas_Time'] = [DU_Time]
                                        Pol_Trans[Sample_Name]['T_DD']['File'] = [DD_filenumber]
                                        Pol_Trans[Sample_Name]['T_DD']['Meas_Time'] = [DD_Time]
                                        Pol_Trans[Sample_Name]['T_UD']['File'] = [UD_filenumber]
                                        Pol_Trans[Sample_Name]['T_UD']['Meas_Time'] = [UD_Time]
                                        Pol_Trans[Sample_Name]['T_SM']['File'] = [SM_filenumber]
                                        Pol_Trans[Sample_Name]['Config'] = [Config]
                                    else:
                                        Pol_Trans[Sample_Name]['T_UU']['File'].append(UU_filenumber)
                                        Pol_Trans[Sample_Name]['T_UU']['Meas_Time'].append(UU_Time)
                                        Pol_Trans[Sample_Name]['T_DU']['File'].append(DU_filenumber)
                                        Pol_Trans[Sample_Name]['T_DU']['Meas_Time'].append(DU_Time)
                                        Pol_Trans[Sample_Name]['T_DD']['File'].append(DD_filenumber)
                                        Pol_Trans[Sample_Name]['T_DD']['Meas_Time'].append(DD_Time)
                                        Pol_Trans[Sample_Name]['T_UD']['File'].append(UD_filenumber)
                                        Pol_Trans[Sample_Name]['T_UD']['Meas_Time'].append(UD_Time)
                                        Pol_Trans[Sample_Name]['T_SM']['File'].append(SM_filenumber)
                                        Pol_Trans[Sample_Name]['Config'].append(Config)

                        elif 'HE3' in Purpose:
                            if YesNoManualHe3Entry:
                                if filenumber in New_HE3_Files:
                                    ScaledOpacity = MuValues[CellIdentifier]
                                    TE = TeValues[CellIdentifier]
                                    CellTimeIdentifier = TimeOfMeasurement
                                    HE3Insert_Time = TimeOfMeasurement
                                    CellIdentifier += 1
                                    CellName = CellTimeIdentifier
                            else:
                                CellTimeIdentifier = f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                                CellName = str(f['entry/DAS_logs/backPolarization/name'][0])
                                CellName = CellName[2:]
                                CellName = CellName[:-1]
                                CellName = CellName + str(CellTimeIdentifier)
                                if CellTimeIdentifier not in HE3_Trans:
                                    HE3Insert_Time = f['/entry/DAS_logs/backPolarization/timestamp'][0]/3600000 #milliseconds to hours
                                    Opacity = f['/entry/DAS_logs/backPolarization/opacityAt1Ang'][0]
                                    Wavelength = f['/entry/DAS_logs/wavelength/wavelength'][0]
                                    ScaledOpacity = Opacity*Wavelength
                                    TE = f['/entry/DAS_logs/backPolarization/glassTransmission'][0]
                            HE3Type = str(f['entry/sample/description'][()])
                            if 'OUT' in HE3Type:
                                if Sample_Name not in Trans:
                                    Trans[Sample_Name] = {'Intent': Intent, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files' : 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}}}
                                if Config not in Trans[Sample_Name]['Config(s)']:
                                    Trans[Sample_Name]['Config(s)'][Config] = {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files': 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}
                                if 'NA' in Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files']:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files'] = [filenumber]
                                else:
                                    Trans[Sample_Name]['Config(s)'][Config]['Unpol_Files'].append(filenumber)

                                if Sample_Name not in AlignDet_Trans:
                                    AlignDet_Trans[Sample_Name] = {'Temp' : Temp, 'Intent': Intent, 'Sample_Base': Sample_Base, 'Config(s)' : {Config : {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}}}
                                if Config not in AlignDet_Trans[Sample_Name]['Config(s)']:
                                    AlignDet_Trans[Sample_Name]['Config(s)'][Config] = {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}
                                if 'UNPOL' in PolarizationState:
                                    if 'NA' in AlignDet_Trans[Sample_Name]['Config(s)'][Config]['MR_Unpol_Files']:
                                        AlignDet_Trans[Sample_Name]['Config(s)'][Config]['MR_Unpol_Files'] = [filenumber]
                                    else:
                                        AlignDet_Trans[Sample_Name]['Config(s)'][Config]['MR_Unpol_Files'].append(filenumber)
                                   
                                HE3OUT_filenumber = filenumber
                                HE3OUT_config = Config
                                HE3OUT_sample = Sample_Name
                                if 'VSANS' in Instrument:
                                    HE3OUT_attenuators = int(f['entry/instrument/attenuator/num_atten_dropped'][0])
                                elif 'NG7SANS' in Instrument:
                                    HE3OUT_attenuators = int(f['/entry/DAS_logs/counter/actualAttenuatorsDropped'][0])
                            elif 'IN' in HE3Type:
                                HE3IN_filenumber = filenumber
                                HE3IN_config = Config
                                HE3IN_sample = Sample_Name
                                if 'VSANS' in Instrument:
                                    HE3IN_attenuators = int(f['entry/instrument/attenuator/num_atten_dropped'][0])
                                elif 'NG7SANS' in Instrument:
                                    HE3IN_attenuators = int(f['/entry/DAS_logs/counter/actualAttenuatorsDropped'][0])
                                HE3IN_StartTime = TimeOfMeasurement
                                if HE3OUT_filenumber > 0:
                                    if HE3OUT_config == HE3IN_config and HE3OUT_attenuators == HE3IN_attenuators and HE3OUT_sample == HE3IN_sample: #This implies that you must have a 3He out before 3He in of same config and atten
                                        if HE3Insert_Time not in HE3_Trans:
                                            HE3_Trans[CellTimeIdentifier] = {'Te' : TE,
                                                                         'Mu' : ScaledOpacity,
                                                                         'Insert_time' : HE3Insert_Time}
                                        Elasped_time = HE3IN_StartTime - HE3Insert_Time
                                        if "Elasped_time" not in HE3_Trans[CellTimeIdentifier]:
                                            HE3_Trans[CellTimeIdentifier]['Config'] = [HE3IN_config]
                                            HE3_Trans[CellTimeIdentifier]['HE3_OUT_file'] = [HE3OUT_filenumber]
                                            HE3_Trans[CellTimeIdentifier]['HE3_IN_file'] = [HE3IN_filenumber]
                                            HE3_Trans[CellTimeIdentifier]['Elasped_time'] = [Elasped_time]
                                            HE3_Trans[CellTimeIdentifier]['Cell_name'] = [CellName]
                                        else:
                                            HE3_Trans[CellTimeIdentifier]['Config'].append(HE3IN_config)
                                            HE3_Trans[CellTimeIdentifier]['HE3_OUT_file'].append(HE3OUT_filenumber)
                                            HE3_Trans[CellTimeIdentifier]['HE3_IN_file'].append(HE3IN_filenumber)
                                            HE3_Trans[CellTimeIdentifier]['Elasped_time'].append(Elasped_time)
                                            HE3_Trans[CellTimeIdentifier]['Cell_name'].append(CellName)
                    f.close()

    return Sample_Names, Sample_Bases, Configs, BlockBeam, Scatt, Trans, Pol_Trans, AlignDet_Trans, HE3_Trans, start_number, FileNumberList

def sans_share_align_det_trans_catalog(TempDiffAllowedForSharingTrans=20.0, AlignDet_Trans=None, Scatt=None):
    """Fill in missing alignment transmissions by borrowing from sibling samples.

    For every (sample, config) entry in the scattering catalog that has no
    matching aligned-transmission entry, this seeds the entry with empty
    placeholders. Then, for any panel/polarization slot still set to
    ``'NA'``, it looks for another sample with the same base name and a
    temperature within ``TempDiffAllowedForSharingTrans`` K, and copies
    that sample's transmission file number.

    Parameters
    ----------
    TempDiffAllowedForSharingTrans : float, optional
        Maximum temperature difference (K) between two samples for the
        donor's transmission to be reused (default 20.0).
    AlignDet_Trans : dict or None, optional
        Aligned-transmission catalog to mutate (default ``None`` -> ``{}``).
    Scatt : dict or None, optional
        Scattering catalog used to seed missing entries (default ``None``
        -> ``{}``).

    Returns
    -------
    AlignDet_Trans : dict
        Updated aligned-transmission catalog.
    """
    # Set defaults for None parameters
    if AlignDet_Trans is None:
        AlignDet_Trans = {}
    if Scatt is None:
        Scatt = {}
    
    for Sample in Scatt:
        for Config in Scatt[Sample]['Config(s)']:
            if Sample not in AlignDet_Trans:
                Intent2 = Scatt[Sample]['Intent']
                Base2 = Scatt[Sample]['Sample_Base']
                Temp2 = Scatt[Sample]['Temp']
                AlignDet_Trans[Sample] = {'Temp': Temp2, 'Intent': Intent2, 'Sample_Base': Base2, 'Config(s)' : {Config : {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}}}
            else:
                if Config not in AlignDet_Trans[Sample]['Config(s)']:
                    AlignDet_Trans[Sample]['Config(s)'][Config] = {'FR_Unpol_Files': 'NA', 'FR_Pol_Files' : 'NA', 'MR_Unpol_Files': 'NA', 'MR_Pol_Files' : 'NA'}


    for Sample in AlignDet_Trans:
        Base = AlignDet_Trans[Sample]['Sample_Base']
        Temp = AlignDet_Trans[Sample]['Temp']
        if 'Config(s)' in AlignDet_Trans[Sample]:
            for Config in AlignDet_Trans[Sample]['Config(s)']:
                if 'NA' in AlignDet_Trans[Sample]['Config(s)'][Config]['FR_Unpol_Files']:
                    for Sample2 in AlignDet_Trans:
                        Base2 = AlignDet_Trans[Sample2]['Sample_Base']
                        Temp2 = AlignDet_Trans[Sample2]['Temp']
                        Temp_Diff = np.sqrt(np.power(float(Temp) - float(Temp2),2))
                        if Base2 == Base and Temp_Diff <= TempDiffAllowedForSharingTrans and Sample2 != Sample:
                            if 'Config(s)' in AlignDet_Trans[Sample2]:
                                for Config2 in AlignDet_Trans[Sample2]['Config(s)']:
                                    if Config2 == Config:
                                        if 'NA' not in AlignDet_Trans[Sample2]['Config(s)'][Config2]['FR_Unpol_Files']:
                                            AlignDet_Trans[Sample]['Config(s)'][Config]['FR_Unpol_Files'] = [AlignDet_Trans[Sample2]['Config(s)'][Config2]['FR_Unpol_Files'][0]]
                if 'NA' in AlignDet_Trans[Sample]['Config(s)'][Config]['FR_Pol_Files']:
                    for Sample2 in AlignDet_Trans:
                        Base2 = AlignDet_Trans[Sample2]['Sample_Base']
                        Temp2 = AlignDet_Trans[Sample2]['Temp']
                        Temp_Diff = np.sqrt(np.power(float(Temp) - float(Temp2),2))
                        if Base2 == Base and Temp_Diff <= TempDiffAllowedForSharingTrans and Sample2 != Sample:
                            if 'Config(s)' in AlignDet_Trans[Sample2]:
                                for Config2 in AlignDet_Trans[Sample2]['Config(s)']:
                                    if Config2 == Config:
                                        if 'NA' not in AlignDet_Trans[Sample2]['Config(s)'][Config2]['FR_Pol_Files']:
                                            AlignDet_Trans[Sample]['Config(s)'][Config]['FR_Pol_Files'] = [AlignDet_Trans[Sample2]['Config(s)'][Config2]['FR_Pol_Files'][0]]
                if 'NA' in AlignDet_Trans[Sample]['Config(s)'][Config]['MR_Unpol_Files']:
                    for Sample2 in AlignDet_Trans:
                        Base2 = AlignDet_Trans[Sample2]['Sample_Base']
                        Temp2 = AlignDet_Trans[Sample2]['Temp']
                        Temp_Diff = np.sqrt(np.power(float(Temp) - float(Temp2),2))
                        if Base2 == Base and Temp_Diff <= TempDiffAllowedForSharingTrans and Sample2 != Sample:
                            if 'Config(s)' in AlignDet_Trans[Sample2]:
                                for Config2 in AlignDet_Trans[Sample2]['Config(s)']:
                                    if Config2 == Config:
                                        if 'NA' not in AlignDet_Trans[Sample2]['Config(s)'][Config2]['MR_Unpol_Files']:
                                            AlignDet_Trans[Sample]['Config(s)'][Config]['MR_Unpol_Files'] = [AlignDet_Trans[Sample2]['Config(s)'][Config2]['MR_Unpol_Files'][0]]
                if 'NA' in AlignDet_Trans[Sample]['Config(s)'][Config]['MR_Pol_Files']:
                    for Sample2 in AlignDet_Trans:
                        Base2 = AlignDet_Trans[Sample2]['Sample_Base']
                        Temp2 = AlignDet_Trans[Sample2]['Temp']
                        Temp_Diff = np.sqrt(np.power(float(Temp) - float(Temp2),2))
                        if Base2 == Base and Temp_Diff <= TempDiffAllowedForSharingTrans and Sample2 != Sample:
                            if 'Config(s)' in AlignDet_Trans[Sample2]:
                                for Config2 in AlignDet_Trans[Sample2]['Config(s)']:
                                    if Config2 == Config:
                                        if 'NA' not in AlignDet_Trans[Sample2]['Config(s)'][Config2]['MR_Pol_Files']:
                                            AlignDet_Trans[Sample]['Config(s)'][Config]['MR_Pol_Files'] = [AlignDet_Trans[Sample2]['Config(s)'][Config2]['MR_Pol_Files'][0]]
    return AlignDet_Trans

def sans_share_sample_base_trans_catalog(Trans=None, Scatt=None):
    """Fill in missing scaling transmissions by borrowing within the same base.

    Mirrors :func:`sans_share_align_det_trans_catalog` for the scaling
    transmission catalog: any (sample, config) lacking ``Unpol_Files`` or
    ``U_Files`` is back-filled with the first matching entry from another
    sample sharing the same ``Sample_Base`` in the same configuration.
    Temperature is not considered here.

    Parameters
    ----------
    Trans : dict or None, optional
        Transmission catalog to mutate (default ``None`` -> ``{}``).
    Scatt : dict or None, optional
        Scattering catalog used to seed missing entries (default ``None``
        -> ``{}``).

    Returns
    -------
    Trans : dict
        Updated transmission catalog.
    """
    # Set defaults for None parameters
    if Trans is None:
        Trans = {}
    if Scatt is None:
        Scatt = {}
    
    for Sample in Scatt:
        for Config in Scatt[Sample]['Config(s)']:
            if Sample not in Trans:
                Intent2 = Scatt[Sample]['Intent']
                Base2 = Scatt[Sample]['Sample_Base']
                Trans[Sample] = {'Intent': Intent2, 'Sample_Base': Base2, 'Config(s)' : {Config : {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files' : 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}}}
            else:
                if Config not in Trans[Sample]['Config(s)']:
                    Trans[Sample]['Config(s)'][Config] = {'Unpol_Files': 'NA', 'U_Files' : 'NA', 'D_Files': 'NA','Unpol_Trans_Cts': 'NA', 'U_Trans_Cts' : 'NA', 'D_Trans_Cts' : 'NA'}
    UnpolBases = {}
    UnpolAssociatedTrans = {}
    UpBases = {}
    UpAssociatedTrans = {}
    for Sample in Trans:
        Base = Trans[Sample]['Sample_Base']
        if 'Config(s)' in Trans[Sample]:
            for Config in Trans[Sample]['Config(s)']:
                if 'NA' not in Trans[Sample]['Config(s)'][Config]['Unpol_Files']:
                    fn = Trans[Sample]['Config(s)'][Config]['Unpol_Files'][0]
                    if Config not in UnpolBases:
                        UnpolBases[Config] = [Base]
                        UnpolAssociatedTrans[Config] = [fn]
                    elif Base not in UnpolBases[Config]:
                        UnpolBases[Config].append(Base)
                        UnpolAssociatedTrans[Config].append(fn)
                if 'NA' not in Trans[Sample]['Config(s)'][Config]['U_Files']:
                    fn = Trans[Sample]['Config(s)'][Config]['U_Files'][0]
                    if Config not in UpBases:
                        UpBases[Config] = [Base]
                        UpAssociatedTrans[Config] = [fn]
                    elif Base not in UpBases[Config]:
                        UpBases[Config].append(Base)
                        UpAssociatedTrans[Config].append(fn)
    for Sample in Trans:
        Base = Trans[Sample]['Sample_Base']
        if 'Config(s)' in Trans[Sample]:
            for Config in Trans[Sample]['Config(s)']:
                if 'NA' in Trans[Sample]['Config(s)'][Config]['Unpol_Files']:
                    if Config in UnpolBases:
                        if Base in UnpolBases[Config]:
                            for i in [i for i,x in enumerate(UnpolBases[Config]) if x == Base]:
                                Trans[Sample]['Config(s)'][Config]['Unpol_Files'] = [UnpolAssociatedTrans[Config][i]]
                if 'NA' in Trans[Sample]['Config(s)'][Config]['U_Files']:
                    if Config in UpBases:
                        if Base in UpBases[Config]:
                            for i in [i for i,x in enumerate(UpBases[Config]) if x == Base]:
                                Trans[Sample]['Config(s)'][Config]['U_Files'] = [UpAssociatedTrans[Config][i]]
    return Trans

def sans_share_empty_polbeam_scatt_catalog(Scatt=None):
    """For empty-cell entries, mirror missing polarized cross-sections.

    On samples whose ``Intent`` contains ``'Empty'``, this fills in any
    missing ``UU``/``DD`` (each from the other) and any missing ``UD``/``DU``
    (each from the other), copying both the file lists and the matching
    measurement times. Half-pol ``U``/``D`` are mirrored similarly (file
    lists only). Useful when only one spin state was measured on the empty
    cell but full-pol reduction requires all four.

    Parameters
    ----------
    Scatt : dict or None, optional
        Scattering catalog to mutate (default ``None`` -> ``{}``).

    Returns
    -------
    Scatt : dict
        Updated scattering catalog.
    """
    # Set defaults for None parameters
    if Scatt is None:
        Scatt = {}
    
    for Sample_Name in Scatt:
        if str(Scatt[Sample_Name]['Intent']).find("Empty") != -1:
            for CF in Scatt[Sample_Name]['Config(s)']:
                if 'NA' in Scatt[Sample_Name]['Config(s)'][CF]['DD'] and 'NA' not in Scatt[Sample_Name]['Config(s)'][CF]['UU']:
                    Scatt[Sample_Name]['Config(s)'][CF]['DD'] = Scatt[Sample_Name]['Config(s)'][CF]['UU']
                    Scatt[Sample_Name]['Config(s)'][CF]['DD_Time'] = Scatt[Sample_Name]['Config(s)'][CF]['UU_Time']
                elif 'NA' in Scatt[Sample_Name]['Config(s)'][CF]['UU'] and 'NA' not in Scatt[Sample_Name]['Config(s)'][CF]['DD']:
                    Scatt[Sample_Name]['Config(s)'][CF]['UU'] = Scatt[Sample_Name]['Config(s)'][CF]['DD']
                    Scatt[Sample_Name]['Config(s)'][CF]['UU_Time'] = Scatt[Sample_Name]['Config(s)'][CF]['DD_Time']
                if 'NA' in Scatt[Sample_Name]['Config(s)'][CF]['UD'] and 'NA' not in Scatt[Sample_Name]['Config(s)'][CF]['DU']:
                    Scatt[Sample_Name]['Config(s)'][CF]['UD'] = Scatt[Sample_Name]['Config(s)'][CF]['DU']
                    Scatt[Sample_Name]['Config(s)'][CF]['UD_Time'] = Scatt[Sample_Name]['Config(s)'][CF]['DU_Time']
                elif 'NA' in Scatt[Sample_Name]['Config(s)'][CF]['DU'] and 'NA' not in Scatt[Sample_Name]['Config(s)'][CF]['UD']:
                    Scatt[Sample_Name]['Config(s)'][CF]['DU'] = Scatt[Sample_Name]['Config(s)'][CF]['UD']
                    Scatt[Sample_Name]['Config(s)'][CF]['DU_Time'] = Scatt[Sample_Name]['Config(s)'][CF]['UD_Time']

                if 'NA' in Scatt[Sample_Name]['Config(s)'][CF]['D'] and 'NA' not in Scatt[Sample_Name]['Config(s)'][CF]['U']:
                    Scatt[Sample_Name]['Config(s)'][CF]['D'] = Scatt[Sample_Name]['Config(s)'][CF]['U']
                elif 'NA' in Scatt[Sample_Name]['Config(s)'][CF]['U'] and 'NA' not in Scatt[Sample_Name]['Config(s)'][CF]['D']:
                    Scatt[Sample_Name]['Config(s)'][CF]['U'] = Scatt[Sample_Name]['Config(s)'][CF]['D']
                    
    return Scatt

def sans_share_pol_trans_catalog(Detector_Panels, Pol_Trans, Scatt, input_path, Instrument = 'VSANS', SampleDescriptionKeywordsToExclude = [], TempDiffAllowedForSharingTrans = 20.0):
    """Fill in missing polarized-transmission entries from sibling samples.

    For every sample with a UU scatt run but no pol-trans entry, this
    seeds a blank ``Pol_Trans`` record. Then, for any sample whose
    ``T_UU`` file is still ``'NA'``, it looks for another sample with the
    same base name, the same voltage, and a temperature within
    ``TempDiffAllowedForSharingTrans`` K that has a complete set of
    ``T_UU``/``T_DU``/``T_DD``/``T_UD`` files, and copies that record
    wholesale.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Forwarded to :func:`_sans_sample_base_name_descrip`.
    Pol_Trans : dict
        Required. Pol-trans catalog to mutate.
    Scatt : dict
        Required. Scattering catalog used to seed missing entries.
    input_path : str
        Required. Directory containing raw NeXus files.
    Instrument : str, optional
        ``'VSANS'`` or ``'NG7SANS'`` (default ``'VSANS'``).
    SampleDescriptionKeywordsToExclude : list[str], optional
        Keywords stripped from sample descriptions (default ``[]``).
    TempDiffAllowedForSharingTrans : float, optional
        Maximum temperature difference (K) for two samples to share a
        polarized transmission (default 20.0).

    Returns
    -------
    Pol_Trans : dict
        Updated pol-trans catalog.
    """

    for Sample in Scatt:
        for Config in Scatt[Sample]['Config(s)']:
            if 'NA' not in Scatt[Sample]['Config(s)'][Config]['UU']:
                filenumber = Scatt[Sample]['Config(s)'][Config]['UU'][0]
                if Sample not in Pol_Trans:
                    Sample_Base, Sample_Name, Descrip, Listed_Config, Desired_Temp, Voltage = _sans_sample_base_name_descrip(Detector_Panels, Instrument, SampleDescriptionKeywordsToExclude, input_path, filenumber)
                    Pol_Trans[Sample_Name] = {'T_UU' : {'File' : 'NA'},
                                              'T_DD' : {'File' : 'NA'},
                                              'T_UD' : {'File' : 'NA'},
                                              'T_SM' : {'File' : 'NA'}, 'Config' : 'NA', 'Temp' : Desired_Temp, 'Voltage': Voltage, 'Sample_Base': Sample_Base}

    for Sample in Pol_Trans:
        Base = Pol_Trans[Sample]['Sample_Base']
        Temp = Pol_Trans[Sample]['Temp']
        Volt = Pol_Trans[Sample]['Voltage']
        if 'T_UU' in Pol_Trans[Sample]:
            if 'NA' in Pol_Trans[Sample]['T_UU']['File']:
                for Sample2 in Pol_Trans:
                    Base2 = Pol_Trans[Sample2]['Sample_Base']
                    Temp2 = Pol_Trans[Sample2]['Temp']
                    Volt2 = Pol_Trans[Sample2]['Voltage']
                    Temp_Diff = np.sqrt(np.power(float(Temp) - float(Temp2),2))
                    if Base2 == Base and Temp_Diff <= TempDiffAllowedForSharingTrans and Sample2 != Sample and Volt2 == Volt:
                        if 'NA' not in Pol_Trans[Sample2]['T_UU']['File'] and 'NA' not in Pol_Trans[Sample2]['T_DU']['File'] and 'NA' not in Pol_Trans[Sample2]['T_DD']['File'] and 'NA' not in Pol_Trans[Sample2]['T_UD']['File']:
                            Pol_Trans[Sample] = Pol_Trans[Sample2]
                            print('Substituting in pol-trans measurements of ', Sample2, 'for ', Sample)
    return Pol_Trans

def sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, trans_filenumber, BBList, DetectorPanel):
    """Compute the absolute transmission of one file under a beam-stop mask.

    Sums the counts inside the transmission mask (built by
    :func:`sans_make_trans_mask`), subtracts a time-scaled block-beam
    background (computed from ``BBList``), and divides by the monitor
    counts and the attenuator transmission (from
    :func:`sans_attenuator_table`).

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    input_path : str
        Required. Directory containing raw NeXus files.
    trans_filenumber : int
        Required. Transmission run number to integrate.
    BBList : Sequence[int] or dict
        Required. Either a list of block-beam file numbers (typical) or
        the per-config block-beam dict from which an ``ExampleFile`` is
        read.
    DetectorPanel : str
        Required. Short panel on which to apply the transmission mask.

    Returns
    -------
    abs_trans : float
        Monitor-normalized, attenuator-corrected transmission.
    abs_trans_unc : float
        Propagated Poisson uncertainty.
    """

    Config = _sans_config_id(Detector_Panels, Instrument, input_path, trans_filenumber)

    relevant_detectors = list(Detector_Panels)
    CvBYesNo = 0
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')
        CvBYesNo = 1

    Mask = sans_make_trans_mask(Detector_Panels, Instrument, input_path, trans_filenumber,Config, DetectorPanel)
    if Config in BBList:
        examplefilenumber = BBList[Config]['ExampleFile']
    else:
        examplefilenumber = 0
    BB, BB_Unc = sans_blocked_beam_counts_per_second_list_of_files(Detector_Panels, Instrument, input_path, BBList, Config, examplefilenumber)
    f = _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, trans_filenumber)
    if f is not None:
        monitor_counts = f['entry/control/monitor_counts'][0]
        count_time = f['entry/collection_time'][0]
        wavelength = f['entry/DAS_logs/wavelength/wavelength'][0]
        attenuation = f['/entry/DAS_logs/counter/actualAttenuatorsDropped'][0]
        attn_trans = sans_attenuator_table(Instrument, wavelength, attenuation)
        abs_trans = 0
        abs_trans_unc = 0
        for dshort in relevant_detectors:
            if 'VSANS' in Instrument:
                data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
            elif 'NG7SANS' in Instrument:
                data = np.array(f['entry/instrument/detector/data'])
            if dshort in BB and dshort in BB_Unc:
                trans = (data - BB[dshort]*count_time)*Mask[dshort]
                unc = np.sqrt(data + BB_Unc[dshort])*Mask[dshort]
            else:
                trans = (data)*Mask[dshort]
                unc = np.sqrt(data)*Mask[dshort]           
            abs_trans += (np.sum(trans)*1E8/monitor_counts)/attn_trans
            abs_trans_unc += (np.sqrt(np.sum(np.power(unc,2)))*1E8/monitor_counts)/attn_trans
        f.close()

    return abs_trans, abs_trans_unc

def sans_make_trans_mask(Detector_Panels, Instrument, input_path, filenumber, Config, DetectorPanel):
    """Build a per-panel mask selecting the direct-beam region for transmission.

    Computes the pixel-to-beam-center radial distance using the geometry
    of ``filenumber`` and sets the mask to 1.0 inside ``1.2 * R_beamstop/2``
    (or ``2.0 * R_beamstop/2`` for converging-beam configurations) on the
    panel matching ``DetectorPanel``. All other panels return zero masks.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    input_path : str
        Required. Directory containing raw NeXus files.
    filenumber : int
        Required. Run number used to read the geometry.
    Config : str
        Required. Configuration label; ``'CvB'`` widens the mask and
        adds detector ``'B'`` to the panel list.
    DetectorPanel : str
        Required. Short panel on which to mark direct-beam pixels.

    Returns
    -------
    mask_it : dict[str, np.ndarray]
        Per-panel float arrays (1.0 inside the beam-stop region on
        ``DetectorPanel``, 0.0 elsewhere).
    """

    mask_it = {}
    relevant_detectors = list(Detector_Panels)
    CvBYesNo = 0
    if str(Config).find('CvB') != -1:
        relevant_detectors.append('B')
        CvBYesNo = 1
    f = _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, filenumber)
    for dshort in relevant_detectors:
        if 'NG7SANS' in Instrument:
            data = np.array(f['entry/instrument/detector/data'])
            mask_it[dshort] = np.zeros_like(data)
            x_pixel_size = f['entry/instrument/detector/x_pixel_size'][0]/10.0
            y_pixel_size = f['entry/instrument/detector/y_pixel_size'][0]/10.0
            beam_center_x = f['entry/instrument/detector/beam_center_x'][0]
            beam_center_y = f['entry/instrument/detector/beam_center_y'][0]
            beamstop_diameter = f['/entry/DAS_logs/beamStop/size'][0] #says beam stop in inches, but it seems to be in cm;
            beamstop_to_detector = f['/entry/DAS_logs/beamStop/detectorDistance'][0] #in cm
            detector_distance = f['entry/instrument/detector/distance'][0] #in cm
            setback = 0
            vertical_offset = 0
            lateral_offset = f['entry/DAS_logs/areaDetector/offset'][0] #in cm?
            realDistX =  x_pixel_size*(1.0)  + lateral_offset
            realDistY =  y_pixel_size*(1.0)
            X, Y = np.indices(data.shape)
            dimX = X
            dimY = Y
            x0_pos =  realDistX - beam_center_x*x_pixel_size + (X)*x_pixel_size #NG7 Change
            y0_pos =  realDistY - beam_center_y*y_pixel_size + (Y)*y_pixel_size #NG7 Change
            R = np.sqrt(np.power(x0_pos, 2) + np.power(y0_pos, 2))
            R_shadow = (beamstop_diameter/2.0)*detector_distance/(detector_distance - beamstop_to_detector) #NG7 Change
            mask_it[dshort][R <= R_shadow] = 1.0 #NG7 Change
            
        elif 'VSANS' in Instrument:
            data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
            mask_it[dshort] = np.zeros_like(data)
            x_pixel_size = f['entry/instrument/detector_{ds}/x_pixel_size'.format(ds=dshort)][0]/10.0
            y_pixel_size = f['entry/instrument/detector_{ds}/y_pixel_size'.format(ds=dshort)][0]/10.0
            beam_center_x = f['entry/instrument/detector_{ds}/beam_center_x'.format(ds=dshort)][0]
            beam_center_y = f['entry/instrument/detector_{ds}/beam_center_y'.format(ds=dshort)][0]
            dimX = f['entry/instrument/detector_{ds}/pixel_num_x'.format(ds=dshort)][0]
            dimY = f['entry/instrument/detector_{ds}/pixel_num_y'.format(ds=dshort)][0]
            if CvBYesNo == 0:
                beamstop_diameter = f['/entry/DAS_logs/C2BeamStop/diameter'][0]/10.0 #beam stop in cm; sits right in front of middle detector?
            else:
                beamstop_diameter = 10.0
            if dshort == 'MT' or dshort == 'MB' or dshort == 'FT' or dshort == 'FB':
                setback = f['entry/instrument/detector_{ds}/setback'.format(ds=dshort)][0]
                vertical_offset = f['entry/instrument/detector_{ds}/vertical_offset'.format(ds=dshort)][0]
                lateral_offset = 0
            else:
                setback = 0
                vertical_offset = 0
                lateral_offset = f['entry/instrument/detector_{ds}/lateral_offset'.format(ds=dshort)][0]
            
            if dshort != 'B':
                coeffs = f['entry/instrument/detector_{ds}/spatial_calibration'.format(ds=dshort)][0][0]/10.0
                panel_gap = f['entry/instrument/detector_{ds}/panel_gap'.format(ds=dshort)][0]/10.0
            if dshort == 'B':
                realDistX =  x_pixel_size*(0.5)
                realDistY =  y_pixel_size*(0.5)
            else:
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
            if dshort == 'B':
                x0_pos =  realDistX - beam_center_x*x_pixel_size + (X)*x_pixel_size 
                y0_pos =  realDistY - beam_center_y*y_pixel_size + (Y)*y_pixel_size
            else:
                x0_pos =  realDistX - beam_center_x + (X)*x_pixel_size 
                y0_pos =  realDistY - beam_center_y + (Y)*y_pixel_size
            R = np.sqrt(np.power(x0_pos, 2) + np.power(y0_pos, 2))
            if dshort == DetectorPanel and CvBYesNo == 0:
                mask_it[dshort][R <= 1.2*beamstop_diameter/2.0] = 1.0
            if dshort == DetectorPanel and CvBYesNo == 1:
                mask_it[dshort][R <= 2.0*beamstop_diameter/2.0] = 1.0

    if f is not None:
        f.close()
    return mask_it

def sans_blocked_beam_counts_per_second_list_of_files(Detector_Panels, Instrument, input_path, filelist, Config, examplefilenumber):
    """Sum block-beam files and return per-panel counts/second arrays.

    Twin of the function with the same name in
    ``polarization_correction_functions.py``: accumulates counts and live
    time across ``filelist``, then divides to obtain a counts-per-second
    map per panel. If ``filelist`` is empty or unusable, arrays of zeros
    matching ``examplefilenumber`` are returned.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    input_path : str
        Required. Directory containing raw NeXus files.
    filelist : Sequence[int]
        Required. Block-beam run numbers to sum.
    Config : str
        Required. Configuration label; ``'CvB'`` triggers inclusion of the
        back detector.
    examplefilenumber : int
        Required. Fallback run number used to size the zero arrays when
        no block-beam files are usable.

    Returns
    -------
    BB_CountsPerSecond : dict[str, np.ndarray]
        Per-panel counts/second.
    BB_Unc : dict[str, np.ndarray]
        Per-pixel uncertainty (sqrt(counts)/seconds).
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
        f = _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, item)
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
        f = _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, examplefilenumber)
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

def sans_attenuator_table(Instrument, wavelength, attenuation):
    """Look up the attenuator transmission factor for an instrument setup.

    Uses hard-coded VSANS and NG7SANS attenuator tables indexed by
    wavelength (Å) and number of dropped attenuators. The result is
    linearly interpolated in wavelength between the two nearest
    tabulated rows. Wavelength is clamped to the tabulated range
    (4.52-19 Å for VSANS, 5-17 Å for NG7SANS; VSANS additionally
    supports two sentinel wavelengths 5300 and 6200000).

    Parameters
    ----------
    Instrument : str
        Required. ``'VSANS'`` or ``'NG7SANS'``.
    wavelength : float
        Required. Wavelength in Angstroms.
    attenuation : int or float
        Required. Number of attenuators dropped (clamped to 0..15 for
        VSANS, 0..10 for NG7SANS).

    Returns
    -------
    Trans : float
        Attenuator transmission factor (1.0 when none are dropped).
    """

    Trans = 1.0
    if 'VSANS' in Instrument:
        if attenuation <= 0:
            attn_index = 0
        elif attenuation >= 15:
            attn_index = 15
        else:
            attn_index = int(attenuation)
        if wavelength < 4.52:
            wavelength = 4.52
        if wavelength > 19.0 and wavelength != 5300 and wavelength != 6200000:
            wavelength = 19.0        
        Attn_Table = {}
        Attn_Table[4.52] = [1,0.446,0.20605,0.094166,0.042092,0.019362,0.0092358,0.0042485,0.002069,0.00096002,0.00045601,0.00021113,9.67E-05,4.55E-05,2.25E-05,1.11E-05]
        Attn_Table[5.01] = [1,0.431,0.19352,0.085922,0.03729,0.016631,0.0076671,0.0034272,0.0016896,0.00075357,0.00034739,0.00015667,6.97E-05,3.25E-05,1.57E-05,8.00E-06]
        Attn_Table[5.5] = [1,0.418,0.18225,0.078184,0.032759,0.014152,0.0063401,0.0027516,0.0013208,0.00057321,0.00025623,0.00011171,4.92E-05,2.27E-05,1.09E-05,5.77E-06]
        Attn_Table[5.99] = [1,0.406,0.17255,0.071953,0.029501,0.01239,0.0054146,0.0022741,0.0010643,0.0004502,0.00019584,8.38E-05,3.61E-05,1.70E-05,8.35E-06,4.65E-06]
        Attn_Table[6.96] = [1,0.382,0.15471,0.06111,0.023894,0.0094621,0.0039362,0.0015706,0.00069733,0.00027963,0.0001166,4.86E-05,2.12E-05,1.06E-05,5.54E-06,3.65E-06]
        Attn_Table[7.94] = [1,0.364,0.14014,0.052552,0.019077,0.0071919,0.0028336,0.0010796,0.00045883,0.00017711,7.14E-05,2.90E-05,1.37E-05,7.78E-06,4.54E-06,3.56E-06]
        Attn_Table[9] = [1,0.34199,0.12617,0.045063,0.015551,0.0055606,0.0020986,0.00075427,0.00031101,0.00011673,0.000045324,1.91E-05,8.51E-06,4.82E-06,2.85E-06,2.14E-06]
        Attn_Table[11] = [1,0.31805,0.10886,0.035741,0.011411,0.0037545,0.0013263,0.00043766,0.00016884,5.99E-05,2.23E-05,9.44E-06,5.57E-06,4.10E-06,2.79E-06,2.46E-06]
        Attn_Table[13] = [1,0.298,0.096286,0.029689,0.0088395,0.0027373,0.00090878,0.00028892,0.00011004,3.88E-05,1.44E-05,6.91E-06,5.28E-06,4.17E-06,2.91E-06,2.76E-06]
        Attn_Table[15] = [1,0.27964,0.085614,0.024762,0.0069407,0.0020229,0.00064044,0.00019568,7.44E-05,2.79E-05,1.10E-05,6.34E-06,5.47E-06,4.89E-06,3.66E-06,3.45E-06]
        Attn_Table[17] = [1,0.26364,0.075577,0.020525,0.0053394,0.0014753,0.00044466,0.00013278,5.40E-05,2.31E-05,1.04E-05,7.36E-06,7.33E-06,6.69E-06,5.20E-06,4.75E-06]
        Attn_Table[19] = [1,0.24614,0.065873,0.016961,0.0040631,0.0010583,0.00031229,9.87E-05,4.85E-05,2.77E-05,1.68E-05,1.47E-05,1.52E-05,1.44E-05,1.26E-05,1.19E-05]
        Attn_Table[5300] = [1,0.429,0.19219,0.085141,0.037122,0.016668,0.0078004,0.0035414,0.0017742,0.0008126,0.00038273,0.00017682,8.12E-05,3.89E-05,1.95E-05,1.00E-05]
        Attn_Table[6200000] = [1,0.4152,0.18249,0.079458,0.034065,0.014849,0.0067964,0.003016,0.001485,0.00066483,0.00030864,0.00014094,6.38E-05,3.02E-05,1.50E-05,7.73E-06]
        if wavelength == 5300:
            Trans = Attn_Table[5300][attn_index]
        elif wavelength == 6200000:
            Trans = Attn_Table[6200000][attn_index]
        else:
            Wavelength_Min = 4.52
            Wavelength_Max = 4.52
            Max_trip = 0
            for i in Attn_Table:
                if wavelength >= i:
                    Wavelength_Min = i
                if wavelength <= i and Max_trip == 0:
                    Wavelength_Max = i
                    Max_trip = 1  
            Trans_MinWave = Attn_Table[Wavelength_Min][attn_index]
            Trans_MaxWave = Attn_Table[Wavelength_Max][attn_index]
            if Wavelength_Max > Wavelength_Min:
                Trans = Trans_MinWave + (wavelength - Wavelength_Min)*(Trans_MaxWave - Trans_MinWave)/(Wavelength_Max - Wavelength_Min)
            else:
                Trans = Trans_MinWave
                
    elif 'NG7SANS' in Instrument:
        if attenuation <= 0:
            attn_index = 0
        elif attenuation >= 10:
            attn_index = 10
        else:
            attn_index = int(attenuation)

        if wavelength < 5.0:
            wavelength = 5.0
        if wavelength > 17.0:
            wavelength = 17.0
                
        Attn_Table = {}
        Attn_Table[5.0] = [1.0, 0.418, 0.189, 0.0784, 0.0328, 0.0139, 5.90E-3, 1.04E-3, 1.90E-4, 3.58E-5, 7.76E-6]
        Attn_Table[6.0] = [1.0, 0.393, 0.167, 0.0651, 0.0256, 0.0101, 4.07E-3, 6.37E-4, 1.03E-4, 1.87E-5, 4.56E-6]
        Attn_Table[7.0] = [1.0, 0.369, 0.148, 0.0541, 0.0200, 7.43E-3, 2.79E-3, 3.85E-4, 5.71E-5, 1.05E-5, 3.25E-6]
        Attn_Table[8.0] = [1.0, 0.347, 0.132, 0.0456, 0.0159, 5.58E-3, 1.99E-3, 2.46E-4, 3.44E-5, 7.00E-6, 7.00E-6]
        Attn_Table[10.0] = [1.0, 0.313, 0.109, 0.0340, 0.0107, 3.42E-3, 1.11E-3, 1.16E-4, 1.65E-5, 1.65E-5, 1.65E-5]    
        Attn_Table[12.0] = [1.0, 0.291, 0.0945, 0.0273, 7.98E-3, 2.36E-3, 7.13E-4, 6.86E-5, 6.86E-5, 6.86E-5, 6.86E-5]    
        Attn_Table[14.0] = [1.0, 0.271, 0.0830, 0.0223, 6.14E-3, 1.70E-3, 4.91E-4, 4.91E-4, 4.91E-4, 4.91E-4, 4.91E-4]
        Attn_Table[17.0] = [1.0, 0.244, 0.0681, 0.0164, 4.09E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3]
        Attn_Table[17.001] = [1.0, 0.244, 0.0681, 0.0164, 4.09E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3, 1.03E-3]

        Wavelength_Min = 5.0
        Wavelength_Max = 5.0
        Max_trip = 0
        for i in Attn_Table:
            if wavelength >= i:
                Wavelength_Min = i
            if wavelength <= i and Max_trip == 0:
                Wavelength_Max = i
                Max_trip = 1
                
        Trans_MinWave = Attn_Table[Wavelength_Min][attn_index]
        Trans_MaxWave = Attn_Table[Wavelength_Max][attn_index]

        if Wavelength_Max > Wavelength_Min:
            Trans = Trans_MinWave + (wavelength - Wavelength_Min)*(Trans_MaxWave - Trans_MinWave)/(Wavelength_Max - Wavelength_Min)
        else:
            Trans = Trans_MinWave
        
            
    return Trans

def sans_process_he3_trans_catalog(Detector_Panels, Instrument='VSANS', input_path=None, HE3_Trans=None, BlockBeam=None, DetectorPanel=None):
    """Compute 3He IN/OUT transmission ratios for every cell-load entry.

    For each (HE3_IN, HE3_OUT) file pair in ``HE3_Trans``, calls
    :func:`sans_calc_abs_trans_block_beam_list` on each and stores the
    ratio (``IN / OUT``) in a new ``'Transmission'`` list on the cell
    entry. Block-beam files are taken from the corresponding configuration
    in ``BlockBeam`` when available (trans files preferred, then scatt).

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str, optional
        ``'VSANS'`` or ``'NG7SANS'`` (default ``'VSANS'``).
    input_path : str or None, optional
        Directory containing raw NeXus files (default ``None``).
    HE3_Trans : dict or None, optional
        3He transmission catalog to mutate (default ``None`` -> ``{}``).
    BlockBeam : dict or None, optional
        Block-beam catalog (default ``None`` -> ``{}``).
    DetectorPanel : str or None, optional
        Panel used to integrate transmissions (default ``None``).

    Returns
    -------
    HE3_Trans : dict
        Input catalog with new ``'Transmission'`` lists added.
    """
    # Set defaults for None parameters
    if HE3_Trans is None:
        HE3_Trans = {}
    if BlockBeam is None:
        BlockBeam = {}
    
    for Cell in HE3_Trans:
        if 'Elasped_time' in HE3_Trans[Cell]:
            counter = 0
            for InFile in HE3_Trans[Cell]['HE3_IN_file']:
                OutFile = HE3_Trans[Cell]['HE3_OUT_file'][counter]
                Config = HE3_Trans[Cell]['Config'][counter]
                BBList = [0]
                if Config in BlockBeam:
                    if 'NA' not in BlockBeam[Config]['Trans']['File']:
                        BBList = BlockBeam[Config]['Trans']['File']
                    elif 'NA' not in BlockBeam[Config]['Scatt']['File']:
                        BBList = BlockBeam[Config]['Scatt']['File']
                IN_trans, IN_trans_unc = sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, InFile, BBList, DetectorPanel)
                OUT_trans, OUT_trans_unc = sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, OutFile, BBList, DetectorPanel)
                trans = IN_trans / OUT_trans
                if 'Transmission' not in HE3_Trans[Cell]:
                    HE3_Trans[Cell]['Transmission'] = [trans]
                else:
                    HE3_Trans[Cell]['Transmission'].append(trans)                
                counter += 1 
    return HE3_Trans

def sans_process_pol_trans_catalog(Detector_Panels, Instrument='VSANS', input_path=None, Pol_Trans=None, BlockBeam=None, DetectorPanel=None):
    """Compute polarized transmissions (UU/DU/DD/UD divided by SM) per sample.

    For each (UU, DU, DD, UD, SM) file tuple in ``Pol_Trans[Samp]``,
    integrates each file via :func:`sans_calc_abs_trans_block_beam_list`
    and stores ``X / SM`` on ``Pol_Trans[Samp]['T_X']['Trans']`` (plus the
    raw SM count on ``T_SM['Trans_Cts']``).

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str, optional
        ``'VSANS'`` or ``'NG7SANS'`` (default ``'VSANS'``).
    input_path : str or None, optional
        Directory containing raw NeXus files (default ``None``).
    Pol_Trans : dict or None, optional
        Pol-trans catalog to mutate (default ``None`` -> ``{}``).
    BlockBeam : dict or None, optional
        Block-beam catalog (default ``None`` -> ``{}``).
    DetectorPanel : str or None, optional
        Panel used to integrate transmissions (default ``None``).

    Returns
    -------
    Pol_Trans : dict
        Input catalog with new ``'Trans'``/``'Trans_Cts'`` lists added.
    """
    # Set defaults for None parameters
    if Pol_Trans is None:
        Pol_Trans = {}
    if BlockBeam is None:
        BlockBeam = {}
    
    #Uses sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, trans_filenumber, BlockBeam, DetectorPanel) which uses
    #sans_make_trans_mask(Detector_Panels, Instrument, input_path, filenumber, Config, DetectorPanel) and
    #sans_blocked_beam_counts_per_second_list_of_files(Detector_Panels, Instrument, input_path, filelist, Config) and 
    #sans_attenuator_table(Instrument, wavelength, attenuation)
    for Samp in Pol_Trans:
        if 'NA' not in Pol_Trans[Samp]['T_UU']['File']:
            counter = 0
            for UUFile in Pol_Trans[Samp]['T_UU']['File']:
                DUFile = Pol_Trans[Samp]['T_DU']['File'][counter]
                DDFile = Pol_Trans[Samp]['T_DD']['File'][counter]
                UDFile = Pol_Trans[Samp]['T_UD']['File'][counter]
                SMFile = Pol_Trans[Samp]['T_SM']['File'][counter]
                Config = Pol_Trans[Samp]['Config'][counter]
                BBList = [0]
                if Config in BlockBeam:
                    if 'NA' not in BlockBeam[Config]['Trans']['File']:
                        BBList = BlockBeam[Config]['Trans']['File']
                    elif 'NA' not in BlockBeam[Config]['Scatt']['File']:
                        BBList = BlockBeam[Config]['Scatt']['File']
                UU_trans, UU_trans_unc = sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, UUFile, BBList, DetectorPanel) #Masking done within this step
                DU_trans, DU_trans_unc = sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, DUFile, BBList, DetectorPanel) #Masking done within this step
                DD_trans, DD_trans_unc = sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, DDFile, BBList, DetectorPanel) #Masking done within this step
                UD_trans, UD_trans_unc = sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, UDFile, BBList, DetectorPanel) #Masking done within this step
                SM_trans, SM_trans_unc = sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, SMFile, BBList, DetectorPanel) #Masking done within this step
                if 'Trans' not in Pol_Trans[Samp]['T_UU']:
                    Pol_Trans[Samp]['T_UU']['Trans'] = [UU_trans/SM_trans]
                    Pol_Trans[Samp]['T_DU']['Trans'] = [DU_trans/SM_trans]
                    Pol_Trans[Samp]['T_DD']['Trans'] = [DD_trans/SM_trans]
                    Pol_Trans[Samp]['T_UD']['Trans'] = [UD_trans/SM_trans]
                    Pol_Trans[Samp]['T_SM']['Trans_Cts'] = [SM_trans]
                else:
                    Pol_Trans[Samp]['T_UU']['Trans'].append(UU_trans/SM_trans)
                    Pol_Trans[Samp]['T_DU']['Trans'].append(DU_trans/SM_trans)
                    Pol_Trans[Samp]['T_DD']['Trans'].append(DD_trans/SM_trans)
                    Pol_Trans[Samp]['T_UD']['Trans'].append(UD_trans/SM_trans)
                    Pol_Trans[Samp]['T_SM']['Trans_Cts'].append(SM_trans)
                counter += 1   
    return Pol_Trans

def sans_process_trans_catalog(Detector_Panels, Instrument='VSANS', input_path=None, Trans=None, BlockBeam=None, DetectorPanel=None):
    """Compute scaling transmission counts for every (sample, config) entry.

    For each ``Unpol_Files`` and ``U_Files`` list in the transmission
    catalog, calls :func:`sans_calc_abs_trans_block_beam_list` to compute
    a block-beam-subtracted transmission, and stores the result on
    ``Unpol_Trans_Cts`` / ``U_Trans_Cts`` lists.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    Instrument : str, optional
        ``'VSANS'`` or ``'NG7SANS'`` (default ``'VSANS'``).
    input_path : str or None, optional
        Directory containing raw NeXus files (default ``None``).
    Trans : dict or None, optional
        Transmission catalog to mutate (default ``None`` -> ``{}``).
    BlockBeam : dict or None, optional
        Block-beam catalog (default ``None`` -> ``{}``).
    DetectorPanel : str or None, optional
        Panel used to integrate transmissions (default ``None``).

    Returns
    -------
    Trans : dict
        Input catalog with new ``'Unpol_Trans_Cts'`` / ``'U_Trans_Cts'``
        lists added.
    """
    # Set defaults for None parameters
    if Trans is None:
        Trans = {}
    if BlockBeam is None:
        BlockBeam = {}
    
    #Uses sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, trans_filenumber, BlockBeam, DetectorPanel) which uses
    #sans_make_trans_mask(Detector_Panels, Instrument, input_path, filenumber, Config, DetectorPanel) and
    #sans_blocked_beam_counts_per_second_list_of_files(Detector_Panels, Instrument, input_path, filelist, Config) and 
    #sans_attenuator_table(Instrument, wavelength, attenuation)
    for Samp in Trans:
        for Config in Trans[Samp]['Config(s)']:
            BBList = [0]
            if Config in BlockBeam:
                if 'NA' not in BlockBeam[Config]['Trans']['File']:
                    BBList = BlockBeam[Config]['Trans']['File']
                elif 'NA' not in BlockBeam[Config]['Scatt']['File']:
                    BBList = BlockBeam[Config]['Scatt']['File']

            if 'NA' not in Trans[Samp]['Config(s)'][Config]['Unpol_Files']:
                for UNF in Trans[Samp]['Config(s)'][Config]['Unpol_Files']:
                    Unpol_trans, Unpol_trans_unc = sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, UNF, BBList, DetectorPanel)
                    if 'NA' in Trans[Samp]['Config(s)'][Config]['Unpol_Trans_Cts']:
                        Trans[Samp]['Config(s)'][Config]['Unpol_Trans_Cts'] = [Unpol_trans]
                    else:
                        Trans[Samp]['Config(s)'][Config]['Unpol_Trans_Cts'].append(Unpol_trans)   
            if 'NA' not in Trans[Samp]['Config(s)'][Config]['U_Files']:
                    for UF in Trans[Samp]['Config(s)'][Config]['U_Files']:
                        Halfpol_trans, Halfpol_trans_unc = sans_calc_abs_trans_block_beam_list(Detector_Panels, Instrument, input_path, UF, BBList, DetectorPanel)
                        if 'NA' in Trans[Samp]['Config(s)'][Config]['U_Trans_Cts']:
                            Trans[Samp]['Config(s)'][Config]['U_Trans_Cts'] = [Halfpol_trans]
                        else:
                            Trans[Samp]['Config(s)'][Config]['U_Trans_Cts'].append(Halfpol_trans)
    return Trans

def plex_file(Detector_Panels, input_path, start_number, Instrument='VSANS', HighResMinX=240, HighResMaxX=474, HighResMinY=667, HighResMaxY=917, ConvertHighResToSubset=True, HighResGain=100.0):
    """Load the detector-efficiency (PLEX) file, or fall back to ones-arrays.

    Looks for a file in ``input_path`` whose name begins with ``PLEX``.
    If found, returns the per-panel detector arrays (optionally cropping
    the ``'B'`` panel to ``[HighRes* bounds]``). If no PLEX file exists,
    falls back to ones-arrays sized from ``start_number``.

    Parameters
    ----------
    Detector_Panels : Iterable[str]
        Required. Short panel names.
    input_path : str
        Required. Directory to search for the PLEX file and as the source
        for fallback geometry.
    start_number : int
        Required. Run number used to size the fallback ones-arrays.
    Instrument : str, optional
        ``'VSANS'`` or ``'NG7SANS'`` (default ``'VSANS'``).
    HighResMinX, HighResMaxX, HighResMinY, HighResMaxY : int, optional
        High-resolution back-detector pixel bounds (defaults 240, 474,
        667, 917).
    ConvertHighResToSubset : bool, optional
        Crop the ``'B'`` panel to those bounds (default ``True``).
    HighResGain : float, optional
        Kept for API parity; not used here (default 100.0).

    Returns
    -------
    filename : str
        Name of the PLEX file used, or ``'0'`` if a fallback was used.
    PlexData : dict[str, np.ndarray]
        Per-panel plex arrays.
    """

    PlexData = {}
    filename = '0'
    Plex_file = [fn for fn in os.listdir(input_path) if fn.startswith("PLEX")]
    if len(Plex_file) >= 1:
        filename = str(Plex_file[0])
    fullpath = os.path.join(input_path, filename)
    if os.path.isfile(fullpath):
        print('Reading in ', filename)
        f = h5py.File(fullpath)
        for dshort in Detector_Panels:
            if 'VSANS' in Instrument:
                data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
            elif 'NG7SANS' in Instrument:
                data = np.array(f['entry/instrument/detector/data'])
            if ConvertHighResToSubset:
                if dshort == 'B':
                    data_subset = data[HighResMinX:HighResMaxX+1,HighResMinY:HighResMaxY+1]
                    PlexData[dshort] = data_subset
                else:
                    PlexData[dshort] = data
            else:
               PlexData[dshort] = data
        f.close()
    else:
        filenumber = start_number
        f = _sans_get_by_filenumber(Detector_Panels, Instrument, input_path, filenumber)
        if f is not None:
            for dshort in Detector_Panels:
                if 'VSANS' in Instrument:
                    datafieldname = 'entry/instrument/detector_{ds}/data'.format(ds=dshort)
                elif 'NG7SANS' in Instrument:
                    datafieldname = 'entry/instrument/detector/data'
                if datafieldname in f:
                    if 'VSANS' in Instrument:
                        data = np.array(f['entry/instrument/detector_{ds}/data'.format(ds=dshort)])
                    elif 'NG7SANS' in Instrument:
                        data = np.array(f['entry/instrument/detector/data'])
                    data_filler = np.ones_like(data)
                else:
                    if 'VSANS' in Instrument:
                        x_size = f['entry/instrument/detector_{ds}/pixel_num_x'.format(ds=dshort)][0]
                        y_size = f['entry/instrument/detector_{ds}/pixel_num_y'.format(ds=dshort)][0]
                    elif 'NG7SANS' in Instrument:
                        x_size = 128
                        y_size = 128
                    data = np.zeros((x_size, y_size))
                    data_filler = np.ones_like(data)
                                          
                if ConvertHighResToSubset:
                    if dshort == 'B':
                        data_subset = data_filler[HighResMinX:HighResMaxX+1,HighResMinY:HighResMaxY+1]
                        PlexData[dshort] = data_subset
                    else:
                        PlexData[dshort] = data_filler
                else:
                    PlexData[dshort] = data_filler
            f.close()
        print('Plex file not found; populated with ones instead')
        print(' ')
            
    return filename, PlexData

def he3_decay_func(t, p, gamma):
    """Exponential 3He polarization decay model ``p * exp(-t / gamma)``.

    Used as the model function for ``scipy.optimize.curve_fit`` inside
    :func:`he3_decay_curves`.

    Parameters
    ----------
    t : float or np.ndarray
        Required. Elapsed time in hours.
    p : float
        Required. Initial atomic polarization.
    gamma : float
        Required. Decay constant in hours.

    Returns
    -------
    float or np.ndarray
    """

    return p * np.exp(-t / gamma)

def he3_decay_curves(save_path, HE3_Trans):
    """Fit each cell's atomic-polarization decay and save diagnostic plots.

    For every cell in ``HE3_Trans``, inverts the measured IN/OUT
    transmissions back to atomic polarization via the cell's ``Mu`` and
    ``Te``, fits :func:`he3_decay_func` (or falls back to ``gamma = 1000``
    when only one point exists), records ``Atomic_P0``, ``Gamma(hours)``,
    derived neutron polarization, and uncertainties in
    ``HE3_Cell_Summary``, and saves two PNGs per cell (atomic-pol fit and
    predicted T_MAJ/T_MIN).

    Parameters
    ----------
    save_path : str
        Required. Output directory for the PNG plots.
    HE3_Trans : dict
        Required. 3He transmission catalog (output of
        :func:`sans_process_he3_trans_catalog`).

    Returns
    -------
    HE3_Cell_Summary : dict
        ``insert_time -> {'Atomic_P0', 'Atomic_P0_Unc', 'Gamma(hours)',
        'Gamma_Unc', 'Mu', 'Te', 'Name', 'Neutron_P0', 'Neutron_P0_Unc'}``.
    """
    HE3_Cell_Summary = {}
    entry_number = 0
    for entry in HE3_Trans:
        entry_number += 1
        Mu = HE3_Trans[entry]['Mu']
        Te = HE3_Trans[entry]['Te']
        xdata = np.array(HE3_Trans[entry]['Elasped_time'])
        trans_data = np.array(HE3_Trans[entry]['Transmission'])
        ydata = np.arccosh(np.array(trans_data)/(np.e**(-Mu)*Te))/Mu
        if xdata.size < 2:
            P0 = ydata[0]
            gamma = 1000.0
            '''#assumes no appreciable time decay until more data obtained'''
            PCell0 = np.tanh(Mu * P0)
            P0_Unc = 'NA'
            gamma_Unc = 'NA'
            PCell0_Unc = 'NA'
        else:
            popt, pcov = curve_fit(he3_decay_func, xdata, ydata)
            P0, gamma = popt
            P0_Unc, gamma_Unc = np.sqrt(np.diag(pcov))
            PCell0 = np.tanh(Mu * P0)
            PCell0_Max = np.tanh(Mu * (P0+P0_Unc))
            PCell0_Min = np.tanh(Mu * (P0-P0_Unc))
            PCell0_Unc = PCell0_Max - PCell0_Min

        PCell0 = np.tanh(Mu * P0)
        PCell0_Max = np.tanh(Mu * (P0+P0_Unc))
        PCell0_Min = np.tanh(Mu * (P0-P0_Unc))
        PCell0_Unc = PCell0_Max - PCell0_Min

        Name = HE3_Trans[entry]['Cell_name'][0]
        HE3_Cell_Summary[HE3_Trans[entry]['Insert_time']] = {'Atomic_P0' : P0, 'Atomic_P0_Unc' : P0_Unc, 'Gamma(hours)' : gamma, 'Gamma_Unc' : gamma_Unc, 'Mu' : Mu, 'Te' : Te, 'Name' : Name, 'Neutron_P0' : PCell0, 'Neutron_P0_Unc' : PCell0_Unc}
        print('He3Cell Summary for Cell Identity', HE3_Trans[entry]['Cell_name'][0])
        print('PolCell0: ', PCell0, '+/-', PCell0_Unc)
        print('AtomicPol0: ', P0, '+/-', P0_Unc)
        print('Gamma (hours): ', gamma, '+/-', gamma_Unc)
        print('     ')

        if xdata.size >= 2:
            fit = he3_decay_func(xdata, popt[0], popt[1])
            fit_max = he3_decay_func(xdata, popt[0] + P0_Unc, popt[1] + gamma_Unc)
            fit_min = he3_decay_func(xdata, popt[0] - P0_Unc, popt[1] - gamma_Unc)
            fig = plt.figure()
            plt.plot(xdata, ydata, 'b*', label='data')
            plt.plot(xdata, fit_max, 'r-', label='fit of data (upper bounds)')
            plt.plot(xdata, fit, 'y-', label='fit of data (best)')
            plt.plot(xdata, fit_min, 'c-', label='fit of data (lower bounds)')
            plt.xlabel('time (hours)')
            plt.ylabel('3He atomic polarization')
            plt.title('He3 Cell Decay for {name}'.format(name = Name))
            plt.legend()
            file_path = os.path.join(save_path, 'He3Curve_AtomicPolarization_Cell{name}.png'.format(name = Name))
            fig.savefig(file_path)
            plt.show()
            plt.close()

        if xdata.size >= 2:
            TMAJ_data = Te * np.exp(-Mu*(1.0 - ydata))
            TMIN_data = Te * np.exp(-Mu*(1.0 + ydata))
            xdatalonger = HE3_Trans[entry]['Elasped_time']
            L = len(xdata)
            last_time = xdata[L-1]
            for i in range(49):
                extra_time = last_time + i*1
                xdatalonger.append(extra_time)
            xdataextended = np.array(xdatalonger)
            AtomicPol_fitlonger = he3_decay_func(xdataextended, popt[0], popt[1])
            TMAJ_fit = Te * np.exp(-Mu*(1.0 - AtomicPol_fitlonger))
            TMIN_fit = Te * np.exp(-Mu*(1.0 + AtomicPol_fitlonger))
            
            fig = plt.figure()
            plt.plot(xdata, TMAJ_data, 'b*', label='T_MAJ data')
            plt.plot(xdataextended, TMAJ_fit, 'c-', label='T_MAJ predicted')

            plt.plot(xdata, TMIN_data, 'r*', label='T_MIN data')
            plt.plot(xdataextended, TMIN_fit, 'm-', label='T_MIN predicted')
            
            plt.xlabel('time (hours)')
            plt.ylabel('Spin Transmission')
            plt.title('Predicted He3 Cell Transmission for {name}'.format(name = Name))
            plt.legend()
            file_path = os.path.join(save_path, 'He3PredictedDecayCurve_{name}.png'.format(name = Name))
            fig.savefig(file_path)
            plt.show()
            plt.close()

    return HE3_Cell_Summary

def he3_pol_at_given_time(entry_time, HE3_Cell_Summary):
    """Compute 3He cell polarization and transmissions at a given time.

    Duplicate of the function with the same name in
    ``polarization_correction_functions.py``: selects the most recently
    inserted cell relative to ``entry_time`` and evaluates the atomic
    polarization, neutron polarization, unpolarized 3He transmission,
    and majority/minority spin transmissions.

    Parameters
    ----------
    entry_time : float
        Required. Time (hours) at which to evaluate the cell.
    HE3_Cell_Summary : dict
        Required. Mapping ``insert_time -> {'Atomic_P0', 'Gamma(hours)',
        'Mu', 'Te', ...}`` (output of :func:`he3_decay_curves`).

    Returns
    -------
    NeutronPol : float
        ``tanh(Mu * AtomicPol)``.
    UnpolHE3Trans : float
        Transmission of unpolarized neutrons through the cell.
    T_MAJ : float
        Majority (aligned) spin transmission.
    T_MIN : float
        Minority spin transmission.
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

def sans_polarization_supermirror_and_flipper(Pol_Trans, HE3_Cell_Summary, UsePolCorr):
    """Derive the supermirror polarization and flipper efficiency per sample.

    For each cross-section measurement time in ``Pol_Trans``, evaluates
    the cell state via :func:`he3_pol_at_given_time`, then combines the
    four polarized transmissions to extract ``P_SM`` (sample depolarization
    times supermirror polarization) and ``P_F`` (flipper polarization,
    fixed at 1.0 for VSANS). Stores results, neutron-pol and unpol-trans
    arrays back on ``Pol_Trans``. If ``UsePolCorr`` is false, ``P_SM`` and
    ``P_F`` are reset to 1.0.

    Parameters
    ----------
    Pol_Trans : dict
        Required. Pol-trans catalog with ``Trans``/``Meas_Time`` lists for
        each cross-section.
    HE3_Cell_Summary : dict
        Required. Cell summary from :func:`he3_decay_curves`.
    UsePolCorr : bool
        Required. Override ``P_SM`` and ``P_F`` to 1.0 when false.

    Returns
    -------
    Pol_Trans : dict
        Updated catalog with ``P_SM``, ``P_F``, ``abs_scale``, plus
        ``Neutron_Pol`` and ``Unpol_Trans`` per cross-section.
    """
    
    for ID in Pol_Trans:
        if 'Meas_Time' in Pol_Trans[ID]['T_UU']:
            for Time in Pol_Trans[ID]['T_UU']['Meas_Time']:
                NP, UT, T_MAJ, T_MIN = he3_pol_at_given_time(Time, HE3_Cell_Summary)
                if 'Neutron_Pol' not in Pol_Trans[ID]['T_UU']:
                    Pol_Trans[ID]['T_UU']['Neutron_Pol'] = [NP]
                    Pol_Trans[ID]['T_UU']['Unpol_Trans'] = [UT]
                else:
                    Pol_Trans[ID]['T_UU']['Neutron_Pol'].append(NP)
                    Pol_Trans[ID]['T_UU']['Unpol_Trans'].append(UT)
            for Time in Pol_Trans[ID]['T_DD']['Meas_Time']:
                NP, UT, T_MAJ, T_MIN = he3_pol_at_given_time(Time, HE3_Cell_Summary)
                if 'Neutron_Pol' not in Pol_Trans[ID]['T_DD']:
                    Pol_Trans[ID]['T_DD']['Neutron_Pol'] = [NP]
                    Pol_Trans[ID]['T_DD']['Unpol_Trans'] = [UT]
                else:
                    Pol_Trans[ID]['T_DD']['Neutron_Pol'].append(NP)
                    Pol_Trans[ID]['T_DD']['Unpol_Trans'].append(UT)       
            for Time in Pol_Trans[ID]['T_DU']['Meas_Time']:
                NP, UT, T_MAJ, T_MIN = he3_pol_at_given_time(Time, HE3_Cell_Summary)
                if 'Neutron_Pol' not in Pol_Trans[ID]['T_DU']:
                    Pol_Trans[ID]['T_DU']['Neutron_Pol'] = [NP]
                    Pol_Trans[ID]['T_DU']['Unpol_Trans'] = [UT]
                else:
                    Pol_Trans[ID]['T_DU']['Neutron_Pol'].append(NP)
                    Pol_Trans[ID]['T_DU']['Unpol_Trans'].append(UT)     
            for Time in Pol_Trans[ID]['T_UD']['Meas_Time']:
                NP, UT,T_MAJ, T_MIN = he3_pol_at_given_time(Time, HE3_Cell_Summary)
                if 'Neutron_Pol' not in Pol_Trans[ID]['T_UD']:
                    Pol_Trans[ID]['T_UD']['Neutron_Pol'] = [NP]
                    Pol_Trans[ID]['T_UD']['Unpol_Trans'] = [UT]
                else:
                    Pol_Trans[ID]['T_UD']['Neutron_Pol'].append(NP)
                    Pol_Trans[ID]['T_UD']['Unpol_Trans'].append(UT)

    for ID in Pol_Trans:
        if 'Neutron_Pol' in Pol_Trans[ID]['T_UU']:
            ABS = np.array(Pol_Trans[ID]['T_SM']['Trans_Cts'])
            Pol_Trans[ID]['abs_scale'] = np.average(ABS)

            UU = np.array(Pol_Trans[ID]['T_UU']['Trans'])
            UU_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_UU']['Unpol_Trans'])
            UU_NeutronPol = np.array(Pol_Trans[ID]['T_UU']['Neutron_Pol'])
            DD = np.array(Pol_Trans[ID]['T_DD']['Trans'])
            DD_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_DD']['Unpol_Trans'])
            DD_NeutronPol = np.array(Pol_Trans[ID]['T_DD']['Neutron_Pol'])
            UD = np.array(Pol_Trans[ID]['T_UD']['Trans'])
            UD_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_UD']['Unpol_Trans'])
            UD_NeutronPol = np.array(Pol_Trans[ID]['T_UD']['Neutron_Pol'])
            DU = np.array(Pol_Trans[ID]['T_DU']['Trans'])
            DU_UnpolHe3Trans = np.array(Pol_Trans[ID]['T_DU']['Unpol_Trans'])
            DU_NeutronPol = np.array(Pol_Trans[ID]['T_DU']['Neutron_Pol'])
            print('  ')
            print(ID)
            print('UU, DU, DD, UD Trans:', int(np.average(UU)*10000)/10000, " " , int(np.average(DU)*10000)/10000, " ", int(np.average(DD)*10000)/10000, " ", int(np.average(UD)*10000)/10000)
            NPAve = 0.25*(np.average(UU_NeutronPol) + np.average(DU_NeutronPol) + np.average(DD_NeutronPol) + np.average(UD_NeutronPol))
            print('3He Pol (Ave.)', NPAve)
            
            PF = 1.00
            Pol_Trans[ID]['P_F'] = np.average(PF)
            PSMUU = (UU/UU_UnpolHe3Trans - 1.0)/(UU_NeutronPol)
            PSMDD = (DD/DD_UnpolHe3Trans - 1.0)/(DD_NeutronPol)
            PSMUD = (1.0 - UD/UD_UnpolHe3Trans)/(UD_NeutronPol)
            PSMDU = (1.0 - DU/DU_UnpolHe3Trans)/(DU_NeutronPol)
            PSM_Ave = 0.25*(np.average(PSMUU) + np.average(PSMDD) + np.average(PSMUD) + np.average(PSMDU))
            Pol_Trans[ID]['P_SM'] = np.average(PSM_Ave)
            print('Sample Depol * PSM', Pol_Trans[ID]['P_SM'])
            print('Flipping ratios (UU/DU, DD/UD):', int(10000*np.average(UU)/np.average(DU))/10000, int(10000*np.average(DD)/np.average(UD))/10000)
            
            if not UsePolCorr:
                '''#False Means no, turn it off'''
                Pol_Trans[ID]['P_SM'] = 1.0
                Pol_Trans[ID]['P_F'] = 1.0
                print('Manually reset P_SM and P_F to unity')

    print(" ")
    return Pol_Trans

def sans_best_supermirror_polarization(Pol_Trans, UsePolCorr = True, Starting_PSM = 0.9985, YesNoBypassBestGuessPSM = False):
    """Pick the best supermirror polarization estimate to use downstream.

    Starts from ``Starting_PSM`` and, if ``YesNoBypassBestGuessPSM`` is
    true, also considers every ``P_SM`` recorded on samples in
    ``Pol_Trans``. The maximum is returned, clamped to 1.0.

    Parameters
    ----------
    Pol_Trans : dict
        Required. Pol-trans catalog (consulted for measured ``P_SM`` when
        ``YesNoBypassBestGuessPSM`` is true).
    UsePolCorr : bool, optional
        Controls whether the chosen value is printed (default ``True``).
    Starting_PSM : float, optional
        Prior best-known supermirror polarization (default 0.9985).
    YesNoBypassBestGuessPSM : bool, optional
        If true, include every measured ``P_SM`` in the comparison
        (default ``False``).

    Returns
    -------
    Truest_PSM : float
        Best estimate of the supermirror polarization, clamped to 1.0.
    """

    Measured_PSM = [Starting_PSM]
    if YesNoBypassBestGuessPSM:
        for Sample in Pol_Trans:              
            if 'P_SM' in Pol_Trans[Sample]:
                Measured_PSM.append(Pol_Trans[Sample]['P_SM'])
    Truest_PSM = np.amax(Measured_PSM)
    if UsePolCorr:
        print('Best measured PSM value (currently or previously measured) is', Truest_PSM)
    if Truest_PSM > 1:
        Truest_PSM = 1.0
        if UsePolCorr:
            print('Best PSM value set to 1.0')
    print(" ")

    return Truest_PSM

def sans_record_data_processing_steps(save_path, Plex_Name, Scatt, BlockBeam, Trans, Pol_Trans, HE3_Cell_Summary, YesNoManualHe3Entry = False, Contents = 'not used'):
    """Write a human-readable summary of the data reduction inputs.

    Produces ``DataReductionSummary.txt`` in ``save_path`` recording the
    plex file used, per-sample/per-config block-beam and transmission
    files, the four polarized cross-section file lists, full polarization
    results (``P_SM`` and transmission file lists), and the 3He cell
    parameters. Twin of ``vsans_record_data_processing`` in
    ``polarization_correction_functions.py``.

    Parameters
    ----------
    save_path : str
        Required. Output directory.
    Plex_Name : str
        Required. Name of the plex file used.
    Scatt : dict
        Required. Scattering catalog.
    BlockBeam : dict
        Required. Block-beam catalog.
    Trans : dict
        Required. Transmission catalog.
    Pol_Trans : dict
        Required. Pol-trans catalog.
    HE3_Cell_Summary : dict
        Required. 3He cell summary.
    YesNoManualHe3Entry : bool or int, optional
        If truthy, additionally write the (module-level) ``New_HE3_Files``,
        ``MuValues``, ``TeValues`` arrays (default ``False``).
    Contents : str, optional
        Verbatim user-input block embedded at the top of the summary
        (default ``'not used'``).

    Returns
    -------
    None
    """

    file_path = os.path.join(save_path, 'DataReductionSummary.txt')
    file1 = open(file_path, "w+")
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

def reduction_pipeline(input_path, save_path, Instrument = "VSANS",
                        UsePolCorr = True, 
                        SampleDescriptionKeywordsToExclude = None,
                        TransPanel = None,
                        YesNoManualHe3Entry=False,
                        New_HE3_Files = None, 
                        MuValues = None,
                        TeValues = None,
                        Excluded_Filenumbers=None,
                        Min_Filenumber=0,
                        Max_Filenumber=1000000,
                        Min_Scatt_Filenumber=0, 
                        Max_Scatt_Filenumber=1000000, 
                        Min_Trans_Filenumber=0, 
                        Max_Trans_Filenumber=1000000, 
                        ReAssignBlockBeamIntent=None, 
                        ReAssignEmptyIntent=None, 
                        ReAssignOpenIntent=None, 
                        ReAssignSampleIntent=None, 
                        YesNoRenameEmpties=True,
                        TempDiffAllowedForSharingTrans=20.0,
                        HighResMinX=240, 
                        HighResMaxX=474, 
                        HighResMinY=667, 
                        HighResMaxY=917,
                        ConvertHighResToSubset=True, 
                        HighResGain=100.0,
                        Notes = 'not used',
                        Starting_PSM = 0.9985,
                        YesNoBypassBestGuessPSM = False
):
        """Run the full pre-polarization-correction SANS reduction pipeline.

        Composes :func:`_sans_instrument_selection`,
        :func:`_sans_sort_data_automatic`, the three ``sans_share_*``
        catalog completers, :func:`sans_share_pol_trans_catalog`,
        :func:`sans_process_he3_trans_catalog`,
        :func:`sans_process_pol_trans_catalog`,
        :func:`sans_process_trans_catalog`, :func:`plex_file`,
        :func:`he3_decay_curves`,
        :func:`sans_polarization_supermirror_and_flipper`,
        :func:`sans_best_supermirror_polarization`, and
        :func:`sans_record_data_processing_steps`. Writes
        ``ReductionResults.json`` to ``save_path`` and returns the same
        dict.

        Parameters
        ----------
        input_path : str
            Required. Directory containing raw NeXus files.
        save_path : str
            Required. Output directory for the summary, plots, and JSON.
        Instrument : str, optional
            ``'VSANS'`` or ``'NG7SANS'`` (default ``'VSANS'``).
        UsePolCorr : bool, optional
            Apply polarization correction (default ``True``).
        SampleDescriptionKeywordsToExclude : list[str] or None, optional
            Keywords stripped from sample descriptions (default ``None``).
        TransPanel : str or None, optional
            Panel used to integrate transmissions; derived from
            ``Instrument`` when ``None`` (default ``None``).
        YesNoManualHe3Entry : bool, optional
            Use manually supplied 3He values (default ``False``).
        New_HE3_Files : list[int] or None, optional
            Manually flagged 3He cell-load files (default ``None``).
        MuValues : list[float] or None, optional
            Manual ``Mu`` values (default ``None``).
        TeValues : list[float] or None, optional
            Manual ``Te`` values (default ``None``).
        Excluded_Filenumbers : list[int] or None, optional
            Files to skip (default ``None``).
        Min_Filenumber, Max_Filenumber : int, optional
            Global file-number bounds (defaults 0 and 1000000).
        Min_Scatt_Filenumber, Max_Scatt_Filenumber : int, optional
            Scattering-run bounds (defaults 0 and 1000000).
        Min_Trans_Filenumber, Max_Trans_Filenumber : int, optional
            Transmission-run bounds (defaults 0 and 1000000).
        ReAssignBlockBeamIntent, ReAssignEmptyIntent, ReAssignOpenIntent, ReAssignSampleIntent : list[int] or None, optional
            Override auto-detected intents (each default ``None``).
        YesNoRenameEmpties : bool, optional
            Rename empty-cell samples to ``Sample_Name = 'Empty'``
            (default ``True``).
        TempDiffAllowedForSharingTrans : float, optional
            Maximum temperature difference for transmission sharing
            (default 20.0 K).
        HighResMinX, HighResMaxX, HighResMinY, HighResMaxY : int, optional
            High-resolution back-detector pixel bounds (defaults 240, 474,
            667, 917).
        ConvertHighResToSubset : bool, optional
            Crop the back detector to high-res bounds (default ``True``).
        HighResGain : float, optional
            Gain factor for the back detector (default 100.0).
        Notes : str, optional
            Verbatim text embedded in the ``DataReductionSummary.txt``
            (default ``'not used'``).
        Starting_PSM : float, optional
            Prior best supermirror polarization (default 0.9985).
        YesNoBypassBestGuessPSM : bool, optional
            Include measured ``P_SM`` values when picking ``Truest_PSM``
            (default ``False``).

        Returns
        -------
        Results : dict
            Reduction outputs including all catalogs, ``Truest_PSM``, the
            plex arrays, and metadata. Also written as
            ``ReductionResults.json`` in ``save_path``.
        """

        Detector_Panels, TransPanel, Slices = _sans_instrument_selection(Instrument = Instrument)

        (Sample_Names, Sample_Bases, Configs, BlockBeamCatalog, ScattCatalog, TransCatalog, Pol_TransCatalog, 
        AlignDet_Trans, HE3_TransCatalog, start_number, 
        filenumberlisting) = _sans_sort_data_automatic(Detector_Panels = Detector_Panels,
                                                    input_path = input_path, 
                                                    Instrument = Instrument,
                                                    UsePolCorr = UsePolCorr,
                                                    SampleDescriptionKeywordsToExclude = SampleDescriptionKeywordsToExclude,
                                                    TransPanel = TransPanel,
                                                    YesNoManualHe3Entry = YesNoManualHe3Entry,
                                                    New_HE3_Files = New_HE3_Files, 
                                                    MuValues = MuValues,
                                                    TeValues = TeValues,
                                                    Excluded_Filenumbers=Excluded_Filenumbers,
                                                    Min_Filenumber= Min_Filenumber,
                                                    Max_Filenumber=Max_Filenumber,
                                                    Min_Scatt_Filenumber=Min_Scatt_Filenumber, 
                                                    Max_Scatt_Filenumber=Max_Scatt_Filenumber, 
                                                    Min_Trans_Filenumber=Min_Trans_Filenumber, 
                                                    Max_Trans_Filenumber=Max_Trans_Filenumber, 
                                                    ReAssignBlockBeamIntent=ReAssignBlockBeamIntent, 
                                                    ReAssignEmptyIntent=ReAssignEmptyIntent, 
                                                    ReAssignOpenIntent=ReAssignOpenIntent, 
                                                    ReAssignSampleIntent=ReAssignSampleIntent, 
                                                    YesNoRenameEmpties=YesNoRenameEmpties,


        )

        AlignDet_Trans = sans_share_align_det_trans_catalog(
                TempDiffAllowedForSharingTrans = TempDiffAllowedForSharingTrans, 
                AlignDet_Trans = AlignDet_Trans, 
                Scatt = ScattCatalog
                )

        TransCatalog = sans_share_sample_base_trans_catalog(
                Trans = TransCatalog, 
                Scatt = ScattCatalog
                )

        ScattCatalog = sans_share_empty_polbeam_scatt_catalog(Scatt = ScattCatalog)


        Pol_TransCatalog = sans_share_pol_trans_catalog(Detector_Panels = Detector_Panels,
                                                    Pol_Trans = Pol_TransCatalog, 
                                                    Scatt = ScattCatalog,
                                                    input_path = input_path,
                                                    Instrument = Instrument,
                                                    SampleDescriptionKeywordsToExclude = SampleDescriptionKeywordsToExclude,
                                                    TempDiffAllowedForSharingTrans = TempDiffAllowedForSharingTrans

        )

        HE3_TransCatalog = sans_process_he3_trans_catalog(Detector_Panels = Detector_Panels, 
                                                                Instrument = Instrument,
                                                                input_path = input_path, 
                                                                HE3_Trans = HE3_TransCatalog, 
                                                                BlockBeam = BlockBeamCatalog, 
                                                                DetectorPanel = TransPanel
                                                                )
                                                    

        Pol_TransCatalog = sans_process_pol_trans_catalog(Detector_Panels = Detector_Panels,
                                                          Instrument = Instrument, 
                                                          input_path = input_path, 
                                                          Pol_Trans = Pol_TransCatalog, 
                                                          BlockBeam = BlockBeamCatalog, 
                                                          DetectorPanel= TransPanel
                                                          )  
        

        TransCatalog = sans_process_trans_catalog(Detector_Panels = Detector_Panels,
                                                  Instrument = Instrument, 
                                                  input_path = input_path, 
                                                  Trans = TransCatalog, 
                                                  BlockBeam = BlockBeamCatalog, 
                                                  DetectorPanel = TransPanel
                                                  )
    

        Plex_Name, Plex = plex_file(Detector_Panels = Detector_Panels, 
                                    input_path = input_path,
                                    start_number = start_number,
                                    Instrument = Instrument,
                                    HighResMinX = HighResMinX,
                                    HighResMaxX = HighResMaxX,
                                    HighResMinY = HighResMinY,
                                    HighResMaxY = HighResMaxY,
                                    ConvertHighResToSubset = ConvertHighResToSubset,
                                    HighResGain = HighResGain,
        )   

        HE3_Cell_Summary = he3_decay_curves(save_path = save_path, 
                                            HE3_Trans = HE3_TransCatalog
                                            )
        print(HE3_Cell_Summary)

        Pol_TransCatalog = sans_polarization_supermirror_and_flipper(
                Pol_Trans = Pol_TransCatalog, 
                HE3_Cell_Summary = HE3_Cell_Summary, 
                UsePolCorr = UsePolCorr)
        
        Truest_PSM = sans_best_supermirror_polarization(
                Pol_Trans = Pol_TransCatalog, 
                UsePolCorr = UsePolCorr, 
                Starting_PSM = Starting_PSM, 
                YesNoBypassBestGuessPSM = YesNoBypassBestGuessPSM)
        
        sans_record_data_processing_steps(save_path = save_path, 
                                          Plex_Name = Plex_Name, 
                                          Scatt = ScattCatalog, 
                                          BlockBeam = BlockBeamCatalog, 
                                          Trans = TransCatalog, 
                                          Pol_Trans = Pol_TransCatalog, 
                                          HE3_Cell_Summary = HE3_Cell_Summary, 
                                          YesNoManualHe3Entry = YesNoManualHe3Entry, 
                                          Contents = Notes,
                                          )
 

        Results = {
            "input_path": input_path,
            "save_path": save_path,
            "status": "Reduction completed successfully.",
            "Truest_PSM": Truest_PSM,
            "Sample_Names": Sample_Names,
            "Sample_Bases": Sample_Bases,
            "Configs": Configs,
            "BlockBeamCatalog": BlockBeamCatalog,
            "ScattCatalog": ScattCatalog,
            "TransCatalog": TransCatalog,
            "Pol_TransCatalog": Pol_TransCatalog,
            "HE3_TransCatalog": HE3_TransCatalog,
            "HE3_Cell_Summary": HE3_Cell_Summary, 
            "AlignDet_Trans": AlignDet_Trans,
            "start_number": start_number,
            "Detector_Panels": Detector_Panels,
            "Instrument": Instrument,
            "SampleDescriptionKeywordsToExclude": SampleDescriptionKeywordsToExclude,
            "HighResMinX": HighResMinX,
            "HighResMaxX": HighResMaxX,
            "HighResMinY": HighResMinY,
            "HighResMaxY": HighResMaxY,
            "YesNoManualHe3Entry": YesNoManualHe3Entry,
            "UsePolCorr": UsePolCorr,
            "HighResGain": HighResGain,
            "Slices": Slices,
            "Plex": Plex,
        }

        def _json_default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            raise TypeError(f"not JSON-serializable: {type(o).__name__}")

        file_path = os.path.join(save_path, 'ReductionResults.json')
        with open(file_path, 'w') as f:
            json.dump(Results, f, indent=2, default=_json_default)
        print(f"Saved Results to {file_path}")
        return Results

