def results_dict_unwrapped(Results):
        """Unpack the ``Results`` dict returned by ``reduction_pipeline``.

        Pulls a fixed set of keys out of ``Results`` and returns them as a
        positional tuple, in the order expected downstream (e.g. by
        :func:`polarization_correction_pipeline`). Every key listed below
        must be present in ``Results``; ``KeyError`` is raised otherwise.

        Parameters
        ----------
        Results : dict
            Required. Output of
            :func:`sans_reduction_utilites.reduction_functions.reduction_pipeline`
            (or an equivalently structured dict). Must contain the keys:
            ``Detector_Panels``, ``Instrument``,
            ``SampleDescriptionKeywordsToExclude``, ``UsePolCorr``,
            ``YesNoManualHe3Entry``, ``input_path``, ``save_path``,
            ``HighResMinX``, ``HighResMaxX``, ``HighResMinY``,
            ``HighResMaxY``, ``HighResGain``, ``Plex``,
            ``HE3_Cell_Summary``, ``Slices``, ``Truest_PSM``,
            ``ScattCatalog``, ``BlockBeamCatalog``, ``Configs``,
            ``Sample_Names``, ``Sample_Bases``, ``TransCatalog``,
            ``Pol_TransCatalog``, ``AlignDet_Trans``.

        Returns
        -------
        Detector_Panels : list[str]
            Short panel names.
        Instrument : str
            ``'VSANS'`` or ``'NG7SANS'``.
        SampleDescriptionKeywordsToExclude : list[str] or None
            Keywords stripped from sample descriptions.
        UsePolCorr : bool
            Whether polarization correction is enabled.
        YesNoManualHe3Entry : bool
            Whether manual 3He values were used.
        input_path : str
            Directory of raw NeXus files.
        save_path : str
            Output directory.
        HighResMinX, HighResMaxX, HighResMinY, HighResMaxY : int
            High-resolution back-detector pixel bounds.
        HighResGain : float
            Gain factor for the back detector.
        Plex : dict[str, np.ndarray]
            Per-panel plex arrays.
        HE3_Cell_Summary : dict
            3He cell parameter summary keyed by insertion time.
        Slices : list[str]
            Slice keys (``'Vert'``, ``'Horz'``, ``'Diag'``, ``'Circ'``).
        Truest_PSM : float
            Best supermirror polarization estimate.
        ScattCatalog, BlockBeamCatalog, TransCatalog, Pol_TransCatalog, AlignDet_Trans : dict
            The five run catalogs built during reduction.
        Configs : dict[str, int]
            Configuration label -> representative file number.
        Sample_Names : list[str]
        Sample_Bases : list[str]

        Raises
        ------
        KeyError
            If any required key is missing from ``Results``.
        """
        Detector_Panels = Results["Detector_Panels"]
        Instrument = Results["Instrument"]
        SampleDescriptionKeywordsToExclude = Results["SampleDescriptionKeywordsToExclude"]
        UsePolCorr = Results["UsePolCorr"]
        YesNoManualHe3Entry = Results["YesNoManualHe3Entry"]
        input_path = Results["input_path"]
        save_path = Results["save_path"]
        HighResMinX = Results["HighResMinX"]
        HighResMaxX = Results["HighResMaxX"]
        HighResMinY = Results["HighResMinY"]                        
        HighResMaxY = Results["HighResMaxY"]  
        HighResGain = Results["HighResGain"] 
        Plex = Results["Plex"] 
        HE3_Cell_Summary = Results["HE3_Cell_Summary"] 
        Slices = Results["Slices"]
        Truest_PSM = Results["Truest_PSM"]
        ScattCatalog = Results["ScattCatalog"]
        BlockBeamCatalog = Results["BlockBeamCatalog"]
        Configs = Results["Configs"]
        Sample_Names = Results["Sample_Names"]
        Sample_Bases = Results["Sample_Bases"]
        TransCatalog = Results["TransCatalog"]
        Pol_TransCatalog = Results["Pol_TransCatalog"]
        AlignDet_Trans = Results["AlignDet_Trans"]
        return (Detector_Panels, Instrument, SampleDescriptionKeywordsToExclude, UsePolCorr, YesNoManualHe3Entry, input_path, save_path, HighResMinX, HighResMaxX, HighResMinY, HighResMaxY, HighResGain, Plex, HE3_Cell_Summary, Slices, Truest_PSM, ScattCatalog, BlockBeamCatalog, Configs, Sample_Names, Sample_Bases, TransCatalog, Pol_TransCatalog, AlignDet_Trans)
