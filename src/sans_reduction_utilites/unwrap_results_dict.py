import os
import os.path
import json

def results_dict_unwrapped(Results):
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
