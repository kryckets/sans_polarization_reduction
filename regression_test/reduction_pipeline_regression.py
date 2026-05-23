import os
import os.path

from sans_reduction_utilites.reduction_functions import reduction_pipeline

def regression_test_reduction_pipeline(input_path, save_path, Instrument, New_HE3_Files, MuValues, TeValues):

    results_dict =reduction_pipeline(input_path, save_path, Instrument,
                        New_HE3_Files = [], 
                        MuValues = [],
                        TeValues = [],
                        )
    assert results_dict["Instrument"] == Instrument, f"Expected Instrument: {Instrument}, but got: {results_dict['Instrument']}"        
    assert results_dict["ScattCatalog"]["Fe3O4NPs_4.9V_300.0K"]["Config(s)"]["4Gd300cmF1400cmM5.5Ang"]["UU"] == [
            51295,
            51310
          ], f"Expected UU config, but got: {results_dict['ScattCatalog']['Fe3O4NPs_4.9V_300.0K']['Config(s)']['4Gd300cmF1400cmM5.5Ang']['UU']}"
    assert results_dict["ScattCatalog"]["Fe3O4NPs_4.9V_300.0K"]["Config(s)"]["4Gd300cmF1400cmM5.5Ang"]["DU"] == [
            51296,
            51297,
            51298,
            51311,
            51312,
            51313
          ], f"Expected DU config, but got: {results_dict['ScattCatalog']['Fe3O4NPs_4.9V_300.0K']['Config(s)']['4Gd300cmF1400cmM5.5Ang']['DU']}"
    assert results_dict["ScattCatalog"]["Fe3O4NPs_4.9V_300.0K"]["Config(s)"]["4Gd300cmF1400cmM5.5Ang"]["DD"] == [
            51299,
            51314
          ], f"Expected DD config, but got: {results_dict['ScattCatalog']['Fe3O4NPs_4.9V_300.0K']['Config(s)']['4Gd300cmF1400cmM5.5Ang']['DD']}"
    assert results_dict["ScattCatalog"]["Fe3O4NPs_4.9V_300.0K"]["Config(s)"]["4Gd300cmF1400cmM5.5Ang"]["UD"] == [
            51300,
            51301,
            51302,
            51315,
            51316,
            51317
          ], f"Expected UD config, but got: {results_dict['ScattCatalog']['Fe3O4NPs_4.9V_300.0K']['Config(s)']['4Gd300cmF1400cmM5.5Ang']['UD']}"
    

    return

if __name__ == "__main__":
    base_path = os.getcwd()
    parent_path = os.path.dirname(base_path)
    input_path = os.path.join(parent_path, "test_data", "Fe3O4Nanoparticles_VSANS26903", "raw_data")
    save_path = os.path.join(parent_path, "regression_test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    regression_test_reduction_pipeline(input_path = input_path, 
                                       save_path = save_path, 
                                       Instrument = "VSANS",
                                       New_HE3_Files = [77070, 77297, 77566], 
                                       MuValues = [3.105, 3.374, 3.105],
                                       TeValues = [0.86, 0.86, 0.86])
