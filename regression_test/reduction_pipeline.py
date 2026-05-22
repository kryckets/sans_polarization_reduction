import os
import os.path

from sans_reduction_utilites.reduction_functions import reduction_pipeline

def regression_test_reduction_pipeline(input_path, save_path, Instrument, New_HE3_Files, MuValues, TeValues):

    results_dict =reduction_pipeline(input_path, save_path, Instrument,
                        New_HE3_Files = [], 
                        MuValues = [],
                        TeValues = [],
                        )
    #assert results_dict["New_HE3_Files"] == New_HE3_Files, f"Expected New_HE3_Files: {New_HE3_Files}, but got: {results_dict['New_HE3_Files']}"
    #assert results_dict["MuValues"] == MuValues, f"Expected MuValues: {MuValues}, but got: {results_dict['MuValues']}"
    #assert results_dict["TeValues"] == TeValues, f"Expected TeValues: {TeValues}, but got: {results_dict['TeValues']}"      
    #assert results_dict["Instrument"] == Instrument, f"Expected Instrument: {Instrument}, but got: {results_dict['Instrument']}"        
    assert results_dict is not None, "Expected results_dict to be a dictionary, but got None"
    print(results_dict)

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
