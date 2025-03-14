import pandas as pd
import numpy as np
import os
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import h5py


def compute_means(trace):
    """
    Compute the mean of all variables in the PyMC InferenceData trace.
    """
    means = {}
    for group in ["posterior"]:  # Add more groups if needed, e.g., 'posterior_predictive'
        if hasattr(trace, group):
            group_data = getattr(trace, group)
            for var_name in group_data.data_vars:
                means[var_name] = group_data[var_name].mean(dim=["chain", "draw"]).values.tolist()
    return means

def save_model(output_file, material_name, customer_name, model_name,
               trace, train_y, test_y, train_t, test_t, y_max, t_min, t_max):
    with h5py.File(output_file, "w") as f:
        # Create a group for the material-customer pair
        group = f.create_group(f"{material_name}_{customer_name}")
        means = compute_means(trace)
        # Save the datasets (arrays)
        for var_name, mean_values in means.items():
            # Save the mean values as a dataset
            group.create_dataset(var_name, data=np.array(mean_values))
        group.create_dataset("posterior_predictive", data=getattr(trace, 'posterior_predictive').likelihood.mean(dim=['chain', 'draw']).values)
        group.create_dataset("predictions", data=getattr(trace, 'predictions').likelihood.mean(dim=['chain', 'draw']).values)
        
        # Save HDIs
        group.create_dataset("hdi90_posterior", data=az.hdi(getattr(trace, 'posterior_predictive').likelihood, hdi_prob = 0.90).likelihood)
        group.create_dataset("hdi90_predictions", data=az.hdi(getattr(trace, 'predictions').likelihood, hdi_prob = 0.90).likelihood)

        # Save metadata (attributes)
        group.attrs["material_name"] = material_name
        group.attrs["customer_name"] = customer_name
        group.attrs["model_name"] = model_name

        group.create_dataset("train_y", data= train_y.tolist())
        if (len(test_y) > 0):
            group.create_dataset("test_y", data= test_y.tolist())
        group.create_dataset("train_t", data= train_t.tolist())
        if (len(test_t) > 0):
            group.create_dataset("test_t", data= test_t.tolist())
        group.attrs["y_max"] = y_max
        group.attrs["t_min"] = t_min
        group.attrs["t_max"] = t_max

def load_stat_cov(filename, id_column, *kwcol, **kwargs):
    """
    Loads static covariates from a CSV file, given an ID column, target columns, 
    and optional fine selection criteria.
    
    Parameters:
    - filename: Path to the CSV file.
    - id_column: The column to use as the identifier (ID column).
    - *kwcol: One or more target columns to load as static covariates.
    - **kwargs: Optional selection criteria to filter rows (e.g., `Material='MaterialA'`).
    
    Returns:
    - A DataFrame containing the specified columns, filtered by criteria.
    """
    # Read the CSV file

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path
    data_path = os.path.join(script_dir, "../111_input_data/static/", filename)
    data = pd.read_csv(data_path)

    # Ensure that the ID column exists
    if id_column not in data.columns:
        raise ValueError(f"ID column '{id_column}' not found in the CSV file.")

    # Apply the selection criteria (if any) using **kwargs
    for key, value in kwargs.items():
        if key not in data.columns:
            raise ValueError(f"Column '{key}' not found in the CSV file.")
        data = data[data[key] == value]

    # Select the target columns based on the passed arguments
    selected_columns = [id_column] + list(kwcol)
    
    # Ensure that the specified columns exist in the data
    missing_columns = [col for col in selected_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    # Return the relevant columns
    return data[selected_columns]

def load_dyn_cov(filename, date_column, *kwcol, **kwargs):
    """
    Loads dynamic covariates from a CSV file, given a date column, target columns,
    and optional fine selection criteria.
    
    Parameters:
    - filename: Path to the CSV file.
    - date_column: The column to use as the date.
    - *kwcol: One or more target columns to load as dynamic covariates.
    - **kwargs: Optional selection criteria to filter rows (e.g., `Material='MaterialA'`).
    
    Returns:
    - A DataFrame containing the specified columns, filtered by criteria.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path
    data_path = os.path.join(script_dir, "../111_input_data/dynamic/", filename)
    data = pd.read_csv(data_path, parse_dates=[date_column])
    data.reset_index(inplace=True)
    #data[date_column] = pd.to_datetime(data[date_column], format='%Y-%m-%d', errors='coerce')

    # Ensure that the date column exists
    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' not found in the CSV file.")

    # Apply the selection criteria (if any) using **kwargs
    for key, value in kwargs.items():
        if key not in data.columns:
            raise ValueError(f"Column '{key}' not found in the CSV file.")
        data = data[data[key] == value]

    # Select the target columns based on the passed arguments
    selected_columns = [date_column] + list(kwcol)
    print(selected_columns)
    # Ensure that the specified columns exist in the data
    missing_columns = [col for col in selected_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    # Return the relevant columns
    return data[selected_columns]