import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import model_utils
import calendar
import model_utils
import seaborn as sns

def load_from_hdf5(filename="model_means.h5"):
    """
    Load the means from an HDF5 file.
    """
    
    means = {}
    with h5py.File(filename, "r") as f:
        for group_name in f.keys():
            group = f[group_name]
            for var_name in group.keys():
                means[var_name] = group[var_name][()]
            for attr_name, attr_value in group.attrs.items():
                means[attr_name] = attr_value
    return means

def load_means_from_h5(file_path):
    """
    Loads mean values from an HDF5 file, filtering out array values, and returns them as a formatted pandas DataFrame.

    Parameters:
    file_path (str): Path to the HDF5 file.

    Returns:
    pd.DataFrame: DataFrame containing only the single-value means.
    """
    means_dict = {}

    # Load data object (assuming `load_from_hdf5` is a valid function)
    obj = load_from_hdf5(file_path)

    # Open HDF5 file and retrieve the means for the specific material/customer
    with h5py.File(file_path, "r") as f:
        group_name = f"{obj['material_name']}_{obj['customer_name']}"
        if group_name in f:
            group = f[group_name]
            for key in group.keys():
                # Check if the item is a single value (not an array)
                if isinstance(group[key][()], (int, float)):  # Scalars only (int or float)
                    means_dict[key] = group[key][()]  # Load scalar mean data

    # Convert to DataFrame (one row, each key is a column)
    means_df = pd.DataFrame([means_dict], columns=means_dict.keys())

    # Round the DataFrame values for better readability
    means_df = means_df.round(4)  # Round to 4 decimal places

    return means_df

def plot_file(data_sales, posterior_predictive, predictions, obs, model, test_data=None, mult=1, time=None, 
              test_t=None, hdi_post=None, hdi_pred=None,
              y_max=1, customer = "", material="",
              t_min = 0, t_max = 1, 
              x_title="Date", y_title="Sales (hL)"):
    # Internal plotting function for h5 files


    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot()

    material_data = data_sales[(data_sales['Material'] == material) &
                            (data_sales['Kunde'] == customer)]

    original_train_t = pd.to_datetime(time * (t_max - t_min) + t_min, unit='D', origin='1900-01-01')
    original_test_t = pd.to_datetime(test_t * (t_max - t_min) + t_min, unit='D', origin='1900-01-01')

    residuals = np.abs((posterior_predictive - obs) * y_max)

    ax1.fill_between(original_train_t, np.array(residuals * mult), 0, color='tab:red', alpha=0.25)
    ax1.fill_between(original_test_t, np.array(np.abs((predictions - test_data) * y_max) * mult), 0, color='tab:red', alpha=0.25)

    ax1.plot(original_train_t, posterior_predictive * y_max, color="tab:blue", label="Training")
    ax1.plot(original_train_t, obs * y_max, color="tab:red", label="Ground Truth")
    
    ax1.fill_between(original_train_t, *hdi_post.T * mult * y_max, color="tab:blue", alpha=0.25)
    ax1.fill_between(original_test_t, *hdi_pred.T * mult * y_max, color="tab:orange", alpha=0.25)
    
    ax1.plot(original_test_t, predictions * y_max, color="tab:orange", label="Prediction")
    ax1.plot(original_test_t, test_data * y_max, color="tab:red")
    
    kunde_name = model_utils.load_stat_cov('kunde.csv', 'Kunde', 'Name', Kunde=customer).iloc[0]['Name']
    prod_name = model_utils.load_stat_cov('material.csv', 'Material', 'Name', Material=material).iloc[0]['Name']

    ax1.set_title(f"Model: {model}\n \
                    Kunde ID: {customer}, Material ID: {material}\n \
                    Sales of {prod_name} at {kunde_name}")
    
    ax1.set_xlabel(x_title)
    ax1.set_ylabel(y_title)
    ax1.legend()
    
    fig.autofmt_xdate()
    plt.show()
    
    #print(np.abs((posterior_predictive - obs) * y_max).mean())
    #print(np.abs(np.abs((predictions - test_data) * y_max)).mean())

def plot(filename, sales):
    # Plot an h5 file including residuals, training data and predictions
    obj = load_from_hdf5(filename)
    plot_file(sales, obj['posterior_predictive'], obj['predictions'], obj['train_y'], obj['model_name'], obj['test_y'], 1, 
          obj['train_t'], obj['test_t'], obj['hdi90_posterior'], obj['hdi90_predictions'],
          obj['y_max'], obj['customer_name'], obj['material_name'], obj['t_min'], obj['t_max'])
    
def plot_violins(results, sales, column, selector, title = 'Category'):
    categories = []
    values = []

    for root_key, obj in results.items():
        material_data = sales[(sales['Material'] == obj['material_name']) &
                              (sales['Kunde'] == obj['customer_name'])]
        
        if selector == "Material":
            col_val = model_utils.load_stat_cov('material.csv', 'Material', column, Material=obj['material_name']).iloc[0][column]
        else:
            col_val = model_utils.load_stat_cov('kunde.csv', 'Kunde', column, Kunde=obj['customer_name']).iloc[0][column]
        residuals = np.abs(obj['posterior_predictive'] - obj['train_y']).mean()
        values.append(residuals)
        categories.append(col_val)

    df = pd.DataFrame({title: categories, "Residuals": values})
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=title, y="Residuals", data=df, palette="muted")
    plt.xticks(rotation=45,ha='right')
    plt.show()
    return df