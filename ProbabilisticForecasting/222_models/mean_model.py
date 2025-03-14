import pymc as pm
import numpy as np
import pandas as pd
import h5py
import arviz as az
import os
from model_utils import load_stat_cov, load_dyn_cov, save_model

## Simple Mean Model
# 

def train(material_name, customer_name, data_sales, output_path = ''):
    """
    Trains a simple mean model for the given material and customer, loading necessary covariates.
    """
    # Extract and preprocess data
    material_data = data_sales[(data_sales['Material'] == material_name) & 
                                (data_sales['Kunde'] == customer_name)]

    # Normalized time vector
    t = (material_data['Date'] - pd.Timestamp("1900-01-01")).dt.days.to_numpy()
    t_min, t_max = np.min(t), np.max(t)
    t = (t - t_min) / (t_max - t_min)  # Normalize time

    # Normalized sales data
    y = material_data['Sale'].to_numpy()
    y_max = np.max(y)
    y = y / y_max

    # Separating test / training data
    idx = -12
    train_y = y[:idx]
    test_y = y[idx:]
    train_t = t[:idx]
    test_t = t[idx:]
    if (len(y) <= 24):
        train_y = y
        test_y = []
        train_t = t
        test_t = []


    # Bayesian Model definition using PyMC
    try:
        with pm.Model() as model:
            t_pt = pm.MutableData('t', train_t)
            sigma = pm.HalfNormal("sigma", sigma=0.05)
            h = pm.Normal("h", mu=0, sigma=1)
            μ = h 
            

            # Sales random variable
            sales = pm.Normal('likelihood', mu=μ, sigma=sigma, observed=train_y, shape=t_pt.shape)

            # Sampling
            trace = pm.sample()
            #in-sample 
            posterior_predictive = pm.sample_posterior_predictive(trace, extend_inferencedata=True) #, nuts_sampler="numpyro"
            
            #out-of-sample
            pm.set_data({'t':test_t})

            out_sample_pp  = pm.sample_posterior_predictive(trace, extend_inferencedata=True, predictions=True)

            # Save results
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct the absolute path
            data_path = os.path.join(script_dir, "../333_output_data", output_path)

            output_file = f"{data_path}/mean_{material_name}_{customer_name}.h5"
            save_model(output_file, material_name, customer_name, "mean", trace,
                    train_y, test_y, train_t, test_t, y_max, t_min, t_max)
    except Exception as e:
        print(f"Error for Material: {material_name}, Customer: {customer_name}: {e}")
        # Log error and skip to the next one
        return
        
    return trace