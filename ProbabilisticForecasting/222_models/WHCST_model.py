import pymc as pm
import numpy as np
import pandas as pd
import h5py
import arviz as az
import os
from model_utils import load_stat_cov, load_dyn_cov, save_model

## WHCST Model
# 
# Following terms are implemented:
# - Weather
# - Holidays
# - Covid
# - Sail (Hanse Sail Rostock event)
# - Trend

def train(material_name, customer_name, data_sales, output_path = ''):
    """
    Trains a Bayesian model for the given material and customer, loading necessary covariates.
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
        print("Not enough data!")
        return
    
    # Load static covariates
    ort = load_stat_cov('kunde.csv', 'Kunde', 'Ort', Kunde=customer_name).iloc[0]['Ort']
    bund = load_stat_cov('kunde.csv', 'Kunde', 'Bundesland', Kunde=customer_name).iloc[0]['Bundesland']

    # Load dynamic covariates (weather, holidays, covid data)
    weather_data = load_dyn_cov("weather.csv", 
                                 "Date", 
                                 "temperature_air_mean_2m",
                                 "Ort",
                                 Ort = ort)
    
    holiday_data = load_dyn_cov("holidays.csv", 
                                 "Date", 
                                 "Feiertage",
                                 "Bundesland",
                                 Bundesland = bund)
    covid_data = load_dyn_cov("covid.csv", 
                                 "Date", 
                                 "Covid")
    sail_data = load_dyn_cov("sail.csv", 
                                 "Date", 
                                 "Sail")

    # Merge material_data and weather_data on 'Date', keeping all rows from material_data
    merged_data = pd.merge(material_data, weather_data[['Date', 'temperature_air_mean_2m']], on='Date', how='left')
    # Merging holiday data
    merged_data = pd.merge(merged_data, holiday_data[['Date', 'Feiertage']], on='Date', how='left')
    # Merging Covid
    merged_data = pd.merge(merged_data, covid_data[['Date', 'Covid']], on='Date', how='left')
    # Mergin Hanse Sail
    merged_data = pd.merge(merged_data, sail_data[['Date', 'Sail']], on='Date', how='left')
    
    # Extract the temperature values
    weather_values = merged_data['temperature_air_mean_2m'].to_numpy()
    # Normalize the covariates for the model (if needed)
    w = weather_values / np.max(weather_values)
    train_weather = w[:idx]
    test_weather = w[idx:]

    hol_vec = merged_data['Feiertage'].to_numpy()
    hol_vec = hol_vec / np.max(hol_vec)
    train_hol = hol_vec[:idx]
    test_hol = hol_vec[idx:]

    covid_vec = merged_data['Covid'].to_numpy()
    train_covid = covid_vec[:idx]
    test_covid = covid_vec[idx:]

    sail_vec = merged_data['Sail'].to_numpy()
    train_sail = sail_vec[:idx]
    test_sail = sail_vec[idx:]

    # Bayesian Model definition using PyMC
    with pm.Model() as model:
        # Define priors and likelihood
        t_pt = pm.MutableData('t', train_t) # Time input vector [0 ... 1]
        weather = pm.MutableData('weather', train_weather) # Weather input vector
        holiday = pm.MutableData('holiday', train_hol) # Hoildays input vector
        cov = pm.MutableData('covid', train_covid) # Covid input vector
        sail = pm.MutableData('sail', train_sail) # Hanse Sail input vector

        alpha = pm.Normal('alpha', mu=0, sigma=2) # Offset of the trend
        beta = pm.Normal('beta', mu=0, sigma=0.5) # Slope of the trend
        cov_h = pm.Normal('cov_h', mu=0, sigma=1) # Covid influence
        w_h = pm.Normal('w_h', mu=0, sigma=1) # Weather influence 
        hol_h = pm.Normal('hol_h', mu=0, sigma=1) # Holidays influence
        sail_h = pm.Normal("sail_h", mu=0, sigma=1) # Hanse Sail influence

        sigma = pm.HalfNormal('sigma', sigma=0.1) # Standard deviation

        trend = pm.Deterministic('trend', alpha + beta * t_pt) # Trend

        # Sales random variable
        μ = trend + w_h * weather + hol_h * holiday + cov_h * cov + sail_h * sail
        sales = pm.Normal('likelihood', mu=μ, sigma=sigma, observed=train_y, shape=t_pt.shape)

        # Sampling
        trace = pm.sample()
        #in-sample 
        posterior_predictive = pm.sample_posterior_predictive(trace, extend_inferencedata=True) #, nuts_sampler="numpyro"
        
        #out-of-sample
        pm.set_data({'t':test_t})
        pm.set_data({'covid':test_covid})
        pm.set_data({'weather':test_weather})
        pm.set_data({'holiday':test_hol})
        pm.set_data({'sail':test_sail})

        out_sample_pp  = pm.sample_posterior_predictive(trace, extend_inferencedata=True, predictions=True)

        # Save results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path
        data_path = os.path.join(script_dir, "../333_output_data", output_path)

        output_file = f"{data_path}/WHCST_{material_name}_{customer_name}.h5"
        save_model(output_file, material_name, customer_name, "WHCST", trace,
                   train_y, test_y, train_t, test_t, y_max, t_min, t_max)
        
    return trace