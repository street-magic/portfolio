# Creates following csv's from an Excel file in 000_raw_data -> 111_input_data:
# - sales.csv
# - static / kunde.csv
# - static / material.csv

import numpy as np
import pandas as pd
import os
import glob
import errno

#import warnings
#warnings.filterwarnings('ignore')

def mkdir_p(path):
    """
    CReates a dirrectory if it does not exist already
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def fill_missing_sales_data(data_sales, end_date):
    """
    Fills in missing months for each NGV-Kunde and Material combination in the sales data.
    
    Parameters:
    data_sales (pandas.DataFrame): DataFrame containing sales data with columns like NGV-Kunde, Material, KalJahr/Monat, Verkauf Hektoliter.
    
    Returns:
    pandas.DataFrame: Original DataFrame with missing months filled in.
    """
    # Create a DatetimeIndex from the 'KalJahr/Monat' column
    data_sales['KalJahr/Monat'] = pd.to_datetime(data_sales['KalJahr/Monat'])
    data_sales = data_sales.set_index('KalJahr/Monat')
    
    # Get a list of unique NGV-Kunde and Material combinations
    grouping_cols = ['NGV-Kunde', 'Material']
    unique_groups = data_sales.groupby(grouping_cols).groups.keys()
    
    # Iterate through each unique group and fill in missing months
    filled_data = []
    for group in unique_groups:
        NGV_kunde, material = group
        
        # Get the data for this group
        group_data = data_sales.loc[data_sales['NGV-Kunde'] == NGV_kunde]
        group_data = group_data.loc[group_data['Material'] == material]
        
        # Create a full date range from the earliest to latest date
        all_months = pd.date_range(start=group_data.index.min(), end=end_date, freq='MS')
        
        # Reindex the group data to the full date range, filling missing values with 0s
        reindexed = group_data.reindex(all_months, fill_value=0)
        reindexed['NGV-Kunde'] = NGV_kunde
        reindexed['Material'] = material
        
        filled_data.append(reindexed)
    
    # Concatenate the filled data back into a single DataFrame
    filled_df = pd.concat(filled_data)
    filled_df = filled_df.reset_index().rename(columns={'index': 'KalJahr/Monat'})
    
    return filled_df

# Find the first .xlsx file in the folder
file_path = glob.glob('000_raw_data/*.xlsx')[0]

# Read the Excel file and process the data
data_sales = pd.read_excel(file_path, sheet_name='Tabelle16', parse_dates=['KalJahr/Monat'])
data_sales['KalJahr/Monat'] = pd.to_datetime(data_sales['KalJahr/Monat'], format='%m.%Y')

# Format the Ort column
data_sales['Ort'] = data_sales['Ort'].str.title()

# Get the maximum date in 'KalJahr/Monat' column
min_date = data_sales['KalJahr/Monat'].min()
max_date = data_sales['KalJahr/Monat'].max()

# Set negative values in 'Verkauf Hektoliter' to zero
data_sales.loc[data_sales['Verkauf Hektoliter'] < 0, 'Verkauf Hektoliter'] = 0

# Fill missing dates with zero sales
filled_sales = fill_missing_sales_data(data_sales, max_date)

# SALES
sales = filled_sales[['KalJahr/Monat', 'Material', 'NGV-Kunde', 'Verkauf Hektoliter']].copy()
sales.rename(columns={
    'KalJahr/Monat': 'Date',
    'NGV-Kunde': 'Kunde',
    'Verkauf Hektoliter': 'Sale'
}, inplace=True)
sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce', format='%m.%Y').dt.strftime('%Y-%m')
sales.to_csv('111_input_data/sales.csv', index=False)

# KUNDE
kunde = data_sales[['NGV-Kunde', 'Ort', 'Unnamed: 1', 'Postleitzahl', 
                    'Längengrad', 'Breitengrad', 'Terrasse', 
                    'Biergarten', 'Saalbetrieb']].copy()
kunde.rename(columns={
    'NGV-Kunde': 'Kunde',
    'Unnamed: 1': 'Name',
    'Längengrad': 'Lat',
    'Breitengrad': 'Long'
}, inplace=True)

# Retain only unique Kunde entries
kunde = kunde.drop_duplicates(subset=['Kunde'])

# Create a mapping for the cities and their corresponding Bundesland codes
city_to_bundesland = {
    "Rostock": "MV",
    "München": "BY",
    "Lübeck": "SH",
    "Berlin": "BE",
    "Stuttgart": "BW",
    "Bremen": "HB",
    "Frankfurt": "HE",
    "Hamburg": "HH",
    "Düsseldorf": "NW",
    "Mainz": "RP",
    "Kiel": "SH",
    "Saarbrücken": "SL",
    "Leipzig": "SN",
    "Halle": "ST",
    "Erfurt": "TH"
    # Add other cities and their Bundesland codes here as needed
}

# Add a new 'Bundesland' column by mapping the 'Ort' column to the corresponding Bundesland
kunde['Bundesland'] = kunde['Ort'].map(city_to_bundesland)

# Reorder the columns to place 'Bundesland' after 'Ort'
kunde = kunde[['Kunde', 'Ort', 'Bundesland', 'Name', 'Postleitzahl', 'Lat', 'Long', 'Terrasse', 'Biergarten', 'Saalbetrieb']]


# Save to CSV file
mkdir_p('111_input_data/static')
kunde.to_csv('111_input_data/static/kunde.csv', index=False)

# MATERIAL
material = data_sales[['Material', 'Unnamed: 12', 'erw. Warengruppe (STM)', 'Unnamed: 14']].copy()
material.rename(columns={
    'Unnamed: 12': 'Name',
    'erw. Warengruppe (STM)': 'Warengruppe',
    'Unnamed: 14': 'Type'
}, inplace=True)

# Retain only unique Material (ID) entries
material = material.drop_duplicates(subset=['Material'])

# Save to CSV file
mkdir_p('111_input_data/static')
material.to_csv('111_input_data/static/material.csv', index=False)