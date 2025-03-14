## Global model 
#  Selects a local model mased on the sales data

import sys
import pandas as pd
import importlib
import numpy as np
import time
import pymc

sys.path.append("222_models")
import WHCT_model
import WHCST_model
import mean_model
import model_utils

sales = pd.read_csv('111_input_data/sales.csv',parse_dates=['Date'], date_format='%Y-%m')
unique_pairs = sales[['Material', 'Kunde']].drop_duplicates()
total = len(unique_pairs)

total_start = time.time()
# Iterate through each unique pair
i = 0
for _, pair in unique_pairs.iterrows():
    material, customer = pair['Material'], pair['Kunde']
    print(f"Processing pair: Material={material}, NGV-Kunde={customer}, len {len(sales[(sales['Material'] == material) &
                            (sales['Kunde'] == customer)])}")
    start_time = time.time()
    print()

    try:
        if len(sales[(sales['Material'] == material) & 
                           (sales['Kunde'] == customer) & 
                           (sales['Sale'] != 0)]) <= 24:
            
            mean_model.train(material,customer,sales)
        elif model_utils.load_stat_cov('kunde.csv', 'Kunde', 'Ort', Kunde=customer).iloc[0]['Ort'] == "Rostock":
            WHCST_model.train(material,customer,sales)
        else:
            WHCT_model.train(material,customer,sales)
    except Exception as e:
        print(f"Error for Material: {material}, Customer: {customer}: {e}")
        # Log error and skip to the next one
    i += 1
    end_time = time.time()
    duration = end_time - start_time
    with open('progress.txt', 'w') as f:
        f.write(f"{i}/{total} {duration:.1f}s Mat: {material} Cus: {customer}\n")

total_end = time.time()
total_duration = total_end - total_start
with open('progress.txt', 'w') as f:
    f.write(f"Done in {total_duration:.2f} sec\n")

print(len(unique_pairs))