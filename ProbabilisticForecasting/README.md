# Probabilistic Forecast Models
This project aims to to build a system for rapid prototyping of probabilistic forecasting models for beverage sales prediction.
## Project structure
- ```000_raw_data``` - conains raw sales figures in Excel format.
- ```111_input_data``` - conains separated and cleaned csv files, including dynamic (changing in time: covid, holidays, weather) as well as static (Bundesland of the customer) covariates. Static covariates can be obtained via running ```convert_raw_data.py```
- ```222_models``` - contains models implementations as well as utilities for plotting. Those Models are called by the ```global_model.py```
- ```333_output_data``` - Every execution of the training proccess results in a corresponding .h5 file that contains posterior distributions, predictions, etc for a customer/product/model triplet.
- ```999_notebooks``` - miscellaneous Jupyter notebooks, mostly exploring the raw data
 

## How to run
The global model can be run via
``` python global_model.py ```.\
Specific models can be run for any customer/product combination. Exaples of that can be seen in the notebook ``` dashboard.ipynb```

## Installation
It is recommended to use Conda for PyMC5 installation:
```
conda create -c conda-forge -n pymc_env "pymc>=5"
```
After a successful PyMC5 installation the rermaining dependencies can be installed via:
```
conda install --yes --file requirements.txt
```


