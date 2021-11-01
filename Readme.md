## Climate Change Forecasting

[content and explanations]


## Usage

Download the data from kaggle and save them in the `data` folder
https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data

#### Create a virtual environment for this project

Create the environment
```bash
cd <path-to-folder-above-climate_forecasting>
python3 -m venv .
```
Activate the environment
```bash
source bin/activate
```

#### Install the necessary packages
```bash
pip install -r requirements
```


#### Run the main file

Note 1.11.2021: currently the jupyter notebook is the best place to start from.
Packaging the Python files is in progress. 

Make the environment available to jupyter as a kernel.
```bash
python -m ipykernel install --user --name=climate_change
```

Open the jupyter notebook in jupyterlab 

```bash
jupyter lab notebooks/climate_forecasting.ipynb
```


