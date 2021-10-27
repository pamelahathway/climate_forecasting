import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import sweetviz as sv

# todo: add pandas configuration


# load data
data_path = "/Users/ph/Documents/ds_projects/climate_change/data"
filename = "GlobalTemperatures.csv"
glob_temp = pd.read_csv(os.path.join(data_path, filename))
