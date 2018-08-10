import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import DataFrame


datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')
rocks_v_mines = pd.read_csv(datafile, header=None, prefix='V')

# Print head/tail of data frame
print(rocks_v_mines.head())
print(rocks_v_mines.tail())

summary = rocks_v_mines.describe()
print(summary)