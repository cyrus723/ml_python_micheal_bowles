import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import DataFrame


datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')

rocks_vs_mines = pd.read_csv(datafile, header=None, prefix='V')

for i in range(208):
    # assign colors based on M or R labels
    if rocks_vs_mines.iat[i, 60] == "M":
        pcolor = "red"
    else:
        pcolor = "blue"

    # plot rows of data
    data_row = rocks_vs_mines.iloc[i, 0:60]
    data_row.plot(color=pcolor)

plt.xlabel("Attribute  Index")
plt.ylabel(('Attibute values'))
plt.show()
