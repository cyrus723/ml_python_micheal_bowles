import matplotlib.pyplot as plt
import os
import pandas as pd

datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')

rocks_vs_mines = pd.read_csv(datafile, header=None, prefix='V')

data_row2 = rocks_vs_mines.iloc[1, 0:60]     # Row 1 columns 0:60
data_row3 = rocks_vs_mines.iloc[2, 0:60]     # Row 2 columns 0:60

plt.scatter(data_row2, data_row3)

plt.xlabel('2nd attributes')
plt.ylabel('3rd attributes')
plt.show()

data_rows21 = rocks_vs_mines.iloc[20, 0:60]
plt.scatter(data_row2, data_rows21)

plt.xlabel('2nd attribute')
plt.ylabel('21st attribute')
plt.show()