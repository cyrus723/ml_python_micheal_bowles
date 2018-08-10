import matplotlib.pyplot as plt
import os
import pandas as pd
from random import uniform

datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')

rocks_vs_mines = pd.read_csv(datafile, header=None, prefix='V')

# change targets to numeric values
target = []
for i in range(208):
    # Assign 0 to "M" and 1 to "R"
    if rocks_vs_mines.iat[i, 60] == "M":
        target.append(1.0)
    else:
        target.append(0.0)

# Plot 35th attribute
data_row = rocks_vs_mines.iloc[0:208, 35]   # Rows 0:208, column 35
plt.scatter(data_row, target)
plt.xlabel("35th attribute values")
plt.ylabel("target value")
plt.show()


# Updated version for better visualization
target = []
for i in range(208):
    # Assign 0 to "M" and 1 to "R"
    if rocks_vs_mines.iat[i, 60] == "M":
        target.append(1.0 + uniform(-0.1, 0.1))
    else:
        target.append(0.0 + uniform(-0.1, 0.1))

data_row = rocks_vs_mines.iloc[0:208, 35]
plt.scatter(data_row, target, alpha=0.5, s=120)
plt.xlabel('35th attribute value')
plt.ylabel('target value')
plt.show()
