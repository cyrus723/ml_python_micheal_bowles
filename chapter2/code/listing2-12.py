import os
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

datafile = os.path.join(os.getcwd(),
                        '../../datasets/abalone.data')

# read the data
abalone = pd.read_csv(datafile, header=None, prefix='V')
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'While weight',
                   'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

correlation_mat = DataFrame(abalone.iloc[:, 1:9].corr())
plt.pcolor(correlation_mat)
plt.show()
