import os
import numpy as np


def getNtileBoundaries(data, ntiles):
    percentBdry = []
    for i in range(ntiles+1):
        percentBdry.append(np.percentile(data, i*(100)/ntiles))
    return percentBdry

attributes = []
labels = []
# Open the data set
datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')
f = open(datafile, 'r')

for line in f.readlines():
    row = line.strip().split(',')
    attributes.append(row)

nrows= len(attributes)
ncol = len(attributes[0])

type = [0] * 3
colCounts = []

# Get types of entries in column
for col in range(ncol):
    for row in attributes:
        try:
            a = float(row[col])
            if isinstance(a, float):
                # its a float field
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                # its a text field
                type[1] += 1
            else:
                # its neither text, nor float
                type[2] += 1
    colCounts.append(type)
    type = [0] * 3

attribute_types = []    # Each element in this list shows stats how many
# entries of given type are in that column
for types in colCounts:
    attribute_types.append(dict(numbers=type[0], strings=type[1],
                                other=type[2]))

# Generate summary statistics for column 3
col = 3
colData = []
for row in attributes:
    colData.append(float(row[col]))

colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)

print('Mean={0}, S.D={1}'.format(colMean, colsd))

#########################################
# Find the outliers

# Calculate quantile boundaries
percentBdry = getNtileBoundaries(colArray, 4)
# Boundaries of 4 equal intervals
print('Quantile boundaries:\n{0}'.format(percentBdry))

# Boundaries of 10 equal intervals: deciles
percentBdry = getNtileBoundaries(colArray, 10)
# Boundaries of 4 equal intervals
print('10 interval boundaries:\n{0}'.format(percentBdry))

# Last column containing categorical variables
col = 60
col_data = []
for row in attributes:
    col_data.append(row[col])

unique = set(col_data)
print('Unique Labels: {0}'.format(unique))

# count values for each unique label
cat_dict = dict(zip(list(unique), range(len(unique))))
print(cat_dict)
cat_count = [0] * 2

for elt in col_data:
    cat_count[cat_dict[elt]] += 1

print(list(unique))
print(cat_count)
