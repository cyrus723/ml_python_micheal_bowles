from math import sqrt

target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

# Print errors
err = []
for i in range(len(target)):
    err.append(target[i] - prediction[i])

print("Errors: ", err)

# Calculate squared errors and absolute value of errors
sq_err = []
abs_err = []
for val in err:
    sq_err.append(val*val)
    abs_err.append(abs(val))

print("Squared errors: ", sq_err)
print("Absolute errors: ", abs_err)

# Calculate and print mean squared error
print("MSE= ", sum(sq_err)/len(sq_err))

# Calculate and print square root of MSE
print("RMSE= ", sqrt(sum(sq_err)/len(sq_err)))

# Calculate and print absolute error MAE
print("MAE= ", sum(abs_err)/len(abs_err))

# Compare MSE to target variance
# Mean squared deviation
target_deviation = []
target_mean = sum(target)/len(target)
for val in target:
    target_deviation.append((val-target_mean)*(val-target_mean))
# Target variance
print("Target variance= ", sum(target_deviation)/len(target_deviation))

# Print target standard deviation
print("Target standard deviation= ",
      sqrt(sum(target_deviation)/len(target_deviation)))