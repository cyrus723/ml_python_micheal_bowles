import os
import numpy as np
from math import sqrt, fabs, exp
from sklearn.linear_model import enet_path
import matplotlib.pyplot as plt


def S(z,gamma):
    if gamma >= fabs(z):
        return 0.0
    if z > 0.0:
        return z - gamma
    else:
        return z + gamma


def Pr(b0,b,x):
    n = len(x)
    sum = b0
    for i in range(n):
        sum += b[i]*x[i]
        if sum < -100:
            sum = -100
    return 1.0/(1.0 + exp(-sum))


x_list = []
labels = []
# Open the data set
datafile = os.path.join(os.getcwd(),
                        '../../datasets/sonar.all-data')
f = open(datafile, 'r')

for line in f.readlines():
    row = line.strip().split(',')
    x_list.append(row)

x_num = []
labels = []

for row in x_list:
    last_col = row.pop()
    if last_col == 'M':
        labels.append(1.0)
    else:
        labels.append(0.0)
    attr_row = [float(elt) for elt in row]
    x_num.append(attr_row)

# Number of rows and columns
n_row = len(x_num)
n_col = len(x_num[1])

alpha = 0.8

# calculate means and variances
x_means = []
x_sd = []
for i in range(n_col):
    col = [x_num[j][i] for j in range(n_row)]
    mean = sum(col)/n_row
    x_means.append(mean)
    col_diff = [(x_num[j][i] - mean) for j in range(n_row)]
    sum_sq = sum([col_diff[i] ** 2 for i in range(n_row)])
    stdev = sqrt(sum_sq/n_row)
    x_sd.append(stdev)

# Use calculate mean and standard deviation to normalize x_num
x_norm = []
for i in range(n_row):
    row_norm = [(x_num[i][j] - x_means[j])/x_sd[j] for j in range(n_col)]
    x_norm.append(row_norm)

# Normalize labels to center
mean_label = sum(labels)/n_row
sd_label = sqrt(sum([(labels[i] - mean_label) * (labels[i] - mean_label) \
                     for i in range(n_row)])/n_row)


# Initialize probabilities and weights
sum_wxr = [0.0] * n_col
sum_wxx = [0.0] * n_col
sum_wr = 0.0
sum_w = 0.0

# calculate starting points for betas
for i_row in range(n_row):
    p = mean_label
    w = p * (1.0 - p)
    # residual for logistic
    r = (labels[i_row] - p) / w
    x = x_norm[i_row]
    sum_wxr = [sum_wxr + w * x[i] * r for i in range(n_col)]
    sum_wxx = [sum_wxx + w * x[i] * r for i in range(n_col)]
    sum_wr = sum_wr + w*r
    sum_w = sum_w + w

avg_wxr = [sum_wxr[i]/n_row for i in range(n_col)]
avg_wxx = [sum_wxr[i]/n_row for i in range(n_col)]

max_wxr = 0.0
for i in range(n_col):
    val = abs(avg_wxr[i])
    if val > max_wxr:
        max_wxr = val

# calculate starting value for lambda
lam = max_wxr/alpha

# this value of lambda corresponds to beta = list of 0s
# initialize a vector of coefficients beta
beta = [0.0] * n_col
beta0 = sum_wr/sum_w

# initialize matrix of betas at each step
beta_mat = []
beta_mat.append(list(beta))

beta0_list = []
beta0_list.append(beta0)

# begin iteration
n_steps = 100
lam_mult = 0.93     # such that lamba^ 100 ~ 1
nz_list = []
for i_step in range(n_steps):
    # decrement lambda
    lam = lam*lam_mult
    # use incremental change in beta to control inner iteration
    # set middle loop values for betas = to outer values
    # values are used for calculating weights and probabilities
    # inner values are used for calculating penalized regression updates
    # take pass through data to calculate averages over data required
    # for iteration initilize accumulators

    beta_irls = list(beta)
    beta0_irls = beta0
    dist_irls = 100.0
    # middle loop to calculate new betas with fixed IRLS weights and probs.
    iter_irls  = 0
    while dist_irls > 0.01:
        iter_irls += 1
        iter_inner = 0.0

        beta_inner = list(beta_irls)
        beta0_inner = beta0_irls
        dist_inner = 100.0
        while dist_inner > 0.01:
            iter_inner += 1
            if iter_inner > 100:
                break
            # cycle through attributes and update one at a time
            # record starting values for comparison
            beta_start = list(beta_inner)
            for i_col in range(n_col):
                sum_wxr = 0.0
                sum_wxx = 0.0
                sum_wr = 0.0
                sum_w = 0.0

                for i_row in range(n_row):
                    x = list(x_norm[i_row])
                    y = labels[i_row]
                    p = Pr(beta0_irls, beta_irls, x)
                    if abs(p) < 1e-5:
                        p = 0.0
                        w = 1e-5
                    elif abs(1.0 - p) < 1e-5:
                        p = 1.0
                        w = 1e-5
                    else:
                        w = p * (1.0 - p)
                    z = (y - p) / w + beta0_irls \
                        + sum([x[i] * beta_irls for i in range(n_col)])

                    r = z - beta0_inner - \
                        sum([x[i] * beta_inner[i] for i in range(n_col)])
                    sum_wxr += w * x[i_col] * r
                    sum_wxx += w * x[i_col] * r
                    sum_wr += w * r
                    sum_w += w

                avg_wxr = sum_wxr / n_row
                avg_wxx = sum_wxx / n_row

                beta0_inner = beta0_inner + sum_wr / sum_w
                unc_beta = avg_wxr + avg_wxx * beta0_inner[i_col]
                beta0_inner[i_col] = S(unc_beta, lam*alpha) / (avg_wxx + lam
                                                               * (1.0 - alpha))

            sum_diff = sum([abs(beta_inner[n] - beta_start) for n in range(n_col)])
            sum_beta = sum([abs(beta_inner[n]) for n in range(n_col)])
            dist_inner = sum_diff / sum_beta

        # print number of steps for inner and middle loop convergence to
        # monitor behavior
        # print(i_step, iter_irls, iter_inner)

        # if exit inner while loop, then set beta_middle = beta_middle and
        # run through middle loop again

        # Check change in beta_middle to see if IRLS is converged
        a = sum([abs(beta_irls[i] - beta_inner[i]) for i in range(n_col)])
        b = sum([abs(beta_irls[i]) for i in range(n_col)])
        dist_irls = a/(b+0.0001)
        d_beta = [beta_inner[i] - beta_irls[i] for i in range(n_col)]
        grad_step = 1.0
        temp = [beta_irls + grad_step*d_beta[i] for i in range(n_col)]
        beta_irls = list(temp)
    beta = list(beta_irls)
    beta0 = beta0_irls
    beta_mat.append(list(beta))
    beta0_list.append(beta0)

    nz_beta = [i for i in range(n_col) if beta[i] != 0.0]
    for q in nz_beta:
        if q not in nz_list:
            nz_list.appendO(q)

# Make up names for columns of x_num
names = ['V' + str(i) for i in range(n_col)]
names_list = [names[nz_list[i]] for i in range(len(nz_list))]

print('Attributes ordered by how early they enter the model')
print(names_list)
for i in range(n_col):
    # plot range of beta values for each attribute
    coef_curve = [beta_mat[k][i] for k in range(n_steps)]
    xaxis = range(n_steps)
    plt.plot(xaxis, coef_curve)

plt.xlabel('Steps taken')
plt.ylabel('Coefficient values')
plt.show()
