# Building Predictive Models using Penalized Linear Methods
* Penalized linear regression model generation steps
    * Optimize the coefficients
    * check out of sample errors to find best coefficient value
## Python Packages for penalized linear regression
* Scikit-learn has implementations
    * Lasso, LARS, ElsaticNet
    * These packages use techniques like not computing correlations for 
    attributes that are not being used in order to cut down calculations.
    * **sklearn.linear_model**
        * Two inputs:  < numpy arrays > one for attributes and one for labels
    * Some packages automatically normalize the inputs and labels
    * scikit-learn packages may use different parameter names e.g. (for 
    variables mentioned in Chap4) lambda is alpha (scikit), alpha is 
    l1_ratio (scikit)
## Multivariable Regression: Predicting Wine Taste
* Error measures for regression problem
    * Regression error =  average squared error
    * The difference in predicted vs actual value contributed accordingly to 
    the final error
* Error measure for multiclass problem
    * Number of misclassified examples
    * Misclassification contributes equally to the final no matter how much 
    is the deviation between actual and predicted value
### Building and testing a model to predict wine taste
* First step: generate out-of-sample performance measures to see whether they
 are going to meet performance requirements
* After normalization, MSE losses its connection to original data
* It is handy to extract sq.root of MSE and relate it to original labels
* Either normalize X or be wary about not normalizing them 
### Training on whole dataset before deployment
* Training on whole dataset gives you value of alpha for best performance
    * alpha in the python scripts corresponds to penalty term lambda 
    discussed in algorithms
    * magnitude of coefficients determines the relative importance of 
    attributes but this ranking only makes sense when attributes are normalized
    * if an attribute is not normalized, its scale factor determines its 
    usage instead of its inherent value in predicting the labels
### Basis expansion: improving performance by creating new variables from old ones
* In some cases, many variable combinations may be needed to find the best 
performance
    * Synthetic variables can replace the older original variables
    
## Binary Classification: Using penalized linear regression to detect unexploded mines
* Solving binary classification problem using penalized linear regression and
 python elasticnet package
* Convert binary classification problem to regression problem by giving 
real numbers to classification labels
* During cross-validation, calculate error quantity for each fold
    * l1_ratio is determines what fraction of the penalty is sum of absolute 
    values of coefficients
    * l1_ratio -> represents percentage to which sum of absolute values 
    contribute in penalty function
    * 1- l1_ratio -> represents percentage to which sum of squares 
    contribute in penalty function
    * if the values are normalized, there is no need to fit_intercept: 
    Intercept is used to adjust any constant offset between attributes and 
    labels
* Down side of normalizing:
    * MSE calculations become less meaningful relative to the regression 
    problem, but this doesn't affect the classification problem anyways
* Performance measurement using AUC
    * Benefit: maximize performance independent on intended operation i.e. 
    independent about what kind of errors need to be optimized
    * maximizing AUC is not representative of best performance
    * right choice: compare models chosen by AUC, minimizing overall error 
    rate and shape of curves to get confidence in solution
    * **roc_auc_score** program calculates the AUC by itself
* Confusion Matrix
    * can be generated from roc_curve
    * performance varies with threshold chosen
        * smaller threshold: high true positive, false positive
        * larger threshold: high true negative, false negative
        * set threshold based on the cost of errors, choose threshold which 
        minimize the overall costs
        * ensure that positive and negative values are in similar quantity to
         get accurate representation of costs in determining final costs   
* cross validation ensures that model isn't overfit as long as the training 
data is statistically similar to what the model will see in deployment.
* **Penalized logistics regression**
    * calculates the probabilities and likelihood of certain outcome instead 
    of outright claiming it as right prediction
    * weights are derived based on probabiloity etimates for each example in 
    the training set
    * the problem can become weighted least squares regression problem
* **non-penalized logistic regression**: iteratively reweighted least squares.
* Process
    * Read variables
    * normalize
    * estimate probabilities and weights with the coefficients (betas) each 
    time the penalty parameter is decremented
    * probabilities __p__ and weights __w__ are calculated one input example 
    at a time
    * the effects of weights on the sum of products like __attributes times 
    residuals__ and __squares of attributes__ need to be collected
* both plain and penalized logistic regression generates vectors of 
coefficients and then multiple the same attributes by them and compare to a 
threshold
* log odds ratio = natural log of the odds ratio
    * Large and positive log odds value means that there is high probability 
    that given case belongs to a specific class
    
## Multiclass classification: classifying crime scene glass
* more than two labels
* one-vs-all approach
    * as many vectors of labels as there are distinct labels
    * each vector is actually a plane dividing the output space
    * Train as many binary classifiers as there are labels
        * Every classifier checks if the given sample belongs to this label or not
* enet_path
    * eps parameter: used to control the range of penalty parameter values 
    that are covered in the training
        * tells the algorithm where to stop decrementing
        * ratio of the stopping value of the penalty parameter divided by 
        starting values.
    * n_alphas: controls the number of steps
        * too large steps may prevent algorithm convergence 
        * increasing eps will control how the penalty parameter gets decremented
    * minimum should be towards the right of graph, giving enough curve to 
    the graph
    
* Non-linear kernals in SVM ~= basis expansion 
    * non-linear kernels do show better performance than basis expansion
    