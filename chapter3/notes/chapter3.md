# Predictive Model Building: Balancing Performance, Complexity and Big Data
## Understanding Function Approximation
* **Function approximation problems**: 
    * We have two types of variables, those which needed to be predicted and 
    those that can be used for making predictions
* Variables to be predicted are: Target, labels, outcomes
* Variables used for making predictions are: predictors, regressors, 
features, attributes
    
##### Working with Training Data
* Attributes may be factor, categorical  or numeric, real-values
    * Categorical data can be converted tp linear, numerical values as well
* Regression: Targets are real valued
* Classification: Targets are categorical data
##### Accessing Performance of predictive models
* Regression problems
    * Real-valued targets: Performance using Mean Squared Error, Mean abslute 
    Error
* Classification problems
    * Accuracy etc.

##### Factors driving algorithm choices and Performance: Complexity and Data
###### Contrast between a simple and complex PROBLEM
* Out of sample error
    * Test using the sample not involved in training
* Performance depends on the complexity of the problem as well.
    * Complexity of decision boundaries
* Mixture model: Data drawn from several distributions
* Size of data also affect the performance
    * A complicated model with lots of data can show good performance but may
     not perform that good when used with small amounts of data
###### Contrast between simple and complex MODEL
* Having a complex model is not that useful unless the dataset isnt big enough
* Modern algorithms generate lots of models 
* Some times the performance using simple or complex model is roughly the same 


###### Factors Driving Performance of Predictive Algorithms
* **Shape** of data is also important
    * rows >>> columns
* Number of columns = Degree of freedom
    * model complexity is directly proportional to degree of freedoms
    
* Linear vs Non linear algorithms
* Linear models preferred if dataset has more attributes (columns) than rows
    * Training time is much faster
* Non Linear models are preferable when number of rows is much higher than 
number of columns 

##### Measuring the performance of predictive models
* Receiver operating curves
* Area under curve
###### Performance measures for different types of problems
* Regression problems 
    * error: difference between target and prediction
    * Mean squared error
    * Mean absolute error
    * root mean squared error
    * RMSE is more usable to calculate than MSE 
    * If MSE ~ target variance or RMSE ~ target standard deviation then the 
    algorithm is not performing well
    * Histogram of error
* Classification problem
    * Related to misclassification error rates
    * They general output probabilities
    * Confusion matrix, contingency table
* It is important to analyze the cost of False positives and false negatives 
to tune your classification model
* ROC curve
    * Overall performance of classifier instead of misclassification error 
    rate relevant to a particular threshold
    * plots TPR vs FPR
    * low threshold --> more false positives, less false negatives, TPR=1
    * **Diagonal** => random guessing results --> reference point
    
###### Simulating performance of deployed models
* performance is directly proportional to size of dataset
    * so very small training data will result in bad performance overall
* N-fold cross validation
    * Divide data into N partitions. N-1 used for training and 1 used for 
    testing
    * May tak more time to train because 90% data is being used in training
    * Sample (esp. testing sample) should be representative of whole dataset
    * Statistical peculiarities should be preserved in the testing sample 
    with great care, otherwise the results will be biased
        * Infrequent samples are often under/over represented by purely 
        random sample.
    * **Stratified Sampling**: divides data into separate subsets that are 
    separately sampled and then recombined.
        * sample infrequent items independently and then add them to final 
        testing samples
    
###### Achieving Harmony between model and data
* **Ordinary least squares (OLS) regression**
    * supervised machine learning algorithms
    * sometimes overfit the problem
        * overfitting => significant discrepancy between errors in testing 
        and training data
    * Cannot rollback overfitting
    * throttling overfitting
        * __forward stepwise regression__
        * __ridge regression__
    
###### Choosing a model to balance problem complexity, model complexity and data set size
* forward stepwise regression
    * Remove some columns to reduce overfitting 
    * brute-force method is called as __best subset selection__
    
###### Best subset selection: Forward stepwise regression
* Reduce chances of overfitting by controlling the number of attributes
* Choose a number of columns 
    * work on all subsets of that specific number of columns
    * perform OLS regression on all choices
    * identify the choice producing least out-of-sample errors
    * results = list of best choice of column subsets
* issues
    * too much calculation required

* Classifier performance using scatter plot
    * ideally all points should be at a line at 45 degree where predicted 
    labels = actual labels
* Machine learning algorithms perform bad at the edge

###### Ridge Regression: Penalizing regression coefficients
* Coefficient penalized regression: making coefficients for each attribute 
smaller (instead of making it zero) to control the participation of that 
attribute in overfitting the model.
* Range of values for alpha should cover a big range

* **Ridge regression method**: regression method with complexity tuning 
parameter
 
