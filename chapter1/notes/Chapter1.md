## Chapter 1
####Functional approximation: 
supervised learning algorithms e.g. Linear, logistic regression etc.
applications: classifications, filtering, predictions

1. penalized linear regression methods
2. ensemble methods

#####Papers: 
1. An empirical comparison of supervised learning algorithms by Rich Caruana and Alexandru Niculescu-Mizil
2. An Empirical evaluation of supervised learning in High Dimensions by Rich Caruana, Nikos Karampatziakis, and Ainur Yessenalina


Attributes -> features -> used for prediction

**Penalized Linear regression**
* Linear and logistics regression are good for prediction when the number of features is very large
* Speed of training is very fast for penalized linear regression, good for scenarios where speed is very important
* Fast, easy to tune and deploy.   
* indicate which of the features are more important for predictions. 
* Feature engineering and feature selection are most time-consuming stages
* penalized linear regression is derivative of ordinary least squares 
    * **Ordinary least squares**
    * method for estimating unknown parameters in linear regression model
    * Parameters for linear function are chosen based on set of explanatory variables by minimizing sum of squares of differences between dependent variables (in dataset) and variables predicted using linear function
    * Issues
        * Overfitting
        * suffer if the data provided is not enough to get enough generalization
    * Require enough degrees of freedom where degree of freedom is the no. of independent variables available for prediction
    * if no. of degrees of freedom == no. of data points. prediction is awfully bad
    * if no. of samples < degrees of freedom --> penalized linear regression is good. 
    * penalized linear regression reduces the number of degrees of freedom to match the amount of data and complexity of underlying model

**Ensemble methods**
* built of multiple predictive models, where the output is combined by voting, averaging etc.
* individual models = **base learners**
* Used for overcoming the instability of invdividual models e.g. binary decision trees, traditional neural nets
* Binary trees
    * Node conditional values come from training over the input data
    * Bagging = bootstrap aggregating

**How to choose an algorithm**
* penalized lineary regression 
    * pros:
        * train quickly
        * accuracy not as good as ensemble or etc --> good for making _good enough_ guesses
        * speed of training is particularly helpful in the model development stages where multiple iterations are being performed for finding out best outputs/inputs etc.
        * faster predictions
* Choosing the best inputs (_feature selection_, _feature engineering_) is very important for solidfying mathematical problem statement
* Linear models are easy to train

**Process steps for building a predictive model**
* Define problems in words and then convert it to mathematical terms --> stating assumptions --> finding features 
* Steps:
    * Look at the available data to determine which of the data is usable for prediction, using statistical tools etc.
    * develop features
    * start training process
    * make performance estimations
    * improving performance
    
**Framing a machine learning problem**
**Feature extraction and Feature Extraction**
* feature extraction: arranging free-form data to numerically arranged data
* Try various combinations of features to verify the validity of your model
* Get feature rankings
* Train several models to cover the whole spectrum of problem complexity 

 
