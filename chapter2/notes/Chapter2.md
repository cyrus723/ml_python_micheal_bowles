# Understanding the Problem by Understanding the Data
Attributes a.k.a Predictors, features, independent variables, inputs
Labels a.k.a outcomes, targets, dependent variables, responses

Types of attributes
* Numeric variables
    * __penalized linear regression works only with numeric variables__
    * KNN, kernel methods, SVM also work with numeric variables only
* Categorical variables (non-numeric)

Types of Labels
* Numeric labels
    * regression problem
* Categorical labels
    * (binary, multi-class) classification problem
    
####Items to check for new dataset
* Number of rows & columns (Size and shape of dataset)
* Number of categorical variables and unique values for them 
* Missing values 
    * Missing values cause **bias**
    * If data is missing, the row might be discarded or filled depending on the cost of data
    * Filling missing values is called **imputation**
* Summary statistics for attributes and labels 

#### Classification problems
Checks needed to be made on classification problem, before diving in

Check statistical properties of the data and interrelationships between attributes and between attributes and the labels.<br/>
If number of rows <= number of rows, prediction model may not work 
* Find how many rows and columns are there
* Find how many columns are numerical and how many are categorical
* Get statistical summaries of the dataset

**QQplot** to identify any possible outliers, by plotting relative to normal 
distribution using __scipy.stats.probplot__ in python.

Outliers can be correlated with errors in prediction
Correction measures
* Train them as separate class
* replicate poor performing examples to force them into heavy representation
* Edit them out if they are unreasonable 
* Draw quartile boundaries to estimate how big of a problem they are

##### Statistical characterizatino of categorical attributes
* Random forests package as a cutoff of 32 categories for any categorical 
attribute
* Stratified Sampling

##### Python Pandas
* Useful for early stage data inspection and preprocessing
* uses __data_frames__
* data frame is a table, matrix-like structure, where rows = samples, columns
 = attributes
* matrix =/= table because matrix has only numerical entries in all rows and 
columns. However, in a table, entry type is same in a column but it may vary 
across columns
* Single entry can be accessed using indices, rows, columns can be accessed/ 
sliced using names

* Sampling of data may require analysis of how data is available in the first
 place
 
##### Visualization properties of Rocks vs Mines dataset
* Visualizing with **parallel coordinate** plots
    * useful for problems with more than a few attributes
    * good for visualizing high dimensional geometry
    
##### Visualizing interrelationships between attributes and labels
* Crossplotting  pairs of attributes
    * pairwise relationships by cross-ploting the attributes with labels
    * **scatter plots**
* **Pearson's correlation**


##### Visualizing Attribute and label correlation using heatmap
* Find correlation between a large number of variables
* Perfect correlation = duplicate data
* Correlation > 0.7 ==> multicol linearity --> results in unstable estimates
* box and whisker plots == Box plots
    * Shows quartile with outliers
    * whiskers drawn at 1.4 times IQR    
    * data outside whiskers is outlier
* __Normalization__ means centering and scaling each column so that a unit of
 attribute 1 = unit of attribute 2
    * K-means clustering builds clusters based on vector distance between 
    rows of data
    * example of scaling 1 mile = 5280 feet
    * usual normalization  is mean=0 and S.D =1
    
    
##### Parallel coordinates for regression problems -- Visualize variable relationships for abalone problem
* parallel coordinates plot using color-coded line represent rows of data 
using their true classification
* to use them on regression problem, the color coding needs to be shades of 
color to correspond to higher or lower target values. Therefore, real values 
need to be compressed in [0.0, 1.0] range
* logit transform used to convert values in 0,1 range
* This plot normalizes the values of every variable to [0, 1] so that when we
 do a line plot, the difference of scale in values among multiple variables, 
 does not overshadow one or other variable. Gives better visual understanding
  compared to simple line chart.
* Number of colors = Number of classes or possible values taken by regression


##### Correlation Heatmap chart to plot the correlation among variables
* for regression the target labels are real number 
* for classification, the targets are binary, categorical data.

### WINE Tastes dataset
 
### Multiclass classification problem: Glass Type
* More than two target classes
    * Unlike regression, where there is order among the label values 2<3<4, 
    classification problems do not have any order relation among the 
    possible output labels. 
* Dataset
    * Many outliers
    * Unbalanced data
    