# Ensemble Methods
* Multiple models give better performance than a single model if the models 
are independent of one another
* Cumulative binomial probability
* Low level algorithms: **base learners**
    * single ML algorithm that gets used for all of the models that will get 
    generated into an ensemble
    * Inputs to base learner are altered in a way that generated models are 
    independent
* upper level algorithms: Bagging, boosting, random forests
    * random forest is combination of upper level algorithm and particular 
    modification of decision trees
    * binary decision trees are most widely used as base learner
## Binary decision trees
* operate by subjecting attributes to a series of binary (yes/no) decisions
* output = Number of boxes, each box is called a node in decision tree prelance
    * Nodes that either pose a yes/no question, non-terminal nodes
    * Nodes terminate without any further branches, called leaf nodes
        * Leaf node -> mean of outcome of all the training observations 
        wounding up in the leaf node
    * top most node: root node
        * root node represents most contributing feature/attribute
* Depth of tree represents the number of decision needed to be made in order 
to reach the leaf node  
### How binary decision tree generates predictions
* At each non-terminal node, incoming sample is evaluated and based on a 
yes/no response, the left or right branch is processed in the tree, until it 
reaches the terminal or leaf node

### How to train a binary decision tree
* Listing 6-1

* Tree training Equal Split point Selection
    * depth 1 trees = __stump__
        * single split point in stumps is called split point
*How split point selection affects predictions
    * study comparison between predicted and original performance
* Algorithm for selecting split points
    * a Tree consists of three key points i.e. split points, two values in 
    each branch
    * Finding appropriate split point is important for predictions
    * choose branches
        * split point is chosen in a way that each branch (group) minimizes the 
        squared sum squared error
* Multivariable Tree training - Which attribute to split?
    * if there are more than one attribute, all possible split points for all
     attributes are checked to see which split points give best sum squared 
     error for each attribute and which attributes give overall minimum
* choosing split points is most computationally intensive task
    * too many split points result in close split points as well
    * large datasets, split points are coarser
* **Planet: Massively parallel learning of tree ensembles with MapReduce**

* Recursive splitting for more tree depth
    * Increase in splitting decreases the number of examples in the deepest 
    nodes, therefore, sometimes splitting terminates before specified depth

### Overfitting Binary trees
* measuring overfit of binary trees
* estimate overfitting
    * Comparing the number of terminal nodes to the amoutn of data available
    * if number of datapoints ~= number of terminal nodes then its overfittings
* Tree depth calculates the complexity of binary tree model
    * depth gives tradeoff between reproducing underlying relationships and 
    overfitting the problem
    * optimum complexity is a function of dataset size

### Modifications for classification and categorical features
* Measures for classification performance, measure misclassification error
    * **Gini impurity measure**
    * **Information gain**
* training trees on categorical attributes
    * split into all possible combination of categorical variables 
    
* Decision trees are not probably that useful in themselves but when used in 
large numbers (for ensemble), their individual problems go into background 
because of large number of trees considered.
* binary decision trees can result in large variance in performance
## Bootstrap Aggregation: Bagging 
* developed by Leo Breiman
    * binary decision trees as base learner
### How does bagging algorithm works?
* bootstrap sample
    * used for generating sample statistics from a modest dataset
    * it is a random selection of several elements from the data set with 
    replacement i.e. it can contain multiple copies of row from original data
* More than one bootstrap samples are from the training data are used for 
training a base learner, on each of these samples
* for classification problems: models can be averaged, probabilities can be 
developed based on percentages of different classes
* 30% data used for measuring out-of-sample performance, instead of using 
cross validation
* Accuracy vs the number of trees
* **Best MSE for a predictive algorithm is the square of standard deviation**
### Bagging Performance--Bias versus variance
* two types of errors
    * **bias**: errors that don't get smaller as the number of data points 
    increase are called bias errors
        * model accuracy suffers at the edge of data because all split points
         are chosen at the center of data.
        * Bagging reduces variance between models but with depth-1 trees, it 
        gets a bias error, which can not be averaged.
        * It can be overcome by using trees with more depth
    * **variance**: 
* Bagging needs tree depth for performance
    * depth-1 trees can only consider solitary attributes and cannot account 
    for strong interactions between variables
* Bagging reduces variance exhibited by individual binary trees bu the trees 
should be of some depth to ensure that bagging works.
## Gradient Boosting
* Developed by Jerome Friedman
* Tree ensemble is generated by training each of the trees in the ensemble on
 different labels and then combining the trees.
    * for regression problem, where goals is to minimize MSE, each successive
     tree is trained on the errors left over by the collection of earlier 
     trees. 
* Basic principle of gradient boosting algorithm
* can reduce **bias** as well as **variance**
### Parameter settings for Gradient Boosting
* Setting depth of individual trees being trained in gradient boosting ensemble
* tree depth needed ~= enough for significant interaction between variables.
    * once can find the correct depth needed for variable interaction by 
    trying variable tree depths
* **eps** = step size control for gradient descent steps
    * too large steps, optimization will diverge instead of convergence
    * too small stepsize, optimization can take too long
* **residual** prediction errors, residuals are calculated after each step. 
Initially they are 0, meaning that residuals are equal to observed labels
* How gradient boosting iterates toward a predictive model
    * Labels are used on for first iteration and after that, residuals are 
    used for training
    * Further passes predict training labels and substract eps of them from 
    residuals before training
    * Out of sample errors are calculated on held out sample.
### Getting the best performance from gradient boosting
* **bias** error is inherent in using shallow trees to built prediction for a
 number of problems
* gradient boosting gives more attention to the areas where its making mistakes
* More tree depth, more training time
* Bagging experienced bias error due to shallow trees
* Boosting constantly monitors its cumulative error and uses that residual 
for subsequent trainign

## Random Forests
* Generate sequence of models by training them on subsets of data
    * subsets are created by choosing samples at random
    * sometimes even the attributes are chosen at random 
### Random Forests: Bagging plus Random Attribute subsets
* It is often recommended that for each node, a random set of attributes 
should be drawn, but it has been said that it may not give that much advantage
* Random forests performance drivers
* Combination of bagging and random attribute selection modification to the 
binary tree base learners. 
* easier to parallelize compared to bootstrap aggregation
    * individual base learners can be trained independently of one another 
* good performance for wide spare attribute spaces like text mining

