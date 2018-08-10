# Penalized Linear Regression
Resolves the overfitting problem faced by ordinary least squares (OLS)

Good to try on new problems, gives an initial boost to the process
* Vector norm: finding magnitude of vector
    * L2 norm:
        * calculate the shortest path to the point in the plain
        * circle shaped
    * L1 norm: Sum of absolute values of all coefficients
        * Taking streetwise path to a given point in the plane
        * diamond shaped
    * area under L1 norm is less than area captured by L2 norm
    
### Usefulness
* Extremely fast model training
    * Model building is iterative so its time consuming
    * Faster development process
* Variable importance information
    * Ranking attributes used for developing the model
    * Attribute rank is directly proportional to usefulness
* Extremely fast evolution when deployed
    * Linear models are extremely fast in evaluation
* Reliable performance on a wide variety of problems
    * Generates reasonable answers for all different shapes and sizes
    * **sparse model** easy to interpret since most of the coefficients are zero
    * easy to see what attributes are contributing to the predictions
* May require linear model
    * e.g. insurance payouts, models where coefficients and their weights can
     be written as simple equations
* When to yse ensemble methods
    * in complicated problems
    * when second or higher order polynomial describe the relationship 
    between attributes

##### Penalized linear regression: Regulating linear regression for optimum performance
* Linear method work with numeric data only
* Guessing the value of coefficients in linear method is tough.
* Adding a penalty coefficient to OLS formulation
    * similar to ridge regression (minimize using penalty term)
    * solve the problem for various values of lambda and then choose the best
     performing one.
* Ridge penalty coefficient
    * Puts an upper bound on how big the betas can get, to prevent overfitting
    * Uses euclidean geometry i.e. sum of squared errors (L2 Norm)
    * all coefficients are completely populated
    * The shape of constraints is a circle (because its squared betas)
* Manhattan penalty coefficient
    * Used by **Lasso regression**
    * Uses taxicab geometry parameter called manhattan length or L1 (sum of 
    absolute B's)
    * L1 norm is linear sum of betas
    * Lasso's coefficient vector is sparse i.e. many coefficients are zero
    * In lasso, the constraint is shaped diamond (because the betas are 
    absolute)
    * Due to the diamond shape, the tangent will probably be at the edge, of 
    diamond, therefore, there is more chance that some of the betas will 
    become zeros as all corners are at some axis
* The point where the plots of constrained betas and unconstrained betas 
touch (tangent) each other, is the optimal value for beta, which is close 
enough to global minimum, and also within constrained space. Figure 4-1, 4-2 
in book
* Elasticnet penalty coefficient 
    * Includes adjustable blend of both Lasso and Ridge

###### Solving the penalized linear regression problem
* **Least Angle Regression (LARS)**
    * Improvement on forward stepwise algorithm
    * The difference is that it only partially incorporates new attributes 
    instead of using them fully as done in forward stepwise algorithm
    * Process
        * Initialize all betas to zero
        determine which attribute has the largest correlation with residuals
        * increment that variable coefficient by a small amount if the 
        correlation is positive, or decrement otherwise
    * Results ~ similar to Lasso
    * **LARS** generates hundreds of models of varying complexity
    * final result = coefficient curves
        * vertical line intersects all coefficient curves 
        * point where vertical line intersects coefficient curves are 
        coefficients at that step in the evolution of LARS
    * coefficient curves reveal the sparsity of Lasso regression
* Finding useful features: help in feature engineering
* Choosing the best model from Hundreds LARS generates
    * using 10-fold cross validation
    * **Stratified sampling**: Divide data by classes and then take samples 
    from each class and form training, testing data
* Mechanizing cross-validation for model selection in python code
* Accumulating errors on each cross validation fold and evaluating results
    * better solution using LARS --> most conservative solution --> solution 
    with small coefficient values 
* Less complex models have better generalization errors, better performance 
on newer data
* cross-validation is a measure to understand how the complexity of model 
will work with the dataset
####### Glmnet: Very fast and very general 
* Developed by Professor Jerome Friedman
* Solves ElasticNet Problem (Eq.4-11)
    * Uses both Lasso and ridge penalty
    * lambda parameter determines how heavily coefficient penalty is penalized
    * Produces coefficient curves, similar to LARS algorithm
####### Comparison between Glmnet and LARS algorithms
* **LARS**: Find the attribute with the largest magnitude correlation with 
the residual and increment its coefficient by a small fixed amount
    * Once B_{j} stops changing, final values of lambda and alpha are achieved
    * Tries to find which attribute has biggest correlation with residual, 
    and the coefficient corresponding to the attribute with highest correlation

* **Glmnet**: correlation between residuals is used to calculate how much 
coefficient ought to be changed in the magnitude
    * soft limiter: Lasso coefficient shrinkage function
    * Initializing and iterating the Glmnet Algorithm
    * lambda should be reduced such that lambda^100=0.001 -> lambda ~ 0.93
        * if the solution is not converging, then lambda should be updated to
         ~ 1
#### Extensions to linear regression with numeric input
#####Solving classification problems with Penalized Regression
* Binary classification mapped to 0 and 1 
    * Faster training
* Multiclass classifications
    * Can be treated as binary class problem, one-vs-rest, one-vs-all
    * a combination of multi-class problem
##### Understanding basis expansion: using linear methods on non-linear problems
* Basis expansion is used for making linear models work with 
strong non-linearities.
    * Main idea: non linearities in the problem can be approximated as 
    polynomials of the attributes (or sum of other nonlinear functions of 
    attributes); add the attributes that are powers of the original attribute
     and let a linear method determine the best coefficients for the polynomial
    * Linear model yields coefficients in a polynomial function of the 
    original variable
    * Various function series can be developed for the linear models
##### Incorporating non numeric attributes into linear methods
* Standard method for converting categorical variable to numeric data: code 
them into new columns of attributes.
    * N possible values coded to N-1 columns of data
    

  