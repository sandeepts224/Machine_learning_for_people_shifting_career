# Machine_learning_for_people_shifting_career
This page tries to reach ML from the mathematical ground to the python  
**1. Linear and Generalized Linear Models (GLMs)**

Ordinary Least Squares (Linear) Regression: The simplest form, modeling the relationship as a linear combination of features.
Ridge Regression: Adds L2 regularization to reduce overfitting by shrinking coefficients.
Lasso Regression: Adds L1 regularization, encouraging sparsity in coefficients.
Elastic Net Regression: Combines both L1 and L2 regularization.
Bayesian Linear Regression: Introduces prior distributions over parameters to manage uncertainty and overfitting.
Generalized Linear Models (GLMs): Extends linear regression to non-normal error distributions (e.g., Poisson, Gamma), linking predictors to the response via a link function.
**2. Nonlinear and Flexible Regression Techniques**

Polynomial Regression: Uses polynomial terms of the features to capture nonlinear relationships.
Spline Regression: Uses piecewise polynomial segments joined smoothly at “knots” to model complex curves.
Generalized Additive Models (GAMs): Extends linear models by allowing nonlinear functions of predictors while maintaining interpretability.
**3. Kernel and Support Vector-Based Regression**

Support Vector Regression (SVR): Uses kernels (e.g., linear, polynomial, RBF) to find a function that fits most data points within a certain epsilon-tube, minimizing complexity.
Kernel Ridge Regression: Combines ridge regression with kernel methods for non-linear pattern capture.
**4. Tree-Based Methods**

Decision Tree Regression: Uses decision rules based on feature thresholds to segment the space and predict outcomes.
Random Forest Regressor: An ensemble of decision trees trained on bootstrapped samples, improving stability and accuracy.
Gradient Boosted Regression Trees: Sequentially builds an ensemble of weak learners (trees) to minimize a loss function, examples include XGBoost, LightGBM, and CatBoost.
Extra Trees (Extremely Randomized Trees) Regressor: Similar to random forests but with more randomness in tree construction.
5. Nearest Neighbor Methods

k-Nearest Neighbors (kNN) Regression: Predicts the output as the average (or weighted average) of the nearest training points’ values, making no explicit assumptions about the function’s form.
**6. Gaussian Process Regression (GPR)**

Gaussian Process Regressor: A non-parametric Bayesian approach defining a prior over functions. It uses kernels to measure similarity between points, providing uncertainty estimates for predictions.
**7. Neural Network-Based Models**

Multilayer Perceptron (MLP) Regression: A feedforward neural network trained to minimize a loss function for continuous outputs.
Convolutional Neural Networks (CNNs) for Regression: Applied when input data has spatial structure (e.g., images).
Recurrent Neural Networks (RNNs), LSTMs, and GRUs: Used for sequential or time-series regression tasks.
Deep Learning Models with Advanced Architectures: Various architectures tailored for complex regression tasks (transformers, attention-based models, etc.).
**8. Ensemble Methods**

Bagging Regressors: Ensemble of models (often trees) trained on bootstrap samples and averaged to reduce variance.
Boosting Regressors: Like gradient boosting or AdaBoost, which iteratively improve on the previous model’s errors.
Stacked (Blended) Regressors: Combine multiple heterogeneous models’ predictions using a meta-learner.
**9. Dimension Reduction-Based Methods**

Partial Least Squares (PLS) Regression: Reduces predictors to a smaller set of uncorrelated components that explain covariance between predictors and response.
**10. Specialized and Other Methods**

Quantile Regression: Models conditional quantiles of the response, useful for understanding variability and uncertainty.
Bayesian Nonparametric Methods: Such as Gaussian processes (already listed) and others that provide flexible modeling capabilities.
Online and Incremental Learning Regressors: Methods designed for streaming data where models update continuously with new data.
