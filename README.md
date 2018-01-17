# Allstate

These are the codes I written for reaching 95th place (top 4%) in kaggle's [Allstate Claims Severity Prediction Competition](https://www.kaggle.com/c/allstate-claims-severity).

This is a three-level stacking model. The first two levels are xgboost and Keras, while the last level is simple linear regression. All training and stacking are done with K-fold cross-validation to minimize information leakage. Feature engineering and parameter tuning are performed probabilistically through [Bayesian optimization](https://github.com/fmfn/BayesianOptimization/tree/master/bayes_opt).