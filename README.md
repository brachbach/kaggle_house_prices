Using a simple Kaggle challenge (https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) to practice Python and data science skills

The best entry point is to run the Jupyter lab (visualize_predictions.ipynb) 

TODO:
- regularize the features so that they have mean zero -- now that I think about it, this should be quite helpful
- convert non-numerical features into numbers so that they can be used
- try a ** 2, ab, etc. in a linear regression
- try using a simple neural net (maybe)
- switch to K folds or randomly selected samples for the train/validation split
- probably switch to 85% train, 15% test
- switch to a better imputation method, maybe multiple imputation: https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4

Questions for a mentor:
- am I making reasonable use of Jupyter Notebook?
- does it make sense to use `pipenv` like I'm using it?
- in general, is my code Pythonic?
- in general, does my coding style make sense for this sort of exploratory data science?
- is VSCode a reasonable editor to use?
- I know it's possible to use a neural net for a continuous value prediction problem like this -- is it advisable?

Curiosity questions:
- how do r values (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) relate to linear regression?