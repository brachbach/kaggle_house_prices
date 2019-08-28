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
- add tests

Questions for a mentor:
- am I making reasonable use of Jupyter Notebook?
- does it make sense to use `pipenv` like I'm using it?
- in general, is my code Pythonic?
- in general, does my coding style make sense for this sort of exploratory data science?
- is VSCode a reasonable editor to use?
-- how do I get VSCode to tell me if I'm e.g. referring to a variable that doesn't exist?
- can I make better (more) use of types?
- I know it's possible to use a neural net for a continuous value prediction problem like this -- is it advisable?
- best way to do unit and integration tests for data science?

Curiosity questions:
- how do r values (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) relate to linear regression?

Data in a Google Sheet: https://docs.google.com/spreadsheets/d/15wNqB5NCo_7YUDzkEk-orAfL7LChEXrDAsE-YTRDncM/edit#gid=0

Reading about the data (http://jse.amstat.org/v19n3/decock.pdf):
- Potential Pitfalls (Outliers): Although all known errors were corrected in the data, no
observations have been removed due to unusual values and all final residential sales
from the initial data set are included in the data presented with this article. There are
five observations that an instructor may wish to remove from the data set before giving
it to students (a plot of SALE PRICE versus GR LIV AREA will quickly indicate these
points). Three of them are true outliers (Partial Sales that likely don’t represent actual
market values) and two of them are simply unusual sales (very large houses priced
relatively appropriately). I would recommend removing any houses with more than
4000 square feet from the data set (which eliminates these five unusual observations)
before assigning it to students.
- , I
defer all the time series material (distributed throughout our text) until the end of the semester as
this material is unnecessary for the analysis of the project. 
- the link to the description of the data is broken :( -- http://www.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt)