# Evaluation challenge: us-income

## Description
  This was an assignment we received during our training at BeCode.  
  We started from an already cleaning dataset about predicting the income of us-citizens based on 
  14 features. Our goal was to use the model to make correct predictions, using RandomForestClassifier.
  The focus of this project is model evaluation, more than model choice or preprocessing. 

  This will be your baseline accuracy against which you'll measure your improvements.

  Then you can get more complicated and use multiple evaluation metrics, see if there is a hint of overfitting, 
  tune your hyper-parameters to have a better score, etc...




## Installation
### Python version
* Python 3.9

### Databases
https://github.com/becodeorg/GNT-Arai-2.31/tree/master/content/additional_resources/datasets/US%20Income

### Packages used
* pandas
* numpy
* matplotlib.pyplot
* seaborn
* sklearn

[comment]: <> (## Usage)

[comment]: <> (| File                        | Description                                                     |)

[comment]: <> (|-----------------------------|-----------------------------------------------------------------|)

[comment]: <> (| main.py                   | File containing Python code.    <br>Used for cleaning and feature engineering the data |)

[comment]: <> (| plots.py                    | File containing Python code.   <br>Used for making some explanatory plots for this README. |)

[comment]: <> (| utils/model.py              | File containing Python code, using ML - Random Forest.   <br>Fitting our data to the model and use to it make predictions. |)

[comment]: <> (| utils/manipulate_dataset.py | File containing Python code.<br>Functions made for ease of use in a team enviroment. |)

[comment]: <> (| utils/plotting.py           | File containing Python code.<br>Used for getting to know the data.<br>Made plots to find correlations between features. |)

[comment]: <> (| csv_output                  | Folder containing some of the csv-files we used for our coding.<br>Not all of our outputted files are in here,   <br>since Github has a file limit of 100MB. |)

[comment]: <> (| visuals                     | Folder containing plots we deemed interesting and helped us gain   <br>insights on the data. |)

## Project process
### determine base accuracy
First step of the project is to run a default Random Forest classifier over the train set and predict
the test set. In the rest of the project I will try to better this score.

| Classifier model  | Accuracy score      | Set type | Random state | 
|------------------------|:----------------:|:-----:|:--------------:|
| RandomForrestClassifier | 0,99%  | train | 42 | 
| RandomForestClassifier | 0,85% | test  | 42 |

The model is overfitting: 
![](visuals/randomforest_default_score_test_train.png)

Looking at the confusion matrix, it is clear that the prediction of class 1 (income higher than 50k) can 
still do a lot better: 

![](visuals/randomforest_default_confusionmatrix.png)

### first improvement: cross-validation

Cross-validation on the train set leads to a less overfitted accuracy score. 

|cross validation score | number of folds  | standard deviation |
|-----------------------|------------------|--------------------|
|   85.82%              |        10        |     +/- 0.50       |


## Contributors
| Name                  | Github                                 |
|-----------------------|----------------------------------------|
| Maarten Van den Bulcke           | https://github.com/MaartenVdBulcke       |




## Timeline
13/08/2021 - 16/08/2021
