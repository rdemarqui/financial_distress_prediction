## Financial Distress Prediction
### Overview
This is a Kaggle competition that requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.
### Objectives
Build a model that borrowers can use to help make the best financial decisions. **Evaluation:** AUC
### Tecnologies Used
* python 3.9.16
* pandas 1.5.3
* numpy 1.22.4
* matplotlib 3.7.1
* sklearn 1.2.2
* skopt 0.9.0
* imblearn 0.10.1
* shap 0.41.0
* xgboost 1.7.5
* lightgbm 3.3.5
### About the Data
Historical data are provided on 250,000 borrowers.

|Variable Name|Description|Type|
|---|---|---|
|SeriousDlqin2yrs|Person experienced 90 days past due delinquency or worse|Y/N|
|RevolvingUtilizationOfUnsecuredLines|Total balance on credit divided by the sum of credit limits|percentage|
|age|Age of borrower in years|integer|
|NumberOfTime30-59DaysPastDueNotWorse|Number of times borrower has been 30-59 days past due|integer|
|DebtRatio|Monthly debt payments|percentage|
|MonthlyIncome|Monthly income|real|
|NumberOfOpenCreditLinesAndLoans|Number of Open loans|integer|
|NumberOfTimes90DaysLate|Number of times borrower has been 90 days or more past due.|integer|
|NumberRealEstateLoansOrLines|Number of mortgage and real estate loans|integer|
|NumberOfTime60-89DaysPastDueNotWorse|Number of times borrower has been 60-89 days past due|integer|
|NumberOfDependents|Number of dependents in family|integer|


Below, some statistics:




### Methodology
Eleven classification algorithms were tested in order to choose the one with the best performance. Due to data imbalance, we trained the algorithms with two methodologies, firstly with oversampling and the secondly with unbalanced. Then applying optimisation (Bayesian and randomic), we tuned the hyperparameters of the best performing algorithm. Finally, we use SHAP to verify which attributes were most important to algorithm.
### Results and Conclusions
As can be seen below, oversampling did more harm than good to the performance of most algorithms. Therefore, we chose not to use balanced data to train.

LGBMClassifier was choosen given it's best result compared to others. Kaggle returned scores **~0.867** for private score and **~0.861** for public score. It's a good result, considering that winer got **0.86955**.

As we can see above, LGB 

We filled null values with the median of the attributes, but MonthlyIncome and NumberOfDependents might still be related to the customer's life stage, for example, newer customers might have salaries below the median. An improvement that could be done is use a median by age instead of full dataset.

Othar thing that could be done to it's make a stack with top 3 algorithims. We let that for future studies.

### References
* https://www.kaggle.com/competitions/GiveMeSomeCredit
