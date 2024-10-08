<div align="center">
  <h1>Financial Distress Prediction</h1>
</div>

<p align="center">
<img src="images\cover.jpg" class="center" width="40%"/>
</p>

### Overview
<p align="justify">
Credit score is crucial for companies as it helps them assess the creditworthiness of potential customers and partners. It enables informed decisions regarding loans, partnerships, and credit terms, ultimately minimizing financial risks and promoting responsible financial management.

<p align="justify">
This work was done based on the Kaggle competition that requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.

### Objectives
<p align="justify">
Build a model that borrowers can use to help make the best financial decisions. <b>Evaluation:</b> AUC

### Tecnologies Used
* `python 3.9.16`
* `pandas 1.5.3`
* `numpy 1.22.4`
* `matplotlib 3.7.1`
* `sklearn 1.2.2`
* `skopt 0.9.0`
* `imblearn 0.10.1`
* `shap 0.41.0`
* `xgboost 1.7.5`
* `lightgbm 3.3.5`

### About the Data
<p align="justify">
Historical data of 250,000 borrowers were provided, as described below.

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

### Methodology
<p align="justify">
Exploring the dataset, we found that attributes <code>MonthlyIncome</code> and <code>NumberOfDependents</code> had null values and median was used to fill them. Also checking feature correlation we've seen that the features <code>NumberOfTime30-59DaysPastDueNotWorse</code>, <code>NumberOfTime60-89DaysPastDueNotWorse</code> and <code>NumberOfTimes90DaysLate</code> had a high correlation. In this case, we could maintain only <code>NumberOfTime30-59DaysPastDueNotWorse</code>, whereas borrowers who had been late for longer periods were late first in this shorter interval, but in previous tests, we've seen that it costs some accuracy points. As accuracy is a sensitive theme in this type of case, we decided to maintain all features.

<p align="center">
<img src="images\feature_correlation.png" class="center" width="70%"/>
</p>

<p align="justify">
After that, using stratified k-fold cross-validation technique, eleven classification algorithms were tested in order to choose the one with the best performance. Due to data imbalance (N=93% and Y=7%), we trained applying two methodologies, firstly with oversampling and secondly with unbalanced data. Then applying optimisation (Bayesian and randomic), we tuned the hyperparameters of the best performing model. Finally, we used SHAP to verify which attributes were most important to the model.

<p align="justify">
The complete study can be replicated in the notebook <code>Financial_distress_prediction.ipynb</code>, available <a href="https://github.com/rdemarqui/financial_distress_prediction/blob/main/Financial_distress_prediction.ipynb">here</a>.

### Results and Conclusions
<p align="justify">
As can be seen below, oversampling (mean_over) did more harm than good to the performance of most algorithms. Therefore, we chose not to use balanced data to train.

<p align="center">
<img src="images\performance_table.png" class="center" width="50%"/>
</p>
      
**LGBMClassifier** <p align="justify"> was chosen given it's best result compared to others. Kaggle returned AUC scores of ~0.867 for private score and ~0.861 for public score. It's a good result, considering that winer got 0.86955.

<p align="justify">
Analyzing chart below, we can see that the model considered <b>RevolvingUtilizationOfUnsecuredLines</b> as the most important feature, followed by <b>NumberOfTime30-59DaysPastDueNotWorse</b> and <b>age</b>. We found that customers with high credit utilization, a history of late payments, and a young age are much more likely to experience future financial difficulties.

<p align="center">
<img src="images\shap.png" class="center" width="50%"/>
</p>

**Future improvements proposal:** <p align="justify"> We filled null values with the median of their respective attributes, but <code>MonthlyIncome</code> and <code>NumberOfDependents</code> might still be related to the customer's life stage, for example, younger customers might have salaries below the median and no dependents. An improvement that could be made is using a median by age instead of a median of the full column. We tried oversampling method, but undersampling could be tried too. Another thing that could be done it's, instead use only one model, make a stacking with top-ranked models or the best one with different seeds. We let that for future studies.

### References
* https://www.kaggle.com/competitions/GiveMeSomeCredit
