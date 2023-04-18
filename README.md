## Financial Distress Prediction
### Overview
This is a Kaggle competition that requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years.
### Objectives
Build a model that borrowers can use to help make the best financial decisions. **Evaluation:** AUC
### Tecnologies Used
* python
* pandas
* numpy
* matplotlib
* sklearn
* skopt
* imblearn
* shap
* xgboost
* lightgbm
### About the Data
Historical data are provided on 250,000 borrowers. Below, some statistics:




### Methodology
Several classification algorithms were tested in order to choose the one with the best performance. Due to data imbalance, we trained the algorithms with two methodologies, the first with oversampling and the second unbalanced. As can be seen below, oversampling did more harm than good to the performance of most algorithms. Therefore, we chose not to use balanced data. Then, applying Bayes optimisation, we tuned the hyperparameters of the best performing algorithm. Finally, we used SHAP to verify which attributes were most important in the decision making of the algorithm.
### Results and Conclusions
Kaggle returned scores **~0.867** for private score and **~0.861** for public score. It's a good result, considering that winer got **0.86955**.

We filled null values with the median of the attributes, but MonthlyIncome and NumberOfDependents might still be related to the customer's life stage, for example, newer customers might have salaries below the median. An improvement that could be done is use a median by age instead of full dataset.

Othar thing that could be done to it's make a stack with top 3 algorithims. We let that for future studies.

### References
* https://www.kaggle.com/competitions/GiveMeSomeCredit
