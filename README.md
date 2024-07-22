# deep-learning-challenge
The deep-learning-challenge project aims to identify if applicants for a funding scheme can be classified into those that have the best chance of success in their ventures (i.e., 1) or failure (i.e., 0) using supervised machine learning techniques, specifically using deep learning neural networks.


## Installation and Run Instructions:
You may need to install the **scikit-learn** and **keras-tuner** packages before commencing:

### scikit-learn:
1. Open **Gitbash** and activate your virtual environment.
2. Run **conda list scikit-learn**. If the response includes a version number for scikit-learn, then the library is already installed and you can safely skip the following steps. Otherwise:
3. Run **pip install scikit-learn** in the terminal.
4. Confirm installation by running **pip show scikit-learn**.

### keras-tuner:
1. Open **Gitbash** and activate your virtual environment.
2. Run **conda list keras-tuner**. If the response includes a version number for keras-tuner, then the library is already installed and you can safely skip the following steps. Otherwise:
3. Run **!pip install keras-tuner** in the terminal.
4. Confirm installation by running **pip show keras-tuner**.


## Usage Instructions:
This repo contains the following:
1. **lending_data.csv:** training data on 77,536 loans, labelled as either healthy (i.e., low-risk; 0) or unhealthy (i.e., high-risk; 0) as summarised by the **loan_status** column (i.e., the target variable). 
2. **credit_risk_classification.ipynb:** a python script, executed in JupyterLab, to reproduce the analyses.


## Overview of the Analysis:
The overall purposes of this analysis is to understand whether a parsimonous model can be built to assist lenders with classifying potential future loan applicants into those likely to pay their loan back on time (i.e., healthy or low-risk; 0) or not (i.e., unhealthy or high-risk; 1). To assist with this, lenders provided data on 77,536 loans, 75,036 (96.78%) of which were labelled as healthy and 2,500 (3.22%) of which were labelled as unhealthy, as summarised by the **loan_status** column (i.e., the target variable).

Several potential predictors of loan status (i.e, healthy vs. unhealthy) were also included in this training dataset:
1. **loan_size:** the loan amount requested, in AUD;
2. **interest_rate:** the interest rate applied to the loan;
3. **borrower_income:** the borrower's total income, in AUD, at application;
5. **debt_to_income:** the ratio of the borrower's total debt, in AUD, to their income, in AUD, at application;
6. **num_accounts:** the number of current accounts the borrower held at application,
7. **derogatory_marks:** whether any derogatory_marks were held about the borrower at application (boolean format), and
8. **total_debt:** the borrower's total debt, in AUD.

After reading in the original **lending_data.csv**, the following steps were used to build and evaluate the performance of the model:
1. columns were separated into the target variable (y; i.e., **loan_status**) and features (X, i.e., **loan_size**, **interest_rate**,	**borrower_income**,	**debt_to_income**,	**num_of_accounts**,	**derogatory_marks**,	**total_debt**);
2. The data was split using a 75/25 ratio into a training set and a testing set. Owing to the imbalanced nature of the original data, this split was stratified by the target variable to ensure adequate proportions of unhealthy or high-risk loans were represented in the training dataset;
3. A logistic regression model was then built, using data only for the training set;
4. Using this model, loan predictions were then made, using data only for the testing set;
5. Model performance was then evaluated using a confusion matrix and a classification report.


## Results:
Overall, the  accuracy of the model is 0.99, indicating that it correctly classifies 99% of the instances.

**Precision:**
* Healthy Loans: 1.00
* Unhealth Loans: 0.87

**Recall:**
* Healthy Loans: 1.00
* Unhealth Loans: 0.89


## Summary:
The logistic regression model performs exceptionally well in predicting healthy, low-risk loans (i.e., 100% precision, 100% recall). However, perhaps owing to the unbalanced nature of the training dataset, despite stratification, the model predicts unhealthy, high-risk loans less accurately, with lower precision (87%) and recall (89%). Despite this, the overall accuracy of the model is 99%, indicating that the model is highly reliable. The macro average (94%) indicates that the model maintains good performance across both classes, even when considering them equally, and the weighted average (99%) shows that the model's overall performance is excellent, albeit heavily influenced by the larger number of healthy loans. 

In choosing which model to recommend, consideration must be given to the consequences of misclassifying either a healthy or unhealthy loan. For lenders, the consequences of loaning to a potential defaulter far outweigh those of refusing to loan to a good creditor. Therefore, when trying to classify loans, greater consideration should be given to recall than precision. This is because a high recall minimises the chances of missing potential defaulters, ensuring that most of the risky loans are identified, even if it means potentially rejecting some good creditors.


## Credits:
This code was compiled and written by me for the credit-risk-classification challenge project in the 2024 Data Analytics Boot Camp hosted by Monash University. 



### Saving model outputs as HDF5 file:
https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format (Accessed 22 July 2024).

### keras tuner for auto-optimisation of model hyperparameters:
https://keras.io/guides/keras_tuner/getting_started/ (Accessed 22 July 2024).
