# Cardiovascular-Disease-Dataset using Machine Learning

A comprehensive dataset designed to support research and analysis related to cardiovascular diseases. It includes various attributes such as patient demographics, medical history, lifestyle factors, and clinical diagnostics. This dataset can be utilized for predictive modeling, data-driven insights, and healthcare advancements.The process model is implemented in a Python script that using Logistic Regression and Random Forest model, Feature Engineering.

## Problem Statement:
Cardiovascular diseases (CVD) are among the leading causes of mortality worldwide, requiring early diagnosis and preventive measures to mitigate their impact. The Cardiovascular Disease Dataset (id 45547) from OpenML provides a collection of medical and lifestyle variables associated with the presence or absence of CVD. The dataset contains objective, examination, and subjective features, all recorded at the time of medical examination. These features include patient demographics (age, height, weight, gender), medical examination results (blood pressure, cholesterol, glucose levels), and lifestyle habits (smoking, alcohol consumption, physical activity).
The primary challenge is to leverage this dataset to develop predictive models that accurately identify individuals at high risk for cardiovascular disease. Given the mixed nature of the features—objective, examination, and subjective—the dataset presents an opportunity to explore the influence of lifestyle factors alongside traditional medical indicators. Additionally, data quality considerations, including the potential biases in self-reported information and variations in measurement techniques, must be carefully addressed to ensure robust analysis.

The goal is to utilize this dataset to:
1.	Identify key predictors of cardiovascular disease.
2.	Develop predictive models to support early diagnosis and intervention.
3.	Analyze the interplay between medical examination results and lifestyle factors.
4.	Enhance understanding of cardiovascular health through data-driven insights.
By harnessing the insights from this dataset, researchers and healthcare professionals can advance early detection efforts and design targeted interventions to reduce the prevalence of cardiovascular disease globally.

Dataset retrieved from OpenML Api: https://www.openml.org/search?type=data&status=active&id=45547

## Data description:
### There are 3 types of input features:

These categories describe different types of data used in the Cardiovascular Disease dataset:

Objective: These are factual, measurable data points that don't depend on interpretation. Examples might include age, gender, or blood pressure measurements recorded in a standardized manner.

Examination: This refers to results obtained from medical tests or assessments performed on the patient. These can include blood test results, cholesterol levels, ECG findings, or any other diagnostic test that provides quantifiable results.

Subjective: This category includes information provided by the patient, often based on their personal experience. These could be self-reported symptoms, lifestyle habits (like smoking or alcohol consumption), or feelings about their health.

Together, these three types of input features help in assessing cardiovascular health and predicting disease risk by combining unbiased factual data, medically examined results, and patient-reported perspectives. 

The target variable (cardio) is a binary indicator showing whether an individual has cardiovascular disease (1) or not (0). Since all values were collected at the time of examination, the dataset represents a snapshot of each patient’s health at that moment, helping in risk assessment and predictive modeling.
Certain features tend to be strong predictors of cardiovascular disease. Based on research and machine learning models, the most predictive features often include:

* Age: Older individuals generally have a higher risk.
* Systolic and Diastolic Blood Pressure: High blood pressure is a major risk factor.
* Cholesterol Levels: Elevated cholesterol, especially LDL, is strongly linked to heart disease.
* Glucose Levels: High blood sugar levels can indicate diabetes, which increases cardiovascular risk.
* Smoking: A well-established risk factor for heart disease.
* Alcohol Intake: Excessive alcohol consumption can contribute to hypertension and other heart-related issues.
* Physical Activity: Lack of exercise is associated with higher cardiovascular risk.

Machine learning models often use these features to predict cardiovascular disease risk with high accuracy. Some studies have explored advanced techniques like cluster analysis to improve risk stratification. If you're working on a predictive model, feature selection and correlation analysis can help refine which variables are most impactful.

### Features:
1.	Age | Objective Feature | age | int (days)
2.	Height | Objective Feature | height | int (cm) |
3.	Weight | Objective Feature | weight | float (kg) |
4.	Gender | Objective Feature | gender | categorical code |
5.	Systolic blood pressure | Examination Feature | ap_hi | int |
6.	Diastolic blood pressure | Examination Feature | ap_lo | int |
7.	Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
8.	Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
9.	Smoking | Subjective Feature | smoke | binary |
10.	Alcohol intake | Subjective Feature | alco | binary |
11.	Physical activity | Subjective Feature | active | binary |
12.	Presence or absence of cardiovascular disease | Target Variable | cardio | binary |

All of the dataset values were collected at the moment of medical examination.

## The following is a summary of the Cardiovascular-Disease-dataset and its key characteristics:
* Number of instances – The dataset contains 70,000 data points (rows).
* Number of features – There are 12 attributes (columns) used for analysis.
* Number of classes – It has 2 classes, likely indicating a binary classification problem (e.g., presence vs. absence of disease).
* Number of missing values – 0, meaning no missing data.
* Number of instances with missing values – Again 0, confirming data completeness.
* Number of numeric features – 5 features are numeric (e.g., age, blood pressure).
* Number of symbolic features – 7 features are symbolic/categorical (e.g., gender, smoking status).

### This dataset is clean and well-structured, making it a great candidate for machine learning models. The first few rows of the Cardiovascular-Disease-dataset. Here's a breakdown of what each column represents:
* age – The person's age.
* gender – Likely encoded as 1 (male) and 2 (female).
* height & weight – Physical measurements.
* ap_hi & ap_lo – Systolic (ap_hi) and Diastolic (ap_lo) blood pressure readings.
* cholesterol – Categorical values (1: normal, 2: above normal, 3: well above normal).
* gluc – Blood glucose levels (1: normal, 2: above normal, 3: well above normal).
* smoke – Whether the person is a smoker (0: No, 1: Yes).
* alco – Whether the person consumes alcohol (0: No, 1: Yes).
* active – Whether the person is physically active (0: No, 1: Yes).
* cardio – Target variable (0: No cardiovascular disease, 1: Has cardiovascular disease).




## Install required packages

%pip install openml

%pip install pandas 

%pip install scikit-learn 

%pip install matplotlib 

%pip install seaborn

## Load the Data Set

import openml
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier  # Added XGBoost
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans






### Reference:

Feurer, Matthias. (2023). Cardiovascular-Disease-dataset. OpenML. Retrieved from OpenML. 
Matthias Feurer, Ph.D. candidate at the University of Freiburg, Germany. Working on automated machine learning. Creator of the python API for OpenML. Germany 2014-07-02 17:15:35
https://www.openml.org/search?type=data&status=active&id=45547









