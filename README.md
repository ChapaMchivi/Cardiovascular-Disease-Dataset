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

* %pip install openml
* %pip install pandas
* %pip install scikit-learn
* %pip install matplotlib
* %pip install seaborn

## Load the Data Set

* import openml
* import pandas as pd
* import os
* import matplotlib.pyplot as plt
* import seaborn as sns
* from scipy.stats import ttest_ind, chi2_contingency, f_oneway
* from sklearn.preprocessing import StandardScaler
* from sklearn.model_selection import train_test_split
* from sklearn.linear_model import LogisticRegression
* from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
* from xgboost import XGBClassifier  # Added XGBoost
* from sklearn.decomposition import PCA
* from sklearn.cluster import KMeans

## Define dataset ID
dataset_id = 45547
csv_filename = "cardiovascular_disease_dataset.csv"

### Check if dataset exists locally to avoid re-downloading
try:
    if os.path.exists(csv_filename):
        
        print("Loading dataset from local CSV...")
        
        df = pd.read_csv(csv_filename)
    
    else:
        
        print("Fetching dataset from OpenML...")
        
        dataset = openml.datasets.get_dataset(dataset_id)
        
        df, _, _, _ = dataset.get_data()
        
        df.to_csv(csv_filename, index=False)  # Save locally

    # Display first few rows
    print("\nDataset Preview:")
    print(df.head())

    # Show column names for verification
    print("\nDataset Columns:")
    print(df.columns)

    # Check for missing values
    print("\nMissing Values Summary:")
    print(df.isnull().sum())

    # **Data Cleaning and Corrections**
    # Convert age from days to years
    df['age'] = df['age'] // 365  

    # Impute missing values using median strategy (Updated to avoid FutureWarning)
    df = df.fillna(df.median())

    # Convert categorical column for efficiency
    df = df.astype({"cardio": "int8"})  

    # **Feature Engineering**
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['blood_pressure_ratio'] = df['ap_hi'] / df['ap_lo']

    # Filter out unrealistic blood pressure values
    df = df[(df['ap_hi'] > 50) & (df['ap_hi'] < 250)]
    df = df[(df['ap_lo'] > 30) & (df['ap_lo'] < 180)]

    # Filter out unrealistic height and weight values
    df = df[(df['height'] > 100) & (df['height'] < 230)]
    df = df[(df['weight'] > 30) & (df['weight'] < 150)]

    # Display basic statistics of numerical features
    print("\nDataset Summary Statistics:")
    print(df.describe())


## Output
### Loading dataset from local CSV...

### Dataset Preview:
     age  gender  height  weight  ap_hi  ap_lo  cholesterol  gluc  smoke  \
0  18393       2     168    62.0  110.0   80.0            1     1      0   
1  20228       1     156    85.0  140.0   90.0            3     1      0   
2  18857       1     165    64.0  130.0   70.0            3     1      0   
3  17623       2     169    82.0  150.0  100.0            1     1      0   
4  17474       1     156    56.0  100.0   60.0            1     1      0   

   alco  active  cardio  
0     0       1       0  
1     0       1       1  
2     0       0       1  
3     0       1       1  
4     0       0       0  

### Dataset Columns:
Index(['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
       'gluc', 'smoke', 'alco', 'active', 'cardio'],
      dtype='object')

### Missing Values Summary:
age            0
gender         0
height         0
weight         0
ap_hi          0
ap_lo          0
cholesterol    0
gluc           0
smoke          0
alco           0
active         0
cardio         0
dtype: int64

### Dataset Summary Statistics:
                age        gender        height        weight         ap_hi  \
count  68664.000000  68664.000000  68664.000000  68664.000000  68664.000000   
mean      52.830260      1.348669    164.398142     74.040733    126.600504   
std        6.768118      0.476552      7.950253     14.044183     16.741088   
min       29.000000      1.000000    105.000000     31.000000     60.000000   
25%       48.000000      1.000000    159.000000     65.000000    120.000000   
50%       53.000000      1.000000    165.000000     72.000000    120.000000   
75%       58.000000      2.000000    170.000000     82.000000    140.000000   
max       64.000000      2.000000    207.000000    149.000000    240.000000   

              ap_lo   cholesterol          gluc         smoke          alco  \
count  68664.000000  68664.000000  68664.000000  68664.000000  68664.000000   
mean      81.365053      1.364674      1.225955      0.088037      0.053580   
std        9.611748      0.678980      0.571922      0.283351      0.225188   
min       40.000000      1.000000      1.000000      0.000000      0.000000   
25%       80.000000      1.000000      1.000000      0.000000      0.000000   
50%       80.000000      1.000000      1.000000      0.000000      0.000000   
75%       90.000000      2.000000      1.000000      0.000000      0.000000   
max      170.000000      3.000000      3.000000      1.000000      1.000000   

             active        cardio           bmi  blood_pressure_ratio  
count  68664.000000  68664.000000  68664.000000          68664.000000  
mean       0.803347      0.494743     27.439573              1.560855  
std        0.397471      0.499976      5.193942              0.152819  
min        0.000000      0.000000     10.726644              0.470588  
25%        1.000000      0.000000     23.875115              1.500000  
50%        1.000000      0.000000     26.346494              1.500000  
75%        1.000000      1.000000     30.116213              1.625000  
max        1.000000      1.000000     86.776860              4.000000



## **Exploratory Data Analysis (EDA)**
    sns.set_style("whitegrid")

    # Histogram for Age Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()



![Histogram Age Distribution](https://github.com/user-attachments/assets/aa7c5b41-a65e-4724-89e8-0fa821d9b76d)



 ## Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()



![Feature Correlation Matrix](https://github.com/user-attachments/assets/36373d78-5ed8-48ba-9f4a-78beb3d79d97)




## Boxplot for Blood Pressure vs. Cardiovascular Disease
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['cardio'], y=df['ap_hi'], hue=df['cardio'], palette="Set2", legend=False)
    plt.title("Systolic Blood Pressure vs. Cardiovascular Disease")
    plt.xlabel("Cardiovascular Disease (0 = No, 1 = Yes)")
    plt.ylabel("Systolic Blood Pressure (ap_hi)")
    plt.show()


![Box Plot Systolic bp vs  CVD](https://github.com/user-attachments/assets/3f4d8e6d-cce4-4eb8-8558-da26220f2099)


# **Machine Learning Models**
    X = df.drop(columns=['cardio'])
    y = df['cardio']

# Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{name} Accuracy: {accuracy:.4f}")


## Output
Logistic Regression Accuracy: 0.7254

Random Forest Accuracy: 0.7074

Gradient Boosting Accuracy: 0.7324

XGBoost Accuracy: 0.7298


from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif


from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

# **Hyperparameter Tuning using GridSearchCV**

param_grid = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "Gradient Boosting": {
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7]
    },
    "XGBoost": {
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7]
    }
}

best_models = {}
for name, params in param_grid.items():
    grid_search = GridSearchCV(models[name], params, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")


# **Feature Selection using SelectKBest**

feature_selector = SelectKBest(score_func=f_classif, k=5)
X_selected = feature_selector.fit_transform(X_train, y_train)

print("\nSelected top 5 features:", X.columns[feature_selector.get_support()])


## Output

Best parameters for Random Forest: {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 100}
Best parameters for Gradient Boosting: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
Best parameters for XGBoost: {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100}

Selected top 5 features: Index(['age', 'ap_hi', 'ap_lo', 'cholesterol', 'bmi'], dtype='object')

### Reference:

Feurer, Matthias. (2023). Cardiovascular-Disease-dataset. OpenML. Retrieved from OpenML. 
Matthias Feurer, Ph.D. candidate at the University of Freiburg, Germany. Working on automated machine learning. Creator of the python API for OpenML. Germany 2014-07-02 17:15:35
https://www.openml.org/search?type=data&status=active&id=45547









