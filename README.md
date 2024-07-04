# Loan Eligibility Prediction

This project demonstrates the process of predicting loan eligibility based on the applicant's information using a Support Vector Machine (SVM) model. The dataset used for this project is a loan dataset that includes various factors such as applicant's income, coapplicant's income, loan amount, credit history, and more.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Collection and Processing](#data-collection-and-processing)
- [Data Visualization](#data-visualization)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Predictive Model](#predictive-model)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/yourusername/loan-eligibility-prediction.git
cd loan-eligibility-prediction
pip install -r requirements.txt
```

## Usage

To run the project, follow these steps:

1. Load the dataset.
2. Perform data preprocessing and visualization.
3. Train the model.
4. Evaluate the model.
5. Use the predictive model to check loan eligibility.

The following code snippet demonstrates these steps:

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
loan_dataset = pd.read_csv('loan.csv')

# Data preprocessing
loan_dataset = loan_dataset.dropna()
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
loan_dataset.replace(to_replace='3+', value=4, inplace=True)
loan_dataset.replace({'Married': {'No': 0, 'Yes': 1},
                      'Gender': {'Male': 1, 'Female': 0},
                      'Self_Employed': {'No': 0, 'Yes': 1},
                      'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                      'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

# Split data into features and labels
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuracy)

# Predictive model function
def predict_loan_eligibility(input_data):
    input_data_as_dataframe = pd.DataFrame(input_data, index=[0])
    input_data_as_dataframe.replace({'Married': {'No': 0, 'Yes': 1},
                                     'Gender': {'Male': 1, 'Female': 0},
                                     'Self_Employed': {'No': 0, 'Yes': 1},
                                     'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                                     'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
    input_data_as_dataframe = input_data_as_dataframe.replace(to_replace='3+', value=4)
    prediction = classifier.predict(input_data_as_dataframe)
    return "Eligible for loan" if prediction[0] == 1 else "Not eligible for loan"

# Example input data for prediction
new_data = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

# Predict loan eligibility
result = predict_loan_eligibility(new_data)
print(result)
```

## Data Collection and Processing

The dataset is loaded into a Pandas DataFrame from a CSV file. The dataset contains 614 rows and 13 columns. Missing values are handled by dropping rows with missing data. The categorical values are converted to numerical values using label encoding.

## Data Visualization

Seaborn is used to create visualizations to understand the distribution of the data. Key visualizations include count plots for education and marital status versus loan status.

## Model Training

A Support Vector Machine (SVM) with a linear kernel is trained on the training data. The data is split into training and testing sets with a 90-10 ratio.

## Model Evaluation

The accuracy of the model is evaluated using the accuracy score metric. In this example, the model achieves an accuracy of 0.798 on the training data and 0.833 on the test data.

## Predictive Model

A predictive system is built to classify loan eligibility based on user input. The input data is preprocessed and passed through the trained model to predict whether the loan application is eligible or not.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to improve this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
