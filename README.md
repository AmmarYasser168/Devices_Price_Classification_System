# Devices_Price_Classification_System

Overview:
This project aims to classify the price range of devices based on their specifications using machine learning models. The dataset used contains various features of devices, such as screen height and width, RAM, and battery power.

Files:
device_price_classification.ipynb
Loading Test Data and Cleaning: This notebook loads the test data from a CSV file and performs data cleaning tasks such as handling missing values and encoding categorical variables.
Applying Classification Models: Three classification models, namely K-Nearest Neighbors (KNN), Quadratic Discriminant Analysis (QDA), and Artificial Neural Network (ANN), are implemented and evaluated using the cleaned test data.
Choosing the Best Model: Based on the performance metrics, the ANN model is selected as the best-performing model for predicting device prices.
Saving Model Parameters: The parameters of the ANN model are saved to disk for future use.
Testing_model.ipynb
Loading Model Parameters for ANN: This notebook demonstrates how to load the saved ANN model parameters.
Testing the Model: The loaded ANN model is tested using testing data from the training dataset. Performance metrics such as accuracy, precision, recall, and F1 score are calculated.
Prediction for Random Samples: The ANN model is used to predict the price range for 10 random samples from the test dataset.
Post Request: An example of making a POST request to the Spring Boot application for predicting device prices.
Folder: test
Contains a Spring Boot application for deploying the machine learning model and exposing RESTful endpoints for making predictions.

Usage:
Open and run the device_price_classification.ipynb notebook to train and evaluate the classification models.
Save the model parameters using the provided code.
Use the Testing_model.ipynb notebook to load the saved model and perform testing.
The Spring Boot application in the test folder can be used to deploy the model and make predictions via RESTful API endpoints.
Dependencies:
Python libraries: pandas, numpy, scikit-learn, tensorflow, keras
Spring Boot for the Java application.

Contributors:
Ammar Yasser
