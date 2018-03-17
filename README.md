# Activity-Detection

A predictive machine learning model that uses android wear sensor data (accelerometer, gyroscope, etc.) and 
predicts the activity being performed (sitting, standing, walking & laying down).

ML_Model contains the machine learning models, training data and source code. It also has a scoring file that can be deployed to the azure in conjuction with the model to expose it as a web service.


Android application contains the source code for the android wear app that calls the ml model exposed as a web service and displays the results in real time.
