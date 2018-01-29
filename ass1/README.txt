CMPE 452 Assignment 1: Perceptron and ANN

Goal: 
Design and train a perceptron using the data in train.txt to predict the three classes of 
Iris using simple feedback learning. Use test.txt to test the performance of your ANN in 
terms of precision and recall.

Deliverables:
- Submit the program code with associated libraries and data so that the TAs can execute it (otherwise you lose points).
- Once your ANN has been trained and tested, classify all data points in the given order using your ANN. 
Implement a function in your program to create an output text file to log the original and the predicted class values for all data points. 

Submit the text file.
- Add the following to the text file (manually or programmatically):
    1. The initial and final weight vectors.
    2. Total classification error or sum squared error, ∑ (y-d)^2 from classifying all the data points using your trained ANN.
    3. Total number of iterations used for training.
    4. The terminating criteria for the training phase