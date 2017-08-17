This project is an example of using the bootstrapping technique to select an
optimal regressor.  For these examples, we use simple linear regresssion
firstly on randomized generated data, and then using the UCIrvine Computer
Hardware data set.  The project consists of several adhoc experiments within
several files.  The purpose of each is listed below:

  part1.py:
    This experiment uses Linear Least Squares (LLS) to examine the effect on
    Confidence Intervals (CI) by altering the number of models used to
    bootstrap, and then altering the number of random examples generated in
    the experiment.
    
  part3.py:
    This experiment uses Linear Least Squares (LLS) and applies it within a
    bootstrapping technique of 1000 LLS models built to predict the estimated
    performance of computer hardware configurations.
