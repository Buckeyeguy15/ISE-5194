# Assignment 3, Problem 1
# Purpose:  Build a linear regression model that predicts output
#           based on a single input feature.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#
# Function:   estimate_coeffs
# Purpose:    To calculate the intercept and slope of the 
#             regression line based on the training data
# Input:      
#     x:      input training data
#     y:      training output
# Output:
#     ( w0, w1): intercept and slope of the linear regression model
#
def estimate_coeffs(x, y):

    ################################################
    # To Do: Fill out the code for this function.
    ################################################
    
    return ( w0, w1 )
    

#
# Function:   plot_regression_line
# Purpose:    To plot the x and y data along with the 
#             trained linear model.
# Input:      
#     x:      input training data
#     y:      training output
#     w:      [0] intercept and [1] slope of trained model
#
def plot_regression_line(x, y, w):
    # plot data points
    plt.scatter(x, y, color = "gray", alpha=0.5, marker = "o", s = 30)
    
    # calculate prediction vector
    y_pred = w[0] + w[1]*x
    
    # plot the regression line
    plt.plot(x, y_pred, color = "r" )
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()
    

def main():
    # Import data from CSV file.
    y,x = np.loadtxt('./homework3_linearRegression_data.csv', delimiter=',', unpack=True)
    
    ################################################
    # To Do: Split data into training and testing sets.
    ################################################

    
    ################################################
    # To Do: Call the estimate_coeffs function to calculate
    # the weights for the 'best fit' line.
    ################################################


    
    print ( "Estimated Coefficients:" )
    print ( "  w0: " + np.str(w[0]) + "   w1: " + np.str(w[1]) )
    
    # Plot the regression line with data
    plot_regression_line ( X_test, y_test, w )
    
    
if __name__ == "__main__": 
    main() 