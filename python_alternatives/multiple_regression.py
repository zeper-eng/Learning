
'''
After the univariate example with a dataset I downloaded from the CDC I decided to go ahead and grab a reusable
dataset with an actual DOI from kaggle:

Akshay Chaudhary. (2025). Indian Kids Screentime 2025 [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/12412513

So what is a multiple regression? 

Briefly, the idea behind multiple regression is that we still have the one dependent variable but now we have multiple predictors.
i.e. more directions in space! In practice it becomes (conceptually) just a univariate regression with multiple predictors (and each get their own coefficient/slope)
where the actual equation looks something like:

Y=β0​+β1​X1​+β2​X2​+⋯+βp​Xp​+e

The main advantage of multiple regression (as opposed to multi-variate regression which is my next topic)
is that by expressing the equation as a linear algebra problem and with some fancy matrix calculus we can 
also derive the analytical solution for our coefficients,

This comes to us in the form of this equation:
β^​=(X⊤X)−1X⊤y

where 

-β^ = all of our coefficients

-X = the matrix of our values for a given observation (rows) and predictor (column)
    -> this is called the design matrix, which is akin to a feature matrix but with that column of 1's for our intercept
    -> it also has a leading column of 1's for β0​ to be constant

-y = our observed values (different from Y^ which are our predicted values)

Otherwise! this is more or less the same as univariate regression but, it does open the door to start pulling
in some lin-alg libraries into typescript akin to the way numpy leverages them in python.
`
'''

import numpy as np
import pandas as pd

Screentime_df = pd.read_csv('Test_Data/Indian_Kids_Screen_Time.csv')
test_names = ['Age','Educational_to_Recreational_Ratio']

predictors=np.array(Screentime_df[test_names])

