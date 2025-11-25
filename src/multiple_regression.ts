/*
After the univariate example with a dataset I downloaded from just a cdc link I decided to go ahead and grab a reusable
dataset with an actual DOI etc.:

Akshay Chaudhary. (2025). Indian Kids Screentime 2025 [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/12412513

So what is a multiple regression? its just a univariate regression with multiple predictors which ends up requiring
multiple columns (all sketched out in my notebook and can be talked through confidently as well).

Briefly, the idea behind multiple regression is that we still have the one dependent variable but now we have multiple predictors.
i.e. more directions in space! 

The main advantage of multiple regression (as opposed to multi-variate regression which is my next topic)
is that with some fancy matrix calculus we can actually also derive the analytical solution for our coefficients.

This comes to us in the form of this equation:
β^​=(X⊤X)−1X⊤y

where 
β^ = all of our coefficients
X = the matrix of our values for a given observation (rows) and predictor (column)
y = our observed values (different from Y^ which are our predicted values)

Otherwise! this is more or less the same as univariate regression but, it does open the door to start pulling
in some lin-alg libraries into typescript akin to the way numpy leverages them in python.

*/
