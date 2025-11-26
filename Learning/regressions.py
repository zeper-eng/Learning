'''
Initially begun as a mini-project to prepare for the transition into supervised learning,
biostatistics, and other quantitative modeling skills needed for work with the CERH at
Columbia’s Irving Medical Center, I decided to expand this project by implementing
various regression models from scratch to strengthen my linear algebra fundamentals
and provide concrete evidence that I understand the math.

This project was originally started in TypeScript, but after realizing how cumbersome
it is to perform matrix operations in JavaScript, I decided to use Python for the
analytical and mathematical components (via NumPy and pandas). The long-term plan is
to pair a TypeScript/JavaScript dashboard front end with a Python backend for the
core regression computations — since NumPy allows for rapid, transparent prototyping
once the math is broken down.

'''
import numpy as np
import pandas as pd
import os

class multiple_regression():
    '''
    After the univariate example (in typescript) with a dataset I downloaded from the CDC I decided to go ahead and grab a reusable
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

    Otherwise! this is more the same as univariate regression (and actually if you only provide 1 predictor it quite litteraly is a 
    univariate regression in implementation).
    `
    '''

    def __init__(self,predictor_vars=None,response_var=None,df=None):
        '''
        conditionals are mostly for being extra verbose when setting self.attributes since i do it for
        most methods.
        '''
        self.predictor_vars=predictor_vars if predictor_vars is not None else None
        self.response_var=response_var if response_var is not None else None
        self.df=df if df is not None else None
        self.b_hat=None


        return

    def create_design_matrix(self,predictor_vars=None,df=None):
        '''creates a design matrix from a given dataframe and array of predictor variables
        
        Parameters
        ----------
        predictor_vars:arraylike,shape=(n_variables,)
            An array of strings containing column headers you want to use as predictor variables
        
        df:pandas.DataFrame,shape=(n_observations,n_variables)
            A pandas dataframe representing your loaded in dataset of interest


        Returns
        -------
        X:np.ndarray,shape=(n_observations,n_predictors+1)
            Returns the design matrix (X) where the first column is filled with 1's such that our
            intercept remains consistent for now.

        

        Notes
        -----



        Examples
        --------
        '''
        predictor_vars=predictor_vars if predictor_vars is not None else self.predictor_vars
        df=df if df is not None else self.df

        predictors=np.array(df[predictor_vars])
        ones=np.full(shape=(predictors.shape[0],1),fill_value=1)
        X = np.concatenate([ones,predictors],axis=1) #our design matrix

        return X 

    def fit_regression(self,X=None,response_var=None,df=None):
        '''fit a regression to a dataset
        
        Parameters
        ----------
        X:np.ndarray,shape=(n_observations,n_predictors+1)
                Returns the design matrix (X) where the first column is filled with 1's such that our
                intercept remains consistent for now.
        
        response_var:str,
            The name of the column pertaining to the response variable
        
        df:pandas.DataFrame,shape=(n_observations,n_variables)
            A pandas dataframe representing your loaded in dataset of interest


        Returns
        -------
        (predictions,responses):tuple,
            Returns Y^ an Y for further use in the rest of the module

        

        Notes
        -----



        Examples
        --------
        '''
        response_var=response_var if response_var is not None else self.response_var
        df=df if df is not None else self.df
        X=X if X is not None else self.create_design_matrix()

        responses = np.array(df[response_var])

        X_T = np.transpose(X)

        b_hat = np.dot(np.linalg.inv(np.dot(X_T, X)), np.dot(X_T, responses))
        self.b_hat=b_hat #store coefficients for use later
        
        predictions=np.dot(X,b_hat) # automatically interprets it as 
        return predictions,responses

    def compute_SST(self, responses):
        '''
        Computes the **Total Sum of Squares (SST)** — the total variance in the observed 
        response variable before accounting for any predictors.

        Parameters
        ----------
        responses : arraylike, shape = (n_observations,)
            The observed values of the response variable.

        Returns
        -------
        SST : float
            The total variance in `responses`, calculated as the sum of squared deviations 
            from the mean.

        Notes
        -----
        SST represents the total variation present in the dataset. In the context of 
        regression, it can be decomposed into the explained (SSR) and unexplained (SSE) 
        components such that:

            SST = SSR + SSE
        '''
        y_bar = np.mean(responses)
        deviations = (responses - y_bar) ** 2
        SST = np.sum(deviations)
        return SST

    def compute_SSR(self, predictions, responses):
        '''
        Computes the **Regression Sum of Squares (SSR)** — the portion of the total variance 
        in the response variable explained by the fitted regression model.

        Parameters
        ----------
        predictions : arraylike, shape = (n_observations,)
            The model's predicted values of the response variable.

        responses : arraylike, shape = (n_observations,)
            The actual observed values of the response variable.

        Returns
        -------
        SSR : float
            The explained variance in `responses`, calculated as the sum of squared deviations 
            of the predictions from the mean of the observed data.

        Notes
        -----
        SSR measures how much variation the model explains compared to a simple model that 
        only predicts the mean. In combination with SSE and SST, it satisfies:

            SST = SSR + SSE
        '''
        y_bar = np.mean(responses)
        deviations = (predictions - y_bar) ** 2
        SSR = np.sum(deviations)
        return SSR

    def compute_SSE(self, predictions, responses):
        '''
        Computes the **Error Sum of Squares (SSE)** — the portion of the total variance in 
        the response variable that is not explained by the regression model.

        Parameters
        ----------
        predictions : arraylike, shape = (n_observations,)
            The model's predicted values of the response variable.

        responses : arraylike, shape = (n_observations,)
            The actual observed values of the response variable.

        Returns
        -------
        SSE : float
            The unexplained variance in `responses`, calculated as the sum of squared residuals 
            (differences between actual and predicted values).

        Notes
        -----
        SSE quantifies model error — the variance remaining after accounting for the model’s 
        predictions. Along with SSR, it forms part of the total variance decomposition:

            SST = SSR + SSE
        '''
        deviations = (responses - predictions) ** 2
        SSE = np.sum(deviations)
        return SSE







if __name__=='__main__':
    test_names = ['Age','Educational_to_Recreational_Ratio']
    mr_modeler = multiple_regression(df=pd.read_csv('Test_Data/Indian_Kids_Screen_Time.csv'),predictor_vars=test_names,response_var='Avg_Daily_Screen_Time_hr')
    predictions, responses = mr_modeler.fit_regression()
    import matplotlib.pyplot as plt

    plt.scatter(responses, predictions)
    plt.xlabel("Actual Avg Daily Screen Time (hr)")
    plt.ylabel("Predicted Avg Daily Screen Time (hr)")
    plt.title("Multiple Regression: Actual vs Predicted")
    plt.plot([responses.min(), responses.max()],
            [responses.min(), responses.max()],
            'r--')  # ideal 1:1 line

    plt.show()



