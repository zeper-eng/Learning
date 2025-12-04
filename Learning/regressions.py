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

    def __init__(self,predictor_vars=None,target_var=None,df=None):
        '''
        conditionals are mostly for being extra verbose when setting self.attributes since i do it for
        most methods.
        '''
        self.predictor_vars=predictor_vars if predictor_vars is not None else None
        self.target_var=target_var if target_var is not None else None
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

    def fit_regression(self,X=None,target_var=None,df=None):
        '''fit a regression to a dataset
        
        Parameters
        ----------
        X:np.ndarray,shape=(n_observations,n_predictors+1)
                Returns the design matrix (X) where the first column is filled with 1's such that our
                intercept remains consistent for now.
        
        target_var:str,
            The name of the column pertaining to the target variable
        
        df:pandas.DataFrame,shape=(n_observations,n_variables)
            A pandas dataframe representing your loaded in dataset of interest


        Returns
        -------
        (predictions,targets):tuple,
            Returns Y^ an Y for further use in the rest of the module

        

        Notes
        -----



        Examples
        --------
        '''
        target_var=target_var if target_var is not None else self.target_var
        df=df if df is not None else self.df
        X=X if X is not None else self.create_design_matrix()

        targets = np.array(df[target_var])

        X_T = np.transpose(X)

        b_hat = np.dot(np.linalg.inv(np.dot(X_T, X)), np.dot(X_T, targets))
        self.b_hat=b_hat #store coefficients for use later
        
        predictions=np.dot(X,b_hat) # automatically interprets it as 
        return predictions,targets

    def compute_SST(self, targets):
        '''
        Computes the **Total Sum of Squares (SST)** — the total variance in the observed 
        target variable before accounting for any predictors.

        Parameters
        ----------
        targets : arraylike, shape = (n_observations,)
            The observed values of the target variable.

        Returns
        -------
        SST : float
            The total variance in `targets`, calculated as the sum of squared deviations 
            from the mean.

        Notes
        -----
        SST represents the total variation present in the dataset. In the context of 
        regression, it can be decomposed into the explained (SSR) and unexplained (SSE) 
        components such that:

            SST = SSR + SSE
        '''
        y_bar = np.mean(targets)
        deviations = (targets - y_bar) ** 2
        SST = np.sum(deviations)
        return SST

    def compute_SSR(self, predictions, targets):
        '''
        Computes the **Regression Sum of Squares (SSR)** — the portion of the total variance 
        in the target variable explained by the fitted regression model.

        Parameters
        ----------
        predictions : arraylike, shape = (n_observations,)
            The model's predicted values of the target variable.

        targets : arraylike, shape = (n_observations,)
            The actual observed values of the target variable.

        Returns
        -------
        SSR : float
            The explained variance in `targets`, calculated as the sum of squared deviations 
            of the predictions from the mean of the observed data.

        Notes
        -----
        SSR measures how much variation the model explains compared to a simple model that 
        only predicts the mean. In combination with SSE and SST, it satisfies:

            SST = SSR + SSE
        '''
        y_bar = np.mean(targets)
        deviations = (predictions - y_bar) ** 2
        SSR = np.sum(deviations)
        return SSR

    def compute_SSE(self, predictions, targets):
        '''
        Computes the **Error Sum of Squares (SSE)** — the portion of the total variance in 
        the target variable that is not explained by the regression model.

        Parameters
        ----------
        predictions : arraylike, shape = (n_observations,)
            The model's predicted values of the target variable.

        targets : arraylike, shape = (n_observations,)
            The actual observed values of the target variable.

        Returns
        -------
        SSE : float
            The unexplained variance in `targets`, calculated as the sum of squared residuals 
            (differences between actual and predicted values).

        Notes
        -----
        SSE quantifies model error — the variance remaining after accounting for the model’s 
        predictions. Along with SSR, it forms part of the total variance decomposition:

            SST = SSR + SSE
        '''
        deviations = (targets - predictions) ** 2
        SSE = np.sum(deviations)
        return SSE

class logistic_regression():
    '''

    Background
    ----------

    So Logistic regression expands a regression to be able to be used for regression

    We assume a couple things as in the previous example of multiple regression
    -The data is independent of each other 
    -The data is i.i.d ^per above
    
    For this specific case we are also working with a BINARY logistic regression meaning that
    the possible outcomes are 0 or 1!

    Since we express it only in terms of 0 and 1 we can therefore assume it also takes the form of 
    a bernoulli distibution which is the simplest kind of distribution that models a binary outcome.

    For a Bernoulli distribution the Probability Mass Function (PMF) can be described as follows:
        P(Y=y)= P^y(1-P)^1-y where, y{0,1}

    Each case individually can be thought of as
        P(y-1)=p
        p(y-0)=1-p 
    ^and if you simplify the expression above we get theese answers without problem


    Moving on to calculating our logistic regression
    ------------------------------------------------

    It mainly consists of 3 moving parts

    1.We have the equation for our logistic regression:
     
    -y^​=σ(XB) where the sigmoid is defined as σ=1/1+e^-XB  
    
    This is in theory the same as the multiple(or univariate) regression but we are taking our datapoints and 
    "squishing" it into a sigmoid function therefore changing all of our possible values from 0-1.

    By squishing it into the values of 0 and 1 we achieve a probabilistic view which also allows us to 
    use the likelihood function L(P):
    L(B)=i=1∏n​[σ(xi⊤​B)]yi​[1−σ(xi⊤​B)]1−yi​

    Although really multiplying lots of small probabilities is not the best approach so we take the -loglikelihood
    L(B)=i=1∏n​[σ(xi⊤​B)]yi​[1−σ(xi⊤​B)]1−yi​

    And if we take the derivatives of that equation we get the current gradient
    ∇B​J(B)=X⊤(σ(XB)−y).

    ^theese equations are mostly so i can tie them to my notebook they are ofcourse not very well written in i have yet
    to mess around with latex some more.

    
    '''

    def __init__(self,predictor_vars=None,target_var=None,df=None):
        '''
        conditionals are mostly for being extra verbose when setting self.attributes since i do it for
        most methods.
        '''
        self.predictor_vars=predictor_vars if predictor_vars is not None else None
        self.target_var=target_var if target_var is not None else None
        self.df=df if df is not None else None
        self.b_hat=None

        #theese are so they are explicitly declared before assignment
        self.design_matrix=None
        self.targets=None


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

        self.design_matrix=X

        return X 
    
    def _initialize_target_vector(self,df=None,target_var=None):
        target_var=target_var if target_var is not None else self.target_var
        df=df if df is not None else self.df

        '''grab target vector from dataframe'''
        target_vector = np.array(df[self.target_var])
        self.targets=target_vector

        return target_vector

    def _initialize_coefficient_vector(self,):
        '''the idea here is that it will be dependent on what our input features are in scale'''

        return
    
    def _sigmoid(self,x):
        '''internal helper function for sigmoid'''
        return 1 / (1 + np.exp(-x))
    
    def negative_log_likelihood(self,design_matrix=None,coefficient_vector=None,target_vector=None):
        '''takes the negative log likelihood given an input coefficient vector and a design matrix
        
        Parameters
        ----------

        

        Returns
        -------

        
        
        Notes
        -----


        

        Examples
        --------
        
        '''

        design_matrix=design_matrix if design_matrix is not None else self.create_design_matrix()
        coefficient_vector=coefficient_vector if coefficient_vector is not None else self._initialize_coefficient_vector()
        target_vector=target_vector if target_vector is not None else self._initialize_target_vector()
        logits=np.dot(design_matrix,coefficient_vector)
        negative_log_likelihood=np.sum(np.logaddexp(0,logits)-(target_vector*logits))#logaddexp factors out the Max val making this numerically stable by pulling that big number out front then multipluing it by the rest of the result

        return negative_log_likelihood
    
    def sigmoid_squash(self,design_matrix=None,coefficient_vector=None):
        ''' A helper function to perform the operation σ(X​B)

        Parameters
        ----------
        design_matrix:np.ndarray,shape=(n_observations,n_predictors+1)
            Returns the design matrix (X) where the first column is filled with 1's such that our
            intercept remains consistent for now.
        
        coefficient_vector:np.ndarray,shape=(n_observations)
            A vector (or column vector) with the same number of coefficients as features in the design
            matrix for dot product computation


        Returns
        -------
        sigmoid_XTB:np.ndarray,shape=(n_observations)
            The results of squashing our linear model with the sigmoid function resulting in them being mapped
            to values between 0 and 1 or probabilities!


        Notes
        -----


        Examples
        --------
        '''

        design_matrix = design_matrix if design_matrix is not None else self.create_design_matrix()
        coefficient_vector=coefficient_vector if coefficient_vector is not None else self._initialize_coefficient_vector()
        
        #Theese are actually pretty neat prints for visualizing what were doing
        #print(design_matrix)
        XT_B=np.dot(design_matrix,coefficient_vector) #also called logits 
        #print(XT_B)
        sigmoid_XTB=self._sigmoid(XT_B)
        
        #print(sigmoid_XTB.shape)

        return sigmoid_XTB

    def calculate_gradient(self,design_matrix=None,target_vector=None,coefficient_vector=None):
        design_matrix=design_matrix if design_matrix is not None else self.create_design_matrix()
        coefficient_vector=coefficient_vector if coefficient_vector is not None else self._initialize_coefficient_vector()
        target_vector=target_vector if target_vector is not None else self._initialize_target_vector()
        
        logits = design_matrix @ coefficient_vector
        y_hat = self._sigmoid(logits)               # shape (n_samples,)
        nabla_J = design_matrix.T @ (y_hat - target_vector)  # shape (n_features,)
        return nabla_J
    
    def gradient_descent_step(self,design_matrix=None,learning_rate=None,max_iterations=None,target_vector=None,coefficient_vector=None):
        '''
        Parameters
        ----------
        

        

        Returns
        -------


        
        Notes
        -----


        
        Examples
        --------
        '''
        learning_rate = learning_rate if learning_rate is not None else .1
        design_matrix = design_matrix if design_matrix is not None else self.create_design_matrix()
        max_iterations = max_iterations if max_iterations is not None else 100
        target_vector=target_vector if target_vector is not None else self._initialize_target_vector()
        coefficient_vector=coefficient_vector if coefficient_vector is not None else self._initialize_coefficient_vector()
        
        for i in range(max_iterations):

            # Gradient of negative log-likelihood
            gradient = self.calculate_gradient(design_matrix,target_vector,coefficient_vector)

            # Parameter update
            coefficient_vector_new = coefficient_vector - learning_rate * gradient
            coefficient_vector = coefficient_vector_new
        
        final_nll = self.negative_log_likelihood(design_matrix,coefficient_vector,target_vector)

        # Save and return the final coefficients
        self.coefficient_vector = coefficient_vector
        return coefficient_vector
    




if __name__=='__main__':
    test='logistic'

    if test=='logistic':
        from sklearn.datasets import load_iris

        iris = load_iris()        # load the dataset
        y = iris.target 
        X = np.hstack((np.ones((y.shape[0],1)),iris.data))
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target  # add target column
        iris_df = iris_df[(iris_df['target'] == 0) | (iris_df['target'] == 1)]

        lr=logistic_regression(df=iris_df,predictor_vars=iris.feature_names,target_var='target')
        lr.create_design_matrix()
        lr._initialize_target_vector()
        print(lr.design_matrix.shape[0])
        filler_predictions=lr.sigmoid_squash(coefficient_vector=np.array([
            -1.059330756723609,   # intercept
            -2.07245518,          # sepal length
            -6.90686719,          # sepal width
            10.96995493,          # petal length
            5.64537232            # petal width
        ]))
        print(lr.negative_log_likelihood(coefficient_vector=np.array([
            -1.059330756723609,   # intercept
            -2.07245518,          # sepal length
            -6.90686719,          # sepal width
            10.96995493,          # petal length
            5.64537232            # petal width
        ])))
        os._exit(0)


        from simple_viz import plot_regression
        y = iris_df['target'].to_numpy()#just so theese tests work
        y_hat = (filler_predictions >= 0.5).astype(int)
        accuracy = (y_hat == y).mean()
        print("Accuracy:", accuracy)

        # 2. How extreme are the probabilities?
        print("min pred:", filler_predictions.min())
        print("max pred:", filler_predictions.max())
        plot_regression(lr.target_vector,filler_predictions)




    if test=='linear':
        import seaborn as sns
        
        mpg=sns.load_dataset("mpg")
        mpg = mpg.dropna(subset=['weight', 'horsepower', 'displacement', 'mpg'])

        test_names = ['weight','horsepower','displacement']
        mr_modeler = multiple_regression(df=mpg,predictor_vars=test_names,target_var='mpg')
        predictions, targets = mr_modeler.fit_regression()
        import matplotlib.pyplot as plt

        plt.scatter(targets, predictions)
        plt.xlabel("Miles per gallon")
        plt.ylabel("Predicted miles per gallon")
        plt.title("Multiple Regression: Actual vs Predicted")
        plt.plot([targets.min(), targets.max()],
                [targets.min(), targets.max()],
                'r--')  # ideal 1:1 line

        plt.show()



