'''
Initially begun as a mini-project to prepare for the transition into supervised learning,
biostatistics, and other quantitative modeling skills needed for work with the CERH at
Columbia’s Irving Medical Center, I decided to expand this project by implementing
various  models from scratch to strengthen my linear algebra fundamentals
and provide concrete evidence that I understand the math.

This project was originally started in TypeScript, but after realizing how cumbersome
it is to perform matrix operations in JavaScript, I decided to use Python for the
analytical and mathematical components (via NumPy and pandas). The long-term plan is
to pair a TypeScript/JavaScript dashboard front end with a Python backend for the
core regression computations — since NumPy allows for rapid, transparent prototyping
once the math is broken down.

Also of note is that not all implementations are the most robust and well edge case guarded ones yet,
this will continuously be updated with more thorough code and robust examples. At the moment various things
could already be improved like a non-normal equation implementation of linear regression and other optimization algorithm
options for logistic regression.

'''

import numpy as np
import pandas as pd
import os


class Multiple_Regression():
    '''
    This module specifically uses the naive equation implementation for a closed form analytical solution of regression.
    
    So there are a few algebraic assumptions being made here namely being that:
        -Full column rank, i.e. no perfect colinearity (X^T*X is invertible) 
        -usually n>=p so full rank is possible
        -of course no NaNs/inf, nothing is numerically insane no massive or tiny values 
    
    This is of course also in addition to the typical statistical assumptions for why regression works but in practice
    these are probably handled during experimental design, etc.

    There are of course more numerically stable and robust ways to perform this operation but, at the moment this is the one
    I have chosen to go with and will expand from there into other methods.


    '''

    def __init__(self,predictor_vars=None,target_var=None,df=None):
        '''

        Parameters
        ----------

        df=Pandas.DataFrame,shape=(n_variables,)
            A Pandas DataFrame that contains all of the data that you would like to use in this example. For this
            project, some pretty strict assumptions about the format of the DataFrame are made as its mostly for 
            learning purposes. Namely that it ~is~ a pandas dataframe containing only numerical data (float preferred)
            and that each column contains a unique str identifier that can then be used to pull the information you
            need.
        
        predictor_vars=listlike,shape=(n_predictors)
            A list of column headers pertaining to each of the predictors you would like to run your regression with
        
        target_var=str,
            A string denoting the column header for the target variable you would like to use in your regression



        Returns
        -------



        Notes
        -----
        Conditionals are mostly for being extra verbose when setting self.attributes since i do it for
        most methods.



        Examples
        --------



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
        
        df=Pandas.DataFrame,shape=(n_variables,)
            A Pandas DataFrame that contains all of the data that you would like to use in this example. For this
            project, some pretty strict assumptions about the format of the DataFrame are made as its mostly for 
            learning purposes. Namely that it ~is~ a pandas dataframe containing only numerical data (float preferred)
            and that each column contains a unique str identifier that can then be used to pull the information you
            need.


        Returns
        -------
        X:np.ndarray,shape=(n_observations,n_predictors+1)
            Returns the design matrix (X) where the first column is filled with 1's such that our
            intercept remains consistent for now.


        Notes
        -----
        As mentioned in the description of the module there are various assumptions about our datas shape 
        specifically that need to be met in order for this implementation to function correctly.



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
        
        df=Pandas.DataFrame,shape=(n_variables,)
            A Pandas DataFrame that contains all of the data that you would like to use in this example. For this
            project, some pretty strict assumptions about the format of the DataFrame are made as its mostly for 
            learning purposes. Namely that it ~is~ a pandas dataframe containing only numerical data (float preferred)
            and that each column contains a unique str identifier that can then be used to pull the information you
            need.


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

        #technically i think
            #b_hat = np.linalg.solve(X_T @ X, X_T @ targets)
        #Is more stable? but I like this implementation right now because its more conceptually visible and also
        #I wont be running this on large datasets for quite a while but when i come back to it expect this comment to be flipped
        #and mention the old method being used.
        
        b_hat = np.dot(np.linalg.inv(np.dot(X_T, X)), np.dot(X_T, targets))
        self.b_hat=b_hat #store coefficients for use later
        
        predictions=np.dot(X,b_hat) 

        return predictions,targets

    def compute_SST(self, targets):
        '''Computes the Total Sum of Squares (SST)

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
        '''Computes the Regression Sum of Squares (SSR) 

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
        '''Computes the Error Sum of Squares (SSE)

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

class Logistic_Regression():
    '''

    Most of the core of this implementation can be found in the new README for the github repo but, briefly
    this is an implementation of a binary logistic regression.

    This allows for use of use of the following implementation given we assume a bernoulli distribution and,
    it mainly consists of 3 moving parts:

    -squashing our linear function into probability space with a sigmoid function that maps our data between 0-1
    -optimizing the negative log likelihood as a function of our coefficients. This is our "loss function"
    -Using gradient descent to progressively update the coefficient vectors with at a user defined learning rate alpha.

    Other assumptions in terms of the shape of your data include
    -all categorical variables have been coded to be represented numerically
    -pandas dataframe with named columns
    
    '''

    def __init__(self,predictor_vars=None,target_var=None,df=None):
        '''
        df=Pandas.DataFrame,shape=(n_variables,)
            A Pandas DataFrame that contains all of the data that you would like to use in this example. For this
            project, some pretty strict assumptions about the format of the DataFrame are made as its mostly for 
            learning purposes. Namely that it ~is~ a pandas dataframe containing only numerical data (float preferred)
            and that each column contains a unique str identifier that can then be used to pull the information you
            need.
        
        predictor_vars=listlike,shape=(n_predictors)
            A list of column headers pertaining to each of the predictors you would like to run your regression with
        
        target_var=str,
            A string denoting the column header for the target variable you would like to use in your regression

        '''
        self.predictor_vars=predictor_vars if predictor_vars is not None else None
        self.target_var=target_var if target_var is not None else None
        self.df=df if df is not None else None
        self.b_hat=None

        #these are so they are explicitly declared before assignment
        self.design_matrix=None
        self.targets=None
        self.predictions=None
        


        return
    
    def create_design_matrix(self,predictor_vars=None,df=None):
        '''creates a design matrix from a given dataframe and array of predictor variables
        
        Parameters
        ----------
        predictor_vars:arraylike,shape=(n_variables,)
            An array of strings containing column headers you want to use as predictor variables
        
        df:pandas.DataFrame,shape=(n_observations,n_variables)
            A pandas dataframe representing your loaded in dataset of interest.


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
        '''grab target vector from dataframe'''
        target_var=target_var if target_var is not None else self.target_var
        df=df if df is not None else self.df

        target_vector = np.array(df[self.target_var])
        self.targets=target_vector

        return target_vector
    


    def _initialize_coefficient_vector(self,design_matrix=None):
        '''Initializes a coefficient vector with all 0's as a starting point from which to calculate gradient descent steps'''
        design_matrix=design_matrix if design_matrix is not None else self.create_design_matrix()
        B=np.zeros(shape=design_matrix.shape[1])

        return B
    
    def _sigmoid(self,x):
        '''internal helper function for sigmoid'''
        return 1 / (1 + np.exp(-x))
    
    def _initialize_attributes(self):
        self.create_design_matrix()
        self._initialize_target_vector()
        self._initialize_coefficient_vector()

        return
    
    def negative_log_likelihood(self,design_matrix=None,coefficient_vector=None,target_vector=None):
        '''takes the negative log likelihood given an input coefficient vector and a design matrix
        
        Parameters
        ----------
        design_matrix:np.ndarray,shape=(n_observations,n_predictors+1)
            Returns the design matrix (X) where the first column is filled with 1's such that our
            intercept remains consistent for now.
        
        coefficient_vector=arraylike,shape=(n_coefficients,)
            This is our coefficient vector B where each entry 0-i-1 is a coefficient going in the order
            B0,B1,B2,B3 etc. It is important that these specific shapes are followed strictly as the rest
            of the linalg operations depend on it. For example our intercept is the first column in the design 
            matrix not the last, and contextually this may differ from implementation to implementation especially
            depending on what linalg operations are being done under the hood.
        
        target_vector:np.ndarray,shape=(n_samples)
            The column from the pandas DataFrame that contains your target variable (Y) in the form of a Numpy Array.
            Expected dimensions are of course equivalent to the rest of the samples i.               
        

        Returns
        -------

        negative_log_likelihood:float,
            The current negative log likelihood as a function of the current set of coefficients (B/the coefficient vector).
            This is our current loss function.

        
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
        
        coefficient_vector:np.ndarray,shape=(n_features)
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
        
        #these are actually pretty neat prints for visualizing what were doing
        #print(design_matrix)
        XT_B=np.dot(design_matrix,coefficient_vector) #also called logits 
        #print(XT_B)
        sigmoid_XTB=self._sigmoid(XT_B)
        
        #print(sigmoid_XTB.shape)

        return sigmoid_XTB

    def calculate_gradient(self,design_matrix=None,target_vector=None,coefficient_vector=None):
        '''
        Parameters
        ----------
        design_matrix:np.ndarray,shape=(n_observations,n_predictors+1)
            Returns the design matrix (X) where the first column is filled with 1's such that our
            intercept remains consistent for now.

        coefficient_vector:np.ndarray,shape=(n_features)
            A vector (or column vector) with the same number of coefficients as features in the design
            matrix for dot product computation
        
        target_vector:np.ndarray,shape=(n_samples)
            The column from the pandas DataFrame that contains your target variable (Y) in the form of a Numpy Array.
            Expected dimensions are of course equivalent to the rest of the samples i.
        
        
        Returns
        -------
        nabla_j:np.ndarray
            The value for the gradient with the current set of coefficients being used. Can be calculated as part of our
            gradient descent steps or can also just be calulated for some implementation of your coefficients.

            
        Notes
        -----


        
        Examples
        --------



        '''
        design_matrix=design_matrix if design_matrix is not None else self.create_design_matrix()
        coefficient_vector=coefficient_vector if coefficient_vector is not None else self._initialize_coefficient_vector()
        target_vector=target_vector if target_vector is not None else self._initialize_target_vector()

        logits = design_matrix @ coefficient_vector
        y_hat = self._sigmoid(logits)               # shape (n_samples,)
        nabla_J = design_matrix.T @ (y_hat - target_vector)  # shape (n_features,)
      
        return nabla_J
    
    def minimize_coefficients(self,design_matrix=None,learning_rate=None,max_iterations=None,target_vector=None,coefficient_vector=None):
        '''
        Parameters
        ----------
        design_matrix:np.ndarray,shape=(n_observations,n_predictors+1)
            Returns the design matrix (X) where the first column is filled with 1's such that our
            intercept remains consistent for now.

        coefficient_vector:np.ndarray,shape=(n_features)
            A vector (or column vector) with the same number of coefficients as features in the design
            matrix for dot product computation
        
        target_vector:np.ndarray,shape=(n_samples)
            The column from the pandas DataFrame that contains your target variable (Y) in the form of a Numpy Array.
            Expected dimensions are of course equivalent to the rest of the samples i.
        
        max_iterations:float,default=100
            The stopping condition for our gradient descent which will run (T) times where T is equivalent
            to the max_iterations.
        
        learning_rate:float,default=.1
            This is alpha or our default learning rate at which our gradient descent steps will be calculated.
            Generally it is important that alpha is within reasonable size so as to satisfy the assumptions made for
            h in the derivative definition for h. 
        

        

        Returns
        -------
        (coefficient_vector,final_nll):tuple,shape=((n_observations),(1))
            A tuple containing the final optimized coefficient vector and the final negative log likelihood
            after calculating T steps of the gradient descent optimization


        
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
        return coefficient_vector,final_nll
    
    def fit_logistic_regression(self,design_matrix=None,learning_rate=None,max_iterations=None,target_vector=None,coefficient_vector=None):
        '''
        Parameters
        ----------
        design_matrix:np.ndarray,shape=(n_observations,n_predictors+1)
            Returns the design matrix (X) where the first column is filled with 1's such that our
            intercept remains consistent for now.

        coefficient_vector:np.ndarray,shape=(n_features)
            A vector (or column vector) with the same number of coefficients as features in the design
            matrix for dot product computation
        
        target_vector:np.ndarray,shape=(n_samples)
            The column from the pandas DataFrame that contains your target variable (Y) in the form of a Numpy Array.
            Expected dimensions are of course equivalent to the rest of the samples i.
        
        max_iterations:float,default=100
            The stopping condition for our gradient descent which will run (T) times where T is equivalent
            to the max_iterations.
        
        learning_rate:float,default=.1
            This is alpha or our default learning rate at which our gradient descent steps will be calculated.
            Generally it is important that alpha is within reasonable size so as to satisfy the assumptions made for
            h in the derivative definition for h. 
        

        

        Returns
        -------
        y_hat_predictions:np.Ndarray,shape=((n_observations),(1))
            The results of the logistic regression function where we squash the results of a linear model as
            follows:sigma(XB), where in this particular case (as opposed to the sigmoid squash method ) is that
            we fit coefficients derived from the gradient descent optimization.
             

        
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

        coefficient_vector,final_nll=self.minimize_coefficients(design_matrix,learning_rate,max_iterations,target_vector,coefficient_vector)
        
        print(f"The final negative log likelihood for this function was: {final_nll}")
        
        y_hat_predictions = self.sigmoid_squash(design_matrix,coefficient_vector)
        self.predictions=y_hat_predictions

        return y_hat_predictions
    
    def compute_SST(self, targets=None):
        '''Computes the Total Sum of Squares (SST)

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
        targets = targets if targets is not None else self.targets

        y_bar = np.mean(targets)
        deviations = (targets - y_bar) ** 2
        SST = np.sum(deviations)
        return SST

    def compute_SSR(self, predictions=None, targets=None):
        '''Computes the Regression Sum of Squares (SSR) 

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
        targets=targets if targets is not None else self.targets
        predictions= predictions if predictions is not None else self.predictions
                
        y_bar = np.mean(targets)
        deviations = (predictions - y_bar) ** 2
        SSR = np.sum(deviations)
        return SSR

    def compute_SSE(self, predictions, targets):
        '''Computes the Error Sum of Squares (SSE)

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
        
        predictions=predictions if predictions is not None else self.predictions
        targets=targets if targets is not None else self.targets

        deviations = (targets - predictions) ** 2
        SSE = np.sum(deviations)
        return SSE
    




if __name__=='__main__':
    test='linear'

    if test=='logistic':
        from sklearn.datasets import load_iris

        iris = load_iris()        # load the dataset
        y = iris.target 
        X = np.hstack((np.ones((y.shape[0],1)),iris.data))
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target  # add target column
        iris_df = iris_df[(iris_df['target'] == 0) | (iris_df['target'] == 1)]

        lr=logistic_regression(df=iris_df,predictor_vars=iris.feature_names,target_var='target')
        lr._initialize_attributes()
        y_hat_predictions=lr.fit_logistic_regression()

        
        


        from simple_viz import plot_regression
        y = iris_df['target'].to_numpy()#just so these tests work
        y_hat = (y_hat_predictions >= 0.5).astype(int)
        accuracy = (y_hat == y).mean()
        print("Accuracy:", accuracy)

        # 2. How extreme are the probabilities?
        print("min pred:", y_hat_predictions.min())
        print("max pred:", y_hat_predictions.max())
        plot_regression(lr.targets,y_hat_predictions)




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



