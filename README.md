# Learning (Hand-rolled Machine learning and Statistical Analyses)

Initially begun as a small project to learn some typescript and work out implementing some simple statistical analyses, it actually also functions as a nice proof of product for the derivations and
theorems I have been reading and doing in a notebook.

Thus, I have decided to document this a little more thoroughly and use it as proof of product for my ability to do simple numerical computing.

# Implementations so far
Outline below are the different algorithms I have implemented so far. Mostly this is gonna be regressions and GLM's, bread and butter, and work out from there. More than likely add in simple non-parametric (unsupervised learning) things like K-means etc before moving on to more complex methods.

##  Multiple Regression (univariate when 1 predictor is provided)
I have implemented a linear regression as a multiple regression which also functions as a simple(univariate) regression when 1 predictor is provided. In linear regression we model a dependent variable (our target/response variable) as a function of a (or multiple) predictor variable(s). 

(1): ![y = XB](https://latex.codecogs.com/svg.latex?y%20=%20XB)

or

(2): ![y = b0+B1X](https://latex.codecogs.com/svg.latex?y%20=%20b_0+b_1x) (for a univariate regression) 


This implementation uses the linalg definition for ease of implementation, where X is our design matrix. A design matrix is a matrix where each row is one of our observations i and each column is one of our features k. Importantly the first column of the design matrix is all 1's so B0 our intercept is constant across observations (also the reason its different from a feature matrix).

B is our coefficient vector which contains coefficients for each one of our predictors. Here
the optimal predictors can be determined with the closed form solution

(3): ![b-hat = (X^T X)^-1 X^T Y](https://latex.codecogs.com/svg.latex?\hat{b}=(X^{T}X)^{-1}X^{T}Y)

##  Logistic Regression (Binary)

Binary Logistic regression models the probability of an event happening as a linear combination of its predictors. Here the "event" is the probability of the response/target variable having the label it does, i.e. also the reason why we use this method for classification.

The equation for a logistic regression is as follows:

(4): ![y=sigmoid(X^TB)](https://latex.codecogs.com/svg.latex?\{Y}=\sigma(X^{T}B))

We would ideally want to maximize the probability of our particular outcomes occuring. Because we are working with binary outcomes, we can assume a bernoulli distribution and therefore define our loss function as a maximum likelihood estimation where L(B): 

(5): ![maximum likelihood](https://latex.codecogs.com/svg.latex?L(B)=\prod_{i=1}^{n}[\sigma(x_i^{T}B)]^{y_i}(1-\sigma(x_i^{T}B))^{1-y_i})

However in practice, multiplying probabilities like that leads to very small numbers and numerical overflow so we typically take the negative log odds.

(6): 

![negative log loss](https://latex.codecogs.com/svg.latex?J(B)=-\sum_{i=1}^{n}\left[y_i\ln\sigma(x_i^{T}B)+(1-y_i)\ln(1-\sigma(x_i^{T}B))\right])

or

![negative log-likelihood sum-exp form](https://latex.codecogs.com/svg.latex?\ell(B)=\sum_i\left[\log(1+e^{x_i^{T}B})-y_i\,x_i^{T}B\right]) (this form is the one I decided to implement)


Because of the sigmoid function we do not automatically recieve an easy to use closed-form solution
so we must use numerical methods to compute our ideal solution it.

Fortunately the logistic regression function is typically convex and has 1 pretty easy to find global minima so I implemented using a simple T steps gradient descent(steepest descent) where our gradient 
is defined as:

![gradient matrix form](https://latex.codecogs.com/svg.latex?\nabla_B\ell(B)=X^{T}(\sigma(XB)-Y))

and our update rule is:

![update rule](https://latex.codecogs.com/svg.latex?B^{(t+1)}=B^{(t)}-\alpha\nabla_B\ell(B^{(t)}))

# Future directions

## Front end

Some kind of typescript front end GUI/chart displayer mostly to try to get some practice using javascript/typescript although its somewhat included in here and perhaps mysql stuff although its hard to really implement SQL without using a proper database connection via stuff like microsoft azure.

## Algorithms
I plan on continuing implementing mostly GLM's for the purpose of growing my clinical research relevant skillset but, will also try out things like K-means and SVD PCA.

In the near future:

- poisson regression
- Kmeans 
- PCA

