/*
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

*/

import pl, { DataFrame } from "nodejs-polars"; 
import * as tf from '@tensorflow/tfjs-node';
import { Matrix } from 'ml-matrix';

/*reading in new dataset*/
const Screentime_CSV = pl.readCSV(
  "Test_Data/Indian_Kids_Screen_Time.csv"
)

/*While polars was great and all tensorflow is probably the next best
thing it doesnt have linalg operations annoyingly in typescript which is 
probably why python is generally preffered but theres another package ml-matrix
which allows for easy use of thoose operations */

console.log(Screentime_CSV.columns)
Screentime_CSV.getColumns
let test_names = ['Age','Educational_to_Recreational_Ratio']


/**
 * process.exit()
 *
 * @param {pl.DataFrame} polars_df a dataframe for running multiple/univariate regression
 * @param {number[]} predictors A string of column headers for selecting predictors
 * @param {number[]} response A string for selecting your response variable
*/


function create_data_object_from_df(polars_df:pl.DataFrame<any>,
  predictors:string | string[],
  response:string | string[],
)

{
  let dataobject = {

    predictors : polars_df.select(predictors).rows(),
    response : polars_df.select(response).rows()

  }
  
  return(dataobject)

}  

let new_dataobject = create_data_object_from_df(Screentime_CSV,test_names,'Avg_Daily_Screen_Time_hr')
console.log(new_dataobject)


function create_design_matrix_from_predictors(predictors:number[][]){
  /*
  As a quick reminder we are aiming for something like this:
    -Y=β0​+β1​X1​+β2​X2​+⋯+βp​Xp​+e
  */

  const B0 = Array(predictors.length).fill(1);  
  const design_matrix = new Matrix([B0,predictors])


  return design_matrix
}


/*
calling them X and Y forward so that I can continue to stay in line with the actual notation

We know the closed form solution for B^ is as follows:
B^​=(X⊤X)−1X⊤y
*/ 

let X:Matrix = create_design_matrix_from_predictors(new_dataobject.predictors)
console.log(X)
process.exit()
let Y = new_dataobject.response
let X_T = tf.transpose(X) 



