/*
A crude but, working implementation of a simple linear regression using a fake dataset grabbed online.

Preferred for this specific implementation is the analytical definitions of the regression coefficient and the intercept 
B1 = Σ[(X_i - X̄)(Y_i - Ȳ)] / Σ[(X_i - X̄)²]
B0 = Ȳ - B1 * X̄
*/

import pl from "nodejs-polars"; //appearently this is like pandas
import {xy} from "./simple_utilities.ts";
import fs from "node:fs";
/////////////////////////////////////////////////////////
//Some quick reminders on just the super basic
/////////////////////////////////////////////////////////
const a = [];                 // empty
const años = [1915, 1916];   // numbers
const names = ["Infant", "Neonatal"]; // strings
const rows = [{Year:1915, rate:44.4}, {Year:1916, rate:44.1}]; // array of objects

////////////////////////////////////////////////////////////////////
//Tensorflow is NOT as ergonomic as it has been in python
//however, probably better for scaling things later on
//so considering that lets go ahead and start with polars (pandas); 
//i.e. dataframe type stuff
////////////////////////////////////////////////////////////////////

// for the record this is the original name incase anyone is ever wondering:NCHS_-_Infant_and_neonatal_mortality_rates__United_States__1915-2013.csv
const polars_df = pl.readCSV(
  "Test_Data/test_data.csv"
)

// some quick cleaning with previous data management files
// mostly just dataframe ops; new syntax nothing new in practice really
const tidy = polars_df
  .rename({ "Mortality Rate": "rate" })
  .select(pl.col("Type"), pl.col("Year"), pl.col("rate")) 
  .withColumns(
    pl.col("Year").cast(pl.Int32),
    pl.col("rate").cast(pl.Float32),
  );

  
//handy I saved theese
const data_object_Neonatal = xy("Neonatal",tidy)
const data_object_Infant = xy("Infant",tidy)


//console.log("the Neonatal object:\n",data_object_Neonatal);
//console.log("the Infant object:\n",data_object_Infant);


/*
ALRIGHT

So by definiton we can think of the data for a linear regression as meeting 4 basic assumptionss
(We will ignore theese for now in terms of my dataset and then add in normalizations when appropriate etc)
-homogenous variance across independent variable entries
-normality (as do most things)
-independence of observations (i.i.d)
-that the relationship between the independent and dependent variables is linear

With theese assumptions met the simplest linear regression formula is as follows
Y = B0 + B1X + e 

where

-y is the predicted value of the dependent variable (y) for any given value of the independent variable (x).
-B0 is the intercept, the predicted value of y when the x is 0.
-B1 is the regression coefficient – how much we expect y to change as x increases.
-x is the independent variable ( the variable we expect is influencing y).
-e is the error of the estimate, or how much variation there is in our estimate of the regression coefficient.

Linear regression finds the line of best through our data by searching for the regression coefficient (B1) that minimizes the total error (e) of the model.
Where we are trying to minimize the sum of squares equation:

SSE=i∑​(Yi​−(B0​+B1​Xi​))2

In practice we know there are analytical derivations of the optimal solution which can be expressed as follows and
so simplifies the process further:

B1 = Σ[(X_i - X̄)(Y_i - Ȳ)] / Σ[(X_i - X̄)²]
B0 = Ȳ - B1 * X̄

I will add in some functions for estimating a small range of values anyways brute-force-esque just for proof of concept 
but, the analytical approach makes more sense to me in terms of then integrating some SQL and stuff.

*/


import {mean} from "./simple_utilities.ts";

import type { XYSeries } from "./simple_utilities.ts";


//I will define theese in here first and then throw them in the simple utilities stockpile
function calculate_B1(dataobject:XYSeries){
    

    // we only really need to define theese once honestly
    const xBar:number = mean(dataobject.x)//X̄
    const yBar:number = mean(dataobject.y)//Ȳ


    //setting up our sums before iterating
    let running_sum_one:number = 0 
    let running_sum_two:number = 0

    for (let i = 0; i < dataobject.x.length; i++){//iterate for now while i figure out vectorized operations
        let x_i:number =dataobject.x[i]//theese throw errors because in theory the index could return none, throwing a ! at the end would silence
        let y_i:number =dataobject.y[i]//theese throw errors because in theory the index could return none, throwing a ! at the end would silence

        let running_numerator = (x_i-xBar)  * (y_i-yBar) //Σ[(X_i - X̄)(Y_i - Ȳ)]
        let running_denominator = (x_i-xBar) * (x_i-xBar) //Σ[(X_i - X̄)²], all i did was expand 

        running_sum_one = running_sum_one+running_numerator
        running_sum_two = running_sum_two+running_denominator
        //running_sum_one = running_sum_one+product_x_i_y_i

    }
    const B1:number = running_sum_one/running_sum_two
    
    return B1

}

function calculate_B0(dataobject:XYSeries,B1:number){
    const xBar:number = mean(dataobject.x)//X̄
    const yBar:number = mean(dataobject.y)//Ȳ
    
    const B0:number = yBar - B1 * xBar //B0 = Ȳ - B1 * X̄

    return B0
}


function calculate_SSE(errors:number[]){

  let running_sum:number=0
  for(let i=0; i<errors.length; i++){
    
    let e:number=errors[i]
    let e_squared=e*e
    running_sum+=e_squared
  }

  return running_sum;

}

function calculate_SST(observations:number[]){

  let running_sum:number=0
  let y_mean:number = mean(observations)

  for(let i=0; i<observations.length; i++){
    let y_i:number=observations[i]
    let squared_deviation:number = (y_i-y_mean)**2 
    running_sum+=squared_deviation
  }

  return running_sum;

}

function calculate_SSR(predictions:number[],observations:number[]){

  let running_sum:number=0
  let y_mean:number = mean(observations)

  for(let i=0; i<predictions.length; i++){
    let y_hat:number=predictions[i]
    let squared_deviation:number = (y_hat-y_mean)**2 
    running_sum+=squared_deviation
  }

  return running_sum;

}

// I generated theese docs with chatgpt just so I can go ahead and see what the numpydoc style stuff looks like in typescript.
function fit_regression(dataobject:XYSeries){
   
    let B1:number = calculate_B1(dataobject)
    let B0:number = calculate_B0(dataobject,B1)

    let predicted_points:number[] = []
    let errors:number[] = []

    for (let i = 0; i < dataobject.y.length; i++){//iterate for now while i figure out vectorized operations
        
        let y:number = B0+(dataobject.x[i]*B1)//so calculate y
        predicted_points.push(y)//add to our actual predicted points 
        errors.push(dataobject.y[i]-y)//calculate error and add to the list(residuals
    }

    let SSR:number = calculate_SSR(predicted_points,dataobject.y)
    let SST:number = calculate_SST(dataobject.y)
    let SSE:number = calculate_SSE(errors)

    let r_squared:number=SSR/SST


    return {predictions:predicted_points,errors:errors,SSR:SSR,SST:SST,SSE:SSE,r_squared}
}

const regression_Neonatal = fit_regression(data_object_Neonatal)
const regression_Infant = fit_regression(data_object_Infant)

console.log("regression_Neonatal: ",regression_Neonatal)
console.log("regression_Infant: ",regression_Infant)

const scatterNeonatal = {
  x: data_object_Neonatal.x,
  y: data_object_Neonatal.y,
  mode: "markers",
  type: "scatter",
  name: "Neonatal",
};

const scatterInfant = {
  x: data_object_Infant.x,
  y: data_object_Infant.y,
  mode: "markers",
  type: "scatter",
  name: "Infant",
};


const lineNeonatal = {
  x: data_object_Neonatal.x,
  y: regression_Neonatal.predictions,
  mode: "lines",
  type: "scatter",
  name: "Neonatal fit",
};

const lineInfant = {
  x: data_object_Infant.x,
  y: regression_Infant.predictions,
  mode: "lines",
  type: "scatter",
  name: "Infant fit",
};

// Combine traces for main plot
const data = [scatterNeonatal, scatterInfant, lineNeonatal, lineInfant];
const annotations = [
  {
    text: `Neonatal: SSE=${regression_Neonatal.SSE.toFixed(2)}, SSR=${regression_Neonatal.SSR.toFixed(2)}, SST=${regression_Neonatal.SST.toFixed(2)}`,
    xref: "paper",
    yref: "paper",
    x: 0.02,
    y: 1.08,
    showarrow: false,
    font: { size: 12, color: "blue" },
  },
  {
    text: `Infant: SSE=${regression_Infant.SSE.toFixed(2)}, SSR=${regression_Infant.SSR.toFixed(2)}, SST=${regression_Infant.SST.toFixed(2)}`,
    xref: "paper",
    yref: "paper",
    x: 0.02,
    y: 1.02,
    showarrow: false,
    font: { size: 12, color: "orange" },
  },
];

// Layout with annotation overlay
const layout = {
  xaxis: { title: "Year" },
  yaxis: { title: "Mortality rate (per 1k)" },
  annotations: annotations,
};

// Generate the HTML
const html = `
<!doctype html>
<meta charset="utf-8">
<div id="plot" style="width:900px;height:520px"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
  const data = ${JSON.stringify(data)};
  const layout = ${JSON.stringify(layout)};
  Plotly.newPlot("plot", data, layout);
</script>
`;

fs.writeFileSync("plots/test_scatterplot.html", html);
console.log("Open plots/test_scatterplot.html in your browser");
