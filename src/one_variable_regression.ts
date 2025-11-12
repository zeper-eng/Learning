import pl from "nodejs-polars"; //appearently this is like pandas
import * as tf from "@tensorflow/tfjs-node"; //tensorflow instead of numpy but close enough 
import * as Plot from "@observablehq/plot"; //matplotlib but in js
import fs from "node:fs"; //like OS but for js
import path from "node:path";


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
//so considering that lets go ahead and use polars (pandas) for now; 
//i.e. dataframe type stuff
////////////////////////////////////////////////////////////////////

// for the record this is the original name incase anyone is ever wondering:NCHS_-_Infant_and_neonatal_mortality_rates__United_States__1915-2013.csv
const polars_df = pl.readCSV(
  "Test_Data/test_data.csv"
)

console.log("the polars dataframe:\n",polars_df.head());


// build arrays from Polars DataFrame
const years = polars_df.getColumn("Year").toArray() as number[];
const rates = polars_df.getColumn("Mortality Rate").toArray() as number[];

/*
ALRIGHT

So by defenition we can think of a linear regression as


*/



