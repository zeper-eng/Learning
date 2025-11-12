import pl from "nodejs-polars"; //appearently this is like pandas
import * as tf from "@tensorflow/tfjs-node"; //tensorflow instead of numpy but close enough 
import fs from "node:fs"; //like OS but for js
import path from "node:path";


//Reading in data using the various workflows


//////////////////////////////////////
//Lets try out the pandas version
//////////////////////////////////////

// for the record this is the original name incase anyone is ever wondering:NCHS_-_Infant_and_neonatal_mortality_rates__United_States__1915-2013.csv
const polars_df = pl.readCSV(
  "Test_Data/test_data.csv"
)

console.log("the polars dataframe:\n",polars_df.head());

const rate = polars_df.getColumn("Mortality Rate");   // Series
console.log(rate.head().toString());

//////////////////////////////////////
//Lets try out the Tensorflow version
//////////////////////////////////////

// for the record this is the original name incase anyone is ever wondering:NCHS_-_Infant_and_neonatal_mortality_rates__United_States__1915-2013.csv

const csvUrl = "file://" + path.resolve("Test_Data/test_data.csv");//bc its annoying about full paths
console.log(csvUrl)

const tensorflow_df= tf.data.csv(
    csvUrl,
     {
  hasHeader: true,              // CSV has a header row
  // columnConfigs: {            // optional: enforce dtypes or select columns
  //   Year: { dtype: "int32" },
  //   "Mortality Rate": { dtype: "float32" }
  // }
});

// This technically returns an iterable so
const tensorflow_df_as_array= await tensorflow_df.toArray()

console.log("the tensorflow dataframe:\n",tensorflow_df_as_array);


/////////////////////////////////////////////////////////
//Tensorflow is NOT as ergonomic as it has been in python
//however, probably better for scaling things later on
//so considering that lets go ahead and use the polars version
//-->continued in "one_variable_regression.ts"
/////////////////////////////////////////////////////////


// build arrays from your Polars DataFrame
const years = polars_df.getColumn("Year").toArray() as number[];
const rates = polars_df.getColumn("Mortality Rate").toArray() as number[];

// Emit a tiny HTML that loads Plotly from CDN
// I guess this is easier than figuring out all the packages
// Will learn some html as I go along for this?

const html = 
//use standard browser mode
//default charachter encoding
//placeholder DOM for plotly ot generate plot :: 
//fetch plotly from cdn:: 
//script is the actual inline script
//also backticks here ``` allow us to do inline code similar to f"" 

`<!doctype html>
<meta charset="utf-8">
<div id="plot" style="width:900px;height:520px"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
  const data = [{ x: ${JSON.stringify(polars_df.getColumn('Year'))}, y: ${JSON.stringify(polars_df.getColumn('Mortality Rate'))}, mode: "markers", type: "scatter"}];
  Plotly.newPlot('plot', data, { xaxis:{title:'Year'}, yaxis:{title:'Mortality rate)'} });
</script>`;
fs.writeFileSync("plot.html", html);
console.log("Open plot.html in your browser");


