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

// Build arrays per type with Polars
const tidy = polars_df
  .rename({ "Mortality Rate": "rate" })
  .select(pl.col("Type"), pl.col("Year"), pl.col("rate")) 
  .withColumns(
    pl.col("Year").cast(pl.Int32),
    pl.col("rate").cast(pl.Float32),
  );

function xy(type: string) {
  const t = tidy.filter(pl.col("Type").eq(pl.lit(type)));
  return {
    x: t.getColumn("Year").toArray() as number[],
    y: t.getColumn("rate").toArray() as number[]
  };
}

const neon = xy("Neonatal");
const infant = xy("Infant");
const data = [
  { x: neon.x,   y: neon.y,   mode: "markers", type: "scatter", name: "Neonatal", marker:{color:"#1f77b4"} },
  { x: infant.x, y: infant.y, mode: "markers", type: "scatter", name: "Infant",   marker:{color:"#ff7f0e"} }
];

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

// In your HTML <script> (or injected HTML string)



`<!doctype html>
<meta charset="utf-8">
<div id="plot" style="width:900px;height:520px"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
  Plotly.newPlot("plot", ${JSON.stringify(data)}, { xaxis:{title:"Year"}, yaxis:{title:"Mortality rate (per 1k)"} });
</script>`;

//kind of sick, the above code^^uses stringify to litteraly turn the columns into big arrays
//i.e. <script>const data = [{ x: {"0":255,"1":255,"2":255,"3":255,...

fs.writeFileSync("plots/test_scatterplot.html", html);
console.log("Open plot.html in your browser");


