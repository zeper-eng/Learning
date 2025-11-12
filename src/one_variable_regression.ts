import pl from "nodejs-polars"; //appearently this is like pandas
import * as tf from "@tensorflow/tfjs-node"; //tensorflow instead of numpy but close enough 
import * as Plot from "@observablehq/plot"; //matplotlib but in js
import fs from "node:fs"; //like OS but for js
import path from "node:path";

/////////////////////////////////////////////////////////
//Tensorflow is NOT as ergonomic as it has been in python
//however, probably better for scaling things later on
//so considering that lets go ahead and use the polars version
/////////////////////////////////////////////////////////

console.log(polars_df[])