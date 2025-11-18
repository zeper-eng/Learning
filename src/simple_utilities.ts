import pl from "nodejs-polars"; 


// Define the shape of the XY data for use as a type for me
export interface XYSeries {
    //where the variable x(Year) is independent and rate(y) is dependent
  x: Array<number>;
  y: Array<number>;
  label?: string;
}

// use for grabbing stuff from my custom little dataset will generalize later
export function xy(type: string,tidy:pl.DataFrame): XYSeries {
  const t = tidy.filter(pl.col("Type").eq(pl.lit(type)));
  return {
    x: t.getColumn("Year").toArray() as number[],
    y: t.getColumn("rate").toArray() as number[]
  };
}


//by the way {} is equivalent to =>, kind of like R syntax

export const mean = (arr: number[]) =>
  arr.reduce((a, b) => a + b, 0) / arr.length;//so here a is our accumulator and we keep adding b then divide

export const sum = (arr: number[]) =>
  arr.reduce((a, b) => a + b, 0);//so here a is our accumulator and we keep adding b 

export const variance = (arr: number[]) => {
  const m = mean(arr);//use previous functuon for mean
  return arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length;//calculate variance from sum of error
};

export const std = (arr: number[]) =>
  Math.sqrt(variance(arr));//standard deviation is just sqrt of that

