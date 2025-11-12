// this is how I type a comment in javascript



// here we are defining message as a constant
// what does that mean? this means it is "block-scoped", read-only
const message = "Hello World!"; // this is a string literal type, as in the type is "hello world" a specific string, not the type string
console.log(message); // this is the first time we are introduced to console; we can think of this as a global object that lets us print like python/R's print
// message = "Hello World!"; // If we uncomment this you can see that theres an error before we even try to run the stuff werre looking at  


// we cannot re-assign it, but lets say it was an array or an "object" we can still mutate its contents (unless its frozen)
// for example this would be allowed
const user = { name: "Ada" };
console.log(user); // before reassignment

user.name = "Grace"; 
console.log(user); // after reassignment

// Now we can start to explore implicit operations on theese types the way you can in python
// for example
console.log(message.toLowerCase()) //lowercase in here

/*
Example output for the previous code

>>> (base) luis@Luiss-MacBook-Pro pre_startdate_miniproject % npx tsx The_Basics_TypeScript_Handbook.ts
>>> Hello World!
>>> { name: 'Ada' }
>>> { name: 'Grace' }
>>> hello world!

*/


// so some nuance for the previous example in typescript is for example
// if we do

let message2 = "Hello World!";
console.log(message2)

//we can still do
message2="new string";// as long as its still a string
console.log(message2)// the type is a string not a literal

/*
Example output for the previous code

>>> (base) luis@Luiss-MacBook-Pro pre_startdate_miniproject % npx tsx The_Basics_TypeScript_Handbook.ts
>>> Hello World!
>>> new string
*/

// lets say we wanted to define a simple function
function make_uppercase(x: string) { // x implicitly has type 'any' (this is not ideal, i.e. the red underline) so we set it to string ourselves
  return x.toUpperCase();
}

//let is for when we might reassign variables, and also expands typing to string
let string_one = 'something';
console.log('string one before changing: '+string_one)
let uppercase_string= make_uppercase(string_one)// and this breaks because a string doesnt have the method flip
console.log('string one after changing: '+uppercase_string)


//We can achieve the same thing with const if we know we wont reassign later, and also makes this a litteral assignment
let string_one_two = 'something';
console.log('string_one_two before changing: '+string_one_two)
let uppercase_string_two= make_uppercase(string_one_two)// and this breaks because a string doesnt have the method flip
console.log('string_one_two  after changing: '+uppercase_string_two)



process.exit(0)// neat os._exit(0) equivalent