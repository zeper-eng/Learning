// this is how I type a comment in javascript



// here we are defining message as a constant
// what does that mean? this means it is "block-scoped", read-only

const message = "Hello World!"; // this is a string literal type, because the variable is const and holds a primitive literal typescript narrows its type to the exact value "hello" (a subtype of a string)


// we cannot re-assign it, but lets say it was an array or an "object" we can still mutate its contents (unless its frozen)
// for example this would be allowed
const user = { name: "Ada" };
user.name = "Grace"; 
