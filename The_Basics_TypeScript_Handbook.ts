// this is how I type a comment in javascript



// here we are defining message as a constant
// what does that mean? this means it is "block-scoped", read-only
const message = "Hello World!"; // this is a string literal type, as in the type is "hello world" a specific string, not the type string
console.log(message);

// we cannot re-assign it, but lets say it was an array or an "object" we can still mutate its contents (unless its frozen)
// for example this would be allowed
const user = { name: "Ada" };
user.name = "Grace"; 
console.log(user);
