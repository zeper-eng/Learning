/* 

In general when it comes to running local programs using typescript
it is often easiest to just spawn python as a java subprocess.

Most other implementations will expect live server access so running
theese as subprocess for the moment will suffice as learning continues.

*/

import { spawn, exec, fork } from "node:child_process";


//This starts a new process directly (as in no shell)
//it returns a ChildProcess object where
    //.stdin is writable stream (you can send data to the process)
    //.stdout/.stderr are readable streams (you can read output as it comes)
    //so naturally this works well for Good for long-running processes or streaming output.
    //i.e.e useful for real-time logs, process may run, output i slarge, pipe data in and out

spawn('python',["regressions.py",])


process.exit()

//spawn(command: string, args?: readonly string[], options?: SpawnOptions)
spawn() 

//This also starts a process but it makes it one-shot small output
    //Runs a command in a shell (bash/zsh)
    //collects all stdout and stderr into buffers
    //when the command finishes calls a callback with 
        //error (if any)
        //stdout (string)
        //stderr (string)
        //i.e. you just want to run a quick python command with global variables

//exec(command: string, options?: ExecOptions, callback?: ExecCallback)    
exec() 

//special case for node-node runs
    //spawns a Node.js process to run another JS/TS file
        //same as spawn but uses node with your script
        //dedicated communication channel for process.send()
        //useful for splitting work across multiple Node processes
            //i.e. multi-threading CPU-heavy taks,
            //isolated jobs

//fork(modulePath: string, args?: readonly string[], options?: ForkOptions)
fork() //Special case for spawning another Node.js script.