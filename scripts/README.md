This directory stores some scripts to run in the terminal. The julia include should use relative path e.g. "../main.jl", even if running the script in .. folder. Actually, I should run on maybe in the src folder, so that all the data, saved_models, tensorboard_logs folders are shared.

```
julia --project scripts/run.jl
```

I probably want to refactor the code a bit. If I put the notebook files inside some inner folder, the notebook runs will use that inner folder for saved_models, which is probably not what I wanted.

So my plan:
- put notebooks in the top level src folder
- put all source files inside inner folder
- put python source files inside another inner folder as well
- put the python notebooks inside the top level src folder? Because I need to run the 
- finally, remember to run all scripts in the top-level src folder, where notebooks stores

I probably want to create a notebooks folder in the top-level, and put all notebooks there.