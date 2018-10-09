# Hi!

Welcome to my capstone project. I generated predictions for a hard drive failure based on data of backblaze. You can find the analysis in notebook.ipynb.

If you would like to rebuild the results by yourself, I provided a environment.yml file for a anaconda environment. 
The code was running on Windows 10, 64bit with 32MB of RAM. Make sure you meet this requirements, as the notebook can get OOM otherwise. Also you should not run a lot of stuff in parallel, as a lot of snippets will use all available cpus. I have not tested the code on an other os, but I assume that it works as no specific os stuff was used. The raw data is about 5GB in size. Inside the notebook you will be able to download the data, decompress and run the data preprocessing. It is possible that you need to restart the notebook as it also can get OOM...


## Preprocess by yourself
Note that this data preprocessing will take several hours to complete (More than 6h). Just follow the steps in the notebook.

For the most cases the script will not the recalculation the results if they are present, but be sure to not split the data for each hard drive again, as this will take several hours to complete.

Have fun!

