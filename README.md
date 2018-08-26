# Hi!

Welcome to my capstone project. I generated predictions for a hard drive failure based on data of backblaze. You can find the analysis in notebook.ipynb.

If you would like to rebuild the results by yourself, I provided a environment.yml file for a anaconda environment. 
The code was running on Windows 10, 64bit with 32MB of RAM. Make sure you meet this requirements, as the notebook can get OOM otherwise. Also you should not run a lot of stuff in parallel, as a lot of snippets will use all available cpus. I have not tested the code on an other os, but I assume that it works as no specific os stuff was used. The raw data is about 5GB in size. Inside the notebook you will be able to download the data, decompress and run the data preprocessing. It is possible that you need to restart the notebook as it also can get OOM...


## Preprocess by yourself
Note that this data preprocessing will take several hours to complete (More than 6h). Just follow the steps in the notebook.


## Using my preprocessed data
When you are unpatient, I provided the data in a separate repository, its compressed 4GB in size. Clone the code from: https://github.com/dariusgm/machine_learning_engineer_nanodegree_data and run the "decompress.ipynb" to unpack the data. The packed data is about 4GB in size. Make sure the git lfs extension is activated on your system. Move that data into the location of your notebook to have the script access to the pregenerated results. 

For the most cases the script will not the recalculation the results if they are present, but be sure to not split the data for each hard drive again, as this will take several hours to complete.

Have fun!

