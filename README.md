# Search Methods Comparison

## Grid Search vs Randon Search vs Bayesian Optimization
Separate codes implementing each of Grid, Random and Bayesian Search on four different classification and four different regression models, on all datasets of PMLB. Along with this a single algorithm, applying all three searches on the same datasets, so as to find out which one works the best by plotting the comparison for better visualization.

### Requirements 
- All the codes are written in Python.
- Python 3.9 was used. Get all the necessary packages by [clicking here](requirements.txt), then typing the following command in your terminal : 

    ```
    pip install -r requirements.txt
   ```
- The original dataset for benchmarking has been taken from PMLB, and can be found [here](https://epistasislab.github.io/pmlb/).

### How to run
- Install all the necessary packages from the requirements section and the codes mentioned in the repository.
- Add the "main.py" file to your local python environment and run it. That is all there is to be done. 
- The file will then run the comparison of all the methods, taking in all four classification models first, and the regression models second. 
- It will end once all the plots comparing the methods for each model has been saved to your project workspace. 

### Plots
The code runs for the datasets checking whether it is a classification or regression dataset and gives us a plot of the average accuracy score with respect to each of the search methods over the entire set of datasets. The code also gives us a plot of the average time of execution of each search model over the entire dataset. Following are the two sample plots for the Random Forest Classifier method, results on the whole PMLB classification datasets :


 ![sample plot](https://github.com/chinmaydas23/ml_search_methods_comparison/blob/main/Plots/SamplePlotforRF-readme.jpg)