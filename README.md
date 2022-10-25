# Search Methods Comparison

## Grid Search vs Randon Search vs Bayesian Optimization
Separate codes implementing each of Grid, Random and Bayesian Search along with a single algorithm, applying all three searches on the same datasets, so as to find out which one works the best.

### Requirements 
- All the codes are written in Python.
- Python 3.9 was used. Get all the necessary packages by [clicking here](requirements.txt), then typing the following command in your terminal : 

    ```
    pip install -r requirements.txt
   ```
- The original dataset for benchmarking has been taken from PMLB, and can be found [here](https://epistasislab.github.io/pmlb/).
### Plots
The code runs for the datasets checking whether it is a classification or regression dataset and gives us a plot of the average accuracy score with respect to each of the search methods over the entire set of datasets. The code also gives us a plot of the average time of execution of each search model over the entire dataset. Following are the two sample plots for the Random Forest Classifier method, results on the whole PMLB classification datasets : ![](https://github.com/chinmaydas23/ml_search_methods_comparision/blob/main/BarGraphSample.pdf?raw=true)