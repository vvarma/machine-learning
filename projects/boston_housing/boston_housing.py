
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Model Evaluation & Validation
# ## Project 1: Predicting Boston Housing Prices
# 
# Welcome to the first project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been written. You will need to implement additional functionality to successfully answer all of the questions for this project. Unless it is requested, do not modify any of the code that has already been included. In this template code, there are four sections which you must complete to successfully produce a prediction with your model. Each section where you will write code is preceded by a **STEP X** header with comments describing what must be done. Please read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer that relate to the project and your implementation. Each section where you will answer a question is preceded by a **QUESTION X** header. Be sure that you have carefully read each question and provide thorough answers in the text boxes that begin with "**Answer:**". Your project submission will be evaluated based on your answers to each of the questions.  
# 
# A description of the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Housing), which is provided by the **UCI Machine Learning Repository**.

# # Getting Started
# To familiarize yourself with an iPython Notebook, **try double clicking on this cell**. You will notice that the text changes so that all the formatting is removed. This allows you to make edits to the block of text you see here. This block of text (and mostly anything that's not code) is written using [Markdown](http://daringfireball.net/projects/markdown/syntax), which is a way to format text using headers, links, italics, and many other options! Whether you're editing a Markdown text block or a code block (like the one below), you can use the keyboard shortcut **Shift + Enter** or **Shift + Return** to execute the code or text block. In this case, it will show the formatted text.
# 
# Let's start by setting up some code we will need to get the rest of the project up and running. Use the keyboard shortcut mentioned above on the following code block to execute it. Alternatively, depending on your iPython Notebook program, you can press the **Play** button in the hotbar. You'll know the code block executes successfully if the message *"Boston Housing dataset loaded successfully!"* is printed.

# In[18]:

# Importing a few necessary libraries
import numpy as np
import matplotlib.pyplot as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

# Make matplotlib show our plots inline (nicely formatted in the notebook)
get_ipython().magic(u'matplotlib inline')

# Create our client's feature set for which we will be predicting a selling price
CLIENT_FEATURES = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]

# Load the Boston Housing dataset into the city_data variable
city_data = datasets.load_boston()

# Initialize the housing prices and housing features
housing_prices = city_data.target
housing_features = city_data.data

print "Boston Housing dataset loaded successfully!"


# # Statistical Analysis and Data Exploration
# In this first section of the project, you will quickly investigate a few basic statistics about the dataset you are working with. In addition, you'll look at the client's feature set in `CLIENT_FEATURES` and see how this particular sample relates to the features of the dataset. Familiarizing yourself with the data through an explorative process is a fundamental practice to help you better understand your results.

# ## Step 1
# In the code block below, use the imported `numpy` library to calculate the requested statistics. You will need to replace each `None` you find with the appropriate `numpy` coding for the proper statistic to be printed. Be sure to execute the code block each time to test if your implementation is working successfully. The print statements will show the statistics you calculate!

# In[19]:

total_houses,total_features= housing_features.shape
# Minimum housing value in the dataset
minimum_price = housing_prices.min(axis=0)
# Maximum housing value in the dataset
maximum_price =  housing_prices.max(axis=0)

# Mean house value of the dataset
mean_price =  housing_prices.mean(axis=0)

# Median house value of the dataset
median_price = None
sorted_prices= housing_prices.copy() # I was sorting housing prices all along 
sorted_prices.sort(axis=0)
if (total_houses%2==0):
    median_price=(sorted_prices[total_houses/2 -1]+sorted_prices[total_houses/2 ])/2
else:
    median_price=sorted_prices[total_houses/2 ]

# Standard deviation of housing values of the dataset
std_dev =housing_prices.std(axis=0) 

# Show the calculated statistics
print "Boston Housing dataset statistics (in $1000's):\n"
print "Total number of houses:", total_houses
print "Total number of features:", total_features
print "Minimum house price:", minimum_price
print "Maximum house price:", maximum_price
print "Mean house price: {0:.3f}".format(mean_price)
print "Median house price:", median_price
print "Standard deviation of house price: {0:.3f}".format(std_dev)


# ## Question 1
# As a reminder, you can view a description of the Boston Housing dataset [here](https://archive.ics.uci.edu/ml/datasets/Housing), where you can find the different features under **Attribute Information**. The `MEDV` attribute relates to the values stored in our `housing_prices` variable, so we do not consider that a feature of the data.
# 
# *Of the features available for each data point, choose three that you feel are significant and give a brief description for each of what they measure.*
# 
# Remember, you can **double click the text box below** to add your answer!

# Top 3 significant features IMO:
# 1. TAX : should be somewhat propotional to the actual value of the property.
# 2. RM: The number of rooms available in the house. This should also be propotional to the value.
# 3. CRIM: Crime rate of the town. Should have an inverse relation to the value.
# 

# ## Question 2
# *Using your client's feature set `CLIENT_FEATURES`, which values correspond with the features you've chosen above?*  
# **Hint: ** Run the code block below to see the client's data.

# In[20]:

# CLIENT_FEATURES=[[CLIENT_FEATURES[0][9],CLIENT_FEATURES[0][5],CLIENT_FEATURES[0][0]]]
import pandas as pd
client_feat=pd.DataFrame(CLIENT_FEATURES, columns = city_data.feature_names)
print client_feat['TAX'].name,client_feat['TAX'][0]
print client_feat['RM'].name,client_feat['RM'][0]
print client_feat['CRIM'].name,client_feat['CRIM'][0]


# TAX 680.0
# RM 5.609
# CRIM 11.95

# # Evaluating Model Performance
# In this second section of the project, you will begin to develop the tools necessary for a model to make a prediction. Being able to accurately evaluate each model's performance through the use of these tools helps to greatly reinforce the confidence in your predictions.

# ## Step 2
# In the code block below, you will need to implement code so that the `shuffle_split_data` function does the following:
# - Randomly shuffle the input data `X` and target labels (housing values) `y`.
# - Split the data into training and testing subsets, holding 30% of the data for testing.
# 
# If you use any functions not already acessible from the imported libraries above, remember to include your import statement below as well!   
# Ensure that you have executed the code block once you are done. You'll know the `shuffle_split_data` function is working if the statement *"Successfully shuffled and split the data!"* is printed.

# In[21]:

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
# Put any import statements you need for this code block here

def shuffle_split_data(X, y):
    """ Shuffles and splits data into 70% training and 30% testing subsets,
        then returns the training and testing subsets. """

    # Shuffle and split the data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    # Return the training and testing data subsets
    return X_train, y_train, X_test, y_test


# Test shuffle_split_data
try:
    X_train, y_train, X_test, y_test = shuffle_split_data(housing_features, housing_prices)
    print "Successfully shuffled and split the data!"
except Exception, e:
    print "Something went wrong with shuffling and splitting the data."+str(e)


# ## Question 3
# *Why do we split the data into training and testing subsets for our model?*

# To ensure that the model can be tested with data unseen. It ensures the model is generic enough to work with new data and is not overfitted to training data.

# ## Step 3
# In the code block below, you will need to implement code so that the `performance_metric` function does the following:
# - Perform a total error calculation between the true values of the `y` labels `y_true` and the predicted values of the `y` labels `y_predict`.
# 
# You will need to first choose an appropriate performance metric for this problem. See [the sklearn metrics documentation](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) to view a list of available metric functions. **Hint: ** Look at the question below to see a list of the metrics that were covered in the supporting course for this project.
# 
# Once you have determined which metric you will use, remember to include the necessary import statement as well!  
# Ensure that you have executed the code block once you are done. You'll know the `performance_metric` function is working if the statement *"Successfully performed a metric calculation!"* is printed.

# In[22]:

from sklearn.metrics import mean_squared_error
# Put any import statements you need for this code block here

def performance_metric(y_true, y_predict):
    """ Calculates and returns the total error between true and predicted values
        based on a performance metric chosen by the student. """

    error = mean_squared_error(y_true,y_predict)
    return error


# Test performance_metric
try:
    total_error = performance_metric(y_train, y_train)
    print "Successfully performed a metric calculation!"
except Exception,e:
    print "Something went wrong with performing a metric calculation."+ str(e)


# ## Question 4
# *Which performance metric below did you find was most appropriate for predicting housing prices and analyzing the total error. Why?*
# - *Accuracy*
# - *Precision*
# - *Recall*
# - *F1 Score*
# - *Mean Squared Error (MSE)*
# - *Mean Absolute Error (MAE)*

# Between MSE and MAE, while MSE is easier to compute it is affected by outliers as it relies on the square of the error to remove the polarity of the error. On the other hand MAE uses absolute errors due to which it is not affected by the outliers as such. For this reason I would have preferred MAE as the performance metric, but in the decision tree regressor documentation, the only supported criterion mentioned is mse, moreover using mse within the regressor and using mae for the bias variance did not make sense. Hence I am using MSE as the performance metric.
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

# ## Step 4 (Final Step)
# In the code block below, you will need to implement code so that the `fit_model` function does the following:
# - Create a scoring function using the same performance metric as in **Step 3**. See the [sklearn `make_scorer` documentation](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Build a GridSearchCV object using `regressor`, `parameters`, and `scoring_function`. See the [sklearn documentation on GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html).
# 
# When building the scoring function and GridSearchCV object, *be sure that you read the parameters documentation thoroughly.* It is not always the case that a default parameter for a function is the appropriate setting for the problem you are working on.
# 
# Since you are using `sklearn` functions, remember to include the necessary import statements below as well!  
# Ensure that you have executed the code block once you are done. You'll know the `fit_model` function is working if the statement *"Successfully fit a model to the data!"* is printed.

# In[63]:

from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot 
# Put any import statements you need for this code block

def fit_model(X, y):
    """ Tunes a decision tree regressor model using GridSearchCV on the input data X 
        and target labels y and returns this optimal model. """
    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Set up the parameters we wish to tune
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # Make an appropriate scoring function
    scoring_function = make_scorer(mean_squared_error,greater_is_better=False)

    # Make the GridSearchCV object
    reg = GridSearchCV(regressor,param_grid=parameters,scoring='mean_squared_error',verbose=0)

    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X, y)
    #print reg
    return reg.best_estimator_


# Test fit_model on entire dataset
try:
    reg = fit_model(housing_features, housing_prices)
    dot_data = StringIO() 
    tree.export_graphviz(reg, out_file=dot_data) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_pdf("boston.pdf") 
    
    print reg.feature_importances_
    print reg
    print "Successfully fit a model!" 
except  Exception, e:
    print "Something went wrong with fitting a model." + str(e)


# ## Question 5
# *What is the grid search algorithm and when is it applicable?*

# Grid search algorithm enables to permutate over a bunch of parameters and identify the best params for the same model. In a sense it adjusts the knobs of the algorithm to give the best results for the data + model combination.

# ## Question 6
# *What is cross-validation, and how is it performed on a model? Why would cross-validation be helpful when using grid search?*

# For cross validation the training data is further split into train and cross validation sets. The training set is used to first train the model with different sets of parameters that are provided. Further these models are compared on the cross validation data to decide the best parameters for the data. Isolation here ensures that the same benifits of choosing test data in model choosing.
# The basic model of cv is the k-fold cv. In this we split the training data into k bins and use k-1 bins in each iteration for training and test on the held out data. This is used for parameter modeling. This method is helpful for reasons: maximises the data used in the training while providing held out data for parameter optimization. averages the performance accross the multiple iterations, helps in the outlier problem and the 1-off cases.
# Since grid search is a method for optimal parameter selection, using cv ensures that the model is not affected by outliers in any particular bin.

# # Checkpoint!
# You have now successfully completed your last code implementation section. Pat yourself on the back! All of your functions written above will be executed in the remaining sections below, and questions will be asked about various results for you to analyze. To prepare the **Analysis** and **Prediction** sections, you will need to intialize the two functions below. Remember, there's no need to implement any more code, so sit back and execute the code blocks! Some code comments are provided if you find yourself interested in the functionality.

# In[37]:

def learning_curves(X_train, y_train, X_test, y_test):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing error rates for each model are then plotted. """
    
    print "Creating learning curve graphs for max_depths of 1, 3, 6, and 10. . ."
    
    # Create the figure window
    fig = pl.figure(figsize=(10,8))

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.rint(np.linspace(1, len(X_train), 50)).astype(int)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    # Create four different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        for i, s in enumerate(sizes):
            
            # Setup a decision tree regressor so that it learns a tree with max_depth = depth
            regressor = DecisionTreeRegressor(max_depth = depth)
            
            # Fit the learner to the training data
            regressor.fit(X_train[:s], y_train[:s])

            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
            
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test, regressor.predict(X_test))

        # Subplot the learning curve graph
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, test_err, lw = 2, label = 'Testing Error')
        ax.plot(sizes, train_err, lw = 2, label = 'Training Error')
        ax.legend()
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('Total Error')
        ax.set_xlim([0, len(X_train)])
    
    # Visual aesthetics
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize=18, y=1.03)
    fig.tight_layout()
    fig.show()


# In[38]:

def model_complexity(X_train, y_train, X_test, y_test):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """
    
    print "Creating a model complexity graph. . . "

    # We will vary the max_depth of a decision tree model from 1 to 14
    max_depth = np.arange(1, 14)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth = d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(max_depth, test_err, lw=2, label = 'Testing Error')
    pl.plot(max_depth, train_err, lw=2, label = 'Training Error')
    pl.legend()
    pl.xlabel('Maximum Depth')
    pl.ylabel('Total Error')
    pl.show()


# # Analyzing Model Performance
# In this third section of the project, you'll take a look at several models' learning and testing error rates on various subsets of training data. Additionally, you'll investigate one particular algorithm with an increasing `max_depth` parameter on the full training set to observe how model complexity affects learning and testing errors. Graphing your model's performance based on varying criteria can be beneficial in the analysis process, such as visualizing behavior that may not have been apparent from the results alone.

# In[39]:

learning_curves(X_train, y_train, X_test, y_test)


# ## Question 7
# *Choose one of the learning curve graphs that are created above. What is the max depth for the chosen model? As the size of the training set increases, what happens to the training error? What happens to the testing error?*

# Based on the above graphs, minimum testing error is obtained ~ < 40 for depth 3,6 and 9. Looking at the training error, at depths 6 and 9 the error is ~0, without any corresponding drop in testing error, suggesting an overfitting scenario in these cases. So the ideal depth with minimum complexity and testing error matching the training error is at 3. Hence 3 is the max_depth i like to choose.
# As the size of training set increases for max_depth 3 there is a sudden drop in testing error at 50 samples, after this the testing error smoothens out with noise, possibly due to the outliers in the testing set or the model itself. Similarly the training error increases till 50 samples and then smoothens out with noise. But as the number of samples increases the difference between training and testing error decreases.

# ## Question 8
# *Look at the learning curve graphs for the model with a max depth of 1 and a max depth of 10. When the model is using the full training set, does it suffer from high bias or high variance when the max depth is 1? What about when the max depth is 10?*

# At max depth 1, the training error is ~50 and is in the same range as the testing error. Since the error is still high, the model has a bias issue which could mean that the model is not complex enough to fit the data. The gap between testing and training error also suggests some variance in the model. 
# At max depth 10, the model has a low bias as the training data seems to fit the data well enough and has ~0 error. It is still affected by variance though not too high, since the test error is somewhat larger in comparison to train error. Here the training error suggests overfitting since the testing error is not affected by the lowered training error at this complexity.

# In[40]:

model_complexity(X_train, y_train, X_test, y_test)


# ## Question 9
# *From the model complexity graph above, describe the training and testing errors as the max depth increases. Based on your interpretation of the graph, which max depth results in a model that best generalizes the dataset? Why?*

# As the max depth increases a general trend of decreasing training error can be noticed. This suggests that with increasing complexity (upto 12) the bias is reducing, the model is fitting the data better at higher depths. The test error drops to its minimum at max_depth 6. Upto this point the train and test error can be seen to be reducing together. At max depth 7 and beyond the error is not decreasing and oscillates , possibly because the model is overfitted to the training data so much so that it is not generalizing to the test data. THe best depth which best generalizes the model, data would in my opinion be at 6, since at 6 the testing error is at minimum, while training error is also not too low (not overfit) and also has a moderate complexity.
# 

# # Model Prediction
# In this final section of the project, you will make a prediction on the client's feature set using an optimized model from `fit_model`. When applying grid search along with cross-validation to optimize your model, it would typically be performed and validated on a training set and subsequently evaluated on a **dedicated test set**. In this project, the optimization below is performed on the *entire dataset* (as opposed to the training set you made above) due to the many outliers in the data. Using the entire dataset for training provides for a less volatile prediction at the expense of not testing your model's performance. 
# 
# *To answer the following questions, it is recommended that you run the code blocks several times and use the median or mean value of the results.*

# ## Question 10
# *Using grid search on the entire dataset, what is the optimal `max_depth` parameter for your model? How does this result compare to your intial intuition?*  
# **Hint: ** Run the code block below to see the max depth produced by your optimized model.

# In[62]:

print "Final model has an optimal max_depth parameter of", reg.get_params()['max_depth']


# Final model has an optimal max_depth parameter of 4. On running the model multiple times this value oscillates between 4,6,5,7 though 4 and 6 are the ones that are more frequent.
# 
# The bias variance graph showed that after depth of 3 the error was more or less a constant, with increasing issue of overfitting at higher depths. The model complexity graph showed a minimum test error at depth 6, while train error still not low enough to suggest over fitting. This is more or less consistent with the parameter obtained from grid search.

# ## Question 11
# *With your parameter-tuned model, what is the best selling price for your client's home? How does this selling price compare to the basic statistics you calculated on the dataset?*  
# 
# **Hint: ** Run the code block below to have your parameter-tuned model make a prediction on the client's home.

# In[64]:

sale_price = reg.predict(CLIENT_FEATURES)
print "Predicted value of client's home: {0:.3f}".format(sale_price[0])


# Predicted value of client's home: 21.630
# Copying relevant stats from above:
# Maximum house price: 50.0
# Mean house price: 22.533
# Median house price: 21.2
# Standard deviation of house price: 9.188
# 
# The predicted value of the house is around the mean and the median, making it an average priced house.

# ## Question 12 (Final Question):
# *In a few sentences, discuss whether you would use this model or not to predict the selling price of future clients' homes in the Greater Boston area.*

# While this model is a good start to predicting the selling price of homes in Boston, it suffers from residual variance at the best max_depth since it is not able to generalize over the entire data set with. As the depth increases the splits increasingly overfit to the training set. 
# 
# Moreover, a decision tree regressor works by splitting the data on an axis that minimizes the error at each level (has a disadvantage of being unable to use combinations of parameters in a single split . This can be seen in the graph boston.pdf in the project directory. It may be interesting to see how other models such as linear regression work with this dataset with algorithms such as gradient descent.
# 
# 

# In[ ]:



