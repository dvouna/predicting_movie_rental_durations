# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# Load the dataset
movie_rentals = pd.read_csv('rental_info.csv') 
print(movie_rentals.info()) 


# Explore Data 
movie_rentals.head(10) 


# Convert rental_date and return_date to datetime format
movie_rentals['rental_date'] = pd.to_datetime(movie_rentals['rental_date']) 
movie_rentals['return_date'] = pd.to_datetime(movie_rentals['return_date'])

# Create rental length and rental length days column from rental and return dates
movie_rentals['rental_length'] = pd.to_datetime(
    movie_rentals["return_date"]) - pd.to_datetime(movie_rentals["rental_date"]) 
  
movie_rentals['rental_length_days'] = movie_rentals['rental_length'].dt.days 



# Determine the number of unique values in the 'special_features' column
movie_rentals['special_features'].value_counts()


# Create binary columns for special features
movie_rentals["deleted_scenes"] =  np.where(
    movie_rentals["special_features"].str.contains("Deleted Scenes"), 1,0) 
 
movie_rentals["behind_the_scenes"] = np.where(
    movie_rentals["special_features"].str.contains("Behind the Scenes"), 1,0) 
  

# Define the features and target variable
X = movie_rentals.drop(columns=['rental_date', 'return_date', 'rental_length', 'rental_length_days', 'special_features',], axis=1)
y = movie_rentals['rental_length_days']


# Perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)


# Instantiate and fit a Lasso regression model and get the coefficients
lasso = Lasso(alpha=0.3, random_state=9)
lasso_coef = lasso.fit(X, y).coef_ 


# Select the features with non-zero coefficients 
X_lasso_train, X_lasso_test = X_train.iloc[:, lasso_coef > 0], X_test.iloc[:, lasso_coef > 0]


# Instantiate and fit a Linear Regression model using the selected features
lr = LinearRegression()
lr = lr.fit(X_lasso_train, y_train) 


# Predict on the test set and calculate the mean squared error
y_test_pred = lr.predict(X_lasso_test) 
mse_lr_lasso = mean_squared_error(y_test, y_test_pred) 
print(mse_lr_lasso) 


# Random Forest Regressor with Randomized Search for hyperparameter tuning

# Define the parameter grid for Randomized Search
params = {'n_estimators': np.arange(1, 101, 1), 'max_depth':np.arange(1,11,1)} 


# Instantiate the Random Forest Regressor 
rf = RandomForestRegressor()  


# Instantiate the Randomized Search with cross-validation
rand_search = RandomizedSearchCV(rf, param_distributions=params, cv=5, random_state=5) 


# Fit the Randomized Search to the training data
rand_search.fit(X_train, y_train) 


# Get the best parameters from the Randomized Search
best_params = rand_search.best_params_  
print(best_params) 


# Instantiate the Random Forest Regressor with the best parameters
rf = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=9)


# Fit the Random Forest model to the training data
rf.fit(X_train, y_train)  


# Predict on the test set and calculate the mean squared error
rf_pred = rf.predict(X_test) 
mse_rf = mean_squared_error(y_test, rf_pred) 
print(mse_rf)