#Iowa housing price prediction using decision trees

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#Generating a dataframe of the Iowa dataset obtained from the government
iowa_file_path = '../input/train.csv'
iowa_training_data = pd.read_csv(iowa_file_path)
print(iowa_training_data.columns)
print("\n")

#Defining the label/target variable
y = iowa_training_data.SalePrice
print(y.head(5))
print("\n")

#Defining the features/predictors
iowa_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = iowa_training_data[iowa_predictors]
print(x.describe())
print("\n")

#Defining the prediction model and fitting the model on the training dataset
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)
iowa_model = DecisionTreeRegressor()
iowa_model.fit(train_x, train_y)

#Initiating the prediction function for forecasting prices for the validation dataset comprising information on Iowa's housing
print("Making predictions for houses in Iowa, on their prices:")
print(val_x)
print("The values are:")
val_predictions = iowa_model.predict(val_x)
print(val_predictions)
print("\n")

#Calculating the mean absolute error
print("The mean absolute error is:", mean_absolute_error(val_y, val_predictions))

#Utility function for evaluating the model and finding the maximum leaf nodes the model must have to attain the least MAE (mean absolute error)
def get_mae(max_leaf_nodes, trainX, valX, trainY, valY):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(trainX, trainY)
    pred_val = model.predict(valX)
    mae = mean_absolute_error(valY, pred_val)
    return(mae)

#Comparing the models with differing values of maximum leaf nodes, by calling the utility function
for max_leaf_nodes in [5, 50, 500, 5000]:
    mae_obtained = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max Leaf Nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, mae_obtained))
    
#Observation: With 50 leaf nodes, the mean absolute error is the least with the value 27825!"
#--------------------------------------------------------------------------------------------------
#Using the Random Forest Model to predict the housing prices in Iowa

#forest_model = RandomForestRegressor()
#forest_model.fit(train_x, train_y)
#iowa_predictions = forest_model.predict(val_x)
#print(mean_absolute_error(val_y, iowa_predictions))
      
#Observation: Random forest model gives a lower mean absolute error of ~24200, than the decision tree model