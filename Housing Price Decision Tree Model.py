from sklearn.tree import DecisionTreeRegressor
import pandas as pd
#loading melbourne home prices data
melbourne_data = pd.read_csv("melb_data.csv")

# dropping rows with missing values
melbourne_data = melbourne_data.dropna()

#selecting the prediction target(dependent variable)
#y is a "series" which is a one-dimensional array with axis labels
y = melbourne_data.Price

#Choosing "Features"(Independent Variables)
#x is a "DataFrame" which is a two-dimensional array with axis labels
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

# Define model. Specify a number for random_state to ensure same results each run
#random_state is a parameter that allows you to set the random seed
#this is basically training the model about the relationship between x and y
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(x, y)

print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(melbourne_model.predict(x.head()))