from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
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
X = melbourne_data[melbourne_features]

#splitting the data into training and validation data, for both features and target
#The split is based on a random number generator.
#Supplying a numeric value to the random_state argument guarantees we get the same split every time we run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model (Decision Tree Regressor)
melbourne_model = DecisionTreeRegressor()
# Fit model (Training the model)
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
#the predict function returns predicted prices for each house in the validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))