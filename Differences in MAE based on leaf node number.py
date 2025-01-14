from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

#compares MAE scores from different values for max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    # defining the model
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    # fitting the model
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


#main code
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('melb_data.csv')

    '''Dropping rows with missing target (Price) values.

    inplace=True means that the changes are saved to the original dataframe and
      not returned as a new dataframe.'''
    data.dropna(axis=0, subset=['Price'], inplace=True)

    # Select target(y)
    y = data.Price

    # Selecting features(X)
    features =  ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = data[features] # Selecting multiple columns from the dataframe as features

    # Divide data into training and validation subsets
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # compare MAE(Mean Absolute Error) with differing values of max_leaf_nodes of the DecisionTreeRegressor
    '''The best value for max_leaf_nodes is the one that minimizes the MAE.
    Very few leaf nodes causes underfitting (high variation from actual values) and
      very high leaf nodes causes overfitting (model will predict training features very well, but not new data).
    The best value for max_leaf_nodes captures patterns accurately and generalizes well to new data.'''

    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print(f"Max leaf nodes: {max_leaf_nodes}  \t\t Mean Absolute Error:  {round(my_mae)}")
