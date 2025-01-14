# Intro-to-Machine-Learning
Basic example of using Decision Tree Regressor to predict housing prices in Melbourne using pandas and sklearn. I learned this using kaggle.
(https://www.kaggle.com/learn/intro-to-machine-learning)

# Script-Overview
1. **Importing Libraries**:
   - Import `DecisionTreeRegressor` from `sklearn.tree` and `pandas`.

2. **Loading Data**:
   - Read the Melbourne home prices dataset (`melb_data.csv`) into a pandas DataFrame.

3. **Data Cleaning**:
   - Drop rows with missing values using `dropna()` to ensure clean data for the model.

4. **Defining Target Variable (`y`)**:
   - Select the `Price` column as the target variable (`y`).

5. **Selecting Features (`x`)**:
   - Choose relevant independent variables: `Rooms`, `Bathroom`, `Landsize`, `Lattitude`, and `Longtitude`.

6. **Defining the Model**:
   - Initialize a `DecisionTreeRegressor` model with a fixed `random_state` for reproducibility.

7. **Fitting the Model**:
   - Train the model by fitting it to the features (`x`) and target (`y`).

8. **Making Predictions**:
   - Display the first 5 rows of feature data and use the model to predict the prices for these houses.

# Source
The melb_data.csv file was collected from kaggle.com (https://www.kaggle.com/learn/intro-to-machine-learning)
