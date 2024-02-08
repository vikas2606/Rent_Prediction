import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Specify the directory path containing the files
directory_path = 'MLSDataBuiltAfter1995/'

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        df['CloseDate'] = pd.to_datetime(df['CloseDate'])
        df['Year'] = df['CloseDate'].dt.year
        df.fillna(0, inplace=True)
        df = df[(df['YearBuilt'] >= 1998) & (df['YearBuilt'] <= 2024) & (df['YearBuilt'] != 9999)]
        
        # Select features and target
        features = ['CumulativeDaysOnMarket', 'LivingArea', 'LotSizeArea', 'LotSizeSquareFeet', 'RATIO_ClosePrice_By_LivingArea', 'YearBuilt']
        target = 'ClosePrice'
        selected_columns = features + [target]
        selected_data = df[selected_columns]
        
        X = selected_data[features]
        y = selected_data[target]
        
        # Transform features using PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Split data into training and testing sets
        X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Predict on the testing set
        y_pred_poly = model.predict(X_test_poly)
        
        # Calculate MSE and R-squared
        mse_poly = mean_squared_error(y_test, y_pred_poly)
        r_squared_poly = model.score(X_test_poly, y_test)
        
        # Print MSE and R-squared
        print(f'File: {filename}')
        print(f'Mean Squared Error with Polynomial Features: {mse_poly}')
        print(f'R-squared with Polynomial Features: {r_squared_poly}')
        
        # Plot predictions vs true values
        plt.scatter(y_test, y_pred_poly)
        plt.xlabel('True Values')
        plt.ylabel('Predictions with Polynomial Features')
        plt.title(f'Predictions vs True Values for {filename}')
        plt.show()
