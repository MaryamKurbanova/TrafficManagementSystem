import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gzip
from datetime import timedelta

# Load datasets
def load_data():
    with gzip.open('data/filtered_Traffic_Volume.csv.gz', 'rt') as f:
        traffic_volume = pd.read_csv(f)

    with gzip.open('data/sorted_crash_data_from_2020.csv.gz', 'rt') as f:
        collisions_2020 = pd.read_csv(f, low_memory=False)

    with gzip.open('data/sorted_crash_data_before_2020part2.csv.gz', 'rt') as f:
        collisions_before_2020_part2 = pd.read_csv(f, low_memory=False)
    
    # Combine date and time columns into a datetime object
    traffic_volume['Datetime'] = pd.to_datetime(
        traffic_volume[['Yr', 'M', 'D', 'HH', 'MM']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d-%H-%M', errors='coerce'
    )
    return traffic_volume, collisions_2020, collisions_before_2020_part2

def plot_total_traffic_volume(traffic_volume):
    # Convert Vol to numeric in case of data type issues
    traffic_volume['Vol'] = pd.to_numeric(traffic_volume['Vol'], errors='coerce')
    traffic_volume.dropna(subset=['Vol'], inplace=True)  # Drops any NaN values in 'Vol'

    # Groups by year and sum the traffic volume
    yearly_totals = traffic_volume.groupby('Yr')['Vol'].sum()
    print("Yearly Totals:", yearly_totals)  # Debugging step to inspect values

    # Plots as a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(yearly_totals.index, yearly_totals.values)
    plt.xlabel('Year')
    plt.ylabel('Total Traffic Volume')
    plt.title('Total Traffic Volume per Year')
    plt.show()

def analyze_collisions_combined(collisions_2020 , collisions_before_2020_part2):
    # Combines the two collision datasets
    combined_collisions = pd.concat([collisions_2020, collisions_before_2020_part2], ignore_index=True)

    # Converts crash date and time to datetime for combined data
    combined_collisions['Crash_Datetime'] = pd.to_datetime(
        combined_collisions['CRASH DATE'] + ' ' + combined_collisions['CRASH TIME'],
        format='%m/%d/%Y %H:%M', errors='coerce'
    )

    # Extracts the year from the Crash_Datetime
    combined_collisions['Year'] = combined_collisions['Crash_Datetime'].dt.year

    # Counts collisions per year
    yearly_collisions_combined = combined_collisions.groupby('Year').size()

    # Plots the number of collisions per year
    plt.figure(figsize=(12, 6))
    yearly_collisions_combined.plot(kind='bar', color='orange')
    plt.title('Total Collisions by Year (Combined Data)')
    plt.xlabel('Year')
    plt.ylabel('Number of Collisions')

    # Rotates x-axis labels and sets the ticks
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')

    # Shows the plot
    plt.tight_layout()
    plt.show()

# Feature Engineering
def feature_engineering(traffic_volume):
    traffic_volume['Lag_1'] = traffic_volume['Vol'].shift(1)
    traffic_volume['Lag_24'] = traffic_volume['Vol'].shift(96)  # 24-hour lag for 15-min intervals
    traffic_volume['Rolling_Mean_3'] = traffic_volume['Vol'].rolling(window=3).mean()
    traffic_volume['Rolling_Sum_3'] = traffic_volume['Vol'].rolling(window=3).sum()
    
    traffic_volume['Hour'] = traffic_volume['HH']
    traffic_volume['DayOfWeek'] = traffic_volume['Datetime'].dt.dayofweek
    for i in range(7):
        traffic_volume[f'DayOfWeek_{i}'] = (traffic_volume['DayOfWeek'] == i).astype(int)
    
    # Drops rows with NaN values generated by lag and rolling features
    traffic_volume.dropna(inplace=True)
    return traffic_volume

# Prepares data for model training
def prepare_data(traffic_volume):
    # Creates list of features
    features = ['Lag_1', 'Lag_24', 'Rolling_Mean_3', 'Rolling_Sum_3', 'Hour'] + \
               [col for col in traffic_volume.columns if 'DayOfWeek_' in col]
    
    print("Features used for training:", features)
    
    # Defines the target variable
    target = traffic_volume['Vol']
    
    # Splits into training and testing sets based on year
    train_data = traffic_volume[traffic_volume['Yr'] < 2023]
    test_data = traffic_volume[traffic_volume['Yr'] >= 2023]

    # Ensures columns exist before splitting data
    X_train = train_data[features]
    y_train = train_data['Vol']
    X_test = test_data[features]
    y_test = test_data['Vol']
    
    return X_train, X_test, y_train, y_test

# Trains the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# Evaluates the model
def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    
    # Calculates evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Prints metrics
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'R² Score: {r2:.2f}')
    
    # Plot 
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:100], label="Actual Values", color="blue")
    plt.plot(y_pred[:100], label="Potential Values", color="orange")
    plt.xlabel("Sample Index")
    plt.ylabel("Traffic Volume")
    plt.title("Actual vs. Potential Traffic Volume(2025) on Test Set (First 100 Samples)")
    plt.legend()
    plt.show()

# Creates data for 2025 based on historical patterns
def create_future_data(year=2025, traffic_volume=None, model=None):
    # Generates date range for 2025 with 15-minute intervals
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='15min')
    future_data = pd.DataFrame({'Datetime': dates})

    # Extracts features (year, month, day, etc.) for alignment with historical data
    future_data['Yr'] = future_data['Datetime'].dt.year
    future_data['M'] = future_data['Datetime'].dt.month
    future_data['D'] = future_data['Datetime'].dt.day
    future_data['HH'] = future_data['Datetime'].dt.hour
    future_data['MM'] = future_data['Datetime'].dt.minute
    future_data['Hour'] = future_data['HH']
    future_data['DayOfWeek'] = future_data['Datetime'].dt.dayofweek

    # Creates DayOfWeek binary columns for model consistency
    for i in range(7):
        future_data[f'DayOfWeek_{i}'] = (future_data['DayOfWeek'] == i).astype(int)

    # Merges historical data to provide Lag_1, Lag_24, and rolling features
    merged_data = future_data.copy()
    merged_data['DateOnly'] = merged_data['Datetime'].dt.strftime('%m-%d %H:%M')  # use month-day hour:minute as a key

    # Generates a matching key in historical data for alignment
    traffic_volume['DateOnly'] = traffic_volume['M'].astype(str).str.zfill(2) + "-" + \
                                traffic_volume['D'].astype(str).str.zfill(2) + " " + \
                                traffic_volume['HH'].astype(str).str.zfill(2) + ":" + \
                                traffic_volume['MM'].astype(str).str.zfill(2)

    # Merges historical volume data and additional columns to future data based on the DateOnly key
    merged_data = merged_data.merge(traffic_volume[['DateOnly', 'Vol', 'WktGeom', 'street', 'fromSt', 'toSt', 'Direction']],
                                   on='DateOnly', how='left', suffixes=('', '_historical'))

    # Uses historical volume data as Lag_1 and Lag_24 features
    merged_data['Lag_1'] = merged_data['Vol'].shift(1)  # Previous volume for Lag_1
    merged_data['Lag_24'] = merged_data['Vol'].shift(96)  # 24-hour prior volume for Lag_24 (assuming 15-minute intervals)
    merged_data['Rolling_Mean_3'] = merged_data['Vol'].rolling(window=3).mean()
    merged_data['Rolling_Sum_3'] = merged_data['Vol'].rolling(window=3).sum()

    # Prepares features for model
    future_features = merged_data[['Lag_1', 'Lag_24', 'Rolling_Mean_3', 'Rolling_Sum_3', 'Hour'] +
                                 [col for col in merged_data.columns if 'DayOfWeek_' in col]]

    future_data = model.predict(future_features.fillna(0))
    merged_data['Potential_Volume'] = future_data

    # Includes additional location details in the final output
    return merged_data[['Datetime', 'Yr', 'M', 'D', 'HH', 'MM', 'Potential_Volume',
                       'WktGeom', 'street', 'fromSt', 'toSt', 'Direction']]

def main():
    # Unpacks the datasets returned by load_data
    traffic_volume, collisions_2020, collisions_before_2020_part2 = load_data()
    
    # Filters the traffic_volume dataset to the years of interest
    traffic_volume = traffic_volume[(traffic_volume['Yr'] >= 2018) & (traffic_volume['Yr'] <= 2024)]

    # Plots traffic volume trends for each year for analysis
    plot_total_traffic_volume(traffic_volume)

    # Analyzes collisions combined using the two collision datasets
    analyze_collisions_combined(collisions_2020, collisions_before_2020_part2)
    
    # Further filtering
    traffic_volume = traffic_volume[(~traffic_volume['Yr'].between(2020, 2021))]

    traffic_volume = feature_engineering(traffic_volume)
    
    X_train, X_test, y_train, y_test = prepare_data(traffic_volume)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    future_data = create_future_data(year=2025, traffic_volume=traffic_volume, model=model)
    future_data.to_csv('traffic_volume_2025.csv', index=False)
    print("Saved to 'traffic_volume_2025.csv'.")


if __name__ == "__main__":
    main()
