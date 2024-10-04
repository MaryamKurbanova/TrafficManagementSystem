import pandas as pd
import matplotlib.pyplot as plt
import gzip
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV


# Loads datasets from compressed files
with gzip.open('data/Automated_Traffic_Volume_Counts_20241003.csv.gz', 'rt') as f:
    traffic_volume = pd.read_csv(f)

with gzip.open('data/Motor_Vehicle_Collisions_-_Crashes_20241003.csv.gz', 'rt') as f:
    collisions = pd.read_csv(f, low_memory=False)

with gzip.open('data/FHV_Base_Aggregate_Report_20241003.csv.gz', 'rt') as f:
    fhv_data = pd.read_csv(f)

with gzip.open('data/dot_VZV_Leading_Pedestrian_Intervals_20240130.csv.gz', 'rt') as f:
    pedestrian_signals = pd.read_csv(f)

with gzip.open('data/US_Accidents_March23.csv.gz', 'rt') as f:
    us_accidents = pd.read_csv(f)

# Displays the columns of the Traffic Volume dataset
print("Traffic Volume Columns:", traffic_volume.columns)

# Function to handle NaN values in the traffic volume data
def handle_nans(dataframe):

    print("NaN values in each column before dropping 'toSt':\n", dataframe.isnull().sum())
    
    # Drops the 'toSt' column
    dataframe.drop(columns=['toSt'], inplace=True)

    print("NaN values in each column after dropping 'toSt':\n", dataframe.isnull().sum())

    # Confirms the column has been dropped
    print("Columns after dropping 'toSt':", dataframe.columns)

# Analysis function
def analyze_data():
    # Handle NaN values in the traffic volume data
    handle_nans(traffic_volume)

    # Display sample data from each dataset
    print("\nTraffic Volume Data:")
    print(traffic_volume.head())
    
    print("\nCollisions Data:")
    print(collisions.head())
    
    print("\nFor-Hire Vehicle Data:")
    print(fhv_data.head())
    
    print("\nPedestrian Signals Data:")
    print(pedestrian_signals.head())
    
    print("\nUS Accidents Data:")
    print(us_accidents.head())

    # Plots traffic volume over time
    plot_traffic_volume()

# Function to plot the traffic volume over time
def plot_traffic_volume():
    # Creates a Datetime column from separate columns
    traffic_volume['Datetime'] = pd.to_datetime(
        traffic_volume[['Yr', 'M', 'D', 'HH', 'MM']].astype(str).agg('-'.join, axis=1), 
        format='%Y-%m-%d-%H-%M', errors='coerce')

    # Ensures 'Datetime' is not null after conversion
    if traffic_volume['Datetime'].isnull().any():
        print("Warning: Some datetime values could not be parsed. Check the data.")

    # Plots the traffic volume over time
    plt.figure(figsize=(12, 6))
    plt.plot(traffic_volume['Datetime'], traffic_volume['Vol'], marker='o', linestyle='-')
    plt.title('Traffic Volume by Time')
    plt.xlabel('Datetime')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

def analyze_collisions(collisions):
    # Converts crash date and time to datetime
    collisions['Crash_Datetime'] = pd.to_datetime(
        collisions['CRASH DATE'] + ' ' + collisions['CRASH TIME'], 
        format='%m/%d/%Y %H:%M', errors='coerce'
    )

    # Extracts the year from the Crash_Datetime
    collisions['Year'] = collisions['Crash_Datetime'].dt.year

    # Count collisions per year
    yearly_collisions = collisions.groupby('Year').size()

    # Plots the number of collisions per year
    plt.figure(figsize=(12, 6))
    yearly_collisions.plot(kind='bar', color='skyblue')
    plt.title('Total Collisions by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Collisions')

    # Rotates x-axis labels and sets the ticks
    plt.xticks(rotation=45, ha='right')  # Adjust rotation and alignment
    plt.grid(axis='y')

    # Show the plot
    plt.tight_layout()  # Adjust layout
    plt.show()

# Features Engineering
def feature_engineering(traffic_volume):
    # Calculates average and total volume
    traffic_volume['Average_Volume'] = traffic_volume.groupby(['Yr', 'M', 'D', 'HH'])['Vol'].transform('mean')
    traffic_volume['Total_Volume'] = traffic_volume.groupby(['Yr', 'M', 'D'])['Vol'].transform('sum')

    # Creates additional features for the hour and day of the week
    traffic_volume['Hour'] = traffic_volume['HH']
    traffic_volume['DayOfWeek'] = pd.to_datetime(traffic_volume['Datetime']).dt.day_name()  # Extract day name

    # One-hot encode the DayOfWeek
    traffic_volume = pd.get_dummies(traffic_volume, columns=['DayOfWeek'], drop_first=True)

    return traffic_volume

# Runs the analysis
analyze_data()  # Ensures this function is defined and used appropriately
# Calls the function to analyze collisions
analyze_collisions(collisions)

# Applys feature engineering
traffic_volume = feature_engineering(traffic_volume)

# Prepares features and target variable
features = traffic_volume[['Average_Volume', 'Total_Volume', 'Hour'] + [col for col in traffic_volume.columns if 'DayOfWeek_' in col]]  # Include one-hot encoded columns
target = traffic_volume['Vol']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Prints the predicted values
print("Predicted Traffic Volumes:\n", y_pred)


# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

