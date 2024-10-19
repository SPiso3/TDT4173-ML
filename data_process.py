import pandas as pd
from geopy.distance import geodesic

# Load the dataset
ais_data = pd.read_csv(r"C:\Univeristy\7th_semester_ntnu_edition\Machine Learning in Practise\group_project\ais_train.csv", delimiter='|')

# Strip any leading/trailing spaces from the column names
ais_data.columns = ais_data.columns.str.strip()

# Print the column names to verify the presence of 'latitude' and 'longitude'
#print("Columns in the dataset:\n", ais_data.columns)



# Convert time column to datetime
ais_data['time'] = pd.to_datetime(ais_data['time'])




ais_data['hour'] = ais_data['time'].dt.hour
ais_data['day_of_week'] = ais_data['time'].dt.dayofweek  # 0 = Monday, 6 = Sunday
ais_data['month'] = ais_data['time'].dt.month

#print("Columns in the dataset:\n", ais_data.columns)



# Sort by vesselId and time
ais_data = ais_data.sort_values(by=['vesselId', 'time'])

# Calculate the distance traveled between consecutive points for each vessel
def calculate_distance(row, next_row):
    if not pd.isna(row['latitude']) and not pd.isna(next_row['latitude']):
        return geodesic((row['latitude'], row['longitude']),
                        (next_row['latitude'], next_row['longitude'])).meters
    return 0

# Initialize an empty column for distance_traveled
ais_data['distance_traveled'] = 0.0

# Iterate over the dataset to calculate distance traveled for each vessel
for vessel_id, group in ais_data.groupby('vesselId'):
    group = group.reset_index(drop=True)  # Ensure the group has sequential indexing
    for i in range(1, len(group)):
        # Calculate distance between current row and previous row
        ais_data.loc[group.index[i], 'distance_traveled'] = calculate_distance(group.iloc[i-1], group.iloc[i])

# Calculate time difference between consecutive timestamps
ais_data['time_diff'] = ais_data.groupby('vesselId')['time'].diff().dt.total_seconds()  # Time difference in seconds

# Calculate the speed using distance and time difference
ais_data['speed'] = ais_data['distance_traveled'] / ais_data['time_diff']  # Speed in meters per second


ais_data['heading_change'] = ais_data.groupby('vesselId')['heading'].diff().fillna(0)

#fill with zero if NaN in these collumns
ais_data['time_diff'].fillna(0, inplace=True)
ais_data['distance_traveled'].fillna(0, inplace=True)
ais_data['speed'].fillna(0, inplace=True)
ais_data['heading_change'].fillna(0, inplace=True)


print("Feature engineering completed. Here's the head of the DataFrame:\n", ais_data.head(5))



# Define the prediction horizon: shift by 3 rows (to predict 1 hour ahead if data is recorded every 20 minutes)
prediction_horizon = 3

 #Shift latitude and longitude to create future target values for each vessel
ais_data['latitude_future'] = ais_data.groupby('vesselId')['latitude'].shift(-prediction_horizon)
ais_data['longitude_future'] = ais_data.groupby('vesselId')['longitude'].shift(-prediction_horizon)

# Fill NaNs in latitude_future and longitude_future using the last known value (forward fill)
ais_data['latitude_future'].fillna(method='ffill', inplace=True)
ais_data['longitude_future'].fillna(method='ffill', inplace=True)

# Display the updated DataFrame to verify the new target columns
print("DataFrame after target engineering:\n", ais_data[['vesselId', 'time', 'latitude', 'longitude', 'latitude_future', 'longitude_future']].head(20))


print("Columns in the dataset:\n", ais_data.columns)


'''
Summary of the Code Steps
Import Libraries
Load the Dataset
Clean Column Names
Convert time Column to Datetime
Extract Time-Based Features (hour, day_of_week, month)
Sort Data by vesselId and time
Define Distance Calculation Function
Initialize distance_traveled Column
Calculate Distance Traveled for Each Vessel
Calculate Time Difference (time_diff)
Calculate Speed (speed)
Calculate Heading Change (heading_change)
Fill NaNs in Relevant Columns with Zero
Define Prediction Horizon and Shift for Target Engineering
Fill NaNs in Target Columns (latitude_future, longitude_future) with Forward Fill



'''






