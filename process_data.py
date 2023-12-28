import numpy as np
import pandas as pd



# import accelerometer data

# activites: climbing down, climbing up, jumping, lying, running, sitting, walking
acc_climbingdown_ua = pd.read_csv('acc_climbingdown_upperarm.csv')
acc_climbingup_ua   = pd.read_csv('acc_climbingup_upperarm.csv')
acc_jumping_ua      = pd.read_csv('acc_jumping_upperarm.csv')
acc_lying_ua        = pd.read_csv('acc_lying_upperarm.csv')
acc_running_ua      = pd.read_csv('acc_running_upperarm.csv')
acc_sitting_ua      = pd.read_csv('acc_sitting_upperarm.csv')
acc_walking_ua      = pd.read_csv('acc_walking_upperarm.csv')

acc_climbingdown_ua['activity'] = 'climbingdown'
acc_climbingup_ua['activity']   = 'climbingup'
acc_jumping_ua['activity']      = 'jumping'
acc_lying_ua['activity']        = 'lying'
acc_running_ua['activity']      = 'running'
acc_sitting_ua['activity']      = 'sitting'
acc_walking_ua['activity']      = 'walking'

acc_data = acc_climbingdown_ua.merge(acc_climbingup_ua, how='outer').merge(acc_jumping_ua, how='outer').merge(acc_lying_ua, how='outer').merge(acc_running_ua, how='outer').merge(acc_sitting_ua, how='outer').merge(acc_walking_ua, how='outer')
acc_data['attr_time'] = pd.to_datetime(acc_data['attr_time'])
acc_data.set_index('attr_time', inplace=True)

# import GPS data

# activites: climbing down, climbing up, jumping, lying, running, sitting, walking
gps_climbingdown_ua = pd.read_csv('GPS_climbingdown_upperarm.csv')
gps_climbingup_ua   = pd.read_csv('GPS_climbingup_upperarm.csv')
gps_jumping_ua      = pd.read_csv('GPS_jumping_upperarm.csv')
gps_lying_ua        = pd.read_csv('GPS_lying_upperarm.csv')
gps_running_ua      = pd.read_csv('GPS_running_upperarm.csv')
gps_sitting_ua      = pd.read_csv('GPS_sitting_upperarm.csv')
gps_walking_ua      = pd.read_csv('GPS_walking_upperarm.csv')

gps_climbingdown_ua['activity'] = 'climbingdown'
gps_climbingup_ua['activity']   = 'climbingup'
gps_jumping_ua['activity']      = 'jumping'
gps_lying_ua['activity'] = 'lying'
gps_running_ua['activity'] = 'running'
gps_sitting_ua['activity'] = 'sitting'
gps_walking_ua['activity'] = 'walking'

gps_data = gps_climbingdown_ua.merge(gps_climbingup_ua, how='outer').merge(gps_jumping_ua, how='outer').merge(gps_lying_ua, how='outer').merge(gps_running_ua, how='outer').merge(gps_sitting_ua, how='outer').merge(gps_walking_ua, how='outer')
gps_data['attr_time'] = pd.to_datetime(gps_data['attr_time'])
gps_data.set_index('attr_time', inplace=True)

# Drop 'id' column
acc_data = acc_data.drop('id', axis=1)
gps_data = gps_data.drop('id', axis=1)

acc_data = acc_data.sort_values('attr_time')
gps_data = gps_data.sort_values('attr_time')

# Timeseries Alignment:

# Set a time window for merging
time_window = pd.Timedelta('5ms')

# Merge DataFrames within the time window
aligned_df = pd.merge_asof(
    acc_data,
    gps_data,
    on='attr_time',
    tolerance=time_window,
    direction='nearest',
    by='activity'  # Add this line to specify the column for checking equality
)

# make activity the last column
activity = aligned_df.pop('activity')
aligned_df['activity'] = activity

# Randomize row order
df_randomized = aligned_df.sample(frac=1, random_state=42)  # Set a random seed for reproducibility

# Reset the index
aligned_df = df_randomized.reset_index(drop=True)

print(aligned_df)

aligned_df.to_csv('processed_data.csv', index=False)

