import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Dropout, Flatten
from keras.utils import plot_model
from keras.optimizers import Adam
import seaborn as sns


# import accelerometer data

# activites: climbing down, climbing up, jumping, lying, running, sitting, walking
acc_climbingdown_chest = pd.read_csv('acc_climbingdown_chest.csv')
acc_climbingdown_fa    = pd.read_csv('acc_climbingdown_forearm.csv')
acc_climbingdown_head  = pd.read_csv('acc_climbingdown_head.csv')
acc_climbingdown_shin  = pd.read_csv('acc_climbingdown_shin.csv')
acc_climbingdown_ua    = pd.read_csv('acc_climbingdown_upperarm.csv')
acc_climbingdown_waist = pd.read_csv('acc_climbingdown_waist.csv')

acc_climbingup_chest = pd.read_csv('acc_climbingup_chest.csv')
acc_climbingup_fa    = pd.read_csv('acc_climbingup_forearm.csv')
acc_climbingup_head  = pd.read_csv('acc_climbingup_head.csv')
acc_climbingup_shin  = pd.read_csv('acc_climbingup_shin.csv')
acc_climbingup_ua    = pd.read_csv('acc_climbingup_upperarm.csv')
acc_climbingup_waist = pd.read_csv('acc_climbingup_waist.csv')

acc_jumping_chest = pd.read_csv('acc_jumping_chest.csv')
acc_jumping_fa    = pd.read_csv('acc_jumping_forearm.csv')
acc_jumping_head  = pd.read_csv('acc_jumping_head.csv')
acc_jumping_shin  = pd.read_csv('acc_jumping_shin.csv')
acc_jumping_ua    = pd.read_csv('acc_jumping_upperarm.csv')
acc_jumping_waist = pd.read_csv('acc_jumping_waist.csv')

acc_lying_chest = pd.read_csv('acc_lying_chest.csv')
acc_lying_fa    = pd.read_csv('acc_lying_forearm.csv')
acc_lying_head  = pd.read_csv('acc_lying_head.csv')
acc_lying_shin  = pd.read_csv('acc_lying_shin.csv')
acc_lying_ua    = pd.read_csv('acc_lying_upperarm.csv')
acc_lying_waist = pd.read_csv('acc_lying_waist.csv')

acc_running_chest = pd.read_csv('acc_running_chest.csv')
acc_running_fa    = pd.read_csv('acc_running_forearm.csv')
acc_running_head  = pd.read_csv('acc_running_head.csv')
acc_running_shin  = pd.read_csv('acc_running_shin.csv')
acc_running_ua    = pd.read_csv('acc_running_upperarm.csv')
acc_running_waist = pd.read_csv('acc_running_waist.csv')

acc_sitting_chest = pd.read_csv('acc_sitting_chest.csv')
acc_sitting_fa    = pd.read_csv('acc_sitting_forearm.csv')
acc_sitting_head  = pd.read_csv('acc_sitting_head.csv')
acc_sitting_shin  = pd.read_csv('acc_sitting_shin.csv')
acc_sitting_ua    = pd.read_csv('acc_sitting_upperarm.csv')
acc_sitting_waist = pd.read_csv('acc_sitting_waist.csv')

acc_walking_chest = pd.read_csv('acc_walking_chest.csv')
acc_walking_fa    = pd.read_csv('acc_walking_forearm.csv')
acc_walking_head  = pd.read_csv('acc_walking_head.csv')
acc_walking_shin  = pd.read_csv('acc_walking_shin.csv')
acc_walking_ua    = pd.read_csv('acc_walking_upperarm.csv')
acc_walking_waist = pd.read_csv('acc_walking_waist.csv')

acc_climbingdown_chest = acc_climbingdown_chest.rename(columns={"attr_x": "climbingdown_chest_x", "attr_y": "climbingdown_chest_y", "attr_z": "climbingdown_chest_z"})
acc_climbingdown_fa    = acc_climbingdown_fa.rename(   columns={"attr_x": "climbingdown_fa_x",    "attr_y": "climbingdown_fa_y",    "attr_z": "climbingdown_fa_z"})
acc_climbingdown_head  = acc_climbingdown_head.rename( columns={"attr_x": "climbingdown_head_x",  "attr_y": "climbingdown_head_y",  "attr_z": "climbingdown_head_z"})
acc_climbingdown_shin  = acc_climbingdown_shin.rename( columns={"attr_x": "climbingdown_shin_x",  "attr_y": "climbingdown_shin_y",  "attr_z": "climbingdown_shin_z"})
acc_climbingdown_ua    = acc_climbingdown_ua.rename(   columns={"attr_x": "climbingdown_ua_x",    "attr_y": "climbingdown_ua_y",    "attr_z": "climbingdown_ua_z"})
acc_climbingdown_waist = acc_climbingdown_waist.rename(columns={"attr_x": "climbingdown_waist_x", "attr_y": "climbingdown_waist_y", "attr_z": "climbingdown_waist_z"})
acc_climbingdown_data  = acc_climbingdown_chest.merge(acc_climbingdown_fa, how='outer').merge(acc_climbingdown_head, how='outer').merge(acc_climbingdown_shin, how='outer').merge(acc_climbingdown_ua, how='outer').merge(acc_climbingdown_waist, how='outer')
acc_climbingdown_ua['activity'] = 'climbingdown'

acc_climbingup_chest = acc_climbingup_chest.rename(columns={"attr_x": "climbingup_chest_x", "attr_y": "climbingup_chest_y", "attr_z": "climbingup_chest_z"})
acc_climbingup_fa    = acc_climbingup_fa.rename(   columns={"attr_x": "climbingup_fa_x",    "attr_y": "climbingup_fa_y",    "attr_z": "climbingup_fa_z"})
acc_climbingup_head  = acc_climbingup_head.rename( columns={"attr_x": "climbingup_head_x",  "attr_y": "climbingup_head_y",  "attr_z": "climbingup_head_z"})
acc_climbingup_shin  = acc_climbingup_shin.rename( columns={"attr_x": "climbingup_shin_x",  "attr_y": "climbingup_shin_y",  "attr_z": "climbingup_shin_z"})
acc_climbingup_ua    = acc_climbingup_ua.rename(   columns={"attr_x": "climbingup_ua_x",    "attr_y": "climbingup_ua_y",    "attr_z": "climbingup_ua_z"})
acc_climbingup_waist = acc_climbingup_waist.rename(columns={"attr_x": "climbingup_waist_x", "attr_y": "climbingup_waist_y", "attr_z": "climbingup_waist_z"})
acc_climbingup_data  = acc_climbingup_chest.merge(acc_climbingup_fa, how='outer').merge(acc_climbingup_head, how='outer').merge(acc_climbingup_shin, how='outer').merge(acc_climbingup_ua, how='outer').merge(acc_climbingup_waist, how='outer')
acc_climbingup_ua['activity'] = 'climbingup'

acc_jumping_chest = acc_jumping_chest.rename(columns={"attr_x": "jumping_chest_x", "attr_y": "jumping_chest_y", "attr_z": "jumping_chest_z"})
acc_jumping_fa    = acc_jumping_fa.rename(   columns={"attr_x": "jumping_fa_x",    "attr_y": "jumping_fa_y",    "attr_z": "jumping_fa_z"})
acc_jumping_head  = acc_jumping_head.rename( columns={"attr_x": "jumping_head_x",  "attr_y": "jumping_head_y",  "attr_z": "jumping_head_z"})
acc_jumping_shin  = acc_jumping_shin.rename( columns={"attr_x": "jumping_shin_x",  "attr_y": "jumping_shin_y",  "attr_z": "jumping_shin_z"})
acc_jumping_ua    = acc_jumping_ua.rename(   columns={"attr_x": "jumping_ua_x",    "attr_y": "jumping_ua_y",    "attr_z": "jumping_ua_z"})
acc_jumping_waist = acc_jumping_waist.rename(columns={"attr_x": "jumping_waist_x", "attr_y": "jumping_waist_y", "attr_z": "jumping_waist_z"})
acc_jumping_data  = acc_jumping_chest.merge(acc_jumping_fa, how='outer').merge(acc_jumping_head, how='outer').merge(acc_jumping_shin, how='outer').merge(acc_jumping_ua, how='outer').merge(acc_jumping_waist, how='outer')
acc_jumping_ua['activity'] = 'jumping'

acc_lying_chest = acc_lying_chest.rename(columns={"attr_x": "lying_chest_x", "attr_y": "lying_chest_y", "attr_z": "lying_chest_z"})
acc_lying_fa    = acc_lying_fa.rename(   columns={"attr_x": "lying_fa_x",    "attr_y": "lying_fa_y",    "attr_z": "lying_fa_z"})
acc_lying_head  = acc_lying_head.rename( columns={"attr_x": "lying_head_x",  "attr_y": "lying_head_y",  "attr_z": "lying_head_z"})
acc_lying_shin  = acc_lying_shin.rename( columns={"attr_x": "lying_shin_x",  "attr_y": "lying_shin_y",  "attr_z": "lying_shin_z"})
acc_lying_ua    = acc_lying_ua.rename(   columns={"attr_x": "lying_ua_x",    "attr_y": "lying_ua_y",    "attr_z": "lying_ua_z"})
acc_lying_waist = acc_lying_waist.rename(columns={"attr_x": "lying_waist_x", "attr_y": "lying_waist_y", "attr_z": "lying_waist_z"})
acc_lying_data  = acc_lying_chest.merge(acc_lying_fa, how='outer').merge(acc_lying_head, how='outer').merge(acc_lying_shin, how='outer').merge(acc_lying_ua, how='outer').merge(acc_lying_waist, how='outer')
acc_lying_ua['activity'] = 'lying'

acc_running_chest = acc_running_chest.rename(columns={"attr_x": "running_chest_x", "attr_y": "running_chest_y", "attr_z": "running_chest_z"})
acc_running_fa    = acc_running_fa.rename(   columns={"attr_x": "running_fa_x",    "attr_y": "running_fa_y",    "attr_z": "running_fa_z"})
acc_running_head  = acc_running_head.rename( columns={"attr_x": "running_head_x",  "attr_y": "running_head_y",  "attr_z": "running_head_z"})
acc_running_shin  = acc_running_shin.rename( columns={"attr_x": "running_shin_x",  "attr_y": "running_shin_y",  "attr_z": "running_shin_z"})
acc_running_ua    = acc_running_ua.rename(   columns={"attr_x": "running_ua_x",    "attr_y": "running_ua_y",    "attr_z": "running_ua_z"})
acc_running_waist = acc_running_waist.rename(columns={"attr_x": "running_waist_x", "attr_y": "running_waist_y", "attr_z": "running_waist_z"})
acc_running_data  = acc_running_chest.merge(acc_running_fa, how='outer').merge(acc_running_head, how='outer').merge(acc_running_shin, how='outer').merge(acc_running_ua, how='outer').merge(acc_running_waist, how='outer')
acc_running_ua['activity'] = 'running'


acc_sitting_chest = acc_sitting_chest.rename(columns={"attr_x": "sitting_chest_x", "attr_y": "sitting_chest_y", "attr_z": "sitting_chest_z"})
acc_sitting_fa    = acc_sitting_fa.rename(   columns={"attr_x": "sitting_fa_x",    "attr_y": "sitting_fa_y",    "attr_z": "sitting_fa_z"})
acc_sitting_head  = acc_sitting_head.rename( columns={"attr_x": "sitting_head_x",  "attr_y": "sitting_head_y",  "attr_z": "sitting_head_z"})
acc_sitting_shin  = acc_sitting_shin.rename( columns={"attr_x": "sitting_shin_x",  "attr_y": "sitting_shin_y",  "attr_z": "sitting_shin_z"})
acc_sitting_ua    = acc_sitting_ua.rename(   columns={"attr_x": "sitting_ua_x",    "attr_y": "sitting_ua_y",    "attr_z": "sitting_ua_z"})
acc_sitting_waist = acc_sitting_waist.rename(columns={"attr_x": "sitting_waist_x", "attr_y": "sitting_waist_y", "attr_z": "sitting_waist_z"})
acc_sitting_data  = acc_sitting_chest.merge(acc_sitting_fa, how='outer').merge(acc_sitting_head, how='outer').merge(acc_sitting_shin, how='outer').merge(acc_sitting_ua, how='outer').merge(acc_sitting_waist, how='outer')
acc_sitting_ua['activity'] = 'sitting'

acc_walking_chest = acc_walking_chest.rename(columns={"attr_x": "walking_chest_x", "attr_y": "walking_chest_y", "attr_z": "walking_chest_z"})
acc_walking_fa    = acc_walking_fa.rename(   columns={"attr_x": "walking_fa_x",    "attr_y": "walking_fa_y",    "attr_z": "walking_fa_z"})
acc_walking_head  = acc_walking_head.rename( columns={"attr_x": "walking_head_x",  "attr_y": "walking_head_y",  "attr_z": "walking_head_z"})
acc_walking_shin  = acc_walking_shin.rename( columns={"attr_x": "walking_shin_x",  "attr_y": "walking_shin_y",  "attr_z": "walking_shin_z"})
acc_walking_ua    = acc_walking_ua.rename(   columns={"attr_x": "walking_ua_x",    "attr_y": "walking_ua_y",    "attr_z": "walking_ua_z"})
acc_walking_waist = acc_walking_waist.rename(columns={"attr_x": "walking_waist_x", "attr_y": "walking_waist_y", "attr_z": "walking_waist_z"})
acc_walking_data  = acc_walking_chest.merge(acc_walking_fa, how='outer').merge(acc_walking_head, how='outer').merge(acc_walking_shin, how='outer').merge(acc_walking_ua, how='outer').merge(acc_walking_waist, how='outer')
acc_walking_ua['activity'] = 'walking'

acc_data = acc_climbingdown_ua.merge(acc_climbingup_ua, how='outer').merge(acc_jumping_ua, how='outer').merge(acc_lying_ua, how='outer').merge(acc_running_ua, how='outer').merge(acc_sitting_ua, how='outer').merge(acc_walking_ua, how='outer')
acc_data['attr_time'] = pd.to_datetime(acc_data['attr_time'])
acc_data.set_index('attr_time', inplace=True)

# import GPS data

# activites: climbing down, climbing up, jumping, lying, running, sitting, walking
gps_climbingdown_chest = pd.read_csv('GPS_climbingdown_chest.csv')
gps_climbingdown_head  = pd.read_csv('GPS_climbingdown_head.csv')
gps_climbingdown_shin  = pd.read_csv('GPS_climbingdown_shin.csv')
gps_climbingdown_ua    = pd.read_csv('GPS_climbingdown_upperarm.csv')
gps_climbingdown_waist = pd.read_csv('GPS_climbingdown_waist.csv')

gps_climbingup_chest = pd.read_csv('GPS_climbingup_chest.csv')
gps_climbingup_head  = pd.read_csv('GPS_climbingup_head.csv')
gps_climbingup_shin  = pd.read_csv('GPS_climbingup_shin.csv')
gps_climbingup_ua    = pd.read_csv('GPS_climbingup_upperarm.csv')
gps_climbingup_waist = pd.read_csv('GPS_climbingup_waist.csv')

gps_jumping_chest = pd.read_csv('GPS_jumping_chest.csv')
gps_jumping_head  = pd.read_csv('GPS_jumping_head.csv')
gps_jumping_shin  = pd.read_csv('GPS_jumping_shin.csv')
gps_jumping_ua    = pd.read_csv('GPS_jumping_upperarm.csv')
gps_jumping_waist = pd.read_csv('GPS_jumping_waist.csv')

gps_lying_chest = pd.read_csv('GPS_lying_chest.csv')
gps_lying_head  = pd.read_csv('GPS_lying_head.csv')
gps_lying_shin  = pd.read_csv('GPS_lying_shin.csv')
gps_lying_ua    = pd.read_csv('GPS_lying_upperarm.csv')
gps_lying_waist = pd.read_csv('GPS_lying_waist.csv')

gps_running_chest = pd.read_csv('GPS_running_chest.csv')
gps_running_head  = pd.read_csv('GPS_running_head.csv')
gps_running_shin  = pd.read_csv('GPS_running_shin.csv')
gps_running_ua    = pd.read_csv('GPS_running_upperarm.csv')
gps_running_waist = pd.read_csv('GPS_running_waist.csv')

gps_sitting_chest = pd.read_csv('GPS_sitting_chest.csv')
gps_sitting_head  = pd.read_csv('GPS_sitting_head.csv')
gps_sitting_shin  = pd.read_csv('GPS_sitting_shin.csv')
gps_sitting_ua    = pd.read_csv('GPS_sitting_upperarm.csv')
gps_sitting_waist = pd.read_csv('GPS_sitting_waist.csv')

gps_walking_chest = pd.read_csv('GPS_walking_chest.csv')
gps_walking_head  = pd.read_csv('GPS_walking_head.csv')
gps_walking_shin  = pd.read_csv('GPS_walking_shin.csv')
gps_walking_ua    = pd.read_csv('GPS_walking_upperarm.csv')
gps_walking_waist = pd.read_csv('GPS_walking_waist.csv')

gps_climbingdown_chest = gps_climbingdown_chest.rename(columns={"attr_lat": "climbingdown_chest_lat", "attr_lng": "climbingdown_chest_lng"})
gps_climbingdown_head  = gps_climbingdown_head.rename( columns={"attr_lat": "climbingdown_head_lat",  "attr_lng": "climbingdown_head_lng"})
gps_climbingdown_shin  = gps_climbingdown_shin.rename( columns={"attr_lat": "climbingdown_shin_lat",  "attr_lng": "climbingdown_shin_lng"})
gps_climbingdown_ua    = gps_climbingdown_ua.rename(   columns={"attr_lat": "climbingdown_ua_lat",    "attr_lng": "climbingdown_ua_lng"})
gps_climbingdown_waist = gps_climbingdown_waist.rename(columns={"attr_lat": "climbingdown_waist_lat", "attr_lng": "climbingdown_waist_lng"})
gps_climbingdown_data  = gps_climbingdown_chest.merge(gps_climbingdown_head, how='outer').merge(gps_climbingdown_shin, how='outer').merge(gps_climbingdown_ua, how='outer').merge(gps_climbingdown_waist, how='outer')
gps_climbingdown_ua['activity'] = 'climbingdown'

gps_climbingup_chest = gps_climbingup_chest.rename(columns={"attr_lat": "climbingup_chest_lat", "attr_lng": "climbingup_chest_lng"})
gps_climbingup_head  = gps_climbingup_head.rename( columns={"attr_lat": "climbingup_head_lat",  "attr_lng": "climbingup_head_lng"})
gps_climbingup_shin  = gps_climbingup_shin.rename( columns={"attr_lat": "climbingup_shin_lat",  "attr_lng": "climbingup_shin_lng"})
gps_climbingup_ua    = gps_climbingup_ua.rename(   columns={"attr_lat": "climbingup_ua_lat",    "attr_lng": "climbingup_ua_lng"})
gps_climbingup_waist = gps_climbingup_waist.rename(columns={"attr_lat": "climbingup_waist_lat", "attr_lng": "climbingup_waist_lng"})
gps_climbingup_data  = gps_climbingup_chest.merge(gps_climbingup_head, how='outer').merge(gps_climbingup_shin, how='outer').merge(gps_climbingup_ua, how='outer').merge(gps_climbingup_waist, how='outer')
gps_climbingup_ua['activity'] = 'climbingup'

gps_jumping_chest = gps_jumping_chest.rename(columns={"attr_lat": "jumping_chest_lat", "attr_lng": "jumping_chest_lng"})
gps_jumping_head  = gps_jumping_head.rename( columns={"attr_lat": "jumping_head_lat",  "attr_lng": "jumping_head_lng"})
gps_jumping_shin  = gps_jumping_shin.rename( columns={"attr_lat": "jumping_shin_lat",  "attr_lng": "jumping_shin_lng"})
gps_jumping_ua    = gps_jumping_ua.rename(   columns={"attr_lat": "jumping_ua_lat",    "attr_lng": "jumping_ua_lng"})
gps_jumping_waist = gps_jumping_waist.rename(columns={"attr_lat": "jumping_waist_lat", "attr_lng": "jumping_waist_lng"})
gps_jumping_data  = gps_jumping_chest.merge(gps_jumping_head, how='outer').merge(gps_jumping_shin, how='outer').merge(gps_jumping_ua, how='outer').merge(gps_jumping_waist, how='outer')
gps_jumping_ua['activity'] = 'jumping'

gps_lying_chest = gps_lying_chest.rename(columns={"attr_lat": "lying_chest_lat", "attr_lng": "lying_chest_lng"})
gps_lying_head  = gps_lying_head.rename( columns={"attr_lat": "lying_head_lat",  "attr_lng": "lying_head_lng"})
gps_lying_shin  = gps_lying_shin.rename( columns={"attr_lat": "lying_shin_lat",  "attr_lng": "lying_shin_lng"})
gps_lying_ua    = gps_lying_ua.rename(   columns={"attr_lat": "lying_ua_lat",    "attr_lng": "lying_ua_lng"})
gps_lying_waist = gps_lying_waist.rename(columns={"attr_lat": "lying_waist_lat", "attr_lng": "lying_waist_lng"})
gps_lying_data  = gps_lying_chest.merge(gps_lying_head, how='outer').merge(gps_lying_shin, how='outer').merge(gps_lying_ua, how='outer').merge(gps_lying_waist, how='outer')
gps_lying_ua['activity'] = 'lying'

gps_running_chest = gps_running_chest.rename(columns={"attr_lat": "running_chest_lat", "attr_lng": "running_chest_lng"})
gps_running_head  = gps_running_head.rename( columns={"attr_lat": "running_head_lat",  "attr_lng": "running_head_lng"})
gps_running_shin  = gps_running_shin.rename( columns={"attr_lat": "running_shin_lat",  "attr_lng": "running_shin_lng"})
gps_running_ua    = gps_running_ua.rename(   columns={"attr_lat": "running_ua_lat",    "attr_lng": "running_ua_lng"})
gps_running_waist = gps_running_waist.rename(columns={"attr_lat": "running_waist_lat", "attr_lng": "running_waist_lng"})
gps_running_data  = gps_running_chest.merge(gps_running_head, how='outer').merge(gps_running_shin, how='outer').merge(gps_running_ua, how='outer').merge(gps_running_waist, how='outer')
gps_running_ua['activity'] = 'running'


gps_sitting_chest = gps_sitting_chest.rename(columns={"attr_lat": "sitting_chest_lat", "attr_lng": "sitting_chest_lng"})
gps_sitting_head  = gps_sitting_head.rename( columns={"attr_lat": "sitting_head_lat",  "attr_lng": "sitting_head_lng"})
gps_sitting_shin  = gps_sitting_shin.rename( columns={"attr_lat": "sitting_shin_lat",  "attr_lng": "sitting_shin_lng"})
gps_sitting_ua    = gps_sitting_ua.rename(   columns={"attr_lat": "sitting_ua_lat",    "attr_lng": "sitting_ua_lng"})
gps_sitting_waist = gps_sitting_waist.rename(columns={"attr_lat": "sitting_waist_lat", "attr_lng": "sitting_waist_lng"})
gps_sitting_data  = gps_sitting_chest.merge(gps_sitting_head, how='outer').merge(gps_sitting_shin, how='outer').merge(gps_sitting_ua, how='outer').merge(gps_sitting_waist, how='outer')
gps_sitting_ua['activity'] = 'sitting'

gps_walking_chest = gps_walking_chest.rename(columns={"attr_lat": "walking_chest_lat", "attr_lng": "walking_chest_lng"})
gps_walking_head  = gps_walking_head.rename( columns={"attr_lat": "walking_head_lat",  "attr_lng": "walking_head_lng"})
gps_walking_shin  = gps_walking_shin.rename( columns={"attr_lat": "walking_shin_lat",  "attr_lng": "walking_shin_lng"})
gps_walking_ua    = gps_walking_ua.rename(   columns={"attr_lat": "walking_ua_lat",    "attr_lng": "walking_ua_lng"})
gps_walking_waist = gps_walking_waist.rename(columns={"attr_lat": "walking_waist_lat", "attr_lng": "walking_waist_lng"})
gps_walking_data  = gps_walking_chest.merge(gps_walking_head, how='outer').merge(gps_walking_shin, how='outer').merge(gps_walking_ua, how='outer').merge(gps_walking_waist, how='outer')
gps_walking_ua['activity'] = 'walking'

gps_data = gps_climbingdown_ua.merge(gps_climbingup_ua, how='outer').merge(gps_jumping_ua, how='outer').merge(gps_lying_ua, how='outer').merge(gps_running_ua, how='outer').merge(gps_sitting_ua, how='outer').merge(gps_walking_ua, how='outer')
gps_data['attr_time'] = pd.to_datetime(gps_data['attr_time'])
gps_data.set_index('attr_time', inplace=True)

# interpolate missing values
acc_int = acc_data.interpolate(method='linear', axis=0).infer_objects(copy=False)
gps_int = gps_data.interpolate(method='linear', axis=0).infer_objects(copy=False)
aligned_df = pd.merge(acc_int, gps_int, how='outer')

# make activity the last column
activity = aligned_df.pop('activity')
aligned_df['activity'] = activity

# what does this 2 lines do?
aligned_df.reset_index(inplace=True)
aligned_df.rename(columns={'Timestamp': 'common_timestamp'}, inplace=True)



# split dataset 80/20
X_train = aligned_df.iloc[:151602,:-1]
Y_train = aligned_df.iloc[:151602,-1]
X_test = aligned_df.iloc[151602:,:-1]
Y_test = aligned_df.iloc[151602:,-1]

# Normalize X data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.fit_transform(X_test)
X_train_normalized = X_train.fillna(0)
X_test_normalized = X_test.fillna(0)
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train_normalized)
X_test_normalized = scaler.fit_transform(X_test_normalized)

# Convert Y labels to one-hot encoding
encoder = LabelEncoder()
Y_train_encoded = encoder.fit_transform(Y_train)
Y_train_onehot = to_categorical(Y_train_encoded)

# Reshape X_train to have the correct input shape (samples, time_steps, features)
X_train_reshaped = np.reshape(X_train_normalized, (X_train_normalized.shape[0], X_train_normalized.shape[1], 1))

# Shape of the input data
n_timesteps, n_features = X_train_reshaped.shape[1], X_train_reshaped.shape[2]
n_outputs = Y_train_onehot.shape[1]

# Create the model
model = Sequential()

# LSTM layer
model.add(LSTM(100, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.5))

# Dense layers
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))

# Compile the model with gradient clipping
optimizer = Adam(clipnorm=1.0)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Plot the model architecture
plot_model(model, show_shapes=True, show_layer_names=True)

# Train the model
train_epochs = 20
batch_size = 32
validation_split = 0.2

history = model.fit(X_train_reshaped, Y_train_onehot, epochs=train_epochs, batch_size=batch_size, verbose=True, validation_split=validation_split, shuffle=True)

# Plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training Data', 'Validation Data'], loc='upper left')
plt.show()


def display_results(testy,predictions,classNames):
    y_test=np.argmax(testy, axis=1)
    y_scores=np.argmax(predictions, axis=1)
    classes = len(classNames)
    cm = confusion_matrix(y_test, y_scores)
    print(" Confusion Matrix ")
    print(cm)
    print(" Classification Report ")
    print(classification_report(y_test, y_scores, target_names=classNames))
    con = np.zeros((classes,classes))
    for x in range(classes):
        for y in range(classes):
            con[x,y] = cm[x,y] #/np.sum(cm[x,:])

    plt.figure(figsize=(10,8))
    df = sns.heatmap(con, annot=True, fmt='g', cmap='Blues',xticklabels= classNames , yticklabels= classNames)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

classNames=["1","2","3","4","5","6"]

test_loss, test_accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
print("Accuracy of the CNN model", test_accuracy)
predictions=model.predict(X_test)

display_results(Y_test,predictions,classNames)

sample = X_test[0]
sample.shape
predictionForSample = model.predict(sample.reshape(1,sample.shape[0],sample.shape[1]))
samplePredictedClass = np.argmax(predictionForSample, axis=1)
sampleActualClass = np.argmax(Y_test, axis=1)
print("Actual class: ",sampleActualClass[0])
print("Predicted class: ",samplePredictedClass[0])