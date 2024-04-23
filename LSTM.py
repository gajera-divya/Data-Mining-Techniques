import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError 
from math import sqrt
import warnings
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, Dropout, BatchNormalization, GRU
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

warnings.filterwarnings("ignore")


df = pd.read_csv("mood_data_cleaned.csv")
df.drop(columns=['Unnamed: 0', 'index'], inplace=True)
df.head()

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

# Select the columns to be normalized
df['feature_mood'] = df['mood']
cols_to_normalize = df.columns.difference(['id', 'date', 'mood'])

# Apply the scaler to the selected columns
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# sort the dataframe by id and date
df.sort_values(['id', 'date'])

# Define the number of steps in each input sequence
n_steps = 3

# Prepare columns to be used as features
feature_columns = ['feature_mood', 'screen', 'activity', 'circumplex.arousal',
                   'circumplex.valence', 'call', 'sms', 'appCat.communication',
                   'appCat.professional', 'appCat.recreation', 'appCat.convenience',
                   'appCat.plain_usage']

user_ids = df['id'].unique()

df_test = df[(df['id'] == 'AS14.05') | (df['id'] == 'AS14.23')].copy()

df = df[~df['id'].isin(['AS14.05', 'AS14.23'])]

# Initialize lists to hold the input and output data
X_train, y_train = [], []
X_val, y_val = [], []
# Iterate over each unique ID
for uid in df['id'].unique():
    X, y = [], []
    user_df = df[df['id'] == uid]  # Extract data for the current ID
    
    # Check if there are enough rows to form at least one sequence
    if len(user_df) > n_steps:
        for i in range(len(user_df) - n_steps):
            # Extract the rows for the input sequence
            X.append(user_df[feature_columns].iloc[i:i + n_steps].values)

            # Get the mood of the day following the last day in the input sequence
            y.append(user_df['mood'].iloc[i + n_steps])
            
    split_index = int((1 - 0.1) * len(X))
    #print(X)
    X_train.extend(X[:split_index])
    X_val.extend(X[split_index:])
    y_train.extend(y[:split_index])
    y_val.extend(y[split_index:])

# Convert lists to numpy arrays for use with machine learning models
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)


# Print the shape of the input and output data
print(X_train.shape, y_train.shape)

# Split data into training and validation sets
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, len(feature_columns))),
    #Dropout(0.02),
    LSTM(32, activation='relu', return_sequences=True),
    #Dropout(0.02),
    LSTM(16),
    #Dropout(0.02),
    Dense(1)
])

# Define the learning rate
learning_rate = 0.0001  # You can change this value as needed

# Create the ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, min_lr=0.000001, verbose=1)

# Create the optimizer with the specified learning rate
optimizer = Adam(learning_rate=learning_rate)
#model.compile(optimizer=optimizer,loss='mse')
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=[MeanSquaredError(), MeanAbsoluteError()])

early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
# Train the model
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), callbacks=[reduce_lr, early_stopping])

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()


# Make predictions on the test set
def plot_user_predictions(user_id, model, df_test, feature_columns, n_steps):
    
    user_df = df_test[df_test['id'] == user_id]
    user_X, user_actual = [], []

    # for i in range(len(user_df) - n_steps):
    #     user_X.append(user_df[feature_columns].iloc[i:i + n_steps].values)
    #     user_actual.append(user_df['mood'].iloc[i + n_steps])

    # user_X = np.array(user_X)
    # user_actual = np.array(user_actual)
    
    #X, y = [], []
    # Iterate over each unique ID
    for uid in df_test['id'].unique():
        
        user_df = df_test[df_test['id'] == uid]  # Extract data for the current ID
        # Check if there are enough rows to form at least one sequence
        if len(user_df) > n_steps:
            for i in range(len(user_df) - n_steps):
                # Extract the rows for the input sequence
                user_X.append(user_df[feature_columns].iloc[i:i + n_steps].values)

                # Get the mood of the day following the last day in the input sequence
                user_actual.append(user_df['mood'].iloc[i + n_steps])


    # Convert lists to numpy arrays for use with machine learning models
    user_X = np.array(user_X)
    user_actual = np.array(user_actual)
    
        
    user_predictions = model.predict(user_X)
    rmse = (mean_squared_error(user_actual, user_predictions))
    r2 = r2_score(user_actual, user_predictions)
    mae = mean_absolute_error(user_actual, user_predictions)
    print(f'RMSE : {rmse}')
    print(f'R2 : {r2}')
    print(f'MAE : {mae}')
    print(user_actual[0])

    plt.figure(figsize=(10, 5))
    plt.plot(user_actual, label='Actual Mood')
    plt.plot(user_predictions, label='Predicted Mood', linestyle='--')
    plt.title(f'Mood Prediction vs Actual for User {user_id}')
    plt.ylabel('Mood')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.show() 

# Example usage
plot_user_predictions('AS14.05', model, df_test, feature_columns, n_steps)
train_metrics = model.evaluate(X_train, y_train, verbose=0)
val_metrics = model.evaluate(X_val, y_val, verbose=0)

print(f"Training Loss: {train_metrics[0]}, Training MSE: {train_metrics[1]}, Training MAE: {train_metrics[2]}")
print(f"Validation Loss: {val_metrics[0]}, Validation MSE: {val_metrics[1]}, Validation MAE: {val_metrics[2]}")
