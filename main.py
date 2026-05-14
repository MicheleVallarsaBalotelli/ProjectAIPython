import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel('ERROR')


FILE_NAME = 'coin_Bitcoin.csv'
PREDICTION_DAYS = 60
EPOCHS = 100
BATCH_SIZE = 32


if not os.path.exists(FILE_NAME):
    print(f"ERRORE: Assicurati che il file {FILE_NAME} sia nella cartella del progetto!")
    exit()


print("📊 Caricamento dati...\n")
df = pd.read_csv(FILE_NAME)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df[['Date', 'Close', 'Volume', 'High', 'Low']]


print("📈 Calcolo indicatori tecnici...\n")
df['MA20'] = df['Close'].rolling(window=20).mean()
df['RSI'] = 100 - (100 / (1 + df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() /
                          -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean()))
df.dropna(inplace=True)


print("🔄 Normalizzazione dati...\n")
features = ['Close', 'Volume', 'MA20', 'RSI']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features].values)


print("📊 Preparazione dataset...\n")
x_train, y_train = [], []
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x, :])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

split_idx = int(len(x_train) * 0.8)
x_train_split = x_train[:split_idx]
y_train_split = y_train[:split_idx]
x_val = x_train[split_idx:]
y_val = y_train[split_idx:]

print(f"Training: {len(x_train_split)} | Validation: {len(x_val)}\n")


print("🤖 Costruzione modello LSTM...\n")
model = Sequential([
    Input(shape=(x_train_split.shape[1], x_train_split.shape[2])),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


print("⏳ Addestramento in corso...\n")
history = model.fit(
    x_train_split, y_train_split,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

print(f"\n✅ Training completato - Epoche: {len(history.history['loss'])}\n")


print("📊 Validazione modello...\n")
val_predictions = model.predict(x_val, verbose=0)

val_pred_scaled = np.zeros((len(val_predictions), len(features)))
val_pred_scaled[:, 0] = val_predictions[:, 0]
val_pred_actual = scaler.inverse_transform(val_pred_scaled)[:, 0]

val_actual_scaled = np.zeros((len(y_val), len(features)))
val_actual_scaled[:, 0] = y_val
val_actual = scaler.inverse_transform(val_actual_scaled)[:, 0]

mse = mean_squared_error(val_actual, val_pred_actual)
mae = mean_absolute_error(val_actual, val_pred_actual)
rmse = np.sqrt(mse)

print(f"--- METRICHE DI VALIDAZIONE ---")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE:  ${mae:.2f}\n")


print("🔮 Generazione previsioni (7 giorni)...\n")
current_batch = scaled_data[-PREDICTION_DAYS:]
current_batch = np.reshape(current_batch, (1, PREDICTION_DAYS, len(features)))

future_predictions = []

for i in range(7):
    prediction = model.predict(current_batch, verbose=0)
    current_pred = prediction[0, 0]
    future_predictions.append(current_pred)

    new_row = np.copy(current_batch[0, -1, :])
    new_row[0] = current_pred
    new_row = np.reshape(new_row, (1, 1, len(features)))
    current_batch = np.append(current_batch[:, 1:, :], new_row, axis=1)


final_prices = []
for p in future_predictions:
    dummy = np.zeros((1, len(features)))
    dummy[0, 0] = p
    final_prices.append(scaler.inverse_transform(dummy)[0, 0])


print("=" * 50)
print("💰 PREVISIONI BITCOIN - PROSSIMI 7 GIORNI 💰")
print("=" * 50)
for i, price in enumerate(final_prices, 1):
    print(f"Giorno {i}: ${price:.2f}")
print("=" * 50 + "\n")


print("📈 Generazione grafici...\n")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
axes[0].set_title('Loss durante l\'addestramento', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)


axes[1].plot(range(1, 8), final_prices, marker='o', linestyle='-', color='green',
             linewidth=3, markersize=10, label='Previsioni')
axes[1].fill_between(range(1, 8), final_prices, alpha=0.3, color='green')
axes[1].set_title('Previsione Bitcoin - Prossimi 7 Giorni', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Giorni', fontsize=11)
axes[1].set_ylabel('Prezzo ($)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

for i, price in enumerate(final_prices, 1):
    axes[1].text(i, price + 500, f'${price:.0f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Processo completato!")