import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path


class ModelLSTM:

    # gestisce il modello LSTM per predizione di prezzi Bitcoin
    def __init__(self, forma_ingresso: tuple, nome_modello: str = 'bitcoin_modello.h5'):
        self.forma_ingresso = forma_ingresso
        self.nome_modello = nome_modello
        self.modello = None
        self.cronologia = None  # Contiene i dati di training (loss, mae, ecc)

    def costruisci_modello(self) -> None:

        # Costruzione architettura del modello LSTM
        print("🤖 Costruzione modello LSTM...\n")

        self.modello = Sequential([
            Input(shape=self.forma_ingresso),

            # primo layer LSTM
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),  # Disattiva casualmente il 20% dei neuroni

            # Layer denso finale per output singolo
            Dense(units=1)  # Output: un singolo prezzo
        ])

        self.modello.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        print("✅ Modello costruito\n")

    def addestra(self, x_train: np.ndarray, y_train: np.ndarray,
                 x_validazione: np.ndarray, y_validazione: np.ndarray,
                 epoche: int = 100, dimensione_batch: int = 32) -> bool:

        # addestramento modello LSTM sui dati di training
        try:
            print("⏳ Addestramento in corso...\n")

            # Early stopping: ferma se val_loss non migliora per 10 epoche consecutive
            interruzione_precoce = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            self.cronologia = self.modello.fit(
                x_train, y_train,
                validation_data=(x_validazione, y_validazione),
                epochs=epoche,
                batch_size=dimensione_batch,
                callbacks=[interruzione_precoce],
                verbose=1
            )

            num_epoche = len(self.cronologia.history['loss'])
            print(f"\n✅ Training completato - Epoche eseguite: {num_epoche}\n")
            return True

        except Exception as errore:
            print(f"❌ ERRORE nel training: {str(errore)}")
            return False

    def valida(self, x_validazione: np.ndarray, y_validazione: np.ndarray,
               gestore_dati: 'GestoreDati') -> dict:

        # valida il modello su dati non visti durante il training
        try:
            print("📊 Validazione modello...\n")

            # previsioni sui dati di validazione
            previsioni = self.modello.predict(x_validazione, verbose=0)

            # denormalizza i prezzi predetti e reali
            prezzi_predetti = gestore_dati.denormalizza_prezzi(previsioni.flatten())
            prezzi_reali = gestore_dati.denormalizza_prezzi(y_validazione.flatten())  # !!! MODIFICATO (Aggiunto .flatten()) !!!

            # metriche di errore
            mse = mean_squared_error(prezzi_reali, prezzi_predetti)
            mae = mean_absolute_error(prezzi_reali, prezzi_predetti)
            rmse = np.sqrt(mse)  # Root Mean Square Error

            metriche = {'RMSE': rmse, 'MAE': mae, 'MSE': mse}

            print(f"--- METRICHE DI VALIDAZIONE ---")
            print(f"RMSE: ${rmse:.2f}")  # Errore medio (sensibile ai grandi errori)
            print(f"MAE:  ${mae:.2f}\n")  # Errore medio assoluto

            return metriche

        except Exception as errore:
            print(f"❌ ERRORE nella validazione: {str(errore)}")
            return {}

    def predici_futuro(self, ultima_sequenza: np.ndarray, giorni: int = 7) -> np.ndarray:

        # previsioni per i prossimi 'giorni' giorni.
        try:
            print(f"🔮 Generazione previsioni ({giorni} giorni)...\n")

            batch_attuale = ultima_sequenza.copy()
            previsioni = []

            for _ in range(giorni):
                previsione = self.modello.predict(batch_attuale, verbose=0)
                previsioni.append(previsione[0, 0])

                nuova_riga = np.copy(batch_attuale[0, -1, :])

                nuova_riga[0] = previsione[0, 0]

                batch_attuale = np.append(
                    batch_attuale[:, 1:, :],
                    np.reshape(nuova_riga, (1, 1, nuova_riga.shape[0])),
                    axis=1
                )

            return np.array(previsioni)

        except Exception as errore:
            print(f"❌ ERRORE nella predizione: {str(errore)}")
            return np.array([])

    def salva_modello(self) -> bool:
        # salva il modello su disco per riutilizzi futuri
        try:
            self.modello.save(self.nome_modello)
            print(f"✅ Modello salvato: {self.nome_modello}\n")
            return True
        except Exception as errore:
            print(f"❌ ERRORE nel salvataggio: {str(errore)}")
            return False

    def carica_modello(self) -> bool:
        try:
            if not Path(self.nome_modello).exists():
                print(f"❌ File modello '{self.nome_modello}' non trovato!")
                return False

            self.modello = load_model(self.nome_modello)
            print(f"✅ Modello caricato: {self.nome_modello}\n")
            return True
        except Exception as errore:
            print(f"❌ ERRORE nel caricamento modello: {str(errore)}")
            return False