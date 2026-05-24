import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path


class ModelLSTM:
    # Gestisce il modello LSTM

    def __init__(self, forma_ingresso: tuple, nome_modello: str = 'bitcoin_modello.h5'):
        self.forma_ingresso = forma_ingresso
        self.nome_modello = nome_modello
        self.modello = None
        self.cronologia = None

    # modello LSTM a 3 layer (50 unità ciascuno) con Dropout 0.2, seguito da 2 layer Dense.
    def costruisci_modello(self) -> None:
        print("🤖 Costruzione modello LSTM...\n")

        self.modello = Sequential([
            Input(shape=self.forma_ingresso),

            LSTM(units=50, return_sequences=True),
            Dropout(0.2),

            LSTM(units=50, return_sequences=True),
            Dropout(0.2),

            LSTM(units=50),
            Dropout(0.2),

            Dense(units=25),
            Dense(units=1)
        ])

        self.modello.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        print("✅ Modello costruito\n")


    # Addestra il modello sui dati di training con validazione in tempo reale
    def addestra(self, x_train: np.ndarray, y_train: np.ndarray,
                 x_validazione: np.ndarray, y_validazione: np.ndarray,
                 epoche: int = 100, dimensione_batch: int = 32) -> bool:
        try:
            print("⏳ Addestramento in corso...\n")

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


    # Valida il modello su dati sconosciuti e calcola le metriche RMSE, MAE, MSE
    def valida(self, x_validazione: np.ndarray, y_validazione: np.ndarray,
               gestore_dati: 'GestoreDati') -> dict:
        try:
            print("📊 Validazione modello...\n")

            previsioni = self.modello.predict(x_validazione, verbose=0)

            prezzi_predetti = gestore_dati.denormalizza_prezzi(previsioni.flatten())
            prezzi_reali = gestore_dati.denormalizza_prezzi(y_validazione)

            mse = mean_squared_error(prezzi_reali, prezzi_predetti)
            mae = mean_absolute_error(prezzi_reali, prezzi_predetti)
            rmse = np.sqrt(mse)

            metriche = {'RMSE': rmse, 'MAE': mae, 'MSE': mse}

            print(f"--- METRICHE DI VALIDAZIONE ---")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAE:  ${mae:.2f}\n")

            return metriche

        except Exception as errore:
            print(f"❌ ERRORE nella validazione: {str(errore)}")
            return {}


    def predici_futuro(self, ultima_sequenza: np.ndarray, giorni: int = 7) -> np.ndarray:
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

    # Salva il modello addestrato su disco in formato .h5 per poterlo riutilizzare senza riaddestrare
    def salva_modello(self) -> bool:
        try:
            self.modello.save(self.nome_modello)
            print(f"✅ Modello salvato: {self.nome_modello}\n")
            return True
        except Exception as errore:
            print(f"❌ ERRORE nel salvataggio: {str(errore)}")
            return False

    # Carica modello precedentemente salvato da disco per evitare di riaddestrare il modello
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