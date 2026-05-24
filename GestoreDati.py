import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class GestoreDati:
    """Gestisce il caricamento, preparazione e normalizzazione dei dati Bitcoin"""

    def __init__(self, nome_file: str, giorni_predizione: int = 60):
        self.nome_file = nome_file
        self.giorni_predizione = giorni_predizione
        self.dataframe = None
        self.normalizzatore = MinMaxScaler(feature_range=(0, 1))
        self.dati_normalizzati = None
        self.caratteristiche = ['Close', 'Volume', 'MA20', 'RSI']


    # Carica il file CSV e valida che contenga le colonne richieste (Date, Close, Volume, High, Low).
    # Ritorna True se successo, False altrimenti.
    def carica_dati(self) -> bool:
        try:
            if not Path(self.nome_file).exists():
                print(f"❌ ERRORE: File '{self.nome_file}' non trovato!")
                return False

            print("📊 Caricamento dati...\n")
            self.dataframe = pd.read_csv(self.nome_file)

            colonne_richieste = ['Date', 'Close', 'Volume', 'High', 'Low']
            if not all(col in self.dataframe.columns for col in colonne_richieste):
                print(f"❌ ERRORE: Il CSV deve contenere le colonne: {colonne_richieste}")
                return False

            self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'])
            self.dataframe = self.dataframe.sort_values('Date')
            self.dataframe = self.dataframe[colonne_richieste]

            print(f"✅ Dati caricati: {len(self.dataframe)} righe\n")
            return True

        except Exception as errore:
            print(f"❌ ERRORE nel caricamento: {str(errore)}")
            return False

    # Calcola gli indicatori tecnici MA20 e RSI
    def calcola_indicatori(self) -> bool:
        try:
            print("📈 Calcolo indicatori tecnici...\n")

            self.dataframe['MA20'] = self.dataframe['Close'].rolling(window=20).mean()

            variazione = self.dataframe['Close'].diff()
            guadagni = (variazione.where(variazione > 0, 0)).rolling(window=14).mean()
            perdite = (-variazione.where(variazione < 0, 0)).rolling(window=14).mean()

            rapporto_forza = np.where(perdite != 0, guadagni / perdite, 0)
            self.dataframe['RSI'] = 100 - (100 / (1 + rapporto_forza))

            self.dataframe.dropna(inplace=True)

            print(f"✅ Indicatori calcolati: {len(self.dataframe)} righe rimanenti\n")
            return True

        except Exception as errore:
            print(f"❌ ERRORE nel calcolo indicatori: {str(errore)}")
            return False

    # Normalizza i dati nel range 0-1 usando MinMaxScaler per preparare i dati al training
    def normalizza_dati(self) -> bool:
        try:
            print("🔄 Normalizzazione dati...\n")

            self.dati_normalizzati = self.normalizzatore.fit_transform(
                self.dataframe[self.caratteristiche].values
            )

            print(f"✅ Dati normalizzati (range: 0-1)\n")
            return True

        except Exception as errore:
            print(f"❌ ERRORE nella normalizzazione: {str(errore)}")
            return False

    # Crea sequenze temporali di lunghezza 'giorni_predizione' per il training.
    def crea_sequenze(self, percentuale_test: float = 0.2) -> tuple:
        try:
            print("📊 Preparazione dataset...\n")

            dati_ingresso = []
            dati_uscita = []

            for i in range(self.giorni_predizione, len(self.dati_normalizzati)):
                dati_ingresso.append(self.dati_normalizzati[i - self.giorni_predizione:i, :])
                dati_uscita.append(self.dati_normalizzati[i, 0])

            dati_ingresso = np.array(dati_ingresso)
            dati_uscita = np.array(dati_uscita)

            indice_split = int(len(dati_ingresso) * (1 - percentuale_test))

            x_train = dati_ingresso[:indice_split]
            y_train = dati_uscita[:indice_split]
            x_validazione = dati_ingresso[indice_split:]
            y_validazione = dati_uscita[indice_split:]

            print(f"Training: {len(x_train)} | Validazione: {len(x_validazione)}\n")

            return (x_train, y_train), (x_validazione, y_validazione)

        except Exception as errore:
            print(f"❌ ERRORE nella preparazione dataset: {str(errore)}")
            return None, None

    # Converte prezzi normalizzati
    def denormalizza_prezzi(self, prezzi_normalizzati: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(prezzi_normalizzati), len(self.caratteristiche)))
        dummy[:, 0] = prezzi_normalizzati
        return self.normalizzatore.inverse_transform(dummy)[:, 0]