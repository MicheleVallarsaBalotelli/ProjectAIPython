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

    def carica_dati(self) -> bool:

        # Carica il file CSV e valida le colonne richieste.Ritorna True se successo altrimenti false
        try:
            if not Path(self.nome_file).exists():
                print(f"❌ ERRORE: File '{self.nome_file}' non trovato!")
                return False

            print("📊 Caricamento dati...\n")
            self.dataframe = pd.read_csv(self.nome_file)

            # verifica CSV contenga le colonne necessarie
            colonne_richieste = ['Date', 'Close', 'Volume', 'High', 'Low']
            if not all(col in self.dataframe.columns for col in colonne_richieste):
                print(f"❌ ERRORE: Il CSV deve contenere le colonne: {colonne_richieste}")
                return False

            # data in formato datetime e ordina
            self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'])
            self.dataframe = self.dataframe.sort_values('Date')
            self.dataframe = self.dataframe[colonne_richieste]

            print(f"✅ Dati caricati: {len(self.dataframe)} righe\n")
            return True

        except Exception as errore:
            print(f"❌ ERRORE nel caricamento: {str(errore)}")
            return False

    def calcola_indicatori(self) -> bool:

        # indicatori tecnici (MA20 - RSI)
        try:
            print("📈 Calcolo indicatori tecnici...\n")

            # MA20
            self.dataframe['MA20'] = self.dataframe['Close'].rolling(window=20).mean()

            # RSI
            variazione = self.dataframe['Close'].diff()

            # guadagni
            guadagni = (variazione.where(variazione > 0, 0)).rolling(window=14).mean()

            # perdite
            perdite = (-variazione.where(variazione < 0, 0)).rolling(window=14).mean()

            # calcolo RSI evitando divisione per zero
            rapporto_forza = np.where(perdite != 0, guadagni / perdite, 0)
            self.dataframe['RSI'] = 100 - (100 / (1 + rapporto_forza))

            # rimozione righe con valori NaN (primi 20-33 giorni)
            self.dataframe.dropna(inplace=True)

            print(f"✅ Indicatori calcolati: {len(self.dataframe)} righe rimanenti\n")
            return True

        except Exception as errore:
            print(f"❌ ERRORE nel calcolo indicatori: {str(errore)}")
            return False

    def normalizza_dati(self) -> bool:

        # normalizzazione dati range 0-1
        try:
            print("🔄 Normalizzazione dati...\n")

            # MinMaxScaler alle caratteristiche selezionate
            self.dati_normalizzati = self.normalizzatore.fit_transform(
                self.dataframe[self.caratteristiche].values
            )

            print(f"✅ Dati normalizzati (range: 0-1)\n")
            return True

        except Exception as errore:
            print(f"❌ ERRORE nella normalizzazione: {str(errore)}")
            return False

    def crea_sequenze(self, percentuale_test: float = 0.2) -> tuple:

        # sequenze di dati per training e validazione.
        try:
            print("📊 Preparazione dataset...\n")

            dati_ingresso = []
            dati_uscita = []

            for i in range(self.giorni_predizione, len(self.dati_normalizzati)):
                # ultimi 60 giorni come input
                dati_ingresso.append(self.dati_normalizzati[i - self.giorni_predizione:i, :])

                # prezzo di chiusura del giorno successivo è l'output (indice 0 è Close)
                dati_uscita.append(self.dati_normalizzati[i, 0])

            dati_ingresso = np.array(dati_ingresso)
            dati_uscita = np.array(dati_uscita)

            # Split 80% training, 20% validazione
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

    def denormalizza_prezzi(self, prezzi_normalizzati: np.ndarray) -> np.ndarray:

        # prezzi normalizzati (0-1) in valori reali in dollari

        dummy = np.zeros((len(prezzi_normalizzati), len(self.caratteristiche)))
        dummy[:, 0] = prezzi_normalizzati
        return self.normalizzatore.inverse_transform(dummy)[:, 0]