import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from GestoreDati import GestoreDati
from ModelloLSTM import ModelLSTM
from Visualizzatore import Visualizzatore

tf.get_logger().setLevel('ERROR')


def main():

    # CONFIGURAZIONE ---
    NOME_FILE = 'coin_Bitcoin.csv'
    GIORNI_PREDIZIONE = 60  # quanti giorni passati guarda per predire il prossimo
    EPOCHE = 100  # quante volte l'algoritmo vede il dataset
    DIMENSIONE_BATCH = 32  # quanti campioni elabora prima di aggiornare i pesi
    GIORNI_FUTURI = 7  # quanti giorni vogliamo predire

    print("=" * 70)
    print("🚀 PREVISIONE PREZZI BITCOIN CON LSTM")
    print("=" * 70 + "\n")

    # CARICAMENTO DATI ---
    print("📥 CARICAMENTO E PREPARAZIONE DATI\n")
    print("-" * 70 + "\n")

    # istanza gestore dati
    gestore = GestoreDati(NOME_FILE, GIORNI_PREDIZIONE)

    # carica il file CSV
    if not gestore.carica_dati():
        print("❌ Impossibile continuare senza i dati!")
        return

    # calcola indicatori tecnici (MA20 e RSI)
    if not gestore.calcola_indicatori():
        print("❌ Impossibile continuare senza gli indicatori!")
        return

    # normalizza i dati nel range 0-1
    if not gestore.normalizza_dati():
        print("❌ Impossibile continuare senza la normalizzazione!")
        return

    # crea sequenze per training e validazione
    (x_train, y_train), (x_validazione, y_validazione) = gestore.crea_sequenze()

    if x_train is None:
        print("❌ Impossibile continuare senza le sequenze!")
        return

    # COSTRUZIONE MODELLO ---
    print("🛠️ COSTRUZIONE MODELLO LSTM\n")
    print("-" * 70 + "\n")

    # istanza del modello LSTM
    # forma_ingresso = (numero_giorni, numero_caratteristiche) es --> (60, 4) = 60 giorni con 4 caratteristiche (Close, Volume, MA20, RSI)
    modello = ModelLSTM(
        forma_ingresso=(x_train.shape[1], x_train.shape[2]),
        nome_modello='bitcoin_modello.h5'
    )
    modello.costruisci_modello()

    #  ADDESTRAMENTO ---
    print("🎓 ADDESTRAMENTO MODELLO\n")
    print("-" * 70 + "\n")

    if not modello.addestra(x_train, y_train, x_validazione, y_validazione,
                            EPOCHE, DIMENSIONE_BATCH):
        print("❌ Impossibile continuare senza l'addestramento!")
        return

    # VALIDAZIONE ---
    print("✅ VALIDAZIONE MODELLO\n")
    print("-" * 70 + "\n")

    # validazione modello su dati che non ha mai visto
    metriche = modello.valida(x_validazione, y_validazione, gestore)

    # PREVISIONI FUTURE ---
    print("🔮 GENERAZIONE PREVISIONI\n")
    print("-" * 70 + "\n")

    # ultima sequenza (ultimi 60 giorni per predire il prossimo)
    ultima_sequenza = gestore.dati_normalizzati[-GIORNI_PREDIZIONE:]
    ultima_sequenza = np.reshape(ultima_sequenza,
                                 (1, GIORNI_PREDIZIONE, len(gestore.caratteristiche)))

    # genera previsioni per i prossimi 7 giorni
    previsioni_normalizzate = modello.predici_futuro(ultima_sequenza, GIORNI_FUTURI)

    # c i prezzi normalizzati in prezzi reali ($)
    prezzi_finali = gestore.denormalizza_prezzi(previsioni_normalizzate)

    # RISULTATI ---
    print("💰 VISUALIZZAZIONE RISULTATI\n")
    print("-" * 70 + "\n")

    # stampa previsioni
    print("=" * 70)
    print(f"💰 PREVISIONI BITCOIN - PROSSIMI {GIORNI_FUTURI} GIORNI 💰")
    print("=" * 70)

    for i, prezzo in enumerate(prezzi_finali, 1):
        print(f"Giorno {i}: ${prezzo:.2f}")

    print("=" * 70 + "\n")

    # VISUALIZZAZIONE GRAFICI ---
    print("📊 GENERAZIONE GRAFICI\n")
    print("-" * 70 + "\n")

    # mostra il grafico della cronologia di training
    print("📈 Mostrando grafico cronologia training...\n")
    Visualizzatore.mostra_cronologia_training(modello.cronologia)

    # mostra il grafico delle previsioni future
    print("📈 Mostrando grafico previsioni future...\n")
    Visualizzatore.mostra_previsioni(
        prezzi_finali.tolist(),
        f"Previsione Bitcoin - Prossimi {GIORNI_FUTURI} Giorni"
    )

    #  SALVATAGGIO ---
    print("💾 SALVATAGGIO MODELLO\n")
    print("-" * 70 + "\n")

    # Salvataggio modello addestrato per futuri utilizzi
    modello.salva_modello()

    print("=" * 70)
    print("✅ PROCESSO COMPLETATO CON SUCCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()