import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from GestoreDati import GestoreDati
from ModelloLSTM import ModelLSTM
from Visualizzatore import Visualizzatore

tf.get_logger().setLevel('ERROR')


def main():

    # CONFIGURAZIONE
    NOME_FILE = 'coin_Bitcoin.csv'
    GIORNI_PREDIZIONE = 60
    EPOCHE = 100
    DIMENSIONE_BATCH = 32
    GIORNI_FUTURI = 7

    print("=" * 70)
    print("🚀 PREVISIONE PREZZI BITCOIN CON LSTM")
    print("=" * 70 + "\n")

    # CARICAMENTO DATI
    print("📥 FASE 1: CARICAMENTO E PREPARAZIONE DATI\n")
    print("-" * 70 + "\n")

    gestore = GestoreDati(NOME_FILE, GIORNI_PREDIZIONE)

    if not gestore.carica_dati():
        print("❌ Impossibile continuare senza i dati!")
        return

    if not gestore.calcola_indicatori():
        print("❌ Impossibile continuare senza gli indicatori!")
        return

    if not gestore.normalizza_dati():
        print("❌ Impossibile continuare senza la normalizzazione!")
        return

    (x_train, y_train), (x_validazione, y_validazione) = gestore.crea_sequenze()

    if x_train is None:
        print("❌ Impossibile continuare senza le sequenze!")
        return

    # COSTRUZIONE MODELLO
    print("🛠️  FASE 2: COSTRUZIONE MODELLO LSTM (3 LAYER)\n")
    print("-" * 70 + "\n")

    modello = ModelLSTM(
        forma_ingresso=(x_train.shape[1], x_train.shape[2]),
        nome_modello='bitcoin_modello.h5'
    )
    modello.costruisci_modello()

    # ADDESTRAMENTO
    print("🎓 FASE 3: ADDESTRAMENTO MODELLO\n")
    print("-" * 70 + "\n")

    if not modello.addestra(x_train, y_train, x_validazione, y_validazione,
                            EPOCHE, DIMENSIONE_BATCH):
        print("❌ Impossibile continuare senza l'addestramento!")
        return

    #  VALIDAZIONE
    print("✅ FASE 4: VALIDAZIONE MODELLO\n")
    print("-" * 70 + "\n")

    metriche = modello.valida(x_validazione, y_validazione, gestore)

    # PREVISIONI FUTURE
    print("🔮 FASE 5: GENERAZIONE PREVISIONI\n")
    print("-" * 70 + "\n")

    ultima_sequenza = gestore.dati_normalizzati[-GIORNI_PREDIZIONE:]
    ultima_sequenza = np.reshape(ultima_sequenza,
                                 (1, GIORNI_PREDIZIONE, len(gestore.caratteristiche)))

    previsioni_normalizzate = modello.predici_futuro(ultima_sequenza, GIORNI_FUTURI)

    prezzi_finali = gestore.denormalizza_prezzi(previsioni_normalizzate)

    # RISULTATI
    print("💰 FASE 6: VISUALIZZAZIONE RISULTATI\n")
    print("-" * 70 + "\n")

    print("=" * 70)
    print(f"💰 PREVISIONI BITCOIN - PROSSIMI {GIORNI_FUTURI} GIORNI 💰")
    print("=" * 70)

    for i, prezzo in enumerate(prezzi_finali, 1):
        print(f"Giorno {i}: ${prezzo:.2f}")

    print("=" * 70 + "\n")

    # VISUALIZZAZIONE GRAFICI
    print("📊 FASE 7: GENERAZIONE GRAFICI\n")
    print("-" * 70 + "\n")

    print("📈 Mostrando grafico cronologia training...\n")
    Visualizzatore.mostra_cronologia_training(modello.cronologia)

    print("📈 Mostrando grafico previsioni future...\n")
    Visualizzatore.mostra_previsioni(
        prezzi_finali.tolist(),
        f"Previsione Bitcoin - Prossimi {GIORNI_FUTURI} Giorni"
    )

    # SALVATAGGIO
    print("💾 FASE 8: SALVATAGGIO MODELLO\n")
    print("-" * 70 + "\n")

    modello.salva_modello()

    print("=" * 70)
    print("✅ PROCESSO COMPLETATO CON SUCCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()