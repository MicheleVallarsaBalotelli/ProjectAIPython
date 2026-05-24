import matplotlib.pyplot as plt


class Visualizzatore:

    """Gestisce la creazione e visualizzazione dei grafici di training e previsioni"""
    # Mostra due grafici affiancati: Loss (training vs validation) e MAE durante l'addestramento del modello.

    @staticmethod
    def mostra_cronologia_training(cronologia, titolo: str = "Cronologia Training") -> None:
        try:
            fig, assi = plt.subplots(1, 2, figsize=(15, 5))

            epoche_effettive = range(1, len(cronologia.history['loss']) + 1)

            # Loss (errore di predizione)
            assi[0].plot(epoche_effettive, cronologia.history['loss'], label='Training Loss', linewidth=2, color='blue')
            assi[0].plot(epoche_effettive, cronologia.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
            assi[0].set_title('Loss durante l\'addestramento', fontsize=14, fontweight='bold')
            assi[0].set_xlabel('Epoch', fontsize=11)
            assi[0].set_ylabel('Loss', fontsize=11)
            assi[0].legend(fontsize=10)
            assi[0].grid(True, alpha=0.3)

            # MAE (Mean Absolute Error)
            assi[1].plot(epoche_effettive, cronologia.history['mae'], label='Training MAE', linewidth=2, color='green')
            assi[1].plot(epoche_effettive, cronologia.history['val_mae'], label='Validation MAE', linewidth=2, color='orange')
            assi[1].set_title('MAE durante l\'addestramento', fontsize=14, fontweight='bold')
            assi[1].set_xlabel('Epoch', fontsize=11)
            assi[1].set_ylabel('MAE', fontsize=11)
            assi[1].legend(fontsize=10)
            assi[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as errore:
            print(f"❌ ERRORE nel plotting: {str(errore)}")


    # Mostra un grafico a linea con i prezzi predetti dei prossimi giorni
    @staticmethod
    def mostra_previsioni(prezzi: list, titolo: str = "Previsioni Bitcoin") -> None:
        try:
            fig, asse = plt.subplots(figsize=(12, 6))

            giorni = range(1, len(prezzi) + 1)

            asse.plot(giorni, prezzi, marker='o', linestyle='-', color='green',
                      linewidth=3, markersize=10, label='Previsioni')

            asse.fill_between(giorni, prezzi, alpha=0.2, color='green')

            asse.set_title(titolo, fontsize=14, fontweight='bold')
            asse.set_xlabel('Giorni', fontsize=12)
            asse.set_ylabel('Prezzo ($)', fontsize=12)
            asse.grid(True, alpha=0.3)
            asse.legend(fontsize=11)

            for i, prezzo in enumerate(prezzi, 1):
                offset = (max(prezzi) * 0.01)
                asse.text(i, prezzo + offset, f'${prezzo:.0f}',
                          ha='center', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.show()

        except Exception as errore:
            print(f"❌ ERRORE nel plotting: {str(errore)}")