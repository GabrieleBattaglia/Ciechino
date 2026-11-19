"""
Ciechino, utilità: prepara l'archivio della release per l'aggiornamento automatico.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di zip_maker.py di MESS. Comprime dist/Ciechino in
Ciechino.zip, coi file alla radice e il manifesto di _internal: tutto il mestiere sta in
crea_archivio_release di GBUtils, così la regola delle esclusioni è una sola per tutti i progetti
(prontuario, fase 7). Qui restano i nomi di Ciechino: i file che nascono accanto all'eseguibile
provandolo, cioè classifica, statistiche e impostazioni, con le loro copie, la lingua scelta, i
registri e il file dei registri per il PUB, perché l'aggiornamento non porti i dati di chi ha provato
il pacchetto. I .log, i .bak e i .tmp li toglie già GBUtils, ma non il registro ruotato, che finisce
in .log.1 e sta in FUORI come in Ciechino.iss.
Si lancia a mano, dopo PyInstaller: importarlo non fa niente.
"""

import sys

from costanti import NOME

FUORI = [
    # Classifica, statistiche, impostazioni, i registri, il file per il PUB e le copie illeggibili messe
    # da parte: tutti i file che Ciechino scrive cominciano col suo nome e un trattino.
    f"{NOME} - *",
    # La lingua scelta, che polipo di GBUtils salva accanto ai dati (lingua.py).
    "selected_language.json",
    "*.log.1",
]


def main():
    from GBUtils import crea_archivio_release

    try:
        crea_archivio_release(NOME, escludi=FUORI)
    except (FileNotFoundError, OSError) as e:
        print(f"Archivio non creato: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
