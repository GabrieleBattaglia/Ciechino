"""
I percorsi di Ciechino: dove il programma scrive e dove trova le risorse che legge soltanto.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di MeTeOra e di MESS.
Segue lo schema del parco software: sta nella radice del progetto, chiama le utilità di GBUtils
senza parametri e ricalcola la cartella a ogni chiamata, mai una volta sola all'importazione,
così vale sia da sorgente sia da eseguibile compilato.
Ciò che il programma scrive, cioè classifica, statistiche, impostazioni e registri, sta accanto al
programma: accanto all'eseguibile quando è compilato, accanto ai sorgenti altrimenti. Ciò che il
programma legge soltanto, cioè manuale, changelog e cataloghi delle lingue, da compilato viaggia
dentro il pacchetto e lì va cercato per primo.
"""

import os

from GBUtils import cartella_applicazione
from GBUtils import percorso_risorsa as _percorso_risorsa


def cartella_programma():
    """La cartella dell'eseguibile compilato, oppure quella dei sorgenti."""
    return cartella_applicazione()


def percorso_dati(nome_file):
    """Un file che Ciechino scrive: classifica, statistiche, impostazioni, registri."""
    return os.path.join(cartella_programma(), nome_file)


def percorso_risorsa(nome_file):
    """Un file che Ciechino legge soltanto: manuale, changelog, cataloghi."""
    return _percorso_risorsa(nome_file)
