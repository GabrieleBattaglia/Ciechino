"""
Le impostazioni di Ciechino: il volume, la lettura dei resoconti con la sintesi, l'uscita della sintesi, l'ultimo nome usato in classifica
e la forma d'onda del proiettile.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di impostazioni.py di MESS.
Si salvano in «Ciechino - Impostazioni.json», accanto al programma, con la scrittura sicura di
classifica.py: un file mancante o rovinato non ferma niente, valgono i predefiniti, e il file
rovinato si mette da parte invece di essere sovrascritto. valide riporta ogni chiave nel suo
intervallo, invece di buttarla, perché un volume a 120 scritto a mano vuol dire il massimo; e tiene
le chiavi che questo Ciechino non conosce, perché un Ciechino più nuovo, sulla stessa cartella, non
perda le sue.
La chiave uscita è l'uscita della sintesi di GBsintesi, non la scheda audio: il suono esce sempre
dal dispositivo che sceglie scegli_dispositivo_audio di GBUtils, e uscita dice soltanto chi legge i
resoconti, automatica oppure nvda, jaws o sapi5, la voce di Windows; un nome che GBsintesi non
conosce vale automatica.
La chiave proiettile è la forma d'onda del proiettile in volo, una di costanti.PROIETTILE_FORME (D17,
decisione di Gabriele del 19 novembre 2026); una forma che non sta lì vale quella di serie,
PROIETTILE_FORMA.
"""

import costanti
import percorsi
from classifica import leggi_dati, numero_valido, pulisci_nome, scrivi_dati
from lingua import _

# Il volume va da zero, il silenzio, a cento, il pieno: è la scala delle impostazioni, non una
# taratura del suono.
VOLUME_MINIMO = 0
VOLUME_MASSIMO = 100
# L'uscita della sintesi che lascia la scelta a GBsintesi: automatica, oppure nvda, jaws o sapi5.
USCITA_AUTOMATICA = "automatica"
PREDEFINITE = {"volume": costanti.VOLUME_PREDEFINITO, "sintesi": True, "uscita": USCITA_AUTOMATICA, "ultimo_nome": "", "proiettile": costanti.PROIETTILE_FORMA}
# L'avviso dell'ultimo caricamento, None se è andato tutto bene: carica restituisce soltanto le
# impostazioni, come vuole il contratto, e la finestra può dire qui perché sono tornate ai predefiniti.
ultimo_avviso = None


def _percorso(percorso):
    # Calcolato a ogni chiamata, mai all'importazione: da compilato e nelle prove la cartella cambia.
    return percorso or percorsi.percorso_dati(costanti.FILE_IMPOSTAZIONI)


def valide(dati):
    """Le impostazioni ricevute, sistemate: il volume riportato fra VOLUME_MINIMO e VOLUME_MASSIMO,
    la sintesi vera o falsa, l'uscita della sintesi un nome non vuoto, l'ultimo nome pulito come in
    classifica, il proiettile una delle forme di PROIETTILE_FORME; ciò che manca, ha il tipo
    sbagliato o è un numero oltre NUMERO_MASSIMO di classifica.py prende il predefinito, e le chiavi
    sconosciute restano."""
    risultato = dict(PREDEFINITE)
    if not isinstance(dati, dict):
        return risultato
    risultato.update({chiave: valore for chiave, valore in dati.items() if chiave not in PREDEFINITE})
    volume = dati.get("volume")
    if numero_valido(volume):
        risultato["volume"] = min(VOLUME_MASSIMO, max(VOLUME_MINIMO, round(volume)))
    if isinstance(dati.get("sintesi"), bool):
        risultato["sintesi"] = dati["sintesi"]
    uscita = dati.get("uscita")
    if isinstance(uscita, str) and uscita.strip():
        risultato["uscita"] = uscita.strip()
    if isinstance(dati.get("ultimo_nome"), str):
        risultato["ultimo_nome"] = pulisci_nome(dati["ultimo_nome"])
    proiettile = dati.get("proiettile")
    if isinstance(proiettile, str) and proiettile.strip() in costanti.PROIETTILE_FORME:
        risultato["proiettile"] = proiettile.strip()
    return risultato


def _controlla(grezzi):
    if not isinstance(grezzi, dict):
        raise ValueError(_("il file non contiene le impostazioni"))
    return valide(grezzi)


def carica(percorso=None):
    """Le impostazioni salvate, o i predefiniti se il file manca o non si legge. Non solleva mai;
    il perché di un ripiego resta in ultimo_avviso."""
    global ultimo_avviso
    dati, ultimo_avviso = leggi_dati(_percorso(percorso), _controlla)
    return dati if dati is not None else valide(None)


def salva(dati, percorso=None):
    """Salva le impostazioni, sistemate da valide; vero se è riuscito."""
    return scrivi_dati(valide(dati), _percorso(percorso))
