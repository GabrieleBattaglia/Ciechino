"""
Ciechino, l'aggiornamento automatico: all'avvio il controllo con l'auto updater di GBUtils.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di aggiornamento.py di MESS: chi installa la prima
release deve poter ricevere le seguenti. Lo conduce gestisci_aggiornamento di GBUtils, come in MESS,
MeTeOra, Tornello e Terminal Beast: controlla l'ultima release su GitHub e, se è più nuova, la propone
con le note della release; se si accetta la scarica, chiude Ciechino, sostituisce i file e lo riapre.
Solo dall'eseguibile compilato: da sorgente non si fa niente. Il controllo gira in un filo; la
proposta e gli avvisi si portano nel filo della finestra, l'unico che possa aprirli, e il filo aspetta
la risposta. Le poche frasi che GBUtils dice da sé passano da traduci: quelle che arrivano a chi gioca
diventano frasi di Ciechino, con gli accenti veri al posto degli apostrofi, e passano da lingua.py,
così si traducono con il resto del programma.
La finestra deve avere l'attributo chiusa e quattro metodi, che il filo chiama sempre nel filo della
finestra: proponi_l_aggiornamento(versione_attuale, versione_nuova, note, attesa=None), che risponde
vero o falso e allo scadere dei secondi di attesa si chiude rispondendo falso;
avvisa_dell_aggiornamento(testo), che torna quando chi gioca ha letto; avanzamento_dell_aggiornamento
(presi, totale), in byte; chiudi_per_aggiornare(). Se la finestra ha anche aspetta_i_dialoghi_d_avvio(),
il filo la chiama prima della proposta e di ogni avviso, e aspetta lì: così la proposta non si apre sopra
gli Ascolti della prima apertura (D23), ma dopo.
"""

import sys
import threading

from costanti import NOME, VERSIONE
from lingua import _

APP = NOME
API = "https://api.github.com/repos/GabrieleBattaglia/Ciechino/releases/latest"
# Le frasi di gestisci_aggiornamento che, con la proposta nella finestra, arrivano ancora a chi gioca:
# sono i tre esiti, e la finestra li riconosce dalla frase, con esito(), per dar loro il suono giusto.
FRASE_NON_PRONTO = _("ma il pacchetto non è ancora pronto: te la ripropongo a un prossimo avvio.")
FRASE_PRONTO = _("Aggiornamento scaricato: Ciechino si chiude per installarlo e si riapre da solo.")
FRASE_NON_RIUSCITO = _("Aggiornamento non riuscito: continui con questa versione, e te lo ripropongo al prossimo avvio.")
# Le frasi di GBUtils, scritte con l'apostrofo al posto dell'accento, come Ciechino le dice. Le altre,
# che con la proposta nella finestra GBUtils tace, sono qui lo stesso: se un giorno arrivassero, sarebbero
# già scritte bene e tradotte.
FRASI = {
    "Controllo aggiornamenti.": _("Controllo gli aggiornamenti."),
    "Hai gia' l'ultima versione,": _("Hai già l'ultima versione,"),
    "Controllo non riuscito, si prosegue.": _("Controllo degli aggiornamenti non riuscito, si prosegue."),
    "Disponibile la versione": _("Disponibile la versione"),
    "ma il pacchetto non e' ancora pronto.": FRASE_NON_PRONTO,
    "Scarico l'aggiornamento.": _("Scarico l'aggiornamento."),
    "Aggiornamento pronto, il programma si chiude per applicarlo.": FRASE_PRONTO,
    "Aggiornamento non riuscito, si prosegue con questa versione.": FRASE_NON_RIUSCITO,
}
PRONTO = "pronto"
NON_RIUSCITO = "non riuscito"
NON_PRONTO = "non pronto"
# La versione nuova arriva davanti alla frase del pacchetto non pronto, quindi le frasi si cercano in fondo al testo.
ESITI = ((FRASE_PRONTO, PRONTO), (FRASE_NON_RIUSCITO, NON_RIUSCITO), (FRASE_NON_PRONTO, NON_PRONTO))
# Le parole scritte con l'apostrofo nelle frasi di GBUtils che non stanno in FRASI. Il verbo e' va per
# ultimo: perche' finisce allo stesso modo, e prima diventerebbe perchè, con l'accento grave.
_ACCENTI = (("perche'", "perché"), ("gia'", "già"), ("Novita'", "Novità"), ("piu'", "più"), ("puo'", "può"), ("e' ", "è "))


def esito(testo):
    """Quale esito dice un testo che gestisci_aggiornamento manda alla finestra: PRONTO, NON_RIUSCITO, NON_PRONTO, o None se non è un esito."""
    return next((nome for frase, nome in ESITI if testo.endswith(frase)), None)


def traduci(testo):
    """Una frase di GBUtils com'è detta in Ciechino: quelle di FRASI riscritte, le altre con gli accenti al posto degli apostrofi."""
    if testo in FRASI:
        return FRASI[testo]
    for apostrofo, accento in _ACCENTI:
        testo = testo.replace(apostrofo, accento)
    return testo


def controlla(finestra, chiama_dopo=None):
    """
    All'avvio, solo dall'eseguibile compilato, il controllo in un filo. La finestra fa la proposta
    (proponi_l_aggiornamento), dice gli esiti (avvisa_dell_aggiornamento) e l'avanzamento dello
    scaricamento (avanzamento_dell_aggiornamento), e si chiude per farsi sostituire
    (chiudi_per_aggiornare). chiama_dopo porta una funzione nel filo della finestra: wx.CallAfter,
    se non si dice altro. Restituisce il filo, o None se il controllo non parte.
    """
    if not getattr(sys, "frozen", False):
        return None
    try:
        from GBUtils import gestisci_aggiornamento
    except ImportError:
        # Senza GBUtils Ciechino parte lo stesso, senza controllo.
        return None
    if chiama_dopo is None:
        import wx

        chiama_dopo = wx.CallAfter

    def nella_finestra(funzione, *argomenti):
        """Esegue funzione nel filo della finestra e ne aspetta il valore; None se intanto la finestra si è chiusa.
        Prima aspetta che i dialoghi dell'avvio della finestra, come gli Ascolti della prima apertura, si siano chiusi."""
        aspetta = getattr(finestra, "aspetta_i_dialoghi_d_avvio", None)
        if callable(aspetta):
            aspetta()
        esito, fatto = [], threading.Event()

        def lavoro():
            try:
                if finestra and not finestra.chiusa:
                    esito.append(funzione(*argomenti))
            finally:
                fatto.set()

        chiama_dopo(lavoro)
        fatto.wait()
        return esito[0] if esito else None

    def proponi(versione_attuale, versione_nuova, note, attesa=None):
        return bool(nella_finestra(finestra.proponi_l_aggiornamento, versione_attuale, versione_nuova, note, attesa))

    def avvisa(testo):
        nella_finestra(finestra.avvisa_dell_aggiornamento, testo)

    def avanzamento(presi, totale):
        chiama_dopo(finestra.avanzamento_dell_aggiornamento, presi, totale)

    def lavoro():
        if gestisci_aggiornamento(APP, VERSIONE, API, proponi=proponi, avvisa=avvisa, avanzamento=avanzamento, traduci=traduci):
            chiama_dopo(finestra.chiudi_per_aggiornare)

    filo = threading.Thread(target=lavoro, name=f"{NOME}, aggiornamento", daemon=True)
    filo.start()
    return filo
