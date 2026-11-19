"""
Ciechino, il tiro a segno a orecchio: il punto d'avvio.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di Showdown.py di MESS, con l'ordine che descrive
istanza.py. La copia che si riapre, con --attendi e il numero di un processo, aspetta prima che quel
processo sia uscito. Poi, se un Ciechino è già aperto, questa copia lo porta in primo piano e si chiude
senza toccare niente, perché due copie insieme suonerebbero due partite sovrapposte e scriverebbero gli
stessi file; altrimenti si mette lei in ascolto delle copie che verranno. Poi si apre il registro degli
errori e dei crash accanto al programma, e un errore imprevisto, della finestra o di un altro filo, vi
lascia il suo resoconto completo. Poi la finestra e, a finestra comparsa, solo dall'eseguibile
compilato, il controllo dell'aggiornamento. Un errore che ferma Ciechino si scrive nel registro e si dice
in un messaggio, col suono dell'errore. All'uscita pulita l'ascolto si ferma e il registro si chiude,
così il file del crash se ne va.
"""

import contextlib
import sys
import threading
import time
import traceback

# Come finisce avvia_finestra, e per ciascun esito il codice d'uscita e la riga del registro. Le righe
# del registro restano in italiano, come quelle di registro.py: le legge chi sviluppa.
APERTA = "aperta"
RIAPERTA = "riaperta"
AGGIORNATA = "aggiornata"
ESITI = {
    APERTA: (0, "Ciechino si chiude."),
    RIAPERTA: (0, "Ciechino si chiude per riaprirsi, con la lingua nuova."),
    AGGIORNATA: (0, "Ciechino si chiude per installare l'aggiornamento."),
}
# Quanto aspetta al più il suono dell'errore che ferma Ciechino, in secondi, prima del messaggio.
ATTESA_DEL_SUONO_D_ERRORE = 2.0


def avvia_finestra(crash_precedente=None, finestre=None):
    """
    Apre la finestra fino all'uscita, e restituisce uno degli ESITI. crash_precedente è il resoconto del
    crash della volta prima, che la vista dice all'apertura; finestre, se data, riceve la finestra appena
    nasce, per l'ascolto dell'istanza unica.
    """
    import wx

    import aggiornamento
    from gui import finestra as modulo_finestra

    # Una wx.App sola per processo: quella delle prove, se c'è già.
    app = wx.GetApp() or wx.App(False)
    finestra = modulo_finestra.FinestraPrincipale(crash_precedente=crash_precedente)
    if finestre is not None:
        finestre.append(finestra)
    _rete_per_gli_errori(finestra)
    # Massimizzata, come vuole Gabriele per tutte le finestre: dentro_area_utile, nel costruttore, ha già
    # dato la misura a cui torna se la si ripristina.
    finestra.Maximize()
    finestra.Show()
    wx.CallAfter(finestra.suoni_d_avvio)
    # A finestra comparsa: solo dall'eseguibile compilato, in un filo, e la finestra intanto risponde.
    wx.CallAfter(aggiornamento.controlla, finestra)
    app.MainLoop()
    congedo = modulo_finestra.USCITA
    # La finestra è distrutta: un attributo che non si legge più vale l'uscita di sempre.
    with contextlib.suppress(RuntimeError):
        congedo = finestra.congedo
    return {modulo_finestra.RIAPERTURA: RIAPERTA, modulo_finestra.PER_AGGIORNARE: AGGIORNATA}.get(congedo, APERTA)


def _rete_per_gli_errori(finestra):
    """
    Le eccezioni che sfuggono a un comando della finestra: wx le passa a sys.excepthook, e il comando
    muore in silenzio. Il resoconto completo va nel registro degli errori, e da sorgente anche sulla
    console; dal programma compilato no, perché lì lo stderr è già il registro e lo scriverebbe due
    volte. Poi la finestra fa sentire il suono dell'errore e mostra un messaggio. Le eccezioni degli
    altri fili, come quello dell'aggiornamento, vanno soltanto nel registro: non sono un comando di chi gioca.
    """
    import wx

    import registro

    compilato = getattr(sys, "frozen", False)
    precedente, precedente_dei_fili = sys.excepthook, threading.excepthook

    def gancio(tipo, valore, traccia):
        registro.problema(tipo, valore, traccia)
        if not compilato:
            precedente(tipo, valore, traccia)
        if finestra:
            wx.CallAfter(finestra.errore_imprevisto, tipo, valore)

    def negli_altri_fili(argomenti):
        registro.problema(argomenti.exc_type, argomenti.exc_value, argomenti.exc_traceback)
        if not compilato:
            precedente_dei_fili(argomenti)

    sys.excepthook = gancio
    threading.excepthook = negli_altri_fili


def _in_primo_piano(finestre):
    """Nel filo dell'ascolto: un'altra copia di Ciechino chiede di aprirsi, e la finestra, se c'è già, torna in primo piano nel suo filo."""
    import wx

    if finestre:
        wx.CallAfter(finestre[0].porta_in_primo_piano)


def _suona_l_errore():
    """Il suono dell'errore, con un motore tutto suo: quello della finestra, a questo punto, può non esserci più."""
    with contextlib.suppress(Exception):
        from motore_audio import Motore
        from suoni import Suoni

        motore = Motore()
        try:
            motore.avvia()
            voce = Suoni(motore).suona("errore")
            if voce is not None and not motore.muto:
                attesa = (voce.inizio + len(voce.buffer) - motore.campione_udibile()) / motore.fs
                time.sleep(min(max(attesa, 0.0), ATTESA_DEL_SUONO_D_ERRORE))
        finally:
            motore.chiudi()


def _errore_fatale(tipo, valore, traccia):
    """
    L'ultima rete del programma: l'errore che ferma Ciechino si scrive nel registro, e da sorgente anche
    sulla console; poi lo dicono il suono dell'errore e un messaggio, che dice dove sta il resoconto.
    Senza wx, o senza schermo, resta il registro.
    """
    import registro

    registro.problema(tipo, valore, traccia)
    registro.scrivi("Ciechino si ferma per un errore fatale.")
    if not getattr(sys, "frozen", False):
        traceback.print_exception(tipo, valore, traccia)
    _suona_l_errore()
    with contextlib.suppress(Exception):
        import wx

        from gui import dialoghi
        from lingua import _

        _app = wx.GetApp() or wx.App(False)
        # Traduttori: {tipo} è il nome dell'errore, in inglese, {valore} la sua descrizione, {resoconto} dove sta il resoconto.
        testo = _("Ciechino si è fermato per un errore imprevisto: {tipo}, {valore}. {resoconto}").format(
            tipo=tipo.__name__, valore=valore, resoconto=dialoghi.resoconto_degli_errori())
        dialoghi.messaggio(None, testo, _("Ciechino si è fermato"), wx.OK | wx.ICON_ERROR)


def main(argomenti=None):
    """
    L'avvio di Ciechino. Restituisce il codice d'uscita: zero se tutto è andato bene, anche quando
    questa copia lascia il posto a quella già aperta; uno se Ciechino si è fermato per un errore.
    argomenti sono quelli della riga di comando, senza il nome del programma: con --attendi e un numero
    di processo, la copia che si riapre aspetta prima che quel processo sia uscito.
    """
    import istanza

    argomenti = sys.argv[1:] if argomenti is None else argomenti
    da_attendere = istanza.da_attendere(argomenti)
    if da_attendere is not None:
        istanza.aspetta_l_uscita(da_attendere)
    # Prima di tutto, anche del registro: un Ciechino già aperto viene in primo piano, e questa copia non tocca niente.
    if istanza.manda():
        return 0
    finestre = []
    ascolto = istanza.Ascolto(lambda: _in_primo_piano(finestre))
    # La pipe è già presa: un'altra copia è partita nello stesso istante, e ascolta lei.
    if not ascolto.attiva and istanza.manda():
        return 0
    import registro
    from costanti import VERSIONE

    crash_precedente = registro.avvia()
    registro.scrivi(f"Ciechino {VERSIONE} si avvia.")
    try:
        codice, riga = ESITI[avvia_finestra(crash_precedente, finestre)]
        registro.scrivi(riga)
    except Exception:  # noqa: BLE001 - ultima rete del programma: lo scrive, lo dice e chiude
        _errore_fatale(*sys.exc_info())
        codice = 1
    finally:
        ascolto.ferma()
        registro.chiudi()
    return codice


if __name__ == "__main__":
    sys.exit(main())
