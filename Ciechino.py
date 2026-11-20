"""
Ciechino, il tiro a segno a orecchio: il punto d'avvio.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di Showdown.py di MESS, con l'ordine che descrive
istanza.py. La copia che si riapre, con --attendi e il numero di un processo, aspetta prima che quel
processo sia uscito. Poi, se un Ciechino è già aperto, questa copia lo porta in primo piano e si chiude
senza toccare niente, perché due copie insieme suonerebbero due partite sovrapposte e scriverebbero gli
stessi file; altrimenti si mette lei in ascolto delle copie che verranno. Poi si apre il registro degli
errori e dei crash accanto al programma, e un errore imprevisto, della finestra o di un altro filo, vi
lascia il suo resoconto completo. Poi la finestra e, a finestra comparsa, i suoni dell'avvio, gli Ascolti
della prima apertura (D23), che si aprono da soli finché non si è cominciata una partita, e, solo
dall'eseguibile compilato, il controllo dell'aggiornamento, la cui proposta aspetta gli Ascolti. Un errore che ferma Ciechino si scrive nel registro e si dice
in un messaggio, col suono dell'errore. All'uscita pulita l'ascolto si ferma e il registro si chiude,
così il file del crash se ne va.
Una copia lanciata mentre questa si sta già chiudendo, durante il congedo, non troverebbe nessuno da
portare in primo piano, e se ne andrebbe senza aprire niente. Perciò la finestra dice a main quando
comincia a chiudersi, e la richiesta che arriva da lì in poi si rimanda: all'uscita pulita main lancia,
con istanza.riapri, la copia nuova, che aspetta l'uscita di questa e poi si apre.
La scheda audio scelta da chi gioca (D20) la legge la finestra dalle impostazioni, prima di far nascere
il motore; il motore di riserva per il suono dell'errore fatale la rilegge da sé.
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


class Finestre(list):
    """
    La finestra di Ciechino, quando c'è, per il filo dell'ascolto dell'istanza unica, con i due segnali
    che quel filo e main si scambiano: in_chiusura, che la finestra accende quando comincia a chiudersi
    davvero, e rimandata, che si accende se un'altra copia ha chiesto di aprirsi proprio da allora.
    """

    def __init__(self):
        super().__init__()
        self.in_chiusura = threading.Event()
        self.rimandata = threading.Event()


def avvia_finestra(crash_precedente=None, finestre=None):
    """
    Apre la finestra fino all'uscita, e restituisce uno degli ESITI. crash_precedente è il resoconto del
    crash della volta prima, che la vista dice all'apertura; finestre, se data, riceve la finestra appena
    nasce, per l'ascolto dell'istanza unica, e se è una Finestre la finestra le accende in_chiusura
    quando comincia a chiudersi.
    """
    import wx

    import aggiornamento
    from gui import finestra as modulo_finestra

    # Una wx.App sola per processo: quella delle prove, se c'è già.
    app = wx.GetApp() or wx.App(False)
    finestra = modulo_finestra.FinestraPrincipale(crash_precedente=crash_precedente)
    if finestre is not None:
        finestre.append(finestra)
        in_chiusura = getattr(finestre, "in_chiusura", None)
        if in_chiusura is not None:
            finestra.al_congedo = in_chiusura.set
    _rete_per_gli_errori(finestra)
    # Massimizzata, come vuole Gabriele per tutte le finestre: dentro_area_utile, nel costruttore, ha già
    # dato la misura a cui torna se la si ripristina.
    finestra.Maximize()
    finestra.Show()
    wx.CallAfter(finestra.suoni_d_avvio)
    # Senza partite cominciate gli Ascolti si apriranno da soli, finiti i suoni dell'avvio (D23); la proposta
    # dell'aggiornamento li aspetta, perciò si preparano prima del controllo.
    wx.CallAfter(finestra.ascolti_alla_prima_apertura)
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


def _da_rimandare(finestre):
    """Se la finestra ha già cominciato a chiudersi: allora la richiesta della copia nuova si rimanda all'uscita, e si annota."""
    in_chiusura = getattr(finestre, "in_chiusura", None)
    if in_chiusura is None or not in_chiusura.is_set():
        return False
    finestre.rimandata.set()
    return True


def _in_primo_piano(finestre):
    """
    Nel filo dell'ascolto: un'altra copia di Ciechino chiede di aprirsi, e la finestra, se c'è già, torna
    in primo piano nel suo filo; se si sta chiudendo, la richiesta si rimanda all'uscita. Il filo non
    tocca la finestra: legge soltanto il segnale.
    """
    import wx

    if _da_rimandare(finestre):
        return
    if finestre:
        wx.CallAfter(_porta_in_primo_piano, finestre)


def _porta_in_primo_piano(finestre):
    """Nel filo della finestra: la finestra torna in primo piano, se nel frattempo non ha cominciato a chiudersi, o non è già distrutta."""
    if _da_rimandare(finestre):
        return
    with contextlib.suppress(RuntimeError):
        finestre[0].porta_in_primo_piano()


def _apri_la_copia_rimandata():
    """All'uscita pulita, la copia chiesta durante la chiusura: parte adesso, aspetta che questa sia uscita, e si apre."""
    import istanza
    import registro

    try:
        istanza.riapri()
    except OSError as errore:
        registro.scrivi(f"La copia di Ciechino chiesta durante la chiusura non è partita: {errore}.")
    else:
        registro.scrivi("Una copia di Ciechino ha chiesto di aprirsi durante la chiusura: si apre appena questa è uscita.")


def _suona_l_errore():
    """
    Il suono dell'errore, con un motore tutto suo: quello della finestra, a questo punto, può non esserci
    più. Il motore nasce con la scheda audio delle impostazioni (D20), così l'errore si sente dove si
    sentiva il gioco, per esempio nelle cuffie; se quella non c'è, sulla scheda automatica.
    """
    with contextlib.suppress(Exception):
        import impostazioni
        from motore_audio import Motore
        from suoni import Suoni

        motore = Motore(scelta=impostazioni.carica().get("scheda"))
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
    finestre = Finestre()
    ascolto = istanza.Ascolto(lambda: _in_primo_piano(finestre))
    # La pipe è già presa: un'altra copia è partita nello stesso istante, e ascolta lei.
    if not ascolto.attiva and istanza.manda():
        return 0
    import registro
    from costanti import VERSIONE

    crash_precedente = registro.avvia()
    registro.scrivi(f"Ciechino {VERSIONE} si avvia.")
    try:
        esito = avvia_finestra(crash_precedente, finestre)
        codice, riga = ESITI[esito]
        registro.scrivi(riga)
        # Solo dopo un'uscita pulita: chi si riapre per la lingua ha già la sua copia, e chi si aggiorna
        # lascia il posto allo script di sostituzione.
        if esito == APERTA and finestre.rimandata.is_set():
            _apri_la_copia_rimandata()
    except Exception:  # noqa: BLE001 - ultima rete del programma: lo scrive, lo dice e chiude
        _errore_fatale(*sys.exc_info())
        codice = 1
    finally:
        ascolto.ferma()
        registro.chiudi()
    return codice


if __name__ == "__main__":
    sys.exit(main())
