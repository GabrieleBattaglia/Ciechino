"""
La mira di Ciechino: dal campione che si sente quando si preme spazio al valore scelto con una scivolata.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
La funzione è una sola, la stessa per il gioco e per il banco della mira, così ciò che Gabriele
tara al banco vale identico in partita. Non sa niente di audio né di finestre: riceve il campione
udibile dal motore e il buffer da sintesi.scivolata, e fa il conto. Il modulo resta leggero, senza
numpy, perché lo importano anche le prove e i banchi che non suonano.
"""

from costanti import COMPENSAZIONE_DEL_TASTO, FS, MARGINE_DOPO_LA_SCIVOLATA

# I quattro esiti di valore_scelto, nell'ordine in cui il buffer li attraversa.
ESITI = ("pausa", "minimo", "scelto", "massimo")


def valore_scelto(campione_udibile, inizio_buffer, scivolata, compensazione=None):
    """Il valore scelto premendo spazio quando dalle casse esce campione_udibile, per una scivolata
    il cui buffer è partito al campione inizio_buffer del motore. Restituisce (valore, esito).
    Al campione udibile si tolgono compensazione secondi (COMPENSAZIONE_DEL_TASTO se è None), il
    ritardo fra il dito e il programma, perché conta ciò che si sentiva quando il dito è sceso.
    Prima del buffer, cioè nella pausa fra le barre, lo spazio non conta: (None, "pausa"). Nei
    bip e nel silenzio che li segue sceglie il minimo: (0.0, "minimo"). Nella scala sceglie la
    frazione percorsa, continua, senza gradini: (frazione, "scelto"). Dopo la fine della scala,
    ma prima che il battito se ne accorga, sceglie il massimo: (1.0, "massimo").
    scivolata è una sintesi.Scivolata, o qualunque oggetto con inizio_scivolata,
    campioni_scivolata e, se diversa da FS, la frequenza di campionamento fs."""
    if compensazione is None:
        compensazione = COMPENSAZIONE_DEL_TASTO
    fs = getattr(scivolata, "fs", FS)
    posizione = float(campione_udibile) - float(compensazione) * fs - float(inizio_buffer)
    if posizione < 0.0:
        return None, "pausa"
    inizio_scala = scivolata.inizio_scivolata
    if posizione < inizio_scala:
        return 0.0, "minimo"
    if posizione < inizio_scala + scivolata.campioni_scivolata:
        return (posizione - inizio_scala) / scivolata.campioni_scivolata, "scelto"
    return 1.0, "massimo"


def scaduta(campione_udibile, inizio_buffer, scivolata, margine=None):
    """Vero quando il campione udibile ha passato la fine della scala di margine secondi
    (MARGINE_DOPO_LA_SCIVOLATA se è None): la scivolata che nessuno ha fermato vale il massimo
    (D3), e la partita passa alla barra seguente. Qui non si toglie la compensazione del tasto,
    che riguarda il dito, non il tempo che passa."""
    if margine is None:
        margine = MARGINE_DOPO_LA_SCIVOLATA
    fs = getattr(scivolata, "fs", FS)
    fine = float(inizio_buffer) + scivolata.inizio_scivolata + scivolata.campioni_scivolata
    return float(campione_udibile) >= fine + float(margine) * fs
