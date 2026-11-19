"""
La lingua di Ciechino: i testi per l'utente passano da qui, per tradurli con gettext.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di lingua.py di MeTeOra 3.0.0: Ciechino nasce in
italiano, predisposto perché i collaboratori che lo vogliano lo traducano (scelta di Gabriele).

Ogni testo per l'utente passa da _ e, quando dipende da un numero, da ngettext o da plurale:
l'italiano è la lingua sorgente, e un catalogo in locales/<lingua>/LC_MESSAGES/messages.mo lo
traduce. I segnaposto sono fra graffe, con un nome, e si riempiono con format dopo la traduzione:
il traduttore può spostarli nella frase, ma non cambiarne il nome.

La lingua la sceglie polipo di GBUtils, all'importazione di questo modulo, prima che gli altri
costruiscano le loro tabelle di testi: quella scelta nelle impostazioni, che polipo salva in
selected_language.json accanto ai dati; al primo avvio quella di Windows, se qualcuno l'ha tradotta;
altrimenti l'italiano. Una lingua scelta vale dal riavvio. Senza la cartella locales, come oggi,
c'è soltanto l'italiano e polipo non si chiama nemmeno.
"""

import ctypes
import os

import percorsi

SORGENTE = "it"
DOMINIO = "messages"
CARTELLA = "locales"
# Il nome della lingua nella lingua stessa, per GetLocaleInfoEx: italiano, English.
LOCALE_SNATIVELANGUAGENAME = 0x04


def cartella_dei_cataloghi():
    return percorsi.percorso_risorsa(CARTELLA)


def _polipo(lingua=None):
    if not os.path.isdir(cartella_dei_cataloghi()):
        return SORGENTE, lambda testo: testo, lambda singolare, plurale_, n: singolare if n == 1 else plurale_
    from GBUtils import polipo

    return polipo(domain=DOMINIO, localedir=cartella_dei_cataloghi(), source_language=SORGENTE, config_path=percorsi.cartella_programma(),
                  interattivo=False, lingua=lingua, plurali=True)


CODICE, _gettext, _ngettext = _polipo()
# La lingua scelta nelle impostazioni durante questa sessione, che vale dal riavvio.
prossima = None


def _(testo):
    """Il testo nella lingua di Ciechino."""
    return _gettext(testo)


def ngettext(singolare, plurale, n):
    """Il testo con il numero n, nella forma giusta della lingua."""
    return _ngettext(singolare, plurale, n)


def plurale(n, singolare, plurale_, come=None):
    """Il caso più comune di ngettext, il numero con la sua parola:
    plurale(3, "{n} colpo", "{n} colpi") fa "3 colpi". Il numero decide la
    forma; come, se c'è, è il testo da scrivere al suo posto, per esempio
    "12.340" col punto delle migliaia."""
    return _ngettext(singolare, plurale_, n).format(n=n if come is None else come)


def lingue():
    """I codici delle lingue che Ciechino sa parlare: l'italiano e quelle che hanno il catalogo compilato."""
    cartella = cartella_dei_cataloghi()
    try:
        nomi = os.listdir(cartella)
    except OSError:
        nomi = []
    return sorted({SORGENTE} | {n for n in nomi if os.path.isfile(os.path.join(cartella, n, "LC_MESSAGES", f"{DOMINIO}.mo"))})


def nome(codice):
    """Il nome della lingua nella lingua stessa, come lo dice Windows. Senza risposta, il codice."""
    testo = ctypes.create_unicode_buffer(85)
    try:
        if ctypes.WinDLL("kernel32").GetLocaleInfoEx(codice, LOCALE_SNATIVELANGUAGENAME, testo, len(testo)):
            return testo.value
    except (OSError, AttributeError):
        pass
    return codice


def scegli(codice):
    """La voce Lingua delle impostazioni: polipo salva la scelta, che vale dal riavvio."""
    global prossima
    _polipo(codice)
    prossima = None if codice == CODICE else codice
