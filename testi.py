"""
I testi di Ciechino: numeri, misure, date e durate all'italiana, le informazioni sul programma e le novità.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di testi.py di MESS e delle parole di statistiche.py di MeTeOra.
Sono funzioni pure, senza wx e senza audio: ricevono numeri e date e restituiscono frasi, così si
collaudano da sole e ogni modulo dice le stesse cose nello stesso modo. I numeri sono all'italiana,
la virgola per i decimali e il punto per le migliaia, le percentuali col simbolo, 64%, come vuole la
regola del parco del 23 ottobre 2026; ogni parola passa da lingua.py, perché i collaboratori possano
tradurla, e i numeri entrano nelle frasi già scritti, al posto del segnaposto.
"""

import datetime
import math
import re

import version
from lingua import _, ngettext, plurale

# Il repository pubblico, deciso da Gabriele il 19 novembre 2026 (D1 di docs/progetto.txt).
REPOSITORY = "https://github.com/GabrieleBattaglia/Ciechino"

MESI = (_("gennaio"), _("febbraio"), _("marzo"), _("aprile"), _("maggio"), _("giugno"), _("luglio"), _("agosto"), _("settembre"), _("ottobre"),
        _("novembre"), _("dicembre"))
# Nell'ordine di weekday(), dal lunedì.
GIORNI = (_("lunedì"), _("martedì"), _("mercoledì"), _("giovedì"), _("venerdì"), _("sabato"), _("domenica"))

# Il punteggio che chiude una frase: chi la riceve non deve aggiungerne un altro.
_FINE_DI_FRASE = (".", ":", "!", "?", ";")


# I numeri.

def numero(x, decimali=1):
    """Un numero all'italiana: 3,2; 12.340,5. Uno zero negativo si scrive senza il meno, perché
    -0,0 a orecchio non vuol dire niente."""
    valore = float(x)
    if not math.isfinite(valore):
        return str(valore)
    if round(valore, decimali) == 0:
        valore = 0.0
    testo = f"{valore:,.{decimali}f}"
    return testo.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def intero(n):
    """Un intero col punto delle migliaia: 12.340. Un numero con la virgola si arrotonda."""
    valore = float(n)
    if not math.isfinite(valore):
        return str(valore)
    return f"{round(valore):,}".replace(",", ".")


def percento(frazione, decimali=0):
    """Una frazione da 0 a 1 come percentuale col simbolo: 0,64 fa 64%."""
    # Traduttori: una percentuale; valore è già scritto, come 64 o 1,5.
    return _("{valore}%").format(valore=numero(float(frazione) * 100, decimali))


def metri(x, decimali=1):
    """Una distanza: 3,2 metri; il numero intero senza decimali, e 1 metro solo per l'uno esatto."""
    valore = float(x)
    if valore.is_integer():
        return plurale(int(valore), "{n} metro", "{n} metri", come=intero(valore))
    # Traduttori: una distanza con i decimali, come «3,2 metri».
    return _("{n} metri").format(n=numero(valore, decimali))


def gradi(x, decimali=1):
    """Un angolo: 12,4 gradi; il numero intero senza decimali, e 1 grado solo per l'uno esatto."""
    valore = float(x)
    if valore.is_integer():
        return plurale(int(valore), "{n} grado", "{n} gradi", come=intero(valore))
    # Traduttori: un angolo con i decimali, come «12,4 gradi».
    return _("{n} gradi").format(n=numero(valore, decimali))


def conta(n, singolare, plurale):
    """Il numero con la sua parola, nella forma giusta: conta(3, "{n} colpo", "{n} colpi") fa
    3 colpi. Se le forme non hanno il segnaposto {n}, il numero va davanti: conta(1, "colpo",
    "colpi") fa 1 colpo. Il numero si scrive col punto delle migliaia. Le forme passano da
    ngettext: per estrarle, pybabel deve conoscere la parola chiave conta:2,3."""
    testo = ngettext(singolare, plurale, n)
    if "{n}" in testo:
        return testo.format(n=intero(n))
    return f"{intero(n)} {testo}"


def elenco(parti):
    """Un elenco a parole: a; a e b; a, b e c. Le parti vuote non contano."""
    parti = [p for p in parti if p]
    if len(parti) <= 1:
        return "".join(parti)
    if len(parti) == 2:
        # Traduttori: un elenco di due voci, come «3 minuti e 12 secondi».
        return _("{prima} e {seconda}").format(prima=parti[0], seconda=parti[1])
    # Traduttori: un elenco di tre voci o più; prime sono tutte le voci meno l'ultima, già separate da virgole: «2 ore, 3 minuti e 5 secondi».
    return _("{prime} e {ultima}").format(prime=", ".join(parti[:-1]), ultima=parti[-1])


# Le date e le durate.

def leggi_data(valore):
    """Una data o un istante: un date, un datetime o un testo ISO; None se non si legge."""
    if isinstance(valore, datetime.date):
        return valore
    if isinstance(valore, str):
        try:
            return datetime.datetime.fromisoformat(valore.strip())
        except ValueError:
            return None
    return None


def _orario(quando):
    """L'ora del giorno con le preposizioni giuste: alle 18:03, all'1:16."""
    orario = f"{quando.hour}:{quando.minute:02d}"
    if quando.hour == 1:
        # Traduttori: il momento di un giorno quando l'ora è l'una: «all'1:16».
        return _("all'{orario}").format(orario=orario)
    # Traduttori: il momento di un giorno, con ore e minuti: «alle 18:03».
    return _("alle {orario}").format(orario=orario)


def data_lunga(data, ora=False):
    """Una data a parole, col giorno della settimana: giovedì 19 novembre 2026. Con ora vero, e
    un istante che l'ha, anche l'ora: giovedì 19 novembre 2026 alle 18:03."""
    quando = leggi_data(data)
    if quando is None:
        return _("una data che non si legge")
    # Traduttori: una data; giorno_della_settimana e mese sono nomi: «giovedì 19 novembre 2026».
    testo = _("{giorno_della_settimana} {giorno} {mese} {anno}").format(giorno_della_settimana=GIORNI[quando.weekday()], giorno=quando.day,
                                                                      mese=MESI[quando.month - 1], anno=quando.year)
    if ora and isinstance(quando, datetime.datetime):
        # Traduttori: una data con l'ora; alle_ore è come «alle 18:03».
        testo = _("{data} {alle_ore}").format(data=testo, alle_ore=_orario(quando))
    return testo


def eta_in_giorni(giorni):
    """L'età di un record: oggi, 1 giorno fa, 12 giorni fa. Un'età negativa, che viene da un
    orologio spostato indietro, vale oggi."""
    n = int(giorni or 0)
    if n <= 0:
        return _("oggi")
    return plurale(n, "{n} giorno fa", "{n} giorni fa", come=intero(n))


def durata(secondi):
    """Una durata a parole. Sotto l'ora minuti e secondi, come 3 minuti e 12 secondi; dall'ora in
    su giorni, ore e minuti, come 2 ore e 5 minuti, perché lì i secondi non dicono più niente."""
    valore = float(secondi or 0)
    if not math.isfinite(valore) or valore <= 0:
        return plurale(0, "{n} secondo", "{n} secondi")
    if valore < 1:
        return _("meno di un secondo")
    if round(valore) < 3600:
        minuti, resto = divmod(round(valore), 60)
        return elenco([plurale(minuti, "{n} minuto", "{n} minuti") if minuti else "", plurale(resto, "{n} secondo", "{n} secondi") if resto else ""])
    giorni, resto = divmod(round(valore / 60), 1440)
    ore, minuti = divmod(resto, 60)
    return elenco([plurale(giorni, "{n} giorno", "{n} giorni", come=intero(giorni)) if giorni else "",
                   plurale(ore, "{n} ora", "{n} ore") if ore else "", plurale(minuti, "{n} minuto", "{n} minuti") if minuti else ""])


def durata_in_cifre(secondi, vuoto=""):
    """Una durata in cifre, come un cronometro, da formatta_durata di GBUtils: 12:05, e dall'ora in
    su 01:02:03. vuoto è ciò che si scrive per una durata che non c'è, zero, negativa o non un
    numero: una durata che non si conosce non è una durata di zero."""
    from GBUtils import formatta_durata

    if isinstance(secondi, bool) or not isinstance(secondi, int | float) or not math.isfinite(secondi):
        return vuoto
    return formatta_durata(secondi, vuoto=vuoto)


# Le informazioni e le novità.

def informazioni():
    """Le informazioni sul programma, una per riga: nome, versione e data del rilascio, autori,
    che cos'è, licenza e repository: la finestra di F3; la vista della partita, all'apertura, ne prende le prime due."""
    righe = [
        # Traduttori: la prima riga delle informazioni; data è come «giovedì 19 novembre 2026».
        _("{nome}, il tiro a segno a orecchio, versione {versione}, rilasciata {data}.").format(nome=version.__app_name__, versione=version.__version__,
                                                                                            data=data_lunga(version.__date__)),
        _("Autori: {autori}.").format(autori=version.__author__),
        _("Ciechino è un gioco di tiro per chi non vede: i bersagli si sentono nello spazio, e il colpo si mira a orecchio con tre valori, "
          "direzione, alzo e potenza, mossi coi tasti finché li tieni giù."),
        _("Ciechino è software libero, con licenza GNU GPL versione 3: puoi usarlo, studiarlo, modificarlo e ridistribuirlo alle sue condizioni."),
        # Traduttori: l'indirizzo resta in fondo alla riga, senza punto dopo, perché si copi intero.
        _("Il codice è su GitHub, all'indirizzo {indirizzo}").format(indirizzo=REPOSITORY),
    ]
    return "\n".join(righe)


# Le righe orizzontali del markdown, fatte di soli trattini, asterischi o sottolineature, e i titoli
# sottolineati con gli uguali: sono separatori grafici, e la vista non li vuole.
_RIGA_ORIZZONTALE = re.compile(r"^(?:[-*_=]\s*){3,}$")
# Le definizioni dei collegamenti in fondo al changelog, come «[0.1.0]: https://...».
_DEFINIZIONE_DI_COLLEGAMENTO = re.compile(r"^\[[^\]]+\]:\s*\S+")
# Fra la versione e la data può stare il trattino, o la lineetta breve o lunga che certi editor
# mettono al suo posto: scritte coi codici, perché a vista si confondono.
_TITOLO_DI_VERSIONE = re.compile(r"^\[([^\]]+)\]\s*(?:[-\N{EN DASH}\N{EM DASH}]\s*(.+))?$")
_IMMAGINE = re.compile(r"!\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
_COLLEGAMENTO = re.compile(r"\[([^\]]+)\]\(([^)\s]+)[^)]*\)")
_GRASSETTO = re.compile(r"(\*\*|__)(\S(?:.*?\S)?)\1")
_CORSIVO = re.compile(r"(?<![\w*])([*_])(\S(?:.*?\S)?)\1(?![\w*])")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>|])")
_PUNTO_ELENCO = re.compile(r"^[-*+]\s+")


def _collegamento(trovato):
    testo, indirizzo = trovato.group(1).strip(), trovato.group(2).strip()
    if not indirizzo or testo == indirizzo:
        return testo or indirizzo
    # Traduttori: un collegamento del changelog, col testo e l'indirizzo fra parentesi.
    return _("{testo} ({indirizzo})").format(testo=testo, indirizzo=indirizzo)


def _togli_markdown(riga):
    """Una riga senza i simboli del markdown. Il codice fra accenti gravi resta com'è, senza gli
    accenti: i nomi come nuova_partita non devono perdere le sottolineature."""
    parti = riga.split("`")
    for i in range(0, len(parti), 2):
        parte = _IMMAGINE.sub(lambda t: t.group(1), parti[i])
        parte = _COLLEGAMENTO.sub(_collegamento, parte)
        parte = _GRASSETTO.sub(r"\2", parte)
        parte = _CORSIVO.sub(r"\2", parte)
        parti[i] = _ESCAPE.sub(r"\1", parte)
    return "".join(parti).strip()


def _titolo(testo):
    """Un titolo del changelog come frase: «[0.1.0] - 2026-11-19» diventa «Versione 0.1.0,
    giovedì 19 novembre 2026.»; gli altri titoli prendono il punto finale."""
    versione = _TITOLO_DI_VERSIONE.match(testo)
    if versione:
        numero_di_versione, quando = versione.group(1).strip(), (versione.group(2) or "").strip()
        if not quando:
            # Traduttori: il titolo di una versione senza data nel changelog.
            return _("Versione {versione}.").format(versione=numero_di_versione)
        data = leggi_data(quando[:10])
        # Traduttori: il titolo di una versione nel changelog; data è come «giovedì 19 novembre 2026».
        return _("Versione {versione}, {data}.").format(versione=numero_di_versione, data=data_lunga(data) if data else _togli_markdown(quando))
    testo = _togli_markdown(testo)
    if testo and not testo.endswith(_FINE_DI_FRASE):
        testo += "."
    return testo


def novita(testo_del_changelog):
    """Il CHANGELOG.md in forma leggibile, una riga per frase o voce: i titoli diventano frasi,
    senza cancelletti; i punti elenco, il grassetto, il corsivo, gli accenti gravi e le parentesi dei
    collegamenti se ne vanno; niente righe vuote e niente righe orizzontali."""
    righe = []
    for grezza in str(testo_del_changelog or "").splitlines():
        riga = grezza.strip()
        if not riga or _RIGA_ORIZZONTALE.match(riga) or _DEFINIZIONE_DI_COLLEGAMENTO.match(riga):
            continue
        livello = len(riga) - len(riga.lstrip("#"))
        if livello and riga[livello:livello + 1] in (" ", ""):
            riga = _titolo(riga[livello:].strip().rstrip("#").strip())
        else:
            riga = riga.lstrip(">").strip()
            riga = _togli_markdown(_PUNTO_ELENCO.sub("", riga))
        if riga:
            righe.append(riga)
    return "\n".join(righe)
