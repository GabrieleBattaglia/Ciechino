"""
I resoconti di Ciechino: i testi della partita, dal resoconto di ogni tiro ai messaggi del livello, dei
bonus, dei primati e della fine, e le righe della barra di stato.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Sono funzioni pure, senza wx e senza audio: ricevono il volo, i bersagli e i numeri della partita e
restituiscono un Messaggio, che la finestra mette in coda alla vista e, se l'impostazione lo vuole,
fa leggere alla sintesi. Il testo è per la vista, anche su più righe, una cosa per riga; l'annuncio
è breve, perché la sintesi lo legge mentre si torna ad ascoltare i bersagli, e un annuncio lungo
coprirebbe il loro suono. Ogni misura è in metri, gradi e secondi (D7), coi numeri all'italiana di
testi.py; ogni parola passa da lingua.py, e nessun testo contiene separatori grafici.
Le misure si arrotondano prima di scriverle: così 1,04 metri diventa «1 metro» e non «1,0 metri», e
uno scarto di tre centimetri a destra si dice «al centro» invece di «0,0 metri a destra».
Il resoconto di un tiro comincia col sommario di D21, con le misure dell'artigliere di volo.Misure
viste dal bersaglio, sempre nello stesso ordine, laterale, verticale, profondità e inclinazione: nei
primi livelli i numeri con un decimale e il segno, poi soltanto le parole, poi tace. Da D24 la riga coi
numeri si legge a blocchi sul display braille: nel primo, di BLOCCO_DEL_BRAILLE caratteri, i valori,
ciascuno con la sua iniziale e i due punti, dalla 0.11.8 senza spazio, come
«L:+1,9 V:+15,8 P:+24,7 I:-18,5 G:61,2%»; dal carattere seguente la legenda. La riga a parole tiene le
parole, e dalla 0.11.8 la precisione globale e la legenda vanno sempre sotto, su una riga loro. Nell'annuncio i valori o le parole
vengono dopo l'esito, senza la legenda. La riga del mancato di D6 dice le stesse misure, nello stesso
ordine, a parole.
La precisione globale di D26, quanto il proiettile si è avvicinato al bersaglio nell'intero cubo, sta
nel resoconto di ogni tiro che ha un bersaglio vicino: nel sommario coi numeri come G:, in quello a parole
sulla seconda riga, e dove il sommario tace nella riga dell'esito; l'annuncio la dice sempre. Non l'hanno
il proiettile perduto, che non ha volo, e il tiro senza bersagli vivi.
Nella palestra delle attese di D22 i messaggi di un tiro finiscono con la riga che dice che cosa farà
lo spazio, e la barra di stato lo ripete; fuori dalla palestra Ctrl+P ha il suo messaggio di rifiuto.
Da D25 il proiettile perduto, quando una barra della mira passa PASSAGGI_DELLA_BARRA volte senza lo
spazio, ha il suo resoconto, e la mira annullata con Esc il suo messaggio.
"""

import math
from dataclasses import dataclass, field

import testi
from bersagli import durata_del_bonus, quanti_bersagli, raggio_del_livello, velocita_del_livello
from costanti import (
    AIUTO_MANCATO_FINO_AL_LIVELLO,
    BLOCCO_DEL_BRAILLE,
    CENTRI_PER_LIVELLO,
    COLPI_BONUS,
    MOLTIPLICATORE_BONUS,
    PASSAGGI_DELLA_BARRA,
    SOMMARIO_NUMERI_FINO_AL_LIVELLO,
    SOMMARIO_PAROLE_FINO_AL_LIVELLO,
    SOMMARIO_SOGLIA_INCLINAZIONE,
    SOMMARIO_SOGLIA_LATERALE,
    SOMMARIO_SOGLIA_PROFONDITA,
    SOMMARIO_SOGLIA_VERTICALE,
)
from fisica import CANNONE
from lingua import _, ngettext, plurale
from statistiche import NOMI_DEI_PARAMETRI, NOMI_DELLE_FORME

# I tipi dei messaggi, come li vuole il contratto in docs/progetto.txt: la finestra può trattarli in
# modo diverso, per esempio aggiungere la classifica dopo quello di fine.
TIPI = ("info", "resoconto", "livello", "bonus", "fine", "avviso")

# Le sottofasi della mira come le dice la barra di stato: la pausa muta prima della scivolata, i tre
# bip col silenzio che li segue, e la scala.
NOMI_DELLE_SOTTOFASI = {
    # Traduttori: da qui, le parti di una barra della mira, nella barra di stato: «Mira: alzo, bip».
    "pausa": _("pausa"), "conto": _("bip"), "scivolata": _("scivolata"),
}

# I gradi della palestra di D21, dal più aiutato al meno: il sommario coi numeri, il sommario a sole
# parole, nessun sommario.
GRADI_DELLA_PALESTRA = ("numeri", "parole", "niente")

# Le tre barre della mira come le dice il resoconto del proiettile perduto (D25), con l'articolo.
NOMI_DELLE_BARRE = {
    # Traduttori: da qui, le barre della mira nel resoconto del proiettile perduto: «La barra dell'alzo è passata 5 volte senza lo Spazio».
    "direzione": _("la barra della direzione"), "alzo": _("la barra dell'alzo"), "potenza": _("la barra della potenza"),
}


@dataclass
class Messaggio:
    """Un messaggio della partita per la finestra. tipo è uno di TIPI; testo va nella vista, anche su
    più righe; annuncio è la frase breve per la sintesi, o vuoto; dati porta i numeri, per chi li vuole
    usare senza leggere il testo: il messaggio di fine vi porta il riepilogo della partita."""

    tipo: str
    testo: str
    annuncio: str = ""
    dati: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.tipo not in TIPI:
            raise ValueError(f"tipo di messaggio sconosciuto: {self.tipo!r}; i tipi sono {', '.join(TIPI)}")


# Le righe a blocchi per il display braille (D24).

def in_blocchi(primo, secondo="", larghezza=None):
    """Una riga in due blocchi per il display braille (D24): primo, poi spazi fino al carattere larghezza
    compreso, BLOCCO_DEL_BRAILLE se è None, così secondo comincia dal carattere seguente, nella prima cella
    del blocco dopo. Se primo non ci sta con almeno uno spazio dopo di sé, il primo blocco si allunga e
    secondo lo segue dopo uno spazio: Gabriele accetta che non sia sempre possibile. Senza secondo resta primo."""
    if not secondo:
        return primo
    larghezza = BLOCCO_DEL_BRAILLE if larghezza is None else larghezza
    if len(primo) < larghezza:
        return primo.ljust(larghezza) + secondo
    return f"{primo} {secondo}"


def _sta_nel_blocco(testo):
    """Vero se il testo sta nel primo blocco con almeno uno spazio dopo di sé, prima del secondo blocco."""
    return len(testo) < BLOCCO_DEL_BRAILLE


def _maiuscola(testo):
    return testo[:1].upper() + testo[1:]


# Le misure.

def _arrotondato(x, decimali=1):
    """Il valore arrotondato come lo si scriverà, senza lo zero negativo."""
    valore = round(float(x), decimali)
    return 0.0 if valore == 0 else valore


def metri(x, decimali=1):
    """Una distanza, arrotondata prima di scriverla: 3,2 metri; 1 metro; 12 metri."""
    return testi.metri(_arrotondato(x, decimali), decimali)


def gradi(x, decimali=1):
    """Un angolo, arrotondato prima di scriverlo: 12,4 gradi; 1 grado; 45 gradi."""
    return testi.gradi(_arrotondato(x, decimali), decimali)


def metri_al_secondo(x, decimali=1):
    """Una velocità: 27,5 metri al secondo; 1 metro al secondo; 10 metri al secondo."""
    valore = _arrotondato(x, decimali)
    if valore.is_integer():
        return plurale(int(valore), "{n} metro al secondo", "{n} metri al secondo", come=testi.intero(valore))
    # Traduttori: una velocità con i decimali, come «27,5 metri al secondo».
    return _("{n} metri al secondo").format(n=testi.numero(valore, decimali))


def secondi(x, decimali=1):
    """Una durata breve, col decimale: 2,8 secondi; 1 secondo; 14 secondi. Le durate lunghe, come
    quella di una partita, le dice testi.durata."""
    valore = _arrotondato(x, decimali)
    if valore.is_integer():
        return plurale(int(valore), "{n} secondo", "{n} secondi", come=testi.intero(valore))
    # Traduttori: una durata breve con i decimali, come «2,8 secondi».
    return _("{n} secondi").format(n=testi.numero(valore, decimali))


def _colpi(n):
    return plurale(n, "{n} colpo", "{n} colpi", come=testi.intero(n))


def _punti(n):
    return plurale(round(n), "{n} punto", "{n} punti", come=testi.intero(n))


def lato(x):
    """Dove sta un punto rispetto alla linea che va dal cannone dritta davanti: «3,1 metri a destra del
    centro», «2 metri a sinistra del centro», o «al centro» quando la misura si arrotonda a zero."""
    valore = _arrotondato(x)
    if valore > 0:
        # Traduttori: misura è come «3,1 metri».
        return _("{misura} a destra del centro").format(misura=metri(valore))
    if valore < 0:
        return _("{misura} a sinistra del centro").format(misura=metri(-valore))
    return _("al centro")


def direzione(azimut):
    """L'azimut di un tiro: «12,4 gradi a destra», «3 gradi a sinistra» o «al centro»."""
    valore = _arrotondato(azimut)
    if valore > 0:
        # Traduttori: angolo è come «12,4 gradi».
        return _("{angolo} a destra").format(angolo=gradi(valore))
    if valore < 0:
        return _("{angolo} a sinistra").format(angolo=gradi(-valore))
    return _("al centro")


def scarti(scarto):
    """Le parti dello scarto di un tiro a vuoto rispetto al centro del bersaglio, dato come
    (radiale, laterale, verticale), l'ordine di fisica.scomponi, e detto a parole nell'ordine del
    sommario di D21: «a destra di 0,8 metri» o «a sinistra di»; «sopra di 1,7 metri» o «sotto di»;
    «lungo di 1,2 metri» o «corto di». Le parti che mancano, None, e quelle che si arrotondano a zero
    non si dicono."""
    radiale, laterale, verticale = (0.0 if c is None else _arrotondato(c) for c in scarto)
    parti = []
    if laterale > 0:
        # Traduttori: da qui, le parti dello scarto di un tiro a vuoto, dette nell'ordine laterale, verticale, profondità; misura è come «0,8 metri».
        parti.append(_("a destra di {misura}").format(misura=metri(laterale)))
    elif laterale < 0:
        parti.append(_("a sinistra di {misura}").format(misura=metri(-laterale)))
    if verticale > 0:
        parti.append(_("sopra di {misura}").format(misura=metri(verticale)))
    elif verticale < 0:
        parti.append(_("sotto di {misura}").format(misura=metri(-verticale)))
    if radiale > 0:
        parti.append(_("lungo di {misura}").format(misura=metri(radiale)))
    elif radiale < 0:
        parti.append(_("corto di {misura}").format(misura=metri(-radiale)))
    return parti


def con_segno(x, decimali=1):
    """Un numero col segno davanti e sempre coi suoi decimali, all'italiana: +13,2; -2,4; +1,0; e 0,0,
    senza segno, quando si arrotonda a zero. Il meno è il trattino, che NVDA legge meno e il braille
    mostra in una cella sola, come il più."""
    valore = _arrotondato(x, decimali)
    testo = testi.numero(valore, decimali)
    return f"+{testo}" if valore > 0 else testo


def scarto_principale(scarto):
    """La parte più grande dello scarto (radiale, laterale, verticale), a parole, come «corto di 3,4
    metri»; le parti che mancano, None, non contano, e a parità vince la prima nell'ordine del
    sommario, laterale, verticale, profondità. Vuoto se non c'è."""
    if scarto is None:
        return ""
    presenti = [(indice, float(scarto[indice])) for indice in (1, 2, 0) if scarto[indice] is not None]
    if not presenti:
        return ""
    indice, valore = max(presenti, key=lambda presente: abs(presente[1]))
    sola = [None, None, None]
    sola[indice] = valore
    parti = scarti(sola)
    return parti[0] if parti else ""


def scarto_del_tiro(volo):
    """Lo scarto di un tiro a vuoto come lo dicono la riga del mancato di D6 e l'annuncio: da D21
    quello delle misure dell'artigliere, (profondità, laterale, verticale) con None per la misura che
    manca, così torna coi numeri del sommario; per un volo senza misure, come quelli costruiti a mano,
    lo scarto nel punto più vicino. None se non c'è nessuno dei due."""
    if volo.misure is not None:
        return volo.misure.scarto
    return volo.scarto


# I bersagli.

def nome_della_forma(forma):
    """La forma dell'LFO a parole, la stessa della pagina delle statistiche: «dente di sega»."""
    return NOMI_DELLE_FORME.get(forma, str(forma).replace("_", " "))


def descrizione_del_bersaglio(bersaglio):
    """Il tipo di un bersaglio: «forma dente di sega, raggio 3,6 metri, fermo»; il mobile dice anche
    la velocità, che vale punti; il bonus non dice né fermo né mobile, perché si muove sempre e il suo
    nome lo dice già."""
    forma = nome_della_forma(bersaglio.forma)
    raggio = metri(bersaglio.raggio)
    if bersaglio.bonus:
        # Traduttori: da qui, il tipo di un bersaglio; forma è come «dente di sega», raggio come «2,9 metri».
        return _("forma {forma}, raggio {raggio}").format(forma=forma, raggio=raggio)
    if bersaglio.mobile:
        # Traduttori: velocita è come «2,3 metri al secondo».
        return _("forma {forma}, raggio {raggio}, mobile a {velocita}").format(forma=forma, raggio=raggio, velocita=metri_al_secondo(bersaglio.velocita_scalare))
    return _("forma {forma}, raggio {raggio}, fermo").format(forma=forma, raggio=raggio)


def bersagli_del_livello(livello):
    """I bersagli normali di un livello, quanti, fermi o mobili, e di che raggio: «1 bersaglio fermo,
    raggio 5 metri», «2 bersagli mobili a 4,5 metri al secondo, raggio 1,9 metri»."""
    quanti = quanti_bersagli(livello)
    raggio = metri(raggio_del_livello(livello))
    velocita = velocita_del_livello(livello)
    if velocita > 0:
        chi = plurale(quanti, "{n} bersaglio mobile", "{n} bersagli mobili", come=testi.intero(quanti))
        # Traduttori: bersagli è come «2 bersagli mobili», velocita come «4,5 metri al secondo», raggio come «1,9 metri».
        return _("{bersagli} a {velocita}, raggio {raggio}").format(bersagli=chi, velocita=metri_al_secondo(velocita), raggio=raggio)
    chi = plurale(quanti, "{n} bersaglio fermo", "{n} bersagli fermi", come=testi.intero(quanti))
    # Traduttori: bersagli è come «1 bersaglio fermo», raggio come «5 metri».
    return _("{bersagli}, raggio {raggio}").format(bersagli=chi, raggio=raggio)


# Il sommario di un tiro (D21).

def grado_della_palestra(livello):
    """Il grado del sommario per un tiro sparato a quel livello (D21): «numeri» fino a
    SOMMARIO_NUMERI_FINO_AL_LIVELLO, «parole» fino a SOMMARIO_PAROLE_FINO_AL_LIVELLO, poi «niente».
    I primi livelli sono la palestra di Gabriele: l'aiuto si semplifica, poi sparisce."""
    if livello <= SOMMARIO_NUMERI_FINO_AL_LIVELLO:
        return "numeri"
    if livello <= SOMMARIO_PAROLE_FINO_AL_LIVELLO:
        return "parole"
    return "niente"


def _parola(valore, soglia, positiva, negativa, giusta):
    """La parola di un asse del sommario: positiva da soglia in su, negativa da meno soglia in giù,
    giusta in mezzo. Con la soglia a zero lo zero esatto resta giusto. Una misura che manca, None,
    non ha parola: None."""
    if valore is None:
        return None
    if valore > 0 and valore >= soglia:
        return positiva
    if valore < 0 and valore <= -soglia:
        return negativa
    return giusta


def parole_delle_misure(misure):
    """Le quattro parole del sommario a parole, per le volo.Misure, nell'ordine laterale, verticale,
    profondità e inclinazione: «a destra», «a sinistra» o «in linea»; «sopra», «sotto» o «all'altezza
    giusta»; «lungo», «corto» o «alla distanza giusta»; «in salita», «in discesa» o «in cima». Le
    soglie sotto le quali un asse va bene stanno in costanti.py (D21). Al posto della parola di una
    misura che manca c'è None.
    pybabel attacca un commento per i traduttori soltanto al primo testo della riga che lo segue: per
    questo ogni parola ha la sua riga e il suo commento."""
    return [
        _parola(misure.laterale, SOMMARIO_SOGLIA_LATERALE,
                # Traduttori: da qui, le dodici parole del sommario di un tiro (D21), che dicono dove il proiettile è passato rispetto al centro del bersaglio, e come andava in quel punto. Di lato: passato a destra del bersaglio.
                _("a destra"),
                # Traduttori: sommario di un tiro, di lato: passato a sinistra del bersaglio.
                _("a sinistra"),
                # Traduttori: sommario di un tiro, di lato: né a destra né a sinistra del bersaglio.
                _("in linea")),
        _parola(misure.verticale, SOMMARIO_SOGLIA_VERTICALE,
                # Traduttori: sommario di un tiro, in altezza: passato più in alto del centro del bersaglio.
                _("sopra"),
                # Traduttori: sommario di un tiro, in altezza: passato più in basso del centro del bersaglio.
                _("sotto"),
                # Traduttori: sommario di un tiro, in altezza: né sopra né sotto il bersaglio.
                _("all'altezza giusta")),
        _parola(misure.profondita, SOMMARIO_SOGLIA_PROFONDITA,
                # Traduttori: sommario di un tiro, in profondità: aggettivo del tiro, passato oltre il bersaglio visto dal cannone, non la preposizione.
                _("lungo"),
                # Traduttori: sommario di un tiro, in profondità: aggettivo del tiro, passato prima del bersaglio visto dal cannone.
                _("corto"),
                # Traduttori: sommario di un tiro, in profondità: né lungo né corto.
                _("alla distanza giusta")),
        _parola(misure.inclinazione, SOMMARIO_SOGLIA_INCLINAZIONE,
                # Traduttori: sommario di un tiro, l'inclinazione: in quel punto il proiettile saliva.
                _("in salita"),
                # Traduttori: sommario di un tiro, l'inclinazione: in quel punto il proiettile scendeva.
                _("in discesa"),
                # Traduttori: sommario di un tiro, l'inclinazione: in quel punto il proiettile era nel tratto quasi piano vicino alla cima della sua curva.
                _("in cima")),
    ]


@dataclass(frozen=True)
class Sommario:
    """Il sommario di un tiro in pezzi (D21, D24): primo, il primo blocco della riga, coi valori o le
    parole; secondo, il blocco che comincia dal carattere seguente, con la legenda; annuncio, ciò che ne
    dice l'annuncio del tiro, i valori senza la legenda. riga è la riga intera, a blocchi; con a_capo,
    il sommario a parole dalla 0.11.8, secondo va invece sotto, su una riga sua, e riga ne ha due."""

    primo: str
    secondo: str
    annuncio: str
    a_capo: bool = False

    @property
    def riga(self):
        if self.a_capo and self.secondo:
            return f"{self.primo}\n{self.secondo}"
        return in_blocchi(self.primo, self.secondo)


def frase_della_globale(globale):
    """La precisione globale di D26 in una frase, col decimale: «Precisione globale 61,2%.»."""
    # Traduttori: la precisione globale di un tiro (D26), quanto il proiettile si è avvicinato al bersaglio nell'intero cubo; percentuale è come «61,2%».
    return _("Precisione globale {percentuale}.").format(percentuale=testi.percento(globale, 1))


def _valori_coi_numeri(misure):
    """Il primo blocco del sommario coi numeri (D24): ogni misura con la sua iniziale e i due punti, senza
    spazio dalla 0.11.8, perché i valori stiano tutti nel primo blocco, G compresa, col segno e un
    decimale, nell'ordine laterale, verticale, profondità e inclinazione; quella che manca con la sua
    iniziale e la parola manca."""
    def scritto(valore):
        if valore is None:
            # Traduttori: al posto del numero di una misura che manca, nel sommario coi numeri: «L:manca»; la legenda dice perché.
            return _("manca")
        return con_segno(valore)

    return " ".join((
        # Traduttori: da qui, i valori del sommario coi numeri (D24), ognuno con la sua iniziale e i due punti, senza spazio, perché stiano nel primo blocco del display braille; valore è come «+1,9», «-18,5», «0,0» o «manca». L, il laterale, in metri, positivo a destra del bersaglio.
        _("L:{valore}").format(valore=scritto(misure.laterale)),
        # Traduttori: V, il verticale, in metri, positivo sopra il bersaglio.
        _("V:{valore}").format(valore=scritto(misure.verticale)),
        # Traduttori: P, la profondità, in metri, positiva se il tiro è lungo, oltre il bersaglio visto dal cannone.
        _("P:{valore}").format(valore=scritto(misure.profondita)),
        # Traduttori: I, l'inclinazione del proiettile, in gradi, positiva in salita.
        _("I:{valore}").format(valore=scritto(misure.inclinazione)),
    ))


def _legenda_dei_numeri(misure, con_la_globale):
    """Il secondo blocco del sommario coi numeri (D24): che cosa dicono le iniziali, e perché manca una misura che manca."""
    parti = [
        # Traduttori: la legenda del sommario coi numeri (D24), dal 41° carattere della riga: le iniziali dei valori del primo blocco, con le loro unità; le misure sono viste dal bersaglio.
        _("L laterale, V verticale, P profondità, in metri; I inclinazione, in gradi; dal bersaglio"),
    ]
    if con_la_globale:
        # Traduttori: la voce della legenda per G, la precisione globale (D26).
        parti.append(_("G precisione globale"))
    if misure.arrivo is None:
        # Traduttori: la voce della legenda per il tiro che cade a terra prima di arrivare alla distanza del bersaglio dal cannone, e così non ha laterale, verticale né inclinazione.
        parti.append(_("L, V e I mancano: il tiro cade prima della distanza del bersaglio"))
    if misure.ricaduta is None:
        # Traduttori: la voce della legenda per il tiro che, scendendo, non ripassa mai all'altezza del centro del bersaglio, e così non ha la profondità.
        parti.append(_("P manca: scendendo, il tiro resta sempre sotto il centro del bersaglio"))
    return "; ".join(parti) + "."


def _sommario_coi_numeri(misure, globale):
    """Il sommario al grado «numeri» (D24): i valori nel primo blocco, la legenda nel secondo. La precisione
    globale di D26, G:, segue i valori nel primo blocco, dove coi valori senza spazio dopo i due punti ci
    sta quasi sempre; nella riga eccezionale che non ci sta con almeno uno spazio dopo di sé apre il
    secondo. L'annuncio la dice sempre dopo i valori."""
    valori = _valori_coi_numeri(misure)
    legenda = _legenda_dei_numeri(misure, globale is not None)
    if globale is None:
        return Sommario(valori, legenda, valori)
    # Traduttori: la precisione globale nel sommario coi numeri (D26), con la sua iniziale G e i due punti, senza spazio come gli altri valori; percentuale è come «61,2%».
    sigla = _("G:{percentuale}").format(percentuale=testi.percento(globale, 1))
    con_la_globale = f"{valori} {sigla}"
    if _sta_nel_blocco(con_la_globale):
        return Sommario(con_la_globale, legenda, con_la_globale)
    return Sommario(valori, f"{sigla}. {legenda}", con_la_globale)


def _parole_del_sommario(misure):
    """Il primo blocco del sommario a parole: le parole di parole_delle_misure, con le frasi per le misure
    che mancano, con la maiuscola."""
    laterale, verticale, profondita, inclinazione = parole_delle_misure(misure)
    if misure.arrivo is not None and misure.ricaduta is not None:
        # Traduttori: il sommario a parole di un tiro nei livelli di mezzo (D21), come «a destra, sopra, lungo, in discesa.»; la maiuscola la mette il programma.
        testo = _("{laterale}, {verticale}, {profondita}, {inclinazione}.").format(laterale=laterale, verticale=verticale, profondita=profondita,
                                                                                    inclinazione=inclinazione)
    elif misure.arrivo is not None:
        # Traduttori: il sommario a parole di un tiro che, scendendo, resta sempre sotto il centro del bersaglio, come «a destra, sotto, profondità non misurabile, in salita.».
        testo = _("{laterale}, {verticale}, profondità non misurabile, {inclinazione}.").format(laterale=laterale, verticale=verticale, inclinazione=inclinazione)
    elif misure.ricaduta is not None:
        # Traduttori: il sommario a parole di un tiro che cade a terra prima di arrivare alla distanza del bersaglio dal cannone, come «cade prima della sua distanza, corto.».
        testo = _("cade prima della sua distanza, {profondita}.").format(profondita=profondita)
    else:
        # Traduttori: il sommario a parole di un tiro che cade prima di arrivare alla distanza del bersaglio dal cannone e, scendendo, resta sempre sotto il suo centro: non ha nessuna misura.
        testo = _("cade prima della sua distanza, profondità non misurabile.")
    return _maiuscola(testo)


def _sommario_a_parole(misure, globale):
    """Il sommario al grado «parole» (D24): le parole sulla prima riga; sulla seconda la precisione globale
    di D26, se c'è, e il punto di vista con l'ordine delle parole. Dalla 0.11.8 la seconda parte va sempre
    sotto, su una riga sua, per scelta di Gabriele del 20 novembre 2026: «Tagliamo la testa al toro e
    mettiamola sempre sotto, su una seconda riga.». L'annuncio ha le parole e la globale."""
    parole = _parole_del_sommario(misure)
    # Traduttori: la seconda riga del sommario a parole (D24), sotto le parole: il punto di vista e l'ordine delle quattro parole della riga sopra.
    legenda = _("Dal bersaglio: lato, altezza, profondità, inclinazione.")
    if globale is None:
        return Sommario(parole, legenda, parole, a_capo=True)
    frase = frase_della_globale(globale)
    return Sommario(parole, f"{frase} {legenda}", f"{parole} {frase}", a_capo=True)


def sommario_del_tiro(volo, grado="numeri"):
    """Il Sommario del tiro (D21, D24), coi suoi blocchi e il suo annuncio, o None al grado «niente» e
    senza misure, quando nessun bersaglio vivo c'era. Dice dove il proiettile è passato rispetto al
    centro del bersaglio, quello colpito o quello passato più vicino, con le misure dell'artigliere di
    volo.Misure, sempre in quest'ordine: il laterale, il verticale, la profondità e l'inclinazione. Al
    grado «numeri» coi valori e le loro iniziali, un decimale e il segno, e la precisione globale di D26;
    al grado «parole» con le parole, come «A destra, sopra, lungo, in discesa.», e a_capo vero, perché la
    precisione globale e la legenda vanno sulla riga sotto. Un grado fuori da GRADI_DELLA_PALESTRA solleva
    ValueError."""
    if grado not in GRADI_DELLA_PALESTRA:
        raise ValueError(f"grado della palestra sconosciuto: {grado!r}; i gradi sono {', '.join(GRADI_DELLA_PALESTRA)}")
    misure = volo.misure
    if misure is None or grado == "niente":
        return None
    if grado == "numeri":
        return _sommario_coi_numeri(misure, volo.precisione_globale)
    return _sommario_a_parole(misure, volo.precisione_globale)


def riga_del_sommario(volo, grado="numeri"):
    """Il sommario del tiro (D21), in testa al resoconto. Al grado «numeri» una riga a blocchi per il
    display braille (D24), «L:+8,8 V:+4,5 P:+3,9 I:-45,8 G:99,1%», poi spazi fino al carattere
    BLOCCO_DEL_BRAILLE compreso e la legenda; al grado «parole» due righe, «A destra, sopra, lungo, in
    discesa.» e sotto la precisione globale col punto di vista; al grado «niente», o se nessun bersaglio
    vivo c'era, vuoto.
    La misura che manca al grado «numeri» ha la sua iniziale e la parola manca, che la legenda spiega; a
    parole si dice: il tiro che cade prima di arrivare alla distanza del bersaglio «cade prima della sua
    distanza», e quello che, scendendo, resta sempre sotto il centro del bersaglio «profondità non
    misurabile». Un grado fuori da GRADI_DELLA_PALESTRA solleva ValueError."""
    sommario = sommario_del_tiro(volo, grado)
    return "" if sommario is None else sommario.riga


# Il resoconto di un tiro.

def _riga_dell_esito(numero, livello, volo, punti, totale, riserva, colpi_bonus, globale=None):
    """La riga dell'esito del resoconto: il numero del tiro, il livello e com'è andata. globale è la
    precisione globale di D26, da dire qui dove il sommario tace; None la tace."""
    tiro, livello = testi.intero(numero), testi.intero(livello)
    colpito = volo.colpito
    scritta = testi.percento(globale, 1) if globale is not None else None
    if colpito is not None:
        precisione = testi.percento(volo.precisione or 0.0)
        bersaglio = descrizione_del_bersaglio(colpito)
        if colpito.bonus:
            if scritta is not None:
                # Traduttori: il resoconto di un centro sul bonus dove il sommario tace, con la precisione globale (D26), globale come «99,4%»; il resto come nel resoconto senza.
                return _("Tiro {tiro}, livello {livello}: colpito il bonus, {bersaglio}; {punti}, precisione {precisione}, precisione globale {globale}, {colpi} in "
                         "più, totale {totale}.").format(tiro=tiro, livello=livello, bersaglio=bersaglio, punti=_punti(punti), precisione=precisione,
                                                         globale=scritta, colpi=_colpi(colpi_bonus), totale=_punti(totale))
            # Traduttori: il resoconto di un centro sul bonus; bersaglio è come «forma seno, raggio 2,9 metri», punti e totale come «1.150 punti», precisione come «64%», colpi come «5 colpi».
            return _("Tiro {tiro}, livello {livello}: colpito il bonus, {bersaglio}; {punti}, precisione {precisione}, {colpi} in più, totale {totale}.").format(
                tiro=tiro, livello=livello, bersaglio=bersaglio, punti=_punti(punti), precisione=precisione, colpi=_colpi(colpi_bonus), totale=_punti(totale))
        if scritta is not None:
            # Traduttori: il resoconto di un centro dove il sommario tace, con la precisione globale (D26), globale come «99,4%»; il resto come nel resoconto senza.
            return _("Tiro {tiro}, livello {livello}: colpito un bersaglio, {bersaglio}; {punti}, precisione {precisione}, precisione globale {globale}, "
                     "totale {totale}.").format(tiro=tiro, livello=livello, bersaglio=bersaglio, punti=_punti(punti), precisione=precisione, globale=scritta,
                                                totale=_punti(totale))
        # Traduttori: il resoconto di un centro; bersaglio è come «forma dente di sega, raggio 3,6 metri, fermo», punti e totale come «230 punti», precisione come «64%».
        return _("Tiro {tiro}, livello {livello}: colpito un bersaglio, {bersaglio}; {punti}, precisione {precisione}, totale {totale}.").format(
            tiro=tiro, livello=livello, bersaglio=bersaglio, punti=_punti(punti), precisione=precisione, totale=_punti(totale))
    if riserva > 0:
        restano = plurale(riserva, "resta {n} colpo", "restano {n} colpi", come=testi.intero(riserva))
        if scritta is not None:
            # Traduttori: il resoconto di un tiro a vuoto dove il sommario tace, con la precisione globale (D26); globale è come «93,2%», restano come «restano 8 colpi».
            return _("Tiro {tiro}, livello {livello}: a vuoto, precisione globale {globale}; nella riserva {restano}.").format(tiro=tiro, livello=livello,
                                                                                                                              globale=scritta, restano=restano)
        # Traduttori: il resoconto di un tiro a vuoto; restano è come «restano 8 colpi».
        return _("Tiro {tiro}, livello {livello}: a vuoto; nella riserva {restano}.").format(tiro=tiro, livello=livello, restano=restano)
    if scritta is not None:
        # Traduttori: il resoconto dell'ultimo tiro a vuoto dove il sommario tace, con la precisione globale (D26), come «93,2%».
        return _("Tiro {tiro}, livello {livello}: a vuoto, precisione globale {globale}; la riserva è vuota.").format(tiro=tiro, livello=livello, globale=scritta)
    return _("Tiro {tiro}, livello {livello}: a vuoto; la riserva è vuota.").format(tiro=tiro, livello=livello)


def riga_della_mira(traiettoria, valori):
    """La mira del tiro: la direzione in gradi a destra o a sinistra, o al centro, l'alzo in gradi e la
    potenza in metri al secondo, ognuno con la percentuale della sua scala (D7)."""
    quanto_direzione, quanto_alzo, quanto_potenza = (testi.percento(min(1.0, max(0.0, float(v)))) for v in valori)
    # Traduttori: la mira di un tiro; direzione è come «12,4 gradi a destra» o «al centro», alzo come «38,2 gradi», potenza come «27,5 metri al secondo», e ognuno ha dopo la percentuale della sua scala, come «64%».
    return _("Mira: direzione {direzione}, {quanto_direzione} della scala; alzo {alzo}, {quanto_alzo}; potenza {potenza}, {quanto_potenza}.").format(
        direzione=direzione(traiettoria.azimut), quanto_direzione=quanto_direzione, alzo=gradi(traiettoria.alzo), quanto_alzo=quanto_alzo,
        potenza=metri_al_secondo(traiettoria.velocita), quanto_potenza=quanto_potenza)


def riga_della_traiettoria(volo):
    """La traiettoria: quanto è salito il colpo, dove è finito, a che distanza dal cannone e da che
    parte, e quanto è durato il volo."""
    valori = {"altezza": metri(volo.altezza_massima), "distanza": metri(volo.gittata), "lato": lato(volo.laterale_finale), "durata": secondi(volo.durata)}
    if volo.colpito is not None:
        # Traduttori: la traiettoria di un centro; altezza e distanza sono come «23,4 metri», lato come «3,1 metri a destra del centro» o «al centro», durata come «2,8 secondi».
        return _("Traiettoria: altezza massima {altezza}; impatto a {distanza} dal cannone, {lato}, dopo {durata} di volo.").format(**valori)
    # Traduttori: la traiettoria di un tiro a vuoto, con le stesse parti di quella di un centro.
    return _("Traiettoria: altezza massima {altezza}; a terra a {distanza} dal cannone, {lato}, dopo {durata} di volo.").format(**valori)


def riga_del_bersaglio(punto):
    """Dove stava il bersaglio colpito, nel momento dell'impatto: la distanza dal cannone, misurata in
    orizzontale come quella della traiettoria, l'altezza e il lato."""
    x, y, z = (float(c) for c in punto)
    distanza = math.hypot(x - CANNONE[0], z - CANNONE[2])
    # Traduttori: distanza e altezza sono come «48,3 metri», lato come «5,2 metri a sinistra del centro» o «al centro».
    return _("Il bersaglio stava a {distanza} dal cannone, a {altezza} d'altezza, {lato}.").format(distanza=metri(distanza), altezza=metri(y), lato=lato(x - CANNONE[0]))


def riga_del_mancato(volo):
    """Di quanto e dove si è mancato il bersaglio più vicino (D6): la distanza dalla sua superficie, nel
    punto più vicino, poi lo scarto dal suo centro nell'ordine del sommario, a destra o a sinistra,
    sopra o sotto, lungo o corto, con le stesse misure dell'artigliere del sommario (D21), così i due
    dicono gli stessi numeri; le misure che mancano non si dicono. Vuoto se nessun bersaglio vivo è
    passato vicino al tiro."""
    scarto = scarto_del_tiro(volo)
    if volo.vicino is None or scarto is None:
        return ""
    distanza = metri(max(0.0, float(volo.distanza_dal_vicino or 0.0)))
    parti = scarti(scarto)
    if not parti:
        # Traduttori: distanza è come «2,3 metri».
        return _("Il bersaglio più vicino è stato mancato di {distanza}.").format(distanza=distanza)
    # Traduttori: distanza è come «2,3 metri»; scarti è un elenco come «a sinistra di 0,8 metri, sopra di 1,7 metri e lungo di 1,2 metri».
    return _("Il bersaglio più vicino è stato mancato di {distanza}: rispetto al suo centro il tiro è passato {scarti}.").format(distanza=distanza,
                                                                                                                              scarti=testi.elenco(parti))


def resoconto_del_tiro(numero, livello, volo, valori, punti=0, totale=0, riserva=0, colpi_bonus=None, aiuto=None, palestra=None):
    """Il messaggio «resoconto» dopo un tiro: il sommario (D21), la riga dell'esito, la mira, la
    traiettoria, e dopo un centro dove stava il bersaglio, dopo un tiro a vuoto di quanto si è
    mancato il più vicino.
    numero è il numero del tiro, livello quello in cui è stato sparato; valori i tre valori da 0 a 1
    della mira; punti quelli del colpo, totale il punteggio dopo il colpo, riserva i colpi rimasti;
    colpi_bonus i colpi dati dal bonus, COLPI_BONUS se è None. aiuto dice se spiegare il tiro a vuoto,
    e se è None vale fino al livello AIUTO_MANCATO_FINO_AL_LIVELLO (D6); serve anche all'annuncio,
    che allora aggiunge lo scarto principale. palestra è il grado del sommario, uno di
    GRADI_DELLA_PALESTRA, e se è None lo dà il livello del tiro con grado_della_palestra; il sommario,
    quando c'è, apre il testo: la prima riga, a blocchi, coi numeri (D24), le prime due a parole. La
    precisione globale di D26 sta nel sommario, e dove il sommario tace nella riga dell'esito. L'annuncio
    per la voce e il braille dice prima l'esito, poi i valori o le parole del sommario, senza la legenda,
    per scelta di Gabriele, con la precisione globale; col sommario coi numeri l'esito non ripete lo scarto principale, che il
    sommario dice già. Senza sommario l'annuncio è l'esito e la precisione globale. Nei dati c'è anche
    precisione_globale, None senza un bersaglio vivo vicino."""
    if aiuto is None:
        aiuto = livello <= AIUTO_MANCATO_FINO_AL_LIVELLO
    if palestra is None:
        palestra = grado_della_palestra(livello)
    if colpi_bonus is None:
        colpi_bonus = COLPI_BONUS
    colpito = volo.colpito
    globale = volo.precisione_globale
    sommario = sommario_del_tiro(volo, palestra)
    esito = _riga_dell_esito(numero, livello, volo, punti, totale, riserva, colpi_bonus, globale=globale if sommario is None else None)
    righe = [esito, riga_della_mira(volo.traiettoria, valori), riga_della_traiettoria(volo)]
    if sommario is not None:
        righe.insert(0, sommario.riga)
    if colpito is not None:
        righe.append(riga_del_bersaglio(colpito.posizione(volo.fine)))
        if colpito.bonus:
            # Traduttori: l'annuncio breve di un centro sul bonus; punti è come «1.150 punti».
            annuncio = _("Colpito il bonus, {punti}.").format(punti=_punti(punti))
        else:
            # Traduttori: l'annuncio breve di un centro; punti è come «230 punti».
            annuncio = _("Colpito, {punti}.").format(punti=_punti(punti))
    else:
        mancato = riga_del_mancato(volo) if aiuto else ""
        if mancato:
            righe.append(mancato)
        # Il sommario coi numeri dice già lo scarto: l'annuncio non lo ripete (D21).
        principale = scarto_principale(scarto_del_tiro(volo)) if aiuto and not (sommario is not None and palestra == "numeri") else ""
        if riserva <= 0:
            annuncio = _("A vuoto, riserva vuota.")
        elif principale:
            # Traduttori: l'annuncio breve di un tiro a vuoto nei livelli bassi; riserva è un numero, scarto come «corto di 3,4 metri».
            annuncio = _("A vuoto, riserva {riserva}, {scarto}.").format(riserva=testi.intero(riserva), scarto=principale)
        else:
            # Traduttori: l'annuncio breve di un tiro a vuoto; riserva è un numero.
            annuncio = _("A vuoto, riserva {riserva}.").format(riserva=testi.intero(riserva))
    if sommario is not None:
        # Traduttori: l'annuncio di un tiro col sommario (D21, D24): prima l'esito, annuncio, come «A vuoto, riserva 8.», poi il primo blocco del sommario, come «L:+1,9 V:+15,8 P:+24,7 I:-18,5 G:61,2%».
        annuncio = _("{annuncio} {sommario}").format(annuncio=annuncio, sommario=sommario.annuncio)
    elif globale is not None:
        # Traduttori: l'annuncio di un tiro senza sommario (D26): prima l'esito, annuncio, come «A vuoto, riserva 8.», poi la precisione globale, come «Precisione globale 93,2%.».
        annuncio = _("{annuncio} {globale}").format(annuncio=annuncio, globale=frase_della_globale(globale))
    dati = {"tiro": numero, "livello": livello, "colpito": colpito is not None, "bonus": bool(colpito is not None and colpito.bonus), "punti": punti,
            "totale": totale, "riserva": riserva, "precisione": volo.precisione, "precisione_globale": globale}
    return Messaggio("resoconto", "\n".join(righe), annuncio, dati)


def proiettile_perduto(numero, livello, parametro, passaggi, riserva, totale=0):
    """Il messaggio «resoconto» del proiettile perduto (D25): la barra del parametro è passata passaggi
    volte senza lo spazio, il colpo non è partito, e conta come un tiro a vuoto. numero è il numero del
    tiro, livello quello della mira, riserva i colpi rimasti dopo, totale il punteggio. Non c'è volo,
    quindi niente sommario, mira, traiettoria né precisione globale."""
    tiro, scritto_il_livello = testi.intero(numero), testi.intero(livello)
    if riserva > 0:
        restano = plurale(riserva, "resta {n} colpo", "restano {n} colpi", come=testi.intero(riserva))
        # Traduttori: il resoconto del proiettile perduto (D25), quando una barra della mira è passata cinque volte senza lo Spazio; restano è come «restano 8 colpi».
        esito = _("Tiro {tiro}, livello {livello}: proiettile perduto; nella riserva {restano}.").format(tiro=tiro, livello=scritto_il_livello, restano=restano)
        # Traduttori: l'annuncio breve del proiettile perduto; riserva è un numero.
        annuncio = _("Proiettile perduto, riserva {riserva}.").format(riserva=testi.intero(riserva))
    else:
        esito = _("Tiro {tiro}, livello {livello}: proiettile perduto; la riserva è vuota.").format(tiro=tiro, livello=scritto_il_livello)
        annuncio = _("Proiettile perduto, riserva vuota.")
    barra = NOMI_DELLE_BARRE.get(parametro, str(parametro).replace("_", " "))
    volte = plurale(passaggi, "{n} volta", "{n} volte", come=testi.intero(passaggi))
    # Traduttori: barra è come «la barra dell'alzo», con la maiuscola messa dal programma, volte come «5 volte».
    perche = _("{barra} è passata {volte} senza lo Spazio: il colpo non è partito, e il proiettile conta come un tiro a vuoto.").format(
        barra=_maiuscola(barra), volte=volte)
    dati = {"tiro": numero, "livello": livello, "colpito": False, "bonus": False, "punti": 0, "totale": totale, "riserva": riserva, "precisione": None,
            "precisione_globale": None, "perduto": True, "parametro": parametro, "passaggi": passaggi}
    return Messaggio("resoconto", "\n".join((esito, perche)), annuncio, dati)


# La partita: inizio, livelli, bonus, primati e fine.

def inizio_partita(livello, riserva):
    """Il messaggio d'inizio di una partita: livello, riserva, bersagli e come si mira."""
    righe = [
        # Traduttori: colpi è come «10 colpi».
        _("Partita nuova, livello {livello}: nella riserva {colpi}.").format(livello=testi.intero(livello), colpi=_colpi(riserva)),
        # Traduttori: bersagli è come «1 bersaglio fermo, raggio 5 metri».
        _("In volo: {bersagli}.").format(bersagli=bersagli_del_livello(livello)),
        _("Ascolta i bersagli e premi Spazio per mirare: tre scivolate, direzione, alzo e potenza, ognuna fermata con lo Spazio."),
    ]
    return Messaggio("info", "\n".join(righe), _("Partita nuova: Spazio per mirare."), {"livello": livello, "riserva": riserva})


def livello_nuovo(livello, colpi, riserva):
    """Il messaggio «livello» quando i centri chiudono un livello: il numero nuovo, i colpi in più, la
    riserva e i bersagli del livello nuovo."""
    righe = [
        # Traduttori: precedente è il numero del livello appena chiuso.
        _("Livello {livello}! Il livello {precedente} è superato.").format(livello=testi.intero(livello), precedente=testi.intero(livello - 1)),
        # Traduttori: colpi è come «10 colpi», riserva è un numero.
        _("{colpi} in più nella riserva, che ora ne ha {riserva}.").format(colpi=_colpi(colpi), riserva=testi.intero(riserva)),
        _("In volo: {bersagli}.").format(bersagli=bersagli_del_livello(livello)),
    ]
    # Traduttori: l'annuncio breve del livello nuovo; livello e riserva sono numeri.
    annuncio = _("Livello {livello}, riserva {riserva}.").format(livello=testi.intero(livello), riserva=testi.intero(riserva))
    return Messaggio("livello", "\n".join(righe), annuncio, {"livello": livello, "colpi": colpi, "riserva": riserva})


def bonus_apparso(bonus, livello):
    """Il messaggio «bonus» quando compare un bersaglio bonus: da che parte entra, com'è, quanto
    resta, quanto vale. Colpito vale MOLTIPLICATORE_BONUS volte un bersaglio normale del livello in
    cui è nato, con la stessa precisione (D13): partita.punti_del_centro fa proprio questo conto, e
    la frase lo dice con le stesse parole."""
    da_sinistra = float(bonus.velocita[0]) > 0
    da = _("da sinistra") if da_sinistra else _("da destra")
    righe = [
        # Traduttori: da è «da sinistra» o «da destra», descrizione come «forma seno, raggio 2,9 metri».
        _("È apparso un bersaglio bonus, che entra {da}: {descrizione}.").format(da=da, descrizione=descrizione_del_bersaglio(bonus)),
        # Traduttori: durata è come «14 secondi», volte come «5 volte», colpi come «5 colpi».
        _("Attraversa lo spazio dei bersagli in {durata}, poi esce; colpito vale {volte} i punti di un bersaglio di questo livello e dà {colpi} in più.").format(
            durata=secondi(durata_del_bonus(livello)), volte=plurale(MOLTIPLICATORE_BONUS, "{n} volta", "{n} volte"), colpi=_colpi(COLPI_BONUS)),
    ]
    # Traduttori: l'annuncio breve del bonus apparso; da è «da sinistra» o «da destra».
    return Messaggio("bonus", "\n".join(righe), _("Bonus {da}.").format(da=da), {"da_sinistra": da_sinistra, "durata": durata_del_bonus(livello)})


def bonus_uscito():
    """Il messaggio «bonus» quando il bonus esce senza essere stato colpito."""
    return Messaggio("bonus", _("Il bersaglio bonus è uscito senza essere colpito."), _("Bonus uscito."))


def ultimo_colpo():
    """L'avviso dell'ultimo colpo nella riserva."""
    return Messaggio("avviso", _("Nella riserva resta l'ultimo colpo: se va a vuoto, la partita finisce."), _("Ultimo colpo."))


def _primati():
    """Il nome di ogni primato e come se ne scrive il valore, con le parole della pagina delle statistiche."""
    def dal_centro(x):
        # Traduttori: misura è come «0,05 metri».
        return _("{misura} dal centro").format(misura=metri(x, 2))

    def col_decimale(x):
        return testi.percento(x, 1)

    return {
        # Traduttori: da qui, i nomi dei primati, dopo «Nuovo primato: ».
        "punteggio": (_("il punteggio più alto in una partita"), _punti),
        "livello": (_("il livello più alto raggiunto"), testi.intero),
        "serie": (_("la serie più lunga di centri consecutivi"), testi.intero),
        "precisione": (_("il centro più preciso"), testi.percento),
        # Traduttori: i due primati della precisione globale (D26), quella di un tiro e la media di una partita.
        "precisione_globale": (_("la precisione globale migliore in un tiro"), col_decimale),
        "precisione_globale_media": (_("la precisione globale media migliore in una partita"), col_decimale),
        "punti_in_un_tiro": (_("più punti in un tiro"), _punti),
        "centri_in_una_partita": (_("più centri in una partita"), testi.intero),
        "partita_piu_lunga": (_("la partita più lunga"), testi.durata),
        "vicino": (_("il tiro passato più vicino al centro di un bersaglio"), dal_centro),
        "mancato": (_("il tiro a vuoto passato più vicino al centro di un bersaglio"), dal_centro),
    }


def riga_del_primato(chiave, valore=None):
    """Un primato battuto: «Nuovo primato: il centro più preciso, 98%.». Un primato che questa tabella
    non conosce si dice con la sua chiave; senza valore, solo col nome."""
    nome, scrivi = _primati().get(chiave, (str(chiave).replace("_", " "), testi.numero))
    if valore is None:
        # Traduttori: nome è come «il centro più preciso».
        return _("Nuovo primato: {nome}.").format(nome=nome)
    # Traduttori: nome è come «il centro più preciso», valore come «98%».
    return _("Nuovo primato: {nome}, {valore}.").format(nome=nome, valore=scrivi(valore))


def primato_scritto(chiave, valore):
    """Il valore di un primato come lo scrive il messaggio, «98%» o «0,05 metri dal centro»: due valori
    che si scrivono uguali sono, a parole, lo stesso primato, e la partita non ne annuncia uno nuovo."""
    _nome, scrivi = _primati().get(chiave, (None, testi.numero))
    return scrivi(valore)


def righe_dei_primati(chiavi, valori=None):
    """Una riga per primato battuto, nell'ordine ricevuto; valori dà il valore di ogni chiave."""
    valori = valori or {}
    return [riga_del_primato(chiave, valori.get(chiave)) for chiave in chiavi]


def primati_battuti(chiavi, valori=None):
    """Il messaggio dei primati battuti da un tiro, o None se non ce ne sono."""
    chiavi = list(chiavi)
    if not chiavi:
        return None
    righe = righe_dei_primati(chiavi, valori)
    if len(chiavi) == 1:
        annuncio = righe[0]
    else:
        annuncio = plurale(len(chiavi), "{n} nuovo primato.", "{n} nuovi primati.")
    return Messaggio("info", "\n".join(righe), annuncio, {"primati": chiavi})


def sommario(riepilogo):
    """Il riassunto di una partita: «1.230 punti, livello 3, 12 centri su 20 tiri, 60%, in 4 minuti e
    12 secondi»."""
    tiri, centri = int(riepilogo.get("tiri", 0)), int(riepilogo.get("centri", 0))
    if tiri:
        # Traduttori: centri è come «12 centri», tiri come «20 tiri», percentuale come «60%».
        bilancio = _("{centri} su {tiri}, {percentuale}").format(centri=plurale(centri, "{n} centro", "{n} centri", come=testi.intero(centri)),
                                                                tiri=plurale(tiri, "{n} tiro", "{n} tiri", come=testi.intero(tiri)),
                                                                percentuale=testi.percento(centri / tiri))
    else:
        bilancio = _("nessun tiro")
    # Traduttori: punti è come «1.230 punti», tiri come «12 centri su 20 tiri, 60%», durata come «4 minuti e 12 secondi».
    return _("{punti}, livello {livello}, {tiri}, in {durata}").format(punti=_punti(riepilogo.get("punteggio", 0)), livello=testi.intero(riepilogo.get("livello", 0)),
                                                                      tiri=bilancio, durata=testi.durata(riepilogo.get("durata_s", 0)))


def fine_partita(riepilogo, primati=(), valori=None):
    """Il messaggio «fine» a riserva vuota: il riassunto della partita e i primati che ha battuto. In
    dati c'è il riepilogo, al quale la finestra aggiunge la classifica."""
    righe = [_("Partita finita: {sommario}.").format(sommario=sommario(riepilogo)), *righe_dei_primati(primati, valori)]
    # Traduttori: l'annuncio breve della fine; punti è come «1.230 punti».
    annuncio = _("Partita finita, {punti}.").format(punti=_punti(riepilogo.get("punteggio", 0)))
    return Messaggio("fine", "\n".join(righe), annuncio, dict(riepilogo))


def partita_abbandonata(riepilogo, primati=(), valori=None):
    """Il messaggio di una partita abbandonata, col suo riassunto e i primati che ha battuto."""
    righe = [_("Partita abbandonata: {sommario}.").format(sommario=sommario(riepilogo)), *righe_dei_primati(primati, valori)]
    return Messaggio("info", "\n".join(righe), _("Partita abbandonata."), dict(riepilogo))


# La pausa.

def in_pausa(nella_mira=False):
    """La pausa chiesta con Ctrl+P (D14). Nella mira dice anche che la barra interrotta, alla ripresa,
    ricomincia da capo, e che i valori già presi restano."""
    if nella_mira:
        return Messaggio("info", _("Partita in pausa: la mira è sospesa e i bersagli tacciono. Alla ripresa la barra in corso ricomincia da capo, "
                                   "e i valori già presi restano. Ctrl+P per riprendere."), _("Pausa."))
    return Messaggio("info", _("Partita in pausa: il tempo è fermo e i bersagli tacciono. Ctrl+P per riprendere."), _("Pausa."))


def ripresa():
    return Messaggio("info", _("La partita riprende."), _("Si riprende."))


def nessuna_partita():
    return Messaggio("avviso", _("Nessuna partita in corso: Spazio per cominciare."), _("Nessuna partita in corso."))


def pausa_solo_in_palestra(fino_al_livello):
    """Ctrl+P fuori dalla palestra delle attese (D22), quando la pausa non c'è: la pausa con Ctrl+P si
    prende soltanto fino al livello fino_al_livello, mentre quella automatica resta sempre."""
    # Traduttori: livello è un numero, l'ultimo livello della palestra, dove dopo ogni tiro la partita aspetta lo Spazio.
    testo = _("La pausa con Ctrl+P c'è soltanto nei livelli della palestra, fino al livello {livello}.").format(livello=testi.intero(fino_al_livello))
    return Messaggio("avviso", testo, _("Pausa solo in palestra."), {"fino_al_livello": fino_al_livello})


# La mira annullata con Esc (D25).

def mira_annullata(riserva, in_pausa=False):
    """Esc nella mira (D25): la mira è annullata, il proiettile resta nella riserva, che ha riserva colpi,
    e i bersagli tornano a cantare; in pausa la partita resta ferma, fino a Ctrl+P."""
    colpi = _colpi(riserva)
    if in_pausa:
        # Traduttori: Esc nella mira messa in pausa con Ctrl+P (D25); colpi è come «10 colpi».
        testo = _("Mira annullata con Esc: il proiettile resta nella riserva, che ha {colpi}, e la partita resta in pausa. Ctrl+P per riprendere.").format(
            colpi=colpi)
    else:
        # Traduttori: Esc nella mira (D25): il colpo non parte, e si torna ad ascoltare; colpi è come «10 colpi».
        testo = _("Mira annullata con Esc: il proiettile resta nella riserva, che ha {colpi}, e i bersagli tornano a cantare. Spazio per mirare di "
                  "nuovo.").format(colpi=colpi)
    return Messaggio("info", testo, _("Mira annullata."), {"riserva": riserva})


# La palestra delle attese (D22).

def attesa_della_palestra(nasceranno=0):
    """L'ultimo messaggio di un tiro nella palestra delle attese (D22): la partita aspetta lo spazio, e
    lo spazio chiamerà i bersagli nuovi, nasceranno, uno dopo un centro, tutti quelli del livello dopo
    il centro che lo chiude; o riporterà all'ascolto, dopo un tiro a vuoto o un centro sul bonus.
    Senza annuncio: la sintesi ha già letto il resoconto, e la barra di stato dice lo stesso."""
    if nasceranno == 1:
        testo = _("Palestra: la partita aspetta lo Spazio, che chiama il bersaglio nuovo.")
    elif nasceranno > 1:
        # Traduttori: n è un numero, almeno 2: i bersagli del livello nuovo, che nascono allo Spazio.
        testo = ngettext("Palestra: la partita aspetta lo Spazio, che chiama {n} bersaglio nuovo.",
                         "Palestra: la partita aspetta lo Spazio, che chiama {n} bersagli nuovi.", nasceranno).format(n=testi.intero(nasceranno))
    else:
        testo = _("Palestra: la partita aspetta lo Spazio, che ti riporta all'ascolto.")
    return Messaggio("info", testo, "", {"nasceranno": nasceranno})


def riga_dell_attesa(attesa_ferma=False, nasceranno=0):
    """La riga della fase nell'attesa della palestra (D22): prima che la partita si fermi, mentre l'effetto
    e gli avvisi suonano ancora, che cosa si sta ascoltando; poi che cosa fa lo spazio."""
    if not attesa_ferma:
        return _("Palestra: ascolta l'esito del tiro")
    if nasceranno == 1:
        return _("Palestra: Spazio per il bersaglio nuovo")
    if nasceranno > 1:
        # Traduttori: n è un numero, almeno 2; la riga sta nella barra di stato, e deve restare breve.
        return ngettext("Palestra: Spazio per {n} bersaglio nuovo", "Palestra: Spazio per {n} bersagli nuovi", nasceranno).format(n=testi.intero(nasceranno))
    return _("Palestra: Spazio per tornare all'ascolto")


# La barra di stato.

def riga_della_fase(fase, in_pausa=False, parametro=None, sottofase=None, attesa_ferma=False, nasceranno=0, passaggio=None):
    """La riga della fase, l'ultima della barra di stato; nell'attesa della palestra, quella di
    riga_dell_attesa. Nella mira dice anche che Esc annulla, e dal secondo passaggio della barra, nel
    secondo blocco del display braille, quale passaggio suona su PASSAGGI_DELLA_BARRA (D25)."""
    if fase == "pronta":
        return _("Nessuna partita: Spazio per cominciare")
    if fase == "finita":
        return _("Partita finita: Spazio per ricominciare")
    if in_pausa:
        return _("In pausa: Ctrl+P per riprendere")
    if fase == "attesa":
        return riga_dell_attesa(attesa_ferma, nasceranno)
    if fase == "ascolto":
        return _("Ascolto: Spazio per mirare")
    if fase == "mira":
        nome = NOMI_DEI_PARAMETRI.get(parametro, str(parametro or "").replace("_", " "))
        if sottofase in NOMI_DELLE_SOTTOFASI:
            # Traduttori: parametro è direzione, alzo o potenza; parte è pausa, bip o scivolata. La riga sta nei 40 caratteri del display braille, e Esc annulla la mira (D25).
            riga = _("Mira: {parametro}, {parte}; Esc annulla").format(parametro=nome, parte=NOMI_DELLE_SOTTOFASI[sottofase])
        else:
            # Traduttori: parametro è direzione, alzo o potenza.
            riga = _("Mira: {parametro}; Esc annulla").format(parametro=nome)
        if passaggio is not None and passaggio > 1:
            # Traduttori: il secondo blocco della riga della mira (D25): la barra ricomincia da capo senza lo Spazio, e n e totale sono numeri, come «passaggio 2 di 5».
            return in_blocchi(riga, _("passaggio {n} di {totale}").format(n=testi.intero(passaggio), totale=testi.intero(PASSAGGI_DELLA_BARRA)))
        return riga
    return _("Volo")


def righe_della_barra(fase, in_pausa=False, livello=0, punti=0, riserva=0, centri_del_livello=0, bersagli=0, bonus=False, parametro=None, sottofase=None,
                      attesa_ferma=False, nasceranno=0, passaggio=None):
    """Le righe della barra di stato, brevi, una cosa per riga: livello e punti; riserva e centri del
    livello su CENTRI_PER_LIVELLO; i bersagli in volo, col bonus se c'è; la fase. Senza partita c'è
    soltanto la fase; a partita finita i bersagli non ci sono più, e la riga si omette. attesa_ferma
    e nasceranno servono alla fase «attesa» della palestra (D22), come in riga_dell_attesa; passaggio
    alla mira, il passaggio della barra in corso (D25), come in riga_della_fase."""
    if fase == "pronta":
        return [riga_della_fase(fase)]
    righe = [
        # Traduttori: punti è come «1.230 punti».
        _("Livello {livello}, {punti}").format(livello=testi.intero(livello), punti=_punti(punti)),
        # Traduttori: riserva, centri e su sono numeri: «Riserva 8, centri 4 su 10».
        _("Riserva {riserva}, centri {centri} su {su}").format(riserva=testi.intero(riserva), centri=testi.intero(centri_del_livello), su=testi.intero(CENTRI_PER_LIVELLO)),
    ]
    if fase != "finita":
        if bonus:
            # Traduttori: n è il numero dei bersagli normali.
            righe.append(_("Bersagli in volo: {n} e il bonus").format(n=testi.intero(bersagli)))
        else:
            righe.append(_("Bersagli in volo: {n}").format(n=testi.intero(bersagli)))
    righe.append(riga_della_fase(fase, in_pausa, parametro, sottofase, attesa_ferma, nasceranno, passaggio))
    return righe
