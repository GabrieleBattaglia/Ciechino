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
"""

import math
from dataclasses import dataclass, field

import testi
from bersagli import durata_del_bonus, quanti_bersagli, raggio_del_livello, velocita_del_livello
from costanti import AIUTO_MANCATO_FINO_AL_LIVELLO, CENTRI_PER_LIVELLO, COLPI_BONUS, MOLTIPLICATORE_BONUS
from fisica import CANNONE
from lingua import _, plurale
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
    """Le parti dello scarto di un tiro a vuoto rispetto al centro del bersaglio, (radiale, laterale,
    verticale) come le dà fisica.scomponi, a parole e in quest'ordine: «lungo di 1,2 metri» o «corto
    di»; «a destra di 0,8 metri» o «a sinistra di»; «sopra di 1,7 metri» o «sotto di». Le parti che si
    arrotondano a zero non si dicono."""
    radiale, laterale, verticale = (_arrotondato(c) for c in scarto)
    parti = []
    # Traduttori: da qui, le parti dello scarto di un tiro a vuoto; misura è come «1,2 metri».
    if radiale > 0:
        parti.append(_("lungo di {misura}").format(misura=metri(radiale)))
    elif radiale < 0:
        parti.append(_("corto di {misura}").format(misura=metri(-radiale)))
    if laterale > 0:
        parti.append(_("a destra di {misura}").format(misura=metri(laterale)))
    elif laterale < 0:
        parti.append(_("a sinistra di {misura}").format(misura=metri(-laterale)))
    if verticale > 0:
        parti.append(_("sopra di {misura}").format(misura=metri(verticale)))
    elif verticale < 0:
        parti.append(_("sotto di {misura}").format(misura=metri(-verticale)))
    return parti


def scarto_principale(scarto):
    """La parte più grande dello scarto, a parole, come «corto di 3,4 metri»; vuoto se non c'è."""
    if scarto is None:
        return ""
    componenti = [float(c) for c in scarto]
    indice = max(range(len(componenti)), key=lambda i: abs(componenti[i]))
    sola = [0.0] * len(componenti)
    sola[indice] = componenti[indice]
    parti = scarti(sola)
    return parti[0] if parti else ""


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


# Il resoconto di un tiro.

def _riga_dell_esito(numero, livello, volo, punti, totale, riserva, colpi_bonus):
    """La prima riga del resoconto: il numero del tiro, il livello e com'è andata."""
    tiro, livello = testi.intero(numero), testi.intero(livello)
    colpito = volo.colpito
    if colpito is not None:
        precisione = testi.percento(volo.precisione or 0.0)
        bersaglio = descrizione_del_bersaglio(colpito)
        if colpito.bonus:
            # Traduttori: il resoconto di un centro sul bonus; bersaglio è come «forma seno, raggio 2,9 metri», punti e totale come «1.150 punti», precisione come «64%», colpi come «5 colpi».
            return _("Tiro {tiro}, livello {livello}: colpito il bonus, {bersaglio}; {punti}, precisione {precisione}, {colpi} in più, totale {totale}.").format(
                tiro=tiro, livello=livello, bersaglio=bersaglio, punti=_punti(punti), precisione=precisione, colpi=_colpi(colpi_bonus), totale=_punti(totale))
        # Traduttori: il resoconto di un centro; bersaglio è come «forma dente di sega, raggio 3,6 metri, fermo», punti e totale come «230 punti», precisione come «64%».
        return _("Tiro {tiro}, livello {livello}: colpito un bersaglio, {bersaglio}; {punti}, precisione {precisione}, totale {totale}.").format(
            tiro=tiro, livello=livello, bersaglio=bersaglio, punti=_punti(punti), precisione=precisione, totale=_punti(totale))
    if riserva > 0:
        restano = plurale(riserva, "resta {n} colpo", "restano {n} colpi", come=testi.intero(riserva))
        # Traduttori: il resoconto di un tiro a vuoto; restano è come «restano 8 colpi».
        return _("Tiro {tiro}, livello {livello}: a vuoto; nella riserva {restano}.").format(tiro=tiro, livello=livello, restano=restano)
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
    """Di quanto e dove si è mancato il bersaglio più vicino (D6): la distanza dalla sua superficie, poi
    lo scarto dal suo centro, lungo o corto, a destra o a sinistra, sopra o sotto. Vuoto se nessun
    bersaglio vivo è passato vicino al tiro."""
    if volo.vicino is None or volo.scarto is None:
        return ""
    distanza = metri(max(0.0, float(volo.distanza_dal_vicino or 0.0)))
    parti = scarti(volo.scarto)
    if not parti:
        # Traduttori: distanza è come «2,3 metri».
        return _("Il bersaglio più vicino è stato mancato di {distanza}.").format(distanza=distanza)
    # Traduttori: distanza è come «2,3 metri»; scarti è un elenco come «lungo di 1,2 metri, a sinistra di 0,8 metri e sopra di 1,7 metri».
    return _("Il bersaglio più vicino è stato mancato di {distanza}: rispetto al suo centro il tiro è passato {scarti}.").format(distanza=distanza,
                                                                                                                              scarti=testi.elenco(parti))


def resoconto_del_tiro(numero, livello, volo, valori, punti=0, totale=0, riserva=0, colpi_bonus=None, aiuto=None):
    """Il messaggio «resoconto» dopo un tiro: la riga dell'esito, la mira, la traiettoria, e dopo un
    centro dove stava il bersaglio, dopo un tiro a vuoto di quanto si è mancato il più vicino.
    numero è il numero del tiro, livello quello in cui è stato sparato; valori i tre valori da 0 a 1
    della mira; punti quelli del colpo, totale il punteggio dopo il colpo, riserva i colpi rimasti;
    colpi_bonus i colpi dati dal bonus, COLPI_BONUS se è None. aiuto dice se spiegare il tiro a vuoto,
    e se è None vale fino al livello AIUTO_MANCATO_FINO_AL_LIVELLO (D6); serve anche all'annuncio,
    che allora aggiunge lo scarto principale."""
    if aiuto is None:
        aiuto = livello <= AIUTO_MANCATO_FINO_AL_LIVELLO
    if colpi_bonus is None:
        colpi_bonus = COLPI_BONUS
    colpito = volo.colpito
    righe = [_riga_dell_esito(numero, livello, volo, punti, totale, riserva, colpi_bonus), riga_della_mira(volo.traiettoria, valori), riga_della_traiettoria(volo)]
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
        principale = scarto_principale(volo.scarto) if aiuto else ""
        if riserva <= 0:
            annuncio = _("A vuoto, riserva vuota.")
        elif principale:
            # Traduttori: l'annuncio breve di un tiro a vuoto nei livelli bassi; riserva è un numero, scarto come «corto di 3,4 metri».
            annuncio = _("A vuoto, riserva {riserva}, {scarto}.").format(riserva=testi.intero(riserva), scarto=principale)
        else:
            # Traduttori: l'annuncio breve di un tiro a vuoto; riserva è un numero.
            annuncio = _("A vuoto, riserva {riserva}.").format(riserva=testi.intero(riserva))
    dati = {"tiro": numero, "livello": livello, "colpito": colpito is not None, "bonus": bool(colpito is not None and colpito.bonus), "punti": punti,
            "totale": totale, "riserva": riserva, "precisione": volo.precisione}
    return Messaggio("resoconto", "\n".join(righe), annuncio, dati)


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
    resta, quanto vale."""
    da_sinistra = float(bonus.velocita[0]) > 0
    da = _("da sinistra") if da_sinistra else _("da destra")
    righe = [
        # Traduttori: da è «da sinistra» o «da destra», descrizione come «forma seno, raggio 2,9 metri».
        _("È apparso un bersaglio bonus, che entra {da}: {descrizione}.").format(da=da, descrizione=descrizione_del_bersaglio(bonus)),
        # Traduttori: durata è come «14 secondi», volte come «5 volte», colpi come «5 colpi».
        _("Attraversa lo spazio dei bersagli in {durata}, poi esce; colpito vale {volte} i punti e dà {colpi} in più.").format(
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

    return {
        # Traduttori: da qui, i nomi dei primati, dopo «Nuovo primato: ».
        "punteggio": (_("il punteggio più alto in una partita"), _punti),
        "livello": (_("il livello più alto raggiunto"), testi.intero),
        "serie": (_("la serie più lunga di centri consecutivi"), testi.intero),
        "precisione": (_("il centro più preciso"), testi.percento),
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

def in_pausa():
    return Messaggio("info", _("Partita in pausa: il tempo è fermo e i bersagli tacciono. Ctrl+P per riprendere."), _("Pausa."))


def ripresa():
    return Messaggio("info", _("La partita riprende."), _("Si riprende."))


def pausa_nella_mira():
    return Messaggio("avviso", _("Durante la mira il tempo è già fermo: la pausa si chiede mentre si ascoltano i bersagli o durante il volo."),
                     _("Il tempo è già fermo."))


def nessuna_partita():
    return Messaggio("avviso", _("Nessuna partita in corso: Spazio per cominciare."), _("Nessuna partita in corso."))


# La barra di stato.

def riga_della_fase(fase, in_pausa=False, parametro=None, sottofase=None):
    """La riga della fase, l'ultima della barra di stato."""
    if fase == "pronta":
        return _("Nessuna partita: Spazio per cominciare")
    if fase == "finita":
        return _("Partita finita: Spazio per ricominciare")
    if in_pausa:
        return _("In pausa: Ctrl+P per riprendere")
    if fase == "ascolto":
        return _("Ascolto: Spazio per mirare")
    if fase == "mira":
        nome = NOMI_DEI_PARAMETRI.get(parametro, str(parametro or "").replace("_", " "))
        if sottofase in NOMI_DELLE_SOTTOFASI:
            # Traduttori: parametro è direzione, alzo o potenza; parte è pausa, bip o scivolata.
            return _("Mira: {parametro}, {parte}").format(parametro=nome, parte=NOMI_DELLE_SOTTOFASI[sottofase])
        # Traduttori: parametro è direzione, alzo o potenza.
        return _("Mira: {parametro}").format(parametro=nome)
    return _("Volo")


def righe_della_barra(fase, in_pausa=False, livello=0, punti=0, riserva=0, centri_del_livello=0, bersagli=0, bonus=False, parametro=None, sottofase=None):
    """Le righe della barra di stato, brevi, una cosa per riga: livello e punti; riserva e centri del
    livello su CENTRI_PER_LIVELLO; i bersagli in volo, col bonus se c'è; la fase. Senza partita c'è
    soltanto la fase; a partita finita i bersagli non ci sono più, e la riga si omette."""
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
    righe.append(riga_della_fase(fase, in_pausa, parametro, sottofase))
    return righe
