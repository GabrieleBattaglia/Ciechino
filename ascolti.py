"""
Gli Ascolti di Ciechino, il tutorial audio: i passi, ognuno col titolo, il testo e il suono, senza finestre.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-20 con la 0.9.0, dalla decisione D23 del 20 novembre 2026. Gabriele: «Facciamo una cosa
ibrida. Il gioco lo lasciamo così ma nelle impostazioni aggiungiamo la voce ascolti.» Un passo alla volta,
raccontato da un testo, si sentono il test stereo, il bersaglio nel mezzo della cupola che va da un lato
all'altro, l'altezza e la distanza, tre punti a caso, la grandezza, le forme dell'oscillazione, i timbri, la
mira manuale e il suo tempo, il proiettile in volo, i colpi a segno e il bonus che attraversa la cupola. Il dialogo
che li mostra sta in gui/dialoghi.py, e la finestra lo apre dal menu Aiuto, l'unica voce dalla 0.11.10.
I suoni passano dal motore del gioco con le sue voci, VoceTono per i bersagli e il proiettile, VoceBuffer per
le tacche, i tick e i preset, e con le stesse regole dello spazio sonoro: fisica.suono_di per panoramica, nota e
volume, l'LFO, la grandezza e il timbro dei bersagli di bersagli.py, la forma del proiettile scelta nelle
impostazioni, gli effetti collocati nello spazio da Suoni.pronto. Ogni voce segue il tempo dell'audio, non
quello di gioco, che col dialogo aperto è fermo; ogni tono entra ed esce con la rampa ASCOLTI_RAMPA, contro
i click. Gli effetti si preparano con conta_subito falso e non si contano mai fra i suoni della partita:
gli Ascolti hanno i loro contatori, passi_ascoltati qui, ascolti_aperti e ascolti_finiti nella finestra.
I numeri di durata e i raggi stanno in costanti.py, coi nomi che cominciano con ASCOLTI_.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import fisica
import testi
from bersagli import Bersaglio, durata_del_bonus, mezza_escursione_ottave, portanti_senza, raggio_del_livello
from costanti import (
    ALZO_FREQUENZE,
    ASCOLTI_CRESCITA,
    ASCOLTI_FORMA,
    ASCOLTI_FORME_PER_PASSO,
    ASCOLTI_FREQUENZA_LFO,
    ASCOLTI_GRANDEZZE,
    ASCOLTI_LIVELLO_DEL_BONUS,
    ASCOLTI_PAUSA,
    ASCOLTI_PUNTI,
    ASCOLTI_RAGGIO,
    ASCOLTI_RAMPA,
    ASCOLTI_SOSTA,
    ASCOLTI_SOSTA_AI_CAPI,
    ASCOLTI_STEREO_DB,
    ASCOLTI_STEREO_FORMA,
    ASCOLTI_STEREO_FREQUENZA,
    ASCOLTI_STEREO_SOSTA,
    ASCOLTI_STEREO_VIAGGIO,
    ASCOLTI_SUONO,
    ASCOLTI_VELOCITA,
    AZIMUT_DEL_PAN,
    COLPI_BONUS,
    CUPOLA_RAGGIO,
    CUPOLA_RAGGIO_INTERNO,
    DURATA_BONUS_MINIMA,
    DURATA_DELLA_SCALA,
    FORME_LFO,
    METRI_PER_OTTAVA,
    MOLTIPLICATORE_BONUS,
    PARAMETRI,
    PARTENZA_DELLA_MIRA,
    POTENZA_DB_MINIMO,
    PROIETTILE_FORMA,
    PROIETTILE_FORME,
    PUNTATORE_FORMA,
    RAGGIO_BONUS_FATTORE,
    RIDUZIONE_DEL_TEMPO_DELLA_MIRA,
    SETTORE_AZIMUT,
    SFUMATURA_STOP,
    TACCHE,
    TEMPO_DELLA_MIRA_INIZIALE,
    TEMPO_DELLA_MIRA_MINIMO,
    VELOCITA_DEL_TEMPO,
    VELOCITA_DELLA_MIRA,
)
from lingua import _, plurale
from motore_audio import VoceTono
from sintesi import db_in_lineare, pan_della_direzione, parametri_del_puntatore, tacca_dell_asse, tick_del_tempo
from statistiche import NOMI_DELLE_FORME, NOMI_DELLE_PORTANTI

# Il mezzo della cupola dei bersagli (D28), da cui partono e dove tornano le tratte: la sua distanza dal centro
# della cupola e la sua elevazione, per le tratte che girano su un arco.
CENTRO = fisica.MEZZO_DELLA_CUPOLA
DISTANZA_DEL_MEZZO = (CUPOLA_RAGGIO_INTERNO + CUPOLA_RAGGIO) / 2.0
ELEVAZIONE_DEL_MEZZO = 30.0
# I passi della bisezione che cerca i capi delle tratte, dove il bersaglio tocca le pareti della sua zona: una
# finezza di calcolo, non una taratura.
PASSI_DEI_CAPI = 60
# Il seme dei bersagli degli Ascolti, che conta per le forme casuale e deriva: fisso, così il tutorial suona
# sempre uguale. Non è una taratura.
SEME_DEI_BERSAGLI = 23
# Lo scarto laterale, in metri, sotto il quale un punto a caso si dice al centro: meno di quanto il decimale
# dei metri sa dire.
QUASI_AL_CENTRO = 0.05
# I semitoni di un'ottava, per dire di quanto oscilla la nota.
SEMITONI_PER_OTTAVA = 12


@dataclass(frozen=True)
class Passo:
    """Un passo degli Ascolti: nome, l'identificativo, titolo e testo per il dialogo, su più righe, e
    costruisci(regia, inizio), che mette nel motore il suono del passo dal campione inizio e restituisce
    l'Ascolto."""

    nome: str
    titolo: str
    testo: str
    costruisci: Callable


@dataclass
class Ascolto:
    """Il suono di un passo messo nel motore: dal campione inizio al campione fine, quando anche l'ultima
    voce ha finito, rampe comprese. voci sono tutte le voci aggiunte, che ferma zittisce; eventi le coppie
    (evento, campione, posizione) dei preset suonati; bersagli i Bersaglio delle voci dei bersagli,
    nell'ordine in cui suonano; percorso, per le tratte e la grandezza, il Percorso seguito, coi tempi dal
    campione inizio; rivelazione il testo da aggiungere dopo l'ascolto, come i punti a caso, o vuoto."""

    passo: str
    inizio: int
    fine: int
    voci: list = field(default_factory=list)
    eventi: list = field(default_factory=list)
    bersagli: list = field(default_factory=list)
    percorso: object = None
    rivelazione: str = ""


class Percorso:
    """Un cammino a tappe, nel tempo: tappe è un elenco di coppie (valore, sosta), dove valore è un
    punto, o un numero solo come un raggio, e sosta i secondi in cui si resta lì; fra una tappa e la
    seguente si va a velocita unità al secondo, partendo e arrivando dolci, col coseno rialzato. Il
    percorso comincia fermo sulla prima tappa e finisce fermo sull'ultima.
    Attributi: durata, in secondi; arrivi, il tempo in cui si arriva a ogni tappa. Chiamato con dei
    tempi, scalari o array, restituisce un array (N, dimensione) dei valori, tenuti nei capi fuori
    dalla durata."""

    def __init__(self, tappe, velocita):
        tappe = [(np.asarray(valore, dtype=float).reshape(-1), float(sosta)) for valore, sosta in tappe]
        if not tappe:
            raise ValueError("Un percorso degli Ascolti vuole almeno una tappa.")
        if velocita <= 0:
            raise ValueError(f"La velocità di un percorso è positiva, non {velocita!r}.")
        inizi, durate, da, a, arrivi = [], [], [], [], []
        tempo = 0.0
        precedente = None
        for valore, sosta in tappe:
            if precedente is not None:
                distanza = float(np.linalg.norm(valore - precedente))
                if distanza > 0.0:
                    viaggio = distanza / velocita
                    inizi.append(tempo)
                    durate.append(viaggio)
                    da.append(precedente)
                    a.append(valore)
                    tempo += viaggio
            arrivi.append(tempo)
            inizi.append(tempo)
            durate.append(max(sosta, 0.0))
            da.append(valore)
            a.append(valore)
            tempo += max(sosta, 0.0)
            precedente = valore
        self.durata = tempo
        self.arrivi = tuple(arrivi)
        self._inizi = np.array(inizi)
        self._durate = np.array(durate)
        self._da = np.array(da)
        self._a = np.array(a)

    def __call__(self, tempi):
        tempi = np.clip(np.atleast_1d(np.asarray(tempi, dtype=float)), 0.0, self.durata)
        quale = np.clip(np.searchsorted(self._inizi, tempi, side="right") - 1, 0, len(self._inizi) - 1)
        durate = self._durate[quale]
        aperte = durate > 0.0
        frazioni = np.where(aperte, np.clip((tempi - self._inizi[quale]) / np.where(aperte, durate, 1.0), 0.0, 1.0), 1.0)
        dolci = (1.0 - np.cos(math.pi * frazioni)) / 2.0
        return self._da[quale] + (self._a[quale] - self._da[quale]) * dolci[:, None]


def bersaglio_degli_ascolti(portante, forma=ASCOLTI_FORMA, raggio=ASCOLTI_RAGGIO, punto=CENTRO, velocita=(0.0, 0.0, 0.0), bonus=False):
    """Un bersaglio come quelli del gioco, coi parametri di serie della sua forma, l'LFO a
    ASCOLTI_FREQUENZA_LFO con la fase a zero e il seme fisso, nato al tempo zero."""
    return Bersaglio(numero=0, forma=forma, raggio=raggio, partenza=punto, velocita=velocita, bonus=bonus, frequenza_lfo=ASCOLTI_FREQUENZA_LFO, fase_lfo=0.0,
                     seme=SEME_DEI_BERSAGLI, portante=portante)


def zona():
    """La fisica.Zona dove sta il centro del bersaglio degli Ascolti, quella del gioco (D28)."""
    return fisica.Zona(ASCOLTI_RAGGIO)


def _sferico(distanza, azimut, elevazione):
    """Il punto della cupola a quella distanza dal suo centro, con quell'azimut e quell'elevazione: (3,) per
    scalari, (N, 3) per array."""
    return fisica.punto_della_cupola(distanza, azimut, elevazione)


def _limite(punto_di, dentro, fuori):
    """Il valore più vicino a fuori, cercato per bisezione da dentro, per cui punto_di(valore) sta ancora
    nella zona del bersaglio: il capo di una tratta."""
    zona_del_bersaglio = zona()
    for _passo in range(PASSI_DEI_CAPI):
        medio = (dentro + fuori) / 2.0
        if zona_del_bersaglio.contiene(punto_di(medio), tolleranza=0.0):
            dentro = medio
        else:
            fuori = medio
    return dentro


def _capi():
    """I capi delle tre tratte, dal mezzo della cupola: gli azimut di sinistra e di destra, sull'arco alla
    distanza e all'elevazione del mezzo; le elevazioni di terra e di cima, sull'arco alla distanza del mezzo,
    davanti; le profondità di fronte e di fondo, alla quota del mezzo, davanti. Fin dove il bersaglio sta
    tutto nella cupola."""
    distanza, elevazione = DISTANZA_DEL_MEZZO, ELEVAZIONE_DEL_MEZZO
    _x, altezza, profondita = CENTRO
    return {
        "sinistra": _limite(lambda azimut: _sferico(distanza, azimut, elevazione), 0.0, -SETTORE_AZIMUT),
        "destra": _limite(lambda azimut: _sferico(distanza, azimut, elevazione), 0.0, SETTORE_AZIMUT),
        "terra": _limite(lambda gradi: _sferico(distanza, 0.0, gradi), elevazione, 0.0),
        "cima": _limite(lambda gradi: _sferico(distanza, 0.0, gradi), elevazione, 90.0),
        "fronte": _limite(lambda z: (0.0, altezza, z), profondita, 0.0),
        "fondo": _limite(lambda z: (0.0, altezza, z), profondita, CUPOLA_RAGGIO),
    }


def punti_dei_capi():
    """I punti dei capi delle tratte, coi nomi di _capi, per i testi e gli effetti."""
    capi = _capi()
    distanza, elevazione = DISTANZA_DEL_MEZZO, ELEVAZIONE_DEL_MEZZO
    _x, altezza, _z = CENTRO
    punti = {lato: _sferico(distanza, capi[lato], elevazione) for lato in ("sinistra", "destra")}
    punti.update({quota: _sferico(distanza, 0.0, capi[quota]) for quota in ("terra", "cima")})
    punti.update({fila: np.array((0.0, altezza, capi[fila])) for fila in ("fronte", "fondo")})
    return {nome: tuple(float(c) for c in punto) for nome, punto in punti.items()}


def tratta(nome):
    """La tratta di un passo, come (tappe, velocità, punto_di): i valori delle tappe, dal mezzo al primo capo,
    al secondo e di nuovo al mezzo, con le soste; la velocità nelle unità di quei valori al secondo, perché il
    bersaglio vada a ASCOLTI_VELOCITA metri al secondo; e la funzione che fa dei valori i punti. L'orizzontale
    gira sull'azimut, l'altezza sull'elevazione, la distanza va in profondità a quota costante."""
    capi = _capi()
    distanza, elevazione = DISTANZA_DEL_MEZZO, ELEVAZIONE_DEL_MEZZO
    _x, altezza, profondita = CENTRO
    if nome == "orizzontale":
        mezzo, primo, secondo = 0.0, capi["sinistra"], capi["destra"]
        velocita = math.degrees(ASCOLTI_VELOCITA / (distanza * math.cos(math.radians(elevazione))))

        def punto_di(valori):
            return _sferico(distanza, valori, elevazione)
    elif nome == "altezza":
        mezzo, primo, secondo = elevazione, capi["terra"], capi["cima"]
        velocita = math.degrees(ASCOLTI_VELOCITA / distanza)

        def punto_di(valori):
            return _sferico(distanza, 0.0, valori)
    elif nome == "distanza":
        mezzo, primo, secondo = profondita, capi["fronte"], capi["fondo"]
        velocita = ASCOLTI_VELOCITA

        def punto_di(valori):
            valori = np.asarray(valori, dtype=float)
            return np.stack(np.broadcast_arrays(0.0, altezza, valori), axis=-1)
    else:
        raise ValueError(f"tratta sconosciuta: {nome!r}")
    tappe = [((mezzo,), ASCOLTI_SOSTA), ((primo,), ASCOLTI_SOSTA_AI_CAPI), ((secondo,), ASCOLTI_SOSTA_AI_CAPI), ((mezzo,), ASCOLTI_SOSTA)]
    return tappe, velocita, punto_di


def _metri(x):
    return testi.metri(x)


def _distanza(punto):
    return fisica.distanza_dal_cannone(np.asarray(punto, dtype=float))


def _semitoni(raggio):
    return testi.numero(SEMITONI_PER_OTTAVA * mezza_escursione_ottave(raggio), 1)


def _testo(*righe):
    """Le righe di un testo, una per riga, senza righe vuote."""
    return "\n".join(riga for riga in righe if riga)


# I testi dei passi. Ogni numero viene dalle costanti e dalla fisica del gioco, così il testo dice sempre il vero.

def _testo_stereo():
    return _testo(
        _("Gli Ascolti ti fanno sentire, un passo alla volta, come suona lo spazio di Ciechino: prima di giocare, o quando vuoi ritarare "
          "l'orecchio."),
        _("Il primo è il test stereo: un suono parte tutto a sinistra, resta lì un attimo, scivola piano fino a tutto a destra, e lì si ferma."),
        _("Se lo senti partire a destra e arrivare a sinistra, le cuffie sono indossate al contrario, o gli altoparlanti sono scambiati: gira le "
          "cuffie, o scambia i due altoparlanti, perché nel gioco destra e sinistra dicono dove mirare. Se lo senti sempre in mezzo, o da un lato "
          "solo, controlla i cavi e il bilanciamento dell'audio di Windows."),
    )


def _testo_orizzontale():
    capi = _capi()
    _x, altezza, _z = CENTRO
    return _testo(
        # Traduttori: distanza e altezza sono misure come «55 metri», raggio come «5 metri».
        _("Un bersaglio fermo nel mezzo della cupola, {distanza} davanti a te e a {altezza} d'altezza, grande come quelli del primo livello, "
          "col raggio di {raggio}.").format(distanza=_metri(_distanza(CENTRO)), altezza=_metri(altezza), raggio=_metri(ASCOLTI_RAGGIO)),
        # Traduttori: angolo è un angolo, come «34 gradi».
        _("Poi gira verso sinistra fino al lato della cupola, a {angolo} a sinistra, attraversa la cupola fino al lato destro e torna al mezzo, "
          "sempre alla stessa distanza da te e alla stessa altezza.").format(angolo=testi.gradi(round(-capi["sinistra"]))),
        # Traduttori: massimo è come «40 gradi», angolo come «34 gradi».
        _("Ascolta la panoramica: è l'unica cosa che cambia, e segue l'angolo del bersaglio visto dal cannone. Tutto da un lato vuol dire il "
          "lato della cupola, a {massimo}; il bersaglio, che è grande, ci arriva vicino, a {angolo}, perché resta tutto dentro.").format(
            massimo=testi.gradi(AZIMUT_DEL_PAN), angolo=testi.gradi(round(capi["destra"]))),
    )


def _testo_altezza():
    punti = punti_dei_capi()
    terra, cima = punti["terra"][1], punti["cima"][1]
    return _testo(
        # Traduttori: terra e cima sono altezze, come «5 metri» e «52 metri»; distanza come «55 metri».
        _("Lo stesso bersaglio, dal mezzo, scende fino a terra, dove il suo centro sta a {terra} dal prato, perché il bersaglio resti tutto sopra "
          "il suolo; poi sale fino a {cima} d'altezza, e torna al mezzo, sempre a {distanza} dal centro della cupola, sulla sua volta.").format(
            terra=_metri(terra), cima=_metri(cima), distanza=_metri(DISTANZA_DEL_MEZZO)),
        # Traduttori: ottava è una misura come «20,3 metri»; grave e acuto sono frequenze già scritte, come 262 e 1.315.
        _("Ascolta la nota: più è grave, più il bersaglio è in basso. Sale di un'ottava ogni {ottava}, da circa {grave} hertz a terra a circa "
          "{acuto} hertz in alto, e oscilla attorno a quella nota, perché il bersaglio è grande.").format(
            ottava=testi.metri(METRI_PER_OTTAVA, 2), grave=testi.intero(fisica.frequenza_di(terra)), acuto=testi.intero(fisica.frequenza_di(cima))),
    )


def _testo_distanza():
    punti = punti_dei_capi()
    fronte, fondo = punti["fronte"], punti["fondo"]
    decibel = fisica.guadagno_db_di(_distanza(fronte)) - fisica.guadagno_db_di(_distanza(fondo))
    return _testo(
        # Traduttori: vicino e lontano sono distanze, come «35 metri» e «75 metri»; altezza come «27,5 metri».
        _("Lo stesso bersaglio, dal mezzo, viene vicino al massimo, a {vicino} da te; poi va lontano al massimo, a {lontano}; e torna al mezzo, "
          "sempre davanti a te e a {altezza} d'altezza.").format(vicino=_metri(_distanza(fronte)), lontano=_metri(_distanza(fondo)),
                                                                 altezza=_metri(fronte[1])),
        # Traduttori: decibel è un numero già scritto, come 10.
        _("La nota e la panoramica restano quelle: cambia soltanto il volume, di circa {decibel} decibel fra il vicino e il lontano. Sono i due capi "
          "della scala della distanza, nel mezzo della cupola.").format(decibel=testi.intero(decibel)),
        _("Il volume che senti dipende anche dal volume del computer, da quello di Ciechino e dalle cuffie o dagli altoparlanti: quando ne cambi "
          "uno, l'orecchio perde la sua misura della distanza. Per ritararlo torna qui, dal menu Aiuto, e riascolta questo "
          "passo, il vicino e il lontano, prima di giocare."),
    )


def _testo_punti():
    return _testo(
        # Traduttori: punti è come «3 punti».
        _("Il bersaglio salta in {punti} presi a caso nella cupola, uno dopo l'altro, staccati da un breve silenzio. Prova a capire dove sta ognuno: "
          "a destra o a sinistra, in alto o in basso, vicino o lontano.").format(punti=plurale(ASCOLTI_PUNTI, "{n} punto", "{n} punti")),
        _("Quando l'ultimo ha finito di suonare, in fondo a questo testo compare dove erano, in metri, e il cursore va lì; poi lo spazio te li fa "
          "riascoltare, sapendo dove sono."),
    )


def _testo_grandezza():
    piccolo, primo, gigante = ASCOLTI_GRANDEZZE
    return _testo(
        # Traduttori: piccolo, primo e gigante sono raggi, come «0,5 metri», «5 metri» e «10 metri».
        _("Un bersaglio fermo nel mezzo della cupola, che cresce: parte piccolissimo, col raggio di {piccolo}, il più piccolo del gioco; cresce fino a "
          "{primo}, il raggio dei bersagli del primo livello, i più grandi del gioco; poi diventa gigantesco, {gigante}, più di ogni bersaglio "
          "vero.").format(piccolo=_metri(piccolo), primo=_metri(primo), gigante=_metri(gigante)),
        # Traduttori: piccolo e gigante sono numeri di semitoni già scritti, come 0,4 e 7,2.
        _("Ascolta l'oscillazione della nota: col bersaglio piccolissimo è quasi un tremolio, di {piccolo} semitoni sopra e sotto il centro; col "
          "gigantesco va su e giù di {gigante} semitoni. Più è larga, più il bersaglio è grande; la nota del centro non cambia, perché il centro "
          "resta alla stessa altezza.").format(piccolo=_semitoni(piccolo), gigante=_semitoni(gigante)),
    )


def _parti_delle_forme():
    """Le forme dell'LFO divise in passi di ASCOLTI_FORME_PER_PASSO, nell'ordine di FORME_LFO."""
    return [FORME_LFO[inizio:inizio + ASCOLTI_FORME_PER_PASSO] for inizio in range(0, len(FORME_LFO), ASCOLTI_FORME_PER_PASSO)]


def _nomi(chiavi, nomi):
    return testi.elenco([nomi.get(chiave, chiave.replace("_", " ")) for chiave in chiavi])


def _testo_forme(posto, forme, portante):
    durata = testi.numero(ASCOLTI_SUONO, 1)
    if posto == 1:
        prima = _testo(
            # Traduttori: quante è il numero delle forme, come 12; per_passo come 6; timbro è un timbro, come «organo»; secondi è un numero già scritto, come 2,5.
            _("Il modo in cui la nota oscilla è la forma del bersaglio: ogni bersaglio ne prende una a caso, fra {quante}. Qui le senti tutte, "
              "{per_passo} per passo, nello stesso punto, nel mezzo della cupola, con la stessa grandezza e sempre il timbro {timbro}: ognuna suona "
              "per {secondi} secondi, staccata dalla seguente da un breve silenzio.").format(quante=len(FORME_LFO), per_passo=ASCOLTI_FORME_PER_PASSO,
                                                                                            timbro=NOMI_DELLE_PORTANTI.get(portante, portante),
                                                                                            secondi=durata),
            _("Nel gioco ogni bersaglio prende a caso anche i dettagli della sua forma e la velocità dell'oscillazione; qui le forme suonano nel loro "
              "disegno di serie, tutte alla stessa velocità."),
        )
    else:
        # Traduttori: quante è un numero, come 6.
        prima = _("Altre {quante} forme dell'oscillazione, allo stesso modo.").format(quante=len(forme))
    # Traduttori: forme è l'elenco dei nomi delle forme, come «seno, dente di sega e triangolo».
    return _testo(prima, _("In quest'ordine: {forme}.").format(forme=_nomi(forme, NOMI_DELLE_FORME)))


def _testo_timbri(proiettile):
    timbri = portanti_senza(proiettile)
    return _testo(
        # Traduttori: quanti è un numero, come 3; secondi un numero già scritto, come 2,5; timbri l'elenco dei timbri, come «organo, triangolo e sega morbida».
        _("Il timbro della voce, la sua forma d'onda, ogni bersaglio lo prende a caso, fra {quanti}: qui li senti tutti, nel mezzo della cupola, con "
          "la stessa oscillazione, ognuno per {secondi} secondi. In quest'ordine: {timbri}.").format(quanti=len(timbri), secondi=testi.numero(ASCOLTI_SUONO, 1),
                                                                                                  timbri=_nomi(timbri, NOMI_DELLE_PORTANTI)),
        # Traduttori: proiettile è un timbro, come «seno».
        _("Il timbro {proiettile} è quello del tuo proiettile, che scegli con Proiettile nel menu Impostazioni: nessun bersaglio lo usa, così il "
          "colpo non si confonde con loro. Con qualunque timbro la nota dice l'altezza allo stesso modo.").format(
            proiettile=NOMI_DELLE_PORTANTI.get(proiettile, proiettile)),
    )


def _testo_mira():
    return _testo(
        # Traduttori: basso e alto sono frequenze già scritte, come 150 e 2.400; decibel è un numero già scritto, come 20.
        _("La mira manuale, come in partita, ma senza sparare: prima il respiro che avvia la mira, coi tre valori al centro della loro scala. "
          "Poi il puntatore, il suono della mira: una voce sola che porta sempre tutti e tre i valori, la direzione nella posizione fra le "
          "orecchie, l'alzo nell'altezza della nota, da {basso} a {alto} hertz, e la potenza nel volume, {decibel} decibel dal minimo al "
          "massimo.").format(basso=testi.intero(ALZO_FREQUENZE[0]), alto=testi.intero(ALZO_FREQUENZE[1]), decibel=testi.intero(-POTENZA_DB_MINIMO)),
        _("Qui si muove un valore alla volta, con gli altri due al centro. Prima la direzione, come tenendo giù la freccia destra e poi la "
          "sinistra: il puntatore va dal mezzo a tutto a destra, dove si ferma col suo colpo sordo, e poi fino a tutto a sinistra; poi l'alzo, "
          "con le frecce su e giù, la nota che sale fino in cima e scende fino in fondo; poi la potenza, con Fine e Home, che cresce fino al "
          "pieno e cala fino al minimo, dove si sente ancora."),
        # Traduttori: durata è come «3 secondi».
        _("Ogni valore percorre tutta la sua scala in {durata}, con una tacca a ogni decimo che attraversa, sempre più acuta verso il "
          "massimo, che arriva dal punto in cui sta il puntatore.").format(durata=testi.durata(DURATA_DELLA_SCALA)),
        _("In partita un valore si muove finché tieni giù il suo tasto, e si ferma quando lo lasci, mentre gli altri restano dove li hai "
          "lasciati, anche nel suono del puntatore; due tasti di valori diversi li muovono insieme. Invio spara coi tre valori di quel "
          "momento; Esc annulla la mira, e il proiettile resta nella riserva."),
    )


def _testo_tempo():
    return _testo(
        # Traduttori: iniziale, riduzione e minimo sono durate, come «30 secondi», «1 secondo» e «10 secondi».
        _("Fuori dalla palestra la mira ha un tempo, che si accorcia salendo di livello: {iniziale} al primo livello dopo la palestra, "
          "{riduzione} in meno a ogni livello, fino a {minimo}. Nella palestra la mira non ha tempo.").format(
            iniziale=testi.durata(TEMPO_DELLA_MIRA_INIZIALE), riduzione=testi.durata(RIDUZIONE_DEL_TEMPO_DELLA_MIRA),
            minimo=testi.durata(TEMPO_DELLA_MIRA_MINIMO)),
        # Traduttori: minimo è una durata, come «10 secondi».
        _("Qui senti la mira più breve, di {minimo}: un tick a ogni secondo, sempre più acuto, e allo scadere lo sparo, perché il colpo parte "
          "da solo, coi valori di quel momento.").format(minimo=testi.durata(TEMPO_DELLA_MIRA_MINIMO)),
    )


def _testo_volo(proiettile):
    from partita import volo_d_esempio

    volo = volo_d_esempio()
    return _testo(
        # Traduttori: altezza e gittata sono misure come «25,3 metri»; secondi è un numero già scritto, come 4,9.
        _("Un tiro dal cannone verso il mezzo della cupola, dove non c'è nessun bersaglio: lo sparo ai tuoi piedi, poi il proiettile che sale fino a "
          "{altezza}, si allontana e ricade, e il tonfo a terra, a {gittata} davanti a te, dopo {secondi} secondi di volo.").format(
            altezza=_metri(volo.altezza_massima), gittata=_metri(volo.gittata), secondi=testi.numero(volo.durata / VELOCITA_DEL_TEMPO, 1)),
        # Traduttori: forma è un timbro, come «seno».
        _("Il proiettile suona in {forma}, la forma che hai scelto con Proiettile nel menu Impostazioni, e segue le regole dello spazio come i "
          "bersagli: la panoramica per il lato, la nota per l'altezza, il volume per la distanza. Lo sparo e il tonfo arrivano dal punto in cui "
          "avvengono.").format(forma=NOMI_DELLE_PORTANTI.get(proiettile, proiettile)),
    )


def _testo_colpi():
    punti = punti_dei_capi()
    return _testo(
        # Traduttori: vicino e lontano sono distanze, come «35 metri» e «75 metri».
        _("Gli effetti dei colpi a segno, ognuno col suo jingle: prima il bersaglio colpito, tre note, vicino, a {vicino}, e lontano, a "
          "{lontano}; poi il bonus colpito, otto note più acute e più brillanti, vicino e lontano.").format(
            vicino=_metri(_distanza(punti["fronte"])), lontano=_metri(_distanza(punti["fondo"]))),
        _("Anche gli effetti vengono dal punto dello spazio dove avvengono: più sono lontani, più suonano piano."),
    )


def _testo_bonus():
    livello = ASCOLTI_LIVELLO_DEL_BONUS
    return _testo(
        # Traduttori: durata è come «14 secondi», livello un numero, come 3.
        _("Il bersaglio bonus: compare col suo suono, entra dal lato sinistro della cupola, la attraversa passando davanti a te, nel mezzo, ed "
          "esce dal lato destro, in {durata}, come al livello {livello}, il primo dei bonus; quando esce, senti il suono del bonus uscito.").format(
            durata=testi.durata(durata_del_bonus(livello)), livello=livello),
        # Traduttori: minima è come «6 secondi»; volte e colpi sono numeri, come 5.
        _("Nel gioco il bonus entra da un lato preso a caso, a un'altezza e a una distanza prese a caso, e più si sale di livello più è svelto, "
          "fino a {minima}. Un centro sul bonus vale {volte} volte un bersaglio normale del suo livello, e regala {colpi} colpi.").format(
            minima=testi.durata(DURATA_BONUS_MINIMA), volte=MOLTIPLICATORE_BONUS, colpi=COLPI_BONUS),
        _("È l'ultimo passo: gli Ascolti si ritrovano quando vuoi, nel menu Aiuto."),
    )


def riga_del_punto(numero, punto):
    """La riga che dice dove era un punto a caso: di lato, in altezza e in profondità, in metri."""
    x, y, z = (float(c) for c in punto)
    if abs(x) < QUASI_AL_CENTRO:
        lato = _("al centro")
    elif x < 0:
        # Traduttori: metri è una misura, come «12,3 metri».
        lato = _("{metri} a sinistra").format(metri=testi.metri(-x))
    else:
        # Traduttori: metri è una misura, come «12,3 metri».
        lato = _("{metri} a destra").format(metri=testi.metri(x))
    # Traduttori: numero è il numero del punto, come 1; lato è come «12,3 metri a sinistra»; altezza e distanza sono misure, come «31,0 metri».
    return _("Punto {numero}: {lato}, a {altezza} d'altezza, a {distanza} davanti a te.").format(numero=numero, lato=lato, altezza=testi.metri(y),
                                                                                               distanza=testi.metri(z))


def rivelazione_dei_punti(punti):
    """Il testo che dice dove erano i punti a caso, una riga per punto, dopo una riga che lo annuncia."""
    righe = [_("Ecco dove erano i punti che hai sentito:")]
    righe.extend(riga_del_punto(numero, punto) for numero, punto in enumerate(punti, 1))
    return "\n".join(righe)


# I suoni dei passi: ognuno riceve la regia e il campione da cui comincia, e restituisce l'Ascolto.

def _suono_stereo(regia, inizio):
    ascolto = Ascolto("stereo", inizio, inizio)
    partenza = inizio / regia.motore.fs
    livello = db_in_lineare(ASCOLTI_STEREO_DB)

    def parametri(tempi_di_gioco, tempi_audio):
        tempi = np.atleast_1d(np.asarray(tempi_audio, dtype=float)) - partenza
        frazioni = np.clip((tempi - ASCOLTI_STEREO_SOSTA) / ASCOLTI_STEREO_VIAGGIO, 0.0, 1.0)
        return np.full(tempi.shape, ASCOLTI_STEREO_FREQUENZA), 2.0 * frazioni - 1.0, np.full(tempi.shape, livello)

    regia.tono(ascolto, parametri, ASCOLTI_STEREO_FORMA, inizio, 2 * ASCOLTI_STEREO_SOSTA + ASCOLTI_STEREO_VIAGGIO)
    return ascolto


def _suono_del_percorso(nome):
    """Il costruttore di una tratta: il bersaglio dal mezzo al primo capo, al secondo e di nuovo al mezzo."""

    def costruisci(regia, inizio):
        ascolto = Ascolto(nome, inizio, inizio)
        tappe, velocita, punto_di = tratta(nome)
        percorso = Percorso(tappe, velocita)
        bersaglio = bersaglio_degli_ascolti(regia.portante)
        partenza = inizio / regia.motore.fs

        def parametri(tempi_di_gioco, tempi_audio):
            tempi = np.atleast_1d(np.asarray(tempi_audio, dtype=float)) - partenza
            return bersaglio.suono_in(punto_di(percorso(tempi)[:, 0]), tempi)

        ascolto.percorso = percorso
        ascolto.bersagli.append(bersaglio)
        regia.tono(ascolto, parametri, bersaglio.portante, inizio, percorso.durata)
        return ascolto

    return costruisci


def _suono_dei_punti(regia, inizio):
    ascolto = Ascolto("punti", inizio, inizio)
    punti = regia.punti_a_caso()
    campione = inizio
    pausa = round(ASCOLTI_PAUSA * regia.motore.fs)
    for punto in punti:
        bersaglio = bersaglio_degli_ascolti(regia.portante, punto=punto)
        ascolto.bersagli.append(bersaglio)
        campione = regia.tono(ascolto, _parametri_fermi(bersaglio, campione / regia.motore.fs), bersaglio.portante, campione, ASCOLTI_SUONO) + pausa
    ascolto.rivelazione = rivelazione_dei_punti(punti)
    return ascolto


def _parametri_fermi(bersaglio, partenza, raggio=None):
    """I parametri di un bersaglio fermo nel suo punto, con l'LFO che parte da zero a partenza, in secondi
    dell'audio; raggio, se c'è, è una funzione dei secondi dalla partenza, per un bersaglio che cresce."""

    def parametri(tempi_di_gioco, tempi_audio):
        tempi = np.atleast_1d(np.asarray(tempi_audio, dtype=float)) - partenza
        posizioni = np.broadcast_to(bersaglio.partenza, (len(tempi), 3))
        return bersaglio.suono_in(posizioni, tempi, None if raggio is None else raggio(tempi))

    return parametri


def _suono_della_grandezza(regia, inizio):
    ascolto = Ascolto("grandezza", inizio, inizio)
    percorso = Percorso([((raggio,), ASCOLTI_SOSTA) for raggio in ASCOLTI_GRANDEZZE], ASCOLTI_CRESCITA)
    bersaglio = bersaglio_degli_ascolti(regia.portante, raggio=max(ASCOLTI_GRANDEZZE))
    ascolto.percorso = percorso
    ascolto.bersagli.append(bersaglio)
    regia.tono(ascolto, _parametri_fermi(bersaglio, inizio / regia.motore.fs, lambda tempi: percorso(tempi)[:, 0]), bersaglio.portante, inizio, percorso.durata)
    return ascolto


def _suono_delle_forme(nome, forme):
    """Il costruttore di un passo delle forme: una dopo l'altra, al centro, staccate da ASCOLTI_PAUSA."""

    def costruisci(regia, inizio):
        ascolto = Ascolto(nome, inizio, inizio)
        campione = inizio
        pausa = round(ASCOLTI_PAUSA * regia.motore.fs)
        for forma in forme:
            bersaglio = bersaglio_degli_ascolti(regia.portante, forma=forma)
            ascolto.bersagli.append(bersaglio)
            campione = regia.tono(ascolto, _parametri_fermi(bersaglio, campione / regia.motore.fs), bersaglio.portante, campione, ASCOLTI_SUONO) + pausa
        return ascolto

    return costruisci


def _suono_dei_timbri(regia, inizio):
    ascolto = Ascolto("timbri", inizio, inizio)
    campione = inizio
    pausa = round(ASCOLTI_PAUSA * regia.motore.fs)
    for portante in portanti_senza(regia.proiettile):
        bersaglio = bersaglio_degli_ascolti(portante)
        ascolto.bersagli.append(bersaglio)
        campione = regia.tono(ascolto, _parametri_fermi(bersaglio, campione / regia.motore.fs), portante, campione, ASCOLTI_SUONO) + pausa
    return ascolto


# Le corse di ogni valore nella mira degli Ascolti: dal centro al massimo, poi dal massimo al minimo, come
# tenendo giù un tasto e poi l'altro.
CORSE_DELLA_MIRA = ((PARTENZA_DELLA_MIRA, 1.0), (1.0, 0.0))


def _suono_della_mira(regia, inizio):
    """Come in partita: il respiro della mira, poi ogni valore che corre, con le sue tacche e il suono del capo,
    una corsa dopo l'altra, staccate da ASCOLTI_PAUSA (D29)."""
    ascolto = Ascolto("mira", inizio, inizio)
    pausa = round(ASCOLTI_PAUSA * regia.motore.fs)
    campione = regia.effetto(ascolto, "mira_avviata", inizio) + pausa
    for parametro in PARAMETRI:
        for da, a in CORSE_DELLA_MIRA:
            campione = regia.corsa(ascolto, parametro, da, a, campione) + pausa
    return ascolto


def _suono_del_tempo(regia, inizio):
    """La mira a tempo più breve, TEMPO_DELLA_MIRA_MINIMO: un tick a ogni secondo, sempre più acuto, e allo
    scadere lo sparo dal cannone (D29)."""
    ascolto = Ascolto("tempo", inizio, inizio)
    fs = regia.motore.fs
    secondi = TEMPO_DELLA_MIRA_MINIMO
    for secondo in range(1, math.ceil(secondi)):
        regia.buffer(ascolto, tick_del_tempo(secondo, secondi, fs=fs), inizio + round(secondo * fs))
    scadenza = inizio + round(secondi * fs)
    ascolto.fine = max(ascolto.fine, scadenza)
    regia.effetto(ascolto, "sparo", scadenza, fisica.CANNONE)
    return ascolto


def _suono_del_volo(regia, inizio):
    """Il tiro di partita.volo_d_esempio, come in partita: lo sparo dal cannone e la voce del proiettile, nella
    forma delle impostazioni, dallo stesso campione; all'arrivo a terra la voce si ferma e parte il tonfo,
    dal punto dove cade."""
    from partita import volo_d_esempio
    from volo import parametri_del_proiettile

    ascolto = Ascolto("volo", inizio, inizio)
    fs = regia.motore.fs
    volo = volo_d_esempio()
    suono_del_volo = parametri_del_proiettile(volo)
    partenza = inizio / fs

    def parametri(tempi_di_gioco, tempi_audio):
        tempi = np.atleast_1d(np.asarray(tempi_audio, dtype=float))
        return suono_del_volo(volo.partenza + (tempi - partenza) * VELOCITA_DEL_TEMPO)

    regia.effetto(ascolto, "sparo", inizio, fisica.CANNONE)
    a_terra = inizio + round(volo.durata / VELOCITA_DEL_TEMPO * fs)
    regia.tono(ascolto, parametri, regia.proiettile, inizio, (a_terra - inizio) / fs, attacco=SFUMATURA_STOP, sfumatura=SFUMATURA_STOP)
    regia.effetto(ascolto, "a_terra", a_terra, volo.punto_finale)
    return ascolto


def _suono_dei_colpi(regia, inizio):
    ascolto = Ascolto("colpi", inizio, inizio)
    capi = punti_dei_capi()
    pausa = round(ASCOLTI_PAUSA * regia.motore.fs)
    campione = inizio
    for evento in ("colpito", "colpito_bonus"):
        for capo in ("fronte", "fondo"):
            campione = regia.effetto(ascolto, evento, campione, capi[capo]) + pausa
    return ascolto


def bonus_degli_ascolti(portante):
    """Il bonus che attraversa la cupola, da sinistra a destra, su una corda alla quota e alla profondità del
    mezzo, col raggio e la velocità del livello ASCOLTI_LIVELLO_DEL_BONUS, nato al tempo zero: parte e arriva
    a un raggio fuori dai lati, come quello del gioco (D28)."""
    livello = ASCOLTI_LIVELLO_DEL_BONUS
    raggio = raggio_del_livello(livello) * RAGGIO_BONUS_FATTORE
    settore = math.radians(SETTORE_AZIMUT)
    _x, altezza, profondita = CENTRO
    fuori = (profondita * math.sin(settore) + raggio) / math.cos(settore)
    rapidita = 2.0 * fuori / durata_del_bonus(livello)
    return bersaglio_degli_ascolti(portante, raggio=raggio, punto=(-fuori, altezza, profondita), velocita=(rapidita, 0.0, 0.0), bonus=True)


def _suono_del_bonus(regia, inizio):
    """Come in partita: il bonus nasce col suo avviso, attraversa la cupola, e all'uscita la sua voce si ferma e
    suona il bonus uscito, dal punto dove esce."""
    ascolto = Ascolto("bonus", inizio, inizio)
    fs = regia.motore.fs
    bonus = bonus_degli_ascolti(regia.portante)
    partenza = inizio / fs

    def parametri(tempi_di_gioco, tempi_audio):
        tempi = np.atleast_1d(np.asarray(tempi_audio, dtype=float)) - partenza
        return bonus.suono_in(bonus.posizione(bonus.nato + tempi * VELOCITA_DEL_TEMPO), tempi)

    ascolto.bersagli.append(bonus)
    uscita = inizio + round((bonus.tempo_di_uscita - bonus.nato) / VELOCITA_DEL_TEMPO * fs)
    regia.effetto(ascolto, "bonus_appare", inizio)
    regia.tono(ascolto, parametri, bonus.portante, inizio, (uscita - inizio) / fs, sfumatura=SFUMATURA_STOP)
    regia.effetto(ascolto, "bonus_uscito", uscita, bonus.posizione(bonus.tempo_di_uscita))
    return ascolto


def passi(proiettile=PROIETTILE_FORMA):
    """I passi degli Ascolti, nell'ordine, coi testi per quella forma del proiettile, che i bersagli non usano mai."""
    portante = portanti_senza(proiettile)[0]
    elenco = [
        Passo("stereo", _("Il test stereo"), _testo_stereo(), _suono_stereo),
        Passo("orizzontale", _("Il mezzo della cupola, da un lato all'altro"), _testo_orizzontale(), _suono_del_percorso("orizzontale")),
        Passo("altezza", _("L'altezza, a terra e in alto"), _testo_altezza(), _suono_del_percorso("altezza")),
        Passo("distanza", _("La distanza, vicino e lontano"), _testo_distanza(), _suono_del_percorso("distanza")),
        Passo("punti", _("I punti a caso"), _testo_punti(), _suono_dei_punti),
        Passo("grandezza", _("La grandezza, da piccolissimo a gigantesco"), _testo_grandezza(), _suono_della_grandezza),
    ]
    parti = _parti_delle_forme()
    for posto, forme in enumerate(parti, 1):
        if len(parti) == 1:
            titolo = _("Le forme dell'oscillazione")
        else:
            # Traduttori: parte e parti sono numeri: le forme dell'oscillazione stanno in più passi.
            titolo = _("Le forme dell'oscillazione, parte {parte} di {parti}").format(parte=posto, parti=len(parti))
        nome = "forme" if posto == 1 else f"forme_{posto}"
        elenco.append(Passo(nome, titolo, _testo_forme(posto, forme, portante), _suono_delle_forme(nome, forme)))
    elenco.extend((
        Passo("timbri", _("I timbri dei bersagli"), _testo_timbri(proiettile), _suono_dei_timbri),
        Passo("mira", _("La mira manuale"), _testo_mira(), _suono_della_mira),
        Passo("tempo", _("Il tempo della mira"), _testo_tempo(), _suono_del_tempo),
        Passo("volo", _("Il proiettile in volo"), _testo_volo(proiettile), _suono_del_volo),
        Passo("colpi", _("Il centro e il bonus, vicino e lontano"), _testo_colpi(), _suono_dei_colpi),
        Passo("bonus", _("Il bonus che attraversa la cupola"), _testo_bonus(), _suono_del_bonus),
    ))
    return tuple(elenco)


class Regia:
    """La regia degli Ascolti: fa suonare i passi nel motore e li ferma. motore è il Motore del gioco; suoni,
    se c'è, i Suoni della finestra, da cui vengono i preset degli effetti, senza contarli; senza, gli effetti
    tacciono. proiettile è la forma delle impostazioni, una di PROIETTILE_FORME, altrimenti ValueError; rng il
    random.Random dei punti a caso; conta, se c'è, riceve "passi_ascoltati" a ogni passo fatto suonare.
    Attributi: passi, i Passo; proiettile; portante, il timbro dei bersagli degli Ascolti, il primo di
    BERSAGLIO_PORTANTI che non è quello del proiettile; in_corso, l'Ascolto che suona o ha suonato per
    ultimo, None dopo ferma; punti, i punti a caso, estratti al primo ascolto del loro passo e tenuti per
    tutta la regia, così lo spazio li fa riascoltare uguali."""

    def __init__(self, motore, suoni=None, proiettile=None, rng=None, conta=None):
        proiettile = PROIETTILE_FORMA if proiettile is None else proiettile
        if proiettile not in PROIETTILE_FORME:
            raise ValueError(f"Forma del proiettile sconosciuta: {proiettile!r}; le forme sono {', '.join(PROIETTILE_FORME)}.")
        self.motore = motore
        self.suoni = suoni
        self.proiettile = proiettile
        self.portante = portanti_senza(proiettile)[0]
        self.rng = random.Random() if rng is None else rng
        self.conta = conta
        self.passi = passi(proiettile)
        self.in_corso = None
        self.punti = None

    def suona(self, indice):
        """Ferma il suono di prima e fa partire quello del passo indice dal prossimo campione sicuro del
        motore; restituisce l'Ascolto."""
        self.ferma()
        passo = self.passi[indice]
        ascolto = passo.costruisci(self, self.motore.prossimo_campione())
        self.in_corso = ascolto
        if self.conta is not None:
            self.conta("passi_ascoltati")
        return ascolto

    def ferma(self):
        """Zittisce ogni voce del suono in corso, con la rampa ASCOLTI_RAMPA. Senza suono non fa niente."""
        ascolto, self.in_corso = self.in_corso, None
        if ascolto is None:
            return
        for voce in ascolto.voci:
            self.motore.ferma(voce, sfumatura=ASCOLTI_RAMPA)

    def sentito(self):
        """Vero se il suono in corso è finito anche per le orecchie: il campione udibile ha passato la sua fine."""
        return self.in_corso is not None and self.motore.campione_udibile() >= self.in_corso.fine

    def rivelazione(self):
        """Il testo da aggiungere dopo l'ascolto, come i punti a caso, quando il suono è finito; altrimenti vuoto."""
        if not self.sentito():
            return ""
        return self.in_corso.rivelazione

    def punti_a_caso(self):
        """I punti a caso, ASCOLTI_PUNTI, nella zona del bersaglio degli Ascolti, estratti come quelli del gioco,
        con fisica.Zona.punto_a_caso, una volta sola."""
        if self.punti is None:
            zona_del_bersaglio = zona()
            self.punti = tuple(tuple(float(c) for c in zona_del_bersaglio.punto_a_caso(self.rng)) for _punto in range(ASCOLTI_PUNTI))
        return self.punti

    def corsa(self, ascolto, parametro, da, a, inizio):
        """Un valore della mira che corre da da ad a, come col suo tasto giù (D29): il puntatore dal campione
        inizio, con gli altri due valori al centro (D31), a VELOCITA_DELLA_MIRA, le tacche dei decimi che
        attraversa e, se arriva a un capo, il suono fine_scala, tutti dal punto del puntatore. Restituisce il
        campione della fine, il suono del capo compreso."""
        fs = self.motore.fs
        verso = 1.0 if a >= da else -1.0
        secondi = abs(a - da) / VELOCITA_DELLA_MIRA
        partenza = inizio / fs

        def parametri(tempi_di_gioco, tempi_audio):
            tempi = np.atleast_1d(np.asarray(tempi_audio, dtype=float)) - partenza
            valori = dict.fromkeys(PARAMETRI, PARTENZA_DELLA_MIRA)
            valori[parametro] = da + verso * VELOCITA_DELLA_MIRA * np.clip(tempi, 0.0, secondi)
            return parametri_del_puntatore(*(valori[nome] for nome in PARAMETRI))

        fine = self.tono(ascolto, parametri, PUNTATORE_FORMA, inizio, secondi, attacco=SFUMATURA_STOP, sfumatura=SFUMATURA_STOP)
        for k in range(1, TACCHE):
            soglia = k / TACCHE
            if min(da, a) < soglia < max(da, a):
                self.buffer(ascolto, tacca_dell_asse(parametro, k, fs=fs), inizio + round(abs(soglia - da) / VELOCITA_DELLA_MIRA * fs))
        if a in (0.0, 1.0):
            pan = pan_della_direzione(a if parametro == "direzione" else PARTENZA_DELLA_MIRA)
            fine = max(fine, self.effetto(ascolto, "fine_scala", fine, pan=pan))
        return fine

    def tono(self, ascolto, parametri, forma, inizio, secondi, attacco=ASCOLTI_RAMPA, sfumatura=ASCOLTI_RAMPA):
        """Una VoceTono di forma, coi parametri, dal campione inizio per tanti secondi: entra con la rampa
        attacco ed esce con la sfumatura, che finisce proprio alla fine. Restituisce il campione della fine."""
        fs = self.motore.fs
        voce = VoceTono(parametri, forma, inizio)
        voce.attacco = attacco
        self.motore.aggiungi(voce)
        fine = inizio + round(secondi * fs)
        self.motore.ferma(voce, max(fine - round(sfumatura * fs), inizio), sfumatura)
        ascolto.voci.append(voce)
        ascolto.fine = max(ascolto.fine, fine)
        return fine

    def buffer(self, ascolto, buffer, inizio):
        """Un suono già pronto dal campione inizio; restituisce il campione della sua fine."""
        voce = self.motore.suona(buffer, inizio=inizio)
        ascolto.voci.append(voce)
        fine = inizio + len(voce.buffer)
        ascolto.fine = max(ascolto.fine, fine)
        return fine

    def effetto(self, ascolto, evento, inizio, posizione=None, pan=None):
        """Il preset di un evento dal campione inizio, collocato nello spazio se c'è la posizione, o nel punto
        del pan, preparato senza contarlo; restituisce il campione della fine, o inizio se l'evento non ha
        suono."""
        if self.suoni is None:
            return inizio
        suono = self.suoni.pronto(evento, posizione=posizione, conta_subito=False, pan=pan)
        if suono is None or not len(suono):
            return inizio
        ascolto.eventi.append((evento, inizio, None if posizione is None else tuple(float(c) for c in posizione)))
        return self.buffer(ascolto, suono, inizio)
