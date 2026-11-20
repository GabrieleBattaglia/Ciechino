"""
Gli Ascolti di Ciechino, il tutorial audio: i passi, ognuno col titolo, il testo e il suono, senza finestre.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-20 con la 0.9.0, dalla decisione D23 del 20 novembre 2026. Gabriele: «Facciamo una cosa
ibrida. Il gioco lo lasciamo così ma nelle impostazioni aggiungiamo la voce ascolti.» Un passo alla volta,
raccontato da un testo, si sentono il test stereo, il bersaglio al centro del cubo con la tratta orizzontale,
l'altezza e la distanza, tre punti a caso, la grandezza, le forme dell'oscillazione, i timbri, le tre
scivolate della mira, il proiettile in volo, i colpi a segno e il bonus che attraversa il cubo. Il dialogo
che li mostra sta in gui/dialoghi.py, e la finestra lo apre dal menu Impostazioni e dal menu Aiuto.
I suoni passano dal motore del gioco con le sue voci, VoceTono per i bersagli e il proiettile, VoceBuffer per
le scivolate e i preset, e con le stesse regole dello spazio sonoro: fisica.suono_di per panoramica, nota e
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
    CUBO_LATO,
    CUBO_X,
    CUBO_Y,
    CUBO_Z,
    DURATA_BONUS_MINIMA,
    FORME_LFO,
    MARGINE_DOPO_LA_SCIVOLATA,
    METRI_PER_OTTAVA,
    MOLTIPLICATORE_BONUS,
    PARAMETRI,
    PASSAGGI_DELLA_BARRA,
    PAUSA_FRA_BARRE,
    PROIETTILE_FORMA,
    PROIETTILE_FORME,
    RAGGIO_BONUS_FATTORE,
    SCIVOLATA_DURATA,
    SFUMATURA_STOP,
    TACCHE,
    VELOCITA_DEL_TEMPO,
)
from lingua import _, plurale
from motore_audio import VoceTono
from sintesi import db_in_lineare, scivolata
from statistiche import NOMI_DELLE_FORME, NOMI_DELLE_PORTANTI

# Il centro del cubo dei bersagli, da cui partono e dove tornano le tratte.
CENTRO = tuple((basso + alto) / 2 for basso, alto in (CUBO_X, CUBO_Y, CUBO_Z))
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
    """(minimi, massimi) della zona dove sta il centro del bersaglio degli Ascolti, quella del gioco."""
    return fisica.zona_del_centro(ASCOLTI_RAGGIO)


def _capi():
    """I punti dei capi delle tre tratte, al centro degli altri due assi: sinistra, destra, terra,
    soffitto, fronte e fondo della zona del bersaglio."""
    minimi, massimi = zona()
    x, y, z = CENTRO
    return {"sinistra": (minimi[0], y, z), "destra": (massimi[0], y, z), "terra": (x, minimi[1], z), "soffitto": (x, massimi[1], z),
            "fronte": (x, y, minimi[2]), "fondo": (x, y, massimi[2])}


def tappe_della_tratta(primo, secondo):
    """Le tappe di una tratta: dal centro al primo capo, poi al secondo, poi di nuovo al centro, con le soste."""
    capi = _capi()
    return [(CENTRO, ASCOLTI_SOSTA), (capi[primo], ASCOLTI_SOSTA_AI_CAPI), (capi[secondo], ASCOLTI_SOSTA_AI_CAPI), (CENTRO, ASCOLTI_SOSTA)]


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
    angolo = math.degrees(math.atan2(capi["destra"][0] - fisica.CANNONE[0], capi["destra"][2] - fisica.CANNONE[2]))
    return _testo(
        # Traduttori: distanza e altezza sono misure come «55 metri», raggio come «5 metri».
        _("Un bersaglio fermo al centro del cubo, {distanza} davanti a te e a {altezza} d'altezza, grande come quelli del primo livello, col "
          "raggio di {raggio}.").format(distanza=_metri(CENTRO[2]), altezza=_metri(CENTRO[1]), raggio=_metri(ASCOLTI_RAGGIO)),
        # Traduttori: lato è una misura, come «20 metri».
        _("Poi va a sinistra fino alla parete del cubo, {lato} a sinistra del centro, attraversa tutto il cubo fino alla parete di destra, e torna "
          "al centro.").format(lato=_metri(-capi["sinistra"][0])),
        # Traduttori: distanza è come «55 metri», angolo e massimo come «20 gradi» e «45 gradi».
        _("Ascolta la panoramica: il suono segue l'angolo del bersaglio visto dal cannone, e a {distanza} la parete del cubo sta a {angolo} dal "
          "centro, quindi il bersaglio non arriva mai tutto da un lato, che sarebbe a {massimo}. Più un bersaglio è lontano, più la sua panoramica "
          "si stringe verso il mezzo.").format(distanza=_metri(CENTRO[2]), angolo=testi.gradi(round(angolo)), massimo=testi.gradi(AZIMUT_DEL_PAN)),
    )


def _testo_altezza():
    capi = _capi()
    terra, soffitto = capi["terra"][1], capi["soffitto"][1]
    return _testo(
        # Traduttori: terra e soffitto sono altezze, come «5 metri» e «45 metri».
        _("Lo stesso bersaglio, dal centro, scende fino a terra, dove il suo centro sta a {terra} dal prato, perché il bersaglio resti tutto sopra "
          "il suolo; poi sale fino al soffitto del cubo, col centro a {soffitto} d'altezza, e torna al centro.").format(terra=_metri(terra),
                                                                                                                      soffitto=_metri(soffitto)),
        # Traduttori: ottava è una misura come «16,7 metri»; grave e acuto sono frequenze già scritte, come 271 e 1.429.
        _("Ascolta la nota: più è grave, più il bersaglio è in basso. Sale di un'ottava ogni {ottava}, da circa {grave} hertz a terra a circa "
          "{acuto} hertz al soffitto, e oscilla attorno a quella nota, perché il bersaglio è grande.").format(
            ottava=testi.metri(METRI_PER_OTTAVA), grave=testi.intero(fisica.frequenza_di(terra)), acuto=testi.intero(fisica.frequenza_di(soffitto))),
    )


def _testo_distanza():
    capi = _capi()
    fronte, fondo = capi["fronte"], capi["fondo"]
    decibel = fisica.guadagno_db_di(_distanza(fronte)) - fisica.guadagno_db_di(_distanza(fondo))
    return _testo(
        # Traduttori: vicino e lontano sono distanze, come «35 metri» e «75 metri».
        _("Lo stesso bersaglio, dal centro, viene vicino al massimo, sul fronte del cubo, a {vicino} davanti a te; poi va lontano al massimo, sul "
          "fondo, a {lontano}; e torna al centro.").format(vicino=_metri(fronte[2]), lontano=_metri(fondo[2])),
        # Traduttori: decibel è un numero già scritto, come 9.
        _("La nota e la panoramica restano quelle: cambia soltanto il volume, di circa {decibel} decibel fra il vicino e il lontano. Sono i due capi "
          "della scala della distanza, al centro del cubo.").format(decibel=testi.intero(decibel)),
        _("Il volume che senti dipende anche dal volume del computer, da quello di Ciechino e dalle cuffie o dagli altoparlanti: quando ne cambi "
          "uno, l'orecchio perde la sua misura della distanza. Per ritararlo torna qui, dal menu Impostazioni o dal menu Aiuto, e riascolta questo "
          "passo, il vicino e il lontano, prima di giocare."),
    )


def _testo_punti():
    return _testo(
        # Traduttori: punti è come «3 punti».
        _("Il bersaglio salta in {punti} presi a caso nel cubo, uno dopo l'altro, staccati da un breve silenzio. Prova a capire dove sta ognuno: "
          "a destra o a sinistra, in alto o in basso, vicino o lontano.").format(punti=plurale(ASCOLTI_PUNTI, "{n} punto", "{n} punti")),
        _("Quando l'ultimo ha finito di suonare, in fondo a questo testo compare dove erano, in metri, e il cursore va lì; poi lo spazio te li fa "
          "riascoltare, sapendo dove sono."),
    )


def _testo_grandezza():
    piccolo, primo, gigante = ASCOLTI_GRANDEZZE
    return _testo(
        # Traduttori: piccolo, primo e gigante sono raggi, come «0,5 metri», «5 metri» e «10 metri».
        _("Un bersaglio fermo al centro del cubo, che cresce: parte piccolissimo, col raggio di {piccolo}, il più piccolo del gioco; cresce fino a "
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
              "{per_passo} per passo, nello stesso punto, al centro del cubo, con la stessa grandezza e sempre il timbro {timbro}: ognuna suona "
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
        _("Il timbro della voce, la sua forma d'onda, ogni bersaglio lo prende a caso, fra {quanti}: qui li senti tutti, al centro del cubo, con "
          "la stessa oscillazione, ognuno per {secondi} secondi. In quest'ordine: {timbri}.").format(quanti=len(timbri), secondi=testi.numero(ASCOLTI_SUONO, 1),
                                                                                                  timbri=_nomi(timbri, NOMI_DELLE_PORTANTI)),
        # Traduttori: proiettile è un timbro, come «seno».
        _("Il timbro {proiettile} è quello del tuo proiettile, che scegli con Proiettile nel menu Impostazioni: nessun bersaglio lo usa, così il "
          "colpo non si confonde con loro. Con qualunque timbro la nota dice l'altezza allo stesso modo.").format(
            proiettile=NOMI_DELLE_PORTANTI.get(proiettile, proiettile)),
    )


def _testo_mira():
    return _testo(
        # Traduttori: pausa è una durata, come «1 secondo»; basso e alto sono frequenze già scritte, come 150 e 2.400.
        _("Le tre scivolate della mira, come in partita, ma senza sparare: prima il respiro che avvia la mira; poi, dopo {pausa} di silenzio, la "
          "direzione, un suono che va da tutto a sinistra a tutto a destra; poi l'alzo, che sale dal grave all'acuto, da {basso} a {alto} hertz; "
          "poi la potenza, che cresce dal quasi silenzio al pieno.").format(pausa=testi.durata(PAUSA_FRA_BARRE), basso=testi.intero(ALZO_FREQUENZE[0]),
                                                                           alto=testi.intero(ALZO_FREQUENZE[1])),
        # Traduttori: durata è come «3 secondi», tacche un numero, come 10.
        _("Ognuna comincia con tre bip, ciascuna coi suoi, e scivola per {durata}, con {tacche} tacche che salgono, una ogni decimo della scala. "
          "Qui nessuno la ferma: arriva in fondo, e lo scatto del valore preso passa alla seguente.").format(durata=testi.durata(SCIVOLATA_DURATA),
                                                                                                         tacche=TACCHE),
        # Traduttori: passaggi è un numero, come 5: quante volte al più suona una barra della mira in partita (D25).
        _("In partita lo spazio ferma ogni scivolata sul valore che senti in quell'istante; una barra che arriva in fondo senza lo spazio "
          "ricomincia da capo, al più {passaggi} volte, e poi il proiettile è perduto. Esc annulla la mira, e il proiettile resta nella "
          "riserva.").format(passaggi=PASSAGGI_DELLA_BARRA),
    )


def _testo_volo(proiettile):
    from partita import volo_d_esempio

    volo = volo_d_esempio()
    return _testo(
        # Traduttori: altezza e gittata sono misure come «25,3 metri»; secondi è un numero già scritto, come 4,9.
        _("Un tiro dal cannone verso il centro del cubo, dove non c'è nessun bersaglio: lo sparo ai tuoi piedi, poi il proiettile che sale fino a "
          "{altezza}, si allontana e ricade, e il tonfo a terra, a {gittata} davanti a te, dopo {secondi} secondi di volo.").format(
            altezza=_metri(volo.altezza_massima), gittata=_metri(volo.gittata), secondi=testi.numero(volo.durata / VELOCITA_DEL_TEMPO, 1)),
        # Traduttori: forma è un timbro, come «seno».
        _("Il proiettile suona in {forma}, la forma che hai scelto con Proiettile nel menu Impostazioni, e segue le regole dello spazio come i "
          "bersagli: la panoramica per il lato, la nota per l'altezza, il volume per la distanza. Lo sparo e il tonfo arrivano dal punto in cui "
          "avvengono.").format(forma=NOMI_DELLE_PORTANTI.get(proiettile, proiettile)),
    )


def _testo_colpi():
    capi = _capi()
    return _testo(
        # Traduttori: vicino e lontano sono distanze, come «35 metri» e «75 metri».
        _("Gli effetti dei colpi a segno, ognuno col suo jingle: prima il bersaglio colpito, tre note, vicino, sul fronte del cubo, a {vicino}, e "
          "lontano, sul fondo, a {lontano}; poi il bonus colpito, otto note più acute e più brillanti, vicino e lontano.").format(
            vicino=_metri(capi["fronte"][2]), lontano=_metri(capi["fondo"][2])),
        _("Anche gli effetti vengono dal punto dello spazio dove avvengono: più sono lontani, più suonano piano."),
    )


def _testo_bonus():
    livello = ASCOLTI_LIVELLO_DEL_BONUS
    return _testo(
        # Traduttori: durata è come «14 secondi», livello un numero, come 3.
        _("Il bersaglio bonus: compare col suo suono, entra dalla parete di sinistra, attraversa il cubo passando dal centro ed esce dalla parete "
          "di destra, in {durata}, come al livello {livello}, il primo dei bonus; quando esce, senti il suono del bonus uscito.").format(
            durata=testi.durata(durata_del_bonus(livello)), livello=livello),
        # Traduttori: minima è come «6 secondi»; volte e colpi sono numeri, come 5.
        _("Nel gioco il bonus entra da un lato preso a caso, a un'altezza e a una distanza prese a caso, e più si sale di livello più è svelto, "
          "fino a {minima}. Un centro sul bonus vale {volte} volte un bersaglio normale del suo livello, e regala {colpi} colpi.").format(
            minima=testi.durata(DURATA_BONUS_MINIMA), volte=MOLTIPLICATORE_BONUS, colpi=COLPI_BONUS),
        _("È l'ultimo passo: gli Ascolti si ritrovano quando vuoi, nel menu Impostazioni e nel menu Aiuto."),
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


def _suono_del_percorso(nome, primo, secondo):
    """Il costruttore di una tratta: il bersaglio dal centro al primo capo, al secondo e di nuovo al centro."""

    def costruisci(regia, inizio):
        ascolto = Ascolto(nome, inizio, inizio)
        percorso = Percorso(tappe_della_tratta(primo, secondo), ASCOLTI_VELOCITA)
        bersaglio = bersaglio_degli_ascolti(regia.portante)
        partenza = inizio / regia.motore.fs

        def parametri(tempi_di_gioco, tempi_audio):
            tempi = np.atleast_1d(np.asarray(tempi_audio, dtype=float)) - partenza
            return bersaglio.suono_in(percorso(tempi), tempi)

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


def _suono_della_mira(regia, inizio):
    """Come in partita: il respiro della mira, poi ogni barra dopo PAUSA_FRA_BARRE, che arriva in fondo, e
    lo scatto del valore preso quando la scivolata è scaduta, MARGINE_DOPO_LA_SCIVOLATA dopo la sua fine,
    che qui fa passare alla barra seguente; in partita, da D25, la barra scaduta ricomincerebbe."""
    ascolto = Ascolto("mira", inizio, inizio)
    fs = regia.motore.fs
    regia.effetto(ascolto, "mira_avviata", inizio)
    campione = inizio
    for parametro in PARAMETRI:
        barra = regia.scivolata(parametro)
        campione += round(PAUSA_FRA_BARRE * fs)
        regia.buffer(ascolto, barra.buffer, campione)
        campione += barra.fine_scivolata + round(MARGINE_DOPO_LA_SCIVOLATA * fs)
        regia.effetto(ascolto, "valore_preso", campione)
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
    capi = _capi()
    pausa = round(ASCOLTI_PAUSA * regia.motore.fs)
    campione = inizio
    for evento in ("colpito", "colpito_bonus"):
        for capo in ("fronte", "fondo"):
            campione = regia.effetto(ascolto, evento, campione, capi[capo]) + pausa
    return ascolto


def bonus_degli_ascolti(portante):
    """Il bonus che attraversa il cubo, da sinistra a destra, all'altezza e alla profondità del centro, col
    raggio e la velocità del livello ASCOLTI_LIVELLO_DEL_BONUS, nato al tempo zero."""
    livello = ASCOLTI_LIVELLO_DEL_BONUS
    raggio = raggio_del_livello(livello) * RAGGIO_BONUS_FATTORE
    rapidita = (CUBO_LATO + 2.0 * raggio) / durata_del_bonus(livello)
    return bersaglio_degli_ascolti(portante, raggio=raggio, punto=(CUBO_X[0] - raggio, CENTRO[1], CENTRO[2]), velocita=(rapidita, 0.0, 0.0), bonus=True)


def _suono_del_bonus(regia, inizio):
    """Come in partita: il bonus nasce col suo avviso, attraversa il cubo, e all'uscita la sua voce si ferma e
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
        Passo("orizzontale", _("Il centro del cubo e la tratta orizzontale"), _testo_orizzontale(), _suono_del_percorso("orizzontale", "sinistra", "destra")),
        Passo("altezza", _("L'altezza, a terra e al soffitto"), _testo_altezza(), _suono_del_percorso("altezza", "terra", "soffitto")),
        Passo("distanza", _("La distanza, vicino e lontano"), _testo_distanza(), _suono_del_percorso("distanza", "fronte", "fondo")),
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
        Passo("mira", _("Le tre scivolate della mira"), _testo_mira(), _suono_della_mira),
        Passo("volo", _("Il proiettile in volo"), _testo_volo(proiettile), _suono_del_volo),
        Passo("colpi", _("Il centro e il bonus, vicino e lontano"), _testo_colpi(), _suono_dei_colpi),
        Passo("bonus", _("Il bonus che attraversa il cubo"), _testo_bonus(), _suono_del_bonus),
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
        self._scivolate = {}

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
        """I punti a caso, ASCOLTI_PUNTI, uniformi nella zona del bersaglio degli Ascolti, estratti una volta sola."""
        if self.punti is None:
            minimi, massimi = zona()
            self.punti = tuple(tuple(self.rng.uniform(basso, alto) for basso, alto in zip(minimi, massimi, strict=True)) for _punto in range(ASCOLTI_PUNTI))
        return self.punti

    def scivolata(self, parametro):
        """La Scivolata di sintesi.scivolata per il parametro, quella della partita, fatta una volta sola."""
        if parametro not in self._scivolate:
            self._scivolate[parametro] = scivolata(parametro, fs=self.motore.fs)
        return self._scivolate[parametro]

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

    def effetto(self, ascolto, evento, inizio, posizione=None):
        """Il preset di un evento dal campione inizio, collocato nello spazio se c'è la posizione, preparato
        senza contarlo; restituisce il campione della fine, o inizio se l'evento non ha suono."""
        if self.suoni is None:
            return inizio
        suono = self.suoni.pronto(evento, posizione=posizione, conta_subito=False)
        if suono is None or not len(suono):
            return inizio
        ascolto.eventi.append((evento, inizio, None if posizione is None else tuple(float(c) for c in posizione)))
        return self.buffer(ascolto, suono, inizio)
