"""
Il volo del proiettile di Ciechino: dove va, che cosa colpisce, con quanta precisione, di quanto
manca il bersaglio più vicino quando non colpisce nulla, e le misure dell'artigliere rispetto al
bersaglio colpito o mancato, per il sommario del resoconto (D21).
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Niente wx e niente audio. Al momento dello sparo il volo è deterministico: la traiettoria è una
parabola esatta e i bersagli hanno la posizione in funzione del tempo di gioco, quindi tutto il
tragitto si esamina subito, vettorizzato con numpy a passi di PASSO_DEL_VOLO, e il motore riceve già
l'istante dell'impatto, da programmare al campione giusto.
"""

import math
from dataclasses import dataclass

import numpy as np

import fisica
from bersagli import Bersaglio
from costanti import PASSO_DEL_VOLO, PROIETTILE_DB, RAGGIO_PROIETTILE

# Le misure dell'artigliere si cercano prima sui passi del volo, poi fra i due passi che contengono
# l'istante giusto, per bisezione, fino a questa finezza in secondi. Non è una taratura del gioco:
# la parabola e il moto dei bersagli si calcolano esatti in qualunque istante, quindi la misura non
# dipende dal passo, e un miliardesimo di secondo vale meno di un decimo di millimetro.
FINEZZA_DELLE_MISURE = 1e-9


@dataclass(frozen=True)
class Misure:
    """Le misure dell'artigliere di un tiro rispetto a un bersaglio, quelle del sommario del
    resoconto (D21): Gabriele le ha scelte il 19 novembre 2026 perché ogni segno porti alla
    correzione giusta, in salita come in discesa.
    L'arrivo è il primo istante in cui il proiettile arriva alla distanza orizzontale dal cannone del
    centro del bersaglio. Lì, contro il centro del bersaglio in quello stesso istante, si misurano il
    laterale, orizzontale e perpendicolare alla direzione del bersaglio vista dal cannone, positivo a
    destra, come in fisica.scomponi; il verticale, positivo sopra; e l'inclinazione del proiettile,
    in gradi, positiva in salita, zero nell'apice, negativa in discesa. La ricaduta è il primo
    istante in cui il proiettile, scendendo, ripassa dall'alto in basso l'altezza del centro del
    bersaglio in quello stesso istante. Lì la profondità è la distanza orizzontale dal cannone del
    proiettile meno quella del bersaglio, positiva se lungo. arrivo e ricaduta sono tempi di volo
    contati dallo sparo.
    I segni dicono la correzione, nei bersagli fermi e in quelli in moto, in salita e in discesa: il
    laterale positivo vuol dire meno direzione; il verticale positivo meno potenza, e meno alzo
    soltanto nei tiri tesi, perché nei tiri arcuati, più ripidi del tiro più dolce, meno alzo
    allunga il tiro e lo alza; la profondità positiva meno potenza. Un solo caso fa disaccordo fra
    due segni: il tiro ripido che all'arrivo sta ancora salendo, sotto il bersaglio, e ricade oltre,
    sotto e lungo insieme.
    Le regole dei casi limite.
    Le misure guardano la parabola intera, fino a terra: in un centro anche il tratto dopo l'impatto,
    come la precisione, quindi l'arrivo e la ricaduta possono venire dopo la fine del volo.
    Il bersaglio sta dove la sua posizione lo mette in ogni istante. Per un bersaglio in moto le due
    distanze dal cannone cambiano col tempo, e l'arrivo è il primo istante in cui quella del
    proiettile raggiunge quella del bersaglio; un bonus già uscito dal cubo prosegue dritto, come nel
    suo moto, e le misure dicono dove sarebbe stato.
    Un tiro che cade prima di arrivare alla distanza del bersaglio non ha l'arrivo: arrivo, laterale,
    verticale e inclinazione sono None. Un tiro che, scendendo, resta sempre sotto il centro del
    bersaglio, perché non arriva mai alla sua altezza o la supera soltanto salendo, non ha la
    ricaduta: ricaduta e profondità sono None. Un bersaglio fermo, superato salendo, si ripassa per
    forza scendendo, perché il proiettile finisce a terra: lo si supera soltanto salendo quando il
    bersaglio sale più in fretta del proiettile.
    Un bersaglio proprio sopra il cannone ha l'arrivo nell'istante dello sparo, e la sua direzione,
    che non c'è, si prende davanti, come in fisica.scomponi; un bersaglio quasi sopra il cannone ha
    l'arrivo subito dopo lo sparo, e il laterale, lì, non supera la sua distanza orizzontale dal
    cannone. Nel cubo dei bersagli, che comincia a CUBO_DISTANZA metri, non succede mai."""

    arrivo: float | None
    laterale: float | None
    verticale: float | None
    inclinazione: float | None
    ricaduta: float | None
    profondita: float | None

    def __post_init__(self):
        dell_arrivo = (self.arrivo, self.laterale, self.verticale, self.inclinazione)
        if any(valore is None for valore in dell_arrivo) and any(valore is not None for valore in dell_arrivo):
            raise ValueError("Le misure dell'arrivo, laterale, verticale e inclinazione, ci sono tutte insieme o mancano tutte insieme.")
        if (self.ricaduta is None) != (self.profondita is None):
            raise ValueError("La profondità c'è soltanto con la ricaduta.")

    @property
    def scarto(self):
        """Le tre misure in metri nell'ordine di fisica.scomponi, (profondità, laterale, verticale),
        con None per quelle che mancano: è lo scarto che il resoconto dice nella riga del mancato di
        D6, nell'ordine del sommario."""
        return (self.profondita, self.laterale, self.verticale)


@dataclass(eq=False)
class Volo:
    """L'esito di un tiro. partenza è il tempo di gioco dello sparo, durata i secondi di volo fino
    all'impatto o a terra. In un centro ci sono colpito, precisione e distanza_al_centro; in un tiro
    a vuoto vicino, distanza_dal_vicino (dalla superficie del bersaglio, cioè la distanza fra i
    centri meno il raggio) e scarto (radiale, laterale, verticale) nel punto più vicino, se un
    bersaglio vivo c'era. misure, in fondo e di serie None, così i Volo costruiti senza restano
    validi, sono le Misure dell'artigliere rispetto al bersaglio colpito o al vicino (D21); senza
    bersagli vivi restano None."""

    traiettoria: fisica.Traiettoria
    partenza: float
    durata: float
    colpito: Bersaglio | None
    punto_finale: np.ndarray
    precisione: float | None
    distanza_al_centro: float | None
    vicino: Bersaglio | None
    distanza_dal_vicino: float | None
    scarto: tuple[float, float, float] | None
    altezza_massima: float
    gittata: float
    laterale_finale: float
    misure: Misure | None = None

    @property
    def fine(self):
        """Il tempo di gioco della fine del volo, da programmare nel motore."""
        return self.partenza + self.durata


def _contatto(distanze, vivi, soglia, tempi):
    """Il primo istante in cui il proiettile tocca il bersaglio vivo, come (tempo di volo, indice
    del primo passo dentro), oppure None. Fra il passo fuori e quello dentro il tempo si interpola
    linearmente sulla distanza; se il passo prima il bersaglio non era ancora vivo, il contatto è
    nel passo stesso in cui nasce."""
    dentro = vivi & (distanze <= soglia)
    if not dentro.any():
        return None
    indice = int(np.argmax(dentro))
    if indice == 0 or not vivi[indice - 1]:
        return float(tempi[indice]), indice
    prima, dopo = float(distanze[indice - 1]), float(distanze[indice])
    frazione = (prima - soglia) / (prima - dopo)
    return float(tempi[indice - 1] + frazione * (tempi[indice] - tempi[indice - 1])), indice


def _minimo_raffinato(distanze, indice):
    """Il minimo della distanza attorno a un passo, con la parabola per i tre quadrati vicini: nel
    breve tratto di un passo il moto relativo è quasi rettilineo, e il quadrato della distanza quasi
    esattamente una parabola. Senza, un centro pieno potrebbe sembrare a qualche centimetro dal
    centro solo perché nessun passo cade proprio sul punto più vicino. Restituisce (minimo, frazione):
    frazione dice dove sta il vertice della parabola, in passi dal passo indice, fra meno mezzo e
    mezzo, perché il passo è il più vicino dei tre; lo scarto di un tiro a vuoto si prende proprio lì."""
    prima, mezzo, dopo = (float(d) ** 2 for d in distanze[indice - 1:indice + 2])
    curvatura = prima - 2.0 * mezzo + dopo
    if curvatura <= 0.0:
        return float(distanze[indice]), 0.0
    quadrato = mezzo - (dopo - prima) ** 2 / (8.0 * curvatura)
    frazione = min(max((prima - dopo) / (2.0 * curvatura), -0.5), 0.5)
    return math.sqrt(min(max(quadrato, 0.0), mezzo)), frazione


def _minimo_nel_tratto(distanze, ammessi, inizio):
    """Il minimo della distanza nel tratto che comincia a inizio e prosegue finché ammessi resta
    vero, raffinato quando il minimo non cade su un capo del tratto."""
    usciti = np.flatnonzero(~ammessi[inizio:])
    fine = inizio + (int(usciti[0]) if usciti.size else len(ammessi) - inizio)
    indice = inizio + int(np.argmin(distanze[inizio:fine]))
    if inizio < indice < fine - 1:
        return _minimo_raffinato(distanze, indice)[0]
    return float(distanze[indice])


def _tempo_fra_i_passi(tempi, indice, frazione):
    """Il tempo di volo a frazione di passo dal passo indice, verso il seguente se è positiva."""
    if frazione > 0.0:
        return float(tempi[indice] + frazione * (tempi[indice + 1] - tempi[indice]))
    if frazione < 0.0:
        return float(tempi[indice] + frazione * (tempi[indice] - tempi[indice - 1]))
    return float(tempi[indice])


def _scarto(traiettoria, bersaglio, partenza, tempo):
    """Lo scarto (radiale, laterale, verticale) di fisica.scomponi al tempo di volo dato: il
    proiettile meno il centro del bersaglio dove sta in quello stesso istante."""
    centro = np.asarray(bersaglio.posizione(partenza + tempo), dtype=float)
    return tuple(float(c) for c in fisica.scomponi(traiettoria.posizione(tempo) - centro, centro))


def _zero_fra(funzione, basso, alto):
    """Il punto fra basso e alto dove funzione passa da una parte all'altra dello zero, per
    bisezione, fino a FINEZZA_DELLE_MISURE: funzione(basso) e funzione(alto) stanno da parti
    diverse, e lo zero conta dalla parte dei positivi."""
    sopra_in_basso = funzione(basso) >= 0.0
    while alto - basso > FINEZZA_DELLE_MISURE:
        mezzo = 0.5 * (basso + alto)
        if not basso < mezzo < alto:
            break
        if (funzione(mezzo) >= 0.0) == sopra_in_basso:
            basso = mezzo
        else:
            alto = mezzo
    return 0.5 * (basso + alto)


def _misure(traiettoria, bersaglio, partenza, tempi, posizioni, centri):
    """Le Misure dell'artigliere, dai passi del volo già calcolati: tempi dallo sparo fino a terra,
    le posizioni del proiettile e i centri del bersaglio in quei tempi. Sui passi si cerca il primo
    cambio, poi lo si stringe per bisezione con la parabola e il moto esatti."""
    def proiettile(t):
        return traiettoria.posizione(t)

    def centro(t):
        return np.asarray(bersaglio.posizione(partenza + t), dtype=float)

    def avanzo(t):
        """Di quanto il proiettile è più lontano dal cannone del bersaglio, in orizzontale."""
        return fisica.distanza_orizzontale(proiettile(t)) - fisica.distanza_orizzontale(centro(t))

    def quota(t):
        """Di quanto il proiettile sta più in alto del centro del bersaglio."""
        return float(proiettile(t)[1] - centro(t)[1])

    arrivo = laterale = verticale = inclinazione = None
    arrivati = fisica.distanza_orizzontale(posizioni) >= fisica.distanza_orizzontale(centri)
    if arrivati.any():
        indice = int(np.argmax(arrivati))
        arrivo = float(tempi[0]) if indice == 0 else _zero_fra(avanzo, float(tempi[indice - 1]), float(tempi[indice]))
        _radiale, laterale, verticale = _scarto(traiettoria, bersaglio, partenza, arrivo)
        inclinazione = float(traiettoria.inclinazione(arrivo))
    ricaduta = profondita = None
    # La discesa comincia nell'apice: lì e nei passi che lo seguono si cerca il primo passaggio dal
    # sopra al sotto il centro del bersaglio, e a terra il proiettile sta sotto ogni bersaglio.
    tempo_apice, _altezza = traiettoria.apice()
    dopo = np.flatnonzero(tempi > tempo_apice)
    discesa = np.concatenate(([tempo_apice], tempi[dopo]))
    sopra = np.concatenate(([quota(tempo_apice)], posizioni[dopo, 1] - centri[dopo, 1])) >= 0.0
    cambi = np.flatnonzero(sopra[:-1] & ~sopra[1:])
    if cambi.size:
        passo = int(cambi[0])
        ricaduta = _zero_fra(quota, float(discesa[passo]), float(discesa[passo + 1]))
        profondita = float(avanzo(ricaduta))
    return Misure(arrivo=arrivo, laterale=laterale, verticale=verticale, inclinazione=inclinazione, ricaduta=ricaduta, profondita=profondita)


def _tempi_del_volo(traiettoria):
    """I tempi di volo dei passi, dallo sparo a terra compresi, a passi di PASSO_DEL_VOLO o appena
    meno, perché l'ultimo cada proprio a terra."""
    tempo_a_terra = traiettoria.tempo_a_terra()
    passi = max(1, math.ceil(tempo_a_terra / PASSO_DEL_VOLO))
    return np.linspace(0.0, tempo_a_terra, passi + 1)


def misure_dell_artigliere(traiettoria, bersaglio, partenza):
    """Le Misure dell'artigliere del tiro sparato al tempo di gioco partenza, rispetto a quel
    bersaglio, sulla parabola intera fino a terra (D21). calcola_volo le dà al bersaglio colpito o al
    vicino; da sola serve alle prove e ai banchi."""
    partenza = float(partenza)
    tempi = _tempi_del_volo(traiettoria)
    return _misure(traiettoria, bersaglio, partenza, tempi, traiettoria.posizione(tempi), bersaglio.posizione(partenza + tempi))


def _altezza_massima(traiettoria, durata):
    """L'altezza più alta raggiunta nei primi durata secondi di volo."""
    tempo_apice, altezza_apice = traiettoria.apice()
    if tempo_apice <= durata:
        return altezza_apice
    return float(traiettoria.posizione(durata)[1])


def calcola_volo(traiettoria, bersagli, partenza):
    """Il volo di un tiro sparato al tempo di gioco partenza fra i bersagli dati. Si esamina il
    tragitto fino a terra a passi di PASSO_DEL_VOLO, e per ogni bersaglio soltanto gli istanti in
    cui è vivo. Il contatto è il primo istante in cui la distanza fra i centri scende a raggio più
    RAGGIO_PROIETTILE; se due bersagli si toccano nello stesso passo vince il contatto più precoce,
    e a parità quello che viene prima nell'elenco. La precisione è 1 meno la distanza minima fra i
    centri nel tratto in cui il proiettile resterebbe dentro il bersaglio, divisa per la soglia del
    contatto. Senza contatto il volo finisce a terra, e si cerca il bersaglio passato più vicino,
    con lo scarto nel punto più vicino, preso al vertice della parabola quando il minimo cade fra due
    passi. In tutti e due i casi il Volo porta le misure dell'artigliere di D21, rispetto al
    bersaglio colpito o al vicino."""
    partenza = float(partenza)
    tempi = _tempi_del_volo(traiettoria)
    tempo_a_terra = float(tempi[-1])
    posizioni = traiettoria.posizione(tempi)
    tempi_di_gioco = partenza + tempi
    esami = []
    for bersaglio in bersagli:
        centri = bersaglio.posizione(tempi_di_gioco)
        differenze = posizioni - centri
        distanze = np.sqrt(np.einsum("ij,ij->i", differenze, differenze))
        vivi = np.asarray(bersaglio.vivo(tempi_di_gioco), dtype=bool)
        soglia = bersaglio.raggio + RAGGIO_PROIETTILE
        esami.append((bersaglio, centri, distanze, vivi, soglia, _contatto(distanze, vivi, soglia, tempi)))
    toccati = [(esame[5][0], ordine) for ordine, esame in enumerate(esami) if esame[5] is not None]
    if toccati:
        _, ordine = min(toccati)
        bersaglio, centri, distanze, vivi, soglia, (durata, indice) = esami[ordine]
        distanza_al_centro = _minimo_nel_tratto(distanze, vivi & (distanze <= soglia), indice)
        precisione = min(max(1.0 - distanza_al_centro / soglia, 0.0), 1.0)
        punto_finale = traiettoria.posizione(durata)
        misure = _misure(traiettoria, bersaglio, partenza, tempi, posizioni, centri)
        return _volo(traiettoria, partenza, durata, punto_finale, colpito=bersaglio, precisione=precisione, distanza_al_centro=distanza_al_centro, misure=misure)
    vicino = distanza_dal_vicino = scarto = misure = centri_del_vicino = None
    # Il confronto fra i bersagli si fa sui passi, uguale per tutti; il raffinamento viene dopo.
    migliore = math.inf
    for bersaglio, centri, distanze, vivi, _soglia, _toccato in esami:
        if not vivi.any():
            continue
        dalla_superficie = np.where(vivi, distanze - bersaglio.raggio, np.inf)
        indice = int(np.argmin(dalla_superficie))
        if dalla_superficie[indice] < migliore:
            migliore = float(dalla_superficie[indice])
            # Il minimo si raffina come nei centri, ma soltanto se anche i passi accanto sono vivi;
            # lo scarto si prende nello stesso punto, il vertice della parabola, non sul passo.
            minimo, frazione = float(distanze[indice]), 0.0
            if 0 < indice < len(vivi) - 1 and vivi[indice - 1] and vivi[indice + 1]:
                minimo, frazione = _minimo_raffinato(distanze, indice)
            vicino = bersaglio
            distanza_dal_vicino = minimo - bersaglio.raggio
            scarto = _scarto(traiettoria, bersaglio, partenza, _tempo_fra_i_passi(tempi, indice, frazione))
            centri_del_vicino = centri
    # Le misure dell'artigliere si cercano una volta sola, per il vicino scelto.
    if vicino is not None:
        misure = _misure(traiettoria, vicino, partenza, tempi, posizioni, centri_del_vicino)
    return _volo(traiettoria, partenza, tempo_a_terra, traiettoria.punto_a_terra(), vicino=vicino, distanza_dal_vicino=distanza_dal_vicino, scarto=scarto,
                 misure=misure)


def _volo(traiettoria, partenza, durata, punto_finale, colpito=None, precisione=None, distanza_al_centro=None, vicino=None, distanza_dal_vicino=None, scarto=None,
          misure=None):
    """Compone il Volo, con le misure del tragitto che valgono per i centri e per i tiri a vuoto."""
    origine = traiettoria.origine
    gittata = math.hypot(float(punto_finale[0] - origine[0]), float(punto_finale[2] - origine[2]))
    return Volo(traiettoria=traiettoria, partenza=partenza, durata=float(durata), colpito=colpito, punto_finale=np.asarray(punto_finale, dtype=float),
                precisione=precisione, distanza_al_centro=distanza_al_centro, vicino=vicino, distanza_dal_vicino=distanza_dal_vicino, scarto=scarto,
                altezza_massima=_altezza_massima(traiettoria, durata), gittata=gittata, laterale_finale=float(punto_finale[0]), misure=misure)


def parametri_del_proiettile(volo):
    """La funzione parametri(tempi_di_gioco, tempi_audio) per la voce del proiettile: la posizione
    al tempo di volo, cioè il tempo di gioco meno la partenza tenuto fra 0 e la durata, diventa
    (frequenze, pan, guadagni lineari) con fisica.suono_di, più PROIETTILE_DB. Tre array lunghi come
    i tempi; il tempo dell'audio non conta, ma si accetta per avere la firma di ogni voce del motore."""
    traiettoria = volo.traiettoria
    partenza = float(volo.partenza)
    durata = float(volo.durata)
    guadagno_del_proiettile = 10.0 ** (PROIETTILE_DB / 20.0)

    def parametri(tempi_di_gioco, tempi_audio=None):
        gioco = np.atleast_1d(np.asarray(tempi_di_gioco, dtype=float))
        if tempi_audio is not None:
            gioco, _ = np.broadcast_arrays(gioco, np.atleast_1d(np.asarray(tempi_audio, dtype=float)))
        tempi_di_volo = np.clip(gioco - partenza, 0.0, durata)
        frequenze, pan, guadagni = fisica.suono_di(traiettoria.posizione(tempi_di_volo))
        return np.asarray(frequenze), np.asarray(pan), np.asarray(guadagni) * guadagno_del_proiettile

    return parametri
