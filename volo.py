"""
Il volo del proiettile di Ciechino: dove va, che cosa colpisce, con quanta precisione, e di quanto
manca il bersaglio più vicino quando non colpisce nulla.
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


@dataclass(eq=False)
class Volo:
    """L'esito di un tiro. partenza è il tempo di gioco dello sparo, durata i secondi di volo fino
    all'impatto o a terra. In un centro ci sono colpito, precisione e distanza_al_centro; in un tiro
    a vuoto vicino, distanza_dal_vicino (dalla superficie del bersaglio, cioè la distanza fra i
    centri meno il raggio) e scarto (radiale, laterale, verticale) nel punto più vicino, se un
    bersaglio vivo c'era."""

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
    centro solo perché nessun passo cade proprio sul punto più vicino."""
    prima, mezzo, dopo = (float(d) ** 2 for d in distanze[indice - 1:indice + 2])
    curvatura = prima - 2.0 * mezzo + dopo
    if curvatura <= 0.0:
        return float(distanze[indice])
    quadrato = mezzo - (dopo - prima) ** 2 / (8.0 * curvatura)
    return math.sqrt(min(max(quadrato, 0.0), mezzo))


def _minimo_nel_tratto(distanze, ammessi, inizio):
    """Il minimo della distanza nel tratto che comincia a inizio e prosegue finché ammessi resta
    vero, raffinato quando il minimo non cade su un capo del tratto. Restituisce (minimo, indice)."""
    usciti = np.flatnonzero(~ammessi[inizio:])
    fine = inizio + (int(usciti[0]) if usciti.size else len(ammessi) - inizio)
    indice = inizio + int(np.argmin(distanze[inizio:fine]))
    if inizio < indice < fine - 1:
        return _minimo_raffinato(distanze, indice), indice
    return float(distanze[indice]), indice


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
    contatto. Senza contatto il volo finisce a terra, e si cerca il bersaglio passato più vicino."""
    partenza = float(partenza)
    tempo_a_terra = traiettoria.tempo_a_terra()
    passi = max(1, math.ceil(tempo_a_terra / PASSO_DEL_VOLO))
    tempi = np.linspace(0.0, tempo_a_terra, passi + 1)
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
        bersaglio, _centri, distanze, vivi, soglia, (durata, indice) = esami[ordine]
        distanza_al_centro, _ = _minimo_nel_tratto(distanze, vivi & (distanze <= soglia), indice)
        precisione = min(max(1.0 - distanza_al_centro / soglia, 0.0), 1.0)
        punto_finale = traiettoria.posizione(durata)
        return _volo(traiettoria, partenza, durata, punto_finale, colpito=bersaglio, precisione=precisione, distanza_al_centro=distanza_al_centro)
    vicino = distanza_dal_vicino = scarto = None
    # Il confronto fra i bersagli si fa sui passi, uguale per tutti; il raffinamento viene dopo.
    migliore = math.inf
    for bersaglio, centri, distanze, vivi, _soglia, _toccato in esami:
        if not vivi.any():
            continue
        dalla_superficie = np.where(vivi, distanze - bersaglio.raggio, np.inf)
        indice = int(np.argmin(dalla_superficie))
        if dalla_superficie[indice] < migliore:
            migliore = float(dalla_superficie[indice])
            # Il minimo si raffina come nei centri, ma soltanto se anche i passi accanto sono vivi.
            minimo = float(distanze[indice])
            if 0 < indice < len(vivi) - 1 and vivi[indice - 1] and vivi[indice + 1]:
                minimo = _minimo_raffinato(distanze, indice)
            vicino = bersaglio
            distanza_dal_vicino = minimo - bersaglio.raggio
            scarto = tuple(float(c) for c in fisica.scomponi(posizioni[indice] - centri[indice], centri[indice]))
    return _volo(traiettoria, partenza, tempo_a_terra, traiettoria.punto_a_terra(), vicino=vicino, distanza_dal_vicino=distanza_dal_vicino, scarto=scarto)


def _volo(traiettoria, partenza, durata, punto_finale, colpito=None, precisione=None, distanza_al_centro=None, vicino=None, distanza_dal_vicino=None, scarto=None):
    """Compone il Volo, con le misure del tragitto che valgono per i centri e per i tiri a vuoto."""
    origine = traiettoria.origine
    gittata = math.hypot(float(punto_finale[0] - origine[0]), float(punto_finale[2] - origine[2]))
    return Volo(traiettoria=traiettoria, partenza=partenza, durata=float(durata), colpito=colpito, punto_finale=np.asarray(punto_finale, dtype=float),
                precisione=precisione, distanza_al_centro=distanza_al_centro, vicino=vicino, distanza_dal_vicino=distanza_dal_vicino, scarto=scarto,
                altezza_massima=_altezza_massima(traiettoria, durata), gittata=gittata, laterale_finale=float(punto_finale[0]))


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
