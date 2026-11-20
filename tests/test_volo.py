"""
Le prove di volo.py: centri pieni e di striscio, tiri corti, lunghi, a destra e sopra, bersagli
mobili, bonus già usciti, la voce del proiettile e il tempo di calcolo; lo scarto nel punto più
vicino, confrontato con un minimo cercato senza i passi del volo, anche fra due passi; e le misure
dell'artigliere del sommario (D21), confrontate con una ricerca fine senza i passi del volo e, per i
bersagli fermi, con le formule chiuse della parabola, in salita e in discesa, coi segni che portano
alla correzione giusta e coi casi limite; e la precisione globale di D26 con la scala del terzo
collaudo, dal centro del bersaglio all'angolo più lontano del cubo, coi casi costruiti a mano: il
bersaglio nel centro del cubo e in un angolo, in moto, il bonus fuori dalla parete, il centro pieno
e il tiro che passa più lontano dell'angolo.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
"""

import itertools
import math
import random
import time

import numpy as np
import pytest

import fisica
import volo
from bersagli import Bersaglio, genera
from costanti import ALTEZZA_CANNONE, CUBO_X, CUBO_Y, CUBO_Z, GRAVITA, PROIETTILE_DB, RAGGIO_INIZIALE, RAGGIO_PROIETTILE

PARTENZA = 20.0
# Quante volte si ripete il calcolo nella prova del tempo: conta il più veloce, che non risente
# degli altri processi in corsa sulla macchina.
RIPETIZIONI_DEL_CRONOMETRO = 5
TEMPO_MASSIMO_DEL_CALCOLO = 0.05
# Con passi di 5 centesimi di secondo, cinquanta volte quelli del gioco, il vertice della parabola
# sbaglia di qualche millimetro: il centimetro basta a distinguerlo dal passo, che sbaglia di decimetri.
TOLLERANZA_FRA_DUE_PASSI = 0.01
# La ricerca fine delle misure dell'artigliere nelle prove: un campione ogni centomillesimo di
# secondo, e l'interpolazione lineare fra i due attorno al cambio.
PASSO_FINE = 1e-5
# Quanto le misure del volo possono scostarsi da quelle della ricerca fine: in metri e in gradi, e
# in secondi per i tempi.
TOLLERANZA_DELLE_MISURE = 1e-4
TOLLERANZA_DEI_TEMPI = 1e-6
# Il passo, nei valori da 0 a 1 delle scivolate, delle derivate nella prova dei segni.
PASSO_DELLE_DERIVATE = 1e-4
# La precisione globale di D26, dal terzo collaudo: gli otto angoli del cubo, il suo centro e mezza diagonale, la distanza dal centro
# del cubo al suo angolo più lontano, la più piccola possibile; quanto l'istante del passaggio più vicino può scostarsi da quello della
# ricerca senza i passi del volo.
ANGOLI_DEL_CUBO = np.array(list(itertools.product(CUBO_X, CUBO_Y, CUBO_Z)))
CENTRO_DEL_CUBO = ((CUBO_X[0] + CUBO_X[1]) / 2, (CUBO_Y[0] + CUBO_Y[1]) / 2, (CUBO_Z[0] + CUBO_Z[1]) / 2)
MEZZA_DIAGONALE = (CUBO_X[1] - CUBO_X[0]) * math.sqrt(3) / 2
TOLLERANZA_DEL_PASSAGGIO = 1e-4
# La distanza del passaggio più vicino del volo sbaglia al più di un decimo di millimetro, e la precisione globale di qualche
# milionesimo: si confronta fino a un centomillesimo.
TOLLERANZA_DELLA_GLOBALE = 1e-5


def _fermo(punto, raggio=1.0, **altro):
    return Bersaglio(numero=altro.pop("numero", 1), forma="seno", raggio=raggio, partenza=punto, **altro)


def _tiro_per(punto, alzo):
    """La traiettoria che a quell'alzo passa per il punto."""
    velocita = fisica.velocita_per(punto, alzo)
    assert not math.isnan(velocita)
    return fisica.Traiettoria(math.degrees(math.atan2(punto[0], punto[2])), alzo, velocita)


def _tiro_con_apice(punto):
    """La traiettoria che ha l'apice nel punto: lì la tangente è orizzontale."""
    salita = punto[1] - ALTEZZA_CANNONE
    verticale = math.sqrt(2 * GRAVITA * salita)
    orizzontale = math.hypot(punto[0], punto[2]) / (verticale / GRAVITA)
    return fisica.Traiettoria(math.degrees(math.atan2(punto[0], punto[2])), math.degrees(math.atan2(verticale, orizzontale)), math.hypot(verticale, orizzontale))


def _minimo_esatto(traiettoria, bersaglio, da=0.0, a=None):
    """Il tempo di volo della distanza minima fra il proiettile e il centro del bersaglio, dove il
    bersaglio sta in quell'istante, cercato senza i passi del volo: un campionamento a un decimillesimo
    di secondo, poi la sezione aurea attorno al campione migliore."""
    a = traiettoria.tempo_a_terra() if a is None else a

    def distanza(t):
        return float(np.linalg.norm(traiettoria.posizione(t) - bersaglio.posizione(PARTENZA + t)))

    tempi = np.arange(da, a, 1e-4)
    distanze = np.linalg.norm(traiettoria.posizione(tempi) - bersaglio.posizione(PARTENZA + tempi), axis=1)
    migliore = int(np.argmin(distanze))
    basso, alto = float(tempi[max(migliore - 1, 0)]), float(tempi[min(migliore + 1, len(tempi) - 1)])
    aurea = (math.sqrt(5.0) - 1.0) / 2.0
    for _giro in range(80):
        sinistra, destra = alto - aurea * (alto - basso), basso + aurea * (alto - basso)
        if distanza(sinistra) < distanza(destra):
            alto = destra
        else:
            basso = sinistra
    return (basso + alto) / 2.0


def _scarto_vero(traiettoria, bersaglio, tempo):
    """Lo scarto (radiale, laterale, verticale) del proiettile dal centro del bersaglio in quell'istante."""
    centro = bersaglio.posizione(PARTENZA + tempo)
    return fisica.scomponi(traiettoria.posizione(tempo) - centro, centro)


def _controlla_scarto(esito, traiettoria, bersaglio, tolleranza=1e-4):
    """Lo scarto di un tiro a vuoto è quello vero nel punto più vicino, nell'istante della distanza
    minima, contro il centro del bersaglio in quello stesso istante, e torna con la distanza dal
    vicino. Restituisce l'istante."""
    tempo = _minimo_esatto(traiettoria, bersaglio)
    assert esito.scarto == pytest.approx(_scarto_vero(traiettoria, bersaglio, tempo), abs=tolleranza)
    assert math.sqrt(sum(c * c for c in esito.scarto)) - bersaglio.raggio == pytest.approx(esito.distanza_dal_vicino, abs=tolleranza)
    return tempo


def _cambio_fine(tempi, valori, indice):
    """Il tempo del cambio di segno fra il campione indice meno uno e indice, interpolato."""
    prima, dopo = valori[indice - 1], valori[indice]
    return float(tempi[indice - 1] + (tempi[indice] - tempi[indice - 1]) * prima / (prima - dopo))


def _misure_vere(traiettoria, bersaglio, passo=PASSO_FINE):
    """D21: le misure dell'artigliere cercate in un altro modo da quello del volo: un campione ogni
    centomillesimo di secondo fino a terra, il primo cambio, e l'interpolazione lineare. Restituisce
    un dizionario coi campi di volo.Misure."""
    a_terra = traiettoria.tempo_a_terra()
    tempi = np.append(np.arange(0.0, a_terra, passo), a_terra)
    proiettile = traiettoria.posizione(tempi)
    centri = bersaglio.posizione(PARTENZA + tempi)
    avanzo = np.hypot(proiettile[:, 0], proiettile[:, 2]) - np.hypot(centri[:, 0], centri[:, 2])
    quota = proiettile[:, 1] - centri[:, 1]
    vere = dict.fromkeys(("arrivo", "laterale", "verticale", "inclinazione", "ricaduta", "profondita"))
    arrivati = np.flatnonzero(avanzo >= 0.0)
    if arrivati.size:
        indice = int(arrivati[0])
        arrivo = 0.0 if indice == 0 else _cambio_fine(tempi, avanzo, indice)
        centro = bersaglio.posizione(PARTENZA + arrivo)
        _radiale, laterale, verticale = fisica.scomponi(traiettoria.posizione(arrivo) - centro, centro)
        vere.update(arrivo=arrivo, laterale=laterale, verticale=verticale, inclinazione=traiettoria.inclinazione(arrivo))
    scende = traiettoria.inclinazione(tempi) <= 0.0
    ricadute = np.flatnonzero(scende[1:] & (quota[:-1] >= 0.0) & (quota[1:] < 0.0)) + 1
    if ricadute.size:
        ricaduta = _cambio_fine(tempi, quota, int(ricadute[0]))
        profondita = fisica.distanza_orizzontale(traiettoria.posizione(ricaduta)) - fisica.distanza_orizzontale(bersaglio.posizione(PARTENZA + ricaduta))
        vere.update(ricaduta=ricaduta, profondita=profondita)
    return vere


def _misure_chiuse(traiettoria, punto):
    """D21: l'arrivo e la ricaduta di un bersaglio fermo, dalle formule chiuse della parabola: la
    distanza orizzontale cresce col tempo alla velocità orizzontale, e la ricaduta è la radice più
    grande di origine più vy t meno g t al quadrato mezzi uguale all'altezza. None dove non ci sono."""
    orizzontale = math.hypot(traiettoria.iniziale[0], traiettoria.iniziale[2])
    arrivo = fisica.distanza_orizzontale(punto) / orizzontale
    verticale = float(traiettoria.iniziale[1])
    discriminante = verticale * verticale - 2.0 * GRAVITA * (punto[1] - ALTEZZA_CANNONE)
    ricaduta = (verticale + math.sqrt(discriminante)) / GRAVITA if discriminante >= 0.0 else None
    return (arrivo if arrivo <= traiettoria.tempo_a_terra() else None), ricaduta


def _controlla_misure(misure, traiettoria, bersaglio):
    """D21: le misure del volo sono quelle vere, che mancano dove mancano quelle della ricerca fine;
    per un bersaglio fermo anche l'arrivo e la ricaduta delle formule chiuse."""
    vere = _misure_vere(traiettoria, bersaglio)
    for nome, vera in vere.items():
        misura = getattr(misure, nome)
        if vera is None:
            assert misura is None, nome
            continue
        tolleranza = TOLLERANZA_DEI_TEMPI if nome in ("arrivo", "ricaduta") else TOLLERANZA_DELLE_MISURE
        assert misura == pytest.approx(float(vera), abs=tolleranza), nome
    if not bersaglio.mobile:
        arrivo, ricaduta = _misure_chiuse(traiettoria, bersaglio.posizione(0.0))
        assert (misure.arrivo is None) == (arrivo is None) and (misure.ricaduta is None) == (ricaduta is None)
        if arrivo is not None:
            assert misure.arrivo == pytest.approx(arrivo, abs=1e-8)
        if ricaduta is not None:
            assert misure.ricaduta == pytest.approx(ricaduta, abs=1e-8)
    assert misure.scarto == (misure.profondita, misure.laterale, misure.verticale)
    return misure


def _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio):
    assert esito.colpito is None
    assert esito.precisione is None
    assert esito.distanza_al_centro is None
    assert esito.durata == pytest.approx(traiettoria.tempo_a_terra())
    assert esito.punto_finale[1] == 0.0
    assert esito.vicino is bersaglio
    assert esito.distanza_dal_vicino > RAGGIO_PROIETTILE
    assert esito.fine == pytest.approx(PARTENZA + esito.durata)
    # Lo scarto nel punto più vicino, anche fra due passi, e le misure dell'artigliere rispetto allo
    # stesso bersaglio (D21).
    _controlla_scarto(esito, traiettoria, bersaglio)
    return _controlla_misure(esito.misure, traiettoria, bersaglio)


def test_centro_pieno_su_bersaglio_fermo():
    traiettoria = fisica.Traiettoria(6.0, 42.0, 31.0)
    centro = traiettoria.posizione(1.8)
    bersaglio = _fermo(centro, raggio=2.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    assert esito.colpito is bersaglio
    assert esito.precisione > 0.995
    assert esito.distanza_al_centro < 0.01
    assert esito.vicino is None
    assert esito.scarto is None
    assert esito.durata < 1.8
    # La fine è il contatto, a raggio più RAGGIO_PROIETTILE dal centro, sulla traiettoria.
    assert np.allclose(esito.punto_finale, traiettoria.posizione(esito.durata))
    assert np.linalg.norm(esito.punto_finale - centro) == pytest.approx(2.0 + RAGGIO_PROIETTILE, abs=1e-3)
    assert esito.laterale_finale == pytest.approx(esito.punto_finale[0])
    assert esito.gittata == pytest.approx(math.hypot(esito.punto_finale[0], esito.punto_finale[2]))
    assert esito.altezza_massima == pytest.approx(max(traiettoria.posizione(np.linspace(0, esito.durata, 5001))[:, 1]), abs=1e-4)
    assert esito.fine == pytest.approx(PARTENZA + esito.durata)
    # D21: le misure guardano la parabola intera, anche dopo il contatto. Il proiettile arriva alla
    # distanza del centro proprio nel centro, a 1,8 secondi, ancora in salita: laterale e verticale
    # nulli. La ricaduta, scendendo, all'altezza del centro, è simmetrica rispetto all'apice, e lì il
    # tiro è lungo: le misure dell'artigliere vogliono il tiro che arriva scendendo.
    misure = _controlla_misure(esito.misure, traiettoria, bersaglio)
    assert misure.arrivo == pytest.approx(1.8, abs=1e-8) and misure.arrivo > esito.durata
    assert misure.laterale == pytest.approx(0.0, abs=1e-6) and misure.verticale == pytest.approx(0.0, abs=1e-6)
    assert misure.inclinazione == pytest.approx(traiettoria.inclinazione(1.8)) and misure.inclinazione > 0.0
    apice, _altezza = traiettoria.apice()
    assert misure.ricaduta == pytest.approx(2.0 * apice - 1.8, abs=1e-8)
    orizzontale = math.hypot(traiettoria.iniziale[0], traiettoria.iniziale[2])
    assert misure.profondita == pytest.approx(orizzontale * (misure.ricaduta - 1.8), abs=1e-6) and misure.profondita > 10.0


def test_tiro_di_striscio():
    traiettoria = fisica.Traiettoria(-4.0, 38.0, 30.0)
    sul_tiro = traiettoria.posizione(1.5)
    tangente = traiettoria.iniziale + np.array((0.0, -GRAVITA * 1.5, 0.0))
    di_lato = np.cross(tangente, (0.0, 1.0, 0.0))
    di_lato /= np.linalg.norm(di_lato)
    raggio = 1.5
    centro = sul_tiro + di_lato * (raggio + RAGGIO_PROIETTILE) * 0.95
    bersaglio = _fermo(centro, raggio=raggio)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    assert esito.colpito is bersaglio
    assert esito.precisione < 0.1
    assert esito.distanza_al_centro == pytest.approx((raggio + RAGGIO_PROIETTILE) * 0.95, abs=1e-3)
    # D21: il bersaglio sta a sinistra del tiro, di lato rispetto al piano verticale della parabola,
    # appena più lontano dal cannone del punto a 1,5 secondi: il proiettile ci arriva subito dopo,
    # alla sua destra di quasi tutta la distanza della precisione, e quasi alla sua altezza.
    misure = _controlla_misure(esito.misure, traiettoria, bersaglio)
    assert 1.5 < misure.arrivo < 1.51
    assert misure.laterale == pytest.approx(esito.distanza_al_centro, rel=0.01)
    assert abs(misure.verticale) < 0.05
    assert misure.inclinazione == pytest.approx(traiettoria.inclinazione(1.5), abs=0.1)
    # Un po' più in là non si tocca più.
    lontano = _fermo(sul_tiro + di_lato * (raggio + RAGGIO_PROIETTILE) * 1.05, raggio=raggio)
    assert volo.calcola_volo(traiettoria, [lontano], PARTENZA).colpito is None


def test_tiro_corto():
    centro = (0.0, 12.0, 55.0)
    bersaglio = _fermo(centro)
    traiettoria = _tiro_per((0.0, 12.0, 52.0), 80.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio)
    radiale, laterale, verticale = esito.scarto
    assert radiale < -2.5
    assert abs(radiale) > abs(verticale)
    assert abs(laterale) < 1e-6
    assert esito.distanza_dal_vicino == pytest.approx(math.sqrt(radiale**2 + laterale**2 + verticale**2) - 1.0, abs=0.01)
    # D21: il tiro ripassa all'altezza del bersaglio scendendo a 52 metri dal cannone, 3 prima di lui,
    # e cade a terra prima di arrivare alla sua distanza: niente arrivo, quindi né laterale né
    # verticale né inclinazione, e la profondità vale proprio meno 3.
    misure = esito.misure
    assert misure.arrivo is None and misure.laterale is None and misure.verticale is None and misure.inclinazione is None
    assert misure.profondita == pytest.approx(-3.0, abs=1e-6)
    assert misure.scarto == (misure.profondita, None, None)


def test_tiro_lungo():
    centro = (0.0, 12.0, 55.0)
    bersaglio = _fermo(centro)
    traiettoria = _tiro_per((0.0, 12.0, 58.0), 80.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    misure = _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio)
    radiale, _, verticale = esito.scarto
    assert radiale > 2.5
    assert abs(radiale) > abs(verticale)
    # D21: lungo di 3 metri all'altezza del bersaglio, e alla sua distanza, a 80 gradi d'alzo, il
    # proiettile ricade quasi a piombo, ancora molto più in alto: sopra e lungo, meno potenza.
    assert misure.profondita == pytest.approx(3.0, abs=1e-6)
    assert misure.verticale > 10.0 and misure.inclinazione < -60.0
    assert misure.laterale == pytest.approx(0.0, abs=1e-9)


def test_tiro_a_destra():
    centro = (-5.0, 15.0, 50.0)
    bersaglio = _fermo(centro)
    traiettoria = _tiro_per((-2.0, 15.0, 50.0), 50.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio)
    radiale, laterale, verticale = esito.scarto
    assert laterale > 2.5
    assert abs(laterale) > abs(radiale)
    assert abs(laterale) > abs(verticale)
    # Il tiro passa a 3 metri dal centro, di lato: 2 dalla superficie di un bersaglio di raggio 1.
    assert esito.distanza_dal_vicino == pytest.approx(2.0, abs=0.05)
    # D21: alla distanza del bersaglio il proiettile è 3 metri alla sua destra: la distanza
    # orizzontale del bersaglio per il seno dell'angolo fra le due direzioni viste dal cannone.
    angolo = math.atan2(-2.0, 50.0) - math.atan2(-5.0, 50.0)
    assert esito.misure.laterale == pytest.approx(math.hypot(5.0, 50.0) * math.sin(angolo), abs=1e-6)
    assert esito.misure.laterale == pytest.approx(3.0, abs=0.01)


def test_tiro_sopra():
    centro = (4.0, 15.0, 50.0)
    bersaglio = _fermo(centro)
    traiettoria = _tiro_con_apice((4.0, 18.0, 50.0))
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio)
    radiale, laterale, verticale = esito.scarto
    assert verticale == pytest.approx(3.0, abs=0.01)
    assert abs(radiale) < 0.05
    assert abs(laterale) < 0.05
    assert esito.distanza_dal_vicino == pytest.approx(2.0, abs=1e-3)
    assert esito.altezza_massima == pytest.approx(18.0)
    # D21: il proiettile arriva alla distanza del bersaglio nell'apice, 3 metri sopra, con
    # l'inclinazione nulla, e ricade alla sua altezza oltre: sopra e lungo.
    misure = esito.misure
    assert misure.verticale == pytest.approx(3.0, abs=1e-6) and misure.laterale == pytest.approx(0.0, abs=1e-6)
    assert misure.arrivo == pytest.approx(traiettoria.apice()[0], abs=1e-8)
    assert misure.inclinazione == pytest.approx(0.0, abs=1e-6)
    assert misure.profondita > 0.0


def test_tiro_in_salita_e_in_discesa():
    """D21: lo stesso tiro passa sotto e sopra quattro bersagli uguali, 3 metri dalla traiettoria, due
    un secondo prima dell'apice e due un secondo dopo. Le misure dell'artigliere si prendono dove il
    proiettile arriva alla loro distanza, quindi proprio lì, in salita nei primi e in discesa negli
    altri, con la stessa inclinazione e il segno opposto, perché la parabola è simmetrica. La
    profondità si prende dove il proiettile ripassa scendendo alla loro altezza, quindi col segno che
    porta alla correzione giusta anche in salita: il tiro che passa sopra è lungo, e non corto come
    nel punto più vicino, dove lo scarto sta di traverso al volo."""
    traiettoria = fisica.Traiettoria(0.0, 60.0, 30.0)
    apice, _altezza = traiettoria.apice()
    orizzontale = math.hypot(traiettoria.iniziale[0], traiettoria.iniziale[2])
    misure = {}
    for quota in (3.0, -3.0):
        for verso in (-1.0, 1.0):
            bersaglio = _fermo(traiettoria.posizione(apice + verso) + np.array((0.0, quota, 0.0)))
            esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
            misure[quota, verso] = _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio)
            assert misure[quota, verso].arrivo == pytest.approx(apice + verso, abs=1e-8)
            assert misure[quota, verso].verticale == pytest.approx(-quota, abs=1e-6)
            # La ricaduta all'altezza del bersaglio: mezzo secondo al quadrato di gravità sotto
            # l'apice, meno la quota, cioè tanti secondi dopo l'apice quanti dice la radice.
            ricaduta = apice + math.sqrt(2.0 * (GRAVITA / 2.0 - quota) / GRAVITA)
            assert misure[quota, verso].ricaduta == pytest.approx(ricaduta, abs=1e-8)
            assert misure[quota, verso].profondita == pytest.approx(orizzontale * (ricaduta - apice - verso), abs=1e-6)
    for quota in (3.0, -3.0):
        salita, discesa = misure[quota, -1.0], misure[quota, 1.0]
        assert salita.inclinazione > 30.0 and discesa.inclinazione < -30.0
        assert salita.inclinazione == pytest.approx(-discesa.inclinazione, abs=1e-6)
    # Sopra vuol dire lungo, in salita come in discesa: meno potenza. Nel punto più vicino, in salita,
    # lo scarto diceva corto.
    assert misure[-3.0, -1.0].profondita > 0.0 and misure[-3.0, 1.0].profondita > 0.0
    bersaglio = _fermo(traiettoria.posizione(apice - 1.0) + np.array((0.0, -3.0, 0.0)))
    assert volo.calcola_volo(traiettoria, [bersaglio], PARTENZA).scarto[0] < 0.0
    # Sotto, in discesa, è corto; sotto in salita è il solo caso di disaccordo: il tiro ripido che
    # alla distanza del bersaglio sale ancora, sotto di lui, e ricade oltre, lungo.
    assert misure[3.0, 1.0].profondita < 0.0 < misure[3.0, -1.0].profondita


def test_il_vicino_e_il_piu_vicino_dalla_superficie():
    traiettoria = _tiro_con_apice((0.0, 20.0, 50.0))
    # Il primo è più vicino fra i centri, il secondo dalla superficie, perché è più grande.
    piccolo = _fermo((0.0, 24.0, 50.0), raggio=0.5, numero=1)
    grande = _fermo((0.0, 26.0, 50.0), raggio=4.0, numero=2)
    esito = volo.calcola_volo(traiettoria, [piccolo, grande], PARTENZA)
    assert esito.colpito is None
    assert esito.vicino is grande
    assert esito.distanza_dal_vicino == pytest.approx(2.0, abs=1e-3)


def test_vince_il_primo_contatto():
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    dopo = _fermo(traiettoria.posizione(2.0), numero=1)
    prima = _fermo(traiettoria.posizione(1.0), numero=2)
    esito = volo.calcola_volo(traiettoria, [dopo, prima], PARTENZA)
    assert esito.colpito is prima


def test_bersaglio_mobile_che_attraversa_la_traiettoria():
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    incrocio = traiettoria.posizione(1.6)
    rapidita = 4.0
    # Il bersaglio nasce prima dello sparo e arriva sulla traiettoria proprio quando passa il proiettile.
    nato = PARTENZA - 1.0
    partenza = incrocio - np.array((rapidita * (PARTENZA + 1.6 - nato), 0.0, 0.0))
    mobile = Bersaglio(numero=1, forma="sega", raggio=1.0, partenza=partenza, velocita=(rapidita, 0.0, 0.0), nato=nato)
    assert np.allclose(mobile.posizione(PARTENZA + 1.6), incrocio)
    esito = volo.calcola_volo(traiettoria, [mobile], PARTENZA)
    assert esito.colpito is mobile
    assert esito.precisione > 0.9
    # Fermo dov'era alla partenza non si colpisce, e nemmeno se arriva un secondo dopo.
    fermo = _fermo(mobile.posizione(PARTENZA))
    assert volo.calcola_volo(traiettoria, [fermo], PARTENZA).colpito is None
    in_ritardo = Bersaglio(numero=2, forma="sega", raggio=1.0, partenza=partenza - np.array((rapidita, 0.0, 0.0)), velocita=(rapidita, 0.0, 0.0), nato=nato)
    esito = volo.calcola_volo(traiettoria, [in_ritardo], PARTENZA)
    assert esito.colpito is None
    assert esito.vicino is in_ritardo
    assert esito.scarto[1] > 0.0


def test_misure_su_un_bersaglio_che_si_muove():
    """D21: con un bersaglio in moto le misure sono dal suo centro nello stesso istante, non da dove
    stava allo sparo, nei tiri a vuoto e nei centri."""
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    incrocio = traiettoria.posizione(1.6)
    rapidita = 4.0
    nato = PARTENZA - 1.0
    # Arriva sulla traiettoria un secondo dopo il proiettile, venendo da sinistra: il proiettile gli
    # passa davanti, alla sua destra.
    in_ritardo = Bersaglio(numero=1, forma="sega", raggio=1.0, partenza=incrocio - np.array((rapidita * (PARTENZA + 2.6 - nato), 0.0, 0.0)),
                           velocita=(rapidita, 0.0, 0.0), nato=nato)
    esito = volo.calcola_volo(traiettoria, [in_ritardo], PARTENZA)
    misure = _controlla_tiro_a_vuoto(esito, traiettoria, in_ritardo)
    assert misure.laterale > 2.5
    # Contro il centro di partenza il laterale sarebbe tutt'altro.
    allo_sparo = in_ritardo.posizione(PARTENZA)
    fermo = fisica.scomponi(traiettoria.posizione(misure.arrivo) - allo_sparo, allo_sparo)
    assert abs(fermo[1] - misure.laterale) > 1.0
    # Un centro sullo stesso moto, che arriva insieme al proiettile ma un metro più in alto.
    in_orario = Bersaglio(numero=2, forma="sega", raggio=1.5, partenza=incrocio + np.array((-rapidita * (PARTENZA + 1.6 - nato), 1.0, 0.0)),
                          velocita=(rapidita, 0.0, 0.0), nato=nato)
    centro = volo.calcola_volo(traiettoria, [in_orario], PARTENZA)
    assert centro.colpito is in_orario
    misure = _controlla_misure(centro.misure, traiettoria, in_orario)
    assert misure.verticale == pytest.approx(-1.0, abs=0.05) and abs(misure.laterale) < 0.05


def test_misure_di_un_bersaglio_che_va_e_viene():
    """D21: per un bersaglio in moto le due distanze dal cannone cambiano col tempo, e l'arrivo è il
    primo istante in cui quella del proiettile raggiunge la sua. Uno che viene incontro si raggiunge
    prima di quello fermo dov'era allo sparo; uno che scappa più in fretta del proiettile, mai."""
    traiettoria = fisica.Traiettoria(0.0, 70.0, 30.0)
    orizzontale = math.hypot(traiettoria.iniziale[0], traiettoria.iniziale[2])
    punto = np.array((0.0, 20.0, 75.0))
    incontro = Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=punto, velocita=(0.0, 0.0, -6.0), nato=PARTENZA)
    misure = _controlla_misure(volo.misure_dell_artigliere(traiettoria, incontro, PARTENZA), traiettoria, incontro)
    # La distanza del proiettile cresce, quella del bersaglio cala, che non arriva a rimbalzare: si
    # incontrano a 75 metri diviso la somma delle due velocità.
    assert misure.arrivo == pytest.approx(75.0 / (orizzontale + 6.0), abs=1e-8)
    # Fermo dov'era allo sparo, invece, il tiro cade prima della sua distanza.
    assert _misure_chiuse(traiettoria, punto)[0] is None
    assert volo.misure_dell_artigliere(traiettoria, _fermo(punto), PARTENZA).arrivo is None
    # Uno che scappa dritto, senza rimbalzi, come un bonus che non esce mai dai lati.
    scappa = Bersaglio(numero=2, forma="seno", raggio=1.0, partenza=(0.0, 20.0, 45.0), velocita=(0.0, 0.0, orizzontale + 1.0), nato=PARTENZA, bonus=True)
    misure = _controlla_misure(volo.misure_dell_artigliere(traiettoria, scappa, PARTENZA), traiettoria, scappa)
    assert misure.arrivo is None and misure.laterale is None


def test_scarto_e_misure_fra_due_passi(monkeypatch):
    """Con passi grossi il minimo cade fra due passi, e lo scarto del tiro a vuoto si prende lo stesso
    nel punto giusto, al vertice della parabola, non sul passo più vicino. Le misure dell'artigliere
    (D21) non dipendono dal passo: si stringono fra due passi con la parabola e il moto esatti."""
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    fermo = _fermo((-4.0, 20.0, 45.0))
    in_moto = Bersaglio(numero=2, forma="seno", raggio=1.0, partenza=(10.0, 20.0, 50.0), velocita=(-3.0, 1.0, -2.0), nato=PARTENZA - 1.0)
    fini = [volo.calcola_volo(traiettoria, [bersaglio], PARTENZA).misure for bersaglio in (fermo, in_moto)]
    monkeypatch.setattr(volo, "PASSO_DEL_VOLO", 0.05)
    for bersaglio, fine in zip((fermo, in_moto), fini, strict=True):
        esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
        assert esito.colpito is None
        tempo = _controlla_scarto(esito, traiettoria, bersaglio, tolleranza=TOLLERANZA_FRA_DUE_PASSI)
        # Sul passo più vicino lo scarto sbaglierebbe di decimetri: la prova guarda proprio fra due passi.
        passi = np.linspace(0.0, traiettoria.tempo_a_terra(), math.ceil(traiettoria.tempo_a_terra() / 0.05) + 1)
        sul_passo = passi[int(np.argmin(np.abs(passi - tempo)))]
        assert np.abs(np.array(_scarto_vero(traiettoria, bersaglio, sul_passo)) - np.array(esito.scarto)).max() > 0.1
        _controlla_misure(esito.misure, traiettoria, bersaglio)
        for nome in ("arrivo", "laterale", "verticale", "inclinazione", "ricaduta", "profondita"):
            assert getattr(esito.misure, nome) == pytest.approx(getattr(fine, nome), abs=1e-6), nome
    # Un centro a un metro dal centro di un bersaglio abbastanza grande perché il proiettile resti
    # dentro per qualche passo, col punto più vicino a 1,53 secondi, fra due passi.
    sul_tiro = traiettoria.posizione(1.53)
    tangente = traiettoria.iniziale + np.array((0.0, -GRAVITA * 1.53, 0.0))
    di_lato = np.cross(tangente, (0.0, 1.0, 0.0))
    di_lato /= np.linalg.norm(di_lato)
    largo = _fermo(sul_tiro + di_lato * 1.0, raggio=3.0)
    esito = volo.calcola_volo(traiettoria, [largo], PARTENZA)
    assert esito.colpito is largo
    assert esito.distanza_al_centro == pytest.approx(1.0, abs=1e-3)
    _controlla_misure(esito.misure, traiettoria, largo)
    passi = np.linspace(0.0, traiettoria.tempo_a_terra(), math.ceil(traiettoria.tempo_a_terra() / 0.05) + 1)
    assert np.abs(passi - 1.53).min() > 0.005


def test_misure_casi_limite():
    """D21, le regole dei casi limite del docstring di volo.Misure."""
    # Un tiro piatto e piano cade a terra prima della distanza del bersaglio e sotto la sua altezza:
    # nessuna misura.
    piatto = fisica.Traiettoria(0.0, 0.0, 10.0)
    lontano = _fermo((0.0, 15.0, 50.0))
    nessuna = _controlla_misure(volo.misure_dell_artigliere(piatto, lontano, PARTENZA), piatto, lontano)
    assert nessuna.scarto == (None, None, None) and nessuna.inclinazione is None and nessuna.ricaduta is None
    # Un tiro teso arriva alla distanza di un bersaglio alto, ma non alla sua altezza: niente ricaduta.
    teso = fisica.Traiettoria(0.0, 10.0, 45.0)
    alto = _fermo((0.0, 30.0, 40.0))
    misure = _controlla_misure(volo.misure_dell_artigliere(teso, alto, PARTENZA), teso, alto)
    assert misure.verticale < -20.0 and misure.profondita is None and misure.ricaduta is None
    # Un bersaglio che sale più in fretta del proiettile: il proiettile lo supera soltanto salendo, poi
    # resta sotto, e scendendo non ripassa mai alla sua altezza.
    ripido = fisica.Traiettoria(0.0, math.degrees(math.atan2(20.0, 15.0)), 25.0)
    sale = Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=(0.0, 3.0, 40.0), velocita=(0.0, 12.0, 0.0), nato=PARTENZA)
    tempi = np.linspace(0.0, ripido.apice()[0], 2001)
    assert (ripido.posizione(tempi)[:, 1] > sale.posizione(PARTENZA + tempi)[:, 1]).any()
    misure = _controlla_misure(volo.misure_dell_artigliere(ripido, sale, PARTENZA), ripido, sale)
    assert misure.ricaduta is None and misure.profondita is None and misure.verticale < 0.0
    # Un bersaglio proprio sopra il cannone: l'arrivo è lo sparo, col proiettile nel cannone, laterale
    # nullo e l'inclinazione dell'alzo.
    tiro = fisica.Traiettoria(10.0, 60.0, 25.0)
    sopra = _fermo((0.0, 20.0, 0.0))
    misure = _controlla_misure(volo.misure_dell_artigliere(tiro, sopra, PARTENZA), tiro, sopra)
    assert misure.arrivo == 0.0 and misure.laterale == 0.0
    assert misure.verticale == pytest.approx(ALTEZZA_CANNONE - 20.0) and misure.inclinazione == pytest.approx(60.0)
    assert misure.profondita > 0.0
    # Quasi sopra, a mezzo metro: l'arrivo subito dopo lo sparo, e il laterale non supera il mezzo metro.
    quasi = _fermo((0.3, 20.0, 0.4))
    misure = _controlla_misure(volo.misure_dell_artigliere(tiro, quasi, PARTENZA), tiro, quasi)
    assert misure.arrivo == pytest.approx(0.5 / math.hypot(tiro.iniziale[0], tiro.iniziale[2]), abs=1e-8)
    assert abs(misure.laterale) <= 0.5
    assert misure.laterale == pytest.approx(0.5 * math.sin(math.radians(10.0) - math.atan2(0.3, 0.4)), abs=1e-6)
    # Le misure dell'arrivo ci sono tutte o nessuna, e la profondità soltanto con la ricaduta.
    with pytest.raises(ValueError):
        volo.Misure(arrivo=1.0, laterale=None, verticale=0.0, inclinazione=0.0, ricaduta=None, profondita=None)
    with pytest.raises(ValueError):
        volo.Misure(arrivo=None, laterale=None, verticale=None, inclinazione=None, ricaduta=2.0, profondita=None)


def _misure_coi_valori(valori, bersaglio, partenza):
    return volo.misure_dell_artigliere(fisica.Traiettoria(*fisica.parametri_del_tiro(*valori)), bersaglio, partenza)


def _derivata(valori, indice, bersaglio, partenza, nome):
    """La derivata di una misura rispetto a uno dei tre valori della mira, o None se la misura sparisce."""
    su, giu = list(valori), list(valori)
    su[indice] += PASSO_DELLE_DERIVATE
    giu[indice] -= PASSO_DELLE_DERIVATE
    alta = getattr(_misure_coi_valori(su, bersaglio, partenza), nome)
    bassa = getattr(_misure_coi_valori(giu, bersaglio, partenza), nome)
    if alta is None or bassa is None:
        return None
    return (alta - bassa) / (2.0 * PASSO_DELLE_DERIVATE)


def test_i_segni_portano_alla_correzione_giusta():
    """D21: ogni segno porta alla correzione giusta, in salita come in discesa, coi bersagli fermi e in
    moto: il laterale cresce con la direzione, quindi positivo vuol dire meno direzione; il verticale e
    la profondità crescono con la potenza, quindi positivi vogliono dire meno potenza. Tiri a caso
    attorno al tiro giusto, con un seme fisso; lo script di verifica di ClaudIA ne ha provati
    seimila, senza un segno sbagliato."""
    rng = random.Random(1911)
    giudicati = {"salita": 0, "discesa": 0}
    for prova in range(36):
        livello = (1, 2, 3, 5, 9, 14)[prova % 6]
        bersaglio = genera(livello, rng, 1, PARTENZA)
        punto = bersaglio.posizione(PARTENZA + 2.0)
        giusti = fisica.soluzione(punto)
        assert giusti is not None
        valori = [min(1.0, max(0.0, g + rng.gauss(0.0, s))) for g, s in zip(giusti, (0.02, 0.05, 0.08), strict=True)]
        misure = _misure_coi_valori(valori, bersaglio, PARTENZA)
        fase = "salita" if misure.inclinazione is not None and misure.inclinazione > 0.0 else "discesa"
        for nome, indice in (("laterale", 0), ("verticale", 2), ("profondita", 2)):
            valore = getattr(misure, nome)
            if valore is None or abs(valore) < 0.05:
                continue
            derivata = _derivata(valori, indice, bersaglio, PARTENZA, nome)
            if derivata is not None:
                assert derivata > 0.0, (nome, livello, valori, misure)
                giudicati[fase] += 1
    # Anche un tiro in salita, voluto: troppo potente, alzo basso, sopra il bersaglio.
    traiettoria = fisica.Traiettoria(0.0, 30.0, 40.0)
    valori = fisica.valori_del_tiro(traiettoria.azimut, traiettoria.alzo, traiettoria.velocita)
    bersaglio = _fermo(traiettoria.posizione(1.0) + np.array((0.0, -2.0, 0.0)))
    misure = _misure_coi_valori(valori, bersaglio, PARTENZA)
    assert misure.inclinazione > 0.0 and misure.verticale > 0.0 and misure.profondita > 0.0
    for nome in ("verticale", "profondita"):
        assert _derivata(valori, 2, bersaglio, PARTENZA, nome) > 0.0
    assert giudicati["discesa"] > 20


def test_bonus_gia_uscito_non_si_colpisce():
    # Un tiro che passa fuori dal cubo, oltre la parete di destra.
    punto = (CUBO_X[1] + 5.0, 12.0, 45.0)
    traiettoria = _tiro_per(punto, 45.0)
    tempo = math.hypot(punto[0], punto[2]) / (traiettoria.velocita * math.cos(math.radians(45.0)))
    raggio = 1.0
    rapidita = 6.0
    nato = 0.0
    partenza = np.array(punto) - np.array((rapidita * (PARTENZA + tempo - nato), 0.0, 0.0))
    bonus = Bersaglio(numero=1, forma="quadra", raggio=raggio, partenza=partenza, velocita=(rapidita, 0.0, 0.0), nato=nato, bonus=True)
    # Senza l'uscita sarebbe proprio lì quando passa il proiettile.
    assert np.allclose(bonus.posizione(PARTENZA + tempo), punto)
    assert not bonus.vivo(PARTENZA + tempo)
    esito = volo.calcola_volo(traiettoria, [bonus], PARTENZA)
    assert esito.colpito is None
    assert esito.durata == pytest.approx(traiettoria.tempo_a_terra())
    # D21: per le misure il bonus già uscito prosegue dritto, come nel suo moto: il tiro arriva alla
    # sua distanza proprio dove sarebbe stato, quindi né a destra né a sinistra, né sopra né sotto.
    assert esito.vicino is bonus
    misure = _controlla_misure(esito.misure, traiettoria, bonus)
    assert misure.arrivo == pytest.approx(tempo, abs=1e-8)
    assert misure.laterale == pytest.approx(0.0, abs=1e-6) and misure.verticale == pytest.approx(0.0, abs=1e-6)


def test_bonus_non_ancora_nato_non_si_colpisce():
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    bonus = Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=traiettoria.posizione(1.0), nato=PARTENZA + 1.5, bonus=True)
    esito = volo.calcola_volo(traiettoria, [bonus], PARTENZA)
    assert esito.colpito is None
    # Non era mai vivo durante il volo prima di 1,5 secondi, e dopo era lontano: è il vicino ugualmente.
    assert esito.vicino is bonus


def test_senza_bersagli():
    traiettoria = fisica.Traiettoria(10.0, 30.0, 25.0)
    esito = volo.calcola_volo(traiettoria, [], PARTENZA)
    assert esito.colpito is None
    assert esito.vicino is None
    assert esito.distanza_dal_vicino is None
    assert esito.scarto is None
    assert esito.misure is None
    assert esito.durata == pytest.approx(traiettoria.tempo_a_terra())
    assert np.allclose(esito.punto_finale, traiettoria.punto_a_terra())
    assert esito.altezza_massima == pytest.approx(traiettoria.apice()[1])
    assert esito.gittata == pytest.approx(math.hypot(*traiettoria.punto_a_terra()[[0, 2]]))
    assert esito.laterale_finale > 0.0
    # Senza bersaglio niente passaggio più vicino, e niente precisione globale (D26).
    assert esito.tempo_del_passaggio is None and esito.centro_nel_passaggio is None and esito.precisione_globale is None


def _piu_lontano(punto):
    """La distanza fra il punto e il più lontano degli otto angoli del cubo, cercata angolo per angolo."""
    return float(np.max(np.linalg.norm(ANGOLI_DEL_CUBO - np.asarray(punto, dtype=float), axis=1)))


def _globale_attesa(esito, bersaglio):
    """La precisione globale che il terzo collaudo vuole, ricavata qui dalle sue parti: la distanza del passaggio più vicino, cercata
    senza i passi del volo, sulla distanza fra il centro del bersaglio in quell'istante e l'angolo più lontano, mai sotto zero."""
    tempo = _minimo_esatto(esito.traiettoria, bersaglio)
    assert esito.tempo_del_passaggio == pytest.approx(tempo, abs=TOLLERANZA_DEL_PASSAGGIO)
    centro = bersaglio.posizione(PARTENZA + tempo)
    assert esito.centro_nel_passaggio == pytest.approx(centro, abs=1e-3)
    distanza = float(np.linalg.norm(esito.traiettoria.posizione(tempo) - centro))
    assert esito.distanza_dal_centro == pytest.approx(distanza, abs=1e-4)
    return max(0.0, 1.0 - distanza / _piu_lontano(centro))


def _a_mano(bersaglio, distanza, tempo=0.0, colpito=False):
    """Un volo costruito a mano, passato a distanza metri dal centro del bersaglio, al tempo di volo dato."""
    campi = {"traiettoria": fisica.Traiettoria(0.0, 40.0, 30.0), "partenza": PARTENZA, "durata": 3.0, "punto_finale": np.array((0.0, 0.0, 70.0)),
             "scarto": None, "altezza_massima": 25.0, "gittata": 70.0, "laterale_finale": 0.0, "tempo_del_passaggio": tempo}
    if colpito:
        return volo.Volo(colpito=bersaglio, precisione=0.5, distanza_al_centro=distanza, vicino=None, distanza_dal_vicino=None, **campi)
    return volo.Volo(colpito=None, precisione=None, distanza_al_centro=None, vicino=bersaglio, distanza_dal_vicino=distanza - bersaglio.raggio, **campi)


def test_precisione_globale_nel_centro_e_in_un_angolo():
    """D26, il terzo collaudo di Gabriele: «Considero retta che va dal centro del bersaglio al punto più lontano dello spazio colpibile
    nel cubo.» Per un bersaglio nel centro del cubo quella retta è mezza diagonale, 43,3 metri, per uno in un angolo la diagonale
    intera, 86,6: lo stesso tiro vale di meno vicino al centro, dove sbagliare di tanto è più difficile. Oltre la retta si resta a zero."""
    nel_centro = _fermo(CENTRO_DEL_CUBO, raggio=RAGGIO_INIZIALE)
    nell_angolo = _fermo(ANGOLI_DEL_CUBO[0], raggio=RAGGIO_INIZIALE)
    for distanza in (0.0, 5.0, 10.0, 20.0, 47.0):
        assert _a_mano(nel_centro, distanza).precisione_globale == pytest.approx(max(0.0, 1.0 - distanza / MEZZA_DIAGONALE))
        assert _a_mano(nell_angolo, distanza).precisione_globale == pytest.approx(1.0 - distanza / (2.0 * MEZZA_DIAGONALE))
    # I numeri del manuale: a 10 metri il 76,9% nel centro e l'88,5% nell'angolo, a 47 metri lo 0% e il 45,7%.
    assert round(_a_mano(nel_centro, 10.0).precisione_globale, 3) == 0.769 and round(_a_mano(nell_angolo, 10.0).precisione_globale, 3) == 0.885
    assert _a_mano(nel_centro, 47.0).precisione_globale == 0.0 and round(_a_mano(nell_angolo, 47.0).precisione_globale, 3) == 0.457
    # Un centro, nel caso peggiore, il bersaglio nel centro del cubo sfiorato sul bordo, sta sopra 1 meno raggio più RAGGIO_PROIETTILE
    # diviso mezza diagonale: l'88,2% al primo livello.
    sfiorato = _a_mano(nel_centro, RAGGIO_INIZIALE + RAGGIO_PROIETTILE, colpito=True)
    assert sfiorato.precisione_globale == pytest.approx(1.0 - (RAGGIO_INIZIALE + RAGGIO_PROIETTILE) / MEZZA_DIAGONALE)
    assert round(sfiorato.precisione_globale, 3) == 0.882
    # Un Volo costruito senza l'istante del passaggio prende il bersaglio alla fine del volo, che per uno fermo è lo stesso punto.
    senza = _a_mano(nell_angolo, 10.0, tempo=None)
    assert senza.centro_nel_passaggio == pytest.approx(ANGOLI_DEL_CUBO[0]) and senza.precisione_globale == _a_mano(nell_angolo, 10.0).precisione_globale


def test_precisione_globale_del_centro_pieno():
    """D26: il proiettile che passa proprio per il centro del bersaglio, nel centro del cubo, vale quasi il 100%, e l'istante del
    passaggio più vicino è quello in cui ci passa, dopo il contatto, che avviene sulla superficie."""
    centro = np.array(CENTRO_DEL_CUBO)
    traiettoria = _tiro_per(centro, 45.0)
    bersaglio = _fermo(centro, raggio=RAGGIO_INIZIALE)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    assert esito.colpito is bersaglio and esito.distanza_al_centro < 1e-3
    assert esito.tempo_del_passaggio > esito.durata
    assert esito.precisione_globale == pytest.approx(_globale_attesa(esito, bersaglio), abs=TOLLERANZA_DELLA_GLOBALE)
    assert esito.precisione_globale > 0.9999


def test_precisione_globale_dei_tiri_a_vuoto_nel_centro_e_in_un_angolo():
    """D26: due tiri a vuoto veri, passati sopra un bersaglio nel centro del cubo e sopra uno vicino a un angolo; il secondo vale di
    più, perché da lì il cubo lasciava sbagliare di più. Un centro di striscio sul bersaglio del primo livello nel centro del cubo sta
    sopra l'88,2% del caso peggiore."""
    globali = []
    for punto in (np.array(CENTRO_DEL_CUBO), np.array((CUBO_X[0] + 1.0, 3.0, CUBO_Z[0] + 1.0))):
        # Il tiro mira otto metri più in alto del centro del bersaglio, e gli passa sopra.
        traiettoria = _tiro_per(punto + np.array((0.0, 8.0, 0.0)), 45.0)
        bersaglio = _fermo(punto, raggio=1.0)
        esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
        assert esito.colpito is None and esito.vicino is bersaglio and 1.0 < esito.distanza_dal_centro < 8.0 + 1e-6
        assert esito.precisione_globale == pytest.approx(_globale_attesa(esito, bersaglio), abs=TOLLERANZA_DELLA_GLOBALE)
        globali.append((esito.distanza_dal_centro, esito.precisione_globale, _piu_lontano(punto)))
    (d_centro, g_centro, lontano_centro), (d_angolo, g_angolo, lontano_angolo) = globali
    assert lontano_centro == pytest.approx(MEZZA_DIAGONALE) and lontano_angolo > 1.9 * MEZZA_DIAGONALE
    assert d_angolo / lontano_angolo < d_centro / lontano_centro and g_angolo > g_centro
    # Di striscio sul bersaglio del primo livello nel centro del cubo: il tiro mira al suo bordo di sopra.
    centro = np.array(CENTRO_DEL_CUBO)
    traiettoria = _tiro_per(centro + np.array((0.0, RAGGIO_INIZIALE, 0.0)), 45.0)
    bersaglio = _fermo(centro, raggio=RAGGIO_INIZIALE)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    assert esito.colpito is bersaglio
    assert esito.precisione_globale == pytest.approx(_globale_attesa(esito, bersaglio), abs=TOLLERANZA_DELLA_GLOBALE)
    assert esito.precisione_globale > 1.0 - (RAGGIO_INIZIALE + RAGGIO_PROIETTILE) / MEZZA_DIAGONALE


def test_precisione_globale_di_un_bersaglio_in_moto():
    """D26: per un bersaglio in moto l'angolo più lontano si misura dal suo centro nell'istante del passaggio più vicino, non da dove
    stava allo sparo, nei tiri a vuoto e nei centri."""
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    incrocio = traiettoria.posizione(1.6)
    rapidita = 6.0
    nato = PARTENZA - 1.0
    # Arriva sulla traiettoria un secondo dopo il proiettile, venendo da sinistra.
    in_ritardo = Bersaglio(numero=1, forma="sega", raggio=1.0, partenza=incrocio - np.array((rapidita * (PARTENZA + 2.6 - nato), 0.0, 0.0)),
                           velocita=(rapidita, 0.0, 0.0), nato=nato)
    esito = volo.calcola_volo(traiettoria, [in_ritardo], PARTENZA)
    assert esito.colpito is None and esito.vicino is in_ritardo
    assert esito.precisione_globale == pytest.approx(_globale_attesa(esito, in_ritardo), abs=TOLLERANZA_DELLA_GLOBALE)
    # Dal centro dello sparo la misura sarebbe un'altra.
    allo_sparo = max(0.0, 1.0 - esito.distanza_dal_centro / _piu_lontano(in_ritardo.posizione(PARTENZA)))
    assert abs(allo_sparo - esito.precisione_globale) > 1e-3
    # Un centro sullo stesso moto, un metro più in alto della traiettoria nell'istante dell'incrocio.
    in_orario = Bersaglio(numero=2, forma="sega", raggio=1.5, partenza=incrocio + np.array((-rapidita * (PARTENZA + 1.6 - nato), 1.0, 0.0)),
                          velocita=(rapidita, 0.0, 0.0), nato=nato)
    centro = volo.calcola_volo(traiettoria, [in_orario], PARTENZA)
    assert centro.colpito is in_orario
    assert centro.precisione_globale == pytest.approx(_globale_attesa(centro, in_orario), abs=TOLLERANZA_DELLA_GLOBALE)


def test_precisione_globale_del_bonus_appena_fuori_dalla_parete():
    """D26: il bonus si misura dal suo centro nell'istante del passaggio più vicino anche quando è appena fuori da una parete, dove è
    ancora vivo finché non è uscito del tutto; il suo angolo più lontano sta più in là di quello del punto sulla parete."""
    punto = np.array((CUBO_X[1] + 0.5, 12.0, 45.0))
    traiettoria = _tiro_per(punto, 45.0)
    tempo = math.hypot(punto[0], punto[2]) / (traiettoria.velocita * math.cos(math.radians(45.0)))
    raggio = 1.5
    rapidita = 6.0
    # Il bonus passa un metro sopra il punto della traiettoria, proprio quando ci arriva il proiettile.
    sopra = punto + np.array((0.0, 1.0, 0.0))
    bonus = Bersaglio(numero=1, forma="quadra", raggio=raggio, partenza=sopra - np.array((rapidita * (PARTENZA + tempo), 0.0, 0.0)),
                      velocita=(rapidita, 0.0, 0.0), bonus=True)
    assert bonus.vivo(PARTENZA + tempo) and np.allclose(bonus.posizione(PARTENZA + tempo), sopra)
    esito = volo.calcola_volo(traiettoria, [bonus], PARTENZA)
    assert esito.colpito is bonus and esito.distanza_dal_centro < 1.0 + 1e-6
    assert esito.precisione_globale == pytest.approx(_globale_attesa(esito, bonus), abs=TOLLERANZA_DELLA_GLOBALE)
    assert esito.centro_nel_passaggio[0] > CUBO_X[1]
    sulla_parete = np.array((CUBO_X[1], *esito.centro_nel_passaggio[1:]))
    assert fisica.distanza_dall_angolo_piu_lontano(esito.centro_nel_passaggio) > fisica.distanza_dall_angolo_piu_lontano(sulla_parete)


def test_precisione_globale_a_zero_oltre_l_angolo():
    """D26: un tiro sparato lontano passa dal bersaglio più lontano del suo angolo più lontano, e la precisione globale resta a zero,
    mai sotto; un tiro a vuoto vicino, invece, vale più di zero."""
    bersaglio = _fermo(CENTRO_DEL_CUBO, raggio=RAGGIO_INIZIALE)
    # Tutto a sinistra, basso e debole: cade a pochi metri dal cannone.
    lontano = volo.calcola_volo(fisica.Traiettoria(-45.0, 10.0, 12.0), [bersaglio], PARTENZA)
    assert lontano.colpito is None and lontano.vicino is bersaglio
    assert lontano.distanza_dal_centro > MEZZA_DIAGONALE
    assert lontano.precisione_globale == 0.0
    assert _globale_attesa(lontano, bersaglio) == 0.0
    vicino = volo.calcola_volo(_tiro_per(np.array(CENTRO_DEL_CUBO) + np.array((0.0, 10.0, 0.0)), 45.0), [bersaglio], PARTENZA)
    assert vicino.colpito is None and 0.0 < vicino.precisione_globale < 1.0


def test_parametri_del_proiettile():
    traiettoria = fisica.Traiettoria(-8.0, 40.0, 28.0)
    esito = volo.calcola_volo(traiettoria, [], PARTENZA)
    parametri = volo.parametri_del_proiettile(esito)
    tempi_di_gioco = np.linspace(PARTENZA - 1.0, PARTENZA + esito.durata + 1.0, 4000)
    tempi_audio = np.linspace(100.0, 101.0, 4000)
    frequenze, pan, guadagni = parametri(tempi_di_gioco, tempi_audio)
    assert frequenze.shape == pan.shape == guadagni.shape == (4000,)
    tempi_di_volo = np.clip(tempi_di_gioco - PARTENZA, 0.0, esito.durata)
    attese = fisica.suono_di(traiettoria.posizione(tempi_di_volo))
    assert np.allclose(frequenze, attese[0])
    assert np.allclose(pan, attese[1])
    assert np.allclose(guadagni, attese[2] * 10 ** (PROIETTILE_DB / 20))
    # Prima dello sparo sta nel cannone, dopo la fine resta nel punto finale.
    prima = fisica.suono_di(traiettoria.posizione(0.0))
    dopo = fisica.suono_di(esito.punto_finale)
    assert frequenze[0] == pytest.approx(prima[0])
    assert frequenze[-1] == pytest.approx(dopo[0])
    assert pan[-1] == pytest.approx(dopo[1])
    # Il tempo di gioco fermo dà un suono fermo, lungo quanto il blocco dell'audio.
    fermo = parametri(PARTENZA + 0.5, tempi_audio)
    assert all(valori.shape == (4000,) for valori in fermo)
    assert np.all(fermo[0] == fermo[0][0])


def test_parametri_del_proiettile_dopo_un_centro():
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    bersaglio = _fermo(traiettoria.posizione(1.2), raggio=2.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    parametri = volo.parametri_del_proiettile(esito)
    frequenze, pan, _ = parametri(np.array([PARTENZA + 5.0]), np.array([0.0]))
    attese = fisica.suono_di(esito.punto_finale)
    assert frequenze[0] == pytest.approx(attese[0])
    assert pan[0] == pytest.approx(attese[1])


def test_tempo_di_calcolo():
    """Dieci secondi di volo fra cinque bersagli, mobili e bonus compresi, in meno di 50 millesimi:
    il volo si calcola nel filo principale, allo sparo, e non deve farsi sentire."""
    verticale = 49.0
    traiettoria = fisica.Traiettoria(-30.0, math.degrees(math.atan2(verticale, 5.0)), math.hypot(verticale, 5.0))
    assert traiettoria.tempo_a_terra() == pytest.approx(10.0, abs=0.1)
    gruppo = [
        Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=(20.0, 30.0, 60.0), velocita=(6.0, 3.0, -2.0)),
        Bersaglio(numero=2, forma="deriva", raggio=0.5, partenza=(10.0, 40.0, 70.0), velocita=(-4.0, 5.0, 8.0), seme=3),
        Bersaglio(numero=3, forma="quadra", raggio=2.0, partenza=(15.0, 10.0, 40.0)),
        Bersaglio(numero=4, forma="casuale", raggio=1.5, partenza=(5.0, 20.0, 50.0), velocita=(11.0, -2.0, 1.0)),
        Bersaglio(numero=5, forma="sega", raggio=0.8, partenza=(-25.8, 30.0, 60.0), velocita=(4.0, 0.0, 0.0), bonus=True),
    ]
    volo.calcola_volo(traiettoria, gruppo, 0.0)
    durate = []
    for _ in range(RIPETIZIONI_DEL_CRONOMETRO):
        inizio = time.perf_counter()
        esito = volo.calcola_volo(traiettoria, gruppo, 0.0)
        durate.append(time.perf_counter() - inizio)
    assert esito.durata > 9.9
    assert min(durate) < TEMPO_MASSIMO_DEL_CALCOLO, f"il volo più veloce ha chiesto {min(durate) * 1000:.1f} millesimi"
