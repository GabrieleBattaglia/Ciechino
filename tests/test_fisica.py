"""
Le prove di fisica.py: il tiro e la sua inversa, la parabola, il suono dello spazio, la distanza
orizzontale delle misure dell'artigliere (D21), la scomposizione dello scarto, la copertura dello spazio
dei bersagli e la distanza dal suo punto più lontano, la misura della precisione globale (D26).
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Dalla 2.0.0 lo spazio dei bersagli è la cupola di D28, e il cubo esce anche dalle prove: al posto della
zona a scatola e degli otto angoli del cubo vengono la tabella delle velocità minime e la scala della
potenza, ogni combinazione dei tre valori che attraversa la cupola, la Zona con le sue pareti e i suoi
rimbalzi, il Percorso dei bersagli mobili, gli spigoli della cupola e i suoi otto angoli, e il suono del
tiro più alto, che sale fino all'apice.
La prova della copertura è quella che il progetto vuole: ogni punto dove può stare il centro di un
bersaglio si raggiunge con i tre valori dentro i margini, e il tiro trovato ci passa davvero.
"""

import itertools
import math
import random
import threading

import numpy as np
import pytest

import fisica
from costanti import (
    ALTEZZA_CANNONE,
    ALTEZZA_MINIMA_BERSAGLIO,
    ALTEZZA_TETTO_DEL_SUONO,
    ALZO_MASSIMO,
    ALZO_MINIMO,
    AZIMUT_DEL_PAN,
    AZIMUT_MASSIMO,
    CUPOLA_RAGGIO,
    CUPOLA_RAGGIO_INTERNO,
    DB_PER_METRO,
    ELEVAZIONE_MASSIMA,
    FREQUENZA_AL_SUOLO,
    GRAVITA,
    GUADAGNO_DB_MINIMO,
    GUADAGNO_DB_VICINO,
    MARGINE_DI_COPERTURA,
    METRI_PER_OTTAVA,
    OTTAVE_DEL_SUONO,
    RAGGIO_BONUS_FATTORE,
    RAGGIO_INIZIALE,
    RAGGIO_MINIMO,
    RIDUZIONE_DEL_RAGGIO,
    SETTORE_AZIMUT,
    VELOCITA_MASSIMA,
)


def _raggi_del_gioco():
    """Tutti i raggi dei bersagli normali, dal primo livello al mezzo metro: RAGGIO_INIZIALE ridotto a ogni
    livello, fino a RAGGIO_MINIMO."""
    raggi, raggio = [], RAGGIO_INIZIALE
    while raggio > RAGGIO_MINIMO:
        raggi.append(raggio)
        raggio *= RIDUZIONE_DEL_RAGGIO
    return (*raggi, RAGGIO_MINIMO)


RAGGI_DEL_GIOCO = _raggi_del_gioco()
# I campioni di una traiettoria nelle prove che cercano dove entra nella cupola: fitti, perché lo 0% della
# potenza la sfiora appena.
CAMPIONI_FITTI = 20000
# Quanti punti a caso per raggio nella prova della copertura, oltre ai bordi della zona; e quanti punti per
# coordinata sui bordi.
PUNTI_A_CASO = 400
PUNTI_SUI_BORDI = 9


def _sferico(distanza, azimut_gradi, elevazione_gradi):
    """Il punto a quella distanza dall'origine, con quell'azimut, positivo a destra, e quell'elevazione,
    riscritto qui senza fisica."""
    azimut, elevazione = math.radians(azimut_gradi), math.radians(elevazione_gradi)
    return np.array((distanza * math.cos(elevazione) * math.sin(azimut), distanza * math.sin(elevazione), distanza * math.cos(elevazione) * math.cos(azimut)))


def _angoli_della_cupola():
    """Gli otto angoli della cupola, come quelli del cubo: le due sfere, a terra, cioè ad
    ALTEZZA_MINIMA_BERSAGLIO, e sul cono di ELEVAZIONE_MASSIMA, ai due lati del settore."""
    angoli = []
    for distanza, azimut in itertools.product((CUPOLA_RAGGIO_INTERNO, CUPOLA_RAGGIO), (-SETTORE_AZIMUT, SETTORE_AZIMUT)):
        for elevazione in (math.degrees(math.asin(ALTEZZA_MINIMA_BERSAGLIO / distanza)), ELEVAZIONE_MASSIMA):
            angoli.append(_sferico(distanza, azimut, elevazione))
    return np.array(angoli)


def _bordi(zona, quanti=PUNTI_SUI_BORDI):
    """I punti sulle pareti di una zona: una griglia di distanza, elevazione e azimut, tenuti soltanto i
    punti dove almeno una coordinata sta a un capo del suo intervallo. L'elevazione va da quella del
    pavimento a quella del cono della zona; l'azimut fino a dove il centro sta a un raggio dal lato."""
    punti = []
    for distanza in np.linspace(zona.interno, zona.esterno, quanti):
        dal_pavimento = math.degrees(math.asin(zona.pavimento / distanza))
        for k, elevazione in enumerate(np.linspace(dal_pavimento, zona.elevazione, quanti)):
            limite = SETTORE_AZIMUT - math.degrees(math.asin(zona.raggio / (distanza * math.cos(math.radians(elevazione)))))
            for j, azimut in enumerate(np.linspace(-limite, limite, quanti)):
                if distanza in (zona.interno, zona.esterno) or k in (0, quanti - 1) or j in (0, quanti - 1):
                    punti.append(_sferico(distanza, azimut, elevazione))
    return np.array(punti)


def _entra_nella_cupola(azimut, alzo, velocita, campioni=CAMPIONI_FITTI):
    """Vero se il tiro passa per un punto della Zona di raggio zero, cercato su tanti campioni fino a terra."""
    traiettoria = fisica.Traiettoria(azimut, alzo, velocita)
    tempi = np.linspace(0.0, traiettoria.tempo_a_terra(), campioni)
    return bool(fisica.Zona(0.0).contiene(traiettoria.posizione(tempi)).any())


def _minima_vera(alzo):
    """La velocità più bassa che a quell'alzo porta il proiettile nella cupola, cercata qui per bisezione
    sulla Zona di raggio zero, con campioni dieci volte più fitti di quelli della tabella."""
    basso, alto = 0.0, VELOCITA_MASSIMA
    for _passo in range(40):
        medio = (basso + alto) / 2.0
        if _entra_nella_cupola(0.0, alzo, medio):
            alto = medio
        else:
            basso = medio
    return alto


def test_parametri_del_tiro_ai_capi_e_al_mezzo():
    """D28: la potenza va dalla velocità minima di quell'alzo a VELOCITA_MASSIMA."""
    assert fisica.parametri_del_tiro(0.0, 0.0, 0.0) == pytest.approx((-AZIMUT_MASSIMO, ALZO_MINIMO, fisica.velocita_minima(ALZO_MINIMO)))
    assert fisica.parametri_del_tiro(1.0, 1.0, 1.0) == pytest.approx((AZIMUT_MASSIMO, ALZO_MASSIMO, VELOCITA_MASSIMA))
    azimut, alzo, velocita = fisica.parametri_del_tiro(0.5, 0.5, 0.5)
    assert azimut == pytest.approx(0.0)
    assert alzo == pytest.approx((ALZO_MINIMO + ALZO_MASSIMO) / 2)
    assert velocita == pytest.approx((fisica.velocita_minima(alzo) + VELOCITA_MASSIMA) / 2)
    for valore in (0.0, 0.13, 0.5, 0.77, 1.0):
        _, alzo, minima = fisica.parametri_del_tiro(0.5, valore, 0.0)
        assert minima == pytest.approx(fisica.velocita_minima(alzo))
        assert fisica.parametri_del_tiro(0.5, valore, 1.0)[2] == pytest.approx(VELOCITA_MASSIMA)


def test_scalari_restano_scalari_e_array_restano_array():
    valori = fisica.parametri_del_tiro(0.3, 0.4, 0.5)
    assert all(isinstance(v, float) for v in valori)
    valori = fisica.parametri_del_tiro(np.array([0.1, 0.9]), np.array([0.2, 0.8]), np.array([0.3, 0.7]))
    assert all(isinstance(v, np.ndarray) and v.shape == (2,) for v in valori)
    assert isinstance(fisica.pan_di(1.0, 5.0), float)
    assert isinstance(fisica.frequenza_di(3.0), float)
    assert isinstance(fisica.guadagno_db_di(3.0), float)
    assert isinstance(fisica.guadagno_di(3.0), float)
    assert isinstance(fisica.distanza_dal_cannone((1.0, 2.0, 3.0)), float)
    assert fisica.frequenza_di(np.zeros((4, 2))).shape == (4, 2)
    assert isinstance(fisica.velocita_minima(40.0), float)
    assert fisica.velocita_minima(np.array([20.0, 40.0, 60.0])).shape == (3,)
    assert isinstance(fisica.distanza_dal_punto_piu_lontano(fisica.MEZZO_DELLA_CUPOLA), float)
    assert fisica.distanza_dal_punto_piu_lontano(np.zeros((2, 4, 3))).shape == (2, 4)


def test_valori_del_tiro_inversa():
    rng = np.random.default_rng(7)
    direzione, alzo, potenza = rng.random(100), rng.random(100), rng.random(100)
    indietro = fisica.valori_del_tiro(*fisica.parametri_del_tiro(direzione, alzo, potenza))
    assert np.allclose(indietro[0], direzione)
    assert np.allclose(indietro[1], alzo)
    assert np.allclose(indietro[2], potenza)
    assert fisica.valori_del_tiro(0.0, ALZO_MINIMO, fisica.velocita_minima(ALZO_MINIMO)) == pytest.approx((0.5, 0.0, 0.0))
    assert fisica.valori_del_tiro(-SETTORE_AZIMUT, ALZO_MASSIMO, VELOCITA_MASSIMA) == pytest.approx((0.0, 1.0, 1.0))


def test_valori_del_tiro_fuori_scala_non_si_tagliano():
    direzione, alzo, potenza = fisica.valori_del_tiro(2 * AZIMUT_MASSIMO, 10.0, VELOCITA_MASSIMA + 10.0)
    assert direzione > 1.0
    assert alzo < 0.0
    assert potenza > 1.0
    # Sotto la velocità minima di quell'alzo la potenza è negativa.
    assert fisica.valori_del_tiro(0.0, 45.0, fisica.velocita_minima(45.0) - 1.0)[2] < 0.0


def test_traiettoria_parte_dal_cannone_e_va_a_destra():
    traiettoria = fisica.Traiettoria(20.0, 30.0, 25.0)
    assert np.allclose(traiettoria.origine, (0.0, ALTEZZA_CANNONE, 0.0))
    assert np.allclose(traiettoria.posizione(0.0), traiettoria.origine)
    assert np.linalg.norm(traiettoria.iniziale) == pytest.approx(25.0)
    punto = traiettoria.posizione(1.0)
    assert punto.shape == (3,)
    assert punto[0] > 0.0
    assert punto[2] > 0.0
    assert math.degrees(math.atan2(punto[0], punto[2])) == pytest.approx(20.0)
    assert traiettoria.posizione(np.linspace(0, 1, 7)).shape == (7, 3)


def test_traiettoria_parabola_esatta():
    traiettoria = fisica.Traiettoria(-10.0, 40.0, 30.0, origine=(1.0, 2.0, 3.0))
    t = 1.7
    alzo, azimut = math.radians(40.0), math.radians(-10.0)
    atteso = np.array((1.0 + 30.0 * math.cos(alzo) * math.sin(azimut) * t, 2.0 + 30.0 * math.sin(alzo) * t - GRAVITA * t * t / 2,
                       3.0 + 30.0 * math.cos(alzo) * math.cos(azimut) * t))
    assert np.allclose(traiettoria.posizione(t), atteso)


def test_tempo_a_terra_apice_punto_a_terra():
    traiettoria = fisica.Traiettoria(5.0, 35.0, 28.0)
    tempo = traiettoria.tempo_a_terra()
    assert tempo > 0.0
    assert traiettoria.posizione(tempo)[1] == pytest.approx(0.0, abs=1e-9)
    assert traiettoria.posizione(tempo * 0.999)[1] > 0.0
    t_apice, altezza = traiettoria.apice()
    campioni = traiettoria.posizione(np.linspace(0.0, tempo, 20001))
    assert altezza == pytest.approx(campioni[:, 1].max(), abs=1e-4)
    assert traiettoria.posizione(t_apice)[1] == pytest.approx(altezza)
    a_terra = traiettoria.punto_a_terra()
    assert a_terra[1] == 0.0
    assert np.allclose(a_terra[[0, 2]], traiettoria.posizione(tempo)[[0, 2]])


def test_inclinazione_della_traiettoria():
    """D21: l'angolo della velocità sull'orizzontale, positivo in salita: alla bocca è l'alzo, nell'apice
    zero, poi negativo, e l'azimut non conta. Uno scalare per t scalare, un array per un array."""
    traiettoria = fisica.Traiettoria(-20.0, 50.0, 30.0)
    t_apice, _altezza = traiettoria.apice()
    assert traiettoria.inclinazione(0.0) == pytest.approx(50.0)
    assert traiettoria.inclinazione(t_apice) == pytest.approx(0.0, abs=1e-9)
    assert isinstance(traiettoria.inclinazione(1.0), float)
    tempi = np.array((0.5, t_apice, t_apice + 0.5, 2 * t_apice))
    angoli = traiettoria.inclinazione(tempi)
    assert isinstance(angoli, np.ndarray) and angoli.shape == (4,)
    assert angoli[0] > 0.0 and angoli[2] < 0.0
    # Simmetria della parabola: a pari distanza dall'apice, pari inclinazione col segno opposto.
    assert angoli[0] == pytest.approx(-traiettoria.inclinazione(2 * t_apice - 0.5))
    assert angoli[3] == pytest.approx(-50.0)
    # Dalla derivata della posizione, a differenze finite.
    passo = 1e-6
    velocita = (traiettoria.posizione(1.3 + passo) - traiettoria.posizione(1.3 - passo)) / (2 * passo)
    assert traiettoria.inclinazione(1.3) == pytest.approx(math.degrees(math.atan2(velocita[1], math.hypot(velocita[0], velocita[2]))), abs=1e-6)
    assert fisica.Traiettoria(10.0, 30.0, 20.0).inclinazione(0.0) == pytest.approx(fisica.Traiettoria(-35.0, 30.0, 20.0).inclinazione(0.0))


def test_tiro_orizzontale_e_rasoterra():
    piatto = fisica.Traiettoria(0.0, 0.0, 20.0)
    assert piatto.apice() == (0.0, ALTEZZA_CANNONE)
    assert piatto.tempo_a_terra() == pytest.approx(math.sqrt(2 * ALTEZZA_CANNONE / GRAVITA))
    da_terra = fisica.Traiettoria(0.0, 30.0, 20.0, origine=(0.0, 0.0, 0.0))
    assert da_terra.tempo_a_terra() == pytest.approx(2 * 20.0 * math.sin(math.radians(30.0)) / GRAVITA)


def test_pan_di():
    """La panoramica arriva ai capi del settore (D28), e coincide con la direzione che mira il punto."""
    assert AZIMUT_DEL_PAN == AZIMUT_MASSIMO == SETTORE_AZIMUT
    assert fisica.pan_di(0.0, 40.0) == pytest.approx(0.0)
    venti = math.radians(20.0)
    assert fisica.pan_di(math.sin(venti), math.cos(venti)) == pytest.approx(20.0 / AZIMUT_DEL_PAN)
    assert fisica.pan_di(-math.sin(venti), math.cos(venti)) == pytest.approx(-20.0 / AZIMUT_DEL_PAN)
    for lato in (-1.0, 1.0):
        sul_lato = _sferico(55.0, lato * SETTORE_AZIMUT, 20.0)
        assert fisica.pan_di(sul_lato[0], sul_lato[2]) == pytest.approx(lato)
    assert fisica.pan_di(100.0, 1.0) == 1.0
    assert fisica.pan_di(-100.0, 1.0) == -1.0
    pan = fisica.pan_di(np.array([-5.0, 0.0, 5.0]), np.array([50.0, 50.0, 50.0]))
    assert pan[0] == pytest.approx(-pan[2])
    assert pan[1] == 0.0
    # Chi porta la direzione sulla panoramica del bersaglio mira giusto: lungo il tiro la panoramica vale
    # due volte la direzione meno uno.
    for direzione in (0.0, 0.2, 0.5, 0.9, 1.0):
        traiettoria = fisica.Traiettoria(*fisica.parametri_del_tiro(direzione, 0.5, 0.5))
        punti = traiettoria.posizione(np.linspace(0.1, traiettoria.tempo_a_terra(), 7))
        assert np.allclose(fisica.pan_di(punti[:, 0], punti[:, 2]), 2.0 * direzione - 1.0)


def test_frequenza_di():
    assert fisica.frequenza_di(0.0) == pytest.approx(FREQUENZA_AL_SUOLO)
    assert fisica.frequenza_di(METRI_PER_OTTAVA) == pytest.approx(2 * FREQUENZA_AL_SUOLO)
    assert fisica.frequenza_di(-5.0) == pytest.approx(FREQUENZA_AL_SUOLO)
    tetto = fisica.frequenza_di(ALTEZZA_TETTO_DEL_SUONO)
    assert tetto == pytest.approx(FREQUENZA_AL_SUOLO * 2 ** OTTAVE_DEL_SUONO)
    assert fisica.frequenza_di(ALTEZZA_TETTO_DEL_SUONO + 100.0) == pytest.approx(tetto)


def test_il_tetto_del_suono_e_l_apice_piu_alto():
    """D28: la velocità massima è la radice di 2 g R, 39,6 m/s, e il tetto delle quattro ottave sta all'apice
    del tiro più alto possibile, quello verticale, un metro sopra la cima della cupola: nessun tiro della mira
    sale oltre, e la nota non si ferma mai."""
    assert math.sqrt(2.0 * GRAVITA * CUPOLA_RAGGIO) == pytest.approx(VELOCITA_MASSIMA)
    assert round(VELOCITA_MASSIMA, 1) == 39.6
    assert math.isclose(ALTEZZA_TETTO_DEL_SUONO, ALTEZZA_CANNONE + CUPOLA_RAGGIO)
    assert fisica.Traiettoria(0.0, 90.0, VELOCITA_MASSIMA).apice()[1] == pytest.approx(ALTEZZA_TETTO_DEL_SUONO)
    assert math.isclose(METRI_PER_OTTAVA * OTTAVE_DEL_SUONO, ALTEZZA_TETTO_DEL_SUONO)
    alzi, potenze = np.meshgrid(np.linspace(0.0, 1.0, 29), np.linspace(0.0, 1.0, 11))
    _, alzi_gradi, velocita = fisica.parametri_del_tiro(0.5, alzi.ravel(), potenze.ravel())
    apici = ALTEZZA_CANNONE + (velocita * np.sin(np.radians(alzi_gradi))) ** 2 / (2.0 * GRAVITA)
    assert apici.max() < ALTEZZA_TETTO_DEL_SUONO
    # Il tiro più alto della mira, l'alzo e la potenza al massimo, sale quasi fino in cima alla cupola.
    _tempo, altezza = fisica.Traiettoria(*fisica.parametri_del_tiro(0.5, 1.0, 1.0)).apice()
    assert altezza == pytest.approx(apici.max())
    assert CUPOLA_RAGGIO < altezza < ALTEZZA_TETTO_DEL_SUONO


def test_il_suono_del_tiro_piu_alto_sale_fino_all_apice():
    """D28, dall'ascolto di Gabriele: fino alla 1.0.0 la nota del tiro più alto si fermava al tetto delle
    ottave, 66,7 metri, e a 85 gradi restava piatta per 5,5 secondi su 9,2 di volo. Ora sale per tutta la
    salita fino all'apice e scende per tutta la discesa, senza un tratto fermo, e all'apice resta sotto il tetto."""
    traiettoria = fisica.Traiettoria(*fisica.parametri_del_tiro(0.5, 1.0, 1.0))
    tempo_apice, altezza = traiettoria.apice()
    tempi = np.linspace(0.0, traiettoria.tempo_a_terra(), 20001)
    frequenze = fisica.frequenza_di(traiettoria.posizione(tempi)[:, 1])
    salti = np.diff(frequenze)
    assert np.all(salti[tempi[1:] <= tempo_apice] > 0.0)
    assert np.all(salti[tempi[:-1] >= tempo_apice] < 0.0)
    assert frequenze.max() == pytest.approx(fisica.frequenza_di(altezza), rel=1e-6)
    assert fisica.frequenza_di(altezza) < fisica.frequenza_di(ALTEZZA_TETTO_DEL_SUONO)
    # Nessun campione uguale al seguente: nessun tratto piatto, nemmeno vicino all'apice.
    assert np.count_nonzero(salti == 0.0) == 0


def test_guadagni():
    assert fisica.guadagno_db_di(0.0) == pytest.approx(GUADAGNO_DB_VICINO)
    assert fisica.guadagno_db_di(10.0) == pytest.approx(GUADAGNO_DB_VICINO - 10.0 * DB_PER_METRO)
    assert fisica.guadagno_db_di(1e6) == pytest.approx(GUADAGNO_DB_MINIMO)
    distanze = np.array([0.0, 20.0, 80.0, 1000.0])
    assert np.allclose(fisica.guadagno_di(distanze), 10 ** (fisica.guadagno_db_di(distanze) / 20))
    assert np.all(np.diff(fisica.guadagno_di(distanze)) <= 0.0)


def test_distanza_e_suono_di():
    assert fisica.distanza_dal_cannone((0.0, ALTEZZA_CANNONE, 0.0)) == 0.0
    assert fisica.distanza_dal_cannone((3.0, ALTEZZA_CANNONE + 4.0, 0.0)) == pytest.approx(5.0)
    posizioni = np.array([[0.0, 10.0, 40.0], [-20.0, 30.0, 60.0], [15.0, 45.0, 75.0]])
    frequenze, pan, guadagni = fisica.suono_di(posizioni)
    assert frequenze.shape == pan.shape == guadagni.shape == (3,)
    for k, punto in enumerate(posizioni):
        f, p, g = fisica.suono_di(punto)
        assert isinstance(f, float)
        assert frequenze[k] == pytest.approx(f) == pytest.approx(fisica.frequenza_di(punto[1]))
        assert pan[k] == pytest.approx(p) == pytest.approx(fisica.pan_di(punto[0], punto[2]))
        assert guadagni[k] == pytest.approx(g) == pytest.approx(fisica.guadagno_di(fisica.distanza_dal_cannone(punto)))


def test_distanza_orizzontale():
    """D21: la distanza dal cannone senza l'altezza, quella delle misure dell'artigliere."""
    assert fisica.distanza_orizzontale((0.0, 37.0, 0.0)) == 0.0
    assert fisica.distanza_orizzontale((3.0, 12.0, -4.0)) == pytest.approx(5.0)
    assert isinstance(fisica.distanza_orizzontale((3.0, 12.0, 4.0)), float)
    posizioni = np.array([[0.0, 10.0, 40.0], [-20.0, 30.0, 60.0], [15.0, 45.0, 75.0]])
    distanze = fisica.distanza_orizzontale(posizioni)
    assert distanze.shape == (3,)
    assert distanze == pytest.approx(np.hypot(posizioni[:, 0], posizioni[:, 2]))
    # L'altezza non conta: la stessa distanza dal suolo alla cima della cupola.
    assert fisica.distanza_orizzontale((6.0, 0.0, 8.0)) == fisica.distanza_orizzontale((6.0, CUPOLA_RAGGIO, 8.0)) == pytest.approx(10.0)


def test_scomponi_bersaglio_davanti():
    davanti = (0.0, 20.0, 50.0)
    assert fisica.scomponi((0.0, 0.0, 2.0), davanti) == pytest.approx((2.0, 0.0, 0.0))
    assert fisica.scomponi((0.0, 0.0, -2.0), davanti) == pytest.approx((-2.0, 0.0, 0.0))
    assert fisica.scomponi((1.5, 0.0, 0.0), davanti) == pytest.approx((0.0, 1.5, 0.0))
    assert fisica.scomponi((0.0, -3.0, 0.0), davanti) == pytest.approx((0.0, 0.0, -3.0))


def test_scomponi_bersaglio_di_lato_e_vettori():
    # Un bersaglio a 45 gradi a destra: lungo è verso destra e avanti, destra è verso destra e indietro.
    di_lato = (30.0, 10.0, 30.0)
    radiale, laterale, verticale = fisica.scomponi((1.0, 0.0, 1.0), di_lato)
    assert radiale == pytest.approx(math.sqrt(2))
    assert laterale == pytest.approx(0.0, abs=1e-12)
    assert verticale == 0.0
    radiale, laterale, _ = fisica.scomponi((1.0, 0.0, -1.0), di_lato)
    assert radiale == pytest.approx(0.0, abs=1e-12)
    assert laterale == pytest.approx(math.sqrt(2))
    delta = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 0.0]])
    punti = np.array([[0.0, 5.0, 40.0], [0.0, 5.0, 40.0]])
    radiali, laterali, verticali = fisica.scomponi(delta, punti)
    assert np.allclose(radiali, (1.0, 0.0))
    assert np.allclose(laterali, (0.0, 1.0))
    assert np.allclose(verticali, (0.0, 2.0))
    # Sopra il cannone non c'è una direzione orizzontale: vale come davanti.
    assert fisica.scomponi((0.0, 0.0, 1.0), (0.0, 30.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0))


def test_velocita_per_passa_dal_punto():
    punto = (8.0, 22.0, 55.0)
    for alzo in (40.0, 55.0, 70.0):
        velocita = fisica.velocita_per(punto, alzo)
        azimut = math.degrees(math.atan2(punto[0], punto[2]))
        traiettoria = fisica.Traiettoria(azimut, alzo, velocita)
        tempo = math.hypot(punto[0], punto[2]) / (velocita * math.cos(math.radians(alzo)))
        assert np.allclose(traiettoria.posizione(tempo), punto)
    assert math.isnan(fisica.velocita_per(punto, 5.0))
    assert fisica.velocita_per(punto, np.array([5.0, 50.0])).shape == (2,)


# La tabella delle velocità minime e la scala della potenza (D28).

def test_tabella_delle_minime():
    """Un alzo ogni quarto di grado, capi compresi, calcolata una volta sola e in sola lettura; ogni
    minima fra zero e la massima, e la funzione la interpola, fuori dalla scala dell'alzo col capo più vicino."""
    alzi, minime = fisica.tabella_delle_minime()
    assert fisica.tabella_delle_minime() is fisica.tabella_delle_minime()
    assert alzi[0] == ALZO_MINIMO and alzi[-1] == ALZO_MASSIMO
    assert len(alzi) == round((ALZO_MASSIMO - ALZO_MINIMO) * fisica.PUNTI_PER_GRADO_DELLE_MINIME) + 1
    assert np.allclose(np.diff(alzi), 1.0 / fisica.PUNTI_PER_GRADO_DELLE_MINIME)
    assert not alzi.flags.writeable and not minime.flags.writeable
    assert np.all(minime > 0.0) and np.all(minime < VELOCITA_MASSIMA)
    assert fisica.velocita_minima(alzi) == pytest.approx(minime)
    mezzo = (alzi[10] + alzi[11]) / 2.0
    assert fisica.velocita_minima(mezzo) == pytest.approx((minime[10] + minime[11]) / 2.0)
    assert fisica.velocita_minima(5.0) == fisica.velocita_minima(ALZO_MINIMO)
    assert fisica.velocita_minima(90.0) == fisica.velocita_minima(ALZO_MASSIMO)


def test_le_minime_scendono_fino_al_tiro_piu_dolce_poi_salgono():
    """Salendo d'alzo la velocità minima cala fino al tiro più dolce verso il punto più vicino della cupola,
    il bordo della sfera interna ad ALTEZZA_MINIMA_BERSAGLIO, poco sotto i 47 gradi; poi cresce fino agli
    85. Vicino al minimo la curva è piatta, e la bisezione sui campioni ci lascia un rumore di qualche
    millesimo: la prova guarda fino ai 45 gradi e dai 50. Il minimo vale la formula chiusa del tiro più dolce,
    più MARGINE_DELLA_MINIMA."""
    alzi, minime = fisica.tabella_delle_minime()
    prima, dopo = alzi <= 45.0, alzi >= 50.0
    assert np.all(np.diff(minime[prima]) < 0.0)
    assert np.all(np.diff(minime[dopo]) > 0.0)
    dislivello = ALTEZZA_MINIMA_BERSAGLIO - ALTEZZA_CANNONE
    distanza = math.sqrt(CUPOLA_RAGGIO_INTERNO ** 2 - ALTEZZA_MINIMA_BERSAGLIO ** 2)
    piu_dolce = math.sqrt(GRAVITA * (dislivello + math.hypot(dislivello, distanza)))
    alzo_piu_dolce = math.degrees(math.atan((dislivello + math.hypot(dislivello, distanza)) / distanza))
    assert piu_dolce + fisica.MARGINE_DELLA_MINIMA <= minime.min() <= piu_dolce + 2 * fisica.MARGINE_DELLA_MINIMA
    assert alzi[int(np.argmin(minime))] == pytest.approx(alzo_piu_dolce, abs=1.0)


def test_i_numeri_della_scala_della_potenza():
    """I numeri di D28: la scala della potenza, dalla minima alla massima, è larga 21,9 m/s a 45 gradi, 16 a
    20 gradi, 11,6 a 15; D28 diceva 11,7, col MARGINE_DELLA_MINIMA di allora, 0,01 invece di 0,03. Sotto i 15
    si stringerebbe troppo: a 10 gradi resterebbero circa 2,4 m/s."""
    for alzo, larghezza in ((45.0, 21.9), (20.0, 16.0), (15.0, 11.6)):
        assert VELOCITA_MASSIMA - fisica.velocita_minima(alzo) == pytest.approx(larghezza, abs=0.05), alzo
    assert VELOCITA_MASSIMA - _minima_vera(10.0) == pytest.approx(2.4, abs=0.1)


def test_la_velocita_minima_e_proprio_la_minima():
    """Lo 0% della potenza porta il proiettile nella cupola, anche fra due punti della tabella, e poco sotto
    non ci arriva più: la minima sta sopra la vera, e di nulla di più che MARGINE_DELLA_MINIMA e qualche
    millesimo, così nessun pezzo della scala va sprecato. Fra gli alzi provati ci sono i due capi, il tiro più
    dolce e il ginocchio della curva, poco sopra i 72 gradi, dove la pendenza cala di colpo fra due punti della
    tabella e l'interpolazione lineare passa sotto la curva vera."""
    rng = random.Random(28)
    sbagliate = []
    for alzo in (ALZO_MINIMO, ALZO_MASSIMO, 46.9, 72.1, 72.125, *(rng.uniform(ALZO_MINIMO, ALZO_MASSIMO) for _ in range(5))):
        minima = fisica.velocita_minima(alzo)
        vera = _minima_vera(alzo)
        entra = _entra_nella_cupola(0.0, alzo, minima)
        if not entra or not 0.0 < minima - vera <= fisica.MARGINE_DELLA_MINIMA + 0.01:
            sbagliate.append(f"a {alzo:.3f} gradi la minima è {minima:.4f} e la vera {vera:.4f}, " + ("e lo 0% entra" if entra else "e lo 0% non entra nella cupola"))
    assert not sbagliate, "; ".join(sbagliate)


def test_ogni_combinazione_dei_tre_valori_passa_nella_cupola():
    """D28: col cubo passava dove un bersaglio può stare il 45,2% delle combinazioni delle tre barre; con la
    cupola ogni combinazione, ai capi delle tre scale e in mezzo, attraversa la zona dei bersagli. La
    direzione copre esattamente il settore: ai capi il tiro corre proprio sul lato."""
    rng = random.Random(452)
    alzi = (0.0, 1.0, 0.5, *(rng.random() for _ in range(19)))
    potenze = (0.0, 1.0, 0.5, rng.random(), rng.random())
    fuori = [(direzione, alzo, potenza) for direzione, alzo, potenza in itertools.product((0.0, 0.5, 1.0), alzi, potenze)
             if not _entra_nella_cupola(*fisica.parametri_del_tiro(direzione, alzo, potenza))]
    assert not fuori, f"{len(fuori)} combinazioni non passano nella cupola, la prima {fuori[0]}"


# Il punto della cupola e la Zona.

def test_punto_della_cupola_e_il_suo_mezzo():
    assert fisica.punto_della_cupola(55.0, 20.0, 30.0) == pytest.approx(_sferico(55.0, 20.0, 30.0))
    assert fisica.punto_della_cupola(55.0, 20.0, 30.0).shape == (3,)
    elevazioni = np.array([10.0, 30.0, 60.0])
    punti = fisica.punto_della_cupola(40.0, -15.0, elevazioni)
    assert punti.shape == (3, 3)
    assert np.allclose(np.linalg.norm(punti, axis=1), 40.0)
    assert np.allclose(np.degrees(np.arcsin(punti[:, 1] / 40.0)), elevazioni)
    assert np.allclose(np.degrees(np.arctan2(punti[:, 0], punti[:, 2])), -15.0)
    # Il mezzo della cupola: a metà fra le due sfere, 55 metri, davanti, a 30 gradi, 27,5 metri d'altezza.
    mezzo = np.array(fisica.MEZZO_DELLA_CUPOLA)
    assert np.linalg.norm(mezzo) == pytest.approx((CUPOLA_RAGGIO_INTERNO + CUPOLA_RAGGIO) / 2.0)
    assert mezzo[0] == 0.0 and mezzo[1] == pytest.approx(27.5)
    assert fisica.Zona(RAGGIO_INIZIALE).contiene(mezzo)
    assert fisica.soluzione(mezzo) is not None


def test_zona_le_sue_misure():
    """D28: il centro di un bersaglio sta a un raggio dalle due sfere e dai lati, almeno ad
    ALTEZZA_MINIMA_BERSAGLIO e a un raggio da terra, sotto il cono stretto dell'angolo che il raggio sottende
    alla distanza più corta. Il raggio zero è tutta la cupola, uno negativo vale zero."""
    for raggio in (0.0, RAGGIO_MINIMO, RAGGIO_INIZIALE, 4.0):
        zona = fisica.Zona(raggio)
        assert zona.raggio == raggio
        assert zona.interno == pytest.approx(CUPOLA_RAGGIO_INTERNO + raggio)
        assert zona.esterno == pytest.approx(CUPOLA_RAGGIO - raggio)
        assert zona.pavimento == pytest.approx(max(ALTEZZA_MINIMA_BERSAGLIO, raggio))
        assert zona.elevazione == pytest.approx(ELEVAZIONE_MASSIMA - math.degrees(math.asin(raggio / (CUPOLA_RAGGIO_INTERNO + raggio))))
        assert repr(zona) == f"Zona(raggio={zona.raggio})"
        for normale in (zona.normale_destra, zona.normale_sinistra):
            assert np.linalg.norm(normale) == pytest.approx(1.0) and normale[1] == 0.0
    assert fisica.Zona(-3.0).raggio == 0.0
    assert fisica.Zona().raggio == 0.0
    assert fisica.Zona.PARETI == ("esterna", "interna", "pavimento", "destra", "sinistra", "cono")
    # Le normali dei lati guardano fuori dal settore: poco oltre il lato la loro proiezione è positiva.
    zona = fisica.Zona(0.0)
    assert zona.normale_destra @ _sferico(50.0, SETTORE_AZIMUT + 1.0, 20.0) > 0.0 > zona.normale_destra @ _sferico(50.0, SETTORE_AZIMUT - 1.0, 20.0)
    assert zona.normale_sinistra @ _sferico(50.0, -SETTORE_AZIMUT - 1.0, 20.0) > 0.0 > zona.normale_sinistra @ _sferico(50.0, 1.0 - SETTORE_AZIMUT, 20.0)


def test_zona_scarti_e_contiene():
    raggio = 2.0
    zona = fisica.Zona(raggio)
    mezzo = np.array(fisica.MEZZO_DELLA_CUPOLA)
    assert zona.scarti(mezzo).shape == (6,)
    assert np.all(zona.scarti(mezzo) < 0.0)
    assert zona.contiene(mezzo) is True
    # Un punto oltre ogni parete, e soltanto quella.
    oltre = {
        "esterna": _sferico(zona.esterno + 1.0, 0.0, 30.0),
        "interna": _sferico(zona.interno - 1.0, 0.0, 30.0),
        "pavimento": np.array((0.0, zona.pavimento - 0.5, 50.0)),
        "destra": _sferico(55.0, SETTORE_AZIMUT, 30.0),
        "sinistra": _sferico(55.0, -SETTORE_AZIMUT, 30.0),
        "cono": _sferico(55.0, 0.0, 85.0),
    }
    for parete, punto in oltre.items():
        scarti = zona.scarti(punto)
        indice = fisica.Zona.PARETI.index(parete)
        assert scarti[indice] > 0.0, parete
        assert np.all(np.delete(scarti, indice) < 0.0), parete
        assert zona.contiene(punto) is False
    assert zona.scarti(oltre["esterna"])[0] == pytest.approx(1.0)
    assert zona.scarti(oltre["interna"])[1] == pytest.approx(1.0)
    assert zona.scarti(oltre["pavimento"])[2] == pytest.approx(0.5)
    # Su un lato il centro sta a tanto quanto il raggio oltre la parete della zona.
    assert zona.scarti(oltre["destra"])[3] == pytest.approx(raggio)
    tutti = np.array([*oltre.values(), mezzo])
    assert zona.scarti(tutti).shape == (7, 6)
    assert np.array_equal(zona.contiene(tutti), [False] * 6 + [True])
    # La tolleranza: un pelo fuori è ancora dentro.
    appena = _sferico(zona.esterno + 1e-7, 0.0, 30.0)
    assert zona.contiene(appena) and not zona.contiene(appena, tolleranza=0.0)


def test_zona_normali_uscenti():
    """La normale di ogni parete, sui punti della parete, è lunga uno e guarda fuori: un passo lungo di lei
    fa crescere lo scarto di quella parete, un passo all'indietro lo fa calare."""
    for raggio in (0.0, 1.5):
        zona = fisica.Zona(raggio)
        for punto in _bordi(zona, 5):
            scarti = zona.scarti(punto)
            for indice in np.flatnonzero(np.abs(scarti) < 1e-9):
                parete = fisica.Zona.PARETI[indice]
                normale = zona.normale(parete, punto)
                assert np.linalg.norm(normale) == pytest.approx(1.0)
                assert zona.scarti(punto + 1e-3 * normale)[indice] > 0.0 > zona.scarti(punto - 1e-3 * normale)[indice], (parete, punto)
    # Sulle sfere e sui piani lo scarto è proprio la distanza dalla parete.
    zona = fisica.Zona(1.0)
    punto = _sferico(zona.esterno, 10.0, 30.0)
    assert zona.scarti(punto + 0.5 * zona.normale("esterna", punto))[0] == pytest.approx(0.5)
    assert zona.normale("pavimento", punto) == pytest.approx((0.0, -1.0, 0.0))
    # Il cono sopra il centro, dove non ha una direzione orizzontale, guarda in su.
    assert zona.normale("cono", (0.0, 50.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))


def test_zona_uscita():
    """Il moto rettilineo da un punto della zona esce nel primo istante su una parete verso fuori: fino a lì
    resta dentro, lì sta sulla parete e dentro tutte le altre, subito dopo è fuori. Da fermo non esce mai."""
    rng = random.Random(33)
    pareti_viste = set()
    for prova in range(300):
        zona = fisica.Zona(rng.choice((0.0, RAGGIO_MINIMO, 2.0, RAGGIO_INIZIALE)))
        punto = zona.punto_a_caso(rng)
        velocita = np.array([rng.gauss(0.0, 1.0) for _ in range(3)]) * rng.uniform(0.5, 12.0)
        tempo, parete = zona.uscita(punto, velocita)
        assert 0.0 < tempo < math.inf and parete in fisica.Zona.PARETI, prova
        pareti_viste.add(parete)
        toccato = punto + velocita * tempo
        scarti = zona.scarti(toccato)
        assert scarti[fisica.Zona.PARETI.index(parete)] == pytest.approx(0.0, abs=1e-6), (prova, parete)
        assert np.all(scarti <= 1e-6), (prova, parete, scarti)
        assert zona.normale(parete, toccato) @ velocita > 0.0
        assert zona.contiene(punto + velocita * np.linspace(0.0, tempo, 2001)[:, None]).all(), prova
        assert not zona.contiene(punto + velocita * (tempo + 1e-3), tolleranza=0.0), prova
    # Dal mezzo della cupola verso un punto oltre ogni parete si esce proprio da quella.
    zona = fisica.Zona(2.0)
    mezzo = np.array(fisica.MEZZO_DELLA_CUPOLA)
    for parete, oltre in (("esterna", _sferico(zona.esterno + 1.0, 0.0, 30.0)), ("interna", _sferico(zona.interno - 1.0, 0.0, 30.0)),
                          ("pavimento", (0.0, zona.pavimento - 0.5, 50.0)), ("destra", _sferico(55.0, SETTORE_AZIMUT, 30.0)),
                          ("sinistra", _sferico(55.0, -SETTORE_AZIMUT, 30.0)), ("cono", _sferico(55.0, 0.0, 85.0))):
        tempo, uscita = zona.uscita(mezzo, np.asarray(oltre) - mezzo)
        assert uscita == parete and 0.0 < tempo < 1.0
        pareti_viste.add(parete)
    assert pareti_viste == set(fisica.Zona.PARETI)
    assert fisica.Zona(1.0).uscita(fisica.MEZZO_DELLA_CUPOLA, (0.0, 0.0, 0.0)) == (math.inf, None)


def test_zona_rimbalzo():
    """Sulla parete la velocità si riflette: la componente lungo la normale cambia segno, il resto e il
    modulo restano. In uno spigolo si riflette su tutte e due le pareti verso cui andrebbe ancora fuori."""
    zona = fisica.Zona(1.0)
    punto = _sferico(zona.esterno, 10.0, 30.0)
    normale = zona.normale("esterna", punto)
    velocita = np.array((2.0, 3.0, 4.0))
    assert velocita @ normale > 0.0
    riflessa = zona.rimbalzo(punto, velocita, "esterna")
    assert riflessa == pytest.approx(velocita - 2.0 * (velocita @ normale) * normale)
    assert np.linalg.norm(riflessa) == pytest.approx(np.linalg.norm(velocita))
    assert riflessa @ normale == pytest.approx(-(velocita @ normale))
    # Lo spigolo fra la sfera esterna e il pavimento, davanti: la velocità va fuori da tutte e due.
    spigolo = np.array((0.0, zona.pavimento, math.sqrt(zona.esterno ** 2 - zona.pavimento ** 2)))
    verso_lo_spigolo = np.array((0.0, -3.0, 4.0))
    for parete in ("esterna", "pavimento"):
        riflessa = zona.rimbalzo(spigolo, verso_lo_spigolo, parete)
        assert np.linalg.norm(riflessa) == pytest.approx(5.0)
        assert riflessa @ zona.normale("esterna", spigolo) <= 1e-12
        assert riflessa @ zona.normale("pavimento", spigolo) <= 1e-12


def test_zona_punto_a_caso():
    """D28: il punto si estrae nell'ordine scritto, la distanza uniforme fra le due sfere della zona, l'azimut
    uniforme nel settore, il seno dell'elevazione uniforme fino a quello di ELEVAZIONE_MASSIMA, e si
    riestrae finché non sta dentro. Copre tutta la zona."""
    for raggio in (RAGGIO_MINIMO, RAGGIO_INIZIALE):
        zona = fisica.Zona(raggio)
        usato, rifatto = random.Random(raggio), random.Random(raggio)
        punti = []
        for _ in range(500):
            punto = zona.punto_a_caso(usato)
            while True:
                distanza = rifatto.uniform(zona.interno, zona.esterno)
                azimut = rifatto.uniform(-SETTORE_AZIMUT, SETTORE_AZIMUT)
                elevazione = math.degrees(math.asin(rifatto.uniform(0.0, math.sin(math.radians(ELEVAZIONE_MASSIMA)))))
                atteso = _sferico(distanza, azimut, elevazione)
                if zona.contiene(atteso, tolleranza=0.0):
                    break
            assert punto == pytest.approx(atteso)
            assert zona.contiene(punto, tolleranza=0.0)
            punti.append(punto)
        assert usato.random() == rifatto.random()
        punti = np.array(punti)
        distanze = np.linalg.norm(punti, axis=1)
        azimut = np.degrees(np.arctan2(punti[:, 0], punti[:, 2]))
        assert distanze.min() < zona.interno + 3.0 and distanze.max() > zona.esterno - 3.0
        assert azimut.min() < -SETTORE_AZIMUT + 8.0 and azimut.max() > SETTORE_AZIMUT - 8.0
        assert punti[:, 1].min() < zona.pavimento + 3.0 and punti[:, 1].max() > 0.8 * zona.esterno


# Il percorso dei bersagli mobili (D28).

def _percorso_a_caso(rng, raggio):
    zona = fisica.Zona(raggio)
    velocita = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
    velocita *= rng.uniform(3.0, 12.0) / np.linalg.norm(velocita)
    return fisica.Percorso(zona, zona.punto_a_caso(rng), velocita)


def test_percorso_resta_nella_zona_a_velocita_costante():
    """I mobili vanno dritti e rimbalzano sulle pareti della loro zona: in qualunque istante dentro, il moto
    continuo e a velocità costante, come nel cubo; tranne i passi a cavallo di un rimbalzo, tutti lunghi uguali."""
    rng = random.Random(28)
    passo = 0.005
    tempi = np.arange(0.0, 240.0, passo)
    for prova in range(24):
        raggio = (0.0, RAGGIO_MINIMO, 2.0, RAGGIO_INIZIALE)[prova % 4]
        percorso = _percorso_a_caso(rng, raggio)
        partenza, velocita = percorso._punti[0], percorso._velocita[0]
        rapidita = float(np.linalg.norm(velocita))
        posizioni = percorso.posizione(tempi)
        assert percorso.rimbalzi > 3, prova
        assert np.all(percorso.zona.scarti(posizioni) <= 1e-6), prova
        assert np.allclose(posizioni[0], partenza)
        passi = np.linalg.norm(np.diff(posizioni, axis=0), axis=1)
        assert passi.max() <= rapidita * passo * (1 + 1e-9), prova
        assert np.mean(np.isclose(passi, rapidita * passo, rtol=1e-6)) > 0.99, prova
        assert np.allclose(np.linalg.norm(percorso._velocita, axis=1), rapidita)


def test_percorso_coi_rimbalzi_sulle_pareti():
    """Ogni tratto comincia su una parete della zona, dove il tratto prima ne usciva: lì la velocità nuova
    va dentro, e cambia soltanto lungo le normali delle pareti toccate. Fino al primo rimbalzo il moto è
    quello rettilineo, e il primo rimbalzo è dove la Zona dice che il moto esce."""
    rng = random.Random(29)
    for prova in range(12):
        percorso = _percorso_a_caso(rng, (RAGGIO_MINIMO, RAGGIO_INIZIALE, 1.0)[prova % 3])
        zona = percorso.zona
        tempo, _parete = zona.uscita(percorso._punti[0], percorso._velocita[0])
        assert percorso._inizi[1] == pytest.approx(tempo)
        prima = np.linspace(0.0, tempo, 50)
        assert percorso.posizione(prima) == pytest.approx(percorso._punti[0] + percorso._velocita[0] * prima[:, None])
        for k in range(1, percorso.rimbalzi + 1):
            punto, prima_del_rimbalzo, dopo = percorso._punti[k], percorso._velocita[k - 1], percorso._velocita[k]
            assert punto == pytest.approx(percorso._punti[k - 1] + prima_del_rimbalzo * (percorso._inizi[k] - percorso._inizi[k - 1]))
            scarti = zona.scarti(punto)
            toccate = [parete for parete, scarto in zip(fisica.Zona.PARETI, scarti, strict=True) if abs(scarto) <= 1e-4]
            assert toccate and np.all(scarti <= 1e-6), (prova, k, scarti)
            normali = np.array([zona.normale(parete, punto) for parete in toccate])
            assert np.all(normali @ dopo <= 1e-9), (prova, k, toccate)
            # Il cambio di velocità sta tutto lungo le normali delle pareti toccate.
            cambio = dopo - prima_del_rimbalzo
            coefficienti = np.linalg.lstsq(normali.T, cambio, rcond=None)[0]
            assert normali.T @ coefficienti == pytest.approx(cambio, abs=1e-9), (prova, k, toccate)


def test_percorso_oltre_l_orizzonte_e_in_qualunque_ordine():
    """I tratti nascono fino a ORIZZONTE_DEL_PERCORSO e si allungano quando un istante va oltre; la posizione
    non dipende dall'ordine in cui si chiedono gli istanti. Mai prima della partenza."""
    rng = random.Random(30)
    zona = fisica.Zona(1.0)
    partenza = zona.punto_a_caso(rng)
    velocita = np.array((7.0, -2.0, 5.0))
    avanti = fisica.Percorso(zona, partenza, velocita)
    rimbalzi_alla_nascita = avanti.rimbalzi
    assert avanti._fine >= fisica.ORIZZONTE_DEL_PERCORSO
    tardi = np.array((3000.0, 3000.5))
    in_fondo = avanti.posizione(tardi)
    assert avanti.rimbalzi > rimbalzi_alla_nascita
    assert np.all(zona.scarti(in_fondo) <= 1e-6)
    tempi = np.linspace(0.0, 3001.0, 30001)
    indietro = fisica.Percorso(zona, partenza, velocita)
    assert indietro.posizione(tempi[::-1])[::-1] == pytest.approx(avanti.posizione(tempi))
    assert avanti.posizione(1234.5).shape == (3,)
    assert avanti.posizione(np.zeros((2, 5))).shape == (2, 5, 3)
    assert avanti.posizione(-10.0) == pytest.approx(partenza)


def test_percorso_libero_e_fermo():
    """Un moto che parte fuori dalla zona va dritto e basta, anche attraverso la zona; uno fermo resta dov'è."""
    zona = fisica.Zona(1.0)
    fuori = np.array((-60.0, 20.0, 40.0))
    velocita = np.array((6.0, 0.0, 0.0))
    libero = fisica.Percorso(zona, fuori, velocita)
    tempi = np.linspace(0.0, 30.0, 301)
    assert libero.posizione(tempi) == pytest.approx(fuori + velocita * tempi[:, None])
    assert zona.contiene(libero.posizione(10.0)) and libero.rimbalzi == 0
    fermo = fisica.Percorso(zona, fisica.MEZZO_DELLA_CUPOLA, (0.0, 0.0, 0.0))
    assert fermo.posizione(tempi) == pytest.approx(np.tile(fisica.MEZZO_DELLA_CUPOLA, (len(tempi), 1)))
    assert fermo.rimbalzi == 0


def test_percorso_che_parte_su_una_parete():
    """Un moto che parte proprio su una parete della zona, dove contiene lo dice dentro, e va verso fuori,
    rimbalza subito, come se fosse appena arrivato lì, e non esce: da un punto nel mezzo di ogni parete, con
    la velocità lungo la sua normale uscente, resta nella zona."""
    zona = fisica.Zona(1.0)
    limite = SETTORE_AZIMUT - math.degrees(math.asin(zona.raggio / (55.0 * math.cos(math.radians(30.0)))))
    sulle_pareti = {
        "esterna": _sferico(zona.esterno, 0.0, 30.0),
        "interna": _sferico(zona.interno, 0.0, 30.0),
        "pavimento": np.array((0.0, zona.pavimento, 50.0)),
        "destra": _sferico(55.0, limite, 30.0),
        "sinistra": _sferico(55.0, -limite, 30.0),
        "cono": _sferico(55.0, 0.0, zona.elevazione),
    }
    scappati = []
    for parete, punto in sulle_pareti.items():
        assert zona.contiene(punto) and abs(zona.scarti(punto)[fisica.Zona.PARETI.index(parete)]) < 1e-9, parete
        percorso = fisica.Percorso(zona, punto, 5.0 * zona.normale(parete, punto))
        peggiore = float(zona.scarti(percorso.posizione(np.arange(0.0, 30.0, 0.01))).max())
        if peggiore > 1e-6:
            scappati.append(f"dalla parete {parete} fino a {peggiore:.1f} metri fuori")
    assert not scappati, f"{len(scappati)} moti su {len(sulle_pareti)} escono dalla zona: " + "; ".join(scappati)


def test_percorso_da_piu_fili():
    """Il callback dell'audio e la finestra chiedono la posizione insieme: i tratti calcolati sotto il
    lucchetto sono gli stessi di un filo solo."""
    zona = fisica.Zona(RAGGIO_MINIMO)
    partenza = np.array(fisica.MEZZO_DELLA_CUPOLA)
    velocita = np.array((11.0, 3.0, -4.0))
    condiviso = fisica.Percorso(zona, partenza, velocita)
    tempi = [np.arange(0.0, 2000.0, 0.37) + scarto for scarto in (0.0, 0.11, 0.23, 0.31)]
    risultati = [None] * len(tempi)
    partenza_insieme = threading.Barrier(len(tempi))

    def chiedi(indice):
        partenza_insieme.wait()
        risultati[indice] = condiviso.posizione(tempi[indice])

    fili = [threading.Thread(target=chiedi, args=(indice,)) for indice in range(len(tempi))]
    for filo in fili:
        filo.start()
    for filo in fili:
        filo.join()
    for indice, chiesti in enumerate(tempi):
        assert risultati[indice] == pytest.approx(fisica.Percorso(zona, partenza, velocita).posizione(chiesti))


# Gli spigoli della cupola e il suo punto più lontano, la misura della precisione globale (D26, D28).

def test_punti_degli_spigoli():
    """Gli spigoli sono le curve dove due pareti della cupola s'incontrano: ogni punto sta nella Zona di raggio
    zero e su almeno due pareti, e fra loro ci sono gli otto angoli. Calcolati una volta sola, in sola lettura."""
    spigoli = fisica.punti_degli_spigoli()
    assert fisica.punti_degli_spigoli() is spigoli
    assert spigoli.ndim == 2 and spigoli.shape[1] == 3 and not spigoli.flags.writeable
    zona = fisica.Zona(0.0)
    assert zona.contiene(spigoli).all()
    assert np.all(np.count_nonzero(np.abs(zona.scarti(spigoli)) <= 1e-6, axis=1) >= 2)
    for angolo in _angoli_della_cupola():
        assert np.linalg.norm(spigoli - angolo, axis=1).min() < 1e-9


def test_distanza_dal_punto_piu_lontano():
    """D26 e D28: la distanza fra un punto e il punto della cupola più lontano da lui. La cupola, come il cubo,
    ha otto angoli, e da un punto davanti al cannone il più lontano è sempre uno di loro: i punti a caso della
    cupola, dentro e appena fuori dai lati come il bonus, lo dicono contro gli otto angoli contati qui, e
    nessun punto delle pareti, dei bordi o del dentro sta più lontano. Dall'origine, il centro della cupola,
    ogni punto della sfera esterna dista proprio CUPOLA_RAGGIO."""
    assert fisica.distanza_dal_punto_piu_lontano((0.0, 0.0, 0.0)) == pytest.approx(CUPOLA_RAGGIO)
    angoli = _angoli_della_cupola()
    rng = random.Random(26)
    zona = fisica.Zona(0.0)
    dentro = np.array([zona.punto_a_caso(rng) for _ in range(400)])
    di_lato = np.array([_sferico(rng.uniform(35.0, 75.0), lato * (SETTORE_AZIMUT + rng.uniform(0.0, 3.0)), rng.uniform(5.0, 60.0))
                        for lato in (-1.0, 1.0) for _ in range(20)])
    punti = np.concatenate((dentro, di_lato, angoli))
    distanze = fisica.distanza_dal_punto_piu_lontano(punti)
    assert isinstance(distanze, np.ndarray) and distanze.shape == (len(punti),)
    attese = np.max(np.linalg.norm(punti[:, None, :] - angoli[None, :, :], axis=-1), axis=1)
    assert distanze == pytest.approx(attese, abs=1e-9)
    for punto, distanza in zip(punti[:5], distanze[:5], strict=True):
        assert fisica.distanza_dal_punto_piu_lontano(punto) == pytest.approx(distanza)
    # Nessun punto della cupola, sulle pareti o dentro, sta più lontano.
    campione = np.concatenate((_bordi(zona, 13), np.array([zona.punto_a_caso(rng) for _ in range(3000)])))
    for punto in punti[::11]:
        assert np.linalg.norm(campione - punto, axis=1).max() <= fisica.distanza_dal_punto_piu_lontano(punto) + 1e-9
    # Nessun punto della cupola sta a meno di 60 metri dal suo punto più lontano: è la retta più corta della
    # precisione globale, che col cubo era mezza diagonale, 43,3 metri.
    assert fisica.distanza_dal_punto_piu_lontano(dentro).min() > 60.0


# La copertura: ogni punto dove può stare il centro di un bersaglio si raggiunge.

def _passa_dal_punto(valori, punto):
    """La distanza fra il punto e il tiro dei valori, nell'istante in cui il tiro ne raggiunge la
    distanza orizzontale."""
    azimut, alzo, velocita = fisica.parametri_del_tiro(*valori)
    traiettoria = fisica.Traiettoria(azimut, alzo, velocita)
    tempo = math.hypot(punto[0], punto[2]) / (velocita * math.cos(math.radians(alzo)))
    return float(np.linalg.norm(traiettoria.posizione(tempo) - np.asarray(punto)))


def _punti_della_zona(raggio, seme):
    zona = fisica.Zona(raggio)
    rng = random.Random(seme)
    return np.concatenate((_bordi(zona), np.array([zona.punto_a_caso(rng) for _ in range(PUNTI_A_CASO)])))


@pytest.mark.parametrize("raggio", [*RAGGI_DEL_GIOCO, RAGGIO_MINIMO * RAGGIO_BONUS_FATTORE])
def test_ogni_punto_della_cupola_si_raggiunge(raggio, monkeypatch):
    """D28: ogni punto della cupola resta raggiungibile. Per ogni raggio del gioco, e per il bonus più piccolo,
    i punti a caso della zona e quelli dei suoi bordi si raggiungono con i tre valori dentro le scale, da 0 a 1,
    e il tiro trovato ci passa davvero. Qui senza MARGINE_DI_COPERTURA: la prova coi margini è la seguente."""
    monkeypatch.setattr(fisica, "MARGINE_DI_COPERTURA", 0.0)
    scoperti = []
    for punto in _punti_della_zona(raggio, f"cupola {raggio}"):
        valori = fisica.soluzione(punto)
        if valori is None:
            scoperti.append(tuple(round(float(c), 2) for c in punto))
            continue
        assert all(0.0 <= v <= 1.0 for v in valori), (punto, valori)
        assert _passa_dal_punto(valori, punto) < 0.01, (punto, valori)
    assert not scoperti, f"{len(scoperti)} punti senza soluzione, il primo {scoperti[0]}"


def test_copertura_della_cupola_dentro_i_margini():
    """La copertura che il progetto vuole, scritta in costanti.py accanto a MARGINE_DI_COPERTURA: ogni punto della
    cupola dove può stare il centro di un bersaglio si raggiunge con i tre valori lontani almeno il margine dai
    capi delle scale, a ogni raggio del gioco, sui punti a caso della zona e sui suoi bordi. Il messaggio dice,
    raggio per raggio, quanti punti restano fuori dai margini, e se per la direzione, vicino ai lati del settore,
    o per l'alzo e la potenza, provati col punto girato davanti al cannone, perché la direzione non li cambia."""
    margine = MARGINE_DI_COPERTURA
    azimut_coi_margini = (1.0 - 2.0 * margine) * AZIMUT_MASSIMO
    difetti = []
    for raggio in RAGGI_DEL_GIOCO:
        di_lato, alzo_o_potenza, azimut_massimo, primo = 0, 0, 0.0, None
        for punto in _punti_della_zona(raggio, f"margini {raggio}"):
            valori = fisica.soluzione(punto)
            if valori is not None:
                assert all(margine <= v <= 1 - margine for v in valori), (punto, valori)
                assert _passa_dal_punto(valori, punto) < 0.01, (punto, valori)
                continue
            azimut = math.degrees(math.atan2(punto[0], punto[2]))
            if abs(azimut) > azimut_coi_margini:
                di_lato += 1
                azimut_massimo = max(azimut_massimo, abs(azimut))
            if fisica.soluzione((0.0, punto[1], math.hypot(punto[0], punto[2]))) is None:
                alzo_o_potenza += 1
                primo = primo or tuple(round(float(c), 2) for c in punto)
        if di_lato or alzo_o_potenza:
            riga = f"raggio {raggio:.3f}:"
            if di_lato:
                riga += f" {di_lato} punti per la direzione, con l'azimut fino a {azimut_massimo:.2f} gradi, e i margini arrivano a {azimut_coi_margini:.2f};"
            if alzo_o_potenza:
                riga += f" {alzo_o_potenza} punti per l'alzo o la potenza, il primo {primo};"
            difetti.append(riga)
    assert not difetti, "Punti fuori dai margini di copertura. " + " ".join(difetti)


def test_soluzione_sceglie_la_potenza_piu_bassa():
    punto = (5.0, 15.0, 45.0)
    valori = fisica.soluzione(punto)
    _, alzo_scelto, velocita_scelta = fisica.parametri_del_tiro(*valori)
    alzi = np.linspace(ALZO_MINIMO + 1.0, ALZO_MASSIMO - 1.0, 500)
    velocita = fisica.velocita_per(punto, alzi)
    assert velocita_scelta <= np.nanmin(velocita) + 1e-9
    distanza, dislivello = math.hypot(punto[0], punto[2]), punto[1] - ALTEZZA_CANNONE
    assert velocita_scelta == pytest.approx(math.sqrt(GRAVITA * (dislivello + math.hypot(dislivello, distanza))))
    assert alzo_scelto > 45.0


def test_soluzione_impossibile():
    assert fisica.soluzione((0.0, 500.0, 50.0)) is None
    assert fisica.soluzione((0.0, 10.0, -40.0)) is None
    assert fisica.soluzione((60.0, 10.0, 10.0)) is None
    # Troppo lontano: chiede più della velocità massima.
    assert fisica.soluzione((0.0, ALTEZZA_MINIMA_BERSAGLIO, 2.0 * CUPOLA_RAGGIO)) is None
