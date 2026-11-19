"""
Le prove di fisica.py: il tiro e la sua inversa, la parabola, il suono dello spazio, la scomposizione
dello scarto e la copertura del cubo.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
La prova della copertura è quella che il progetto vuole: ogni punto dove può stare il centro di un
bersaglio si raggiunge con i tre valori dentro i margini, e il tiro trovato ci passa davvero.
"""

import math

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
    CUBO_X,
    CUBO_Y,
    CUBO_Z,
    DB_PER_METRO,
    FREQUENZA_AL_SUOLO,
    GRAVITA,
    GUADAGNO_DB_MINIMO,
    GUADAGNO_DB_VICINO,
    MARGINE_DI_COPERTURA,
    METRI_PER_OTTAVA,
    RAGGIO_BONUS_FATTORE,
    RAGGIO_INIZIALE,
    RAGGIO_MINIMO,
    VELOCITA_MASSIMA,
    VELOCITA_MINIMA,
)

# I punti della griglia per asse nella prova della copertura: undici per lato, cioè un punto ogni
# 4 o 5 metri, capi compresi, dove i tiri sono più difficili.
PUNTI_PER_ASSE = 11


def test_parametri_del_tiro_ai_capi_e_al_mezzo():
    assert fisica.parametri_del_tiro(0.0, 0.0, 0.0) == pytest.approx((-AZIMUT_MASSIMO, ALZO_MINIMO, VELOCITA_MINIMA))
    assert fisica.parametri_del_tiro(1.0, 1.0, 1.0) == pytest.approx((AZIMUT_MASSIMO, ALZO_MASSIMO, VELOCITA_MASSIMA))
    azimut, alzo, _ = fisica.parametri_del_tiro(0.5, 0.5, 0.5)
    assert azimut == pytest.approx(0.0)
    assert alzo == pytest.approx((ALZO_MINIMO + ALZO_MASSIMO) / 2)


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


def test_valori_del_tiro_inversa():
    rng = np.random.default_rng(7)
    direzione, alzo, potenza = rng.random(100), rng.random(100), rng.random(100)
    indietro = fisica.valori_del_tiro(*fisica.parametri_del_tiro(direzione, alzo, potenza))
    assert np.allclose(indietro[0], direzione)
    assert np.allclose(indietro[1], alzo)
    assert np.allclose(indietro[2], potenza)
    assert fisica.valori_del_tiro(0.0, 0.0, VELOCITA_MINIMA) == pytest.approx((0.5, 0.0, 0.0))


def test_valori_del_tiro_fuori_scala_non_si_tagliano():
    direzione, _, potenza = fisica.valori_del_tiro(2 * AZIMUT_MASSIMO, 10.0, VELOCITA_MASSIMA + 10.0)
    assert direzione > 1.0
    assert potenza > 1.0


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


def test_tiro_orizzontale_e_rasoterra():
    piatto = fisica.Traiettoria(0.0, 0.0, 20.0)
    assert piatto.apice() == (0.0, ALTEZZA_CANNONE)
    assert piatto.tempo_a_terra() == pytest.approx(math.sqrt(2 * ALTEZZA_CANNONE / GRAVITA))
    da_terra = fisica.Traiettoria(0.0, 30.0, 20.0, origine=(0.0, 0.0, 0.0))
    assert da_terra.tempo_a_terra() == pytest.approx(2 * 20.0 * math.sin(math.radians(30.0)) / GRAVITA)


def test_pan_di():
    assert fisica.pan_di(0.0, 40.0) == pytest.approx(0.0)
    assert fisica.pan_di(10.0, 10.0) == pytest.approx(45.0 / AZIMUT_DEL_PAN)
    assert fisica.pan_di(-10.0, 10.0) == pytest.approx(-45.0 / AZIMUT_DEL_PAN)
    assert fisica.pan_di(100.0, 1.0) == 1.0
    assert fisica.pan_di(-100.0, 1.0) == -1.0
    pan = fisica.pan_di(np.array([-5.0, 0.0, 5.0]), np.array([50.0, 50.0, 50.0]))
    assert pan[0] == pytest.approx(-pan[2])
    assert pan[1] == 0.0


def test_frequenza_di():
    assert fisica.frequenza_di(0.0) == pytest.approx(FREQUENZA_AL_SUOLO)
    assert fisica.frequenza_di(METRI_PER_OTTAVA) == pytest.approx(2 * FREQUENZA_AL_SUOLO)
    assert fisica.frequenza_di(-5.0) == pytest.approx(FREQUENZA_AL_SUOLO)
    tetto = fisica.frequenza_di(ALTEZZA_TETTO_DEL_SUONO)
    assert fisica.frequenza_di(ALTEZZA_TETTO_DEL_SUONO + 100.0) == pytest.approx(tetto)


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


def test_zona_del_centro():
    for raggio in (RAGGIO_MINIMO, RAGGIO_INIZIALE):
        minimi, massimi = fisica.zona_del_centro(raggio)
        assert np.allclose(minimi, (CUBO_X[0] + raggio, max(CUBO_Y[0] + raggio, ALTEZZA_MINIMA_BERSAGLIO), CUBO_Z[0] + raggio))
        assert np.allclose(massimi, (CUBO_X[1] - raggio, CUBO_Y[1] - raggio, CUBO_Z[1] - raggio))
    minimi, massimi = fisica.zona_del_centro(1000.0)
    assert np.all(minimi <= massimi)


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


def _passa_dal_punto(valori, punto):
    """La distanza fra il punto e il tiro dei valori, nell'istante in cui il tiro ne raggiunge la
    distanza orizzontale."""
    azimut, alzo, velocita = fisica.parametri_del_tiro(*valori)
    traiettoria = fisica.Traiettoria(azimut, alzo, velocita)
    tempo = math.hypot(punto[0], punto[2]) / (velocita * math.cos(math.radians(alzo)))
    return float(np.linalg.norm(traiettoria.posizione(tempo) - np.asarray(punto)))


def _griglia(minimi, massimi):
    assi = [np.linspace(minimi[k], massimi[k], PUNTI_PER_ASSE) for k in range(3)]
    return np.stack(np.meshgrid(*assi, indexing="ij"), axis=-1).reshape(-1, 3)


def _controlla_copertura(punti):
    margine = MARGINE_DI_COPERTURA
    scoperti = []
    for punto in punti:
        valori = fisica.soluzione(punto)
        if valori is None:
            scoperti.append(tuple(punto))
            continue
        assert all(margine <= v <= 1 - margine for v in valori), (punto, valori)
        assert _passa_dal_punto(valori, punto) < 0.01, (punto, valori)
    assert not scoperti, f"{len(scoperti)} punti senza soluzione, il primo {scoperti[0]}"


@pytest.mark.parametrize("raggio", [RAGGIO_MINIMO, RAGGIO_INIZIALE, RAGGIO_MINIMO * RAGGIO_BONUS_FATTORE])
def test_copertura_del_cubo(raggio):
    """Ogni centro possibile di un bersaglio, al raggio più piccolo, a quello del primo livello e a
    quello del bonus più piccolo, si raggiunge con i tre valori dentro i margini di copertura."""
    _controlla_copertura(_griglia(*fisica.zona_del_centro(raggio)))


def test_copertura_della_traversata_del_bonus():
    """Il bonus entra ed esce dal cubo: anche i centri fuori dalle pareti laterali, finché tocca il
    cubo, si raggiungono."""
    raggio = RAGGIO_INIZIALE * RAGGIO_BONUS_FATTORE
    minimi, massimi = fisica.zona_del_centro(raggio)
    minimi[0], massimi[0] = CUBO_X[0] - raggio, CUBO_X[1] + raggio
    _controlla_copertura(_griglia(minimi, massimi))


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
