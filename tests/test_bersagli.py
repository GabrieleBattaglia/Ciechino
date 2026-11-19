"""
Le prove di bersagli.py: i livelli, le forme dell'LFO, il rimbalzo, il bonus, la generazione a caso
e i parametri sonori.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
"""

import itertools
import math
import random

import numpy as np
import pytest

import bersagli
import fisica
from costanti import (
    BERSAGLIO_DB,
    CUBO_LATO,
    CUBO_X,
    DURATA_BONUS_INIZIALE,
    DURATA_BONUS_MINIMA,
    FATTORE_ESCURSIONE,
    FORME_LFO,
    FREQUENZA_LFO,
    LIVELLI_CON_UN_BERSAGLIO_IN_PIU,
    LIVELLO_DEI_BONUS,
    LIVELLO_DEI_MOBILI,
    METRI_PER_OTTAVA,
    RAGGIO_BONUS_FATTORE,
    RAGGIO_INIZIALE,
    RAGGIO_MINIMO,
    VELOCITA_INIZIALE_BERSAGLI,
    VELOCITA_MASSIMA_BERSAGLI,
)

FS_DI_PROVA = 44100


def _tempi(secondi=3.0, inizio=0.0):
    return inizio + np.arange(int(secondi * FS_DI_PROVA)) / FS_DI_PROVA


def test_raggio_del_livello():
    assert bersagli.raggio_del_livello(1) == pytest.approx(RAGGIO_INIZIALE)
    raggi = [bersagli.raggio_del_livello(n) for n in range(1, 60)]
    assert all(a >= b for a, b in itertools.pairwise(raggi))
    assert raggi[-1] == pytest.approx(RAGGIO_MINIMO)
    assert min(raggi) >= RAGGIO_MINIMO


def test_velocita_del_livello():
    assert bersagli.velocita_del_livello(LIVELLO_DEI_MOBILI - 1) == 0.0
    assert bersagli.velocita_del_livello(LIVELLO_DEI_MOBILI) == pytest.approx(VELOCITA_INIZIALE_BERSAGLI)
    assert bersagli.velocita_del_livello(1000) == pytest.approx(VELOCITA_MASSIMA_BERSAGLI)


def test_quanti_bersagli():
    assert bersagli.quanti_bersagli(1) == 1
    for k, livello in enumerate(LIVELLI_CON_UN_BERSAGLIO_IN_PIU, start=2):
        assert bersagli.quanti_bersagli(livello - 1) == k - 1
        assert bersagli.quanti_bersagli(livello) == k


def test_durata_del_bonus():
    assert bersagli.durata_del_bonus(LIVELLO_DEI_BONUS) == pytest.approx(DURATA_BONUS_INIZIALE)
    assert bersagli.durata_del_bonus(1000) == pytest.approx(DURATA_BONUS_MINIMA)


@pytest.mark.parametrize("forma", FORME_LFO)
def test_lfo_fra_meno_uno_e_uno_e_forma_conservata(forma):
    tempi = _tempi()
    valori = bersagli.lfo(forma, tempi, 4.3, fase=0.37, seme=12345)
    assert valori.shape == tempi.shape
    assert np.all(valori >= -1.0)
    assert np.all(valori <= 1.0)
    assert valori.max() - valori.min() > 0.5
    assert isinstance(bersagli.lfo(forma, 0.1, 4.3, seme=1), float)
    assert bersagli.lfo(forma, tempi.reshape(-1, 100), 4.3, seme=1).shape == (len(tempi) // 100, 100)


@pytest.mark.parametrize("forma", FORME_LFO)
def test_lfo_senza_stato(forma):
    """A pezzi o tutto insieme, i valori sono gli stessi: il motore lo chiede blocco per blocco."""
    tempi = _tempi(2.0)
    tutto = bersagli.lfo(forma, tempi, 3.1, fase=0.2, seme=99)
    pezzi = np.concatenate([bersagli.lfo(forma, pezzo, 3.1, fase=0.2, seme=99) for pezzo in np.array_split(tempi, 37)])
    assert np.array_equal(tutto, pezzi)


def test_lfo_forme_note():
    frequenza = 2.0
    quarto = 1.0 / (4 * frequenza)
    assert bersagli.lfo("seno", quarto, frequenza) == pytest.approx(1.0)
    assert bersagli.lfo("triangolo", 0.0, frequenza) == pytest.approx(0.0)
    assert bersagli.lfo("triangolo", quarto, frequenza) == pytest.approx(1.0)
    assert bersagli.lfo("triangolo", 3 * quarto, frequenza) == pytest.approx(-1.0)
    assert bersagli.lfo("sega", 0.0, frequenza) == pytest.approx(-1.0)
    assert bersagli.lfo("sega", 2 * quarto, frequenza) == pytest.approx(0.0)
    assert bersagli.lfo("sega_inversa", 0.0, frequenza) == pytest.approx(1.0)
    assert bersagli.lfo("quadra", quarto, frequenza) == pytest.approx(1.0)
    assert bersagli.lfo("quadra", 3 * quarto, frequenza) == pytest.approx(-1.0)
    # La fase è una frazione di ciclo: mezzo ciclo rovescia il seno.
    assert bersagli.lfo("seno", quarto, frequenza, fase=0.5) == pytest.approx(-1.0)
    sega = bersagli.lfo("sega", np.linspace(0.0, 0.49, 100), frequenza)
    assert np.all(np.diff(sega) > 0.0)
    inversa = bersagli.lfo("sega_inversa", np.linspace(0.0, 0.49, 100), frequenza)
    assert np.all(np.diff(inversa) < 0.0)


def test_quadra_coi_fronti_smussati():
    frequenza = 6.0
    tempi = _tempi(2.0)
    quadra = bersagli.lfo("quadra", tempi, frequenza)
    # Una quadra netta salterebbe di 2 in un campione; il raccordo distribuisce il salto.
    salto_massimo = np.abs(np.diff(quadra)).max()
    assert salto_massimo < 0.05
    durata_fronte = np.count_nonzero(np.abs(quadra) < 0.99) / (2 * frequenza * tempi[-1]) / FS_DI_PROVA
    assert durata_fronte < 2 * bersagli.RACCORDO_DELLA_QUADRA
    assert np.count_nonzero(np.abs(quadra) > 0.999) / len(quadra) > 0.9


def test_casuale_tenuto_nel_ciclo_e_deterministico():
    frequenza = 5.0
    tempi = _tempi(4.0)
    valori = bersagli.lfo("casuale", tempi, frequenza, seme=42)
    cicli = np.floor(tempi * frequenza).astype(int)
    for ciclo in np.unique(cicli):
        nel_ciclo = valori[cicli == ciclo]
        assert np.all(nel_ciclo == nel_ciclo[0])
    per_ciclo = [valori[cicli == c][0] for c in np.unique(cicli)]
    assert len(set(per_ciclo)) == len(per_ciclo)
    assert np.array_equal(valori, bersagli.lfo("casuale", tempi, frequenza, seme=42))
    assert not np.array_equal(valori, bersagli.lfo("casuale", tempi, frequenza, seme=43))


def test_casuale_ben_distribuito():
    valori = bersagli.lfo("casuale", np.arange(20000) + 0.5, 1.0, seme=7)
    assert abs(float(np.mean(valori))) < 0.03
    assert valori.min() < -0.99
    assert valori.max() > 0.99


def test_deriva_continua_e_raccordata():
    frequenza = 4.0
    tempi = _tempi(4.0)
    deriva = bersagli.lfo("deriva", tempi, frequenza, seme=5)
    assert np.abs(np.diff(deriva)).max() < 2 * math.pi * frequenza / FS_DI_PROVA
    assert np.array_equal(deriva, bersagli.lfo("deriva", tempi, frequenza, seme=5))
    # All'inizio di ogni ciclo deriva vale ciò che casuale tiene in quel ciclo.
    inizi = np.arange(10) / frequenza
    assert np.allclose(bersagli.lfo("deriva", inizi, frequenza, seme=5), bersagli.lfo("casuale", inizi, frequenza, seme=5))


def test_lfo_forma_sconosciuta():
    with pytest.raises(ValueError, match="sconosciuta"):
        bersagli.lfo("zigzag", 0.0, 3.0)
    with pytest.raises(ValueError, match="sconosciuta"):
        bersagli.Bersaglio(numero=1, forma="zigzag", raggio=1.0, partenza=(0.0, 10.0, 50.0))


def test_rimbalzo_dentro_continuo_a_velocita_costante():
    rng = np.random.default_rng(3)
    minimi = np.array((-20.0, 4.0, 35.0))
    massimi = np.array((20.0, 45.0, 75.0))
    passo = 0.001
    tempi = np.arange(0.0, 60.0, passo)
    for _ in range(20):
        partenza = rng.uniform(minimi, massimi)
        velocita = rng.normal(size=3)
        velocita *= rng.uniform(0.5, 12.0) / np.linalg.norm(velocita)
        posizioni = bersagli.rimbalzo(partenza + velocita * tempi[:, None], minimi, massimi)
        assert np.all(posizioni >= minimi - 1e-9)
        assert np.all(posizioni <= massimi + 1e-9)
        assert np.allclose(posizioni[0], partenza)
        passi = np.linalg.norm(np.diff(posizioni, axis=0), axis=1)
        attesa = np.linalg.norm(velocita) * passo
        # Continua: nessun passo più lungo del moto libero. Velocità costante: tranne i passi a
        # cavallo di un rimbalzo, tutti lunghi uguali; e su ogni asse il passo vale la componente.
        assert passi.max() <= attesa * (1 + 1e-9)
        assert np.mean(np.isclose(passi, attesa, rtol=1e-6)) > 0.99
        per_asse = np.abs(np.diff(posizioni, axis=0))
        assert np.all(per_asse <= np.abs(velocita) * passo * (1 + 1e-9) + 1e-12)


def test_rimbalzo_asse_chiuso():
    posizioni = bersagli.rimbalzo(np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]), (0.0, 5.0, 0.0), (10.0, 5.0, 10.0))
    assert np.allclose(posizioni[:, 1], 5.0)


def test_bersaglio_fermo():
    bersaglio = bersagli.Bersaglio(numero=1, forma="seno", raggio=2.0, partenza=(3.0, 20.0, 50.0), nato=10.0)
    assert not bersaglio.mobile
    assert bersaglio.velocita_scalare == 0.0
    assert bersaglio.tempo_di_uscita is None
    assert np.allclose(bersaglio.posizione(100.0), (3.0, 20.0, 50.0))
    assert bersaglio.posizione(np.array([10.0, 20.0, 30.0])).shape == (3, 3)
    assert bersaglio.vivo(10.0) is True
    assert bersaglio.vivo(1e6) is True
    assert bersaglio.vivo(9.99) is False
    assert np.array_equal(bersaglio.vivo(np.array([9.0, 10.0, 11.0])), [False, True, True])
    assert bersaglio.mezza_escursione_ottave() == pytest.approx(2.0 * FATTORE_ESCURSIONE / METRI_PER_OTTAVA)
    assert bersaglio == bersaglio
    assert bersaglio != bersagli.Bersaglio(numero=1, forma="seno", raggio=2.0, partenza=(3.0, 20.0, 50.0), nato=10.0)


def test_bersaglio_mobile_resta_nella_zona():
    raggio = 1.5
    bersaglio = bersagli.Bersaglio(numero=2, forma="sega", raggio=raggio, partenza=(0.0, 25.0, 55.0), velocita=(9.0, -4.0, 6.0), nato=5.0)
    assert bersaglio.mobile
    assert bersaglio.velocita_scalare == pytest.approx(math.sqrt(81 + 16 + 36))
    assert bersaglio.tempo_di_uscita is None
    minimi, massimi = fisica.zona_del_centro(raggio)
    posizioni = bersaglio.posizione(np.arange(5.0, 200.0, 0.01))
    assert np.all(posizioni >= minimi - 1e-9)
    assert np.all(posizioni <= massimi + 1e-9)
    assert np.allclose(posizioni[0], (0.0, 25.0, 55.0))
    # Prima di un rimbalzo il moto è rettilineo.
    assert np.allclose(bersaglio.posizione(5.5), (4.5, 23.0, 58.0))


def test_genera_bersagli_nella_zona():
    rng = random.Random(11)
    for livello in (1, LIVELLO_DEI_MOBILI - 1, LIVELLO_DEI_MOBILI, 12, 30):
        for numero in range(40):
            bersaglio = bersagli.genera(livello, rng, numero, 7.5)
            raggio = bersagli.raggio_del_livello(livello)
            minimi, massimi = fisica.zona_del_centro(raggio)
            assert bersaglio.numero == numero
            assert bersaglio.nato == 7.5
            assert not bersaglio.bonus
            assert bersaglio.raggio == pytest.approx(raggio)
            assert bersaglio.forma in FORME_LFO
            assert FREQUENZA_LFO[0] <= bersaglio.frequenza_lfo <= FREQUENZA_LFO[1]
            assert 0.0 <= bersaglio.fase_lfo < 1.0
            assert np.all(bersaglio.partenza >= minimi)
            assert np.all(bersaglio.partenza <= massimi)
            assert bersaglio.velocita_scalare == pytest.approx(bersagli.velocita_del_livello(livello))
            assert bersaglio.mobile == (livello >= LIVELLO_DEI_MOBILI)


def test_genera_deterministico_e_vario():
    primi = [bersagli.genera(5, random.Random(21), n, 0.0) for n in range(3)]
    secondi = [bersagli.genera(5, random.Random(21), n, 0.0) for n in range(3)]
    for a, b in zip(primi, secondi, strict=True):
        assert np.array_equal(a.partenza, b.partenza)
        assert np.array_equal(a.velocita, b.velocita)
        assert (a.forma, a.frequenza_lfo, a.fase_lfo, a.seme) == (b.forma, b.frequenza_lfo, b.fase_lfo, b.seme)
    rng = random.Random(4)
    forme = {bersagli.genera(1, rng, n, 0.0).forma for n in range(300)}
    assert forme == set(FORME_LFO)
    direzioni = np.array([bersagli.genera(LIVELLO_DEI_MOBILI, rng, n, 0.0).velocita for n in range(2000)])
    assert np.allclose(direzioni.mean(axis=0), 0.0, atol=0.15)


def test_genera_bonus_attraversa_il_cubo():
    rng = random.Random(8)
    lati = set()
    for numero in range(60):
        livello = LIVELLO_DEI_BONUS + numero % 20
        bonus = bersagli.genera_bonus(livello, rng, numero, 30.0)
        raggio = bersagli.raggio_del_livello(livello) * RAGGIO_BONUS_FATTORE
        durata = bersagli.durata_del_bonus(livello)
        assert bonus.bonus
        assert bonus.mobile
        assert bonus.raggio == pytest.approx(raggio)
        assert bonus.velocita[1] == 0.0
        assert bonus.velocita[2] == 0.0
        assert abs(bonus.velocita[0]) == pytest.approx((CUBO_LATO + 2 * raggio) / durata)
        assert bonus.tempo_di_uscita == pytest.approx(30.0 + durata)
        minimi, massimi = fisica.zona_del_centro(raggio)
        assert minimi[1] <= bonus.partenza[1] <= massimi[1]
        assert minimi[2] <= bonus.partenza[2] <= massimi[2]
        # Nasce tangente al cubo da fuori, ed esce del tutto dall'altra parte.
        inizio = bonus.posizione(30.0)
        fine = bonus.posizione(bonus.tempo_di_uscita)
        assert abs(inizio[0]) == pytest.approx(CUBO_X[1] + raggio)
        assert fine[0] == pytest.approx(-inizio[0])
        assert np.allclose(fine[1:], inizio[1:])
        lati.add(math.copysign(1.0, bonus.velocita[0]))
        # Vivo dal suo nato all'uscita, non prima e non dopo.
        assert bonus.vivo(30.0)
        assert bonus.vivo(30.0 + durata / 2)
        assert bonus.vivo(bonus.tempo_di_uscita)
        assert not bonus.vivo(29.9)
        assert not bonus.vivo(bonus.tempo_di_uscita + 0.01)
        mezzo = bonus.posizione(30.0 + durata / 2)
        assert mezzo[0] == pytest.approx(0.0, abs=1e-9)
    assert lati == {-1.0, 1.0}


def test_bonus_fermo_non_esce_mai():
    bonus = bersagli.Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=(0.0, 10.0, 50.0), bonus=True)
    assert bonus.tempo_di_uscita == math.inf
    assert bonus.vivo(1e9)


def test_parametri_sonori():
    raggio = 3.0
    bersaglio = bersagli.Bersaglio(numero=1, forma="triangolo", raggio=raggio, partenza=(-6.0, 20.0, 50.0), velocita=(2.0, 0.0, 0.0),
                                   nato=0.0, frequenza_lfo=3.0, fase_lfo=0.1, seme=9)
    tempi_audio = _tempi(1.0, inizio=12.0)
    tempi_di_gioco = np.linspace(0.0, 1.0, len(tempi_audio))
    frequenze, pan, guadagni = bersaglio.parametri_sonori(tempi_di_gioco, tempi_audio)
    assert frequenze.shape == pan.shape == guadagni.shape == tempi_audio.shape
    posizioni = bersaglio.posizione(tempi_di_gioco)
    centro, pan_atteso, guadagni_attesi = fisica.suono_di(posizioni)
    assert np.allclose(pan, pan_atteso)
    assert np.allclose(guadagni, guadagni_attesi * 10 ** (BERSAGLIO_DB / 20))
    assert np.allclose(frequenze, centro * 2.0 ** (bersaglio.lfo(tempi_audio) * bersaglio.mezza_escursione_ottave()))
    # L'oscillazione copre le altezze del bersaglio, da y meno r a y più r.
    rapporto = frequenze / centro
    assert rapporto.max() == pytest.approx(fisica.frequenza_di(20.0 + raggio * FATTORE_ESCURSIONE) / fisica.frequenza_di(20.0), rel=1e-3)
    assert rapporto.min() == pytest.approx(fisica.frequenza_di(20.0 - raggio * FATTORE_ESCURSIONE) / fisica.frequenza_di(20.0), rel=1e-3)
    # Il pan si muove col bersaglio verso destra.
    assert pan[-1] > pan[0]


def test_parametri_sonori_col_tempo_di_gioco_fermo():
    bersaglio = bersagli.Bersaglio(numero=1, forma="seno", raggio=2.0, partenza=(0.0, 20.0, 50.0), velocita=(5.0, 0.0, 0.0), frequenza_lfo=4.0)
    tempi_audio = _tempi(0.5)
    frequenze, pan, guadagni = bersaglio.parametri_sonori(7.0, tempi_audio)
    assert frequenze.shape == pan.shape == guadagni.shape == tempi_audio.shape
    assert np.all(pan == pan[0])
    assert np.all(guadagni == guadagni[0])
    assert frequenze.max() > frequenze.min()
