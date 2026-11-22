"""
Le prove di bersagli.py: i livelli, le forme dell'LFO coi loro parametri a caso, il moto coi rimbalzi, il
bonus, la generazione a caso, con le portanti e l'escursione, e i parametri sonori.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Dalla 2.0.0 i bersagli stanno nella cupola di D28: la funzione rimbalzo del cubo e le sue prove escono, e
le prove nuove guardano la Zona e il Percorso di ogni bersaglio, la partenza estratta con Zona.punto_a_caso,
il bonus che attraversa il settore su una corda a quota e profondità costanti, il suo tempo di uscita dal
lato verso cui va, e la traversata che si raggiunge tutta con la mira.
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
    BERSAGLIO_FORMA,
    BERSAGLIO_PORTANTI,
    DURATA_BONUS_INIZIALE,
    DURATA_BONUS_MINIMA,
    ELEVAZIONE_MASSIMA,
    ESCURSIONE_A_CASO,
    FATTORE_ESCURSIONE,
    FORME_LFO,
    FREQUENZA_LFO,
    FRONTE_DELLA_QUADRA,
    LARGHEZZA_DELLA_QUADRA,
    LIVELLI_CON_UN_BERSAGLIO_IN_PIU,
    LIVELLO_DEI_BONUS,
    LIVELLO_DEI_MOBILI,
    METRI_PER_OTTAVA,
    PESO_DELLA_DOPPIA,
    RAGGIO_BONUS_FATTORE,
    RAGGIO_INIZIALE,
    RAGGIO_MINIMO,
    RAPPORTI_DELLA_DOPPIA,
    SETTORE_AZIMUT,
    SPINTA_DEL_CASO,
    VALORI_DEL_CASUALE,
    VELOCITA_INIZIALE_BERSAGLI,
    VELOCITA_MASSIMA_BERSAGLI,
)
from sintesi import FORME_PORTANTI

FS_DI_PROVA = 44100
# Le forme della 0.1.0, che coi parametri di serie suonano come allora, e le nuove di D18.
FORME_DELLA_0_1_0 = ("seno", "sega", "sega_inversa", "triangolo", "quadra", "casuale", "deriva")
FORME_NUOVE = tuple(forma for forma in FORME_LFO if forma not in FORME_DELLA_0_1_0)
# Le forme che saltano per natura, senza raccordo, come nella 0.1.0; e quelle fatte di valori a caso,
# che arrivano vicino agli estremi in qualche decina di cicli invece che in uno o due.
FORME_COI_SALTI = ("sega", "sega_inversa", "casuale")
FORME_A_CASO = ("casuale", "deriva")
PUNTO = (0.0, 20.0, 50.0)


def _tempi(secondi=3.0, inizio=0.0):
    return inizio + np.arange(int(secondi * FS_DI_PROVA)) / FS_DI_PROVA


def _estrazioni(forma, quante=12):
    """Tante voci a caso di una forma, come nel gioco: parametri, frequenza, fase e seme, sempre le stesse."""
    rng = random.Random(f"prove di {forma}")
    for _ in range(quante):
        yield bersagli.parametri_a_caso(forma, rng), rng.uniform(*FREQUENZA_LFO), rng.random(), rng.getrandbits(32)


def _lfo_della_0_1_0(forma, tempi, frequenza, fase=0.0, seme=0):
    """L'LFO della 0.1.0, riscritto qui com'era: i parametri di serie devono rendere proprio lui."""
    fasi = np.asarray(tempi, dtype=float) * frequenza + fase
    cicli = np.floor(fasi)
    frazioni = fasi - cicli
    triangolo = 1.0 - 4.0 * np.abs(np.mod(fasi + 0.25, 1.0) - 0.5)
    if forma == "seno":
        return np.sin(2.0 * math.pi * fasi)
    if forma == "sega":
        return 2.0 * frazioni - 1.0
    if forma == "sega_inversa":
        return 1.0 - 2.0 * frazioni
    if forma == "triangolo":
        return triangolo
    if forma == "quadra":
        mezzo = min(bersagli.RACCORDO_DELLA_QUADRA * frequenza / 2.0, 0.25)
        return np.sin(math.pi / 2.0 * np.clip(triangolo / (4.0 * mezzo), -1.0, 1.0))
    if forma == "casuale":
        return bersagli._caso(seme, cicli)
    da, verso = bersagli._caso(seme, cicli), bersagli._caso(seme, cicli + 1.0)
    return da + (verso - da) * (1.0 - np.cos(math.pi * frazioni)) / 2.0


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


# Le forme dell'LFO.

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
    parametri, frequenza, fase, seme = next(_estrazioni(forma))
    for scelti in (None, parametri):
        tutto = bersagli.lfo(forma, tempi, frequenza, fase=fase, seme=seme, parametri=scelti)
        pezzi = np.concatenate([bersagli.lfo(forma, pezzo, frequenza, fase=fase, seme=seme, parametri=scelti) for pezzo in np.array_split(tempi, 37)])
        assert np.array_equal(tutto, pezzi)


@pytest.mark.parametrize("forma", FORME_DELLA_0_1_0)
def test_i_parametri_di_serie_rendono_la_forma_della_0_1_0(forma):
    tempi = _tempi(2.0, inizio=3.3)
    attesi = _lfo_della_0_1_0(forma, tempi, 4.3, 0.37, 12345)
    assert np.allclose(bersagli.lfo(forma, tempi, 4.3, fase=0.37, seme=12345), attesi, rtol=0.0, atol=1e-9)
    bersaglio = bersagli.Bersaglio(numero=1, forma=forma, raggio=1.0, partenza=PUNTO, frequenza_lfo=4.3, fase_lfo=0.37, seme=12345)
    assert np.allclose(bersaglio.lfo(tempi), attesi, rtol=0.0, atol=1e-9)
    assert bersaglio.portante == BERSAGLIO_FORMA and bersaglio.escursione == 1.0
    assert bersaglio.parametri_lfo == bersagli.parametri_di_serie(forma)


@pytest.mark.parametrize("forma", FORME_LFO)
def test_ogni_forma_coi_parametri_a_caso_sta_fra_gli_estremi_e_li_tocca(forma):
    """Fra meno 1 e 1 sempre; gli estremi toccati, o quasi, in due cicli, e casuale e deriva in sessanta,
    così l'escursione dice la grandezza."""
    for parametri, frequenza, fase, seme in _estrazioni(forma):
        cicli = 60 if forma in FORME_A_CASO else 2
        tempi = np.arange(int(cicli / frequenza * FS_DI_PROVA)) / FS_DI_PROVA
        valori = bersagli.lfo(forma, tempi, frequenza, fase=fase, seme=seme, parametri=parametri)
        assert np.all(np.abs(valori) <= 1.0), parametri
        vicino = 0.75 if forma in FORME_A_CASO else 0.99
        assert valori.max() >= vicino and valori.min() <= -vicino, (parametri, valori.max(), valori.min())


@pytest.mark.parametrize("forma", FORME_LFO)
def test_ogni_forma_coi_parametri_a_caso_ha_la_media_vicina_a_zero(forma):
    """La media nel tempo entro 0,1: l'altezza percepita è quella del centro del bersaglio."""
    for parametri, frequenza, fase, seme in _estrazioni(forma):
        # Quattromila cicli, trentadue punti a ciclo, scostati dall'inizio perché non battano coi salti.
        tempi = (np.arange(4000 * 32) + 0.37) / (32 * frequenza)
        media = float(np.mean(bersagli.lfo(forma, tempi, frequenza, fase=fase, seme=seme, parametri=parametri)))
        assert abs(media) <= 0.1, (parametri, media)


@pytest.mark.parametrize("forma", [forma for forma in FORME_LFO if forma not in FORME_COI_SALTI])
def test_ogni_forma_continua_non_scatta(forma):
    """Dove la forma è continua, e dove i salti hanno il raccordo, nessun campione scatta: al massimo
    qualche centesimo della scala, anche a 6 hertz."""
    for parametri, _frequenza, fase, seme in _estrazioni(forma):
        tempi = _tempi(1.0)
        valori = bersagli.lfo(forma, tempi, FREQUENZA_LFO[1], fase=fase, seme=seme, parametri=parametri)
        assert np.abs(np.diff(valori)).max() < 0.05, parametri


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
    assert bersagli.lfo("impulsi", quarto, frequenza) == pytest.approx(1.0)
    assert bersagli.lfo("impulsi", 3 * quarto, frequenza) == pytest.approx(-1.0)
    assert bersagli.lfo("molla", 0.0, frequenza) == pytest.approx(0.0, abs=1e-12)
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
    # All'inizio di ogni ciclo deriva vale ciò che casuale tiene in quel ciclo, anche con la stessa spinta.
    inizi = np.arange(10) / frequenza
    assert np.allclose(bersagli.lfo("deriva", inizi, frequenza, seme=5), bersagli.lfo("casuale", inizi, frequenza, seme=5))
    spinta = {"spinta": 0.5}
    assert np.allclose(bersagli.lfo("deriva", inizi, frequenza, seme=5, parametri=spinta), bersagli.lfo("casuale", inizi, frequenza, seme=5, parametri=spinta))


def test_lfo_forma_sconosciuta():
    with pytest.raises(ValueError, match="sconosciuta"):
        bersagli.lfo("zigzag", 0.0, 3.0)
    with pytest.raises(ValueError, match="sconosciuta"):
        bersagli.Bersaglio(numero=1, forma="zigzag", raggio=1.0, partenza=(0.0, 10.0, 50.0))
    with pytest.raises(ValueError, match="sconosciuta"):
        bersagli.parametri_di_serie("zigzag")


# I parametri delle forme fanno ciò che dicono.

def _al_centro_dei_posti(forma, parametri, quanti, frequenza=3.0, cicli=2):
    """I valori al centro di ciascun posto del ciclo, lontano dai raccordi."""
    tempi = (np.arange(quanti * cicli) + 0.5) / (quanti * frequenza)
    return bersagli.lfo(forma, tempi, frequenza, parametri=parametri)


def test_i_parametri_delle_forme_della_0_1_0():
    frequenza = 2.0
    tempi = np.arange(4000) / 4000 / frequenza
    # Il seno con l'armonica: la seconda armonica pesa quanto dice, e il minimo è il meno del massimo.
    for armonica in (-0.3, 0.2):
        seno = bersagli.lfo("seno", tempi, frequenza, parametri={"armonica": armonica})
        spettro = np.abs(np.fft.rfft(seno))
        assert spettro[2] / spettro[1] == pytest.approx(abs(armonica), rel=1e-6)
        assert seno.max() == pytest.approx(1.0, abs=1e-5) and seno.min() == pytest.approx(-1.0, abs=1e-5)
    # La sega curvata passa per gli stessi capi e per il centro, ma a metà rampa vale meno o più.
    meta = 0.25 / frequenza
    assert bersagli.lfo("sega", meta, frequenza, parametri={"curvatura": 0.6}) == pytest.approx(-(0.5 ** 0.6))
    assert bersagli.lfo("sega_inversa", meta, frequenza, parametri={"curvatura": 1.6}) == pytest.approx(0.5 ** 1.6)
    # Il triangolo sale per la parte pendenza del ciclo.
    for pendenza in (0.3, 0.7):
        triangolo = bersagli.lfo("triangolo", tempi, frequenza, parametri={"pendenza": pendenza})
        assert np.mean(np.diff(triangolo) > 0.0) == pytest.approx(pendenza, abs=0.002)
    # La quadra sta in alto per la parte larghezza del ciclo, e il fronte allarga le rampe.
    quadra = bersagli.lfo("quadra", tempi, frequenza, parametri={"larghezza": 0.46, "fronte": 0.2})
    assert np.mean(quadra > 0.0) == pytest.approx(0.46, abs=0.002)
    assert np.mean(np.abs(quadra) < 0.999) > 0.3
    # Il casuale cambia valore tante volte per ciclo quante dice valori.
    for valori in (2, 3):
        casuale = bersagli.lfo("casuale", tempi, frequenza, seme=3, parametri={"valori": valori})
        assert np.count_nonzero(np.diff(casuale)) == valori - 1
    # La deriva tiene il valore fermo per la parte del ciclo che la morbidezza lascia.
    deriva = bersagli.lfo("deriva", tempi, frequenza, seme=3, parametri={"morbidezza": 0.4})
    assert np.all(deriva[:int(0.6 * len(tempi))] == deriva[0]) and not np.all(deriva == deriva[0])
    # La spinta porta i valori a caso verso gli estremi.
    lunghi = np.arange(20000) + 0.5
    uniformi = np.abs(bersagli.lfo("casuale", lunghi, 1.0, seme=7))
    spinti = np.abs(bersagli.lfo("casuale", lunghi, 1.0, seme=7, parametri={"spinta": 0.4}))
    assert np.mean(spinti > 0.8) > 2 * np.mean(uniformi > 0.8)


def test_la_molla_guizza_in_su_e_poi_in_giu():
    frequenza = 3.0
    parametri = {"oscillazioni": 2, "smorzamento": 4.0}
    tempi = np.arange(2 * 3000) / 3000 / frequenza
    molla = bersagli.lfo("molla", tempi, frequenza, parametri=parametri)
    primo, secondo = molla[:3000], molla[3000:]
    assert np.allclose(secondo, -primo)
    assert primo.max() == pytest.approx(1.0, abs=1e-5) and -0.99 < primo.min() < -0.3
    assert secondo.min() == pytest.approx(-1.0, abs=1e-5)
    # Parte da zero e ci torna, quindi il ciclo seguente comincia dove l'altro finisce.
    assert bersagli.lfo("molla", np.arange(6) / frequenza, frequenza, parametri=parametri) == pytest.approx([0.0] * 6, abs=1e-12)


def test_i_gradini_salgono_o_scendono():
    for gradini in (3, 5):
        livelli = np.linspace(-1.0, 1.0, gradini)
        salita = _al_centro_dei_posti("gradini", {"gradini": gradini, "verso": 1}, gradini)
        discesa = _al_centro_dei_posti("gradini", {"gradini": gradini, "verso": -1}, gradini)
        assert np.allclose(salita, np.tile(livelli, 2)) and np.allclose(discesa, np.tile(livelli[::-1], 2))


def test_il_trillo_alterna_due_note_in_alto_e_in_basso():
    for note, profondita in ((2, 0.25), (3, 0.5)):
        alto = [1.0 - profondita * (posto % 2) for posto in range(note)]
        trillo = _al_centro_dei_posti("trillo", {"note": note, "profondita": profondita}, 2 * note)
        assert np.allclose(trillo, np.tile(alto + [-x for x in alto], 2))


def test_gli_impulsi_lasciano_il_centro_fermo():
    frequenza = 2.0
    tempi = np.arange(8000) / 4000 / frequenza
    for larghezza in (0.1, 0.3):
        impulsi = bersagli.lfo("impulsi", tempi, frequenza, parametri={"larghezza": larghezza})
        assert np.mean(impulsi == 0.0) == pytest.approx(1.0 - 2.0 * larghezza, abs=0.002)
        assert impulsi.max() == pytest.approx(1.0) and impulsi.min() == pytest.approx(-1.0)


def test_la_doppia_ha_due_frequenze():
    frequenza = 2.0
    tempi = np.arange(8000) / 2000 / frequenza
    for rapporto, peso in ((1.5, 0.4), (3.5, 0.8)):
        doppia = bersagli.lfo("doppia", tempi, frequenza, parametri={"rapporto": rapporto, "peso": peso})
        spettro = np.abs(np.fft.rfft(doppia))
        # Quattro cicli: la prima frequenza è la riga 4, la seconda la riga 4 per il rapporto.
        seconda = round(4 * rapporto)
        assert set(np.argsort(spettro)[-2:]) == {4, seconda}
        assert spettro[seconda] / spettro[4] == pytest.approx(peso, rel=1e-6)
        assert doppia.max() == pytest.approx(1.0, abs=1e-4) and doppia.min() == pytest.approx(-1.0, abs=1e-4)


def test_parametri_per_ogni_forma():
    assert set(bersagli.PARAMETRI_LFO) == set(FORME_LFO)
    assert len(FORME_NUOVE) >= 4
    for forma, descrizioni in bersagli.PARAMETRI_LFO.items():
        assert descrizioni, forma
        nomi = [p.nome for p in descrizioni]
        assert len(set(nomi)) == len(nomi)
        for parametro in descrizioni:
            assert parametro.tipo in ("reale", "intero", "scelta")
            basso, alto = parametro.limiti
            assert all(basso <= x <= alto for x in parametro.intervallo), (forma, parametro.nome)
            assert basso <= parametro.di_serie <= alto
            if parametro.tipo != "scelta":
                assert parametro.intervallo[0] < parametro.intervallo[1], (forma, parametro.nome)


@pytest.mark.parametrize("forma", FORME_LFO)
def test_parametri_a_caso_negli_intervalli(forma):
    rng = random.Random(f"intervalli di {forma}")
    estratti = [bersagli.parametri_a_caso(forma, rng) for _ in range(300)]
    for parametro in bersagli.PARAMETRI_LFO[forma]:
        valori = [e[parametro.nome] for e in estratti]
        if parametro.tipo == "reale":
            basso, alto = parametro.intervallo
            assert all(isinstance(v, float) and basso <= v <= alto for v in valori)
            assert max(valori) - min(valori) > 0.9 * (alto - basso)
        elif parametro.tipo == "intero":
            assert set(valori) == set(range(parametro.intervallo[0], parametro.intervallo[1] + 1))
            assert all(type(v) is int for v in valori)
        else:
            assert set(valori) == set(parametro.intervallo)
    # Lo stesso seme, gli stessi parametri.
    rng = random.Random(f"intervalli di {forma}")
    assert estratti == [bersagli.parametri_a_caso(forma, rng) for _ in range(300)]


def test_parametri_a_caso_nell_ordine_della_tabella():
    for forma, attesi in (("quadra", lambda r: {"larghezza": r.uniform(*LARGHEZZA_DELLA_QUADRA), "fronte": r.uniform(*FRONTE_DELLA_QUADRA)}),
                          ("casuale", lambda r: {"valori": r.randint(*VALORI_DEL_CASUALE), "spinta": r.uniform(*SPINTA_DEL_CASO)}),
                          ("doppia", lambda r: {"rapporto": r.choice(RAPPORTI_DELLA_DOPPIA), "peso": r.uniform(*PESO_DELLA_DOPPIA)})):
        uno, altro = random.Random(9), random.Random(9)
        assert bersagli.parametri_a_caso(forma, uno) == attesi(altro)
        assert uno.random() == altro.random()


def test_parametri_completi_e_i_limiti():
    assert bersagli.parametri_completi("quadra", {"fronte": 0.1}) == {"larghezza": 0.5, "fronte": 0.1}
    assert bersagli.parametri_completi("gradini", {"gradini": 4.0, "verso": -1.0}) == {"gradini": 4, "verso": -1}
    assert bersagli.parametri_completi("seno") == bersagli.parametri_di_serie("seno") == {"armonica": 0.0}
    for forma, sbagliati, motivo in (("seno", {"pendenza": 0.5}, "non ha i parametri"), ("triangolo", {"pendenza": 1.0}, "fuori"),
                                     ("casuale", {"valori": 1.5}, "intero"), ("doppia", {"peso": "molto"}, "numero"),
                                     ("deriva", {"morbidezza": math.nan}, "fuori")):
        with pytest.raises(ValueError, match=motivo):
            bersagli.parametri_completi(forma, sbagliati)
        with pytest.raises(ValueError, match=motivo):
            bersagli.lfo(forma, 0.0, 3.0, parametri=sbagliati)
        with pytest.raises(ValueError, match=motivo):
            bersagli.Bersaglio(numero=1, forma=forma, raggio=1.0, partenza=PUNTO, parametri_lfo=sbagliati)
    bersaglio = bersagli.Bersaglio(numero=1, forma="trillo", raggio=1.0, partenza=PUNTO, frequenza_lfo=3.0, parametri_lfo={"note": 3})
    assert bersaglio.parametri_lfo == {"note": 3, "profondita": bersagli.parametri_di_serie("trillo")["profondita"]}
    tempi = _tempi(0.5)
    assert np.array_equal(bersaglio.lfo(tempi), bersagli.lfo("trillo", tempi, 3.0, parametri={"note": 3}))


# Il moto, coi rimbalzi sulle pareti della cupola (D28).

def test_bersaglio_fermo():
    bersaglio = bersagli.Bersaglio(numero=1, forma="seno", raggio=2.0, partenza=(3.0, 20.0, 50.0), nato=10.0)
    assert not bersaglio.mobile
    assert bersaglio.velocita_scalare == 0.0
    assert bersaglio.tempo_di_uscita is None
    # La sua zona nasce con lui, col suo raggio; un fermo non ha percorso.
    assert isinstance(bersaglio.zona, fisica.Zona) and bersaglio.zona.raggio == 2.0 and bersaglio.zona is bersaglio.zona
    assert bersaglio._percorso is None
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
    """D28: il mobile va dritto e rimbalza sulle pareti della sua zona, col suo fisica.Percorso: dentro in ogni
    istante, continuo e a velocità costante, come rimbalzava sulle pareti del cubo."""
    raggio = 1.5
    bersaglio = bersagli.Bersaglio(numero=2, forma="sega", raggio=raggio, partenza=(0.0, 25.0, 55.0), velocita=(9.0, -4.0, 6.0), nato=5.0)
    assert bersaglio.mobile
    rapidita = math.sqrt(81 + 16 + 36)
    assert bersaglio.velocita_scalare == pytest.approx(rapidita)
    assert bersaglio.tempo_di_uscita is None
    assert isinstance(bersaglio._percorso, fisica.Percorso) and bersaglio._percorso.zona is bersaglio.zona
    assert bersaglio.zona.raggio == raggio
    passo = 0.01
    tempi = np.arange(5.0, 200.0, passo)
    posizioni = bersaglio.posizione(tempi)
    assert posizioni.shape == (len(tempi), 3)
    assert np.all(bersaglio.zona.scarti(posizioni) <= 1e-6)
    assert np.allclose(posizioni[0], (0.0, 25.0, 55.0))
    assert bersaglio._percorso.rimbalzi > 5
    passi = np.linalg.norm(np.diff(posizioni, axis=0), axis=1)
    assert passi.max() <= rapidita * passo * (1 + 1e-9)
    assert np.mean(np.isclose(passi, rapidita * passo, rtol=1e-6)) > 0.99
    # Prima di un rimbalzo il moto è rettilineo, e il tempo del percorso si conta dalla nascita.
    assert np.allclose(bersaglio.posizione(5.5), (4.5, 23.0, 58.0))
    assert bersaglio.posizione(123.4) == pytest.approx(bersaglio._percorso.posizione(123.4 - 5.0))
    assert bersaglio.posizione(77.0).shape == (3,)


def test_mobile_nato_fuori_dalla_zona_e_bonus_vanno_dritti():
    """Un mobile che nasce fuori dalla sua zona va dritto, come il moto libero di fisica.Percorso, e il bonus
    va dritto sempre, senza percorso, perché entra ed esce dal settore."""
    fuori = bersagli.Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=(-60.0, 20.0, 40.0), velocita=(6.0, 0.0, 0.0))
    tempi = np.linspace(0.0, 30.0, 61)
    assert fuori.posizione(tempi) == pytest.approx(np.array((-60.0, 20.0, 40.0)) + np.outer(tempi, (6.0, 0.0, 0.0)))
    bonus = bersagli.Bersaglio(numero=2, forma="seno", raggio=1.0, partenza=(0.0, 20.0, 50.0), velocita=(0.0, 0.0, 9.0), bonus=True, nato=3.0)
    assert bonus._percorso is None
    assert bonus.posizione(13.0) == pytest.approx((0.0, 20.0, 140.0))


# La generazione a caso.

def test_genera_bersagli_nella_zona():
    rng = random.Random(11)
    for livello in (1, LIVELLO_DEI_MOBILI - 1, LIVELLO_DEI_MOBILI, 12, 30):
        for numero in range(40):
            bersaglio = bersagli.genera(livello, rng, numero, 7.5)
            raggio = bersagli.raggio_del_livello(livello)
            assert bersaglio.numero == numero
            assert bersaglio.nato == 7.5
            assert not bersaglio.bonus
            assert bersaglio.raggio == pytest.approx(raggio)
            assert bersaglio.forma in FORME_LFO
            assert FREQUENZA_LFO[0] <= bersaglio.frequenza_lfo <= FREQUENZA_LFO[1]
            assert 0.0 <= bersaglio.fase_lfo < 1.0
            # D28: la partenza sta nella zona del suo raggio, cioè col bersaglio tutto dentro la cupola.
            assert bersaglio.zona.raggio == pytest.approx(raggio)
            assert bersaglio.zona.contiene(bersaglio.partenza, tolleranza=0.0)
            assert bersaglio.velocita_scalare == pytest.approx(bersagli.velocita_del_livello(livello))
            assert bersaglio.mobile == (livello >= LIVELLO_DEI_MOBILI)
            assert (bersaglio._percorso is not None) == bersaglio.mobile
            assert bersaglio.portante in BERSAGLIO_PORTANTI
            assert ESCURSIONE_A_CASO[0] <= bersaglio.escursione <= ESCURSIONE_A_CASO[1]
            assert set(bersaglio.parametri_lfo) == {p.nome for p in bersagli.PARAMETRI_LFO[bersaglio.forma]}


def test_genera_deterministico_e_vario():
    def impronta(b):
        return (b.forma, b.frequenza_lfo, b.fase_lfo, b.seme, b.parametri_lfo, b.escursione, b.portante, tuple(b.partenza), tuple(b.velocita))

    primi = [bersagli.genera(5, random.Random(21), n, 0.0) for n in range(3)]
    secondi = [bersagli.genera(5, random.Random(21), n, 0.0) for n in range(3)]
    for a, b in zip(primi, secondi, strict=True):
        assert impronta(a) == impronta(b)
    assert impronta(bersagli.genera_bonus(5, random.Random(21), 1, 0.0)) == impronta(bersagli.genera_bonus(5, random.Random(21), 1, 0.0))
    assert impronta(primi[0]) != impronta(bersagli.genera(5, random.Random(22), 0, 0.0))
    rng = random.Random(4)
    generati = [bersagli.genera(1, rng, n, 0.0) for n in range(600)]
    assert {b.forma for b in generati} == set(FORME_LFO)
    assert {b.portante for b in generati} == set(BERSAGLIO_PORTANTI)
    # Due bersagli della stessa forma non suonano mai uguali: i parametri cambiano.
    for forma in FORME_LFO:
        stessa = [b.parametri_lfo for b in generati if b.forma == forma]
        assert len({tuple(sorted(p.items())) for p in stessa}) > 1, forma
    direzioni = np.array([bersagli.genera(LIVELLO_DEI_MOBILI, rng, n, 0.0).velocita for n in range(2000)])
    assert np.allclose(direzioni.mean(axis=0), 0.0, atol=0.15)


def _voce_attesa(rng, portanti):
    """Le estrazioni del suono di un bersaglio, rifatte a mano nell'ordine che il docstring di genera promette."""
    forma = rng.choice(FORME_LFO)
    frequenza = rng.uniform(*FREQUENZA_LFO)
    fase = rng.random()
    seme = rng.getrandbits(32)
    parametri = bersagli.parametri_a_caso(forma, rng)
    escursione = rng.uniform(*ESCURSIONE_A_CASO)
    portante = rng.choice(portanti)
    return forma, frequenza, fase, seme, parametri, escursione, portante


def _voce(bersaglio):
    return (bersaglio.forma, bersaglio.frequenza_lfo, bersaglio.fase_lfo, bersaglio.seme, bersaglio.parametri_lfo, bersaglio.escursione, bersaglio.portante)


def _partenza_attesa(rng, raggio):
    """La partenza di genera rifatta a mano nell'ordine di fisica.Zona.punto_a_caso (D28): la distanza, l'azimut
    e il seno dell'elevazione, uniformi, ripetuti finché il punto non sta nella zona del raggio. Restituisce
    anche quante estrazioni sono servite."""
    zona = fisica.Zona(raggio)
    giri = 0
    while True:
        giri += 1
        distanza = rng.uniform(zona.interno, zona.esterno)
        azimut = math.radians(rng.uniform(-SETTORE_AZIMUT, SETTORE_AZIMUT))
        elevazione = math.asin(rng.uniform(0.0, math.sin(math.radians(ELEVAZIONE_MASSIMA))))
        punto = distanza * np.array((math.cos(elevazione) * math.sin(azimut), math.sin(elevazione), math.cos(elevazione) * math.cos(azimut)))
        if zona.contiene(punto, tolleranza=0.0):
            return punto, giri


def _bordo_della_corda(raggio, profondita):
    """Fin dove va, da una parte e dall'altra, la corda del bonus dentro il settore, col centro a un raggio dai due lati."""
    settore = math.radians(SETTORE_AZIMUT)
    return (profondita * math.sin(settore) - raggio) / math.cos(settore)


def test_genera_estrae_nell_ordine_scritto():
    portanti = ("seno", "triangolo")
    for livello in (1, LIVELLO_DEI_MOBILI + 2):
        usato, rifatto = random.Random(77), random.Random(77)
        bersaglio = bersagli.genera(livello, usato, 3, 1.5, portanti=portanti)
        partenza, _giri = _partenza_attesa(rifatto, bersagli.raggio_del_livello(livello))
        direzione = np.array([rifatto.gauss(0.0, 1.0) for _ in range(3)])
        voce = _voce_attesa(rifatto, portanti)
        assert bersaglio.partenza == pytest.approx(partenza)
        assert np.allclose(bersaglio.velocita, direzione / np.linalg.norm(direzione) * bersagli.velocita_del_livello(livello))
        assert _voce(bersaglio) == voce
        # E nient'altro: dopo, i due generatori sono allo stesso punto.
        assert usato.random() == rifatto.random()
    # Una partenza che chiede più di un'estrazione, perché la prima cade fuori dalla zona.
    for seme in itertools.count(100):
        if _partenza_attesa(random.Random(seme), RAGGIO_INIZIALE)[1] > 1:
            break
    usato, rifatto = random.Random(seme), random.Random(seme)
    assert bersagli.genera(1, usato, 1, 0.0).partenza == pytest.approx(_partenza_attesa(rifatto, RAGGIO_INIZIALE)[0])
    # Il bonus: il lato, poi l'altezza e la profondità, ripetute finché la corda non sta nella zona, poi il suono.
    for seme in (78, 79, 80, 81):
        usato, rifatto = random.Random(seme), random.Random(seme)
        bonus = bersagli.genera_bonus(LIVELLO_DEI_BONUS, usato, 4, 2.0, portanti=portanti)
        raggio = bersagli.raggio_del_livello(LIVELLO_DEI_BONUS) * RAGGIO_BONUS_FATTORE
        zona = fisica.Zona(raggio)
        da_sinistra = rifatto.random() < 0.5
        while True:
            altezza, profondita = rifatto.uniform(zona.pavimento, zona.esterno), rifatto.uniform(0.0, zona.esterno)
            bordo = _bordo_della_corda(raggio, profondita)
            if bordo > 0.0 and zona.contiene((0.0, altezza, profondita), tolleranza=0.0) and zona.contiene((bordo, altezza, profondita), tolleranza=0.0):
                break
        assert (bool(bonus.velocita[0] > 0.0), bonus.partenza[1], bonus.partenza[2]) == (da_sinistra, altezza, profondita)
        assert _voce(bonus) == _voce_attesa(rifatto, portanti)
        assert usato.random() == rifatto.random()


def test_genera_con_le_portanti():
    rng = random.Random(31)
    solo_seno = [bersagli.genera(2, rng, n, 0.0, portanti=["seno"]) for n in range(30)]
    assert {b.portante for b in solo_seno} == {"seno"}
    senza = bersagli.portanti_senza("seno")
    assert senza == tuple(p for p in BERSAGLIO_PORTANTI if p != "seno")
    assert bersagli.portanti_senza("violino") == BERSAGLIO_PORTANTI
    scelte = {bersagli.genera(2, rng, n, 0.0, portanti=senza).portante for n in range(200)}
    scelte_del_bonus = {bersagli.genera_bonus(LIVELLO_DEI_BONUS, rng, n, 0.0, portanti=senza).portante for n in range(200)}
    assert scelte == scelte_del_bonus == set(senza)
    # Un elenco vuoto, o una portante che la sintesi non conosce, non toccano rng.
    for sbagliate, motivo in (((), "nessuna portante"), ([], "nessuna portante"), (("seno", "violino"), "sconosciuta")):
        stato = rng.getstate()
        with pytest.raises(ValueError, match=motivo):
            bersagli.genera(2, rng, 1, 0.0, portanti=sbagliate)
        with pytest.raises(ValueError, match=motivo):
            bersagli.genera_bonus(LIVELLO_DEI_BONUS, rng, 1, 0.0, portanti=sbagliate)
        assert rng.getstate() == stato
    with pytest.raises(ValueError, match="sconosciuta"):
        bersagli.Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=PUNTO, portante="violino")
    assert set(BERSAGLIO_PORTANTI) <= set(FORME_PORTANTI)
    assert BERSAGLIO_FORMA in BERSAGLIO_PORTANTI


def test_genera_bonus_attraversa_il_settore():
    """D28: il bonus nasce col centro a un raggio fuori dal lato di sinistra o di destra, già tangente al lato da
    fuori, attraversa il settore lungo x a quota e profondità costanti, su una corda che dentro il settore sta
    tutta nella zona del suo raggio, ed esce del tutto dall'altro lato in durata_del_bonus secondi."""
    rng = random.Random(8)
    lati = set()
    settore = math.radians(SETTORE_AZIMUT)
    for numero in range(60):
        livello = LIVELLO_DEI_BONUS + numero % 20
        bonus = bersagli.genera_bonus(livello, rng, numero, 30.0)
        raggio = bersagli.raggio_del_livello(livello) * RAGGIO_BONUS_FATTORE
        durata = bersagli.durata_del_bonus(livello)
        zona = bonus.zona
        assert bonus.bonus
        assert bonus.mobile
        assert bonus._percorso is None
        assert bonus.raggio == pytest.approx(raggio) and zona.raggio == pytest.approx(raggio)
        assert bonus.velocita[1] == 0.0
        assert bonus.velocita[2] == 0.0
        altezza, profondita = bonus.partenza[1], bonus.partenza[2]
        assert zona.pavimento <= altezza <= zona.esterno and 0.0 <= profondita <= zona.esterno
        # La corda intera va da un raggio fuori da un lato a un raggio fuori dall'altro, in tanti secondi.
        fuori = (profondita * math.sin(settore) + raggio) / math.cos(settore)
        assert abs(bonus.velocita[0]) == pytest.approx(2.0 * fuori / durata)
        assert bonus.tempo_di_uscita == pytest.approx(30.0 + durata)
        assert bonus.portante in BERSAGLIO_PORTANTI
        assert ESCURSIONE_A_CASO[0] <= bonus.escursione <= ESCURSIONE_A_CASO[1]
        # Nasce tangente da fuori al lato da cui parte, ed esce del tutto dall'altro.
        da_sinistra = bonus.velocita[0] > 0.0
        partenza_dal_lato, arrivo_dal_lato = (zona.normale_sinistra, zona.normale_destra) if da_sinistra else (zona.normale_destra, zona.normale_sinistra)
        inizio = bonus.posizione(30.0)
        fine = bonus.posizione(bonus.tempo_di_uscita)
        assert inizio @ partenza_dal_lato == pytest.approx(raggio)
        assert fine @ arrivo_dal_lato == pytest.approx(raggio)
        assert fine[0] == pytest.approx(-inizio[0])
        assert np.allclose(fine[1:], inizio[1:])
        lati.add(math.copysign(1.0, bonus.velocita[0]))
        # Dentro il settore, col centro a un raggio dai due lati, la corda sta tutta nella zona, e non è vuota.
        tempi = np.linspace(30.0, bonus.tempo_di_uscita, 801)
        posizioni = bonus.posizione(tempi)
        nel_settore = (posizioni @ zona.normale_destra <= -raggio) & (posizioni @ zona.normale_sinistra <= -raggio)
        assert nel_settore.sum() > 10
        assert zona.contiene(posizioni[nel_settore]).all()
        # Vivo dal suo nato all'uscita, non prima e non dopo.
        assert bonus.vivo(30.0)
        assert bonus.vivo(30.0 + durata / 2)
        assert bonus.vivo(bonus.tempo_di_uscita)
        assert not bonus.vivo(29.9)
        assert not bonus.vivo(bonus.tempo_di_uscita + 0.01)
        mezzo = bonus.posizione(30.0 + durata / 2)
        assert mezzo[0] == pytest.approx(0.0, abs=1e-9)
    assert lati == {-1.0, 1.0}


def test_tempo_di_uscita_dal_lato_verso_cui_va():
    """D28: il bonus esce del tutto dal settore quando il suo centro sta a un raggio oltre il lato verso cui va,
    da qualunque punto parta e con qualunque velocità che vada verso un lato; un bonus che non va verso nessun
    lato non esce mai, e gli altri bersagli non hanno un tempo di uscita."""
    settore = math.radians(SETTORE_AZIMUT)
    raggio = 1.0
    for partenza, velocita, nato in (((0.0, 20.0, 50.0), (-5.0, 0.0, 0.0), 0.0), ((-10.0, 30.0, 40.0), (4.0, 0.0, 0.0), 12.0),
                                     ((5.0, 25.0, 45.0), (3.0, 1.0, 2.0), 7.0), ((-3.0, 15.0, 60.0), (-2.0, -0.5, 1.0), 1.0)):
        bonus = bersagli.Bersaglio(numero=1, forma="seno", raggio=raggio, partenza=partenza, velocita=velocita, nato=nato, bonus=True)
        lato = bonus.zona.normale_destra if velocita[0] > 0.0 else bonus.zona.normale_sinistra
        uscita = bonus.tempo_di_uscita
        assert uscita > nato
        assert bonus.posizione(uscita) @ lato == pytest.approx(raggio)
        assert bonus.posizione(uscita - 0.01) @ lato < raggio
        assert bonus.vivo(uscita) and not bonus.vivo(uscita + 0.01)
    # Già fuori del tutto, e diretto ancora più in là: esce subito, quando nasce.
    oltre = (100.0 * math.sin(settore + 0.3), 20.0, 100.0 * math.cos(settore + 0.3))
    assert bersagli.Bersaglio(numero=2, forma="seno", raggio=raggio, partenza=oltre, velocita=(5.0, 0.0, 0.0), nato=4.0, bonus=True).tempo_di_uscita == 4.0
    # Dritto in avanti non va verso nessun lato.
    avanti = bersagli.Bersaglio(numero=3, forma="seno", raggio=raggio, partenza=(0.0, 20.0, 40.0), velocita=(0.0, 0.0, 6.0), bonus=True)
    assert avanti.tempo_di_uscita == math.inf and avanti.vivo(1e9)
    normale = bersagli.Bersaglio(numero=4, forma="seno", raggio=raggio, partenza=(0.0, 20.0, 40.0), velocita=(5.0, 0.0, 0.0))
    assert normale.tempo_di_uscita is None


def test_bonus_fermo_non_esce_mai():
    bonus = bersagli.Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=(0.0, 10.0, 50.0), bonus=True)
    assert bonus.tempo_di_uscita == math.inf
    assert bonus.vivo(1e9)


def test_copertura_della_traversata_del_bonus(monkeypatch):
    """D28: ogni punto della cupola resta raggiungibile, anche quelli della corda del bonus. Per bonus di ogni
    livello, ogni centro della traversata dentro il settore, dove la corda sta tutta nella zona del suo raggio,
    si raggiunge con i tre valori dentro le scale, e il tiro trovato ci passa. Come nella prova della cupola di
    test_fisica, qui senza MARGINE_DI_COPERTURA."""
    monkeypatch.setattr(fisica, "MARGINE_DI_COPERTURA", 0.0)
    rng = random.Random(13)
    scoperti = []
    for numero in range(40):
        livello = LIVELLO_DEI_BONUS + numero % 20
        bonus = bersagli.genera_bonus(livello, rng, numero, 0.0)
        bordo = _bordo_della_corda(bonus.raggio, bonus.partenza[2])
        for x in np.linspace(-bordo, bordo, 11):
            centro = np.array((x, bonus.partenza[1], bonus.partenza[2]))
            valori = fisica.soluzione(centro)
            if valori is None:
                scoperti.append((livello, tuple(round(float(c), 2) for c in centro)))
                continue
            traiettoria = fisica.Traiettoria(*fisica.parametri_del_tiro(*valori))
            tempo = math.hypot(centro[0], centro[2]) / math.hypot(traiettoria.iniziale[0], traiettoria.iniziale[2])
            assert np.linalg.norm(traiettoria.posizione(tempo) - centro) < 0.01, (livello, centro, valori)
    assert not scoperti, f"{len(scoperti)} centri del bonus senza soluzione, il primo {scoperti[0]}"


# L'escursione e i parametri sonori.

def test_escursione_a_caso_e_mezza_escursione():
    rng = random.Random(41)
    generati = [bersagli.genera(3, rng, n, 0.0) for n in range(300)]
    escursioni = [b.escursione for b in generati]
    assert all(ESCURSIONE_A_CASO[0] <= e <= ESCURSIONE_A_CASO[1] for e in escursioni)
    assert max(escursioni) - min(escursioni) > 0.9 * (ESCURSIONE_A_CASO[1] - ESCURSIONE_A_CASO[0])
    for b in generati[:20]:
        assert b.mezza_escursione_ottave() == pytest.approx(b.raggio * FATTORE_ESCURSIONE * b.escursione / METRI_PER_OTTAVA)
    # Lo stesso raggio, l'escursione più ampia: l'oscillazione in ottave cresce in proporzione.
    tempi_audio = _tempi(1.0)
    ampiezze = []
    for escursione in (ESCURSIONE_A_CASO[0], 1.0, ESCURSIONE_A_CASO[1]):
        bersaglio = bersagli.Bersaglio(numero=1, forma="seno", raggio=3.0, partenza=PUNTO, frequenza_lfo=3.0, escursione=escursione)
        frequenze, _pan, _guadagni = bersaglio.parametri_sonori(0.0, tempi_audio)
        ampiezze.append(math.log2(frequenze.max() / frequenze.min()) / 2.0)
        assert ampiezze[-1] == pytest.approx(3.0 * FATTORE_ESCURSIONE * escursione / METRI_PER_OTTAVA, rel=1e-4)
    assert ampiezze == sorted(ampiezze)
    for sbagliata in (-0.5, math.inf, math.nan):
        with pytest.raises(ValueError, match="escursione"):
            bersagli.Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=PUNTO, escursione=sbagliata)


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


def test_parametri_sonori_coi_parametri_della_forma():
    """Il suono usa i parametri della forma del bersaglio: due bersagli che differiscono solo per quelli suonano diversi."""
    tempi_audio = _tempi(0.5)
    liscio = bersagli.Bersaglio(numero=1, forma="quadra", raggio=2.0, partenza=PUNTO, frequenza_lfo=3.0)
    smussato = bersagli.Bersaglio(numero=2, forma="quadra", raggio=2.0, partenza=PUNTO, frequenza_lfo=3.0, parametri_lfo={"fronte": 0.2})
    frequenze_lisce = liscio.parametri_sonori(0.0, tempi_audio)[0]
    frequenze_smussate = smussato.parametri_sonori(0.0, tempi_audio)[0]
    assert not np.allclose(frequenze_lisce, frequenze_smussate)
    assert frequenze_lisce.max() == pytest.approx(frequenze_smussate.max()) and frequenze_lisce.min() == pytest.approx(frequenze_smussate.min())


def test_parametri_sonori_col_tempo_di_gioco_fermo():
    bersaglio = bersagli.Bersaglio(numero=1, forma="seno", raggio=2.0, partenza=(0.0, 20.0, 50.0), velocita=(5.0, 0.0, 0.0), frequenza_lfo=4.0)
    tempi_audio = _tempi(0.5)
    frequenze, pan, guadagni = bersaglio.parametri_sonori(7.0, tempi_audio)
    assert frequenze.shape == pan.shape == guadagni.shape == tempi_audio.shape
    assert np.all(pan == pan[0])
    assert np.all(guadagni == guadagni[0])
    assert frequenze.max() > frequenze.min()
