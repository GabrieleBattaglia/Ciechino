"""
Le prove di sintesi.py: le portanti, la panoramica e i bip, contati al campione, e i suoni della mira manuale,
il puntatore che porta i tre valori, le tacche e il tick del tempo, misurati sull'onda.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0; dalla 2.0.0 prova i suoni della mira manuale di D29, e non più il buffer
delle scivolate, coi suoi bip e le sue tacche; dalla 2.0.1 il puntatore di D31, parametri_del_puntatore e
pan_della_direzione, al posto della voce di ogni valore, con tacca_mono, tacca_dell_asse e tick_del_tempo.
"""

import numpy as np
import pytest

import sintesi
from costanti import (
    ALZO_FREQUENZE,
    DURATA_DELLA_SCALA,
    FS,
    PARAMETRI,
    PARTENZA_DELLA_MIRA,
    POTENZA_DB_MINIMO,
    PUNTATORE_DB,
    PUNTATORE_FORMA,
    TACCA_DB,
    TACCA_DURATA,
    TACCA_FORMA,
    TACCA_FREQUENZE,
    TACCHE,
    TICK_DB,
    TICK_DURATA,
    TICK_FORMA,
    TICK_FREQUENZE,
)
from mira import Asse

# I secondi delle mire a tempo che si provano: quella del primo livello dopo la palestra, la più corta, una di
# mezzo e una che non è un numero intero di secondi.
SECONDI_DELLE_MIRE = (30.0, 10.0, 17.0, 12.5)


def _campioni(secondi):
    return round(secondi * FS)


def _frequenza_dagli_zeri(onda, fs=FS):
    """Gli istanti dei passaggi per lo zero in salita, interpolati, e la frequenza fra due di fila."""
    sotto = onda[:-1] < 0.0
    sopra = onda[1:] >= 0.0
    indici = np.flatnonzero(sotto & sopra)
    frazioni = -onda[indici] / (onda[indici + 1] - onda[indici])
    istanti = indici + frazioni
    return (istanti[:-1] + istanti[1:]) / 2.0, fs / np.diff(istanti)


def _frequenza_media(onda, fs=FS):
    """La frequenza di un suono fermo, dai passaggi per lo zero in salita."""
    _istanti, frequenze = _frequenza_dagli_zeri(np.asarray(onda, dtype=np.float64), fs)
    return float(np.median(frequenze))


def _pan_stimato(stereo):
    """La panoramica di un blocco stereo, da meno uno a uno, rifatta dall'energia dei due lati con la legge di
    sintesi.panoramica."""
    dati = np.asarray(stereo, dtype=np.float64)
    sinistra = np.sqrt(np.mean(dati[:, 0] ** 2))
    destra = np.sqrt(np.mean(dati[:, 1] ** 2))
    return float(np.arctan2(destra, sinistra) * 4.0 / np.pi - 1.0)


def _mono(stereo):
    """Il mono di un blocco stereo con la panoramica ferma: con la legge a potenza costante, la radice della somma dei
    quadrati dei due lati, col segno del lato più forte."""
    dati = np.asarray(stereo, dtype=np.float64)
    forte = dati[:, 0] if np.sum(dati[:, 0] ** 2) >= np.sum(dati[:, 1] ** 2) else dati[:, 1]
    return np.sign(forte) * np.sqrt(np.sum(dati ** 2, axis=1))


def _terza_armonica(onda, frequenza, fs=FS):
    """Il rapporto fra l'ampiezza della terza armonica e quella della fondamentale, ai minimi quadrati, con le prime
    cinque armoniche nel modello: zero per il seno, un nono per il triangolo."""
    tempi = np.arange(len(onda)) / fs
    colonne = []
    for numero in range(1, 6):
        colonne.append(np.sin(2.0 * np.pi * numero * frequenza * tempi))
        colonne.append(np.cos(2.0 * np.pi * numero * frequenza * tempi))
    coefficienti = np.linalg.lstsq(np.stack(colonne, axis=1), onda, rcond=None)[0]
    ampiezze = np.hypot(coefficienti[0::2], coefficienti[1::2])
    return float(ampiezze[2] / ampiezze[0])


def _senza_rampe(onda):
    """L'onda di un bip senza le due rampe di un millesimo, dove il livello è pieno."""
    rampa = round(sintesi.RAMPA_DEL_BIP * FS)
    return onda[rampa:len(onda) - rampa]


def test_forme_e_picchi():
    assert sintesi.FORME_PORTANTI == ("seno", "triangolo", "sega_morbida", "organo")
    for forma in sintesi.FORME_PORTANTI:
        for frequenza in (55.0, 440.0, 3000.0, 9000.0, 17500.0):
            onda = sintesi.Oscillatore(forma).rendi(frequenza, 8192)
            assert np.max(np.abs(onda)) <= 1.0
            assert np.max(np.abs(onda)) > 0.0
    with pytest.raises(ValueError):
        sintesi.Oscillatore("piffero")


def test_fase_continua_fra_i_blocchi():
    """Un blocco intero e lo stesso blocco a pezzi danno gli stessi campioni."""
    frequenze = np.geomspace(200.0, 2000.0, 6000)
    intero = sintesi.Oscillatore("sega_morbida").rendi(frequenze)
    oscillatore = sintesi.Oscillatore("sega_morbida")
    pezzi = np.concatenate([oscillatore.rendi(frequenze[a:a + 512]) for a in range(0, 6000, 512)])
    assert np.allclose(intero, pezzi, atol=1e-9)


def test_armoniche_tolte_sopra_il_limite():
    """Una sega a 7.000 hertz: la seconda armonica, 14.000, resta; la terza, 21.000, sparisce."""
    onda = sintesi.Oscillatore("sega_morbida").rendi(7000.0, FS)
    spettro = np.abs(np.fft.rfft(onda)) / len(onda)
    frequenze = np.fft.rfftfreq(len(onda), 1.0 / FS)
    def ampiezza(hertz):
        return spettro[np.argmin(np.abs(frequenze - hertz))]
    assert ampiezza(7000.0) > 0.1
    assert ampiezza(14000.0) > 0.01
    assert ampiezza(21000.0) < 1e-6
    # La quarta e la quinta armonica, 28.000 e 35.000 hertz, si piegherebbero a 16.100 e 9.100.
    assert ampiezza(FS - 28000.0) < 1e-6
    assert ampiezza(FS - 35000.0) < 1e-6
    assert np.max(spettro[frequenze > sintesi.FREQUENZA_LIMITE_ARMONICHE]) < 1e-6


def test_frequenza_nulla_e_silenzio():
    onda = sintesi.Oscillatore("organo").rendi(np.zeros(100))
    assert np.all(onda == 0.0)


def test_panoramica_a_potenza_costante():
    mono = np.ones(5)
    stereo = sintesi.panoramica(mono, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
    assert stereo.dtype == np.float32
    assert stereo.shape == (5, 2)
    assert stereo[0, 0] == pytest.approx(1.0) and stereo[0, 1] == pytest.approx(0.0, abs=1e-7)
    assert stereo[4, 1] == pytest.approx(1.0) and stereo[4, 0] == pytest.approx(0.0, abs=1e-7)
    assert stereo[2, 0] == pytest.approx(np.sqrt(0.5)) and stereo[2, 1] == pytest.approx(np.sqrt(0.5))
    assert np.allclose(np.sum(stereo.astype(np.float64) ** 2, axis=1), 1.0, atol=1e-6)
    assert np.allclose(sintesi.panoramica(mono, 0.0), np.sqrt(0.5))


def test_decibel():
    assert sintesi.db_in_lineare(0.0) == pytest.approx(1.0)
    assert sintesi.db_in_lineare(-20.0) == pytest.approx(0.1)
    assert sintesi.lineare_in_db(0.5) == pytest.approx(-6.0206, abs=1e-3)
    assert sintesi.lineare_in_db(0.0) == pytest.approx(sintesi.PAVIMENTO_IN_DB)
    assert np.allclose(sintesi.lineare_in_db(sintesi.db_in_lineare(np.array([-40.0, -3.0, 0.0]))), [-40.0, -3.0, 0.0])


def test_bip_lunghezza_livello_e_rampe():
    onda = sintesi.bip(1000.0, 0.012, -12.0)
    assert onda.dtype == np.float32
    assert len(onda) == _campioni(0.012)
    assert abs(onda[0]) < 1e-3 and abs(onda[-1]) < 1e-2
    assert np.max(np.abs(onda)) == pytest.approx(sintesi.db_in_lineare(-12.0), rel=0.02)


def _puntatore(**valori):
    """parametri_del_puntatore coi valori dati per nome, e gli altri al centro, da dove parte la mira."""
    tre = dict.fromkeys(PARAMETRI, PARTENZA_DELLA_MIRA) | valori
    return sintesi.parametri_del_puntatore(*(tre[nome] for nome in PARAMETRI))


def test_la_direzione_nella_panoramica_del_puntatore():
    """D31: la direzione è la panoramica del puntatore, da tutto a sinistra a tutto a destra, e non cambia né la nota
    né il volume, che restano quelli dell'alzo e della potenza."""
    valori = np.linspace(0.0, 1.0, 11)
    for alzo, potenza in ((0.5, 0.5), (0.0, 1.0), (0.8, 0.2)):
        frequenze, pan, guadagni = _puntatore(direzione=valori, alzo=alzo, potenza=potenza)
        assert frequenze.shape == pan.shape == guadagni.shape == valori.shape
        np.testing.assert_allclose(pan, 2.0 * valori - 1.0, rtol=0, atol=1e-12)
        assert pan[0] == -1.0 and pan[5] == pytest.approx(0.0, abs=1e-12) and pan[-1] == 1.0
        np.testing.assert_allclose(frequenze, _puntatore(alzo=alzo)[0][0])
        np.testing.assert_allclose(guadagni, _puntatore(potenza=potenza)[2][0])
    np.testing.assert_allclose(sintesi.pan_della_direzione(valori), 2.0 * valori - 1.0, rtol=0, atol=1e-12)
    assert sintesi.pan_della_direzione(0.25) == -0.5 and isinstance(sintesi.pan_della_direzione(0.25), float)
    assert sintesi.pan_della_direzione(-1.0) == -1.0 and sintesi.pan_della_direzione(2.0) == 1.0


def test_l_alzo_nell_altezza_del_puntatore():
    """L'alzo è l'altezza del puntatore, da ALZO_FREQUENZE[0] a ALZO_FREQUENZE[1] su scala esponenziale: a passi uguali
    del valore intervalli uguali, a metà scala la media geometrica dei due capi; panoramica e volume restano quelli
    della direzione e della potenza."""
    basso, alto = ALZO_FREQUENZE
    valori = np.linspace(0.0, 1.0, 21)
    for direzione, potenza in ((0.5, 0.5), (1.0, 0.0), (0.1, 0.9)):
        frequenze, pan, guadagni = _puntatore(direzione=direzione, alzo=valori, potenza=potenza)
        assert frequenze[0] == pytest.approx(basso) and frequenze[-1] == pytest.approx(alto)
        assert frequenze[10] == pytest.approx(np.sqrt(basso * alto))
        np.testing.assert_allclose(frequenze, basso * (alto / basso) ** valori)
        rapporti = frequenze[1:] / frequenze[:-1]
        np.testing.assert_allclose(rapporti, rapporti[0])
        assert np.all(rapporti > 1.0)
        np.testing.assert_allclose(pan, 2.0 * direzione - 1.0)
        np.testing.assert_allclose(guadagni, _puntatore(potenza=potenza)[2][0])


def test_la_potenza_nel_volume_del_puntatore():
    """La potenza è il volume del puntatore, che cresce lineare in decibel da POTENZA_DB_MINIMO sotto PUNTATORE_DB fino a
    PUNTATORE_DB: 20 decibel, così anche al minimo il puntatore si sente; panoramica e nota restano quelle della
    direzione e dell'alzo, e il volume non le compensa."""
    assert POTENZA_DB_MINIMO == -20.0 and PUNTATORE_DB == -9.0
    valori = np.linspace(0.0, 1.0, 21)
    for direzione, alzo in ((0.5, 0.5), (0.0, 1.0), (0.7, 0.0)):
        frequenze, pan, guadagni = _puntatore(direzione=direzione, alzo=alzo, potenza=valori)
        decibel = sintesi.lineare_in_db(guadagni)
        assert decibel[0] == pytest.approx(PUNTATORE_DB + POTENZA_DB_MINIMO)
        assert decibel[-1] == pytest.approx(PUNTATORE_DB)
        np.testing.assert_allclose(decibel, PUNTATORE_DB + POTENZA_DB_MINIMO * (1.0 - valori), rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.diff(decibel), -POTENZA_DB_MINIMO / 20.0, rtol=1e-9)
        np.testing.assert_allclose(pan, 2.0 * direzione - 1.0)
        np.testing.assert_allclose(frequenze, _puntatore(alzo=alzo)[0][0])


def test_i_valori_del_puntatore_fuori_scala_e_insieme():
    """Un valore oltre i capi vale il capo; tre numeri danno tre array di un elemento; due valori che si muovono
    insieme cambiano insieme lo stesso suono."""
    fuori = sintesi.parametri_del_puntatore([-0.5, 1.5], [1.5, -0.5], [-2.0, 3.0])
    capi = sintesi.parametri_del_puntatore([0.0, 1.0], [1.0, 0.0], [0.0, 1.0])
    for a, b in zip(fuori, capi, strict=True):
        np.testing.assert_array_equal(a, b)
    solo = sintesi.parametri_del_puntatore(0.3, 0.6, 0.9)
    assert all(isinstance(array, np.ndarray) and array.shape == (1,) for array in solo)
    insieme = np.linspace(0.0, 1.0, 5)
    frequenze, pan, guadagni = sintesi.parametri_del_puntatore(insieme, 0.5, insieme)
    assert frequenze.shape == pan.shape == guadagni.shape == insieme.shape
    np.testing.assert_allclose(pan, 2.0 * insieme - 1.0)
    np.testing.assert_allclose(sintesi.lineare_in_db(guadagni), PUNTATORE_DB + POTENZA_DB_MINIMO * (1.0 - insieme), atol=1e-9)
    np.testing.assert_allclose(frequenze, np.sqrt(ALZO_FREQUENZE[0] * ALZO_FREQUENZE[1]))


def _scala(parametro):
    """Il puntatore mentre un valore corre da un capo all'altro, in DURATA_DELLA_SCALA, con gli altri due al centro, come
    lo fa la voce del motore: lo stato di un Asse, i parametri di ogni campione e l'Oscillatore di PUNTATORE_FORMA."""
    asse = Asse(0.0)
    asse.muovi(1, 0.0)
    tempi = np.arange(_campioni(DURATA_DELLA_SCALA)) / FS
    frequenze, pan, guadagni = _puntatore(**{parametro: asse.valore(tempi)})
    mono = sintesi.Oscillatore(PUNTATORE_FORMA).rendi(frequenze) * guadagni
    return sintesi.panoramica(mono, pan).astype(np.float64)


def test_la_direzione_suona_da_sinistra_a_destra():
    scala = _scala("direzione")
    finestra = _campioni(0.1)
    pan = []
    for dove in range(0, len(scala) - finestra + 1, finestra):
        stimato = _pan_stimato(scala[dove:dove + finestra])
        atteso = 2.0 * (dove + finestra / 2.0) / len(scala) - 1.0
        assert stimato == pytest.approx(atteso, abs=0.02)
        pan.append(stimato)
    assert np.all(np.diff(pan) > 0)
    assert pan[0] < -0.9 and pan[-1] > 0.9


def test_la_frequenza_istantanea_dell_alzo():
    scala = _scala("alzo")[:, 0]
    istanti, frequenze = _frequenza_dagli_zeri(scala)
    attese = ALZO_FREQUENZE[0] * (ALZO_FREQUENZE[1] / ALZO_FREQUENZE[0]) ** (istanti / len(scala))
    assert np.allclose(frequenze, attese, rtol=0.01)
    assert frequenze[0] == pytest.approx(ALZO_FREQUENZE[0], rel=0.02)
    assert frequenze[-1] == pytest.approx(ALZO_FREQUENZE[1], rel=0.02)


def test_l_inviluppo_della_potenza_in_decibel():
    """Il puntatore con l'alzo al centro, a 600 hertz: un decimo di secondo sono 60 periodi esatti, e il livello in
    decibel sale in linea retta di POTENZA_DB_MINIMO, cambiato di segno, lungo la scala della potenza."""
    scala = _scala("potenza")
    finestra = _campioni(0.1)
    centri, livelli = [], []
    for dove in range(0, len(scala) - finestra + 1, finestra):
        pezzo = scala[dove:dove + finestra, 0]
        centri.append((dove + finestra / 2.0) / len(scala))
        livelli.append(20.0 * np.log10(np.sqrt(np.mean(pezzo ** 2))))
    centri, livelli = np.array(centri), np.array(livelli)
    relativi = livelli - livelli[-1]
    attesi = -POTENZA_DB_MINIMO * (centri - centri[-1])
    assert np.allclose(relativi, attesi, atol=0.2)
    pendenza = np.polyfit(centri, livelli, 1)[0]
    assert pendenza == pytest.approx(-POTENZA_DB_MINIMO, rel=0.01)


def test_le_frequenze_delle_tacche():
    frequenze = sintesi.frequenze_delle_tacche()
    assert len(frequenze) == TACCHE
    assert frequenze[0] == pytest.approx(TACCA_FREQUENZE[0]) and frequenze[-1] == pytest.approx(TACCA_FREQUENZE[1])
    assert np.all(np.diff(frequenze) > 0)
    rapporti = frequenze[1:] / frequenze[:-1]
    assert np.allclose(rapporti, rapporti[0])
    assert len(sintesi.frequenze_delle_tacche(0)) == 0
    np.testing.assert_allclose(sintesi.frequenze_delle_tacche(1), [TACCA_FREQUENZE[0]])


@pytest.mark.parametrize("parametro", PARAMETRI)
def test_ogni_tacca_ha_la_sua_frequenza_e_la_sua_forma(parametro):
    """La tacca k, da 1 a TACCHE meno 1, è un bip di TACCA_DURATA a TACCA_DB, alla k-esima frequenza di
    frequenze_delle_tacche(TACCHE - 1), nella forma del suo valore: le nove tacche dei decimi coprono tutta la scala, la
    prima a TACCA_FREQUENZE[0] e la nona a TACCA_FREQUENZE[1], 2400 hertz. Misurate sull'onda, la frequenza dai
    passaggi per lo zero, il livello dal picco e la forma dalla terza armonica, che il seno non ha e il triangolo ha a
    un nono."""
    frequenze = sintesi.frequenze_delle_tacche(TACCHE - 1)
    assert len(frequenze) == TACCHE - 1
    assert frequenze[0] == pytest.approx(TACCA_FREQUENZE[0]) and frequenze[-1] == pytest.approx(TACCA_FREQUENZE[1]) == 2400.0
    forma = TACCA_FORMA[parametro]
    misurate = []
    for k in range(1, TACCHE):
        tacca = sintesi.tacca_dell_asse(parametro, k)
        assert tacca.dtype == np.float32 and tacca.shape == (_campioni(TACCA_DURATA), 2), k
        mono = _mono(tacca)
        misurate.append(_frequenza_media(mono))
        assert misurate[-1] == pytest.approx(frequenze[k - 1], rel=0.01), k
        pieno = _senza_rampe(mono)
        # Il triangolo acuto perde qualche armonica, e il suo picco scende un poco: mai sopra TACCA_DB.
        assert 0.95 * sintesi.db_in_lineare(TACCA_DB) < np.max(np.abs(pieno)) <= sintesi.db_in_lineare(TACCA_DB), k
        terza = _terza_armonica(pieno, frequenze[k - 1])
        if forma == "triangolo":
            assert terza == pytest.approx(1.0 / 9.0, rel=0.1), k
        else:
            assert terza < 0.01, k
        # Al campione, è il bip della sua frequenza, nella sua forma, a TACCA_DB, con la panoramica della tacca.
        pan = 2.0 * k / TACCHE - 1.0 if parametro == "direzione" else 0.0
        atteso = sintesi.panoramica(sintesi.bip(frequenze[k - 1], TACCA_DURATA, TACCA_DB, forma), pan)
        np.testing.assert_allclose(tacca, atteso, rtol=0, atol=1e-6)
        mono = sintesi.tacca_mono(parametro, k)
        assert mono.dtype == np.float32 and mono.shape == (_campioni(TACCA_DURATA),), k
        np.testing.assert_allclose(mono, sintesi.bip(frequenze[k - 1], TACCA_DURATA, TACCA_DB, forma), rtol=0, atol=1e-6)
    assert np.all(np.diff(misurate) > 0)


def test_la_panoramica_delle_tacche():
    """Le tacche della direzione stanno nel punto del loro decimo, da sinistra a destra; quelle dell'alzo e della potenza
    al centro, coi due lati uguali."""
    stimati = []
    for k in range(1, TACCHE):
        stimati.append(_pan_stimato(sintesi.tacca_dell_asse("direzione", k)))
        assert stimati[-1] == pytest.approx(2.0 * k / TACCHE - 1.0, abs=1e-3), k
        for parametro in ("alzo", "potenza"):
            tacca = sintesi.tacca_dell_asse(parametro, k)
            np.testing.assert_array_equal(tacca[:, 0], tacca[:, 1])
    assert np.all(np.diff(stimati) > 0)
    assert stimati[0] < -0.7 and stimati[4] == pytest.approx(0.0, abs=1e-3) and stimati[-1] > 0.7


def test_le_tacche_dal_punto_del_puntatore():
    """D31: col pan dato, quello del puntatore nell'istante della tacca, la tacca di ogni valore viene da lì, ed è la
    tacca mono collocata con sintesi.panoramica."""
    for parametro in PARAMETRI:
        for pan in (-1.0, -0.4, 0.0, 0.6, 1.0):
            tacca = sintesi.tacca_dell_asse(parametro, 3, pan=pan)
            assert tacca.dtype == np.float32
            assert _pan_stimato(tacca) == pytest.approx(pan, abs=1e-3), (parametro, pan)
            np.testing.assert_allclose(tacca, sintesi.panoramica(sintesi.tacca_mono(parametro, 3), pan), rtol=0, atol=1e-7)


def test_le_tacche_fuori_dalla_scala():
    """La tacca zero e la decima sono i capi, che hanno il suono fine_scala: tacca_dell_asse non le fa."""
    for k in (0, TACCHE, -1, TACCHE + 1):
        with pytest.raises(ValueError):
            sintesi.tacca_dell_asse("alzo", k)
        with pytest.raises(ValueError):
            sintesi.tacca_mono("potenza", k)
    with pytest.raises(ValueError):
        sintesi.tacca_dell_asse("velocità", 1)
    with pytest.raises(ValueError):
        sintesi.tacca_mono("velocità", 1)


def test_le_tacche_a_un_altra_frequenza_di_campionamento():
    tacca = sintesi.tacca_dell_asse("potenza", 3, fs=22050)
    assert tacca.shape == (round(TACCA_DURATA * 22050), 2)
    assert _frequenza_media(_mono(tacca), fs=22050) == pytest.approx(sintesi.frequenze_delle_tacche(TACCHE - 1)[2], rel=0.01)


@pytest.mark.parametrize("secondi", SECONDI_DELLE_MIRE)
def test_il_tick_del_tempo_sale(secondi):
    """Un tick a ogni secondo che passa, dal primo a secondi meno uno, al centro, nella forma TICK_FORMA: la frequenza
    sale da TICK_FREQUENZE[0] a TICK_FREQUENZE[1] su scala esponenziale."""
    basso, alto = TICK_FREQUENZE
    secondi_dei_tick = list(range(1, int(np.ceil(secondi))))
    misurate = []
    for secondo in secondi_dei_tick:
        tick = sintesi.tick_del_tempo(secondo, secondi)
        assert tick.dtype == np.float32 and tick.shape == (_campioni(TICK_DURATA), 2), secondo
        np.testing.assert_array_equal(tick[:, 0], tick[:, 1])
        mono = _mono(tick)
        misurate.append(_frequenza_media(mono))
        frazione = min(1.0, (secondo - 1) / (secondi - 2))
        assert misurate[-1] == pytest.approx(basso * (alto / basso) ** frazione, rel=0.01), secondo
        pieno = _senza_rampe(mono)
        assert 0.95 * sintesi.db_in_lineare(TICK_DB) < np.max(np.abs(pieno)) <= sintesi.db_in_lineare(TICK_DB), secondo
    assert misurate[0] == pytest.approx(basso, rel=0.01)
    assert np.all(np.diff(misurate) > 0)
    if float(secondi).is_integer():
        assert misurate[-1] == pytest.approx(alto, rel=0.01)


def test_la_forma_del_tick():
    assert TICK_FORMA == "triangolo"
    for secondo in (1, 15, 29):
        tick = sintesi.tick_del_tempo(secondo, 30.0)
        frazione = (secondo - 1) / 28.0
        frequenza = TICK_FREQUENZE[0] * (TICK_FREQUENZE[1] / TICK_FREQUENZE[0]) ** frazione
        assert _terza_armonica(_senza_rampe(_mono(tick)), frequenza) == pytest.approx(1.0 / 9.0, rel=0.1), secondo


def test_il_tick_fuori_dal_tempo():
    """Prima del primo secondo il tick è il più grave, dopo l'ultimo il più acuto; una mira di due secondi o meno ha il
    tick più grave."""
    np.testing.assert_array_equal(sintesi.tick_del_tempo(0, 30.0), sintesi.tick_del_tempo(1, 30.0))
    np.testing.assert_array_equal(sintesi.tick_del_tempo(40, 30.0), sintesi.tick_del_tempo(29, 30.0))
    for secondi in (2.0, 1.0):
        np.testing.assert_array_equal(sintesi.tick_del_tempo(1, secondi), sintesi.tick_del_tempo(1, 30.0))


def test_nessun_buffer_oltre_uno():
    """Le tacche di ogni valore, i tick di ogni mira, il suono di ogni valore lungo la scala e un bip a zero decibel
    restano dentro più o meno uno, senza un campione che non sia un numero."""
    buffer = [sintesi.tacca_dell_asse(parametro, k) for parametro in PARAMETRI for k in range(1, TACCHE)]
    buffer += [sintesi.tick_del_tempo(secondo, secondi) for secondi in SECONDI_DELLE_MIRE for secondo in range(0, int(secondi) + 1)]
    buffer += [_scala(parametro) for parametro in PARAMETRI]
    buffer += [sintesi.bip(frequenza, 0.05, 0.0, forma) for frequenza in (150.0, 2400.0, 9000.0) for forma in sintesi.FORME_PORTANTI]
    for dati in buffer:
        assert len(dati) > 0
        assert np.all(np.isfinite(dati))
        assert np.max(np.abs(dati)) <= 1.0
    griglia = np.linspace(0.0, 1.0, 11)
    _frequenze, pan, guadagni = sintesi.parametri_del_puntatore(griglia[:, None, None], griglia[None, :, None], griglia[None, None, :])
    assert pan.shape == guadagni.shape == (11, 11, 11)
    assert np.all((pan >= -1.0) & (pan <= 1.0)) and np.all((guadagni > 0.0) & (guadagni <= 1.0))

