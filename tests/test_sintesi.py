"""
Le prove di sintesi.py: le portanti, la panoramica, i bip e il buffer della mira, contati al campione.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
"""

import numpy as np
import pytest

import sintesi
from costanti import (
    ALZO_FREQUENZE,
    BIP_DISTACCO,
    BIP_DURATA,
    BIP_FREQUENZE,
    BIP_SILENZIO_FINALE,
    FS,
    PARAMETRI,
    POTENZA_DB_MINIMO,
    SCIVOLATA_DURATA,
    TACCA_DURATA,
    TACCHE,
)


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


def test_lunghezze_al_campione():
    for parametro in PARAMETRI:
        barra = sintesi.scivolata(parametro)
        quanti = len(BIP_FREQUENZE[parametro])
        atteso = quanti * _campioni(BIP_DURATA) + (quanti - 1) * _campioni(BIP_DISTACCO) + _campioni(BIP_SILENZIO_FINALE)
        assert barra.parametro == parametro
        assert barra.inizio_scivolata == atteso
        assert barra.campioni_scivolata == _campioni(SCIVOLATA_DURATA)
        assert barra.buffer.shape == (atteso + _campioni(SCIVOLATA_DURATA), 2)
        assert barra.buffer.dtype == np.float32
        assert barra.fine_scivolata == len(barra.buffer)
    corta = sintesi.scivolata("alzo", bip_durata=0.02)
    assert corta.inizio_scivolata == 3 * _campioni(0.02) + 2 * _campioni(BIP_DISTACCO) + _campioni(BIP_SILENZIO_FINALE)


def test_bip_e_silenzi_al_loro_posto():
    barra = sintesi.scivolata("direzione")
    sonoro = np.any(barra.buffer != 0.0, axis=1)
    lungo = _campioni(BIP_DURATA)
    distacco = _campioni(BIP_DISTACCO)
    for i in range(3):
        dove = i * (lungo + distacco)
        assert sonoro[dove:dove + lungo].sum() >= lungo - 2
        fine_silenzio = dove + lungo + (distacco if i < 2 else _campioni(BIP_SILENZIO_FINALE))
        assert not sonoro[dove + lungo:fine_silenzio].any()


def test_posizione_delle_tacche():
    con = sintesi.scivolata("alzo")
    senza = sintesi.scivolata("alzo", tacche=False)
    differenza = np.any(con.buffer != senza.buffer, axis=1)
    assert len(con.tacche) == TACCHE and senza.tacche == ()
    lunga = _campioni(TACCA_DURATA)
    dentro = np.zeros(len(differenza), dtype=bool)
    for k, (inizio, fine) in enumerate(con.tacche, start=1):
        assert fine == con.inizio_scivolata + round(k * con.campioni_scivolata / TACCHE)
        assert fine - inizio == lunga
        dentro[inizio:fine] = True
        toccati = np.flatnonzero(differenza[inizio:fine])
        assert len(toccati) >= lunga - 4
        assert toccati[-1] >= lunga - 3
    assert not differenza[~dentro].any()
    assert con.tacche[-1][1] == con.fine_scivolata


def test_tacche_salgono_di_frequenza():
    frequenze = sintesi.frequenze_delle_tacche()
    assert len(frequenze) == TACCHE
    assert np.all(np.diff(frequenze) > 0)
    rapporti = frequenze[1:] / frequenze[:-1]
    assert np.allclose(rapporti, rapporti[0])


def _scala(barra):
    return barra.buffer[barra.inizio_scivolata:].astype(np.float64)


def test_panoramica_della_direzione_da_sinistra_a_destra():
    barra = sintesi.scivolata("direzione", tacche=False)
    scala = _scala(barra)
    finestra = _campioni(0.1)
    pan = []
    for dove in range(0, len(scala) - finestra + 1, finestra):
        pezzo = scala[dove:dove + finestra]
        sinistra = np.sqrt(np.mean(pezzo[:, 0] ** 2))
        destra = np.sqrt(np.mean(pezzo[:, 1] ** 2))
        stimato = np.arctan2(destra, sinistra) * 4.0 / np.pi - 1.0
        atteso = 2.0 * (dove + finestra / 2.0) / len(scala) - 1.0
        assert stimato == pytest.approx(atteso, abs=0.02)
        pan.append(stimato)
    assert np.all(np.diff(pan) > 0)
    assert pan[0] < -0.9 and pan[-1] > 0.9


def test_frequenza_istantanea_dell_alzo_esponenziale():
    barra = sintesi.scivolata("alzo", tacche=False)
    scala = _scala(barra)[:, 0]
    istanti, frequenze = _frequenza_dagli_zeri(scala)
    attese = ALZO_FREQUENZE[0] * (ALZO_FREQUENZE[1] / ALZO_FREQUENZE[0]) ** (istanti / len(scala))
    assert np.allclose(frequenze, attese, rtol=0.01)
    assert frequenze[0] == pytest.approx(ALZO_FREQUENZE[0], rel=0.02)
    assert frequenze[-1] == pytest.approx(ALZO_FREQUENZE[1], rel=0.02)


def test_alzo_lineare_per_i_banchi():
    barra = sintesi.scivolata("alzo", tacche=False, curva="lineare")
    scala = _scala(barra)[:, 0]
    istanti, frequenze = _frequenza_dagli_zeri(scala)
    attese = ALZO_FREQUENZE[0] + (ALZO_FREQUENZE[1] - ALZO_FREQUENZE[0]) * istanti / len(scala)
    assert np.allclose(frequenze, attese, rtol=0.01)


def _livelli_in_db(barra, finestra):
    scala = _scala(barra)
    centri, livelli = [], []
    for dove in range(0, len(scala) - finestra + 1, finestra):
        pezzo = scala[dove:dove + finestra, 0]
        centri.append((dove + finestra / 2.0) / len(scala))
        livelli.append(20.0 * np.log10(np.sqrt(np.mean(pezzo ** 2))))
    return np.array(centri), np.array(livelli)


def test_inviluppo_della_potenza_in_decibel():
    """La potenza a 330 hertz: un decimo di secondo sono 33 periodi esatti, e il livello in decibel
    sale in linea retta di POTENZA_DB_MINIMO, cambiato di segno, lungo la scala."""
    barra = sintesi.scivolata("potenza", tacche=False)
    centri, livelli = _livelli_in_db(barra, _campioni(0.1))
    # La prima e l'ultima finestra contengono le rampe della scala, e restano fuori dal conto.
    relativi = livelli[1:-1] - livelli[-2]
    attesi = -POTENZA_DB_MINIMO * (centri[1:-1] - centri[-2])
    assert np.allclose(relativi, attesi, atol=0.2)
    pendenza = np.polyfit(centri[1:-1], livelli[1:-1], 1)[0]
    assert pendenza == pytest.approx(-POTENZA_DB_MINIMO, rel=0.01)


def test_potenza_varianti():
    ridotta = sintesi.scivolata("potenza", tacche=False, db_minimo=-20.0)
    centri, livelli = _livelli_in_db(ridotta, _campioni(0.1))
    assert np.polyfit(centri[1:-1], livelli[1:-1], 1)[0] == pytest.approx(20.0, rel=0.01)
    in_ampiezza = sintesi.scivolata("potenza", tacche=False, curva="ampiezza")
    centri, livelli = _livelli_in_db(in_ampiezza, _campioni(0.1))
    ampiezze = 10.0 ** (livelli / 20.0)
    assert np.polyfit(centri[1:-1], ampiezze[1:-1], 2)[0] == pytest.approx(0.0, abs=0.01)
    with pytest.raises(ValueError):
        sintesi.scivolata("potenza", curva="esponenziale")
    with pytest.raises(ValueError):
        sintesi.scivolata("velocità")


def test_nessun_campione_oltre_uno():
    varianti = [
        {},
        {"forma": "sega_morbida"},
        {"forma": "organo", "frequenze": (100.0, 15000.0)},
        {"tacca_db": 0.0},
        {"bip_durata": 0.03},
    ]
    for parametro in PARAMETRI:
        for variante in varianti:
            barra = sintesi.scivolata(parametro, **variante)
            assert np.all(np.isfinite(barra.buffer))
            assert np.max(np.abs(barra.buffer)) <= 1.0
