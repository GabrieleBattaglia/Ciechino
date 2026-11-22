"""
Le prove di mira.py, la mira manuale (D29): ogni valore corre a VELOCITA_DELLA_MIRA nel verso del suo tasto
e si ferma ai capi; due tasti dello stesso valore danno il verso all'ultimo sceso, e lasciato quello torna
l'altro; la ripetizione di un tasto già giù non cambia niente; lascia_tutto ferma tutti; le tacche e il capo
davanti a un valore cadono ai tempi giusti, in salita e in discesa.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0; dalla 2.0.0 prova Asse e Mira, e non più valore_scelto, la compensazione
del tasto, il margine e la scadenza delle scivolate, usciti con D29.
"""

import itertools

import numpy as np
import pytest

from costanti import DURATA_DELLA_SCALA, PARAMETRI, PARTENZA_DELLA_MIRA, TACCHE, VELOCITA_DELLA_MIRA
from mira import Asse, Mira

# Un tempo d'audio qualunque, in secondi, lontano dallo zero: così ogni conto parte da un dal che non vale zero.
INIZIO = 12.5


def _tempo(valore_da, valore_a):
    """I secondi che servono a un valore per andare da valore_da a valore_a."""
    return abs(valore_a - valore_da) / VELOCITA_DELLA_MIRA


def test_le_costanti_della_mira():
    """D29: i tre valori partono dal centro e percorrono la scala in 3 secondi, quanto durava una scivolata."""
    assert PARAMETRI == ("direzione", "alzo", "potenza")
    assert PARTENZA_DELLA_MIRA == 0.5
    assert DURATA_DELLA_SCALA == 3.0
    assert round(VELOCITA_DELLA_MIRA * DURATA_DELLA_SCALA, 12) == 1.0
    assert TACCHE == 10


def test_l_asse_parte_fermo_al_centro():
    asse = Asse()
    assert asse.stato == (PARTENZA_DELLA_MIRA, 0, 0.0)
    assert asse.verso == 0
    for tempo in (0.0, INIZIO, 1000.0):
        assert asse.valore(tempo) == PARTENZA_DELLA_MIRA
    assert asse.capo() is None and asse.tempo_al_capo() is None
    assert asse.tacche_davanti() == []
    # Un valore di partenza fuori dalla scala si porta al capo più vicino.
    assert Asse(1.7).valore(0.0) == 1.0
    assert Asse(-0.3).valore(0.0) == 0.0
    assert Asse(0.2).valore(INIZIO) == pytest.approx(0.2)


def test_il_valore_corre_alla_velocita_della_mira_e_si_ferma_ai_capi():
    su = Asse()
    su.muovi(1, INIZIO)
    assert su.valore(INIZIO) == pytest.approx(0.5)
    assert su.valore(INIZIO + 0.3) == pytest.approx(0.5 + 0.3 * VELOCITA_DELLA_MIRA)
    assert su.valore(INIZIO + 0.9) == pytest.approx(0.8)
    # Dal centro al capo ci vuole metà della scala, poi il valore resta sul capo.
    assert su.valore(INIZIO + DURATA_DELLA_SCALA / 2) == pytest.approx(1.0)
    for dopo in (2.0, 5.0, 100.0):
        assert su.valore(INIZIO + dopo) == 1.0
    # Prima del tasto il valore è quello di partenza: non corre indietro nel tempo.
    assert su.valore(INIZIO - 1.0) == pytest.approx(0.5)
    giu = Asse()
    giu.muovi(-1, INIZIO)
    assert giu.valore(INIZIO + 0.75) == pytest.approx(0.25)
    assert giu.valore(INIZIO + DURATA_DELLA_SCALA / 2) == pytest.approx(0.0, abs=1e-12)
    for dopo in (2.0, 5.0, 100.0):
        assert giu.valore(INIZIO + dopo) == 0.0


def test_da_un_capo_all_altro_in_durata_della_scala():
    asse = Asse(0.0)
    asse.muovi(1, INIZIO)
    assert asse.valore(INIZIO + DURATA_DELLA_SCALA - 0.001) < 1.0
    assert asse.valore(INIZIO + DURATA_DELLA_SCALA) == pytest.approx(1.0)
    assert asse.tempo_al_capo() == pytest.approx(INIZIO + DURATA_DELLA_SCALA)
    asse = Asse(1.0)
    asse.muovi(-1, INIZIO)
    assert asse.valore(INIZIO + DURATA_DELLA_SCALA - 0.001) > 0.0
    assert asse.valore(INIZIO + DURATA_DELLA_SCALA) == pytest.approx(0.0, abs=1e-12)
    assert asse.tempo_al_capo() == pytest.approx(INIZIO + DURATA_DELLA_SCALA)


def test_il_valore_di_un_array_di_tempi():
    """Il motore chiede il valore di ogni campione di un blocco in una volta: lo stesso conto, campione per campione."""
    asse = Asse(0.3)
    asse.muovi(1, INIZIO)
    tempi = INIZIO + np.linspace(-1.0, 4.0, 501)
    valori = asse.valore(tempi)
    assert isinstance(valori, np.ndarray) and valori.shape == tempi.shape
    np.testing.assert_allclose(valori, [asse.valore(float(tempo)) for tempo in tempi], rtol=0, atol=1e-15)
    assert np.all(np.diff(valori) >= 0.0)
    assert valori[0] == pytest.approx(0.3) and valori[-1] == 1.0
    assert np.all((valori >= 0.0) & (valori <= 1.0))
    # Un tempo solo, anche come array di dimensione zero, dà un float.
    assert type(asse.valore(INIZIO + 1.0)) is float
    assert type(asse.valore(np.float64(INIZIO + 1.0))) is float
    assert type(asse.valore(np.array(INIZIO + 1.0))) is float
    # Un blocco a due dimensioni resta della sua forma.
    assert asse.valore(np.full((4, 3), INIZIO + 0.3)).shape == (4, 3)


def test_muovi_riparte_dal_valore_del_momento():
    asse = Asse()
    asse.muovi(1, INIZIO)
    asse.muovi(-1, INIZIO + 0.6)
    valore, verso, dal = asse.stato
    assert valore == pytest.approx(0.7) and verso == -1 and dal == INIZIO + 0.6
    # Il cambio di verso non fa saltare il valore: prima e dopo dell'istante, lo stesso.
    assert asse.valore(INIZIO + 0.6) == pytest.approx(0.7)
    assert asse.valore(INIZIO + 0.9) == pytest.approx(0.6)
    asse.ferma(INIZIO + 1.2)
    assert asse.verso == 0
    assert asse.stato[0] == pytest.approx(0.5)
    for dopo in (1.2, 3.0, 60.0):
        assert asse.valore(INIZIO + dopo) == pytest.approx(0.5)


def test_capo_e_tempo_al_capo():
    for partenza, verso in itertools.product((0.0, 0.2, 0.5, 0.75, 1.0), (-1, 1)):
        asse = Asse(partenza)
        asse.muovi(verso, INIZIO)
        capo = 1.0 if verso > 0 else 0.0
        assert asse.capo() == capo, (partenza, verso)
        assert asse.tempo_al_capo() == pytest.approx(INIZIO + _tempo(partenza, capo)), (partenza, verso)
        assert asse.valore(asse.tempo_al_capo()) == pytest.approx(capo, abs=1e-12), (partenza, verso)
    # Un valore già al capo ci arriva subito, al tempo del tasto.
    asse = Asse(1.0)
    asse.muovi(1, INIZIO)
    assert asse.tempo_al_capo() == INIZIO
    asse = Asse(0.0)
    asse.muovi(-1, INIZIO)
    assert asse.tempo_al_capo() == INIZIO
    # Fermo, nessun capo.
    asse.ferma(INIZIO + 1.0)
    assert asse.capo() is None and asse.tempo_al_capo() is None


def test_tacche_davanti_in_salita_e_in_discesa():
    """Dal centro, in su, le tacche dal sesto al nono decimo, una ogni 0,3 secondi; in giù dal quarto al primo. La
    tacca del quinto decimo, su cui il valore sta già, non suona, e il decimo decimo è il capo, col suo suono."""
    su = Asse()
    su.muovi(1, INIZIO)
    tacche = su.tacche_davanti()
    assert [k for _tempo_, k in tacche] == [6, 7, 8, 9]
    np.testing.assert_allclose([tempo for tempo, _k in tacche], INIZIO + np.array([0.3, 0.6, 0.9, 1.2]), rtol=0, atol=1e-9)
    giu = Asse()
    giu.muovi(-1, INIZIO)
    tacche = giu.tacche_davanti()
    assert [k for _tempo_, k in tacche] == [4, 3, 2, 1]
    np.testing.assert_allclose([tempo for tempo, _k in tacche], INIZIO + np.array([0.3, 0.6, 0.9, 1.2]), rtol=0, atol=1e-9)
    # Da un capo all'altro tutte e nove, nei due sensi.
    asse = Asse(0.0)
    asse.muovi(1, INIZIO)
    assert [k for _tempo_, k in asse.tacche_davanti()] == list(range(1, TACCHE))
    asse = Asse(1.0)
    asse.muovi(-1, INIZIO)
    assert [k for _tempo_, k in asse.tacche_davanti()] == list(range(TACCHE - 1, 0, -1))
    # Da un punto fra due tacche, la prima tacca arriva prima di 0,3 secondi.
    asse = Asse(0.25)
    asse.muovi(1, INIZIO)
    assert asse.tacche_davanti()[0] == (pytest.approx(INIZIO + 0.15), 3)
    asse.muovi(-1, INIZIO)
    assert asse.tacche_davanti() == [(pytest.approx(INIZIO + 0.15), 2), (pytest.approx(INIZIO + 0.45), 1)]


def test_tacche_davanti_dal_capo_o_da_fermo():
    """Spinto oltre il capo su cui sta, il valore non ha tacche davanti; fermo nemmeno."""
    asse = Asse(1.0)
    asse.muovi(1, INIZIO)
    assert asse.tacche_davanti() == []
    asse = Asse(0.0)
    asse.muovi(-1, INIZIO)
    assert asse.tacche_davanti() == []
    asse = Asse(0.4)
    assert asse.tacche_davanti() == []


def test_le_tacche_dopo_un_cambio_di_verso():
    """Fermato o girato su una tacca, con il conto in virgola mobile, il valore non la risuona: la tacca del settimo
    decimo, appena passata in salita, non torna in discesa."""
    asse = Asse()
    asse.muovi(1, INIZIO)
    asse.muovi(-1, INIZIO + 0.6)
    tacche = asse.tacche_davanti()
    assert [k for _tempo_, k in tacche] == [6, 5, 4, 3, 2, 1]
    assert tacche[0][0] == pytest.approx(INIZIO + 0.9)
    # Fermato sulla tacca del quinto decimo, e ripartito in su, comincia dal sesto.
    asse.ferma(INIZIO + 1.2)
    asse.muovi(1, INIZIO + 2.0)
    assert [k for _tempo_, k in asse.tacche_davanti()] == [6, 7, 8, 9]


@pytest.mark.parametrize("verso", [-1, 1])
def test_ogni_tacca_cade_quando_il_valore_la_attraversa(verso):
    """Per partenze a passi di un quarantesimo, che cadono anche sulle tacche: davanti ci sono soltanto i decimi
    oltre il valore nel verso del moto, ciascuno al tempo in cui il valore vale proprio quel decimo, in ordine di
    tempo e prima del capo."""
    for partenza in np.linspace(0.0, 1.0, 41):
        asse = Asse(partenza)
        asse.muovi(verso, INIZIO)
        tacche = asse.tacche_davanti()
        attese = [k for k in range(1, TACCHE) if (k / TACCHE - partenza) * verso > 1e-9]
        assert sorted(k for _tempo_, k in tacche) == attese, partenza
        tempi = [tempo for tempo, _k in tacche]
        assert tempi == sorted(tempi)
        for tempo, k in tacche:
            assert isinstance(k, int)
            assert tempo > INIZIO
            assert tempo == pytest.approx(INIZIO + _tempo(partenza, k / TACCHE))
            assert asse.valore(tempo) == pytest.approx(k / TACCHE)
            assert tempo < asse.tempo_al_capo()


def test_la_mira_parte_ferma_al_centro():
    m = Mira()
    assert list(m.assi) == list(PARAMETRI)
    assert len({id(asse) for asse in m.assi.values()}) == len(PARAMETRI)
    assert m.valori(INIZIO) == [PARTENZA_DELLA_MIRA] * 3
    assert m.valori(INIZIO + 100.0) == [PARTENZA_DELLA_MIRA] * 3
    assert m.tasti_giu() == []
    assert m.pressioni == 0
    assert Mira(partenza=0.2).valori(0.0) == pytest.approx([0.2] * 3)


def test_un_tasto_muove_soltanto_il_suo_valore():
    m = Mira()
    assert m.premi("direzione", 1, INIZIO) is True
    assert m.valori(INIZIO + 0.3) == pytest.approx([0.6, 0.5, 0.5])
    assert m.tasti_giu() == [("direzione", 1)]
    assert m.lascia("direzione", 1, INIZIO + 0.6) is True
    assert m.tasti_giu() == []
    for dopo in (0.6, 2.0, 30.0):
        assert m.valori(INIZIO + dopo) == pytest.approx([0.7, 0.5, 0.5])
    assert m.pressioni == 1


def test_ogni_tasto_nel_suo_verso_fino_al_capo():
    """Le frecce sinistra e giù, e Home, portano il loro valore verso il minimo; destra, su e Fine verso il massimo."""
    for parametro, verso in itertools.product(PARAMETRI, (-1, 1)):
        m = Mira()
        m.premi(parametro, verso, INIZIO)
        indice = PARAMETRI.index(parametro)
        assert m.valori(INIZIO + 0.6)[indice] == pytest.approx(0.5 + 0.2 * verso)
        capo = 1.0 if verso > 0 else 0.0
        for dopo in (DURATA_DELLA_SCALA / 2, 2.0, 10.0):
            assert m.valori(INIZIO + dopo)[indice] == pytest.approx(capo, abs=1e-12)
        # Gli altri due restano al centro.
        assert [v for i, v in enumerate(m.valori(INIZIO + 10.0)) if i != indice] == [0.5, 0.5]


def test_due_valori_si_muovono_insieme():
    m = Mira()
    m.premi("direzione", 1, INIZIO)
    m.premi("alzo", -1, INIZIO + 0.3)
    m.premi("potenza", 1, INIZIO + 0.6)
    assert m.valori(INIZIO + 0.9) == pytest.approx([0.8, 0.3, 0.6])
    m.lascia("alzo", -1, INIZIO + 0.9)
    assert m.valori(INIZIO + 1.2) == pytest.approx([0.9, 0.3, 0.7])
    assert m.tasti_giu() == [("direzione", 1), ("potenza", 1)]
    assert m.pressioni == 3


def test_due_tasti_dello_stesso_valore_vale_l_ultimo_sceso():
    """Il docstring di Mira: tenendo giù la freccia sinistra e poi anche la destra il valore va a destra, e lasciata la
    destra torna a sinistra; lasciata anche la sinistra si ferma."""
    m = Mira()
    direzione = m.assi["direzione"]
    assert m.premi("direzione", -1, INIZIO) is True
    assert direzione.verso == -1
    assert m.premi("direzione", 1, INIZIO + 0.3) is True
    assert direzione.verso == 1
    assert m.tasti_giu() == [("direzione", -1), ("direzione", 1)]
    assert m.valori(INIZIO + 0.3)[0] == pytest.approx(0.4)
    assert m.valori(INIZIO + 0.9)[0] == pytest.approx(0.6)
    assert m.lascia("direzione", 1, INIZIO + 0.9) is True
    assert direzione.verso == -1
    assert m.tasti_giu() == [("direzione", -1)]
    assert m.valori(INIZIO + 1.2)[0] == pytest.approx(0.5)
    assert m.lascia("direzione", -1, INIZIO + 1.2) is True
    assert direzione.verso == 0 and m.tasti_giu() == []
    assert m.valori(INIZIO + 5.0)[0] == pytest.approx(0.5)
    assert m.pressioni == 2


def test_lasciato_il_tasto_piu_vecchio_comanda_ancora_l_ultimo():
    """Con i due tasti giù, lasciare quello sceso prima non cambia il moto: premi e lascia dicono vero soltanto
    quando il moto cambia."""
    m = Mira()
    m.premi("alzo", -1, INIZIO)
    m.premi("alzo", 1, INIZIO + 0.3)
    stato = m.assi["alzo"].stato
    assert m.lascia("alzo", -1, INIZIO + 0.6) is False
    assert m.assi["alzo"].stato == stato
    assert m.tasti_giu() == [("alzo", 1)]
    assert m.valori(INIZIO + 0.9)[1] == pytest.approx(0.6)


def test_la_ripetizione_di_un_tasto_gia_giu_non_cambia_niente():
    m = Mira()
    m.premi("potenza", 1, INIZIO)
    stato = m.assi["potenza"].stato
    for ripetizione in range(1, 20):
        assert m.premi("potenza", 1, INIZIO + 0.03 * ripetizione) is False
    assert m.assi["potenza"].stato == stato
    assert m.tasti_giu() == [("potenza", 1)]
    assert m.pressioni == 1
    assert m.valori(INIZIO + 0.9)[2] == pytest.approx(0.8)
    # Nemmeno la ripetizione del tasto sceso prima, mentre comanda l'altro, cambia il verso o l'ordine.
    m.premi("potenza", -1, INIZIO + 0.9)
    stato = m.assi["potenza"].stato
    assert m.premi("potenza", 1, INIZIO + 1.0) is False
    assert m.assi["potenza"].stato == stato and m.assi["potenza"].verso == -1
    assert m.tasti_giu() == [("potenza", 1), ("potenza", -1)]
    assert m.pressioni == 2


def test_lasciare_un_tasto_che_non_era_giu_non_cambia_niente():
    m = Mira()
    assert m.lascia("potenza", 1, INIZIO) is False
    m.premi("direzione", 1, INIZIO)
    stato = m.assi["direzione"].stato
    assert m.lascia("direzione", -1, INIZIO + 0.3) is False
    assert m.lascia("alzo", 1, INIZIO + 0.3) is False
    assert m.assi["direzione"].stato == stato
    assert m.tasti_giu() == [("direzione", 1)]


def test_lascia_tutto_ferma_tutti():
    m = Mira()
    m.premi("direzione", 1, INIZIO)
    m.premi("alzo", -1, INIZIO)
    m.premi("alzo", 1, INIZIO + 0.15)
    assert m.lascia_tutto(INIZIO + 0.3) == ["direzione", "alzo"]
    assert m.tasti_giu() == []
    assert all(asse.verso == 0 for asse in m.assi.values())
    for dopo in (0.3, 1.0, 30.0):
        assert m.valori(INIZIO + dopo) == pytest.approx([0.6, 0.5, 0.5])
    # Tutto fermo, una seconda volta non cambia niente, e un tasto lasciato dopo nemmeno.
    assert m.lascia_tutto(INIZIO + 0.5) == []
    assert m.lascia("direzione", 1, INIZIO + 0.6) is False
    # Dopo, lo stesso tasto è una pressione nuova, non una ripetizione.
    assert m.premi("direzione", 1, INIZIO + 1.0) is True
    assert m.valori(INIZIO + 1.3)[0] == pytest.approx(0.7)
    assert m.pressioni == 4


def test_i_valori_nell_ordine_dei_parametri():
    m = Mira()
    m.premi("potenza", 1, INIZIO)
    m.premi("direzione", -1, INIZIO)
    valori = m.valori(INIZIO + 0.6)
    assert valori == pytest.approx([0.3, 0.5, 0.7])
    assert valori == [m.assi[parametro].valore(INIZIO + 0.6) for parametro in PARAMETRI]
    assert all(type(valore) is float for valore in valori)


def test_parametro_o_verso_sbagliati():
    m = Mira()
    with pytest.raises(ValueError):
        m.premi("velocità", 1, INIZIO)
    with pytest.raises(ValueError):
        m.lascia("velocità", 1, INIZIO)
    for verso in (0, 2, -2):
        with pytest.raises(ValueError):
            m.premi("alzo", verso, INIZIO)
        with pytest.raises(ValueError):
            m.lascia("alzo", verso, INIZIO)
    assert m.tasti_giu() == [] and m.pressioni == 0
    assert m.valori(INIZIO + 1.0) == [PARTENZA_DELLA_MIRA] * 3

