"""
Le prove della classe Suoni di suoni.py: la voce al campione chiesto, i buffer collocati nello spazio,
il guadagno, la memoria dei preset, gli eventi sconosciuti e il conteggio; e, sulla collezione vera in
sola lettura, che ogni preset degli eventi ci sia e porti la firma di Ciechino.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0; dalla 2.0.0 i punti delle prove stanno nella cupola di D28, e il suono
frequente che si conta è fine_scala della mira manuale di D29, al posto del valore preso, uscito con lei.
Nessuna prova sintetizza né suona un preset vero: _rendi_preset è sostituito da una fabbrica finta,
che dà a ogni preset lo stesso buffer al centro e si ricorda chi glielo ha chiesto; il motore è
quello muto del conftest o quello col flusso finto di aiuti_audio, che dice al campione che cosa esce.
Le due prove sulla collezione vera la leggono soltanto: falliscono finché strumenti/crea_preset.py
non ha aggiunto e firmato i preset di Ciechino nella collezione condivisa di GBUtils, ed è giusto
così, perché senza quel passo il gioco resta muto.
"""

import json

import numpy as np
import pytest
from aiuti_audio import motore_finto

import suoni
from costanti import CUPOLA_RAGGIO, CUPOLA_RAGGIO_INTERNO, FS, SETTORE_AZIMUT
from fisica import Zona, distanza_dal_cannone, guadagno_di, punto_della_cupola
from motore_audio import Motore, VoceBuffer
from sintesi import db_in_lineare, panoramica
from statistiche import Statistiche
from strumenti import crea_preset
from suoni import EVENTI, Suoni

# Il buffer finto: una rampa che scende, così ogni campione ha il suo valore e un suono spostato
# anche di un solo campione si riconosce; al centro, come quasi tutti i preset veri, e con un picco
# lontano dal tetto morbido del motore, che altrimenti lo piegherebbe.
LUNGHEZZA_FINTA = 2_000
AMPIEZZA_FINTA = 0.5
# Un punto proprio davanti al cannone, uno vicino, pure davanti, e uno lontano a destra, dentro la cupola dei
# bersagli: il vicino appena oltre la sfera interna, il lontano appena dentro quella esterna e il lato destro.
DAVANTI = (0.0, 20.0, 50.0)
VICINO = tuple(float(c) for c in punto_della_cupola(CUPOLA_RAGGIO_INTERNO + 1.0, 0.0, 15.0))
LONTANO_A_DESTRA = tuple(float(c) for c in punto_della_cupola(CUPOLA_RAGGIO - 1.0, SETTORE_AZIMUT - 5.0, 10.0))
# Il campione da cui far partire una voce col motore muto: dieci secondi avanti, così il filo muto,
# che corre col tempo vero, non lo passa prima che il comando arrivi.
AVANTI_COL_MOTORE_MUTO = 10 * FS


def buffer_finto():
    mono = AMPIEZZA_FINTA * (1.0 - np.arange(LUNGHEZZA_FINTA) / LUNGHEZZA_FINTA)
    buffer = panoramica(mono, 0.0)
    buffer.flags.writeable = False
    return buffer


class Fabbrica:
    """La _rendi_preset finta: lo stesso buffer per ogni preset, tranne quelli vuoti, che non hanno
    suono, e quelli guasti, che non si sintetizzano; chiesti sono i nomi, nell'ordine."""

    def __init__(self):
        self.chiesti = []
        self.vuoti = set()
        self.guasti = set()

    def __call__(self, nome):
        self.chiesti.append(nome)
        if nome in self.guasti:
            raise RuntimeError("sintesi guasta")
        if nome in self.vuoti:
            return np.zeros((0, 2), dtype=np.float32)
        return buffer_finto()


@pytest.fixture
def fabbrica(monkeypatch):
    finta = Fabbrica()
    monkeypatch.setattr(suoni, "_rendi_preset", finta)
    return finta


@pytest.fixture
def finto():
    motore, flusso = motore_finto()
    yield motore, flusso
    motore.chiudi()


def _energia(buffer, lato=None):
    dati = np.asarray(buffer, dtype=np.float64)
    if lato is not None:
        dati = dati[:, lato]
    return float(np.sum(dati * dati))


def test_i_punti_delle_prove_stanno_nella_cupola():
    zona = Zona()
    for punto in (DAVANTI, VICINO, LONTANO_A_DESTRA):
        assert zona.contiene(np.array(punto)), punto
    assert VICINO[0] == DAVANTI[0] == 0.0 and LONTANO_A_DESTRA[0] > 0.0
    assert distanza_dal_cannone(np.array(VICINO)) < distanza_dal_cannone(np.array(LONTANO_A_DESTRA))


def test_suona_parte_al_campione_chiesto(fabbrica, finto):
    motore, flusso = finto
    voce = Suoni(motore).suona("sparo", inizio=700)
    assert isinstance(voce, VoceBuffer) and voce.inizio == 700
    uscita = flusso.avanza_fino(700 + LUNGHEZZA_FINTA)
    assert np.all(uscita[:700] == 0.0)
    np.testing.assert_allclose(uscita[700:700 + LUNGHEZZA_FINTA], buffer_finto(), rtol=0, atol=1e-7)
    assert np.all(uscita[700 + LUNGHEZZA_FINTA:] == 0.0)


def test_senza_inizio_parte_al_prossimo_campione(fabbrica, finto):
    motore, flusso = finto
    flusso.avanza(3)
    atteso = motore.prossimo_campione()
    voce = Suoni(motore).suona("mira_avviata")
    assert voce.inizio == atteso
    flusso.avanza_fino(atteso + LUNGHEZZA_FINTA)
    uscita = flusso.uscita
    assert np.all(uscita[:atteso] == 0.0)
    np.testing.assert_allclose(uscita[atteso:atteso + LUNGHEZZA_FINTA], buffer_finto(), rtol=0, atol=1e-7)


def test_suona_col_motore_muto(fabbrica):
    motore = Motore()
    try:
        assert motore.avvia() is False and motore.muto
        inizio = motore.campione_scritto + AVANTI_COL_MOTORE_MUTO
        voce = Suoni(motore).suona("pausa", inizio=inizio)
        assert isinstance(voce, VoceBuffer) and voce.inizio == inizio
        np.testing.assert_array_equal(voce.buffer, buffer_finto())
    finally:
        motore.chiudi()


def test_suona_con_la_posizione(fabbrica, finto):
    motore, flusso = finto
    voce = Suoni(motore).suona("colpito", posizione=LONTANO_A_DESTRA, inizio=0)
    uscita = flusso.avanza_fino(LUNGHEZZA_FINTA)
    np.testing.assert_allclose(uscita[:LUNGHEZZA_FINTA], Suoni(None).pronto("colpito", posizione=LONTANO_A_DESTRA), rtol=0, atol=1e-7)
    assert voce.inizio == 0


def test_pronto_lontano_a_destra_e_piu_forte_a_destra_e_piu_debole(fabbrica):
    # pronto non tocca il motore: prepara soltanto.
    lontano = Suoni(None).pronto("colpito", posizione=LONTANO_A_DESTRA)
    vicino = Suoni(None).pronto("colpito", posizione=VICINO)
    assert lontano.shape == vicino.shape == (LUNGHEZZA_FINTA, 2)
    assert lontano.dtype == np.float32
    assert _energia(lontano, 1) > 2.0 * _energia(lontano, 0)
    assert _energia(lontano) < _energia(vicino)
    # Il punto vicino sta davanti al cannone, e i due lati si sentono uguali.
    assert _energia(vicino, 0) == pytest.approx(_energia(vicino, 1), rel=1e-6)


def test_pronto_davanti_e_il_preset_per_il_guadagno_della_distanza(fabbrica):
    # Il preset al centro, fatto mono e rimesso al centro, non perde niente: resta soltanto il
    # guadagno della distanza.
    buffer = Suoni(None).pronto("a_terra", posizione=DAVANTI)
    atteso = buffer_finto() * guadagno_di(distanza_dal_cannone(np.array(DAVANTI)))
    np.testing.assert_allclose(buffer, atteso, rtol=0, atol=1e-6)


def test_il_guadagno_in_decibel(fabbrica):
    suoni_ = Suoni(None)
    np.testing.assert_allclose(suoni_.pronto("sparo", guadagno_db=-6.0), buffer_finto() * db_in_lineare(-6.0), rtol=0, atol=1e-7)
    collocato = suoni_.pronto("colpito", posizione=VICINO)
    np.testing.assert_allclose(suoni_.pronto("colpito", posizione=VICINO, guadagno_db=-6.0), collocato * db_in_lineare(-6.0), rtol=0, atol=1e-7)
    # Il buffer della memoria non cambia per un guadagno chiesto una volta.
    np.testing.assert_array_equal(suoni_.pronto("sparo"), buffer_finto())


def test_la_memoria_rende_ogni_preset_una_volta_sola(fabbrica, finto):
    motore, _flusso = finto
    suoni_ = Suoni(motore)
    for _volta in range(3):
        suoni_.suona("sparo")
        suoni_.pronto("sparo")
        suoni_.pronto("sparo", posizione=VICINO, guadagno_db=-3.0)
        suoni_.suona("sparo", posizione=LONTANO_A_DESTRA)
    assert fabbrica.chiesti == [EVENTI["sparo"]]
    # Senza posizione e a zero decibel il buffer è quello della memoria, e nessuno lo può cambiare.
    primo = suoni_.pronto("sparo")
    assert primo is suoni_.pronto("sparo")
    assert not primo.flags.writeable
    suoni_.suona("pausa")
    assert fabbrica.chiesti == [EVENTI["sparo"], EVENTI["pausa"]]


def test_precarica(fabbrica, finto):
    motore, _flusso = finto
    suoni_ = Suoni(motore)
    assert suoni_.precarica() == len(EVENTI)
    assert sorted(fabbrica.chiesti) == sorted(set(EVENTI.values()))
    fabbrica.chiesti.clear()
    for evento in EVENTI:
        suoni_.suona(evento)
        suoni_.pronto(evento, posizione=VICINO)
    assert fabbrica.chiesti == []
    assert suoni_.precarica(["sparo", "colpito"]) == 2


def test_evento_sconosciuto_non_solleva(fabbrica, finto, capsys):
    motore, flusso = finto
    contati = []
    suoni_ = Suoni(motore, conta=contati.append)
    for _volta in range(3):
        assert suoni_.suona("evento_che_non_c_e") is None
        assert suoni_.pronto("evento_che_non_c_e") is None
        assert suoni_.pronto("evento_che_non_c_e", posizione=VICINO) is None
    assert suoni_.suona(["un", "evento", "che", "non", "si", "cerca"]) is None
    assert suoni_.suona(None) is None
    assert suoni_.precarica(["evento_che_non_c_e", "sparo"]) == 1
    errori = capsys.readouterr().err
    # Una volta sola per evento, per non riempire il registro.
    assert errori.count("evento_che_non_c_e") == 1
    assert "evento sonoro sconosciuto" in errori
    assert contati == []
    assert fabbrica.chiesti == [EVENTI["sparo"]]
    assert np.all(flusso.avanza(6) == 0.0)


def test_il_conteggio(fabbrica, finto):
    motore, _flusso = finto
    contati = []
    suoni_ = Suoni(motore, conta=contati.append)
    suoni_.suona("sparo")
    suoni_.pronto("colpito", posizione=LONTANO_A_DESTRA)
    suoni_.suona("fine_scala")
    suoni_.suona("fine_scala", posizione=VICINO, guadagno_db=-3.0)
    assert contati == ["sparo", "colpito", "fine_scala", "fine_scala"]
    # precarica prepara, non suona: non conta.
    suoni_.precarica()
    assert len(contati) == 4


def test_gli_eventi_usciti_con_la_mira_manuale_non_suonano(fabbrica, finto, capsys):
    """D29: il valore preso e il proiettile perduto non sono più eventi del gioco, e suonarli è come suonare un evento
    sconosciuto, che non si conta; il capo della scala ha il suo evento, fine_scala."""
    motore, _flusso = finto
    contati = []
    suoni_ = Suoni(motore, conta=contati.append)
    for evento in ("valore_preso", "proiettile_perduto"):
        assert evento not in EVENTI
        assert suoni_.suona(evento) is None
    assert capsys.readouterr().err.count("evento sonoro sconosciuto") == 2
    assert contati == [] and fabbrica.chiesti == []
    assert isinstance(suoni_.suona("fine_scala"), VoceBuffer)
    assert fabbrica.chiesti == ["ciechino_fine_scala"] and contati == ["fine_scala"]


def test_un_effetto_preparato_si_conta_quando_suona(fabbrica, finto, capsys):
    """L'impatto di un proiettile si prepara allo sparo e suona soltanto se l'evento programmato
    scatta: preparato con conta_subito falso non si conta, e si conta con suonato quando è partito."""
    motore, _flusso = finto
    contati = []
    suoni_ = Suoni(motore, conta=contati.append)
    contato = suoni_.pronto("a_terra", posizione=VICINO)
    rimandato = suoni_.pronto("a_terra", posizione=VICINO, conta_subito=False)
    np.testing.assert_array_equal(rimandato, contato)
    assert contati == ["a_terra"]
    # La partita abbandonata in volo: l'effetto non suona, e non si conta mai. Quello sentito sì.
    suoni_.suonato("colpito")
    assert contati == ["a_terra", "colpito"]
    # Un evento sconosciuto o senza suono non si conta, come in suona e in pronto.
    fabbrica.vuoti.add(EVENTI["pausa"])
    suoni_.suonato("pausa")
    suoni_.suonato("evento_che_non_c_e")
    assert contati == ["a_terra", "colpito"]
    assert capsys.readouterr().err.count("evento_che_non_c_e") == 1


def test_il_conteggio_collegato_dopo_alle_statistiche(fabbrica, finto):
    motore, _flusso = finto
    statistiche = Statistiche()
    suoni_ = Suoni(motore)
    suoni_.suona("nuova_partita")
    suoni_.conta = statistiche.suono
    suoni_.suona("sparo")
    suoni_.suona("sparo")
    suoni_.pronto("a_terra", posizione=VICINO)
    assert statistiche.dati["suoni"] == {"sparo": 2, "a_terra": 1}


def test_un_conteggio_guasto_non_toglie_il_suono(fabbrica, finto, capsys):
    motore, _flusso = finto

    def conta(evento):
        raise RuntimeError("statistiche guaste")

    suoni_ = Suoni(motore, conta=conta)
    assert isinstance(suoni_.suona("sparo"), VoceBuffer)
    assert isinstance(suoni_.suona("sparo"), VoceBuffer)
    assert suoni_.pronto("colpito", posizione=VICINO) is not None
    errori = capsys.readouterr().err
    assert errori.count("statistiche guaste") == 2


def test_preset_vuoto_o_guasto(fabbrica, finto, capsys):
    motore, flusso = finto
    fabbrica.vuoti.add(EVENTI["pausa"])
    fabbrica.guasti.add(EVENTI["ripresa"])
    contati = []
    suoni_ = Suoni(motore, conta=contati.append)
    for _volta in range(3):
        assert suoni_.suona("pausa") is None
        assert suoni_.suona("ripresa", posizione=VICINO) is None
        assert suoni_.pronto("pausa").shape == (0, 2)
        assert suoni_.pronto("ripresa", posizione=VICINO, guadagno_db=-3.0).shape == (0, 2)
    # Il preset guasto non si riprova, e il registro lo sente una volta sola.
    assert fabbrica.chiesti.count(EVENTI["ripresa"]) == 1
    assert fabbrica.chiesti.count(EVENTI["pausa"]) == 1
    errori = capsys.readouterr().err
    assert errori.count(EVENTI["ripresa"]) == 1 and "sintesi guasta" in errori
    assert contati == []
    assert suoni_.precarica() == len(EVENTI) - 2
    assert np.all(flusso.avanza(6) == 0.0)


def test_una_posizione_sbagliata_solleva(fabbrica):
    # Un punto che non ha tre coordinate è un errore di chi chiama, non un evento sconosciuto.
    with pytest.raises(ValueError):
        Suoni(None).pronto("colpito", posizione=(1.0, 2.0))


def _collezione_vera():
    """La collezione condivisa di GBUtils, letta soltanto."""
    with open(crea_preset.percorso_condiviso(), encoding="utf-8") as file:
        return json.load(file)


def test_ogni_preset_degli_eventi_e_nella_collezione_vera():
    collezione = _collezione_vera()
    mancano = sorted(preset for preset in set(EVENTI.values()) if preset not in collezione)
    assert not mancano, f"nella collezione vera mancano {len(mancano)} preset di Ciechino, {', '.join(mancano)}: si aggiungono con python strumenti\\crea_preset.py"


def test_ogni_preset_degli_eventi_porta_la_firma_di_ciechino():
    collezione = _collezione_vera()
    senza = sorted(preset for preset in set(EVENTI.values()) if not crea_preset.firmato(collezione.get(preset)))
    assert not senza, f"nella collezione vera {len(senza)} preset degli eventi non portano la firma {crea_preset.APP} in usato_da, {', '.join(senza)}: la mette python strumenti\\crea_preset.py"
