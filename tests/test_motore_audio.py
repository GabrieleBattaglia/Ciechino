"""
Le prove di motore_audio.py: voci al campione, attacco e fermate sfumate, tempo di gioco, eventi
programmati e annullati, campione udibile, tetto, notifiche per il registro, modo muto e guardia del
flusso perso, col flusso finto di aiuti_audio e senza mai aprire la scheda audio.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
"""

import importlib.util
import math
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
from aiuti_audio import LATENZA_FINTA, FlussoFinto, apri_finto, motore_finto

import motore_audio
from costanti import AUDIO_BLOCCO, AUDIO_LATENZA, FS, RAMPA_DEL_GUADAGNO, SFUMATURA_STOP, SOGLIA_DEL_TETTO, VELOCITA_DEL_TEMPO
from motore_audio import FLUSSO_FERMO_DOPO, Motore, Voce, VoceBuffer, VoceTono
from sintesi import Oscillatore, panoramica

BLOCCO = AUDIO_BLOCCO
# Quanto aspettano al più, in tempo vero, le prove che guardano il filo muto.
ATTESA_DEL_FILO = 2.0


@pytest.fixture
def finto():
    motore, flusso = motore_finto()
    yield motore, flusso
    motore.chiudi()


def _costante(valore, lunghezza=20_000):
    return np.full((lunghezza, 2), valore, dtype=np.float32)


def test_avvio_sul_flusso_finto(finto):
    motore, flusso = finto
    assert flusso.attivo
    assert motore.stato()["aperto"] and not motore.stato()["muto"]
    assert motore.stato()["interfaccia"] == "finta"
    assert motore.stato()["latenza"] == pytest.approx(LATENZA_FINTA)
    assert motore.campione_scritto == 0
    assert motore.prossimo_campione() == 2 * BLOCCO
    flusso.avanza(3)
    assert motore.campione_scritto == 3 * BLOCCO
    assert motore.prossimo_campione(1) == 4 * BLOCCO


def test_voce_parte_al_campione_esatto(finto):
    motore, flusso = finto
    voce = motore.suona(_costante(0.5), inizio=700)
    uscita = flusso.avanza(4)
    assert np.all(uscita[:700] == 0.0)
    assert np.all(uscita[700:] == 0.5)
    assert voce.inizio == 700 and not voce.finita


def test_voce_finisce_col_suo_buffer(finto):
    motore, flusso = finto
    voce = motore.suona(_costante(0.25, 1000), inizio=300)
    uscita = flusso.avanza(4)
    assert np.all(uscita[300:1300] == 0.25)
    assert np.all(uscita[1300:] == 0.0)
    assert voce.finita and voce.finita_al == 1300
    notizie = [n for n in motore.notifiche() if n.tipo == "voce_finita"]
    assert len(notizie) == 1 and notizie[0].oggetto is voce and notizie[0].campione == 1300
    assert motore.stato()["voci"] == 0


def test_voce_in_ritardo_parte_subito(finto):
    motore, flusso = finto
    flusso.avanza(2)
    motore.suona(_costante(0.5, 100), inizio=10)
    uscita = flusso.avanza(1)
    assert np.all(uscita[:100] == 0.5) and np.all(uscita[100:] == 0.0)


def test_fermata_con_sfumatura(finto):
    motore, flusso = finto
    voce = motore.suona(_costante(0.5), inizio=0)
    motore.ferma(voce, campione=1000)
    uscita = flusso.avanza(5)[:, 0]
    sfumatura = round(SFUMATURA_STOP * FS)
    assert np.all(uscita[:1000] == 0.5)
    attesa = 0.5 * (1000 + sfumatura - np.arange(1000, 1000 + sfumatura)) / sfumatura
    assert np.allclose(uscita[1000:1000 + sfumatura], attesa, atol=1e-6)
    assert np.all(np.diff(uscita[1000:1000 + sfumatura]) < 0)
    assert np.all(uscita[1000 + sfumatura:] == 0.0)
    assert voce.finita and voce.finita_al == 1000 + sfumatura


def test_fermata_subito_e_prima_dell_inizio(finto):
    motore, flusso = finto
    voce = motore.suona(_costante(0.5), inizio=0)
    flusso.avanza(1)
    motore.ferma(voce, sfumatura=0.0)
    assert np.all(flusso.avanza(1) == 0.0)
    assert voce.finita and voce.finita_al == BLOCCO
    muta = motore.suona(_costante(0.5), inizio=5000)
    motore.ferma(muta, campione=10)
    assert np.all(flusso.avanza(12) == 0.0)
    assert muta.finita


def test_guadagno_con_rampa(finto):
    motore, flusso = finto
    voce = motore.suona(_costante(0.5, 40_000), inizio=0)
    flusso.avanza(1)
    motore.guadagno_voce(voce, 0.5)
    rampa = round(RAMPA_DEL_GUADAGNO * FS)
    uscita = flusso.avanza(-(-rampa // BLOCCO) + 2)[:, 0]
    assert uscita[0] == pytest.approx(0.5, abs=1e-3)
    assert np.all(np.diff(uscita[:rampa]) <= 0)
    assert np.allclose(uscita[rampa:], 0.25)
    assert voce.guadagno == 0.5


def test_voce_tono_e_due_orologi(finto):
    motore, flusso = finto
    visti = {"gioco": [], "audio": []}

    def parametri(tempi_di_gioco, tempi_audio):
        visti["gioco"].append(np.array(tempi_di_gioco))
        visti["audio"].append(np.array(tempi_audio))
        return 440.0, 0.0, 0.5

    voce = VoceTono(parametri, "seno", 256)
    motore.aggiungi(voce)
    uscita = flusso.avanza(3)
    assert np.all(uscita[:256] == 0.0) and np.any(uscita[256:] != 0.0)
    assert np.max(np.abs(uscita)) <= 0.5 * math.sqrt(0.5) + 1e-6
    assert visti["audio"][0][0] == pytest.approx(256 / FS)
    assert len(visti["gioco"][0]) == BLOCCO - 256
    tutti = np.concatenate(visti["audio"])
    assert np.allclose(np.diff(tutti), 1.0 / FS)
    with pytest.raises(ValueError):
        VoceTono(parametri, "piffero", 0)


def test_voce_tono_comincia_con_l_attacco(finto):
    """Una VoceTono sale da zero in SFUMATURA_STOP, senza il click di un inviluppo che va dal silenzio
    al pieno in un campione; il resto del suono è quello dell'oscillatore."""
    motore, flusso = finto
    voce = VoceTono(lambda tempi_di_gioco, tempi_audio: (1000.0, 0.0, 0.5), "seno", 700)
    assert voce.attacco == SFUMATURA_STOP
    motore.aggiungi(voce)
    uscita = flusso.avanza(4)[:, 0]
    assert np.all(uscita[:700] == 0.0)
    campioni_d_attacco = round(SFUMATURA_STOP * FS)
    suono = panoramica(Oscillatore("seno", FS).rendi(np.full(len(uscita) - 700, 1000.0)) * 0.5, 0.0)[:, 0]
    rampa = np.clip(np.arange(len(suono)) / campioni_d_attacco, 0.0, 1.0)
    np.testing.assert_allclose(uscita[700:], suono * rampa, rtol=0, atol=1e-6)
    # L'inviluppo dell'attacco cresce da zero: nessun campione supera la sua rampa.
    picco = 0.5 * math.sqrt(0.5)
    assert np.all(np.abs(uscita[700:700 + campioni_d_attacco]) <= picco * np.arange(campioni_d_attacco) / campioni_d_attacco + 1e-6)
    assert np.max(np.abs(uscita[700 + campioni_d_attacco:])) == pytest.approx(picco, rel=1e-2)


def test_attacco_di_una_voce_in_ritardo_e_buffer_intatto(finto):
    """Una voce arrivata tardi comincia dal primo campione del blocco, e l'attacco con lei; un
    buffer già pronto non ha attacco e suona intatto al campione."""
    motore, flusso = finto
    flusso.avanza(2)
    voce = motore.aggiungi(VoceTono(lambda tempi_di_gioco, tempi_audio: (1000.0, 0.0, 0.5), "seno", 10))
    uscita = flusso.avanza(1)[:, 0]
    assert voce.inizio == 2 * BLOCCO and uscita[0] == 0.0
    campioni_d_attacco = round(SFUMATURA_STOP * FS)
    assert np.max(np.abs(uscita[:campioni_d_attacco // 4])) < 0.25 * 0.5 * math.sqrt(0.5) + 1e-6
    assert VoceBuffer(_costante(0.5, 10), 0).attacco == 0.0
    motore.ferma(voce, sfumatura=0.0)
    motore.suona(_costante(0.5, 100), inizio=motore.campione_scritto)
    uscita = flusso.avanza(1)
    assert np.all(uscita[:100] == 0.5) and np.all(uscita[100:] == 0.0)


def test_tempo_di_gioco_fermato_e_ripreso_esatto(finto):
    motore, flusso = finto
    assert motore.tempo_scorre
    assert motore.tempo_di_gioco(FS) == pytest.approx(VELOCITA_DEL_TEMPO)
    assert motore.ferma_tempo(campione=FS) == FS
    assert not motore.tempo_scorre
    assert motore.tempo_di_gioco(FS) == VELOCITA_DEL_TEMPO * 1.0
    assert motore.tempo_di_gioco(2 * FS) == VELOCITA_DEL_TEMPO * 1.0
    assert motore.fai_scorrere_tempo(campione=2 * FS) == 2 * FS
    assert motore.tempo_scorre
    assert motore.tempo_di_gioco(2 * FS) == pytest.approx(VELOCITA_DEL_TEMPO)
    assert motore.tempo_di_gioco(2 * FS + FS // 10) == pytest.approx(1.1 * VELOCITA_DEL_TEMPO)
    assert motore.tempo_di_gioco(FS // 2) == pytest.approx(0.5 * VELOCITA_DEL_TEMPO)
    visti = []

    def parametri(tempi_di_gioco, tempi_audio):
        visti.append(np.array(tempi_di_gioco))
        return 440.0, 0.0, 0.1

    motore.aggiungi(VoceTono(parametri, "seno", 0))
    flusso.avanza_fino(2 * FS + FS // 10)
    tempi = np.concatenate(visti)
    campioni = np.arange(len(tempi))
    attesi = np.where(campioni < FS, campioni / FS, 1.0)
    attesi = np.where(campioni >= 2 * FS, 1.0 + (campioni - 2 * FS) / FS, attesi) * VELOCITA_DEL_TEMPO
    assert np.allclose(tempi, attesi, atol=1e-12)
    assert np.all(tempi[FS:2 * FS] == tempi[FS])


def test_ancora_nuova_prima_di_una_futura(finto):
    motore, flusso = finto
    motore.ferma_tempo(campione=10_000)
    motore.ferma_tempo(campione=5000)
    assert motore.tempo_di_gioco(8000) == pytest.approx(5000 / FS * VELOCITA_DEL_TEMPO)
    flusso.avanza(4)
    # Il campione 100 è già suonato: l'ancora va a campione_scritto, e cancella la fermata futura.
    assert motore.fai_scorrere_tempo(campione=100) == 4 * BLOCCO
    assert motore.tempo_scorre
    assert motore.tempo_di_gioco(8000) == pytest.approx(8000 / FS * VELOCITA_DEL_TEMPO)


def test_evento_al_campione_esatto_con_voce_dallo_stesso_campione(finto):
    motore, flusso = finto
    tempo = 0.0301
    atteso = math.ceil(tempo / VELOCITA_DEL_TEMPO * FS)
    while atteso / FS * VELOCITA_DEL_TEMPO < tempo:
        atteso += 1
    while (atteso - 1) / FS * VELOCITA_DEL_TEMPO >= tempo:
        atteso -= 1
    scattati = []

    def all_impatto(campione):
        scattati.append(campione)
        motore.suona(_costante(0.5, 300), inizio=campione)

    evento = motore.programma(tempo, all_impatto)
    uscita = flusso.avanza_fino(atteso + 400)
    assert scattati == [atteso]
    assert evento.scattato and evento.campione == atteso
    assert np.all(uscita[:atteso] == 0.0)
    assert np.all(uscita[atteso:atteso + 300] == 0.5)
    assert np.all(uscita[atteso + 300:] == 0.0)
    notizie = motore.notifiche()
    assert [n.campione for n in notizie if n.tipo == "evento"] == [atteso]


def test_evento_che_ferma_una_voce_e_il_tempo(finto):
    motore, flusso = finto
    proiettile = motore.suona(_costante(0.3), inizio=0)

    def all_impatto(campione):
        motore.ferma(proiettile, campione, sfumatura=0.0)
        motore.ferma_tempo(campione)

    motore.programma(1000.5 / FS * VELOCITA_DEL_TEMPO, all_impatto)
    uscita = flusso.avanza(4)
    assert np.all(uscita[:1001] == 0.3)
    assert np.all(uscita[1001:] == 0.0)
    assert not motore.tempo_scorre
    assert motore.tempo_di_gioco(1500) == pytest.approx(1001 / FS * VELOCITA_DEL_TEMPO)


def test_evento_annullato_e_tempo_fermo(finto):
    motore, flusso = finto
    motore.ferma_tempo(campione=0)
    fermo = motore.programma(0.001, lambda campione: None)
    annullato = motore.programma(0.0, lambda campione: None)
    motore.annulla(annullato)
    flusso.avanza(10)
    assert not fermo.scattato and not annullato.scattato
    motore.fai_scorrere_tempo(campione=10 * BLOCCO)
    flusso.avanza(1)
    assert fermo.scattato and fermo.campione == 10 * BLOCCO + math.ceil(0.001 * FS / VELOCITA_DEL_TEMPO)
    assert not annullato.scattato


def test_evento_annullato_dopo_i_comandi_del_blocco_non_scatta(finto, monkeypatch):
    """Il filo principale prende il GIL dentro il callback, dopo i comandi d'inizio blocco, e annulla
    l'evento che sta per scattare: il comando arriverebbe al blocco seguente, ma l'evento non scatta."""
    motore, flusso = finto
    chiamate = []
    evento = motore.programma(0.0, chiamate.append)
    originale = motore._tempi_del_blocco

    def tempi_col_filo_principale_in_mezzo(campione, n):
        tempi = originale(campione, n)
        if evento in motore._eventi and not evento.annullato:
            motore.annulla(evento)
        return tempi

    monkeypatch.setattr(motore, "_tempi_del_blocco", tempi_col_filo_principale_in_mezzo)
    flusso.avanza(3)
    assert evento.annullato and not evento.scattato and evento.campione is None
    assert chiamate == []
    assert not [n for n in motore.notifiche() if n.tipo in ("evento", "errore")]


def test_campione_udibile_col_flusso_finto(finto):
    motore, flusso = finto
    assert motore.campione_udibile() == 0.0
    flusso.avanza(3)
    atteso = 3 * BLOCCO - LATENZA_FINTA * FS
    assert motore.campione_udibile() == pytest.approx(atteso)
    flusso.time += 0.005
    assert motore.campione_udibile() == pytest.approx(atteso + 0.005 * FS)
    flusso.time += 1.0
    assert motore.campione_udibile() == motore.campione_scritto
    assert motore.tempo_di_gioco() == pytest.approx(motore.campione_scritto / FS * VELOCITA_DEL_TEMPO)


def test_campione_udibile_senza_tempo_del_dac(finto):
    motore, flusso = finto
    flusso.dac_zero = True
    flusso.avanza(3)
    assert motore.campione_udibile() == pytest.approx(3 * BLOCCO - LATENZA_FINTA * FS)


def test_tetto():
    assert SOGLIA_DEL_TETTO < 1.0
    campioni = np.array([[0.0, 0.5], [SOGLIA_DEL_TETTO, -SOGLIA_DEL_TETTO], [1.2, -1.6], [40.0, -1e9]], dtype=np.float32)
    piegati = motore_audio.tetto(campioni.copy())
    assert np.all(piegati[:2] == campioni[:2])
    assert np.all(np.abs(piegati) <= 1.0)
    assert SOGLIA_DEL_TETTO < piegati[2, 0] < 1.0 and -1.0 < piegati[2, 1] < -SOGLIA_DEL_TETTO
    assert piegati[2, 1] < -piegati[2, 0]


def test_tetto_sulla_somma(finto):
    motore, flusso = finto
    motore.suona(_costante(0.7), inizio=0)
    motore.suona(_costante(0.7), inizio=0)
    uscita = flusso.avanza(2)
    assert np.all(uscita < 1.0) and np.all(uscita > SOGLIA_DEL_TETTO)


def test_volume_e_silenzio(finto):
    motore, flusso = finto
    motore.suona(_costante(0.5, 100_000), inizio=0)
    flusso.avanza(1)
    motore.volume = 0.5
    rampa = round(RAMPA_DEL_GUADAGNO * FS)
    blocchi = -(-rampa // BLOCCO) + 1
    uscita = flusso.avanza(blocchi)
    assert uscita[-1, 0] == pytest.approx(0.25)
    assert motore.volume == 0.5
    motore.silenzia(True)
    assert motore.silenziato
    uscita = flusso.avanza(blocchi)
    assert np.all(uscita[-BLOCCO:] == 0.0)
    motore.silenzia(False)
    uscita = flusso.avanza(blocchi)
    assert uscita[-1, 0] == pytest.approx(0.25)


def test_notifiche_dei_buchi(finto):
    motore, flusso = finto
    flusso.avanza(1, buco=True)
    flusso.avanza(1)
    flusso.avanza(1, buco=True)
    notizie = motore.notifiche()
    # Ogni buco porta il conto dall'avvio, lo stesso dello stato.
    assert [(n.tipo, n.campione, n.volte) for n in notizie] == [("buco", 0, 1), ("buco", 2 * BLOCCO, 2)]
    assert motore.stato()["buchi"] == 2
    assert notizie[1].riga() == f"Buco nell'audio al campione {2 * BLOCCO}: la scheda è rimasta senza campioni, 2 volte dall'avvio."
    assert motore.notifiche() == []


class VoceGuasta(Voce):
    def _produci(self, campione, quanti, tempi_di_gioco, tempi_audio):
        raise RuntimeError("guasto voluto")


class VoceNonFinita(Voce):
    def _produci(self, campione, quanti, tempi_di_gioco, tempi_audio):
        return np.full((quanti, 2), np.nan, dtype=np.float32)


def test_voce_che_solleva_si_toglie_e_si_notifica(finto):
    motore, flusso = finto
    buona = motore.suona(_costante(0.5), inizio=0)
    guasta = motore.aggiungi(VoceGuasta(0))
    nan = motore.aggiungi(VoceNonFinita(0))
    uscita = flusso.avanza(2)
    assert np.all(uscita == 0.5)
    errori = [n for n in motore.notifiche() if n.tipo == "errore"]
    assert {id(n.oggetto) for n in errori} == {id(guasta), id(nan)}
    assert all(n.campione == 0 for n in errori)
    assert "guasto voluto" in next(n.testo for n in errori if n.oggetto is guasta)
    assert guasta.finita and nan.finita and not buona.finita
    assert motore.stato()["voci"] == 1


def test_evento_che_solleva(finto):
    motore, flusso = finto

    def guasto(campione):
        raise RuntimeError("evento guasto")

    motore.programma(0.0, guasto)
    motore.suona(_costante(0.5, 100), inizio=0)
    uscita = flusso.avanza(1)
    assert np.all(uscita[:100] == 0.5)
    tipi = [n.tipo for n in motore.notifiche()]
    assert "errore" in tipi and "evento" in tipi


def test_comando_sbagliato_non_ferma_il_callback(finto):
    motore, flusso = finto
    motore.ferma("non sono una voce")
    motore.suona(_costante(0.5, 100), inizio=0)
    uscita = flusso.avanza(1)
    assert np.all(uscita[:100] == 0.5)
    errori = [n for n in motore.notifiche() if n.tipo == "errore"]
    assert len(errori) == 1 and errori[0].oggetto == "non sono una voce"
    assert errori[0].origine == "comando ferma" and errori[0].testo.startswith("AttributeError: ")
    assert errori[0].riga().startswith("Errore nel filo dell'audio al campione 0 (comando ferma, str): AttributeError: ")


# Le notifiche degli errori, con ciò che serve al registro.

def test_errore_di_una_voce_porta_origine_volte_e_pila(finto):
    """Un guasto nei parametri di una voce: la prima notifica porta la pila, le seguenti il conto."""
    motore, flusso = finto

    def parametri_rotti(tempi_di_gioco, tempi_audio):
        raise ValueError("parametri rotti")

    voci = []
    for numero in range(3):
        voci.append(motore.aggiungi(VoceTono(parametri_rotti, "seno", numero * BLOCCO)))
        flusso.avanza(1)
    errori = [n for n in motore.notifiche() if n.tipo == "errore"]
    assert [n.oggetto for n in errori] == voci and all(voce.finita for voce in voci)
    assert [(n.origine, n.testo, n.volte) for n in errori] == [("voce", "ValueError: parametri rotti", volte) for volte in (1, 2, 3)]
    assert errori[0].pila is not None and errori[1].pila is None and errori[2].pila is None
    prima = errori[0].riga()
    assert prima.startswith("Errore nel filo dell'audio al campione 0 (voce, VoceTono di ")
    assert "parametri_rotti, dal campione 0): ValueError: parametri rotti\nPila delle chiamate:\n" in prima
    # Le righe del sorgente si leggono quando la riga si scrive, nel filo principale.
    assert 'raise ValueError("parametri rotti")' in prima and "_produci" in prima
    assert errori[2].riga().endswith("ValueError: parametri rotti, 3 volte dall'avvio")
    assert "Pila" not in errori[2].riga()


def test_errore_di_un_evento_e_del_blocco_intero(finto, monkeypatch):
    motore, flusso = finto

    def evento_guasto(campione):
        raise RuntimeError("evento guasto")

    motore.programma(0.0, evento_guasto)
    flusso.avanza(1)
    errori = [n for n in motore.notifiche() if n.tipo == "errore"]
    assert len(errori) == 1 and errori[0].origine == "evento" and errori[0].volte == 1
    assert "evento_guasto al tempo di gioco 0.000): RuntimeError: evento guasto" in errori[0].riga()

    def esegui_rotto(inizio):
        raise RuntimeError("blocco rotto")

    # Un guasto fuori dalle voci: il blocco intero esce muto, e si ripete a ogni blocco.
    monkeypatch.setattr(motore, "_esegui_comandi", esegui_rotto)
    motore.suona(_costante(0.5, 10_000), inizio=motore.campione_scritto)
    assert np.all(flusso.avanza(3) == 0.0)
    blocchi = [n for n in motore.notifiche() if n.tipo == "errore"]
    assert [(n.origine, n.volte, n.pila is not None, n.oggetto) for n in blocchi] == [("blocco", 1, True, None), ("blocco", 2, False, None), ("blocco", 3, False, None)]
    assert blocchi[0].riga().startswith(f"Errore nel filo dell'audio al campione {BLOCCO} (blocco): RuntimeError: blocco rotto\nPila delle chiamate:\n")
    assert motore.campione_scritto == 4 * BLOCCO


def test_ferma_tutte(finto):
    motore, flusso = finto
    voci = [motore.suona(_costante(0.2), inizio=0) for _ in range(3)]
    flusso.avanza(1)
    motore.ferma_tutte()
    flusso.avanza(2)
    assert all(voce.finita for voce in voci)
    assert motore.stato()["voci"] == 0


def test_buffer_mono_al_centro():
    voce = VoceBuffer(np.ones(10, dtype=np.float32), 0)
    assert voce.buffer.shape == (10, 2)
    assert np.allclose(voce.buffer, math.sqrt(0.5))
    with pytest.raises(ValueError):
        VoceBuffer(np.ones((10, 3)), 0)


def test_chiudi_idempotente():
    motore, flusso = motore_finto()
    motore.chiudi()
    motore.chiudi()
    assert flusso.chiuso and motore.chiuso
    assert not motore.stato()["aperto"]
    assert not motore.avvia()


def test_senza_flusso_lavora_muto_davvero():
    """Il conftest fa rispondere None ad apri_flusso: il motore va col suo filo, al passo del tempo vero."""
    motore = Motore()
    try:
        assert motore.avvia() is False
        assert motore.stato()["muto"] and not motore.stato()["aperto"]
        voce = motore.suona(_costante(0.5, 200))
        evento = motore.programma(0.01, lambda campione: None)
        partenza = time.perf_counter()
        while time.perf_counter() - partenza < 2.0 and not (voce.finita and evento.scattato and motore.campione_scritto > 0.04 * FS):
            time.sleep(0.005)
        trascorso = time.perf_counter() - partenza
        assert voce.finita and evento.scattato
        assert motore.campione_scritto <= (trascorso + 0.1) * FS + 2 * BLOCCO
        udibile = motore.campione_udibile()
        assert 0.0 < udibile <= motore.campione_scritto
        tipi = {n.tipo for n in motore.notifiche()}
        assert {"voce_finita", "evento"} <= tipi
        filo = motore._filo_muto
        assert filo is not None and filo.is_alive() and filo.daemon
    finally:
        motore.chiudi()
    assert not motore._filo_muto.is_alive()
    fermo = motore.campione_scritto
    time.sleep(0.03)
    assert motore.campione_scritto == fermo


def test_apri_che_solleva_va_muto():
    def rotto(callback, fs, blocco):
        raise OSError("niente scheda")

    motore = Motore(apri=rotto)
    try:
        assert motore.avvia() is False
        assert motore.muto and motore.ultimo_errore.endswith("apertura: OSError: niente scheda")
        assert motore.stato()["errore"] == motore.ultimo_errore
    finally:
        motore.chiudi()


def test_avvio_muto_dice_perche(monkeypatch):
    """Il perché che apri_flusso lascia nel modulo finisce in ultimo_errore, per il registro; senza
    un perché, come con l'apertura finta del conftest, lo dice una frase generica."""
    monkeypatch.setattr(motore_audio, "ULTIMO_ERRORE_DI_APERTURA", "")
    motore = Motore()
    try:
        assert motore.avvia() is False
        assert motore.ultimo_errore.endswith("apertura: nessuna scheda audio si è aperta")
    finally:
        motore.chiudi()

    def apri(callback, fs, blocco):
        motore_audio.ULTIMO_ERRORE_DI_APERTURA = "dispositivo predefinito, OSError: occupato"

    monkeypatch.setattr(motore_audio, "apri_flusso", apri)
    motore = Motore()
    try:
        assert motore.avvia() is False
        assert motore.ultimo_errore.endswith("apertura: dispositivo predefinito, OSError: occupato")
    finally:
        motore.chiudi()


# La guardia del flusso perso.

def _aspetta(condizione):
    """Aspetta in tempo vero, al più ATTESA_DEL_FILO, che la condizione diventi vera."""
    scadenza = time.perf_counter() + ATTESA_DEL_FILO
    while not condizione() and time.perf_counter() < scadenza:
        time.sleep(0.005)
    return condizione()


def test_flusso_non_piu_attivo_passa_al_muto():
    """La cuffia si stacca e PortAudio dà il flusso per non più attivo: al primo controllo il motore
    lo lascia e va avanti muto dal campione scritto, con le voci, il tempo di gioco e gli eventi."""
    motore, flusso = motore_finto()
    try:
        voce = motore.suona(_costante(0.5, 10 * FS), inizio=0)
        flusso.avanza(10)
        assert motore.controlla() is False and not motore.muto
        flusso.active = False
        assert motore.controlla() is True
        assert motore.muto and motore.flusso_perso and motore.flusso is None
        stato = motore.stato()
        assert not stato["aperto"] and stato["muto"] and stato["perso"] and stato["interfaccia"] == "finta"
        assert motore.ultimo_errore == "flusso perso: PortAudio dice che il flusso non è più attivo, sulla scheda finta"
        perse = [n for n in motore.notifiche() if n.tipo == "flusso_perso"]
        assert len(perse) == 1 and perse[0].campione == 10 * BLOCCO
        assert perse[0].riga() == f"Flusso audio perso al campione {10 * BLOCCO}: PortAudio dice che il flusso non è più attivo, sulla scheda finta. Ciechino continua senza suono."
        # Il flusso perso si chiude in un filo a parte.
        motore._filo_della_chiusura.join(ATTESA_DEL_FILO)
        assert flusso.chiuso
        # Il filo muto riparte dal campione scritto: il campione udibile si muove e gli eventi scattano.
        evento = motore.programma(motore.tempo_di_gioco(motore.campione_scritto) + 0.02, lambda campione: None)
        assert _aspetta(lambda: motore.campione_udibile() > 10 * BLOCCO and evento.scattato)
        assert evento.campione > 10 * BLOCCO and not voce.finita
        # Un callback del flusso perso che si rifacesse vivo scrive silenzio e non tocca niente.
        assert np.all(flusso.avanza(2) == 0.0)
        assert motore.controlla() is False
        assert not [n for n in motore.notifiche() if n.tipo in ("errore", "flusso_perso")]
    finally:
        motore.chiudi()
    assert not motore._filo_muto.is_alive()


def test_callback_che_tace_passa_al_muto(finto):
    """Il callback smette di arrivare mentre l'orologio del flusso va avanti, fra un controllo e
    l'altro: oltre FLUSSO_FERMO_DOPO il flusso è perso, e il campione udibile riparte."""
    motore, flusso = finto
    flusso.avanza(10)
    fermo = motore.campione_udibile()
    passo = 0.6 * FLUSSO_FERMO_DOPO
    assert motore.controlla() is False
    flusso.time += passo
    assert motore.controlla() is False
    # Senza la guardia il campione udibile resterebbe qui per sempre.
    assert motore.campione_udibile() == motore.campione_scritto >= fermo
    flusso.time += passo
    assert motore.controlla() is True
    silenzio = f"{2 * passo:.2f}".replace(".", ",")
    assert motore.muto and motore.ultimo_errore == f"flusso perso: il callback non arriva da {silenzio} secondi, sulla scheda finta"
    assert [n.tipo for n in motore.notifiche()] == ["flusso_perso"]
    assert _aspetta(lambda: motore.campione_udibile() > 10 * BLOCCO)


def test_callback_vivo_e_risveglio_non_perdono_il_flusso(finto):
    """Un flusso che chiede i suoi blocchi non si perde mai; un controllo che arriva dopo un lungo
    sonno del filo principale, per un sistema sospeso, non giudica, e dà al callback altro tempo."""
    motore, flusso = finto
    for _blocco in range(round(4 * FLUSSO_FERMO_DOPO * FS / BLOCCO)):
        flusso.avanza(1)
        assert motore.controlla() is False
    passo = 0.6 * FLUSSO_FERMO_DOPO
    flusso.time += 10 * FLUSSO_FERMO_DOPO
    assert motore.controlla() is False
    flusso.time += passo
    assert motore.controlla() is False
    # Il callback torna, e il conto del silenzio riparte da lì.
    flusso.avanza(1)
    flusso.time += passo
    assert motore.controlla() is False
    flusso.time += passo
    assert motore.controlla() is False
    # Un orologio che torna indietro non è un silenzio.
    flusso.time -= 4 * passo
    assert motore.controlla() is False
    assert not motore.muto and motore.flusso is flusso
    flusso.time += passo
    assert motore.controlla() is False
    flusso.time += passo
    assert motore.controlla() is True


class _FlussoSenzaStato(FlussoFinto):
    """Un flusso il cui stato non si legge, come un flusso guasto di PortAudio."""

    @property
    def active(self):
        raise RuntimeError("Error checking stream state")


def test_stato_che_non_si_legge_e_motore_muto_o_chiuso():
    motore = Motore(apri=lambda callback, fs, blocco: _FlussoSenzaStato(callback, fs, blocco))
    try:
        assert motore.avvia() is True
        assert motore.controlla() is True
        assert motore.ultimo_errore == "flusso perso: lo stato del flusso non si legge, RuntimeError: Error checking stream state, sulla scheda finta"
    finally:
        motore.chiudi()
    # Un motore muto dall'avvio, uno chiuso e uno mai avviato non hanno un flusso da perdere.
    muto = Motore()
    try:
        muto.avvia()
        assert muto.controlla() is False and not muto.flusso_perso
    finally:
        muto.chiudi()
    assert muto.controlla() is False
    assert Motore().controlla() is False


def test_apri_cercato_al_momento_di_avvia(monkeypatch):
    aperti = []

    def apri(callback, fs, blocco):
        aperti.append((fs, blocco))
        return apri_finto(callback, fs, blocco)

    motore = Motore(fs=48000, blocco=256)
    monkeypatch.setattr(motore_audio, "apri_flusso", apri)
    try:
        assert motore.avvia() is True
        assert aperti == [(48000, 256)]
    finally:
        motore.chiudi()


# apri_flusso vera, con un sounddevice finto: il conftest sostituisce quella del modulo, quindi qui
# se ne carica una copia fresca dal file.

class _FlussoDiProva:
    def __init__(self, registro, fallisce, **opzioni):
        if fallisce(opzioni):
            raise RuntimeError("il dispositivo non si apre")
        self.opzioni = opzioni
        self.avviato = False
        registro.append(self)

    def start(self):
        self.avviato = True

    def close(self):
        pass


def _sounddevice_finto(fallisce):
    modulo = types.ModuleType("sounddevice")
    modulo.aperti = []

    class WasapiSettings:
        def __init__(self, auto_convert=False):
            self.auto_convert = auto_convert

    modulo.WasapiSettings = WasapiSettings
    modulo.OutputStream = lambda **opzioni: _FlussoDiProva(modulo.aperti, fallisce, **opzioni)
    return modulo


def _apri_flusso_vera():
    percorso = Path(motore_audio.__file__)
    specifica = importlib.util.spec_from_file_location("motore_audio_copia_di_prova", percorso)
    copia = importlib.util.module_from_spec(specifica)
    specifica.loader.exec_module(copia)
    return copia.apri_flusso


@pytest.mark.parametrize(("scelta", "fallisce", "atteso", "perche"), [
    ((7, "Windows WASAPI"), lambda o: False, (7, True), ""),
    ((3, "Windows DirectSound"), lambda o: False, (3, False), ""),
    ((7, "Windows WASAPI"), lambda o: o["device"] == 7, (None, False), "dispositivo 7 di Windows WASAPI, RuntimeError: il dispositivo non si apre"),
    (RuntimeError("niente scelta"), lambda o: False, (None, False), "scelta del dispositivo, RuntimeError: niente scelta"),
])
def test_apri_flusso_sceglie_e_ripiega(monkeypatch, scelta, fallisce, atteso, perche):
    import GBUtils

    finto = _sounddevice_finto(fallisce)
    monkeypatch.setitem(sys.modules, "sounddevice", finto)
    chieste = []

    def scegli(**opzioni):
        chieste.append(opzioni)
        if isinstance(scelta, Exception):
            raise scelta
        return scelta

    monkeypatch.setattr(GBUtils, "scegli_dispositivo_audio", scegli)
    apri_flusso = _apri_flusso_vera()
    callback = object()
    flusso = apri_flusso(callback, FS, BLOCCO)
    assert chieste == [{"modo": "callback"}]
    assert flusso is finto.aperti[-1] and flusso.avviato
    dispositivo, wasapi = atteso
    opzioni = flusso.opzioni
    assert opzioni["device"] == dispositivo
    assert (opzioni["extra_settings"] is not None and opzioni["extra_settings"].auto_convert) == wasapi
    assert opzioni["callback"] is callback
    assert opzioni["samplerate"] == FS and opzioni["blocksize"] == BLOCCO
    assert opzioni["channels"] == 2 and opzioni["dtype"] == "float32" and opzioni["latency"] == AUDIO_LATENZA
    # Il perché dei tentativi falliti resta nel modulo, per il registro; vuoto se il primo è riuscito.
    assert apri_flusso.__globals__["ULTIMO_ERRORE_DI_APERTURA"] == perche


def test_apri_flusso_senza_niente(monkeypatch):
    import GBUtils

    monkeypatch.setitem(sys.modules, "sounddevice", _sounddevice_finto(lambda o: True))
    monkeypatch.setattr(GBUtils, "scegli_dispositivo_audio", lambda **opzioni: (7, "Windows WASAPI"))
    apri_flusso = _apri_flusso_vera()
    assert apri_flusso(object(), FS, BLOCCO) is None
    assert apri_flusso.__globals__["ULTIMO_ERRORE_DI_APERTURA"] == ("dispositivo 7 di Windows WASAPI, RuntimeError: il dispositivo non si apre; "
                                                                    "dispositivo predefinito, RuntimeError: il dispositivo non si apre")


def test_comandi_da_un_altro_filo(finto):
    """Il filo principale manda comandi mentre il callback gira: niente si perde e niente solleva."""
    motore, flusso = finto
    voci = []

    def manda():
        for _ in range(200):
            voci.append(motore.suona(_costante(0.001, 50)))

    filo = threading.Thread(target=manda)
    filo.start()
    while filo.is_alive():
        flusso.avanza(1)
    filo.join()
    flusso.avanza(4)
    assert all(voce.finita for voce in voci)
    assert not [n for n in motore.notifiche() if n.tipo == "errore"]
