"""
Le prove degli Ascolti di ascolti.py (D23): i passi coi loro testi, e i loro suoni sul motore col flusso finto di
aiuti_audio, mai con la scheda vera.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-20 con la 0.9.0. I suoni si guardano in due modi: coi parametri delle voci dei bersagli, chiesti
negli istanti che contano, e con ciò che il motore scrive, blocco per blocco, sul flusso finto. Nessun preset vero si
sintetizza: _rendi_preset dà a tutti lo stesso buffer finto, e annota chi lo chiede.
"""

import itertools
import random
import re

import numpy as np
import pytest
from aiuti_audio import LATENZA_FINTA, motore_finto

import ascolti
import fisica
import suoni
import testi
from bersagli import Bersaglio, durata_del_bonus, mezza_escursione_ottave, portanti_senza
from costanti import (
    ASCOLTI_DURATA_MASSIMA,
    ASCOLTI_GRANDEZZE,
    ASCOLTI_LIVELLO_DEL_BONUS,
    ASCOLTI_PAUSA,
    ASCOLTI_PUNTI,
    ASCOLTI_RAGGIO,
    ASCOLTI_STEREO_FORMA,
    ASCOLTI_STEREO_SOSTA,
    ASCOLTI_STEREO_VIAGGIO,
    ASCOLTI_SUONO,
    BERSAGLIO_DB,
    BERSAGLIO_PORTANTI,
    FORME_LFO,
    MARGINE_DOPO_LA_SCIVOLATA,
    PARAMETRI,
    PAUSA_FRA_BARRE,
    PROIETTILE_FORME,
)
from motore_audio import VoceBuffer, VoceTono
from sintesi import scivolata
from statistiche import NOMI_DELLE_FORME, NOMI_DELLE_PORTANTI

# Il buffer finto di ogni preset: corto, al centro, ben sotto il tetto del motore.
CAMPIONI_FINTI = 2_000
SEPARATORE = re.compile(r"-{3,}|={3,}|_{3,}")
NOMI_DEI_PASSI = ["stereo", "orizzontale", "altezza", "distanza", "punti", "grandezza", "forme", "forme_2", "timbri", "mira", "volo", "colpi", "bonus"]
# Un millesimo di secondo, per prendere un istante appena dentro una sosta.
UN_MILLESIMO = 0.001


@pytest.fixture
def finto():
    motore, flusso = motore_finto()
    yield motore, flusso
    motore.chiudi()


@pytest.fixture
def chiesti(monkeypatch):
    """I preset che i passi chiedono, in ordine: tutti suonano lo stesso buffer finto."""
    nomi = []

    def rendi(nome):
        nomi.append(nome)
        buffer = np.full((CAMPIONI_FINTI, 2), 0.1, dtype=np.float32)
        buffer.flags.writeable = False
        return buffer

    monkeypatch.setattr(suoni, "_rendi_preset", rendi)
    return nomi


def _regia(motore, proiettile="seno", **opzioni):
    return ascolti.Regia(motore, suoni.Suoni(motore), proiettile, rng=random.Random(7), **opzioni)


def _indice(nome):
    return NOMI_DEI_PASSI.index(nome)


def _parametri(voce, secondi):
    """(frequenze, pan, guadagni) di una VoceTono negli istanti dati, in secondi dal suo inizio."""
    tempi = voce.inizio / voce.fs + np.atleast_1d(np.asarray(secondi, dtype=float))
    return voce.parametri(np.zeros_like(tempi), tempi)


def _nota_del_centro(bersaglio, voce, secondi):
    """La frequenza della nota del centro del bersaglio, cioè senza l'oscillazione dell'LFO."""
    frequenze, _pan, _guadagni = _parametri(voce, secondi)
    return frequenze / 2.0 ** (np.asarray(bersaglio.lfo(np.atleast_1d(secondi))) * bersaglio.mezza_escursione_ottave())


def _udito(motore, flusso, campione):
    """Fa scrivere al flusso finto abbastanza blocchi perché il campione sia già uscito dalle casse."""
    flusso.avanza_fino(campione + round(LATENZA_FINTA * motore.fs) + motore.blocco)
    assert motore.campione_udibile() >= campione


# I passi e i testi.

def test_i_passi_coi_titoli_i_testi_e_i_suoni():
    passi = ascolti.passi("seno")
    assert [passo.nome for passo in passi] == NOMI_DEI_PASSI
    for passo in passi:
        assert passo.titolo.strip() and not passo.titolo.endswith("."), passo.nome
        righe = passo.testo.splitlines()
        assert righe and all(riga.strip() for riga in righe), passo.nome
        assert "{" not in passo.testo and "}" not in passo.testo and not SEPARATORE.search(passo.testo), passo.nome
        # I numeri all'italiana: mai un punto fra due cifre seguito da meno di tre cifre, come 2.5.
        assert not re.search(r"\d\.\d{1,2}(?!\d)", passo.testo), passo.nome
        assert callable(passo.costruisci)
    testi_dei_passi = {passo.nome: passo.testo for passo in passi}
    # Il passo della distanza dice come si ritara l'orecchio quando cambia il volume.
    assert "ritararlo" in testi_dei_passi["distanza"] and "volume" in testi_dei_passi["distanza"]
    # Il test stereo dice che cosa fare se è al contrario.
    assert "al contrario" in testi_dei_passi["stereo"]
    # Le forme e i timbri, ognuno col suo nome, nell'ordine in cui suonano.
    forme = testi_dei_passi["forme"] + testi_dei_passi["forme_2"]
    posti = [forme.index(NOMI_DELLE_FORME[forma]) for forma in FORME_LFO]
    assert posti == sorted(posti)
    assert f"{testi.numero(ASCOLTI_SUONO, 1)} secondi" in testi_dei_passi["forme"]
    for portante in portanti_senza("seno"):
        assert NOMI_DELLE_PORTANTI[portante] in testi_dei_passi["timbri"]


def test_il_proiettile_sconosciuto_non_ha_ascolti(finto):
    motore, _flusso = finto
    with pytest.raises(ValueError, match="proiettile"):
        ascolti.Regia(motore, None, "organo")


def test_ogni_passo_suona_e_dura_poco(finto, chiesti):
    """Ogni passo mette le sue voci nel motore e dura al massimo ASCOLTI_DURATA_MASSIMA: oltre, si divide in due."""
    motore, _flusso = finto
    regia = _regia(motore)
    for indice, passo in enumerate(regia.passi):
        ascolto = regia.suona(indice)
        assert ascolto.passo == passo.nome and ascolto.voci and regia.in_corso is ascolto
        assert 0 < (ascolto.fine - ascolto.inizio) / motore.fs <= ASCOLTI_DURATA_MASSIMA, passo.nome
        assert all(isinstance(voce, VoceTono | VoceBuffer) for voce in ascolto.voci)


def test_il_test_stereo_va_da_tutto_a_sinistra_a_tutto_a_destra(finto):
    motore, flusso = finto
    regia = ascolti.Regia(motore)
    ascolto = regia.suona(_indice("stereo"))
    (voce,) = ascolto.voci
    assert voce.forma == ASCOLTI_STEREO_FORMA
    secondi = np.linspace(0.0, 2 * ASCOLTI_STEREO_SOSTA + ASCOLTI_STEREO_VIAGGIO, 400)
    frequenze, pan, _guadagni = _parametri(voce, secondi)
    assert np.all(np.diff(pan) >= 0.0) and len(set(frequenze.tolist())) == 1
    assert _parametri(voce, ASCOLTI_STEREO_SOSTA / 2)[1][0] == -1.0
    assert _parametri(voce, ASCOLTI_STEREO_SOSTA + ASCOLTI_STEREO_VIAGGIO / 2)[1][0] == pytest.approx(0.0)
    assert _parametri(voce, ASCOLTI_STEREO_SOSTA * 1.5 + ASCOLTI_STEREO_VIAGGIO)[1][0] == 1.0
    # Quel che esce: all'inizio soltanto l'orecchio sinistro, alla fine soltanto il destro.
    _udito(motore, flusso, ascolto.fine)
    uscita = flusso.uscita
    fs = motore.fs
    inizio = uscita[ascolto.inizio + round(0.1 * fs):ascolto.inizio + round(0.9 * ASCOLTI_STEREO_SOSTA * fs)]
    fine = uscita[ascolto.fine - round(0.9 * ASCOLTI_STEREO_SOSTA * fs):ascolto.fine - round(0.1 * fs)]
    assert np.abs(inizio[:, 0]).max() > 0.1 and np.abs(inizio[:, 1]).max() < 1e-6
    assert np.abs(fine[:, 1]).max() > 0.1 and np.abs(fine[:, 0]).max() < 1e-6
    # Finito, tace.
    assert np.abs(uscita[ascolto.fine:]).max() == 0.0


def _tappe(ascolto):
    """I punti delle tappe di una tratta, negli istanti d'arrivo, e quegli istanti, appena dentro la sosta."""
    istanti = np.array(ascolto.percorso.arrivi) + UN_MILLESIMO
    return ascolto.percorso(istanti), istanti


def test_la_tratta_orizzontale_dal_centro_a_sinistra_a_destra_e_al_centro(finto):
    motore, _flusso = finto
    ascolto = ascolti.Regia(motore).suona(_indice("orizzontale"))
    (voce,), (bersaglio,) = ascolto.voci, ascolto.bersagli
    minimi, massimi = fisica.zona_del_centro(ASCOLTI_RAGGIO)
    punti, istanti = _tappe(ascolto)
    centro = np.array(ascolti.CENTRO)
    attesi = [centro, (minimi[0], centro[1], centro[2]), (massimi[0], centro[1], centro[2]), centro]
    assert np.allclose(punti, attesi)
    _frequenze, pan, guadagni = _parametri(voce, istanti)
    assert np.allclose(pan, [fisica.pan_di(x, z) for x, _y, z in attesi])
    assert pan[1] < 0.0 < pan[2] and pan[0] == pytest.approx(0.0)
    # Su tutta la tratta la panoramica non esce dai capi, e la nota del centro resta quella dell'altezza del centro.
    secondi = np.linspace(0.0, ascolto.percorso.durata, 2000)
    tutte = _parametri(voce, secondi)[1]
    assert tutte.min() == pytest.approx(fisica.pan_di(minimi[0], centro[2])) and tutte.max() == pytest.approx(fisica.pan_di(massimi[0], centro[2]))
    assert np.allclose(_nota_del_centro(bersaglio, voce, secondi), fisica.frequenza_di(centro[1]))
    # Il guadagno è quello della distanza, con BERSAGLIO_DB.
    assert guadagni[0] == pytest.approx(fisica.guadagno_di(fisica.distanza_dal_cannone(centro)) * 10 ** (BERSAGLIO_DB / 20))


def test_l_altezza_a_terra_e_al_soffitto_col_minimo_e_il_massimo_della_nota(finto):
    motore, _flusso = finto
    ascolto = ascolti.Regia(motore).suona(_indice("altezza"))
    (voce,), (bersaglio,) = ascolto.voci, ascolto.bersagli
    minimi, massimi = fisica.zona_del_centro(ASCOLTI_RAGGIO)
    punti, istanti = _tappe(ascolto)
    assert punti[1][1] == pytest.approx(minimi[1]) and punti[2][1] == pytest.approx(massimi[1])
    assert np.allclose(_nota_del_centro(bersaglio, voce, istanti), [fisica.frequenza_di(y) for y in punti[:, 1]])
    secondi = np.arange(0.0, ascolto.percorso.durata, 1.0 / motore.fs)
    frequenze, pan, _guadagni = _parametri(voce, secondi)
    mezza = bersaglio.mezza_escursione_ottave()
    # La nota più grave e la più acuta: quella di terra e quella del soffitto, ciascuna in fondo e in cima all'oscillazione.
    assert frequenze.min() == pytest.approx(fisica.frequenza_di(minimi[1]) * 2.0 ** -mezza, rel=1e-4)
    assert frequenze.max() == pytest.approx(fisica.frequenza_di(massimi[1]) * 2.0 ** mezza, rel=1e-4)
    assert np.allclose(pan, 0.0)


def test_la_distanza_vicino_e_lontano_cambia_soltanto_il_volume(finto):
    motore, _flusso = finto
    ascolto = ascolti.Regia(motore).suona(_indice("distanza"))
    (voce,), (bersaglio,) = ascolto.voci, ascolto.bersagli
    minimi, massimi = fisica.zona_del_centro(ASCOLTI_RAGGIO)
    punti, istanti = _tappe(ascolto)
    vicino, lontano = punti[1], punti[2]
    assert vicino[2] == pytest.approx(minimi[2]) and lontano[2] == pytest.approx(massimi[2])
    _frequenze, pan, guadagni = _parametri(voce, istanti)
    pieno = 10 ** (BERSAGLIO_DB / 20)
    assert guadagni[1] == pytest.approx(fisica.guadagno_di(fisica.distanza_dal_cannone(vicino)) * pieno)
    assert guadagni[2] == pytest.approx(fisica.guadagno_di(fisica.distanza_dal_cannone(lontano)) * pieno)
    assert guadagni[1] > guadagni[0] > guadagni[2]
    secondi = np.linspace(0.0, ascolto.percorso.durata, 2000)
    tutti = _parametri(voce, secondi)
    assert tutti[2].max() == pytest.approx(guadagni[1]) and tutti[2].min() == pytest.approx(guadagni[2])
    assert np.allclose(tutti[1], 0.0) and np.allclose(pan, 0.0)
    assert np.allclose(_nota_del_centro(bersaglio, voce, secondi), fisica.frequenza_di(ascolti.CENTRO[1]))
    # Il testo dice di quanti decibel, coi numeri della fisica.
    decibel = fisica.guadagno_db_di(fisica.distanza_dal_cannone(vicino)) - fisica.guadagno_db_di(fisica.distanza_dal_cannone(lontano))
    assert f"circa {testi.intero(decibel)} decibel" in ascolti.passi()[_indice("distanza")].testo


def test_i_punti_a_caso_stanno_nel_cubo_e_si_rivelano_dopo_l_ascolto(finto):
    motore, flusso = finto
    regia = ascolti.Regia(motore, rng=random.Random(11))
    ascolto = regia.suona(_indice("punti"))
    assert len(ascolto.voci) == len(ascolto.bersagli) == len(regia.punti) == ASCOLTI_PUNTI
    minimi, massimi = fisica.zona_del_centro(ASCOLTI_RAGGIO)
    fs = motore.fs
    for numero, (voce, punto) in enumerate(zip(ascolto.voci, regia.punti, strict=True)):
        assert np.all(np.asarray(punto) >= minimi) and np.all(np.asarray(punto) <= massimi)
        # Ogni punto suona ASCOLTI_SUONO secondi, staccato dal seguente da ASCOLTI_PAUSA di silenzio.
        assert voce.inizio == ascolto.inizio + numero * round(ASCOLTI_SUONO * fs + ASCOLTI_PAUSA * fs)
        _frequenze, pan, guadagni = _parametri(voce, [0.5, 1.5])
        assert np.allclose(pan, fisica.pan_di(punto[0], punto[2]))
        assert np.allclose(guadagni, fisica.guadagno_di(fisica.distanza_dal_cannone(punto)) * 10 ** (BERSAGLIO_DB / 20))
    # Finché il suono non è finito il testo non si rivela; poi sì, una riga per punto.
    _udito(motore, flusso, ascolto.inizio + round(ASCOLTI_SUONO * fs))
    assert not regia.sentito() and regia.rivelazione() == ""
    _udito(motore, flusso, ascolto.fine)
    assert regia.sentito()
    righe = regia.rivelazione().splitlines()
    assert righe[0] == "Ecco dove erano i punti che hai sentito:" and len(righe) == 1 + ASCOLTI_PUNTI
    assert righe[1:] == [ascolti.riga_del_punto(numero, punto) for numero, punto in enumerate(regia.punti, 1)]
    # Lo spazio li fa riascoltare uguali.
    prima = regia.punti
    assert regia.suona(_indice("punti")).rivelazione == ascolto.rivelazione and regia.punti is prima


def test_la_riga_di_un_punto_in_metri_all_italiana():
    assert ascolti.riga_del_punto(1, (-12.34, 31.04, 47.5)) == "Punto 1: 12,3 metri a sinistra, a 31,0 metri d'altezza, a 47,5 metri davanti a te."
    assert ascolti.riga_del_punto(2, (8.0, 5.0, 35.0)) == "Punto 2: 8 metri a destra, a 5 metri d'altezza, a 35 metri davanti a te."
    assert ascolti.riga_del_punto(3, (0.01, 25.0, 55.0)).startswith("Punto 3: al centro, a 25 metri d'altezza")


def test_la_grandezza_cresce_da_piccolissima_a_gigantesca(finto):
    motore, _flusso = finto
    ascolto = ascolti.Regia(motore).suona(_indice("grandezza"))
    (voce,), (bersaglio,) = ascolto.voci, ascolto.bersagli
    percorso = ascolto.percorso
    assert np.allclose(percorso(np.array(percorso.arrivi) + UN_MILLESIMO)[:, 0], ASCOLTI_GRANDEZZE)
    raggi = percorso(np.linspace(0.0, percorso.durata, 1000))[:, 0]
    assert np.all(np.diff(raggi) >= 0.0)
    # L'oscillazione della nota, in ottave, ciclo per ciclo dell'LFO: cresce col raggio, e la nota del centro resta.
    centro = fisica.frequenza_di(ascolti.CENTRO[1])
    ciclo = 1.0 / bersaglio.frequenza_lfo
    ampiezze = []
    for inizio in np.arange(0.0, percorso.durata - ciclo, ciclo):
        secondi = np.linspace(inizio, inizio + ciclo, 400)
        frequenze = _parametri(voce, secondi)[0]
        ampiezze.append(np.abs(np.log2(frequenze / centro)).max())
    assert all(dopo >= prima - 1e-6 for prima, dopo in itertools.pairwise(ampiezze))
    assert ampiezze[0] == pytest.approx(mezza_escursione_ottave(ASCOLTI_GRANDEZZE[0]), rel=1e-3)
    assert ampiezze[-1] == pytest.approx(mezza_escursione_ottave(ASCOLTI_GRANDEZZE[-1]), rel=1e-3)


def test_tutte_le_forme_dell_oscillazione_e_tutti_i_timbri(finto):
    motore, _flusso = finto
    regia = ascolti.Regia(motore)
    forme = []
    for nome in ("forme", "forme_2"):
        ascolto = regia.suona(_indice(nome))
        forme.extend(bersaglio.forma for bersaglio in ascolto.bersagli)
        assert {voce.forma for voce in ascolto.voci} == {regia.portante}
    assert forme == list(FORME_LFO)
    # I timbri dei bersagli: tutti quelli che un bersaglio può avere con quel proiettile, e mai quello del proiettile.
    for proiettile in PROIETTILE_FORME:
        regia = ascolti.Regia(motore, proiettile=proiettile)
        ascolto = regia.suona(_indice("timbri"))
        timbri = [voce.forma for voce in ascolto.voci]
        assert timbri == [bersaglio.portante for bersaglio in ascolto.bersagli] == list(portanti_senza(proiettile))
        assert proiettile not in timbri and set(timbri) | {proiettile} == set(BERSAGLIO_PORTANTI)
        assert regia.portante != proiettile


def test_le_tre_scivolate_della_mira_come_in_partita(finto, chiesti):
    motore, _flusso = finto
    ascolto = _regia(motore).suona(_indice("mira"))
    fs = motore.fs
    barre = [voce for voce in ascolto.voci if len(voce.buffer) != CAMPIONI_FINTI]
    campione = ascolto.inizio
    attesi = [("mira_avviata", ascolto.inizio, None)]
    for voce, parametro in zip(barre, PARAMETRI, strict=True):
        buffer = scivolata(parametro, fs=fs)
        campione += round(PAUSA_FRA_BARRE * fs)
        assert voce.inizio == campione and np.array_equal(voce.buffer, buffer.buffer)
        campione += buffer.fine_scivolata + round(MARGINE_DOPO_LA_SCIVOLATA * fs)
        attesi.append(("valore_preso", campione, None))
    assert ascolto.eventi == attesi


def test_il_proiettile_in_volo_dal_cannone_a_terra(finto, chiesti):
    from partita import volo_d_esempio

    motore, flusso = finto
    volo = volo_d_esempio()
    for proiettile in PROIETTILE_FORME:
        ascolto = _regia(motore, proiettile).suona(_indice("volo"))
        # I comandi arrivano al motore col blocco seguente: la fine della voce la mette lui.
        flusso.avanza(1)
        a_terra = ascolto.inizio + round(volo.durata * motore.fs)
        assert ascolto.eventi == [("sparo", ascolto.inizio, fisica.CANNONE), ("a_terra", a_terra, tuple(volo.punto_finale.tolist()))]
        (voce,) = [voce for voce in ascolto.voci if isinstance(voce, VoceTono)]
        assert voce.forma == proiettile and voce.inizio == ascolto.inizio
        # La voce comincia al cannone e finisce dove il proiettile tocca terra, con le regole dello spazio.
        frequenze, pan, _guadagni = _parametri(voce, [0.0, volo.durata])
        partenza, arrivo = fisica.suono_di(volo.traiettoria.posizione(0.0)), fisica.suono_di(volo.punto_finale)
        assert frequenze[0] == pytest.approx(partenza[0]) and frequenze[1] == pytest.approx(arrivo[0]) and pan[1] == pytest.approx(arrivo[1])
        assert voce.fine == a_terra


def test_i_colpi_vicino_e_lontano_e_il_bonus_che_attraversa(finto, chiesti):
    motore, flusso = finto
    regia = _regia(motore)
    minimi, massimi = fisica.zona_del_centro(ASCOLTI_RAGGIO)
    x, y, _z = ascolti.CENTRO
    vicino, lontano = (x, y, minimi[2]), (x, y, massimi[2])
    ascolto = regia.suona(_indice("colpi"))
    pausa = round(ASCOLTI_PAUSA * motore.fs)
    attesi, campione = [], ascolto.inizio
    for evento in ("colpito", "colpito_bonus"):
        for punto in (vicino, lontano):
            attesi.append((evento, campione, tuple(float(c) for c in punto)))
            campione += CAMPIONI_FINTI + pausa
    assert ascolto.eventi == attesi
    # Il bonus: compare col suo avviso, attraversa il cubo da sinistra a destra nella durata del suo livello, ed esce.
    ascolto = regia.suona(_indice("bonus"))
    flusso.avanza(1)
    (voce,), (bonus,) = [v for v in ascolto.voci if isinstance(v, VoceTono)], ascolto.bersagli
    assert bonus.bonus and bonus.raggio < ASCOLTI_RAGGIO
    durata = durata_del_bonus(ASCOLTI_LIVELLO_DEL_BONUS)
    uscita = ascolto.inizio + round(durata * motore.fs)
    assert [evento[:2] for evento in ascolto.eventi] == [("bonus_appare", ascolto.inizio), ("bonus_uscito", uscita)]
    assert ascolto.eventi[1][2][0] > massimi[0] and voce.fine == uscita
    pan = _parametri(voce, np.linspace(0.0, durata, 200))[1]
    assert pan[0] < -0.5 and pan[-1] > 0.5 and np.all(np.diff(pan) > 0.0)


def test_lo_spazio_ferma_il_suono_di_prima_e_ferma_zittisce_tutto(finto, chiesti):
    motore, flusso = finto
    regia = _regia(motore)
    primo = regia.suona(_indice("orizzontale"))
    flusso.avanza(4)
    secondo = regia.suona(_indice("volo"))
    flusso.avanza(4)
    # La voce di prima si spegne subito, molto prima della sua fine.
    assert primo.voci[0].fine < primo.fine
    regia.ferma()
    assert regia.in_corso is None and regia.rivelazione() == "" and not regia.sentito()
    flusso.avanza_secondi(0.2)
    # Il proiettile tace, e il tonfo, che doveva venire alla fine del volo, non suona più.
    assert np.abs(flusso.avanza_secondi(ASCOLTI_DURATA_MASSIMA)).max() == 0.0
    assert all(voce.finita for voce in secondo.voci)
    # Senza suono in corso, ferma non fa niente.
    regia.ferma()


def test_si_contano_i_passi_ascoltati_e_mai_gli_effetti_fra_i_suoni(finto, chiesti):
    motore, _flusso = finto
    contati, suonati = [], []
    regia = ascolti.Regia(motore, suoni.Suoni(motore, conta=suonati.append), conta=contati.append)
    for nome in ("mira", "volo", "colpi", "bonus"):
        regia.suona(_indice(nome))
    assert contati == ["passi_ascoltati"] * 4 and suonati == []
    # I preset sono quelli degli eventi del gioco, nella collezione.
    assert set(chiesti) == {suoni.EVENTI[evento] for evento in ("mira_avviata", "valore_preso", "sparo", "a_terra", "colpito", "colpito_bonus", "bonus_appare",
                                                                "bonus_uscito")}
    # Senza i suoni della finestra gli effetti tacciono, e le voci dei bersagli suonano lo stesso.
    muta = ascolti.Regia(motore)
    assert muta.suona(_indice("volo")).eventi == [] and muta.in_corso.voci


def test_il_percorso_a_tappe():
    percorso = ascolti.Percorso([((0.0,), 1.0), ((10.0,), 0.5)], 5.0)
    assert percorso.durata == pytest.approx(3.5) and percorso.arrivi == pytest.approx((0.0, 3.0))
    valori = percorso([0.5, 1.5, 2.0, 3.2, 99.0, -1.0])[:, 0]
    # Ferma, poi parte dolce, è a metà a metà viaggio, poi ferma all'arrivo e oltre la fine; prima dell'inizio sta alla partenza.
    assert valori[0] == 0.0 and 0.0 < valori[1] < 2.5 and valori[2] == pytest.approx(5.0) and valori[3] == 10.0 and valori[4] == 10.0 and valori[5] == 0.0
    with pytest.raises(ValueError):
        ascolti.Percorso([], 1.0)
    with pytest.raises(ValueError):
        ascolti.Percorso([((0.0,), 1.0)], 0.0)


def test_suono_in_suona_come_il_bersaglio_nel_suo_punto():
    """D23: Bersaglio.suono_in, che gli Ascolti usano per i loro cammini, dà nel punto del bersaglio quel che dà parametri_sonori."""
    bersaglio = Bersaglio(numero=1, forma="trillo", raggio=3.0, partenza=(4.0, 20.0, 50.0), frequenza_lfo=3.3, fase_lfo=0.2, seme=5, portante="triangolo",
                          escursione=1.1)
    tempi = np.linspace(0.0, 2.0, 500)
    attesi = bersaglio.parametri_sonori(tempi, tempi)
    ottenuti = bersaglio.suono_in(bersaglio.posizione(tempi), tempi)
    assert all(np.allclose(a, b) for a, b in zip(attesi, ottenuti, strict=True))
    # Col raggio dato, l'oscillazione è quella di quel raggio.
    doppio = bersaglio.suono_in(bersaglio.posizione(tempi), tempi, raggio=np.full(len(tempi), 6.0))[0]
    assert np.allclose(np.log2(doppio / attesi[0]), np.asarray(bersaglio.lfo(tempi)) * (mezza_escursione_ottave(6.0, 1.1) - bersaglio.mezza_escursione_ottave()))
