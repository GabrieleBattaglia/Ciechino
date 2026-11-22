"""
Le prove degli Ascolti di ascolti.py (D23): i passi coi loro testi, e i loro suoni sul motore col flusso finto di
aiuti_audio, mai con la scheda vera.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-20 con la 0.9.0. I suoni si guardano in due modi: coi parametri delle voci dei bersagli, chiesti
negli istanti che contano, e con ciò che il motore scrive, blocco per blocco, sul flusso finto. Nessun preset vero si
sintetizza: _rendi_preset dà a tutti lo stesso buffer finto, e annota chi lo chiede, oppure un silenzio lungo quanto
il preset vero, dalla sua partitura, per misurare la durata dei passi.
Dalla 2.0.0 i passi stanno nella cupola di D28: le tratte partono dal suo mezzo e girano sull'azimut, sull'elevazione
o in profondità, restano nella zona del bersaglio e cambiano una cosa sola; la mira manuale di D29 suona le corse dei
tre valori, con le tacche e il suono del capo; il passo nuovo del tempo dice i tick e lo sparo; il bonus attraversa
la cupola su una corda. I numeri dei testi si rifanno col codice.
"""

import itertools
import math
import random
import re

import numpy as np
import pytest
from aiuti_audio import LATENZA_FINTA, motore_finto

import ascolti
import fisica
import suoni
import testi
from bersagli import Bersaglio, durata_del_bonus, mezza_escursione_ottave, portanti_senza, raggio_del_livello
from costanti import (
    ALTEZZA_CANNONE,
    ALTEZZA_MINIMA_BERSAGLIO,
    ALZO_FREQUENZE,
    ASCOLTI_DURATA_MASSIMA,
    ASCOLTI_FORME_PER_PASSO,
    ASCOLTI_GRANDEZZE,
    ASCOLTI_LIVELLO_DEL_BONUS,
    ASCOLTI_PAUSA,
    ASCOLTI_PUNTI,
    ASCOLTI_RAGGIO,
    ASCOLTI_SOSTA,
    ASCOLTI_SOSTA_AI_CAPI,
    ASCOLTI_STEREO_FORMA,
    ASCOLTI_STEREO_SOSTA,
    ASCOLTI_STEREO_VIAGGIO,
    ASCOLTI_SUONO,
    ASCOLTI_VELOCITA,
    AZIMUT_DEL_PAN,
    BERSAGLIO_DB,
    BERSAGLIO_PORTANTI,
    COLPI_BONUS,
    CUPOLA_RAGGIO,
    CUPOLA_RAGGIO_INTERNO,
    DB_PER_METRO,
    DURATA_BONUS_MINIMA,
    DURATA_DELLA_SCALA,
    ELEVAZIONE_MASSIMA,
    FORME_LFO,
    FS,
    METRI_PER_OTTAVA,
    MOLTIPLICATORE_BONUS,
    PARAMETRI,
    PARTENZA_DELLA_MIRA,
    POTENZA_DB_MINIMO,
    PROIETTILE_FORME,
    PUNTATORE_FERMO_DB,
    RAGGIO_BONUS_FATTORE,
    RIDUZIONE_DEL_TEMPO_DELLA_MIRA,
    SETTORE_AZIMUT,
    SFUMATURA_STOP,
    TACCHE,
    TEMPO_DELLA_MIRA_INIZIALE,
    TEMPO_DELLA_MIRA_MINIMO,
    VELOCITA_DEL_TEMPO,
    VELOCITA_DELLA_MIRA,
)
from motore_audio import VoceBuffer, VoceTono
from sintesi import db_in_lineare, pan_della_direzione, parametri_del_puntatore, tacca_dell_asse, tick_del_tempo
from statistiche import NOMI_DELLE_FORME, NOMI_DELLE_PORTANTI
from strumenti import crea_preset

# Il buffer finto di ogni preset: corto, al centro, ben sotto il tetto del motore.
CAMPIONI_FINTI = 2_000
SEPARATORE = re.compile(r"-{3,}|={3,}|_{3,}")
NOMI_DEI_PASSI = ["stereo", "orizzontale", "altezza", "distanza", "punti", "grandezza", "forme", "forme_2", "timbri", "mira", "tempo", "volo", "colpi",
                  "bonus"]
# Le tratte, coi loro due capi nell'ordine in cui il bersaglio ci arriva.
TRATTE = [("orizzontale", "sinistra", "destra"), ("altezza", "terra", "cima"), ("distanza", "fronte", "fondo")]
# Un millesimo di secondo, per prendere un istante appena dentro una sosta.
UN_MILLESIMO = 0.001
# Di quanto si va oltre un capo, nelle unità della sua tratta, gradi o metri, per uscire dalla zona; e la tolleranza dei punti sui
# confini, dove la bisezione dei capi si ferma a una finezza di calcolo.
OLTRE_IL_CAPO = 1e-3
TOLLERANZA = 1e-9
# I campioni della trasformata che cerca la frequenza più forte di un tick: molti zeri in coda, per una lettura fine.
PUNTI_DELLA_TRASFORMATA = 1 << 16


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


@pytest.fixture
def lunghi_come_i_veri(monkeypatch):
    """Ogni preset suona un silenzio lungo quanto il preset vero, che la sua partitura in strumenti/crea_preset.py dice: così la durata
    di un passo è quella che si sentirà, senza sintetizzare niente."""

    def rendi(nome):
        buffer = np.zeros((round(crea_preset.durata(crea_preset.PRESET[nome]["score"]) * FS), 2), dtype=np.float32)
        buffer.flags.writeable = False
        return buffer

    monkeypatch.setattr(suoni, "_rendi_preset", rendi)


def _regia(motore, proiettile="seno", **opzioni):
    return ascolti.Regia(motore, suoni.Suoni(motore), proiettile, rng=random.Random(7), **opzioni)


def _indice(nome):
    return NOMI_DEI_PASSI.index(nome)


def _parametri(voce, secondi):
    """(frequenze, pan, guadagni) di una VoceTono negli istanti dati, in secondi dal suo inizio."""
    tempi = voce.inizio / voce.fs + np.atleast_1d(np.asarray(secondi, dtype=float))
    return voce.parametri(np.zeros_like(tempi), tempi)


def _puntatore_con(parametro, valore):
    """(frequenze, pan, guadagni) del puntatore col valore del parametro dato e gli altri due al centro, come negli Ascolti (D31)."""
    return parametri_del_puntatore(*(valore if nome == parametro else PARTENZA_DELLA_MIRA for nome in PARAMETRI))


def _pan_del_buffer(stereo):
    """La panoramica di un buffer stereo, da meno uno a uno, dall'energia dei due lati con la legge di sintesi.panoramica."""
    dati = np.asarray(stereo, dtype=np.float64)
    return float(np.arctan2(np.sqrt(np.mean(dati[:, 1] ** 2)), np.sqrt(np.mean(dati[:, 0] ** 2))) * 4.0 / np.pi - 1.0)


def _nota_del_centro(bersaglio, voce, secondi):
    """La frequenza della nota del centro del bersaglio, cioè senza l'oscillazione dell'LFO."""
    frequenze, _pan, _guadagni = _parametri(voce, secondi)
    return frequenze / 2.0 ** (np.asarray(bersaglio.lfo(np.atleast_1d(secondi))) * bersaglio.mezza_escursione_ottave())


def _udito(motore, flusso, campione):
    """Fa scrivere al flusso finto abbastanza blocchi perché il campione sia già uscito dalle casse."""
    flusso.avanza_fino(campione + round(LATENZA_FINTA * motore.fs) + motore.blocco)
    assert motore.campione_udibile() >= campione


def _pieno():
    """Il guadagno lineare di BERSAGLIO_DB, che i bersagli hanno sopra quello della distanza."""
    return 10 ** (BERSAGLIO_DB / 20)


def _frequenza_dominante(mono, fs):
    """La frequenza più forte di un suono breve, dalla trasformata con gli zeri in coda."""
    spettro = np.abs(np.fft.rfft(np.asarray(mono, dtype=float), n=PUNTI_DELLA_TRASFORMATA))
    return float(np.argmax(spettro)) * fs / PUNTI_DELLA_TRASFORMATA


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
        # Dalla 2.0.0 lo spazio dei bersagli è la cupola (D28), e la mira non ha più scivolate, barre né bip (D29).
        assert not re.search(r"\bcub[oi]\b|scivolat|\bbarr[ae]\b|\bbip\b", f"{passo.titolo} {passo.testo}"), passo.nome
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
    for portante in portanti_senza("seno"):
        assert NOMI_DELLE_PORTANTI[portante] in testi_dei_passi["timbri"]
    # La mira manuale dice i suoi tasti, l'Invio ed Esc; il tempo dice lo sparo allo scadere.
    mira = testi_dei_passi["mira"]
    assert all(parola in mira for parola in ("freccia destra", "frecce su e giù", "Fine e Home", "Invio spara", "Esc annulla"))
    # Il puntatore di D31 porta i tre valori, e negli Ascolti gli altri due restano al centro.
    assert all(parola in mira for parola in ("il puntatore", "porta sempre tutti e tre i valori", "con gli altri due al centro"))
    assert "tick" in testi_dei_passi["tempo"] and "sparo" in testi_dei_passi["tempo"] and "palestra" in testi_dei_passi["tempo"]


def test_i_numeri_dei_testi_vengono_dal_codice():
    """D23: ogni numero dei testi viene dalle costanti e dalla fisica del gioco, così il testo dice sempre il vero; qui si rifà con le
    stesse funzioni, e il testo lo deve dire."""
    from partita import nella_palestra, tempo_della_mira, volo_d_esempio

    testi_dei_passi = {passo.nome: passo.testo for passo in ascolti.passi("seno")}
    capi, punti = ascolti._capi(), ascolti.punti_dei_capi()
    centro = np.array(ascolti.CENTRO)
    vicino, lontano = fisica.distanza_dal_cannone(punti["fronte"]), fisica.distanza_dal_cannone(punti["fondo"])
    decibel = fisica.guadagno_db_di(vicino) - fisica.guadagno_db_di(lontano)
    piccolo, gigante = ASCOLTI_GRANDEZZE[0], ASCOLTI_GRANDEZZE[-1]
    volo = volo_d_esempio()
    attesi = {
        "orizzontale": [f"{testi.metri(fisica.distanza_dal_cannone(centro))} davanti a te", f"a {testi.metri(centro[1])} d'altezza",
                        f"col raggio di {testi.metri(ASCOLTI_RAGGIO)}", f"a {testi.gradi(round(-capi['sinistra']))} a sinistra",
                        f"a {testi.gradi(AZIMUT_DEL_PAN)};", f"a {testi.gradi(round(capi['destra']))}, perché"],
        "altezza": [f"sta a {testi.metri(punti['terra'][1])} dal prato", f"fino a {testi.metri(punti['cima'][1])} d'altezza",
                    f"sempre a {testi.metri(ascolti.DISTANZA_DEL_MEZZO)} dal centro della cupola, sulla sua volta",
                    f"ogni {testi.metri(METRI_PER_OTTAVA, 2)}", f"ogni {testi.numero(METRI_PER_OTTAVA, 2)} metri,",
                    f"circa {testi.intero(fisica.frequenza_di(punti['terra'][1]))} hertz a terra",
                    f"circa {testi.intero(fisica.frequenza_di(punti['cima'][1]))} hertz in alto"],
        "distanza": [f"a {testi.metri(vicino)} da te", f"a {testi.metri(lontano)};", f"a {testi.metri(punti['fronte'][1])} d'altezza",
                     f"circa {testi.intero(decibel)} decibel"],
        "punti": [f"in {ASCOLTI_PUNTI} punti"],
        "grandezza": [f"raggio di {testi.metri(piccolo)}", f"fino a {testi.metri(ASCOLTI_GRANDEZZE[1])}", f"gigantesco, {testi.metri(gigante)}",
                      f"di {testi.numero(ascolti.SEMITONI_PER_OTTAVA * mezza_escursione_ottave(piccolo), 1)} semitoni",
                      f"di {testi.numero(ascolti.SEMITONI_PER_OTTAVA * mezza_escursione_ottave(gigante), 1)} semitoni"],
        "forme": [f"fra {len(FORME_LFO)}.", f"{ASCOLTI_FORME_PER_PASSO} per passo", f"per {testi.numero(ASCOLTI_SUONO, 1)} secondi"],
        "forme_2": [f"Altre {len(FORME_LFO) - ASCOLTI_FORME_PER_PASSO} forme"],
        "timbri": [f"fra {len(portanti_senza('seno'))}:", f"per {testi.numero(ASCOLTI_SUONO, 1)} secondi"],
        "mira": [f"da {testi.intero(ALZO_FREQUENZE[0])} a {testi.intero(ALZO_FREQUENZE[1])} hertz",
                 f"{testi.intero(-POTENZA_DB_MINIMO)} decibel dal minimo al massimo", f"in {testi.durata(DURATA_DELLA_SCALA)}"],
        "tempo": [f"{testi.durata(TEMPO_DELLA_MIRA_INIZIALE)} al primo livello dopo la palestra",
                  f"{testi.durata(RIDUZIONE_DEL_TEMPO_DELLA_MIRA)} in meno a ogni livello, fino a {testi.durata(TEMPO_DELLA_MIRA_MINIMO)}.",
                  f"di {testi.durata(TEMPO_DELLA_MIRA_MINIMO)}:"],
        "volo": [f"fino a {testi.metri(volo.altezza_massima)}", f"a {testi.metri(volo.gittata)} davanti a te",
                 f"dopo {testi.numero(volo.durata / VELOCITA_DEL_TEMPO, 1)} secondi"],
        "colpi": [f"vicino, a {testi.metri(vicino)}", f"lontano, a {testi.metri(lontano)}"],
        "bonus": [f"in {testi.durata(durata_del_bonus(ASCOLTI_LIVELLO_DEL_BONUS))}", f"al livello {ASCOLTI_LIVELLO_DEL_BONUS},",
                  f"fino a {testi.durata(DURATA_BONUS_MINIMA)}", f"{MOLTIPLICATORE_BONUS} volte", f"{COLPI_BONUS} colpi"],
    }
    for nome, pezzi in attesi.items():
        for pezzo in pezzi:
            assert pezzo in testi_dei_passi[nome], (nome, pezzo)
    # Il tempo della mira dice quello della partita (D29): nessun tempo nella palestra, TEMPO_DELLA_MIRA_INIZIALE al primo livello
    # dopo, RIDUZIONE_DEL_TEMPO_DELLA_MIRA in meno a ogni livello, «1 secondo in meno a ogni livello», fino a TEMPO_DELLA_MIRA_MINIMO.
    primo = next(livello for livello in itertools.count(1) if not nella_palestra(livello))
    assert all(tempo_della_mira(livello) is None for livello in range(1, primo))
    assert tempo_della_mira(primo) == TEMPO_DELLA_MIRA_INIZIALE and tempo_della_mira(primo) - tempo_della_mira(primo + 1) == RIDUZIONE_DEL_TEMPO_DELLA_MIRA == 1
    assert "1 secondo in meno a ogni livello" in testi_dei_passi["tempo"]
    assert min(tempo_della_mira(livello) for livello in range(primo, primo + 100)) == TEMPO_DELLA_MIRA_MINIMO


def test_il_proiettile_sconosciuto_non_ha_ascolti(finto):
    motore, _flusso = finto
    with pytest.raises(ValueError, match="proiettile"):
        ascolti.Regia(motore, None, "organo")


def test_ogni_passo_suona_e_dura_poco(finto, lunghi_come_i_veri):
    """Ogni passo mette le sue voci nel motore e dura al massimo ASCOLTI_DURATA_MASSIMA, coi preset lunghi come quelli veri: oltre, si
    divide in due."""
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


# Le tratte nella cupola (D28).

def _oltre(nome, valore):
    """Il punto di un capo spostato appena oltre, nel verso in cui la sua tratta usciva dalla zona."""
    distanza, elevazione = ascolti.DISTANZA_DEL_MEZZO, ascolti.ELEVAZIONE_DEL_MEZZO
    _x, altezza, profondita = ascolti.CENTRO
    if nome in ("sinistra", "destra"):
        return fisica.punto_della_cupola(distanza, valore + math.copysign(OLTRE_IL_CAPO, valore), elevazione)
    if nome in ("terra", "cima"):
        return fisica.punto_della_cupola(distanza, 0.0, valore + math.copysign(OLTRE_IL_CAPO, valore - elevazione))
    return np.array((0.0, altezza, valore + math.copysign(OLTRE_IL_CAPO, valore - profondita)))


def test_i_capi_delle_tratte_sul_bordo_della_zona_del_bersaglio():
    """D28: la zona degli Ascolti è quella del gioco per il loro bersaglio, e le tratte partono dal mezzo della cupola, a metà fra le
    due sfere. I capi stanno dove il bersaglio, tutto dentro, tocca una parete: dentro la zona, e appena oltre fuori. Sono quelli dei
    conti a mano sulla cupola: il lato del settore alla distanza orizzontale del mezzo, il suolo, il cono di ELEVAZIONE_MASSIMA
    ristretto dall'angolo del raggio, e le due sfere alla quota del mezzo."""
    zona = ascolti.zona()
    assert isinstance(zona, fisica.Zona) and zona.raggio == ASCOLTI_RAGGIO
    assert ascolti.CENTRO == fisica.MEZZO_DELLA_CUPOLA and zona.contiene(ascolti.CENTRO, tolleranza=0.0)
    distanza, elevazione = ascolti.DISTANZA_DEL_MEZZO, ascolti.ELEVAZIONE_DEL_MEZZO
    assert distanza == (CUPOLA_RAGGIO_INTERNO + CUPOLA_RAGGIO) / 2 and np.linalg.norm(ascolti.CENTRO) == pytest.approx(distanza)
    _x, altezza, _z = ascolti.CENTRO
    assert altezza == pytest.approx(distanza * math.sin(math.radians(elevazione)))
    orizzontale = distanza * math.cos(math.radians(elevazione))
    destra = SETTORE_AZIMUT - math.degrees(math.asin(ASCOLTI_RAGGIO / orizzontale))
    attesi = {
        "sinistra": -destra, "destra": destra,
        "terra": math.degrees(math.asin(max(ALTEZZA_MINIMA_BERSAGLIO, ASCOLTI_RAGGIO) / distanza)),
        "cima": ELEVAZIONE_MASSIMA - math.degrees(math.asin(ASCOLTI_RAGGIO / (CUPOLA_RAGGIO_INTERNO + ASCOLTI_RAGGIO))),
        "fronte": math.sqrt((CUPOLA_RAGGIO_INTERNO + ASCOLTI_RAGGIO) ** 2 - altezza ** 2),
        "fondo": math.sqrt((CUPOLA_RAGGIO - ASCOLTI_RAGGIO) ** 2 - altezza ** 2),
    }
    capi = ascolti._capi()
    assert capi == pytest.approx(attesi, abs=1e-9)
    punti = ascolti.punti_dei_capi()
    assert set(punti) == set(capi)
    for nome, valore in capi.items():
        assert zona.contiene(punti[nome], tolleranza=TOLLERANZA), nome
        assert not zona.contiene(_oltre(nome, valore), tolleranza=0.0), nome
    # I punti dei capi: i lati sull'arco del mezzo, terra e cima sulla volta davanti, fronte e fondo alla quota del mezzo.
    assert np.allclose(punti["destra"], fisica.punto_della_cupola(distanza, destra, elevazione))
    assert np.allclose(punti["sinistra"], fisica.punto_della_cupola(distanza, -destra, elevazione))
    assert np.allclose(punti["terra"], fisica.punto_della_cupola(distanza, 0.0, attesi["terra"]))
    assert np.allclose(punti["cima"], fisica.punto_della_cupola(distanza, 0.0, attesi["cima"]))
    assert np.allclose(punti["fronte"], (0.0, altezza, attesi["fronte"])) and np.allclose(punti["fondo"], (0.0, altezza, attesi["fondo"]))
    with pytest.raises(ValueError, match="tratta"):
        ascolti.tratta("diagonale")


@pytest.mark.parametrize(("nome", "primo", "secondo"), TRATTE)
def test_ogni_tratta_dal_mezzo_ai_due_capi_e_ritorno_dentro_la_zona(finto, nome, primo, secondo):
    """Ogni tratta va dal mezzo della cupola al primo capo, al secondo e di nuovo al mezzo, ferma ASCOLTI_SOSTA nel mezzo e
    ASCOLTI_SOSTA_AI_CAPI ai capi; fra una tappa e l'altra il bersaglio fa la sua strada a ASCOLTI_VELOCITA metri al secondo, in
    media, partendo e arrivando dolce, e resta sempre dentro la zona del bersaglio."""
    motore, _flusso = finto
    tappe, _velocita, punto_di = ascolti.tratta(nome)
    capi, punti_dei_capi = ascolti._capi(), ascolti.punti_dei_capi()
    (mezzo,), _sosta = tappe[0]
    assert [sosta for _valore, sosta in tappe] == [ASCOLTI_SOSTA, ASCOLTI_SOSTA_AI_CAPI, ASCOLTI_SOSTA_AI_CAPI, ASCOLTI_SOSTA]
    assert [valore for (valore,), _sosta in tappe] == [mezzo, capi[primo], capi[secondo], mezzo]
    attesi = np.array([ascolti.CENTRO, punti_dei_capi[primo], punti_dei_capi[secondo], ascolti.CENTRO])
    assert np.allclose(punto_di(np.array([mezzo, capi[primo], capi[secondo], mezzo])), attesi)
    ascolto = ascolti.Regia(motore).suona(_indice(nome))
    percorso = ascolto.percorso
    assert np.allclose(punto_di(percorso(np.array(percorso.arrivi) + UN_MILLESIMO)[:, 0]), attesi)
    assert ascolto.fine - ascolto.inizio == round(percorso.durata * motore.fs) and percorso.durata <= ASCOLTI_DURATA_MASSIMA
    secondi = np.linspace(0.0, percorso.durata, 4000)
    assert np.all(ascolti.zona().contiene(punto_di(percorso(secondi)[:, 0]), tolleranza=TOLLERANZA))
    soste = [sosta for _valore, sosta in tappe]
    for k in range(1, len(tappe)):
        partenza = percorso.arrivi[k - 1] + soste[k - 1]
        tratto = punto_di(percorso(np.linspace(partenza, percorso.arrivi[k], 2001))[:, 0])
        strada = float(np.sum(np.linalg.norm(np.diff(tratto, axis=0), axis=1)))
        assert strada / (percorso.arrivi[k] - partenza) == pytest.approx(ASCOLTI_VELOCITA, rel=1e-4), (nome, k)


def _cammino(motore, nome, quanti=2000):
    """L'ascolto di una tratta, la sua voce e il suo bersaglio, quanti istanti lungo tutto il cammino e i punti del bersaglio in quegli
    istanti."""
    ascolto = ascolti.Regia(motore).suona(_indice(nome))
    (voce,), (bersaglio,) = ascolto.voci, ascolto.bersagli
    _tappe, _velocita, punto_di = ascolti.tratta(nome)
    secondi = np.linspace(0.0, ascolto.percorso.durata, quanti)
    return ascolto, voce, bersaglio, secondi, punto_di(ascolto.percorso(secondi)[:, 0])


def test_la_tratta_orizzontale_cambia_soltanto_la_panoramica(finto):
    """La tratta orizzontale gira sull'arco del mezzo, alla stessa distanza dal centro della cupola, dal cannone e alla stessa altezza:
    cambia soltanto la panoramica, che è l'azimut visto dal cannone diviso AZIMUT_DEL_PAN, prima a sinistra e poi a destra."""
    motore, _flusso = finto
    ascolto, voce, bersaglio, secondi, punti = _cammino(motore, "orizzontale")
    centro = np.array(ascolti.CENTRO)
    assert np.allclose(np.linalg.norm(punti, axis=1), ascolti.DISTANZA_DEL_MEZZO) and np.allclose(punti[:, 1], centro[1])
    assert np.allclose(fisica.distanza_dal_cannone(punti), fisica.distanza_dal_cannone(centro))
    _frequenze, pan, guadagni = _parametri(voce, secondi)
    assert np.allclose(_nota_del_centro(bersaglio, voce, secondi), fisica.frequenza_di(centro[1]))
    assert np.allclose(guadagni, fisica.guadagno_di(fisica.distanza_dal_cannone(centro)) * _pieno())
    assert np.allclose(pan, np.degrees(np.arctan2(punti[:, 0], punti[:, 2] - fisica.CANNONE[2])) / AZIMUT_DEL_PAN)
    capi = ascolti._capi()
    assert pan.min() == pytest.approx(capi["sinistra"] / AZIMUT_DEL_PAN) and pan.max() == pytest.approx(capi["destra"] / AZIMUT_DEL_PAN)
    nelle_tappe = _parametri(voce, np.array(ascolto.percorso.arrivi) + UN_MILLESIMO)[1]
    assert nelle_tappe[0] == pytest.approx(0.0) and nelle_tappe[1] < 0.0 < nelle_tappe[2] and nelle_tappe[3] == pytest.approx(0.0)


def test_la_tratta_dell_altezza_cambia_soltanto_l_altezza_a_distanza_costante(finto):
    """La tratta dell'altezza scende a terra e sale in cima sulla volta davanti, alla distanza del mezzo dal centro della cupola: la
    panoramica resta al centro, la nota segue l'altezza, dalla più grave a terra alla più acuta in cima, e il volume cambia appena,
    soltanto perché il cannone sta ALTEZZA_CANNONE sopra il centro della cupola."""
    motore, _flusso = finto
    ascolto, voce, bersaglio, secondi, punti = _cammino(motore, "altezza")
    assert np.allclose(np.linalg.norm(punti, axis=1), ascolti.DISTANZA_DEL_MEZZO) and np.allclose(punti[:, 0], 0.0)
    _frequenze, pan, guadagni = _parametri(voce, secondi)
    assert np.allclose(pan, 0.0)
    assert np.allclose(_nota_del_centro(bersaglio, voce, secondi), fisica.frequenza_di(punti[:, 1]))
    assert np.allclose(guadagni, fisica.guadagno_di(fisica.distanza_dal_cannone(punti)) * _pieno())
    decibel = 20 * np.log10(guadagni)
    assert decibel.max() - decibel.min() <= DB_PER_METRO * 2 * ALTEZZA_CANNONE
    # La nota più grave e la più acuta: quella di terra e quella della cima, ciascuna in fondo e in cima all'oscillazione.
    terra, cima = ascolti.punti_dei_capi()["terra"][1], ascolti.punti_dei_capi()["cima"][1]
    tutti = np.arange(0.0, ascolto.percorso.durata, 1.0 / motore.fs)
    frequenze = _parametri(voce, tutti)[0]
    mezza = bersaglio.mezza_escursione_ottave()
    assert frequenze.min() == pytest.approx(fisica.frequenza_di(terra) * 2.0 ** -mezza, rel=1e-4)
    assert frequenze.max() == pytest.approx(fisica.frequenza_di(cima) * 2.0 ** mezza, rel=1e-4)


def test_la_tratta_della_distanza_cambia_soltanto_il_volume(finto):
    """La tratta della distanza va in profondità, davanti, alla quota del mezzo: la panoramica resta al centro e la nota del centro
    quella della quota; cambia soltanto il volume, il più forte al fronte e il più piano al fondo, e il testo dice di quanti decibel."""
    motore, _flusso = finto
    ascolto, voce, bersaglio, secondi, punti = _cammino(motore, "distanza")
    centro = np.array(ascolti.CENTRO)
    assert np.allclose(punti[:, 0], 0.0) and np.allclose(punti[:, 1], centro[1])
    _frequenze, pan, guadagni = _parametri(voce, secondi)
    assert np.allclose(pan, 0.0)
    assert np.allclose(_nota_del_centro(bersaglio, voce, secondi), fisica.frequenza_di(centro[1]))
    assert np.allclose(guadagni, fisica.guadagno_di(fisica.distanza_dal_cannone(punti)) * _pieno())
    capi = ascolti.punti_dei_capi()
    vicino, lontano = fisica.distanza_dal_cannone(capi["fronte"]), fisica.distanza_dal_cannone(capi["fondo"])
    assert guadagni.max() == pytest.approx(fisica.guadagno_di(vicino) * _pieno()) and guadagni.min() == pytest.approx(fisica.guadagno_di(lontano) * _pieno())
    nelle_tappe = _parametri(voce, np.array(ascolto.percorso.arrivi) + UN_MILLESIMO)[2]
    assert nelle_tappe[1] > nelle_tappe[0] > nelle_tappe[2]
    decibel = fisica.guadagno_db_di(vicino) - fisica.guadagno_db_di(lontano)
    assert f"circa {testi.intero(decibel)} decibel" in ascolti.passi()[_indice("distanza")].testo


def test_i_punti_a_caso_stanno_nella_cupola_e_si_rivelano_dopo_l_ascolto(finto):
    motore, flusso = finto
    regia = ascolti.Regia(motore, rng=random.Random(11))
    ascolto = regia.suona(_indice("punti"))
    assert len(ascolto.voci) == len(ascolto.bersagli) == len(regia.punti) == ASCOLTI_PUNTI
    # Si estraggono come quelli del gioco (D28), con Zona.punto_a_caso, e stanno tutti nella zona del bersaglio.
    rng = random.Random(11)
    zona = ascolti.zona()
    assert regia.punti == tuple(tuple(float(c) for c in zona.punto_a_caso(rng)) for _punto in range(ASCOLTI_PUNTI))
    fs = motore.fs
    for numero, (voce, punto) in enumerate(zip(ascolto.voci, regia.punti, strict=True)):
        assert zona.contiene(punto, tolleranza=0.0)
        # Ogni punto suona ASCOLTI_SUONO secondi, staccato dal seguente da ASCOLTI_PAUSA di silenzio.
        assert voce.inizio == ascolto.inizio + numero * round(ASCOLTI_SUONO * fs + ASCOLTI_PAUSA * fs)
        _frequenze, pan, guadagni = _parametri(voce, [0.5, 1.5])
        assert np.allclose(pan, fisica.pan_di(punto[0], punto[2]))
        assert np.allclose(guadagni, fisica.guadagno_di(fisica.distanza_dal_cannone(punto)) * _pieno())
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
    # Il gigante, fermo nel mezzo della cupola, ci sta tutto dentro.
    assert fisica.Zona(max(ASCOLTI_GRANDEZZE)).contiene(bersaglio.partenza, tolleranza=0.0)


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


# La mira manuale e il suo tempo (D29).

def test_la_mira_manuale_come_in_partita_con_le_corse_le_tacche_e_i_capi(finto, chiesti):
    """D29: il respiro che avvia la mira, poi per ogni valore, nell'ordine di PARAMETRI, le corse di CORSE_DELLA_MIRA, dal centro al
    massimo e dal massimo al minimo, come col suo tasto giù, a VELOCITA_DELLA_MIRA; la tacca di tacca_dell_asse a ogni decimo che
    attraversa, nell'istante in cui lo passa; e il suono fine_scala al capo, dal punto del puntatore. Il puntatore di D31 è una voce
    sola per ogni valore, dalla prima corsa alla fine dell'ultima, nel timbro del proiettile (D32), con gli altri due valori al
    centro: pieno mentre il valore corre, e PUNTATORE_FERMO_DB più piano fra le due corse, col valore fermo dove l'ha lasciato. Ogni
    corsa parte ASCOLTI_PAUSA dopo la fine della precedente, capo compreso, e niente spara."""
    assert ascolti.CORSE_DELLA_MIRA == ((PARTENZA_DELLA_MIRA, 1.0), (1.0, 0.0))
    motore, _flusso = finto
    fs = motore.fs
    regia = _regia(motore)
    ascolto = regia.suona(_indice("mira"))
    pausa = round(ASCOLTI_PAUSA * fs)
    fermo = float(db_in_lineare(PUNTATORE_FERMO_DB))
    voci = iter(ascolto.voci)
    respiro = next(voci)
    assert isinstance(respiro, VoceBuffer) and respiro.inizio == ascolto.inizio
    campione = ascolto.inizio + CAMPIONI_FINTI + pausa
    eventi = [("mira_avviata", ascolto.inizio, None)]
    puntatori = {}
    for parametro in PARAMETRI:
        via = campione
        corse = []
        for da, a in ascolti.CORSE_DELLA_MIRA:
            fine = campione + round(abs(a - da) / VELOCITA_DELLA_MIRA * fs)
            # Le tacche: i decimi che il valore attraversa, esclusi quello da cui parte e il capo, che ha il suo suono.
            decimi = sorted((k for k in range(1, TACCHE) if min(da, a) < k / TACCHE < max(da, a)), key=lambda k: abs(k / TACCHE - da))
            tacche = sorted(itertools.islice(voci, len(decimi)), key=lambda tacca: tacca.inizio)
            assert [tacca.inizio for tacca in tacche] == [campione + round(abs(k / TACCHE - da) / VELOCITA_DELLA_MIRA * fs) for k in decimi]
            assert all(isinstance(tacca, VoceBuffer) and campione < tacca.inizio < fine for tacca in tacche)
            assert all(np.array_equal(tacca.buffer, tacca_dell_asse(parametro, k, fs=fs)) for tacca, k in zip(tacche, decimi, strict=True))
            capo = next(voci)
            assert isinstance(capo, VoceBuffer) and capo.inizio == fine
            pan = pan_della_direzione(a if parametro == "direzione" else PARTENZA_DELLA_MIRA)
            assert _pan_del_buffer(capo.buffer) == pytest.approx(pan, abs=1e-3), (parametro, a)
            eventi.append(("fine_scala", fine, None))
            corse.append((campione, fine, da, a))
            campione = fine + CAMPIONI_FINTI + pausa
        voce = next(voci)
        assert isinstance(voce, VoceTono) and voce.forma == regia.proiettile and voce.inizio == via, parametro
        # Nelle corse il puntatore suona pieno il valore che corre: dopo l'entrata, a metà e prima dell'uscita.
        for dal, al, da, a in corse:
            secondi, verso = (al - dal) / fs, (1.0 if a >= da else -1.0)
            for istante in (2 * SFUMATURA_STOP, secondi / 2, secondi - 2 * SFUMATURA_STOP):
                ottenuti = _parametri(voce, (dal - via) / fs + istante)
                attesi = _puntatore_con(parametro, da + verso * VELOCITA_DELLA_MIRA * istante)
                assert all(np.allclose(x, y) for x, y in zip(ottenuti, attesi, strict=True)), (parametro, da, a, istante)
        # Fra le due corse il valore resta dove la prima l'ha lasciato, e il puntatore suona PUNTATORE_FERMO_DB più piano.
        (_dal, al, _da, a), _seconda = corse
        frequenze, pan, guadagni = _parametri(voce, (al - via) / fs + (CAMPIONI_FINTI + pausa) / (2 * fs))
        attesi = _puntatore_con(parametro, a)
        assert np.allclose(frequenze, attesi[0]) and np.allclose(pan, attesi[1]) and np.allclose(guadagni, attesi[2] * fermo), parametro
        puntatori[parametro] = (voce, via, corse)
    assert next(voci, None) is None
    assert ascolto.eventi == eventi and ascolto.fine == campione - pausa
    # La direzione va dal mezzo a tutto a destra, e poi a tutto a sinistra, come dice il testo.
    voce, via, corse = puntatori["direzione"]
    capi = [_parametri(voce, (al - via) / fs)[1][0] for _dal, al, _da, _a in corse]
    assert _parametri(voce, 0.0)[1][0] == pytest.approx(0.0) and capi == pytest.approx([1.0, -1.0])


def test_una_corsa_che_non_arriva_al_capo_non_ha_il_suo_suono(finto, chiesti):
    """Regia.corsa: un valore che si ferma prima del capo ha il puntatore e le tacche dei decimi che passa, e non il suono fine_scala."""
    motore, _flusso = finto
    fs = motore.fs
    regia = _regia(motore)
    ascolto = ascolti.Ascolto("prova", 0, 0)
    inizio = motore.prossimo_campione()
    da, a = 0.45, 0.2
    fine = regia.corsa(ascolto, "alzo", da, a, inizio)
    assert fine == inizio + round(abs(a - da) / VELOCITA_DELLA_MIRA * fs) and ascolto.fine == fine and ascolto.eventi == []
    (voce,) = [voce for voce in ascolto.voci if isinstance(voce, VoceTono)]
    assert voce.inizio == inizio and voce.forma == regia.proiettile
    tacche = sorted((voce for voce in ascolto.voci if isinstance(voce, VoceBuffer)), key=lambda tacca: tacca.inizio)
    assert [tacca.inizio for tacca in tacche] == [inizio + round(abs(k / TACCHE - da) / VELOCITA_DELLA_MIRA * fs) for k in (4, 3)]
    assert np.array_equal(tacche[0].buffer, tacca_dell_asse("alzo", 4, fs=fs)) and np.array_equal(tacche[1].buffer, tacca_dell_asse("alzo", 3, fs=fs))


def test_il_tempo_della_mira_coi_tick_e_lo_sparo(finto, chiesti):
    """D29: il passo del tempo è la mira a tempo più breve, TEMPO_DELLA_MIRA_MINIMO secondi: un tick a ogni secondo, quello di
    tick_del_tempo come in partita, sempre più acuto, dal primo secondo fino a un secondo prima della scadenza, e allo scadere lo
    sparo dal cannone, perché il colpo parte da solo. Niente altro suona."""
    motore, _flusso = finto
    fs = motore.fs
    ascolto = _regia(motore).suona(_indice("tempo"))
    secondi = TEMPO_DELLA_MIRA_MINIMO
    scadenza = ascolto.inizio + round(secondi * fs)
    assert ascolto.eventi == [("sparo", scadenza, tuple(float(c) for c in fisica.CANNONE))]
    assert all(isinstance(voce, VoceBuffer) for voce in ascolto.voci)
    tick = [voce for voce in ascolto.voci if voce.inizio != scadenza]
    assert len(tick) == len(ascolto.voci) - 1 == math.ceil(secondi) - 1
    assert [voce.inizio for voce in tick] == [ascolto.inizio + round(secondo * fs) for secondo in range(1, math.ceil(secondi))]
    assert all(np.array_equal(voce.buffer, tick_del_tempo(secondo, secondi, fs=fs)) for secondo, voce in enumerate(tick, 1))
    altezze = [_frequenza_dominante(voce.buffer[:, 0], fs) for voce in tick]
    assert all(dopo > prima for prima, dopo in itertools.pairwise(altezze)), altezze
    assert ascolto.fine == scadenza + CAMPIONI_FINTI


# Il volo, i colpi e il bonus.

def test_il_proiettile_in_volo_dal_cannone_a_terra(finto, chiesti):
    from partita import volo_d_esempio

    motore, flusso = finto
    volo = volo_d_esempio()
    for proiettile in PROIETTILE_FORME:
        ascolto = _regia(motore, proiettile).suona(_indice("volo"))
        # I comandi arrivano al motore col blocco seguente: la fine della voce la mette lui.
        flusso.avanza(1)
        a_terra = ascolto.inizio + round(volo.durata / VELOCITA_DEL_TEMPO * motore.fs)
        assert ascolto.eventi == [("sparo", ascolto.inizio, fisica.CANNONE), ("a_terra", a_terra, tuple(volo.punto_finale.tolist()))]
        (voce,) = [voce for voce in ascolto.voci if isinstance(voce, VoceTono)]
        assert voce.forma == proiettile and voce.inizio == ascolto.inizio
        # La voce comincia al cannone e finisce dove il proiettile tocca terra, con le regole dello spazio.
        frequenze, pan, _guadagni = _parametri(voce, [0.0, volo.durata])
        partenza, arrivo = fisica.suono_di(volo.traiettoria.posizione(0.0)), fisica.suono_di(volo.punto_finale)
        assert frequenze[0] == pytest.approx(partenza[0]) and frequenze[1] == pytest.approx(arrivo[0]) and pan[1] == pytest.approx(arrivo[1])
        assert voce.fine == a_terra


def test_i_colpi_vicino_e_lontano(finto, chiesti):
    """Il centro e il bonus colpiti, ognuno vicino e lontano: ai due capi della tratta della distanza, staccati da ASCOLTI_PAUSA."""
    motore, _flusso = finto
    capi = ascolti.punti_dei_capi()
    ascolto = _regia(motore).suona(_indice("colpi"))
    pausa = round(ASCOLTI_PAUSA * motore.fs)
    attesi, campione = [], ascolto.inizio
    for evento in ("colpito", "colpito_bonus"):
        for capo in ("fronte", "fondo"):
            attesi.append((evento, campione, capi[capo]))
            campione += CAMPIONI_FINTI + pausa
    assert ascolto.eventi == attesi


def test_il_bonus_attraversa_la_cupola_su_una_corda(finto, chiesti):
    """Il bonus compare col suo avviso, entra a un raggio fuori dal lato sinistro, attraversa la cupola su una corda alla quota e alla
    profondità del mezzo, davanti al cannone, ed esce a un raggio fuori dal lato destro nella durata del livello
    ASCOLTI_LIVELLO_DEL_BONUS, col raggio di quel livello; dentro il settore sta tutto nella cupola. All'uscita suona il bonus uscito."""
    motore, flusso = finto
    fs = motore.fs
    ascolto = _regia(motore).suona(_indice("bonus"))
    flusso.avanza(1)
    (voce,), (bonus,) = [voce for voce in ascolto.voci if isinstance(voce, VoceTono)], ascolto.bersagli
    _x, altezza, profondita = ascolti.CENTRO
    assert bonus.bonus and bonus.raggio == pytest.approx(raggio_del_livello(ASCOLTI_LIVELLO_DEL_BONUS) * RAGGIO_BONUS_FATTORE)
    assert np.allclose(bonus.partenza[1:], (altezza, profondita)) and bonus.velocita[0] > 0.0 and np.allclose(bonus.velocita[1:], 0.0)
    durata = durata_del_bonus(ASCOLTI_LIVELLO_DEL_BONUS)
    assert bonus.tempo_di_uscita - bonus.nato == pytest.approx(durata)
    cupola = fisica.Zona(0.0)
    sinistra, destra = fisica.Zona.PARETI.index("sinistra"), fisica.Zona.PARETI.index("destra")
    assert cupola.scarti(bonus.posizione(bonus.nato))[sinistra] == pytest.approx(bonus.raggio)
    assert cupola.scarti(bonus.posizione(bonus.tempo_di_uscita))[destra] == pytest.approx(bonus.raggio)
    assert np.allclose(bonus.posizione(bonus.nato + durata / 2), (0.0, altezza, profondita))
    tempi = np.linspace(bonus.nato, bonus.tempo_di_uscita, 1000)
    scarti = fisica.Zona(bonus.raggio).scarti(bonus.posizione(tempi))
    altre = [indice for indice, parete in enumerate(fisica.Zona.PARETI) if parete not in ("sinistra", "destra")]
    assert np.all(scarti[:, altre] <= 0.0)
    uscita = ascolto.inizio + round(durata / VELOCITA_DEL_TEMPO * fs)
    assert [evento[:2] for evento in ascolto.eventi] == [("bonus_appare", ascolto.inizio), ("bonus_uscito", uscita)]
    assert ascolto.eventi[1][2] == pytest.approx(tuple(bonus.posizione(bonus.tempo_di_uscita))) and voce.fine == uscita
    # Da tutto a sinistra a tutto a destra: la panoramica sale sempre, piena fuori dal settore.
    pan = _parametri(voce, np.linspace(0.0, durata, 400))[1]
    assert pan[0] == -1.0 and pan[-1] == 1.0 and np.all(np.diff(pan) >= 0.0)
    dentro = np.abs(pan) < 1.0
    assert dentro.sum() > len(pan) / 2 and np.all(np.diff(pan[dentro]) > 0.0)


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
    nomi = ("mira", "tempo", "volo", "colpi", "bonus")
    for nome in nomi:
        regia.suona(_indice(nome))
    assert contati == ["passi_ascoltati"] * len(nomi) and suonati == []
    # I preset sono quelli degli eventi del gioco, nella collezione; dalla 2.0.0 il capo della scala, e non più il valore preso.
    assert set(chiesti) == {suoni.EVENTI[evento] for evento in ("mira_avviata", "fine_scala", "sparo", "a_terra", "colpito", "colpito_bonus", "bonus_appare",
                                                                "bonus_uscito")}
    # Senza i suoni della finestra gli effetti tacciono, e le voci dei bersagli, i suoni dei valori e i tick suonano lo stesso.
    muta = ascolti.Regia(motore)
    for nome in ("volo", "mira", "tempo"):
        assert muta.suona(_indice(nome)).eventi == [] and muta.in_corso.voci, nome


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
