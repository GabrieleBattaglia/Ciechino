"""
Le prove dei suoni di Ciechino nella collezione condivisa di GBUtils: le tabelle di suoni.py,
strumenti/crea_preset.py su una copia della collezione, _rendi_preset e l'elenco dei suoni per
Acu_Maker.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di tests/test_suoni.py di MESS.
La collezione vera non si scrive mai: per tutto il modulo una guardia sostituisce Acusticator.aggiorna,
da cui passano anche le firme, e fa fallire la prova che provasse a scriverla. Lo strumento lavora su
una copia in una cartella temporanea, dalla quale si tolgono prima i preset di Ciechino e la sua
firma: così le prove valgono uguali prima e dopo che lo strumento giri sulla collezione vera.
Nessuna prova suona: i preset si sintetizzano soltanto, per misurarli.
"""

import copy
import json
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
from GBUtils import Acusticator, parse_banda, scomponi_nota

import suoni
from costanti import FS
from strumenti import crea_preset, elenco_suoni

# Gli eventi della 0.1.0, come li elenca docs/progetto.txt: per il gioco, poi per la finestra.
EVENTI_DEL_CONTRATTO = (
    "nuova_partita", "mira_avviata", "valore_preso", "sparo", "colpito", "colpito_bonus", "a_terra", "livello_su", "bonus_appare",
    "bonus_uscito", "ultimo_colpo", "partita_finita", "in_classifica", "fuori_classifica", "pausa", "ripresa", "non_ora",
    "avvio", "uscita", "classifica", "statistiche", "guida", "manuale", "novita", "informazioni", "caffe", "pub", "volume",
    "impostazioni", "vai_alla_vista", "vai_alla_barra", "primo_piano", "errore",
)
# Un buffer che arriva qui è saturo, o quasi: la sintesi taglia a 1, e il motore somma altre voci.
PICCO_MASSIMO = 0.98
# Un buffer che non arriva qui non si sente.
PICCO_MINIMO = 0.05
# Due preset con la stessa onda sono la stessa figura, soltanto trasposta, quando ogni elemento dura
# quanto l'altro entro questi secondi e gli intervalli, dentro gli elementi e fra l'uno e l'altro,
# coincidono entro questi semitoni: sono i margini della prova di MESS.
SCARTO_DI_DURATA = 0.025
SCARTO_DI_INTERVALLO = 0.5
KIND_DI_RUMORE = (5, 6, 7, 8)
# Le altezze in semitoni, col la 4 a 440 hertz, che è il 69 del MIDI.
SEMITONI_PER_OTTAVA = 12
LA_4 = 440.0
LA_4_MIDI = 69
# Una riga di trattini, di uguali o di sottolineature: i separatori grafici che non si scrivono.
SEPARATORE = re.compile(r"-{3,}|={3,}|_{3,}")


def _leggi(percorso):
    with open(percorso, encoding="utf-8") as file:
        return json.load(file)


def _scrivi(percorso, dati):
    """Nella forma in cui la scrive Acusticator: rientro di quattro, accenti in chiaro, un a capo in fondo."""
    with open(percorso, "w", encoding="utf-8") as file:
        json.dump(dati, file, indent=4, ensure_ascii=False)
        file.write("\n")


def _vergine(dati):
    """La collezione senza i preset di Ciechino e senza la sua firma, com'era prima dello strumento."""
    for nome in crea_preset.PRESET:
        dati.pop(nome, None)
    for preset in dati.values():
        if isinstance(preset, dict) and isinstance(preset.get("usato_da"), list):
            firme = [app for app in preset["usato_da"] if str(app).strip().lower() != crea_preset.APP]
            if firme:
                preset["usato_da"] = firme
            else:
                del preset["usato_da"]
    return dati


@pytest.fixture(scope="module", autouse=True)
def guardia():
    """Per tutto il modulo Acusticator.aggiorna rifiuta la collezione vera, e anche il percorso None, che vorrebbe dire
    lei; alla fine Acusticator dimentica le collezioni che ha in memoria, copie comprese."""
    vera = os.path.normcase(os.path.abspath(crea_preset.percorso_condiviso()))
    originale = Acusticator.aggiorna

    def aggiorna(cambia, percorso=None):
        if percorso is None or os.path.normcase(os.path.abspath(percorso)) == vera:
            raise AssertionError(f"una prova ha tentato di scrivere la collezione vera: {percorso!r}")
        return originale(cambia, percorso)

    with pytest.MonkeyPatch.context() as cambi:
        cambi.setattr(Acusticator, "aggiorna", aggiorna)
        yield
    Acusticator.reload()


@pytest.fixture(scope="module")
def originale():
    """La collezione vera, letta e ripulita di Ciechino: la base di ogni copia. Le prove non la cambiano."""
    return _vergine(_leggi(crea_preset.percorso_condiviso()))


def _copia(cartella, dati):
    percorso = cartella / Acusticator.NOME_COLLEZIONE
    _scrivi(percorso, dati)
    return percorso


@pytest.fixture(scope="module")
def creata(tmp_path_factory, originale):
    """Una copia della collezione su cui lo strumento è passato una volta, con il suo esito e la collezione che ne esce."""
    percorso = _copia(tmp_path_factory.mktemp("collezione"), originale)
    esito = crea_preset.crea(str(percorso))
    return SimpleNamespace(percorso=percorso, esito=esito, dati=_leggi(percorso))


def _sintetizza(dati, nome):
    score, kind, adsr = Acusticator.prepara(dati, nome=nome)
    return Acusticator.sintetizza(score, kind, adsr, fs=FS)


def _altezze(nota, rumore):
    """L'altezza d'inizio e quella di fine di un elemento, in semitoni; per il rumore il centro della banda. None se non si legge."""
    if rumore:
        try:
            banda = parse_banda(nota)
        except ValueError:
            return None
        if banda is None or None in banda:
            return None
        basso_inizio, basso_fine, alto_inizio, alto_fine = banda
        centri = (math.sqrt(basso_inizio * alto_inizio), math.sqrt(basso_fine * alto_fine))
        return tuple(SEMITONI_PER_OTTAVA * math.log2(centro / LA_4) + LA_4_MIDI for centro in centri)
    altezze = []
    for pezzo in str(nota).split("."):
        scomposta = scomponi_nota(pezzo)
        if scomposta is None:
            return None
        altezze.append(scomposta[0] + scomposta[1])
    return altezze[0], altezze[-1]


def _figura(preset):
    """L'onda e la figura di un preset senza le pause: per ogni elemento l'altezza d'inizio, quella di fine e la durata."""
    kind = preset["kind"]
    elementi = []
    for nota, durata, _panorama, _volume in preset["score"]:
        if str(nota).lower() == "p":
            continue
        altezze = _altezze(nota, kind in KIND_DI_RUMORE)
        if altezze is None:
            return kind, None
        elementi.append((*altezze, float(durata)))
    return kind, elementi


def _stessa_figura(a, b):
    """Vero se le due figure hanno gli stessi elementi, con le stesse scivolate, gli stessi salti e quasi le stesse durate."""
    if not a or len(a) != len(b):
        return False
    for (inizio_a, fine_a, durata_a), (inizio_b, fine_b, durata_b) in zip(a, b, strict=True):
        if abs(durata_a - durata_b) > SCARTO_DI_DURATA + 1e-9 or abs((fine_a - inizio_a) - (fine_b - inizio_b)) > SCARTO_DI_INTERVALLO:
            return False
    return all(abs((a[k][0] - a[k - 1][1]) - (b[k][0] - b[k - 1][1])) <= SCARTO_DI_INTERVALLO for k in range(1, len(a)))


def test_gli_eventi_del_contratto():
    assert set(suoni.EVENTI) == set(EVENTI_DEL_CONTRATTO)
    assert len(suoni.EVENTI) == len(EVENTI_DEL_CONTRATTO)
    # Nessun evento sta in due gruppi: la mappa piatta li ha tutti.
    assert sum(len(gruppo) for _titolo, gruppo in suoni.GRUPPI) == len(suoni.EVENTI)
    assert isinstance(suoni.GRUPPI, list)
    assert all(isinstance(titolo, str) and titolo.strip() for titolo, _gruppo in suoni.GRUPPI)


def test_ogni_evento_ha_la_sua_azione():
    assert set(suoni.AZIONI) == set(suoni.EVENTI)
    for evento, azione in suoni.AZIONI.items():
        assert isinstance(azione, str) and azione.strip(), evento
        assert not SEPARATORE.search(azione), evento


def test_un_preset_per_evento_e_mai_due_eventi_sullo_stesso():
    preset = list(suoni.EVENTI.values())
    assert len(set(preset)) == len(preset)
    assert suoni.EVENTI[crea_preset.EVENTO_DEL_CAFFE] == crea_preset.PRESET_DEL_CAFFE
    for evento, nome in suoni.EVENTI.items():
        if evento != crea_preset.EVENTO_DEL_CAFFE:
            assert nome.startswith(crea_preset.PREFISSO) and nome in crea_preset.PRESET, evento
    assert set(crea_preset.PRESET) == set(preset) - {crea_preset.PRESET_DEL_CAFFE}


def test_la_tabella_dei_preset_non_ha_problemi():
    assert crea_preset.controlla_tabella() == []


def test_lo_strumento_aggiunge_e_firma_sulla_copia(creata, originale):
    esito = creata.esito
    assert esito.errore is None and not esito.problemi
    assert esito.scritto
    assert esito.aggiunti == list(crea_preset.PRESET)
    assert esito.firmati == crea_preset.nomi_da_firmare()
    assert not esito.sostituiti and not esito.diversi and not esito.mancanti
    for nome in suoni.EVENTI.values():
        assert nome in creata.dati, nome
        assert crea_preset.APP in creata.dati[nome]["usato_da"], nome
    # Donazione tiene le firme degli altri programmi, e aggiunge quella di Ciechino.
    prima = originale[crea_preset.PRESET_DEL_CAFFE].get("usato_da", [])
    assert creata.dati[crea_preset.PRESET_DEL_CAFFE]["usato_da"] == sorted({*prima, crea_preset.APP})


def test_i_preset_nuovi_sono_quelli_della_tabella(creata):
    for nome in crea_preset.PRESET:
        preset = creata.dati[nome]
        assert {chiave: preset[chiave] for chiave in ("descrizione", "score", "kind", "adsr")} == crea_preset.campi(nome)
        # Le firme subito dopo la descrizione, come vuole il formato della collezione.
        assert list(preset) == ["descrizione", "usato_da", "score", "kind", "adsr"]
        assert preset["usato_da"] == [crea_preset.APP]


def test_niente_onda_quadra_salvo_donazione(creata):
    quadre = [nome for nome in suoni.EVENTI.values() if creata.dati[nome].get("kind") == crea_preset.ONDA_QUADRA]
    assert quadre == [crea_preset.PRESET_DEL_CAFFE]


def test_le_descrizioni_cominciano_col_nome_di_ciechino(creata):
    for nome in crea_preset.PRESET:
        descrizione = creata.dati[nome]["descrizione"]
        assert descrizione.startswith(crea_preset.INIZIO_DESCRIZIONE), nome
        assert not SEPARATORE.search(descrizione), nome


@pytest.mark.parametrize("evento", list(suoni.EVENTI))
def test_ogni_preset_si_sintetizza_senza_saturare(creata, evento):
    nome = suoni.EVENTI[evento]
    buffer = _sintetizza(creata.dati[nome], nome)
    assert buffer is not None and buffer.ndim == 2 and buffer.shape[1] == 2 and len(buffer) > 0
    assert buffer.dtype == np.float32
    assert np.all(np.isfinite(buffer))
    picco = float(np.max(np.abs(buffer)))
    assert PICCO_MINIMO < picco < PICCO_MASSIMO, f"{nome}: picco {picco:.3f}"
    # La lunghezza è quella dello score, al campione.
    assert abs(len(buffer) - crea_preset.durata(creata.dati[nome]["score"]) * FS) <= len(creata.dati[nome]["score"])


def test_gli_eventi_frequenti_sono_brevi(creata):
    for evento in crea_preset.EVENTI_FREQUENTI:
        nome = suoni.EVENTI[evento]
        assert crea_preset.durata(creata.dati[nome]["score"]) < crea_preset.DURATA_MASSIMA_FREQUENTI, evento
        assert len(_sintetizza(creata.dati[nome], nome)) / FS < crea_preset.DURATA_MASSIMA_FREQUENTI, evento


def test_sparo_colpiti_e_terra_sono_di_rumore(creata):
    for evento in ("sparo", "colpito", "colpito_bonus", "a_terra"):
        assert creata.dati[suoni.EVENTI[evento]]["kind"] in KIND_DI_RUMORE, evento
    # Ciascuno col suo colore, così gli esiti si distinguono anche per il timbro.
    colori = [creata.dati[suoni.EVENTI[evento]]["kind"] for evento in ("sparo", "colpito", "colpito_bonus", "a_terra")]
    assert len(set(colori)) == len(colori)


def test_nessun_preset_nuovo_e_uguale_a_uno_che_c_era(creata, originale):
    def impronta(preset):
        return json.dumps([preset.get("score"), preset.get("kind"), preset.get("adsr")])

    impronte = {impronta(preset): nome for nome, preset in originale.items() if isinstance(preset, dict)}
    uguali = {nome: impronte[impronta(creata.dati[nome])] for nome in crea_preset.PRESET if impronta(creata.dati[nome]) in impronte}
    assert not uguali, f"preset nuovi uguali a uno che c'era: {uguali}"
    nuove = [impronta(creata.dati[nome]) for nome in crea_preset.PRESET]
    assert len(set(nuove)) == len(nuove)


def test_nessun_preset_degli_eventi_e_la_figura_trasposta_di_un_altro(creata):
    nomi = list(suoni.EVENTI.values())
    figure = {nome: _figura(creata.dati[nome]) for nome in nomi}
    assert all(figure[nome][1] is not None for nome in crea_preset.PRESET)
    coppie = []
    for i, a in enumerate(nomi):
        for b in nomi[i + 1:]:
            (kind_a, figura_a), (kind_b, figura_b) = figure[a], figure[b]
            if kind_a == kind_b and figura_a is not None and figura_b is not None and _stessa_figura(figura_a, figura_b):
                coppie.append((a, b))
    assert not coppie, f"preset che sono la stessa figura trasposta: {coppie}"


def test_lo_strumento_e_idempotente(creata):
    prima = creata.percorso.read_bytes()
    orario = creata.percorso.stat().st_mtime_ns
    esito = crea_preset.crea(str(creata.percorso))
    assert esito.errore is None
    assert not (esito.aggiunti or esito.sostituiti or esito.diversi or esito.firmati or esito.scritto)
    assert creata.percorso.read_bytes() == prima
    assert creata.percorso.stat().st_mtime_ns == orario


def test_gli_altri_preset_restano_intatti(creata, originale):
    dati = creata.dati
    # Quelli che c'erano restano al loro posto, e i nuovi vanno in coda, nell'ordine della tabella.
    assert list(dati)[:len(originale)] == list(originale)
    assert list(dati)[len(originale):] == list(crea_preset.PRESET)
    for nome, prima in originale.items():
        dopo = dati[nome]
        if nome in suoni.EVENTI.values():
            # Donazione cambia soltanto nelle firme.
            assert {k: v for k, v in dopo.items() if k != "usato_da"} == {k: v for k, v in prima.items() if k != "usato_da"}
        else:
            assert dopo == prima, nome
            assert list(dopo) == list(prima), nome


def test_la_prova_non_scrive(tmp_path, originale):
    percorso = _copia(tmp_path, originale)
    prima = percorso.read_bytes()
    esito = crea_preset.crea(str(percorso), prova=True)
    assert esito.prova and esito.errore is None and not esito.scritto
    assert esito.aggiunti == list(crea_preset.PRESET)
    assert esito.firmati == crea_preset.nomi_da_firmare()
    assert percorso.read_bytes() == prima
    assert not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp"))


def test_un_preset_ritoccato_resta_finche_non_si_sostituisce(tmp_path, creata):
    dati = copy.deepcopy(creata.dati)
    dati["ciechino_sparo"]["descrizione"] = "Ciechino, ritoccato con Acu_Maker."
    dati["ciechino_sparo"]["score"][0][3] = 0.1
    percorso = _copia(tmp_path, dati)
    esito = crea_preset.crea(str(percorso))
    assert esito.diversi == ["ciechino_sparo"] and not esito.aggiunti and not esito.sostituiti and not esito.scritto
    assert _leggi(percorso)["ciechino_sparo"] == dati["ciechino_sparo"]
    esito = crea_preset.crea(str(percorso), sostituisci=True)
    assert esito.sostituiti == ["ciechino_sparo"] and esito.scritto
    rimesso = _leggi(percorso)["ciechino_sparo"]
    assert {k: v for k, v in rimesso.items() if k != "usato_da"} == crea_preset.campi("ciechino_sparo")
    assert rimesso["usato_da"] == [crea_preset.APP]
    assert list(rimesso) == ["descrizione", "usato_da", "score", "kind", "adsr"]


def test_la_firma_mancante_si_rimette(tmp_path, creata):
    dati = copy.deepcopy(creata.dati)
    del dati["ciechino_pausa"]["usato_da"]
    percorso = _copia(tmp_path, dati)
    esito = crea_preset.crea(str(percorso))
    assert esito.firmati == ["ciechino_pausa"] and not esito.aggiunti and esito.scritto
    assert _leggi(percorso)["ciechino_pausa"]["usato_da"] == [crea_preset.APP]


def test_una_collezione_che_non_c_e_non_nasce(tmp_path):
    percorso = tmp_path / "manca.json"
    esito = crea_preset.crea(str(percorso))
    assert esito.errore and not esito.scritto
    assert not percorso.exists()


def test_main_con_la_prova(tmp_path, originale, capsys):
    percorso = _copia(tmp_path, originale)
    prima = percorso.read_bytes()
    assert crea_preset.main(["--collezione", str(percorso), "--prova"]) == 0
    uscita = capsys.readouterr().out
    assert "Prova" in uscita and "ciechino_sparo" in uscita
    assert not SEPARATORE.search(uscita)
    assert percorso.read_bytes() == prima


def test_main_scrive_la_copia(tmp_path, originale, capsys):
    percorso = _copia(tmp_path, originale)
    assert crea_preset.main(["--collezione", str(percorso)]) == 0
    assert "Collezione scritta." in capsys.readouterr().out
    assert set(crea_preset.PRESET) <= set(_leggi(percorso))


def test_rendi_preset_sintetizza_una_volta_sola(monkeypatch, creata):
    chiesti = []

    def preset(nome, volume=None, pan=None):
        chiesti.append(nome)
        return Acusticator.prepara(creata.dati[nome], volume=volume, pan=pan, nome=nome)

    monkeypatch.setattr(Acusticator, "preset", preset)
    monkeypatch.setattr(suoni, "_BUFFER_DEI_PRESET", {})
    primo = suoni._rendi_preset("ciechino_sparo")
    secondo = suoni._rendi_preset("ciechino_sparo")
    assert primo is secondo and chiesti == ["ciechino_sparo"]
    assert primo.dtype == np.float32 and primo.ndim == 2 and primo.shape[1] == 2
    assert len(primo) == round(crea_preset.durata(creata.dati["ciechino_sparo"]["score"]) * FS)
    # Il buffer è lo stesso per tutti: chi lo vuole cambiare ne fa una copia.
    assert not primo.flags.writeable
    with pytest.raises(ValueError):
        primo[0, 0] = 1.0


def test_rendi_preset_di_un_preset_che_non_c_e(monkeypatch):
    monkeypatch.setattr(Acusticator, "preset", lambda nome, volume=None, pan=None: (None, None, None))
    monkeypatch.setattr(suoni, "_BUFFER_DEI_PRESET", {})
    buffer = suoni._rendi_preset("ciechino_che_non_esiste")
    assert buffer.shape == (0, 2) and buffer.dtype == np.float32
    assert suoni._rendi_preset("ciechino_che_non_esiste") is buffer


def test_l_elenco_dei_suoni_e_aggiornato():
    percorso = elenco_suoni.percorso()
    assert percorso.exists(), f"manca {elenco_suoni.NOME_DEL_FILE}: si scrive con python strumenti/elenco_suoni.py"
    assert percorso.read_text(encoding="utf-8") == elenco_suoni.testo(), f"{elenco_suoni.NOME_DEL_FILE} non è aggiornato: python strumenti/elenco_suoni.py"


def test_l_elenco_dei_suoni_ha_una_riga_per_azione():
    righe = elenco_suoni.righe()
    assert all(riga.strip() and not SEPARATORE.search(riga) for riga in righe)
    for evento, preset in suoni.EVENTI.items():
        voce = f": {preset}."
        assert sum(riga.endswith(voce) for riga in righe) == 1, evento
    # L'intestazione, poi per ogni gruppo il titolo e le sue voci.
    assert len(righe) == 2 + len(suoni.GRUPPI) + len(suoni.EVENTI)
