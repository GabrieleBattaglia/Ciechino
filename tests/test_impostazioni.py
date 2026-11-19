"""
Le prove di impostazioni.py: i predefiniti, i valori riportati nel loro intervallo, il file e le sue copie.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0. Il conftest dirotta la cartella del programma in una cartella temporanea.
"""

import json

import costanti
import impostazioni


def test_predefinite():
    assert impostazioni.PREDEFINITE == {"volume": costanti.VOLUME_PREDEFINITO, "sintesi": True, "uscita": "automatica", "ultimo_nome": ""}
    assert impostazioni.valide(None) == impostazioni.PREDEFINITE
    assert impostazioni.valide([1, 2]) == impostazioni.PREDEFINITE
    # valide restituisce una copia: cambiarla non tocca i predefiniti.
    copia = impostazioni.valide(None)
    copia["volume"] = 3
    assert impostazioni.PREDEFINITE["volume"] == costanti.VOLUME_PREDEFINITO


def test_valide_riporta_nell_intervallo():
    assert impostazioni.valide({"volume": 150})["volume"] == impostazioni.VOLUME_MASSIMO
    assert impostazioni.valide({"volume": -5})["volume"] == impostazioni.VOLUME_MINIMO
    assert impostazioni.valide({"volume": 55.6})["volume"] == 56
    assert impostazioni.valide({"volume": True})["volume"] == costanti.VOLUME_PREDEFINITO
    assert impostazioni.valide({"volume": "forte"})["volume"] == costanti.VOLUME_PREDEFINITO
    assert impostazioni.valide({"volume": float("nan")})["volume"] == costanti.VOLUME_PREDEFINITO
    assert impostazioni.valide({"sintesi": False})["sintesi"] is False
    assert impostazioni.valide({"sintesi": 0})["sintesi"] is True
    assert impostazioni.valide({"uscita": "  Altoparlanti (Realtek)  "})["uscita"] == "Altoparlanti (Realtek)"
    assert impostazioni.valide({"uscita": "   "})["uscita"] == "automatica"
    assert impostazioni.valide({"uscita": 3})["uscita"] == "automatica"
    assert impostazioni.valide({"ultimo_nome": "  Gabry\t B\x07 "})["ultimo_nome"] == "Gabry B"
    assert impostazioni.valide({"ultimo_nome": "x" * 40})["ultimo_nome"] == "x" * costanti.NOME_MASSIMO
    assert impostazioni.valide({"ultimo_nome": None})["ultimo_nome"] == ""


def test_valide_tiene_le_chiavi_sconosciute():
    dati = impostazioni.valide({"volume": 20, "chiave_del_futuro": [1, 2], "lingua": "en"})
    assert dati["chiave_del_futuro"] == [1, 2]
    assert dati["lingua"] == "en"
    assert dati["volume"] == 20


def test_carica_senza_file(cartella_di_prova):
    assert impostazioni.carica() == impostazioni.PREDEFINITE
    assert impostazioni.ultimo_avviso is None


def test_salva_e_carica(cartella_di_prova):
    assert impostazioni.salva({"volume": 33, "sintesi": False, "uscita": "Cuffie", "ultimo_nome": "Gab", "extra": 1})
    file = cartella_di_prova / costanti.FILE_IMPOSTAZIONI
    assert json.loads(file.read_text(encoding="utf-8"))["volume"] == 33
    assert impostazioni.carica() == {"volume": 33, "sintesi": False, "uscita": "Cuffie", "ultimo_nome": "Gab", "extra": 1}


def test_salva_sistema_i_valori(cartella_di_prova):
    assert impostazioni.salva({"volume": 999})
    assert impostazioni.carica()["volume"] == impostazioni.VOLUME_MASSIMO


def test_percorso_esplicito(tmp_path):
    file = tmp_path / "mie.json"
    assert impostazioni.salva({"volume": 10}, str(file))
    assert impostazioni.carica(str(file))["volume"] == 10


def test_file_rovinato_non_solleva(cartella_di_prova):
    assert impostazioni.salva({"volume": 40})
    assert impostazioni.salva({"volume": 41})
    file = cartella_di_prova / costanti.FILE_IMPOSTAZIONI
    file.write_text("{{{", encoding="utf-8")
    assert impostazioni.carica()["volume"] == 40
    assert "copia di riserva" in impostazioni.ultimo_avviso
    (cartella_di_prova / (costanti.FILE_IMPOSTAZIONI + ".bak")).write_text("rotto", encoding="utf-8")
    file.write_text(json.dumps(7), encoding="utf-8")
    assert impostazioni.carica() == impostazioni.PREDEFINITE
    assert "non contiene le impostazioni" in impostazioni.ultimo_avviso


def test_ripiego_sulla_copia_avvisa_una_volta(cartella_di_prova):
    assert impostazioni.salva({"volume": 40})
    assert impostazioni.salva({"volume": 41})
    file = cartella_di_prova / costanti.FILE_IMPOSTAZIONI
    file.write_text("{{{", encoding="utf-8")
    assert impostazioni.carica()["volume"] == 40
    assert "copia di riserva" in impostazioni.ultimo_avviso
    # Il file torna al suo posto dalla copia, e il caricamento seguente non ha niente da dire.
    assert json.loads(file.read_text(encoding="utf-8"))["volume"] == 40
    assert (cartella_di_prova / (costanti.FILE_IMPOSTAZIONI + ".bak")).exists()
    assert impostazioni.carica()["volume"] == 40
    assert impostazioni.ultimo_avviso is None


def test_intero_enorme_non_solleva(cartella_di_prova):
    enorme = 10 ** 400
    assert impostazioni.valide({"volume": enorme})["volume"] == costanti.VOLUME_PREDEFINITO
    assert impostazioni.valide({"volume": -enorme})["volume"] == costanti.VOLUME_PREDEFINITO
    # Un numero grande ma dentro il confine dei file vale ancora il massimo, come il 120 scritto a mano.
    assert impostazioni.valide({"volume": 1e15})["volume"] == impostazioni.VOLUME_MASSIMO
    file = cartella_di_prova / costanti.FILE_IMPOSTAZIONI
    file.write_text(json.dumps({"volume": enorme, "sintesi": False}), encoding="utf-8")
    dati = impostazioni.carica()
    assert dati["volume"] == costanti.VOLUME_PREDEFINITO and dati["sintesi"] is False
    assert impostazioni.ultimo_avviso is None


def test_salva_su_cartella_che_non_ce(tmp_path):
    assert not impostazioni.salva({"volume": 1}, str(tmp_path / "manca" / "impostazioni.json"))
