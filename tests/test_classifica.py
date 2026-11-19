"""
Le prove di classifica.py: il file con le sue copie, le posizioni, l'inserimento, le righe e i nomi.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0. Il conftest dirotta la cartella del programma in una cartella
temporanea, che le prove ricevono come cartella_di_prova.
"""

import json
from datetime import date, datetime

import classifica
import costanti


def voce(nome, punteggio, data="2026-11-10T12:00:00", **altro):
    return {"nome": nome, "punteggio": punteggio, "data": data, "livello": 1, "tiri": 10, "centri": 5, "durata_s": 60.0, **altro}


def percorso_predefinito(cartella):
    return cartella / costanti.FILE_CLASSIFICA


def test_carica_senza_file(cartella_di_prova):
    assert classifica.carica() == ([], None)


def test_salva_e_carica(cartella_di_prova):
    voci = [voce("Gabry", 500), voce("Gab", 300)]
    assert classifica.salva(voci)
    file = percorso_predefinito(cartella_di_prova)
    contenuto = json.loads(file.read_text(encoding="utf-8"))
    assert contenuto["formato"] == costanti.FORMATO_CLASSIFICA
    assert classifica.carica() == (voci, None)
    # Il secondo salvataggio lascia la copia di riserva, e nessun .tmp.
    assert classifica.salva([voce("Gabry", 600)])
    assert (cartella_di_prova / (costanti.FILE_CLASSIFICA + ".bak")).exists()
    assert not (cartella_di_prova / (costanti.FILE_CLASSIFICA + ".tmp")).exists()


def test_percorso_esplicito(tmp_path):
    file = tmp_path / "altra.json"
    assert classifica.salva([voce("Gabry", 10)], str(file))
    voci, avviso = classifica.carica(str(file))
    assert [v["nome"] for v in voci] == ["Gabry"]
    assert avviso is None


def test_file_rovinato_ripiega_sulla_copia_e_si_rinomina(cartella_di_prova):
    assert classifica.salva([voce("Prima", 100)])
    assert classifica.salva([voce("Seconda", 200)])
    file = percorso_predefinito(cartella_di_prova)
    file.write_text("{ non è JSON", encoding="utf-8")
    voci, avviso = classifica.carica()
    assert [v["nome"] for v in voci] == ["Prima"]
    assert "non si legge" in avviso and "copia di riserva" in avviso
    illeggibile = cartella_di_prova / (costanti.FILE_CLASSIFICA + ".illeggibile")
    assert illeggibile.read_text(encoding="utf-8") == "{ non è JSON"
    assert not file.exists()
    # Il salvataggio seguente non tocca il file rovinato, e la copia di riserva resta quella buona.
    assert classifica.salva(voci)
    assert illeggibile.read_text(encoding="utf-8") == "{ non è JSON"


def test_un_secondo_file_rovinato_non_sovrascrive_il_primo(cartella_di_prova):
    file = percorso_predefinito(cartella_di_prova)
    file.write_text("primo guasto", encoding="utf-8")
    voci, avviso = classifica.carica()
    assert voci == [] and "Riparto da zero" in avviso
    file.write_text("secondo guasto", encoding="utf-8")
    classifica.carica()
    assert (cartella_di_prova / (costanti.FILE_CLASSIFICA + ".illeggibile")).read_text(encoding="utf-8") == "primo guasto"
    assert (cartella_di_prova / (costanti.FILE_CLASSIFICA + ".illeggibile.2")).read_text(encoding="utf-8") == "secondo guasto"


def test_contenuto_sbagliato_e_copia_rovinata(cartella_di_prova):
    file = percorso_predefinito(cartella_di_prova)
    file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (cartella_di_prova / (costanti.FILE_CLASSIFICA + ".bak")).write_text("rotta", encoding="utf-8")
    voci, avviso = classifica.carica()
    assert voci == []
    assert "non contiene una classifica" in avviso
    assert "Riparto da zero" in avviso
    assert (cartella_di_prova / (costanti.FILE_CLASSIFICA + ".bak.illeggibile")).exists()


def test_salvataggio_interrotto_usa_il_tmp(cartella_di_prova):
    tmp = cartella_di_prova / (costanti.FILE_CLASSIFICA + ".tmp")
    tmp.write_text(json.dumps({"formato": 1, "voci": [voce("Recente", 900)]}), encoding="utf-8")
    (cartella_di_prova / (costanti.FILE_CLASSIFICA + ".bak")).write_text(json.dumps({"formato": 1, "voci": [voce("Vecchia", 100)]}), encoding="utf-8")
    voci, avviso = classifica.carica()
    assert [v["nome"] for v in voci] == ["Recente"]
    assert "manca" in avviso and "ultimo salvataggio" in avviso


def test_voci_rovinate_scartate_e_ordinate(cartella_di_prova):
    grezze = [
        voce("Bassa", 10),
        voce("Alta", 900),
        voce("Pari nuova", 500, data="2026-11-15T10:00:00"),
        voce("Pari vecchia", 500, data="2026-11-01T10:00:00"),
        {"nome": "", "punteggio": 5, "data": "2026-11-01T10:00:00"},
        {"nome": "Senza data", "punteggio": 5},
        {"nome": "Punti falsi", "punteggio": True, "data": "2026-11-01T10:00:00"},
        {"nome": "Negativo", "punteggio": -3, "data": "2026-11-01T10:00:00"},
        "non una voce",
        {"nome": "Campi storti", "punteggio": 7.6, "data": "2026-11-01T10:00:00", "livello": "tre", "tiri": -2, "durata_s": "x"},
    ]
    percorso_predefinito(cartella_di_prova).write_text(json.dumps({"formato": 1, "voci": grezze}), encoding="utf-8")
    voci, avviso = classifica.carica()
    assert avviso is None
    assert [v["nome"] for v in voci] == ["Alta", "Pari vecchia", "Pari nuova", "Bassa", "Campi storti"]
    storta = voci[-1]
    assert storta["punteggio"] == 8 and storta["livello"] == 0 and storta["tiri"] == 0 and storta["durata_s"] == 0.0


def test_posizione_a_parita_dietro_le_vecchie():
    voci = [voce("A", 900), voce("B", 500), voce("C", 500), voce("D", 100)]
    assert classifica.posizione(voci, 1000) == 1
    assert classifica.posizione(voci, 500) == 4
    assert classifica.posizione(voci, 501) == 2
    assert classifica.posizione(voci, 1) == 5
    assert classifica.posizione(voci, 0) is None
    assert classifica.posizione([], 0) is None
    assert classifica.posizione([], 1) == 1


def test_posizione_oltre_i_posti():
    piena = [voce(f"G{i}", 1000 - i) for i in range(costanti.POSTI_IN_CLASSIFICA)]
    ultima = piena[-1]["punteggio"]
    assert classifica.posizione(piena, ultima) is None
    assert classifica.posizione(piena, ultima + 1) == costanti.POSTI_IN_CLASSIFICA


def test_inserisci_taglia_e_non_tocca_la_lista():
    piena = [voce(f"G{i}", 1000 - 2 * i) for i in range(costanti.POSTI_IN_CLASSIFICA)]
    nuove, dove = classifica.inserisci(piena, voce("  Gabry\t B  ", 999))
    assert dove == 2
    assert len(nuove) == costanti.POSTI_IN_CLASSIFICA
    assert nuove[1]["nome"] == "Gabry B"
    assert len(piena) == costanti.POSTI_IN_CLASSIFICA
    assert nuove[-1] is piena[-2]
    fuori, dove = classifica.inserisci(piena, voce("Ultimo", 1))
    assert dove is None and fuori == piena


def test_inserisci_mette_la_data_e_il_nome_di_ripiego():
    nuove, dove = classifica.inserisci([], {"nome": "\x07", "punteggio": 50})
    assert dove == 1
    assert nuove[0]["nome"] == "Anonimo"
    assert datetime.fromisoformat(nuove[0]["data"])


def test_nuova_voce_dal_riepilogo():
    riepilogo = {"punteggio": 1234, "livello": 3, "tiri": 40, "centri": 22, "durata_s": 312.25}
    v = classifica.nuova_voce(" Gabry ", riepilogo, data="2026-11-19T18:03:00")
    assert v == {"nome": "Gabry", "punteggio": 1234, "data": "2026-11-19T18:03:00", "livello": 3, "tiri": 40, "centri": 22, "durata_s": 312.25}


def test_righe():
    voci = [voce("Gabry", 12340, data="2026-11-16T09:00:00"), voce("Vecchio", 50, data="2026-11-18T23:59:00"), voce("Gab", 1, data="2026-11-19T08:00:00")]
    righe = classifica.righe(voci, oggi=datetime(2026, 11, 19, 20, 0), nuova=3)
    assert righe == [
        "Classifica, 3 voci, ordinata per punteggio.",
        "1. Gabry, 12.340 punti, 3 giorni fa.",
        "2. Vecchio, 50 punti, 1 giorno fa.",
        "3. Gab, 1 punto, oggi, nuova.",
    ]


def test_righe_con_la_voce_nuova_e_una_data():
    voci, _dove = classifica.inserisci([voce("Gabry", 100)], voce("Nuova", 200, data="2026-11-19T10:00:00"))
    righe = classifica.righe(voci, oggi=date(2026, 11, 19), nuova=voci[0])
    assert righe[0] == "Classifica, 2 voci, ordinata per punteggio."
    assert righe[1] == "1. Nuova, 200 punti, oggi, nuova."
    assert righe[2] == "2. Gabry, 100 punti, 9 giorni fa."


def test_righe_vuote_e_una_voce():
    assert len(classifica.righe([])) == 1
    assert "vuota" in classifica.righe([])[0]
    assert classifica.righe([voce("Solo", 7)], oggi=date(2026, 11, 10))[0] == "Classifica, 1 voce, ordinata per punteggio."


def test_pulisci_nome():
    assert classifica.pulisci_nome("  Gabry  ") == "Gabry"
    assert classifica.pulisci_nome("Ga\x00b\x1bry") == "Gabry"
    assert classifica.pulisci_nome("Gabriele\t\n  Battaglia") == "Gabriele Battaglia"
    assert classifica.pulisci_nome("GaBrY") == "GaBrY"
    assert classifica.pulisci_nome(None) == ""
    assert classifica.pulisci_nome("x" * 50) == "x" * costanti.NOME_MASSIMO
    # Il taglio non lascia uno spazio in fondo.
    assert classifica.pulisci_nome("a" * (costanti.NOME_MASSIMO - 1) + " b") == "a" * (costanti.NOME_MASSIMO - 1)
    assert classifica.nome_valido("Gab")
    assert not classifica.nome_valido(" \t ")


def test_salva_su_cartella_che_non_ce(tmp_path):
    assert not classifica.salva([voce("Gabry", 1)], str(tmp_path / "manca" / "classifica.json"))
