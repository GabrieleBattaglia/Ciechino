"""
Le prove di statistiche.py: i conti, i primati, le ultime partite, il file con le sue copie e la pagina.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0. Il conftest dirotta la cartella del programma in una cartella
temporanea; l'orologio delle statistiche è finto, così le date delle righe sono note.
"""

import ast
import json
import re
from datetime import datetime
from pathlib import Path

import costanti
import statistiche


class Orologio:
    """Un orologio che va avanti a mano."""

    def __init__(self, quando=datetime(2026, 11, 19, 18, 3)):
        self.quando = quando

    def __call__(self):
        return self.quando


def esito(colpito=True, **altro):
    base = {"colpito": colpito, "bonus": False, "forma": "seno", "mobile": False, "precisione": 0.5 if colpito else None, "punti": 150 if colpito else 0,
            "distanza_dal_vicino": 1.25, "serie": 1 if colpito else 0}
    return {**base, **altro}


def nuove():
    return statistiche.Statistiche(ora=Orologio())


def test_nuove_statistiche_a_zero():
    s = nuove()
    assert s.dati["formato"] == costanti.FORMATO_STATISTICHE
    assert all(v == 0 for v in s.dati["conti"].values())
    assert set(s.dati["mira"]) == set(costanti.PARAMETRI)
    assert set(s.dati["forme"]) == set(costanti.FORME_LFO)
    assert s.dati["dal"] == "2026-11-19T18:03:00"


def test_conta_e_suono():
    s = nuove()
    s.conta("partite_iniziate")
    s.conta("colpi_guadagnati", 10)
    s.conta("chiave_nuova", 2)
    s.conta("secondi_strani", 0.25)
    s.suono("sparo")
    s.suono("sparo")
    s.suono("colpito")
    assert s.dati["conti"]["partite_iniziate"] == 1
    assert s.dati["conti"]["colpi_guadagnati"] == 10
    assert s.dati["conti"]["chiave_nuova"] == 2
    assert s.dati["conti"]["secondi_strani"] == 0.25
    assert s.dati["suoni"] == {"sparo": 2, "colpito": 1}


def test_tiri_e_primati():
    s = nuove()
    assert s.tiro((0.5, 0.4, 0.8), esito(serie=1, precisione=0.5)) == []
    assert s.tiro((0.6, 0.4, 0.8), esito(serie=2, precisione=0.9, punti=300, forma="quadra", mobile=True, distanza_dal_vicino=0.3)) == [
        "precisione", "punti_in_un_tiro", "vicino", "serie"]
    battuti = s.tiro((0.1, 0.1, 0.1), esito(colpito=False, distanza_dal_vicino=2.0))
    assert battuti == []
    assert s.tiro(None, esito(colpito=False, distanza_dal_vicino=0.8)) == ["mancato"]
    conti = s.dati["conti"]
    assert (conti["tiri"], conti["centri"], conti["tiri_a_vuoto"], conti["punti"]) == (4, 2, 2, 450)
    assert s.dati["forme"]["seno"]["colpiti"] == 1 and s.dati["forme"]["quadra"]["colpiti"] == 1
    assert s.dati["moto"]["fermi"]["colpiti"] == 1 and s.dati["moto"]["mobili"]["colpiti"] == 1
    assert s.dati["primati"]["serie"]["valore"] == 2
    assert s.dati["primati"]["precisione"]["valore"] == 0.9
    assert s.dati["primati"]["vicino"]["valore"] == 0.3
    assert s.dati["primati"]["mancato"]["valore"] == 0.8
    assert s.dati["somme"]["centri_misurati"] == 2
    assert s.dati["mira"]["direzione"]["nei_centri"] == 2
    assert abs(s.dati["mira"]["direzione"]["somma_nei_centri"] - 1.1) < 1e-9


def test_tiro_bonus_e_valori_per_parametro():
    s = nuove()
    s.tiro({"alzo": 0.25}, esito(bonus=True))
    assert s.dati["conti"]["bonus_colpiti"] == 1
    assert s.dati["mira"]["alzo"]["nei_centri"] == 1
    assert s.dati["mira"]["direzione"]["nei_centri"] == 0


def test_tiro_con_esito_sbagliato_non_solleva():
    s = nuove()
    s.tiro("niente", "niente")
    s.tiro((1, 2), {"colpito": True, "precisione": "alta", "punti": None, "distanza_dal_vicino": float("nan"), "serie": -1})
    assert s.dati["conti"]["tiri"] == 2
    assert s.dati["conti"]["centri"] == 1


def test_mira_istogramma_e_media():
    s = nuove()
    s.mira("direzione", 0.0, "minimo")
    s.mira("direzione", 0.55, "scelto")
    s.mira("direzione", 0.58, "scelto")
    s.mira("direzione", 1.0, "massimo")
    s.mira("direzione", None, "pausa")
    s.mira("potenza", 1.7, "scelto")
    m = s.dati["mira"]["direzione"]
    assert (m["minimo"], m["scelto"], m["massimo"]) == (1, 2, 1)
    assert m["classi"] == [1, 0, 0, 0, 0, 2, 0, 0, 0, 1]
    assert abs(m["somma"] - 2.13) < 1e-9
    assert s.dati["mira"]["potenza"]["classi"][-1] == 1


def test_bersagli_apparsi():
    s = nuove()
    s.bersaglio_apparso("seno", False, False)
    s.bersaglio_apparso("deriva", True, True)
    s.bersaglio_apparso("forma_futura", True, False)
    assert s.dati["forme"]["seno"]["apparsi"] == 1
    assert s.dati["forme"]["forma_futura"]["apparsi"] == 1
    assert s.dati["moto"]["mobili"]["apparsi"] == 2
    assert s.dati["conti"]["bonus_apparsi"] == 1
    # Senza la portante il timbro non si conta: le chiamate di prima di D18 restano valide.
    assert all(coppia == {"apparsi": 0, "colpiti": 0} for coppia in s.dati["portanti"].values())


def test_ogni_forma_e_ogni_timbro_hanno_il_loro_nome():
    """D9 e D18: le dodici forme dell'LFO e i quattro timbri passano da _, con i nomi del manuale."""
    assert set(statistiche.NOMI_DELLE_FORME) == set(costanti.FORME_LFO)
    assert set(statistiche.NOMI_DELLE_PORTANTI) == set(costanti.BERSAGLIO_PORTANTI)
    assert statistiche.NOMI_DELLE_FORME["gradini"] == "scala a gradini" and statistiche.NOMI_DELLE_FORME["doppia"] == "onda doppia"


def test_i_timbri_dei_bersagli_apparsi_e_colpiti():
    """D18: ogni bersaglio ha un timbro a caso, e le statistiche lo contano, apparsi e colpiti, timbro per timbro."""
    s = nuove()
    assert set(s.dati["portanti"]) == set(costanti.BERSAGLIO_PORTANTI)
    s.bersaglio_apparso("seno", False, False, portante="organo")
    s.bersaglio_apparso("trillo", False, True, portante="sega_morbida")
    s.bersaglio_apparso("gradini", True, False, portante="organo")
    s.tiro((0.5, 0.5, 0.5), esito(forma="trillo", portante="sega_morbida", bonus=True))
    s.tiro((0.5, 0.5, 0.5), esito(colpito=False))
    assert s.dati["portanti"]["organo"] == {"apparsi": 2, "colpiti": 0}
    assert s.dati["portanti"]["sega_morbida"] == {"apparsi": 1, "colpiti": 1}
    righe = s.righe()
    assert "Timbro organo: 2 apparsi, 0 colpiti, 0%." in righe
    assert "Timbro sega morbida: 1 apparso, 1 colpito, 100%." in righe
    assert not any(r.startswith("Timbro seno") for r in righe)
    assert "Forma trillo: 1 apparso, 1 colpito, 100%." in righe and "Forma scala a gradini: 1 apparso, 0 colpiti, 0%." in righe
    # I timbri vengono dopo le forme, nella sezione dei bersagli.
    assert righe.index("Timbro organo: 2 apparsi, 0 colpiti, 0%.") > righe.index("Forma trillo: 1 apparso, 1 colpito, 100%.")
    # Un file di prima di D18, senza i timbri, si legge e li parte da zero; un timbro sconosciuto resta.
    grezzi = dict(s.dati)
    del grezzi["portanti"]
    assert statistiche.Statistiche(dati=grezzi).dati["portanti"]["organo"] == {"apparsi": 0, "colpiti": 0}
    grezzi["portanti"] = {"futuro": {"apparsi": 2, "colpiti": "tanti"}}
    assert statistiche.Statistiche(dati=grezzi).dati["portanti"]["futuro"] == {"apparsi": 2, "colpiti": 0}


def test_partita_finita_e_ultime_partite():
    orologio = Orologio()
    s = statistiche.Statistiche(ora=orologio)
    assert s.partita_finita({"punteggio": 1230, "livello": 3, "tiri": 20, "centri": 12, "durata_s": 252.0, "motivo": "abbandonata"}) == []
    assert s.partita_finita({"punteggio": 2000, "livello": 2, "tiri": 30, "centri": 11, "durata_s": 100.0, "motivo": "riserva_vuota"}) == ["punteggio"]
    assert s.dati["conti"]["partite_abbandonate"] == 1 and s.dati["conti"]["partite_finite"] == 1
    assert s.dati["tempo_di_gioco"] == 352.0
    assert s.dati["primati"]["livello"]["valore"] == 3
    for i in range(costanti.ULTIME_PARTITE + 3):
        s.partita_finita({"punteggio": i, "livello": 1, "tiri": 1, "centri": 0, "durata_s": 1.0, "motivo": "riserva_vuota"})
    assert len(s.dati["ultime"]) == costanti.ULTIME_PARTITE
    assert s.dati["ultime"][-1]["punteggio"] == costanti.ULTIME_PARTITE + 2
    s.partita_finita(None)
    assert s.dati["ultime"][-1]["motivo"] == "riserva_vuota"


def test_salva_e_carica(cartella_di_prova):
    s, avviso = statistiche.Statistiche.carica(ora=Orologio())
    assert avviso is None
    assert s.dati["conti"]["avvii"] == 1
    s.tiro((0.5, 0.5, 0.5), esito())
    s.partita_finita({"punteggio": 150, "livello": 1, "tiri": 1, "centri": 1, "durata_s": 30.0, "motivo": "abbandonata"})
    s.suono("sparo")
    assert s.salva()
    file = cartella_di_prova / costanti.FILE_STATISTICHE
    assert json.loads(file.read_text(encoding="utf-8"))["salvato"] == "2026-11-19T18:03:00"
    di_nuovo, avviso = statistiche.carica()
    assert avviso is None
    assert di_nuovo.dati["conti"]["avvii"] == 2
    assert di_nuovo.dati["conti"]["tiri"] == 1
    assert di_nuovo.dati["suoni"] == {"sparo": 1}
    assert di_nuovo.dati["ultime"] == s.dati["ultime"]
    senza_avvio, _avviso = statistiche.Statistiche.carica(avvio=False)
    assert senza_avvio.dati["conti"]["avvii"] == 1


def test_file_rovinato_ripiega_sulla_copia(cartella_di_prova):
    s, _avviso = statistiche.carica()
    s.conta("tiri", 7)
    assert s.salva()
    assert s.salva()
    file = cartella_di_prova / costanti.FILE_STATISTICHE
    file.write_text("[rotto", encoding="utf-8")
    letto, avviso = statistiche.carica(avvio=False)
    assert letto.dati["conti"]["tiri"] == 7
    assert "copia di riserva" in avviso
    assert (cartella_di_prova / (costanti.FILE_STATISTICHE + ".illeggibile")).read_text(encoding="utf-8") == "[rotto"
    # Il file torna al suo posto dalla copia, e il caricamento seguente non ha niente da dire.
    assert json.loads(file.read_text(encoding="utf-8"))["conti"]["tiri"] == 7
    assert (cartella_di_prova / (costanti.FILE_STATISTICHE + ".bak")).exists()
    di_nuovo, avviso = statistiche.carica(avvio=False)
    assert avviso is None and di_nuovo.dati["conti"]["tiri"] == 7


def test_intero_enorme_non_solleva(cartella_di_prova):
    enorme = 10 ** 400
    grezzi = {"conti": {"tiri": enorme, "centri": 3}, "tempo_di_gioco": enorme, "somme": {"punteggi": enorme},
              "primati": {"punteggio": {"valore": enorme, "quando": "2026-11-18T10:00:00"}}, "suoni": {"sparo": enorme, "colpito": 2},
              "mira": {"alzo": {"minimo": enorme, "classi": [enorme] * statistiche.CLASSI}}, "forme": {"seno": {"apparsi": enorme}},
              "ultime": [{"quando": "2026-11-18T10:00:00", "punteggio": enorme, "durata_s": enorme}]}
    (cartella_di_prova / costanti.FILE_STATISTICHE).write_text(json.dumps(grezzi), encoding="utf-8")
    letto, avviso = statistiche.carica(avvio=False)
    assert avviso is None
    assert letto.dati["conti"]["tiri"] == 0
    assert letto.dati["conti"]["centri"] == 3
    assert letto.dati["tempo_di_gioco"] == 0.0 and letto.dati["somme"]["punteggi"] == 0
    assert "punteggio" not in letto.dati["primati"]
    assert letto.dati["suoni"] == {"colpito": 2}
    assert letto.dati["mira"]["alzo"]["minimo"] == 0 and letto.dati["mira"]["alzo"]["classi"] == [0] * statistiche.CLASSI
    assert letto.dati["forme"]["seno"]["apparsi"] == 0
    assert letto.dati["ultime"][0]["punteggio"] == 0
    # La pagina si compone senza sollevare, e un tiro con numeri enormi non entra nei conti.
    assert letto.righe()
    letto.tiro((enorme, 0.5, 0.5), esito(precisione=enorme, punti=enorme, distanza_dal_vicino=enorme, serie=enorme))
    assert letto.dati["conti"]["punti"] == 0 and "serie" not in letto.dati["primati"]


def test_file_e_copia_rovinati_ripartono_da_zero(cartella_di_prova):
    file = cartella_di_prova / costanti.FILE_STATISTICHE
    file.write_text(json.dumps("non un dizionario"), encoding="utf-8")
    (cartella_di_prova / (costanti.FILE_STATISTICHE + ".bak")).write_text("rotta", encoding="utf-8")
    letto, avviso = statistiche.carica()
    assert letto.dati["conti"]["avvii"] == 1
    assert letto.dati["conti"]["tiri"] == 0
    assert "non contiene le statistiche" in avviso and "Riparto da zero" in avviso


def test_completa_risana_i_tipi_sbagliati():
    grezzi = {
        "dal": "non una data",
        "conti": {"tiri": 5, "centri": -1, "avvii": "tre", "chiave_futura": 4, "falso": True},
        "tempo_di_gioco": "molto",
        "somme": [],
        "mira": {"alzo": {"minimo": 2, "classi": [1, 2]}, "direzione": "rotta"},
        "forme": {"seno": {"apparsi": 3, "colpiti": "x"}, "nuova": {"apparsi": 1}},
        "moto": {"fermi": None},
        "primati": {"punteggio": {"valore": 10, "quando": "2026-11-18T10:00:00"}, "rotto": {"valore": 1, "quando": "ieri"}},
        "ultime": [{"quando": "2026-11-18T10:00:00", "punteggio": 10, "motivo": "boh"}, {"quando": "no"}, 3],
        "suoni": {"sparo": 2.0, "rotto": "x"},
    }
    dati = statistiche.completa(grezzi, "2026-11-19T18:03:00")
    assert dati["dal"] == "2026-11-19T18:03:00"
    assert dati["conti"]["tiri"] == 5 and dati["conti"]["centri"] == 0 and dati["conti"]["avvii"] == 0
    assert dati["conti"]["chiave_futura"] == 4 and "falso" not in dati["conti"]
    assert dati["tempo_di_gioco"] == 0.0
    assert dati["mira"]["alzo"]["minimo"] == 2 and dati["mira"]["alzo"]["classi"] == [0] * statistiche.CLASSI
    assert dati["mira"]["direzione"]["minimo"] == 0
    assert dati["forme"]["seno"] == {"apparsi": 3, "colpiti": 0}
    assert dati["forme"]["nuova"] == {"apparsi": 1, "colpiti": 0}
    assert list(dati["primati"]) == ["punteggio"]
    assert len(dati["ultime"]) == 1 and dati["ultime"][0]["motivo"] == "riserva_vuota"
    assert dati["suoni"] == {"sparo": 2}
    assert statistiche.completa("rotto", "2026-11-19T18:03:00")["conti"]["tiri"] == 0


def test_righe_vuote():
    righe = nuove().righe()
    assert righe[0] == f"Statistiche di Ciechino {costanti.VERSIONE}, raccolte da giovedì 19 novembre 2026: 1 giorno, oggi compreso."
    assert righe[1].startswith("Le sezioni: 1 In breve; 2 Le partite;")
    titoli = [r for r in righe if r.startswith("Sezione ")]
    assert titoli[0] == "Sezione 1 di 8, in breve."
    assert len(titoli) == 8
    assert "Ancora niente da dire." in righe


def test_righe_piene():
    orologio = Orologio()
    s = statistiche.Statistiche(ora=orologio)
    s.conta("avvii")
    s.conta("partite_iniziate", 2)
    s.conta("mire_avviate", 3)
    s.conta("caratteristica_nuova", 4)
    s.bersaglio_apparso("seno", False, False)
    s.bersaglio_apparso("sega_inversa", True, True)
    s.mira("direzione", 0.55, "scelto")
    s.mira("alzo", 0.0, "minimo")
    s.mira("potenza", 1.0, "massimo")
    s.tiro((0.55, 0.0, 1.0), esito(precisione=0.64, punti=12340, distanza_dal_vicino=0.05, serie=1))
    s.tiro((0.2, 0.2, 0.2), esito(colpito=False, distanza_dal_vicino=3.2))
    s.suono("sparo")
    s.suono("sparo")
    s.suono("colpito")
    orologio.quando = datetime(2026, 11, 21, 9, 5)
    s.partita_finita({"punteggio": 12340, "livello": 1, "tiri": 2, "centri": 1, "durata_s": 192.0, "motivo": "abbandonata"})
    righe = s.righe(nomi_dei_suoni={"sparo": "lo sparo del cannone"})
    testo = "\n".join(righe)
    assert righe[0].endswith("raccolte da giovedì 19 novembre 2026: 3 giorni, oggi compreso.")
    assert "Sezione 9 di 9, gli altri conti." in righe
    assert "Caratteristica nuova: 4." in righe
    assert "Centri: 1, 50% dei tiri." in righe
    assert "Precisione media dei centri: 64%." in righe
    assert "Punti fatti: 12.340." in righe
    assert "Partite abbandonate: 1." in righe
    assert "Partite iniziate e non concluse, perché sono in corso o il programma si è chiuso: 1." in righe
    assert "Durata media di una partita: 3 minuti e 12 secondi." in righe
    assert "Direzione, valori presi: 1." in righe
    assert "Direzione, scelti con lo spazio nella scivolata: 1, 100%." in righe
    assert "Alzo, fermati al minimo nei bip: 1, 100%." in righe
    # Da D25 il massimo si prende soltanto con lo spazio, appena finita la scivolata.
    assert "Potenza, presi al massimo con lo spazio appena finita la scivolata: 1, 100%." in righe
    assert "Direzione, valore medio: 55% della scala." in righe
    assert "Direzione, valori per fasce della scala: da 50 a 60%: 1." in righe
    assert "Bersagli mobili: 1 apparso, 0 colpiti, 0%." in righe
    assert "Forma dente di sega inverso: 1 apparso, 0 colpiti, 0%." in righe
    assert "Forma seno: 1 apparso, 1 colpito, 100%." in righe
    assert "Punteggio più alto in una partita: 12.340 punti, sabato 21 novembre 2026." in righe
    assert "Il tiro passato più vicino al centro di un bersaglio: 0,05 metri dal centro, giovedì 19 novembre 2026." in righe
    assert "Il tiro a vuoto passato più vicino al centro di un bersaglio: 3,20 metri dal centro, giovedì 19 novembre 2026." in righe
    assert "Sabato 21 novembre 2026 alle 9:05: 12.340 punti, livello 1, 1 centro su 2 tiri, 3 minuti e 12 secondi, abbandonata." in righe
    assert "Eventi sonori in tutto: 3." in righe
    assert "Lo sparo del cannone: 2." in righe
    assert "Colpito: 1." in righe
    # Niente righe vuote, niente separatori grafici, nessun segnaposto rimasto.
    assert all(r.strip() for r in righe)
    assert not re.search(r"[-_=]{3,}", testo)
    assert "{" not in testo and "}" not in testo


def test_contatori_della_finestra_e_delle_pause_con_la_loro_etichetta():
    s = nuove()
    s.conta("avvii")
    s.conta("lingua_cambiata")
    s.conta("mail_al_pub_pronte", 3)
    s.conta("registri_al_pub", 2)
    s.conta("ingressi_in_classifica", 4)
    s.conta("pause", 5)
    s.conta("pause_automatiche", 6)
    s.conta("proiettile_cambiato", 7)
    s.conta("partite_col_seno", 8)
    s.conta("partite_col_triangolo", 9)
    s.conta("partite_con_la_sega_morbida", 10)
    s.conta("scheda_cambiata", 11)
    s.conta("scheda_non_aperta", 12)
    s.conta("pause_rifiutate", 13)
    s.conta("attese_della_palestra", 14)
    s.conta("ascolti_aperti", 15)
    s.conta("ascolti_d_avvio", 16)
    s.conta("passi_ascoltati", 17)
    s.conta("ascolti_finiti", 18)
    righe = s.righe()
    # Gli Ascolti (D23), in breve, dopo le schede audio e prima delle mail al PUB, nell'ordine.
    ascolti = ["Aperture degli Ascolti: 15.", "Ascolti aperti da soli all'avvio, prima della prima partita: 16.", "Passi degli Ascolti ascoltati: 17.",
               "Ascolti seguiti fino all'ultimo passo: 18."]
    primo = righe.index(ascolti[0])
    assert righe[primo:primo + 4] == ascolti and righe[primo - 1] == "Schede audio scelte che non si sono aperte: 12." and righe[primo + 4] == "Mail al PUB preparate: 3."
    for riga in ("Cambi di lingua: 1.", "Mail al PUB preparate: 3.", "Mail d'aiuto al PUB coi registri allegati: 2.", "Partite entrate in classifica: 4.",
                 "Pause chieste con Ctrl+P: 5.", "Pause automatiche, per un dialogo o un menu aperto o un'altra finestra: 6.", "Cambi della forma del proiettile: 7.",
                 "Partite col proiettile in seno: 8.", "Partite col proiettile in triangolo: 9.", "Partite col proiettile in sega morbida: 10.",
                 "Cambi della scheda audio: 11.", "Schede audio scelte che non si sono aperte: 12.",
                 "Pause rifiutate, chieste con Ctrl+P fuori dalla palestra: 13.", "Attese della palestra dopo un tiro: 14."):
        assert riga in righe
    # D22: le due voci della palestra stanno con le pause, il rifiuto subito dopo quelle chieste, le attese dopo quelle automatiche.
    assert righe.index("Pause rifiutate, chieste con Ctrl+P fuori dalla palestra: 13.") == righe.index("Pause chieste con Ctrl+P: 5.") + 1
    assert righe.index("Attese della palestra dopo un tiro: 14.") == righe.index("Pause automatiche, per un dialogo o un menu aperto o un'altra finestra: 6.") + 1
    # Il cambio del proiettile sta in breve, subito dopo quello della lingua (D17), e dopo di lui le due voci della scheda audio (D20).
    assert righe.index("Cambi della forma del proiettile: 7.") == righe.index("Cambi di lingua: 1.") + 1
    assert righe.index("Cambi della scheda audio: 11.") == righe.index("Cambi della forma del proiettile: 7.") + 1
    assert righe.index("Schede audio scelte che non si sono aperte: 12.") == righe.index("Cambi della scheda audio: 11.") + 1
    # Nessuno di loro finisce negli altri conti, che restano per le chiavi nuove.
    assert not any("altri conti" in r for r in righe)
    assert [r for r in righe if r.startswith("Sezione ")][-1] == "Sezione 8 di 8, i suoni."


def test_la_mira_di_d25_con_le_sue_etichette():
    """D25: le mire annullate con Esc, le barre ripartite senza lo spazio e i proiettili perduti stanno nei tiri, dopo le mire avviate."""
    s = nuove()
    s.conta("mire_avviate", 9)
    s.conta("mire_annullate", 2)
    s.conta("barre_ripartite", 7)
    s.conta("proiettili_perduti", 1)
    righe = s.righe()
    mire = righe.index("Mire avviate: 9.")
    assert righe[mire + 1:mire + 4] == ["Mire annullate con Esc: 2.", "Barre della mira ripartite da capo senza lo spazio: 7.",
                                        "Proiettili perduti, con una barra mai fermata dallo spazio: 1."]
    assert not any("altri conti" in r for r in righe)


def test_la_precisione_globale(cartella_di_prova):
    """D26: la precisione globale di ogni tiro dà la media dei tiri e il primato della migliore; quella media di ogni partita va fra
    le ultime partite, dà la media per partita e il suo primato. Un tiro o una partita senza, come il proiettile perduto o le
    partite di prima di D26, non contano nelle medie, e i dati di prima si leggono."""
    orologio = Orologio()
    s = statistiche.Statistiche(ora=orologio)
    assert s.tiro((0.5, 0.5, 0.5), esito(precisione_globale=0.95)) == []
    assert s.tiro((0.1, 0.1, 0.1), esito(colpito=False, distanza_dal_vicino=9.0, precisione_globale=0.8)) == []
    assert s.tiro((0.5, 0.5, 0.5), esito(precisione=0.6, precisione_globale=0.975, distanza_dal_vicino=0.4)) == ["precisione", "vicino", "precisione_globale"]
    assert s.tiro((0.2, 0.2, 0.2), esito(colpito=False, distanza_dal_vicino=None, precisione_globale=None)) == []
    somme = s.dati["somme"]
    assert somme["tiri_col_globale"] == 3 and abs(somme["precisione_globale"] - 2.725) < 1e-9
    assert s.dati["primati"]["precisione_globale"]["valore"] == 0.975
    assert s.partita_finita({"punteggio": 300, "livello": 1, "tiri": 4, "centri": 2, "durata_s": 60.0, "motivo": "abbandonata",
                             "precisione_globale": 0.90833333}) == []
    assert s.partita_finita({"punteggio": 100, "livello": 1, "tiri": 1, "centri": 1, "durata_s": 10.0, "motivo": "riserva_vuota",
                             "precisione_globale": 0.95}) == ["precisione_globale_media"]
    s.partita_finita({"punteggio": 0, "livello": 1, "tiri": 1, "centri": 0, "durata_s": 5.0, "motivo": "riserva_vuota", "precisione_globale": None})
    s.partita_finita({"punteggio": 0, "livello": 1, "tiri": 1, "centri": 0, "durata_s": 5.0, "motivo": "riserva_vuota", "precisione_globale": 1.7})
    assert [p["precisione_globale"] for p in s.dati["ultime"]] == [0.9083, 0.95, None, None]
    assert somme["partite_col_globale"] == 2 and abs(somme["globale_delle_partite"] - 1.8583) < 1e-9
    righe = s.righe()
    assert "Precisione globale media dei tiri: 90,8%." in righe
    assert "Precisione globale media per partita: 92,9%." in righe
    assert "La precisione globale migliore in un tiro: 97,5%, giovedì 19 novembre 2026." in righe
    assert "La precisione globale media migliore in una partita: 95,0%, giovedì 19 novembre 2026." in righe
    assert "Giovedì 19 novembre 2026 alle 18:03: 100 punti, livello 1, 1 centro su 1 tiro, precisione globale media 95,0%, 10 secondi, finita a riserva vuota." in righe
    assert "Giovedì 19 novembre 2026 alle 18:03: 0 punti, livello 1, 0 centri su 1 tiro, 5 secondi, finita a riserva vuota." in righe
    # I file di prima di D26, senza le somme nuove e senza la precisione delle ultime partite, si leggono da zero.
    grezzi = json.loads(json.dumps(s.dati))
    for chiave in ("precisione_globale", "tiri_col_globale", "globale_delle_partite", "partite_col_globale"):
        del grezzi["somme"][chiave]
    for partita in grezzi["ultime"]:
        del partita["precisione_globale"]
    letto = statistiche.Statistiche(dati=grezzi, ora=orologio)
    assert letto.dati["somme"]["tiri_col_globale"] == 0 and all(p["precisione_globale"] is None for p in letto.dati["ultime"])
    assert letto.righe()


def test_ogni_chiave_dei_conti_ha_la_sua_riga():
    s = nuove()
    for chiave in statistiche.CHIAVI_DEI_CONTI:
        s.conta(chiave, 3)
    righe = s.righe()
    assert not any("altri conti" in r for r in righe)


def _chiavi_contate(percorso):
    """Le chiavi scritte per esteso nelle chiamate a conta e _conta di un modulo: ogni testo nel
    primo argomento, anche dentro un'espressione come «"a" if x else "b"»."""
    albero = ast.parse(percorso.read_text(encoding="utf-8"))
    chiavi = set()
    for nodo in ast.walk(albero):
        if not isinstance(nodo, ast.Call) or not nodo.args:
            continue
        funzione = nodo.func
        nome = funzione.attr if isinstance(funzione, ast.Attribute) else getattr(funzione, "id", None)
        if nome not in ("conta", "_conta"):
            continue
        chiavi.update(c.value for c in ast.walk(nodo.args[0]) if isinstance(c, ast.Constant) and isinstance(c.value, str))
    return chiavi


def test_ogni_chiave_contata_dalla_partita_e_dalla_finestra_ha_un_etichetta():
    radice = Path(statistiche.__file__).resolve().parent
    contate = _chiavi_contate(radice / "partita.py") | _chiavi_contate(radice / "gui" / "finestra.py") | _chiavi_contate(radice / "ascolti.py")
    # La prova deve trovare davvero le chiamate: se non ne trova, il modo di contare è cambiato.
    assert {"partite_iniziate", "ingressi_in_classifica", "proiettile_cambiato", "scheda_cambiata", "scheda_non_aperta"} <= contate
    # Gli Ascolti (D23): la finestra conta le aperture, quelle dell'avvio e quelli finiti, la regia i passi.
    assert {"ascolti_aperti", "ascolti_d_avvio", "ascolti_finiti", "passi_ascoltati"} <= contate
    assert contate - set(statistiche.CHIAVI_DEI_CONTI) == set()


def test_i_conti_delle_partite_per_forma_del_proiettile_hanno_un_etichetta():
    """La partita conta le partite di ogni forma del proiettile con una chiave presa da CONTI_DEL_PROIETTILE: la prova
    con ast non la vede scritta per esteso, e la guarda qui (D17)."""
    import partita

    assert set(partita.CONTI_DEL_PROIETTILE) == set(costanti.PROIETTILE_FORME)
    assert set(partita.CONTI_DEL_PROIETTILE.values()) <= set(statistiche.CHIAVI_DEI_CONTI)
