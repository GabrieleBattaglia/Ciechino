"""
Le prove di schede.py (D20): l'elenco delle uscite, le etichette con gli avvisi, la scelta da salvare, da
ripulire e da ritrovare, e la rilettura, con GBUtils sostituito.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di tests/test_schede_audio.py di MeTeOra, senza le prove di mpv
e del mixer condiviso, che Ciechino non usa.
Le funzioni audio di GBUtils le sostituisce per tutte le prove il conftest, con una SchedeFinte di
aiuti_audio che le annota: l'elenco finto ricalca quello della macchina di Gabriele, e un elenco chiesto
con una prova di apertura fa fallire la prova invece di arrivare alla scheda.
"""

import GBUtils
import pytest
from aiuti_audio import ALTOPARLANTI, ASIO, CUFFIE, CUFFIE_FINTE, DIGITALE, VIVAVOCE, WDM_KS

import schede


def _uscita(indice):
    return next(uscita for uscita in schede.uscite() if uscita["indice"] == indice)


def test_uscite_in_ordine_di_latenza_con_wdm_ks(schede_finte):
    elenco = schede.uscite()
    # Il modo è quello del motore, a callback, e niente si apre per provarlo.
    assert schede_finte.chiamate == [("elenco", "callback")]
    # Dentro la stessa latenza resta l'ordine di GBUtils: la scheda già in uso davanti, poi per indice.
    assert [uscita["indice"] for uscita in elenco] == [18, 19, 24, 34, 16, 4, 3, 12, 15]
    # A callback WDM-KS si apre, e resta nell'elenco, con la sua esclusiva.
    assert [uscita["dispositivo"] for uscita in elenco if uscita["breve"] == "WDM-KS"] == [WDM_KS, VIVAVOCE.replace("\r\n", " ")]
    assert all(uscita["esclusiva"] == (uscita["breve"] in ("ASIO", "WDM-KS")) for uscita in elenco)
    assert all(set(uscita) == {"indice", "dispositivo", "interfaccia", "breve", "latenza", "esclusiva"} for uscita in elenco)
    assert elenco[0] == {"indice": 18, "dispositivo": ALTOPARLANTI, "interfaccia": "Windows WASAPI", "breve": "WASAPI", "latenza": 3.0, "esclusiva": False}
    # L'a capo nel nome diventa uno spazio.
    assert _uscita(15)["dispositivo"] == "Cuffie (Bluetooth)"
    assert not any("\r" in uscita["dispositivo"] or "\n" in uscita["dispositivo"] for uscita in elenco)
    # Una latenza che manca va in fondo.
    schede_finte.elenco[0] = {**schede_finte.elenco[0], "latenza": None}
    assert schede.uscite()[-1]["indice"] == 18


def test_l_elenco_non_apre_le_schede(schede_finte):
    # La finta del conftest rifiuta l'elenco con una prova di apertura, come farebbe aprire la scheda vera.
    with pytest.raises(AssertionError):
        GBUtils.elenco_dispositivi_audio(prova="predefinito", modo="callback")
    assert schede_finte.chiamate == []


def test_etichette():
    assert schede.etichetta(_uscita(18)) == "Altoparlanti (Realtek(R) Audio), WASAPI, 3 ms"
    assert schede.etichetta(_uscita(16)) == "Realtek ASIO, ASIO, 23,2 ms, esclusiva: può zittire NVDA"
    assert schede.etichetta(_uscita(24)) == "Speakers (Realtek HD Audio output), WDM-KS, 10 ms, esclusiva: può zittire NVDA"
    assert schede.etichetta(_uscita(15)) == "Cuffie (Bluetooth), DirectSound, 120 ms"
    voce = {"indice": 1, "dispositivo": "Scheda", "interfaccia": "Windows WASAPI", "breve": "WASAPI", "latenza": 2.96, "esclusiva": False}
    assert schede.etichetta(voce) == "Scheda, WASAPI, 3 ms"
    assert schede.etichetta({**voce, "latenza": 0.54}) == "Scheda, WASAPI, 0,5 ms"
    assert schede.etichetta({**voce, "latenza": 1234.0}) == "Scheda, WASAPI, 1.234 ms"
    # Senza latenza e senza nome breve: l'interfaccia senza Windows, e niente millesimi.
    assert schede.etichetta({"indice": 1, "dispositivo": "Scheda", "interfaccia": "Windows WASAPI", "latenza": None}) == "Scheda, WASAPI"


def test_etichette_del_vivavoce():
    voce = {"indice": 1, "dispositivo": "Auricolare (WH-1000XM4 Hands-Free AG Audio)", "interfaccia": "Windows WASAPI", "breve": "WASAPI",
            "latenza": 3.0, "esclusiva": False}
    assert schede.etichetta(voce) == "Auricolare (WH-1000XM4 Hands-Free AG Audio), WASAPI, 3 ms, vivavoce: suona come un telefono"
    # Il nome che WDM-KS dà al vivavoce di un telefono, con l'a capo dentro: esclusiva e vivavoce insieme, prima l'esclusiva.
    assert schede.e_vivavoce(VIVAVOCE)
    assert schede.etichetta(_uscita(34)).endswith(", WDM-KS, 10 ms, esclusiva: può zittire NVDA, vivavoce: suona come un telefono")
    assert schede.e_vivavoce("Vivavoce (Jabra)") and schede.e_vivavoce("Headset (HandsFree)")
    assert not schede.e_vivavoce(CUFFIE) and not schede.e_vivavoce(None) and not schede.e_vivavoce("")


def test_nome_della_scheda():
    assert schede.nome_della_scheda(_uscita(16)) == "Realtek ASIO, ASIO"
    # Una scelta salvata non ha il nome breve: l'interfaccia perde Windows davanti.
    assert schede.nome_della_scheda({"dispositivo": DIGITALE, "interfaccia": "Windows WASAPI"}) == f"{DIGITALE}, WASAPI"
    assert schede.nome_della_scheda({"dispositivo": "Cuffie\r\n(Bluetooth)", "interfaccia": "Windows DirectSound"}) == "Cuffie (Bluetooth), DirectSound"
    assert schede.nome_della_scheda({"dispositivo": "Scheda", "interfaccia": None}) == "Scheda"
    # L'automatica, e una scheda senza dispositivo, non hanno nome.
    assert schede.nome_della_scheda({}) == "" and schede.nome_della_scheda(None) == ""
    assert schede.nome_della_scheda({"dispositivo": None, "interfaccia": "MME"}) == ""


def test_valida():
    assert schede.valida({"dispositivo": ALTOPARLANTI, "interfaccia": "Windows WASAPI"}) == {"dispositivo": ALTOPARLANTI, "interfaccia": "Windows WASAPI"}
    # Spazi e a capo si ripuliscono, le chiavi in più se ne vanno.
    assert schede.valida({"dispositivo": " Cuffie\r\n(Bluetooth) ", "interfaccia": "Windows DirectSound ", "indice": 15}) == {
        "dispositivo": "Cuffie (Bluetooth)", "interfaccia": "Windows DirectSound"}
    # Tutto il resto è la scelta automatica.
    for sbagliata in (None, {}, [], "Altoparlanti", {"dispositivo": ALTOPARLANTI}, {"dispositivo": ALTOPARLANTI, "interfaccia": ""},
                      {"dispositivo": "  ", "interfaccia": "MME"}, {"dispositivo": 18, "interfaccia": "Windows WASAPI"}):
        assert schede.valida(sbagliata) == {}
    # Sempre un dizionario nuovo.
    scelta = {"dispositivo": ASIO, "interfaccia": "ASIO"}
    assert schede.valida(scelta) is not scelta


def test_ritrova_e_da_salvare():
    elenco = schede.uscite()
    # Lo stesso nome in tre interfacce: conta la coppia.
    assert schede.ritrova({"dispositivo": ALTOPARLANTI, "interfaccia": "MME"}, elenco)["indice"] == 4
    assert schede.ritrova({"dispositivo": ALTOPARLANTI, "interfaccia": "Windows WASAPI"}, elenco)["indice"] == 18
    assert schede.ritrova({"dispositivo": ALTOPARLANTI, "interfaccia": "ASIO"}, elenco) is None
    assert schede.ritrova({"dispositivo": CUFFIE, "interfaccia": "Windows WASAPI"}, elenco) is None
    assert schede.ritrova({}, elenco) is None
    assert schede.ritrova(None, elenco) is None
    assert schede.ritrova("Altoparlanti", elenco) is None
    # Si salvano i nomi, non l'indice, e la scelta salvata si ritrova.
    for uscita in elenco:
        scelta = schede.da_salvare(uscita)
        assert scelta == {"dispositivo": uscita["dispositivo"], "interfaccia": uscita["interfaccia"]}
        assert schede.valida(scelta) == scelta
        assert schede.ritrova(scelta, elenco) is uscita
    assert schede.da_salvare(None) == {}
    # Un nome scritto a mano con l'a capo si ritrova lo stesso, anche nell'elenco grezzo di GBUtils.
    assert schede.ritrova({"dispositivo": "Cuffie\r\n(Bluetooth)", "interfaccia": "Windows DirectSound"}, elenco)["indice"] == 15
    assert schede.ritrova({"dispositivo": "Cuffie (Bluetooth)", "interfaccia": "Windows DirectSound"}, GBUtils.elenco_dispositivi_audio(prova="nessuno"))["indice"] == 15


def test_rileggi_fa_comparire_le_cuffie(schede_finte):
    scelta = {"dispositivo": CUFFIE, "interfaccia": "Windows WASAPI"}
    schede_finte.arrivate = [CUFFIE_FINTE]
    # Le cuffie accese dopo l'avvio: PortAudio non le vede finché non si rilegge.
    assert schede.ritrova(scelta, schede.uscite()) is None
    esito = schede.rileggi()
    assert esito == {"prima": 9, "dopo": 10, "arrivate": [(CUFFIE, "Windows WASAPI")], "sparite": [], "mixer": None}
    assert schede.ritrova(scelta, schede.uscite())["indice"] == 40
    assert ("rileggi",) in schede_finte.chiamate
    # Un guasto della rilettura arriva a chi la chiede.
    schede_finte.rotta = RuntimeError("il filo del mixer non si è fermato")
    with pytest.raises(RuntimeError):
        schede.rileggi()


def test_nessuna_funzione_scrive_in_console(capsys):
    elenco = schede.uscite()
    for uscita in elenco:
        schede.etichetta(uscita)
        schede.nome_della_scheda(uscita)
        schede.ritrova(schede.da_salvare(uscita), elenco)
    schede.valida({"dispositivo": "x", "interfaccia": "y"})
    schede.rileggi()
    assert capsys.readouterr() == ("", "")
