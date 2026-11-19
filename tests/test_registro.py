"""
Le prove del registro degli errori e dei crash, e dei registri che la richiesta d'aiuto al PUB allega,
nella cartella temporanea di ogni prova: senza avvia non fa niente; avvia senza cartella scrive in
quella del programma; problema scrive il resoconto con la pila; il registro gira con la sua copia e
non supera la misura; crash_vero riconosce i resoconti fatali e ignora le eccezioni di Windows gestite;
chiudi toglie il file del crash; il crash della volta prima si mette da parte, e se non si può resta
nel file del crash, che l'uscita non toglie; lo stderr che il programma compilato non ha diventa il
registro. Poi pub.py: quali registri si allegano, in che ordine, e ripuliti dai nomi di utente e
computer. Alla fine di ogni prova il registro si chiude, e faulthandler torna com'era per pytest.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di test_registro.py di MESS.
"""

import faulthandler
import os
import sys

import pytest

import pub
import registro

FATALE = "Fatal Python error: Segmentation fault\n\nThread 0x00001234 (most recent call first):\n  File \"motore_audio.py\", line 1 in _callback\n"
GESTITA = "Windows fatal exception: code 0x8001010d\n\nCurrent thread 0x00001234 (most recent call first):\n  File \"suoni.py\", line 9 in suona\n"


@pytest.fixture(autouse=True)
def registro_chiuso():
    acceso = faulthandler.is_enabled()
    yield
    registro.chiudi()
    if acceso and not faulthandler.is_enabled():
        faulthandler.enable(file=sys.__stderr__)


def _leggi(percorso):
    return percorso.read_text(encoding="utf-8")


def test_i_nomi_dei_file():
    assert registro.FILE_DEGLI_ERRORI == "Ciechino - Errori.log"
    assert registro.FILE_DEL_CRASH == "Ciechino - Crash.log"
    assert registro.FILE_DEL_CRASH_PRECEDENTE == "Ciechino - Crash precedente.log"


def test_senza_avvia_non_fa_niente(tmp_path):
    assert not registro.avviato()
    registro.scrivi("una riga")
    registro.problema(ValueError, ValueError("prova"), None)
    registro.chiudi()
    assert list(tmp_path.iterdir()) == []


def test_avvia_scrive_nella_cartella_del_programma(cartella_di_prova):
    # Il conftest fa della cartella temporanea la cartella del programma.
    assert registro.avvia() is None and registro.avviato()
    assert (cartella_di_prova / registro.FILE_DEL_CRASH).exists()
    registro.scrivi("Ciechino 0.1.0 si avvia.")
    registro.chiudi()
    assert not registro.avviato()
    assert "Ciechino 0.1.0 si avvia." in _leggi(cartella_di_prova / registro.FILE_DEGLI_ERRORI)


def test_problema_scrive_la_pila(tmp_path):
    assert registro.avvia(str(tmp_path)) is None
    registro.scrivi("Ciechino 0.1.0 si avvia.")
    try:
        raise ValueError("prova")
    except ValueError:
        registro.problema(*sys.exc_info())
    registro.chiudi()
    testo = _leggi(tmp_path / registro.FILE_DEGLI_ERRORI)
    assert "Ciechino 0.1.0 si avvia." in testo and "Traceback (most recent call last):" in testo
    assert "in test_problema_scrive_la_pila" in testo and testo.rstrip().endswith("ValueError: prova")


def test_il_registro_gira_e_non_supera_la_misura(tmp_path):
    misura = 4000
    registro.avvia(str(tmp_path), misura=misura)
    for numero in range(200):
        registro.scrivi(f"Riga {numero} " + "x" * 80)
    registro.chiudi()
    errori = tmp_path / registro.FILE_DEGLI_ERRORI
    copia = tmp_path / f"{registro.FILE_DEGLI_ERRORI}.1"
    assert errori.exists() and copia.exists()
    # Una copia sola: niente .log.2.
    assert not (tmp_path / f"{registro.FILE_DEGLI_ERRORI}.2").exists()
    assert errori.stat().st_size + copia.stat().st_size <= misura
    # L'ultima riga scritta è nel registro, la più vecchia se n'è andata.
    assert "Riga 199 " in _leggi(errori) and "Riga 0 " not in _leggi(errori) + _leggi(copia)


@pytest.mark.parametrize(("testo", "vero"), [
    (FATALE, True),
    ("Windows fatal exception: access violation\n", True),
    ("Windows fatal exception: stack overflow\n", True),
    ("Windows fatal exception: code 0xc0000374\n", True),
    (GESTITA, False),
    ("Windows fatal exception: code 0xe24c4a02\n", False),
    ("Windows fatal exception: code 0x406d1388\n", False),
    ("", False),
    (GESTITA + FATALE, True),
])
def test_crash_vero(testo, vero):
    assert registro.crash_vero(testo) is vero


def test_chiudi_toglie_il_file_del_crash(tmp_path):
    registro.avvia(str(tmp_path))
    assert (tmp_path / registro.FILE_DEL_CRASH).exists()
    registro.chiudi()
    assert not (tmp_path / registro.FILE_DEL_CRASH).exists()
    # Senza errori, il registro degli errori non nasce nemmeno.
    assert not (tmp_path / registro.FILE_DEGLI_ERRORI).exists()


def test_il_crash_precedente_si_mette_da_parte(tmp_path):
    (tmp_path / registro.FILE_DEL_CRASH).write_text(FATALE, encoding="utf-8")
    precedente = registro.avvia(str(tmp_path))
    assert precedente == str(tmp_path / registro.FILE_DEL_CRASH_PRECEDENTE)
    assert _leggi(tmp_path / registro.FILE_DEL_CRASH_PRECEDENTE) == FATALE
    assert _leggi(tmp_path / registro.FILE_DEL_CRASH) == ""
    registro.chiudi()
    assert not (tmp_path / registro.FILE_DEL_CRASH).exists() and (tmp_path / registro.FILE_DEL_CRASH_PRECEDENTE).exists()


def test_un_crash_nuovo_prende_il_posto_del_vecchio(tmp_path):
    (tmp_path / registro.FILE_DEL_CRASH_PRECEDENTE).write_text("Fatal Python error: il vecchio\n", encoding="utf-8")
    (tmp_path / registro.FILE_DEL_CRASH).write_text(FATALE, encoding="utf-8")
    registro.avvia(str(tmp_path))
    assert _leggi(tmp_path / registro.FILE_DEL_CRASH_PRECEDENTE) == FATALE


def test_le_eccezioni_gestite_non_sono_un_crash(tmp_path):
    (tmp_path / registro.FILE_DEL_CRASH).write_text(GESTITA, encoding="utf-8")
    assert registro.avvia(str(tmp_path)) is None
    assert _leggi(tmp_path / registro.FILE_DEL_CRASH) == ""
    assert not (tmp_path / registro.FILE_DEL_CRASH_PRECEDENTE).exists()


def test_il_resoconto_che_non_si_sposta_non_si_perde(tmp_path, monkeypatch):
    crash = tmp_path / registro.FILE_DEL_CRASH
    crash.write_text(FATALE, encoding="utf-8")
    monkeypatch.setattr(registro, "_metti_da_parte", lambda *_argomenti: None)
    assert registro.avvia(str(tmp_path)) == str(crash)
    assert _leggi(crash) == FATALE
    registro.chiudi()
    # L'uscita pulita lo riporta al solo resoconto e lo lascia, per il prossimo avvio.
    assert _leggi(crash) == FATALE
    assert f"resta in {registro.FILE_DEL_CRASH}" in _leggi(tmp_path / registro.FILE_DEGLI_ERRORI)


def test_lo_stderr_del_programma_compilato(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stderr", None)
    registro.avvia(str(tmp_path))
    assert sys.stderr is not None
    print("Un avviso di una libreria", file=sys.stderr)
    print("su due righe", file=sys.stderr)
    registro.chiudi()
    assert sys.stderr is None
    testo = _leggi(tmp_path / registro.FILE_DEGLI_ERRORI)
    assert "Un avviso di una libreria" in testo and "su due righe" in testo


def test_il_registro_che_non_si_scrive_non_va_in_ricorsione(tmp_path, monkeypatch):
    # Al posto del file del registro una cartella: ogni scrittura fallisce, e logging lo racconta su
    # stderr, che nel programma compilato è il registro stesso.
    (tmp_path / registro.FILE_DEGLI_ERRORI).mkdir()
    monkeypatch.setattr(sys, "stderr", None)
    registro.avvia(str(tmp_path))
    assert sys.stderr.encoding == "utf-8" and not sys.stderr.isatty()
    print("Un avviso di una libreria", file=sys.stderr)
    registro.scrivi("Ciechino 0.1.0 si avvia.")
    registro.problema(ValueError, ValueError("prova"), None)
    registro.chiudi()
    assert sys.stderr is None


# I registri per il PUB.

def test_i_registri_da_allegare_e_il_loro_ordine(tmp_path):
    assert pub.registri_presenti(str(tmp_path)) == []
    assert pub.scrivi_i_registri(str(tmp_path), "0.1.0", "Gabriele", "GABRYBAT") is None
    (tmp_path / registro.FILE_DEGLI_ERRORI).write_text("l'errore nuovo\n", encoding="utf-8")
    (tmp_path / f"{registro.FILE_DEGLI_ERRORI}.1").write_text("l'errore vecchio\n", encoding="utf-8")
    (tmp_path / registro.FILE_DEL_CRASH_PRECEDENTE).write_text(FATALE, encoding="utf-8")
    # Un registro vuoto non si allega, e il crash di adesso nemmeno: è aperto da faulthandler.
    (tmp_path / registro.FILE_DEL_CRASH).write_text("", encoding="utf-8")
    presenti = [os.path.basename(p) for p in pub.registri_presenti(str(tmp_path))]
    assert presenti == [f"{registro.FILE_DEGLI_ERRORI}.1", registro.FILE_DEGLI_ERRORI, registro.FILE_DEL_CRASH_PRECEDENTE]


def test_i_registri_ripuliti_dai_nomi(tmp_path):
    (tmp_path / registro.FILE_DEGLI_ERRORI).write_text("File \"C:\\Users\\Gabriele\\Ciechino\\partita.py\" su GABRYBAT\nUtente contento\n",
                                                      encoding="utf-8")
    (tmp_path / registro.FILE_DEL_CRASH_PRECEDENTE).write_text(FATALE, encoding="utf-8")
    percorso = pub.scrivi_i_registri(str(tmp_path), "0.1.0", "Gabriele", "GABRYBAT")
    assert percorso == str(tmp_path / pub.FILE_DEI_REGISTRI)
    testo = _leggi(tmp_path / pub.FILE_DEI_REGISTRI)
    assert testo.startswith("Registri di Ciechino 0.1.0, preparati il ")
    assert "C:\\Users\\<utente>\\Ciechino" in testo and "su <computer>" in testo
    assert "Gabriele" not in testo.replace("Gabriele Battaglia", "") and "GABRYBAT" not in testo
    assert "Dal registro degli errori, tutto:" in testo and f"Dal resoconto del crash {registro.FILE_DEL_CRASH_PRECEDENTE}, tutto:" in testo


def test_ripulisci_lascia_le_parole_comuni():
    # Utente è anche una parola: si toglie solo accanto a un separatore dei percorsi.
    testo = pub.ripulisci("L'Utente apre C:\\Users\\Utente\\Desktop su PC-UFFICIO, non su PC-UFFICIO2.", "Utente", "PC-UFFICIO")
    assert testo == "L'Utente apre C:\\Users\\<utente>\\Desktop su <computer>, non su PC-UFFICIO2."


def test_le_mail_del_pub_parlano_di_ciechino(tmp_path):
    iscrizione = pub.mail_d_iscrizione()
    assert [p[1] for p in iscrizione["passi"]][-1] is None and "{programma}" in iscrizione["spiegazione"]
    aiuto = pub.mail_d_aiuto("0.1.0", True, {}, scheda_audio="Altoparlanti (WASAPI)", casella=True)
    assert aiuto["oggetto"] == "Chiedo per Ciechino 0.1.0" and len(aiuto["passi"]) == pub.PASSI_SENZA_REGISTRI
    assert "Programma: Ciechino 0.1.0, pacchetto compilato" in aiuto["corpo"] and "Uscita audio: Altoparlanti (WASAPI)" in aiuto["corpo"]
    assert "spunta la casella" in aiuto["spiegazione"] and "appunti" not in aiuto
    registri = str(tmp_path / pub.FILE_DEI_REGISTRI)
    con_registri = pub.mail_d_aiuto("0.1.0", False, {}, registri=registri)
    assert len(con_registri["passi"]) == pub.PASSI_CON_REGISTRI and con_registri["appunti"] == registri
    assert con_registri["passi"][3][0].startswith("Passo 4 di 5.") and con_registri["passi"][-1][0].startswith("Passo 5 di 5.")
    # Niente di MeTeOra rimasto nelle parole, e niente righe vuote fra i paragrafi.
    for mail in (iscrizione, aiuto, con_registri):
        tutto = "\n".join([mail["titolo"], mail["oggetto"], mail["corpo"], mail["spiegazione"], *(p[0] for p in mail["passi"])])
        assert "MeTeOra" not in tutto and "\n\n" not in tutto
