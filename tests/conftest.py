"""
Le prove di Ciechino: mette la radice del progetto nel percorso di importazione, dirotta in una
cartella temporanea tutto ciò che il programma scrive, tiene mute le casse e fa nascere le finestre
delle prove su un desktop di Windows nascosto.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, dal conftest di MESS, che a sua volta viene da Tornello 10.13.37.
Windows dà il primo piano anche a una finestra mai mostrata quando riceve il fuoco, se il processo
è partito dal terminale che aveva il primo piano: il 26 settembre 2026 finestre invisibili delle
prove di Tornello hanno preso il primo piano sullo schermo di Gabriele, e NVDA gli leggeva i loro
titoli. Perciò, prima che wx crei qualunque finestra, il thread principale passa su un desktop tutto
suo, che nessuno vede. Se il passaggio non riesce la suite non parte.
Nessuna prova apre la scheda audio vera: motore_audio.apri_flusso risponde None, e il motore
lavora muto; le prove che vogliono guidare il tempo usano il flusso finto di tests/aiuti_audio.py.
Nessuna prova parla: la sintesi di GBsintesi è sostituita da un registratore.
"""

import os
import sys
from pathlib import Path

import pytest

RADICE = Path(__file__).resolve().parent.parent
if str(RADICE) not in sys.path:
    sys.path.insert(0, str(RADICE))
CARTELLA_DELLE_PROVE = Path(__file__).resolve().parent
if str(CARTELLA_DELLE_PROVE) not in sys.path:
    sys.path.insert(0, str(CARTELLA_DELLE_PROVE))

NOME_DEL_DESKTOP = f"ciechino_prove_{os.getpid()}"
GENERIC_ALL = 0x10000000
# La maniglia del desktop nascosto resta aperta finché il processo vive: chiusa, il desktop
# sparirebbe sotto le finestre delle prove.
DESKTOP_DELLE_PROVE = {}


def _user32():
    """user32 con le firme delle funzioni dei desktop, in una copia tutta sua."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateDesktopW.restype = wintypes.HANDLE
    user32.CreateDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    user32.SetThreadDesktop.restype = wintypes.BOOL
    user32.SetThreadDesktop.argtypes = [wintypes.HANDLE]
    user32.CloseDesktop.restype = wintypes.BOOL
    user32.CloseDesktop.argtypes = [wintypes.HANDLE]
    return user32


def _non_disponibile(azione, funzione, codice):
    import ctypes

    return (f"Le prove non partono: non si è potuto {azione} ({funzione}, errore di Windows {codice}: {ctypes.FormatError(codice).strip()}). "
            "Le finestre delle prove nascono su un desktop di Windows nascosto: senza, una finestra mai mostrata che riceve il fuoco "
            "prenderebbe il primo piano sullo schermo di chi lancia le prove. Il passaggio fallisce se questo thread ha già delle finestre.")


def pytest_configure(config):
    """Il passaggio al desktop nascosto, una volta per processo, prima che le prove creino wx.App."""
    if sys.platform != "win32" or DESKTOP_DELLE_PROVE:
        return
    import ctypes

    user32 = _user32()
    desktop = user32.CreateDesktopW(NOME_DEL_DESKTOP, None, None, 0, GENERIC_ALL, None)
    if not desktop:
        raise pytest.UsageError(_non_disponibile("creare il desktop nascosto", "CreateDesktopW", ctypes.get_last_error()))
    if not user32.SetThreadDesktop(desktop):
        codice = ctypes.get_last_error()
        user32.CloseDesktop(desktop)
        raise pytest.UsageError(_non_disponibile("spostare le prove sul desktop nascosto", "SetThreadDesktop", codice))
    DESKTOP_DELLE_PROVE["nascosto"] = desktop


@pytest.fixture(autouse=True)
def cartella_di_prova(tmp_path, monkeypatch):
    """La cartella del programma, dove Ciechino scrive classifica, statistiche, impostazioni e
    registri, durante le prove è una cartella temporanea, e la prova la riceve."""
    import percorsi

    monkeypatch.setattr(percorsi, "cartella_programma", lambda: str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def casse_mute(monkeypatch):
    """Nessuna prova apre la scheda audio: il motore, senza flusso, lavora muto.
    Finché motore_audio non esiste ancora, la fissatura non fa niente."""
    try:
        import motore_audio
    except ImportError:
        return
    monkeypatch.setattr(motore_audio, "apri_flusso", lambda *argomenti, **opzioni: None)


class Detti(list):
    """I testi che la sintesi avrebbe letto, in ordine."""


@pytest.fixture(autouse=True)
def voce_finta(monkeypatch):
    """Nessuna prova parla: Sintesi.dici di GBsintesi registra il testo invece di mandarlo allo
    screen reader, e la prova riceve l'elenco."""
    detti = Detti()
    try:
        import GBsintesi
    except ImportError:
        return detti

    def dici(self, testo, chiave=GBsintesi.AUTOMATICA, dove="entrambi", interrompi=False):
        detti.append(testo)
        return True

    monkeypatch.setattr(GBsintesi.Sintesi, "dici", dici)
    return detti


@pytest.fixture(scope="session")
def app_wx():
    """Una wx.App sola per tutta la sessione delle prove, sul desktop nascosto."""
    import wx

    app = wx.GetApp() or wx.App(False)
    yield app
