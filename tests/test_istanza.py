"""
Le prove dell'istanza unica, con una pipe e un mutex di prova scelti da CIECHINO_ISTANZA, diversi per
ogni prova, che non parlano con il Ciechino magari aperto: senza ascolto manda è falso; con l'ascolto
è vero e la prima copia riceve la richiesta di mostrarsi, il mutex che cerca l'installatore c'è, e una
seconda copia partita nello stesso istante non prende la pipe; ferma libera la pipe e il mutex. Poi la
riapertura, senza lanciare niente: gli argomenti di --attendi, la riga di comando da sorgente e dal
compilato, e riapri con un Popen finto. AllowSetForegroundWindow, che darebbe a ogni processo il
permesso di prendersi il primo piano sullo schermo vero, è sostituito da un registratore.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di test_istanza.py di MESS.
"""

import ctypes
import os
import sys
import threading
import time
import types
import uuid
from ctypes import wintypes

import pytest

import istanza

SYNCHRONIZE = 0x00100000
# Quanti secondi una prova aspetta che il filo dell'ascolto risponda: molti più del necessario.
ATTESA = 10


@pytest.fixture
def pipe_di_prova(monkeypatch):
    """Il nome di prova di pipe e mutex, e l'elenco delle volte che si è chiesto il permesso del primo piano."""
    nome = f"ciechino_prove_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv(istanza.VARIABILE_DI_PROVA, nome)
    permessi = []
    monkeypatch.setattr(ctypes.windll.user32, "AllowSetForegroundWindow", lambda chi: permessi.append(chi) or 1)
    return permessi


def _mutex_c_e(nome):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = wintypes.HANDLE
    kernel32.OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    maniglia = kernel32.OpenMutexW(SYNCHRONIZE, False, nome)
    if maniglia:
        kernel32.CloseHandle(maniglia)
    return bool(maniglia)


def test_i_nomi_di_prova(pipe_di_prova, monkeypatch):
    proprio = os.environ[istanza.VARIABILE_DI_PROVA]
    assert istanza.VARIABILE_DI_PROVA == "CIECHINO_ISTANZA"
    assert istanza.nome() == rf"\\.\pipe\{proprio}" and istanza.nome_del_mutex() == f"Ciechino-{proprio}"
    monkeypatch.delenv(istanza.VARIABILE_DI_PROVA)
    monkeypatch.setenv("USERNAME", "gabriele")
    # Il mutex vero ha lo stesso nome di AppMutex in Ciechino.iss: lo controlla anche test_pacchetto.
    assert istanza.nome() == r"\\.\pipe\Ciechino-gabriele" and istanza.nome_del_mutex() == "Ciechino"


def test_senza_ascolto_manda_e_falso(pipe_di_prova):
    assert not istanza.manda()
    assert pipe_di_prova == []


def test_con_l_ascolto_la_seconda_copia_si_presenta(pipe_di_prova):
    chiamata = threading.Event()
    ascolto = istanza.Ascolto(chiamata.set)
    try:
        assert ascolto.attiva and _mutex_c_e(istanza.nome_del_mutex())
        assert istanza.manda()
        assert chiamata.wait(ATTESA)
        assert pipe_di_prova == [-1]
        # Una copia partita nello stesso istante trova la pipe già presa.
        altra = istanza.Ascolto(lambda: None)
        assert not altra.attiva
        altra.ferma()
    finally:
        ascolto.ferma()
    assert not _mutex_c_e(istanza.nome_del_mutex())
    # Dopo ferma nessuno ascolta più: una copia nuova non trova la prima.
    assert not istanza.manda()


def test_la_chiave_sbagliata_non_mostra(pipe_di_prova):
    from multiprocessing import AuthenticationError
    from multiprocessing.connection import Client

    chiamate = []
    ascolto = istanza.Ascolto(lambda: chiamate.append(True))
    try:
        with pytest.raises((AuthenticationError, OSError, EOFError)):
            Client(istanza.nome(), family="AF_PIPE", authkey=b"un'altra chiave").close()
        # L'ascolto resta vivo e risponde alla copia vera.
        assert istanza.manda()
        scadenza = time.monotonic() + ATTESA
        while not chiamate and time.monotonic() < scadenza:
            time.sleep(0.01)
        assert chiamate == [True]
    finally:
        ascolto.ferma()


def test_ferma_libera_la_pipe(pipe_di_prova):
    istanza.Ascolto(lambda: None).ferma()
    # Il filo dell'ascolto si chiude da sé: entro pochi istanti la pipe è di nuovo libera.
    scadenza = time.monotonic() + ATTESA
    nuovo = istanza.Ascolto(lambda: None)
    while not nuovo.attiva and time.monotonic() < scadenza:
        time.sleep(0.05)
        nuovo = istanza.Ascolto(lambda: None)
    try:
        assert nuovo.attiva
    finally:
        nuovo.ferma()


@pytest.mark.parametrize(("argomenti", "pid"), [
    ([], None),
    (["--attendi", "1234"], 1234),
    (["Ciechino.py", "--attendi", "77"], 77),
    (["--attendi"], None),
    (["--attendi", "dopo"], None),
])
def test_da_attendere(argomenti, pid):
    assert istanza.da_attendere(argomenti) == pid


def test_aspetta_l_uscita():
    # Questo processo è vivo: con zero secondi l'attesa scade subito.
    assert not istanza.aspetta_l_uscita(os.getpid(), 0)
    # Un processo che non si apre vale come già uscito.
    assert istanza.aspetta_l_uscita(0, 0)


def test_il_comando_di_riapertura(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    cartella = os.path.dirname(os.path.abspath(istanza.__file__))
    assert istanza.comando_di_riapertura(42) == [sys.executable, os.path.join(cartella, "Ciechino.py"), "--attendi", "42"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert istanza.comando_di_riapertura(42) == [sys.executable, "--attendi", "42"]
    assert istanza.comando_di_riapertura()[-1] == str(os.getpid())


def test_riapri_non_lancia_niente_di_vero(pipe_di_prova, monkeypatch):
    lanciati = []

    def popen_finto(comando, **opzioni):
        """Al posto di subprocess.Popen: annota il comando e le opzioni, e non lancia niente."""
        lanciati.append((comando, opzioni))
        return types.SimpleNamespace(pid=4321)

    monkeypatch.setattr(istanza.subprocess, "Popen", popen_finto)
    monkeypatch.delattr(sys, "frozen", raising=False)
    processo = istanza.riapri()
    comando, opzioni = lanciati[-1]
    assert comando == istanza.comando_di_riapertura() and processo.pid == 4321
    assert "env" not in opzioni and opzioni["cwd"] == os.path.dirname(comando[1])
    # La copia nuova ha il permesso del primo piano.
    assert pipe_di_prova == [4321]
    # Dal compilato riparte come un Ciechino aperto da capo, nella cartella dell'eseguibile.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    istanza.riapri()
    comando, opzioni = lanciati[-1]
    assert comando[0] == sys.executable and opzioni["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert opzioni["cwd"] == os.path.dirname(sys.executable)
