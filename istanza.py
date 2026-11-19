"""
Ciechino, l'istanza unica: la seconda copia porta la prima in primo piano e si chiude.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di istanza.py di MESS, che viene da quello di MeTeOra.
Due copie aperte insieme suonerebbero due partite sovrapposte, che a orecchio non si distinguono, e
scriverebbero la stessa classifica e le stesse statistiche, così che una cancellerebbe il lavoro
dell'altra. La prima copia tiene un mutex con nome, che anche l'installatore di Inno Setup cerca con
AppMutex di Ciechino.iss, e ascolta su una pipe con nome di Windows, per utente e con una chiave; la
seconda copia ci si presenta, le permette di venire in primo piano e si chiude. Sul filo passano solo
byte JSON, mai oggetti di Python. La variabile d'ambiente CIECHINO_ISTANZA cambia i nomi della pipe e
del mutex, per le prove: così non parlano con il Ciechino che magari è aperto.
Come MESS, Ciechino sa anche riaprirsi, riapri: la lingua scelta nelle impostazioni vale dal riavvio
(lingua.py), e la finestra può offrire di ripartire subito. La copia nuova parte con --attendi e il
numero del processo di questa, e prima di presentarsi alla pipe aspetta che questa sia uscita,
aspetta_l_uscita: altrimenti la troverebbe ancora aperta, la porterebbe in primo piano e si
chiuderebbe lei. Da sorgente si riapre l'interprete di adesso con Ciechino.py, dal programma
compilato Ciechino.exe, con PYINSTALLER_RESET_ENVIRONMENT, come un Ciechino aperto da capo e non come
un figlio di questo.
"""

import contextlib
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import threading
from ctypes import wintypes
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener

from costanti import NOME

# La variabile d'ambiente delle prove, che dà a pipe e mutex un nome tutto loro.
VARIABILE_DI_PROVA = "CIECHINO_ISTANZA"
# Lo script dei sorgenti che la copia riaperta lancia con l'interprete.
SCRIPT = f"{NOME}.py"
# AllowSetForegroundWindow: qualunque processo, cioè la prima copia.
_ASFW_ANY = -1
# Quante volte ferma() sblocca il filo dell'ascolto, e quanti secondi aspetta ogni volta che finisca:
# 40 per 50 millesimi, due secondi al più, la misura che in MESS e in MeTeOra basta sempre.
LIMITE_DEGLI_SBLOCCHI = 40
PAUSA_DELLO_SBLOCCO = 0.05
# Il messaggio più lungo che l'ascolto accetta da una copia che si presenta, in byte: il suo è di
# pochi byte, e un limite tiene fuori chi mandasse altro sulla pipe.
MESSAGGIO_MASSIMO = 1 << 16
# L'argomento della copia che si riapre, seguito dal numero del processo da aspettare, e quanti secondi
# lo aspetta al più: dopo, si presenta lo stesso, e se la vecchia è ancora aperta la porta in primo piano.
ATTENDI = "--attendi"
ATTESA_DELLA_RIAPERTURA = 60
# OpenProcess col solo diritto di aspettare il processo, e la risposta di WaitForSingleObject quando è finito.
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0
_MILLESIMI = 1000


def nome():
    """Il nome della pipe, per utente."""
    proprio = os.environ.get(VARIABILE_DI_PROVA)
    if proprio:
        return rf"\\.\pipe\{proprio}"
    return rf"\\.\pipe\{NOME}-{os.environ.get('USERNAME', 'utente')}"


def nome_del_mutex():
    """Il mutex di Ciechino aperto: lo stesso nome di AppMutex in Ciechino.iss; con CIECHINO_ISTANZA, uno di prova."""
    proprio = os.environ.get(VARIABILE_DI_PROVA)
    return f"{NOME}-{proprio}" if proprio else NOME


def _chiave(nome_della_pipe):
    return hashlib.sha256(f"{NOME} {nome_della_pipe}".encode()).digest()


def manda(nome_della_pipe=None):
    """Se un Ciechino è già aperto, gli chiede di venire in primo piano e torna vero; falso se non ce n'è uno."""
    nome_della_pipe = nome_della_pipe or nome()
    try:
        connessione = Client(nome_della_pipe, family="AF_PIPE", authkey=_chiave(nome_della_pipe))
    except (OSError, EOFError, AuthenticationError):
        # Nessuno ascolta, o sulla pipe c'è qualcuno che non conosce la chiave di Ciechino.
        return False
    with contextlib.suppress(Exception):
        ctypes.windll.user32.AllowSetForegroundWindow(_ASFW_ANY)
    try:
        with connessione:
            connessione.send_bytes(json.dumps({"mostra": True}).encode("utf-8"))
    except (OSError, EOFError):
        return False
    return True


def da_attendere(argomenti):
    """Il numero del processo da aspettare, dopo --attendi fra gli argomenti; None se non c'è o non è un numero."""
    if ATTENDI not in argomenti:
        return None
    posto = argomenti.index(ATTENDI) + 1
    if posto >= len(argomenti) or not argomenti[posto].isdigit():
        return None
    return int(argomenti[posto])


def aspetta_l_uscita(pid, secondi=ATTESA_DELLA_RIAPERTURA):
    """Aspetta che il processo pid sia uscito, al più per i secondi dati: vero se è uscito, o non c'era già più."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    maniglia = kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
    if not maniglia:
        return True
    try:
        return kernel32.WaitForSingleObject(maniglia, max(0, round(secondi * _MILLESIMI))) == _WAIT_OBJECT_0
    finally:
        kernel32.CloseHandle(maniglia)


def _cartella_dei_sorgenti():
    """La cartella di Ciechino.py, che sta accanto a questo modulo: non quella dei dati, che le prove spostano."""
    return os.path.dirname(os.path.abspath(__file__))


def comando_di_riapertura(pid=None):
    """La riga di comando della copia che si riapre dopo questa: Ciechino.exe dal programma compilato, l'interprete con Ciechino.py da sorgente."""
    pid = os.getpid() if pid is None else pid
    if getattr(sys, "frozen", False):
        return [sys.executable, ATTENDI, str(pid)]
    return [sys.executable, os.path.join(_cartella_dei_sorgenti(), SCRIPT), ATTENDI, str(pid)]


def riapri():
    """
    Lancia la copia di Ciechino che si apre quando questa sarà uscita, e la restituisce; OSError se non
    parte. Questa copia ha il primo piano e le permette di prenderlo: senza, la finestra nuova potrebbe
    comparire dietro, e NVDA non la leggerebbe. Dal programma compilato la copia nuova riparte come un
    Ciechino aperto da capo: PYINSTALLER_RESET_ENVIRONMENT le dice di non usare l'ambiente che
    PyInstaller ha preparato per questa.
    """
    comando = comando_di_riapertura()
    if getattr(sys, "frozen", False):
        opzioni = {"cwd": os.path.dirname(sys.executable), "close_fds": True, "env": {**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"}}
    else:
        opzioni = {"cwd": _cartella_dei_sorgenti(), "close_fds": True}
    processo = subprocess.Popen(comando, **opzioni)  # noqa: S603 - il comando è Ciechino stesso
    with contextlib.suppress(Exception):
        ctypes.windll.user32.AllowSetForegroundWindow(processo.pid)
    return processo


class Ascolto:
    """
    La prima copia ascolta le altre: mostra() è chiamata nel filo dell'ascolto, per ogni copia che si
    presenta, e deve portare il lavoro nel filo della finestra, per esempio con wx.CallAfter. Se la
    pipe c'è già, perché un'altra copia è partita nello stesso istante, attiva è falso.
    """

    def __init__(self, mostra, nome_della_pipe=None):
        self._mostra = mostra
        self._nome = nome_della_pipe or nome()
        self._fermo = False
        self._mutex = None
        self._filo = None
        try:
            self._ascoltatore = Listener(self._nome, family="AF_PIPE", authkey=_chiave(self._nome))
        except OSError:
            self._ascoltatore = None
        self.attiva = self._ascoltatore is not None
        if self.attiva:
            if nome_della_pipe is None:
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CreateMutexW.restype = wintypes.HANDLE
                kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
                self._mutex = kernel32.CreateMutexW(None, False, nome_del_mutex()) or None
            self._filo = threading.Thread(target=self._ascolta, name=f"{NOME}, istanza unica", daemon=True)
            self._filo.start()

    def _ascolta(self):
        while not self._fermo:
            try:
                connessione = self._ascoltatore.accept()
            except Exception:  # noqa: BLE001 - una copia che si presenta male, o con la chiave sbagliata
                if self._fermo:
                    return
                continue
            with connessione, contextlib.suppress(Exception):
                messaggio = json.loads(connessione.recv_bytes(MESSAGGIO_MASSIMO).decode("utf-8"))
                if isinstance(messaggio, dict) and messaggio.get("mostra") and not self._fermo:
                    self._mostra()

    def ferma(self):
        """All'uscita: l'ascolto finisce, e il mutex si lascia."""
        self._fermo = True
        if self._mutex is not None:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle(self._mutex)
            self._mutex = None
        if self._ascoltatore is not None:
            # Prima il filo, poi il Listener. Il filo si sblocca aprendo e chiudendo la pipe senza la stretta
            # di mano di Client, che aspetterebbe per sempre una sfida (la lezione della 1.96.11 di MeTeOra),
            # più volte, finché non è finito: se arriva in accept dopo un'apertura, aspetterebbe la seguente.
            # Solo dopo si chiude il Listener: chiuso prima, i suoi handle restavano in coda chiusi mentre il
            # filo li usava, Windows ne riusava i numeri, e un'istanza della pipe poteva restare aperta,
            # così che nessuno, nello stesso processo, poteva più ascoltare con quel nome (MESS).
            for _ in range(LIMITE_DEGLI_SBLOCCHI):
                if self._filo is None or not self._filo.is_alive():
                    break
                with contextlib.suppress(OSError), open(self._nome, "rb", buffering=0):
                    pass
                self._filo.join(PAUSA_DELLO_SBLOCCO)
            with contextlib.suppress(Exception):
                self._ascoltatore.close()
            self._ascoltatore = None
