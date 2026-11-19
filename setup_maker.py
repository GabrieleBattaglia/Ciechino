"""
Ciechino, utilità: compila l'installatore con Inno Setup.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di setup_maker.py di MESS: il pacchetto ha un
installatore, che chiede di accettare la licenza. Compila Ciechino.iss col compilatore di Inno Setup 6,
passandogli la versione di version.py, e scrive Ciechino-Setup-<versione>.exe nella cartella del
progetto, accanto a Ciechino.zip. Si lancia dopo PyInstaller e zip_maker.py, con dist ripulita
(prontuario, fase 7.0 e 7.5). Il compilatore si cerca nella variabile CIECHINO_ISCC, poi sulla path,
poi nelle cartelle dove Inno Setup si installa.
Si lancia a mano: importarlo non fa niente.
"""

import os
import shutil
import subprocess
import sys

from costanti import NOME
from version import __version__

# La cartella del progetto, dove stanno Ciechino.iss e dist e dove nasce l'installatore, da qualunque
# cartella si lanci lo script.
RADICE = os.path.dirname(os.path.abspath(__file__))
VARIABILE_DEL_COMPILATORE = "CIECHINO_ISCC"
RICETTA = f"{NOME}.iss"


def compilatore():
    """Il percorso di ISCC.exe, o None."""
    candidati = [os.environ.get(VARIABILE_DEL_COMPILATORE), shutil.which("iscc")]
    if os.environ.get("LOCALAPPDATA"):
        candidati.append(os.path.join(os.environ["LOCALAPPDATA"], "Programs", "Inno Setup 6", "ISCC.exe"))
    for radice in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES")):
        if radice:
            candidati.append(os.path.join(radice, "Inno Setup 6", "ISCC.exe"))
    return next((c for c in candidati if c and os.path.isfile(c)), None)


def main():
    iscc = compilatore()
    if iscc is None:
        print(f"Installatore non creato: non trovo ISCC.exe di Inno Setup 6. Indica il percorso con la variabile {VARIABILE_DEL_COMPILATORE}.")
        return 1
    if not os.path.isfile(os.path.join(RADICE, "dist", NOME, f"{NOME}.exe")):
        print(f"Installatore non creato: manca dist\\{NOME}\\{NOME}.exe. Compila prima con PyInstaller.")
        return 1
    esito = subprocess.run([iscc, "/Q", f"/DVersione={__version__}", RICETTA], cwd=RADICE, check=False)  # noqa: S603 - il compilatore trovato qui sopra
    if esito.returncode:
        print(f"Installatore non creato: ISCC ha finito con il codice {esito.returncode}.")
        return esito.returncode
    print(f"Installatore creato: {NOME}-Setup-{__version__}.exe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
