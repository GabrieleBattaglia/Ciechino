"""
Le prove del pacchetto, senza compilare niente: PyInstaller, Inno Setup e zip_maker veri non partono
mai. La ricetta Ciechino.spec si compila con compile() senza eseguirla, ed è a cartella con COLLECT,
senza console, coi suoi datas; l'installatore Ciechino.iss ha AppMutex uguale al mutex di istanza.py,
un AppId suo, diverso da quelli di MESS, di MeTeOra e degli altri programmi del parco, e conosce i file
dei dati di costanti.py e i registri di registro.py; zip_maker e setup_maker si importano senza
effetti, e le loro funzioni si provano con il compressore e il compilatore sostituiti; babel.cfg porta
al catalogo i moduli giusti; i requisiti dicono le librerie di Ciechino.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
"""

import ast
import fnmatch
import importlib
import re
import subprocess
import sys
import uuid
from pathlib import Path

import GBUtils
import pytest

import istanza
import pub
import registro
from costanti import FILE_CLASSIFICA, FILE_IMPOSTAZIONI, FILE_STATISTICHE, NOME
from version import __version__

RADICE = Path(__file__).resolve().parent.parent
SPEC = RADICE / "Ciechino.spec"
ISS = RADICE / "Ciechino.iss"
# Gli AppId di MESS e di MeTeOra, scritti qui perché la prova valga anche dove i loro repository non ci sono.
APPID_ALTRUI = {"C442DC21-A3A1-4E14-844C-645899A77571", "64985198-5673-4D67-A490-35A62886C1F2"}
# Tutti i file che Ciechino scrive accanto all'eseguibile, che né lo zip né l'installatore devono portare.
DATI = (FILE_CLASSIFICA, f"{FILE_CLASSIFICA}.bak", f"{FILE_CLASSIFICA}.tmp", FILE_STATISTICHE, f"{FILE_STATISTICHE}.bak", FILE_IMPOSTAZIONI,
        f"{FILE_IMPOSTAZIONI}.bak", "selected_language.json", registro.FILE_DEGLI_ERRORI, f"{registro.FILE_DEGLI_ERRORI}.1", registro.FILE_DEL_CRASH,
        registro.FILE_DEL_CRASH_PRECEDENTE, pub.FILE_DEI_REGISTRI)


def _testo(percorso):
    return percorso.read_text(encoding="utf-8")


def _chiamate(albero):
    """Le chiamate per nome della ricetta, con le loro opzioni: {"EXE": {"console": False, ...}}."""
    trovate = {}
    for nodo in ast.walk(albero):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name):
            trovate[nodo.func.id] = {k.arg: k.value for k in nodo.keywords}
    return trovate


# La ricetta di PyInstaller.

def test_la_ricetta_si_compila_senza_eseguirla():
    testo = _testo(SPEC)
    codice = compile(testo, str(SPEC), "exec")
    assert codice.co_filename == str(SPEC)
    assert "# ruff: noqa: F821" in testo.splitlines()
    chiamate = _chiamate(ast.parse(testo))
    assert {"Analysis", "PYZ", "EXE", "COLLECT"} <= set(chiamate)
    # A cartella: l'eseguibile senza le librerie, raccolte da COLLECT; mai onefile.
    exe = chiamate["EXE"]
    assert ast.literal_eval(exe["exclude_binaries"]) is True
    assert ast.literal_eval(exe["console"]) is False and ast.literal_eval(exe["upx"]) is False
    assert ast.literal_eval(chiamate["COLLECT"]["upx"]) is False
    assert re.search(r'^NOME = "Ciechino"$', testo, re.MULTILINE)


def test_i_datas_della_ricetta():
    testo = _testo(SPEC)
    for voce in ('"manuale.html"', '"CHANGELOG.md"', '"Acu_Collection.json"', "GBUtils.__file__", '"locales"', '"messages.mo"',
                 'collect_data_files("accessible_output2", include_py_files=False)', 'collect_submodules("accessible_output2.outputs")'):
        assert voce in testo, voce
    # Le DLL che non si possono ridistribuire restano fuori, e la licenza va accanto all'eseguibile.
    assert '"pctkusr.dll"' in testo and '"nvdacontrollerclient32.dll"' in testo
    assert 'shutil.copyfile(os.path.join(SPECPATH, "LICENSE"), os.path.join(DISTPATH, NOME, "LICENSE"))' in testo


# L'installatore.

def _appid(testo):
    trovato = re.search(r"^AppId=\{\{([0-9A-Fa-f-]{36})\}$", testo, re.MULTILINE)
    return trovato.group(1).upper() if trovato else None


def test_l_installatore_e_il_mutex_di_istanza(monkeypatch):
    monkeypatch.delenv(istanza.VARIABILE_DI_PROVA, raising=False)
    righe = _testo(ISS).splitlines()
    assert f"AppMutex={istanza.nome_del_mutex()}" in righe
    assert [r for r in righe if r.startswith("AppMutex=")] == ["AppMutex=Ciechino"]


def test_l_appid_e_nuovo():
    appid = _appid(_testo(ISS))
    assert appid is not None and str(uuid.UUID(appid)).upper() == appid
    altrui = set(APPID_ALTRUI)
    # E, dove ci sono, quelli di tutti gli altri programmi del parco, accanto a Ciechino.
    for altro in RADICE.parent.glob("*/*.iss"):
        if altro.resolve() != ISS.resolve():
            suo = _appid(altro.read_text(encoding="utf-8", errors="replace"))
            if suo:
                altrui.add(suo)
    assert appid not in altrui


def test_l_installatore_dice_le_scelte_di_ciechino():
    righe = _testo(ISS).splitlines()
    for riga in ("AppName=Ciechino", "PrivilegesRequired=lowest", r"DefaultDirName={autopf}\Ciechino", "LicenseFile=LICENSE",
                 "OutputBaseFilename=Ciechino-Setup-{#Versione}", r"UninstallDisplayIcon={app}\Ciechino.exe"):
        assert riga in righe, riga
    assert any("Italian.isl" in r for r in righe)


def test_l_installatore_conosce_i_dati():
    testo = _testo(ISS)
    # I nomi di costanti.py, nelle costanti del codice di Inno Setup.
    for nome in (FILE_CLASSIFICA, FILE_STATISTICHE, FILE_IMPOSTAZIONI):
        assert f"'{nome}';" in testo, nome
    # I registri di registro.py e il file per il PUB, che la disinstallazione toglie se si risponde Sì.
    for nome in (registro.FILE_DEGLI_ERRORI, f"{registro.FILE_DEGLI_ERRORI}.1", registro.FILE_DEL_CRASH, registro.FILE_DEL_CRASH_PRECEDENTE,
                 pub.FILE_DEI_REGISTRI, "selected_language.json"):
        assert f"'\\{nome}'" in testo, nome
    assert "MsgBox('Cancello anche i dati di Ciechino?" in testo and "MB_DEFBUTTON2" in testo
    # Nessun file dei dati entra nell'installatore.
    esclusi = re.search(r'Excludes: "([^"]+)"', testo).group(1).split(",")
    motivi = [m.lstrip("\\").lower() for m in esclusi]
    for nome in DATI:
        assert any(fnmatch.fnmatch(nome.lower(), m) for m in motivi), nome


# Lo zip e il compilatore dell'installatore.

@pytest.fixture
def senza_effetti(monkeypatch):
    """Il compressore di GBUtils e il lancio dei processi sostituiti da registratori: niente si comprime e niente parte."""
    fatti = {"archivi": [], "processi": []}

    def archivio(nome_app, **opzioni):
        fatti["archivi"].append((nome_app, opzioni))
        return 1, []

    def lancia(comando, **opzioni):
        fatti["processi"].append((comando, opzioni))
        return subprocess.CompletedProcess(comando, fatti.get("codice", 0))

    monkeypatch.setattr(GBUtils, "crea_archivio_release", archivio)
    monkeypatch.setattr(subprocess, "run", lancia)
    return fatti


@pytest.mark.parametrize("modulo", ["zip_maker", "setup_maker"])
def test_si_importano_senza_effetti(senza_effetti, monkeypatch, modulo):
    monkeypatch.delitem(sys.modules, modulo, raising=False)
    importato = importlib.import_module(modulo)
    assert callable(importato.main)
    assert senza_effetti == {"archivi": [], "processi": []}


def test_zip_maker_lascia_fuori_i_dati(senza_effetti, capsys):
    import zip_maker

    assert zip_maker.main() == 0
    assert senza_effetti["archivi"] == [(NOME, {"escludi": zip_maker.FUORI})]
    # crea_archivio_release toglie da sé .log, .bak e .tmp; FUORI tutto il resto.
    estensioni_di_gbutils = (".bak", ".tmp", ".log")
    for nome in DATI:
        assert nome.lower().endswith(estensioni_di_gbutils) or any(fnmatch.fnmatch(nome.lower(), m.lower()) for m in zip_maker.FUORI), nome


def test_zip_maker_senza_dist(monkeypatch, capsys):
    import zip_maker

    def manca(*_argomenti, **_opzioni):
        raise FileNotFoundError("Cartella da comprimere assente: dist\\Ciechino")

    monkeypatch.setattr(GBUtils, "crea_archivio_release", manca)
    assert zip_maker.main() == 1
    assert capsys.readouterr().out.startswith("Archivio non creato: ")


def test_setup_maker_trova_il_compilatore(tmp_path, monkeypatch):
    import setup_maker

    iscc = tmp_path / "ISCC.exe"
    iscc.write_bytes(b"")
    monkeypatch.setenv("CIECHINO_ISCC", str(iscc))
    assert setup_maker.VARIABILE_DEL_COMPILATORE == "CIECHINO_ISCC"
    assert setup_maker.compilatore() == str(iscc)


def test_setup_maker_compila_con_la_versione(senza_effetti, tmp_path, monkeypatch, capsys):
    import setup_maker

    iscc = str(tmp_path / "ISCC.exe")
    monkeypatch.setattr(setup_maker, "RADICE", str(tmp_path))
    monkeypatch.setattr(setup_maker, "compilatore", lambda: None)
    assert setup_maker.main() == 1 and "CIECHINO_ISCC" in capsys.readouterr().out
    monkeypatch.setattr(setup_maker, "compilatore", lambda: iscc)
    # Senza l'eseguibile di PyInstaller non si compila.
    assert setup_maker.main() == 1 and "dist\\Ciechino\\Ciechino.exe" in capsys.readouterr().out
    assert senza_effetti["processi"] == []
    eseguibile = tmp_path / "dist" / "Ciechino" / "Ciechino.exe"
    eseguibile.parent.mkdir(parents=True)
    eseguibile.write_bytes(b"")
    assert setup_maker.main() == 0
    assert senza_effetti["processi"] == [([iscc, "/Q", f"/DVersione={__version__}", "Ciechino.iss"], {"cwd": str(tmp_path), "check": False})]
    assert capsys.readouterr().out.strip() == f"Installatore creato: Ciechino-Setup-{__version__}.exe"
    senza_effetti["codice"] = 2
    assert setup_maker.main() == 2


# Il catalogo dei testi e i requisiti.

def test_babel_porta_al_catalogo_i_moduli_giusti():
    pytest.importorskip("babel")
    from babel.messages.frontend import parse_mapping_cfg
    from babel.util import pathmatch

    with open(RADICE / "babel.cfg", encoding="utf-8") as file:
        metodi, _opzioni = parse_mapping_cfg(file)

    def metodo(percorso):
        """Come pybabel: vale il primo motivo che combacia, e un file che non ne ha nessuno resta fuori."""
        return next((nome for motivo, nome in metodi if pathmatch(motivo, percorso)), None)

    for percorso in ("aggiornamento.py", "pub.py", "gui/finestra.py", "gui\\dialoghi.py"):
        assert metodo(percorso) == "python", percorso
    for percorso in ("zip_maker.py", "setup_maker.py"):
        assert metodo(percorso) == "ignore", percorso
    for percorso in ("tests/test_pacchetto.py", "strumenti/crea_preset.py"):
        assert metodo(percorso) is None, percorso


def test_i_requisiti():
    def nomi(file):
        righe = _testo(RADICE / file).splitlines()
        return {re.split(r"[<>=!~]", r)[0].strip().lower() for r in righe if r.strip() and not r.startswith(("#", "-r"))}

    programma = nomi("requirements.txt")
    sviluppo = nomi("requirements-dev.txt")
    assert {"wxpython", "numpy", "scipy", "sounddevice", "accessible-output2", "pywin32", "requests", "certifi"} == programma
    assert {"pytest", "pyinstaller", "babel"} <= sviluppo and not programma & sviluppo
    assert "-r requirements.txt" in _testo(RADICE / "requirements-dev.txt").splitlines()
