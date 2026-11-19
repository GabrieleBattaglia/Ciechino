"""
Le prove dei documenti di Ciechino per chi gioca: il manuale, il README e il CHANGELOG, che F2 mostra
come novità, dicono gli stessi numeri del codice e restano leggibili con NVDA.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, dalla revisione avversaria: il CHANGELOG diceva 33 suoni quando
erano già 54, e il manuale prometteva un bonus a cinque volte con i numeri della formula di prima.
Un numero scritto a mano in un documento invecchia in silenzio; queste prove lo confrontano con le
tabelle di suoni.py e con partita.punti_del_centro, così chi cambia il codice se ne accorge.
Il manuale si controlla anche come pagina: ben formato, con ogni collegamento interno che porta a un
titolo che esiste, e senza separatori grafici nel testo né righe vuote nel sorgente, come vuole
Gabriele per lo screen reader.
Dopo i banchi d'ascolto del 19 novembre 2026 il manuale dice anche la scala dell'alzo (D19), quante
sono le forme dei bersagli e di quanto varia la loro escursione (D18), e le forme del proiettile (D17):
anche questi numeri si confrontano con costanti.py.
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import suoni
import testi
from bersagli import raggio_del_livello, velocita_del_livello
from costanti import (
    ALZO_FREQUENZE,
    ESCURSIONE_A_CASO,
    FORME_LFO,
    LIVELLO_DEI_BONUS,
    PROIETTILE_FORMA,
    PROIETTILE_FORME,
    RAGGIO_BONUS_FATTORE,
    VERSIONE,
)
from partita import punti_del_centro

RADICE = Path(__file__).resolve().parent.parent
# Una riga di trattini, di uguali o di sottolineature: i separatori grafici che non si scrivono.
SEPARATORE = re.compile(r"-{3,}|={3,}|_{3,}")
# I numeri che il manuale scrive a parole, per contare le forme dei bersagli.
A_PAROLE = ("zero", "una", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "undici", "dodici", "tredici",
            "quattordici", "quindici", "sedici", "diciassette", "diciotto", "diciannove", "venti")
# Come il manuale dice le forme del proiettile, nell'elenco delle Impostazioni.
PROIETTILI_A_PAROLE = {"seno": "il seno", "triangolo": "il triangolo", "sega_morbida": "la sega morbida"}
# Gli elementi HTML che non si chiudono.
VUOTI = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


def _leggi(nome):
    return (RADICE / nome).read_text(encoding="utf-8")


def _blocco_della_versione(changelog, versione):
    """Il testo del blocco del CHANGELOG di quella versione, fino al titolo seguente."""
    trovato = re.search(rf"^## \[{re.escape(versione)}\][^\n]*\n(.*?)(?=^## |\Z)", changelog, re.MULTILINE | re.DOTALL)
    assert trovato, f"nel CHANGELOG manca il blocco della {versione}"
    return trovato.group(1)


class _Pagina(HTMLParser):
    """Legge il manuale: gli elementi aperti e chiusi in ordine, gli id, i collegamenti interni e il testo visibile."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.aperti = []
        self.errori = []
        self.id = []
        self.collegamenti = []
        self.testo = []
        self._dentro_lo_stile = False

    def handle_starttag(self, tag, attributi):
        attributi = dict(attributi)
        if "id" in attributi:
            self.id.append(attributi["id"])
        if tag == "a" and str(attributi.get("href", "")).startswith("#"):
            self.collegamenti.append(attributi["href"][1:])
        if tag == "style":
            self._dentro_lo_stile = True
        if tag not in VUOTI:
            self.aperti.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag, attributi):
        attributi = dict(attributi)
        if "id" in attributi:
            self.id.append(attributi["id"])

    def handle_endtag(self, tag):
        if tag == "style":
            self._dentro_lo_stile = False
        if tag in VUOTI:
            return
        if not self.aperti or self.aperti[-1][0] != tag:
            self.errori.append(f"riga {self.getpos()[0]}: chiude {tag}, ma è aperto {self.aperti[-1] if self.aperti else 'niente'}")
            return
        self.aperti.pop()

    def handle_data(self, dati):
        if not self._dentro_lo_stile:
            self.testo.append(dati)


def _manuale():
    pagina = _Pagina()
    pagina.feed(_leggi("manuale.html"))
    pagina.close()
    return pagina


def test_il_changelog_dice_quanti_sono_i_suoni():
    blocco = _blocco_della_versione(_leggi("CHANGELOG.md"), VERSIONE)
    trovato = re.search(r"(\d+) in tutto, in (\d+) gruppi", blocco)
    assert trovato, "il blocco della versione non dice più quanti sono i suoni: «N in tutto, in M gruppi»"
    assert (int(trovato.group(1)), int(trovato.group(2))) == (len(suoni.EVENTI), len(suoni.GRUPPI))


def test_il_manuale_dice_quanti_sono_i_suoni():
    trovato = re.search(r"in tutto sono (\d+), in (\d+) gruppi", _leggi("manuale.html"))
    assert trovato, "il manuale non dice più quanti sono i suoni: «in tutto sono N, in M gruppi»"
    assert (int(trovato.group(1)), int(trovato.group(2))) == (len(suoni.EVENTI), len(suoni.GRUPPI))


def test_il_manuale_da_i_punti_veri_del_bonus():
    """D13: il bonus vale cinque volte un normale del livello, e il manuale lo dice coi numeri di punti_del_centro."""
    livello = LIVELLO_DEI_BONUS
    bonus = SimpleNamespace(bonus=True, raggio=raggio_del_livello(livello) * RAGGIO_BONUS_FATTORE, velocita_scalare=velocita_del_livello(livello) + 7.0)
    di_striscio, nel_centro = (punti_del_centro(bonus, precisione, livello=livello) for precisione in (0.0, 1.0))
    frase = f"da {testi.intero(di_striscio)} a {testi.intero(nel_centro)} punti"
    assert frase in _leggi("manuale.html"), f"il manuale non dice «{frase}» per il bonus al livello {livello}"


def test_il_manuale_e_ben_formato():
    pagina = _manuale()
    assert not pagina.errori, pagina.errori
    assert not pagina.aperti, f"elementi rimasti aperti: {pagina.aperti}"


def test_gli_id_del_manuale_sono_unici_e_ogni_collegamento_interno_porta_a_uno():
    pagina = _manuale()
    doppi = sorted({i for i in pagina.id if pagina.id.count(i) > 1})
    assert not doppi, doppi
    mancanti = sorted(set(pagina.collegamenti) - set(pagina.id))
    assert not mancanti, mancanti


def test_il_manuale_non_ha_righe_vuote():
    vuote = [numero for numero, riga in enumerate(_leggi("manuale.html").splitlines(), 1) if not riga.strip()]
    assert not vuote, f"righe vuote nel manuale: {vuote}"


def test_il_manuale_da_la_scala_vera_dell_alzo():
    """D19: l'alzo sale da ALZO_FREQUENZE[0] a ALZO_FREQUENZE[1] hertz, e il manuale lo dice."""
    frase = f"da {testi.intero(ALZO_FREQUENZE[0])} a {testi.intero(ALZO_FREQUENZE[1])} hertz"
    assert frase in _leggi("manuale.html"), f"il manuale non dice «{frase}» per l'alzo"


def test_il_manuale_conta_le_forme_dei_bersagli():
    """D18: tante forme nel titolo e nell'elenco quante in FORME_LFO, e la variazione vera dell'escursione."""
    manuale = _leggi("manuale.html")
    quante = A_PAROLE[len(FORME_LFO)]
    assert f"Le {quante} forme</h3>" in manuale, f"il titolo delle forme non dice «Le {quante} forme»"
    sezione = re.search(r'<h3 id="forme">.*?<ul>(.*?)</ul>', manuale, re.DOTALL)
    assert sezione, "nel manuale manca l'elenco delle forme"
    assert sezione.group(1).count("<li>") == len(FORME_LFO)
    variazione = round(max(1.0 - ESCURSIONE_A_CASO[0], ESCURSIONE_A_CASO[1] - 1.0) * 100)
    assert f"al massimo del {variazione}%" in manuale, f"il manuale non dice che l'escursione varia al massimo del {variazione}%"


def test_il_manuale_dice_le_forme_del_proiettile():
    """D17: le forme di PROIETTILE_FORME, una per una, e quella di serie."""
    manuale = _leggi("manuale.html")
    assert set(PROIETTILI_A_PAROLE) == set(PROIETTILE_FORME), "una forma del proiettile nuova va detta anche nel manuale"
    for forma in PROIETTILE_FORME:
        assert f"{PROIETTILI_A_PAROLE[forma]}, " in manuale, f"il manuale non descrive la forma {forma} del proiettile"
    assert f"All'inizio è {PROIETTILI_A_PAROLE[PROIETTILE_FORMA]}." in manuale


def test_niente_separatori_grafici_nei_documenti():
    testo_del_manuale = "".join(_manuale().testo)
    for nome, testo in (("manuale.html", testo_del_manuale), ("README.md", _leggi("README.md")), ("CHANGELOG.md", _leggi("CHANGELOG.md"))):
        trovato = SEPARATORE.search(testo)
        assert trovato is None, f"{nome}: {trovato.group(0) if trovato else ''}"
