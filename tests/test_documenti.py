"""
Le prove dei documenti di Ciechino per chi gioca: il manuale, il README e il CHANGELOG, che F2 mostra
come novità, dicono gli stessi numeri del codice e restano leggibili con NVDA.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, dalla revisione avversaria: il CHANGELOG diceva 33 suoni quando
erano già 54, e il manuale prometteva un bonus a cinque volte con i numeri della formula di prima.
Un numero scritto a mano in un documento invecchia in silenzio; queste prove lo confrontano con le
tabelle di suoni.py e con partita.punti_del_centro, così chi cambia il codice se ne accorge.
Il manuale si controlla anche come pagina: ben formato, con ogni collegamento interno che porta a un
titolo che esiste, e senza separatori grafici nel testo, come vuole Gabriele per lo screen reader.
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import suoni
import testi
from bersagli import raggio_del_livello, velocita_del_livello
from costanti import LIVELLO_DEI_BONUS, RAGGIO_BONUS_FATTORE, VERSIONE
from partita import punti_del_centro

RADICE = Path(__file__).resolve().parent.parent
# Una riga di trattini, di uguali o di sottolineature: i separatori grafici che non si scrivono.
SEPARATORE = re.compile(r"-{3,}|={3,}|_{3,}")
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


def test_niente_separatori_grafici_nei_documenti():
    testo_del_manuale = "".join(_manuale().testo)
    for nome, testo in (("manuale.html", testo_del_manuale), ("README.md", _leggi("README.md")), ("CHANGELOG.md", _leggi("CHANGELOG.md"))):
        trovato = SEPARATORE.search(testo)
        assert trovato is None, f"{nome}: {trovato.group(0) if trovato else ''}"
