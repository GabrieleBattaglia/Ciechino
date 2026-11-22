"""
Le prove dei documenti di Ciechino per chi gioca: il manuale, il README e il CHANGELOG, che F2 mostra
come novità, dicono gli stessi numeri del codice e restano leggibili con NVDA.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, dalla revisione avversaria: il CHANGELOG diceva 33 suoni quando
erano già 54, e il manuale prometteva un bonus a cinque volte con i numeri della formula di prima.
Un numero scritto a mano in un documento invecchia in silenzio; queste prove lo confrontano con le
tabelle di suoni.py, con costanti.py, con la fisica e con le funzioni che scrivono i testi, così chi
cambia il codice se ne accorge.
Il manuale si controlla anche come pagina: ben formato, con ogni collegamento interno che porta a un
titolo che esiste, e senza separatori grafici nel testo né righe vuote nel sorgente, come vuole
Gabriele per lo screen reader.
Dalle versioni di prima restano i controlli che valgono ancora: la scala dell'alzo e le forme dei
bersagli e del proiettile (D17, D18, D19), le voci dei menu col loro tasto (D20), i livelli del sommario e
della palestra (D21, D22), gli Ascolti (D23), il secondo di silenzio, i blocchi del braille e la
legenda del sommario (D24), Esc nella mira (D25), la classifica in griglia (D27); e ogni versione ha il
suo blocco nel CHANGELOG, che è la storia e resta com'era, coi numeri di allora.
Dalla 2.0.0, del 22 novembre 2026, la cupola (D28), la mira manuale col suo tempo (D29) e la riserva più
larga (D30): il manuale, il README e il blocco della 2.0.0 dicono il guscio, il settore, i valori del tiro
in numeri, i tiri più dolci di fisica.soluzione, il suono dello spazio, i tasti della mira, le tacche, il
tempo della mira a ogni livello, la riserva, la precisione globale con la retta fino al punto della cupola
più lontano, con gli esempi calcolati da volo.Volo per un bersaglio nel mezzo della cupola e uno in un
angolo in fondo, le statistiche nuove e gli Ascolti coi loro quattordici passi. Il cubo e le scivolate
restano soltanto dove il testo dice che erano di prima.
"""

import importlib.util
import math
import random
import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ascolti
import classifica
import fisica
import resoconti
import suoni
import testi
import version
from bersagli import durata_del_bonus, mezza_escursione_ottave, quanti_bersagli, raggio_del_livello, velocita_del_livello
from costanti import (
    ALTEZZA_MINIMA_BERSAGLIO,
    ALTEZZA_TETTO_DEL_SUONO,
    ALZO_FREQUENZE,
    ALZO_MASSIMO,
    ALZO_MINIMO,
    ASCOLTI_FORME_PER_PASSO,
    ASCOLTI_GRANDEZZE,
    ASCOLTI_LIVELLO_DEL_BONUS,
    ASCOLTI_PUNTI,
    AZIMUT_DEL_PAN,
    AZIMUT_MASSIMO,
    BLOCCO_DEL_BRAILLE,
    COLPI_BONUS,
    COLPI_INIZIALI,
    COLPI_PER_LIVELLO,
    CUPOLA_RAGGIO,
    CUPOLA_RAGGIO_INTERNO,
    DB_PER_METRO,
    DURATA_DELLA_SCALA,
    ELEVAZIONE_MASSIMA,
    ESCURSIONE_A_CASO,
    ESPONENTE_POTENZA,
    FORME_LFO,
    FREQUENZA_AL_SUOLO,
    GRAVITA,
    GRAZIA_DELL_ATTESA,
    GUADAGNO_DB_MINIMO,
    GUADAGNO_DB_VICINO,
    LIVELLO_DEI_BONUS,
    MARGINE_DI_COPERTURA,
    METRI_PER_OTTAVA,
    OTTAVE_DEL_SUONO,
    PALESTRA_FINO_AL_LIVELLO,
    PARTENZA_DELLA_MIRA,
    POTENZA_DB_MINIMO,
    PROIETTILE_FORMA,
    PROIETTILE_FORME,
    PUNTATORE_FORMA,
    RAGGIO_BONUS_FATTORE,
    RAGGIO_INIZIALE,
    RAGGIO_MINIMO,
    RAGGIO_PROIETTILE,
    RIDUZIONE_DEL_TEMPO_DELLA_MIRA,
    RITARDO_DEL_RESOCONTO,
    SETTORE_AZIMUT,
    SOMMARIO_NUMERI_FINO_AL_LIVELLO,
    SOMMARIO_PAROLE_FINO_AL_LIVELLO,
    SOMMARIO_SOGLIA_INCLINAZIONE,
    SOMMARIO_SOGLIA_LATERALE,
    SOMMARIO_SOGLIA_PROFONDITA,
    SOMMARIO_SOGLIA_VERTICALE,
    TACCA_DURATA,
    TACCA_FORMA,
    TACCA_FREQUENZE,
    TACCHE,
    TEMPO_DELLA_MIRA_MINIMO,
    TICK_FREQUENZE,
    TIRI_PER_IL_PRIMATO_DELLA_MEDIA,
    VELOCITA_DELLA_MIRA,
    VELOCITA_MASSIMA,
    VERSIONE,
)
from partita import punti_del_centro, tempo_della_mira, volo_d_esempio
from sintesi import frequenze_delle_tacche

RADICE = Path(__file__).resolve().parent.parent
# Una riga di trattini, di uguali o di sottolineature: i separatori grafici che non si scrivono.
SEPARATORE = re.compile(r"-{3,}|={3,}|_{3,}")
# I numeri che i documenti scrivono a parole, per contare le forme, i passi e i colpi.
A_PAROLE = ("zero", "una", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "undici", "dodici", "tredici",
            "quattordici", "quindici", "sedici", "diciassette", "diciotto", "diciannove", "venti")
# Le decine a parole della prima pagina del manuale, che dice la cupola senza cifre.
DECINE_A_PAROLE = {30: "trenta", 40: "quaranta", 80: "ottanta"}
# Come il manuale dice le forme del proiettile, nell'elenco delle Impostazioni.
PROIETTILI_A_PAROLE = {"seno": "il seno", "triangolo": "il triangolo", "sega_morbida": "la sega morbida"}
# I livelli come il manuale li conta, per la palestra, il sommario e il tempo della mira: «dal ventiquattresimo».
ORDINALI = ("", "primo", "secondo", "terzo", "quarto", "quinto", "sesto", "settimo", "ottavo", "nono", "decimo", "undicesimo", "dodicesimo",
            "tredicesimo", "quattordicesimo", "quindicesimo", "sedicesimo", "diciassettesimo", "diciottesimo", "diciannovesimo", "ventesimo",
            "ventunesimo", "ventiduesimo", "ventitreesimo", "ventiquattresimo")
# Come il manuale dice la soglia in metri del sommario, uguale per i tre assi, e la grazia dello spazio nella palestra: un
# valore nuovo va scritto anche qui, e nel manuale.
SOGLIE_A_PAROLE = {0.5: "Uno scarto sotto il mezzo metro"}
GRAZIE_A_PAROLE = {0.5: "o nel mezzo secondo che segue"}
# Come il manuale dice il ritardo del resoconto dopo l'effetto (D24).
RITARDI_A_PAROLE = {1.0: "dopo un secondo di silenzio"}
# Come il manuale dice quanto un valore della mira mette per un decimo della scala, e per andare dal centro a un capo (D29).
DECIMI_A_PAROLE = {0.3: "un decimo ogni tre decimi di secondo"}
MEZZE_SCALE_A_PAROLE = {1.5: "arriva a un capo in un secondo e mezzo"}
# I tasti fuori dai menu, coi nomi della guida F1, come li scrive il manuale nel suo elenco.
TASTI_NEL_MANUALE = {
    "Spazio": "<kbd>Spazio</kbd>",
    "Frecce sinistra e destra": "<kbd>Freccia sinistra</kbd> e <kbd>Freccia destra</kbd>",
    "Frecce su e giù": "<kbd>Freccia su</kbd> e <kbd>Freccia giù</kbd>",
    "Home e Fine": "<kbd>Home</kbd> e <kbd>Fine</kbd>",
    "Invio": "<kbd>Invio</kbd>",
    "Esc": "<kbd>Esc</kbd>",
    "F5": "<kbd>F5</kbd>",
    "F6": "<kbd>F6</kbd>",
    "F7": "<kbd>F7</kbd>",
}
# Gli elementi HTML che non si chiudono.
VUOTI = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
# Le parole delle versioni di prima: dove i documenti le dicono, la frase dice anche che sono di prima. Le scivolate della mira
# si dicono con l'articolo, perché l'onda quadra può sembrare fatta di «brevi scivolate», che col cubo non c'entrano.
PAROLE_DI_PRIMA = ("cubo", "le scivolate", "la scivolata", "proiettile perduto", "proiettili perduti")
SEGNI_DI_PRIMA = ("prima della", "fino alla", "di prima")


def _leggi(nome):
    return (RADICE / nome).read_text(encoding="utf-8")


# Il mezzo della cupola, dove sta il bersaglio degli Ascolti, e l'angolo in fondo, a terra contro il lato sinistro: gli
# esempi della precisione globale (D26, D28).
MEZZO = np.array(fisica.MEZZO_DELLA_CUPOLA)
ANGOLO_IN_FONDO = fisica.punto_della_cupola(CUPOLA_RAGGIO, -SETTORE_AZIMUT, math.degrees(math.asin(ALTEZZA_MINIMA_BERSAGLIO / CUPOLA_RAGGIO)))


def _lontananza_minima():
    """(distanza, punto): dove la retta della precisione globale è più corta, cioè il punto con la distanza più piccola dal
    punto della cupola più lontano da lui. Quella distanza è convessa, e la cupola è simmetrica: il punto sta sul piano
    davanti, x uguale a zero, e una griglia che si stringe intorno al migliore lo trova."""
    centro, passo = np.array((0.0, MEZZO[1], MEZZO[2])), 4.0
    for _giro in range(8):
        griglia = np.array([(0.0, y, z) for y in centro[1] + passo * np.arange(-5, 6) for z in centro[2] + passo * np.arange(-5, 6)])
        centro = griglia[int(np.argmin(fisica.distanza_dal_punto_piu_lontano(griglia)))]
        passo /= 4.0
    return float(fisica.distanza_dal_punto_piu_lontano(centro)), centro


def _globale(punto, distanza):
    """La precisione globale, scritta come la scrive il gioco, di un tiro a vuoto costruito a mano, passato a distanza metri dal centro
    di un bersaglio fermo in punto: la calcola volo.Volo, come in partita."""
    from bersagli import Bersaglio
    from fisica import Traiettoria
    from volo import Volo

    bersaglio = Bersaglio(numero=1, forma="seno", raggio=RAGGIO_INIZIALE, partenza=punto)
    volo = Volo(traiettoria=Traiettoria(0.0, 40.0, 30.0), partenza=0.0, durata=3.0, colpito=None, punto_finale=(0.0, 0.0, 70.0), precisione=None,
                distanza_al_centro=None, vicino=bersaglio, distanza_dal_vicino=distanza - RAGGIO_INIZIALE, scarto=None, altezza_massima=25.0, gittata=70.0,
                laterale_finale=0.0, tempo_del_passaggio=0.0)
    return testi.percento(volo.precisione_globale, 1)


def _vocale(percentuale):
    """Vero se la percentuale si legge con una vocale davanti: otto, ottanta, uno, undici."""
    intera = percentuale.split(",")[0].rstrip("%")
    return intera.startswith("8") or intera in ("1", "11")


def _con_articolo(percentuale):
    """La percentuale con l'articolo che vuole davanti: «l'88,2%», «l'11,0%», «il 98,6%», «lo 0%»."""
    if percentuale in ("0,0%", "0%"):
        return "lo 0%"
    return f"l'{percentuale}" if _vocale(percentuale) else f"il {percentuale}"


def _al(percentuale):
    """La percentuale con la preposizione: «al 48%», «all'8%», «all'80%»."""
    return f"all'{percentuale}" if _vocale(percentuale) else f"al {percentuale}"


def _secondi(x):
    """Un numero di secondi come lo scrive il manuale: 13 o 13,5."""
    return testi.intero(x) if float(x).is_integer() else testi.numero(x)


def _blocco_della_versione(changelog, versione):
    """Il testo del blocco del CHANGELOG di quella versione, fino al titolo seguente."""
    trovato = re.search(rf"^## \[{re.escape(versione)}\][^\n]*\n(.*?)(?=^## |\Z)", changelog, re.MULTILINE | re.DOTALL)
    assert trovato, f"nel CHANGELOG manca il blocco della {versione}"
    return trovato.group(1)


def _sezione(manuale, identificativo):
    """L'HTML di un capitolo del manuale, dal suo titolo h2 o h3 al titolo seguente."""
    trovato = re.search(rf'<h[23] id="{re.escape(identificativo)}">.*?(?=<h[23] |</body>)', manuale, re.DOTALL)
    assert trovato, f"nel manuale manca il capitolo {identificativo}"
    return trovato.group(0)


def _senza_tag(html):
    return re.sub(r"<[^>]+>", "", html)


def _voci(html):
    """Le voci di un elenco, cioè il contenuto di ogni li, nell'ordine."""
    return re.findall(r"<li>(.*?)</li>", html, re.DOTALL)


def _tiro(direzione, alzo, potenza):
    return fisica.Traiettoria(*fisica.parametri_del_tiro(direzione, alzo, potenza))


def _punto(distanza, altezza, azimut=0.0):
    """Il punto a quella distanza da chi gioca, a quell'altezza e con quell'azimut, positivo a destra."""
    return fisica.punto_della_cupola(distanza, azimut, math.degrees(math.asin(altezza / distanza)))


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


def _volo_d_esempio_del_sommario():
    """Il tiro a vuoto costruito a mano degli esempi del sommario: un bersaglio di 2 metri, passato a 15 metri dal centro."""
    from bersagli import Bersaglio
    from fisica import Traiettoria
    from volo import Misure, Volo

    bersaglio = Bersaglio(numero=1, forma="seno", raggio=2.0, partenza=(0.0, 20.0, 50.0))
    misure = Misure(arrivo=2.0, laterale=1.9, verticale=15.8, inclinazione=-18.5, ricaduta=2.5, profondita=24.7)
    return Volo(traiettoria=Traiettoria(0.0, 40.0, 30.0), partenza=0.0, durata=3.0, colpito=None, punto_finale=(0.0, 0.0, 70.0), precisione=None,
                distanza_al_centro=None, vicino=bersaglio, distanza_dal_vicino=13.0, scarto=None, altezza_massima=25.0, gittata=70.0, laterale_finale=0.0,
                misure=misure)


class _ComandiFinti:
    """Al posto della finestra, per leggere la tabella dei menu e i tasti della guida senza aprirla: ogni comando vale None."""

    def __getattr__(self, nome):
        return None


# Le prove di sempre: la pagina, i suoni, i punti, le forme e i menu.

def test_il_changelog_dice_quanti_sono_i_suoni():
    """Il blocco più recente del CHANGELOG che dice quanti sono i suoni, «N in tutto, in M gruppi», dice i numeri veri: una versione
    senza suoni nuovi, come la 0.11.0, non lo ripete."""
    changelog = _leggi("CHANGELOG.md")
    assert re.search(rf"^## \[{re.escape(VERSIONE)}\] - {re.escape(version.__date__)}$", changelog, re.MULTILINE), \
        f"nel CHANGELOG manca il blocco della {VERSIONE} del {version.__date__}"
    trovato = re.search(r"(\d+) in tutto, in (\d+) gruppi", changelog)
    assert trovato, "il CHANGELOG non dice più quanti sono i suoni: «N in tutto, in M gruppi»"
    assert (int(trovato.group(1)), int(trovato.group(2))) == (len(suoni.EVENTI), len(suoni.GRUPPI))


def test_il_manuale_dice_quanti_sono_i_suoni():
    trovato = re.search(r"in tutto sono (\d+), in (\d+) gruppi", _leggi("manuale.html"))
    assert trovato, "il manuale non dice più quanti sono i suoni: «in tutto sono N, in M gruppi»"
    assert (int(trovato.group(1)), int(trovato.group(2))) == (len(suoni.EVENTI), len(suoni.GRUPPI))


def test_l_esempio_dell_elenco_dei_suoni_e_quello_vero():
    """Il manuale cita una riga di suoni_di_ciechino.txt, quella dello sparo: è la riga che scrive strumenti/elenco_suoni.py dalle
    tabelle di suoni.py, non quella di un file vecchio."""
    specifica = importlib.util.spec_from_file_location("elenco_suoni_della_prova", RADICE / "strumenti" / "elenco_suoni.py")
    elenco = importlib.util.module_from_spec(specifica)
    specifica.loader.exec_module(elenco)
    riga = elenco._voce(suoni.AZIONI["sparo"], suoni.EVENTI["sparo"])
    assert f"per esempio: {riga}" in _leggi("manuale.html"), f"il manuale non cita la riga vera dello sparo: «{riga}»"


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


def test_la_versione_del_manuale_e_quella_vera():
    """L'intestazione del manuale dice la versione e il suo giorno, come version.py."""
    frase = f"<p>Versione {VERSIONE}, {testi.data_lunga(version.__date__)}.</p>"
    assert frase in _leggi("manuale.html"), f"il manuale non comincia con «{frase}»"


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


def test_il_manuale_dice_ogni_voce_dei_menu_col_suo_tasto():
    """D20: Scheda audio con Ctrl+Maiusc+X, e ogni altra voce dei menu con un tasto rapido, sta nel manuale come
    «<li>Voce, <kbd>Tasto</kbd>», col tasto detto come nella guida F1, Maiusc al posto di Shift."""
    from gui.finestra import FinestraPrincipale

    finta = _ComandiFinti()
    finta.voci_menu = lambda: FinestraPrincipale.voci_menu(finta)
    manuale = _leggi("manuale.html")
    con_tasto = [(voce, tasto) for _titolo, comandi in FinestraPrincipale.voci_guida(finta) for voce, tasto in comandi if tasto]
    assert ("Scheda audio", "Ctrl+Maiusc+X") in con_tasto, "nei menu manca Scheda audio con Ctrl+Maiusc+X"
    mancanti = [f"{voce}, {tasto}" for voce, tasto in con_tasto if f"<li>{voce}, <kbd>{tasto}</kbd>" not in manuale]
    assert not mancanti, f"voci dei menu che il manuale non dice col loro tasto: {mancanti}"


def test_il_manuale_dice_i_tasti_della_guida_nel_loro_ordine():
    """D25 e D29: l'elenco dei tasti fuori dai menu del manuale ha una voce per ogni tasto della guida F1, nello stesso ordine, che
    comincia col tasto e con quello che la guida dice, fino al primo punto e virgola; i tasti della mira compresi."""
    from gui.finestra import FinestraPrincipale

    tasti = FinestraPrincipale.tasti_fuori_dai_menu(_ComandiFinti())
    assert [tasto for tasto, _cosa in tasti] == list(TASTI_NEL_MANUALE), "un tasto nuovo della guida va detto anche nel manuale, e in TASTI_NEL_MANUALE"
    voci = _voci(re.search(r"<ul>.*?</ul>", _sezione(_leggi("manuale.html"), "tasti"), re.DOTALL).group(0))
    assert len(voci) == len(tasti), f"l'elenco dei tasti del manuale ha {len(voci)} voci, la guida {len(tasti)} tasti"
    for voce, (tasto, cosa) in zip(voci, tasti, strict=True):
        assert voce.startswith(f"{TASTI_NEL_MANUALE[tasto]}: "), f"la voce del tasto {tasto} non comincia col tasto: {voce[:60]}"
        attesa = f"{_senza_tag(TASTI_NEL_MANUALE[tasto])}: {cosa.split(';')[0]}"
        assert _senza_tag(voce).startswith(attesa), f"il manuale non dice per {tasto} quello che dice la guida: «{attesa}»"


def test_il_manuale_dice_i_livelli_della_palestra_e_del_sommario():
    """D21 e D22: i livelli del sommario coi numeri e a parole, la palestra delle attese e di Ctrl+P, le soglie del
    sommario e la grazia dello spazio, detti nel manuale coi valori di costanti.py; e l'avviso di Ctrl+P fuori dalla
    palestra, citato con le parole della vista."""
    manuale = _leggi("manuale.html")
    numeri, parole, palestra = SOMMARIO_NUMERI_FINO_AL_LIVELLO, SOMMARIO_PAROLE_FINO_AL_LIVELLO, PALESTRA_FINO_AL_LIVELLO
    frasi = [
        f"Nei primi {A_PAROLE[parole]} livelli il resoconto di ogni tiro comincia con un sommario",
        f"Nella palestra, i primi {A_PAROLE[numeri]} livelli, il sommario dà i numeri, in metri e in gradi, con un decimale e il segno",
        f"Dal {ORDINALI[numeri + 1]} al {ORDINALI[parole]} livello restano soltanto le parole",
        f"Dal {ORDINALI[parole + 1]} livello il sommario non c'è più",
        f'<h3 id="palestra">La palestra, i primi {A_PAROLE[palestra]} livelli</h3>',
        f"I primi {A_PAROLE[palestra]} livelli sono una palestra",
        f"Dal primo tiro del {ORDINALI[palestra + 1]} livello la palestra è finita",
        resoconti.pausa_solo_in_palestra(palestra).testo,
        f"un'inclinazione sotto i {testi.intero(SOMMARIO_SOGLIA_INCLINAZIONE)} gradi",
    ]
    soglie = {SOMMARIO_SOGLIA_LATERALE, SOMMARIO_SOGLIA_VERTICALE, SOMMARIO_SOGLIA_PROFONDITA}
    assert len(soglie) == 1, "il manuale dice una soglia sola per laterale, verticale e profondità"
    soglia = soglie.pop()
    assert soglia in SOGLIE_A_PAROLE, f"la soglia del sommario di {soglia} metri va detta nel manuale, e in SOGLIE_A_PAROLE"
    assert GRAZIA_DELL_ATTESA in GRAZIE_A_PAROLE, f"la grazia di {GRAZIA_DELL_ATTESA} secondi va detta nel manuale, e in GRAZIE_A_PAROLE"
    frasi.extend((SOGLIE_A_PAROLE[soglia], GRAZIE_A_PAROLE[GRAZIA_DELL_ATTESA]))
    mancanti = [frase for frase in frasi if frase not in manuale]
    assert not mancanti, f"frasi della palestra e del sommario che il manuale non dice: {mancanti}"
    # La palestra delle attese e quella dei numeri del sommario oggi coincidono, e il manuale lo dà per inteso.
    assert numeri == palestra, "il sommario coi numeri e la palestra non vanno più fino allo stesso livello: rileggi il manuale"


def test_il_manuale_e_il_changelog_dicono_gli_ascolti():
    """D23, rifatti con la 2.0.0: il capitolo degli Ascolti, la loro voce soltanto nel menu Aiuto, un passo dell'elenco per ogni passo
    vero, col suo titolo, le forme in una voce sola, e i numeri dei testi presi dalle stesse funzioni degli Ascolti; la riga della prima
    apertura con le parole della vista; il passo da riascoltare per ritarare l'orecchio; il blocco della 0.9.0 e quello della 2.0.0
    nel CHANGELOG."""
    from gui.dialoghi import intestazione_del_passo
    from gui.finestra import frase_della_prima_apertura

    manuale = _leggi("manuale.html")
    assert '<h2 id="ascolti">Gli Ascolti</h2>' in manuale and manuale.count("<li>Ascolti: ") == 1
    impostazioni = re.search(r'<h3 id="menu-impostazioni">.*?<ul>(.*?)</ul>', manuale, re.DOTALL)
    aiuto = re.search(r'<h3 id="menu-aiuto">.*?<ul>(.*?)</ul>', manuale, re.DOTALL)
    assert impostazioni and aiuto and "<li>Ascolti: " not in impostazioni.group(1) and "<li>Ascolti: " in aiuto.group(1)
    elenco = ascolti.passi()
    assert intestazione_del_passo(elenco[0], 1, len(elenco)) in manuale
    sezione = _sezione(manuale, "ascolti-passi")
    # Le forme dell'oscillazione stanno in due passi veri e in una voce sola dell'elenco.
    forme = [passo for passo in elenco if passo.nome.startswith("forme")]
    assert sezione.count("<li>") == len(elenco) - (len(forme) - 1)
    assert f"I passi sono {A_PAROLE[len(elenco)]}" in sezione
    senza_titolo = [passo.titolo for passo in elenco if passo not in forme and f"<li>{passo.titolo}: " not in sezione]
    assert not senza_titolo, f"passi degli Ascolti che l'elenco del manuale non dice col loro titolo: {senza_titolo}"
    assert f"<li>Le forme dell'oscillazione, in {A_PAROLE[len(forme)]} passi, {A_PAROLE[ASCOLTI_FORME_PER_PASSO]} per volta" in sezione
    capi = {nome: np.array(punto) for nome, punto in ascolti.punti_dei_capi().items()}
    lato = math.degrees(math.atan2(capi["destra"][0], capi["destra"][2]))
    vicino, lontano = (fisica.distanza_dal_cannone(capi[capo]) for capo in ("fronte", "fondo"))
    piccolo, primo, gigante = ASCOLTI_GRANDEZZE
    volo = volo_d_esempio()
    assert piccolo == RAGGIO_MINIMO == 0.5, "il manuale dice il raggio più piccolo degli Ascolti a parole, mezzo metro"
    frasi = [
        f"a {testi.metri(fisica.distanza_dal_cannone(ascolti.CENTRO))} dal cannone e a {testi.metri(ascolti.CENTRO[1])} d'altezza",
        f"arriva fino a {testi.gradi(round(lato))}, vicino ai {testi.intero(SETTORE_AZIMUT)} del lato",
        # Come il testo del passo, dalle prove del 22 novembre 2026: la distanza è dal centro della cupola, sotto il cannone, non da te.
        f"sempre a {testi.intero(ascolti.DISTANZA_DEL_MEZZO)} metri dal centro della cupola, sotto i tuoi piedi, scende fino a terra, col centro a "
        f"{testi.metri(capi['terra'][1])} dal prato, "
        f"e sale lungo la volta della cupola fino a {testi.metri(capi['cima'][1])} d'altezza",
        f"la nota va da circa {testi.intero(fisica.frequenza_di(capi['terra'][1]))} a circa {testi.intero(fisica.frequenza_di(capi['cima'][1]))} hertz",
        f"viene vicino al massimo, a {testi.metri(vicino)} dal cannone, e va lontano al massimo, a {testi.metri(lontano)}",
        f"cambia soltanto il volume, di circa {testi.intero(fisica.guadagno_db_di(vicino) - fisica.guadagno_db_di(lontano))} decibel",
        f"salta in {A_PAROLE[ASCOLTI_PUNTI]} punti presi a caso nella cupola",
        ascolti.riga_del_punto(1, (-12.3, 31.0, 47.5)),
        f"a {testi.intero(primo)} metri, il più grande, poi a {testi.intero(gigante)} metri, gigantesco",
        f"da un tremolio di {testi.numero(12 * mezza_escursione_ottave(piccolo), 1)} semitoni a {testi.numero(12 * mezza_escursione_ottave(gigante), 1)} "
        "semitoni sopra e sotto il centro",
        f"la mira più breve, quella di {testi.intero(TEMPO_DELLA_MIRA_MINIMO)} secondi",
        f"che sale fino a {testi.metri(volo.altezza_massima)} e cade a {testi.metri(volo.gittata)} davanti a te, dopo {testi.numero(volo.durata)} secondi di volo",
        f"ciascuno vicino, a {testi.metri(vicino)}, e lontano, a {testi.metri(lontano)}",
        f"in {testi.intero(durata_del_bonus(ASCOLTI_LIVELLO_DEL_BONUS))} secondi, come al {ORDINALI[ASCOLTI_LIVELLO_DEL_BONUS]} livello",
    ]
    mancanti = [frase for frase in frasi if frase not in sezione]
    assert not mancanti, f"numeri dei passi degli Ascolti che il manuale non dice come il codice: {mancanti}"
    # Il passo della mira fa correre ogni valore dal centro al massimo e dal massimo al minimo, come dice il manuale.
    assert ascolti.CORSE_DELLA_MIRA == ((PARTENZA_DELLA_MIRA, 1.0), (1.0, 0.0)) and "dal centro fino al massimo, e dal massimo fino al minimo" in sezione
    nomi = [passo.nome for passo in elenco]
    ritarare = _sezione(manuale, "ritarare")
    assert f"al passo {nomi.index('distanza') + 1}, La distanza" in ritarare and f"il test stereo, il passo {nomi.index('stereo') + 1}" in ritarare
    frase = frase_della_prima_apertura()
    assert frase in manuale and "Esc" in frase and "Impostazioni" not in frase and "menu Aiuto" in frase
    assert "sala briefing dei suoni" in manuale
    changelog = _leggi("CHANGELOG.md")
    assert "Ascolti" in _blocco_della_versione(changelog, "0.9.0") and "Ascolti" in _leggi("README.md")
    blocco = _blocco_della_versione(changelog, "2.0.0")
    assert "Gli Ascolti sono rifatti" in blocco and "Il tempo della mira" in blocco and f"i passi sono adesso {A_PAROLE[len(elenco)]}" in blocco


def test_il_secondo_collaudo_nel_manuale_e_nel_changelog():
    """D24, D25 e D26 nel manuale coi valori veri: il ritardo del resoconto, i blocchi del display braille con la legenda di
    resoconti.py e la G calcolata dal volo, gli annunci d'esempio di resoconti.resoconto_del_tiro, Esc nella mira col messaggio della
    vista; e un blocco del CHANGELOG per ogni versione del collaudo."""
    manuale = _leggi("manuale.html")
    assert RITARDO_DEL_RESOCONTO in RITARDI_A_PAROLE, f"il ritardo di {RITARDO_DEL_RESOCONTO} secondi va detto nel manuale, e in RITARDI_A_PAROLE"
    volo = _volo_d_esempio_del_sommario()
    sommario = resoconti.sommario_del_tiro(volo, "numeri")
    # Dalla 0.11.8 la G sta coi valori nel primo blocco, e il secondo è la legenda sola; da D28 la G divide per la cupola.
    globale = testi.percento(1.0 - volo.distanza_dal_centro / fisica.distanza_dal_punto_piu_lontano(volo.centro_nel_passaggio), 1)
    assert sommario.primo.endswith(f" G:{globale}") and len(sommario.primo) < BLOCCO_DEL_BRAILLE
    legenda = sommario.secondo.rstrip(".")
    annunci = [resoconti.resoconto_del_tiro(2, livello, volo, [0.5, 0.5, 0.5], riserva=18).annuncio for livello in (1, 5, 8)]
    frasi = [
        RITARDI_A_PAROLE[RITARDO_DEL_RESOCONTO],
        f"a blocchi di {BLOCCO_DEL_BRAILLE} caratteri",
        f"{sommario.primo}; dopo",
        legenda,
        f"al primo livello: {annunci[0]}. Al quinto: {annunci[1]} All'ottavo: {annunci[2]}",
        f"I:-45,8 G:{testi.percento(1.0 - 0.4 / fisica.distanza_dal_punto_piu_lontano(MEZZO), 1)}, un centro quasi pieno nel mezzo della cupola",
        resoconti.mira_annullata(COLPI_INIZIALI).testo,
        '<li><kbd>Esc</kbd>: durante la mira annulla il tiro',
        '<h3 id="precisione-globale">La precisione globale</h3>',
    ]
    mancanti = [frase for frase in frasi if frase not in manuale]
    assert not mancanti, f"frasi del secondo collaudo che il manuale non dice: {mancanti}"
    changelog = _leggi("CHANGELOG.md")
    for versione in ("0.9.1", "0.9.2", "0.10.0", "0.11.0"):
        assert _blocco_della_versione(changelog, versione).strip()
    assert "Esc" in _blocco_della_versione(changelog, "0.10.0") and "precisione globale" in _blocco_della_versione(changelog, "0.11.0")


def test_le_correzioni_della_0_11_8_nei_documenti():
    """Le decisioni di Gabriele del 20 novembre 2026 che valgono ancora: il sommario a parole con la seconda riga sotto, detta con le
    parole vere di resoconti.py, e il primato della media soltanto dalle partite di TIRI_PER_IL_PRIMATO_DELLA_MEDIA tiri; i blocchi
    della 0.11.0 e della 0.11.8 del CHANGELOG restano com'erano, coi numeri di allora: il proiettile perduto senza la precisione
    globale, la riga coi valori attaccati e i tre decimi di secondo del margine delle scivolate."""
    manuale, changelog = _leggi("manuale.html"), _leggi("CHANGELOG.md")
    parole = resoconti.sommario_del_tiro(_volo_d_esempio_del_sommario(), "parole")
    assert parole.a_capo and parole.riga == f"{parole.primo}\n{parole.secondo}"
    frasi = [
        f"Sulla riga sotto, sempre, come ha deciso Gabriele, vengono la precisione globale e il punto di vista: {parole.secondo}",
        f"che vale soltanto per le partite di almeno {A_PAROLE[TIRI_PER_IL_PRIMATO_DELLA_MEDIA]} tiri",
    ]
    mancanti = [frase for frase in frasi if frase not in manuale]
    assert not mancanti, f"frasi della 0.11.8 che il manuale non dice: {mancanti}"
    assert "tranne quello del proiettile perduto" in _blocco_della_versione(changelog, "0.11.0")
    blocco = _blocco_della_versione(changelog, "0.11.8")
    assert "L:+2,5 V:+7,7 P:+12,1 I:-23,2 G:91,4%" in blocco and "tre decimi di secondo" in blocco and "almeno dieci tiri" in blocco


def test_le_correzioni_della_0_11_10_restano_nel_changelog():
    """Il terzo collaudo di Gabriele, del 20 novembre 2026, fu la retta fino all'angolo più lontano del cubo: il blocco della 0.11.10 la
    racconta coi numeri di quel cubo, 43,3 metri nel centro e 86,6 in un angolo, che la cupola ha cambiato, e dice anche la voce
    Ascolti soltanto nel menu Aiuto. Il manuale non dice più la diagonale come misura di oggi."""
    blocco = _blocco_della_versione(_leggi("CHANGELOG.md"), "0.11.10")
    frasi = ["lunga 43,3 metri", "per uno in un angolo 86,6", "vale il 76,9% nel primo caso e l'88,5% nel secondo",
             "vale lo 0% nel centro del cubo e il 45,7% in un angolo", "menu Aiuto", "Impostazioni"]
    mancanti = [frase for frase in frasi if frase not in blocco]
    assert not mancanti, f"frasi della 0.11.10 sparite dal suo blocco: {mancanti}"
    assert "sulla diagonale del cubo" not in _sezione(_leggi("manuale.html"), "precisione-globale")


def test_la_classifica_ricca_della_0_12_0_nei_documenti():
    """La issue 1, decisione D27: l'esempio della griglia nel manuale è quello che scrive classifica.righe, le colonne dell'elenco
    sono quelle del codice e nel suo ordine, e l'annuncio a parole della voce nuova è quello vero; il blocco della 0.12.0 del
    CHANGELOG e il README dicono la griglia, e il CHANGELOG anche le voci vecchie senza precisione."""
    manuale, readme, changelog = _leggi("manuale.html"), _leggi("README.md"), _leggi("CHANGELOG.md")
    voci = [
        {"nome": "Gabry", "punteggio": 12340, "data": "2026-11-17T10:00:00", "livello": 4, "tiri": 37, "centri": 25, "durata_s": 725.0,
         "precisione_globale": 0.7134},
        {"nome": "Mario", "punteggio": 8120, "data": "2026-11-20T10:00:00", "livello": 3, "tiri": 30, "centri": 19, "durata_s": 581.0,
         "precisione_globale": 0.64},
        {"nome": "Anna", "punteggio": 5002, "data": "2026-11-09T10:00:00", "livello": 2, "tiri": 21, "centri": 12, "durata_s": 373.0,
         "precisione_globale": None},
    ]
    esempio = re.search(r"<pre>(.*?)</pre>", manuale, re.DOTALL)
    assert esempio and esempio.group(1).splitlines() == classifica.righe(voci, oggi=date(2026, 11, 20))
    sezione = re.search(r'<h3 id="classifica">.*?</ul>', manuale, re.DOTALL)
    assert sezione and re.findall(r"<li>([^:<]+):", sezione.group(0)) == classifica._colonne()
    assert classifica.annuncio(voci[1], 2) in manuale
    blocco = _blocco_della_versione(changelog, "0.12.0")
    for testo in (blocco, readme):
        assert "griglia allineata" in testo and "precisione globale media" in testo
    assert "n.d." in blocco and f"{testi.durata_in_cifre(725.0)}, e dall'ora in su {testi.durata_in_cifre(3725.0)}" in blocco


# La 2.0.0: la cupola, la mira manuale e la riserva più larga (D28, D29, D30).

def test_la_cupola_nei_documenti():
    """D28: il guscio fra le due sfere, il settore, la cima, il cono lasciato fuori, l'altezza minima dei centri e la zona di un
    bersaglio del primo livello, nel manuale coi valori di costanti.py e della fisica, a cifre nel capitolo della cupola e a parole
    nella prima pagina; il README e il blocco della 2.0.0 del CHANGELOG dicono la stessa cupola."""
    manuale, readme = _leggi("manuale.html"), _leggi("README.md")
    blocco = _blocco_della_versione(_leggi("CHANGELOG.md"), "2.0.0")
    tutta, prima = fisica.Zona(0.0), fisica.Zona(RAGGIO_INIZIALE)
    cima = testi.metri(tutta.esterno * math.sin(math.radians(tutta.elevazione)))
    vicina, lontana, lato = (testi.intero(x) for x in (CUPOLA_RAGGIO_INTERNO, CUPOLA_RAGGIO, SETTORE_AZIMUT))
    sezione = _sezione(manuale, "cupola")
    frasi = [
        f"un guscio fra due sfere, una vicina, di {vicina} metri di raggio, e una lontana, di {lontana}",
        f"da {lato} gradi a sinistra a {lato} gradi a destra, e sale dal prato fino a {cima} d'altezza",
        f"sopra gli {testi.intero(ELEVAZIONE_MASSIMA)} gradi d'elevazione, cioè a meno di {testi.intero(90 - ELEVAZIONE_MASSIMA)} gradi dalla verticale",
        f"almeno {testi.intero(ALTEZZA_MINIMA_BERSAGLIO)} metri sopra il prato",
        f"coi bersagli di {testi.intero(RAGGIO_INIZIALE)} metri, fra {testi.intero(prima.interno)} e {testi.intero(prima.esterno)} metri da te",
        f"{testi.numero(GRAVITA, 2)} metri al secondo per secondo",
    ]
    mancanti = [frase for frase in frasi if frase not in sezione]
    assert not mancanti, f"frasi della cupola che il manuale non dice coi numeri del codice: {mancanti}"
    parole = {valore: DECINE_A_PAROLE.get(int(valore)) for valore in (CUPOLA_RAGGIO_INTERNO, CUPOLA_RAGGIO, SETTORE_AZIMUT)}
    assert None not in parole.values(), "una misura nuova della cupola va scritta a parole anche in DECINE_A_PAROLE, e nella prima pagina"
    assert (f"comincia a {parole[CUPOLA_RAGGIO_INTERNO]} metri e finisce a {parole[CUPOLA_RAGGIO]}, si allarga per {parole[SETTORE_AZIMUT]} gradi a "
            f"sinistra e {parole[SETTORE_AZIMUT]} a destra") in _sezione(manuale, "che-cos-e")
    assert f"un guscio fra due sfere di {vicina} e {lontana} metri di raggio" in readme and f"da {lato} gradi a sinistra a {lato} gradi a destra" in readme
    for frase in (f"la vicina di {vicina} metri di raggio e la lontana di {lontana}", f"da {lato} gradi a sinistra a {lato} gradi a destra",
                  f"fino a {cima} d'altezza", f"sopra gli {testi.intero(ELEVAZIONE_MASSIMA)} gradi d'elevazione"):
        assert frase in blocco, f"il blocco della 2.0.0 non dice «{frase}»"


def test_i_valori_del_tiro_in_numeri():
    """D28: la direzione copre il settore, l'alzo va da ALZO_MINIMO ad ALZO_MASSIMO, la potenza dalla minima di ogni alzo alla
    VELOCITA_MASSIMA, con le larghezze della scala e dei decimi, come le calcola fisica; il blocco della 2.0.0 dice gli stessi
    numeri."""
    manuale = _leggi("manuale.html")
    sezione = _sezione(manuale, "valori")
    assert AZIMUT_MASSIMO == SETTORE_AZIMUT and ESPONENTE_POTENZA == 1.0, "il manuale dice la direzione sui lati della cupola e la potenza lineare"
    azimut, decimo_di_direzione = testi.intero(AZIMUT_MASSIMO), 2 * AZIMUT_MASSIMO / TACCHE
    alzo_medio = ALZO_MINIMO + 0.5 * (ALZO_MASSIMO - ALZO_MINIMO)
    minima = {alzo: fisica.velocita_minima(alzo) for alzo in (45.0, ALZO_MINIMO, ALZO_MASSIMO)}
    larga = {alzo: VELOCITA_MASSIMA - velocita for alzo, velocita in minima.items()}
    frasi = [
        f"La direzione va da {azimut} gradi a sinistra a {azimut} gradi a destra, i due lati della cupola",
        f"sono {testi.intero(decimo_di_direzione)} gradi, che a {testi.intero(ascolti.DISTANZA_DEL_MEZZO)} metri da te fanno "
        f"{testi.metri(2 * ascolti.DISTANZA_DEL_MEZZO * math.sin(math.radians(decimo_di_direzione / 2)))} di lato",
        f"L'alzo va da {testi.intero(ALZO_MINIMO)} gradi, un tiro radente, a {testi.intero(ALZO_MASSIMO)} gradi, quasi verticale",
        f"ogni decimo della scala sono {testi.intero((ALZO_MASSIMO - ALZO_MINIMO) / TACCHE)} gradi, e il 50% è {testi.intero(alzo_medio)} gradi",
        f"Sotto i {testi.intero(ALZO_MINIMO)} gradi non si scende",
        f"arriva sempre a {testi.numero(VELOCITA_MASSIMA)} metri al secondo, la velocità con cui un tiro verso l'alto salirebbe di "
        f"{testi.intero(VELOCITA_MASSIMA ** 2 / (2 * GRAVITA))} metri, il raggio della cupola",
        f"{testi.numero(minima[45.0])} metri al secondo a 45 gradi, {testi.numero(minima[ALZO_MINIMO])} a {testi.intero(ALZO_MINIMO)} gradi e "
        f"{testi.numero(minima[ALZO_MASSIMO])} a {testi.intero(ALZO_MASSIMO)}",
        f"A 45 gradi la scala della potenza è quindi larga {testi.numero(larga[45.0])} metri al secondo, e ogni decimo vale "
        f"{testi.numero(larga[45.0] / TACCHE)} metri al secondo; a {testi.intero(ALZO_MINIMO)} gradi è larga {testi.numero(larga[ALZO_MINIMO])}, e ogni "
        f"decimo vale {testi.numero(larga[ALZO_MINIMO] / TACCHE)}",
    ]
    mancanti = [frase for frase in frasi if frase not in sezione]
    assert not mancanti, f"valori del tiro che il manuale non dice come il codice: {mancanti}"
    assert round(VELOCITA_MASSIMA ** 2 / (2 * GRAVITA), 6) == CUPOLA_RAGGIO
    assert f"una tacca ogni {testi.intero(decimo_di_direzione)} gradi" in _sezione(manuale, "orientarsi")
    blocco = _blocco_della_versione(_leggi("CHANGELOG.md"), "2.0.0")
    assert (f"l'alzo va da {testi.intero(ALZO_MINIMO)} a {testi.intero(ALZO_MASSIMO)} gradi, e la potenza arriva sempre a {testi.numero(VELOCITA_MASSIMA)} "
            f"metri al secondo, mentre il suo minimo segue l'alzo, {testi.numero(minima[45.0])} metri al secondo a 45 gradi e "
            f"{testi.numero(minima[ALZO_MINIMO])} a {testi.intero(ALZO_MINIMO)}") in blocco


def test_i_tiri_per_orientarsi():
    """I tiri più dolci del manuale sono quelli di fisica.soluzione, coi valori in percentuale e in gradi e metri al secondo; il tiro coi
    tre valori a metà, da dove parte ogni mira, entra nella sfera vicina ed esce dalla lontana; e i voli durano quanto dice il
    capitolo dello sparo."""
    manuale = _leggi("manuale.html")
    sezione = _sezione(manuale, "orientarsi")
    meta = _tiro(PARTENZA_DELLA_MIRA, PARTENZA_DELLA_MIRA, PARTENZA_DELLA_MIRA)
    tempi = np.linspace(0.0, meta.tempo_a_terra(), 20001)
    posizioni = meta.posizione(tempi)
    dentro = np.flatnonzero(fisica.Zona(0.0).contiene(posizioni))
    entrata, uscita = posizioni[dentro[0]], posizioni[dentro[-1]]
    assert abs(np.linalg.norm(entrata) - CUPOLA_RAGGIO_INTERNO) < 0.1 and abs(np.linalg.norm(uscita) - CUPOLA_RAGGIO) < 0.1
    frasi = [f"con {testi.gradi(meta.alzo)} di alzo e {testi.numero(meta.velocita)} metri al secondo, sale fino a {testi.metri(meta.apice()[1])} e cade a "
             f"{testi.metri(fisica.distanza_orizzontale(meta.punto_a_terra()))}; entra nella cupola sulla sfera vicina a {testi.metri(entrata[1])} "
             f"d'altezza, e ne esce dalla sfera lontana a {testi.metri(uscita[1])}"]
    punti = {
        "mezzo": (f"Il mezzo della cupola, dritto davanti a {testi.intero(ascolti.DISTANZA_DEL_MEZZO)} metri da te e alto {testi.numero(MEZZO[1])}: ", MEZZO),
        "basso vicino": ("In basso e vicino, a 35 metri e alto 5: ", _punto(35.0, 5.0)),
        "basso lontano": ("In basso e lontano, a 75 metri e alto 5: ", _punto(75.0, 5.0)),
        "alto vicino": ("In alto e vicino, a 35 metri e alto 25: ", _punto(35.0, 25.0)),
        "alto lontano": ("In alto e lontano, a 75 metri e alto 60: ", _punto(75.0, 60.0)),
    }
    soluzioni = {}
    for nome, (inizio, punto) in punti.items():
        valori = fisica.soluzione(punto)
        assert valori is not None, f"il punto {nome} del manuale non ha un tiro dentro i margini"
        soluzioni[nome] = valori
        _azimut, alzo, velocita = fisica.parametri_del_tiro(*valori)
        percentuali = [testi.percento(valore) for valore in valori]
        if nome == "mezzo":
            frasi.append(f"{inizio}direzione {_al(percentuali[0])}, alzo {_al(percentuali[1])}, cioè {testi.gradi(alzo)}, potenza {_al(percentuali[2])}, cioè "
                         f"{testi.numero(velocita)} metri al secondo")
        else:
            frasi.append(f"{inizio}alzo {_al(percentuali[1])}, {testi.gradi(alzo)}, potenza {_al(percentuali[2])}, {testi.numero(velocita)} metri al secondo")
    a_sinistra = fisica.soluzione(fisica.punto_della_cupola(ascolti.DISTANZA_DEL_MEZZO, -20.0, ascolti.ELEVAZIONE_DEL_MEZZO))
    assert a_sinistra is not None and np.allclose(a_sinistra[1:], soluzioni["mezzo"][1:])
    frasi.append(f"spostato di 20 gradi a sinistra: direzione {_al(testi.percento(a_sinistra[0]))}, alzo {_al(testi.percento(a_sinistra[1]))} e potenza "
                 f"{_al(testi.percento(a_sinistra[2]))}, come nel mezzo")
    mancanti = [frase for frase in frasi if frase not in sezione]
    assert not mancanti, f"tiri per orientarsi che il manuale non dice come fisica.soluzione: {mancanti}"
    durate = [_tiro(*soluzioni[nome]).tempo_a_terra() for nome in ("basso vicino", "mezzo")]
    piu_alto = fisica.Traiettoria(0.0, ALZO_MASSIMO, VELOCITA_MASSIMA).tempo_a_terra()
    frase = (f"{testi.numero(durate[0])} secondi per il tiro più dolce verso un bersaglio basso e vicino, {testi.numero(durate[1])} per quello verso il mezzo "
             f"della cupola, e fino a {testi.numero(piu_alto)} per il tiro più alto")
    assert frase in _sezione(manuale, "sparo-volo"), f"il capitolo dello sparo non dice «{frase}»"


def test_il_suono_dello_spazio():
    """D28: la panoramica ai lati del settore, la nota fino al tetto sopra l'apice del tiro più alto, il volume fra le due sfere e la
    grandezza nell'oscillazione, coi numeri della fisica e delle costanti; il blocco della 2.0.0 dice la nota."""
    manuale = _leggi("manuale.html")
    assert AZIMUT_DEL_PAN == SETTORE_AZIMUT, "il manuale dice la panoramica tutta da un lato sui lati della cupola"
    lato = SETTORE_AZIMUT - math.degrees(math.asin(RAGGIO_INIZIALE / math.hypot(MEZZO[0], MEZZO[2])))
    panoramica = _sezione(manuale, "panoramica")
    for frase in (f"tutto a sinistra a {testi.intero(AZIMUT_DEL_PAN)} gradi a sinistra", f"I lati della cupola stanno a {testi.intero(SETTORE_AZIMUT)} gradi a qualunque distanza",
                  f"al primo livello, nel mezzo della cupola, fino a {testi.gradi(round(lato))}"):
        assert frase in panoramica, f"il capitolo della panoramica non dice «{frase}»"
    tutta = fisica.Zona(0.0)
    cima = tutta.esterno * math.sin(math.radians(tutta.elevazione))
    apice = fisica.Traiettoria(0.0, ALZO_MASSIMO, VELOCITA_MASSIMA).apice()[1]
    assert apice < ALTEZZA_TETTO_DEL_SUONO, "il tiro più alto deve restare sotto il tetto della nota"
    nota = _sezione(manuale, "nota")
    frasi = [
        f"al suolo un La a {testi.intero(FREQUENZA_AL_SUOLO)} hertz, e un'ottava più su ogni {testi.numero(METRI_PER_OTTAVA, 2)} metri, "
        f"{A_PAROLE[OTTAVE_DEL_SUONO]} ottave fino a {testi.metri(ALTEZZA_TETTO_DEL_SUONO)} d'altezza, dove la nota arriva a "
        f"{testi.intero(fisica.frequenza_di(ALTEZZA_TETTO_DEL_SUONO))} hertz",
        f"Un bersaglio nel mezzo della cupola, a {testi.metri(MEZZO[1])}, canta attorno ai {testi.intero(fisica.frequenza_di(MEZZO[1]))} hertz; uno in cima, a "
        f"{testi.metri(cima)}, attorno ai {testi.intero(fisica.frequenza_di(cima))}",
        f"con {testi.intero(ALZO_MASSIMO)} gradi di alzo e tutta la potenza sale fino a {testi.metri(apice)}",
    ]
    mancanti = [frase for frase in frasi if frase not in nota]
    assert not mancanti, f"frasi della nota che il manuale non dice come il codice: {mancanti}"
    gittata = max(fisica.distanza_orizzontale(fisica.Traiettoria(0.0, alzo, VELOCITA_MASSIMA).punto_a_terra()) for alzo in np.linspace(ALZO_MINIMO, ALZO_MASSIMO, 701))
    pavimento = (GUADAGNO_DB_VICINO - GUADAGNO_DB_MINIMO) / DB_PER_METRO
    assert gittata > pavimento, "il manuale dice che soltanto il proiettile va oltre il pavimento del volume"
    volume = _sezione(manuale, "volume-distanza")
    frasi = [
        f"ogni {testi.intero(1 / DB_PER_METRO)} metri in più il suono scende di un decibel",
        f"sulla sfera vicina, a {testi.intero(CUPOLA_RAGGIO_INTERNO)} metri, suona circa {testi.numero(DB_PER_METRO * (CUPOLA_RAGGIO - CUPOLA_RAGGIO_INTERNO))} "
        f"decibel più forte di uno sulla sfera lontana, a {testi.intero(CUPOLA_RAGGIO)}",
        f"oltre i {testi.intero(pavimento)} metri, dove arriva soltanto il proiettile nei tiri più lunghi, fino a {testi.intero(gittata)} metri",
    ]
    mancanti = [frase for frase in frasi if frase not in volume]
    assert not mancanti, f"frasi del volume che il manuale non dice come il codice: {mancanti}"
    assert RAGGIO_MINIMO == 0.5, "il manuale dice il raggio più piccolo a parole, mezzo metro"
    grandezza = _sezione(manuale, "grandezza")
    assert (f"Al primo livello il raggio è di {testi.intero(RAGGIO_INIZIALE)} metri, e la nota si allarga di circa "
            f"{testi.numero(12 * mezza_escursione_ottave(RAGGIO_INIZIALE), 0)} semitoni sopra e sotto il centro; al raggio più piccolo, mezzo metro, "
            f"l'oscillazione è di circa {testi.numero(12 * mezza_escursione_ottave(RAGGIO_MINIMO), 1)} semitoni") in grandezza
    blocco = _blocco_della_versione(_leggi("CHANGELOG.md"), "2.0.0")
    assert (f"sale di un'ottava ogni {testi.numero(METRI_PER_OTTAVA, 2)} metri, {A_PAROLE[OTTAVE_DEL_SUONO]} ottave fino a {testi.intero(ALTEZZA_TETTO_DEL_SUONO)} "
            f"metri, appena sopra l'apice del tiro più alto, che arriva a {testi.numero(apice)}") in blocco


def test_la_mira_manuale_nei_documenti():
    """D29: la mira che parte dal centro, i tasti, la velocità dei valori, le loro voci, le tacche coi loro numeri, i capi e la
    copertura dell'alzo e della potenza, il controllo Mira, la barra di stato di cinque righe con la riga dei valori di resoconti.py;
    il README e il blocco della 2.0.0."""
    from gui.finestra import RIGHE_DELLA_BARRA

    manuale, readme = _leggi("manuale.html"), _leggi("README.md")
    blocco = _blocco_della_versione(_leggi("CHANGELOG.md"), "2.0.0")
    assert f"partono tutti dal centro della loro scala, al {testi.percento(PARTENZA_DELLA_MIRA)}" in _sezione(manuale, "avviare-la-mira")
    tasti = _sezione(manuale, "tasti-mira")
    decimo, mezza_scala = round(DURATA_DELLA_SCALA / TACCHE, 6), round((1.0 - PARTENZA_DELLA_MIRA) / VELOCITA_DELLA_MIRA, 6)
    assert decimo in DECIMI_A_PAROLE and mezza_scala in MEZZE_SCALE_A_PAROLE, "un tempo nuovo della scala va detto nel manuale, e qui a parole"
    frasi = [
        "<li><kbd>Freccia sinistra</kbd> e <kbd>Freccia destra</kbd> muovono la direzione",
        "<li><kbd>Freccia su</kbd> e <kbd>Freccia giù</kbd> alzano e abbassano l'alzo",
        "<li><kbd>Fine</kbd> alza la potenza, verso il massimo, e <kbd>Home</kbd> la abbassa, verso il minimo",
        "<li><kbd>Invio</kbd> fa partire il colpo coi tre valori di quel momento",
        "<li><kbd>Esc</kbd> annulla la mira, e il proiettile resta nella riserva",
        f"in {testi.intero(DURATA_DELLA_SCALA)} secondi, cioè {DECIMI_A_PAROLE[decimo]}",
        MEZZE_SCALE_A_PAROLE[mezza_scala],
    ]
    mancanti = [frase for frase in frasi if frase not in tasti]
    assert not mancanti, f"frasi dei tasti della mira che il manuale non dice: {mancanti}"
    # Il puntatore di D31: una voce sola in onda triangolare, che porta la direzione nel punto, l'alzo nella nota e la potenza nel volume.
    assert PUNTATORE_FORMA == "triangolo", "il manuale dice il puntatore in onda triangolare"
    voci = _sezione(manuale, "suoni-mira")
    frasi = [
        "una voce sola, in onda triangolare, che porta sempre tutti e tre i valori",
        "La direzione è il punto da cui arriva il puntatore",
        f"da {testi.intero(ALZO_FREQUENZE[0])} a {testi.intero(ALZO_FREQUENZE[1])} hertz, {A_PAROLE[round(math.log2(ALZO_FREQUENZE[1] / ALZO_FREQUENZE[0]))]} ottave",
        f"La potenza è il volume, {testi.intero(-POTENZA_DB_MINIMO)} decibel dal minimo al massimo",
    ]
    mancanti = [frase for frase in frasi if frase not in voci]
    assert not mancanti, f"frasi delle voci della mira che il manuale non dice: {mancanti}"
    # Le tacche tengono la forma del loro valore, e arrivano dal punto del puntatore, come il suono del capo (D31).
    assert TACCA_FORMA == {"direzione": "seno", "alzo": "seno", "potenza": "triangolo"}, "il manuale dice le forme delle tacche"
    assert "in seno per la direzione e per l'alzo e in onda triangolare per la potenza, e arrivano dal punto del puntatore" in _sezione(manuale, "capi-scala")
    assert "un doppio colpo sordo e grave, anche lui dal punto del puntatore" in _sezione(manuale, "capi-scala")
    # Le nove tacche dei decimi coprono tutta la scala, da TACCA_FREQUENZE[0] a TACCA_FREQUENZE[1], un'ottava (dalle prove del 22 novembre 2026).
    frequenze = frequenze_delle_tacche(TACCHE - 1)
    assert math.isclose(frequenze[-1], TACCA_FREQUENZE[1]) and TACCA_FREQUENZE[1] == 2 * TACCA_FREQUENZE[0], "il manuale dice un'ottava sulle nove tacche"
    assert 5 / TACCHE == PARTENZA_DELLA_MIRA, "il manuale dice che la quinta tacca è la metà della scala, da dove parte la mira"
    # Con la mira manuale ogni valore arriva al capo e lì si ferma: il margine di copertura è zero, e i capi si scelgono come gli altri valori.
    assert MARGINE_DI_COPERTURA == 0.0, "il manuale dice che ogni punto della cupola si raggiunge coi valori dentro le scale, capi compresi"
    capi = _sezione(manuale, "capi-scala")
    frasi = [
        f"un bip piccolissimo di {testi.intero(TACCA_DURATA * 1000)} millesimi",
        f"Le tacche sono {A_PAROLE[TACCHE - 1]}, dal {ORDINALI[1]} al {ORDINALI[TACCHE - 1]} decimo",
        f"da {testi.intero(frequenze[0])} a {testi.intero(frequenze[-1])} hertz, un'ottava sulle {A_PAROLE[TACCHE - 1]} tacche",
        "La quinta tacca è la metà della scala",
        f"si raggiunge coi tre valori dentro le loro scale, dallo {testi.percento(MARGINE_DI_COPERTURA)} al {testi.percento(1 - MARGINE_DI_COPERTURA)}, "
        "capi compresi",
    ]
    mancanti = [frase for frase in frasi if frase not in capi]
    assert not mancanti, f"frasi delle tacche e dei capi che il manuale non dice: {mancanti}"
    assert "2%" not in capi and "98%" not in capi
    # La copertura: ogni punto dove può stare il centro del bersaglio più piccolo, il bonus coi bersagli di mezzo metro, si raggiunge con
    # l'alzo e la potenza dentro le scale; la direzione non conta qui, perché copre il settore esattamente, e il punto si porta davanti.
    rng = random.Random(28)
    zona = fisica.Zona(RAGGIO_MINIMO * RAGGIO_BONUS_FATTORE)
    davanti = [(0.0, float(punto[1]), float(math.hypot(punto[0], punto[2]))) for punto in (zona.punto_a_caso(rng) for _punto in range(200))]
    scoperti = [punto for punto in davanti if fisica.soluzione(punto) is None]
    assert not scoperti, f"punti della cupola che alzo e potenza dentro i margini non raggiungono: {scoperti[:3]}"
    assert "NVDA dice il suo nome, Mira" in _sezione(manuale, "fuoco-mira")
    # Dalle prove del 22 novembre 2026: i tasti dei valori muovono la mira soltanto col fuoco sulla Mira, dove F6 lo riporta; nella vista
    # e nella barra leggono, e Invio ed Esc valgono anche da lì. Il valore non va più a scatti, perché da lì non si muove.
    fuoco = _sezione(manuale, "fuoco-mira")
    frasi = [
        "fanno quello che fanno sempre nelle aree di testo, muovono il cursore e leggono, e non muovono i valori",
        "<kbd>Invio</kbd> spara ed <kbd>Esc</kbd> annulla la mira anche dalla vista e dalla barra",
        "<kbd>F6</kbd> riporta il cursore sulla Mira, con il suo suono",
        "da lì i tasti dei valori muovono di nuovo la mira",
        "Fuori dalla mira non c'è niente da puntare, e <kbd>F6</kbd> fa sentire il non ora",
    ]
    mancanti = [frase for frase in frasi if frase not in fuoco]
    assert not mancanti and "a scatti" not in fuoco, f"frasi del fuoco sulla Mira che il manuale non dice: {mancanti}"
    assert "I tasti dei valori muovono la mira soltanto col cursore sulla Mira" in tasti
    barra = _sezione(manuale, "barra")
    alzo_medio = ALZO_MINIMO + PARTENZA_DELLA_MIRA * (ALZO_MASSIMO - ALZO_MINIMO)
    quarto = PALESTRA_FINO_AL_LIVELLO + 1
    frasi = [
        f"se ne vedono {A_PAROLE[RIGHE_DELLA_BARRA]} senza scorrere",
        "I gradi hanno sempre un decimale, anche quando sono interi",
        "<kbd>F6</kbd> riporta il cursore sulla Mira",
        f"all'inizio di una mira al {ORDINALI[quarto]} livello: {resoconti.riga_dei_valori_della_mira(0.0, alzo_medio, PARTENZA_DELLA_MIRA, tempo_della_mira(quarto))}",
        resoconti.riga_dei_valori_della_mira(-12.5, 45.0, 0.63, 23.4),
        f"L'ultima riga, nella mira, dice {resoconti.riga_della_fase('mira')}.",
    ]
    mancanti = [frase for frase in frasi if frase not in barra]
    assert not mancanti, f"frasi della barra di stato che il manuale non dice come resoconti.py: {mancanti}"
    assert f"alta {A_PAROLE[RIGHE_DELLA_BARRA]} righe" in _sezione(manuale, "finestra")
    assert len(resoconti.righe_della_barra("mira", livello=quarto, mira=(0.0, alzo_medio, PARTENZA_DELLA_MIRA, 30.0))) == RIGHE_DELLA_BARRA
    for frase in ("le frecce sinistra e destra la direzione, le frecce su e giù l'alzo, Home e Fine la potenza", "Invio spara", "controllo Mira",
                  "F6", "in gradi, sempre con un decimale, e in percentuale", "capi compresi"):
        assert frase in readme and frase in blocco, f"il README o il blocco della 2.0.0 non dicono «{frase}»"
    assert "Col cursore sulla Mira, ogni valore si muove" in readme and "I tasti dei valori muovono la mira soltanto col cursore sulla Mira" in blocco
    assert f"da {testi.intero(frequenze[0])} a {testi.intero(frequenze[-1])} hertz" in blocco
    assert f"da un capo all'altro della scala in {testi.intero(DURATA_DELLA_SCALA)} secondi" in blocco
    assert f"la barra di stato, che adesso ha {A_PAROLE[RIGHE_DELLA_BARRA]} righe" in blocco


def test_il_tempo_della_mira_nei_documenti():
    """D29: nessun tempo nella palestra, poi TEMPO_DELLA_MIRA_INIZIALE, un secondo in meno a livello, fino al minimo, coi tick dalla
    prima alla seconda frequenza; l'elenco dei livelli dice il tempo e i punti di ognuno, coi numeri di partita.py; il README e il
    blocco della 2.0.0."""
    manuale, readme = _leggi("manuale.html"), _leggi("README.md")
    blocco = _blocco_della_versione(_leggi("CHANGELOG.md"), "2.0.0")
    quarto = PALESTRA_FINO_AL_LIVELLO + 1
    assert all(tempo_della_mira(livello) is None for livello in range(1, quarto))
    assert RIDUZIONE_DEL_TEMPO_DELLA_MIRA == 1.0, "i documenti dicono «uno in meno a ogni livello»"
    dal_minimo = next(livello for livello in range(quarto, 200) if tempo_della_mira(livello) == TEMPO_DELLA_MIRA_MINIMO)
    iniziale, minimo = testi.intero(tempo_della_mira(quarto)), testi.intero(TEMPO_DELLA_MIRA_MINIMO)
    tempo = _sezione(manuale, "tempo-mira")
    frasi = [
        f"Nella palestra, i primi {A_PAROLE[PALESTRA_FINO_AL_LIVELLO]} livelli, la mira non ha tempo",
        f"{iniziale} secondi al {ORDINALI[quarto]} livello, uno in meno a ogni livello, fino a {minimo} secondi dal {ORDINALI[dal_minimo]} in poi",
        f"da {testi.intero(TICK_FREQUENZE[0])} hertz al primo secondo a {testi.intero(TICK_FREQUENZE[1])} all'ultimo tick",
        # Dalle prove del 22 novembre 2026: il tempo parte dopo gli avvisi e il resoconto, e fino a lì i secondi restano tutti.
        "Il tempo della mira parte dopo gli avvisi e il resoconto",
        "il tempo e i suoi tick aspettano che finiscano, così non li coprono, e fino a lì i secondi restano tutti",
    ]
    mancanti = [frase for frase in frasi if frase not in tempo]
    assert not mancanti, f"frasi del tempo della mira che il manuale non dice: {mancanti}"
    voci = _voci(re.search(r"<ul>.*?</ul>", _sezione(manuale, "livelli"), re.DOTALL).group(0))
    ultimo = 18
    assert len(voci) == ultimo + 1, "l'elenco dei livelli va dal primo al diciottesimo, poi una voce per gli altri"
    for livello, voce in enumerate(voci[:ultimo], 1):
        assert voce.startswith(f"Livello {livello}: ")
        normale = SimpleNamespace(bonus=False, raggio=raggio_del_livello(livello), velocita_scalare=velocita_del_livello(livello))
        attese = [f"da {testi.intero(punti_del_centro(normale, 0.0))} a {testi.intero(punti_del_centro(normale, 1.0))} punti"]
        attese.append("mira senza tempo" if tempo_della_mira(livello) is None else f"mira di {testi.intero(tempo_della_mira(livello))} secondi")
        if livello >= LIVELLO_DEI_BONUS:
            attese.append(f"in {_secondi(durata_del_bonus(livello))} secondi")
        if quanti_bersagli(livello) > 1:
            attese.append(f"{A_PAROLE[quanti_bersagli(livello)]} bersagli")
        mancanti = [attesa for attesa in attese if attesa not in voce]
        assert not mancanti, f"il livello {livello} del manuale non dice {mancanti}"
    assert (f"da {testi.intero(tempo_della_mira(ultimo + 1))} secondi al {ORDINALI[ultimo + 1]} fino ai {minimo} secondi del {ORDINALI[dal_minimo]}") in voci[ultimo]
    assert f"dal {ORDINALI[quarto]} ne ha {iniziale} secondi, uno in meno a ogni livello, fino a {minimo} dal {ORDINALI[dal_minimo]}" in readme
    assert f"dal {ORDINALI[quarto]} livello ne ha {iniziale} secondi, uno in meno a ogni livello, fino a {minimo} dal {ORDINALI[dal_minimo]} in poi" in blocco
    assert "Il tempo parte dopo gli avvisi e il resoconto del tiro di prima" in readme
    assert "Il tempo della mira parte dopo gli avvisi e il resoconto del tiro di prima" in blocco


def test_la_riserva_nei_documenti():
    """D30: 20 colpi all'inizio, 15 a ogni livello chiuso, 8 per un centro sul bonus, nel manuale, a cifre e a parole, nel README e nel
    blocco della 2.0.0; il messaggio di Esc nella mira, con la riserva iniziale, è quello della vista."""
    manuale, readme = _leggi("manuale.html"), _leggi("README.md")
    blocco = _blocco_della_versione(_leggi("CHANGELOG.md"), "2.0.0")
    iniziali, per_livello, bonus = (testi.intero(n) for n in (COLPI_INIZIALI, COLPI_PER_LIVELLO, COLPI_BONUS))
    parole = {n: A_PAROLE[n] for n in (COLPI_INIZIALI, COLPI_PER_LIVELLO, COLPI_BONUS)}
    frasi = [
        (_sezione(manuale, "riserva"), f"Si parte con {iniziali} colpi"),
        (_sezione(manuale, "riserva"), f"aggiungono {per_livello} colpi"),
        (_sezione(manuale, "riserva"), f"ne aggiunge altri {bonus}"),
        (_sezione(manuale, "che-cos-e"), f"se ne parte con {parole[COLPI_INIZIALI]}"),
        (_sezione(manuale, "che-cos-e"), f"regalano altri {parole[COLPI_PER_LIVELLO]} colpi"),
        (_sezione(manuale, "che-cos-e"), f"regala {parole[COLPI_BONUS]} colpi"),
        (_sezione(manuale, "cominciare"), f"con {iniziali} colpi nella riserva"),
        (_sezione(manuale, "menu-partita"), f"con il livello uno e {parole[COLPI_INIZIALI]} colpi"),
        (_sezione(manuale, "bonus"), f"aggiunge {parole[COLPI_BONUS]} colpi alla riserva"),
        (_sezione(manuale, "che-cosa-dice"), f"per il bonus anche gli {parole[COLPI_BONUS]} colpi in più"),
        (_sezione(manuale, "esc-mira"), resoconti.mira_annullata(COLPI_INIZIALI).testo),
        (readme, f"La riserva parte da {parole[COLPI_INIZIALI]} colpi"),
        (readme, f"danno altri {parole[COLPI_PER_LIVELLO]} colpi, un centro sul bonus {parole[COLPI_BONUS]}"),
        (blocco, f"si parte con {iniziali} colpi, ogni livello chiuso ne dà {per_livello} e un centro sul bonus {bonus}"),
        (blocco, f"il primato della precisione globale media vale ancora per le partite di almeno {A_PAROLE[TIRI_PER_IL_PRIMATO_DELLA_MEDIA]} tiri"),
    ]
    mancanti = [frase for testo, frase in frasi if frase not in testo and f"{frase[0].upper()}{frase[1:]}" not in testo]
    assert not mancanti, f"frasi della riserva che i documenti non dicono coi numeri del codice: {mancanti}"


def test_la_precisione_globale_della_cupola():
    """D26 con la cupola di D28: la retta dal centro del bersaglio al punto della cupola più lontano, lunga quanto dice
    fisica.distanza_dal_punto_piu_lontano nel mezzo della cupola, nel punto dove è più corta e nell'angolo in fondo; gli esempi
    calcolati da volo.Volo; il centro sempre sopra il minimo, col bersaglio nel punto più sfavorevole; il README e il blocco della
    2.0.0."""
    manuale, readme = _leggi("manuale.html"), _leggi("README.md")
    blocco = _blocco_della_versione(_leggi("CHANGELOG.md"), "2.0.0")
    nel_mezzo, nell_angolo = fisica.distanza_dal_punto_piu_lontano(MEZZO), fisica.distanza_dal_punto_piu_lontano(ANGOLO_IN_FONDO)
    piu_corta, dove = _lontananza_minima()
    # Il punto più favorevole sta poco più vicino e poco più in alto del mezzo, e dentro la zona dei bersagli grandi e piccoli.
    assert np.linalg.norm(dove) < np.linalg.norm(MEZZO) and dove[1] > MEZZO[1]
    assert fisica.Zona(RAGGIO_INIZIALE).contiene(dove) and fisica.Zona(RAGGIO_MINIMO).contiene(dove)
    # Il punto più lontano dall'angolo in fondo a sinistra sta in cima al lato destro.
    spigoli = fisica.punti_degli_spigoli()
    lontano = spigoli[int(np.argmax(np.linalg.norm(spigoli - ANGOLO_IN_FONDO, axis=1)))]
    assert lontano[0] > 0 and math.isclose(lontano[1], float(np.max(spigoli[:, 1])))
    elevazione = math.degrees(math.atan2(MEZZO[1], math.hypot(MEZZO[0], MEZZO[2])))

    def coppia(distanza):
        return _con_articolo(_globale(MEZZO, distanza)), _con_articolo(_globale(ANGOLO_IN_FONDO, distanza))

    cinque, dieci, venti, lontanissimo = (coppia(distanza) for distanza in (5.0, 10.0, 20.0, 64.0))
    assert lontanissimo[0] == "lo 0%" and nel_mezzo < 64.0

    def centro(raggio):
        return _con_articolo(testi.percento(1 - (raggio + RAGGIO_PROIETTILE) / piu_corta, 1))

    sezione = _sezione(manuale, "precisione-globale")
    frasi = [
        f"La retta è lunga {testi.numero(nel_mezzo)} metri per un bersaglio nel mezzo della cupola, a {testi.intero(np.linalg.norm(MEZZO))} metri da te e a "
        f"{testi.intero(elevazione)} gradi d'elevazione, e {testi.numero(piu_corta)} metri nel punto dove è più corta, poco più vicino e poco più in alto del "
        f"mezzo; arriva a {testi.numero(nell_angolo)} metri per un bersaglio in un angolo in fondo alla cupola, a terra contro il lato sinistro, il cui punto "
        "più lontano è in cima al lato destro",
        f"un tiro passato a 5 metri dal centro del bersaglio vale {cinque[0]} con un bersaglio nel mezzo della cupola, e {cinque[1]} con uno nell'angolo in fondo",
        f"a 10 metri {dieci[0]} e {dieci[1]}; a 20 metri {venti[0]} e {venti[1]}; a 64 metri {lontanissimo[0]}, perché passa più lontano del punto più "
        f"lontano, e {lontanissimo[1]}",
        f"con bersagli di {testi.intero(RAGGIO_INIZIALE)} metri, sopra {centro(RAGGIO_INIZIALE)}, e coi bersagli di mezzo metro sopra {centro(RAGGIO_MINIMO)}",
        "per il bonus anche quando è appena fuori da un lato",
    ]
    mancanti = [frase for frase in frasi if frase not in sezione]
    assert not mancanti, f"frasi della precisione globale che il manuale non dice come il codice: {mancanti}"
    for nome, testo in (("README.md", readme), ("CHANGELOG.md", blocco)):
        assert (f"lunga {testi.numero(nel_mezzo)} metri per un bersaglio nel mezzo della cupola e {testi.numero(nell_angolo)} per uno in un angolo in fondo"
                in testo), nome
        assert f"vale {dieci[0]} nel primo caso e {dieci[1]} nel secondo" in testo, nome
    assert f"uno passato a 64 metri lo 0% e {lontanissimo[1]}" in readme


def test_le_statistiche_della_mira_nel_manuale():
    """D29: le voci nuove della pagina delle statistiche, con le parole di statistiche.py, e quelle delle scivolate di prima della
    2.0.0, che restano con la loro etichetta."""
    sorgente = _leggi("statistiche.py")
    manuale = _sezione(_leggi("manuale.html"), "statistiche").lower()
    voci = ("Mire finite col colpo", "Mire scadute, col colpo partito da solo allo scadere del tempo", "Tempo medio di una mira", "Tasti premuti nella mira",
            "scelti con la mira manuale", "lasciati a un capo della scala", "Barre della mira ripartite da capo senza lo spazio, prima della 2.0.0",
            "Proiettili perduti, con una barra mai fermata dallo spazio, prima della 2.0.0")
    for voce in voci:
        assert voce in sorgente, f"la voce «{voce}» non sta più in statistiche.py: rileggi il manuale"
        parte = voce.split(", prima della")[0].lower()
        assert parte in manuale, f"il manuale non dice la voce delle statistiche «{voce}»"


def test_il_cubo_e_le_scivolate_restano_nella_storia():
    """Dalla 2.0.0 il cubo, le scivolate e il proiettile perduto non ci sono più: nel manuale e nel README una frase che li nomina dice
    anche che erano di prima."""
    for nome, testo in (("manuale.html", " ".join(_manuale().testo)), ("README.md", _leggi("README.md"))):
        frasi = re.split(r"(?<=[.;:])\s+", re.sub(r"\s+", " ", testo))
        vecchie = [frase for frase in frasi if any(parola in frase.lower() for parola in PAROLE_DI_PRIMA) and not any(segno in frase for segno in SEGNI_DI_PRIMA)]
        assert not vecchie, f"{nome} dice ancora come di oggi: {vecchie}"


def test_niente_separatori_grafici_nei_documenti():
    testo_del_manuale = "".join(_manuale().testo)
    for nome, testo in (("manuale.html", testo_del_manuale), ("README.md", _leggi("README.md")), ("CHANGELOG.md", _leggi("CHANGELOG.md"))):
        trovato = SEPARATORE.search(testo)
        assert trovato is None, f"{nome}: {trovato.group(0) if trovato else ''}"
