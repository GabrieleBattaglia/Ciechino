"""
Le prove della finestra principale di Ciechino e del suo avvio, sul desktop nascosto del conftest.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di tests/test_finestra.py di MESS. La finestra nasce con
una partita finta, che ha l'interfaccia del contratto e annota ciò che le si chiede; il motore è muto,
come vuole il conftest, o quello del flusso finto quando la prova vuole la scheda; i preset non si
sintetizzano, e ogni suono si annota. I dialoghi modali e i messaggi di wx sono sostituiti, perché una
prova che aspetta un tasto non finirebbe mai; i tasti sono l'evento wxEVT_CHAR_HOOK che il gancio di wx
crea per un tasto vero, rifatto dalla prova: SendInput li manderebbe allo schermo vero. Nessuna prova
parla: la voce finta del conftest raccoglie gli annunci. Classifica, statistiche e impostazioni si
scrivono nella cartella temporanea del conftest.
Dalla 2.0.0 la mira manuale di D29: la partita finta conosce muovi, lascia, spara e tasti_giu; le prove
guardano i tasti della mira e l'Invio che vanno alla partita e non all'area di testo, il rilascio che
arriva al controllo col fuoco, i tasti lasciati che la tastiera dice su, con wx.GetKeyState sostituito,
il fuoco che va sul controllo Mira e torna alla vista, la barra di cinque righe e la guida coi tasti nuovi.
"""

import itertools
import json
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pytest
import wx
from aiuti_audio import ALTOPARLANTI, ASIO, CUFFIE, CUFFIE_FINTE, DIGITALE, LATENZA_FINTA, FlussoFinto, apri_finto, motore_finto

import aggiornamento
import classifica
import costanti
import impostazioni
import partita as modulo_partita
import percorsi
import resoconti
import schede
import suoni
from gui import dialoghi
from gui import finestra as modulo_finestra
from gui.finestra import FinestraPrincipale, guadagno_del_volume
from motore_audio import Motore, VoceTono

# Il buffer che prende il posto di ogni preset: corto, e non vuoto, così la voce nasce e il suono si annota.
CAMPIONI_FINTI = 64


@dataclass
class Messaggio:
    """Un messaggio della partita, con i campi del contratto."""

    tipo: str
    testo: str
    annuncio: str = ""
    dati: dict = field(default_factory=dict)


class PartitaFinta:
    """Una partita con l'interfaccia del contratto: annota le chiamate e restituisce i messaggi preparati dalla prova."""

    def __init__(self, motore, suoni_, statistiche):
        self.motore = motore
        self.suoni = suoni_
        self.statistiche = statistiche
        self.fase = "pronta"
        self.in_pausa = False
        self.chiamate = []
        # I messaggi del prossimo battito e del prossimo spazio.
        self.in_arrivo = []
        self.allo_spazio = []
        self.barra = ["Pronta. Livello 1, 0 punti.", "Riserva: 10 colpi."]
        self.guasta = False
        # La pausa automatica (D14): se c'è, e le volte in cui la partita si è fermata o ripresa da sola.
        self.automatica = False
        self.automatiche = []
        # Le forme del proiettile ricevute dalla finestra (D17), in ordine; proiettile è quella della
        # partita in corso, o di quella che comincerà, come nella partita vera.
        self.proiettili = []
        self.proiettile = self._proiettile_scelto = costanti.PROIETTILE_FORMA
        # La mira manuale (D29): i tasti che la partita crede giù, come coppie (parametro, verso), i messaggi del
        # prossimo Invio, e le fasi in cui vanno la partita allo spazio e allo sparo, se la prova le vuole.
        self.giu = []
        self.allo_sparo = []
        self.fase_allo_spazio = None
        self.fase_allo_sparo = None

    def in_corso(self):
        return self.fase not in ("pronta", "finita")

    def scegli_proiettile(self, forma):
        """Come la partita vera: la forma vale subito senza partita in corso, dalla seguente con una partita in corso."""
        self.proiettili.append(forma)
        self._proiettile_scelto = forma
        if self.in_corso():
            return False
        self.proiettile = forma
        return True

    def pausa_automatica(self, attiva):
        """Come la partita vera: si ferma se è in corso e non già in pausa; riprende soltanto la pausa messa così."""
        if attiva and self.in_corso() and not self.in_pausa:
            self.in_pausa = self.automatica = True
            self.automatiche.append("ferma")
        elif not attiva and self.automatica:
            self.in_pausa = self.automatica = False
            self.automatiche.append("riprende")
        return []

    def nuova(self):
        self.chiamate.append("nuova")
        self.proiettile = self._proiettile_scelto
        self.fase = "ascolto"
        self.in_pausa = self.automatica = False
        return [Messaggio("info", "Comincia la partita, livello 1.", "Si comincia.")]

    def spazio(self):
        self.chiamate.append("spazio")
        if self.fase_allo_spazio is not None:
            self.fase = self.fase_allo_spazio
        messaggi, self.allo_spazio = self.allo_spazio, []
        return messaggi

    def muovi(self, parametro, verso):
        """Il tasto di un valore della mira scende (D29): si annota, e il tasto è giù."""
        self.chiamate.append(("muovi", parametro, verso))
        if (parametro, verso) not in self.giu:
            self.giu.append((parametro, verso))
        return []

    def lascia(self, parametro, verso):
        """Il tasto di un valore della mira sale (D29): si annota, e il tasto non è più giù."""
        self.chiamate.append(("lascia", parametro, verso))
        if (parametro, verso) in self.giu:
            self.giu.remove((parametro, verso))
        return []

    def spara(self):
        """Invio nella mira (D29): si annota, e restituisce i messaggi preparati dalla prova."""
        self.chiamate.append("spara")
        if self.fase_allo_sparo is not None:
            self.fase = self.fase_allo_sparo
        messaggi, self.allo_sparo = self.allo_sparo, []
        return messaggi

    def tasti_giu(self):
        """Come la partita vera: i tasti giù soltanto nella mira."""
        return list(self.giu) if self.fase == "mira" else []

    def battito(self):
        if self.guasta:
            raise ValueError("battito guasto")
        messaggi, self.in_arrivo = self.in_arrivo, []
        return messaggi

    def pausa(self):
        self.chiamate.append("pausa")
        self.in_pausa = not self.in_pausa
        return [Messaggio("info", "Partita in pausa." if self.in_pausa else "La partita riprende.")]

    def annulla_mira(self):
        """Esc nella mira (D25), come la partita vera: si torna all'ascolto, col messaggio."""
        self.chiamate.append("annulla_mira")
        self.fase = "ascolto"
        return [Messaggio("info", "Mira annullata con Esc.", "Mira annullata.")]

    def abbandona(self):
        self.chiamate.append("abbandona")
        if not self.in_corso():
            return []
        self.fase = "pronta"
        self.in_pausa = self.automatica = False
        return [Messaggio("info", "Partita abbandonata.", "Partita abbandonata.", self.riepilogo())]

    def righe_della_barra(self):
        return list(self.barra)

    def riepilogo(self):
        return {"punteggio": 0, "livello": 1, "tiri": 0, "centri": 0, "durata_s": 0.0}


class Sentiti(list):
    """Gli eventi sonori della finestra, in ordine; inizi annota anche il campione chiesto."""

    def __init__(self):
        super().__init__()
        self.inizi = []


class Messaggi(list):
    """I messaggi di wx, al posto di dialoghi.messaggio: si annotano; le domande rispondono risposta."""

    def __init__(self):
        super().__init__()
        self.risposta = wx.NO

    def __call__(self, genitore, testo, titolo, stile=0):
        self.append((testo, titolo, stile))
        return self.risposta if stile & wx.YES_NO else wx.OK


@pytest.fixture
def sentiti(monkeypatch):
    """Nessun preset si sintetizza, e ogni suono si annota: la voce nasce lo stesso sul motore, che è muto o finto."""
    registro = Sentiti()
    monkeypatch.setattr(suoni, "_rendi_preset", lambda nome: np.zeros((CAMPIONI_FINTI, 2), dtype=np.float32))
    originale = suoni.Suoni.suona

    def suona(self, evento, posizione=None, guadagno_db=0.0, inizio=None):
        registro.append(evento)
        registro.inizi.append(inizio)
        return originale(self, evento, posizione, guadagno_db, inizio)

    monkeypatch.setattr(suoni.Suoni, "suona", suona)
    return registro


@pytest.fixture
def messaggi(monkeypatch):
    registro = Messaggi()
    monkeypatch.setattr(dialoghi, "messaggio", registro)
    return registro


@pytest.fixture
def senza_modali(monkeypatch):
    """Un dialogo che nessuna prova ha preparato non si apre davvero: risponde Annulla."""
    monkeypatch.setattr(wx.Dialog, "ShowModal", lambda self: wx.ID_CANCEL)


def _crea(**opzioni):
    finestra = FinestraPrincipale(fabbrica_della_partita=PartitaFinta, **opzioni)
    finestra.Show()
    return finestra


def _chiudi(finestra):
    motore = finestra.motore if finestra else None
    if finestra and not finestra.chiusa:
        finestra.Close(force=True)
    if motore is not None:
        motore.chiudi()
    wx.Yield()


@pytest.fixture
def finestra(app_wx, sentiti, messaggi, senza_modali, monkeypatch):
    f = _crea()
    # Sul desktop nascosto nessuna finestra ha il primo piano: la finestra si dice attiva, come quando si gioca.
    monkeypatch.setattr(f, "IsActive", lambda: True)
    yield f
    _chiudi(f)


def _tasto(finestra, codice, controllo=False, maiuscolo=False):
    """Un tasto come lo vede wx: il gancio parte dalla vista, che ha il fuoco, e sale fino alla finestra."""
    evento = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
    evento.SetEventObject(finestra.vista)
    evento.SetKeyCode(codice)
    evento.SetControlDown(controllo)
    evento.SetShiftDown(maiuscolo)
    finestra.vista.GetEventHandler().ProcessEvent(evento)
    return evento


def _riga_del_cursore(controllo):
    _trovata, colonna, riga = controllo.PositionToXY(controllo.GetInsertionPoint())
    return controllo.GetLineText(riga), colonna


def _leggi(nome):
    with open(percorsi.percorso_dati(nome), encoding="utf-8") as file:
        return json.load(file)


# La finestra com'è all'apertura.

def test_la_finestra_nasce_coi_controlli_etichettati(finestra):
    assert finestra.GetTitle() == "Ciechino"
    figli = list(finestra.pannello.GetChildren())
    for controllo, nome in ((finestra.vista, "Partita"), (finestra.barra, "Barra di stato"), (finestra.controllo_mira, "Mira")):
        etichetta = figli[figli.index(controllo) - 1]
        assert isinstance(etichetta, wx.StaticText) and etichetta.GetLabel() == nome
        assert controllo.GetName() == nome
    for controllo in (finestra.vista, finestra.barra):
        stile = controllo.GetWindowStyleFlag()
        assert stile & wx.TE_MULTILINE and stile & wx.TE_READONLY and stile & wx.TE_RICH2
    assert finestra.barra.GetWindowStyleFlag() & wx.TE_DONTWRAP and not finestra.vista.GetWindowStyleFlag() & wx.TE_DONTWRAP
    sizer = finestra.pannello.GetSizer()
    assert sizer.GetItem(finestra.vista).GetProportion() == 1
    barra = sizer.GetItem(finestra.barra)
    assert barra.GetProportion() == 0 and barra.GetFlag() & wx.EXPAND
    # Alta quanto le sue righe nel suo carattere, non una misura fissa; da D29 le righe sono cinque, perché nella mira, col bonus in
    # volo, c'è anche quella dei tre valori.
    assert finestra.barra.GetMinSize().height >= finestra.barra.GetCharHeight() * modulo_finestra.RIGHE_DELLA_BARRA
    righe_della_mira = resoconti.righe_della_barra("mira", livello=4, bersagli=1, bonus=True, mira=(-12.5, 38.2, 0.63, 23.2))
    assert len(righe_della_mira) == modulo_finestra.RIGHE_DELLA_BARRA == 5
    # Il controllo Mira (D29) non è un testo, così NVDA lascia passare frecce, Home e Fine: un pannello che prende i tasti, che
    # accetta il fuoco ma non da Tab, sotto la barra, alto almeno una riga del suo carattere.
    mira = finestra.controllo_mira
    assert isinstance(mira, modulo_finestra.ControlloDellaMira) and not isinstance(mira, wx.TextCtrl)
    assert mira.GetLabel() == "Mira" and mira.GetWindowStyleFlag() & wx.WANTS_CHARS
    assert mira.AcceptsFocus() and not mira.AcceptsFocusFromKeyboard()
    assert figli.index(mira) > figli.index(finestra.barra) and sizer.GetItem(mira).GetProportion() == 0
    assert mira.GetMinSize().height >= mira.GetCharHeight()


def test_i_menu_coi_tasti_giusti(finestra):
    menu = finestra.GetMenuBar()
    assert [menu.GetMenuLabelText(i) for i in range(menu.GetMenuCount())] == ["Partita", "Impostazioni", "Aiuto"]
    tasti = {}
    for i in range(menu.GetMenuCount()):
        for voce in menu.GetMenu(i).GetMenuItems():
            if voce.IsSeparator():
                continue
            rapido = voce.GetAccel()
            tasti[voce.GetItemLabelText()] = rapido.ToString() if rapido is not None else None
    assert tasti == {
        "Nuova partita": "Ctrl+N", "Pausa": "Ctrl+P", "Classifica": "Ctrl+K", "Statistiche": "Ctrl+T", "Esci": "Ctrl+Q",
        "Volume...": None, "Scheda audio...": "Ctrl+Shift+X", "Proiettile...": None, "Leggi i resoconti con la sintesi": None,
        "Lingua...": None, "Guida ai comandi": "F1", "Manuale": "Shift+F1", "Novità": "F2", "Informazioni": "F3", "Ascolti...": None, "Offrimi un caffè...": None,
        "Entra al PUB, il gruppo di supporto...": None, "Chiedi aiuto al PUB...": None,
    }
    sintesi = finestra.voci_del_menu[finestra.cambia_sintesi]
    assert sintesi.IsCheckable() and sintesi.IsChecked() == impostazioni.PREDEFINITE["sintesi"]


def test_la_vista_comincia_con_le_informazioni_utili(finestra):
    righe = finestra.vista.GetValue().splitlines()
    assert righe[0].startswith(f"Ciechino, il tiro a segno a orecchio, versione {costanti.VERSIONE}, rilasciata ")
    assert righe[1].startswith("Autori: Gabriele Battaglia (IZ4APU) & ClaudIA")
    assert righe[2].startswith("Oggi è ")
    assert "Spazio" in righe[3] and "F1" in righe[3]
    assert righe[4] == classifica.righe([])[0]
    # Il conftest tiene il motore muto: senza scheda, il gioco è muto e lo dice.
    assert righe[5] == "Non trovo una scheda audio che si apra: Ciechino funziona lo stesso, ma senza suono."
    assert all(riga.strip() for riga in righe)
    assert finestra.vista.GetInsertionPoint() == 0
    assert finestra.barra.GetValue() == "Pronta. Livello 1, 0 punti.\nRiserva: 10 colpi."


def test_la_vista_col_record_la_scheda_e_il_crash(app_wx, sentiti, messaggi, senza_modali):
    voce = classifica.nuova_voce("Gabry", {"punteggio": 1234, "livello": 2, "tiri": 9, "centri": 5, "durata_s": 61.0})
    assert classifica.salva([voce])
    motore, _flusso = motore_finto()
    f = _crea(motore=motore, crash_precedente="C:\\Ciechino\\Ciechino - Crash precedente.log")
    try:
        testo = f.vista.GetValue()
        assert "Il record della classifica è di Gabry, con 1.234 punti, oggi." in testo
        # Il flusso finto ha interfaccia e latenza: 20 millesimi.
        assert "Il suono esce da finta, con una latenza di 20 millesimi di secondo." in testo
        assert "La volta scorsa Ciechino si è chiuso per un errore grave" in testo and "Crash precedente.log" in testo
        # All'avvio il suono d'apertura, poi la notizia del crash, staccata: il motore la mette dopo la fine dell'avvio.
        f.suoni_d_avvio()
        assert sentiti[-2:] == ["avvio", "crash_precedente"]
        assert sentiti.inizi[-1] >= CAMPIONI_FINTI + round(modulo_finestra.PAUSA_DOPO_L_AVVIO * motore.fs)
    finally:
        _chiudi(f)
    assert motore.chiuso


def test_un_file_rovinato_lo_dice_all_apertura(app_wx, sentiti, messaggi, senza_modali):
    with open(percorsi.percorso_dati(costanti.FILE_IMPOSTAZIONI), "w", encoding="utf-8") as file:
        file.write("{ rovinato")
    f = _crea()
    try:
        assert "non si legge" in f.vista.GetValue() and f.avvisi_dei_dati
        f.suoni_d_avvio()
        assert sentiti[-2:] == ["avvio", "dati_rovinati"]
    finally:
        _chiudi(f)


# Lo spazio, i tasti e il battito.

def test_lo_spazio_va_alla_partita_e_non_va_avanti(finestra):
    evento = _tasto(finestra, wx.WXK_SPACE)
    assert finestra.partita.chiamate == ["spazio"] and not evento.GetSkipped()
    # Gli altri tasti, e lo spazio coi modificatori, vanno avanti come sempre, verso i controlli e i tasti rapidi.
    assert _tasto(finestra, ord("A")).GetSkipped()
    assert _tasto(finestra, wx.WXK_SPACE, controllo=True).GetSkipped()
    assert _tasto(finestra, wx.WXK_SPACE, maiuscolo=True).GetSkipped()
    assert finestra.partita.chiamate == ["spazio"]


def test_lo_spazio_non_e_della_partita_con_un_dialogo_aperto_o_la_finestra_non_attiva(finestra, monkeypatch):
    finestra._modali = 1
    assert _tasto(finestra, wx.WXK_SPACE).GetSkipped()
    finestra._modali = 0
    monkeypatch.setattr(finestra, "IsActive", lambda: False)
    assert _tasto(finestra, wx.WXK_SPACE).GetSkipped()
    assert finestra.partita.chiamate == []


def test_lo_spazio_porta_i_messaggi_nella_vista(finestra, voce_finta):
    finestra.partita.allo_spazio = [Messaggio("info", "Mira: la direzione.", "Direzione.")]
    _tasto(finestra, wx.WXK_SPACE)
    assert finestra.vista.GetValue().endswith("\nMira: la direzione.")
    assert voce_finta == ["Direzione."]


def test_esc_nella_mira_la_annulla_e_non_va_avanti(finestra, monkeypatch, voce_finta):
    """D25: Esc, durante la mira, a finestra attiva e senza dialoghi, è della partita, che annulla la mira, e non va avanti; il
    messaggio va nella vista e alla sintesi. Tenuto giù conta una volta. Fuori dalla mira, con un dialogo aperto, coi
    modificatori o a finestra non attiva, Esc va avanti come sempre."""
    assert _tasto(finestra, wx.WXK_ESCAPE).GetSkipped()
    finestra.partita.fase = "ascolto"
    assert _tasto(finestra, wx.WXK_ESCAPE).GetSkipped() and finestra.partita.chiamate == []
    finestra.partita.fase = "mira"
    finestra._modali = 1
    assert _tasto(finestra, wx.WXK_ESCAPE).GetSkipped()
    finestra._modali = 0
    assert _tasto(finestra, wx.WXK_ESCAPE, controllo=True).GetSkipped() and finestra.partita.chiamate == []
    monkeypatch.setattr(finestra, "IsActive", lambda: False)
    assert _tasto(finestra, wx.WXK_ESCAPE).GetSkipped() and finestra.partita.chiamate == []
    monkeypatch.setattr(finestra, "IsActive", lambda: True)
    assert not _tasto(finestra, wx.WXK_ESCAPE).GetSkipped()
    assert finestra.partita.chiamate == ["annulla_mira"] and finestra.partita.fase == "ascolto"
    assert finestra.vista.GetValue().endswith("\nMira annullata con Esc.") and voce_finta == ["Mira annullata."]
    # Ripetuto dal tasto tenuto giù, nella mira, non conta, ma resta della partita.
    finestra.partita.fase = "mira"
    evento = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
    evento.SetEventObject(finestra.vista)
    evento.SetKeyCode(wx.WXK_ESCAPE)
    monkeypatch.setattr(evento, "IsAutoRepeat", lambda: True)
    finestra._tasto(evento)
    assert not evento.GetSkipped() and finestra.partita.chiamate == ["annulla_mira"]


def _ripetuto(finestra, codice, monkeypatch):
    """Un tasto ripetuto dalla tastiera, perché il dito lo tiene giù: arriva al gancio come gli altri, con IsAutoRepeat vero."""
    evento = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
    evento.SetEventObject(finestra.vista)
    evento.SetKeyCode(codice)
    monkeypatch.setattr(evento, "IsAutoRepeat", lambda: True)
    finestra._tasto(evento)
    return evento


def _rilascio(controllo, codice):
    """Il rilascio di un tasto, come lo riceve il controllo col fuoco."""
    evento = wx.KeyEvent(wx.wxEVT_KEY_UP)
    evento.SetEventObject(controllo)
    evento.SetKeyCode(codice)
    controllo.GetEventHandler().ProcessEvent(evento)
    return evento


# I tasti della mira manuale di D29, nell'ordine in cui la finestra li tiene, coi valori che muovono.
TASTI_ATTESI_DELLA_MIRA = {wx.WXK_LEFT: ("direzione", -1), wx.WXK_RIGHT: ("direzione", 1), wx.WXK_UP: ("alzo", 1), wx.WXK_DOWN: ("alzo", -1),
                           wx.WXK_HOME: ("potenza", -1), wx.WXK_END: ("potenza", 1)}


def test_i_tasti_della_mira_vanno_alla_partita_e_non_all_area_di_testo(finestra, monkeypatch, voce_finta):
    """D29: nella mira, a finestra attiva e senza dialoghi, col fuoco sul controllo Mira, le frecce orizzontali muovono la direzione,
    quelle verticali l'alzo, Home e Fine la potenza, e Invio spara: sono della partita e si fermano nel gancio, così nessuna area di
    testo li riceve. La ripetizione del tasto tenuto giù non chiama di nuovo la partita. Fuori dalla mira, coi modificatori, con un
    dialogo aperto o a finestra non attiva vanno avanti come sempre. Il tastierino resta fuori, perché NVDA se lo tiene. Col fuoco
    nella vista o nella barra li prova test_col_fuoco_nella_vista_o_nella_barra_i_tasti_dei_valori_leggono."""
    assert modulo_finestra.TASTI_DELLA_MIRA == TASTI_ATTESI_DELLA_MIRA
    assert set(modulo_finestra.TASTI_DELLO_SPARO) == {wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER}
    tastierino = {wx.WXK_NUMPAD_LEFT, wx.WXK_NUMPAD_RIGHT, wx.WXK_NUMPAD_UP, wx.WXK_NUMPAD_DOWN, wx.WXK_NUMPAD_HOME, wx.WXK_NUMPAD_END}
    assert not tastierino & set(modulo_finestra.TASTI_DELLA_MIRA)
    partita = finestra.partita
    tutti = [*TASTI_ATTESI_DELLA_MIRA, *modulo_finestra.TASTI_DELLO_SPARO]
    for fase in ("pronta", "ascolto", "volo"):
        partita.fase = fase
        assert all(_tasto(finestra, codice).GetSkipped() for codice in tutti), fase
    assert partita.chiamate == []
    # La mira che comincia porta il fuoco sul controllo Mira, al battito come allo spazio.
    partita.fase = "mira"
    finestra._al_battito()
    assert finestra.controllo_mira.HasFocus()
    vista = finestra.vista
    prima = (vista.GetValue(), vista.GetInsertionPoint())
    for codice in TASTI_ATTESI_DELLA_MIRA:
        assert not _tasto(finestra, codice).GetSkipped()
    assert partita.chiamate == [("muovi", parametro, verso) for parametro, verso in TASTI_ATTESI_DELLA_MIRA.values()]
    assert (vista.GetValue(), vista.GetInsertionPoint()) == prima
    # Tenuti giù, si ripetono senza chiamare di nuovo la partita, e restano suoi.
    for codice in tutti:
        assert not _ripetuto(finestra, codice, monkeypatch).GetSkipped()
    assert len(partita.chiamate) == len(TASTI_ATTESI_DELLA_MIRA)
    # Coi modificatori, con un dialogo aperto o a finestra non attiva, vanno avanti.
    assert _tasto(finestra, wx.WXK_LEFT, controllo=True).GetSkipped() and _tasto(finestra, wx.WXK_END, maiuscolo=True).GetSkipped()
    assert _tasto(finestra, wx.WXK_RETURN, controllo=True).GetSkipped()
    finestra._modali = 1
    assert all(_tasto(finestra, codice).GetSkipped() for codice in tutti)
    finestra._modali = 0
    monkeypatch.setattr(finestra, "IsActive", lambda: False)
    assert all(_tasto(finestra, codice).GetSkipped() for codice in tutti)
    monkeypatch.setattr(finestra, "IsActive", lambda: True)
    assert len(partita.chiamate) == len(TASTI_ATTESI_DELLA_MIRA)
    # Invio spara, e i messaggi del tiro vanno nella vista e alla sintesi; anche l'Invio del tastierino.
    partita.allo_sparo = [Messaggio("resoconto", "Tiro 1, livello 1: colpito.", "Colpito.")]
    assert not _tasto(finestra, wx.WXK_RETURN).GetSkipped()
    assert partita.chiamate[-1] == "spara" and vista.GetValue().endswith("\nTiro 1, livello 1: colpito.") and voce_finta == ["Colpito."]
    assert not _tasto(finestra, wx.WXK_NUMPAD_ENTER).GetSkipped() and partita.chiamate[-2:] == ["spara", "spara"]
    # Lo spazio nella mira resta della partita, che decide lei che cosa farne.
    assert not _tasto(finestra, wx.WXK_SPACE).GetSkipped() and partita.chiamate[-1] == "spazio"


def test_il_rilascio_dei_tasti_della_mira_arriva_alla_partita(finestra):
    """D29: il rilascio di un tasto della mira arriva al controllo col fuoco, la vista, la barra o il controllo Mira, e la partita
    lascia il valore dove si sentiva. Fuori dalla mira, e per gli altri tasti, il rilascio va avanti come sempre."""
    partita = finestra.partita
    controlli = (finestra.vista, finestra.barra, finestra.controllo_mira)
    partita.fase = "ascolto"
    assert all(_rilascio(controllo, wx.WXK_LEFT).GetSkipped() for controllo in controlli) and partita.chiamate == []
    partita.fase = "mira"
    for controllo in controlli:
        for codice, (parametro, verso) in TASTI_ATTESI_DELLA_MIRA.items():
            assert not _rilascio(controllo, codice).GetSkipped()
            assert partita.chiamate[-1] == ("lascia", parametro, verso), controllo.GetName()
        assert _rilascio(controllo, ord("A")).GetSkipped() and _rilascio(controllo, wx.WXK_RETURN).GetSkipped()
    assert len(partita.chiamate) == len(controlli) * len(TASTI_ATTESI_DELLA_MIRA)


def test_col_fuoco_nella_vista_o_nella_barra_i_tasti_dei_valori_leggono(finestra, monkeypatch):
    """D29, dalle prove del 22 novembre 2026: i tasti dei valori, frecce, Home e Fine, muovono la mira soltanto col fuoco sul controllo
    Mira. Col fuoco nella vista o nella barra, portato lì con F5 o F7, proseguono come sempre, per leggere, anche ripetuti, e la
    partita non li riceve; il rilascio dei tasti della mira arriva alla partita anche da lì, e l'Invio spara e l'Esc annulla la mira
    da tutti e due."""
    partita = finestra.partita
    partita.fase_allo_sparo = "volo"
    for tasto, controllo in ((wx.WXK_F5, finestra.vista), (wx.WXK_F7, finestra.barra)):
        partita.fase = "mira"
        finestra._al_battito()
        assert finestra.controllo_mira.HasFocus()
        assert not _tasto(finestra, tasto).GetSkipped() and controllo.HasFocus()
        prima = len(partita.chiamate)
        assert all(_tasto(finestra, codice).GetSkipped() for codice in TASTI_ATTESI_DELLA_MIRA), controllo.GetName()
        assert all(_ripetuto(finestra, codice, monkeypatch).GetSkipped() for codice in TASTI_ATTESI_DELLA_MIRA), controllo.GetName()
        assert partita.chiamate[prima:] == [] and controllo.HasFocus()
        # Il rilascio di un tasto premuto sulla Mira, che sale dopo F5 o F7, arriva lo stesso.
        assert not _rilascio(controllo, wx.WXK_RIGHT).GetSkipped() and partita.chiamate[prima:] == [("lascia", "direzione", 1)]
        # L'Invio spara anche da qui, e il fuoco resta dov'era.
        assert not _tasto(finestra, wx.WXK_RETURN).GetSkipped() and partita.chiamate[-1] == "spara" and partita.fase == "volo"
        assert controllo.HasFocus()
        # In una mira nuova, portato il fuoco di nuovo qui, l'Esc la annulla.
        partita.fase = "mira"
        finestra._al_battito()
        assert not _tasto(finestra, tasto).GetSkipped() and controllo.HasFocus()
        assert not _tasto(finestra, wx.WXK_ESCAPE).GetSkipped() and partita.chiamate[-1] == "annulla_mira" and partita.fase == "ascolto"


def test_f6_riporta_il_fuoco_sulla_mira(finestra, sentiti):
    """D29, dalle prove del 22 novembre 2026: F6 è un tasto del fuoco, fra F5 e F7. Durante la mira riporta il fuoco sul controllo
    Mira col suono vai_alla_mira, dalla vista come dalla barra, e da lì i tasti dei valori muovono di nuovo la mira; fuori dalla mira
    non c'è niente da puntare: suona il non ora, e il fuoco resta dov'è. Con un dialogo aperto F6 va avanti, come F5 e F7."""
    assert modulo_finestra.TASTI_DEL_FUOCO == (wx.WXK_F5, wx.WXK_F6, wx.WXK_F7)
    partita = finestra.partita
    vista, barra, mira = finestra.vista, finestra.barra, finestra.controllo_mira
    for fase in ("pronta", "ascolto", "volo"):
        partita.fase = fase
        vista.SetFocus()
        prima = len(sentiti)
        assert not _tasto(finestra, wx.WXK_F6).GetSkipped() and sentiti[prima:] == ["non_ora"] and vista.HasFocus(), fase
    partita.fase = "mira"
    finestra._al_battito()
    assert mira.HasFocus()
    for tasto, controllo in ((wx.WXK_F7, barra), (wx.WXK_F5, vista)):
        assert not _tasto(finestra, tasto).GetSkipped() and controllo.HasFocus()
        prima = len(sentiti)
        assert not _tasto(finestra, wx.WXK_F6).GetSkipped() and sentiti[prima:] == ["vai_alla_mira"] and mira.HasFocus()
        chiamate = len(partita.chiamate)
        assert not _tasto(finestra, wx.WXK_LEFT).GetSkipped() and partita.chiamate[chiamate:] == [("muovi", "direzione", -1)]
    # Sulla Mira F6 la lascia lì, col suo suono.
    prima = len(sentiti)
    assert not _tasto(finestra, wx.WXK_F6).GetSkipped() and sentiti[prima:] == ["vai_alla_mira"] and mira.HasFocus()
    finestra._modali = 1
    assert all(_tasto(finestra, codice).GetSkipped() for codice in modulo_finestra.TASTI_DEL_FUOCO)
    finestra._modali = 0


def test_i_tasti_rimasti_giu_senza_rilascio_si_lasciano_al_battito(finestra, monkeypatch):
    """D29: il rilascio arriva soltanto al controllo col fuoco, e può perdersi; a ogni battito i tasti che la partita crede giù e che
    la tastiera, chiesta con wx.GetKeyState, dice su valgono come lasciati. Quelli ancora giù restano, e fuori dalla mira la tastiera
    non si chiede."""
    partita = finestra.partita
    partita.fase = "mira"
    partita.giu = [("alzo", 1), ("potenza", -1), ("direzione", 1)]
    davvero_giu = {wx.WXK_UP, wx.WXK_RIGHT}
    chiesti = []

    def stato(codice):
        chiesti.append(codice)
        return codice in davvero_giu

    monkeypatch.setattr(wx, "GetKeyState", stato)
    finestra._al_battito()
    assert sorted(chiesti) == sorted([wx.WXK_UP, wx.WXK_HOME, wx.WXK_RIGHT])
    assert partita.chiamate == [("lascia", "potenza", -1)] and partita.giu == [("alzo", 1), ("direzione", 1)]
    # Il battito dopo, la freccia su è salita anche lei.
    davvero_giu.discard(wx.WXK_UP)
    finestra._al_battito()
    assert partita.chiamate[-1] == ("lascia", "alzo", 1) and partita.giu == [("direzione", 1)]
    chiesti.clear()
    partita.fase = "volo"
    finestra._al_battito()
    assert chiesti == [] and partita.chiamate.count(("lascia", "direzione", 1)) == 0


def test_il_fuoco_va_sul_controllo_mira_e_torna_alla_vista(finestra, sentiti):
    """D29: quando la mira comincia il fuoco va sul controllo Mira, che non è un testo; quando finisce, con l'Invio, con Esc o al
    battito, torna alla vista, se era ancora sulla Mira. F5 e F7 lo portano dove si vuole, e lì resta anche a mira finita."""
    partita = finestra.partita
    vista, mira = finestra.vista, finestra.controllo_mira
    vista.SetFocus()
    partita.fase = "ascolto"
    partita.fase_allo_spazio = "mira"
    _tasto(finestra, wx.WXK_SPACE)
    assert mira.HasFocus()
    partita.fase_allo_sparo = "volo"
    _tasto(finestra, wx.WXK_RETURN)
    assert vista.HasFocus()
    # Con Esc: la partita finta torna all'ascolto.
    _tasto(finestra, wx.WXK_SPACE)
    assert mira.HasFocus()
    _tasto(finestra, wx.WXK_ESCAPE)
    assert partita.fase == "ascolto" and vista.HasFocus()
    # La mira che comincia o finisce al battito, come quella che scade.
    partita.fase = "mira"
    finestra._al_battito()
    assert mira.HasFocus()
    partita.fase = "volo"
    finestra._al_battito()
    assert vista.HasFocus()
    # F7 nella mira porta il fuoco nella barra, e a mira finita resta lì.
    partita.fase = "mira"
    finestra._al_battito()
    _tasto(finestra, wx.WXK_F7)
    assert finestra.barra.HasFocus()
    partita.fase = "ascolto"
    finestra._al_battito()
    assert finestra.barra.HasFocus()


def test_f5_e_f7_portano_il_fuoco_col_loro_suono(finestra, sentiti):
    assert not _tasto(finestra, wx.WXK_F5).GetSkipped()
    assert not _tasto(finestra, wx.WXK_F7).GetSkipped()
    assert sentiti == ["vai_alla_vista", "vai_alla_barra"]


def test_il_battito_porta_i_messaggi_in_coda_col_cursore_sul_nuovo(finestra, voce_finta):
    prima = finestra.vista.GetValue()
    finestra.partita.in_arrivo = [Messaggio("resoconto", "Colpito, al centro pieno.\nPunti: 120.", "Colpito.")]
    finestra._al_battito()
    assert finestra.vista.GetValue() == prima + "\nColpito, al centro pieno.\nPunti: 120."
    assert _riga_del_cursore(finestra.vista) == ("Colpito, al centro pieno.", 0)
    # Due messaggi nello stesso battito: il cursore va all'inizio del primo, così Freccia giù li legge in
    # ordine, e la sintesi li dice tutti e due.
    finestra.partita.in_arrivo = [Messaggio("livello", "Livello 2: altri 10 colpi.", "Livello 2."), Messaggio("bonus", "Un bonus!", "")]
    finestra._al_battito()
    assert finestra.vista.GetValue().endswith("Punti: 120.\nLivello 2: altri 10 colpi.\nUn bonus!")
    assert _riga_del_cursore(finestra.vista) == ("Livello 2: altri 10 colpi.", 0)
    assert voce_finta == ["Colpito.", "Livello 2."]
    # Un battito senza messaggi non tocca la vista.
    valore = finestra.vista.GetValue()
    finestra._al_battito()
    assert finestra.vista.GetValue() == valore


def test_la_sintesi_spenta_non_legge_i_resoconti(finestra, voce_finta, sentiti):
    finestra._esegui(finestra.cambia_sintesi)
    assert not finestra.impostazioni["sintesi"] and not finestra.voci_del_menu[finestra.cambia_sintesi].IsChecked()
    assert _leggi(costanti.FILE_IMPOSTAZIONI)["sintesi"] is False
    assert sentiti == ["sintesi_spenta"]
    finestra.partita.in_arrivo = [Messaggio("resoconto", "Mancato, troppo lungo.", "Mancato.")]
    finestra._al_battito()
    assert finestra.vista.GetValue().endswith("\nMancato, troppo lungo.") and voce_finta == []
    finestra._esegui(finestra.cambia_sintesi)
    assert finestra.impostazioni["sintesi"] and sentiti[-1] == "impostazioni"
    assert voce_finta == ["La sintesi legge i resoconti della partita."]


def test_la_barra_si_riscrive_solo_se_cambia_tenendo_il_cursore(finestra, monkeypatch):
    barra = finestra.barra
    barra.SetInsertionPoint(barra.XYToPosition(3, 1))
    riscritture = []
    originale = barra.ChangeValue
    monkeypatch.setattr(barra, "ChangeValue", lambda testo: (riscritture.append(testo), originale(testo)))
    finestra._al_battito()
    assert riscritture == []
    finestra.partita.barra = ["Ascolto. Livello 2, 120 punti.", "Riserva: 9 colpi, 1 bersaglio."]
    finestra._al_battito()
    assert barra.GetValue() == "Ascolto. Livello 2, 120 punti.\nRiserva: 9 colpi, 1 bersaglio."
    assert barra.PositionToXY(barra.GetInsertionPoint()) == (True, 3, 1)
    # Una riga che si accorcia porta il cursore a fine riga; meno righe, sull'ultima.
    barra.SetInsertionPoint(barra.XYToPosition(20, 1))
    finestra.partita.barra = ["Volo.", "Riserva."]
    finestra._al_battito()
    assert barra.PositionToXY(barra.GetInsertionPoint()) == (True, len("Riserva."), 1)
    finestra.partita.barra = ["Finita."]
    finestra._al_battito()
    assert barra.PositionToXY(barra.GetInsertionPoint())[2] == 0
    assert len(riscritture) == 3


def test_un_battito_guasto_ferma_il_timer_e_la_partita_nuova_lo_riavvia(finestra):
    finestra.partita.guasta = True
    with pytest.raises(ValueError, match="battito guasto"):
        finestra._al_battito()
    assert not finestra.timer.IsRunning() and not finestra._nel_battito
    finestra.partita.guasta = False
    finestra._esegui(finestra.nuova_partita)
    assert finestra.timer.IsRunning() and finestra.partita.chiamate == ["nuova"]


def test_una_barra_guasta_dice_l_errore_una_volta_sola(finestra, sentiti, messaggi, monkeypatch):
    """Una riga della barra che non si compone, come un segnaposto sbagliato in una traduzione: la barra
    resta com'era, il registro riceve il guasto una volta, chi gioca un messaggio solo, e il battito va avanti."""
    import registro

    problemi = []
    monkeypatch.setattr(registro, "problema", lambda tipo, valore, traccia: problemi.append(tipo))
    prima = finestra.barra.GetValue()

    def guasta():
        raise KeyError("part")

    monkeypatch.setattr(finestra.partita, "righe_della_barra", guasta)
    for _giro in range(5):
        finestra._al_battito()
    wx.Yield()
    assert finestra.barra.GetValue() == prima and finestra.timer.IsRunning() and not finestra._nel_battito
    assert problemi == [KeyError]
    assert len(messaggi) == 1 and "KeyError" in messaggi[0][0] and sentiti == ["errore"]


def test_un_messaggio_guasto_non_ferma_il_battito(finestra, messaggi, monkeypatch):
    """Un messaggio che non si scrive nella vista: è già consumato, e il battito seguente va avanti; il messaggio d'errore è uno."""
    import registro

    problemi = []
    monkeypatch.setattr(registro, "problema", lambda tipo, valore, traccia: problemi.append(tipo))
    for _giro in range(3):
        finestra.partita.in_arrivo = [Messaggio("info", 5)]
        finestra._al_battito()
    wx.Yield()
    assert finestra.timer.IsRunning() and not finestra._nel_battito and problemi == [TypeError] and len(messaggi) == 1
    finestra.partita.in_arrivo = [Messaggio("info", "Si va avanti.")]
    finestra._al_battito()
    assert finestra.vista.GetValue().endswith("\nSi va avanti.")


# La pausa della finestra (D14).

def test_un_dialogo_ferma_la_partita_e_alla_chiusura_riprende(finestra, sentiti, monkeypatch):
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    durante = []
    monkeypatch.setattr(dialoghi.Lettura, "ShowModal", lambda dialogo: durante.append((partita.in_pausa, partita.automatica)) or wx.ID_OK)
    finestra._esegui(finestra.mostra_classifica)
    assert durante == [(True, True)] and partita.automatiche == ["ferma"]
    # La ripresa arriva dopo il comando, quando wx smista le chiamate rimandate, ed è silenziosa. Sul
    # desktop nascosto, nello stesso giro, la finestra può anche perdere e riprendere l'attivazione.
    wx.Yield()
    assert not partita.in_pausa and partita.automatiche[-1] == "riprende"
    assert sentiti == ["classifica"] and partita.chiamate == ["nuova"]


def test_le_domande_i_messaggi_e_la_mail_fermano_la_partita(finestra, messaggi, monkeypatch):
    import GBUtils
    import GBwx

    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    stati = []

    def chiede(genitore, testo, titolo, stile=0):
        stati.append(partita.in_pausa)
        return messaggi(genitore, testo, titolo, stile)

    monkeypatch.setattr(dialoghi, "messaggio", chiede)
    # Ctrl+N con una partita in corso: la domanda la ferma, e al no il battito la riprende.
    finestra._esegui(finestra.nuova_partita)
    assert stati == [True] and partita.in_pausa
    finestra._al_battito()
    assert not partita.in_pausa
    finestra.avvisa_dell_aggiornamento(aggiornamento.FRASE_NON_RIUSCITO)
    assert stati == [True, True]
    finestra._al_battito()
    monkeypatch.setattr(GBUtils, "informazioni_di_sistema", lambda separatore=".": [("windows", "Windows 11 Pro")])
    monkeypatch.setattr(GBwx, "finestra_mail", lambda *argomenti, **opzioni: stati.append(partita.in_pausa) or True)
    finestra._esegui(finestra.chiedi_aiuto_al_pub)
    assert stati == [True, True, True]
    finestra._al_battito()
    assert not partita.in_pausa and partita.automatiche == ["ferma", "riprende"] * 3 and "pausa" not in partita.chiamate


def test_la_finestra_non_attiva_ferma_la_partita_ma_ctrl_p_resta(finestra, monkeypatch):
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    attiva = [False]
    monkeypatch.setattr(finestra, "IsActive", lambda: attiva[0])
    finestra.GetEventHandler().ProcessEvent(wx.ActivateEvent(wx.wxEVT_ACTIVATE, False))
    assert partita.in_pausa and partita.automatiche == ["ferma"]
    # Finché la finestra non torna attiva, il battito la tiene ferma.
    finestra._al_battito()
    assert partita.in_pausa
    attiva[0] = True
    finestra.GetEventHandler().ProcessEvent(wx.ActivateEvent(wx.wxEVT_ACTIVATE, True))
    wx.Yield()
    assert not partita.in_pausa and partita.automatiche[-1] == "riprende"
    # Una pausa chiesta con Ctrl+P resta: né il ritorno della finestra né il battito la tolgono.
    finestra._esegui(finestra.pausa)
    assert partita.in_pausa
    prima = list(partita.automatiche)
    attiva[0] = False
    finestra.GetEventHandler().ProcessEvent(wx.ActivateEvent(wx.wxEVT_ACTIVATE, False))
    attiva[0] = True
    finestra.GetEventHandler().ProcessEvent(wx.ActivateEvent(wx.wxEVT_ACTIVATE, True))
    wx.Yield()
    finestra._al_battito()
    assert partita.in_pausa and not partita.automatica and partita.automatiche == prima


# I menu (D22): un menu aperto ferma la partita come un dialogo.

def _evento_del_menu(finestra, tipo):
    """L'evento che wx manda alla finestra quando il primo menu della barra si apre o si chiude."""
    evento = wx.MenuEvent(tipo, 0, finestra.GetMenuBar().GetMenu(0))
    evento.SetEventObject(finestra)
    finestra.GetEventHandler().ProcessEvent(evento)


def test_un_menu_aperto_ferma_la_partita_e_alla_chiusura_riprende(finestra, sentiti, monkeypatch):
    """Il menu aperto con Alt o F10 ferma la partita, in silenzio, come un dialogo; il battito la tiene
    ferma finché il menu resta aperto, e la chiusura la fa riprendere. Qui Windows non sa dire dei menu,
    e valgono gli eventi di wx."""
    monkeypatch.setattr(modulo_finestra, "menu_di_windows", lambda: None)
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
    assert partita.in_pausa and partita.automatiche == ["ferma"] and finestra._finestra_modale_aperta()
    finestra._al_battito()
    wx.Yield()
    assert partita.in_pausa and partita.automatiche == ["ferma"]
    _evento_del_menu(finestra, wx.wxEVT_MENU_CLOSE)
    wx.Yield()
    assert not partita.in_pausa and partita.automatiche == ["ferma", "riprende"] and not finestra._finestra_modale_aperta()
    assert sentiti == [] and "pausa" not in partita.chiamate


def test_il_menu_lo_dice_windows_anche_senza_gli_eventi(finestra, monkeypatch):
    """Con Alt la barra dei menu si evidenzia senza tendina, e wx non manda eventi: lo dice Windows, e il
    battito ferma la partita; quando Windows dice che il menu non c'è più, la riprende, anche se la
    chiusura di wx si è persa."""
    aperto = [True]
    monkeypatch.setattr(modulo_finestra, "menu_di_windows", lambda: aperto[0])
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    finestra._al_battito()
    assert partita.in_pausa and partita.automatiche == ["ferma"]
    aperto[0] = False
    finestra._al_battito()
    assert not partita.in_pausa and partita.automatiche == ["ferma", "riprende"]
    # L'apertura di wx, poi la chiusura persa: la parola di Windows vale.
    aperto[0] = True
    _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
    finestra._al_battito()
    assert partita.in_pausa
    aperto[0] = False
    finestra._al_battito()
    assert not partita.in_pausa and not finestra._menu_dagli_eventi
    # L'apertura di wx persa, col menu che Windows dice aperto: il battito non fa ripartire la partita.
    aperto[0] = True
    finestra._al_battito()
    assert partita.in_pausa and partita.automatiche == ["ferma", "riprende"] * 2 + ["ferma"]


def test_gli_eventi_dei_menu_persi_o_spaiati(finestra, sentiti, monkeypatch):
    """Dove Windows non sa dire dei menu: due aperture e una chiusura, come quando ci si sposta da un menu
    all'altro, fanno riprendere la partita alla chiusura, perché il menu aperto non è un conto; una
    chiusura persa la tiene ferma finché un tasto non arriva alla finestra, cosa che a menu aperto non
    succede."""
    monkeypatch.setattr(modulo_finestra, "menu_di_windows", lambda: None)
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
    _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
    _evento_del_menu(finestra, wx.wxEVT_MENU_CLOSE)
    wx.Yield()
    assert not partita.in_pausa and partita.automatiche == ["ferma", "riprende"]
    _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
    for _battito in range(3):
        finestra._al_battito()
    assert partita.in_pausa
    evento = _tasto(finestra, wx.WXK_F5)
    assert not evento.GetSkipped() and sentiti[-1] == "vai_alla_vista"
    finestra._al_battito()
    assert not partita.in_pausa and partita.automatiche == ["ferma", "riprende"] * 2


def test_il_comando_del_menu_che_apre_un_dialogo_tiene_ferma_la_partita(finestra, sentiti, monkeypatch):
    """wx manda la chiusura del menu prima del comando: la prova della ripresa arriva dentro il dialogo
    aperto dal comando, e lo lascia stare; la partita riprende quando il dialogo si chiude, con una pausa
    automatica sola."""
    monkeypatch.setattr(modulo_finestra, "menu_di_windows", lambda: None)
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    durante = []

    def legge(dialogo):
        wx.Yield()
        durante.append((partita.in_pausa, partita.automatica))
        return wx.ID_OK

    monkeypatch.setattr(dialoghi.Lettura, "ShowModal", legge)
    _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
    _evento_del_menu(finestra, wx.wxEVT_MENU_CLOSE)
    finestra._esegui(finestra.mostra_classifica)
    assert durante == [(True, True)]
    wx.Yield()
    assert not partita.in_pausa and partita.automatiche == ["ferma", "riprende"] and sentiti == ["classifica"]


def test_il_tasto_rapido_non_e_un_menu_aperto(finestra, monkeypatch):
    """Un tasto rapido dei menu, come Ctrl+P, passa da TranslateAccelerator di Windows, e wx manda
    l'apertura del menu, il comando e la chiusura, ma per Windows nessun menu si è aperto: la partita non
    si ferma, e la chiusura non ha niente da riprendere. Ctrl+P arriva alla partita in corso, non ferma."""
    monkeypatch.setattr(modulo_finestra, "menu_di_windows", lambda: False)
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
    assert not partita.in_pausa and partita.automatiche == [] and not finestra._menu_dagli_eventi
    evento = wx.CommandEvent(wx.wxEVT_MENU, finestra.voci_del_menu[finestra.pausa].GetId())
    evento.SetEventObject(finestra)
    finestra.GetEventHandler().ProcessEvent(evento)
    _evento_del_menu(finestra, wx.wxEVT_MENU_CLOSE)
    wx.Yield()
    finestra._al_battito()
    assert partita.chiamate[-1] == "pausa" and partita.in_pausa and partita.automatiche == []


def test_menu_di_windows_risponde():
    """La domanda vera a Windows: nelle prove nessun menu è aperto; fuori da Windows non si sa."""
    if sys.platform == "win32":
        assert modulo_finestra.menu_di_windows() is False
    else:
        assert modulo_finestra.menu_di_windows() is None


class PartitaConLaFine(PartitaFinta):
    """Una partita finta finita col tonfo dell'ultimo tiro che suona ancora: la fine aspetta, come nella
    partita vera, e la finestra la chiede con messaggi_in_attesa (D22)."""

    def __init__(self, motore, suoni_, statistiche):
        super().__init__(motore, suoni_, statistiche)
        self.fase = "finita"
        self.aspettano = [Messaggio("resoconto", "Tiro 9, livello 2: a vuoto; la riserva è vuota.", "A vuoto, riserva vuota."), _fine(1234)]

    def messaggi_in_attesa(self):
        messaggi, self.aspettano = self.aspettano, []
        return messaggi


def test_la_fine_che_aspetta_il_tonfo_arriva_prima_della_partita_nuova_e_dell_uscita(app_wx, sentiti, messaggi, senza_modali, monkeypatch):
    """D22: Ctrl+N a partita appena finita, col tonfo che suona ancora: prima la fine, con la sua
    classifica, poi la partita nuova; e lo stesso all'uscita, dove la fine va in classifica col nome."""
    monkeypatch.setattr(dialoghi.Nome, "ShowModal", lambda dialogo: (dialogo.testo.SetValue("Gabry"), dialogo.conferma(), dialogo.GetReturnCode())[-1])
    f = FinestraPrincipale(fabbrica_della_partita=PartitaConLaFine)
    try:
        f.Show()
        monkeypatch.setattr(f, "IsActive", lambda: True)
        f._esegui(f.nuova_partita)
        testo = f.vista.GetValue()
        assert testo.index("\nPartita finita: 1234 punti.") < testo.index("\nComincia la partita, livello 1.")
        assert [(v["nome"], v["punteggio"]) for v in classifica.carica()[0]] == [("Gabry", 1234)] and f.partita.chiamate == ["nuova"]
        assert sentiti == ["in_classifica", "confermato"]
    finally:
        _chiudi(f)
    f = FinestraPrincipale(fabbrica_della_partita=PartitaConLaFine)
    try:
        f.Show()
        f.partita.aspettano[-1] = _fine(900)
        f.Close()
        assert f.chiusa and [v["punteggio"] for v in classifica.carica()[0]] == [1234, 900]
    finally:
        _chiudi(f)


# La partita, la classifica e le letture.

def _fine(punteggio):
    return Messaggio("fine", f"Partita finita: {punteggio} punti.", "Partita finita.",
                     {"punteggio": punteggio, "livello": 2, "tiri": 9, "centri": 5, "durata_s": 61.0, "precisione_globale": 0.6134})


def test_a_partita_finita_il_nome_e_la_classifica_nella_vista(finestra, sentiti, voce_finta, monkeypatch):
    finestra.impostazioni["ultimo_nome"] = "Gabry"
    proposti = []

    def scrive_il_nome(dialogo):
        proposti.append((dialogo.testo.GetValue(), dialogo.GRAZIA_ALL_APERTURA))
        dialogo.testo.SetValue("Gabriele")
        dialogo.conferma()
        return dialogo.GetReturnCode()

    monkeypatch.setattr(dialoghi.Nome, "ShowModal", scrive_il_nome)
    finestra.partita.in_arrivo = [_fine(1234)]
    finestra._al_battito()
    assert proposti == [("Gabry", True)]
    voci = _leggi(costanti.FILE_CLASSIFICA)["voci"]
    assert len(voci) == 1 and voci[0]["nome"] == "Gabriele" and voci[0]["punteggio"] == 1234 and voci[0]["centri"] == 5
    assert voci[0]["precisione_globale"] == 0.6134
    assert _leggi(costanti.FILE_IMPOSTAZIONI)["ultimo_nome"] == "Gabriele"
    statistiche = _leggi(costanti.FILE_STATISTICHE)
    assert statistiche["conti"]["ingressi_in_classifica"] == 1
    # D27: la griglia in coda alla vista, col cursore sulla riga della voce nuova; alla sintesi la voce a parole.
    nuova = "  1 Gabriele 1.234       2    9      5      61,3% 01:01      0 nuova"
    assert finestra.vista.GetValue().endswith("\nPartita finita: 1234 punti.\nClassifica, 1 voce, ordinata per punteggio.\n"
                                              f"Pos Nome     Punti Livello Tiri Centri Precisione Tempo Giorni\n{nuova}")
    assert _riga_del_cursore(finestra.vista) == (nuova, 0)
    assert sentiti == ["in_classifica", "confermato"]
    assert voce_finta == ["Partita finita.", "1. Gabriele, 1.234 punti, livello 2, 9 tiri, 5 centri, precisione globale media 61,3%, 1 minuto e 1 secondo, nuova."]


def test_la_voce_nuova_in_mezzo_alla_classifica(finestra, monkeypatch):
    vecchie = [classifica.nuova_voce(nome, {"punteggio": punti}) for nome, punti in (("Anna", 5000), ("Bruno", 100))]
    assert classifica.salva(vecchie)
    monkeypatch.setattr(dialoghi.Nome, "ShowModal", lambda dialogo: (dialogo.conferma(), dialogo.GetReturnCode())[-1])
    finestra.impostazioni["ultimo_nome"] = "Gabry"
    finestra.partita.in_arrivo = [_fine(800)]
    finestra._al_battito()
    assert [v["nome"] for v in _leggi(costanti.FILE_CLASSIFICA)["voci"]] == ["Anna", "Gabry", "Bruno"]
    riga, colonna = _riga_del_cursore(finestra.vista)
    assert colonna == 0 and riga.startswith("  2 Gabry   800 ") and riga.endswith(" nuova")


def test_la_partita_misura_il_posto_sulla_classifica_della_finestra(finestra, monkeypatch):
    """D33: la finestra dà alla partita la sua classifica in memoria, per il posto nella situazione dopo ogni tiro; dopo una voce
    nuova la partita vede la classifica nuova."""
    assert finestra.partita.classifica() is finestra._voci_della_classifica and finestra._voci_della_classifica == []
    monkeypatch.setattr(dialoghi.Nome, "ShowModal", lambda dialogo: (dialogo.conferma(), dialogo.GetReturnCode())[-1])
    finestra.impostazioni["ultimo_nome"] = "Gabry"
    finestra.partita.in_arrivo = [_fine(800)]
    finestra._al_battito()
    assert [(v["nome"], v["punteggio"]) for v in finestra.partita.classifica()] == [("Gabry", 800)]


def test_fuori_classifica_e_nome_annullato(finestra, sentiti, monkeypatch):
    aperti = []
    monkeypatch.setattr(dialoghi.Nome, "ShowModal", lambda dialogo: aperti.append(dialogo) or wx.ID_CANCEL)
    finestra.partita.in_arrivo = [_fine(0)]
    finestra._al_battito()
    assert aperti == [] and sentiti == ["fuori_classifica"]
    assert finestra.vista.GetValue().endswith("\nIl punteggio di questa partita non entra in classifica.")
    finestra.partita.in_arrivo = [_fine(500)]
    finestra._al_battito()
    assert len(aperti) == 1 and sentiti[-2:] == ["in_classifica", "annullato"]
    assert classifica.carica()[0] == []
    assert finestra.vista.GetValue().endswith("questa partita resta fuori dalla classifica.")


def test_la_classifica_aspetta_che_il_dialogo_aperto_si_chiuda(finestra, sentiti, monkeypatch):
    monkeypatch.setattr(dialoghi.Nome, "ShowModal", lambda dialogo: (dialogo.conferma(), dialogo.GetReturnCode())[-1])
    finestra.impostazioni["ultimo_nome"] = "Gabry"
    finestra._modali = 1
    finestra.partita.in_arrivo = [_fine(300)]
    finestra._al_battito()
    assert finestra._partite_da_concludere and sentiti == []
    finestra._modali = 0
    finestra._al_battito()
    assert not finestra._partite_da_concludere and sentiti == ["in_classifica", "confermato"]


def _porta_udibile_a(motore, flusso, campione):
    """Il flusso finto prepara il blocco che contiene il campione, e l'orologio si sposta finché il campione udibile è quello."""
    flusso.avanza_fino(campione)
    inizio, dac = motore._riferimento
    flusso.time = dac + (campione - inizio) / flusso.fs


def test_il_nome_si_chiede_dopo_il_suono_dell_ingresso(app_wx, sentiti, messaggi, senza_modali, monkeypatch):
    """Con la scheda, il dialogo del nome aspetta la fine del suono dell'ingresso; intanto conta come un
    dialogo aperto, e lo spazio non comincia una partita nuova. Se l'orologio del motore si fermasse,
    il nome si chiede alla scadenza in tempo vero; all'uscita si chiede subito, dopo il suono."""
    motore, flusso = motore_finto()
    f = _crea(motore=motore)
    monkeypatch.setattr(f, "IsActive", lambda: True)
    aperti = []
    monkeypatch.setattr(dialoghi.Nome, "ShowModal", lambda dialogo: aperti.append(dialogo.testo.GetValue()) or (dialogo.conferma(), dialogo.GetReturnCode())[-1])
    try:
        f.impostazioni["ultimo_nome"] = "Gabry"
        f.partita.in_arrivo = [_fine(700)]
        f._al_battito()
        attesa = f._nome_in_attesa
        assert sentiti == ["in_classifica"] and aperti == [] and attesa is not None and f._dialogo_aperto()
        # Il suono parte dal prossimo campione sicuro, perché la partita finta non ha avvisi da far finire prima.
        assert attesa.fine == motore.prossimo_campione() + CAMPIONI_FINTI > motore.campione_udibile()
        assert _tasto(f, wx.WXK_SPACE).GetSkipped() and f.partita.chiamate == []
        _porta_udibile_a(motore, flusso, attesa.fine - 1)
        f._al_battito()
        assert aperti == []
        _porta_udibile_a(motore, flusso, attesa.fine)
        f._al_battito()
        assert aperti == ["Gabry"] and f._nome_in_attesa is None and sentiti[-1] == "confermato"
        assert [v["punteggio"] for v in classifica.carica()[0]] == [700]
        # L'orologio fermo: alla scadenza il nome si chiede lo stesso.
        f.partita.in_arrivo = [_fine(600)]
        f._al_battito()
        assert f._nome_in_attesa is not None
        f._nome_in_attesa = f._nome_in_attesa._replace(scadenza=0.0)
        f._al_battito()
        assert len(aperti) == 2 and f._nome_in_attesa is None
        # All'uscita, con la partita che aspetta ancora il nome: si chiede prima di chiudere.
        f.partita.in_arrivo = [_fine(500)]
        f._al_battito()
        assert f._nome_in_attesa is not None
        f.Close()
        assert f.chiusa and len(aperti) == 3 and [v["punteggio"] for v in classifica.carica()[0]] == [700, 600, 500]
    finally:
        _chiudi(f)


def test_la_partita_finita_durante_la_domanda_d_uscita_entra_in_classifica(app_wx, sentiti, messaggi, senza_modali, monkeypatch):
    """Il colpo cade mentre la domanda d'uscita aspetta, e la partita finisce: al sì il nome si chiede e la voce si scrive, poi Ciechino esce."""
    f = _crea()
    monkeypatch.setattr(f, "IsActive", lambda: True)
    monkeypatch.setattr(dialoghi.Nome, "ShowModal", lambda dialogo: (dialogo.testo.SetValue("Gabry"), dialogo.conferma(), dialogo.GetReturnCode())[-1])
    congedi = []
    f.al_congedo = lambda: congedi.append(f.chiusa)
    try:
        f.partita.fase = "volo"

        def finisce_mentre_chiede(*argomenti):
            # Dentro la domanda il timer batte ancora: l'ultimo colpo cade, e la partita finisce.
            f.partita.fase = "finita"
            f.partita.in_arrivo = [_fine(900)]
            f._al_battito()
            assert f._partite_da_concludere
            return messaggi(*argomenti)

        monkeypatch.setattr(dialoghi, "messaggio", finisce_mentre_chiede)
        messaggi.risposta = wx.YES
        f.Close()
        assert f.chiusa and congedi == [True]
        voci = classifica.carica()[0]
        assert [(v["nome"], v["punteggio"]) for v in voci] == [("Gabry", 900)]
        assert sentiti == ["domanda", "in_classifica", "confermato", "uscita"]
    finally:
        _chiudi(f)


def test_gli_avvisi_dei_dati_vanno_anche_alla_sintesi(finestra, voce_finta, monkeypatch):
    monkeypatch.setattr(finestra.statistiche, "salva", lambda: False)
    monkeypatch.setattr(impostazioni, "salva", lambda dati, percorso=None: False)
    assert not finestra._salva_le_statistiche() and not finestra._salva_le_impostazioni()
    assert voce_finta == ["Le statistiche non si sono potute salvare: le ritroverai come erano all'ultimo salvataggio riuscito.",
                          "Le impostazioni non si sono potute salvare: valgono fino all'uscita."]
    avviso = "La classifica non si legge: riprendo dalla copia di riserva."
    monkeypatch.setattr(classifica, "carica", lambda percorso=None: ([], avviso))
    monkeypatch.setattr(classifica, "salva", lambda voci, percorso=None: False)
    monkeypatch.setattr(dialoghi.Nome, "ShowModal", lambda dialogo: (dialogo.conferma(), dialogo.GetReturnCode())[-1])
    finestra.impostazioni["ultimo_nome"] = "Gabry"
    finestra.partita.in_arrivo = [_fine(500)]
    finestra._al_battito()
    assert avviso in voce_finta and "La classifica non si è potuta salvare: la voce nuova vale fino all'uscita." in voce_finta
    # La voce nuova vale fino all'uscita: Ctrl+K la mostra, anche se il file non la contiene.
    letti = []
    monkeypatch.setattr(dialoghi.Lettura, "ShowModal", lambda dialogo: letti.append(dialogo.testo.GetValue()) or wx.ID_OK)
    finestra.mostra_classifica()
    assert len(letti) == 1 and "Gabry" in letti[0] and avviso not in letti[0]


def test_classifica_e_statistiche_si_leggono_nel_dialogo(finestra, sentiti, monkeypatch):
    letti = []
    monkeypatch.setattr(dialoghi.Lettura, "ShowModal", lambda dialogo: letti.append((dialogo.GetTitle(), dialogo.testo.GetValue(),
                                                                                       dialogo.IsMaximized(), dialogo.testo.GetFont().IsFixedWidth())) or wx.ID_OK)
    finestra._esegui(finestra.mostra_classifica)
    finestra._esegui(finestra.mostra_statistiche)
    # La classifica, una griglia (D27), sta in un carattere a larghezza fissa, alla grandezza di quello del sistema.
    assert letti[0] == ("Classifica", classifica.righe([])[0], True, True)
    assert letti[1][0] == "Statistiche" and letti[1][2] and "\n\n" not in letti[1][1] and len(letti[1][1].splitlines()) > 1 and not letti[1][3]
    assert sentiti == ["classifica", "statistiche"]


def test_la_griglia_della_classifica_arriva_intera_nel_dialogo(finestra, monkeypatch):
    """Il testo scritto dopo il carattere a larghezza fissa arriva tutto, spazi in testa compresi."""
    voci = [classifica.nuova_voce(nome, {"punteggio": punti, "livello": 1, "tiri": 10, "centri": 3, "durata_s": 95.0, "precisione_globale": 0.5})
            for nome, punti in (("Anna", 5000), ("Bruno", 100))]
    assert classifica.salva(voci)
    letti = []
    monkeypatch.setattr(dialoghi.Lettura, "ShowModal", lambda dialogo: letti.append(dialogo.testo.GetValue()) or wx.ID_OK)
    finestra._esegui(finestra.mostra_classifica)
    assert letti[0].splitlines() == classifica.righe(voci)


def test_nuova_partita_e_pausa(finestra, messaggi, sentiti):
    finestra._esegui(finestra.nuova_partita)
    assert finestra.partita.chiamate == ["nuova"] and messaggi == []
    # Con una partita in corso, prima la domanda, col no già scelto: al no non succede niente.
    finestra._esegui(finestra.nuova_partita)
    assert len(messaggi) == 1 and messaggi[-1][2] & wx.NO_DEFAULT and finestra.partita.chiamate == ["nuova"]
    assert sentiti == ["domanda", "annullato"]
    messaggi.risposta = wx.YES
    finestra._esegui(finestra.nuova_partita)
    assert finestra.partita.chiamate == ["nuova", "abbandona", "nuova"]
    assert "Partita abbandonata." in finestra.vista.GetValue()
    finestra._esegui(finestra.pausa)
    assert finestra.partita.chiamate[-1] == "pausa" and finestra.vista.GetValue().endswith("\nPartita in pausa.")


# L'uscita.

def test_l_uscita_con_una_partita_in_corso_chiede(app_wx, sentiti, messaggi, senza_modali, monkeypatch):
    f = _crea()
    try:
        f.partita.fase = "ascolto"
        f.Close()
        # Il no, già scelto: la finestra resta, e la partita anche.
        assert f and not f.chiusa and f.partita.chiamate == []
        assert len(messaggi) == 1 and messaggi[0][2] & wx.YES_NO and messaggi[0][2] & wx.NO_DEFAULT
        assert sentiti == ["domanda", "annullato"]
        messaggi.risposta = wx.YES
        motore, partita = f.motore, f.partita
        f.Close()
        assert partita.chiamate == ["abbandona"] and f.chiusa and f.congedo == modulo_finestra.USCITA
        assert sentiti[-2:] == ["domanda", "uscita"] and motore.chiuso
        assert _leggi(costanti.FILE_STATISTICHE)["conti"]["avvii"] == 1
    finally:
        _chiudi(f)
    assert not f


def test_l_uscita_senza_partita_non_chiede(app_wx, sentiti, messaggi, senza_modali):
    f = _crea()
    try:
        f.Close()
        assert messaggi == [] and f.chiusa and sentiti == ["uscita"]
    finally:
        _chiudi(f)


def test_chiudi_per_aggiornare_non_chiede_e_abbandona(app_wx, sentiti, messaggi, senza_modali):
    f = _crea()
    try:
        f.partita.fase = "volo"
        f.chiudi_per_aggiornare()
        assert messaggi == [] and f.chiusa and f.congedo == modulo_finestra.PER_AGGIORNARE and f.partita.chiamate == ["abbandona"]
    finally:
        _chiudi(f)


def test_chiudi_per_aggiornare_aspetta_il_nome_in_classifica(finestra, monkeypatch):
    """Il dialogo del nome non si chiude come annullato, e la chiusura aspetta anche le partite che attendono la classifica o il nome."""
    riprove = []
    monkeypatch.setattr(wx, "CallLater", lambda millesimi, funzione, *argomenti: riprove.append(funzione))
    nome = dialoghi.Nome(finestra, "Gabry")
    lettura = dialoghi.Lettura(finestra, "Classifica", "Nessuna voce.")
    chiusi = []
    try:
        with monkeypatch.context() as provvisorio:
            provvisorio.setattr(wx.Dialog, "IsModal", lambda dialogo: dialogo in (nome, lettura))
            provvisorio.setattr(wx.Dialog, "EndModal", lambda dialogo, codice: chiusi.append((type(dialogo).__name__, codice)))
            finestra.chiudi_per_aggiornare()
        assert chiusi == [("Lettura", wx.ID_CANCEL)] and not finestra.chiusa and len(riprove) == 1
    finally:
        nome.Destroy()
        lettura.Destroy()
    finestra._partite_da_concludere.append({"punteggio": 300})
    finestra.chiudi_per_aggiornare()
    assert not finestra.chiusa and len(riprove) == 2
    finestra._partite_da_concludere.clear()
    finestra._nome_in_attesa = modulo_finestra.NomeInAttesa(None, 0.0, {"punteggio": 300}, [], 1)
    finestra.chiudi_per_aggiornare()
    assert not finestra.chiusa and len(riprove) == 3
    finestra._nome_in_attesa = None
    finestra.chiudi_per_aggiornare()
    assert finestra.chiusa and finestra.congedo == modulo_finestra.PER_AGGIORNARE


def test_la_riga_dello_scaricamento_se_ne_va_se_l_aggiornamento_non_riesce(finestra, sentiti, messaggi, monkeypatch):
    finestra.avanzamento_dell_aggiornamento(63, 100)
    assert finestra.barra.GetValue().splitlines()[-1] == "Aggiornamento scaricato al 63%."
    finestra.avvisa_dell_aggiornamento(aggiornamento.FRASE_NON_RIUSCITO)
    assert "Aggiornamento scaricato" not in finestra.barra.GetValue()
    # Anche quando la chiusura per l'aggiornamento rinuncia, perché un messaggio è rimasto aperto troppo a lungo.
    monkeypatch.setattr(wx, "CallLater", lambda *argomenti: None)
    finestra.avanzamento_dell_aggiornamento(1, 2)
    assert finestra.barra.GetValue().splitlines()[-1] == "Aggiornamento scaricato al 50%."
    finestra._modali = 1
    finestra.chiudi_per_aggiornare()
    finestra._chiusura_dal -= modulo_finestra.LIMITE_DELLA_CHIUSURA
    finestra.chiudi_per_aggiornare()
    finestra._modali = 0
    assert "Aggiornamento scaricato" not in finestra.barra.GetValue() and not finestra.chiusa
    assert sentiti[-1] == "aggiornamento_non_riuscito"


# Le impostazioni.

def test_la_curva_del_volume():
    assert guadagno_del_volume(0) == 0.0 and guadagno_del_volume(-5) == 0.0
    assert guadagno_del_volume(100) == pytest.approx(1.0) and guadagno_del_volume(150) == pytest.approx(1.0)
    assert guadagno_del_volume(50) == pytest.approx(10 ** (-modulo_finestra.DINAMICA_DEL_VOLUME_DB / 2 / 20))
    valori = [guadagno_del_volume(v) for v in range(impostazioni.VOLUME_MINIMO, impostazioni.VOLUME_MASSIMO + 1)]
    assert all(b > a for a, b in itertools.pairwise(valori))


def test_il_volume_si_sente_mentre_cambia_e_annulla_torna_indietro(finestra, sentiti, monkeypatch):
    def ritocca(dialogo, conferma):
        dialogo.numero.SetValue(30)
        dialogo.prova()
        assert finestra.motore.volume == pytest.approx(guadagno_del_volume(30))
        if conferma:
            dialogo.conferma()
            return dialogo.GetReturnCode()
        return wx.ID_CANCEL

    monkeypatch.setattr(dialoghi.Volume, "ShowModal", lambda dialogo: ritocca(dialogo, False))
    finestra._esegui(finestra.cambia_volume)
    assert finestra.motore.volume == pytest.approx(guadagno_del_volume(costanti.VOLUME_PREDEFINITO))
    assert sentiti == ["volume", "volume", "annullato"]
    monkeypatch.setattr(dialoghi.Volume, "ShowModal", lambda dialogo: ritocca(dialogo, True))
    finestra._esegui(finestra.cambia_volume)
    assert finestra.motore.volume == pytest.approx(guadagno_del_volume(30)) and _leggi(costanti.FILE_IMPOSTAZIONI)["volume"] == 30
    assert sentiti[-3:] == ["volume", "volume", "confermato"] and finestra.vista.GetValue().endswith("\nIl volume è 30.")


def test_la_lingua_con_una_lingua_sola(finestra, sentiti, monkeypatch):
    import lingua

    monkeypatch.setattr(lingua, "lingue", lambda: ["it"])
    monkeypatch.setattr(dialoghi.SceltaLingua, "ShowModal", lambda dialogo: (dialogo.conferma(), dialogo.GetReturnCode())[-1])
    finestra._esegui(finestra.cambia_lingua)
    assert sentiti == ["lingua", "confermato"]
    assert "per ora Ciechino parla solo questa" in finestra.vista.GetValue().splitlines()[-1]


def test_la_lingua_nuova_offre_di_riaprire(finestra, sentiti, messaggi, monkeypatch):
    import istanza
    import lingua

    scelte, riaperture = [], []
    monkeypatch.setattr(lingua, "lingue", lambda: ["en", "it"])
    monkeypatch.setattr(lingua, "scegli", scelte.append)
    monkeypatch.setattr(istanza, "riapri", lambda: riaperture.append(True))

    def sceglie_l_inglese(dialogo):
        dialogo.elenco.SetSelection(0)
        dialogo.conferma()
        return dialogo.GetReturnCode()

    monkeypatch.setattr(dialoghi.SceltaLingua, "ShowModal", sceglie_l_inglese)
    finestra._esegui(finestra.cambia_lingua)
    # Al no la lingua vale dal prossimo avvio, e Ciechino resta aperto.
    assert scelte == ["en"] and sentiti == ["lingua", "lingua_scelta", "domanda", "annullato"] and not finestra.chiusa and riaperture == []
    messaggi.risposta = wx.YES
    finestra._esegui(finestra.cambia_lingua)
    assert finestra.chiusa and riaperture == [True] and finestra.congedo == modulo_finestra.RIAPERTURA and sentiti[-1] == "riavvio"


def _spia_del_motore(finestra, monkeypatch):
    """Le voci calcolate aggiunte al motore e le loro fermate, in ordine: ("aggiungi", voce) e ("ferma", voce, campione).
    I suoni già pronti degli eventi, che il motore suona come VoceBuffer, non contano."""
    motore = finestra.motore
    visti = []
    aggiungi, ferma = motore.aggiungi, motore.ferma

    def spia_aggiungi(voce):
        if isinstance(voce, VoceTono):
            visti.append(("aggiungi", voce))
        return aggiungi(voce)

    def spia_ferma(voce, campione=None, sfumatura=costanti.SFUMATURA_STOP):
        if isinstance(voce, VoceTono):
            visti.append(("ferma", voce, campione))
        return ferma(voce, campione, sfumatura)

    monkeypatch.setattr(motore, "aggiungi", spia_aggiungi)
    monkeypatch.setattr(motore, "ferma", spia_ferma)
    return visti


def _scelta_del_proiettile(monkeypatch, indici, conferma=True):
    """Il dialogo del proiettile al posto di quello vero: si sposta sulle righe indici, fermandosi su ognuna, poi preme OK o Annulla."""
    aperti = []

    def sceglie(dialogo):
        aperti.append(dialogo)
        for indice in indici:
            dialogo.elenco.SetSelection(indice)
            dialogo.prova()
        if conferma:
            dialogo.conferma()
            return dialogo.GetReturnCode()
        return wx.ID_CANCEL

    monkeypatch.setattr(dialoghi.SceltaProiettile, "ShowModal", sceglie)
    return aperti


def test_il_proiettile_si_sceglie_ascoltando_l_esempio(finestra, sentiti, voce_finta, monkeypatch):
    """D17: senza partita in corso la forma vale subito; ogni riga su cui ci si ferma fa sentire il suo esempio, che si ferma quando parte il seguente e quando il dialogo si chiude."""
    partita = finestra.partita
    # All'apertura la partita riceve la forma delle impostazioni.
    assert partita.proiettili == [costanti.PROIETTILE_FORMA]
    visti = _spia_del_motore(finestra, monkeypatch)
    aperti = _scelta_del_proiettile(monkeypatch, (2, 1))
    finestra._esegui(finestra.cambia_proiettile)
    assert len(aperti) == 1 and aperti[0].forme == costanti.PROIETTILE_FORME
    # Due esempi, uno per riga, con la voce del gioco: il primo si ferma prima che parta il secondo, il secondo alla chiusura.
    voci = [passo[1] for passo in visti if passo[0] == "aggiungi"]
    assert [voce.forma for voce in voci] == ["sega_morbida", "triangolo"]
    durata = round(modulo_partita.DURATA_DELL_ESEMPIO * finestra.motore.fs) - round(modulo_partita.SFUMATURA_DELL_ESEMPIO * finestra.motore.fs)
    primo, secondo = voci
    assert visti == [("aggiungi", primo), ("ferma", primo, primo.inizio + durata), ("ferma", primo, None),
                     ("aggiungi", secondo), ("ferma", secondo, secondo.inizio + durata), ("ferma", secondo, None)]
    assert finestra._esempio_del_proiettile is None
    assert sentiti == ["proiettile", "confermato"]
    assert finestra.impostazioni["proiettile"] == "triangolo" and _leggi(costanti.FILE_IMPOSTAZIONI)["proiettile"] == "triangolo"
    assert partita.proiettili == [costanti.PROIETTILE_FORMA, "triangolo"]
    assert finestra.statistiche.dati["conti"]["proiettile_cambiato"] == 1
    testo = "Il proiettile adesso suona in triangolo: lo senti dalla prossima partita."
    assert finestra.vista.GetValue().endswith("\n" + testo) and voce_finta[-1] == testo


def test_il_proiettile_scelto_a_partita_in_corso_vale_dalla_seguente(finestra, sentiti, voce_finta, monkeypatch):
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    durante = []
    aperti = _scelta_del_proiettile(monkeypatch, (2,))
    originale = dialoghi.SceltaProiettile.ShowModal

    def guarda_e_sceglie(dialogo):
        # Col dialogo aperto la partita è ferma, come con ogni dialogo (D14).
        durante.append((partita.in_pausa, partita.automatica))
        return originale(dialogo)

    monkeypatch.setattr(dialoghi.SceltaProiettile, "ShowModal", guarda_e_sceglie)
    finestra._esegui(finestra.cambia_proiettile)
    assert len(aperti) == 1 and durante == [(True, True)]
    assert sentiti == ["proiettile", "confermato"]
    assert partita.proiettili[-1] == "sega_morbida" and _leggi(costanti.FILE_IMPOSTAZIONI)["proiettile"] == "sega_morbida"
    testo = "Il proiettile suonerà in sega morbida dalla prossima partita: quella in corso resta in seno."
    assert finestra.vista.GetValue().endswith("\n" + testo) and voce_finta[-1] == testo
    # Chiuso il dialogo, la partita riprende da sola.
    wx.Yield()
    assert not partita.in_pausa


def test_due_cambi_nella_stessa_partita_dicono_la_forma_della_partita(finestra, sentiti, voce_finta, monkeypatch):
    """Dopo un primo cambio a partita in corso le impostazioni hanno già la forma nuova, e la partita
    tiene la sua: il secondo cambio e l'annullamento dicono la forma della partita, non quella
    delle impostazioni."""
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    assert partita.proiettile == "seno"
    _scelta_del_proiettile(monkeypatch, (1,))
    finestra._esegui(finestra.cambia_proiettile)
    assert voce_finta[-1] == "Il proiettile suonerà in triangolo dalla prossima partita: quella in corso resta in seno."
    _scelta_del_proiettile(monkeypatch, (2,))
    finestra._esegui(finestra.cambia_proiettile)
    assert partita.proiettile == "seno" and finestra.impostazioni["proiettile"] == "sega_morbida"
    assert voce_finta[-1] == "Il proiettile suonerà in sega morbida dalla prossima partita: quella in corso resta in seno."
    _scelta_del_proiettile(monkeypatch, (0,))
    finestra._esegui(finestra.cambia_proiettile)
    assert voce_finta[-1] == "Il proiettile suonerà in seno dalla prossima partita: quella in corso resta in seno."
    # Annulla, con la forma delle impostazioni diversa da quella della partita: si dicono tutte e due.
    _scelta_del_proiettile(monkeypatch, (1,))
    finestra._esegui(finestra.cambia_proiettile)
    _scelta_del_proiettile(monkeypatch, (2,), conferma=False)
    finestra._esegui(finestra.cambia_proiettile)
    assert sentiti[-1] == "annullato" and finestra.impostazioni["proiettile"] == "triangolo"
    assert voce_finta[-1] == "Il proiettile suonerà in triangolo dalla prossima partita: quella in corso resta in seno."
    # Finita la partita, la forma della prossima è una sola.
    partita.fase = "finita"
    _scelta_del_proiettile(monkeypatch, (2,), conferma=False)
    finestra._esegui(finestra.cambia_proiettile)
    assert voce_finta[-1] == "Il proiettile resta in triangolo."
    assert finestra.statistiche.dati["conti"]["proiettile_cambiato"] == 4


def test_il_proiettile_annullato_o_uguale_resta(finestra, sentiti, monkeypatch):
    visti = _spia_del_motore(finestra, monkeypatch)
    _scelta_del_proiettile(monkeypatch, (1,), conferma=False)
    finestra._esegui(finestra.cambia_proiettile)
    # Annulla dopo un esempio: l'esempio si ferma, la forma resta, niente si salva né si conta.
    assert [passo[0] for passo in visti] == ["aggiungi", "ferma", "ferma"] and visti[-1][2] is None
    assert sentiti == ["proiettile", "annullato"]
    assert finestra.impostazioni["proiettile"] == costanti.PROIETTILE_FORMA and finestra.partita.proiettili == [costanti.PROIETTILE_FORMA]
    assert finestra.vista.GetValue().endswith("\nIl proiettile resta in seno.")
    # OK sulla forma che c'era già: confermato, e la forma resta, senza un cambio da contare.
    _scelta_del_proiettile(monkeypatch, ())
    finestra._esegui(finestra.cambia_proiettile)
    assert sentiti[2:] == ["proiettile", "confermato"] and finestra.partita.proiettili == [costanti.PROIETTILE_FORMA]
    assert finestra.vista.GetValue().endswith("\nIl proiettile resta in seno.\nIl proiettile resta in seno.")
    assert finestra.statistiche.dati["conti"]["proiettile_cambiato"] == 0


def test_il_proiettile_non_salvato_vale_fino_all_uscita(finestra, sentiti, monkeypatch):
    monkeypatch.setattr(impostazioni, "salva", lambda dati, percorso=None: False)
    _scelta_del_proiettile(monkeypatch, (1,))
    finestra._esegui(finestra.cambia_proiettile)
    assert sentiti == ["proiettile", "salvataggio_non_riuscito"]
    assert finestra.impostazioni["proiettile"] == "triangolo" and finestra.partita.proiettili[-1] == "triangolo"
    righe = finestra.vista.GetValue().splitlines()
    assert righe[-2:] == ["Le impostazioni non si sono potute salvare: valgono fino all'uscita.", "Il proiettile adesso suona in triangolo: lo senti dalla prossima partita."]


def test_la_forma_salvata_arriva_alla_partita_all_apertura(app_wx, sentiti, messaggi, senza_modali):
    with open(percorsi.percorso_dati(costanti.FILE_IMPOSTAZIONI), "w", encoding="utf-8") as file:
        json.dump({"proiettile": "sega_morbida"}, file)
    finestra = _crea()
    try:
        assert finestra.impostazioni["proiettile"] == "sega_morbida" and finestra.partita.proiettili == ["sega_morbida"]
    finally:
        _chiudi(finestra)


# Gli Ascolti (D23).

FRASE_DELLA_PRIMA_APERTURA = modulo_finestra.frase_della_prima_apertura()


def _voci_del_menu(finestra, etichetta):
    """Le voci dei menu con quell'etichetta: il titolo del menu e l'identificativo della voce."""
    barra = finestra.GetMenuBar()
    trovate = []
    for i in range(barra.GetMenuCount()):
        for voce in barra.GetMenu(i).GetMenuItems():
            if not voce.IsSeparator() and voce.GetItemLabelText() == etichetta:
                trovate.append((barra.GetMenuLabelText(i), voce.GetId()))
    return trovate


def test_gli_ascolti_dal_menu_aiuto_fermano_la_partita_e_alla_chiusura_tacciono(finestra, sentiti, monkeypatch):
    finestra._esegui(finestra.nuova_partita)
    partita = finestra.partita
    # Dal terzo collaudo, il 20 novembre 2026, la voce Ascolti sta soltanto nel menu Aiuto, con la sua a; qui si apre due volte di fila.
    voci = _voci_del_menu(finestra, "Ascolti...")
    assert [menu for menu, _identificativo in voci] == ["Aiuto"]
    aiuto = finestra.GetMenuBar().GetMenu(finestra.GetMenuBar().FindMenu("Aiuto"))
    assert aiuto.FindItemById(voci[0][1]).GetItemLabel() == "&Ascolti..."
    voci = voci * 2
    motore = finestra.motore
    fermate = []
    ferma = motore.ferma

    def spia(voce, campione=None, sfumatura=costanti.SFUMATURA_STOP):
        fermate.append((voce, campione))
        return ferma(voce, campione, sfumatura)

    monkeypatch.setattr(motore, "ferma", spia)
    aperti = []

    def ascolta(dialogo):
        ascolto = dialogo.ascolta()
        aperti.append((partita.in_pausa, partita.automatica, dialogo.regia, ascolto, len(fermate)))
        return wx.ID_OK

    monkeypatch.setattr(dialoghi.Ascolti, "ShowModal", ascolta)
    for _menu, identificativo in voci:
        evento = wx.CommandEvent(wx.wxEVT_MENU, identificativo)
        evento.SetEventObject(finestra)
        finestra.GetEventHandler().ProcessEvent(evento)
        # Chiusi gli Ascolti, la partita riprende da sola (D14).
        wx.Yield()
        assert not partita.in_pausa
    assert len(aperti) == 2
    for in_pausa, automatica, regia, ascolto, prima in aperti:
        # Con gli Ascolti aperti la partita è ferma, come con ogni dialogo, e gli Ascolti suonano sul motore del gioco.
        assert in_pausa and automatica and regia.motore is motore and regia.proiettile == finestra.impostazioni["proiettile"]
        assert ascolto.passo == "stereo" and ascolto.voci
        # Alla chiusura ogni voce degli Ascolti tace: una fermata subito, dopo quelle che il passo aveva messo in programma.
        subito = {id(voce) for voce, campione in fermate[prima:] if campione is None}
        assert {id(voce) for voce in ascolto.voci} <= subito
    assert sentiti.count("ascolti") == 2
    conti = finestra.statistiche.dati["conti"]
    assert conti["ascolti_aperti"] == 2 and conti["passi_ascoltati"] == 2 and conti["ascolti_finiti"] == 0 and conti["ascolti_d_avvio"] == 0


def test_gli_ascolti_letti_dalla_sintesi_e_seguiti_fino_in_fondo(finestra, voce_finta, monkeypatch):
    letti, passi = [], []

    def fino_in_fondo(dialogo):
        passi.extend(dialogo.passi)
        dialogo.seguente()
        letti.append(voce_finta[-1])
        dialogo.indice = len(dialogo.passi) - 1
        dialogo.seguente()
        return dialogo.GetReturnCode()

    monkeypatch.setattr(dialoghi.Ascolti, "ShowModal", fino_in_fondo)
    finestra._esegui(finestra.ascolti)
    # Il passo nuovo va alla sintesi, perché la lettura dei resoconti è accesa. Dalla 2.0.0 i passi sono quattordici, col tempo della
    # mira, e il secondo gira nel mezzo della cupola (D28).
    assert len(passi) == 14 and passi[1].nome == "orizzontale"
    assert letti and letti[0].startswith("Passo 2 di 14. Il mezzo della cupola, da un lato all'altro.")
    conti = finestra.statistiche.dati["conti"]
    assert conti["ascolti_aperti"] == 1 and conti["ascolti_finiti"] == 1 and conti["passi_ascoltati"] == 0


def test_alla_prima_apertura_gli_ascolti_si_aprono_da_soli_dopo_i_suoni_d_avvio(app_wx, sentiti, messaggi, senza_modali, monkeypatch):
    """D23, da Gabriele: con zero partite cominciate gli Ascolti si aprono da soli, una volta per avvio, finiti i suoni dell'avvio e
    l'avviso dei dati rovinati, a finestra attiva; la vista lo dice, e intanto la proposta dell'aggiornamento aspetta."""
    with open(percorsi.percorso_dati(costanti.FILE_IMPOSTAZIONI), "w", encoding="utf-8") as file:
        file.write("{ rovinato")
    motore, flusso = motore_finto()
    f = _crea(motore=motore)
    try:
        monkeypatch.setattr(f, "IsActive", lambda: True)
        aperti = []
        monkeypatch.setattr(dialoghi.Ascolti, "ShowModal", lambda dialogo: aperti.append(f.aspetta_i_dialoghi_d_avvio(0)) or wx.ID_OK)
        f.suoni_d_avvio()
        assert sentiti[-2:] == ["avvio", "dati_rovinati"]
        assert f.ascolti_alla_prima_apertura() and not f.aspetta_i_dialoghi_d_avvio(0)
        # Finché suonano l'avvio e l'avviso dei dati rovinati, gli Ascolti aspettano.
        f._al_battito()
        wx.Yield()
        assert aperti == []
        flusso.avanza_fino(f._fine_dei_suoni_d_avvio + round(LATENZA_FINTA * motore.fs) + motore.blocco)
        f._al_battito()
        wx.Yield()
        # Aperti una volta, e con loro aperti la strada dell'aggiornamento è chiusa; chiusi, è libera.
        assert aperti == [False] and f.aspetta_i_dialoghi_d_avvio(0)
        assert sentiti.count("ascolti") == 1
        assert f.vista.GetValue().endswith("\n" + FRASE_DELLA_PRIMA_APERTURA) and _riga_del_cursore(f.vista) == (FRASE_DELLA_PRIMA_APERTURA, 0)
        conti = f.statistiche.dati["conti"]
        assert conti["ascolti_d_avvio"] == 1 and conti["ascolti_aperti"] == 1
        f._al_battito()
        wx.Yield()
        assert aperti == [False]
    finally:
        _chiudi(f)


def test_con_una_partita_cominciata_gli_ascolti_non_si_aprono_da_soli(finestra, sentiti, monkeypatch):
    aperti = []
    monkeypatch.setattr(dialoghi.Ascolti, "ShowModal", lambda dialogo: aperti.append(dialogo) or wx.ID_OK)
    # Le statistiche contano una partita: gli Ascolti non aspettano, e nemmeno l'aggiornamento.
    finestra.statistiche.conta("partite_iniziate")
    finestra.suoni_d_avvio()
    assert not finestra.ascolti_alla_prima_apertura() and finestra.aspetta_i_dialoghi_d_avvio(0)
    finestra._al_battito()
    wx.Yield()
    # Senza partite, ma con una partita cominciata con lo spazio prima che si aprano: non si aprono più.
    finestra.statistiche.dati["conti"]["partite_iniziate"] = 0
    assert finestra.ascolti_alla_prima_apertura()
    finestra._esegui(finestra.nuova_partita)
    finestra._al_battito()
    wx.Yield()
    assert aperti == [] and finestra.aspetta_i_dialoghi_d_avvio(0) and not finestra._ascolti_d_avvio
    assert "ascolti" not in sentiti and FRASE_DELLA_PRIMA_APERTURA not in finestra.vista.GetValue()


def test_la_proposta_dell_aggiornamento_arriva_dopo_gli_ascolti_della_prima_apertura(app_wx, sentiti, messaggi, senza_modali, monkeypatch):
    """L'ordine dei dialoghi dell'avvio: prima gli Ascolti della prima apertura, poi la proposta dell'aggiornamento, che li aspetta."""
    import GBUtils

    motore, flusso = motore_finto()
    f = _crea(motore=motore)
    filo = None
    try:
        monkeypatch.setattr(f, "IsActive", lambda: True)
        ordine = []
        monkeypatch.setattr(dialoghi.Ascolti, "ShowModal", lambda dialogo: ordine.append("ascolti") or wx.ID_OK)
        monkeypatch.setattr(dialoghi.Aggiornamento, "ShowModal", lambda dialogo: ordine.append("aggiornamento") or wx.ID_NO)
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(GBUtils, "gestisci_aggiornamento", lambda app, versione, api, **opzioni: opzioni["proponi"](versione, "9.9.9", "Note.", attesa=None))
        f.suoni_d_avvio()
        assert f.ascolti_alla_prima_apertura()
        filo = aggiornamento.controlla(f)
        # Il filo aspetta gli Ascolti: la proposta non arriva, nemmeno dando a wx il tempo di portarla.
        for _giro in range(5):
            wx.Yield()
            time.sleep(0.01)
        assert ordine == [] and filo.is_alive()
        flusso.avanza_fino(f._fine_dei_suoni_d_avvio + round(LATENZA_FINTA * motore.fs) + motore.blocco)
        f._al_battito()
        for _giro in range(300):
            wx.Yield()
            if len(ordine) == 2:
                break
            time.sleep(0.01)
        assert ordine == ["ascolti", "aggiornamento"]
        filo.join(5)
        assert not filo.is_alive()
    finally:
        _chiudi(f)


# La scheda audio (D20).

SCELTA_CUFFIE = {"dispositivo": CUFFIE, "interfaccia": "Windows WASAPI"}
SCELTA_DIGITALE = {"dispositivo": DIGITALE, "interfaccia": "Windows WASAPI"}
SCELTA_ASIO = {"dispositivo": ASIO, "interfaccia": "ASIO"}
# Le frasi della vista, con la latenza del flusso finto, 20 millesimi.
SUL_DIGITALE = "Il suono esce da Realtek Digital Output (Realtek(R) Audio), WASAPI, con una latenza di 20 millesimi di secondo."
SUGLI_ALTOPARLANTI = "Il suono esce da Altoparlanti (Realtek(R) Audio), WASAPI, con una latenza di 20 millesimi di secondo."
SULLA_FINTA = "Il suono esce da finta, con una latenza di 20 millesimi di secondo."
MUTO = "Non trovo una scheda audio che si apra: Ciechino funziona lo stesso, ma senza suono."


def apri_con_ripiego(callback, fs, blocco, scelta=None, riprova=False, esito=None):
    """Come apri_flusso, ma finta: le cuffie mancano, l'ASIO c'è ma non si apre, e allora suonano gli altoparlanti, la
    scelta automatica; le altre schede scelte si aprono, come col flusso finto di apri_finto."""
    if scelta and scelta["dispositivo"] in (CUFFIE, ASIO):
        esito.update(mancante=scelta["dispositivo"] == CUFFIE, non_si_apre=scelta["dispositivo"] == ASIO, quale="automatica",
                     errore=f"scheda scelta {scelta['dispositivo']}, non si apre")
        flusso = FlussoFinto(callback, fs, blocco)
        flusso.device, flusso.interfaccia = ALTOPARLANTI, "Windows WASAPI"
        flusso.start()
        return flusso
    return apri_finto(callback, fs, blocco, scelta=scelta, riprova=riprova)


def _scrivi_la_scheda(scelta):
    with open(percorsi.percorso_dati(costanti.FILE_IMPOSTAZIONI), "w", encoding="utf-8") as file:
        json.dump({"scheda": scelta}, file)


def _con_la_scheda(monkeypatch, apri=apri_finto, scelta=None):
    """La finestra col motore del flusso finto, che apre le schede con apri e nasce con la scelta, attiva come quando si gioca."""
    f = _crea(motore=Motore(apri=apri, scelta=scelta))
    monkeypatch.setattr(f, "IsActive", lambda: True)
    return f


class Schede:
    """Il dialogo delle schede al posto di quello vero: annota ciò che mostra e com'è la partita, poi sceglie la riga
    della scheda indicata, o l'automatica con {}, e preme OK; con None preme Annulla."""

    def __init__(self, monkeypatch, finestra, *scelte):
        self.finestra = finestra
        self.scelte = list(scelte)
        self.visti = []
        # Una funzione, perché diventi un metodo del dialogo e riceva il dialogo stesso.
        monkeypatch.setattr(dialoghi.SceltaScheda, "ShowModal", lambda dialogo: self(dialogo))

    def __call__(self, dialogo):
        partita, motore = self.finestra.partita, self.finestra.motore
        etichette = [c.GetLabel() for c in dialogo.pannello.GetChildren() if isinstance(c, wx.StaticText)]
        self.visti.append({"righe": list(dialogo.righe), "scelta": dialogo.elenco.GetSelection(), "notizie": etichette[1:-1],
                           "in_pausa": getattr(partita, "in_pausa", None), "muto": motore.muto, "flusso": motore.flusso})
        scelta = self.scelte.pop(0)
        if scelta is None:
            return wx.ID_CANCEL
        indice = 0 if not scelta else next(n for n, u in enumerate(dialogo.uscite, 1) if schede.da_salvare(u) == scelta)
        dialogo.elenco.SetSelection(indice)
        dialogo.conferma()
        return dialogo.GetReturnCode()


def test_il_motore_nasce_con_la_scheda_delle_impostazioni(app_wx, sentiti, messaggi, senza_modali, voce_finta):
    """La scheda salvata arriva al motore che la finestra fa nascere; qui non si apre, perché il conftest tiene le casse
    mute, e lo dicono la vista, il suono e la sintesi, senza cancellare la scelta."""
    _scrivi_la_scheda(SCELTA_CUFFIE)
    f = _crea()
    try:
        assert f.impostazioni["scheda"] == SCELTA_CUFFIE and f.motore.scelta == SCELTA_CUFFIE
        perche = f"La scheda audio scelta, {CUFFIE}, WASAPI, non si apre."
        riga = f"{perche} {MUTO}"
        # Nella vista una frase per riga, il perché e poi il dove; alla sintesi una frase sola.
        assert f.vista.GetValue().splitlines()[5:7] == [perche, MUTO]
        f.suoni_d_avvio()
        assert sentiti == ["avvio", "scheda_non_aperta"] and voce_finta == [riga]
        assert sentiti.inizi[-1] >= CAMPIONI_FINTI + round(modulo_finestra.PAUSA_DOPO_L_AVVIO * f.motore.fs)
        assert f.statistiche.dati["conti"]["scheda_non_aperta"] == 1 and f.impostazioni["scheda"] == SCELTA_CUFFIE
    finally:
        _chiudi(f)
    # Senza scelta, nessuna notizia della scheda, né suono né sintesi.
    _scrivi_la_scheda({})
    f = _crea()
    try:
        assert f.motore.scelta == {} and f.vista.GetValue().splitlines()[5] == MUTO
        sentiti.clear()
        voce_finta.clear()
        f.suoni_d_avvio()
        assert sentiti == ["avvio"] and voce_finta == [] and f.statistiche.dati["conti"]["scheda_non_aperta"] == 1
    finally:
        _chiudi(f)


@pytest.mark.parametrize(("scelta", "perche"), [(SCELTA_CUFFIE, "non c'è"), (SCELTA_ASIO, "non si apre")])
def test_la_scheda_scelta_che_manca_suona_sull_automatica(app_wx, sentiti, messaggi, senza_modali, voce_finta, monkeypatch, scelta, perche):
    f = _con_la_scheda(monkeypatch, apri_con_ripiego, scelta)
    try:
        nome = schede.nome_della_scheda(scelta)
        prima = f"La scheda audio scelta, {nome}, {perche}: Ciechino suona su quella automatica."
        riga = f"{prima} {SUGLI_ALTOPARLANTI}"
        righe = f.vista.GetValue().splitlines()
        assert righe[righe.index(prima) + 1] == SUGLI_ALTOPARLANTI
        f.crash_precedente = "C:\\Ciechino\\Ciechino - Crash precedente.log"
        f.suoni_d_avvio()
        # Le notizie in fila, ognuna staccata dalla precedente.
        assert sentiti == ["avvio", "crash_precedente", "scheda_non_aperta"] and voce_finta == [riga]
        stacco = round(modulo_finestra.PAUSA_DOPO_L_AVVIO * f.motore.fs)
        assert sentiti.inizi[2] == sentiti.inizi[1] + CAMPIONI_FINTI + stacco
        assert f.statistiche.dati["conti"]["scheda_non_aperta"] == 1
    finally:
        _chiudi(f)


def test_ctrl_maiusc_x_resta_ai_menu(finestra):
    """Il gancio dei tasti prende soltanto lo spazio, F5 e F7 senza modificatori: Ctrl+Maiusc+X va avanti, al tasto rapido del menu."""
    evento = _tasto(finestra, ord("X"), controllo=True, maiuscolo=True)
    assert evento.GetSkipped() and finestra.partita.chiamate == []
    voce = finestra.voci_del_menu[finestra.scegli_la_scheda]
    assert voce.GetAccel().ToString() == "Ctrl+Shift+X" and voce.GetItemLabelText() == "Scheda audio..."


def test_la_scheda_si_sceglie_dopo_la_rilettura(app_wx, sentiti, messaggi, senza_modali, voce_finta, schede_finte, monkeypatch):
    """Ctrl+Maiusc+X: la partita si ferma, il motore rilegge le schede, così ci sono le cuffie accese dopo l'avvio, poi
    l'elenco con la scheda in uso. La scheda scelta suona subito, si salva e si conta; la vista e la sintesi dicono dove."""
    f = _con_la_scheda(monkeypatch)
    try:
        f._esegui(f.nuova_partita)
        schede_finte.arrivate = [CUFFIE_FINTE]
        dialogo = Schede(monkeypatch, f, SCELTA_DIGITALE)
        f._esegui(f.scegli_la_scheda)
        visto = dialogo.visti[0]
        # La rilettura prima dell'elenco, e la partita ferma, come con ogni dialogo (D14).
        chiamate = [chiamata[0] for chiamata in schede_finte.chiamate]
        assert "rileggi" in chiamate and chiamate.index("rileggi") < chiamate.index("elenco")
        assert visto["in_pausa"] is True
        assert visto["righe"][0] == "Automatica, la scheda di Windows, in uso: finta" and visto["scelta"] == 0
        assert f"{CUFFIE}, WASAPI, 3 ms" in visto["righe"] and visto["notizie"] == [SULLA_FINTA]
        # Le righe in ordine di latenza, come le dà schede.uscite().
        assert visto["righe"][1:] == [dialoghi.riga_della_scheda(u) for u in schede.uscite()]
        assert sentiti == ["scheda_audio", "confermato"]
        assert f.motore.scelta == SCELTA_DIGITALE and f.motore.flusso.scelta == SCELTA_DIGITALE
        assert f.impostazioni["scheda"] == SCELTA_DIGITALE and _leggi(costanti.FILE_IMPOSTAZIONI)["scheda"] == SCELTA_DIGITALE
        assert f.vista.GetValue().endswith("\n" + SUL_DIGITALE) and voce_finta[-1] == SUL_DIGITALE
        assert f.statistiche.dati["conti"]["scheda_cambiata"] == 1 and f.statistiche.dati["conti"]["scheda_non_aperta"] == 0
        wx.Yield()
        assert not f.partita.in_pausa
        # Di nuovo: la riga della scheda in uso lo dice, ed è già scelta; OK lì non cambia niente.
        dialogo = Schede(monkeypatch, f, SCELTA_DIGITALE, None, {})
        f._esegui(f.scegli_la_scheda)
        visto = dialogo.visti[0]
        in_uso = [n for n, riga in enumerate(visto["righe"]) if riga.endswith(", in uso")]
        assert in_uso == [visto["scelta"]] and visto["righe"][visto["scelta"]].startswith("Realtek Digital Output (Realtek(R) Audio), WASAPI, ")
        assert visto["righe"][0] == "Automatica, la scheda di Windows"
        assert sentiti[-2:] == ["scheda_audio", "confermato"] and voce_finta[-1] == "La scelta della scheda audio resta quella di prima. " + SUL_DIGITALE
        # Annulla lascia tutto com'era.
        f._esegui(f.scegli_la_scheda)
        assert sentiti[-2:] == ["scheda_audio", "annullato"] and voce_finta[-1] == "La scelta della scheda audio non cambia. " + SUL_DIGITALE
        assert f.motore.scelta == SCELTA_DIGITALE and _leggi(costanti.FILE_IMPOSTAZIONI)["scheda"] == SCELTA_DIGITALE
        # L'automatica.
        f._esegui(f.scegli_la_scheda)
        assert sentiti[-2:] == ["scheda_audio", "confermato"] and f.motore.scelta == {}
        assert voce_finta[-1] == "Ciechino suona sulla scheda automatica, quella di Windows. " + SULLA_FINTA
        assert _leggi(costanti.FILE_IMPOSTAZIONI)["scheda"] == {} and f.statistiche.dati["conti"]["scheda_cambiata"] == 2
        assert dialogo.scelte == []
    finally:
        _chiudi(f)


def test_la_scheda_scelta_che_non_si_apre_resta_salvata(app_wx, sentiti, messaggi, senza_modali, voce_finta, monkeypatch):
    f = _con_la_scheda(monkeypatch, apri_con_ripiego)
    try:
        dialogo = Schede(monkeypatch, f, SCELTA_ASIO, SCELTA_ASIO)
        f._esegui(f.scegli_la_scheda)
        riga = f"La scheda audio scelta, Realtek ASIO, ASIO, non si apre: Ciechino suona su quella automatica. {SUGLI_ALTOPARLANTI}"
        assert sentiti == ["scheda_audio", "scheda_non_aperta"] and voce_finta[-1] == riga and f.vista.GetValue().endswith("\n" + riga)
        # La scelta resta, nel motore e sul disco: le cuffie, o la scheda occupata, possono tornare.
        assert f.motore.scelta == SCELTA_ASIO and _leggi(costanti.FILE_IMPOSTAZIONI)["scheda"] == SCELTA_ASIO
        conti = f.statistiche.dati["conti"]
        assert (conti["scheda_cambiata"], conti["scheda_non_aperta"]) == (1, 1)
        # Riaprendo l'elenco, la riga in uso è l'automatica; l'ASIO di nuovo, che non suona, si riprova, ma la scelta non cambia.
        f._esegui(f.scegli_la_scheda)
        visto = dialogo.visti[1]
        assert visto["righe"][0] == f"Automatica, la scheda di Windows, in uso: {ALTOPARLANTI}, WASAPI" and visto["scelta"] == 0
        # Il dialogo dice anche che la scheda scelta, appena riprovata dalla rilettura, non si apre.
        assert visto["notizie"] == [riga]
        assert sentiti[-2:] == ["scheda_audio", "scheda_non_aperta"] and voce_finta[-1] == riga
        assert (conti["scheda_cambiata"], conti["scheda_non_aperta"]) == (1, 2)
    finally:
        _chiudi(f)


def test_dopo_il_flusso_perso_la_scelta_della_scheda_ridà_il_suono(app_wx, sentiti, messaggi, senza_modali, voce_finta, monkeypatch):
    f = _con_la_scheda(monkeypatch, scelta=SCELTA_DIGITALE)
    try:
        perso = "La scheda audio non risponde più, forse si è staccata: Ciechino continua senza suono. Per riavere il suono scegli la scheda con Ctrl+Maiusc+X."
        f.motore.flusso.active = False
        f._al_battito()
        assert f.motore.muto and f.motore.flusso is None and voce_finta[-1] == perso
        dialogo = Schede(monkeypatch, f, None)
        f._esegui(f.scegli_la_scheda)
        # La rilettura ha già ridato il suono: il dialogo si apre col motore sulla scheda scelta, ed è lei la riga in uso.
        visto = dialogo.visti[0]
        assert not visto["muto"] and visto["flusso"] is f.motore.flusso and visto["notizie"] == [SUL_DIGITALE]
        assert visto["righe"][visto["scelta"]].endswith(", in uso")
        assert voce_finta[-1] == "La scelta della scheda audio non cambia. " + SUL_DIGITALE and f.motore.stato()["aperto"]
        # Il flusso che si perde di nuovo si dice di nuovo.
        f.motore.flusso.active = False
        f._al_battito()
        assert voce_finta[-1] == perso and voce_finta.count(perso) == 2
    finally:
        _chiudi(f)


def test_il_flusso_perso_a_elenco_aperto_torna_con_ok_sulla_stessa_riga(app_wx, sentiti, messaggi, senza_modali, voce_finta, monkeypatch):
    """Il battito gira anche dentro il dialogo: se la guardia del motore perde il flusso mentre l'elenco è aperto, OK sulla
    riga che era in uso, con la stessa scelta, riapre la scheda invece di lasciare il gioco muto; niente si salva né si conta."""
    _scrivi_la_scheda(SCELTA_DIGITALE)
    f = _con_la_scheda(monkeypatch, scelta=SCELTA_DIGITALE)
    try:
        perso = "La scheda audio non risponde più, forse si è staccata: Ciechino continua senza suono. Per riavere il suono scegli la scheda con Ctrl+Maiusc+X."
        visti = []

        def sceglie(dialogo):
            # La scheda si stacca a elenco aperto, e il battito, che wx smista anche qui, lo vede.
            f.motore.flusso.active = False
            f._al_battito()
            selezionata = dialogo.elenco.GetSelection()
            visti.append((dialogo.righe[selezionata], f.motore.muto))
            dialogo.conferma()
            return dialogo.GetReturnCode()

        monkeypatch.setattr(dialoghi.SceltaScheda, "ShowModal", sceglie)
        f._esegui(f.scegli_la_scheda)
        riga, muto = visti[0]
        assert riga.startswith("Realtek Digital Output (Realtek(R) Audio), WASAPI, ") and riga.endswith(", in uso") and muto
        assert perso in voce_finta
        stato = f.motore.stato()
        assert stato["aperto"] and not stato["perso"] and not f.motore.muto and f.motore.scelta == SCELTA_DIGITALE
        assert sentiti[-2:] == ["scheda_audio", "confermato"] and voce_finta[-1] == SUL_DIGITALE
        assert f.statistiche.dati["conti"]["scheda_cambiata"] == 0 and _leggi(costanti.FILE_IMPOSTAZIONI)["scheda"] == SCELTA_DIGITALE
    finally:
        _chiudi(f)


def test_l_elenco_che_non_si_legge_e_la_rilettura_che_non_riesce(app_wx, sentiti, messaggi, senza_modali, schede_finte, monkeypatch):
    import registro

    righe = []
    monkeypatch.setattr(registro, "scrivi", righe.append)
    f = _con_la_scheda(monkeypatch)
    try:
        schede_finte.rotta = RuntimeError("PortAudio non si riapre")
        dialogo = Schede(monkeypatch, f, None, {})
        f._esegui(f.scegli_la_scheda)
        visto = dialogo.visti[0]
        assert visto["notizie"] == [SULLA_FINTA, "Non ho potuto rileggere le schede audio: quelle accese da poco, come delle cuffie Bluetooth, possono mancare."]
        assert any(riga.startswith("Motore audio, cambio di scheda: rilettura: RuntimeError: PortAudio non si riapre") for riga in righe)
        schede_finte.rotta = None

        def elenco_rotto():
            raise OSError("niente PortAudio")

        monkeypatch.setattr(schede, "uscite", elenco_rotto)
        f._esegui(f.scegli_la_scheda)
        visto = dialogo.visti[1]
        assert visto["righe"] == ["Automatica, la scheda di Windows, in uso: finta"]
        assert visto["notizie"][-1] == "Non riesco a leggere l'elenco delle schede audio: niente PortAudio. Resta la scelta automatica."
        # OK sull'automatica, che era già in uso e già scelta: niente cambia.
        assert sentiti[-1] == "confermato" and f.statistiche.dati["conti"]["scheda_cambiata"] == 0
    finally:
        _chiudi(f)


def test_il_suono_dell_errore_fatale_esce_dalla_scheda_scelta(monkeypatch):
    import Ciechino
    import motore_audio

    _scrivi_la_scheda(SCELTA_CUFFIE)
    nati = []

    class MotoreFinto:
        muto = True

        def __init__(self, scelta=None):
            nati.append(scelta)

        def avvia(self):
            return False

        def chiudi(self):
            nati.append("chiuso")

    monkeypatch.setattr(motore_audio, "Motore", MotoreFinto)
    monkeypatch.setattr(suoni.Suoni, "suona", lambda self, evento, *argomenti, **opzioni: None)
    Ciechino._suona_l_errore()
    assert nati == [SCELTA_CUFFIE, "chiuso"]


# L'aiuto.

def test_la_guida_nasce_dalla_tabella_dei_menu(finestra, sentiti):
    finestra._esegui(finestra.guida)
    testo = finestra.vista.GetValue()
    assert "\nGuida ai comandi di Ciechino." in testo
    righe = testo.split("\nGuida ai comandi di Ciechino.")[1].splitlines()
    for titolo, comandi in finestra.voci_guida():
        assert any(riga.startswith(f"Il menu {titolo}, con {len(comandi)} comand") for riga in righe), titolo
        for voce, tasto in comandi:
            assert (f"{voce}: {tasto}." if tasto else f"{voce}: dal menu, senza tasto rapido.") in righe, voce
    assert "Manuale: Maiusc+F1." in righe and "Volume: dal menu, senza tasto rapido." in righe
    # Le voci nuove del proiettile (D17) e della scheda audio (D20) entrano nella guida da sole, perché la guida nasce dalla tabella dei menu.
    assert "Il menu Impostazioni, con 5 comandi:" in righe and "Proiettile: dal menu, senza tasto rapido." in righe
    assert "Scheda audio: Ctrl+Maiusc+X." in righe
    # Gli Ascolti (D23), dal terzo collaudo, stanno soltanto nel menu Aiuto, e la guida li dice una volta, fra le sue voci.
    assert "Il menu Aiuto, con 8 comandi:" in righe and righe.count("Ascolti: dal menu, senza tasto rapido.") == 1
    aiuto = righe.index("Il menu Aiuto, con 8 comandi:")
    assert righe.index("Ascolti: dal menu, senza tasto rapido.") > aiuto > righe.index("Il menu Impostazioni, con 5 comandi:")
    assert any(riga.startswith("Spazio: ") for riga in righe) and any(riga.startswith("F5: ") for riga in righe)
    # Dalla 2.0.0 i tasti della mira manuale (D29) stanno fra i tasti fuori dai menu, dopo lo spazio, che non ferma più le scivolate,
    # e prima di Esc (D25), che annulla il tiro; poi F5, F6, che riporta il fuoco sulla Mira, e F7. I tasti dei valori muovono la
    # mira soltanto col fuoco sulla Mira, e la guida lo dice.
    # La prima riga è il resto di quella del titolo, vuoto.
    fuori = righe[1:righe.index(next(riga for riga in righe if riga.startswith("Il menu ")))]
    assert fuori == [
        "I tasti fuori dai menu:",
        "Spazio: comincia la partita e, mentre ascolti i bersagli, avvia la mira; nei livelli della palestra, dopo ogni tiro, fa ripartire la "
        "partita, che aspetta ferma.",
        "Frecce sinistra e destra: nella mira, col fuoco sulla Mira, muovono la direzione, finché le tieni giù.",
        "Frecce su e giù: nella mira, col fuoco sulla Mira, alzano e abbassano l'alzo, finché le tieni giù.",
        "Home e Fine: nella mira, col fuoco sulla Mira, abbassano e alzano la potenza, finché le tieni giù.",
        "Invio: nella mira fa partire il colpo coi tre valori di quel momento.",
        "Esc: durante la mira annulla il tiro: il colpo non parte, e il proiettile resta nella riserva; nelle finestre interne le chiude.",
        "F5: porta il fuoco nella vista della partita.",
        "F6: durante la mira riporta il fuoco sulla Mira, dove le frecce, Home e Fine muovono i valori.",
        "F7: porta il fuoco nella barra di stato.",
    ]
    guida = "\n".join(righe)
    assert "scivolat" not in guida and "tornano a cantare" not in guida
    assert _riga_del_cursore(finestra.vista) == ("Guida ai comandi di Ciechino.", 0)
    assert sentiti == ["guida"]


def test_manuale_novita_e_informazioni(finestra, sentiti, monkeypatch, tmp_path):
    import os

    aperti = []
    monkeypatch.setattr(os, "startfile", aperti.append, raising=False)
    risorse = tmp_path / "risorse"
    risorse.mkdir()
    monkeypatch.setattr(percorsi, "percorso_risorsa", lambda nome: str(risorse / nome))
    finestra._esegui(finestra.manuale)
    finestra._esegui(finestra.novita)
    assert sentiti == ["file_mancante", "file_mancante"] and aperti == []
    (risorse / "manuale.html").write_text("<html></html>", encoding="utf-8")
    (risorse / "CHANGELOG.md").write_text("# Novità\n\n## [0.1.0] - 2026-11-19\n\n- La **prima** versione.\n", encoding="utf-8")
    finestra._esegui(finestra.manuale)
    assert aperti == [str(risorse / "manuale.html")] and sentiti[-1] == "manuale"
    finestra._esegui(finestra.novita)
    assert finestra.vista.GetValue().endswith("\nNovità.\nVersione 0.1.0, giovedì 19 novembre 2026.\nLa prima versione.")
    assert _riga_del_cursore(finestra.vista) == ("Novità.", 0) and sentiti[-1] == "novita"
    finestra._esegui(finestra.informazioni)
    assert "GNU GPL" in finestra.vista.GetValue() and sentiti[-1] == "informazioni"


def test_il_caffe(finestra, sentiti, monkeypatch):
    import GBUtils

    import lingua

    chiesti, letti = [], []
    monkeypatch.setattr(GBUtils, "Donazione", lambda **opzioni: chiesti.append(opzioni) or "Offrimi un caffè.")
    monkeypatch.setattr(dialoghi.Caffe, "ShowModal", lambda dialogo: letti.append(dialogo.testo.GetValue()) or wx.ID_OK)
    finestra._esegui(finestra.caffe)
    assert chiesti == [{"lang": lingua.CODICE, "probabilita": 100, "stampa": False}] and letti == ["Offrimi un caffè."] and sentiti == ["caffe"]
    # Con una traduzione installata l'invito è nella lingua in cui Ciechino parla, non in italiano.
    monkeypatch.setattr(lingua, "CODICE", "en")
    finestra._esegui(finestra.caffe)
    assert chiesti[-1]["lang"] == "en"


def test_le_due_voci_del_pub(finestra, sentiti, monkeypatch):
    import GBUtils
    import GBwx

    mail = []
    monkeypatch.setattr(GBUtils, "informazioni_di_sistema", lambda separatore=".": [("windows", "Windows 11 Pro")])
    monkeypatch.setattr(GBwx, "finestra_mail", lambda genitore, titolo, destinatario, oggetto, corpo, spiegazione, passi, pulsanti, senza_posta="",
                        casella=None, **opzioni: mail.append((titolo, oggetto, corpo, casella)) or True)
    finestra._esegui(finestra.entra_al_pub)
    finestra._esegui(finestra.chiedi_aiuto_al_pub)
    assert mail[0][0] == "Entra al PUB, il gruppo di supporto"
    assert mail[1][0] == "Chiedi aiuto al PUB" and costanti.VERSIONE in mail[1][1] and "Windows 11 Pro" in mail[1][2]
    # Senza registri accanto al programma, niente casella.
    assert mail[1][3] is None
    assert sentiti == ["pub", "confermato", "pub_aiuto", "confermato"]
    assert finestra.statistiche.dati["conti"]["mail_al_pub_pronte"] == 2


# L'aggiornamento, l'istanza unica e gli errori.

def test_la_proposta_dell_aggiornamento(finestra, sentiti, monkeypatch):
    risposte = [wx.ID_YES, wx.ID_NO]
    monkeypatch.setattr(dialoghi.Aggiornamento, "ShowModal", lambda dialogo: risposte.pop(0))
    assert finestra.proponi_l_aggiornamento("0.1.0", "0.2.0", "## [0.2.0] - 2026-12-01\n- Novità.", 120) is True
    assert sentiti == ["aggiornamento_disponibile", "aggiornamento_scelto"]
    assert finestra.proponi_l_aggiornamento("0.1.0", "0.2.0", "", 120) is False
    assert sentiti[-1] == "aggiornamento_rimandato"
    monkeypatch.setattr(dialoghi.Aggiornamento, "ShowModal", lambda dialogo: (setattr(dialogo, "scaduta", True), wx.ID_NO)[-1])
    assert finestra.proponi_l_aggiornamento("0.1.0", "0.2.0", "", 120) is False
    assert sentiti[-1] == "aggiornamento_scaduto"


def test_gli_esiti_e_l_avanzamento_dell_aggiornamento(finestra, sentiti, messaggi):
    for frase, suono, icona in ((aggiornamento.FRASE_PRONTO, "aggiornamento_pronto", wx.ICON_INFORMATION),
                                (aggiornamento.FRASE_NON_RIUSCITO, "aggiornamento_non_riuscito", wx.ICON_ERROR),
                                ("Disponibile la versione 0.2.0, " + aggiornamento.FRASE_NON_PRONTO, "aggiornamento_non_pronto", wx.ICON_WARNING)):
        finestra.avvisa_dell_aggiornamento(frase)
        assert sentiti[-1] == suono and messaggi[-1][0] == frase and messaggi[-1][2] & icona
        assert finestra.vista.GetValue().endswith(frase)
    finestra.avanzamento_dell_aggiornamento(512, 1024)
    assert finestra.barra.GetValue().splitlines()[-1] == "Aggiornamento scaricato al 50%."
    # La finestra ha ciò che aggiornamento.py si aspetta.
    for metodo in ("proponi_l_aggiornamento", "avvisa_dell_aggiornamento", "avanzamento_dell_aggiornamento", "chiudi_per_aggiornare"):
        assert callable(getattr(finestra, metodo))
    assert finestra.chiusa is False


def test_primo_piano_ed_errore_imprevisto(finestra, sentiti, messaggi):
    finestra.porta_in_primo_piano()
    assert sentiti == ["primo_piano"]
    finestra.errore_imprevisto(ValueError, ValueError("qualcosa di storto"))
    assert sentiti[-1] == "errore" and "ValueError, qualcosa di storto" in messaggi[-1][0] and messaggi[-1][2] & wx.ICON_ERROR


# Le notizie del motore audio.

def test_le_notizie_del_motore_vanno_nel_registro_riassunte(finestra, voce_finta, monkeypatch):
    import registro
    from motore_audio import Notifica

    righe, controlli, coda = [], [], []
    monkeypatch.setattr(registro, "scrivi", righe.append)
    monkeypatch.setattr(finestra.motore, "controlla", lambda: controlli.append(True), raising=False)

    def notifiche():
        prese = list(coda)
        coda.clear()
        return prese

    monkeypatch.setattr(finestra.motore, "notifiche", notifiche)
    errore = Notifica("errore", 100, None, "ValueError: rotto")
    coda.extend([errore] * 3 + [Notifica("buco", 200)] * 5 + [Notifica("voce_finita", 300), Notifica("evento", 400)])
    finestra._al_battito()
    # A ogni battito il motore guarda il flusso; un errore va nel registro la prima volta, con la riga
    # che la notifica sa dire, e i buchi e le ripetizioni si contano.
    assert controlli == [True]
    assert righe == [errore.riga()] and righe[0].startswith("Errore nel filo dell'audio al campione 100")
    finestra._ultimo_riassunto -= modulo_finestra.INTERVALLO_DEI_RIASSUNTI
    finestra._al_battito()
    assert len(righe) == 3
    assert righe[1].startswith("Buchi nel flusso audio: 5 negli ultimi ") and righe[1].endswith(", 5 da quando Ciechino è aperto.")
    assert righe[2].startswith("L'errore del filo dell'audio in blocco si è ripetuto ancora 2 volte negli ultimi ") and righe[2].endswith(": ValueError: rotto")
    # Niente di nuovo, niente righe; lo stesso errore che torna si conta soltanto.
    finestra._ultimo_riassunto -= modulo_finestra.INTERVALLO_DEI_RIASSUNTI
    coda.append(Notifica("errore", 500, None, "ValueError: rotto"))
    finestra._al_battito()
    assert len(righe) == 4 and "si è ripetuto ancora 1 volta negli ultimi " in righe[3]
    # Il flusso perso: il registro, la vista e la sintesi lo dicono, una volta.
    testo = "La scheda audio non risponde più, forse si è staccata: Ciechino continua senza suono. Per riavere il suono scegli la scheda con Ctrl+Maiusc+X."
    coda.extend([Notifica("flusso_perso", 600, None, "flusso perso")] * 2)
    finestra._al_battito()
    assert righe[-1].startswith("Flusso audio perso al campione 600: flusso perso") and righe[-1].endswith(". Ciechino continua senza suono.")
    assert finestra.vista.GetValue().endswith("\n" + testo) and voce_finta == [testo]
    assert sum(1 for riga in righe if riga.startswith("Flusso audio perso")) == 1
    # All'uscita l'ultimo riassunto, anche prima dell'intervallo.
    coda.extend([Notifica("buco", 700)] * 2)
    finestra._al_battito()
    finestra.Close(force=True)
    assert righe[-1].startswith("Buchi nel flusso audio: 2 negli ultimi ") and righe[-1].endswith(", 7 da quando Ciechino è aperto.")


def test_una_voce_guasta_del_motore_finisce_nel_registro(app_wx, sentiti, messaggi, senza_modali, monkeypatch):
    """Il filo dell'audio non arriva alla rete degli errori: la voce guasta diventa una notifica, e il battito la scrive nel registro."""
    import registro
    from motore_audio import VoceTono

    righe = []
    monkeypatch.setattr(registro, "scrivi", righe.append)
    motore, flusso = motore_finto()
    f = _crea(motore=motore)
    try:
        def parametri(tempi_di_gioco, tempi_audio):
            raise ValueError("parametri guasti")

        motore.aggiungi(VoceTono(parametri, "seno", motore.prossimo_campione()))
        flusso.avanza(3)
        f._al_battito()
        errori = [riga for riga in righe if riga.startswith("Errore nel filo dell'audio al campione ")]
        assert len(errori) == 1 and "ValueError: parametri guasti" in errori[0]
    finally:
        _chiudi(f)


def test_il_motivo_del_motore_muto_va_nel_registro(app_wx, sentiti, messaggi, senza_modali, monkeypatch):
    import registro
    from motore_audio import Motore

    righe = []
    monkeypatch.setattr(registro, "scrivi", righe.append)

    def non_si_apre(callback, fs, blocco):
        raise OSError("scheda occupata")

    f = _crea(motore=Motore(apri=non_si_apre))
    try:
        assert any(riga.startswith("Motore audio: ") and "scheda occupata" in riga for riga in righe)
        assert "Non trovo una scheda audio che si apra" in f.vista.GetValue()
    finally:
        _chiudi(f)


# Il punto d'avvio.

def test_ciechino_lascia_il_posto_alla_copia_aperta(monkeypatch):
    import Ciechino
    import istanza
    import registro

    avviati = []
    monkeypatch.setattr(istanza, "manda", lambda: True)
    monkeypatch.setattr(registro, "avvia", lambda *a, **k: avviati.append(True))
    assert Ciechino.main([]) == 0 and avviati == []


def test_ciechino_registra_l_uscita_e_ferma_tutto(monkeypatch):
    import Ciechino
    import istanza
    import registro

    fatti = []

    class AscoltoFinto:
        attiva = True

        def __init__(self, mostra):
            fatti.append("ascolto")

        def ferma(self):
            fatti.append("ferma")

    monkeypatch.setattr(istanza, "manda", lambda: False)
    monkeypatch.setattr(istanza, "Ascolto", AscoltoFinto)
    monkeypatch.setattr(registro, "avvia", lambda *a, **k: fatti.append("registro") or None)
    monkeypatch.setattr(registro, "scrivi", lambda riga: fatti.append(riga))
    monkeypatch.setattr(registro, "chiudi", lambda: fatti.append("chiudi"))
    monkeypatch.setattr(Ciechino, "avvia_finestra", lambda crash, finestre: Ciechino.RIAPERTA)
    assert Ciechino.main([]) == 0
    assert fatti == ["ascolto", "registro", f"Ciechino {costanti.VERSIONE} si avvia.", Ciechino.ESITI[Ciechino.RIAPERTA][1], "ferma", "chiudi"]
    # Un errore che ferma Ciechino: l'ultima rete lo dice, e il codice è uno.
    fatali = []
    monkeypatch.setattr(Ciechino, "avvia_finestra", lambda crash, finestre: 1 / 0)
    monkeypatch.setattr(Ciechino, "_errore_fatale", lambda *eccezione: fatali.append(eccezione[0]))
    assert Ciechino.main([]) == 1 and fatali == [ZeroDivisionError] and fatti[-2:] == ["ferma", "chiudi"]


def test_una_copia_chiesta_durante_la_chiusura_si_apre_dopo(monkeypatch):
    """Una copia di Ciechino che si presenta quando questa si sta già chiudendo non resta senza risposta:
    all'uscita pulita main lancia la copia nuova, che aspetta l'uscita di questa."""
    import Ciechino
    import istanza
    import registro

    fatti, mostre, rimandate = [], [], []

    class AscoltoFinto:
        attiva = True

        def __init__(self, mostra):
            mostre.append(mostra)

        def ferma(self):
            fatti.append("ferma")

    class FinestraFinta:
        def porta_in_primo_piano(self):
            fatti.append("primo piano")

    monkeypatch.setattr(istanza, "manda", lambda: False)
    monkeypatch.setattr(istanza, "Ascolto", AscoltoFinto)
    monkeypatch.setattr(istanza, "riapri", lambda: fatti.append("riapri"))
    monkeypatch.setattr(registro, "avvia", lambda *a, **k: None)
    monkeypatch.setattr(registro, "scrivi", lambda riga: fatti.append(riga))
    monkeypatch.setattr(registro, "chiudi", lambda: fatti.append("chiudi"))
    monkeypatch.setattr(wx, "CallAfter", lambda funzione, *argomenti: rimandate.append((funzione, argomenti)))

    def si_chiude(crash, finestre, esito=Ciechino.APERTA):
        finestre.append(FinestraFinta())
        # Aperta: la copia nuova la porta in primo piano, nel filo della finestra.
        mostre[-1]()
        funzione, argomenti = rimandate.pop()
        funzione(*argomenti)
        # Una richiesta partita prima della chiusura e arrivata nel filo della finestra dopo: si rimanda.
        mostre[-1]()
        funzione, argomenti = rimandate.pop()
        finestre.in_chiusura.set()
        funzione(*argomenti)
        # Dalla chiusura in poi il filo dell'ascolto non tocca più la finestra.
        mostre[-1]()
        assert rimandate == [] and finestre.rimandata.is_set()
        return esito

    monkeypatch.setattr(Ciechino, "avvia_finestra", si_chiude)
    assert Ciechino.main([]) == 0
    assert fatti == [f"Ciechino {costanti.VERSIONE} si avvia.", "primo piano", Ciechino.ESITI[Ciechino.APERTA][1], "riapri",
                     "Una copia di Ciechino ha chiesto di aprirsi durante la chiusura: si apre appena questa è uscita.", "ferma", "chiudi"]
    # Chi si riapre per la lingua, o si aggiorna, non lancia un'altra copia.
    for esito in (Ciechino.RIAPERTA, Ciechino.AGGIORNATA):
        fatti.clear()
        monkeypatch.setattr(Ciechino, "avvia_finestra", lambda crash, finestre, esito=esito: si_chiude(crash, finestre, esito))
        assert Ciechino.main([]) == 0 and "riapri" not in fatti


def test_la_rete_per_gli_errori(monkeypatch):
    import Ciechino
    import registro

    problemi, chiamati = [], []
    monkeypatch.setattr(sys, "excepthook", lambda *a: chiamati.append("prima"))
    monkeypatch.setattr(threading, "excepthook", lambda argomenti: chiamati.append("prima nei fili"))
    monkeypatch.setattr(registro, "problema", lambda tipo, valore, traccia: problemi.append(tipo))
    monkeypatch.setattr(wx, "CallAfter", lambda funzione, *argomenti: chiamati.append((funzione, argomenti)))

    class FinestraFinta:
        def errore_imprevisto(self, tipo, valore):
            pass

    finestra = FinestraFinta()
    Ciechino._rete_per_gli_errori(finestra)
    errore = KeyError("x")
    sys.excepthook(KeyError, errore, None)
    assert problemi == [KeyError] and chiamati[0] == "prima" and chiamati[1] == (finestra.errore_imprevisto, (KeyError, errore))
    threading.excepthook(threading.ExceptHookArgs((OSError, OSError("disco"), None, None)))
    assert problemi[-1] is OSError and chiamati[-1] == "prima nei fili"
