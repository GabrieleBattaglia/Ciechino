"""
Le prove dei dialoghi di Ciechino, sul desktop nascosto del conftest.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di tests/test_finestre_interne.py di MESS. Ogni dialogo
di gui/dialoghi.py si apre con ShowModal, come lo apre la finestra, e dentro il suo ciclo si guarda che
sia massimizzato e si preme Esc, che deve chiuderlo: col codice del pulsante Chiudi, se c'è, altrimenti
come Annulla, o come Non adesso nella proposta dell'aggiornamento. Una rete di sicurezza chiude il
dialogo che Esc non ha chiuso, perché una prova che aspetta un tasto non finirebbe mai. Esc e Invio
sono l'evento wxEVT_CHAR_HOOK che il gancio dei tasti di wx crea per un tasto vero, rifatto dalla prova:
SendInput li manderebbe allo schermo vero. Nessun dialogo suona: ricevono un registratore al posto
della funzione dei suoni. I messaggi di wx sono sostituiti.
"""

import inspect
from types import SimpleNamespace

import pytest
import wx
from aiuti_audio import ALTOPARLANTI, ASIO, CUFFIE, CUFFIE_FINTE, DIGITALE, USCITE_FINTE

import costanti
import impostazioni
import schede
from gui import dialoghi

# Il codice con cui la rete di sicurezza chiude un dialogo che Esc non ha chiuso, dopo ATTESA millisecondi.
NON_CHIUSO = 4242
ATTESA = 3000


class Sentiti(list):
    """I suoni che un dialogo avrebbe fatto, in ordine."""

    def __call__(self, evento, *argomenti, **opzioni):
        self.append(evento)


class Messaggi(list):
    """I messaggi di wx, al posto di dialoghi.messaggio: si annotano, e rispondono OK."""

    def __call__(self, genitore, testo, titolo, stile=0):
        self.append((testo, titolo, stile))
        return wx.OK


class RegiaFinta:
    """La regia degli Ascolti vista dal dialogo (D23): i passi coi titoli e i testi, e le chiamate annotate; da_rivelare è il
    testo che rivelazione restituisce."""

    def __init__(self, quanti=3):
        self.passi = tuple(SimpleNamespace(nome=f"passo_{n}", titolo=f"Il passo {n}", testo=f"Il testo del passo {n}.\nLa seconda riga del passo {n}.")
                           for n in range(1, quanti + 1))
        self.chiamate = []
        self.da_rivelare = ""

    def suona(self, indice):
        self.chiamate.append(("suona", indice))
        return indice

    def ferma(self):
        self.chiamate.append(("ferma",))

    def rivelazione(self):
        return self.da_rivelare


@pytest.fixture
def sentiti():
    return Sentiti()


@pytest.fixture
def messaggi(monkeypatch):
    registro = Messaggi()
    monkeypatch.setattr(dialoghi, "messaggio", registro)
    return registro


# Le fabbriche di ogni dialogo, come li apre la finestra.
FABBRICHE = {
    "Lettura": lambda suona: dialoghi.Lettura(None, "Classifica", "La classifica è ancora vuota.", suona=suona),
    "Ascolti": lambda suona: dialoghi.Ascolti(None, RegiaFinta(), suona=suona),
    "Caffe": lambda suona: dialoghi.Caffe(None, "Offrimi un caffè, se ti va.", suona),
    "Aggiornamento": lambda suona: dialoghi.Aggiornamento(None, "0.1.0", "0.2.0", "Versione 0.2.0.", 120, suona),
    "Nome": lambda suona: dialoghi.Nome(None, "Gabry", "Con 1.234 punti entri in classifica al posto 1: scrivi il tuo nome.", suona),
    "Volume": lambda suona: dialoghi.Volume(None, 80, None, suona),
    "SceltaLingua": lambda suona: dialoghi.SceltaLingua(None, ["italiano, codice it"], 0, suona),
    "SceltaProiettile": lambda suona: dialoghi.SceltaProiettile(None, costanti.PROIETTILE_FORMA, None, suona),
    "SceltaScheda": lambda suona: dialoghi.SceltaScheda(None, [dict(u) for u in USCITE_FINTE], 0, "Altoparlanti (Realtek(R) Audio), WASAPI",
                                                        suona=suona),
}


def _dialoghi_del_modulo():
    """Le classi dei dialoghi di gui/dialoghi.py che si aprono: tutte quelle che derivano dalla base, tranne la base."""
    return {nome for nome, classe in inspect.getmembers(dialoghi, inspect.isclass)
            if issubclass(classe, dialoghi._Dialogo) and classe is not dialoghi._Dialogo}


def _apri(dialogo, azione):
    """
    Apre il dialogo con ShowModal e, dentro il suo ciclo, guarda se è massimizzato e fa l'azione.
    Restituisce il codice di chiusura, NON_CHIUSO se l'azione non lo ha chiuso, e se era massimizzato.
    """
    visto = {}

    def dentro():
        visto["massimizzato"] = dialogo.IsMaximized()
        azione(dialogo)

    rete = wx.CallLater(ATTESA, lambda: dialogo.IsModal() and dialogo.EndModal(NON_CHIUSO))
    wx.CallAfter(dentro)
    try:
        codice = dialogo.ShowModal()
    finally:
        rete.Stop()
    return codice, visto.get("massimizzato")


def _tasto(dialogo, codice, controllo=None):
    """Un tasto come lo vede wx: il gancio parte dal controllo dato, o da quello che ha il fuoco, o dal dialogo, e sale."""
    fuoco = controllo or wx.Window.FindFocus()
    bersaglio = fuoco if fuoco is not None and fuoco.GetTopLevelParent() is dialogo else dialogo
    evento = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
    evento.SetEventObject(bersaglio)
    evento.SetKeyCode(codice)
    bersaglio.GetEventHandler().ProcessEvent(evento)
    return evento


def _esc(dialogo):
    _tasto(dialogo, wx.WXK_ESCAPE)


def _premi(pulsante):
    evento = wx.CommandEvent(wx.wxEVT_BUTTON, pulsante.GetId())
    evento.SetEventObject(pulsante)
    pulsante.GetEventHandler().ProcessEvent(evento)


def test_la_rassegna_conta_tutti_i_dialoghi():
    # Un dialogo nuovo prende le regole dalla base, ma la rassegna lo deve aprire per dirlo.
    assert _dialoghi_del_modulo() == set(FABBRICHE)


@pytest.mark.parametrize("nome", sorted(FABBRICHE))
def test_ogni_dialogo_si_apre_massimizzato_ed_esc_lo_chiude(app_wx, sentiti, messaggi, nome):
    dialogo = FABBRICHE[nome](sentiti)
    try:
        chiudi = dialogo.pulsante_chiudi
        codice, massimizzato = _apri(dialogo, _esc)
        assert massimizzato, f"{nome} non si apre massimizzato"
        assert codice != NON_CHIUSO, f"{nome}: Esc non lo chiude"
        if isinstance(dialogo, dialoghi.Aggiornamento):
            # Esc vale Non adesso: aggiornare si sceglie solo col suo pulsante.
            assert codice == wx.ID_NO
        elif chiudi is not None:
            assert dialogo.GetEscapeId() == chiudi.GetId() and codice == wx.ID_OK
        else:
            assert codice == wx.ID_CANCEL, f"{nome}: Esc chiude con {codice}, non come Annulla"
    finally:
        dialogo.Destroy()
    assert sentiti == [] and messaggi == []


def test_lettura_ha_il_testo_etichettato_in_sola_lettura_e_chiudi(app_wx):
    dialogo = dialoghi.Lettura(None, "Statistiche", "Sezione 1 di 2.\nAvvii: 3.")
    try:
        figli = list(dialogo.pannello.GetChildren())
        posto = figli.index(dialogo.testo)
        etichetta = figli[posto - 1]
        assert isinstance(etichetta, wx.StaticText) and etichetta.GetLabel() == "Statistiche"
        assert dialogo.testo.GetName() == "Statistiche"
        stile = dialogo.testo.GetWindowStyleFlag()
        assert stile & wx.TE_READONLY and stile & wx.TE_MULTILINE and stile & wx.TE_RICH2
        assert dialogo.testo.GetValue() == "Sezione 1 di 2.\nAvvii: 3."
        assert [b.GetLabelText() for b in dialogo.bottoni] == ["Chiudi"]
        # Chiudi, premuto, chiude con lo stesso codice di Esc.
        codice, _massimizzato = _apri(dialogo, lambda d: _premi(d.bottoni[-1]))
        assert codice == wx.ID_OK
    finally:
        dialogo.Destroy()


def test_caffe_dona_con_paypal(app_wx, sentiti, monkeypatch):
    aperti = []
    monkeypatch.setattr(dialoghi.webbrowser, "open", lambda indirizzo, *a, **k: aperti.append(indirizzo))
    dialogo = dialoghi.Caffe(None, "Offrimi un caffè, se ti va.", sentiti)
    try:
        assert [b.GetLabelText() for b in dialogo.bottoni] == ["Dona con PayPal", "Chiudi"]
        # Invio chiude: Chiudi è il predefinito, non Dona.
        assert dialogo.GetDefaultItem() is dialogo.bottoni[-1]
        codice, _massimizzato = _apri(dialogo, lambda d: _premi(d.bottoni[0]))
    finally:
        dialogo.Destroy()
    assert codice == wx.ID_OK and aperti == [dialoghi.PAYPAL] and sentiti == ["caffe_paypal"]


def test_caffe_invio_nel_testo_chiude_ma_non_quello_tenuto_giu(app_wx, monkeypatch):
    orologio = [100.0]
    monkeypatch.setattr(dialoghi, "_adesso", lambda: orologio[0])
    dialogo = dialoghi.Caffe(None, "Offrimi un caffè, se ti va.")
    try:
        # L'Invio rimasto dalla voce del menu, dentro la grazia dell'apertura, non chiude.
        orologio[0] += dialoghi.GRAZIA_INVIO / 2
        _tasto(dialogo, wx.WXK_RETURN, dialogo.testo)
        assert dialogo.GetReturnCode() == 0
        orologio[0] += dialoghi.GRAZIA_INVIO * 2
        _tasto(dialogo, wx.WXK_RETURN, dialogo.testo)
        assert dialogo.GetReturnCode() == wx.ID_OK
    finally:
        dialogo.Destroy()


TASTI_DEGLI_ASCOLTI = ["Spazio: fa sentire il passo, e lo rifà sentire quante volte vuoi.", "Invio: ferma il suono e passa al passo seguente.",
                       "Esc: chiude gli Ascolti in ogni momento."]


def test_ascolti_area_etichettata_testo_del_passo_e_pulsanti(app_wx, sentiti):
    """D23: l'area in sola lettura «Ascolti» con «Passo N di M», il titolo, il testo e i tasti in fondo; Ascolta, Passo seguente, il
    predefinito, e Chiudi, che vale anche per Esc."""
    dialogo = dialoghi.Ascolti(None, RegiaFinta(), suona=sentiti)
    try:
        assert dialogo.GetTitle() == "Ascolti"
        figli = list(dialogo.pannello.GetChildren())
        etichetta = figli[figli.index(dialogo.testo) - 1]
        assert isinstance(etichetta, wx.StaticText) and etichetta.GetLabel() == "Ascolti" and dialogo.testo.GetName() == "Ascolti"
        stile = dialogo.testo.GetWindowStyleFlag()
        assert stile & wx.TE_READONLY and stile & wx.TE_MULTILINE and stile & wx.TE_RICH2
        assert dialogo.testo.GetValue().splitlines() == ["Passo 1 di 3. Il passo 1.", "Il testo del passo 1.", "La seconda riga del passo 1.", *TASTI_DEGLI_ASCOLTI]
        assert [b.GetLabelText() for b in dialogo.bottoni] == ["Ascolta", "Passo seguente", "Chiudi"]
        assert all(b.GetLabelText() for b in dialogo.bottoni)
        assert dialogo.GetDefaultItem() is dialogo.bottoni[1] and dialogo.pulsante_chiudi is dialogo.bottoni[-1]
        assert dialogo.indice == 0 and not dialogo.finiti
    finally:
        dialogo.Destroy()
    assert sentiti == []


def test_ascolti_spazio_invio_ed_esc(app_wx, sentiti, monkeypatch):
    """Lo spazio fa sentire il passo, quante volte si vuole; Invio ferma il suono e passa al seguente, col cursore in cima e la
    sintesi, e all'ultimo passo chiude; un Invio o uno spazio entro la grazia da quello di prima, o dall'apertura, non contano."""
    orologio = [100.0]
    monkeypatch.setattr(dialoghi, "_adesso", lambda: orologio[0])
    regia, detti = RegiaFinta(), []
    dialogo = dialoghi.Ascolti(None, regia, detti.append, sentiti)
    try:
        testo = dialogo.testo
        # L'Invio tenuto giù dalla voce del menu non salta il primo passo, e non arriva all'area di testo.
        orologio[0] += dialoghi.GRAZIA_INVIO / 2
        assert not _tasto(dialogo, wx.WXK_RETURN, testo).GetSkipped()
        assert dialogo.indice == 0 and regia.chiamate == []
        # Lo spazio fa sentire il passo, e si ferma nel gancio: niente eco, niente spazio nel testo.
        assert not _tasto(dialogo, wx.WXK_SPACE, testo).GetSkipped()
        assert regia.chiamate == [("suona", 0)]
        # Lo spazio ripetuto subito, come un tasto tenuto giù che NVDA rimanda, non rifà partire il suono; dopo la grazia sì.
        orologio[0] += dialoghi.GRAZIA_INVIO / 2
        _tasto(dialogo, wx.WXK_SPACE, testo)
        assert regia.chiamate == [("suona", 0)]
        orologio[0] += dialoghi.GRAZIA_INVIO * 2
        _tasto(dialogo, wx.WXK_SPACE, testo)
        assert regia.chiamate == [("suona", 0), ("suona", 0)]
        # Invio: il suono si ferma, il testo è quello del passo 2, col cursore in cima, e la sintesi lo legge senza i tasti.
        testo.SetInsertionPoint(testo.GetLastPosition())
        orologio[0] += dialoghi.GRAZIA_INVIO * 2
        _tasto(dialogo, wx.WXK_RETURN, testo)
        assert regia.chiamate[-1] == ("ferma",) and dialogo.indice == 1
        assert testo.GetValue().splitlines()[0] == "Passo 2 di 3. Il passo 2." and testo.GetInsertionPoint() == 0
        assert detti == ["Passo 2 di 3. Il passo 2. Il testo del passo 2. La seconda riga del passo 2."]
        _tasto(dialogo, wx.WXK_SPACE, testo)
        assert regia.chiamate[-1] == ("suona", 1)
        # All'ultimo passo Invio chiude: lo dicono il testo e il pulsante, che diventa Fine.
        orologio[0] += dialoghi.GRAZIA_INVIO * 2
        _tasto(dialogo, wx.WXK_NUMPAD_ENTER, testo)
        assert dialogo.indice == 2 and testo.GetValue().splitlines()[-2] == "Invio: ferma il suono e chiude gli Ascolti."
        assert dialogo.bottoni[1].GetLabelText() == "Fine"
        orologio[0] += dialoghi.GRAZIA_INVIO * 2
        codice, _massimizzato = _apri(dialogo, lambda d: _tasto(d, wx.WXK_RETURN, d.testo))
        assert codice == wx.ID_OK and dialogo.finiti and regia.chiamate[-1] == ("ferma",)
    finally:
        dialogo.Destroy()
    assert sentiti == []


def test_ascolti_esc_e_i_pulsanti(app_wx, sentiti, monkeypatch):
    """Esc chiude in ogni momento e ferma il suono, senza contare come finiti; i pulsanti fanno ciò che fanno i tasti, e sui pulsanti
    lo spazio e Invio restano loro."""
    monkeypatch.setattr(dialoghi, "_adesso", lambda: 100.0)
    regia = RegiaFinta()
    dialogo = dialoghi.Ascolti(None, regia, suona=sentiti)
    try:
        ascolta, seguente, _chiudi = dialogo.bottoni
        # Sul pulsante lo spazio resta suo: il gancio lo lascia andare, e lo preme il pulsante, come in Windows.
        assert _tasto(dialogo, wx.WXK_SPACE, seguente).GetSkipped() and regia.chiamate == []
        _premi(ascolta)
        _premi(seguente)
        assert regia.chiamate == [("suona", 0), ("ferma",)] and dialogo.indice == 1
        codice, _massimizzato = _apri(dialogo, _esc)
        assert codice == wx.ID_OK and not dialogo.finiti and regia.chiamate[-1] == ("ferma",)
    finally:
        dialogo.Destroy()
    assert sentiti == []


def test_ascolti_il_testo_che_si_rivela_dopo_l_ascolto(app_wx):
    """Quando il suono del passo è finito, il testo della regia va in fondo all'area, col cursore sul suo inizio, e alla sintesi,
    una volta per passo."""
    regia, detti = RegiaFinta(), []
    dialogo = dialoghi.Ascolti(None, regia, detti.append)
    try:
        dialogo.controlla()
        assert detti == [] and dialogo.testo.GetValue().splitlines()[-1] == TASTI_DEGLI_ASCOLTI[-1]
        regia.da_rivelare = "Ecco dove erano i punti che hai sentito:\nPunto 1: al centro."
        dialogo.controlla()
        dialogo.controlla()
        righe = dialogo.testo.GetValue().splitlines()
        assert righe[-3:] == [TASTI_DEGLI_ASCOLTI[-1], "Ecco dove erano i punti che hai sentito:", "Punto 1: al centro."]
        _trovata, _colonna, riga = dialogo.testo.PositionToXY(dialogo.testo.GetInsertionPoint())
        assert dialogo.testo.GetLineText(riga) == "Ecco dove erano i punti che hai sentito:"
        assert detti == ["Ecco dove erano i punti che hai sentito: Punto 1: al centro."]
        # Al passo seguente il testo nuovo non ha la rivelazione di prima, e una nuova si può dire.
        dialogo.seguente()
        assert "Punto 1: al centro." not in dialogo.testo.GetValue()
        dialogo.controlla()
        assert dialogo.testo.GetValue().endswith("\nPunto 1: al centro.") and len(detti) == 3
    finally:
        dialogo.Destroy()
    # Chiuso il dialogo, il timer non chiede più niente alla regia, e il suono è fermo.
    assert regia.chiamate[-1] == ("ferma",)


def test_aggiornamento_aggiorna_non_adesso_e_scadenza(app_wx):
    dialogo = dialoghi.Aggiornamento(None, "0.1.0", "0.2.0", "Versione 0.2.0, con la classifica nuova.", 120)
    try:
        testo = dialogo.testo.GetValue()
        assert "Ciechino 0.2.0" in testo and "0.1.0" in testo and "2 minuti" in testo and "Versione 0.2.0, con la classifica nuova." in testo
        assert "\n\n" not in testo
        assert [b.GetLabelText() for b in dialogo.bottoni] == ["Aggiorna", "Non adesso"]
        assert dialogo.GetDefaultItem() is dialogo.bottoni[-1] and dialogo.GetEscapeId() == wx.ID_NO
        codice, _massimizzato = _apri(dialogo, lambda d: _premi(d.bottoni[0]))
        assert codice == wx.ID_YES and not dialogo.scaduta
    finally:
        dialogo.Destroy()
    dialogo = dialoghi.Aggiornamento(None, "0.1.0", "0.2.0", "", 120)
    try:
        assert "Nessuna nota scritta per questa versione." in dialogo.testo.GetValue()
        codice, _massimizzato = _apri(dialogo, lambda d: d._scaduta())
        assert codice == wx.ID_NO and dialogo.scaduta
    finally:
        dialogo.Destroy()


def test_nome_proposto_etichetta_e_conferma(app_wx, sentiti, messaggi):
    dialogo = dialoghi.Nome(None, "  Gabry  ", "Con 1.234 punti entri in classifica al posto 1: scrivi il tuo nome.", sentiti)
    try:
        figli = list(dialogo.pannello.GetChildren())
        etichetta = figli[figli.index(dialogo.testo) - 1]
        atteso = f"&Nome, da {costanti.NOME_MINIMO} a {costanti.NOME_MASSIMO} caratteri"
        assert etichetta.GetLabel() == atteso and dialogo.testo.GetName() == atteso.replace("&", "")
        # Il nome proposto è l'ultimo usato, pulito e tutto selezionato, così si sostituisce scrivendo.
        assert dialogo.testo.GetValue() == "Gabry" and dialogo.testo.GetSelection() == (0, len("Gabry"))
        assert dialogo.GRAZIA_ALL_APERTURA and dialogo.GetEscapeId() == wx.ID_ANY
        assert dialogo.GetDefaultItem().GetId() == wx.ID_OK
        # Un nome vuoto avvisa, col suo suono, e il dialogo resta aperto.
        dialogo.testo.SetValue("   ")
        dialogo.conferma()
        assert dialogo.nome is None and dialogo.GetReturnCode() == 0
        assert sentiti == ["campo_da_correggere"] and len(messaggi) == 1 and messaggi[0][2] & wx.ICON_WARNING
        dialogo.testo.SetValue("  Gabriele   Battaglia ")
        dialogo.conferma()
        assert dialogo.nome == "Gabriele Battaglia" and dialogo.GetReturnCode() == wx.ID_OK
    finally:
        dialogo.Destroy()


def test_nome_la_grazia_mangia_lo_spazio_e_l_invio_dell_apertura(app_wx, monkeypatch):
    orologio = [500.0]
    monkeypatch.setattr(dialoghi, "_adesso", lambda: orologio[0])
    dialogo = dialoghi.Nome(None, "Gabry")
    try:
        # Lo spazio dell'ultimo tiro, appena il dialogo è aperto: non arriva al campo.
        orologio[0] += dialoghi.GRAZIA_INVIO / 2
        assert not _tasto(dialogo, wx.WXK_SPACE, dialogo.testo).GetSkipped()
        assert not _tasto(dialogo, wx.WXK_RETURN, dialogo.testo).GetSkipped()
        # Dopo la grazia lo spazio va al campo, come ogni lettera. La ripetizione di wx non si rifà da Python: IsAutoRepeat
        # legge un campo che il KeyEvent di wxPython non lascia scrivere.
        orologio[0] += dialoghi.GRAZIA_INVIO * 2
        assert _tasto(dialogo, wx.WXK_SPACE, dialogo.testo).GetSkipped()
        assert _tasto(dialogo, ord("G"), dialogo.testo).GetSkipped()
    finally:
        dialogo.Destroy()


def test_volume_campo_e_prova(app_wx, monkeypatch):
    provati = []
    # Il suono di prova arriva quando ci si ferma: qui subito, al posto del timer.
    monkeypatch.setattr(dialoghi._Dialogo, "suona_fra_poco", lambda self, funzione: funzione())
    dialogo = dialoghi.Volume(None, 80, provati.append)
    try:
        figli = list(dialogo.pannello.GetChildren())
        etichetta = figli[figli.index(dialogo.numero) - 1]
        assert etichetta.GetLabel() == f"&Volume, da {impostazioni.VOLUME_MINIMO} a {impostazioni.VOLUME_MASSIMO}"
        assert dialogo.numero.GetName() == etichetta.GetLabel().replace("&", "")
        assert (dialogo.numero.GetMin(), dialogo.numero.GetMax(), dialogo.numero.GetValue()) == (impostazioni.VOLUME_MINIMO, impostazioni.VOLUME_MASSIMO, 80)
        dialogo.numero.SetValue(35)
        dialogo.prova()
        assert provati[-1] == 35
        dialogo.conferma()
        assert dialogo.valore == 35 and dialogo.GetReturnCode() == wx.ID_OK
    finally:
        dialogo.Destroy()


def test_scelta_della_lingua(app_wx):
    dialogo = dialoghi.SceltaLingua(None, ["English, codice en", "italiano, codice it"], 1)
    try:
        figli = list(dialogo.pannello.GetChildren())
        assert figli[figli.index(dialogo.elenco) - 1].GetLabel() == "&Lingua" and dialogo.elenco.GetName() == "Lingua"
        assert dialogo.elenco.GetSelection() == 1
        dialogo.elenco.SetSelection(0)
        dialogo.conferma()
        assert dialogo.scelta == 0 and dialogo.GetReturnCode() == wx.ID_OK
    finally:
        dialogo.Destroy()


def test_scelta_del_proiettile_etichette_ed_elenco(app_wx, sentiti):
    """D17: la spiegazione sopra, l'elenco etichettato delle tre forme dette a parole, la forma attuale già scelta, OK e Annulla, la grazia dell'apertura."""
    dialogo = dialoghi.SceltaProiettile(None, "triangolo", None, sentiti)
    try:
        figli = list(dialogo.pannello.GetChildren())
        posto = figli.index(dialogo.elenco)
        etichetta, spiegazione = figli[posto - 1], figli[posto - 2]
        assert isinstance(etichetta, wx.StaticText) and etichetta.GetLabel() == "&Forma del proiettile"
        assert dialogo.elenco.GetName() == "Forma del proiettile"
        assert isinstance(spiegazione, wx.StaticText) and "prossima partita" in spiegazione.GetLabel() and "esempio" in spiegazione.GetLabel()
        assert dialogo.GetTitle() == "Proiettile"
        assert dialogo.forme == costanti.PROIETTILE_FORME
        righe = dialogo.elenco.GetStrings()
        assert righe[0].startswith("Seno, ") and righe[1].startswith("Triangolo, ") and righe[2].startswith("Sega morbida, ")
        # Ogni riga dice anche il carattere del suono, senza segnaposti rimasti né separatori.
        assert all(len(riga.split(", ", 1)) == 2 and "{" not in riga and "_" not in riga and "---" not in riga for riga in righe)
        assert dialogo.elenco.GetSelection() == costanti.PROIETTILE_FORME.index("triangolo")
        assert dialogo.GRAZIA_ALL_APERTURA and dialogo.GetEscapeId() == wx.ID_ANY
        assert dialogo.GetDefaultItem().GetId() == wx.ID_OK
        assert [b.GetLabelText() for b in dialogo.pannello.GetChildren() if isinstance(b, wx.Button)] == ["OK", "Annulla"]
    finally:
        dialogo.Destroy()
    # Una forma sconosciuta, come un file scritto a mano, apre il dialogo su quella di serie.
    dialogo = dialoghi.SceltaProiettile(None, "organo")
    try:
        assert dialogo.elenco.GetSelection() == costanti.PROIETTILE_FORME.index(costanti.PROIETTILE_FORMA)
    finally:
        dialogo.Destroy()
    assert sentiti == []


def test_scelta_del_proiettile_prova_al_cambio_e_conferma(app_wx, sentiti, monkeypatch):
    provate = []
    # L'esempio arriva quando ci si ferma: qui subito, al posto del timer.
    monkeypatch.setattr(dialoghi._Dialogo, "suona_fra_poco", lambda self, funzione: funzione())
    dialogo = dialoghi.SceltaProiettile(None, "seno", provate.append, sentiti)
    try:
        # Spostarsi nell'elenco è l'evento della selezione cambiata: fa sentire la forma della riga.
        for indice in (2, 1):
            dialogo.elenco.SetSelection(indice)
            evento = wx.CommandEvent(wx.wxEVT_LISTBOX, dialogo.elenco.GetId())
            evento.SetEventObject(dialogo.elenco)
            evento.SetInt(indice)
            dialogo.elenco.GetEventHandler().ProcessEvent(evento)
        assert provate == ["sega_morbida", "triangolo"]
        dialogo.conferma()
        assert dialogo.scelta == "triangolo" and dialogo.GetReturnCode() == wx.ID_OK
        # Chiuso il dialogo, niente più esempi, nemmeno quello che aspettava il ritardo.
        dialogo.ferma_la_prova()
        dialogo.prova()
        assert provate == ["sega_morbida", "triangolo"]
    finally:
        dialogo.Destroy()
    assert sentiti == []


def test_scelta_del_proiettile_annulla_ed_esc(app_wx, sentiti):
    for azione in (lambda d: _premi(next(b for b in d.pannello.GetChildren() if isinstance(b, wx.Button) and b.GetId() == wx.ID_CANCEL)), _esc):
        dialogo = dialoghi.SceltaProiettile(None, "seno", None, sentiti)
        try:
            dialogo.elenco.SetSelection(2)
            codice, massimizzato = _apri(dialogo, azione)
            assert codice == wx.ID_CANCEL and massimizzato and dialogo.scelta is None
        finally:
            dialogo.Destroy()
    # Il ritardo dell'esempio vero, col timer di wx: chiuso il dialogo prima che scada, l'esempio non suona.
    provate = []
    dialogo = dialoghi.SceltaProiettile(None, "seno", provate.append, sentiti)
    try:
        dialogo._cambiato()
        assert dialogo._timer_del_suono is not None and dialogo._timer_del_suono.IsRunning()
        dialogo.ferma_la_prova()
        assert not dialogo._timer_del_suono.IsRunning()
    finally:
        dialogo.Destroy()
    assert provate == [] and sentiti == []


def _uscite():
    """Le uscite finte con le cuffie accese dopo l'avvio, come le darebbe schede.uscite() dopo la rilettura: copie nuove."""
    return [dict(u) for u in (*USCITE_FINTE, CUFFIE_FINTE)]


def test_scelta_della_scheda_etichette_ed_elenco(app_wx, sentiti):
    """D20: la spiegazione e le notizie sopra, l'elenco etichettato con prima l'automatica e poi le schede dette da
    schede.etichetta, la scheda in uso detta nella sua riga e già scelta, OK e Annulla, la grazia dell'apertura."""
    uscite = _uscite()
    cuffie = len(uscite)
    adesso = "Il suono esce da Cuffie (WH-1000XM4), WASAPI, con una latenza di 20 millesimi di secondo."
    dialogo = dialoghi.SceltaScheda(None, uscite, cuffie, notizie=[adesso, ""], suona=sentiti)
    try:
        assert dialogo.GetTitle() == "Scheda audio"
        figli = list(dialogo.pannello.GetChildren())
        posto = figli.index(dialogo.elenco)
        etichetta, notizia, spiegazione = figli[posto - 1], figli[posto - 2], figli[posto - 3]
        assert isinstance(etichetta, wx.StaticText) and etichetta.GetLabel() == "&Scheda audio" and dialogo.elenco.GetName() == "Scheda audio"
        # La notizia vuota non diventa una riga vuota: sopra l'etichetta c'è quella che dice dove suona Ciechino, e sopra la spiegazione.
        assert notizia.GetLabel() == adesso
        assert "Automatica suona sulla scheda di Windows" in spiegazione.GetLabel() and "prossimi avvii" in spiegazione.GetLabel()
        # La latenza delle righe è quella dichiarata, e la via a orecchio per tornare all'automatica se un'esclusiva zittisce NVDA.
        assert "la latenza che la scheda dichiara" in spiegazione.GetLabel()
        assert "premi Ctrl+Maiusc+X, aspetta il suono della scelta, poi premi Inizio e, dopo un secondo, Invio." in spiegazione.GetLabel()
        # Il secondo della spiegazione e del manuale basta a uscire dalla grazia dell'apertura.
        assert dialoghi.GRAZIA_INVIO < 1
        righe = dialogo.elenco.GetStrings()
        assert righe == dialogo.righe and len(righe) == len(uscite) + 1
        assert righe[0] == "Automatica, la scheda di Windows"
        assert righe[1:-1] == [schede.etichetta(u) for u in uscite[:-1]]
        assert righe[-1] == "Cuffie (WH-1000XM4), WASAPI, 3 ms, in uso"
        # Le righe che avvisano: l'interfaccia esclusiva e il vivavoce del telefono.
        assert righe[4] == "Realtek ASIO, ASIO, 23,2 ms, esclusiva: può zittire NVDA"
        assert any(riga.endswith("vivavoce: suona come un telefono") for riga in righe)
        assert sum(", in uso" in riga for riga in righe) == 1
        assert all("{" not in riga and "\n" not in riga and "---" not in riga for riga in righe)
        assert dialogo.elenco.GetSelection() == cuffie
        assert dialogo.GRAZIA_ALL_APERTURA and dialogo.GetEscapeId() == wx.ID_ANY
        assert dialogo.GetDefaultItem().GetId() == wx.ID_OK
        assert [b.GetLabelText() for b in dialogo.pannello.GetChildren() if isinstance(b, wx.Button)] == ["OK", "Annulla"]
        assert dialogo.indice is None and dialogo.scelta is None
    finally:
        dialogo.Destroy()
    assert sentiti == []


def test_scelta_della_scheda_automatica_in_uso_o_nessuna(app_wx):
    uscite = _uscite()
    dialogo = dialoghi.SceltaScheda(None, uscite, 0, "Altoparlanti (Realtek(R) Audio), WASAPI")
    try:
        assert dialogo.righe[0] == "Automatica, la scheda di Windows, in uso: Altoparlanti (Realtek(R) Audio), WASAPI"
        assert dialogo.elenco.GetSelection() == 0 and sum(", in uso" in riga for riga in dialogo.righe) == 1
    finally:
        dialogo.Destroy()
    # Muto: nessuna riga in uso, e l'elenco si apre sull'automatica, o sulla riga chiesta, come la scelta salvata.
    for selezionata, attesa in ((None, 0), (3, 3), (99, len(uscite))):
        dialogo = dialoghi.SceltaScheda(None, uscite, None, selezionata=selezionata)
        try:
            assert not any("in uso" in riga for riga in dialogo.righe)
            assert dialogo.elenco.GetSelection() == attesa
        finally:
            dialogo.Destroy()
    # Senza schede resta l'automatica.
    dialogo = dialoghi.SceltaScheda(None, [], None)
    try:
        assert dialogo.righe == ["Automatica, la scheda di Windows"] and dialogo.elenco.GetSelection() == 0
    finally:
        dialogo.Destroy()


def test_scelta_della_scheda_conferma(app_wx):
    uscite = _uscite()
    dialogo = dialoghi.SceltaScheda(None, uscite, 0)
    try:
        digitale = next(n for n, u in enumerate(uscite, 1) if u["dispositivo"] == DIGITALE)
        dialogo.elenco.SetSelection(digitale)
        dialogo.conferma()
        assert dialogo.GetReturnCode() == wx.ID_OK and dialogo.indice == digitale
        assert dialogo.scelta == {"dispositivo": DIGITALE, "interfaccia": "Windows WASAPI"}
        # L'automatica si salva come il dizionario vuoto.
        dialogo.elenco.SetSelection(0)
        dialogo.conferma()
        assert dialogo.indice == 0 and dialogo.scelta == {}
    finally:
        dialogo.Destroy()
    # Il doppio clic conferma come OK.
    dialogo = dialoghi.SceltaScheda(None, uscite, 1)
    try:
        asio = next(n for n, u in enumerate(uscite, 1) if u["dispositivo"] == ASIO)
        dialogo.elenco.SetSelection(asio)
        evento = wx.CommandEvent(wx.wxEVT_LISTBOX_DCLICK, dialogo.elenco.GetId())
        evento.SetEventObject(dialogo.elenco)
        dialogo.elenco.GetEventHandler().ProcessEvent(evento)
        assert dialogo.scelta == {"dispositivo": ASIO, "interfaccia": "ASIO"} and dialogo.GetReturnCode() == wx.ID_OK
    finally:
        dialogo.Destroy()


def test_scelta_della_scheda_annulla_esc_e_grazia(app_wx, sentiti, monkeypatch):
    for azione in (lambda d: _premi(next(b for b in d.pannello.GetChildren() if isinstance(b, wx.Button) and b.GetId() == wx.ID_CANCEL)), _esc):
        dialogo = dialoghi.SceltaScheda(None, _uscite(), 1, suona=sentiti)
        try:
            dialogo.elenco.SetSelection(2)
            codice, massimizzato = _apri(dialogo, azione)
            assert codice == wx.ID_CANCEL and massimizzato and dialogo.scelta is None and dialogo.indice is None
        finally:
            dialogo.Destroy()
    # L'Invio tenuto giù dalla voce del menu, dentro la grazia dell'apertura, non sceglie la scheda.
    orologio = [700.0]
    monkeypatch.setattr(dialoghi, "_adesso", lambda: orologio[0])
    dialogo = dialoghi.SceltaScheda(None, _uscite(), 1, suona=sentiti)
    try:
        orologio[0] += dialoghi.GRAZIA_INVIO / 2
        assert not _tasto(dialogo, wx.WXK_RETURN, dialogo.elenco).GetSkipped()
        assert dialogo.GetReturnCode() == 0 and dialogo.scelta is None
        orologio[0] += dialoghi.GRAZIA_INVIO * 2
        assert _tasto(dialogo, wx.WXK_RETURN, dialogo.elenco).GetSkipped()
    finally:
        dialogo.Destroy()
    assert sentiti == []


def test_le_righe_delle_schede():
    altoparlanti = next(u for u in USCITE_FINTE if u["dispositivo"] == ALTOPARLANTI)
    assert dialoghi.riga_della_scheda(altoparlanti) == "Altoparlanti (Realtek(R) Audio), WASAPI, 3 ms"
    assert dialoghi.riga_della_scheda(CUFFIE_FINTE, in_uso=True) == f"{CUFFIE}, WASAPI, 3 ms, in uso"
    assert dialoghi.riga_dell_automatica() == "Automatica, la scheda di Windows"
    assert dialoghi.riga_dell_automatica(True) == "Automatica, la scheda di Windows, in uso"
    assert dialoghi.riga_dell_automatica(True, "Altoparlanti, WASAPI") == "Automatica, la scheda di Windows, in uso: Altoparlanti, WASAPI"


def test_i_nomi_delle_forme_del_proiettile():
    assert set(dialoghi.NOMI_DEI_PROIETTILI) == set(dialoghi.CARATTERI_DEI_PROIETTILI) == set(costanti.PROIETTILE_FORME)
    assert dialoghi.nome_del_proiettile("sega_morbida") == "sega morbida"
    assert dialoghi.nome_del_proiettile("organo") == "organo"
    assert dialoghi.riga_del_proiettile("seno") == "Seno, " + dialoghi.CARATTERI_DEI_PROIETTILI["seno"]


def test_il_resoconto_degli_errori(cartella_di_prova):
    import registro

    assert "non è aperto" in dialoghi.resoconto_degli_errori()
    registro.avvia(str(cartella_di_prova))
    try:
        testo = dialoghi.resoconto_degli_errori()
    finally:
        registro.chiudi()
    assert registro.FILE_DEGLI_ERRORI in testo and str(cartella_di_prova) in testo
