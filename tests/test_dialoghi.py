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

import pytest
import wx

import costanti
import impostazioni
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
    "Caffe": lambda suona: dialoghi.Caffe(None, "Offrimi un caffè, se ti va.", suona),
    "Aggiornamento": lambda suona: dialoghi.Aggiornamento(None, "0.1.0", "0.2.0", "Versione 0.2.0.", 120, suona),
    "Nome": lambda suona: dialoghi.Nome(None, "Gabry", "Con 1.234 punti entri in classifica al posto 1: scrivi il tuo nome.", suona),
    "Volume": lambda suona: dialoghi.Volume(None, 80, None, suona),
    "SceltaLingua": lambda suona: dialoghi.SceltaLingua(None, ["italiano, codice it"], 0, suona),
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


def test_il_resoconto_degli_errori(cartella_di_prova):
    import registro

    assert "non è aperto" in dialoghi.resoconto_degli_errori()
    registro.avvia(str(cartella_di_prova))
    try:
        testo = dialoghi.resoconto_degli_errori()
    finally:
        registro.chiudi()
    assert registro.FILE_DEGLI_ERRORI in testo and str(cartella_di_prova) in testo
