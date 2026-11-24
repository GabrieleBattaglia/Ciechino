"""
Le prove d'integrazione di Ciechino: la finestra principale vera con la partita vera, giocata dall'inizio
alla fine, col motore audio sul flusso finto o muto, e il punto d'avvio.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0; dalla 2.0.0 la prova mira a mano, come vuole D29, coi tasti dei valori che scendono e
salgono e l'Invio che spara, al posto dello spazio sulle tre scivolate, nella cupola di D28 e con la riserva di D30.
Le altre prove guardano un pezzo alla volta: la partita col motore ma senza la finestra, la finestra
con una partita finta. Qui stanno insieme tutti i pezzi veri, FinestraPrincipale, Partita, Motore,
Suoni, Statistiche, classifica e impostazioni, e la prova gioca come Gabriele: lo spazio, i tasti della
mira e l'Invio sono l'evento wxEVT_CHAR_HOOK che il gancio dei tasti di wx crea per un tasto vero,
rifatto dalla prova e mandato al controllo che ha il fuoco, la vista o, nella mira, il controllo Mira,
da cui sale alla finestra, così passa dal gestore vero, _tasto, con i suoi controlli e il rinfresco
della barra; il rilascio è l'evento wxEVT_KEY_UP, che wx manda soltanto al controllo col fuoco, e la
tastiera che la finestra interroga a ogni battito con wx.GetKeyState è quella della prova, coi tasti
che tiene giù. SendInput li manderebbe allo schermo vero. Il battito è il metodo che il wx.Timer
chiama, _al_battito, chiamato a mano: il timer si ferma, perché la prova decide lei quando passa il
tempo, e un battito arrivato da wx.Yield nel momento sbagliato la renderebbe casuale.
Nessuna prova apre la scheda audio né sintetizza un preset vero: _rendi_preset dà a ogni preset lo
stesso buffer breve. Col flusso finto di aiuti_audio il motore avanza soltanto quando la prova lo
chiede; per premere o lasciare un tasto a un campione preciso la prova fa preparare il blocco che lo
contiene e sposta l'orologio finto finché il campione udibile è proprio quello: un valore della mira
corre da un capo all'altro della sua scala in DURATA_DELLA_SCALA secondi, e così arriva esatto dove la
prova lo vuole. Senza scheda il motore è quello muto, col suo filo che chiama il callback quando
l'orologio di motore_audio passa l'ora del blocco: qui quell'orologio lo sposta la prova, a passi più
corti di RITARDO_MASSIMO_MUTO, e aspetta che il filo abbia preparato i blocchi dovuti. Così il percorso
muto è quello vero, filo compreso, e la prova resta esatta e svelta come col flusso finto, invece di
aspettare il tempo vero o di indovinarlo.
Il dialogo del nome si apre quando il suono dell'ingresso in classifica è finito, sull'orologio del
motore: col flusso finto la prova lo porta fin lì, col motore muto non c'è niente da aspettare.
Il nome in classifica lo scrive la prova al posto del dialogo, i messaggi di wx si annotano, la
sintesi la registra la voce finta del conftest; classifica, statistiche e impostazioni si scrivono
nella cartella temporanea del conftest, e la prova le rilegge dal disco.
"""

import importlib
import importlib.util
import json
import math
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import wx
from aiuti_audio import ALTOPARLANTI, CUFFIE, CUFFIE_FINTE, DIGITALE, motore_finto, sounddevice_finto

import classifica
import costanti
import fisica
import motore_audio
import partita as modulo_partita
import percorsi
import resoconti
import suoni
import testi
from costanti import (
    AIUTO_MANCATO_FINO_AL_LIVELLO,
    CENTRI_PER_LIVELLO,
    COLPI_BONUS,
    COLPI_INIZIALI,
    COLPI_PER_LIVELLO,
    PARAMETRI,
    PARTENZA_DELLA_MIRA,
    TACCHE,
    VELOCITA_DELLA_MIRA,
)
from gui import dialoghi
from gui import finestra as modulo_finestra
from gui.finestra import RIGHE_DELLA_BARRA, TASTO_DI_OGNI_VERSO, FinestraPrincipale
from partita import Partita, punti_del_centro, tempo_della_mira
from sintesi import panoramica, tick_del_tempo
from volo import calcola_volo

SEME = 2026
NOME = "Gabriele"
# Il sommario di D21 del tiro a vuoto voluto, con i tre valori al minimo: tutto a sinistra, l'alzo più basso e la potenza
# più bassa, che con la cupola di D28 entra appena nel guscio dei bersagli, lontano da quello del primo livello.
# Da D24 la riga comincia coi valori, ognuno con la sua iniziale; dalla 0.11.8 senza lo spazio dopo i due punti,
# e la precisione globale di D26 ci sta con loro nel primo blocco.
VUOTO = "L:manca V:manca P:manca I:manca"
# I tre valori del tiro a vuoto voluto, tutti al capo più basso della loro scala.
AL_MINIMO = (0.0, 0.0, 0.0)
# La riga della fase durante la mira manuale (D29).
FASE_DELLA_MIRA = "Mira: Invio spara, Esc annulla"
# Il buffer finto di ogni preset: breve e costante, così ogni effetto finisce presto e non resta voce.
LUNGHEZZA_FINTA = 256
AMPIEZZA_FINTA = 0.1
# Quanti secondi di suono al massimo si aspetta un tiro, prima di dire che non si risolve.
SECONDI_MASSIMI = 30
# I blocchi dopo la fine di un tiro in cui finiscono gli effetti finti, lunghi LUNGHEZZA_FINTA.
BLOCCHI_DEGLI_EFFETTI = 10
# Il motore muto: l'orologio parte da qui e si sposta a passi di al più tanti secondi, meno dei
# RITARDO_MASSIMO_MUTO oltre i quali il filo ripartirebbe da adesso; fra un battito e l'altro del volo
# passano tanti blocchi. Il filo che resta indietro si aspetta a passi di tanti secondi veri, al più
# per tanti secondi veri.
INIZIO_DELL_OROLOGIO = 1000.0
PASSO_DELL_OROLOGIO = 0.2
BLOCCHI_PER_BATTITO_MUTO = 8
PASSO_VERO = 0.0005
ATTESA_MASSIMA = 30.0
# Il livello dalla mira più breve, TEMPO_DELLA_MIRA_MINIMO secondi (D29), e quanti secondi di suono restano aperti un
# dialogo, o la finestra in secondo piano, nella prova della pausa automatica: più di una mira intera a quel livello.
LIVELLO_DELLA_MIRA_BREVE = 24
SECONDI_NEL_DIALOGO = 15
# Il bersaglio che si muove, come il bonus, si mira dove sarà al passaggio del proiettile: il punto si ritrova per
# approssimazioni, al più tante, finché il tempo del passaggio non cambia più di tanti secondi.
GIRI_DELLA_MIRA_SUL_MOBILE = 30
FINEZZA_DELLA_MIRA_SUL_MOBILE = 1e-7
# L'Invio della mira sul mobile arriva tanti secondi dopo i tasti, più di una corsa intera da metà scala a un capo: così
# il colpo parte da un istante che la prova conosce prima di scegliere i valori.
SECONDI_PRIMA_DELL_INVIO = 1.6


# Le fissature.

@pytest.fixture
def preset_finti(monkeypatch):
    buffer = np.full((LUNGHEZZA_FINTA, 2), AMPIEZZA_FINTA, dtype=np.float32)
    buffer.flags.writeable = False
    monkeypatch.setattr(suoni, "_rendi_preset", lambda nome: buffer)


class MessaggiWx(list):
    """I messaggi di wx, al posto di dialoghi.messaggio: una partita giocata bene non ne apre nessuno.
    Una domanda risponde risposta, il no se la prova non dice altro."""

    def __init__(self):
        super().__init__()
        self.risposta = wx.NO

    def __call__(self, genitore, testo, titolo, stile=0):
        self.append((testo, titolo, stile))
        return self.risposta if stile & wx.YES_NO else wx.OK


@pytest.fixture
def messaggi_wx(monkeypatch):
    registro = MessaggiWx()
    monkeypatch.setattr(dialoghi, "messaggio", registro)
    # Un dialogo che la prova non ha preparato non si apre davvero: risponde Annulla.
    monkeypatch.setattr(wx.Dialog, "ShowModal", lambda self: wx.ID_CANCEL)
    return registro


@pytest.fixture
def nomi(monkeypatch):
    """Il dialogo del nome: la prova annota il nome proposto, la spiegazione e se è massimizzato, poi
    scrive NOME e preme OK, come farebbe Gabriele."""
    aperti = []

    def scrive_il_nome(dialogo):
        etichette = [c.GetLabel() for c in dialogo.pannello.GetChildren() if isinstance(c, wx.StaticText)]
        aperti.append({"proposto": dialogo.testo.GetValue(), "spiegazione": etichette[0], "massimizzato": dialogo.IsMaximized()})
        dialogo.testo.SetValue(NOME)
        dialogo.conferma()
        return dialogo.GetReturnCode()

    monkeypatch.setattr(dialoghi.Nome, "ShowModal", scrive_il_nome)
    return aperti


class OrologioAMano:
    """Il modulo time come lo vede motore_audio: perf_counter, l'orologio del filo muto e del campione
    udibile, sta fermo finché la prova non lo sposta; il resto è il modulo vero."""

    def __init__(self, valore=INIZIO_DELL_OROLOGIO):
        self.valore = float(valore)

    def perf_counter(self):
        return self.valore

    def __getattr__(self, nome):
        return getattr(time, nome)


@pytest.fixture
def orologio_muto(monkeypatch):
    orologio = OrologioAMano()
    monkeypatch.setattr(motore_audio, "time", orologio)
    return orologio


# La finestra e ciò che la prova ne osserva.

def _col_fuoco(finestra):
    """Il controllo della finestra che ha il fuoco: la vista, la barra o, nella mira, il controllo Mira (D29); la vista
    se il fuoco non è nella finestra."""
    fuoco = wx.Window.FindFocus()
    if fuoco is not None and fuoco.GetTopLevelParent() is finestra:
        return fuoco
    return finestra.vista


def _tasto(finestra, codice, controllo=False):
    """Un tasto come lo vede wx: il gancio parte dal controllo col fuoco e sale fino alla finestra."""
    sorgente = _col_fuoco(finestra)
    evento = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
    evento.SetEventObject(sorgente)
    evento.SetKeyCode(codice)
    evento.SetControlDown(controllo)
    sorgente.GetEventHandler().ProcessEvent(evento)
    return evento


def _rilascio(finestra, codice):
    """Il rilascio di un tasto come lo vede wx: EVT_KEY_UP arriva soltanto al controllo col fuoco, e non sale."""
    sorgente = _col_fuoco(finestra)
    evento = wx.KeyEvent(wx.wxEVT_KEY_UP)
    evento.SetEventObject(sorgente)
    evento.SetKeyCode(codice)
    sorgente.GetEventHandler().ProcessEvent(evento)
    return evento


def _menu(finestra, comando):
    """Una voce dei menu, come la sceglie wx col suo tasto rapido: l'evento del menu, alla finestra."""
    evento = wx.CommandEvent(wx.wxEVT_MENU, finestra.voci_del_menu[comando].GetId())
    evento.SetEventObject(finestra)
    finestra.GetEventHandler().ProcessEvent(evento)


def _evento_del_menu(finestra, tipo):
    """L'evento che wx manda alla finestra quando il primo menu della barra si apre o si chiude (D22)."""
    evento = wx.MenuEvent(tipo, 0, finestra.GetMenuBar().GetMenu(0))
    evento.SetEventObject(finestra)
    finestra.GetEventHandler().ProcessEvent(evento)


def _riga_del_cursore(controllo):
    _trovata, colonna, riga = controllo.PositionToXY(controllo.GetInsertionPoint())
    return controllo.GetLineText(riga), colonna


def _leggi(nome):
    with open(percorsi.percorso_dati(nome), encoding="utf-8") as file:
        return json.load(file)


class TimerFinto:
    """Il timer del battito al posto del wx.Timer: si accende e si spegne come quello vero, con i nomi dei
    suoi metodi, ma non chiama mai, perché il battito lo dà la prova."""

    def __init__(self, acceso):
        self.acceso = acceso

    def Start(self, *argomenti):
        self.acceso = True
        return True

    def Stop(self):
        self.acceso = False

    def IsRunning(self):
        return self.acceso


class Tavolo:
    """La finestra vera con la partita vera, e il tempo del motore in mano alla prova: il flusso finto,
    o l'orologio del motore muto. Annota ogni messaggio che la partita dà alla finestra, con una spia
    attorno a _ricevi, ogni suono con la sua voce, ogni buffer che entra nel motore, come le tacche e i
    tick della mira (D29), e ogni barra di stato letta. Tiene la tastiera che la finestra legge con
    wx.GetKeyState: un tasto è giù se la prova lo tiene in tasti_giu."""

    def __init__(self, finestra, monkeypatch, flusso=None, orologio=None):
        self.finestra = finestra
        self.partita = finestra.partita
        self.motore = finestra.motore
        self.flusso = flusso
        self.orologio = orologio
        self.blocchi_per_battito = 1 if flusso is not None else BLOCCHI_PER_BATTITO_MUTO
        self.ricevuti = []
        # I secondi del mondo della prova in cui la partita è rimasta ferma ad aspettare lo spazio, nella palestra di D22,
        # e l'ora del battito che ha risolto l'ultimo tiro, che i messaggi seguono alla fine dell'effetto.
        self.in_attesa = 0.0
        self.risolto_alle = None
        # L'ora del battito che ha fermato la partita nell'attesa della palestra: da D24 può essere quello stesso che porta i
        # messaggi del tiro, un secondo dopo l'effetto.
        self.fermo_dal = None
        # I messaggi dell'ultimo battito, o comando, che ne ha portati.
        self.ultimo_lotto = []
        self.suonati = []
        self.nel_motore = []
        self.barre = []
        # I tasti della mira che la prova tiene giù, e quante volte ne ha premuto uno (D29).
        self.tasti_giu = set()
        self.pressioni = 0
        ricevi, suona, suona_nel_motore = finestra._ricevi, finestra.suoni.suona, finestra.motore.suona

        def spia(messaggi, con_la_classifica=True):
            self.ricevuti.extend(messaggi or ())
            if messaggi:
                self.ultimo_lotto = list(messaggi)
            return ricevi(messaggi, con_la_classifica)

        def spia_dei_suoni(evento, *argomenti, **opzioni):
            voce = suona(evento, *argomenti, **opzioni)
            self.suonati.append((evento, voce))
            return voce

        def spia_del_motore(buffer, inizio=None, guadagno=1.0):
            voce = suona_nel_motore(buffer, inizio=inizio, guadagno=guadagno)
            self.nel_motore.append(voce)
            return voce

        monkeypatch.setattr(finestra, "_ricevi", spia)
        # La partita e la finestra suonano con gli stessi Suoni: la spia li sente tutti e due.
        monkeypatch.setattr(finestra.suoni, "suona", spia_dei_suoni)
        monkeypatch.setattr(finestra.motore, "suona", spia_del_motore)
        # Sul desktop nascosto la tastiera vera direbbe sempre su: la finestra legge quella della prova.
        monkeypatch.setattr(wx, "GetKeyState", lambda codice: codice in self.tasti_giu)
        # Sul desktop nascosto nessuna finestra ha il primo piano: la finestra si dice attiva, come quando si gioca.
        monkeypatch.setattr(finestra, "IsActive", lambda: True)
        acceso = finestra.timer.IsRunning()
        finestra.timer.Stop()
        monkeypatch.setattr(finestra, "timer", TimerFinto(acceso))
        if orologio is not None:
            self._aspetta_il_filo()

    def voce(self, evento):
        """La voce dell'ultimo suono di quell'evento."""
        return next(voce for nome, voce in reversed(self.suonati) if nome == evento)

    # Ciò che fa chi gioca, e ciò che sente e legge.

    def barra(self):
        righe = self.finestra.barra.GetValue().splitlines()
        self.barre.append(righe)
        return righe

    def fase_della_barra(self):
        return self.barra()[-1]

    def fuoco(self):
        return wx.Window.FindFocus()

    def spazio(self):
        """Lo spazio, dal gestore vero: deve fermarsi lì, senza andare avanti verso la vista."""
        evento = _tasto(self.finestra, wx.WXK_SPACE)
        assert not evento.GetSkipped()

    def premi(self, codice):
        """Un tasto della mira che scende (D29): il gancio lo prende e non va avanti, e per la tastiera resta giù."""
        self.tasti_giu.add(codice)
        self.pressioni += 1
        evento = _tasto(self.finestra, codice)
        assert not evento.GetSkipped()

    def lascia(self, codice):
        """Un tasto della mira che sale, col suo EVT_KEY_UP al controllo col fuoco, che lo prende e non va avanti."""
        self.tasti_giu.discard(codice)
        evento = _rilascio(self.finestra, codice)
        assert not evento.GetSkipped()

    def invio(self):
        """L'Invio, che nella mira spara (D29): il gancio lo prende e non va avanti."""
        evento = _tasto(self.finestra, wx.WXK_RETURN)
        assert not evento.GetSkipped()

    def battito(self):
        self.finestra._al_battito()
        if self.partita.fase == "attesa" and self.partita.attesa_ferma:
            if self.fermo_dal is None:
                self.fermo_dal = self.adesso()
        else:
            self.fermo_dal = None

    def cursore_sul_nuovo(self):
        """Il cursore della vista sta all'inizio del primo messaggio dell'ultimo battito, o comando, che ne
        ha portati: di solito il resoconto del tiro, e sotto, in ordine, il livello, il bonus o l'ultimo colpo.
        La vista va a capo da sé, e una riga lunga, come quella del sommario con la sua legenda (D24), si
        legge a pezzi: la riga del cursore è l'inizio di quella del messaggio."""
        riga, colonna = _riga_del_cursore(self.finestra.vista)
        assert colonna == 0 and riga and self.ultimo_lotto[0].testo.splitlines()[0].startswith(riga)

    # Il tempo del motore.

    def adesso(self):
        """L'ora del mondo della prova: quella del flusso finto, o l'orologio del motore muto."""
        return self.flusso.time if self.flusso is not None else self.orologio.valore

    def _aspetta_il_filo(self):
        """Aspetta che il filo muto abbia preparato tutti i blocchi dovuti all'ora dell'orologio, e che
        sia uscito dal callback: allora il motore sta fermo finché l'orologio non si sposta."""
        passo = self.motore.blocco / self.motore.fs
        scadenza = time.monotonic() + ATTESA_MASSIMA
        while True:
            riferimento = self.motore._riferimento
            if riferimento is not None and riferimento[1] + passo > self.orologio.valore and self.motore.campione_scritto == riferimento[0] + self.motore.blocco:
                return
            assert time.monotonic() < scadenza, "il filo del motore muto non prepara i blocchi"
            time.sleep(PASSO_VERO)

    def _sposta_l_orologio(self, istante):
        """Porta l'orologio del motore muto all'istante, a passi di PASSO_DELL_OROLOGIO al più."""
        while self.orologio.valore < istante:
            self.orologio.valore = min(istante, self.orologio.valore + PASSO_DELL_OROLOGIO)
            self._aspetta_il_filo()
        self.orologio.valore = istante
        self._aspetta_il_filo()

    def avanza(self, blocchi):
        if self.flusso is not None:
            self.flusso.avanza(blocchi)
        else:
            self._sposta_l_orologio(self.orologio.valore + blocchi * self.motore.blocco / self.motore.fs)

    def porta_udibile_a(self, campione):
        """Fa preparare al motore il blocco che contiene il campione, poi sposta l'orologio finché il
        campione udibile è proprio quello."""
        if self.flusso is not None:
            self.flusso.avanza_fino(math.ceil(campione))
            inizio, dac = self.motore._riferimento
            self.flusso.time = dac + (campione - inizio) / self.flusso.fs
        else:
            for _giro in range(2):
                # Il primo giro arriva al blocco del campione, il secondo aggiusta l'ora su quel blocco.
                inizio, dac = self.motore._riferimento
                self._sposta_l_orologio(dac + (campione - inizio) / self.motore.fs)
        assert self.motore.campione_udibile() == pytest.approx(campione, abs=1e-6)


def _fabbrica(partite, orologio=None):
    """La fabbrica della partita vera, con il rng col seme; la finestra la chiama con motore, suoni e statistiche."""
    def crea(motore, suoni_, statistiche):
        partita = Partita(motore, suoni_, statistiche, rng=random.Random(SEME), orologio=orologio)
        partite.append(partita)
        return partita

    return crea


def _chiudi(finestra):
    motore = finestra.motore if finestra else None
    if finestra and not finestra.chiusa:
        finestra.Close(force=True)
    if motore is not None:
        motore.chiudi()
    wx.Yield()


# La mira manuale (D29).

def _campioni_della_corsa(motore, da, a):
    """I campioni in cui un valore della mira, col suo tasto giù, va da da ad a."""
    return abs(a - da) / VELOCITA_DELLA_MIRA * motore.fs


def _valori_adesso(partita):
    """I tre valori della mira nell'istante che si sente."""
    return partita.mira.valori(partita._adesso())


def _riga_dei_valori(partita):
    """La riga dei valori della mira che la barra di stato deve dire adesso, coi gradi e la percentuale (D29)."""
    direzione, alzo, potenza = _valori_adesso(partita)
    azimut, gradi_dell_alzo, _velocita = fisica.parametri_del_tiro(direzione, alzo, potenza)
    return resoconti.riga_dei_valori_della_mira(azimut, gradi_dell_alzo, potenza, partita.secondi_della_mira())


def _barra_della_mira(tavolo):
    """Nella mira la barra di stato ha le sue RIGHE_DELLA_BARRA righe, con quella dei valori sopra quella della fase."""
    righe = tavolo.barra()
    assert len(righe) == RIGHE_DELLA_BARRA and righe[-2:] == [_riga_dei_valori(tavolo.partita), FASE_DELLA_MIRA]


def _mira(tavolo, valori, invio=None):
    """La mira manuale dei tre valori, dal centro della scala dove la mira li mette (D29): i tasti dei valori da muovere
    scendono insieme, ognuno nel suo verso, e ognuno sale quando il suo valore è arrivato, al campione udibile esatto, o
    poco dopo il capo della scala, dove il valore si ferma da sé; a ogni passo la barra di stato dice i valori di quel
    momento. Poi l'Invio, al campione udibile invio se c'è, fa partire il colpo coi valori voluti."""
    partita, motore = tavolo.partita, tavolo.motore
    assert partita.fase == "mira" and partita.tasti_giu() == [] and _valori_adesso(partita) == [PARTENZA_DELLA_MIRA] * len(PARAMETRI)
    dal = motore.campione_udibile()
    corse = []
    for parametro, valore in zip(PARAMETRI, valori, strict=True):
        if valore == PARTENZA_DELLA_MIRA:
            continue
        arrivo = dal + _campioni_della_corsa(motore, PARTENZA_DELLA_MIRA, valore)
        if valore in (0.0, 1.0):
            # Al capo il valore si ferma da sé: il tasto sale dopo il campione in cui suona il capo, che il battito conta.
            arrivo = math.ceil(arrivo) + 1
        corse.append((arrivo, TASTO_DI_OGNI_VERSO[(parametro, 1 if valore > PARTENZA_DELLA_MIRA else -1)]))
    for _arrivo, codice in corse:
        tavolo.premi(codice)
    _barra_della_mira(tavolo)
    for arrivo, codice in sorted(corse):
        tavolo.porta_udibile_a(arrivo)
        tavolo.battito()
        _barra_della_mira(tavolo)
        tavolo.lascia(codice)
    assert partita.tasti_giu() == [] and _valori_adesso(partita) == pytest.approx(list(valori), abs=1e-9)
    if invio is not None:
        tavolo.porta_udibile_a(invio)
        tavolo.battito()
    _barra_della_mira(tavolo)
    tavolo.invio()
    assert partita.fase == "volo" and partita.valori == pytest.approx(list(valori), abs=1e-9)
    assert tavolo.fase_della_barra() == "Volo"


def _valori_sul_centro(tavolo):
    """Avvia la mira e restituisce il bersaglio e i valori che portano il colpo sul suo centro, con la fisica: i bersagli
    dei primi livelli sono fermi, e il tempo di gioco che scorre nella mira non li sposta."""
    partita = tavolo.partita
    assert partita.fase == "ascolto"
    bersaglio = next(b for b in partita.bersagli if not b.bonus)
    tavolo.spazio()
    assert partita.fase == "mira" and not bersaglio.mobile
    valori = fisica.soluzione(bersaglio.posizione(tavolo.motore.tempo_di_gioco(tavolo.motore.prossimo_campione())))
    assert valori is not None
    return bersaglio, valori


def _partenza_all_invio(tavolo, invio):
    """Il tempo di gioco da cui partirà il colpo dell'Invio dato al campione udibile invio: ANTICIPO_DELLO_SPARO dopo il
    prossimo campione sicuro di quell'istante (D29). La prova lo sa prima di mirare, perché il flusso finto prepara i
    blocchi fino a quello del campione, e nella mira il tempo di gioco scorre senza ancore nuove."""
    motore = tavolo.motore
    scritto = max(motore.campione_scritto, -(-(math.ceil(invio) + 1) // motore.blocco) * motore.blocco)
    sicuro = scritto + motore.prossimo_campione() - motore.campione_scritto
    return motore.tempo_di_gioco(sicuro + round(costanti.ANTICIPO_DELLO_SPARO * motore.fs))


def _valori_sul_mobile(partita, bersaglio, partenza):
    """I valori che portano sul centro di un bersaglio che si muove, come il bonus, il colpo che parte a partenza: si mira
    dove il bersaglio sarà al passaggio del proiettile, come vuole D29, e il punto si ritrova per approssimazioni, dalla
    fisica e dal volo vero fra i bersagli vivi, finché il tempo del passaggio non cambia più."""
    vivi = [b for b in partita.bersagli if b.vivo(partenza)]
    tempo = 0.0
    for _giro in range(GIRI_DELLA_MIRA_SUL_MOBILE):
        valori = fisica.soluzione(bersaglio.posizione(partenza + tempo))
        assert valori is not None
        volo = calcola_volo(fisica.Traiettoria(*fisica.parametri_del_tiro(*valori)), vivi, partenza)
        if abs(volo.tempo_del_passaggio - tempo) < FINEZZA_DELLA_MIRA_SUL_MOBILE:
            break
        tempo = volo.tempo_del_passaggio
    assert volo.colpito is bersaglio
    return valori


# Il tiro: il volo fino al resoconto, il riposo.

def _media_globale(tavolo):
    """La precisione globale media dei tiri arrivati finora (D26), al decimillesimo, dai resoconti:
    quella che il riepilogo deve portare in classifica (D27). I tiri senza, se ce ne sono, restano fuori."""
    globali = [m.dati["precisione_globale"] for m in tavolo.ricevuti if m.tipo == "resoconto" and m.dati.get("precisione_globale") is not None]
    return round(sum(globali) / len(globali), 4)


def _volo(tavolo):
    """Il volo, qualche blocco e un battito alla volta, finché il tiro si risolve e i suoi messaggi
    arrivano, quando l'effetto dell'impatto ha finito di suonare (D22); restituisce i messaggi nuovi."""
    partita = tavolo.partita
    da = len(tavolo.ricevuti)
    tavolo.risolto_alle = None
    for _giro in range(SECONDI_MASSIMI * costanti.FS // (costanti.AUDIO_BLOCCO * tavolo.blocchi_per_battito)):
        tavolo.avanza(tavolo.blocchi_per_battito)
        tavolo.battito()
        if partita.fase == "volo":
            assert tavolo.fase_della_barra() in ("Volo", "In pausa: Ctrl+P per riprendere")
            continue
        if tavolo.risolto_alle is None:
            tavolo.risolto_alle = tavolo.adesso()
        if len(tavolo.ricevuti) > da:
            return tavolo.ricevuti[da:]
    raise AssertionError("il tiro non si è risolto")


def _riposo(tavolo):
    """Lascia finire gli effetti dell'ultimo tiro, e gli avvisi che la partita ha messo in fila dopo
    l'impatto, come l'ultimo colpo o il livello nuovo: poi nel motore restano soltanto le voci dei
    bersagli. Nella palestra (D22) la partita si ferma e aspetta lo spazio, e la prova lo dà, come
    Gabriele dopo aver letto il resoconto, con _fuori_dall_attesa."""
    dopo = tavolo.partita.dopo_gli_avvisi()
    if dopo is not None:
        tavolo.avanza(-(-(dopo - tavolo.motore.campione_scritto) // tavolo.motore.blocco))
    tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
    assert tavolo.motore.stato()["voci"] == len(tavolo.partita.bersagli)
    if tavolo.partita.fase == "attesa":
        _fuori_dall_attesa(tavolo)


def _fuori_dall_attesa(tavolo):
    """Nella palestra (D22), con l'effetto e gli avvisi finiti: il battito ferma la partita, e la barra dice
    che cosa fa lo spazio; lo spazio dato subito, nella grazia, non conta e non suona; passata la grazia
    fa ripartire la partita, coi bersagli nuovi dopo un centro. Il tempo dell'attesa va in
    tavolo.in_attesa, perché la durata della partita non lo conta."""
    partita = tavolo.partita
    tavolo.battito()
    assert partita.fase == "attesa" and partita.attesa_ferma
    assert tavolo.fase_della_barra() in ("Palestra: Spazio per il bersaglio nuovo", "Palestra: Spazio per tornare all'ascolto")
    dal = tavolo.fermo_dal
    # La grazia si conta dal campione udibile a cui la partita si è fermata, che può cadere fra due campioni:
    # un campione prima non basta, uno dopo sì, anche con gli arrotondamenti dell'orologio finto.
    grazia = partita._attesa_dal + round(costanti.GRAZIA_DELL_ATTESA * tavolo.motore.fs)
    suonati = len(tavolo.suonati)
    tavolo.spazio()
    tavolo.porta_udibile_a(math.floor(grazia) - 1)
    tavolo.battito()
    tavolo.spazio()
    assert partita.fase == "attesa" and len(tavolo.suonati) == suonati
    tavolo.porta_udibile_a(math.ceil(grazia) + 1)
    tavolo.battito()
    tavolo.spazio()
    assert partita.fase == "ascolto" and tavolo.fase_della_barra() == "Ascolto: Spazio per mirare" and partita.bersagli
    tavolo.in_attesa += tavolo.adesso() - dal


def _centro(tavolo, numero, totale):
    """Un centro sul bersaglio, dalla mira al resoconto; restituisce il totale nuovo e i messaggi."""
    partita = tavolo.partita
    bersaglio, valori = _valori_sul_centro(tavolo)
    _mira(tavolo, valori)
    volo = partita.ultimo_volo
    assert volo.colpito is bersaglio
    nuovi = _volo(tavolo)
    punti = punti_del_centro(bersaglio, volo.precisione)
    totale += punti
    resoconto = nuovi[0]
    assert resoconto.tipo == "resoconto" and resoconto.dati["tiro"] == numero and resoconto.dati["colpito"]
    assert resoconto.dati["punti"] == punti and resoconto.dati["totale"] == totale == partita.punti
    # Al livello 1 il resoconto comincia col sommario dei numeri (D21), coi valori nel primo blocco e la legenda dal carattere 41 (D24),
    # e la precisione globale (D26); nell'annuncio l'esito, poi il primo blocco, senza la legenda.
    # D33: fra il sommario e l'esito la situazione, il livello, poi il tempo e il posto nella classifica che la finestra ha in memoria;
    # il posto va anche nell'annuncio, subito dopo l'esito.
    sommario, livello, tempo, esito = resoconto.testo.splitlines()[:4]
    atteso = resoconti.sommario_del_tiro(volo, "numeri")
    voci = tavolo.finestra._voci_della_classifica
    posto, posto_detto = resoconti.posto_in_classifica(totale, voci)
    assert sommario == atteso.riga and sommario.startswith("L:") and "L laterale, V verticale" in sommario and " G:" in atteso.annuncio
    # Il centro che chiude il livello lo dice superato: i conti del livello sono quelli di prima del livello nuovo, nei dati del resoconto.
    dati = resoconto.dati
    assert livello == resoconti.riga_del_livello(1, dati["tiri_del_livello"], dati["riserva"], dati["centri_del_livello"]) and dati["tiri_del_livello"] >= 1
    assert livello.endswith("livello superato." if dati["centri_del_livello"] == CENTRI_PER_LIVELLO else " al livello 2.")
    assert tempo.startswith("Tempo ") and tempo.endswith(f"; {posto}.") and dati["posizione"] == classifica.posizione(voci, totale)
    assert esito.startswith(f"Tiro {numero}, livello 1: colpito un bersaglio, forma ")
    assert resoconto.annuncio == f"Colpito, {testi.intero(punti)} punti. {posto_detto}. {atteso.annuncio}"
    tavolo.cursore_sul_nuovo()
    return totale, nuovi


def _a_vuoto(tavolo):
    """Un tiro a vuoto voluto, coi tre valori tenuti giù fino al minimo, dove ognuno si ferma col suo suono."""
    tavolo.spazio()
    _mira(tavolo, AL_MINIMO)
    assert tavolo.partita.ultimo_volo.colpito is None


# La partita intera.

def test_una_partita_intera_nella_finestra_vera(app_wx, preset_finti, messaggi_wx, nomi, voce_finta, monkeypatch):
    motore, flusso = motore_finto()
    partite = []
    # La durata vera della partita la misura l'orologio finto del flusso: è il tempo del mondo della prova.
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica(partite, orologio=lambda: flusso.time), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        assert partite == [partita]
        apertura = finestra.vista.GetValue()
        assert "Il suono esce da finta, con una latenza di 20 millesimi di secondo." in apertura.splitlines()
        assert tavolo.barra() == ["Nessuna partita: Spazio per cominciare"]

        # Lo spazio comincia la partita, con la riserva di D30.
        tavolo.spazio()
        inizio = tavolo.adesso()
        assert partita.fase == "ascolto" and [m.tipo for m in tavolo.ricevuti] == ["info"]
        assert (COLPI_INIZIALI, COLPI_PER_LIVELLO, COLPI_BONUS) == (20, 15, 8)
        assert tavolo.barra() == ["Livello 1, 0 punti", f"Riserva {COLPI_INIZIALI}, centri 0 su {CENTRI_PER_LIVELLO}", "Bersagli in volo: 1",
                                  "Ascolto: Spazio per mirare"]
        tavolo.cursore_sul_nuovo()
        _riposo(tavolo)

        # I centri, fino al livello 2; al primo, lo spazio col proiettile in volo non fa niente, e suona non_ora.
        totale = 0
        for numero in range(1, CENTRI_PER_LIVELLO + 1):
            if numero == 1:
                bersaglio, valori = _valori_sul_centro(tavolo)
                _mira(tavolo, valori)
                prima = len(tavolo.ricevuti)
                tavolo.spazio()
                assert len(tavolo.ricevuti) == prima and partita.fase == "volo" and tavolo.fase_della_barra() == "Volo"
                nuovi = _volo(tavolo)
                totale = punti_del_centro(bersaglio, partita.ultimo_volo.precisione)
                assert nuovi[0].dati["tiro"] == 1 and nuovi[0].dati["totale"] == totale == partita.punti
                tavolo.cursore_sul_nuovo()
            else:
                totale, nuovi = _centro(tavolo, numero, totale)
            # Il livello 1 è della palestra (D22): dopo ogni centro la partita aspetta lo spazio, che chiama il bersaglio nuovo.
            assert partita.fase == "attesa" and nuovi[-1].testo == "Palestra: la partita aspetta lo Spazio, che chiama il bersaglio nuovo."
            if numero < CENTRI_PER_LIVELLO:
                assert partita.livello == 1 and partita.bersagli == []
                assert tavolo.barra()[:2] == [f"Livello 1, {testi.intero(totale)} punti", f"Riserva {COLPI_INIZIALI}, centri {numero} su {CENTRI_PER_LIVELLO}"]
            _riposo(tavolo)
        riserva = COLPI_INIZIALI + COLPI_PER_LIVELLO
        assert [m.tipo for m in nuovi][:2] == ["resoconto", "livello"] and nuovi[1].annuncio == f"Livello 2, riserva {riserva}."
        assert (partita.livello, partita.riserva, partita.centri_del_livello) == (2, riserva, 0)
        assert tavolo.barra() == [f"Livello 2, {testi.intero(totale)} punti", f"Riserva {riserva}, centri 0 su {CENTRI_PER_LIVELLO}", "Bersagli in volo: 1",
                                  "Ascolto: Spazio per mirare"]

        # I tiri a vuoto voluti, fino alla riserva vuota; a metà, una pausa col proiettile in volo, con Ctrl+P.
        in_pausa = 0.0
        for vuoto in range(1, riserva + 1):
            _a_vuoto(tavolo)
            if vuoto == riserva // 2:
                _menu(finestra, finestra.pausa)
                dal = tavolo.adesso()
                assert partita.in_pausa and tavolo.fase_della_barra() == "In pausa: Ctrl+P per riprendere"
                tavolo.cursore_sul_nuovo()
                for _blocco in range(costanti.FS // costanti.AUDIO_BLOCCO):
                    tavolo.avanza(1)
                    tavolo.battito()
                assert partita.fase == "volo" and not partita._evento_impatto.scattato
                _menu(finestra, finestra.pausa)
                in_pausa += tavolo.adesso() - dal
                assert not partita.in_pausa and tavolo.fase_della_barra() == "Volo"
            nuovi = _volo(tavolo)
            resoconto = nuovi[0]
            assert resoconto.tipo == "resoconto" and resoconto.dati["tiro"] == CENTRI_PER_LIVELLO + vuoto and not resoconto.dati["colpito"]
            assert resoconto.dati["riserva"] == riserva - vuoto == partita.riserva
            # Al livello 2 il resoconto dice ancora di quanto si è mancato il bersaglio più vicino (D6).
            assert AIUTO_MANCATO_FINO_AL_LIVELLO >= 2 and "Il bersaglio più vicino è stato mancato di " in resoconto.testo
            # Al livello 2 il sommario c'è ancora, coi numeri (D21); il tiro al minimo cade prima della
            # distanza del bersaglio e sotto la sua altezza, e il sommario lo dice senza numeri.
            assert resoconto.testo.splitlines()[0] == resoconti.riga_del_sommario(partita.ultimo_volo, "numeri") and resoconto.testo.startswith(f"{VUOTO} G:")
            # La G è 1 meno la distanza del passaggio più vicino sulla distanza fra il centro del bersaglio in quell'istante e il
            # punto della cupola più lontano da lui, mai sotto zero (D26, D28).
            volo = partita.ultimo_volo
            lontananza = fisica.distanza_dal_punto_piu_lontano(volo.vicino.posizione(volo.partenza + volo.tempo_del_passaggio))
            globale = max(0.0, 1.0 - volo.distanza_dal_centro / lontananza)
            assert resoconto.dati["precisione_globale"] == pytest.approx(globale) and resoconto.testo.startswith(f"{VUOTO} G:{testi.percento(globale, 1)}")
            # Anche il livello 2 è della palestra (D22): dopo il tiro a vuoto lo spazio riporta all'ascolto.
            if partita.riserva > 1:
                assert [m.tipo for m in nuovi] == ["resoconto", "info"] and partita.fase == "attesa"
                # D33: dopo l'esito il posto nella classifica della finestra, poi il sommario.
                posto = resoconti.posto_in_classifica(totale, finestra._voci_della_classifica)[1]
                assert resoconto.annuncio.startswith(f"A vuoto, riserva {partita.riserva}. {posto}. {VUOTO} G:")
                assert nuovi[-1].testo == "Palestra: la partita aspetta lo Spazio, che ti riporta all'ascolto."
            elif partita.riserva == 1:
                assert [m.tipo for m in nuovi] == ["resoconto", "avviso", "info"] and nuovi[1].annuncio == "Ultimo colpo."
            if partita.riserva:
                tavolo.cursore_sul_nuovo()
                _riposo(tavolo)
        # La partita è finita al battito che ha risolto l'ultimo tiro; i messaggi sono arrivati alla fine del tonfo (D22).
        fine = tavolo.risolto_alle

        # La fine: il resoconto, la fine col riepilogo, il nome e la classifica in coda alla vista.
        assert partita.fase == "finita" and not partita.in_corso()
        posto = resoconti.posto_in_classifica(totale, finestra._voci_della_classifica)[1]
        assert [m.tipo for m in nuovi] == ["resoconto", "fine"] and nuovi[0].annuncio.startswith(f"A vuoto, riserva vuota. {posto}. {VUOTO} G:")
        tiri = CENTRI_PER_LIVELLO + riserva
        # La durata non conta né la pausa né le attese della palestra, una dopo ogni tiro tranne l'ultimo (D22).
        assert tavolo.in_attesa > 0
        durata = round(fine - inizio - in_pausa - tavolo.in_attesa, 1)
        riepilogo = {"punteggio": totale, "livello": 2, "tiri": tiri, "centri": CENTRI_PER_LIVELLO, "durata_s": durata,
                     "precisione_globale": _media_globale(tavolo)}
        assert nuovi[1].dati == riepilogo == partita.riepilogo()
        assert nuovi[1].annuncio == f"Partita finita, {testi.intero(totale)} punti."
        # Il suono della classifica segue quello della fine, in fila, staccato: partiti insieme si fonderebbero. Da D24 arriva coi messaggi
        # della fine, un secondo dopo il tonfo, quando la fine può aver già finito di suonare.
        fine_della_partita, ingresso = tavolo.voce("partita_finita"), tavolo.voce("in_classifica")
        assert ingresso.inizio >= fine_della_partita.inizio + LUNGHEZZA_FINTA + round(modulo_partita.STACCO_FRA_GLI_AVVISI * motore.fs)
        # Il nome si chiede quando il suono dell'ingresso è finito: intanto la finestra lo aspetta come un
        # dialogo aperto, e lo spazio, che va avanti verso la vista, non comincia una partita nuova.
        assert nomi == [] and finestra._dialogo_aperto()
        assert _tasto(finestra, wx.WXK_SPACE).GetSkipped() and partita.fase == "finita"
        tavolo.porta_udibile_a(ingresso.inizio + LUNGHEZZA_FINTA - 1)
        tavolo.battito()
        assert nomi == []
        tavolo.porta_udibile_a(ingresso.inizio + LUNGHEZZA_FINTA)
        tavolo.battito()
        assert nomi == [{"proposto": "", "spiegazione": f"Con {testi.intero(totale)} punti entri in classifica al posto 1: scrivi il tuo nome.",
                         "massimizzato": True}]
        assert tavolo.barra() == [f"Livello 2, {testi.intero(totale)} punti", f"Riserva 0, centri 0 su {CENTRI_PER_LIVELLO}",
                                  "Partita finita: Spazio per ricominciare"]
        # Ogni riga della barra, in ogni fase, sta nei quaranta caratteri del display braille, tranne quella dei valori della mira
        # di D29, che li supera.
        valori_della_mira = {righe[-2] for righe in tavolo.barre if righe[-1] == FASE_DELLA_MIRA}
        assert valori_della_mira and all(len(riga) <= 40 for righe in tavolo.barre for riga in righe if riga not in valori_della_mira)

        # La classifica, sul disco e nella vista, col cursore sulla voce nuova.
        voci = _leggi(costanti.FILE_CLASSIFICA)["voci"]
        assert len(voci) == 1
        assert {chiave: voci[0][chiave] for chiave in ("nome", *riepilogo)} == {"nome": NOME, **riepilogo}
        righe_della_classifica = classifica.righe(voci, nuova=1)
        # D27: la riga della griglia dice tutti i campi della partita, in colonna, e finisce con nuova.
        nuova = righe_della_classifica[classifica.PRIMA_VOCE]
        assert nuova.split() == ["1", NOME, testi.intero(totale), "2", str(tiri), str(CENTRI_PER_LIVELLO), testi.percento(riepilogo["precisione_globale"], 1),
                                 testi.durata_in_cifre(riepilogo["durata_s"]), "0", "nuova"]
        assert _riga_del_cursore(finestra.vista) == (nuova, 0)
        assert _leggi(costanti.FILE_IMPOSTAZIONI)["ultimo_nome"] == NOME

        # La vista: l'apertura, poi tutti i messaggi in ordine, una cosa per riga, poi la classifica. La mira non scrive niente.
        righe = [riga for m in tavolo.ricevuti for riga in m.testo.splitlines() if riga.strip()]
        testo = finestra.vista.GetValue()
        assert testo == "\n".join([apertura, *righe, *righe_della_classifica])
        posti = [testo.index(f"\nTiro {n}, livello ") for n in range(1, tiri + 1)]
        assert posti == sorted(posti)
        assert posti[CENTRI_PER_LIVELLO - 1] < testo.index("\nLivello 2! Il livello 1 è superato.") < posti[CENTRI_PER_LIVELLO]
        assert posti[-2] < testo.index("\nNella riserva resta l'ultimo colpo") < posti[-1] < testo.index("\nPartita finita: ")

        # La sintesi ha letto ogni annuncio, nell'ordine, e alla fine la voce nuova della classifica, a parole.
        assert voce_finta == [m.annuncio for m in tavolo.ricevuti if m.annuncio] + [classifica.annuncio(voci[0], 1)]
        assert voce_finta[0] == "Partita nuova: Spazio per mirare."
        # D33: fra i punti e il sommario il posto in classifica, che in questa prova, a classifica vuota, è il primo.
        assert sum(1 for detto in voce_finta if detto.startswith("Colpito, ") and " punti. 1°. L:" in detto and " G:" in detto) == CENTRI_PER_LIVELLO

        # Le statistiche, sul disco, coi numeri della partita.
        statistiche = _leggi(costanti.FILE_STATISTICHE)
        # I contatori a zero non si confrontano: le statistiche ne partono di nuovi da zero quando la
        # pagina impara a dirli, e la prova guarda soltanto quelli che la partita ha mosso.
        assert {chiave: n for chiave, n in statistiche["conti"].items() if n} == {
            "avvii": 1, "partite_iniziate": 1, "partite_finite": 1, "tiri": tiri, "centri": CENTRI_PER_LIVELLO, "tiri_a_vuoto": riserva, "punti": totale,
            "livelli_superati": 1, "colpi_guadagnati": COLPI_PER_LIVELLO, "mire_avviate": tiri, "ingressi_in_classifica": 1, "pause": 1,
            # La partita si è giocata col proiettile di serie, il seno (D17), e tutta nella palestra (D22); ogni mira è finita
            # con l'Invio, coi suoi tasti (D29).
            "partite_col_seno": 1, "attese_della_palestra": tiri - 1, "mire_sparate": tiri, "tasti_della_mira": tavolo.pressioni}
        # Il valore che arriva al capo della scala suona fine_scala: tre volte per ogni tiro a vuoto, coi tre valori al minimo.
        assert statistiche["suoni"] == {"nuova_partita": 1, "mira_avviata": tiri, "fine_scala": 3 * riserva, "sparo": tiri, "colpito": CENTRI_PER_LIVELLO,
                                        "a_terra": riserva, "livello_su": 1, "non_ora": 1, "pausa": 1, "ripresa": 1, "ultimo_colpo": 1,
                                        "partita_finita": 1, "in_classifica": 1, "confermato": 1}
        # In palestra la mira non ha tempo, e nessuna è scaduta; i secondi di mira sono quelli dei tasti, dall'avvio all'Invio.
        assert statistiche["somme"]["secondi_di_mira"] > 0
        for parametro in PARAMETRI:
            presi = statistiche["mira"][parametro]
            # Da D29 i valori della mira manuale vanno in manuali, e quelli rimasti a un capo anche in ai_capi; le chiavi delle
            # scivolate restano a zero.
            assert (presi["manuali"], presi["ai_capi"], presi["nei_centri"], sum(presi["classi"])) == (tiri, riserva, CENTRI_PER_LIVELLO, tiri)
            assert presi["classi"][0] >= riserva
            assert (presi["scelto"], presi["minimo"], presi["massimo_con_lo_spazio"], presi["massimo"]) == (0, 0, 0, 0)
        assert statistiche["moto"] == {"fermi": {"apparsi": CENTRI_PER_LIVELLO + 1, "colpiti": CENTRI_PER_LIVELLO}, "mobili": {"apparsi": 0, "colpiti": 0}}
        primati = statistiche["primati"]
        assert (primati["punteggio"]["valore"], primati["livello"]["valore"], primati["centri_in_una_partita"]["valore"],
                primati["serie"]["valore"]) == (totale, 2, CENTRI_PER_LIVELLO, CENTRI_PER_LIVELLO)
        assert primati["partita_piu_lunga"]["valore"] == pytest.approx(durata) and statistiche["tempo_di_gioco"] == pytest.approx(durata)
        assert statistiche["ultime"][-1]["motivo"] == "riserva_vuota" and statistiche["ultime"][-1]["punteggio"] == totale

        # Finita la partita, e sentita la classifica, il motore non tiene più nessuna voce, e nessun messaggio
        # di wx si è aperto.
        flusso.avanza_fino(ingresso.inizio + LUNGHEZZA_FINTA)
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert motore.stato()["voci"] == 0 and partita.bersagli == []
        assert messaggi_wx == []
        # Lo spazio, a partita finita, ne comincia un'altra.
        tavolo.spazio()
        assert partita.fase == "ascolto" and partita.tiri == 0 and tavolo.ricevuti[-1].annuncio == "Partita nuova: Spazio per mirare."
    finally:
        _chiudi(finestra)
    assert motore.chiuso


def test_una_partita_col_motore_muto(app_wx, preset_finti, messaggi_wx, nomi, voce_finta, orologio_muto, monkeypatch):
    """Senza scheda audio il motore lavora muto, col suo filo, e la partita va uguale: un centro, poi i
    tiri a vuoto fino alla riserva vuota, la classifica e le statistiche. Anche la mira manuale va uguale:
    i valori si muovono sul campione udibile dell'orologio del motore muto."""
    partite = []
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica(partite, orologio=lambda: orologio_muto.valore))
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, orologio=orologio_muto)
        partita, motore = tavolo.partita, tavolo.motore
        assert motore.muto and motore.flusso is None and not motore.stato()["aperto"]
        assert motore._filo_muto is not None and motore._filo_muto.is_alive()
        assert "Non trovo una scheda audio che si apra: Ciechino funziona lo stesso, ma senza suono." in finestra.vista.GetValue().splitlines()
        tavolo.spazio()
        inizio = tavolo.adesso()
        assert partita.fase == "ascolto"
        _riposo(tavolo)
        totale, _nuovi = _centro(tavolo, 1, 0)
        assert partita.centri == 1
        _riposo(tavolo)
        for vuoto in range(1, COLPI_INIZIALI + 1):
            _a_vuoto(tavolo)
            nuovi = _volo(tavolo)
            assert nuovi[0].tipo == "resoconto" and nuovi[0].dati["tiro"] == 1 + vuoto and not nuovi[0].dati["colpito"]
            if partita.riserva:
                tavolo.cursore_sul_nuovo()
                _riposo(tavolo)
        assert partita.fase == "finita" and [m.tipo for m in nuovi] == ["resoconto", "fine"]
        # Le attese della palestra (D22) non contano nella durata.
        riepilogo = {"punteggio": totale, "livello": 1, "tiri": COLPI_INIZIALI + 1, "centri": 1, "durata_s": round(tavolo.risolto_alle - inizio - tavolo.in_attesa, 1),
                     "precisione_globale": _media_globale(tavolo)}
        assert partita.riepilogo() == riepilogo == nuovi[-1].dati
        assert len(nomi) == 1
        voci = _leggi(costanti.FILE_CLASSIFICA)["voci"]
        assert [{chiave: voce[chiave] for chiave in ("nome", *riepilogo)} for voce in voci] == [{"nome": NOME, **riepilogo}]
        nuova = classifica.righe(voci, nuova=1)[classifica.PRIMA_VOCE]
        assert _riga_del_cursore(finestra.vista) == (nuova, 0) and finestra.vista.GetValue().endswith("\n" + nuova)
        assert voce_finta == [m.annuncio for m in tavolo.ricevuti if m.annuncio] + [classifica.annuncio(voci[0], 1)]
        statistiche = _leggi(costanti.FILE_STATISTICHE)
        conti = statistiche["conti"]
        assert (conti["partite_finite"], conti["tiri"], conti["centri"], conti["tiri_a_vuoto"], conti["punti"], conti["ingressi_in_classifica"]) == (
            1, COLPI_INIZIALI + 1, 1, COLPI_INIZIALI, totale, 1)
        assert (conti["mire_sparate"], conti["tasti_della_mira"], conti["mire_scadute"]) == (COLPI_INIZIALI + 1, tavolo.pressioni, 0)
        assert (statistiche["suoni"]["sparo"], statistiche["suoni"]["a_terra"], statistiche["suoni"]["colpito"]) == (COLPI_INIZIALI + 1, COLPI_INIZIALI, 1)
        assert statistiche["suoni"]["fine_scala"] == 3 * COLPI_INIZIALI
        assert tavolo.barra()[-1] == "Partita finita: Spazio per ricominciare"
        assert messaggi_wx == []
    finally:
        _chiudi(finestra)
    assert motore.chiuso and not motore._filo_muto.is_alive()


def test_uscire_col_proiettile_in_volo_abbandona_la_partita(app_wx, preset_finti, messaggi_wx, nomi, monkeypatch):
    """Ctrl+Q con il proiettile in volo: la domanda, col no già scelto; al sì la partita si abbandona,
    le voci si fermano, le statistiche si salvano, la classifica non si chiede e il motore si chiude."""
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        tavolo.spazio()
        _riposo(tavolo)
        _bersaglio, valori = _valori_sul_centro(tavolo)
        _mira(tavolo, valori)
        proiettile, evento = partita._voce_proiettile, partita._evento_impatto
        voci = list(partita._voci.values())
        finestra.Close()
        # Il no, già scelto: si resta in volo.
        assert len(messaggi_wx) == 1 and messaggi_wx[0][2] & wx.YES_NO and messaggi_wx[0][2] & wx.NO_DEFAULT
        assert not finestra.chiusa and partita.fase == "volo"
        messaggi_wx.risposta = wx.YES
        finestra.Close()
        assert finestra.chiusa and partita.fase == "pronta" and evento.annullato
        assert tavolo.ricevuti[-1].annuncio == "Partita abbandonata." and nomi == []
        # Il flusso finto non avanza più, e il motore si è chiuso: le fermate del proiettile e dei bersagli
        # sono rimaste nella coda dei comandi, dove la scheda vera le avrebbe suonate, sfumate, mentre il
        # congedo suonava.
        fermate = [comando[1] for comando in motore._comandi if comando[0] == "ferma"]
        assert all(any(voce is fermata for fermata in fermate) for voce in [proiettile, *voci])
        assert flusso.chiuso and not evento.scattato
        statistiche = _leggi(costanti.FILE_STATISTICHE)
        conti = statistiche["conti"]
        assert (conti["partite_iniziate"], conti["partite_abbandonate"], conti["partite_finite"]) == (1, 1, 0)
        # Il tiro abbandonato in volo non è né nel riepilogo della partita né fra i tiri contati: il suo
        # esito non si è mai sentito, e i due conti dicono lo stesso numero.
        assert statistiche["ultime"][-1]["tiri"] == 0 and conti["tiri"] == 0
        assert classifica.carica() == ([], None)
    finally:
        _chiudi(finestra)
    assert motore.chiuso


# La mira manuale nella finestra (D29).

def test_la_mira_manuale_coi_tasti_nella_finestra_vera(app_wx, preset_finti, messaggi_wx, nomi, voce_finta, monkeypatch):
    """D29 coi pezzi veri sul flusso finto. Lo spazio avvia la mira: il fuoco va sul controllo Mira, i tre valori stanno al
    centro, la barra di stato li dice, i bersagli suonano pieni e il tempo di gioco scorre; lo spazio nella mira suona il
    non ora. Ogni tasto muove il suo valore finché è giù, nel suo verso, alla velocità della scala, con le sue tacche a ogni
    decimo; la ripetizione del tasto tenuto giù non cambia niente, e due tasti giù muovono due valori insieme. Il rilascio
    arriva al controllo col fuoco; quello perso lo riprende il battito, che chiede la tastiera. F7 porta il fuoco nella barra,
    dove i tasti dei valori, dalle prove del 22 novembre 2026, vanno avanti, per leggere, e non muovono la mira; F6 lo riporta
    sulla Mira, da dove la muovono di nuovo. Il rilascio arriva anche alla barra, l'Invio spara anche da lì, e il fuoco resta
    lì anche dopo lo sparo. Alla mira seguente il fuoco torna sulla Mira,
    Esc annulla col tasto giù, il proiettile resta nella riserva e il fuoco torna alla vista; fuori dalla mira Esc e i tasti
    dei valori vanno avanti."""
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        fs = motore.fs
        tavolo.spazio()
        _riposo(tavolo)
        assert tavolo.fuoco() is finestra.vista

        # Lo spazio avvia la mira, col fuoco sul controllo Mira, che NVDA dice col suo nome.
        tavolo.spazio()
        assert partita.fase == "mira" and tavolo.suonati[-1][0] == "mira_avviata"
        assert tavolo.fuoco() is finestra.controllo_mira and finestra.controllo_mira.GetName() == "Mira"
        righe = tavolo.barra()
        assert len(righe) == RIGHE_DELLA_BARRA and righe[-2:] == ["Direzione davanti, alzo 50,0 gradi, potenza 50%, senza tempo", FASE_DELLA_MIRA]
        # Nella mira i bersagli suonano pieni e il tempo di gioco scorre (D29).
        tempo = motore.tempo_di_gioco()
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert motore.tempo_di_gioco() > tempo and all(partita._voci[b].guadagno == 1.0 for b in partita.bersagli)
        suonati = len(tavolo.suonati)
        tavolo.spazio()
        assert partita.fase == "mira" and [nome for nome, _voce in tavolo.suonati[suonati:]] == ["non_ora"]

        # La freccia destra muove la direzione verso destra, con le tacche dei decimi davanti e il capo in fondo.
        destra = TASTO_DI_OGNI_VERSO[("direzione", 1)]
        prima = len(tavolo.nel_motore)
        dal = motore.campione_udibile()
        tavolo.premi(destra)
        assert partita.tasti_giu() == [("direzione", 1)] and partita._voce_del_puntatore is not None
        tacche = partita._tacche["direzione"]
        davanti = {voce.inizio: voce for voce in tavolo.nel_motore[prima:]}
        for k in range(TACCHE // 2 + 1, TACCHE):
            inizio = round((dal / fs + (k / TACCHE - PARTENZA_DELLA_MIRA) / VELOCITA_DELLA_MIRA) * fs)
            # La tacca mono del suo decimo, dal punto del puntatore, quello della direzione che la attraversa (D31).
            assert np.allclose(davanti[inizio].buffer, panoramica(tacche[k], 2.0 * k / TACCHE - 1.0), rtol=0, atol=1e-7)
        assert partita._capi_in_fila["direzione"] == round((dal / fs + (1.0 - PARTENZA_DELLA_MIRA) / VELOCITA_DELLA_MIRA) * fs)
        # La ripetizione del tasto tenuto giù arriva dal gancio, ma non conta: il moto è già partito.
        ripetuto = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        ripetuto.SetEventObject(finestra.controllo_mira)
        ripetuto.SetKeyCode(destra)
        monkeypatch.setattr(ripetuto, "IsAutoRepeat", lambda: True)
        finestra._tasto(ripetuto)
        assert not ripetuto.GetSkipped() and partita.mira.pressioni == 1
        # Dopo un decimo e mezzo di scala anche la freccia su: due valori si muovono insieme.
        su = TASTO_DI_OGNI_VERSO[("alzo", 1)]
        alzo_dal = dal + _campioni_della_corsa(motore, 0.0, 0.15)
        tavolo.porta_udibile_a(alzo_dal)
        tavolo.battito()
        tavolo.premi(su)
        assert sorted(partita.tasti_giu()) == [("alzo", 1), ("direzione", 1)]
        tavolo.porta_udibile_a(alzo_dal + _campioni_della_corsa(motore, 0.0, 0.1))
        tavolo.battito()
        assert _valori_adesso(partita) == pytest.approx([0.75, 0.6, 0.5], abs=1e-9)
        _barra_della_mira(tavolo)
        # Il rilascio della destra, al controllo Mira: la direzione si ferma, e la tacca del 70%, già passata, ha suonato intera.
        tavolo.lascia(destra)
        assert partita.tasti_giu() == [("alzo", 1)]
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert _valori_adesso(partita)[0] == pytest.approx(0.75, abs=1e-9)
        sentita = round((dal / fs + (0.7 - PARTENZA_DELLA_MIRA) / VELOCITA_DELLA_MIRA) * fs)
        assert davanti[sentita].finita and davanti[sentita].finita_al == sentita + len(tacche[7])
        # Il rilascio della freccia su si perde: per la tastiera è su, e il battito lo prende come lasciato, al suo campione.
        tavolo.tasti_giu.discard(su)
        perso = alzo_dal + _campioni_della_corsa(motore, 0.0, 0.2)
        tavolo.porta_udibile_a(perso)
        assert partita.tasti_giu() == [("alzo", 1)]
        tavolo.battito()
        assert partita.tasti_giu() == [] and partita.mira.assi["alzo"].verso == 0
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert _valori_adesso(partita) == pytest.approx([0.75, 0.7, 0.5], abs=1e-9)

        # F7 porta il fuoco nella barra, per leggere: lì Fine va avanti, verso la barra, e la potenza non si muove.
        assert not _tasto(finestra, wx.WXK_F7).GetSkipped() and tavolo.fuoco() is finestra.barra
        fine = TASTO_DI_OGNI_VERSO[("potenza", 1)]
        assert _tasto(finestra, fine).GetSkipped() and partita.tasti_giu() == [] and partita.mira.assi["potenza"].verso == 0
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert _valori_adesso(partita) == pytest.approx([0.75, 0.7, 0.5], abs=1e-9) and partita.mira.pressioni == 2
        # F6 riporta il fuoco sulla Mira, col suo suono, e da lì Fine alza la potenza; tornato il fuoco nella barra con F7, col
        # tasto ancora giù, il rilascio arriva alla barra.
        suonati = len(tavolo.suonati)
        assert not _tasto(finestra, wx.WXK_F6).GetSkipped() and tavolo.fuoco() is finestra.controllo_mira
        assert [nome for nome, _voce in tavolo.suonati[suonati:]] == ["vai_alla_mira"]
        potenza_dal = motore.campione_udibile()
        tavolo.premi(fine)
        tavolo.porta_udibile_a(potenza_dal + _campioni_della_corsa(motore, 0.0, 0.2))
        tavolo.battito()
        assert not _tasto(finestra, wx.WXK_F7).GetSkipped() and tavolo.fuoco() is finestra.barra and partita.fase == "mira"
        tavolo.lascia(fine)
        valori = [0.75, 0.7, 0.7]
        assert _valori_adesso(partita) == pytest.approx(valori, abs=1e-9)
        _barra_della_mira(tavolo)
        # Il tempo passato ha portato l'uscita oltre il capo della direzione: le tacche e il capo che la destra aveva davanti
        # quando è salita sono finiti senza suonare.
        assert motore.campione_scritto > max(davanti)
        assert all(davanti[voce].finita and davanti[voce].finita_al == voce for voce in davanti if voce > sentita)
        # L'Invio dalla barra spara coi valori del momento; il fuoco, che F7 ha messo nella barra, resta lì.
        tavolo.invio()
        assert partita.fase == "volo" and partita.valori == pytest.approx(valori, abs=1e-9) and partita.mira is None
        assert tavolo.fuoco() is finestra.barra and partita._voce_del_puntatore is None
        assert finestra.statistiche.dati["conti"]["tasti_della_mira"] == 3 and finestra.statistiche.dati["conti"]["mire_sparate"] == 1
        nuovi = _volo(tavolo)
        assert nuovi[0].tipo == "resoconto" and nuovi[0].dati["tiro"] == 1
        assert f"; alzo {resoconti.gradi(partita.ultimo_volo.traiettoria.alzo)}, 70%; " in nuovi[0].testo
        _riposo(tavolo)

        # La mira seguente riporta il fuoco sulla Mira; Esc, col tasto sinistro giù, la annulla e non va avanti.
        riserva = partita.riserva
        tavolo.spazio()
        assert partita.fase == "mira" and tavolo.fuoco() is finestra.controllo_mira
        sinistra = TASTO_DI_OGNI_VERSO[("direzione", -1)]
        tavolo.premi(sinistra)
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert not _tasto(finestra, wx.WXK_ESCAPE).GetSkipped()
        assert partita.fase == "ascolto" and partita.riserva == riserva and partita.mira is None and partita.tasti_giu() == []
        assert tavolo.fase_della_barra() == "Ascolto: Spazio per mirare" and tavolo.fuoco() is finestra.vista
        assert tavolo.ultimo_lotto[-1].annuncio == "Mira annullata." and voce_finta[-1] == "Mira annullata." and tavolo.suonati[-1][0] == "mira_annullata"
        tavolo.cursore_sul_nuovo()
        # Fuori dalla mira il rilascio, Esc e i tasti dei valori vanno avanti, alla vista.
        tavolo.tasti_giu.discard(sinistra)
        assert _rilascio(finestra, sinistra).GetSkipped()
        assert _tasto(finestra, wx.WXK_ESCAPE).GetSkipped() and _tasto(finestra, sinistra).GetSkipped() and _tasto(finestra, wx.WXK_RETURN).GetSkipped()
        assert partita.fase == "ascolto" and partita.mira is None
        conti = finestra.statistiche.dati["conti"]
        assert (conti["mire_avviate"], conti["mire_annullate"], conti["mire_sparate"], conti["tiri"]) == (2, 1, 1, 1)
        assert messaggi_wx == []
        # Ctrl+N nella mira, col sì: la partita nuova comincia nell'ascolto, e il fuoco torna dalla Mira alla vista, dopo la domanda.
        tavolo.spazio()
        assert partita.fase == "mira" and tavolo.fuoco() is finestra.controllo_mira
        messaggi_wx.risposta = wx.YES
        _menu(finestra, finestra.nuova_partita)
        assert partita.fase == "ascolto" and partita.tiri == 0 and partita.mira is None and tavolo.fuoco() is finestra.vista
        assert len(messaggi_wx) == 1 and messaggi_wx[0][2] & wx.YES_NO
    finally:
        _chiudi(finestra)
    assert motore.chiuso


def test_la_mira_a_tempo_scade_fuori_dalla_palestra(app_wx, preset_finti, messaggi_wx, nomi, voce_finta, monkeypatch):
    """D29: fuori dalla palestra la mira ha il suo tempo, TEMPO_DELLA_MIRA_INIZIALE secondi al primo livello dopo, che la barra
    di stato conta e che un tick al secondo scandisce, sempre più acuto, in fila dall'avvio. Allo scadere il colpo parte da
    solo coi valori di quell'istante, col tasto ancora giù, il fuoco torna alla vista, e la mira scaduta si conta; il
    rilascio che arriva col proiettile in volo va avanti."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        fs = motore.fs
        tavolo.spazio()
        _riposo(tavolo)
        partita.livello = costanti.PALESTRA_FINO_AL_LIVELLO + 1
        secondi = tempo_della_mira(partita.livello)
        assert secondi == costanti.TEMPO_DELLA_MIRA_INIZIALE == 30
        prima = len(tavolo.nel_motore)
        tavolo.spazio()
        assert partita.fase == "mira" and tavolo.fuoco() is finestra.controllo_mira
        # Finiti gli avvisi della partita nuova, il tempo della mira parte dallo spazio, dove la mira comincia a correre.
        avvio = partita._mira_dal
        scadenza = avvio + round(secondi * fs)
        assert partita._scadenza == scadenza and avvio == partita._mira_corre_dal
        assert tavolo.barra()[-2:] == ["Direzione davanti, alzo 50,0 gradi, potenza 50%, 30 secondi", FASE_DELLA_MIRA]
        # I tick, in fila dall'avvio, uno al secondo fino al penultimo, ognuno col bip del suo secondo, che sale col tempo che scorre.
        tick = [voce for voce in tavolo.nel_motore[prima:] if voce.inizio > avvio]
        assert [voce.inizio for voce in tick] == [avvio + round(s * fs) for s in range(1, round(secondi))]
        assert all(np.array_equal(voce.buffer, tick_del_tempo(s, secondi, fs=fs)) for s, voce in enumerate(tick, 1))
        # La destra per un decimo di scala, poi il tempo passa e la barra lo conta.
        destra = TASTO_DI_OGNI_VERSO[("direzione", 1)]
        dal = motore.campione_udibile()
        tavolo.premi(destra)
        tavolo.porta_udibile_a(dal + _campioni_della_corsa(motore, 0.0, 0.1))
        tavolo.lascia(destra)
        tavolo.porta_udibile_a(avvio + 10.5 * fs)
        tavolo.battito()
        assert partita.fase == "mira" and partita.secondi_della_mira() == pytest.approx(19.5)
        riga = tavolo.barra()[-2]
        assert riga == _riga_dei_valori(partita) and riga == "Direzione 8,0 gradi a destra, alzo 50,0 gradi, potenza 50%, 20 secondi"
        # Home giù nove decimi di secondo prima dello scadere, tre decimi di scala: il tempo scade col tasto ancora giù.
        casa = TASTO_DI_OGNI_VERSO[("potenza", -1)]
        tavolo.porta_udibile_a(scadenza - _campioni_della_corsa(motore, 0.0, 0.3))
        tavolo.battito()
        tavolo.premi(casa)
        tavolo.porta_udibile_a(scadenza - 1)
        tavolo.battito()
        assert partita.fase == "mira" and tavolo.barra()[-2].endswith(", 1 secondo") and tavolo.fuoco() is finestra.controllo_mira
        assert all(voce.finita for voce in tick)
        suonati = len(tavolo.suonati)
        tavolo.porta_udibile_a(scadenza)
        tavolo.battito()
        assert partita.fase == "volo" and partita.valori == pytest.approx([0.6, 0.5, 0.2], abs=1e-9)
        assert [nome for nome, _voce in tavolo.suonati[suonati:]] == ["sparo"] and tavolo.fuoco() is finestra.vista
        assert tavolo.fase_della_barra() == "Volo"
        tavolo.tasti_giu.discard(casa)
        assert _rilascio(finestra, casa).GetSkipped()
        conti = finestra.statistiche.dati["conti"]
        assert (conti["mire_avviate"], conti["mire_sparate"], conti["mire_scadute"], conti["tasti_della_mira"]) == (1, 1, 1, 2)
        assert finestra.statistiche.dati["somme"]["secondi_di_mira"] == pytest.approx(secondi)
        # Fuori dalla palestra il tiro non ha attesa: il resoconto, poi l'ascolto.
        nuovi = _volo(tavolo)
        assert nuovi[0].tipo == "resoconto" and partita.fase == "ascolto" and partita.tiri == 1
        assert messaggi_wx == []
    finally:
        _chiudi(finestra)
    assert motore.chiuso


def test_un_dialogo_e_un_alt_tab_fermano_la_partita_e_la_riprendono(app_wx, preset_finti, messaggi_wx, nomi, voce_finta, monkeypatch):
    """D14: Ctrl+K durante la mira a tempo, col tasto della direzione giù. Il dialogo resta aperto a lungo, più di quanto
    la mira durerebbe, e il timer batte ancora, ma la partita è ferma e muta: la direzione si ferma dove si sentiva, il
    tasto vale come lasciato, il tempo della mira aspetta e nessun colpo parte. Chiuso il dialogo, il tempo della mira
    riprende coi secondi che restavano, il tasto rimasto giù non muove niente, e allo scadere il colpo parte da solo. Poi
    un Alt+Tab nell'ascolto: il tempo di gioco si ferma finché la finestra non torna attiva. Tutto in silenzio."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        fs = motore.fs
        tavolo.spazio()
        _riposo(tavolo)
        riserva = partita.riserva
        partita.livello = LIVELLO_DELLA_MIRA_BREVE
        secondi = tempo_della_mira(partita.livello)
        assert secondi == costanti.TEMPO_DELLA_MIRA_MINIMO < SECONDI_NEL_DIALOGO
        tavolo.spazio()
        assert partita.fase == "mira"
        avvio = partita._mira_dal
        sinistra = TASTO_DI_OGNI_VERSO[("direzione", -1)]
        dal = motore.campione_udibile()
        tavolo.premi(sinistra)
        tavolo.porta_udibile_a(dal + _campioni_della_corsa(motore, 0.0, 0.3))
        tavolo.battito()
        dentro = []

        def legge_a_lungo(dialogo):
            # Più di una mira intera, col battito che arriva anche dentro il dialogo, e il tasto sempre giù.
            for _blocco in range(SECONDI_NEL_DIALOGO * costanti.FS // costanti.AUDIO_BLOCCO):
                tavolo.avanza(1)
                tavolo.battito()
            # La mira è muta: né il puntatore né i tick del tempo.
            muta = partita._voce_del_puntatore is None and partita._tick_in_fila == []
            dentro.append((partita.in_pausa, muta, partita.tasti_giu(), _valori_adesso(partita), partita.secondi_della_mira(), partita.fase, partita.riserva))
            return wx.ID_OK

        monkeypatch.setattr(dialoghi.Lettura, "ShowModal", legge_a_lungo)
        suonati = len(tavolo.suonati)
        _menu(finestra, finestra.mostra_classifica)
        assert len(dentro) == 1
        in_pausa, muta, giu, valori, rimasti, fase, riserva_nel_dialogo = dentro[0]
        assert (in_pausa, muta, giu, fase, riserva_nel_dialogo) == (True, True, [], "mira", riserva)
        # La direzione si è fermata all'apertura del dialogo, dove si sentiva: tre decimi di scala a sinistra del centro.
        assert valori == pytest.approx([PARTENZA_DELLA_MIRA - 0.3, PARTENZA_DELLA_MIRA, PARTENZA_DELLA_MIRA], abs=1e-9)
        assert 0 < rimasti < secondi and partita.fase == "mira"
        # Chiuso il dialogo, con la finestra attiva, il battito riprende la partita: il tempo della mira riparte da dove era, coi tick
        # dei secondi che restano.
        tavolo.battito()
        ripresa = partita._mira_dal
        assert not partita.in_pausa and partita.fase == "mira" and partita._scadenza == ripresa + round(rimasti * fs)
        assert partita._tick_in_fila and all(ripresa < voce.inizio < partita._scadenza for voce in partita._tick_in_fila)
        assert partita.secondi_della_mira() <= rimasti and partita.tasti_giu() == [] and _valori_adesso(partita) == valori
        assert tavolo.barra()[-2:] == [_riga_dei_valori(partita), FASE_DELLA_MIRA]
        # Il tasto rimasto giù non muove niente, e il suo rilascio non cambia niente.
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        tavolo.battito()
        assert _valori_adesso(partita) == valori
        tavolo.lascia(sinistra)
        assert _valori_adesso(partita) == valori and partita.mira.pressioni == 1
        # Il tempo che restava scade, e il colpo parte da solo coi valori fermi.
        tavolo.porta_udibile_a(partita._scadenza)
        tavolo.battito()
        assert partita.fase == "volo" and partita.valori == pytest.approx(valori, abs=1e-12) and partita._mira_dal is None
        assert finestra.statistiche.dati["conti"]["mire_scadute"] == 1 and partita._scadenza is None
        assert avvio < ripresa
        nuovi = _volo(tavolo)
        assert nuovi[0].tipo == "resoconto" and partita.tiri == 1
        _riposo(tavolo)
        # Un Alt+Tab nell'ascolto: il tempo di gioco si ferma, e torna a scorrere col ritorno della finestra.
        attiva = [False]
        monkeypatch.setattr(finestra, "IsActive", lambda: attiva[0])
        tavolo.battito()
        assert partita.in_pausa and partita.fase == "ascolto"
        fermo = motore.tempo_di_gioco(motore.prossimo_campione())
        for _blocco in range(SECONDI_NEL_DIALOGO * costanti.FS // costanti.AUDIO_BLOCCO):
            tavolo.avanza(1)
            tavolo.battito()
        assert partita.in_pausa and motore.tempo_di_gioco() == pytest.approx(fermo)
        attiva[0] = True
        tavolo.battito()
        assert not partita.in_pausa
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert motore.tempo_di_gioco() > fermo
        # Le pause della finestra non suonano e non scrivono: né pausa né ripresa, e messaggi della partita solo i suoi.
        assert not {"pausa", "ripresa"} & {nome for nome, _voce in tavolo.suonati[suonati:]}
        assert "In pausa" not in finestra.vista.GetValue() and messaggi_wx == []
        assert finestra.statistiche.dati["conti"]["pause_automatiche"] == 2 and finestra.statistiche.dati["conti"].get("pause", 0) == 0
    finally:
        _chiudi(finestra)
    assert motore.chiuso


def test_i_secondi_della_mira_non_contano_la_pausa(app_wx, preset_finti, messaggi_wx, nomi, voce_finta, monkeypatch):
    """Il tempo medio di una mira, nelle statistiche, somma i secondi di ogni mira dall'avvio allo sparo (D29): come la durata
    della partita non conta le pause e le attese (D14, D22), e come il tempo della mira a tempo aspetta la ripresa, neanche i
    secondi della mira devono contare il tempo in cui la partita era ferma. Una mira della palestra: un secondo, poi cinque
    secondi di pausa chiesta con Ctrl+P, poi un altro secondo e l'Invio. La mira è durata due secondi, non sette."""
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        fs = motore.fs
        tavolo.spazio()
        _riposo(tavolo)
        tavolo.spazio()
        assert partita.fase == "mira" and partita.secondi_della_mira() is None
        # Nella palestra la mira non ha tempo: corre dallo spazio, e lì comincia a contare i suoi secondi.
        tavolo.porta_udibile_a(partita._mira_corre_dal + fs)
        tavolo.battito()
        _menu(finestra, finestra.pausa)
        assert partita.in_pausa and tavolo.suonati[-1][0] == "pausa"
        for _blocco in range(5 * costanti.FS // costanti.AUDIO_BLOCCO):
            tavolo.avanza(1)
            tavolo.battito()
        _menu(finestra, finestra.pausa)
        assert not partita.in_pausa and partita.fase == "mira"
        tavolo.porta_udibile_a(motore.prossimo_campione() + fs)
        tavolo.battito()
        tavolo.invio()
        assert partita.fase == "volo"
        assert finestra.statistiche.dati["somme"]["secondi_di_mira"] == pytest.approx(2.0, abs=0.1)
    finally:
        _chiudi(finestra)
    assert motore.chiuso


def test_il_proiettile_delle_impostazioni_nella_partita_vera(app_wx, preset_finti, messaggi_wx, voce_finta, monkeypatch):
    """D17 e D18 coi pezzi veri: la forma del proiettile scelta dal menu Impostazioni, col suo esempio che
    suona davvero nel motore, vale subito se nessuna partita è in corso, e i bersagli nascono in un'altra
    portante. Scelta a proiettile in volo, la forma nuova aspetta la partita seguente, e quella in corso
    tiene il suo proiettile fino alla fine. Le impostazioni e le statistiche lo ricordano."""
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        assert partita.proiettile == partita.proiettile_scelto == costanti.PROIETTILE_FORMA
        scelte = ["sega_morbida", "triangolo"]
        esempi = []

        def sceglie(dialogo):
            forma = scelte.pop(0)
            dialogo.elenco.SetSelection(costanti.PROIETTILE_FORME.index(forma))
            dialogo.prova()
            # L'esempio suona davvero, nella forma della riga, mentre il dialogo è aperto e la partita, se c'è, tace.
            voce = finestra._esempio_del_proiettile
            esempi.append(voce)
            uscita = flusso.avanza_fino(voce.inizio + motore.fs // 2)
            assert not voce.finita and np.abs(uscita[-motore.blocco:]).max() > 0
            dialogo.conferma()
            return dialogo.GetReturnCode()

        monkeypatch.setattr(dialoghi.SceltaProiettile, "ShowModal", sceglie)
        _menu(finestra, finestra.cambia_proiettile)
        assert partita.proiettile == partita.proiettile_scelto == "sega_morbida"
        assert esempi[0].forma == "sega_morbida" and finestra._esempio_del_proiettile is None
        # Chiuso il dialogo, l'esempio si ferma prima della sua fine naturale.
        flusso.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert esempi[0].finita and esempi[0].finita_al < esempi[0].inizio + round(modulo_partita.DURATA_DELL_ESEMPIO * motore.fs)
        assert finestra.vista.GetValue().endswith("\nIl proiettile adesso suona in sega morbida: lo senti dalla prossima partita.")

        # La partita: i bersagli in un'altra portante, il proiettile in sega morbida.
        tavolo.spazio()
        assert partita.fase == "ascolto" and partita.proiettile == "sega_morbida"
        assert all(partita._voci[b].forma == b.portante != "sega_morbida" for b in partita.bersagli)
        _riposo(tavolo)
        _bersaglio, valori = _valori_sul_centro(tavolo)
        _mira(tavolo, valori)
        assert partita._voce_proiettile.forma == "sega_morbida"
        # Col proiettile in volo si sceglie il triangolo: la partita in corso tiene la sega morbida.
        _menu(finestra, finestra.cambia_proiettile)
        assert (partita.proiettile, partita.proiettile_scelto) == ("sega_morbida", "triangolo") and esempi[1].forma == "triangolo"
        assert finestra.vista.GetValue().endswith("\nIl proiettile suonerà in triangolo dalla prossima partita: quella in corso resta in sega morbida.")
        tavolo.battito()
        assert not partita.in_pausa and partita.fase == "volo"
        nuovi = _volo(tavolo)
        assert nuovi[0].dati["colpito"]
        # Il bersaglio nato al posto del colpito, allo spazio della palestra (D22), evita ancora la forma della partita in corso.
        _riposo(tavolo)
        assert partita.bersagli and all(partita._voci[b].forma == b.portante != "sega_morbida" for b in partita.bersagli)
        _a_vuoto(tavolo)
        assert partita._voce_proiettile.forma == "sega_morbida"
        _volo(tavolo)
        _riposo(tavolo)

        # Ctrl+N, col sì: la partita nuova ha il proiettile in triangolo, e i bersagli un'altra portante.
        messaggi_wx.risposta = wx.YES
        _menu(finestra, finestra.nuova_partita)
        assert partita.fase == "ascolto" and partita.proiettile == "triangolo"
        assert all(partita._voci[b].forma == b.portante != "triangolo" for b in partita.bersagli)
        _riposo(tavolo)
        _a_vuoto(tavolo)
        assert partita._voce_proiettile.forma == "triangolo"
        _volo(tavolo)

        # Le impostazioni sul disco, le statistiche e i suoni.
        assert _leggi(costanti.FILE_IMPOSTAZIONI)["proiettile"] == "triangolo" and scelte == []
        conti = finestra.statistiche.dati["conti"]
        assert (conti["proiettile_cambiato"], conti["partite_con_la_sega_morbida"], conti["partite_col_triangolo"], conti["partite_col_seno"]) == (2, 1, 1, 0)
        assert [nome for nome, _voce in tavolo.suonati].count("proiettile") == 2
        assert len(messaggi_wx) == 1 and messaggi_wx[0][2] & wx.YES_NO
    finally:
        _chiudi(finestra)
    assert motore.chiuso


def test_il_livello_nuovo_e_il_bonus_suonano_in_fila(app_wx, preset_finti, messaggi_wx, nomi, voce_finta, monkeypatch):
    """Il centro che chiude il livello prima del livello dei bonus: il livello nuovo e il bonus che appare
    nascono nello stesso battito, e i loro avvisi suonano uno dopo l'altro, staccati; la vista e la
    sintesi li dicono nello stesso ordine. Fuori dalla palestra di D22, che qui la costante a zero
    toglie: nella palestra il bonus nascerebbe allo spazio dell'attesa. Poi il bonus, che continua a
    muoversi durante la mira (D29), si colpisce mirando dove sarà al passaggio del proiettile: vale i
    punti del bonus e dà COLPI_BONUS colpi in più, 8 da D30."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 1.0)
    monkeypatch.setattr(modulo_partita, "PALESTRA_FINO_AL_LIVELLO", 0)
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        tavolo.spazio()
        _riposo(tavolo)
        partita.livello = costanti.LIVELLO_DEI_BONUS - 1
        partita.centri_del_livello = CENTRI_PER_LIVELLO - 1
        _bersaglio, valori = _valori_sul_centro(tavolo)
        _mira(tavolo, valori)
        nuovi = _volo(tavolo)
        assert partita.livello == costanti.LIVELLO_DEI_BONUS and partita.bonus is not None
        assert [m.tipo for m in nuovi] == ["resoconto", "livello", "bonus"]
        livello, bonus = tavolo.voce("livello_su"), tavolo.voce("bonus_appare")
        assert bonus.inizio == livello.inizio + LUNGHEZZA_FINTA + round(modulo_partita.STACCO_FRA_GLI_AVVISI * motore.fs)
        assert partita._fine_degli_avvisi == bonus.inizio + LUNGHEZZA_FINTA + round(modulo_partita.STACCO_FRA_GLI_AVVISI * motore.fs)
        assert voce_finta[-2:] == [nuovi[1].annuncio, nuovi[2].annuncio]
        assert f" e dà {COLPI_BONUS} colpi in più." in nuovi[2].testo
        tavolo.cursore_sul_nuovo()
        assert tavolo.barra()[2] == f"Bersagli in volo: {len(partita.bersagli) - 1} e il bonus"
        # I messaggi arrivano un secondo dopo l'effetto (D24): gli avvisi sono già passati, e un suono nuovo parte subito.
        assert partita.dopo_gli_avvisi() is None

        # Il bonus colpito in volo: la mira va dove il bonus sarà al passaggio del proiettile, che parte all'Invio.
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        bersaglio, riserva, punti = partita.bonus, partita.riserva, partita.punti
        tavolo.spazio()
        assert partita.fase == "mira" and partita.secondi_della_mira() == tempo_della_mira(partita.livello)
        invio = motore.campione_udibile() + math.ceil(SECONDI_PRIMA_DELL_INVIO * motore.fs)
        partenza = _partenza_all_invio(tavolo, invio)
        valori = _valori_sul_mobile(partita, bersaglio, partenza)
        posto = bersaglio.posizione(motore.tempo_di_gioco())
        _mira(tavolo, valori, invio=invio)
        volo = partita.ultimo_volo
        assert volo.partenza == pytest.approx(partenza) and volo.colpito is bersaglio
        # Durante la mira il bonus si è mosso, e il colpo è partito verso dove sarebbe stato, non verso dove si sentiva.
        assert not np.allclose(bersaglio.posizione(partenza), posto)
        nuovi = _volo(tavolo)
        guadagno = punti_del_centro(bersaglio, volo.precisione, livello=costanti.LIVELLO_DEI_BONUS)
        assert nuovi[0].tipo == "resoconto" and nuovi[0].dati["bonus"] and nuovi[0].dati["punti"] == guadagno
        assert nuovi[0].annuncio.startswith(f"Colpito il bonus, {testi.intero(guadagno)} punti.")
        assert partita.riserva == nuovi[0].dati["riserva"] == riserva + COLPI_BONUS and partita.punti == punti + guadagno
        assert bersaglio not in partita.bersagli and partita.bonus is not bersaglio
        assert finestra.statistiche.dati["conti"]["bonus_colpiti"] == 1 and finestra.statistiche.dati["suoni"]["colpito_bonus"] == 1
    finally:
        _chiudi(finestra)


def test_la_palestra_e_i_menu_nella_finestra_vera(app_wx, messaggi_wx, nomi, voce_finta, monkeypatch):
    """D22 coi pezzi veri sul flusso finto. Il menu aperto con Alt ferma la partita, in silenzio, e la
    chiusura la riprende; Pausa scelta dal menu fa la pausa chiesta, che resta col menu chiuso. Un
    centro nella palestra: il resoconto arriva quando il jingle ha finito di suonare, nella vista e alla
    sintesi; la partita si ferma finiti gli avvisi, lo spazio nella grazia non conta, poi chiama il
    bersaglio nuovo; la durata non conta né l'attesa né le pause. Fuori dalla palestra Ctrl+P suona il
    non ora e la vista lo dice, il menu ferma la partita lo stesso, e il tiro a vuoto non ha attesa."""
    lungo = np.full((round(0.6 * costanti.FS), 2), AMPIEZZA_FINTA, dtype=np.float32)
    corto = np.full((LUNGHEZZA_FINTA, 2), AMPIEZZA_FINTA, dtype=np.float32)
    effetti = {suoni.EVENTI["colpito"], suoni.EVENTI["a_terra"]}
    monkeypatch.setattr(suoni, "_rendi_preset", lambda nome: lungo if nome in effetti else corto)
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    # Il menu aperto come lo dice Windows: la prova lo apre e lo chiude insieme agli eventi di wx.
    aperto = [False]
    monkeypatch.setattr(modulo_finestra, "menu_di_windows", lambda: aperto[0])
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([], orologio=lambda: flusso.time), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        tavolo.spazio()
        inizio = tavolo.adesso()
        _riposo(tavolo)
        suonati = len(tavolo.suonati)

        # Il menu aperto con Alt ferma la partita, e il battito la tiene ferma finché resta aperto.
        aperto[0] = True
        _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
        assert partita.in_pausa
        dal = tavolo.adesso()
        fermo = motore.tempo_di_gioco(motore.prossimo_campione())
        for _blocco in range(costanti.FS // costanti.AUDIO_BLOCCO):
            tavolo.avanza(1)
            tavolo.battito()
        assert partita.in_pausa and motore.tempo_di_gioco() == pytest.approx(fermo)
        aperto[0] = False
        _evento_del_menu(finestra, wx.wxEVT_MENU_CLOSE)
        wx.Yield()
        assert not partita.in_pausa
        in_pausa = tavolo.adesso() - dal
        # Pausa dal menu: wx chiude il menu prima del comando, e la pausa del menu diventa quella chiesta, che resta.
        aperto[0] = True
        _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
        aperto[0] = False
        _evento_del_menu(finestra, wx.wxEVT_MENU_CLOSE)
        _menu(finestra, finestra.pausa)
        wx.Yield()
        tavolo.battito()
        assert partita.in_pausa and tavolo.fase_della_barra() == "In pausa: Ctrl+P per riprendere"
        _menu(finestra, finestra.pausa)
        assert not partita.in_pausa and [nome for nome, _voce in tavolo.suonati[suonati:]] == ["pausa", "ripresa"]

        # Un centro nella palestra: il resoconto aspetta la fine del jingle, più il secondo di silenzio di D24.
        _bersaglio, valori = _valori_sul_centro(tavolo)
        _mira(tavolo, valori)
        evento = partita._evento_impatto
        da = len(tavolo.ricevuti)
        while partita.fase == "volo":
            tavolo.avanza(1)
            tavolo.battito()
        assert len(tavolo.ricevuti) == da and partita.fase == "attesa" and tavolo.fase_della_barra() == "Palestra: ascolta l'esito del tiro"
        arrivo = evento.campione + len(lungo) + round(costanti.RITARDO_DEL_RESOCONTO * motore.fs)
        tavolo.porta_udibile_a(arrivo - 1)
        tavolo.battito()
        assert len(tavolo.ricevuti) == da
        tavolo.porta_udibile_a(arrivo)
        tavolo.battito()
        nuovi = tavolo.ricevuti[da:]
        assert [m.tipo for m in nuovi] == ["resoconto", "info"] and nuovi[-1].testo == "Palestra: la partita aspetta lo Spazio, che chiama il bersaglio nuovo."
        assert voce_finta[-1] == nuovi[0].annuncio and finestra.vista.GetValue().endswith("\n" + nuovi[-1].testo)
        tavolo.cursore_sul_nuovo()
        # Lo spazio prima che la partita si fermi non conta e non suona.
        suonati = len(tavolo.suonati)
        tavolo.spazio()
        assert partita.fase == "attesa" and len(tavolo.suonati) == suonati
        assert partita._fine_degli_avvisi <= motore.campione_udibile()
        _fuori_dall_attesa(tavolo)
        assert len(partita.bersagli) == 1 and partita.attesa_ferma is False
        # In gioco fin qui: dall'inizio, meno il menu tenuto aperto e l'attesa della palestra.
        assert tavolo.in_attesa > 0 and in_pausa > 0
        assert partita.riepilogo()["durata_s"] == round(tavolo.adesso() - inizio - in_pausa - tavolo.in_attesa, 1)

        # Fuori dalla palestra: Ctrl+P suona il non ora, e la vista dice perché.
        partita.livello = costanti.PALESTRA_FINO_AL_LIVELLO + 1
        rifiuto = f"La pausa con Ctrl+P c'è soltanto nei livelli della palestra, fino al livello {costanti.PALESTRA_FINO_AL_LIVELLO}."
        _menu(finestra, finestra.pausa)
        assert not partita.in_pausa and tavolo.suonati[-1][0] == "non_ora"
        assert finestra.vista.GetValue().endswith("\n" + rifiuto) and voce_finta[-1] == "Pausa solo in palestra."
        # Il menu ferma la partita in ogni livello.
        aperto[0] = True
        _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
        tavolo.battito()
        assert partita.in_pausa
        aperto[0] = False
        _evento_del_menu(finestra, wx.wxEVT_MENU_CLOSE)
        wx.Yield()
        assert not partita.in_pausa
        # Il tiro a vuoto fuori dalla palestra non ha attesa: subito l'ascolto.
        _a_vuoto(tavolo)
        nuovi = _volo(tavolo)
        assert [m.tipo for m in nuovi] == ["resoconto"] and partita.fase == "ascolto" and tavolo.fase_della_barra() == "Ascolto: Spazio per mirare"
        conti = finestra.statistiche.dati["conti"]
        assert (conti["pause"], conti["pause_rifiutate"], conti["pause_automatiche"], conti["attese_della_palestra"]) == (1, 1, 3, 1)
        assert messaggi_wx == []
    finally:
        _chiudi(finestra)
    assert motore.chiuso


def test_ctrl_p_dal_tasto_rapido_nella_finestra_vera(app_wx, preset_finti, messaggi_wx, nomi, voce_finta, monkeypatch):
    """Ctrl+P da tastiera passa da TranslateAccelerator di Windows, e wx manda l'apertura del menu Partita,
    il comando e la chiusura, misurati su un desktop nascosto, ma per Windows nessun menu è aperto (D22).
    Fuori dalla palestra, nella mira, il rifiuto non tocca la mira in corso: il tasto resta giù, il valore
    continua a muoversi e il tempo della mira non si ferma, e nessuna pausa automatica si conta. Nella
    palestra la pausa è quella chiesta, contata una volta, e resta: il tasto vale come lasciato (D29)."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    monkeypatch.setattr(modulo_finestra, "menu_di_windows", lambda: False)
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([], orologio=lambda: flusso.time), motore=motore)

    def ctrl_p():
        _evento_del_menu(finestra, wx.wxEVT_MENU_OPEN)
        _menu(finestra, finestra.pausa)
        _evento_del_menu(finestra, wx.wxEVT_MENU_CLOSE)
        wx.Yield()
        tavolo.battito()

    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        tavolo.spazio()
        _riposo(tavolo)
        partita.livello = costanti.PALESTRA_FINO_AL_LIVELLO + 1
        tavolo.spazio()
        assert partita.fase == "mira"
        destra = TASTO_DI_OGNI_VERSO[("direzione", 1)]
        tavolo.premi(destra)
        mira, voce, scadenza = partita.mira, partita._voce_del_puntatore, partita._scadenza
        ctrl_p()
        assert not partita.in_pausa and tavolo.suonati[-1][0] == "non_ora" and voce_finta[-1] == "Pausa solo in palestra."
        assert partita.mira is mira and partita._voce_del_puntatore is voce and not voce.finita and partita._scadenza == scadenza
        assert partita.tasti_giu() == [("direzione", 1)] and tavolo.fuoco() is finestra.controllo_mira
        prima = _valori_adesso(partita)[0]
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert _valori_adesso(partita)[0] > prima
        conti = finestra.statistiche.dati["conti"]
        assert (conti.get("pause", 0), conti.get("pause_rifiutate", 0), conti.get("pause_automatiche", 0)) == (0, 1, 0)
        # Nella palestra Ctrl+P mette la pausa chiesta, col suo suono: la direzione si ferma, e il tasto vale come lasciato.
        partita.livello = 1
        ctrl_p()
        assert partita.in_pausa and not partita._pausa_automatica and tavolo.suonati[-1][0] == "pausa"
        assert partita.tasti_giu() == [] and partita._voce_del_puntatore is None and partita.fase == "mira"
        ferma = _valori_adesso(partita)
        tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
        assert _valori_adesso(partita) == ferma
        ctrl_p()
        assert not partita.in_pausa and tavolo.suonati[-1][0] == "ripresa" and _valori_adesso(partita) == ferma
        assert (conti.get("pause", 0), conti.get("pause_rifiutate", 0), conti.get("pause_automatiche", 0)) == (1, 1, 0)
    finally:
        _chiudi(finestra)
    assert motore.chiuso


def test_lo_spazio_della_partita_nuova_riavvia_il_battito_guasto(app_wx, preset_finti, messaggi_wx, monkeypatch):
    """Un battito guasto ferma il timer; la partita nuova lo fa ripartire, anche quando la comincia lo spazio."""
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        assert finestra.timer.IsRunning()
        guasto, battito = [True], tavolo.partita.battito

        def forse_guasto():
            if guasto[0]:
                raise ZeroDivisionError("battito guasto")
            return battito()

        monkeypatch.setattr(tavolo.partita, "battito", forse_guasto)
        with pytest.raises(ZeroDivisionError):
            tavolo.battito()
        assert not finestra.timer.IsRunning()
        guasto[0] = False
        tavolo.spazio()
        assert tavolo.partita.fase == "ascolto" and finestra.timer.IsRunning()
        tavolo.battito()
        # Lo spazio che avvia la mira, con la partita già in corso, lascia il timer com'è.
        finestra.timer.Stop()
        tavolo.spazio()
        assert tavolo.partita.fase == "mira" and not finestra.timer.IsRunning()
    finally:
        _chiudi(finestra)


# La scheda audio (D20).

def _apri_flusso_vera():
    """apri_flusso vera, da una copia di motore_audio: quella del modulo il conftest la sostituisce perché non apra la scheda."""
    specifica = importlib.util.spec_from_file_location("motore_audio_copia_d_integrazione", Path(motore_audio.__file__))
    copia = importlib.util.module_from_spec(specifica)
    specifica.loader.exec_module(copia)
    return copia.apri_flusso


def test_la_scheda_audio_scelta_nella_finestra_vera(app_wx, preset_finti, messaggi_wx, voce_finta, schede_finte, monkeypatch):
    """D20 coi pezzi veri: la finestra fa nascere il motore con la scheda delle impostazioni, e apri_flusso vera la cerca
    fra le uscite e la apre con un sounddevice finto. All'avvio le cuffie scelte non ci sono, e Ciechino suona sugli
    altoparlanti, l'automatica, e lo dice. Le cuffie si accendono a partita in corso: Ctrl+Maiusc+X ferma la partita,
    chiude il flusso, rilegge PortAudio a flusso chiuso, e il motore rientra nelle cuffie; nell'elenco si sceglie l'uscita
    digitale, che suona subito, resta sul disco e si conta, e la partita riprende e si gioca. Poi la scheda si stacca, il
    flusso si perde, e Ctrl+Maiusc+X ridà il suono anche se si annulla l'elenco."""
    sd_finto = sounddevice_finto(schede_finte)
    monkeypatch.setitem(sys.modules, "sounddevice", sd_finto)
    monkeypatch.setattr(motore_audio, "apri_flusso", _apri_flusso_vera())
    cuffie = {"dispositivo": CUFFIE, "interfaccia": "Windows WASAPI"}
    digitale = {"dispositivo": DIGITALE, "interfaccia": "Windows WASAPI"}
    sul_digitale = "Il suono esce da Realtek Digital Output (Realtek(R) Audio), WASAPI, con una latenza di 20 millesimi di secondo."
    with open(percorsi.percorso_dati(costanti.FILE_IMPOSTAZIONI), "w", encoding="utf-8") as file:
        json.dump({"scheda": cuffie}, file)
    # A ogni rilettura, quali flussi aperti fin lì erano già chiusi: la rilettura con un flusso aperto può far cadere Ciechino.
    schede_finte.alla_rilettura = lambda: [flusso.chiuso for flusso in sd_finto.aperti]
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]))
    motore = finestra.motore
    try:
        finestra.Show()
        assert motore.scelta == cuffie and motore.flusso.opzioni["device"] == 18
        perche = f"La scheda audio scelta, {CUFFIE}, WASAPI, non c'è: Ciechino suona su quella automatica."
        dove = f"Il suono esce da {ALTOPARLANTI}, WASAPI, con una latenza di 20 millesimi di secondo."
        riga = f"{perche} {dove}"
        # Nella vista il perché e il dove stanno su due righe; la sintesi li ha in una frase.
        righe = finestra.vista.GetValue().splitlines()
        assert righe[righe.index(perche) + 1] == dove
        tavolo = Tavolo(finestra, monkeypatch, flusso=motore.flusso)
        finestra.suoni_d_avvio()
        assert [nome for nome, _voce in tavolo.suonati] == ["avvio", "scheda_non_aperta"] and voce_finta == [riga]
        avviso = tavolo.voce("scheda_non_aperta")
        tavolo.flusso.avanza_fino(avviso.inizio + len(avviso.buffer))
        assert finestra.statistiche.dati["conti"]["scheda_non_aperta"] == 1

        # La partita, e le cuffie che si accendono.
        partita = tavolo.partita
        tavolo.spazio()
        _riposo(tavolo)
        assert partita.fase == "ascolto"
        schede_finte.arrivate = [CUFFIE_FINTE]
        visti = []

        def sceglie(dialogo):
            in_uso = [n for n, riga in enumerate(dialogo.righe) if riga.endswith(", in uso")]
            visti.append((partita.in_pausa, motore.flusso.opzioni["device"], in_uso, dialogo.elenco.GetSelection(), dialogo.righe[in_uso[0]]))
            if len(visti) == 1:
                dialogo.elenco.SetSelection(next(n for n, u in enumerate(dialogo.uscite, 1) if u["dispositivo"] == DIGITALE))
                dialogo.conferma()
                return dialogo.GetReturnCode()
            return wx.ID_CANCEL

        monkeypatch.setattr(dialoghi.SceltaScheda, "ShowModal", sceglie)
        # Ctrl+Maiusc+X: il gancio della finestra lo lascia ai menu, e il tasto rapido manda il comando.
        evento = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        evento.SetEventObject(finestra.vista)
        evento.SetKeyCode(ord("X"))
        evento.SetControlDown(True)
        evento.SetShiftDown(True)
        finestra.vista.GetEventHandler().ProcessEvent(evento)
        assert evento.GetSkipped()
        suonati = len(tavolo.suonati)
        _menu(finestra, finestra.scegli_la_scheda)
        # Nel dialogo la partita era ferma, e il motore era già rientrato nelle cuffie, la riga in uso e già scelta.
        assert visti[0][:4] == (True, 40, [visti[0][3]], visti[0][3]) and visti[0][4] == f"{CUFFIE}, WASAPI, 3 ms, in uso"
        # Alla rilettura il flusso degli altoparlanti era chiuso.
        assert schede_finte.riletture == [[True]]
        assert [nome for nome, _voce in tavolo.suonati[suonati:]] == ["scheda_audio", "confermato"]
        assert motore.scelta == digitale and motore.flusso.opzioni["device"] == 19 and motore.stato()["quale"] == "scelta"
        assert [flusso.chiuso for flusso in sd_finto.aperti] == [True, True, False]
        assert _leggi(costanti.FILE_IMPOSTAZIONI)["scheda"] == digitale
        assert finestra.vista.GetValue().endswith("\n" + sul_digitale) and voce_finta[-1] == sul_digitale
        conti = finestra.statistiche.dati["conti"]
        assert (conti["scheda_cambiata"], conti["scheda_non_aperta"]) == (1, 1)

        # La partita riprende sulla scheda nuova, e si gioca: il tiro arriva sul bersaglio.
        tavolo.flusso = motore.flusso
        tavolo.battito()
        assert not partita.in_pausa and partita.fase == "ascolto"
        assert np.abs(tavolo.flusso.avanza(BLOCCHI_DEGLI_EFFETTI)).max() > 0
        _riposo(tavolo)
        _bersaglio, valori = _valori_sul_centro(tavolo)
        _mira(tavolo, valori)
        nuovi = _volo(tavolo)
        assert nuovi[0].tipo == "resoconto" and nuovi[0].dati["colpito"]
        _riposo(tavolo)

        # La scheda si stacca: il flusso si perde, e la vista dice come riavere il suono.
        perso = ("La scheda audio non risponde più, forse si è staccata: Ciechino continua senza suono. Per riavere il suono scegli la scheda "
                 "con Ctrl+Maiusc+X.")
        tavolo.flusso.active = False
        tavolo.battito()
        assert motore.muto and motore.flusso_perso and finestra.vista.GetValue().endswith("\n" + perso) and voce_finta[-1] == perso
        suonati = len(tavolo.suonati)
        _menu(finestra, finestra.scegli_la_scheda)
        # La rilettura ha chiuso il flusso perso prima di rileggere, e ha ridato il suono già prima dell'elenco, sulla scheda scelta.
        assert schede_finte.riletture[-1] == [True, True, True]
        assert visti[1][1] == 19 and visti[1][2] == [visti[1][3]] and visti[1][4].startswith("Realtek Digital Output (Realtek(R) Audio), WASAPI, ")
        assert [nome for nome, _voce in tavolo.suonati[suonati:]] == ["scheda_audio", "annullato"]
        assert finestra.vista.GetValue().endswith("\nLa scelta della scheda audio non cambia. " + sul_digitale)
        stato = motore.stato()
        assert stato["aperto"] and not stato["perso"] and not motore.muto and motore.scelta == digitale
        assert _leggi(costanti.FILE_IMPOSTAZIONI)["scheda"] == digitale and conti["scheda_cambiata"] == 1
        tavolo.flusso = motore.flusso
        tavolo.battito()
        assert not partita.in_pausa and np.abs(tavolo.flusso.avanza(BLOCCHI_DEGLI_EFFETTI)).max() > 0
        assert messaggi_wx == []
    finally:
        _chiudi(finestra)
    assert motore.chiuso


# Il punto d'avvio.

def test_ciechino_si_importa_senza_effetti(monkeypatch, cartella_di_prova):
    """Importare Ciechino.py non apre finestre, non avvia fili né registri, non tocca i ganci degli errori
    e non scrive niente: tutto comincia da main."""
    monkeypatch.delitem(sys.modules, "Ciechino", raising=False)
    prima = set(sys.modules)
    fili = set(threading.enumerate())
    ganci = (sys.excepthook, threading.excepthook)
    app = wx.GetApp()
    modulo = importlib.import_module("Ciechino")
    assert callable(modulo.main) and modulo.__name__ == "Ciechino"
    # Le sole importazioni nuove sono della libreria standard: wx, la finestra, il registro e l'istanza unica
    # si importano dentro main e dentro avvia_finestra.
    nuovi = {nome.split(".")[0] for nome in set(sys.modules) - prima} - {"Ciechino"}
    assert nuovi <= set(sys.stdlib_module_names)
    assert set(threading.enumerate()) == fili and (sys.excepthook, threading.excepthook) == ganci and wx.GetApp() is app
    assert list(cartella_di_prova.iterdir()) == []


def test_main_con_ciechino_gia_aperto_esce_subito(monkeypatch, cartella_di_prova):
    """Con un Ciechino già aperto, main lo porta in primo piano con istanza.manda e torna zero, senza
    ascoltare, senza registro e senza finestra; con --attendi aspetta prima l'uscita di quel processo."""
    import Ciechino
    import istanza
    import registro

    fatti = []
    monkeypatch.setattr(istanza, "manda", lambda *argomenti: fatti.append("manda") or True)
    monkeypatch.setattr(istanza, "aspetta_l_uscita", lambda pid, *argomenti: fatti.append(("aspetta", pid)) or True)
    monkeypatch.setattr(istanza, "Ascolto", lambda *argomenti, **opzioni: fatti.append("ascolto"))
    monkeypatch.setattr(registro, "avvia", lambda *argomenti, **opzioni: fatti.append("registro"))
    monkeypatch.setattr(registro, "scrivi", lambda riga: fatti.append(riga))
    monkeypatch.setattr(registro, "chiudi", lambda: fatti.append("chiudi"))
    monkeypatch.setattr(Ciechino, "avvia_finestra", lambda *argomenti, **opzioni: fatti.append("finestra"))
    fili = set(threading.enumerate())
    assert Ciechino.main([]) == 0
    assert fatti == ["manda"]
    assert Ciechino.main([istanza.ATTENDI, "4321"]) == 0
    assert fatti == ["manda", ("aspetta", 4321), "manda"]
    assert set(threading.enumerate()) == fili and list(cartella_di_prova.iterdir()) == []
