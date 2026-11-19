"""
Le prove d'integrazione di Ciechino: la finestra principale vera con la partita vera, giocata dall'inizio
alla fine, col motore audio sul flusso finto o muto, e il punto d'avvio.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Le altre prove guardano un pezzo alla volta: la partita col motore ma senza la finestra, la finestra
con una partita finta. Qui stanno insieme tutti i pezzi veri, FinestraPrincipale, Partita, Motore,
Suoni, Statistiche, classifica e impostazioni, e la prova gioca come Gabriele: lo spazio è l'evento
wxEVT_CHAR_HOOK che il gancio dei tasti di wx crea per un tasto vero, rifatto dalla prova e mandato
alla vista, che ha il fuoco, così passa dal gestore vero della finestra, _tasto, con i suoi controlli
e il rinfresco della barra; SendInput lo manderebbe allo schermo vero. Il battito è il metodo che il
wx.Timer chiama, _al_battito, chiamato a mano: il timer si ferma, perché la prova decide lei quando
passa il tempo, e un battito arrivato da wx.Yield nel momento sbagliato la renderebbe casuale.
Nessuna prova apre la scheda audio né sintetizza un preset vero: _rendi_preset dà a ogni preset lo
stesso buffer breve. Col flusso finto di aiuti_audio il motore avanza soltanto quando la prova lo
chiede; per premere spazio a un campione preciso la prova fa preparare il blocco che lo contiene e
sposta l'orologio finto finché il campione udibile è proprio quello. Senza scheda il motore è quello
muto, col suo filo che chiama il callback quando l'orologio di motore_audio passa l'ora del blocco:
qui quell'orologio lo sposta la prova, a passi più corti di RITARDO_MASSIMO_MUTO, e aspetta che il
filo abbia preparato i blocchi dovuti. Così il percorso muto è quello vero, filo compreso, e la prova
resta esatta e svelta come col flusso finto, invece di aspettare il tempo vero o di indovinarlo.
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
import suoni
import testi
from costanti import AIUTO_MANCATO_FINO_AL_LIVELLO, CENTRI_PER_LIVELLO, COLPI_INIZIALI, COLPI_PER_LIVELLO, COMPENSAZIONE_DEL_TASTO, PARAMETRI
from gui import dialoghi
from gui.finestra import FinestraPrincipale
from partita import Partita, punti_del_centro
from statistiche import NOMI_DEI_PARAMETRI

SEME = 2026
NOME = "Gabriele"
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
# Quanti secondi di suono restano aperti un dialogo, o la finestra in secondo piano, nella prova della
# pausa automatica: più di una mira intera, che lasciata a sé scadrebbe in circa dodici secondi e mezzo.
SECONDI_NEL_DIALOGO = 15


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

def _tasto(finestra, codice, controllo=False):
    """Un tasto come lo vede wx: il gancio parte dalla vista, che ha il fuoco, e sale fino alla finestra."""
    evento = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
    evento.SetEventObject(finestra.vista)
    evento.SetKeyCode(codice)
    evento.SetControlDown(controllo)
    finestra.vista.GetEventHandler().ProcessEvent(evento)
    return evento


def _menu(finestra, comando):
    """Una voce dei menu, come la sceglie wx col suo tasto rapido: l'evento del menu, alla finestra."""
    evento = wx.CommandEvent(wx.wxEVT_MENU, finestra.voci_del_menu[comando].GetId())
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
    attorno a _ricevi, ogni suono con la sua voce, e ogni barra di stato letta."""

    def __init__(self, finestra, monkeypatch, flusso=None, orologio=None):
        self.finestra = finestra
        self.partita = finestra.partita
        self.motore = finestra.motore
        self.flusso = flusso
        self.orologio = orologio
        self.blocchi_per_battito = 1 if flusso is not None else BLOCCHI_PER_BATTITO_MUTO
        self.ricevuti = []
        # I messaggi dell'ultimo battito, o comando, che ne ha portati.
        self.ultimo_lotto = []
        self.suonati = []
        self.barre = []
        ricevi, suona = finestra._ricevi, finestra.suoni.suona

        def spia(messaggi, con_la_classifica=True):
            self.ricevuti.extend(messaggi or ())
            if messaggi:
                self.ultimo_lotto = list(messaggi)
            return ricevi(messaggi, con_la_classifica)

        def spia_dei_suoni(evento, *argomenti, **opzioni):
            voce = suona(evento, *argomenti, **opzioni)
            self.suonati.append((evento, voce))
            return voce

        monkeypatch.setattr(finestra, "_ricevi", spia)
        # La partita e la finestra suonano con gli stessi Suoni: la spia li sente tutti e due.
        monkeypatch.setattr(finestra.suoni, "suona", spia_dei_suoni)
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

    def spazio(self):
        """Lo spazio, dal gestore vero: deve fermarsi lì, senza andare avanti verso la vista."""
        evento = _tasto(self.finestra, wx.WXK_SPACE)
        assert not evento.GetSkipped()

    def battito(self):
        self.finestra._al_battito()

    def cursore_sul_nuovo(self):
        """Il cursore della vista sta all'inizio del primo messaggio dell'ultimo battito, o comando, che ne
        ha portati: di solito il resoconto del tiro, e sotto, in ordine, il livello, il bonus o l'ultimo colpo."""
        assert _riga_del_cursore(self.finestra.vista) == (self.ultimo_lotto[0].testo.splitlines()[0], 0)

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


def _campione_del_valore(partita, valore):
    """Il campione udibile a cui lo spazio sceglie quel valore nella barra in corso."""
    barra = partita.scivolate[PARAMETRI[partita.barra]]
    return partita.inizio_barra + barra.inizio_scivolata + valore * barra.campioni_scivolata + COMPENSAZIONE_DEL_TASTO * barra.fs


def _nel_conto(partita):
    """Un campione a metà dei bip della barra in corso, dove lo spazio sceglie il minimo."""
    barra = partita.scivolate[PARAMETRI[partita.barra]]
    return partita.inizio_barra + barra.inizio_scivolata / 2 + COMPENSAZIONE_DEL_TASTO * barra.fs


# Il tiro: la mira al campione, il volo fino al resoconto, il riposo.

def _valori_sul_centro(tavolo):
    """Avvia la mira e restituisce il bersaglio e i valori che portano il colpo sul suo centro, con la fisica."""
    partita = tavolo.partita
    assert partita.fase == "ascolto"
    bersaglio = next(b for b in partita.bersagli if not b.bonus)
    tavolo.spazio()
    assert partita.fase == "mira" and partita.barra == 0
    partenza = tavolo.motore.tempo_di_gioco(tavolo.motore.prossimo_campione())
    valori = fisica.soluzione(bersaglio.posizione(partenza))
    assert valori is not None
    return bersaglio, valori


def _mira(tavolo, valori):
    """Le tre barre, ciascuna attraversata nelle sue parti con la barra di stato che le dice; zero si
    sceglie nei bip, ogni altro valore nella scivolata."""
    partita = tavolo.partita
    for indice, valore in enumerate(valori):
        nome = NOMI_DEI_PARAMETRI[PARAMETRI[indice]]
        assert partita.fase == "mira" and partita.barra == indice
        assert tavolo.fase_della_barra() == f"Mira: {nome}, pausa"
        tavolo.porta_udibile_a(partita.inizio_barra - 1)
        tavolo.battito()
        assert tavolo.fase_della_barra() == f"Mira: {nome}, pausa"
        tavolo.porta_udibile_a(_nel_conto(partita))
        tavolo.battito()
        assert tavolo.fase_della_barra() == f"Mira: {nome}, bip"
        if valore:
            tavolo.porta_udibile_a(_campione_del_valore(partita, valore))
            tavolo.battito()
            assert tavolo.fase_della_barra() == f"Mira: {nome}, scivolata"
        tavolo.spazio()
        assert partita.valori[-1] == pytest.approx(valore, abs=1e-9)
    assert partita.fase == "volo" and tavolo.fase_della_barra() == "Volo"


def _volo(tavolo):
    """Il volo, qualche blocco e un battito alla volta, finché il tiro si risolve; restituisce i messaggi nuovi."""
    partita = tavolo.partita
    da = len(tavolo.ricevuti)
    for _giro in range(SECONDI_MASSIMI * costanti.FS // (costanti.AUDIO_BLOCCO * tavolo.blocchi_per_battito)):
        tavolo.avanza(tavolo.blocchi_per_battito)
        tavolo.battito()
        if partita.fase != "volo":
            return tavolo.ricevuti[da:]
        assert tavolo.fase_della_barra() in ("Volo", "In pausa: Ctrl+P per riprendere")
    raise AssertionError("il tiro non si è risolto")


def _riposo(tavolo):
    """Lascia finire gli effetti dell'ultimo tiro, e gli avvisi che la partita ha messo in fila dopo
    l'impatto, come l'ultimo colpo o il livello nuovo: poi nel motore restano soltanto le voci dei
    bersagli."""
    dopo = tavolo.partita.dopo_gli_avvisi()
    if dopo is not None:
        tavolo.avanza(-(-(dopo - tavolo.motore.campione_scritto) // tavolo.motore.blocco))
    tavolo.avanza(BLOCCHI_DEGLI_EFFETTI)
    assert tavolo.motore.stato()["voci"] == len(tavolo.partita.bersagli)


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
    assert resoconto.testo.startswith(f"Tiro {numero}, livello 1: colpito un bersaglio, forma ")
    assert resoconto.annuncio == f"Colpito, {testi.intero(punti)} punti."
    tavolo.cursore_sul_nuovo()
    return totale, nuovi


def _a_vuoto(tavolo):
    """Un tiro a vuoto voluto, tutto a sinistra, rasoterra e piano, scelto nei bip delle tre barre."""
    tavolo.spazio()
    _mira(tavolo, (0.0, 0.0, 0.0))
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

        # Lo spazio comincia la partita.
        tavolo.spazio()
        inizio = tavolo.adesso()
        assert partita.fase == "ascolto" and [m.tipo for m in tavolo.ricevuti] == ["info"]
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
            if numero < CENTRI_PER_LIVELLO:
                assert partita.livello == 1 and partita.fase == "ascolto"
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
            if partita.riserva > 1:
                assert [m.tipo for m in nuovi] == ["resoconto"] and partita.fase == "ascolto"
                assert resoconto.annuncio.startswith(f"A vuoto, riserva {partita.riserva}, ")
            elif partita.riserva == 1:
                assert [m.tipo for m in nuovi] == ["resoconto", "avviso"] and nuovi[1].annuncio == "Ultimo colpo."
            if partita.riserva:
                tavolo.cursore_sul_nuovo()
                _riposo(tavolo)
        fine = tavolo.adesso()

        # La fine: il resoconto, la fine col riepilogo, il nome e la classifica in coda alla vista.
        assert partita.fase == "finita" and not partita.in_corso()
        assert [m.tipo for m in nuovi] == ["resoconto", "fine"] and nuovi[0].annuncio == "A vuoto, riserva vuota."
        tiri = CENTRI_PER_LIVELLO + riserva
        durata = round(fine - inizio - in_pausa, 1)
        riepilogo = {"punteggio": totale, "livello": 2, "tiri": tiri, "centri": CENTRI_PER_LIVELLO, "durata_s": durata}
        assert nuovi[1].dati == riepilogo == partita.riepilogo()
        assert nuovi[1].annuncio == f"Partita finita, {testi.intero(totale)} punti."
        # Il suono della classifica segue quello della fine, in fila, staccato: partiti insieme si fonderebbero.
        fine_della_partita, ingresso = tavolo.voce("partita_finita"), tavolo.voce("in_classifica")
        assert ingresso.inizio == fine_della_partita.inizio + LUNGHEZZA_FINTA + round(modulo_partita.STACCO_FRA_GLI_AVVISI * motore.fs)
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
        # Ogni riga della barra, in ogni fase, sta nei quaranta caratteri del display braille.
        assert all(len(riga) <= 40 for righe in tavolo.barre for riga in righe)

        # La classifica, sul disco e nella vista, col cursore sulla voce nuova.
        voci = _leggi(costanti.FILE_CLASSIFICA)["voci"]
        assert len(voci) == 1
        assert {chiave: voci[0][chiave] for chiave in ("nome", *riepilogo)} == {"nome": NOME, **riepilogo}
        righe_della_classifica = classifica.righe(voci, nuova=1)
        nuova = f"1. {NOME}, {testi.intero(totale)} punti, oggi, nuova."
        assert righe_della_classifica[1] == nuova
        assert _riga_del_cursore(finestra.vista) == (nuova, 0)
        assert _leggi(costanti.FILE_IMPOSTAZIONI)["ultimo_nome"] == NOME

        # La vista: l'apertura, poi tutti i messaggi in ordine, una cosa per riga, poi la classifica.
        righe = [riga for m in tavolo.ricevuti for riga in m.testo.splitlines() if riga.strip()]
        testo = finestra.vista.GetValue()
        assert testo == "\n".join([apertura, *righe, *righe_della_classifica])
        posti = [testo.index(f"\nTiro {n}, livello ") for n in range(1, tiri + 1)]
        assert posti == sorted(posti)
        assert posti[CENTRI_PER_LIVELLO - 1] < testo.index("\nLivello 2! Il livello 1 è superato.") < posti[CENTRI_PER_LIVELLO]
        assert posti[-2] < testo.index("\nNella riserva resta l'ultimo colpo") < posti[-1] < testo.index("\nPartita finita: ")

        # La sintesi ha letto ogni annuncio, nell'ordine, e alla fine la voce nuova della classifica.
        assert voce_finta == [m.annuncio for m in tavolo.ricevuti if m.annuncio] + [nuova]
        assert voce_finta[0] == "Partita nuova: Spazio per mirare."
        assert sum(1 for detto in voce_finta if detto.startswith("Colpito, ")) == CENTRI_PER_LIVELLO

        # Le statistiche, sul disco, coi numeri della partita.
        statistiche = _leggi(costanti.FILE_STATISTICHE)
        # I contatori a zero non si confrontano: le statistiche ne partono di nuovi da zero quando la
        # pagina impara a dirli, e la prova guarda soltanto quelli che la partita ha mosso.
        assert {chiave: n for chiave, n in statistiche["conti"].items() if n} == {
            "avvii": 1, "partite_iniziate": 1, "partite_finite": 1, "tiri": tiri, "centri": CENTRI_PER_LIVELLO, "tiri_a_vuoto": riserva, "punti": totale,
            "livelli_superati": 1, "colpi_guadagnati": COLPI_PER_LIVELLO, "mire_avviate": tiri, "ingressi_in_classifica": 1, "pause": 1,
            # La partita si è giocata col proiettile di serie, il seno (D17).
            "partite_col_seno": 1}
        assert statistiche["suoni"] == {"nuova_partita": 1, "mira_avviata": tiri, "valore_preso": 3 * tiri, "sparo": tiri, "colpito": CENTRI_PER_LIVELLO,
                                        "a_terra": riserva, "livello_su": 1, "non_ora": 1, "pausa": 1, "ripresa": 1, "ultimo_colpo": 1,
                                        "partita_finita": 1, "in_classifica": 1, "confermato": 1}
        for parametro in PARAMETRI:
            presi = statistiche["mira"][parametro]
            assert (presi["scelto"], presi["minimo"], presi["massimo"], presi["nei_centri"]) == (CENTRI_PER_LIVELLO, riserva, 0, CENTRI_PER_LIVELLO)
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
    tiri a vuoto fino alla riserva vuota, la classifica e le statistiche."""
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
        riepilogo = {"punteggio": totale, "livello": 1, "tiri": COLPI_INIZIALI + 1, "centri": 1, "durata_s": round(tavolo.adesso() - inizio, 1)}
        assert partita.riepilogo() == riepilogo == nuovi[-1].dati
        assert len(nomi) == 1
        voci = _leggi(costanti.FILE_CLASSIFICA)["voci"]
        assert [{chiave: voce[chiave] for chiave in ("nome", *riepilogo)} for voce in voci] == [{"nome": NOME, **riepilogo}]
        nuova = classifica.righe(voci, nuova=1)[1]
        assert _riga_del_cursore(finestra.vista) == (nuova, 0) and finestra.vista.GetValue().endswith("\n" + nuova)
        assert voce_finta == [m.annuncio for m in tavolo.ricevuti if m.annuncio] + [nuova]
        statistiche = _leggi(costanti.FILE_STATISTICHE)
        conti = statistiche["conti"]
        assert (conti["partite_finite"], conti["tiri"], conti["centri"], conti["tiri_a_vuoto"], conti["punti"], conti["ingressi_in_classifica"]) == (
            1, COLPI_INIZIALI + 1, 1, COLPI_INIZIALI, totale, 1)
        assert (statistiche["suoni"]["sparo"], statistiche["suoni"]["a_terra"], statistiche["suoni"]["colpito"]) == (COLPI_INIZIALI + 1, COLPI_INIZIALI, 1)
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


def test_un_dialogo_e_un_alt_tab_fermano_la_partita_e_la_riprendono(app_wx, preset_finti, messaggi_wx, nomi, voce_finta, monkeypatch):
    """D14: Ctrl+K durante la scivolata della direzione. Il dialogo resta aperto a lungo e il timer batte
    ancora, ma la partita è ferma e muta: nessuna barra scade e nessun colpo parte. Chiuso il dialogo,
    la barra della direzione riparte da capo, col suo silenzio, e la mira si finisce. Poi un Alt+Tab
    nell'ascolto: il tempo di gioco si ferma finché la finestra non torna attiva. Tutto in silenzio."""
    motore, flusso = motore_finto()
    finestra = FinestraPrincipale(fabbrica_della_partita=_fabbrica([]), motore=motore)
    try:
        finestra.Show()
        tavolo = Tavolo(finestra, monkeypatch, flusso=flusso)
        partita = tavolo.partita
        tavolo.spazio()
        _riposo(tavolo)
        riserva = partita.riserva
        tavolo.spazio()
        assert partita.fase == "mira" and partita.barra == 0
        tavolo.porta_udibile_a(_campione_del_valore(partita, 0.5))
        tavolo.battito()
        assert tavolo.fase_della_barra() == "Mira: direzione, scivolata"
        dentro = []

        def legge_a_lungo(dialogo):
            # Più di una mira intera, col battito che arriva anche dentro il dialogo.
            for _blocco in range(SECONDI_NEL_DIALOGO * costanti.FS // costanti.AUDIO_BLOCCO):
                tavolo.avanza(1)
                tavolo.battito()
            dentro.append((partita.in_pausa, list(partita.valori), partita.fase, partita.riserva))
            return wx.ID_OK

        monkeypatch.setattr(dialoghi.Lettura, "ShowModal", legge_a_lungo)
        suonati = len(tavolo.suonati)
        _menu(finestra, finestra.mostra_classifica)
        assert dentro == [(True, [], "mira", riserva)]
        # Chiuso il dialogo, con la finestra attiva, il battito riprende la partita: la barra riparte da capo.
        tavolo.battito()
        assert not partita.in_pausa and partita.fase == "mira" and partita.barra == 0 and partita.valori == []
        assert partita.inizio_barra > motore.campione_udibile() and tavolo.fase_della_barra() == "Mira: direzione, pausa"
        _mira(tavolo, (0.0, 0.0, 0.0))
        nuovi = _volo(tavolo)
        assert nuovi[0].tipo == "resoconto" and partita.riserva == riserva - 1
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
        # Il bersaglio nato al posto del colpito evita ancora la forma della partita in corso.
        assert partita.bersagli and all(partita._voci[b].forma == b.portante != "sega_morbida" for b in partita.bersagli)
        _riposo(tavolo)
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
    sintesi li dicono nello stesso ordine."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 1.0)
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
        assert partita.dopo_gli_avvisi() == bonus.inizio + LUNGHEZZA_FINTA + round(modulo_partita.STACCO_FRA_GLI_AVVISI * motore.fs)
        assert voce_finta[-2:] == [nuovi[1].annuncio, nuovi[2].annuncio]
        tavolo.cursore_sul_nuovo()
        assert tavolo.barra()[2] == f"Bersagli in volo: {len(partita.bersagli) - 1} e il bonus"
        # Passati gli avvisi, un suono nuovo parte subito.
        flusso.avanza_fino(partita.dopo_gli_avvisi())
        assert partita.dopo_gli_avvisi() is None
    finally:
        _chiudi(finestra)


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
