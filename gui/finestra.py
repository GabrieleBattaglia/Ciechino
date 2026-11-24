"""
La finestra principale di Ciechino.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello della finestra principale di MESS: due aree di testo in
sola lettura, la vista della partita, grande, e sotto la barra di stato, di poche righe; nessun albero
dei comandi, ma i menu, ciascuna voce con il suo tasto rapido, nati da una tabella sola, da cui nasce
anche la guida ai comandi, così le due cose non possono andare d'accordo a metà.
Diversamente da MESS la vista non si riscrive a ogni comando: si allunga. Ogni messaggio della partita,
e ogni risposta della finestra, va in coda, col cursore all'inizio del testo nuovo, così chi legge con
le frecce trova subito l'ultimo resoconto e, risalendo, tutti quelli di prima. L'annuncio breve del
messaggio va alla sintesi, con GBsintesi, se l'impostazione lo vuole.
Lo spazio è il comando del gioco: si prende alla pressione nel gancio dei tasti della finestra,
EVT_CHAR_HOOK, che wx chiama prima dei controlli e dei tasti rapidi dei menu, e non va avanti, così il
carattere non arriva all'area di testo e NVDA non lo ripete; nella mira, allo stesso modo, Esc la
annulla e conserva il proiettile (D25). Un timer di BATTITO_MS fa avanzare la
partita, porta i suoi messaggi nella vista e rinfresca la barra, che si riscrive soltanto quando cambia,
tenendo il cursore sulla stessa riga e nella stessa colonna.
La finestra possiede il motore audio, i suoni, le statistiche e le impostazioni; la partita la crea con
una fabbrica, che le prove sostituiscono con una partita finta. A partita finita chiede il nome a chi
entra in classifica e mette la classifica in coda alla vista. Come MESS ha l'istanza unica, il registro
degli errori e l'aggiornamento automatico, coi quattro metodi che aggiornamento.py si aspetta e
l'attributo chiusa, che diventa vero quando la finestra comincia a chiudersi.
Il dialogo del nome si apre quando il suono dell'ingresso in classifica è finito: nell'attesa conta
già come un dialogo aperto, così lo spazio non comincia una partita nuova.
La partita si ferma da sola, in silenzio, quando si apre un dialogo o un messaggio, o un menu, con
Alt, F10 o il mouse (D22), o quando la finestra perde l'attivazione, e riprende da sola al ritorno, ma
soltanto se la pausa l'aveva messa la finestra: una pausa chiesta con Ctrl+P resta (D14). Il menu
aperto lo dicono gli eventi dei menu di wx e, a ogni battito, Windows stesso, così un evento perso o
due eventi che non si appaiano non lasciano la partita ferma, né la fanno ripartire a menu aperto.
A ogni battito la finestra chiede anche al motore audio se il flusso è vivo, e ne legge le notizie:
gli errori del filo dell'audio, i buchi e il flusso perso finiscono nel registro, riassunti, perché il
filo di PortAudio non arriva alla rete degli errori.
La scheda audio la sceglie chi gioca (D20, sul modello della voce Scheda audio di MeTeOra): il motore
nasce con la scheda delle impostazioni, e se quella non c'è o non si apre la vista e la sintesi lo
dicono, e Ciechino suona su quella automatica, senza cancellare la scelta, perché le cuffie possono
tornare. Ctrl+Maiusc+X, Scheda audio del menu Impostazioni, fa rileggere le schede al motore, così
compaiono anche le cuffie Bluetooth accese dopo l'avvio, e il suono perso torna; poi apre l'elenco.
Gli Ascolti, il tutorial audio (D23), si aprono da Ascolti del menu Aiuto, l'unica voce dalla 0.11.10, e da soli
all'avvio, una volta, finché le statistiche non contano una partita cominciata: dopo i suoni dell'avvio, a
finestra attiva e senza altri dialoghi; la proposta dell'aggiornamento li aspetta.
"""

import contextlib
import datetime
import functools
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import NamedTuple

import wx
from GBwx import dentro_area_utile

import aggiornamento
import classifica
import costanti
import impostazioni as modulo_impostazioni
import istanza
import lingua
import percorsi
import pub
import registro
import schede
import suoni as modulo_suoni
import testi
from gui import dialoghi
from lingua import _, ngettext, plurale
from motore_audio import Motore
from statistiche import Statistiche

TITOLO = costanti.NOME
# La misura della finestra ripristinata, in pixel al 100%: dentro_area_utile la ferma all'area utile
# dello schermo, e la finestra si apre comunque massimizzata.
MISURA = (1024, 768)
# Quante righe si vedono nella barra di stato senza scorrere: quelle della partita stanno qui, e se
# un giorno fossero di più la barra scorre.
RIGHE_DELLA_BARRA = 5
# La curva del volume: da 1 a 100 la scala è lineare in decibel e copre tanti decibel, così ogni
# passo cambia il volume che l'orecchio sente sempre della stessa misura; zero è il silenzio.
DINAMICA_DEL_VOLUME_DB = 40.0
# Il silenzio fra il suono dell'avvio e quello della notizia che lo segue, in secondi: come in MESS,
# perché i due suoni non si confondano.
PAUSA_DOPO_L_AVVIO = 1.0
# Quanto aspetta al più l'uscita, in secondi, che il suono del congedo finisca prima di chiudere il motore.
ATTESA_MASSIMA_DELL_USCITA = 2.0
# Ogni quanti millesimi la chiusura per l'aggiornamento riprova, finché un messaggio resta aperto, e
# dopo quanti secondi smette: lo script di sostituzione aspetta la chiusura al massimo 30 secondi.
RIPROVA_DELLA_CHIUSURA = 200
LIMITE_DELLA_CHIUSURA = 20
# Il dialogo del nome aspetta la fine del suono dell'ingresso, sull'orologio del motore; se quell'orologio
# si fermasse, si apre comunque dopo tanti secondi veri oltre la fine prevista. All'uscita, invece, si
# aspetta quel suono al più per tanti secondi, fermi, perché il battito non c'è più.
MARGINE_DEL_NOME = 3.0
ATTESA_MASSIMA_DEL_NOME = 3.0
# Ogni quanti secondi, al più, il registro riceve il riassunto dei buchi del flusso audio e degli errori
# ripetuti del filo dell'audio: un buco o un guasto si ripete anche a ogni blocco, e una riga per volta
# riempirebbe il registro.
INTERVALLO_DEI_RIASSUNTI = 60.0
MILLESIMI = 1000
MEGABYTE = 1024 * 1024
# I tasti fuori dai menu, che il gancio dei tasti prende da sé.
TASTI_DEL_FUOCO = (wx.WXK_F5, wx.WXK_F6, wx.WXK_F7)
# I tasti della mira manuale (D29), col valore che muovono e il verso: le frecce orizzontali la direzione, quelle
# verticali l'alzo, Home e Fine la potenza; Invio spara. Il tastierino resta fuori, perché NVDA se lo tiene.
TASTI_DELLA_MIRA = {wx.WXK_LEFT: ("direzione", -1), wx.WXK_RIGHT: ("direzione", 1), wx.WXK_UP: ("alzo", 1), wx.WXK_DOWN: ("alzo", -1),
                    wx.WXK_HOME: ("potenza", -1), wx.WXK_END: ("potenza", 1)}
TASTO_DI_OGNI_VERSO = {valore: codice for codice, valore in TASTI_DELLA_MIRA.items()}
TASTI_DELLO_SPARO = (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER)
# Il suono e l'icona di ogni esito dell'aggiornamento.
SUONI_DEGLI_ESITI = {aggiornamento.PRONTO: "aggiornamento_pronto", aggiornamento.NON_RIUSCITO: "aggiornamento_non_riuscito",
                     aggiornamento.NON_PRONTO: "aggiornamento_non_pronto"}
ICONE_DEGLI_ESITI = {aggiornamento.NON_RIUSCITO: wx.ICON_ERROR, aggiornamento.NON_PRONTO: wx.ICON_WARNING}
# Come si è chiusa la finestra, per Ciechino.py e il registro.
USCITA = "uscita"
RIAPERTURA = "riapertura"
PER_AGGIORNARE = "aggiornamento"
# I segni di GetGUIThreadInfo di Windows che dicono un menu aperto (D22): il filo è nel giro di un menu,
# anche soltanto con la barra dei menu evidenziata da Alt o da F10, senza tendina; nel menu di sistema,
# Alt+Spazio; in un menu a comparsa, come quello del tasto Applicazioni.
GUI_INMENUMODE = 0x4
GUI_SYSTEMMENUMODE = 0x8
GUI_POPUPMENUMODE = 0x10
SEGNI_DEL_MENU = GUI_INMENUMODE | GUI_SYSTEMMENUMODE | GUI_POPUPMENUMODE


def guadagno_del_volume(valore):
    """
    Il volume delle impostazioni, da VOLUME_MINIMO a VOLUME_MASSIMO, come guadagno lineare del motore.
    Il minimo è il silenzio; sopra, la scala è lineare in decibel, da meno DINAMICA_DEL_VOLUME_DB al
    minimo più uno fino a zero decibel al massimo: 100 vale 1, 50 vale meno 20 decibel, cioè un decimo,
    e ogni dieci passi sono 4 decibel. L'orecchio sente i volumi così, a rapporti: con una scala lineare
    in ampiezza la metà alta del cursore cambierebbe pochissimo, e quella bassa tutto.
    """
    minimo, massimo = modulo_impostazioni.VOLUME_MINIMO, modulo_impostazioni.VOLUME_MASSIMO
    v = min(max(float(valore), minimo), massimo)
    if v <= minimo:
        return 0.0
    return 10 ** ((v - massimo) / (massimo - minimo) * DINAMICA_DEL_VOLUME_DB / 20)


def _partita_vera(motore, suoni, statistiche):
    """La fabbrica della partita di Ciechino: partita.py si importa qui, quando serve."""
    from partita import Partita

    return Partita(motore, suoni, statistiche)


@functools.cache
def _chiedi_a_windows():
    """La funzione di Windows che dice lo stato dei menu del filo, con la sua struttura, in una copia di
    user32 tutta sua, perché le firme non tocchino quelle di altri moduli; None fuori da Windows."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class GUITHREADINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
                    ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND), ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
                    ("rcCaret", wintypes.RECT)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetGUIThreadInfo.restype = wintypes.BOOL
    user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    kernel32.GetCurrentThreadId.argtypes = []
    return user32.GetGUIThreadInfo, kernel32.GetCurrentThreadId, GUITHREADINFO, ctypes.byref, ctypes.sizeof


def menu_di_windows():
    """
    Vero se il filo che chiama, quello della finestra, ha un menu aperto secondo Windows, anche soltanto
    la barra dei menu evidenziata da Alt; falso se non ce l'ha; None se non lo si sa, fuori da Windows o
    se la domanda non riesce (D22). Windows lo sa sempre: un evento dei menu di wx perso non lo inganna.
    """
    try:
        funzioni = _chiedi_a_windows()
        if funzioni is None:
            return None
        chiedi, filo, struttura, riferimento, misura = funzioni
        stato = struttura()
        stato.cbSize = misura(struttura)
        if not chiedi(filo(), riferimento(stato)):
            return None
        return bool(stato.flags & SEGNI_DEL_MENU)
    except Exception:  # noqa: BLE001 - il battito chiede a ogni giro: senza risposta valgono gli eventi di wx
        return None


def frase_della_prima_apertura():
    """La riga della vista quando gli Ascolti si aprono da soli all'avvio (D23): che è la prima volta, che Esc li chiude e dove si ritrovano."""
    return _("Sembra la prima volta che apri Ciechino: ti porto negli Ascolti, la sala briefing dei suoni, che Esc chiude e che ritrovi quando vuoi nel "
             "menu Aiuto.")


def _senza_tasto(testo):
    """Il testo di una voce di menu per la guida: senza la e commerciale e senza i puntini del dialogo che apre."""
    return testo.replace("&&", "\x00").replace("&", "").replace("\x00", "&").rstrip(".")


class Voce(NamedTuple):
    """Una voce dei menu: il testo, il tasto rapido o None, il comando e se è una voce a spunta."""

    testo: str
    tasto: str | None
    comando: Callable
    spunta: bool = False


class NomeInAttesa(NamedTuple):
    """
    Una partita entrata in classifica che aspetta il dialogo del nome: fine è il campione dove finisce il
    suono dell'ingresso, o None se non suona; scadenza l'istante di time.monotonic dopo cui il dialogo si
    apre comunque; poi il riepilogo, le voci della classifica e la posizione della voce nuova.
    """

    fine: int | None
    scadenza: float
    riepilogo: dict
    voci: list
    posizione: int


class ControlloDellaMira(wx.Panel):
    """Il controllo che ha il fuoco durante la mira manuale (D29): un pannello vuoto, che non è un testo,
    così NVDA lascia passare frecce, Home e Fine come pressioni vere, giù finché il dito non sale. Accetta
    il fuoco, ma non da Tab: ci arriva la finestra quando la mira comincia. Il nome è quello che NVDA
    dice."""

    def __init__(self, genitore, nome):
        super().__init__(genitore, style=wx.WANTS_CHARS)
        self.SetName(nome)
        self.SetLabel(nome)
        self.SetMinSize(wx.Size(-1, self.GetCharHeight()))

    def AcceptsFocus(self):
        return True

    def AcceptsFocusFromKeyboard(self):
        return False


class FinestraPrincipale(wx.Frame):
    """
    La finestra di Ciechino. crash_precedente è il resoconto del crash della volta prima, che la vista
    dice all'apertura. fabbrica_della_partita(motore, suoni, statistiche) crea la partita: senza, quella
    di partita.py. motore è il motore audio, già avviato o da avviare, con la sua scelta della scheda:
    senza, la finestra ne crea uno con la scheda delle impostazioni (D20).
    """

    def __init__(self, crash_precedente=None, fabbrica_della_partita=None, motore=None):
        super().__init__(None, title=TITOLO)
        self.crash_precedente = crash_precedente
        # chiusa diventa vera quando la finestra comincia a chiudersi: il filo dell'aggiornamento non le
        # chiede più niente. congedo dice a Ciechino.py come se n'è andata.
        self.chiusa = False
        self.congedo = USCITA
        # Chi vuole saperlo, cioè Ciechino.py per l'istanza unica, mette qui una funzione senza argomenti:
        # la finestra la chiama quando comincia a chiudersi davvero, dopo la domanda.
        self.al_congedo = None
        self._per_aggiornare = False
        self._per_riaprire = False
        self._chiusura_dal = None
        self._modali = 0
        # Il menu aperto secondo gli eventi dei menu di wx (D22): vale da solo soltanto dove Windows non sa dirlo.
        self._menu_dagli_eventi = False
        self._comandi_in_corso = 0
        self._nel_battito = False
        # Se la partita era nella mira al giro di prima, perché il fuoco la segua (D29).
        self._in_mira = False
        self._errore_aperto = False
        # I lavori del battito che si sono guastati: il registro riceve una volta sola ogni guasto uguale,
        # e chi gioca un messaggio solo per lavoro, perché il battito li ripeterebbe cento volte al secondo.
        self._guasti_detti = set()
        self._guasti_scritti = set()
        # I dati delle partite finite che aspettano la classifica, perché arrivate con un dialogo aperto.
        self._partite_da_concludere = []
        # La partita entrata in classifica che aspetta la fine del suono dell'ingresso per chiedere il
        # nome, e se lo si chiede all'uscita, senza battito, aspettando quel suono da fermi.
        self._nome_in_attesa = None
        self._all_uscita = False
        # Le notizie del filo dell'audio ancora da riassumere nel registro: per ogni errore, già scritto
        # una volta, quante volte si è ripetuto; i buchi, da quando li si è scritti e in tutto.
        self._errori_del_motore = {}
        self._buchi_da_dire = 0
        self._buchi_in_tutto = 0
        self._ultimo_riassunto = time.monotonic()
        self._flusso_perso_detto = False
        self._testo_barra = None
        # La riga della barra che dice lo scaricamento dell'aggiornamento, quando c'è.
        self._riga_dell_aggiornamento = None
        self._sintesi = None
        # Gli Ascolti all'avvio (D23): se aspettano di aprirsi da soli, il campione dove finiscono i suoni
        # dell'avvio, e il segnale che dice al filo dell'aggiornamento quando la sua proposta può arrivare.
        self._ascolti_d_avvio = False
        self._fine_dei_suoni_d_avvio = None
        self._avvio_libero = threading.Event()
        self._avvio_libero.set()
        self.impostazioni = modulo_impostazioni.carica()
        self.avvisi_dei_dati = [modulo_impostazioni.ultimo_avviso] if modulo_impostazioni.ultimo_avviso else []
        # Statistiche.carica conta da sé l'avvio.
        self.statistiche, avviso = Statistiche.carica()
        if avviso:
            self.avvisi_dei_dati.append(avviso)
        self._voci_della_classifica, avviso = classifica.carica()
        if avviso:
            self.avvisi_dei_dati.append(avviso)
        # Vero dopo un salvataggio della classifica non riuscito: allora vale quella in memoria, fino all'uscita.
        self._classifica_in_memoria = False
        # Il motore nasce con la scheda delle impostazioni, lette qui sopra (D20).
        self.motore = motore if motore is not None else Motore(scelta=self.impostazioni["scheda"])
        if not self.motore.avviato:
            self.motore.avvia()
        # Perché la scheda non si è aperta, o che cosa non è andato all'avvio del motore: la vista dice
        # soltanto che il gioco è muto, o che la scheda scelta non si è aperta, e il motivo lo legge chi
        # sviluppa, nel registro.
        ultimo_errore = getattr(self.motore, "ultimo_errore", None)
        if ultimo_errore:
            registro.scrivi(f"Motore audio: {ultimo_errore}")
        # Com'è andata l'apertura della scheda all'avvio: se la scheda scelta non c'era o non si è
        # aperta, lo dicono la vista, il suono e la sintesi, quando la finestra compare.
        self._esito_all_avvio = dict(self._esito_della_scheda())
        if self._scelta_mancata(self._esito_all_avvio):
            self.statistiche.conta("scheda_non_aperta")
        self.motore.volume = guadagno_del_volume(self.impostazioni["volume"])
        self.suoni = modulo_suoni.Suoni(self.motore, conta=self.statistiche.suono)
        # Tutti i preset si sintetizzano adesso, così il primo sparo non aspetta la sintesi di un rumore.
        self.suoni.precarica()
        self.partita = (fabbrica_della_partita or _partita_vera)(self.motore, self.suoni, self.statistiche)
        # Il posto in classifica nella situazione dopo ogni tiro (D33) si misura sulla classifica di
        # adesso, quella in memoria, che la fine di ogni partita rilegge.
        self.partita.classifica = lambda: self._voci_della_classifica
        # La forma del proiettile delle Impostazioni (D17), e la voce del suo esempio, quando suona.
        self._esempio_del_proiettile = None
        self._dai_il_proiettile(self.impostazioni["proiettile"])
        self._crea_controlli()
        self._crea_menu()
        dentro_area_utile(self, MISURA)
        self.Bind(wx.EVT_CLOSE, self._alla_chiusura)
        self.Bind(wx.EVT_ACTIVATE, self._all_attivazione)
        self.Bind(wx.EVT_MENU_OPEN, self._al_menu_aperto)
        self.Bind(wx.EVT_MENU_CLOSE, self._al_menu_chiuso)
        self.Bind(wx.EVT_CHAR_HOOK, self._tasto)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._al_battito, self.timer)
        self.timer.Start(costanti.BATTITO_MS)
        self.vista.ChangeValue(self.apertura())
        self.vista.SetInsertionPoint(0)
        self.vista.ShowPosition(0)
        self.aggiorna_barra()

    # Costruzione.

    def _crea_controlli(self):
        self.pannello = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        nome_della_vista = _("Partita")
        self.etichetta_vista = wx.StaticText(self.pannello, label=nome_della_vista)
        self.vista = wx.TextCtrl(self.pannello, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2)
        self.vista.SetName(nome_della_vista)
        nome_della_barra = _("Barra di stato")
        self.etichetta_barra = wx.StaticText(self.pannello, label=nome_della_barra)
        # Le righe della barra non vanno mai a capo, come in MESS: una riga lunga scorre di lato, e le
        # frecce su e giù passano da una riga all'altra anche con i caratteri grandi.
        self.barra = wx.TextCtrl(self.pannello, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2 | wx.TE_DONTWRAP)
        self.barra.SetName(nome_della_barra)
        # La mira manuale (D29): mentre si mira il fuoco sta qui, su un controllo che non è un testo,
        # perché sulle aree di testo NVDA si prende le frecce, Home e Fine, le rimanda come pressioni brevi
        # e rilegge la riga a ogni ripetizione. Non mostra niente: i valori li dice la barra di stato.
        nome_della_mira = _("Mira")
        self.etichetta_mira = wx.StaticText(self.pannello, label=nome_della_mira)
        self.controllo_mira = ControlloDellaMira(self.pannello, nome_della_mira)
        sizer.Add(self.etichetta_vista, 0, wx.LEFT | wx.RIGHT | wx.TOP, dialoghi.BORDO)
        sizer.Add(self.vista, 1, wx.EXPAND | wx.ALL, dialoghi.BORDO)
        sizer.Add(self.etichetta_barra, 0, wx.LEFT | wx.RIGHT, dialoghi.BORDO)
        sizer.Add(self.barra, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, dialoghi.BORDO)
        sizer.Add(self.etichetta_mira, 0, wx.LEFT | wx.RIGHT, dialoghi.BORDO)
        sizer.Add(self.controllo_mira, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, dialoghi.BORDO)
        # Il rilascio dei tasti della mira arriva al controllo che ha il fuoco, qualunque sia dei tre.
        for controllo in (self.vista, self.barra, self.controllo_mira):
            controllo.Bind(wx.EVT_KEY_UP, self._tasto_su)
        # La barra è alta quanto le sue righe, nel suo carattere, più il bordo, la barra di scorrimento
        # orizzontale e mezza riga di margine, come in MESS: niente misure fisse in pixel.
        riga = self.barra.GetCharHeight()
        fuori = max(self.barra.GetSize().height - self.barra.GetClientSize().height,
                    self.barra.GetWindowBorderSize().height + wx.SystemSettings.GetMetric(wx.SYS_HSCROLL_Y, self.barra))
        self.barra.SetMinSize(wx.Size(-1, riga * RIGHE_DELLA_BARRA + riga // 2 + fuori))
        self.pannello.SetSizer(sizer)

    def voci_menu(self):
        """La tabella dei menu: per ogni menu il titolo e le voci; None separa due gruppi di voci."""
        return (
            (_("&Partita"), (
                Voce(_("&Nuova partita"), "Ctrl+N", self.nuova_partita),
                Voce(_("&Pausa"), "Ctrl+P", self.pausa),
                None,
                Voce(_("&Classifica"), "Ctrl+K", self.mostra_classifica),
                Voce(_("S&tatistiche"), "Ctrl+T", self.mostra_statistiche),
                None,
                Voce(_("&Esci"), "Ctrl+Q", self.esci),
            )),
            (_("&Impostazioni"), (
                Voce(_("&Volume..."), None, self.cambia_volume),
                # Il tasto di MeTeOra per la stessa scelta (D20): il gancio dei tasti lo lascia ai menu, perché ha i modificatori.
                Voce(_("Scheda &audio..."), "Ctrl+Shift+X", self.scegli_la_scheda),
                Voce(_("&Proiettile..."), None, self.cambia_proiettile),
                Voce(_("&Leggi i resoconti con la sintesi"), None, self.cambia_sintesi, True),
                Voce(_("L&ingua..."), None, self.cambia_lingua),
            )),
            (_("&Aiuto"), (
                Voce(_("&Guida ai comandi"), "F1", self.guida),
                Voce(_("&Manuale"), "Shift+F1", self.manuale),
                Voce(_("&Novità"), "F2", self.novita),
                Voce(_("&Informazioni"), "F3", self.informazioni),
                # Gli Ascolti (D23): dal terzo collaudo, il 20 novembre 2026, soltanto qui, come ha chiesto Gabriele.
                Voce(_("&Ascolti..."), None, self.ascolti),
                None,
                Voce(_("&Offrimi un caffè..."), None, self.caffe),
                None,
                Voce(_("&Entra al PUB, il gruppo di supporto..."), None, self.entra_al_pub),
                Voce(_("&Chiedi aiuto al PUB..."), None, self.chiedi_aiuto_al_pub),
            )),
        )

    def tasti_fuori_dai_menu(self):
        """I tasti che il gancio della finestra prende da sé, con quello che fanno, per la guida."""
        return (
            # Traduttori: il nome del tasto, come lo dice la guida.
            (_("Spazio"), _("comincia la partita e, mentre ascolti i bersagli, avvia la mira; nei livelli della palestra, dopo ogni tiro, fa "
                            "ripartire la partita, che aspetta ferma")),
            # Traduttori: da qui, i tasti della mira manuale (D29), che muovono un valore finché li tieni giù.
            (_("Frecce sinistra e destra"), _("nella mira, col fuoco sulla Mira, muovono la direzione, finché le tieni giù")),
            (_("Frecce su e giù"), _("nella mira, col fuoco sulla Mira, alzano e abbassano l'alzo, finché le tieni giù")),
            (_("Home e Fine"), _("nella mira, col fuoco sulla Mira, abbassano e alzano la potenza, finché le tieni giù")),
            (_("Invio"), _("nella mira fa partire il colpo coi tre valori di quel momento")),
            # Traduttori: Esc nella mira (D25); fuori dalla mira chiude le finestre interne, come sempre.
            ("Esc", _("durante la mira annulla il tiro: il colpo non parte, e il proiettile resta nella riserva; nelle finestre interne le chiude")),
            ("F5", _("porta il fuoco nella vista della partita")),
            # Traduttori: F6 durante la mira manuale (D29), dopo F5 o F7.
            ("F6", _("durante la mira riporta il fuoco sulla Mira, dove le frecce, Home e Fine muovono i valori")),
            ("F7", _("porta il fuoco nella barra di stato")),
        )

    def _crea_menu(self):
        barra_menu = wx.MenuBar()
        # Le voci dei menu per comando: la voce a spunta della sintesi si rimette in pari da qui.
        self.voci_del_menu = {}
        for titolo, voci in self.voci_menu():
            menu = wx.Menu()
            for voce in voci:
                if voce is None:
                    menu.AppendSeparator()
                    continue
                testo = f"{voce.testo}\t{voce.tasto}" if voce.tasto else voce.testo
                elemento = menu.Append(wx.ID_ANY, testo, kind=wx.ITEM_CHECK if voce.spunta else wx.ITEM_NORMAL)
                self.voci_del_menu[voce.comando] = elemento
                self.Bind(wx.EVT_MENU, lambda event, c=voce.comando: self._esegui(c), elemento)
            barra_menu.Append(menu, titolo)
        self.SetMenuBar(barra_menu)
        self.voci_del_menu[self.cambia_sintesi].Check(bool(self.impostazioni["sintesi"]))

    def voci_guida(self):
        """Le voci dei menu per la guida: titoli e testi senza le e commerciali, tasti con i nomi italiani."""
        voci = []
        for titolo, elenco in self.voci_menu():
            comandi = [(_senza_tasto(voce.testo), voce.tasto.replace("Shift", _("Maiusc")) if voce.tasto else None) for voce in filter(None, elenco)]
            voci.append((_senza_tasto(titolo), comandi))
        return voci

    def testo_della_guida(self):
        """La guida ai comandi, una voce per riga: i tasti fuori dai menu, poi ogni menu con le sue voci."""
        righe = [_("Guida ai comandi di Ciechino."), _("I tasti fuori dai menu:")]
        for tasto, cosa in self.tasti_fuori_dai_menu():
            # Traduttori: una riga della guida, come «F5: porta il fuoco nella vista della partita.».
            righe.append(_("{tasto}: {cosa}.").format(tasto=tasto, cosa=cosa))
        for titolo, comandi in self.voci_guida():
            n = len(comandi)
            righe.append(ngettext("Il menu {menu}, con {n} comando:", "Il menu {menu}, con {n} comandi:", n).format(menu=titolo, n=n))
            for testo, tasto in comandi:
                if tasto:
                    righe.append(_("{voce}: {tasto}.").format(voce=testo, tasto=tasto))
                else:
                    righe.append(_("{voce}: dal menu, senza tasto rapido.").format(voce=testo))
        return "\n".join(righe)

    # L'apertura.

    def apertura(self):
        """Le righe con cui comincia la vista: chi è Ciechino, oggi, come si comincia, il record, il suono e le notizie dell'avvio."""
        righe = testi.informazioni().splitlines()[:2]
        # Traduttori: {data} è come «giovedì 19 novembre 2026».
        righe.append(_("Oggi è {data}.").format(data=testi.data_lunga(datetime.datetime.now())))
        righe.append(_("Per cominciare una partita premi Spazio; F1 ti dice tutti i comandi."))
        righe.append(self._riga_del_record())
        # La scheda audio: se quella scelta non c'era o non si è aperta, prima la riga che lo dice, poi quella di dove suona.
        righe.extend(self._righe_della_scheda())
        if self.crash_precedente:
            # Traduttori: {file} è il percorso del resoconto del crash.
            righe.append(_("La volta scorsa Ciechino si è chiuso per un errore grave, e il resoconto è nel file {file}: se chiedi aiuto al PUB, "
                           "dal menu Aiuto, puoi allegarlo.").format(file=self.crash_precedente))
        righe.extend(self.avvisi_dei_dati)
        return "\n".join(righe)

    def _riga_del_record(self):
        voci = self._voci_della_classifica
        if not voci:
            return classifica.righe([])[0]
        primo = voci[0]
        quando = testi.leggi_data(primo.get("data"))
        giorni = (datetime.datetime.now().date() - quando.date()).days if isinstance(quando, datetime.datetime) else 0
        punti = plurale(primo["punteggio"], "{n} punto", "{n} punti", come=testi.intero(primo["punteggio"]))
        # Traduttori: «Il record della classifica è di Gabry, con 12.340 punti, 3 giorni fa.».
        return _("Il record della classifica è di {nome}, con {punti}, {eta}.").format(nome=primo["nome"], punti=punti, eta=testi.eta_in_giorni(giorni))

    def _righe_della_scheda(self, esito=None):
        """
        La scheda audio su cui suona Ciechino, dallo stato del motore, o che il gioco è muto; e prima, se
        la scheda scelta da chi gioca non c'era o non si è aperta, quale e perché (D20): una frase per
        riga, come le vuole l'apertura della vista. esito è quello di un'apertura del motore; None vuol
        dire quello dell'avvio.
        """
        perche = self._perche_non_la_scelta(self._esito_all_avvio if esito is None else esito)
        return [riga for riga in (perche, self._dove_suona()) if riga]

    def _riga_della_scheda(self, esito=None):
        """Le righe di _righe_della_scheda in una frase sola, per la sintesi, per le notizie del dialogo e per i messaggi della vista."""
        return " ".join(self._righe_della_scheda(esito))

    def _dove_suona(self):
        """La frase che dice dove suona Ciechino adesso: la scheda con la latenza, dallo stato del motore, o che il gioco è muto."""
        stato = self.motore.stato()
        if not stato.get("aperto"):
            return _("Non trovo una scheda audio che si apra: Ciechino funziona lo stesso, ma senza suono.")
        scheda = self._nome_in_uso(stato)
        latenza = stato.get("latenza")
        if isinstance(latenza, int | float) and latenza > 0:
            millesimi = round(latenza * MILLESIMI)
            # Traduttori: {scheda} è il dispositivo con l'interfaccia, come «Altoparlanti (Realtek(R) Audio), WASAPI».
            return _("Il suono esce da {scheda}, con una latenza di {latenza}.").format(
                scheda=scheda, latenza=plurale(millesimi, "{n} millesimo di secondo", "{n} millesimi di secondo", come=testi.intero(millesimi)))
        return _("Il suono esce da {scheda}.").format(scheda=scheda)

    @staticmethod
    def _nome_in_uso(stato):
        """Il nome della scheda su cui suona il motore, come lo dice schede.nome_della_scheda: «Altoparlanti (Realtek(R) Audio), WASAPI»."""
        dispositivo, interfaccia = stato.get("dispositivo"), stato.get("interfaccia")
        nome = schede.nome_della_scheda({"dispositivo": dispositivo, "interfaccia": interfaccia}) if dispositivo else ""
        return nome or ", ".join(str(parte) for parte in (dispositivo, interfaccia) if parte) or _("una scheda senza nome")

    def _esito_della_scheda(self):
        """Com'è andata l'ultima apertura della scheda nel motore, con le chiavi di motore_audio.ESITO_VUOTO; vuoto se il motore non lo sa dire."""
        esito = getattr(self.motore, "esito_dell_apertura", None)
        return esito if isinstance(esito, dict) else {}

    @staticmethod
    def _scelta_mancata(esito):
        """Vero se l'apertura chiedeva una scheda scelta da chi gioca, e quella non c'era o non si è aperta (D20)."""
        return bool(esito and esito.get("richiesta")) and esito.get("quale") != "scelta"

    @classmethod
    def _perche_non_la_scelta(cls, esito):
        """La frase che dice che la scheda scelta non c'è o non si apre, e se Ciechino suona su quella automatica; vuota se la scelta suona, o è l'automatica."""
        if not cls._scelta_mancata(esito):
            return ""
        scheda = schede.nome_della_scheda(esito.get("richiesta"))
        if esito.get("mancante"):
            if esito.get("aperta"):
                # Traduttori: {scheda} è la scheda audio scelta, come «Cuffie (WH-1000XM4), WASAPI»: è sparita, e Ciechino suona sulla scelta automatica.
                return _("La scheda audio scelta, {scheda}, non c'è: Ciechino suona su quella automatica.").format(scheda=scheda)
            # Traduttori: {scheda} è la scheda audio scelta, come «Cuffie (WH-1000XM4), WASAPI»; segue la frase che dice che il gioco è muto.
            return _("La scheda audio scelta, {scheda}, non c'è.").format(scheda=scheda)
        if esito.get("aperta"):
            # Traduttori: {scheda} è la scheda audio scelta, come «Realtek ASIO, ASIO»: c'è, ma non si apre, e Ciechino suona sulla scelta automatica.
            return _("La scheda audio scelta, {scheda}, non si apre: Ciechino suona su quella automatica.").format(scheda=scheda)
        # Traduttori: {scheda} è la scheda audio scelta, come «Realtek ASIO, ASIO»; segue la frase che dice che il gioco è muto.
        return _("La scheda audio scelta, {scheda}, non si apre.").format(scheda=scheda)

    def suoni_d_avvio(self):
        """
        Il suono dell'avvio, da chiamare quando la finestra è comparsa; poi, ognuna staccata di
        PAUSA_DOPO_L_AVVIO dalla precedente, le notizie: la più importante fra il crash della volta prima
        e i dati che non si leggevano, se c'è, e la scheda audio scelta che non c'è o non si apre (D20),
        che dice anche la sintesi. Il motore mette ogni notizia al campione giusto, così non si sovrappongono.
        Il campione dove l'ultimo di questi suoni finisce resta in _fine_dei_suoni_d_avvio: gli Ascolti
        dell'avvio aspettano lui (D23).
        """
        voce = self.suoni.suona("avvio")
        notizie = []
        if self.crash_precedente:
            notizie.append("crash_precedente")
        elif self.avvisi_dei_dati:
            notizie.append("dati_rovinati")
        scheda_mancata = self._scelta_mancata(self._esito_all_avvio)
        if scheda_mancata:
            notizie.append("scheda_non_aperta")
        for notizia in notizie:
            inizio = None
            if voce is not None:
                inizio = voce.inizio + len(voce.buffer) + round(PAUSA_DOPO_L_AVVIO * self.motore.fs)
            voce = self.suoni.suona(notizia, inizio=inizio) or voce
        if voce is not None:
            self._fine_dei_suoni_d_avvio = voce.inizio + len(voce.buffer)
        if scheda_mancata:
            self._dici(self._riga_della_scheda())

    # Gli Ascolti all'avvio (D23), chiesti da Gabriele il 20 novembre 2026: «quando Ciechino viene aperto e le
    # partite fatte sono 0, quindi presumibilmente è la prima apertura, porta direttamente il giocatore alla
    # nostra sala briefing dei suoni.»

    def ascolti_alla_prima_apertura(self):
        """
        Ciechino.py la chiama all'avvio, dopo suoni_d_avvio e prima del controllo dell'aggiornamento. Se le
        statistiche non contano nessuna partita cominciata, gli Ascolti si apriranno da soli, una volta per
        avvio: il battito li apre appena i suoni dell'avvio sono finiti, la finestra è attiva e nessun
        dialogo o menu è aperto. Intanto la proposta dell'aggiornamento aspetta, con
        aspetta_i_dialoghi_d_avvio, che arriva soltanto quando gli Ascolti si sono chiusi; e una partita
        cominciata prima li toglie. Restituisce vero se gli Ascolti aspettano di aprirsi.
        """
        if self.chiusa or self._partite_cominciate():
            return False
        self._ascolti_d_avvio = True
        self._avvio_libero.clear()
        return True

    def aspetta_i_dialoghi_d_avvio(self, attesa=None):
        """
        Per il filo dell'aggiornamento, prima della proposta e degli avvisi: aspetta, al più attesa secondi,
        o finché serve se è None, che gli Ascolti dell'avvio si siano chiusi, o che non debbano aprirsi più.
        Vero se la strada è libera. Si chiama soltanto fuori dal filo della finestra, che altrimenti si fermerebbe.
        """
        return self._avvio_libero.wait(attesa)

    def _partite_cominciate(self):
        """Vero se le statistiche contano una partita cominciata, o se una partita è in corso adesso."""
        conti = self.statistiche.dati.get("conti", {})
        return bool(conti.get("partite_iniziate")) or bool(self.partita.in_corso())

    def _libera_l_avvio(self):
        """Gli Ascolti dell'avvio non aspettano più: la proposta dell'aggiornamento può arrivare."""
        self._ascolti_d_avvio = False
        self._avvio_libero.set()

    def _pronti_gli_ascolti_d_avvio(self):
        """Vero quando gli Ascolti dell'avvio si possono aprire: i suoni dell'avvio sono finiti, la finestra è attiva e niente è aperto."""
        fine = self._fine_dei_suoni_d_avvio or 0
        return self.IsActive() and not self._comandi_in_corso and not self._dialogo_aperto() and self.motore.campione_udibile() >= fine

    def _forse_gli_ascolti_d_avvio(self):
        """
        A ogni battito, se gli Ascolti dell'avvio aspettano: una partita cominciata li toglie; quando sono
        pronti si aprono, ma fuori dal battito, che deve continuare a girare anche col dialogo aperto.
        """
        if not self._ascolti_d_avvio:
            return
        if self.chiusa or self._partite_cominciate():
            self._libera_l_avvio()
            return
        if self._pronti_gli_ascolti_d_avvio():
            self._ascolti_d_avvio = False
            wx.CallAfter(self._apri_gli_ascolti_d_avvio)

    def _apri_gli_ascolti_d_avvio(self):
        """
        Gli Ascolti dell'avvio: se intanto è cambiato qualcosa, come un dialogo aperto, il battito riprova;
        altrimenti la vista dice che è la prima volta, come si riaprono e che Esc li chiude, col cursore su
        quella riga, che NVDA ritrova quando il dialogo si chiude, e si aprono, come dal menu. Chiusi, la
        proposta dell'aggiornamento può arrivare.
        """
        if not self or self.IsBeingDeleted():
            return
        if self.chiusa or self._partite_cominciate():
            self._libera_l_avvio()
            return
        if not self._pronti_gli_ascolti_d_avvio():
            self._ascolti_d_avvio = True
            return
        try:
            self.statistiche.conta("ascolti_d_avvio")
            self._in_coda(frase_della_prima_apertura())
            self._esegui(self.ascolti)
        finally:
            self._libera_l_avvio()

    # I comandi e i tasti.

    def _esegui(self, comando):
        self._comandi_in_corso += 1
        try:
            comando()
        finally:
            self._comandi_in_corso -= 1
            self._segui_la_fase_della_mira()
            self.aggiorna_barra()

    def _finestra_modale_aperta(self):
        """
        Vero se è aperto davvero un dialogo, o un messaggio, o un menu (D22): allora la partita si ferma,
        F5 e F7 sono suoi, e il dialogo del nome aspetta che si chiuda.
        """
        if self._modali or self._menu_aperto():
            return True
        return any(isinstance(finestra, wx.Dialog) and finestra.IsShown() for finestra in wx.GetTopLevelWindows())

    def _dialogo_aperto(self):
        """
        Vero se è aperto un dialogo, o un messaggio, o se il dialogo del nome sta per aprirsi, alla fine
        del suono dell'ingresso: allora lo spazio non è della partita, e la classifica aspetta. Senza
        l'attesa del nome, lo spazio premuto nel frattempo, a partita finita, ne comincerebbe un'altra.
        """
        return self._nome_in_attesa is not None or self._finestra_modale_aperta()

    def _spazio_per_la_partita(self):
        """Lo spazio va alla partita soltanto se la finestra principale è attiva e nessun dialogo è aperto."""
        return not self.chiusa and self.IsActive() and not self._dialogo_aperto()

    def _tasto(self, event):
        """
        Il gancio dei tasti della finestra, che wx chiama alla pressione, prima dei controlli e dei tasti
        rapidi dei menu. Lo spazio senza modificatori, a finestra attiva e senza dialoghi aperti, è della
        partita e si ferma qui, senza Skip: il carattere non arriva all'area di testo, e NVDA non lo
        ripete. Lo spazio tenuto giù conta una volta sola, alla prima pressione. Esc, durante la mira e alle
        stesse condizioni dello spazio, annulla la mira (D25); fuori dalla mira, e nei dialoghi, prosegue
        come prima. F5 e F7 portano il fuoco. Ogni altro tasto prosegue come sempre, verso i controlli e i
        tasti rapidi.
        Un tasto che arriva qui dice anche che nessun menu è aperto, perché il menu aperto si tiene la
        tastiera: dove Windows non sa dire dei menu, un evento di chiusura perso finisce qui (D22).
        """
        self._menu_dagli_eventi = False
        codice = event.GetKeyCode()
        if not event.HasAnyModifiers():
            if getattr(self.partita, "fase", None) == "mira" and self._spazio_per_la_partita():
                # La mira manuale (D29): i tasti dei valori, col fuoco sul controllo Mira, e l'Invio, ovunque,
                # sono della partita e si fermano qui; la ripetizione del tasto tenuto giù non cambia niente, il
                # moto è già partito. Col fuoco nella vista o nella barra, portato lì con F5 o F7, le frecce,
                # Home e Fine leggono come sempre, e F6 riporta il fuoco sulla Mira.
                if codice in TASTI_DELLA_MIRA and self.controllo_mira.HasFocus():
                    if not event.IsAutoRepeat():
                        parametro, verso = TASTI_DELLA_MIRA[codice]
                        self._esegui(lambda: self._ricevi(self.partita.muovi(parametro, verso)))
                    return
                if codice in TASTI_DELLO_SPARO:
                    if not event.IsAutoRepeat():
                        self._esegui(lambda: self._ricevi(self.partita.spara()))
                    return
            if codice == wx.WXK_SPACE and self._spazio_per_la_partita():
                if not event.IsAutoRepeat():
                    self._esegui(self.spazio)
                return
            if codice == wx.WXK_ESCAPE and self._spazio_per_la_partita() and getattr(self.partita, "fase", None) == "mira":
                if not event.IsAutoRepeat():
                    self._esegui(self.annulla_la_mira)
                return
            if codice in TASTI_DEL_FUOCO and not self._finestra_modale_aperta():
                self._esegui({wx.WXK_F5: self.vai_alla_vista, wx.WXK_F6: self.vai_alla_mira, wx.WXK_F7: self.vai_alla_barra}[codice])
                return
        event.Skip()

    def _tasto_su(self, event):
        """Il rilascio di un tasto della mira manuale (D29), che arriva al controllo col fuoco: il valore si
        ferma dove si sentiva. Gli altri rilasci proseguono come sempre."""
        codice = event.GetKeyCode()
        if codice in TASTI_DELLA_MIRA and getattr(self.partita, "fase", None) == "mira":
            parametro, verso = TASTI_DELLA_MIRA[codice]
            self._esegui(lambda: self._ricevi(self.partita.lascia(parametro, verso)))
            return
        event.Skip()

    def _controlla_i_tasti_della_mira(self):
        """A ogni battito, i tasti della mira che la partita crede giù e che la tastiera dice su valgono
        come lasciati: il rilascio arriva soltanto al controllo col fuoco, e con un dialogo o un'altra
        finestra davanti può perdersi. Il tasto si legge con wx.GetKeyState, che chiede lo stato vero."""
        for parametro, verso in list(getattr(self.partita, "tasti_giu", list)()):
            codice = TASTO_DI_OGNI_VERSO.get((parametro, verso))
            if codice is not None and not wx.GetKeyState(codice):
                self._ricevi(self.partita.lascia(parametro, verso))

    def _segui_la_fase_della_mira(self):
        """Il fuoco segue la mira manuale (D29): quando la mira comincia va sul controllo Mira, quando
        finisce torna alla vista, se era ancora sulla Mira. In mezzo F5 e F7 lo portano dove si vuole, e
        resta lì."""
        in_mira = getattr(self.partita, "fase", None) == "mira"
        if in_mira == self._in_mira:
            return
        self._in_mira = in_mira
        if self.chiusa or self._finestra_modale_aperta():
            return
        if in_mira:
            self.controllo_mira.SetFocus()
        elif self.controllo_mira.HasFocus():
            self.vista.SetFocus()

    def spazio(self):
        """Lo spazio della partita: senza partita in corso ne comincia una, e allora il battito riparte,
        come con Nuova partita, se un battito guasto l'aveva fermato."""
        nuova = not self.partita.in_corso()
        self._ricevi(self.partita.spazio())
        if nuova and self.partita.in_corso():
            self._riavvia_il_battito()

    def annulla_la_mira(self):
        """Esc nella mira (D25): la partita annulla la mira, col suo suono, e il proiettile resta nella riserva."""
        self._ricevi(self.partita.annulla_mira())

    def _riavvia_il_battito(self):
        """Il timer del battito riparte, se si era fermato: lo ferma soltanto un battito guasto, e una partita nuova merita un'altra prova."""
        if not self.timer.IsRunning():
            self.timer.Start(costanti.BATTITO_MS)

    def _al_battito(self, event=None):
        """
        Il battito: il motore dice se il flusso è vivo, la partita si ferma o riprende secondo i dialoghi
        e l'attivazione della finestra (D14), poi avanza; i suoi messaggi vanno nella vista, le notizie del
        motore nel registro, e la barra si rinfresca. Il timer batte anche dentro i dialoghi modali e i
        messaggi di wx, che smistano WM_TIMER: per questo la pausa la decide ogni battito.
        """
        if self.chiusa or self._nel_battito:
            return
        self._nel_battito = True
        try:
            self._controlla_il_motore()
            self._pausa_della_finestra()
            self._controlla_i_tasti_della_mira()
            try:
                messaggi = self.partita.battito()
            except Exception:
                # Un battito guasto si guasterebbe a ogni giro, e la rete degli errori aprirebbe un messaggio
                # dopo l'altro: il timer si ferma, e l'errore arriva alla rete una volta sola. Una partita
                # nuova lo fa ripartire.
                self.timer.Stop()
                raise
            try:
                self._ricevi(messaggi)
            except Exception:  # noqa: BLE001 - i messaggi sono già consumati e non si ripetono: il battito seguente va avanti
                self._guasto("messaggi")
            self._notizie_del_motore()
            self._forse_gli_ascolti_d_avvio()
        finally:
            self._nel_battito = False
        self._segui_la_fase_della_mira()
        self.aggiorna_barra()

    def _ricevi(self, messaggi, con_la_classifica=True):
        """
        I messaggi della partita, in coda alla vista uno dopo l'altro, col cursore all'inizio del primo e
        l'annuncio alla sintesi. Un messaggio «fine» salva le statistiche e porta la partita finita alla
        classifica; con_la_classifica falso, all'uscita, la classifica non si chiede.
        """
        self._accogli(messaggi)
        if not con_la_classifica:
            self._partite_da_concludere.clear()
            return
        self._concludi_le_partite()

    def _accogli(self, messaggi):
        """
        I messaggi in coda alla vista, senza ancora la classifica. Il cursore va all'inizio del primo: la
        partita mette per primo il resoconto del tiro, e dopo il livello, i primati, l'ultimo colpo o il
        bonus, così Freccia giù li legge tutti in ordine, e le misure del tiro non restano sopra il cursore.
        """
        primo = None
        for messaggio in messaggi or ():
            testo = getattr(messaggio, "testo", "") or ""
            if testo:
                inizio = self._in_coda(testo, getattr(messaggio, "annuncio", "") or "")
                if primo is None:
                    primo = inizio
            if getattr(messaggio, "tipo", None) == "fine":
                dati = getattr(messaggio, "dati", None)
                self._partite_da_concludere.append(dati if isinstance(dati, dict) else {})
                self._salva_le_statistiche()
        if primo is not None:
            self.vista.SetInsertionPoint(primo)
            self.vista.ShowPosition(primo)

    def _guasto(self, lavoro):
        """
        Un guasto, dentro un except, in un lavoro che il battito ripete: il resoconto va nel registro una
        volta sola per guasto uguale, e da sorgente anche sulla console; chi gioca sente il suono
        dell'errore e legge il messaggio una volta sola per lavoro, quando il battito è finito. Altrimenti
        un guasto che si ripete a ogni giro aprirebbe un messaggio dopo l'altro e riempirebbe il registro.
        """
        tipo, valore, traccia = sys.exc_info()
        firma = (lavoro, getattr(tipo, "__name__", str(tipo)), str(valore))
        if firma not in self._guasti_scritti:
            self._guasti_scritti.add(firma)
            registro.problema(tipo, valore, traccia)
            if not getattr(sys, "frozen", False):
                traceback.print_exception(tipo, valore, traccia)
        if lavoro not in self._guasti_detti:
            self._guasti_detti.add(lavoro)
            wx.CallAfter(self.errore_imprevisto, tipo, valore)

    # La vista, la sintesi e la barra.

    def _in_coda(self, testo, annuncio="", riga_del_cursore=0, fuoco=False):
        """
        Mette in coda alla vista un testo, o un elenco di righe, e il cursore all'inizio della riga
        riga_del_cursore del testo nuovo, la prima se non si dice altro; l'annuncio va alla sintesi. Le
        righe vuote non entrano: la vista è una voce per riga. Il cursore si mette dalla posizione che la
        vista dà dopo aver scritto ciò che lo precede, non contando i caratteri, perché con TE_RICH2 le
        posizioni non sono indici della stringa. Restituisce quella posizione, o None se non ha scritto niente.
        """
        righe = [riga for riga in (testo.splitlines() if isinstance(testo, str) else testo) if riga.strip()]
        if not righe:
            return None
        riga_del_cursore = min(max(riga_del_cursore, 0), len(righe) - 1)
        vista = self.vista
        if vista.GetLastPosition():
            vista.AppendText("\n")
        if riga_del_cursore:
            vista.AppendText("\n".join(righe[:riga_del_cursore]) + "\n")
        inizio = vista.GetLastPosition()
        vista.AppendText("\n".join(righe[riga_del_cursore:]))
        vista.SetInsertionPoint(inizio)
        vista.ShowPosition(inizio)
        if fuoco:
            vista.SetFocus()
        self._dici(annuncio)
        return inizio

    def _dici(self, testo):
        """L'annuncio alla sintesi, con l'uscita delle impostazioni, se la lettura dei resoconti è accesa."""
        if not testo or not self.impostazioni.get("sintesi"):
            return
        import GBsintesi

        if self._sintesi is None:
            self._sintesi = GBsintesi.Sintesi()
        uscita = self.impostazioni.get("uscita")
        if uscita not in GBsintesi.USCITE:
            uscita = GBsintesi.AUTOMATICA
        with contextlib.suppress(Exception):
            # Uno screen reader che non risponde non deve fermare la partita.
            self._sintesi.dici(testo, uscita)

    def righe_della_barra(self):
        """Le righe della barra: quelle della partita, e sotto lo scaricamento dell'aggiornamento, quando c'è."""
        righe = [str(riga) for riga in self.partita.righe_della_barra() or ()]
        if self._riga_dell_aggiornamento:
            righe.append(self._riga_dell_aggiornamento)
        return righe

    def aggiorna_barra(self):
        """
        Riscrive la barra di stato, ma solo se è cambiata, e mai a metà di un comando: lo fa la fine del
        comando. Si riscrive anche col cursore sopra, che non disturba NVDA: il cursore resta sulla stessa
        riga e nella stessa colonna, a fine riga se la riga si è accorciata, sull'ultima riga se le righe
        sono meno di prima. Con TE_RICH2 righe e colonne le traduce wx. Righe che non si compongono, per
        esempio per un segnaposto sbagliato in una traduzione, lasciano la barra com'era: l'errore si dice
        una volta sola, e la partita va avanti, invece di un messaggio a ogni battito.
        """
        if not self or self.chiusa or self._comandi_in_corso:
            return
        try:
            testo = "\n".join(self.righe_della_barra())
        except Exception:  # noqa: BLE001 - il battito riproverebbe cento volte al secondo
            self._guasto("barra")
            return
        if testo == self._testo_barra:
            return
        self._testo_barra = testo
        _trovata, colonna, riga = self.barra.PositionToXY(self.barra.GetInsertionPoint())
        self.barra.ChangeValue(testo)
        riga = min(max(riga, 0), max(self.barra.GetNumberOfLines() - 1, 0))
        colonna = min(max(colonna, 0), max(self.barra.GetLineLength(riga), 0))
        posizione = self.barra.XYToPosition(colonna, riga)
        if posizione >= 0:
            self.barra.SetInsertionPoint(posizione)

    def vai_alla_vista(self):
        self.suoni.suona("vai_alla_vista")
        self.vista.SetFocus()

    def vai_alla_mira(self):
        """F6: durante la mira manuale il fuoco torna sul controllo Mira, dove i tasti dei valori muovono la
        mira (D29); fuori dalla mira non c'è niente da puntare, e suona il non ora."""
        if getattr(self.partita, "fase", None) != "mira":
            self.suoni.suona("non_ora")
            return
        self.suoni.suona("vai_alla_mira")
        self.controllo_mira.SetFocus()

    def vai_alla_barra(self):
        """F7: il fuoco va nella barra; se c'è già, la barra si riscrive comunque, tenendo il cursore."""
        self.suoni.suona("vai_alla_barra")
        if self.barra.HasFocus():
            self._testo_barra = None
            self.aggiorna_barra()
        self.barra.SetFocus()

    # I dialoghi e i messaggi.

    def _modale(self, dialogo):
        """
        Mostra un dialogo modale; finché resta aperto, lo spazio non è della partita, la classifica aspetta
        e la partita in corso si ferma, in silenzio (D14). Chiuso il dialogo, riprende da sola, se la
        finestra è di nuovo attiva e la pausa l'aveva messa lei.
        """
        self._modali += 1
        try:
            self._ferma_la_partita()
            return dialogo.ShowModal()
        finally:
            self._modali -= 1
            self._dopo_un_modale()

    def _messaggio(self, testo, stile=wx.OK | wx.ICON_INFORMATION, titolo=TITOLO):
        """Un messaggio di wx, o una domanda, che ferma la partita come un dialogo."""
        self._modali += 1
        try:
            self._ferma_la_partita()
            return dialoghi.messaggio(self, testo, titolo, stile)
        finally:
            self._modali -= 1
            self._dopo_un_modale()

    def _domanda(self, testo):
        """Una domanda sì o no, con il suo suono e il no già scelto: vero per il sì. Al no suona l'annullamento."""
        self.suoni.suona("domanda")
        if self._messaggio(testo, wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION) == wx.YES:
            return True
        self.suoni.suona("annullato")
        return False

    # La pausa della finestra (D14): una partita che chi gioca non può seguire, perché un dialogo ha il
    # fuoco o perché la finestra non è attiva, si ferma e tace; la decisione è della partita, con
    # pausa_automatica, che riprende soltanto le pause messe così e lascia stare quelle di Ctrl+P.

    def _pausa_automatica(self, attiva):
        """Chiede alla partita la pausa automatica, o la ripresa; una partita che non la conosce, come quelle finte delle prove, non si ferma."""
        if self.chiusa:
            return
        pausa = getattr(self.partita, "pausa_automatica", None)
        if not callable(pausa):
            return
        try:
            messaggi = pausa(attiva)
        except Exception:  # noqa: BLE001 - la pausa si chiede anche dentro il messaggio dell'errore: non deve sollevare
            self._guasto("pausa")
            return
        self._accogli(messaggi)

    def _ferma_la_partita(self):
        self._pausa_automatica(True)

    def _riprendi_la_partita(self):
        """La partita riprende da sola soltanto senza dialoghi aperti e con la finestra di nuovo attiva."""
        if self.chiusa or self._finestra_modale_aperta() or not self.IsActive():
            return
        self._pausa_automatica(False)
        self.aggiorna_barra()

    def _pausa_della_finestra(self):
        """A ogni battito, la pausa segue lo stato della finestra: così non dipende da un evento perso."""
        if self._finestra_modale_aperta() or not self.IsActive():
            self._ferma_la_partita()
        else:
            self._riprendi_la_partita()

    def _dopo_un_modale(self):
        """
        Chiuso un dialogo, la ripresa si prova dopo il comando che l'aveva aperto: Ctrl+N col sì abbandona
        la partita subito, e una ripresa nel mezzo farebbe ripartire per un attimo la barra e le voci.
        """
        if not self.chiusa:
            wx.CallAfter(self._riprendi_dopo)

    def _riprendi_dopo(self):
        # Fra la chiamata e adesso la finestra può essere stata distrutta.
        if self and not self.IsBeingDeleted():
            self._riprendi_la_partita()

    def _all_attivazione(self, event):
        """
        La finestra principale perde l'attivazione, per Alt+Tab, per un'altra applicazione o per un dialogo
        che si apre, e la partita si ferma subito; la riprende, a cose fatte, la prova del ritorno.
        """
        event.Skip()
        if self.chiusa:
            return
        if event.GetActive():
            # Durante l'evento wx non ha ancora rimesso il fuoco nella finestra, e IsActive direbbe di no.
            wx.CallAfter(self._riprendi_dopo)
        else:
            self._ferma_la_partita()

    # I menu (D22): un menu aperto, con Alt, F10 o il mouse, ferma la partita come un dialogo, e la
    # ripresa viene quando si chiude, anche se un suo comando apre poi un dialogo, che la tiene ferma
    # fino alla sua chiusura. I tasti rapidi dei menu non sono menu aperti, anche se wx ne manda gli
    # eventi: lo dice Windows.

    def _al_menu_aperto(self, event):
        """
        Si apre un menu della finestra: la partita si ferma subito, in silenzio. Ma un tasto rapido, come
        Ctrl+P, passa da TranslateAccelerator di Windows, che manda l'apertura del menu della sua voce,
        poi il comando, poi la chiusura, senza che nessun menu si apra davvero: lì Windows dice che il
        filo non è in un menu, e la partita non si ferma, così Ctrl+P fuori dalla palestra non fa ripartire
        la barra della mira e non conta una pausa automatica. Un menu vero è già aperto per Windows quando
        arriva l'evento; dove Windows non sa rispondere valgono gli eventi.
        """
        event.Skip()
        if self.chiusa or menu_di_windows() is False:
            return
        self._menu_dagli_eventi = True
        self._ferma_la_partita()

    def _al_menu_chiuso(self, event):
        """
        Si chiude un menu della finestra: la ripresa si prova dopo il comando scelto, se c'è, come dopo un
        dialogo. wx manda la chiusura prima del comando: un comando che apre un dialogo tiene ferma la
        partita, e la prova della ripresa, che arriva dentro il dialogo, la lascia stare.
        """
        event.Skip()
        if self.chiusa:
            return
        self._menu_dagli_eventi = False
        self._dopo_un_modale()

    def _menu_aperto(self):
        """
        Vero se un menu è aperto. Lo dice Windows, che lo sa anche soltanto con la barra dei menu evidenziata
        da Alt, quando wx non manda eventi, e la sua parola vale sugli eventi di wx, che si possono perdere
        o non appaiare: un'apertura persa non fa ripartire la partita a menu aperto, una chiusura persa
        non la lascia ferma. Dove Windows non sa rispondere valgono gli eventi.
        """
        di_windows = menu_di_windows()
        if di_windows is None:
            return self._menu_dagli_eventi
        self._menu_dagli_eventi = di_windows
        return di_windows

    def errore_imprevisto(self, tipo, valore):
        """
        Un'eccezione sfuggita a un comando: il comando muore in silenzio, e il resoconto completo va nel
        registro degli errori, accanto al programma. Qui la dicono il suono dell'errore e un messaggio,
        che dice dove sta il resoconto; uno solo alla volta.
        """
        if not self or self.IsBeingDeleted() or self._errore_aperto:
            return
        self._errore_aperto = True
        try:
            self.suoni.suona("errore")
            # Traduttori: {tipo} è il nome dell'errore, in inglese, {valore} la sua descrizione, {resoconto} dove sta il resoconto.
            self._messaggio(_("Il comando si è interrotto per un errore imprevisto: {tipo}, {valore}. {resoconto}").format(
                tipo=tipo.__name__, valore=valore, resoconto=dialoghi.resoconto_degli_errori()), wx.OK | wx.ICON_ERROR, _("Errore imprevisto"))
        finally:
            self._errore_aperto = False

    def _salva_le_statistiche(self):
        if self.statistiche.salva():
            return True
        self.suoni.suona("salvataggio_non_riuscito")
        # L'avviso va anche alla sintesi, come ogni messaggio della vista: è una perdita di dati.
        testo = _("Le statistiche non si sono potute salvare: le ritroverai come erano all'ultimo salvataggio riuscito.")
        self._in_coda(testo, testo)
        return False

    def _salva_le_impostazioni(self):
        if modulo_impostazioni.salva(self.impostazioni):
            return True
        self.suoni.suona("salvataggio_non_riuscito")
        testo = _("Le impostazioni non si sono potute salvare: valgono fino all'uscita.")
        self._in_coda(testo, testo)
        return False

    # La partita.

    def nuova_partita(self):
        """
        Ctrl+N: una partita nuova; con una partita in corso, prima una domanda, perché quella si abbandona.
        A partita appena finita, la fine che aspetta ancora la fine del tonfo arriva prima, con la sua
        classifica (D22).
        """
        if self.partita.in_corso():
            if not self._domanda(_("Una partita è in corso: la abbandoni e ne cominci una nuova?")):
                return
            self._ricevi(self.partita.abbandona())
        else:
            self._ricevi(self._messaggi_in_attesa())
        self._ricevi(self.partita.nuova())
        self._riavvia_il_battito()

    def pausa(self):
        """Ctrl+P: la pausa e la ripresa le decide la partita, coi loro suoni."""
        self._ricevi(self.partita.pausa())

    def esci(self):
        self.Close()

    def _concludi_le_partite(self):
        """
        Le partite finite vanno in classifica, una dopo l'altra, quando nessun dialogo è aperto: altrimenti
        aspettano il battito seguente. Una partita entrata in classifica aspetta, prima del nome, la fine
        del suono dell'ingresso; intanto le altre aspettano lei.
        """
        while not self.chiusa:
            if self._nome_in_attesa is not None:
                if not self._nome_pronto():
                    return
                self._chiedi_il_nome()
                continue
            if not self._partite_da_concludere or self._dialogo_aperto():
                return
            self._in_classifica(self._partite_da_concludere.pop(0))

    def _secondi_alla_fine(self, fine):
        """Quanti secondi mancano perché si senta fino al campione fine: zero o meno se è già passato."""
        return (fine - self.motore.campione_udibile()) / self.motore.fs

    def _nome_pronto(self):
        """
        Se il dialogo del nome si può aprire: nessun dialogo vero aperto, e il suono dell'ingresso finito,
        sull'orologio del motore; con il motore muto non c'è niente da aspettare, e se l'orologio si fosse
        fermato vale la scadenza in tempo vero. All'uscita il battito non c'è più, e il suono si aspetta da
        fermi, al più per ATTESA_MASSIMA_DEL_NOME.
        """
        if self._finestra_modale_aperta():
            return False
        attesa = self._nome_in_attesa
        if attesa.fine is None or getattr(self.motore, "muto", False):
            return True
        mancano = self._secondi_alla_fine(attesa.fine)
        if self._all_uscita:
            time.sleep(min(max(mancano, 0.0), ATTESA_MASSIMA_DEL_NOME))
            return True
        return mancano <= 0 or time.monotonic() >= attesa.scadenza

    def _chiudi_i_conti(self):
        """
        All'uscita, prima che la finestra si chiuda: il nome che aspettava il suono dell'ingresso si chiede
        adesso, e le partite finite mentre un dialogo era aperto, per esempio la domanda d'uscita, vanno in
        classifica, col nome, invece di perdersi.
        """
        self._all_uscita = True
        try:
            self._concludi_le_partite()
        finally:
            self._all_uscita = False

    def _riepilogo_di(self, dati):
        """Il riepilogo di una partita finita, dai dati del suo messaggio: il riepilogo stesso o la sua chiave, o in mancanza dalla partita."""
        riepilogo = dati.get("riepilogo") if isinstance(dati.get("riepilogo"), dict) else dati
        if "punteggio" not in riepilogo:
            riepilogo = self.partita.riepilogo()
        return riepilogo

    def _dopo_gli_avvisi(self):
        """
        Il campione da cui far partire l'esito della classifica: dopo gli avvisi della partita appena
        finita, che suonano in fila, perché il suono della fine e quello della classifica, partiti allo
        stesso campione, si fonderebbero in uno; None, cioè subito, se non ce ne sono, o se la partita non
        sa dirlo.
        """
        dopo = getattr(self.partita, "dopo_gli_avvisi", None)
        return dopo() if callable(dopo) else None

    def _messaggi_in_attesa(self):
        """
        I messaggi dell'ultimo tiro che la partita tiene finché il suo effetto suona (D22), subito: per la
        partita nuova e per la chiusura, che non devono perdere la fine della partita e la sua classifica.
        Una partita che non li tiene, come quelle finte delle prove, non ne ha.
        """
        prendi = getattr(self.partita, "messaggi_in_attesa", None)
        return prendi() if callable(prendi) else []

    def _classifica_attuale(self):
        """
        La classifica per una partita finita e per Ctrl+K: quella del file, riletta ogni volta; ma dopo un
        salvataggio non riuscito quella in memoria, con la voce nuova, che vale fino all'uscita, come dice
        l'avviso. Il salvataggio seguente, se riesce, scrive anche lei.
        """
        if self._classifica_in_memoria:
            return self._voci_della_classifica, None
        voci, avviso = classifica.carica()
        self._voci_della_classifica = voci
        return voci, avviso

    def _in_classifica(self, dati):
        """
        A partita finita: se il punteggio entra in classifica, il suono in_classifica, e alla sua fine il
        dialogo del nome, che intanto aspetta in _nome_in_attesa. Altrimenti il suono fuori_classifica e
        una riga che lo dice. I due suoni seguono la fine della partita, in fila.
        """
        riepilogo = self._riepilogo_di(dati)
        voci, avviso = self._classifica_attuale()
        if avviso:
            self.suoni.suona("dati_rovinati")
            self._in_coda(avviso, avviso)
        punteggio = riepilogo.get("punteggio")
        dove = classifica.posizione(voci, punteggio)
        if dove is None:
            self.suoni.suona("fuori_classifica", inizio=self._dopo_gli_avvisi())
            testo = _("Il punteggio di questa partita non entra in classifica.")
            self._in_coda(testo, testo)
            return
        voce = self.suoni.suona("in_classifica", inizio=self._dopo_gli_avvisi())
        fine = voce.inizio + len(voce.buffer) if voce is not None else None
        mancano = max(self._secondi_alla_fine(fine), 0.0) if fine is not None else 0.0
        self._nome_in_attesa = NomeInAttesa(fine, time.monotonic() + mancano + MARGINE_DEL_NOME, riepilogo, voci, dove)

    def _chiedi_il_nome(self):
        """
        Il dialogo del nome della partita in attesa, con l'ultimo nome proposto; poi la voce nuova si
        scrive, e la classifica va in coda alla vista col cursore sulla voce nuova.
        """
        attesa, self._nome_in_attesa = self._nome_in_attesa, None
        riepilogo, voci, dove = attesa.riepilogo, attesa.voci, attesa.posizione
        punteggio = riepilogo.get("punteggio")
        punti = plurale(round(punteggio), "{n} punto", "{n} punti", come=testi.intero(punteggio))
        # Traduttori: {punti} è come «1.234 punti», {posizione} un numero.
        spiegazione = _("Con {punti} entri in classifica al posto {posizione}: scrivi il tuo nome.").format(punti=punti, posizione=dove)
        dialogo = dialoghi.Nome(self, self.impostazioni.get("ultimo_nome", ""), spiegazione, self.suoni.suona)
        try:
            codice = self._modale(dialogo)
            nome = dialogo.nome
        finally:
            dialogo.Destroy()
        if codice != wx.ID_OK or not nome:
            self.suoni.suona("annullato")
            testo = _("Il nome non è stato scritto: questa partita resta fuori dalla classifica.")
            self._in_coda(testo, testo)
            return
        voci, posizione = classifica.inserisci(voci, classifica.nuova_voce(nome, riepilogo))
        salvata = classifica.salva(voci)
        self._voci_della_classifica = voci
        self._classifica_in_memoria = not salvata
        self.impostazioni["ultimo_nome"] = nome
        self.statistiche.conta("ingressi_in_classifica")
        self.suoni.suona("confermato" if salvata else "salvataggio_non_riuscito")
        righe = classifica.righe(voci, nuova=posizione)
        if posizione is not None and posizione <= len(voci):
            # Il cursore sulla riga della voce nuova, nella griglia (D27); alla sintesi la stessa voce a parole.
            self._in_coda(righe, classifica.annuncio(voci[posizione - 1], posizione), posizione - 1 + classifica.PRIMA_VOCE)
        else:
            self._in_coda(righe)
        if not salvata:
            testo = _("La classifica non si è potuta salvare: la voce nuova vale fino all'uscita.")
            self._in_coda(testo, testo)
        self._salva_le_impostazioni()
        self._salva_le_statistiche()

    def _lettura(self, titolo, testo, suono, a_larghezza_fissa=False):
        """Un testo da leggere in un dialogo, massimizzato, in sola lettura, con Chiudi ed Esc (D11);
        a_larghezza_fissa per una griglia, come la classifica, che anche sullo schermo resta in colonna."""
        self.suoni.suona(suono)
        dialogo = dialoghi.Lettura(self, titolo, testo, suona=self.suoni.suona, a_larghezza_fissa=a_larghezza_fissa)
        try:
            self._modale(dialogo)
        finally:
            dialogo.Destroy()

    def mostra_classifica(self):
        """Ctrl+K: la classifica, una voce per riga, nella griglia allineata di D27."""
        voci, avviso = self._classifica_attuale()
        righe = classifica.righe(voci)
        if avviso:
            righe.append(avviso)
        self._lettura(_("Classifica"), "\n".join(righe), "classifica", a_larghezza_fissa=True)

    def mostra_statistiche(self):
        """Ctrl+T: le statistiche, a sezioni, una voce per riga, coi nomi dei suoni detti a parole."""
        self._lettura(_("Statistiche"), "\n".join(self.statistiche.righe(nomi_dei_suoni=modulo_suoni.AZIONI)), "statistiche")

    # Le impostazioni.

    def _prova_il_volume(self, valore):
        """Il volume scritto nel dialogo vale subito, e il suo suono lo fa sentire."""
        self.motore.volume = guadagno_del_volume(valore)
        self.suoni.suona("volume")

    def cambia_volume(self):
        """Volume: il dialogo col numero da 0 a 100, che si sente mentre si cambia; Annulla torna al volume di prima."""
        prima = self.impostazioni["volume"]
        self.suoni.suona("volume")
        dialogo = dialoghi.Volume(self, prima, self._prova_il_volume, self.suoni.suona)
        try:
            codice = self._modale(dialogo)
            valore = dialogo.valore
        finally:
            dialogo.Destroy()
        if codice != wx.ID_OK or valore is None:
            self.motore.volume = guadagno_del_volume(prima)
            self.suoni.suona("annullato")
            testo = _("Il volume resta {volume}.").format(volume=prima)
            self._in_coda(testo, testo)
            return
        self.impostazioni["volume"] = valore
        self.motore.volume = guadagno_del_volume(valore)
        if self._salva_le_impostazioni():
            self.suoni.suona("confermato")
        testo = _("Il volume è {volume}.").format(volume=valore)
        self._in_coda(testo, testo)

    # La scheda audio (D20).

    def scegli_la_scheda(self):
        """
        Ctrl+Maiusc+X, Scheda audio del menu Impostazioni, sul modello della voce Scheda audio di
        MeTeOra. La partita si ferma come con ogni dialogo (D14), e prima dell'elenco il motore rilegge
        le schede restando sulla scelta di adesso: così compaiono anche le cuffie Bluetooth accese dopo
        l'avvio, una scheda scelta che era sparita ed è tornata riprende a suonare, e dopo un flusso
        perso il suono torna. Poi, col suo suono, il dialogo delle schede, con la scheda in uso già
        scelta. OK su un'altra riga, o su una scelta diversa da quella salvata, cambia la scheda del
        motore e salva la scelta, che resta anche se non si apre, perché le cuffie possono tornare; la
        vista e la sintesi dicono dove suona Ciechino adesso, o che la scheda scelta non si è aperta.
        OK sulla riga in uso, come Annulla, non cambia niente; la riga in uso è quella di quando il
        dialogo si chiude, perché se il flusso si è perso intanto OK sulla stessa riga la riapre.
        """
        prima = schede.valida(self.impostazioni.get("scheda"))
        # Come _modale, ma la partita si ferma già per la rilettura, e resta ferma fino al cambio di scheda.
        self._modali += 1
        try:
            self._ferma_la_partita()
            rilettura = self._rileggi_le_schede()
            notizie = []
            try:
                elenco = schede.uscite()
            except Exception as errore:  # noqa: BLE001 - PortAudio fallisce in molti modi: resta la scelta automatica
                elenco = []
                # Traduttori: {errore} è il perché, come lo dice PortAudio, spesso in inglese.
                notizie.append(_("Non riesco a leggere l'elenco delle schede audio: {errore}. Resta la scelta automatica.").format(errore=errore))
            in_uso, automatica = self._riga_in_uso(elenco)
            # Dove suona Ciechino dopo la rilettura, e prima, se la scheda scelta non c'è o non si apre, quale.
            notizie.insert(0, self._riga_della_scheda(rilettura or {}))
            if rilettura and rilettura.get("errore_della_rilettura"):
                notizie.append(_("Non ho potuto rileggere le schede audio: quelle accese da poco, come delle cuffie Bluetooth, possono mancare."))
            selezionata = in_uso if in_uso is not None else self._riga_della_scelta(prima, elenco)
            self.suoni.suona("scheda_audio")
            dialogo = dialoghi.SceltaScheda(self, elenco, in_uso, automatica, selezionata, notizie, self.suoni.suona)
            try:
                codice = self._modale(dialogo)
                indice, nuova = dialogo.indice, dialogo.scelta
            finally:
                dialogo.Destroy()
            # Si parla della scelta, non della scheda: la rilettura prima dell'elenco può aver già spostato il suono,
            # per esempio nelle cuffie scelte appena riaccese.
            if codice != wx.ID_OK or nuova is None:
                self.suoni.suona("annullato")
                testo = _("La scelta della scheda audio non cambia.") + " " + self._dove_suona()
                self._in_coda(testo, testo)
                return
            # La riga in uso si ricalcola: il battito gira anche dentro il dialogo, e la guardia del motore può
            # aver perso il flusso; allora OK sulla stessa riga riapre la scheda, invece di lasciare il gioco muto.
            if indice == self._riga_in_uso(elenco)[0] and nuova == prima:
                self.suoni.suona("confermato")
                testo = _("La scelta della scheda audio resta quella di prima.") + " " + self._dove_suona()
                self._in_coda(testo, testo)
                return
            esito = self._cambia_la_scheda(nuova)
        finally:
            self._modali -= 1
            self._dopo_un_modale()
        self._di_la_scheda_nuova(nuova, prima, esito)

    def _rileggi_le_schede(self):
        """
        Il motore rilegge le schede e torna sulla scelta di adesso, con rileggi_le_schede; restituisce
        l'esito, o None per un motore chiuso o che non lo sa fare.
        """
        rileggi = getattr(self.motore, "rileggi_le_schede", None)
        esito = rileggi() if callable(rileggi) else None
        self._dopo_il_cambio(esito)
        return esito

    def _cambia_la_scheda(self, scelta):
        """Il motore passa alla scheda scelta, già riletta; restituisce l'esito, o None per un motore chiuso o che non lo sa fare."""
        cambia = getattr(self.motore, "cambia_scheda", None)
        esito = cambia(scelta) if callable(cambia) else None
        self._dopo_il_cambio(esito)
        return esito

    def _dopo_il_cambio(self, esito):
        """
        Dopo un cambio di scheda, o una rilettura: quel che non è andato va nel registro, come all'avvio,
        e con una scheda che suona un flusso perso si potrà dire di nuovo.
        """
        if not esito:
            return
        if esito.get("aperta"):
            self._flusso_perso_detto = False
        ultimo_errore = getattr(self.motore, "ultimo_errore", None)
        if ultimo_errore:
            registro.scrivi(f"Motore audio, cambio di scheda: {ultimo_errore}")

    def _riga_in_uso(self, elenco):
        """
        La riga dell'elenco delle schede su cui suona Ciechino, zero per l'automatica, o None se è muto o
        se la scheda scelta non sta nell'elenco; e per l'automatica il nome della scheda dove suona.
        """
        stato = self.motore.stato()
        if not stato.get("aperto"):
            return None, ""
        if stato.get("quale") == "scelta":
            uscita = schede.ritrova(stato.get("scelta"), elenco)
            return (elenco.index(uscita) + 1, "") if uscita is not None else (None, "")
        return 0, self._nome_in_uso(stato)

    @staticmethod
    def _riga_della_scelta(scelta, elenco):
        """La riga dell'elenco della scelta salvata, o zero, l'automatica, se la scelta è vuota o la scheda non c'è."""
        uscita = schede.ritrova(scelta, elenco)
        return elenco.index(uscita) + 1 if uscita is not None else 0

    def _di_la_scheda_nuova(self, nuova, prima, esito):
        """
        Dopo il cambio di scheda: la scelta nuova si salva e si conta; poi la vista e la sintesi dicono
        dove suona Ciechino adesso, col suono confermato, o che la scheda scelta non c'è o non si apre,
        col suo suono, che si conta anche lui. Un salvataggio non riuscito lo dice, e la scelta vale
        fino all'uscita.
        """
        salvata = True
        if nuova != prima:
            self.impostazioni["scheda"] = dict(nuova)
            self.statistiche.conta("scheda_cambiata")
            salvata = self._salva_le_impostazioni()
        esito = esito or {}
        if self._scelta_mancata(esito) or not esito.get("aperta"):
            self.statistiche.conta("scheda_non_aperta")
            self.suoni.suona("scheda_non_aperta")
            testo = self._riga_della_scheda(esito)
        else:
            if salvata:
                self.suoni.suona("confermato")
            testo = self._dove_suona()
            if not nuova:
                testo = _("Ciechino suona sulla scheda automatica, quella di Windows.") + " " + testo
        self._in_coda(testo, testo)

    def _dai_il_proiettile(self, forma):
        """
        La forma del proiettile alla partita, che la usa dalla partita seguente, o subito se nessuna è in
        corso (D17). Una partita che non la conosce, come quelle finte delle prove, non la riceve.
        """
        scegli = getattr(self.partita, "scegli_proiettile", None)
        if callable(scegli):
            scegli(forma)

    def _ferma_l_esempio(self):
        """Ferma l'esempio del proiettile che sta suonando, se c'è."""
        voce, self._esempio_del_proiettile = self._esempio_del_proiettile, None
        if voce is not None:
            self.motore.ferma(voce)

    def _prova_il_proiettile(self, forma):
        """L'esempio della forma su cui si è fermato il dialogo del proiettile: quello di prima si ferma, e parte il nuovo."""
        from partita import esempio_del_proiettile

        self._ferma_l_esempio()
        self._esempio_del_proiettile = esempio_del_proiettile(self.motore, forma)

    @staticmethod
    def _proiettile_dalla_prossima(forma, in_gioco):
        """La frase della vista quando la forma per la prossima partita e quella della partita in corso sono diverse."""
        # Traduttori: {forma} e {prima} sono nomi di forme d'onda, come «seno» o «sega morbida»: {forma} vale dalla prossima partita, {prima} è quella della partita in corso.
        return _("Il proiettile suonerà in {forma} dalla prossima partita: quella in corso resta in {prima}.").format(
            forma=dialoghi.nome_del_proiettile(forma), prima=dialoghi.nome_del_proiettile(in_gioco))

    def cambia_proiettile(self):
        """
        Proiettile: il dialogo con le forme d'onda del proiettile (D17), che fa sentire un esempio di
        ognuna. La forma nuova vale dalla prossima partita: se una partita è in corso, questa tiene la sua
        fino alla fine, e la vista lo dice; altrimenti vale da subito, per la partita che comincerà.
        """
        prima = self.impostazioni["proiettile"]
        self.suoni.suona("proiettile")
        dialogo = dialoghi.SceltaProiettile(self, prima, self._prova_il_proiettile, self.suoni.suona)
        try:
            codice = self._modale(dialogo)
            forma = dialogo.scelta
        finally:
            dialogo.ferma_la_prova()
            self._ferma_l_esempio()
            dialogo.Destroy()
        # La forma della partita in corso: dopo un primo cambio a partita in corso le impostazioni hanno
        # già quella nuova, e la partita tiene ancora la sua. Una partita che non la conosce, come
        # quelle finte delle prove, suona nella forma delle impostazioni.
        in_corso = self.partita.in_corso()
        in_gioco = getattr(self.partita, "proiettile", prima) if in_corso else prima
        if codice != wx.ID_OK or forma is None or forma == prima:
            self.suoni.suona("annullato" if codice != wx.ID_OK or forma is None else "confermato")
            if in_gioco != prima:
                testo = self._proiettile_dalla_prossima(prima, in_gioco)
            else:
                # Traduttori: {forma} è il nome di una forma d'onda, come «seno» o «sega morbida».
                testo = _("Il proiettile resta in {forma}.").format(forma=dialoghi.nome_del_proiettile(prima))
            self._in_coda(testo, testo)
            return
        self.impostazioni["proiettile"] = forma
        self._dai_il_proiettile(forma)
        self.statistiche.conta("proiettile_cambiato")
        if self._salva_le_impostazioni():
            self.suoni.suona("confermato")
        if in_corso:
            testo = self._proiettile_dalla_prossima(forma, in_gioco)
        else:
            # Traduttori: {forma} è il nome di una forma d'onda, come «seno» o «sega morbida».
            testo = _("Il proiettile adesso suona in {forma}: lo senti dalla prossima partita.").format(forma=dialoghi.nome_del_proiettile(forma))
        self._in_coda(testo, testo)

    def ascolti(self):
        """
        Ascolti, nel menu Aiuto (D23): il tutorial audio, un passo alla volta. Suona
        ascolti, conta ascolti_aperti, e apre dialoghi.Ascolti con la Regia di ascolti.py, sul motore del gioco,
        coi suoni della finestra e la forma del proiettile delle impostazioni; la regia conta passi_ascoltati,
        e dici è quella della vista, che legge soltanto se l'impostazione lo vuole. La partita in corso si
        ferma come con ogni dialogo (D14); alla chiusura ogni voce degli Ascolti tace. Chiusi con Invio
        all'ultimo passo, si conta ascolti_finiti.
        """
        from ascolti import Regia

        self.suoni.suona("ascolti")
        self.statistiche.conta("ascolti_aperti")
        regia = Regia(self.motore, self.suoni, self.impostazioni["proiettile"], conta=self.statistiche.conta)
        dialogo = dialoghi.Ascolti(self, regia, self._dici, self.suoni.suona)
        finiti = False
        try:
            self._modale(dialogo)
            finiti = dialogo.finiti
        finally:
            dialogo.ferma()
            regia.ferma()
            dialogo.Destroy()
        if finiti:
            self.statistiche.conta("ascolti_finiti")

    def cambia_sintesi(self):
        """Leggi i resoconti con la sintesi: la voce a spunta accende e spegne l'annuncio dei messaggi della partita."""
        acceso = not self.impostazioni.get("sintesi")
        self.impostazioni["sintesi"] = acceso
        self.voci_del_menu[self.cambia_sintesi].Check(acceso)
        if acceso:
            self.suoni.suona("impostazioni")
            testo = _("La sintesi legge i resoconti della partita.")
        else:
            self.suoni.suona("sintesi_spenta")
            testo = _("La sintesi non legge più i resoconti della partita.")
        self._salva_le_impostazioni()
        self._in_coda(testo, testo)

    def cambia_lingua(self):
        """
        Lingua, sul modello di MeTeOra: l'elenco delle lingue che hanno un catalogo. polipo salva la scelta,
        che vale dal riavvio, perché le tabelle dei testi si traducono quando i moduli si caricano; poi una
        domanda offre di riaprire Ciechino subito, con istanza.riapri, all'uscita.
        """
        self.suoni.suona("lingua")
        codici = lingua.lingue() or [lingua.CODICE]
        attuale = lingua.prossima or lingua.CODICE
        # Traduttori: una riga dell'elenco delle lingue, come «italiano, codice it».
        righe = [_("{nome}, codice {codice}").format(nome=lingua.nome(codice), codice=codice) for codice in codici]
        dialogo = dialoghi.SceltaLingua(self, righe, codici.index(attuale) if attuale in codici else 0, self.suoni.suona)
        try:
            codice = self._modale(dialogo)
            indice = dialogo.scelta
        finally:
            dialogo.Destroy()
        if codice != wx.ID_OK or indice is None:
            self.suoni.suona("annullato")
            testo = _("La lingua non cambia.")
            self._in_coda(testo, testo)
            return
        scelta = codici[indice]
        if len(codici) < 2:
            self.suoni.suona("confermato")
            testo = _("La lingua resta {lingua}: per ora Ciechino parla solo questa, e le altre arriveranno con le traduzioni.").format(
                lingua=lingua.nome(scelta))
            self._in_coda(testo, testo)
            return
        lingua.scegli(scelta)
        if scelta == lingua.CODICE:
            self.suoni.suona("confermato")
            testo = _("La lingua resta {lingua}.").format(lingua=lingua.nome(scelta))
            self._in_coda(testo, testo)
            return
        self.statistiche.conta("lingua_cambiata")
        self.suoni.suona("lingua_scelta")
        testo = _("La lingua sarà {lingua} dal prossimo avvio di Ciechino.").format(lingua=lingua.nome(scelta))
        self._in_coda(testo, testo)
        if self._domanda(_("Vuoi riaprire Ciechino adesso, per usare subito la lingua nuova?")):
            self._per_riaprire = True
            self.Close()

    # L'aiuto.

    def guida(self):
        """F1: la guida ai comandi, dalla tabella dei menu, in coda alla vista."""
        self.suoni.suona("guida")
        testo = self.testo_della_guida()
        self._in_coda(testo, testo.splitlines()[0], fuoco=True)

    def manuale(self):
        """Maiusc+F1: il manuale nel browser predefinito; se manca, o Windows non lo apre, lo dice la vista."""
        percorso = percorsi.percorso_risorsa("manuale.html")
        if not os.path.isfile(percorso):
            self.suoni.suona("file_mancante")
            testo = _("Il manuale non c'è: dovrebbe essere il file {percorso}. Reinstallare Ciechino lo rimette al suo posto.").format(percorso=percorso)
            self._in_coda(testo, testo)
            return
        try:
            os.startfile(percorso)  # noqa: S606 - il manuale si apre col browser che Windows associa all'HTML
        except OSError as errore:
            self.suoni.suona("file_mancante")
            testo = _("Il manuale non si apre nel browser: {errore}. È il file {percorso}.").format(errore=errore.strerror or errore, percorso=percorso)
            self._in_coda(testo, testo)
            return
        self.suoni.suona("manuale")
        testo = _("Il manuale si apre nel browser.")
        self._in_coda(testo, testo)

    def novita(self):
        """F2: le novità, dal CHANGELOG.md reso leggibile, in coda alla vista."""
        try:
            with open(percorsi.percorso_risorsa("CHANGELOG.md"), encoding="utf-8") as file:
                testo = testi.novita(file.read())
        except OSError as errore:
            self.suoni.suona("file_mancante")
            testo = _("Il file delle novità non si può leggere: {errore}.").format(errore=errore.strerror or errore)
            self._in_coda(testo, testo)
            return
        self.suoni.suona("novita")
        testo = testo or _("Il file delle novità è vuoto.")
        self._in_coda(testo, testo.splitlines()[0], fuoco=True)

    def informazioni(self):
        """F3: nome, versione, autori, licenza e repository, in coda alla vista."""
        self.suoni.suona("informazioni")
        testo = testi.informazioni()
        self._in_coda(testo, testo.splitlines()[0], fuoco=True)

    def caffe(self):
        """Offrimi un caffè: l'invito, solo a richiesta, nel dialogo con Dona con PayPal e Chiudi."""
        from GBUtils import Donazione

        # La lingua è quella in cui Ciechino parla adesso, come in MeTeOra: con una traduzione installata
        # l'invito non resta in italiano mentre il titolo e i pulsanti del dialogo sono tradotti (D9).
        invito = Donazione(lang=lingua.CODICE, probabilita=100, stampa=False) or _("Grazie se vorrai offrire un caffè all'autore.")
        self.suoni.suona("caffe")
        dialogo = dialoghi.Caffe(self, invito, self.suoni.suona)
        try:
            self._modale(dialogo)
        finally:
            dialogo.Destroy()

    # Il PUB, il gruppo di supporto: indirizzi, informazioni sul computer e mail da GBUtils, la finestra
    # della mail da GBwx, le parole da pub.py. La finestra si apre sempre, anche quando il programma di
    # posta c'è, e senza posta guida un passo alla volta.

    def entra_al_pub(self):
        """Entra al PUB: la mail che iscrive al gruppo di supporto."""
        self.suoni.suona("pub")
        self._mail_al_pub(pub.mail_d_iscrizione(), _("La mail d'iscrizione al PUB è pronta: spediscila tu."), _("Entra al PUB: niente di preparato."))

    def _scheda_audio(self):
        """La scheda audio, come la dice la mail d'aiuto, o None se Ciechino è muto."""
        stato = self.motore.stato()
        if not stato.get("aperto"):
            return None
        return ", ".join(str(parte) for parte in (stato.get("dispositivo"), stato.get("interfaccia")) if parte) or None

    def chiedi_aiuto_al_pub(self):
        """
        Chiedi aiuto al PUB: la mail al gruppo, con l'oggetto che dice la versione, la riga da sostituire
        con la domanda e le informazioni su Ciechino, sulla scheda audio e sul computer. Se ci sono registri
        degli errori o dei crash, la casella li allega, ripuliti dal nome dell'utente e del computer.
        """
        from GBUtils import informazioni_di_sistema

        self.suoni.suona("pub_aiuto")
        argomenti = (costanti.VERSIONE, bool(getattr(sys, "frozen", False)), informazioni_di_sistema(","), self._scheda_audio())
        cartella = percorsi.cartella_programma()
        presenti = pub.registri_presenti(cartella)
        mail = pub.mail_d_aiuto(*argomenti, casella=bool(presenti))
        spuntata = [False]

        def cambi(con_i_registri):
            # Il file si scrive quando la casella si spunta, ripulito dal nome dell'utente e del computer.
            spuntata[0] = con_i_registri
            if not con_i_registri:
                return {}
            try:
                registri = pub.scrivi_i_registri(cartella, costanti.VERSIONE, os.environ.get("USERNAME", ""), os.environ.get("COMPUTERNAME", ""))
            except OSError as errore:
                return {"spiegazione": mail["spiegazione"] + "\n" + _("Non riesco a scrivere il file dei registri: {errore}.").format(
                    errore=errore.strerror or errore)}
            return pub.mail_d_aiuto(*argomenti, casella=True, registri=registri) if registri else {}

        casella = (pub.CASELLA_DEI_REGISTRI, cambi) if presenti else None
        pronta = self._mail_al_pub(mail, _("La mail d'aiuto al PUB è pronta: scrivi la domanda e spediscila tu."), _("Chiedi aiuto al PUB: niente di preparato."),
                                   casella)
        if pronta and spuntata[0]:
            self.statistiche.conta("registri_al_pub")

    def _mail_al_pub(self, mail, pronta, annullata, casella=None):
        """La finestra di GBwx che prepara la mail; la vista dice se è pronta o se si è chiusa prima, col suono dell'annullamento."""
        from GBwx import finestra_mail

        self._modali += 1
        try:
            self._ferma_la_partita()
            riuscita = finestra_mail(self, mail["titolo"], mail["destinatario"], mail["oggetto"], mail["corpo"], mail["spiegazione"], mail["passi"],
                                     pub.PULSANTI, pub.SENZA_POSTA, casella=casella)
        finally:
            self._modali -= 1
            self._dopo_un_modale()
        if riuscita:
            self.statistiche.conta("mail_al_pub_pronte")
            self.suoni.suona("confermato")
        else:
            self.suoni.suona("annullato")
        testo = pronta if riuscita else annullata
        self._in_coda(testo, testo)
        return riuscita

    # L'aggiornamento automatico e l'istanza unica: li conducono aggiornamento.py e istanza.py, dai
    # loro fili, che chiamano questi metodi nel filo della finestra.

    def proponi_l_aggiornamento(self, versione_attuale, versione_nuova, note, attesa=None):
        """
        La proposta dell'aggiornamento, con le note della release rese leggibili come le novità: vero se si
        sceglie Aggiorna. Allo scadere dell'attesa il dialogo si chiude come Non adesso. Aggiorna, Non adesso
        e la proposta lasciata scadere hanno ciascuno il suo suono.
        """
        dialogo = dialoghi.Aggiornamento(self, versione_attuale, versione_nuova, testi.novita(note or ""), attesa, self.suoni.suona)
        try:
            self.suoni.suona("aggiornamento_disponibile")
            scelta = self._modale(dialogo)
            scaduta = dialogo.scaduta
        finally:
            dialogo.Destroy()
        if scelta == wx.ID_YES:
            self.suoni.suona("aggiornamento_scelto")
            testo = _("Scarico Ciechino {versione}: quando è pronto, Ciechino si chiude per installarlo, abbandonando la partita in corso, "
                      "e si riapre da solo.").format(versione=versione_nuova)
        elif scaduta:
            self.suoni.suona("aggiornamento_scaduto")
            testo = _("La proposta di Ciechino {versione} è scaduta: te la ripropongo al prossimo avvio.").format(versione=versione_nuova)
        else:
            self.suoni.suona("aggiornamento_rimandato")
            testo = _("Ciechino {versione} può aspettare: te lo ripropongo al prossimo avvio.").format(versione=versione_nuova)
        self._in_coda(testo, testo)
        return scelta == wx.ID_YES

    def avvisa_dell_aggiornamento(self, testo):
        """
        Un esito dell'aggiornamento, in un messaggio che aspetta l'OK: torna soltanto quando lo si è chiuso,
        perché i 30 secondi dello script di sostituzione partano dopo. Il suono e l'icona li sceglie la
        frase; il testo resta anche nella vista.
        """
        esito = aggiornamento.esito(testo)
        if esito in (aggiornamento.NON_RIUSCITO, aggiornamento.NON_PRONTO):
            # Lo scaricamento è finito male: la sua riga non resta nella barra, che direbbe ancora «scaricato al 63%».
            self._togli_la_riga_dell_aggiornamento()
        self.suoni.suona(SUONI_DEGLI_ESITI.get(esito, "aggiornamento_disponibile"))
        self._in_coda(testo)
        self._messaggio(testo, wx.OK | ICONE_DEGLI_ESITI.get(esito, wx.ICON_INFORMATION), _("Aggiornamento di Ciechino"))

    def avanzamento_dell_aggiornamento(self, presi, totale):
        """A che punto è lo scaricamento, in una riga in fondo alla barra di stato."""
        if not self or self.chiusa:
            return
        if totale:
            # Traduttori: {percentuale} è come 45%.
            self._riga_dell_aggiornamento = _("Aggiornamento scaricato al {percentuale}.").format(percentuale=testi.percento(min(presi / totale, 1.0)))
        else:
            # Traduttori: {mega} è come 3,2.
            self._riga_dell_aggiornamento = _("Aggiornamento: {mega} MB scaricati.").format(mega=testi.numero(presi / MEGABYTE, 1))
        self.aggiorna_barra()

    def _togli_la_riga_dell_aggiornamento(self):
        self._riga_dell_aggiornamento = None
        self.aggiorna_barra()

    def chiudi_per_aggiornare(self):
        """
        GBUtils sta per sostituire Ciechino: la finestra chiude i dialoghi aperti, come annullati, e si
        chiude senza domande, abbandonando la partita in corso. Il dialogo del nome in classifica no: lo
        conferma chi gioca, come un messaggio aperto, che non si può chiudere da qui; e prima di chiudersi
        la finestra aspetta anche le partite finite che attendono la classifica, che il battito porta al
        dialogo del nome. La chiusura riprova, ma per LIMITE_DELLA_CHIUSURA secondi al massimo: dopo, lo
        script ha rinunciato, e la vista lo dice col suono dell'aggiornamento non riuscito.
        """
        if not self or self.chiusa:
            return
        if self._chiusura_dal is None:
            self._chiusura_dal = time.monotonic()
        # La fine di una partita che aspetta ancora la fine del tonfo arriva adesso, e la chiusura aspetta la sua classifica,
        # che il battito porta al dialogo del nome (D22).
        self._accogli(self._messaggi_in_attesa())
        aperti = [d for d in wx.GetTopLevelWindows() if isinstance(d, wx.Dialog) and d.IsModal()]
        for dialogo in aperti:
            # Chiuso come annullato, il nome perderebbe la voce della classifica.
            if not isinstance(dialogo, dialoghi.Nome):
                dialogo.EndModal(wx.ID_CANCEL)
        if aperti or self._modali or self._partite_da_concludere or self._nome_in_attesa is not None:
            if time.monotonic() - self._chiusura_dal < LIMITE_DELLA_CHIUSURA:
                wx.CallLater(RIPROVA_DELLA_CHIUSURA, self.chiudi_per_aggiornare)
            else:
                self._chiusura_dal = None
                self._togli_la_riga_dell_aggiornamento()
                self.suoni.suona("aggiornamento_non_riuscito")
                testo = _("L'aggiornamento non si è installato: Ciechino doveva chiudersi entro 30 secondi, ma un messaggio è rimasto aperto. "
                          "Continui con questa versione, e te lo ripropongo al prossimo avvio.")
                self._in_coda(testo, testo)
            return
        self._per_aggiornare = True
        self.Close(force=True)

    def porta_in_primo_piano(self):
        """Si è provato ad aprire Ciechino una seconda volta: questa finestra torna in primo piano, col suo suono, e l'altra copia si chiude."""
        if not self or self.chiusa:
            return
        if self.IsIconized():
            self.Iconize(False)
        self.Raise()
        self.suoni.suona("primo_piano")

    # Le notizie del motore audio: il callback trasforma ogni eccezione in una notifica, e il filo di
    # PortAudio non arriva mai alla rete degli errori; qui le notifiche diventano righe del registro.

    def _controlla_il_motore(self):
        """A ogni battito il motore guarda se il flusso è vivo: perso, passa al modo muto e lo notifica. Un motore che non lo sa fare non si guarda."""
        controlla = getattr(self.motore, "controlla", None)
        if not callable(controlla):
            return
        try:
            controlla()
        except Exception:  # noqa: BLE001 - si ripeterebbe a ogni battito
            self._guasto("motore")

    def _notizie_del_motore(self):
        """
        Svuota le notifiche del motore. Un errore va nel registro la prima volta, con la riga che la
        notifica sa dire, pila compresa, e poi, per la stessa origine e lo stesso testo, si contano
        soltanto le ripetizioni; i buchi si contano; il flusso perso va nel registro, e la vista e la
        sintesi lo dicono, perché da lì il gioco continua muto. Le voci finite e gli eventi scattati non
        sono notizie per il registro.
        """
        notifiche = getattr(self.motore, "notifiche", None)
        if not callable(notifiche):
            return
        try:
            for notifica in notifiche() or ():
                tipo = getattr(notifica, "tipo", None)
                if tipo == "errore":
                    self._errore_del_motore(notifica)
                elif tipo == "buco":
                    # Il motore conta i buchi da sé, anche quelli delle notifiche perse in una coda piena.
                    self._buchi_da_dire += 1
                    self._buchi_in_tutto = max(self._buchi_in_tutto + 1, int(getattr(notifica, "volte", 0) or 0))
                elif tipo == "flusso_perso":
                    self._flusso_perso(notifica)
            self._riassunto_del_motore()
        except Exception:  # noqa: BLE001 - si ripeterebbe a ogni battito
            self._guasto("notizie")

    @staticmethod
    def _riga_della_notifica(notifica, ripiego):
        """La riga del registro che la notifica sa dire da sé, con la pila alla prima volta; altrimenti il ripiego."""
        riga = getattr(notifica, "riga", None)
        if callable(riga):
            with contextlib.suppress(Exception):
                return riga()
        return ripiego

    def _errore_del_motore(self, notifica):
        oggetto = getattr(notifica, "oggetto", None)
        dove = getattr(notifica, "origine", "") or (type(oggetto).__name__ if oggetto is not None else "blocco")
        testo = getattr(notifica, "testo", "") or ""
        chiave = (dove, testo)
        if chiave in self._errori_del_motore:
            self._errori_del_motore[chiave] += 1
            return
        self._errori_del_motore[chiave] = 0
        registro.scrivi(self._riga_della_notifica(notifica, f"Errore nel filo dell'audio al campione {getattr(notifica, 'campione', '?')} ({dove}): {testo}"))

    def _flusso_perso(self, notifica):
        if self._flusso_perso_detto:
            return
        self._flusso_perso_detto = True
        motivo = getattr(notifica, "testo", "") or getattr(self.motore, "ultimo_errore", None) or "il callback non arriva più"
        registro.scrivi(self._riga_della_notifica(notifica, f"Flusso audio perso al campione {getattr(notifica, 'campione', '?')}: {motivo}. "
                                                            "Ciechino continua senza suono."))
        # Il suono torna scegliendo la scheda con Ctrl+Maiusc+X (D20): il motore la rilegge e la riapre, e il
        # flusso perso, se torna a perdersi, si dice di nuovo.
        testo = _("La scheda audio non risponde più, forse si è staccata: Ciechino continua senza suono. Per riavere il suono scegli la scheda "
                  "con Ctrl+Maiusc+X.")
        self._in_coda(testo, testo)

    def _riassunto_del_motore(self, finale=False):
        """
        Ogni INTERVALLO_DEI_RIASSUNTI secondi, e all'uscita, una riga per i buchi del flusso e una per ogni
        errore che si è ripetuto da quando l'ultima riga l'ha detto; niente, se non c'è niente di nuovo.
        """
        adesso = time.monotonic()
        if not finale and adesso - self._ultimo_riassunto < INTERVALLO_DEI_RIASSUNTI:
            return
        secondi = round(adesso - self._ultimo_riassunto)
        self._ultimo_riassunto = adesso
        if self._buchi_da_dire:
            registro.scrivi(f"Buchi nel flusso audio: {self._buchi_da_dire} negli ultimi {secondi} secondi, {self._buchi_in_tutto} da quando Ciechino è aperto.")
            self._buchi_da_dire = 0
        for (dove, testo), volte in self._errori_del_motore.items():
            if volte:
                registro.scrivi(f"L'errore del filo dell'audio in {dove} si è ripetuto ancora {volte} {'volta' if volte == 1 else 'volte'} "
                                f"negli ultimi {secondi} secondi: {testo}")
                self._errori_del_motore[(dove, testo)] = 0

    # La chiusura.

    def _suona_e_aspetta(self, evento):
        """Il suono del congedo, e l'attesa che finisca, fino a ATTESA_MASSIMA_DELL_USCITA: dopo, il motore si chiude e lo troncherebbe."""
        voce = self.suoni.suona(evento)
        if voce is None or self.motore.muto:
            return
        attesa = self._secondi_alla_fine(voce.inizio + len(voce.buffer))
        time.sleep(min(max(attesa, 0.0), ATTESA_MASSIMA_DELL_USCITA))

    def _alla_chiusura(self, event):
        """
        L'uscita: con una partita in corso prima una domanda, col no già scelto, salvo quando la chiusura
        non si può rifiutare, come per l'aggiornamento. Poi, se la chiusura si poteva rifiutare, le partite
        finite che aspettano ancora il nome o la classifica li ricevono: anche una partita finita mentre la
        domanda aspettava. Poi la partita in corso si abbandona, senza chiedere la classifica, le statistiche
        si salvano, il registro riceve l'ultimo riassunto del motore, suona il congedo, il motore si chiude
        e la finestra se ne va. Se si è scelto di riaprire Ciechino per la lingua nuova, la copia nuova
        parte qui, e aspetta che questa sia uscita.
        """
        if self.chiusa:
            return
        if event.CanVeto() and self.partita.in_corso():
            if not self._domanda(_("Una partita è in corso: se esci, la partita si abbandona. Esci lo stesso?")):
                self._per_riaprire = False
                event.Veto()
                return
        # I messaggi dell'ultimo tiro che aspettavano la fine del suo effetto (D22): la fine di una partita
        # chiusa mentre suona il tonfo va in classifica come le altre.
        self._accogli(self._messaggi_in_attesa())
        if event.CanVeto():
            self._chiudi_i_conti()
        self.chiusa = True
        # Il filo dell'aggiornamento che aspettava gli Ascolti dell'avvio non resta appeso (D23).
        self._libera_l_avvio()
        if callable(self.al_congedo):
            with contextlib.suppress(Exception):
                self.al_congedo()
        self.timer.Stop()
        try:
            if self.partita.in_corso():
                self._ricevi(self.partita.abbandona(), con_la_classifica=False)
        except Exception:  # noqa: BLE001 - una partita guasta non deve impedire l'uscita: il resoconto va nel registro
            registro.problema(*sys.exc_info())
        self._salva_le_statistiche()
        with contextlib.suppress(Exception):
            self._notizie_del_motore()
            self._riassunto_del_motore(finale=True)
        suono = "uscita"
        if self._per_aggiornare:
            self.congedo = PER_AGGIORNARE
        elif self._per_riaprire:
            try:
                istanza.riapri()
            except OSError as errore:
                registro.scrivi(f"La riapertura di Ciechino non è partita: {errore}.")
            else:
                self.congedo = RIAPERTURA
                suono = "riavvio"
        self._suona_e_aspetta(suono)
        self.motore.chiudi()
        self.Destroy()
