"""
I dialoghi di Ciechino: il testo da leggere, il nome per la classifica, il volume, la forma del
proiettile, la scheda audio, la lingua, l'invito al caffè e la proposta dell'aggiornamento automatico.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di gui/dialoghi.py di MESS. Ogni dialogo segue lo schema
di GBwx: stile adattabile, controlli in un pannello che scorre, misura presa dal contenuto con
adatta_finestra, così i caratteri grandi non lasciano niente fuori dallo schermo; poi si apre
massimizzato, come la finestra principale, perché con lo screen reader e il display braille il testo
largo va meno a capo; ed Esc preme il suo pulsante Chiudi, se ce l'ha. Ogni campo ha davanti la sua
etichetta, creata subito prima di lui, da cui NVDA prende il nome, che gli dà anche SetName.
I dialoghi non suonano da sé: ricevono suona, la funzione dei suoni della finestra, così le prove li
aprono muti. I messaggi di wx passano da messaggio, che le prove sostituiscono, perché un messaggio
vero aspetterebbe per sempre un tasto.
Da MESS viene anche la grazia dell'apertura: NVDA, nei campi di testo, rimanda ogni ripetizione di un
tasto tenuto giù come una pressione nuova, e IsAutoRepeat non la vede. Un dialogo con
GRAZIA_ALL_APERTURA ignora gli Invio che arrivano entro GRAZIA_INVIO dall'apertura o dall'Invio di
prima. Quello del nome ignora anche lo spazio, entro la grazia dall'apertura: il nome si chiede a
partita finita, quando chi gioca può avere ancora il dito sullo spazio, e uno spazio arrivato lì
cancellerebbe il nome proposto, che all'apertura è tutto selezionato.
"""

import contextlib
import time
import webbrowser

import wx
from GBwx import STILE_ADATTABILE, adatta_finestra, pannello_scorrevole

import classifica
import costanti
import impostazioni as modulo_impostazioni
import schede
import testi
from lingua import _

# L'indirizzo dell'invito al caffè, lo stesso di MESS e di MeTeOra: GBUtils non ne ha uno suo.
PAYPAL = "https://paypal.me/GabrieleBattaglia780"
# Il tempo entro cui un Invio dopo l'apertura, o dopo l'Invio di prima, vale come tasto tenuto giù, in
# secondi: gli 0,65 della grazia di GBwx 1.1.1, provati da Gabriele con NVDA il 23 ottobre 2026.
GRAZIA_INVIO = 0.65
# Quanto aspetta, dopo l'ultima cifra scritta o l'ultima freccia, la prova del volume, in millesimi:
# così non suona a ogni cifra. È la misura di MESS.
RITARDO_DEL_SUONO_AL_VOLO = 350
# Le misure minime, in pixel al 100%, e quelle a cui un dialogo torna se lo si ripristina: adatta_finestra
# le allarga quando il contenuto non ci sta, e il dialogo si apre comunque massimizzato.
MISURA_DELLA_LETTURA = (560, 420)
MISURA_DEI_CAMPI = (440, 240)
# Il margine attorno ai controlli, in pixel al 100%, come in MESS.
BORDO = 8
SPAZIO_FRA_I_PULSANTI = 4
TASTI_INVIO = (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER)
MILLESIMI = 1000
# Le forme d'onda del proiettile (D17), una per ogni voce di costanti.PROIETTILE_FORME: il nome, come lo
# dicono i messaggi, «il proiettile suona in seno», e il carattere, poche parole sul suo timbro.
NOMI_DEI_PROIETTILI = {
    # Traduttori: da qui, i nomi delle forme d'onda del proiettile, come si dicono dopo «in»: «il proiettile suona in seno».
    "seno": _("seno"), "triangolo": _("triangolo"), "sega_morbida": _("sega morbida"),
}
CARATTERI_DEI_PROIETTILI = {
    # Traduttori: da qui, il carattere del suono di ogni forma d'onda del proiettile, nell'elenco del dialogo.
    "seno": _("il suono più puro e morbido, quasi un fischio"),
    "triangolo": _("un poco più chiaro e vivo del seno, ancora dolce"),
    "sega_morbida": _("il più ricco e brillante, con un leggero ronzio"),
}


def _muto(*_argomenti, **_opzioni):
    """La funzione dei suoni di un dialogo che non ne ha ricevuta una: non suona niente."""


def _adesso():
    return time.monotonic()


def messaggio(genitore, testo, titolo, stile=wx.OK | wx.ICON_INFORMATION):
    """Un messaggio di wx, o una domanda con YES_NO: una funzione a sé, perché le prove la sostituiscono."""
    return wx.MessageBox(testo, titolo, stile, genitore)


def resoconto_degli_errori():
    """Dove sta il resoconto completo di un errore imprevisto, da dire nel messaggio che lo annuncia."""
    import percorsi
    import registro

    if not registro.avviato():
        return _("Il resoconto non si è potuto scrivere, perché il registro degli errori non è aperto.")
    # Traduttori: {file} è il nome del registro degli errori, {cartella} la cartella del programma.
    return _("Il resoconto completo è nel file {file}, nella cartella {cartella}: se chiedi aiuto al PUB, dal menu Aiuto, puoi allegarlo.").format(
        file=registro.FILE_DEGLI_ERRORI, cartella=percorsi.cartella_programma())


def invio_da_ignorare(dialogo, event):
    """Vero se l'Invio arriva entro la grazia dall'apertura del dialogo o dall'Invio precedente: un tasto tenuto giù."""
    adesso = _adesso()
    precedente, dialogo.ultimo_invio = dialogo.ultimo_invio, adesso
    return event.IsAutoRepeat() or adesso - precedente < GRAZIA_INVIO


def etichetta_del_nome():
    """L'etichetta del campo del nome in classifica, con le misure di costanti.py."""
    # Traduttori: {minimo} e {massimo} sono i caratteri del nome, come 1 e 20; la e commerciale segna la lettera del tasto rapido.
    return _("&Nome, da {minimo} a {massimo} caratteri").format(minimo=costanti.NOME_MINIMO, massimo=costanti.NOME_MASSIMO)


def _senza_tasto(testo):
    """Il testo di un'etichetta senza la e commerciale del tasto rapido, per SetName."""
    return testo.replace("&&", "\x00").replace("&", "").replace("\x00", "&")


def nome_del_proiettile(forma):
    """Il nome a parole di una forma d'onda del proiettile, come lo dicono i messaggi: «sega morbida»."""
    return NOMI_DEI_PROIETTILI.get(forma, str(forma).replace("_", " "))


def riga_del_proiettile(forma):
    """La riga dell'elenco del dialogo del proiettile: il nome, con la maiuscola, e il carattere del suono."""
    nome = nome_del_proiettile(forma)
    carattere = CARATTERI_DEI_PROIETTILI.get(forma)
    if not carattere:
        return nome[:1].upper() + nome[1:]
    # Traduttori: una riga dell'elenco delle forme del proiettile, come «Seno, il suono più puro e morbido, quasi un fischio».
    return _("{nome}, {carattere}").format(nome=nome[:1].upper() + nome[1:], carattere=carattere)


def riga_dell_automatica(in_uso=False, scheda=""):
    """
    La prima riga dell'elenco delle schede audio (D20): la scelta automatica, che segue la scheda di
    Windows. Quando Ciechino suona lì, la riga lo dice, col nome della scheda, se c'è.
    """
    if not in_uso:
        return _("Automatica, la scheda di Windows")
    if scheda:
        # Traduttori: la prima riga dell'elenco delle schede audio, quando Ciechino suona sulla scelta automatica; {scheda} è come «Altoparlanti (Realtek(R) Audio), WASAPI».
        return _("Automatica, la scheda di Windows, in uso: {scheda}").format(scheda=scheda)
    return _("Automatica, la scheda di Windows, in uso")


def riga_della_scheda(uscita, in_uso=False):
    """La riga di un'uscita nell'elenco delle schede audio: quella di schede.etichetta, e in coda «in uso» se Ciechino suona lì."""
    riga = schede.etichetta(uscita)
    if not in_uso:
        return riga
    # Traduttori: la riga della scheda audio su cui suona Ciechino, nell'elenco delle schede; {scheda} è come «Cuffie (WH-1000XM4), WASAPI, 3 ms».
    return _("{scheda}, in uso").format(scheda=riga)


class _Dialogo(wx.Dialog):
    """
    La base dei dialoghi: pannello che scorre, sizer verticale, apertura massimizzata, Esc che preme
    Chiudi. suona è la funzione dei suoni della finestra, suona(evento). Un dialogo con
    GRAZIA_ALL_APERTURA vera conta l'apertura come un Invio, e ignora gli Invio che arrivano entro la
    grazia dall'apertura o dall'Invio di prima, ovunque sia il fuoco.
    """

    GRAZIA_ALL_APERTURA = False

    def __init__(self, genitore, titolo, suona=None):
        super().__init__(genitore, title=titolo, style=STILE_ADATTABILE)
        self.suona = suona if suona is not None else _muto
        self.pannello = pannello_scorrevole(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        # Il pulsante che Esc preme, se il dialogo ne ha uno: lo sceglie pulsanti, col suo indice.
        self.pulsante_chiudi = None
        self.ultimo_invio = float("-inf")
        self.aperto_alle = _adesso()
        self._timer_del_suono = None

    def spiegazione(self, testo):
        """Una frase che spiega il dialogo, prima dei campi: NVDA la legge all'apertura."""
        controllo = wx.StaticText(self.pannello, label=testo)
        self.sizer.Add(controllo, 0, wx.ALL, BORDO)
        return controllo

    def etichetta(self, testo):
        controllo = wx.StaticText(self.pannello, label=testo)
        self.sizer.Add(controllo, 0, wx.LEFT | wx.RIGHT | wx.TOP, BORDO)
        return controllo

    def aggiungi(self, controllo, proporzione=0):
        self.sizer.Add(controllo, proporzione, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, BORDO)
        return controllo

    def campo(self, testo, crea, proporzione=0):
        """
        Un campo con la sua etichetta e lo stesso nome per NVDA. crea è la funzione che crea il
        controllo: la si chiama dopo l'etichetta, perché NVDA cerca l'etichetta di un controllo fra
        quelli creati subito prima di lui, e un controllo passato già fatto sarebbe nato prima.
        """
        self.etichetta(testo)
        controllo = crea()
        controllo.SetName(_senza_tasto(testo))
        return self.aggiungi(controllo, proporzione)

    def pulsanti(self, *voci, predefinito=0, chiudi=None):
        """
        Una riga di pulsanti, da coppie di identificativo e testo. predefinito è l'indice del pulsante
        di Invio, None per nessuno; chiudi l'indice del pulsante Chiudi, che Esc preme.
        """
        riga = wx.BoxSizer(wx.HORIZONTAL)
        creati = []
        for identificativo, testo in voci:
            pulsante = wx.Button(self.pannello, identificativo, testo)
            riga.Add(pulsante, 0, wx.ALL, SPAZIO_FRA_I_PULSANTI)
            creati.append(pulsante)
        if predefinito is not None:
            creati[predefinito].SetDefault()
        if chiudi is not None:
            self.pulsante_chiudi = creati[chiudi]
        self.sizer.Add(riga, 0, wx.ALIGN_RIGHT | wx.ALL, SPAZIO_FRA_I_PULSANTI)
        return creati

    def completa(self, misura=MISURA_DEI_CAMPI):
        """
        Dà il sizer al pannello e al dialogo la misura del contenuto, poi le due regole di tutte le
        finestre interne, da MESS: il dialogo si apre massimizzato, e la misura di adatta_finestra resta
        quella a cui torna se lo si ripristina; e se ha un pulsante Chiudi, Esc lo preme, perché wx lo fa
        da sé soltanto quando Chiudi è wx.ID_CANCEL. Un dialogo che sceglie da sé il pulsante di Esc, come
        l'aggiornamento col suo Non adesso, tiene il suo.
        """
        self.pannello.SetSizer(self.sizer)
        adatta_finestra(self, self.pannello, misura)
        self.Maximize()
        if self.pulsante_chiudi is not None and self.GetEscapeId() == wx.ID_ANY:
            self.SetEscapeId(self.pulsante_chiudi.GetId())
        self.aperto_alle = _adesso()
        if self.GRAZIA_ALL_APERTURA:
            # L'ultimo legame del gancio dei tasti: wx lo chiama per primo, prima di ogni altro e del pulsante predefinito.
            self.ultimo_invio = self.aperto_alle
            self.Bind(wx.EVT_CHAR_HOOK, self._grazia)

    def _grazia(self, event):
        """Un Invio entro la grazia dall'apertura o dall'Invio di prima si mangia; ogni altro tasto prosegue."""
        if event.GetKeyCode() in TASTI_INVIO and not event.HasAnyModifiers() and invio_da_ignorare(self, event):
            return
        event.Skip()

    def chiudi(self, codice):
        """Chiude il dialogo con il codice dato; fuori da ShowModal, come nelle prove, lo annota soltanto."""
        if self.IsModal():
            self.EndModal(codice)
        else:
            self.SetReturnCode(codice)

    def avvisa(self, testo, controllo, suono="campo_da_correggere"):
        """Un avviso che lascia il dialogo aperto, con il cursore sul campo da correggere, e il suo suono."""
        self.suona(suono)
        messaggio(self, testo, self.GetTitle(), wx.OK | wx.ICON_WARNING)
        controllo.SetFocus()

    def suona_fra_poco(self, funzione):
        """Chiama funzione poco dopo l'ultima richiesta: un suono che segue un campo suona una volta sola, quando ci si ferma."""
        if self._timer_del_suono is not None and self._timer_del_suono.IsRunning():
            self._timer_del_suono.Restart(RITARDO_DEL_SUONO_AL_VOLO, funzione)
        else:
            self._timer_del_suono = wx.CallLater(RITARDO_DEL_SUONO_AL_VOLO, self._rimandato, funzione)

    @staticmethod
    def _rimandato(funzione):
        # Se nel frattempo il dialogo si è chiuso, i suoi controlli non ci sono più: niente da fare.
        with contextlib.suppress(RuntimeError):
            funzione()


class Lettura(_Dialogo):
    """
    Un testo da leggere in un'area in sola lettura, una voce per riga, con il pulsante Chiudi, che vale
    anche per Esc: la classifica e le statistiche (D11), l'invito al caffè e la proposta
    dell'aggiornamento. pulsanti, se c'è, sostituisce Chiudi con altre coppie di identificativo e testo:
    l'ultima è quella che chiude.
    """

    def __init__(self, genitore, titolo, testo, pulsanti=None, suona=None):
        super().__init__(genitore, titolo, suona)
        self.testo = self.campo(titolo, lambda: wx.TextCtrl(self.pannello, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2, value=testo), 1)
        voci = pulsanti or ((wx.ID_OK, _("&Chiudi")),)
        self.bottoni = self.pulsanti(*voci, predefinito=0, chiudi=len(voci) - 1)
        self.bottoni[-1].Bind(wx.EVT_BUTTON, lambda event: self.chiudi(wx.ID_OK))
        self.Bind(wx.EVT_CHAR_HOOK, self._tasto)
        self.completa(MISURA_DELLA_LETTURA)
        wx.CallAfter(self._al_testo)

    def _al_testo(self):
        with contextlib.suppress(RuntimeError):
            self.testo.SetInsertionPoint(0)
            self.testo.SetFocus()

    def _tasto(self, event):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.chiudi(wx.ID_OK)
        else:
            event.Skip()


class Caffe(Lettura):
    """L'invito a offrire un caffè, solo a richiesta dal menu Aiuto, con Dona con PayPal e Chiudi, il predefinito."""

    def __init__(self, genitore, testo, suona=None):
        super().__init__(genitore, _("Offrimi un caffè"), testo, ((wx.ID_ANY, _("Dona con &PayPal")), (wx.ID_OK, _("&Chiudi"))), suona)
        self.bottoni[0].Bind(wx.EVT_BUTTON, self.dona)
        # Invio chiude: donare deve essere una scelta, non un tasto premuto per abitudine.
        self.bottoni[-1].SetDefault()
        # L'apertura conta come un Invio: quello tenuto giù dalla voce del menu non chiude l'invito appena aperto.
        self.ultimo_invio = _adesso()

    def _tasto(self, event):
        """
        Invio nel testo chiude come Chiudi: l'area in sola lettura su più righe, che ha il fuoco, si tiene
        Invio, e il pulsante predefinito da solo non basta. Sui pulsanti Invio resta loro; un Invio
        tenuto giù non risponde, anche con NVDA.
        """
        if event.GetEventObject() is self.testo and event.GetKeyCode() in TASTI_INVIO and not invio_da_ignorare(self, event):
            self.chiudi(wx.ID_OK)
        else:
            super()._tasto(event)

    def dona(self, event=None):
        self.suona("caffe_paypal")
        with contextlib.suppress(webbrowser.Error):
            webbrowser.open(PAYPAL)
        self.chiudi(wx.ID_OK)


class Aggiornamento(Lettura):
    """
    La proposta dell'aggiornamento automatico: nel testo in sola lettura, che ha il fuoco all'apertura,
    la versione nuova, quella che si ha e le note della release; i pulsanti Aggiorna e Non adesso, che è
    il predefinito. Esc, e Invio nel testo, valgono Non adesso: aggiornare si sceglie solo con il suo
    pulsante. Con attesa, in secondi, il dialogo si chiude da solo come Non adesso allo scadere. Il
    codice di ShowModal è wx.ID_YES per aggiornare, wx.ID_NO altrimenti; scaduta dice se il no è venuto
    dallo scadere dell'attesa.
    """

    def __init__(self, genitore, versione_attuale, versione_nuova, note, attesa=None, suona=None):
        # Traduttori: {nuova} e {attuale} sono versioni, come 0.2.0.
        righe = [_("È disponibile Ciechino {nuova}; tu hai la {attuale}.").format(nuova=versione_nuova, attuale=versione_attuale),
                 _("Aggiorna scarica la versione nuova; poi Ciechino si chiude e si riapre aggiornato, con la tua classifica, le tue statistiche e "
                   "le tue impostazioni.")]
        if attesa:
            # Traduttori: {attesa} è una durata, come 2 minuti.
            righe.append(_("Se non rispondi entro {attesa}, il dialogo si chiude da solo e te lo ripropongo al prossimo avvio.").format(
                attesa=testi.durata(attesa)))
        righe += [_("Le note della release:"), (note or "").strip() or _("Nessuna nota scritta per questa versione.")]
        super().__init__(genitore, _("Aggiornamento di Ciechino"), "\n".join(righe), ((wx.ID_YES, _("&Aggiorna")), (wx.ID_NO, _("&Non adesso"))),
                         suona)
        self.bottoni[0].Bind(wx.EVT_BUTTON, lambda event: self.chiudi(wx.ID_YES))
        # Lettura lega l'ultimo pulsante alla chiusura con OK: qui vale Non adesso.
        self.bottoni[-1].Unbind(wx.EVT_BUTTON)
        self.bottoni[-1].Bind(wx.EVT_BUTTON, lambda event: self.chiudi(wx.ID_NO))
        self.bottoni[-1].SetDefault()
        self.SetEscapeId(wx.ID_NO)
        self.scaduta = False
        # L'apertura conta come un Invio: il dialogo compare da solo all'avvio, magari mentre si preme Invio altrove.
        self.ultimo_invio = _adesso()
        self._scadenza = wx.CallLater(round(attesa * MILLESIMI), self._scaduta) if attesa else None

    def _tasto(self, event):
        """Esc, e Invio nel testo, valgono Non adesso; sui pulsanti Invio resta loro. Un Invio tenuto giù non risponde."""
        tasto = event.GetKeyCode()
        nel_testo = event.GetEventObject() is self.testo and tasto in TASTI_INVIO and not invio_da_ignorare(self, event)
        if tasto == wx.WXK_ESCAPE or nel_testo:
            self.chiudi(wx.ID_NO)
        else:
            event.Skip()

    def chiudi(self, codice):
        if self._scadenza is not None:
            self._scadenza.Stop()
        super().chiudi(codice)

    def _scaduta(self):
        # Il dialogo può essere già chiuso, o distrutto.
        if self and self.IsModal():
            self.scaduta = True
            self.EndModal(wx.ID_NO)


class Nome(_Dialogo):
    """
    Il nome di chi entra in classifica (D11): il campo, con il nome proposto tutto selezionato, OK, il
    predefinito, e Annulla, che vale anche per Esc. OK con un nome che, pulito come in classifica, è
    troppo corto avvisa e resta aperto. In nome resta il nome scelto, pulito, o None.
    """

    GRAZIA_ALL_APERTURA = True

    def __init__(self, genitore, proposto="", spiegazione="", suona=None):
        super().__init__(genitore, _("Il tuo nome in classifica"), suona)
        if spiegazione:
            self.spiegazione(spiegazione)
        self.testo = self.campo(etichetta_del_nome(), lambda: wx.TextCtrl(self.pannello, value=classifica.pulisci_nome(proposto)))
        self.testo.SetMaxLength(costanti.NOME_MASSIMO)
        ok, _annulla = self.pulsanti((wx.ID_OK, _("&OK")), (wx.ID_CANCEL, _("&Annulla")))
        ok.Bind(wx.EVT_BUTTON, self.conferma)
        self.nome = None
        self.completa()
        self.testo.SetFocus()
        self.testo.SelectAll()

    def _grazia(self, event):
        """Oltre all'Invio, lo spazio entro la grazia dall'apertura, o tenuto giù, si mangia: è quello di chi stava giocando."""
        if event.GetKeyCode() == wx.WXK_SPACE and not event.HasAnyModifiers() and (event.IsAutoRepeat() or _adesso() - self.aperto_alle < GRAZIA_INVIO):
            return
        super()._grazia(event)

    def conferma(self, event=None):
        nome = classifica.pulisci_nome(self.testo.GetValue())
        if not classifica.nome_valido(nome):
            # Traduttori: {minimo} e {massimo} sono i caratteri del nome, come 1 e 20.
            self.avvisa(_("Scrivi il tuo nome, da {minimo} a {massimo} caratteri.").format(minimo=costanti.NOME_MINIMO, massimo=costanti.NOME_MASSIMO),
                        self.testo)
            return
        self.nome = nome
        self.chiudi(wx.ID_OK)


class Volume(_Dialogo):
    """
    Il volume di Ciechino, da VOLUME_MINIMO a VOLUME_MASSIMO delle impostazioni, in uno SpinCtrl. A ogni
    ritocco, quando ci si ferma, al_cambio riceve il valore scritto: la finestra lo fa sentire subito.
    In valore resta il volume scelto con OK, o None.
    """

    def __init__(self, genitore, volume, al_cambio=None, suona=None):
        super().__init__(genitore, _("Volume"), suona)
        self.spiegazione(_("Il volume cambia mentre lo scrivi, e lo senti subito: OK lo tiene, Annulla torna a quello di prima."))
        # Traduttori: {minimo} e {massimo} sono i capi della scala del volume, 0 e 100.
        testo = _("&Volume, da {minimo} a {massimo}").format(minimo=modulo_impostazioni.VOLUME_MINIMO, massimo=modulo_impostazioni.VOLUME_MASSIMO)
        self.numero = self.campo(testo, lambda: wx.SpinCtrl(self.pannello, min=modulo_impostazioni.VOLUME_MINIMO, max=modulo_impostazioni.VOLUME_MASSIMO,
                                                             initial=volume))
        ok, _annulla = self.pulsanti((wx.ID_OK, _("&OK")), (wx.ID_CANCEL, _("&Annulla")))
        ok.Bind(wx.EVT_BUTTON, self.conferma)
        self._al_cambio = al_cambio
        self.valore = None
        self.numero.Bind(wx.EVT_SPINCTRL, self._cambiato)
        self.numero.Bind(wx.EVT_TEXT, self._cambiato)
        self.completa()
        self.numero.SetFocus()

    def _cambiato(self, event=None):
        if event is not None:
            event.Skip()
        if self._al_cambio is not None:
            self.suona_fra_poco(self.prova)

    def prova(self):
        """Il volume scritto nel campo, a chi lo fa sentire."""
        if self._al_cambio is not None:
            self._al_cambio(self.numero.GetValue())

    def conferma(self, event=None):
        self.valore = self.numero.GetValue()
        self.chiudi(wx.ID_OK)


class SceltaProiettile(_Dialogo):
    """
    La forma d'onda del proiettile (D17): sotto la spiegazione, l'elenco delle forme di
    costanti.PROIETTILE_FORME, ognuna col nome e il carattere del suono, con la forma attuale già
    scelta; OK e Annulla, che vale anche per Esc. Spostandosi nell'elenco, quando ci si ferma, al_cambio
    riceve la forma della riga: la finestra ne fa sentire un esempio, e lo ferma quando ne parte un
    altro o quando il dialogo si chiude. In scelta resta la forma scelta con OK, o None. Come quello
    della lingua ha la grazia dell'apertura: l'Invio tenuto giù dalla voce del menu non conferma.
    """

    GRAZIA_ALL_APERTURA = True

    def __init__(self, genitore, forma, al_cambio=None, suona=None):
        super().__init__(genitore, _("Proiettile"), suona)
        self.spiegazione(_("La forma d'onda dà il timbro al proiettile in volo, e i bersagli ne usano un'altra, così non li confondi con lui. "
                           "Spostandoti nell'elenco senti un esempio di ogni forma. La forma scelta vale dalla prossima partita."))
        self.forme = tuple(costanti.PROIETTILE_FORME)
        righe = [riga_del_proiettile(f) for f in self.forme]
        self.elenco = self.campo(_("&Forma del proiettile"), lambda: wx.ListBox(self.pannello, choices=righe, style=wx.LB_SINGLE), 1)
        self.elenco.SetSelection(self.forme.index(forma if forma in self.forme else costanti.PROIETTILE_FORMA))
        ok, _annulla = self.pulsanti((wx.ID_OK, _("&OK")), (wx.ID_CANCEL, _("&Annulla")))
        ok.Bind(wx.EVT_BUTTON, self.conferma)
        self.elenco.Bind(wx.EVT_LISTBOX, self._cambiato)
        self.elenco.Bind(wx.EVT_LISTBOX_DCLICK, self.conferma)
        self._al_cambio = al_cambio
        self.scelta = None
        self.completa()
        self.elenco.SetFocus()

    def _cambiato(self, event=None):
        if event is not None:
            event.Skip()
        if self._al_cambio is not None:
            self.suona_fra_poco(self.prova)

    def prova(self):
        """La forma della riga scelta, a chi ne fa sentire l'esempio."""
        if self._al_cambio is None:
            return
        indice = self.elenco.GetSelection()
        if indice != wx.NOT_FOUND:
            self._al_cambio(self.forme[indice])

    def ferma_la_prova(self):
        """
        Niente più esempi: la finestra la chiama quando il dialogo si è chiuso, prima di distruggerlo,
        così l'esempio chiesto un attimo prima della chiusura, che aspettava la fine del ritardo, non
        suona a dialogo chiuso.
        """
        self._al_cambio = None
        if self._timer_del_suono is not None and self._timer_del_suono.IsRunning():
            self._timer_del_suono.Stop()

    def conferma(self, event=None):
        indice = self.elenco.GetSelection()
        if indice == wx.NOT_FOUND:
            self.avvisa(_("Scegli una forma dall'elenco."), self.elenco)
            return
        self.scelta = self.forme[indice]
        self.chiudi(wx.ID_OK)


class SceltaScheda(_Dialogo):
    """
    La scheda audio di Ciechino (D20), sul modello della scelta della lingua: la spiegazione, con la
    latenza dichiarata delle righe e la via a orecchio per tornare all'automatica se una scheda
    esclusiva zittisce NVDA, poi le notizie, frasi che la finestra aggiunge, come quella che dice
    dove suona Ciechino adesso; poi
    l'elenco «Scheda audio», con prima la riga dell'automatica, la scheda di Windows, e poi una riga per
    ogni uscita di uscite, l'elenco di schede.uscite(), detta con schede.etichetta; OK e Annulla, che vale
    anche per Esc; la grazia dell'apertura, perché l'Invio tenuto giù dalla voce del menu non confermi.
    in_uso è la riga su cui suona Ciechino, zero per l'automatica, o None se è muto: quella riga lo dice,
    e per l'automatica con il nome della scheda in automatica. selezionata è la riga scelta
    all'apertura; None vuol dire quella in uso, o l'automatica se nessuna è in uso.
    Attributi uscite e righe; dopo OK indice, la riga scelta, e scelta, la scelta da salvare nelle
    impostazioni, come da schede.da_salvare: il dizionario vuoto per l'automatica. Con Annulla restano
    None.
    """

    GRAZIA_ALL_APERTURA = True

    def __init__(self, genitore, uscite, in_uso=None, automatica="", selezionata=None, notizie=(), suona=None):
        super().__init__(genitore, _("Scheda audio"), suona)
        # La latenza delle righe è quella che la scheda dichiara; la vista, dopo la scelta, dice quella del flusso aperto,
        # col blocco del gioco. Le esclusive possono zittire NVDA a ogni avvio: la via del ritorno va detta, e va a orecchio.
        self.spiegazione(_("Automatica suona sulla scheda di Windows, e la riga della scheda su cui suona Ciechino finisce con in uso. "
                           "Ogni riga dice la latenza che la scheda dichiara, di solito più bassa di quella che la vista dice dopo la scelta. "
                           "La scheda scelta vale da subito e resta anche per i prossimi avvii. Se una scheda esclusiva zittisce NVDA, per "
                           "tornare all'automatica premi Ctrl+Maiusc+X, aspetta il suono della scelta, poi premi Inizio e, dopo un secondo, Invio."))
        for notizia in notizie:
            if notizia:
                self.spiegazione(notizia)
        self.uscite = list(uscite)
        self.righe = [riga_dell_automatica(in_uso == 0, automatica), *(riga_della_scheda(uscita, in_uso == n) for n, uscita in enumerate(self.uscite, 1))]
        self.elenco = self.campo(_("&Scheda audio"), lambda: wx.ListBox(self.pannello, choices=self.righe, style=wx.LB_SINGLE), 1)
        if selezionata is None:
            selezionata = in_uso or 0
        self.elenco.SetSelection(min(max(selezionata, 0), len(self.righe) - 1))
        ok, _annulla = self.pulsanti((wx.ID_OK, _("&OK")), (wx.ID_CANCEL, _("&Annulla")))
        ok.Bind(wx.EVT_BUTTON, self.conferma)
        self.elenco.Bind(wx.EVT_LISTBOX_DCLICK, self.conferma)
        self.indice = None
        self.scelta = None
        self.completa()
        self.elenco.SetFocus()

    def conferma(self, event=None):
        indice = self.elenco.GetSelection()
        if indice == wx.NOT_FOUND:
            self.avvisa(_("Scegli una scheda dall'elenco."), self.elenco)
            return
        self.indice = indice
        self.scelta = schede.da_salvare(self.uscite[indice - 1] if indice else None)
        self.chiudi(wx.ID_OK)


class SceltaLingua(_Dialogo):
    """
    La lingua di Ciechino, sul modello della voce Lingua di MeTeOra: un elenco delle lingue che hanno un
    catalogo, con il nome nella lingua stessa e il codice; la scelta vale dal prossimo avvio. In scelta
    resta l'indice della riga scelta con OK, o None.
    """

    GRAZIA_ALL_APERTURA = True

    def __init__(self, genitore, righe, scelta=0, suona=None):
        super().__init__(genitore, _("Lingua"), suona)
        self.spiegazione(_("La lingua scelta vale dal prossimo avvio di Ciechino."))
        self.elenco = self.campo(_("&Lingua"), lambda: wx.ListBox(self.pannello, choices=list(righe), style=wx.LB_SINGLE), 1)
        if righe:
            self.elenco.SetSelection(min(max(scelta, 0), len(righe) - 1))
        ok, _annulla = self.pulsanti((wx.ID_OK, _("&OK")), (wx.ID_CANCEL, _("&Annulla")))
        ok.Bind(wx.EVT_BUTTON, self.conferma)
        self.elenco.Bind(wx.EVT_LISTBOX_DCLICK, self.conferma)
        self.scelta = None
        self.completa()
        self.elenco.SetFocus()

    def conferma(self, event=None):
        indice = self.elenco.GetSelection()
        if indice == wx.NOT_FOUND:
            self.avvisa(_("Scegli una lingua dall'elenco."), self.elenco)
            return
        self.scelta = indice
        self.chiudi(wx.ID_OK)
