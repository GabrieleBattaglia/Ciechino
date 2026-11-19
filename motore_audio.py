"""
Il motore audio di Ciechino: un flusso sounddevice a callback tutto suo, che somma le voci al campione
giusto e tiene i due orologi del gioco, il campione e il tempo di gioco.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di apri_flusso_uscita di GBAudio in Chitabry per il
flusso a callback, e del _MixerCondiviso di GBUtils per l'apertura con la conversione di WASAPI, il
timer di Windows e la chiusura che aspetta l'uscita da PortAudio.
Perché un motore suo (D12): Ciechino deve sapere al campione che cosa esce dalle casse quando si
preme spazio, suonare voci i cui parametri cambiano nel tempo e mettere ogni suono al campione
giusto. Il mixer condiviso di GBUtils scrive a blocchi e non lo sa dire; qui c'è un flusso solo, e
quindi un orologio solo, che il callback conta campione per campione.
Le regole dei fili: il filo principale non tocca mai le strutture del callback. Manda comandi in
una coda, collections.deque, che il callback svuota all'inizio di ogni blocco e di nuovo dopo ogni
evento programmato, e li applica al campione indicato. Le sole cose che il filo principale scrive
direttamente sono le ancore del tempo di gioco: tuple sostituite intere sotto un lucchetto, che
il callback legge senza lucchetto, perché uno scambio di riferimento in Python è indivisibile.
Senza scheda audio il motore lavora muto: un filo chiama lo stesso callback al passo del tempo
vero e scarta il suono, così la partita funziona uguale e le prove non aprono mai le casse.
La guardia del flusso: se la scheda smette di chiamare il callback, per esempio quando si stacca una
cuffia e WASAPI invalida il dispositivo, il campione udibile resterebbe fermo per sempre, e con lui
il tempo di gioco e gli eventi programmati: una partita bloccata e muta, senza una parola. Per
questo la finestra chiama controlla a ogni battito: un flusso che PortAudio dice non più attivo, o
il cui callback tace da più di FLUSSO_FERMO_DOPO secondi, si dà per perso, e il motore passa al modo
muto dal campione già scritto, così la partita va avanti senza suono e la finestra lo può dire.
Le notizie del filo dell'audio, voci finite, eventi scattati, buchi, errori e il flusso perso,
arrivano al filo principale con notifiche(), perché il filo dell'audio non scrive mai su disco: ogni
errore porta da dove viene, quante volte è già successo e, la prima volta, la pila delle chiamate,
così il registro degli errori ha ciò che serve per capire il guasto.
"""

import atexit
import collections
import contextlib
import threading
import time
import traceback
from typing import NamedTuple

import numpy as np

from costanti import AUDIO_BLOCCO, AUDIO_LATENZA, FS, RAMPA_DEL_GUADAGNO, SFUMATURA_STOP, SOGLIA_DEL_TETTO, VELOCITA_DEL_TEMPO
from sintesi import FORME_PORTANTI, Oscillatore, panoramica

# Quanti secondi di storia del tempo di gioco si tengono: chi chiede il tempo di gioco di un
# campione passato, per esempio quello dello spazio meno la compensazione del tasto, lo chiede
# di qualche decimo fa, non di dieci secondi.
MEMORIA_DELLE_ANCORE = 10.0
# Le notifiche che nessuno legge non devono crescere per sempre: oltre questo numero le più
# vecchie si perdono. Lo stesso tetto vale per gli errori diversi di cui il motore tiene il conto.
NOTIFICHE_MASSIME = 1000
# Quanti eventi programmati possono scattare in un blocco solo: un evento che ne programma un
# altro già scaduto, all'infinito, non deve bloccare il filo dell'audio.
EVENTI_PER_BLOCCO = 64
# Il filo muto, se resta indietro di più di tanti secondi (un sistema sospeso, un debugger), non
# rincorre il tempo perso con una raffica di blocchi: riparte da adesso.
RITARDO_MASSIMO_MUTO = 0.25
# Quanti secondi chiudi aspetta il filo muto, e quello che lascia un flusso perso, prima di andarsene
# comunque.
ATTESA_DELLA_CHIUSURA = 2.0
# La guardia del flusso: un callback che tace per più di tanti secondi dell'orologio del flusso,
# mentre il filo principale è sveglio e controlla, vuol dire una scheda che non risponde più. Mezzo
# secondo basta perché un singhiozzo del sistema non tolga il suono a una scheda viva, ed è abbastanza
# poco perché chi gioca quasi non si accorga della partita ferma. È un numero del motore, non della
# meccanica di gioco, come gli altri qui sopra: resta qui finché nessuno lo vorrà tarare.
FLUSSO_FERMO_DOPO = 0.5

# Perché l'ultima apri_flusso non ha aperto niente, o ha dovuto ripiegare sul dispositivo
# predefinito: un pezzo per tentativo fallito, separati da punto e virgola; vuoto se il primo
# tentativo è riuscito. La firma di apri_flusso non cambia, perché le prove la sostituiscono: il
# motore legge qui il perché, dopo averla chiamata.
ULTIMO_ERRORE_DI_APERTURA = ""


def _testo_dell_errore(errore):
    """Un'eccezione detta in breve, "Tipo: messaggio", com'è nelle notifiche e in ultimo_errore."""
    return f"{type(errore).__name__}: {errore}"


def _nome_della_funzione(funzione):
    """Il nome qualificato di una funzione, come Bersaglio.parametri_sonori, per il registro; per
    un oggetto chiamabile senza nome, il nome della sua classe."""
    return getattr(funzione, "__qualname__", None) or type(funzione).__qualname__


def _descrivi(oggetto):
    """Due parole su ciò che una notifica riguarda, per il registro: la classe di una voce e la
    funzione dei suoi parametri, la funzione di un evento e il suo tempo di gioco, la classe di
    tutto il resto."""
    if oggetto is None:
        return ""
    if isinstance(oggetto, Evento):
        return f"{_nome_della_funzione(oggetto.funzione)} al tempo di gioco {oggetto.tempo:.3f}"
    if isinstance(oggetto, VoceTono):
        return f"{type(oggetto).__name__} di {_nome_della_funzione(oggetto.parametri)}, dal campione {oggetto.inizio}"
    if isinstance(oggetto, Voce):
        return f"{type(oggetto).__name__} dal campione {oggetto.inizio}"
    return type(oggetto).__name__


class Notifica(NamedTuple):
    """Una notizia dal filo dell'audio per il filo principale, che la riceve con Motore.notifiche().
    campione è il campione del flusso a cui è successa; gli altri campi dipendono da tipo.
    "voce_finita": oggetto la voce, campione dove ha finito.
    "evento": oggetto l'Evento scattato, campione dove è scattato.
    "buco": la scheda è rimasta a secco, cioè status.output_underflow; volte è il conto dei buchi
    dall'avvio, lo stesso di stato()["buchi"].
    "errore": un guasto dentro il filo dell'audio, che non lo ha fermato. origine dice dove: "blocco",
    il blocco intero, uscito muto; "voce", e oggetto è la voce, tolta dal motore; "evento", e oggetto
    è l'Evento la cui funzione ha sollevato; "comando " col nome del comando, come "comando ferma", e
    oggetto è ciò che il comando riguardava, rimasto com'era. testo è l'eccezione, "Tipo: messaggio".
    volte è quante volte lo stesso errore, stessa origine e stesso testo, è successo dall'avvio,
    questa compresa: un guasto che si ripete a ogni blocco arriva 86 volte al secondo, e il registro
    lo scrive una volta sola col suo conto. pila, soltanto la prima volta, è la pila delle chiamate,
    una traceback.StackSummary presa senza leggere il sorgente, che si legge quando la si scrive;
    None le altre volte.
    "flusso_perso": la scheda non risponde più, e da campione il motore lavora muto; testo dice
    perché e su quale scheda.
    riga() la dice in una riga per il registro degli errori."""

    tipo: str
    campione: int
    oggetto: object = None
    testo: str = ""
    origine: str = ""
    volte: int = 1
    pila: object = None

    def riga(self):
        """La notifica in una riga per il registro degli errori, che parla a chi sviluppa; un errore
        alla sua prima volta porta sotto la pila delle chiamate, con le righe del sorgente."""
        if self.tipo == "errore":
            dove = self.origine or "blocco"
            descrizione = _descrivi(self.oggetto)
            if descrizione:
                dove = f"{dove}, {descrizione}"
            riga = f"Errore nel filo dell'audio al campione {self.campione} ({dove}): {self.testo}"
            if self.volte > 1:
                riga += f", {self.volte} volte dall'avvio"
            if self.pila:
                with contextlib.suppress(Exception):
                    riga += "\nPila delle chiamate:\n" + "".join(self.pila.format()).rstrip()
            return riga
        if self.tipo == "buco":
            return f"Buco nell'audio al campione {self.campione}: la scheda è rimasta senza campioni, {self.volte} volte dall'avvio."
        if self.tipo == "flusso_perso":
            return f"Flusso audio perso al campione {self.campione}: {self.testo}. Ciechino continua senza suono."
        descrizione = _descrivi(self.oggetto)
        return f"Notizia del motore audio, {self.tipo}, al campione {self.campione}" + (f", {descrizione}." if descrizione else ".")


class Evento:
    """Un evento programmato con Motore.programma: quando il tempo di gioco raggiunge tempo, il
    callback chiama funzione(campione_esatto). Dopo lo scatto scattato è vero e campione dice a
    quale campione è successo: la partita lo confronta col campione udibile per sapere quando
    l'impatto si è davvero sentito."""

    def __init__(self, tempo, funzione):
        self.tempo = float(tempo)
        self.funzione = funzione
        self.scattato = False
        self.annullato = False
        self.campione = None


class _Rampa:
    """Un valore che va da da ad a in durata campioni, a partire dal campione inizio, in linea retta.
    È la stessa cosa per il guadagno di una voce e per il volume generale, e la si calcola su
    qualunque tratto di campioni senza tenere stato fra un blocco e l'altro."""

    __slots__ = ("a", "da", "durata", "inizio")

    def __init__(self, valore):
        self.da = self.a = float(valore)
        self.inizio = 0
        self.durata = 0

    def valore_a(self, campione):
        if self.durata <= 0 or campione >= self.inizio + self.durata:
            return self.a
        if campione <= self.inizio:
            return self.da
        return self.da + (self.a - self.da) * (campione - self.inizio) / self.durata

    def imposta(self, valore, campione, durata):
        """Da campione in poi il valore va verso valore in durata campioni, partendo da dove è."""
        self.da = self.valore_a(campione)
        self.a = float(valore)
        self.inizio = int(campione)
        self.durata = max(0, int(durata))

    def valori(self, campione, quanti):
        """Il valore dei campioni da campione a campione più quanti: un numero se è fermo, un array
        se in quel tratto la rampa sta ancora andando."""
        if self.durata <= 0 or campione >= self.inizio + self.durata:
            return self.a
        posti = (np.arange(campione, campione + quanti) - self.inizio) / self.durata
        return self.da + (self.a - self.da) * np.clip(posti, 0.0, 1.0)


class Voce:
    """Una voce del motore. inizio è il campione del flusso dove comincia; fine il campione dove
    è finita del tutto, None finché nessuno la ferma; guadagno il guadagno lineare verso cui va,
    con una rampa; finita diventa vero quando ha suonato l'ultimo campione, e finita_al dice dove;
    attacco, in secondi, è una rampa da zero a pieno dal campione d'inizio, che moltiplica il
    guadagno senza toccarlo.
    Le sottoclassi scrivono _produci, il suono di un tratto; rendi ci mette intorno inizio e fine
    al campione, il guadagno, l'attacco e la sfumatura della fermata."""

    def __init__(self, inizio, guadagno=1.0):
        self.inizio = int(inizio)
        self.fine = None
        self.finita = False
        self.finita_al = None
        # La frequenza di campionamento la fissa il motore quando la voce entra.
        self.fs = FS
        self._guadagno = _Rampa(guadagno)
        self._sfuma_da = None
        # Un buffer già pronto ha l'attacco nel suo suono, e resta intatto al campione: niente
        # attacco. Le voci calcolate, che cominciano a guadagno pieno, ne hanno uno loro.
        self.attacco = 0.0

    @property
    def guadagno(self):
        return self._guadagno.a

    def imposta_guadagno(self, valore, campione, campioni_di_rampa=0):
        """Il guadagno va verso valore da campione in poi, in tanti campioni."""
        self._guadagno.imposta(valore, campione, campioni_di_rampa)

    def ferma_a(self, campione, campioni_di_sfumatura=0):
        """La voce si spegne da campione in poi, in linea retta, e finisce del tutto tanti campioni
        dopo. Una voce che a quel campione non era ancora cominciata finisce senza suonare. Una
        seconda fermata conta solo se finisce prima della prima."""
        campione = int(campione)
        if campione <= self.inizio:
            fine, da = self.inizio, None
        else:
            fine, da = campione + max(0, int(campioni_di_sfumatura)), campione
        if self.fine is None or fine < self.fine:
            self.fine = fine
            self._sfuma_da = da

    def _chiudi(self, campione):
        self.finita = True
        self.finita_al = int(campione)

    def _produci(self, campione, quanti, tempi_di_gioco, tempi_audio):
        """Il suono dei campioni da campione a campione più quanti, (k, 2) con k al massimo quanti:
        meno di quanti vuol dire che la voce non ha altro da dire."""
        raise NotImplementedError

    def rendi(self, campione, n, tempi_di_gioco, tempi_audio):
        """Il blocco (n, 2) float32 dei campioni da campione a campione più n, già con inizio e fine
        al campione dentro il blocco, oppure None se in questo blocco la voce tace.
        tempi_di_gioco e tempi_audio sono i due orologi, un valore per campione del blocco."""
        if self.finita:
            return None
        prima = max(0, self.inizio - campione)
        if prima >= n:
            return None
        dopo = n
        if self.fine is not None:
            dopo = min(n, self.fine - campione)
            if dopo <= prima:
                self._chiudi(max(self.fine, campione))
                return None
        quanti = dopo - prima
        da = campione + prima
        pezzo = np.array(self._produci(da, quanti, tempi_di_gioco[prima:dopo], tempi_audio[prima:dopo]), dtype=np.float32, copy=True)
        if pezzo.ndim != 2 or pezzo.shape[1] != 2 or len(pezzo) > quanti:
            raise ValueError(f"la voce ha prodotto un blocco di forma {pezzo.shape} invece di ({quanti}, 2)")
        k = len(pezzo)
        if k and not np.isfinite(pezzo).all():
            raise ValueError("la voce ha prodotto campioni non finiti")
        if k:
            guadagno = self._guadagno.valori(da, k)
            if np.ndim(guadagno):
                pezzo *= np.asarray(guadagno, dtype=np.float32)[:, None]
            elif guadagno != 1.0:
                pezzo *= np.float32(guadagno)
            # L'attacco si conta da inizio, che il motore ha già portato al primo campione del
            # blocco se la voce è arrivata in ritardo: la rampa comincia dove la voce comincia davvero.
            campioni_d_attacco = round(self.attacco * self.fs)
            if campioni_d_attacco > 0 and da < self.inizio + campioni_d_attacco:
                attacco = np.clip((np.arange(da, da + k) - self.inizio) / campioni_d_attacco, 0.0, 1.0)
                pezzo *= attacco.astype(np.float32)[:, None]
            lunghezza = 0 if self._sfuma_da is None or self.fine is None else self.fine - self._sfuma_da
            if lunghezza > 0 and da + k > self._sfuma_da:
                sfuma = np.clip((self.fine - np.arange(da, da + k)) / lunghezza, 0.0, 1.0)
                pezzo *= sfuma.astype(np.float32)[:, None]
        if k < quanti:
            self._chiudi(da + k)
        elif self.fine is not None and self.fine <= campione + n:
            self._chiudi(self.fine)
        if not k:
            return None
        blocco = np.zeros((n, 2), dtype=np.float32)
        blocco[prima:prima + k] = pezzo
        return blocco


class VoceBuffer(Voce):
    """Un suono già pronto: un buffer (N, 2), o mono, che si mette al centro a potenza costante."""

    def __init__(self, buffer, inizio, guadagno=1.0):
        super().__init__(inizio, guadagno)
        dati = np.asarray(buffer, dtype=np.float32)
        if dati.ndim == 2 and dati.shape[1] == 1:
            dati = dati[:, 0]
        if dati.ndim == 1:
            dati = panoramica(dati, 0.0)
        if dati.ndim != 2 or dati.shape[1] != 2:
            raise ValueError(f"buffer di forma {dati.shape}: serve (N, 2) o mono")
        self.buffer = dati

    def _produci(self, campione, quanti, tempi_di_gioco, tempi_audio):
        da = campione - self.inizio
        return self.buffer[da:da + quanti]


class VoceTono(Voce):
    """Una voce che si calcola blocco per blocco: parametri(tempi_di_gioco, tempi_audio)
    restituisce (frequenze, pan, guadagni lineari) di ogni campione, o numeri validi per tutti, e
    la voce li suona con un Oscillatore di forma e la panoramica. È la voce dei bersagli e del
    proiettile, che si muovono col tempo di gioco e vibrano col tempo dell'audio."""

    def __init__(self, parametri, forma, inizio, guadagno=1.0):
        super().__init__(inizio, guadagno)
        if forma not in FORME_PORTANTI:
            raise ValueError(f"forma sconosciuta: {forma!r}")
        self.parametri = parametri
        self.forma = forma
        # L'oscillatore nasce al primo blocco, con la frequenza di campionamento del motore.
        self._oscillatore = None
        # La voce nasce a guadagno pieno: senza un attacco l'inviluppo passerebbe dal silenzio al
        # pieno in un campione, un piccolo click a banda larga che si sente quando un bersaglio nasce
        # da solo. L'attacco dura quanto la sfumatura della fermata: la voce comincia come finisce.
        self.attacco = SFUMATURA_STOP

    def _produci(self, campione, quanti, tempi_di_gioco, tempi_audio):
        if self._oscillatore is None:
            self._oscillatore = Oscillatore(self.forma, self.fs)
        frequenze, pan, guadagni = self.parametri(tempi_di_gioco, tempi_audio)
        frequenze = np.broadcast_to(np.asarray(frequenze, dtype=np.float64), (quanti,))
        guadagni = np.broadcast_to(np.asarray(guadagni, dtype=np.float64), (quanti,))
        return panoramica(self._oscillatore.rendi(frequenze) * guadagni, pan)


def tetto(campioni, soglia=SOGLIA_DEL_TETTO):
    """Il tetto morbido della somma, in luogo: sotto soglia niente cambia, sopra il valore piega con
    una tangente iperbolica che parte con la stessa pendenza e non arriva mai oltre uno. Così due
    suoni forti insieme si schiacciano un poco invece di tagliarsi con un gracchio."""
    ampiezze = np.abs(campioni)
    oltre = ampiezze > soglia
    if oltre.any():
        spazio = 1.0 - soglia
        piegati = soglia + spazio * np.tanh((ampiezze[oltre] - soglia) / spazio)
        campioni[oltre] = np.sign(campioni[oltre]) * piegati
    return campioni


def _tempo_scalare(ancore, campione, fs):
    """Il tempo di gioco a un campione, dalle ancore (campione, tempo, scorre) in ordine."""
    scelta = ancore[0]
    for ancora in reversed(ancore):
        if ancora[0] <= campione:
            scelta = ancora
            break
    inizio, tempo, scorre = scelta
    if not scorre:
        return tempo
    return tempo + (campione - inizio) / fs * VELOCITA_DEL_TEMPO


def apri_flusso(callback, fs, blocco):
    """Apre e avvia il flusso di uscita a callback, stereo float32, blocco campioni per blocco, a
    latenza AUDIO_LATENZA, sul dispositivo che scegli_dispositivo_audio di GBUtils trova per il
    modo callback, con la conversione automatica di WASAPI se l'interfaccia è quella; se non si
    apre ripiega sul dispositivo predefinito. Restituisce il flusso, o None se non si apre niente.
    Il perché di ogni tentativo fallito resta in ULTIMO_ERRORE_DI_APERTURA, per il registro: senza,
    chi gioca muto saprebbe soltanto che la scheda non si apre.
    Le prove la sostituiscono, e il motore la cerca qui al momento di avviarsi."""
    global ULTIMO_ERRORE_DI_APERTURA
    ULTIMO_ERRORE_DI_APERTURA = ""
    try:
        import sounddevice as sd
    except Exception as errore:  # noqa: BLE001 - senza sounddevice si gioca muti, non si cade
        ULTIMO_ERRORE_DI_APERTURA = f"sounddevice non si importa, {_testo_dell_errore(errore)}"
        return None
    falliti = []
    tentativi = []
    try:
        from GBUtils import scegli_dispositivo_audio

        dispositivo, interfaccia = scegli_dispositivo_audio(modo="callback")
        if dispositivo is not None:
            tentativi.append((dispositivo, interfaccia))
    except Exception as errore:  # noqa: BLE001 - senza la scelta si va col dispositivo predefinito
        falliti.append(f"scelta del dispositivo, {_testo_dell_errore(errore)}")
    tentativi.append((None, None))
    for dispositivo, interfaccia in tentativi:
        flusso = None
        try:
            # Senza la conversione WASAPI accetta soltanto la frequenza impostata in Windows: un
            # dispositivo tenuto a 48.000 non si aprirebbe a 44.100.
            extra = sd.WasapiSettings(auto_convert=True) if interfaccia and "WASAPI" in interfaccia else None
            flusso = sd.OutputStream(samplerate=fs, blocksize=blocco, channels=2, dtype="float32", latency=AUDIO_LATENZA,
                                     device=dispositivo, extra_settings=extra, callback=callback)
            flusso.start()
        except Exception as errore:  # noqa: BLE001 - il dispositivo audio fallisce in molti modi, e si prova il seguente
            if dispositivo is None:
                quale = "dispositivo predefinito"
            else:
                quale = f"dispositivo {dispositivo}" + (f" di {interfaccia}" if interfaccia else "")
            falliti.append(f"{quale}, {_testo_dell_errore(errore)}")
            if flusso is not None:
                # Un flusso che non si è avviato si chiude come può.
                with contextlib.suppress(Exception):
                    flusso.close()
            continue
        ULTIMO_ERRORE_DI_APERTURA = "; ".join(falliti)
        return flusso
    ULTIMO_ERRORE_DI_APERTURA = "; ".join(falliti)
    return None


def _lascia_il_flusso(flusso):
    """Chiude un flusso perso, nel filo che controlla gli fa partire accanto: prima abort, che non
    aspetta di suonare ciò che resta, poi close. Un guasto non conta, perché la scheda non risponde
    comunque; il filo a parte serve perché un driver bloccato non blocchi la finestra."""
    for azione in ("abort", "close"):
        with contextlib.suppress(Exception):
            getattr(flusso, azione)()


def _descrivi_flusso(flusso):
    """Il nome dell'interfaccia e del dispositivo di un flusso, per stato(); None dove non si sa."""
    interfaccia = getattr(flusso, "interfaccia", None)
    dispositivo = getattr(flusso, "device", None)
    if interfaccia is None and isinstance(dispositivo, int):
        try:
            import sounddevice as sd

            informazioni = sd.query_devices(dispositivo)
            interfaccia = sd.query_hostapis(informazioni["hostapi"])["name"]
            dispositivo = informazioni["name"]
        except Exception:  # noqa: BLE001, S110 - è solo una descrizione, se manca si lascia il numero
            pass
    return interfaccia, dispositivo


class _TempiMuti:
    """Il time_info del filo muto: l'orologio è perf_counter, e il DAC è l'istante previsto per
    il blocco."""

    __slots__ = ("currentTime", "inputBufferAdcTime", "outputBufferDacTime")

    def __init__(self, istante):
        self.currentTime = istante
        self.inputBufferAdcTime = 0.0
        self.outputBufferDacTime = istante


class _StatoMuto:
    """Lo status del filo muto: nessun buco, perché non c'è una scheda da lasciare a secco."""

    output_underflow = False
    output_overflow = False
    priming_output = False


class Motore:
    """Il motore audio: un flusso solo, le voci, il tempo di gioco, gli eventi programmati.
    apri è la funzione che apre il flusso, apri(callback, fs, blocco); None vuol dire
    motore_audio.apri_flusso, cercata al momento di avvia così le prove la sostituiscono.
    Gli attributi che si leggono da fuori: fs e blocco; flusso, None se il motore è muto; muto, vero
    se lavora senza scheda, dall'avvio o da quando il flusso si è perso; flusso_perso, vero da quando
    controlla ha dato il flusso per perso; avviato e chiuso; ultimo_errore, il perché dell'ultimo
    guasto visto dal filo principale, None se non ce ne sono stati: dopo avvia dice perché la scheda
    non si è aperta, o perché il timer di Windows non si è alzato, più pezzi separati da punto e
    virgola; dopo controlla, perché il flusso si è perso. Gli errori del filo dell'audio arrivano
    invece con notifiche()."""

    def __init__(self, fs=FS, blocco=AUDIO_BLOCCO, apri=None):
        self.fs = int(fs)
        self.blocco = int(blocco)
        self._apri = apri
        self.flusso = None
        self.muto = False
        self.flusso_perso = False
        self.avviato = False
        self.chiuso = False
        self.ultimo_errore = None
        self._comandi = collections.deque()
        self._notifiche = collections.deque(maxlen=NOTIFICHE_MASSIME)
        # Il conto di ogni errore del filo dell'audio, per origine e testo: lo tocca soltanto il callback.
        self._errori_visti = {}
        # Il callback della scheda e quello del filo muto non girano mai insieme: dopo un flusso perso,
        # un callback della scheda ancora in corso finisce prima che il filo muto cominci.
        self._lucchetto_del_callback = threading.Lock()
        # La guardia di controlla: i blocchi che la scheda ha chiesto, quanti ne aveva visti l'ultimo
        # controllo, l'ora di quel controllo e l'ora da cui il callback tace, sull'orologio del flusso.
        self._blocchi_della_scheda = 0
        self._blocchi_visti = None
        self._ultimo_controllo = None
        self._vivo_dal = None
        self._filo_della_chiusura = None
        # Le strutture del filo dell'audio: le tocca soltanto il callback.
        self._voci = []
        self._eventi = []
        self._uscita = _Rampa(1.0)
        # Il prossimo campione che il callback preparerà.
        self._campione = 0
        # Il primo campione dell'ultimo blocco preparato e l'istante in cui uscirà dalle casse.
        self._riferimento = None
        # Il tempo di gioco: all'inizio vale zero e scorre, come nella fase d'ascolto.
        self._ancore = ((0, 0.0, True),)
        self._lucchetto_ancore = threading.Lock()
        self._chiave_ancore = None
        self._vettori_ancore = None
        self._volume = 1.0
        self._silenziato = False
        self._buchi = 0
        self._filo_muto = None
        self._ferma_muto = threading.Event()
        self._periodo = None
        self._timer_alzato = False
        self._interfaccia = None
        self._dispositivo = None
        self._latenza = None

    # Avvio e chiusura.

    def avvia(self):
        """Apre il flusso e comincia a contare i campioni. Vero se suona davvero; falso se lavora
        muto, senza scheda, oppure se il motore è già stato chiuso. Se lavora muto, ultimo_errore
        dice perché, "apertura: " col perché di ogni tentativo, per il registro."""
        if self.chiuso:
            return False
        if self.avviato:
            return not self.muto
        self.avviato = True
        problemi = []
        try:
            from GBUtils import _alza_timer

            self._periodo = _alza_timer()
            self._timer_alzato = True
        except Exception as errore:  # noqa: BLE001 - un timer più lasco si sopporta, un motore fermo no
            problemi.append(f"timer: {_testo_dell_errore(errore)}")
        apri = self._apri if self._apri is not None else apri_flusso
        flusso = None
        perche = None
        try:
            flusso = apri(self._callback_della_scheda, self.fs, self.blocco)
        except Exception as errore:  # noqa: BLE001 - senza flusso si gioca muti
            perche = _testo_dell_errore(errore)
            flusso = None
        if flusso is None:
            if perche is None:
                # Il perché lo sa soltanto apri_flusso, che lo lascia nel modulo; un'altra funzione
                # d'apertura, come quelle delle prove, non lo dice.
                perche = (ULTIMO_ERRORE_DI_APERTURA if apri is apri_flusso else "") or "nessuna scheda audio si è aperta"
            problemi.append(f"apertura: {perche}")
        if problemi:
            self.ultimo_errore = "; ".join(problemi)
        atexit.register(self.chiudi)
        if flusso is None:
            self._avvia_muto()
            return False
        self.flusso = flusso
        try:
            self._latenza = float(flusso.latency)
        except (AttributeError, TypeError, ValueError):
            self._latenza = None
        self._interfaccia, self._dispositivo = _descrivi_flusso(flusso)
        return True

    def _avvia_muto(self):
        """Da qui il motore lavora muto: un filo chiama il callback al passo del tempo vero, dal
        campione già scritto, e butta via il suono. Serve all'avvio senza scheda e dopo un flusso perso."""
        self.muto = True
        self._ferma_muto.clear()
        self._filo_muto = threading.Thread(target=self._gira_muto, name="Ciechino-motore-muto", daemon=True)
        self._filo_muto.start()

    def controlla(self):
        """La guardia del flusso, che la finestra chiama dal filo principale a ogni battito.
        Restituisce vero se in questa chiamata il flusso si è perso e il motore è passato al modo
        muto; falso se il flusso è vivo, e sempre per un motore muto, chiuso o mai avviato.
        Il flusso si dà per perso in due casi: PortAudio dice che non è più attivo, come succede con
        WASAPI quando si stacca la cuffia; oppure il callback non chiede blocchi da più di
        FLUSSO_FERMO_DOPO secondi dell'orologio del flusso, contati da un controllo all'altro. Se fra
        due controlli è passato più di tanto, fermo era il filo principale, per un sistema sospeso o
        un debugger, e il callback ha diritto allo stesso tempo da adesso: un risveglio non toglie
        il suono a una scheda viva.
        Col flusso perso il motore lo lascia, e lo chiude in un filo a parte, perché un driver
        bloccato non blocchi la finestra; scrive il perché in ultimo_errore, manda la notifica
        "flusso_perso" e riparte muto dal campione già scritto: le voci, il tempo di gioco e gli
        eventi programmati vanno avanti senza suono, e la partita non resta bloccata. stato()
        risponde allora aperto falso e perso vero. La scheda non si riapre da sola: per riavere il
        suono si riapre Ciechino."""
        if self.chiuso or self.muto or self.flusso is None:
            return False
        motivo = self._motivo_della_perdita()
        if motivo is None:
            return False
        self._perdi_il_flusso(motivo)
        return True

    def _motivo_della_perdita(self):
        """Perché il flusso è da dare per perso, o None se è vivo; aggiorna la guardia."""
        try:
            attivo = bool(self.flusso.active)
        except AttributeError:
            # Un flusso che non lo sa dire, come quelli delle prove, si giudica soltanto dal callback.
            attivo = True
        except Exception as errore:  # noqa: BLE001 - uno stato che non si legge è di un flusso guasto
            return f"lo stato del flusso non si legge, {_testo_dell_errore(errore)}"
        if not attivo:
            return "PortAudio dice che il flusso non è più attivo"
        # L'orologio è quello del flusso, letto sempre dal filo principale: il conto dei blocchi dice
        # se il callback è vivo, l'orologio quanto tempo è passato, e due orologi diversi non si
        # confrontano mai. Col flusso finto delle prove è l'orologio finto, che si sposta a mano.
        adesso = self._orologio()
        if adesso is None:
            return None
        precedente, self._ultimo_controllo = self._ultimo_controllo, adesso
        blocchi = self._blocchi_della_scheda
        if blocchi != self._blocchi_visti or precedente is None or not 0.0 <= adesso - precedente <= FLUSSO_FERMO_DOPO:
            self._blocchi_visti = blocchi
            self._vivo_dal = adesso
            return None
        silenzio = adesso - self._vivo_dal
        if silenzio > FLUSSO_FERMO_DOPO:
            return f"il callback non arriva da {silenzio:.2f} secondi".replace(".", ",")
        return None

    def _perdi_il_flusso(self, motivo):
        """Lascia il flusso perso e passa al modo muto dal campione già scritto."""
        flusso = self.flusso
        # Da qui il callback della scheda, se si rifacesse vivo, scrive silenzio e non tocca niente.
        self.flusso_perso = True
        self.flusso = None
        campione = self._campione
        scheda = ", ".join(str(parte) for parte in (self._dispositivo, self._interfaccia) if parte)
        testo = f"{motivo}, sulla scheda {scheda}" if scheda else motivo
        self.ultimo_errore = f"flusso perso: {testo}"
        # Il campione udibile riparte da quello scritto, sull'orologio del filo muto: senza, fino al
        # primo blocco del filo muto lo si calcolerebbe dal tempo del DAC della scheda persa.
        self._riferimento = (campione, time.perf_counter())
        self._notifica("flusso_perso", campione, None, testo)
        self._avvia_muto()
        self._filo_della_chiusura = threading.Thread(target=_lascia_il_flusso, args=(flusso,), name="Ciechino-flusso-perso", daemon=True)
        self._filo_della_chiusura.start()

    def chiudi(self):
        """Ferma tutto e lascia libera la scheda, aspettando che il callback sia uscito da
        PortAudio: un filo sorpreso là dentro mentre l'interprete smonta i moduli fa morire il
        processo con 0xC0000374, corruzione dell'heap, come insegna il mixer di GBUtils. Si può
        chiamare più volte; la chiama anche atexit. Un motore chiuso resta chiuso."""
        if self.chiuso:
            return
        self.chiuso = True
        # Tolta da atexit, la chiusura non tiene vivo il motore fino alla fine del processo; se non
        # era registrata non c'è niente da togliere.
        with contextlib.suppress(Exception):
            atexit.unregister(self.chiudi)
        self._ferma_muto.set()
        # Il filo muto, e quello che chiude un flusso perso: nessuno dei due deve restare dentro
        # PortAudio o dentro il callback mentre l'interprete smonta i moduli.
        for filo in (self._filo_muto, self._filo_della_chiusura):
            if filo is not None and filo.is_alive() and filo is not threading.current_thread():
                filo.join(ATTESA_DELLA_CHIUSURA)
        flusso = self.flusso
        if flusso is not None:
            for azione in ("stop", "close"):
                try:
                    getattr(flusso, azione)()
                except Exception as errore:  # noqa: BLE001 - alla chiusura si va avanti comunque
                    self.ultimo_errore = f"chiusura: {type(errore).__name__}: {errore}"
        if self._timer_alzato:
            self._timer_alzato = False
            try:
                from GBUtils import _abbassa_timer

                _abbassa_timer(self._periodo)
            except Exception as errore:  # noqa: BLE001 - il timer si rende come si può
                self.ultimo_errore = f"timer: {type(errore).__name__}: {errore}"

    # I campioni.

    @property
    def campione_scritto(self):
        """Il prossimo campione che il callback preparerà."""
        return self._campione

    def prossimo_campione(self, blocchi=2):
        """Un campione futuro sicuro da cui far partire una voce: campione_scritto più tanti
        blocchi, così il comando arriva al callback prima che il blocco si prepari."""
        return self._campione + int(blocchi) * self.blocco

    def _orologio(self):
        """L'orologio del flusso, lo stesso di outputBufferDacTime; da muti perf_counter."""
        if self.muto:
            return time.perf_counter()
        flusso = self.flusso
        if flusso is None:
            return None
        try:
            return float(flusso.time)
        except Exception:  # noqa: BLE001 - un flusso guasto non ha un'ora, e si risponde senza
            return None

    def campione_udibile(self):
        """Il campione che esce dalle casse adesso, con la parte frazionaria: il primo campione
        dell'ultimo blocco, più il tempo passato da quando quel blocco doveva arrivare al DAC. Sta
        fra zero e campione_scritto, perché non si può sentire ciò che non è ancora pronto."""
        riferimento = self._riferimento
        if riferimento is None:
            return 0.0
        campione, dac = riferimento
        adesso = self._orologio()
        if adesso is None:
            return float(campione)
        udibile = campione + (adesso - dac) * self.fs
        return float(min(max(udibile, 0.0), self._campione))

    # Le voci.

    def aggiungi(self, voce):
        """Manda una voce al callback, che la suonerà dal suo campione d'inizio; se quel campione
        è già passato quando il comando arriva, la voce parte dal primo campione del blocco."""
        self._comandi.append(("aggiungi", voce))
        return voce

    def suona(self, buffer, inizio=None, guadagno=1.0):
        """Crea e aggiunge una VoceBuffer, e la restituisce; inizio None vuol dire
        prossimo_campione()."""
        voce = VoceBuffer(buffer, self.prossimo_campione() if inizio is None else inizio, guadagno)
        return self.aggiungi(voce)

    def ferma(self, voce, campione=None, sfumatura=SFUMATURA_STOP):
        """Ferma una voce: da campione si spegne in sfumatura secondi, poi finisce. campione None
        vuol dire subito, al primo campione del blocco in cui il comando arriva."""
        self._comandi.append(("ferma", voce, campione, sfumatura))

    def ferma_tutte(self, sfumatura=SFUMATURA_STOP):
        """Ferma subito tutte le voci, ognuna con la sua sfumatura: serve a chi chiude una partita."""
        self._comandi.append(("ferma_tutte", sfumatura))

    def guadagno_voce(self, voce, valore, rampa=RAMPA_DEL_GUADAGNO):
        """Il guadagno lineare di una voce va verso valore in rampa secondi, dal blocco in cui il
        comando arriva."""
        self._comandi.append(("guadagno", voce, valore, rampa))

    # Il tempo di gioco.

    def tempo_di_gioco(self, campione=None):
        """Il tempo di gioco, in secondi, a un campione; predefinito il campione udibile. Risponde
        subito anche per un campione futuro, con le ancore già chieste."""
        if campione is None:
            campione = self.campione_udibile()
        return _tempo_scalare(self._ancore, campione, self.fs)

    def ferma_tempo(self, campione=None):
        """Il tempo di gioco si ferma da campione in poi, predefinito prossimo_campione().
        Restituisce il campione usato."""
        return self._ancora(campione, False)

    def fai_scorrere_tempo(self, campione=None):
        """Il tempo di gioco riprende a scorrere da campione in poi, predefinito
        prossimo_campione(), dallo stesso valore a cui si era fermato. Restituisce il campione usato."""
        return self._ancora(campione, True)

    @property
    def tempo_scorre(self):
        """Vero se il tempo di gioco, secondo l'ultima ancora chiesta, scorre."""
        return self._ancore[-1][2]

    def _ancora(self, campione, scorre):
        """Mette un'ancora del tempo di gioco. Le ancore stanno in ordine di campione: una nuova
        che cade prima di altre già chieste le rende vane, perché vale l'ultima parola. Un campione
        già preparato dal callback non si può più cambiare, e diventa campione_scritto."""
        with self._lucchetto_ancore:
            campione = self.prossimo_campione() if campione is None else int(campione)
            campione = max(campione, self._campione)
            tenute = [ancora for ancora in self._ancore if ancora[0] < campione] or [self._ancore[0]]
            if tenute[-1][2] != scorre:
                tenute.append((campione, _tempo_scalare(tenute, campione, self.fs), scorre))
            orizzonte = self._campione - round(MEMORIA_DELLE_ANCORE * self.fs)
            primo = 0
            for indice, ancora in enumerate(tenute):
                if ancora[0] <= orizzonte:
                    primo = indice
            self._ancore = tuple(tenute[primo:])
        return campione

    def _tempi_del_blocco(self, campione, n):
        """Il tempo di gioco di ogni campione del blocco, dalle ancore lette adesso."""
        ancore = self._ancore
        if ancore is not self._chiave_ancore:
            self._vettori_ancore = (np.array([a[0] for a in ancore], dtype=np.int64), np.array([a[1] for a in ancore], dtype=np.float64),
                                    np.array([1.0 if a[2] else 0.0 for a in ancore]))
            self._chiave_ancore = ancore
        inizi, tempi, scorre = self._vettori_ancore
        campioni = np.arange(campione, campione + n, dtype=np.int64)
        quale = np.maximum(np.searchsorted(inizi, campioni, side="right") - 1, 0)
        return tempi[quale] + (campioni - inizi[quale]) / self.fs * VELOCITA_DEL_TEMPO * scorre[quale]

    # Gli eventi.

    def programma(self, tempo_di_gioco, funzione):
        """Quando, nel preparare un blocco, il tempo di gioco raggiunge tempo_di_gioco, il callback
        chiama funzione(campione_esatto) nel filo dell'audio, e i comandi che la funzione manda
        valgono già in quel blocco. La funzione deve essere rapida: niente sintesi, solo voci già
        pronte. Un tempo già passato scatta al blocco seguente. Restituisce l'Evento."""
        evento = Evento(tempo_di_gioco, funzione)
        self._comandi.append(("programma", evento))
        return evento

    def annulla(self, evento):
        """Toglie un evento programmato che non è ancora scattato."""
        evento.annullato = True
        self._comandi.append(("annulla", evento))

    # Il volume generale.

    @property
    def volume(self):
        """Il volume generale, da 0 a 1, lineare; i cambi vanno con una rampa di RAMPA_DEL_GUADAGNO."""
        return self._volume

    @volume.setter
    def volume(self, valore):
        self._volume = min(1.0, max(0.0, float(valore)))
        self._comandi.append(("uscita", self._volume, self._silenziato))

    @property
    def silenziato(self):
        return self._silenziato

    def silenzia(self, vero=True):
        """Zittisce l'uscita con una rampa, per la pausa, o la riapre; il tempo e le voci vanno avanti
        come prima, e a fermare il tempo ci pensa chi mette in pausa."""
        self._silenziato = bool(vero)
        self._comandi.append(("uscita", self._volume, self._silenziato))

    # Le notizie.

    def notifiche(self):
        """Svuota e restituisce le notifiche del filo dell'audio, in ordine: voci finite, eventi
        scattati, buchi, errori e il flusso perso, con i campi che Notifica descrive. La finestra le
        legge a ogni battito e manda al registro gli errori, i buchi e il flusso perso, con
        Notifica.riga(); oltre NOTIFICHE_MASSIME non lette, le più vecchie si perdono."""
        raccolte = []
        while True:
            try:
                raccolte.append(self._notifiche.popleft())
            except IndexError:
                return raccolte

    def stato(self):
        """Come sta il motore, per chi lo guarda e per il registro: aperto, vero se suona su un
        flusso; muto; perso, vero se il flusso si è perso durante il gioco; interfaccia, dispositivo e
        latenza della scheda, che dopo un flusso perso dicono quella persa; blocco; buchi, il conto
        dei buchi dall'avvio; voci, quelle che il callback sta suonando; errore, ultimo_errore."""
        return {
            "aperto": self.flusso is not None and not self.chiuso,
            "muto": self.muto,
            "perso": self.flusso_perso,
            "interfaccia": self._interfaccia,
            "dispositivo": self._dispositivo,
            "latenza": self._latenza,
            "blocco": self.blocco,
            "buchi": self._buchi,
            "voci": len(self._voci),
            "errore": self.ultimo_errore,
        }

    def _notifica(self, tipo, campione, oggetto=None, testo="", origine="", volte=1, pila=None):
        self._notifiche.append(Notifica(tipo, int(campione), oggetto, testo, origine, int(volte), pila))

    def _notifica_errore(self, origine, campione, oggetto, errore):
        """La notifica di un errore del filo dell'audio, col conto delle volte. La pila si prende
        soltanto la prima volta, e senza leggere il sorgente, che costerebbe una lettura dal disco
        dentro il callback: le righe le legge il filo principale quando scrive il registro."""
        testo = _testo_dell_errore(errore)
        firma = (origine, testo)
        volte = self._errori_visti.get(firma, 0) + 1
        if volte > 1 or len(self._errori_visti) < NOTIFICHE_MASSIME:
            self._errori_visti[firma] = volte
        pila = None
        if volte == 1:
            with contextlib.suppress(Exception):
                pila = traceback.StackSummary.extract(traceback.walk_tb(errore.__traceback__), lookup_lines=False)
        self._notifica("errore", campione, oggetto, testo, origine, volte, pila)

    # Il filo dell'audio.

    def _callback_della_scheda(self, outdata, frames, time_info, status):
        """Il callback che riceve il flusso della scheda: conta il blocco per la guardia di
        controlla e passa al callback comune, sotto il lucchetto che lo divide dal filo muto. Dopo
        che controlla ha dato il flusso per perso, un callback che si rifacesse vivo scrive silenzio
        e non tocca niente: il conto dei campioni e le voci sono ormai del filo muto."""
        self._blocchi_della_scheda += 1
        with self._lucchetto_del_callback:
            if self.flusso_perso:
                with contextlib.suppress(Exception):
                    outdata.fill(0)
                return
            self._callback(outdata, frames, time_info, status)

    def _callback(self, outdata, frames, time_info, status):
        """Il callback comune della scheda e del filo muto, con la firma di sounddevice. Non solleva
        mai: un guasto diventa silenzio e una notifica, e il conto dei campioni va avanti comunque."""
        inizio = self._campione
        try:
            self._prepara(outdata, int(frames), time_info, status, inizio)
        except Exception as errore:  # noqa: BLE001 - il filo dell'audio non deve morire
            # Se non si scrive nemmeno lo zero, resta ciò che c'è.
            with contextlib.suppress(Exception):
                outdata.fill(0)
            self._notifica_errore("blocco", inizio, None, errore)
        finally:
            self._campione = inizio + int(frames)

    def _prepara(self, outdata, frames, time_info, status, inizio):
        dac = float(getattr(time_info, "outputBufferDacTime", 0.0) or 0.0)
        if not dac:
            # Certi driver lasciano il tempo del DAC a zero: si ripiega sull'ora del callback più
            # la latenza dichiarata dal flusso.
            dac = float(getattr(time_info, "currentTime", 0.0) or 0.0) + (self._latenza or 0.0)
        self._riferimento = (inizio, dac)
        if status is not None and getattr(status, "output_underflow", False):
            self._buchi += 1
            self._notifica("buco", inizio, volte=self._buchi)
        if self.chiuso:
            outdata.fill(0)
            return
        self._esegui_comandi(inizio)
        tempi_di_gioco = self._scatta_eventi(inizio, frames)
        tempi_audio = np.arange(inizio, inizio + frames, dtype=np.float64) / self.fs
        somma = np.zeros((frames, 2), dtype=np.float32)
        restano = []
        for voce in self._voci:
            try:
                pezzo = voce.rendi(inizio, frames, tempi_di_gioco, tempi_audio)
            except Exception as errore:  # noqa: BLE001 - una voce guasta esce, le altre suonano
                voce.finita = True
                voce.finita_al = inizio
                self._notifica_errore("voce", inizio, voce, errore)
                continue
            if pezzo is not None:
                somma += pezzo
            if voce.finita:
                self._notifica("voce_finita", voce.finita_al if voce.finita_al is not None else inizio, voce)
            else:
                restano.append(voce)
        self._voci = restano
        guadagno = self._uscita.valori(inizio, frames)
        if np.ndim(guadagno):
            somma *= np.asarray(guadagno, dtype=np.float32)[:, None]
        elif guadagno != 1.0:
            somma *= np.float32(guadagno)
        tetto(somma)
        if outdata.shape[1] == 2:
            outdata[:] = somma
        else:
            outdata[:] = somma.mean(axis=1, keepdims=True)

    def _esegui_comandi(self, inizio):
        """Applica i comandi arrivati, nell'ordine, al blocco che comincia a inizio."""
        while True:
            try:
                comando = self._comandi.popleft()
            except IndexError:
                return
            try:
                self._esegui(comando, inizio)
            except Exception as errore:  # noqa: BLE001 - un comando sbagliato non ferma gli altri
                self._notifica_errore(f"comando {comando[0]}", inizio, comando[1] if len(comando) > 1 else None, errore)

    def _esegui(self, comando, inizio):
        nome = comando[0]
        if nome == "aggiungi":
            voce = comando[1]
            voce.fs = self.fs
            voce.inizio = max(int(voce.inizio), inizio)
            if voce not in self._voci:
                self._voci.append(voce)
        elif nome == "ferma":
            _nome, voce, campione, sfumatura = comando
            campione = inizio if campione is None else max(int(campione), inizio)
            voce.ferma_a(campione, round(sfumatura * self.fs))
        elif nome == "ferma_tutte":
            for voce in self._voci:
                voce.ferma_a(inizio, round(comando[1] * self.fs))
        elif nome == "guadagno":
            _nome, voce, valore, rampa = comando
            voce.imposta_guadagno(valore, inizio, round(rampa * self.fs))
        elif nome == "programma":
            if not comando[1].annullato:
                self._eventi.append(comando[1])
        elif nome == "annulla":
            self._eventi = [evento for evento in self._eventi if evento is not comando[1]]
        elif nome == "uscita":
            _nome, volume, silenziato = comando
            self._uscita.imposta(0.0 if silenziato else volume, inizio, round(RAMPA_DEL_GUADAGNO * self.fs))
        else:
            raise ValueError(f"comando sconosciuto: {nome!r}")

    def _scatta_eventi(self, inizio, frames):
        """Fa scattare, in ordine di campione, gli eventi il cui tempo di gioco cade in questo
        blocco, e restituisce il tempo di gioco di ogni campione dopo che gli eventi hanno detto la
        loro: una funzione può aver fermato o fatto ripartire il tempo."""
        tempi = self._tempi_del_blocco(inizio, frames)
        for _giro in range(EVENTI_PER_BLOCCO):
            # annulla mette subito annullato a vero, ma il suo comando arriva soltanto al blocco
            # seguente: il filo principale può annullare un evento anche dopo i comandi d'inizio
            # blocco, e un evento annullato non deve scattare.
            self._eventi = [evento for evento in self._eventi if not evento.annullato]
            if not self._eventi or tempi[-1] < min(evento.tempo for evento in self._eventi):
                break
            scelto, posto = None, frames
            for evento in self._eventi:
                if tempi[-1] >= evento.tempo:
                    dove = int(np.searchsorted(tempi, evento.tempo, side="left"))
                    if dove < posto:
                        scelto, posto = evento, dove
            if scelto is None:
                break
            self._eventi = [evento for evento in self._eventi if evento is not scelto]
            # L'ultimo controllo, il più vicino allo scatto: senza un lucchetto resta una finestra di
            # pochi passi, che non si può chiudere del tutto.
            if scelto.annullato:
                continue
            campione = inizio + posto
            scelto.scattato = True
            scelto.campione = campione
            try:
                scelto.funzione(campione)
            except Exception as errore:  # noqa: BLE001 - un evento guasto non ferma il filo dell'audio
                self._notifica_errore("evento", campione, scelto, errore)
            self._notifica("evento", campione, scelto)
            self._esegui_comandi(inizio)
            tempi = self._tempi_del_blocco(inizio, frames)
        return tempi

    def _gira_muto(self):
        """Il filo del motore muto: chiama il callback al passo del tempo vero, con perf_counter
        come orologio del DAC, e butta via il suono. Prende il lucchetto del callback, così dopo un
        flusso perso non gira mai insieme a un callback della scheda rimasto a metà."""
        passo = self.blocco / self.fs
        uscita = np.zeros((self.blocco, 2), dtype=np.float32)
        stato = _StatoMuto()
        prossimo = time.perf_counter()
        while not self._ferma_muto.is_set():
            adesso = time.perf_counter()
            if adesso < prossimo:
                self._ferma_muto.wait(prossimo - adesso)
                continue
            if adesso - prossimo > RITARDO_MASSIMO_MUTO:
                prossimo = adesso
            with self._lucchetto_del_callback:
                self._callback(uscita, self.blocco, _TempiMuti(prossimo), stato)
            prossimo += passo
