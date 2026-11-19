"""
Gli aiuti audio delle prove di Ciechino: un flusso finto che fa avanzare il motore audio a mano,
blocco per blocco, con un orologio finto; le schede audio finte di GBUtils; un sounddevice finto
per provare apri_flusso vera.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0; con D20, lo stesso giorno, le schede finte, il sounddevice finto e
la scelta della scheda in apri_finto.
Le prove non aprono mai la scheda audio: il conftest fa rispondere None a motore_audio.apri_flusso,
e il motore lavora muto con un filo che segue il tempo vero. Chi vuole decidere da sé quando passa
il tempo, campione per campione, dà al motore questo flusso finto, che chiama il callback soltanto
quando la prova glielo chiede. Serve alle prove del motore e a quelle della partita, dei suoni e
della mira, che devono sapere che cosa esce, e quando.
Come si usa, nel caso più comune:
    from aiuti_audio import motore_finto
    motore, flusso = motore_finto()
    voce = motore.suona(buffer, inizio=700)
    uscita = flusso.avanza(4)           # quattro blocchi: un array (4 * blocco, 2) float32
    motore.campione_udibile()           # secondo l'orologio finto e la latenza finta
    motore.chiudi()
Oppure a mano: Motore(apri=apri_finto), poi avvia(), e il flusso è motore.flusso. Con un'altra
latenza: Motore(apri=functools.partial(apri_finto, latenza=0.05)).
Il flusso finto imita sounddevice quanto basta: il callback riceve outdata (blocco, 2) float32 di
zeri da riempire, frames, un time_info con outputBufferDacTime uguale a time più latency (o zero,
con dac_zero vero, come certi driver), currentTime uguale a time, e uno status con
output_underflow vero quando avanza riceve buco=True. Dopo ogni blocco time cresce di blocco
diviso fs secondi. Tutto ciò che il callback scrive resta in uscita, dall'inizio.
apri_finto accetta anche la scelta della scheda (D20), come apri_flusso: il flusso la ricorda, e
prende da lei il nome del dispositivo e dell'interfaccia, così cambia_scheda si prova con lui.
Le schede della macchina, per schede.py e per apri_flusso vera, le dà SchedeFinte, che il conftest
mette al posto di elenco_dispositivi_audio, scegli_dispositivo_audio e rileggi_dispositivi_audio di
GBUtils in tutte le prove: nessuna prova tocca PortAudio. La prova che vuole guardarle o cambiarle
chiede la fissatura schede_finte. sounddevice_finto è un modulo sounddevice che apre flussi finti,
da mettere in sys.modules quando si prova apri_flusso vera, che altrimenti aprirebbe la scheda.
"""

import functools
import types

import numpy as np

# La latenza finta, in secondi: 20 millesimi, un valore realistico per WASAPI.
LATENZA_FINTA = 0.02

# Le uscite finte, come le dà elenco_dispositivi_audio di GBUtils, nel suo ordine: prima la stessa
# scheda del predefinito di Windows, poi per preferenza d'interfaccia (ASIO, WASAPI, WDM-KS,
# DirectSound, MME) e per indice. Ricalcano quelle delle prove di schede_audio di MeTeOra, che vengono
# dalla macchina di Gabriele: un nome WDM-KS e uno DirectSound hanno un a capo dentro, e c'è il vivavoce
# di un telefono Bluetooth.
ALTOPARLANTI = "Altoparlanti (Realtek(R) Audio)"
DIGITALE = "Realtek Digital Output (Realtek(R) Audio)"
ASIO = "Realtek ASIO"
LATENZA_ASIO = 23.219954648526077
WDM_KS = "Speakers (Realtek HD Audio output)"
VIVAVOCE = "Output (@System32\\drivers\\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0\r\n;(iPhone))"
CUFFIE = "Cuffie (WH-1000XM4)"
INTERFACCE = {"ASIO": "ASIO", "WASAPI": "Windows WASAPI", "WDM-KS": "Windows WDM-KS", "DirectSound": "Windows DirectSound", "MME": "MME"}


def uscita_finta(indice, dispositivo, breve, latenza, stessa=False):
    """Una voce di elenco_dispositivi_audio, con tutte le sue chiavi; l'indice 4 è il predefinito."""
    return {"indice": indice, "dispositivo": dispositivo, "interfaccia": INTERFACCE[breve], "breve": breve, "canali": 2, "frequenza": 48000.0,
            "latenza": latenza, "predefinito": indice == 4, "stessa_scheda": stessa, "esclusiva": breve in ("ASIO", "WDM-KS"), "apribile": None,
            "motivo": None}


USCITE_FINTE = (
    uscita_finta(18, ALTOPARLANTI, "WASAPI", 3.0, stessa=True),
    uscita_finta(12, ALTOPARLANTI, "DirectSound", 120.0, stessa=True),
    uscita_finta(4, ALTOPARLANTI, "MME", 90.0, stessa=True),
    uscita_finta(16, ASIO, "ASIO", LATENZA_ASIO),
    uscita_finta(19, DIGITALE, "WASAPI", 3.0),
    uscita_finta(24, WDM_KS, "WDM-KS", 10.0),
    uscita_finta(34, VIVAVOCE, "WDM-KS", 10.0),
    uscita_finta(15, "Cuffie\r\n(Bluetooth)", "DirectSound", 120.0),
    uscita_finta(3, "Microsoft Sound Mapper - Output", "MME", 90.0),
)
# Le cuffie Bluetooth accese dopo l'avvio, che compaiono soltanto dopo una rilettura.
CUFFIE_FINTE = uscita_finta(40, CUFFIE, "WASAPI", 3.0)
# La scelta automatica finta: gli altoparlanti su WASAPI, come sulla macchina di Gabriele.
AUTOMATICA_FINTA = (18, "Windows WASAPI")


class SchedeFinte:
    """Le funzioni delle schede audio di GBUtils, finte, che il conftest mette al posto delle vere
    in tutte le prove. Attributi che la prova può leggere e cambiare: elenco, le uscite, come quelle
    di elenco_dispositivi_audio; automatica, la coppia (indice, interfaccia) che
    scegli_dispositivo_audio restituisce; arrivate, le uscite che si aggiungono all'elenco alla
    prossima rilettura, come cuffie accese dopo l'avvio; rotta, un'eccezione che la rilettura
    solleva; alla_rilettura, una funzione senza argomenti chiamata a ogni rilettura, prima di tutto,
    per guardare com'è il motore in quel momento. Quel che è successo: chiamate, in ordine, le tuple
    ("elenco", modo), ("scegli", modo, riprova) e ("rileggi",); riletture, ciò che alla_rilettura ha
    restituito a ogni rilettura, o None.
    elenco_dispositivi_audio con una prova diversa da "nessuno" solleva AssertionError: GBUtils
    aprirebbe le schede vere."""

    def __init__(self):
        self.elenco = [dict(voce) for voce in USCITE_FINTE]
        self.automatica = AUTOMATICA_FINTA
        self.arrivate = []
        self.rotta = None
        self.alla_rilettura = None
        self.chiamate = []
        self.riletture = []

    def elenco_dispositivi_audio(self, prova="predefinito", modo="scrittura"):
        if prova != "nessuno":
            raise AssertionError(f"elenco_dispositivi_audio con prova={prova!r} aprirebbe le schede vere")
        self.chiamate.append(("elenco", modo))
        return [dict(voce) for voce in self.elenco]

    def scegli_dispositivo_audio(self, api=None, riprova=False, modo="scrittura"):
        self.chiamate.append(("scegli", modo, riprova))
        if api is None:
            return self.automatica
        # Con l'indice o con il nome di un'interfaccia, come quella vera, ma senza aprire niente.
        for voce in self.elenco:
            if voce["indice"] == api or (isinstance(api, str) and voce["breve"].lower() == api.strip().lower()):
                return voce["indice"], voce["interfaccia"]
        raise ValueError(f"interfaccia audio {api} non disponibile")

    def rileggi_dispositivi_audio(self):
        self.chiamate.append(("rileggi",))
        self.riletture.append(self.alla_rilettura() if self.alla_rilettura is not None else None)
        if self.rotta is not None:
            raise self.rotta
        prima = [(voce["dispositivo"], voce["interfaccia"]) for voce in self.elenco]
        self.elenco.extend(dict(voce) for voce in self.arrivate)
        self.arrivate = []
        dopo = [(voce["dispositivo"], voce["interfaccia"]) for voce in self.elenco]
        return {"prima": len(prima), "dopo": len(dopo), "arrivate": [coppia for coppia in dopo if coppia not in prima], "sparite": [], "mixer": None}

    def voce(self, indice):
        """L'uscita dell'elenco con quell'indice, o None."""
        return next((voce for voce in self.elenco if voce["indice"] == indice), None)


class TempiFinti:
    """Il time_info del callback, con i nomi dei campi di PortAudio."""

    def __init__(self, dac, adesso):
        self.outputBufferDacTime = dac
        self.currentTime = adesso
        self.inputBufferAdcTime = 0.0


class StatoFinto:
    """Lo status del callback: buco vero vuol dire che la scheda è rimasta a secco."""

    def __init__(self, buco=False):
        self.output_underflow = bool(buco)
        self.output_overflow = False
        self.priming_output = False

    def __bool__(self):
        return self.output_underflow


class FlussoFinto:
    """Un flusso di uscita finto: callback(outdata, frames, time_info, status) come sounddevice,
    fs la frequenza di campionamento, blocco i campioni di ogni chiamata.
    Attributi: time, l'orologio finto in secondi, che la prova può anche spostare a mano per
    simulare il tempo che passa fra due blocchi; latency, la latenza in secondi; dac_zero, se vero
    il time_info porta outputBufferDacTime a zero; interfaccia e device, per lo stato del motore;
    attivo e chiuso, che dicono se il motore lo ha avviato e chiuso; azioni, i nomi di start, stop,
    abort e close nell'ordine in cui il motore li ha chiamati; blocchi, i blocchi scritti; scelta e
    riprova, ciò che il motore ha chiesto all'apertura; opzioni, quelle di OutputStream, col
    sounddevice finto."""

    def __init__(self, callback, fs, blocco, latenza=LATENZA_FINTA, tempo=0.0):
        self.callback = callback
        self.fs = int(fs)
        self.blocco = int(blocco)
        self.latency = float(latenza)
        self.time = float(tempo)
        self.dac_zero = False
        self.interfaccia = "finta"
        self.device = None
        self.attivo = False
        self.chiuso = False
        self.azioni = []
        self.blocchi = []
        self.scelta = None
        self.riprova = None
        self.opzioni = None

    def start(self):
        self.azioni.append("start")
        self.attivo = True

    def stop(self):
        self.azioni.append("stop")
        self.attivo = False

    def abort(self):
        self.azioni.append("abort")
        self.attivo = False

    def close(self):
        self.azioni.append("close")
        self.attivo = False
        self.chiuso = True

    @property
    def campioni(self):
        """Quanti campioni il callback ha preparato finora."""
        return len(self.blocchi) * self.blocco

    @property
    def uscita(self):
        """Tutto ciò che il callback ha scritto, dall'inizio: un array (campioni, 2) float32."""
        if not self.blocchi:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(self.blocchi)

    def avanza(self, blocchi=1, buco=False):
        """Chiama il callback per tanti blocchi, facendo avanzare l'orologio, e restituisce ciò che
        ha scritto in questi blocchi, (blocchi * blocco, 2) float32. buco vero segnala alla prima
        chiamata che la scheda è rimasta a secco."""
        nuovi = []
        for numero in range(int(blocchi)):
            uscita = np.zeros((self.blocco, 2), dtype=np.float32)
            dac = 0.0 if self.dac_zero else self.time + self.latency
            self.callback(uscita, self.blocco, TempiFinti(dac, self.time), StatoFinto(buco and numero == 0))
            self.blocchi.append(uscita)
            nuovi.append(uscita)
            self.time += self.blocco / self.fs
        if not nuovi:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(nuovi)

    def avanza_fino(self, campione):
        """Avanza finché il callback non ha preparato anche il campione indicato; restituisce ciò che
        ha scritto nel frattempo."""
        mancano = max(0, int(campione) + 1 - self.campioni)
        return self.avanza(-(-mancano // self.blocco))

    def avanza_secondi(self, secondi):
        """Avanza di almeno tanti secondi, in blocchi interi."""
        return self.avanza(-(-round(secondi * self.fs) // self.blocco))


def apri_finto(callback, fs, blocco, latenza=LATENZA_FINTA, scelta=None, riprova=False):
    """La fabbrica da dare al motore, Motore(apri=apri_finto): apre un FlussoFinto e lo avvia. Ogni
    scheda si apre: con una scelta il flusso ne prende il dispositivo e l'interfaccia, e il motore lo
    dà per la scheda scelta; senza, è la scheda "finta"."""
    flusso = FlussoFinto(callback, fs, blocco, latenza)
    flusso.scelta = dict(scelta) if scelta else {}
    flusso.riprova = riprova
    if scelta:
        flusso.device, flusso.interfaccia = scelta["dispositivo"], scelta["interfaccia"]
    flusso.start()
    return flusso


def sounddevice_finto(schede=None, rotte=()):
    """Un modulo sounddevice finto per provare apri_flusso vera, da mettere in sys.modules: senza,
    aprirebbe la scheda vera. OutputStream(**opzioni) apre un FlussoFinto col callback, la frequenza
    e il blocco chiesti, che ricorda le opzioni, e prende il nome del dispositivo e dell'interfaccia
    dalle uscite di schede, una SchedeFinte, cercando l'indice; il dispositivo predefinito, None, è
    "predefinito" di MME. I dispositivi in rotte, indici o None, non si aprono: OutputStream solleva
    RuntimeError. Attributi del modulo: aperti, i flussi aperti; chieste, le opzioni di ogni apertura
    tentata; rotte, l'insieme dei dispositivi che non si aprono, che la prova può cambiare."""
    modulo = types.ModuleType("sounddevice")
    modulo.aperti = []
    modulo.chieste = []
    modulo.rotte = set(rotte)

    class WasapiSettings:
        def __init__(self, auto_convert=False):
            self.auto_convert = auto_convert

    def output_stream(**opzioni):
        modulo.chieste.append(opzioni)
        dispositivo = opzioni.get("device")
        if dispositivo in modulo.rotte:
            raise RuntimeError("il dispositivo non si apre")
        flusso = FlussoFinto(opzioni["callback"], opzioni["samplerate"], opzioni["blocksize"])
        flusso.opzioni = opzioni
        voce = schede.voce(dispositivo) if schede is not None and dispositivo is not None else None
        if voce is not None:
            flusso.device, flusso.interfaccia = voce["dispositivo"], voce["interfaccia"]
        elif dispositivo is None:
            flusso.device, flusso.interfaccia = "predefinito", "MME"
        else:
            flusso.device, flusso.interfaccia = f"dispositivo {dispositivo}", "sconosciuta"
        modulo.aperti.append(flusso)
        return flusso

    modulo.WasapiSettings = WasapiSettings
    modulo.OutputStream = output_stream
    return modulo


def motore_finto(fs=None, blocco=None, latenza=LATENZA_FINTA):
    """Un motore avviato sul flusso finto, e il flusso: (motore, flusso). Chi lo crea lo chiude con
    motore.chiudi() a fine prova, che rende anche il timer di Windows."""
    from costanti import AUDIO_BLOCCO, FS
    from motore_audio import Motore

    motore = Motore(fs=FS if fs is None else fs, blocco=AUDIO_BLOCCO if blocco is None else blocco, apri=functools.partial(apri_finto, latenza=latenza))
    motore.avvia()
    return motore, motore.flusso
