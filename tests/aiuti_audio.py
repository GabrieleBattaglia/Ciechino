"""
Gli aiuti audio delle prove di Ciechino: un flusso finto che fa avanzare il motore audio a mano,
blocco per blocco, con un orologio finto.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
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
"""

import functools

import numpy as np

# La latenza finta, in secondi: 20 millesimi, un valore realistico per WASAPI.
LATENZA_FINTA = 0.02


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
    attivo e chiuso, che dicono se il motore lo ha avviato e chiuso; blocchi, i blocchi scritti."""

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
        self.blocchi = []

    def start(self):
        self.attivo = True

    def stop(self):
        self.attivo = False

    def abort(self):
        self.attivo = False

    def close(self):
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


def apri_finto(callback, fs, blocco, latenza=LATENZA_FINTA):
    """La fabbrica da dare al motore, Motore(apri=apri_finto): apre un FlussoFinto e lo avvia."""
    flusso = FlussoFinto(callback, fs, blocco, latenza)
    flusso.start()
    return flusso


def motore_finto(fs=None, blocco=None, latenza=LATENZA_FINTA):
    """Un motore avviato sul flusso finto, e il flusso: (motore, flusso). Chi lo crea lo chiude con
    motore.chiudi() a fine prova, che rende anche il timer di Windows."""
    from costanti import AUDIO_BLOCCO, FS
    from motore_audio import Motore

    motore = Motore(fs=FS if fs is None else fs, blocco=AUDIO_BLOCCO if blocco is None else blocco, apri=functools.partial(apri_finto, latenza=latenza))
    motore.avvia()
    return motore, motore.flusso
