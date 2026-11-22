"""
La sintesi di Ciechino: le forme d'onda delle portanti, la panoramica, i bip e i suoni della mira manuale.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di _sintetizza di Acusticator in GBUtils per la legge
della panoramica a potenza costante.
Usa soltanto numpy e non suona niente: prepara campioni che il motore audio mette in fila. Le
portanti sono fatte per somma di poche armoniche, e ogni armonica si toglie, campione per campione,
dove supererebbe FREQUENZA_LIMITE_ARMONICHE: così una scivolata o un bersaglio che sale verso
l'acuto non piega mai un'armonica sotto la frequenza di Nyquist, cioè niente aliasing, che a
orecchio è un fischio che scende mentre il suono sale. L'oscillatore tiene la fase in float64 e la
porta da un blocco all'altro, così una voce lunga del motore non ha gradini fra un blocco e il
seguente.
Ogni numero di taratura viene da costanti.py; quelli qui sotto riguardano soltanto il timbro e i
dettagli della sintesi, e restano qui finché qualcuno non vorrà tararli giocando.
"""

import numpy as np

from costanti import (
    ALZO_FREQUENZE,
    FS,
    PARAMETRI,
    POTENZA_DB_MINIMO,
    PUNTATORE_DB,
    TACCA_DB,
    TACCA_DURATA,
    TACCA_FORMA,
    TACCA_FREQUENZE,
    TACCHE,
    TICK_DB,
    TICK_DURATA,
    TICK_FORMA,
    TICK_FREQUENZE,
)

# Le forme delle portanti, per le scivolate, i bersagli e il proiettile.
FORME_PORTANTI = ("seno", "triangolo", "sega_morbida", "organo")
# Sopra questa frequenza un'armonica non c'è più: 18.000 hertz stanno sotto la metà di 44.100 con
# margine, e sopra quasi nessuno sente niente.
FREQUENZA_LIMITE_ARMONICHE = 18000.0
# L'armonica non sparisce di colpo, che in una scivolata farebbe un piccolo click a ogni armonica
# tolta: si spegne su questa fascia di hertz, e al limite vale già zero.
FASCIA_DI_TAGLIO = 1000.0
# Il triangolo: le armoniche dispari, a 1/n² e a segni alterni; sette bastano, l'ottava sarebbe a
# un duecentoventicinquesimo della fondamentale.
ARMONICHE_DEL_TRIANGOLO = 7
# La sega morbida: tutte le armoniche a 1/n, con una caduta in più per ogni armonica, che toglie il
# ronzio tagliente della sega vera.
ARMONICHE_DELLA_SEGA = 12
CADUTA_DELLA_SEGA = 0.8
# L'organo: la fondamentale con la seconda e la terza armonica deboli.
ARMONICHE_DELL_ORGANO = (1.0, 0.25, 0.12)
# Il picco di ogni forma si misura su tanti punti di un periodo; il margine copre l'errore della
# misura, così nessun campione supera mai 1.
PUNTI_PER_IL_PICCO = 8192
MARGINE_DEL_PICCO = 1.001
# La rampa contro i click dei bip, in secondi: un millesimo, come vuole il contratto. Il suono dei
# valori della mira manuale ha l'attacco e la sfumatura delle voci del motore.
RAMPA_DEL_BIP = 0.001
# lineare_in_db di un silenzio perfetto: un pavimento invece di meno infinito.
PAVIMENTO_IN_DB = -240.0


def _armoniche_della_forma(forma):
    """Le coppie (numero, ampiezza) di una forma, prima della normalizzazione."""
    if forma == "seno":
        return ((1, 1.0),)
    if forma == "triangolo":
        return tuple((2 * k + 1, (-1.0) ** k / (2 * k + 1) ** 2) for k in range(ARMONICHE_DEL_TRIANGOLO))
    if forma == "sega_morbida":
        return tuple((n, CADUTA_DELLA_SEGA ** (n - 1) / n) for n in range(1, ARMONICHE_DELLA_SEGA + 1))
    return tuple((n, ampiezza) for n, ampiezza in enumerate(ARMONICHE_DELL_ORGANO, start=1) if ampiezza)


def _normalizza(armoniche):
    """Divide le ampiezze per il picco più alto che la forma può raggiungere.
    Le armoniche si tolgono dall'alto, e il peso di ognuna, campione per campione, non cresce mai col
    numero dell'armonica: il vettore dei pesi sta quindi dentro l'inviluppo convesso delle somme
    parziali, e il valore assoluto, convesso, ha il massimo su una di quelle. Basta perciò il picco
    più alto fra le somme parziali, e nessun campione può uscire da più o meno uno."""
    angoli = np.arange(PUNTI_PER_IL_PICCO) * (2.0 * np.pi / PUNTI_PER_IL_PICCO)
    somma = np.zeros(PUNTI_PER_IL_PICCO)
    picco = 0.0
    for numero, ampiezza in armoniche:
        somma += ampiezza * np.sin(numero * angoli)
        picco = max(picco, float(np.max(np.abs(somma))))
    norma = picco * MARGINE_DEL_PICCO
    return tuple((numero, ampiezza / norma) for numero, ampiezza in armoniche)


_ARMONICHE = {forma: _normalizza(_armoniche_della_forma(forma)) for forma in FORME_PORTANTI}


class Oscillatore:
    """Una portante di forma FORME_PORTANTI che segue una frequenza diversa a ogni campione.
    La fase, in giri, sta in float64 e prosegue da una chiamata di rendi all'altra: la voce del
    motore la chiama un blocco alla volta, e l'onda non deve accorgersene."""

    def __init__(self, forma, fs=FS, fase=0.0):
        if forma not in _ARMONICHE:
            raise ValueError(f"forma sconosciuta: {forma!r}")
        self.forma = forma
        self.fs = int(fs)
        self.fase = float(fase) % 1.0
        self._armoniche = _ARMONICHE[forma]

    def rendi(self, frequenze, n=None):
        """Un blocco mono, float64 fra meno uno e uno, dalle frequenze di ogni campione.
        frequenze è un array, oppure un numero solo insieme a n, la lunghezza del blocco. Una
        frequenza nulla o negativa vale silenzio, e la fase resta dov'è."""
        f = np.asarray(frequenze, dtype=np.float64)
        if f.ndim == 0:
            if n is None:
                raise ValueError("con una frequenza sola serve anche n, quanti campioni")
            f = np.full(int(n), float(f))
        m = len(f)
        if m == 0:
            return np.zeros(0)
        f = np.maximum(f, 0.0)
        passi = f / self.fs
        cumulata = np.cumsum(passi)
        # La fase di ogni campione è quella con cui il campione comincia: il primo ha la fase lasciata
        # dal blocco prima, e il blocco seguente comincerà dalla somma di tutti i passi.
        fasi = (self.fase + cumulata - passi) % 1.0
        self.fase = float((self.fase + cumulata[-1]) % 1.0)
        angoli = 2.0 * np.pi * fasi
        uscita = np.zeros(m)
        f_minima = float(f.min())
        f_massima = float(f.max())
        if f_massima <= 0.0:
            return uscita
        spente = f <= 0.0 if f_minima <= 0.0 else None
        for numero, ampiezza in self._armoniche:
            if f_minima > 0.0 and numero * f_minima >= FREQUENZA_LIMITE_ARMONICHE:
                # Da qui in su tutte le armoniche sono oltre il limite in ogni campione.
                break
            onda = np.sin(numero * angoli)
            if spente is not None or numero * f_massima > FREQUENZA_LIMITE_ARMONICHE - FASCIA_DI_TAGLIO:
                peso = np.clip((FREQUENZA_LIMITE_ARMONICHE - numero * f) / FASCIA_DI_TAGLIO, 0.0, 1.0)
                if spente is not None:
                    peso[spente] = 0.0
                onda *= peso
            uscita += ampiezza * onda
        return uscita


def panoramica(mono, pan):
    """Il blocco stereo (N, 2) float32 di un suono mono, con pan da meno uno, tutto a sinistra, a
    uno, tutto a destra; pan è un numero o un array di N valori. La legge è a potenza costante,
    coseno e seno come in Acusticator: al centro ogni lato sta a meno tre decibel, e un suono che
    attraversa lo spazio non perde volume in mezzo."""
    mono = np.asarray(mono, dtype=np.float64)
    angolo = (np.clip(np.asarray(pan, dtype=np.float64), -1.0, 1.0) + 1.0) * (np.pi / 4.0)
    stereo = np.empty((len(mono), 2), dtype=np.float32)
    stereo[:, 0] = mono * np.cos(angolo)
    stereo[:, 1] = mono * np.sin(angolo)
    return stereo


def db_in_lineare(db):
    """Da decibel a guadagno lineare, per un numero o un array."""
    valore = np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)
    return float(valore) if valore.ndim == 0 else valore


def lineare_in_db(x):
    """Da guadagno lineare a decibel, per un numero o un array; lo zero vale PAVIMENTO_IN_DB."""
    ampiezza = np.maximum(np.abs(np.asarray(x, dtype=np.float64)), 10.0 ** (PAVIMENTO_IN_DB / 20.0))
    valore = 20.0 * np.log10(ampiezza)
    return float(valore) if valore.ndim == 0 else valore


def _rampe(onda, campioni):
    """Apre e chiude un suono con due rampe a coseno rialzato, contro i click, in luogo."""
    campioni = min(int(campioni), len(onda) // 2)
    if campioni <= 0:
        return onda
    rampa = 0.5 - 0.5 * np.cos(np.pi * (np.arange(campioni) + 0.5) / campioni)
    onda[:campioni] *= rampa
    onda[-campioni:] *= rampa[::-1]
    return onda


def _bip(frequenza, campioni, db, forma, fs):
    """Il bip mono in float64, lungo esattamente tanti campioni."""
    campioni = int(campioni)
    if campioni <= 0:
        return np.zeros(0)
    onda = Oscillatore(forma, fs).rendi(frequenza, campioni) * db_in_lineare(db)
    return _rampe(onda, round(RAMPA_DEL_BIP * fs))


def bip(frequenza, durata, db, forma="seno", fs=FS):
    """Un bip mono float32 di durata secondi, arrotondati al campione, a db decibel, con le rampe
    di un millesimo contro i click."""
    return _bip(frequenza, round(durata * fs), db, forma, fs).astype(np.float32)


def frequenze_delle_tacche(quante=TACCHE, frequenze=TACCA_FREQUENZE):
    """Le frequenze delle tacche, dalla prima all'ultima su scala esponenziale."""
    da, a = (float(x) for x in frequenze)
    if quante <= 1:
        return np.full(max(0, quante), da)
    return da * (a / da) ** (np.arange(quante) / (quante - 1))


def _controlla_il_parametro(parametro):
    if parametro not in PARAMETRI:
        raise ValueError(f"parametro sconosciuto: {parametro!r}")


def pan_della_direzione(direzione):
    """La panoramica del puntatore per la direzione da 0 a 1: da meno uno, tutto a sinistra, a uno, tutto a
    destra, la stessa scala della panoramica dei bersagli. Uno scalare per un numero, un array per un array."""
    pan = 2.0 * np.clip(np.asarray(direzione, dtype=float), 0.0, 1.0) - 1.0
    return float(pan) if pan.ndim == 0 else pan


def parametri_del_puntatore(direzione, alzo, potenza):
    """(frequenze, pan, guadagni lineari) del puntatore della mira manuale, la voce sola che porta sempre i
    tre valori da 0 a 1 (D31, dalla 2.0.1): la panoramica dalla direzione, con pan_della_direzione;
    l'altezza dall'alzo, da ALZO_FREQUENZE[0] a ALZO_FREQUENZE[1] su scala esponenziale; il volume dalla
    potenza, da POTENZA_DB_MINIMO sotto PUNTATORE_DB fino a PUNTATORE_DB, lineare in decibel. I valori sono
    numeri o array, che si allargano l'uno sull'altro; tre array della stessa forma, di almeno un elemento.
    Fino alla 2.0.0 ogni valore aveva una voce sua, e gli altri due non si sentivano."""
    direzione, alzo, potenza = np.broadcast_arrays(*(np.clip(np.atleast_1d(np.asarray(valore, dtype=float)), 0.0, 1.0)
                                                     for valore in (direzione, alzo, potenza)))
    basso, alto = ALZO_FREQUENZE
    guadagni = db_in_lineare(PUNTATORE_DB + POTENZA_DB_MINIMO * (1.0 - potenza))
    return basso * (alto / basso) ** alzo, pan_della_direzione(direzione), guadagni


def tacca_mono(parametro, k, fs=FS):
    """La tacca k, da 1 a TACCHE meno 1, di un valore della mira manuale, mono float32: un bip di
    TACCA_DURATA a TACCA_DB, alla frequenza k-esima di frequenze_delle_tacche(TACCHE - 1), cioè le nove
    tacche dei decimi su tutta la scala da TACCA_FREQUENZE[0] a TACCA_FREQUENZE[1], nella forma TACCA_FORMA
    del parametro (D29). La partita le prepara una volta sola, e le colloca nel punto del puntatore quando
    le mette in fila (D31)."""
    _controlla_il_parametro(parametro)
    if not 1 <= int(k) < TACCHE:
        raise ValueError(f"le tacche vanno da 1 a {TACCHE - 1}, non {k!r}")
    frequenza = float(frequenze_delle_tacche(TACCHE - 1)[int(k) - 1])
    return _bip(frequenza, round(TACCA_DURATA * fs), TACCA_DB, TACCA_FORMA[parametro], int(fs)).astype(np.float32)


def tacca_dell_asse(parametro, k, fs=FS, pan=None):
    """La tacca k di tacca_mono, stereo float32, col pan dato, da meno uno a uno: quello del puntatore
    nell'istante della tacca (D31). Senza pan, quello del puntatore con gli altri valori al centro: per la
    direzione la panoramica del suo decimo, per l'alzo e la potenza il centro."""
    mono = tacca_mono(parametro, k, fs)
    if pan is None:
        pan = pan_della_direzione(int(k) / TACCHE) if parametro == "direzione" else 0.0
    return panoramica(mono, pan).astype(np.float32)


def tick_del_tempo(secondo, secondi, fs=FS):
    """Il tick del secondo dato, da 1 a secondi meno 1, di una mira a tempo di tanti secondi (D29): un bip
    stereo float32 di TICK_DURATA a TICK_DB, al centro, nella forma TICK_FORMA, che sale col tempo che
    scorre da TICK_FREQUENZE[0], al primo secondo, a TICK_FREQUENZE[1], all'ultimo, su scala esponenziale."""
    basso, alto = TICK_FREQUENZE
    frazione = 0.0 if secondi <= 2 else min(1.0, max(0.0, (float(secondo) - 1.0) / (float(secondi) - 2.0)))
    mono = _bip(basso * (alto / basso) ** frazione, round(TICK_DURATA * fs), TICK_DB, TICK_FORMA, int(fs))
    return panoramica(mono, 0.0).astype(np.float32)
