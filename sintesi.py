"""
La sintesi di Ciechino: le forme d'onda delle portanti, la panoramica, i bip e i buffer della mira.
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

from dataclasses import dataclass

import numpy as np

from costanti import (
    ALZO_FREQUENZE,
    BIP_DB,
    BIP_DISTACCO,
    BIP_DURATA,
    BIP_FREQUENZE,
    BIP_SILENZIO_FINALE,
    DIREZIONE_FREQUENZA,
    FS,
    PARAMETRI,
    POTENZA_DB_MINIMO,
    POTENZA_FREQUENZA,
    SCIVOLATA_DB,
    SCIVOLATA_DURATA,
    SCIVOLATA_FORMA,
    TACCA_DB,
    TACCA_DURATA,
    TACCA_FREQUENZE,
    TACCHE,
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
# Le rampe contro i click, in secondi: un millesimo per i bip, come vuole il contratto, cinque per
# l'inizio e la fine della scala di una scivolata.
RAMPA_DEL_BIP = 0.001
RAMPA_DELLA_SCALA = 0.005
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


@dataclass
class Scivolata:
    """Il buffer di una barra della mira e dove sta la scala dentro di lui.
    buffer comincia a pausa finita, col primo bip; inizio_scivolata è il campione del buffer dove
    comincia la scala, lunga campioni_scivolata; il buffer finisce con la scala. tacche elenca le
    coppie (inizio, fine) dei bip delle tacche, in campioni del buffer, fine esclusa."""

    parametro: str
    buffer: np.ndarray
    inizio_scivolata: int
    campioni_scivolata: int
    fs: int = FS
    tacche: tuple = ()

    @property
    def fine_scivolata(self):
        """Il campione del buffer dove la scala è finita, che è anche la lunghezza del buffer."""
        return self.inizio_scivolata + self.campioni_scivolata


def _glissando(frequenze, frazioni, esponenziale):
    """Le frequenze della scala: un numero resta fermo, una coppia (da, a) scorre."""
    if np.ndim(frequenze) == 0:
        return np.full(len(frazioni), float(frequenze))
    da, a = (float(x) for x in frequenze)
    if esponenziale:
        return da * (a / da) ** frazioni
    return da + (a - da) * frazioni


def frequenze_delle_tacche(quante=TACCHE, frequenze=TACCA_FREQUENZE):
    """Le frequenze delle tacche, dalla prima all'ultima su scala esponenziale."""
    da, a = (float(x) for x in frequenze)
    if quante <= 1:
        return np.full(max(0, quante), da)
    return da * (a / da) ** (np.arange(quante) / (quante - 1))


def scivolata(parametro, tacche=True, fs=FS):
    """Il buffer della mira per un parametro di PARAMETRI, coi valori di costanti.py.
    Il buffer: i bip di BIP_FREQUENZE del parametro, lunghi BIP_DURATA e separati da BIP_DISTACCO
    di silenzio, poi BIP_SILENZIO_FINALE di silenzio, poi la scala di SCIVOLATA_DURATA secondi,
    arrotondati al campione, nella forma SCIVOLATA_FORMA del parametro. Ogni durata si arrotonda al
    campione da sola, così il conto dei campioni è esatto e il motore, la mira e le prove lo rifanno
    uguale.
    La scala: la direzione va da tutto a sinistra a tutto a destra a DIREZIONE_FREQUENZA; l'alzo
    sale da ALZO_FREQUENZE[0] a ALZO_FREQUENZE[1], al centro, su scala esponenziale in hertz; la
    potenza a POTENZA_FREQUENZA, al centro, cresce da POTENZA_DB_MINIMO sotto SCIVOLATA_DB fino a
    SCIVOLATA_DB, lineare in decibel. Il campione i della scala vale la frazione i diviso
    campioni_scivolata, la stessa che mira.valore_scelto dà allo spazio.
    Le tacche, se tacche è vero: TACCHE bip di TACCA_DURATA a TACCA_DB, al centro, da
    TACCA_FREQUENZE[0] a TACCA_FREQUENZE[1] su scala esponenziale; la k-esima finisce a k decimi
    della scala, quindi l'ultima la chiude. Senza tacche, le prove sentono la scala da sola.
    Fino alla 0.12.0 la funzione prendeva anche le varianti dei banchi d'ascolto, la forma, le
    frequenze, le curve, il minimo della potenza, il volume delle tacche e la durata dei bip: i
    banchi sono usciti con la 1.0.0, e con loro le varianti."""
    if parametro not in PARAMETRI:
        raise ValueError(f"parametro sconosciuto: {parametro!r}")
    forma = SCIVOLATA_FORMA[parametro]
    fs = int(fs)
    campioni_bip = round(BIP_DURATA * fs)
    campioni_distacco = round(BIP_DISTACCO * fs)
    campioni_silenzio = round(BIP_SILENZIO_FINALE * fs)
    campioni_scala = round(SCIVOLATA_DURATA * fs)
    frequenze_bip = BIP_FREQUENZE[parametro]
    quanti_bip = len(frequenze_bip)
    inizio = quanti_bip * campioni_bip + max(0, quanti_bip - 1) * campioni_distacco + campioni_silenzio
    sinistra_destra = np.zeros((inizio + campioni_scala, 2))
    for i, frequenza in enumerate(frequenze_bip):
        dove = i * (campioni_bip + campioni_distacco)
        sinistra_destra[dove:dove + campioni_bip] += panoramica(_bip(frequenza, campioni_bip, BIP_DB, "seno", fs), 0.0)
    frazioni = np.arange(campioni_scala) / campioni_scala
    livello = db_in_lineare(SCIVOLATA_DB)
    pan = 0.0
    if parametro == "direzione":
        hertz = _glissando(DIREZIONE_FREQUENZA, frazioni, True)
        pan = 2.0 * frazioni - 1.0
    elif parametro == "alzo":
        hertz = _glissando(ALZO_FREQUENZE, frazioni, True)
    else:
        hertz = _glissando(POTENZA_FREQUENZA, frazioni, True)
        livello = db_in_lineare(SCIVOLATA_DB + POTENZA_DB_MINIMO * (1.0 - frazioni))
    scala = Oscillatore(forma, fs).rendi(hertz) * livello
    _rampe(scala, round(RAMPA_DELLA_SCALA * fs))
    sinistra_destra[inizio:] += panoramica(scala, pan)
    posizioni = []
    if tacche and TACCHE > 0:
        campioni_tacca = round(TACCA_DURATA * fs)
        for k, frequenza in enumerate(frequenze_delle_tacche(), start=1):
            fine = inizio + round(k * campioni_scala / TACCHE)
            dove = max(inizio, fine - campioni_tacca)
            sinistra_destra[dove:fine] += panoramica(_bip(frequenza, fine - dove, TACCA_DB, "seno", fs), 0.0)
            posizioni.append((dove, fine))
    # Con i valori di costanti.py la somma resta ben sotto uno; una taratura nuova esagerata, per
    # esempio tacche a zero decibel, si abbassa tutta insieme invece di tagliarsi.
    picco = float(np.max(np.abs(sinistra_destra))) if len(sinistra_destra) else 0.0
    if picco > 1.0:
        sinistra_destra /= picco
    return Scivolata(parametro, sinistra_destra.astype(np.float32), inizio, campioni_scala, fs, tuple(posizioni))
