"""
I bersagli di Ciechino: dove stanno, come si muovono, come suonano e come cambiano coi livelli.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Niente wx e niente audio, soltanto numpy e la libreria standard. Tutto si calcola in qualunque
istante senza stato e senza simulare: la posizione dal tempo di gioco, l'LFO dal tempo dell'audio.
Così il motore audio può chiedere i parametri di un blocco qualunque, il volo può esaminare tutto
il tragitto in un colpo solo, e fermando il tempo di gioco i bersagli si fermano mentre il loro
suono continua a vibrare.
Il bonus: nasce col centro appena fuori dal cubo, a un raggio dalla parete di sinistra o di destra,
cioè già tangente al cubo, ed è vivo da subito, dal suo tempo nato; lo attraversa lungo x, alla
velocità (CUBO_LATO più due raggi) diviso durata_del_bonus(livello), e smette di essere vivo a
tempo_di_uscita, quando è uscito del tutto dall'altra parte. Prima di nato nessun bersaglio è vivo,
perché non esisteva ancora.
"""

import math
from dataclasses import dataclass, field

import numpy as np

import fisica
from costanti import (
    AUMENTO_DI_VELOCITA,
    BERSAGLIO_DB,
    CUBO_LATO,
    CUBO_X,
    DURATA_BONUS_INIZIALE,
    DURATA_BONUS_MINIMA,
    FATTORE_ESCURSIONE,
    FORME_LFO,
    FREQUENZA_LFO,
    LIVELLI_CON_UN_BERSAGLIO_IN_PIU,
    LIVELLO_DEI_BONUS,
    LIVELLO_DEI_MOBILI,
    METRI_PER_OTTAVA,
    RAGGIO_BONUS_FATTORE,
    RAGGIO_INIZIALE,
    RAGGIO_MINIMO,
    RIDUZIONE_DEL_RAGGIO,
    RIDUZIONE_DURATA_BONUS,
    VELOCITA_INIZIALE_BERSAGLI,
    VELOCITA_MASSIMA_BERSAGLI,
)

# La durata, in secondi, del raccordo di ogni fronte dell'LFO quadro: il progetto vuole i fronti
# smussati di qualche millesimo, perché un salto netto di frequenza della portante si sente come
# un click. Il raccordo è un quarto di seno, centrato sul fronte, e non supera mai un quarto di ciclo.
RACCORDO_DELLA_QUADRA = 0.004

# Le costanti di splitmix64, il mescolatore di Sebastiano Vigna: trasforma il seme e il numero del
# ciclo in un numero a caso senza stato, così casuale e deriva danno sempre gli stessi valori nello
# stesso ciclo, in qualunque ordine e a pezzi di qualunque lunghezza li si calcoli.
_SPLITMIX_INCREMENTO = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_PRIMO = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_SECONDO = np.uint64(0x94D049BB133111EB)
_MASCHERA_64 = 0xFFFFFFFFFFFFFFFF
# Un numero a caso uniforme usa i 53 bit alti dell'hash, quanti ne tiene la mantissa di un float.
_BIT_DEL_FLOAT = 53


def _splitmix(valori):
    """splitmix64 su un array di uint64, con l'aritmetica modulo 2 alla 64 che il mescolatore vuole."""
    with np.errstate(over="ignore"):
        z = valori + _SPLITMIX_INCREMENTO
        z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_PRIMO
        z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_SECONDO
        return z ^ (z >> np.uint64(31))


def _caso(seme, cicli):
    """Un valore a caso fra meno 1 e 1 per ogni numero di ciclo, deterministico dato il seme."""
    cicli = np.asarray(cicli, dtype=np.int64)
    forma = cicli.shape
    numeri = np.ascontiguousarray(cicli.reshape(-1)).view(np.uint64)
    mescolato_seme = _splitmix(np.array([int(seme) & _MASCHERA_64], dtype=np.uint64))[0]
    hash_ = _splitmix(numeri ^ mescolato_seme)
    uniformi = (hash_ >> np.uint64(64 - _BIT_DEL_FLOAT)).astype(np.float64) / float(2**_BIT_DEL_FLOAT)
    return (2.0 * uniformi - 1.0).reshape(forma)


def _triangolo(fasi):
    """Il triangolo allineato al seno: 0 all'inizio del ciclo, 1 a un quarto, meno 1 a tre quarti."""
    return 1.0 - 4.0 * np.abs(np.mod(fasi + 0.25, 1.0) - 0.5)


def lfo(forma, tempi, frequenza, fase=0.0, seme=0):
    """L'LFO di un bersaglio, fra meno 1 e 1, nei tempi dati in secondi. La fase è una frazione di
    ciclo, da 0 a 1; il seme conta solo per casuale e deriva. Le forme sono quelle di FORME_LFO:
    seno; sega, che sale e ricade; sega_inversa, che scende e risale; triangolo; quadra, coi fronti
    raccordati in RACCORDO_DELLA_QUADRA; casuale, un valore nuovo a ogni ciclo tenuto fermo; deriva,
    un valore a caso a ogni inizio di ciclo, raccordati col coseno. Deriva passa all'inizio di ogni
    ciclo dallo stesso valore che casuale tiene in quel ciclo. Scalare in ingresso, float in uscita."""
    fasi = np.asarray(tempi, dtype=float) * float(frequenza) + float(fase)
    cicli = np.floor(fasi)
    frazioni = fasi - cicli
    if forma == "seno":
        valori = np.sin(2.0 * math.pi * fasi)
    elif forma == "sega":
        valori = 2.0 * frazioni - 1.0
    elif forma == "sega_inversa":
        valori = 1.0 - 2.0 * frazioni
    elif forma == "triangolo":
        valori = _triangolo(fasi)
    elif forma == "quadra":
        # Il triangolo attraversa lo zero sui fronti con pendenza 4 per ciclo: diviso per quattro
        # mezzi raccordi e tagliato a più o meno 1, è una quadra coi fronti a rampa, che il quarto
        # di seno arrotonda.
        mezzo_raccordo = min(RACCORDO_DELLA_QUADRA * abs(float(frequenza)) / 2.0, 0.25)
        if mezzo_raccordo > 0.0:
            valori = np.sin(math.pi / 2.0 * np.clip(_triangolo(fasi) / (4.0 * mezzo_raccordo), -1.0, 1.0))
        else:
            valori = np.where(frazioni < 0.5, 1.0, -1.0)
    elif forma == "casuale":
        valori = _caso(seme, cicli)
    elif forma == "deriva":
        da = _caso(seme, cicli)
        verso = _caso(seme, cicli + 1.0)
        valori = da + (verso - da) * (1.0 - np.cos(math.pi * frazioni)) / 2.0
    else:
        raise ValueError(f"Forma dell'LFO sconosciuta: {forma!r}; le forme sono {', '.join(FORME_LFO)}.")
    return fisica.scalare_o_array(np.clip(valori, -1.0, 1.0))


def rimbalzo(libera, minimi, massimi):
    """La posizione che rimbalza sulle pareti, asse per asse, senza simulare i rimbalzi: la
    posizione libera, quella del moto rettilineo, riflessa dentro [minimi, massimi] da una funzione
    a triangolo di periodo due volte la larghezza. È continua, resta sempre dentro, e la velocità
    scalare resta quella del moto libero perché ogni asse cambia solo di segno. Un asse largo zero
    tiene la posizione sul suo minimo."""
    libera = np.asarray(libera, dtype=float)
    minimi = np.asarray(minimi, dtype=float)
    larghezze = np.asarray(massimi, dtype=float) - minimi
    aperti = larghezze > 0.0
    sicure = np.where(aperti, larghezze, 1.0)
    dentro = np.mod(libera - minimi, 2.0 * sicure)
    riflessa = minimi + sicure - np.abs(dentro - sicure)
    return np.where(aperti, riflessa, minimi)


@dataclass(eq=False)
class Bersaglio:
    """Un bersaglio in volo. Si confronta per identità, non per contenuto: due bersagli con gli
    stessi numeri restano due bersagli, e il volo dice quale ha colpito."""

    numero: int
    forma: str
    raggio: float
    partenza: np.ndarray
    velocita: np.ndarray = field(default_factory=lambda: np.zeros(3))
    nato: float = 0.0
    bonus: bool = False
    frequenza_lfo: float = FREQUENZA_LFO[0]
    fase_lfo: float = 0.0
    seme: int = 0

    def __post_init__(self):
        if self.forma not in FORME_LFO:
            raise ValueError(f"Forma dell'LFO sconosciuta: {self.forma!r}; le forme sono {', '.join(FORME_LFO)}.")
        self.raggio = float(self.raggio)
        self.partenza = np.array(self.partenza, dtype=float).reshape(3)
        self.velocita = np.array(self.velocita, dtype=float).reshape(3)
        self.nato = float(self.nato)
        self.frequenza_lfo = float(self.frequenza_lfo)
        self.fase_lfo = float(self.fase_lfo)
        self.seme = int(self.seme)

    @property
    def velocita_scalare(self):
        """La velocità in metri al secondo, cioè il modulo del vettore velocità."""
        return float(np.linalg.norm(self.velocita))

    @property
    def mobile(self):
        return self.velocita_scalare > 0.0

    @property
    def zona(self):
        """(minimi, massimi) della zona dove può stare il suo centro, quella dei rimbalzi."""
        return fisica.zona_del_centro(self.raggio)

    @property
    def tempo_di_uscita(self):
        """Per il bonus il tempo di gioco in cui è uscito del tutto dal cubo, dalla parte verso cui va;
        per gli altri None. Un bonus fermo non esce mai: infinito."""
        if not self.bonus:
            return None
        verso = float(self.velocita[0])
        if verso > 0.0:
            strada = CUBO_X[1] + self.raggio - float(self.partenza[0])
        elif verso < 0.0:
            strada = float(self.partenza[0]) - (CUBO_X[0] - self.raggio)
        else:
            return math.inf
        return self.nato + max(strada, 0.0) / abs(verso)

    def posizione(self, t):
        """La posizione del centro al tempo di gioco t: (3,) per t scalare, (N, 3) per un array.
        Il fermo sta dov'è nato; il mobile va dritto e rimbalza sulle pareti della sua zona; il bonus
        va dritto e basta, perché entra ed esce dal cubo."""
        trascorso = (np.asarray(t, dtype=float) - self.nato)[..., None]
        libera = self.partenza + self.velocita * trascorso
        if self.bonus or not self.mobile:
            return libera
        minimi, massimi = self.zona
        return rimbalzo(libera, minimi, massimi)

    def vivo(self, t):
        """Vero dal tempo nato in poi; per il bonus fino a tempo_di_uscita compreso. Per un array di
        tempi, un array di booleani."""
        tempi = np.asarray(t, dtype=float)
        vivo = tempi >= self.nato
        uscita = self.tempo_di_uscita
        if uscita is not None:
            vivo = vivo & (tempi <= uscita)
        if vivo.ndim == 0:
            return bool(vivo)
        return vivo

    def lfo(self, t_audio):
        """Il suo LFO, fra meno 1 e 1, nei tempi dell'audio."""
        return lfo(self.forma, t_audio, self.frequenza_lfo, self.fase_lfo, self.seme)

    def mezza_escursione_ottave(self):
        """Di quante ottave l'LFO sposta la frequenza del centro, in su e in giù: il raggio per
        FATTORE_ESCURSIONE, in ottave, così l'oscillazione copre le altezze del bersaglio."""
        return self.raggio * FATTORE_ESCURSIONE / METRI_PER_OTTAVA

    def parametri_sonori(self, tempi_di_gioco, tempi_audio):
        """(frequenze, pan, guadagni lineari) per la voce del motore, tre array lunghi come i tempi:
        la posizione dai tempi di gioco, l'LFO dai tempi dell'audio, BERSAGLIO_DB compreso. Uno dei
        due tempi può essere scalare, per esempio il tempo di gioco fermo durante la mira."""
        gioco, audio = np.broadcast_arrays(np.atleast_1d(np.asarray(tempi_di_gioco, dtype=float)), np.atleast_1d(np.asarray(tempi_audio, dtype=float)))
        frequenze, pan, guadagni = fisica.suono_di(self.posizione(gioco))
        frequenze = np.asarray(frequenze) * 2.0 ** (np.asarray(self.lfo(audio)) * self.mezza_escursione_ottave())
        guadagni = np.asarray(guadagni) * 10.0 ** (BERSAGLIO_DB / 20.0)
        return frequenze, np.asarray(pan, dtype=float), guadagni


def raggio_del_livello(n):
    """Il raggio dei bersagli al livello n: RAGGIO_INIZIALE ridotto a ogni livello, fino a RAGGIO_MINIMO."""
    return max(RAGGIO_MINIMO, RAGGIO_INIZIALE * RIDUZIONE_DEL_RAGGIO ** (n - 1))


def velocita_del_livello(n):
    """La velocità dei bersagli al livello n: fermi prima di LIVELLO_DEI_MOBILI, poi sempre più svelti
    fino a VELOCITA_MASSIMA_BERSAGLI."""
    if n < LIVELLO_DEI_MOBILI:
        return 0.0
    return min(VELOCITA_MASSIMA_BERSAGLI, VELOCITA_INIZIALE_BERSAGLI + AUMENTO_DI_VELOCITA * (n - LIVELLO_DEI_MOBILI))


def quanti_bersagli(n):
    """Quanti bersagli normali volano insieme al livello n."""
    return 1 + sum(1 for livello in LIVELLI_CON_UN_BERSAGLIO_IN_PIU if livello <= n)


def durata_del_bonus(n):
    """I secondi della traversata del bonus al livello n, mai sotto DURATA_BONUS_MINIMA."""
    return max(DURATA_BONUS_MINIMA, DURATA_BONUS_INIZIALE - RIDUZIONE_DURATA_BONUS * (n - LIVELLO_DEI_BONUS))


def _direzione_a_caso(rng):
    """Un versore uniforme sulla sfera: tre gaussiane normalizzate."""
    while True:
        vettore = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
        lunghezza = float(np.linalg.norm(vettore))
        if lunghezza > 1e-9:
            return vettore / lunghezza


def _voce_a_caso(rng):
    """Forma, frequenza, fase e seme dell'LFO, sempre nello stesso ordine di estrazione."""
    forma = rng.choice(FORME_LFO)
    frequenza = rng.uniform(*FREQUENZA_LFO)
    fase = rng.random()
    seme = rng.getrandbits(32)
    return forma, frequenza, fase, seme


def genera(livello, rng, numero, tempo):
    """Un bersaglio normale del livello, nato al tempo di gioco dato: posizione uniforme nella zona
    permessa, direzione del moto uniforme sulla sfera, forma a caso fra FORME_LFO. rng è un
    random.Random; la direzione si estrae anche per i fermi, così la sequenza delle estrazioni non
    dipende dal livello."""
    raggio = raggio_del_livello(livello)
    minimi, massimi = fisica.zona_del_centro(raggio)
    partenza = np.array([rng.uniform(basso, alto) for basso, alto in zip(minimi, massimi, strict=True)])
    velocita = _direzione_a_caso(rng) * velocita_del_livello(livello)
    forma, frequenza, fase, seme = _voce_a_caso(rng)
    return Bersaglio(numero=numero, forma=forma, raggio=raggio, partenza=partenza, velocita=velocita, nato=tempo, bonus=False,
                     frequenza_lfo=frequenza, fase_lfo=fase, seme=seme)


def genera_bonus(livello, rng, numero, tempo):
    """Il bersaglio bonus del livello: raggio ridotto di RAGGIO_BONUS_FATTORE, centro appena fuori
    dalla parete di sinistra o di destra, a caso, altezza e profondità uniformi nella zona permessa;
    attraversa il cubo lungo x in durata_del_bonus(livello) secondi, dal primo contatto all'uscita
    completa."""
    raggio = raggio_del_livello(livello) * RAGGIO_BONUS_FATTORE
    minimi, massimi = fisica.zona_del_centro(raggio)
    da_sinistra = rng.random() < 0.5
    altezza = rng.uniform(minimi[1], massimi[1])
    profondita = rng.uniform(minimi[2], massimi[2])
    rapidita = (CUBO_LATO + 2.0 * raggio) / durata_del_bonus(livello)
    if da_sinistra:
        partenza = np.array((CUBO_X[0] - raggio, altezza, profondita))
        velocita = np.array((rapidita, 0.0, 0.0))
    else:
        partenza = np.array((CUBO_X[1] + raggio, altezza, profondita))
        velocita = np.array((-rapidita, 0.0, 0.0))
    forma, frequenza, fase, seme = _voce_a_caso(rng)
    return Bersaglio(numero=numero, forma=forma, raggio=raggio, partenza=partenza, velocita=velocita, nato=tempo, bonus=True,
                     frequenza_lfo=frequenza, fase_lfo=fase, seme=seme)
