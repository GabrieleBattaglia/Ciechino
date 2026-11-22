"""
I bersagli di Ciechino: dove stanno, come si muovono, come suonano e come cambiano coi livelli.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Niente wx e niente audio, soltanto numpy e la libreria standard. Tutto si calcola in qualunque
istante senza stato e senza simulare: la posizione dal tempo di gioco, l'LFO dal tempo dell'audio.
Così il motore audio può chiedere i parametri di un blocco qualunque, il volo può esaminare tutto
il tragitto in un colpo solo, e fermando il tempo di gioco i bersagli si fermano mentre il loro
suono continua a vibrare.
Da D28 i bersagli stanno nella cupola: nascono in un punto a caso della Zona del loro raggio, e i
mobili vanno dritti e rimbalzano sulle sue pareti, con un fisica.Percorso calcolato a tratti.
Il bonus: nasce col centro appena fuori dal settore, a un raggio dal lato di sinistra o di destra,
cioè già tangente al lato, ed è vivo da subito, dal suo tempo nato; attraversa il settore lungo x, su
una corda a quota e profondità costanti tutta dentro la cupola, in durata_del_bonus(livello) secondi, e
smette di essere vivo a tempo_di_uscita, quando è uscito del tutto dall'altro lato. Prima di nato
nessun bersaglio è vivo, perché non esisteva ancora.
I bersagli sempre un po' diversi (D18, dal banco dei bersagli del 19 novembre 2026): oltre alla forma,
alla frequenza e alla fase dell'LFO, ogni bersaglio estrae a caso i parametri della sua forma, dentro
gli intervalli di costanti.py, il moltiplicatore della sua escursione, in ESCURSIONE_A_CASO, e la sua
portante, fra quelle che chi lo genera gli offre. I valori di serie dei tre campi nuovi di Bersaglio
rendono il bersaglio della 0.1.0: chi lo costruisce senza dirli sente quello di prima.
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple

import numpy as np

import fisica
from costanti import (
    ARMONICA_DEL_SENO,
    AUMENTO_DI_VELOCITA,
    BERSAGLIO_DB,
    BERSAGLIO_FORMA,
    BERSAGLIO_PORTANTI,
    CURVATURA_DELLA_SEGA,
    DURATA_BONUS_INIZIALE,
    DURATA_BONUS_MINIMA,
    ESCURSIONE_A_CASO,
    FATTORE_ESCURSIONE,
    FORME_LFO,
    FREQUENZA_LFO,
    FRONTE_DELLA_QUADRA,
    GRADINI,
    LARGHEZZA_DEGLI_IMPULSI,
    LARGHEZZA_DELLA_QUADRA,
    LIVELLI_CON_UN_BERSAGLIO_IN_PIU,
    LIVELLO_DEI_BONUS,
    LIVELLO_DEI_MOBILI,
    METRI_PER_OTTAVA,
    MORBIDEZZA_DELLA_DERIVA,
    NOTE_DEL_TRILLO,
    OSCILLAZIONI_DELLA_MOLLA,
    PENDENZA_DEL_TRIANGOLO,
    PESO_DELLA_DOPPIA,
    PROFONDITA_DEL_TRILLO,
    RAGGIO_BONUS_FATTORE,
    RAGGIO_INIZIALE,
    RAGGIO_MINIMO,
    RAPPORTI_DELLA_DOPPIA,
    RIDUZIONE_DEL_RAGGIO,
    RIDUZIONE_DURATA_BONUS,
    SETTORE_AZIMUT,
    SMORZAMENTO_DELLA_MOLLA,
    SPINTA_DEL_CASO,
    VALORI_DEL_CASUALE,
    VELOCITA_INIZIALE_BERSAGLI,
    VELOCITA_MASSIMA_BERSAGLI,
    VERSI_DEI_GRADINI,
)
from sintesi import FORME_PORTANTI

# La durata, in secondi, del raccordo di ogni fronte dell'LFO quadro: il progetto vuole i fronti
# smussati di qualche millesimo, perché un salto netto di frequenza della portante si sente come
# un click. Il raccordo è un quarto di seno, centrato sul fronte, e non supera mai un quarto di ciclo.
# Lo stesso raccordo smussa i salti delle forme a scalini, gradini e trillo.
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
# Seno, molla e doppia si dividono per il loro picco, così arrivano a 1: il picco si cerca su una
# griglia di tanti punti per ciclo, che sbaglia di qualche milionesimo, e la doppia lo cerca su tanti
# cicli, quanti bastano a ripetere la forma coi rapporti di RAPPORTI_DELLA_DOPPIA.
_PUNTI_PER_CICLO = 4096
_CICLI_DELLA_DOPPIA = 4


class Parametro(NamedTuple):
    """Un parametro a caso di una forma dell'LFO. tipo dice come si estrae: "reale" uniforme fra i
    due capi di intervallo, "intero" fra i due capi compresi, "scelta" uno degli elementi di
    intervallo. di_serie è il valore di chi non lo dice; limiti sono il più piccolo e il più grande
    valore, compresi, con cui la formula ha ancora senso, più larghi dell'intervallo del gioco."""

    nome: str
    tipo: str
    intervallo: tuple
    di_serie: float
    limiti: tuple


def _mezzo(intervallo):
    return (intervallo[0] + intervallo[1]) / 2.0


def _mezzo_intero(intervallo):
    return (intervallo[0] + intervallo[1]) // 2


_SPINTA = Parametro("spinta", "reale", SPINTA_DEL_CASO, 1.0, (0.1, 10.0))
_CURVATURA = Parametro("curvatura", "reale", CURVATURA_DELLA_SEGA, 1.0, (0.1, 10.0))

# I parametri di ogni forma, nell'ordine in cui si estraggono. Il valore di serie delle sette forme
# della 0.1.0 rende la forma di allora: niente armonica, rampa dritta, triangolo simmetrico, quadra a
# metà coi soli fronti del raccordo, un valore per ciclo senza spinta, deriva sempre in moto. Le forme
# nuove hanno di serie il mezzo del loro intervallo, o la prima scelta.
PARAMETRI_LFO = {
    "seno": (Parametro("armonica", "reale", ARMONICA_DEL_SENO, 0.0, (-1.0, 1.0)),),
    "sega": (_CURVATURA,),
    "sega_inversa": (_CURVATURA,),
    "triangolo": (Parametro("pendenza", "reale", PENDENZA_DEL_TRIANGOLO, 0.5, (0.01, 0.99)),),
    "quadra": (Parametro("larghezza", "reale", LARGHEZZA_DELLA_QUADRA, 0.5, (0.05, 0.95)),
               Parametro("fronte", "reale", FRONTE_DELLA_QUADRA, 0.0, (0.0, 0.45))),
    "casuale": (Parametro("valori", "intero", VALORI_DEL_CASUALE, 1, (1, 64)), _SPINTA),
    "deriva": (Parametro("morbidezza", "reale", MORBIDEZZA_DELLA_DERIVA, 1.0, (0.01, 1.0)), _SPINTA),
    "molla": (Parametro("oscillazioni", "intero", OSCILLAZIONI_DELLA_MOLLA, _mezzo_intero(OSCILLAZIONI_DELLA_MOLLA), (1, 64)),
              Parametro("smorzamento", "reale", SMORZAMENTO_DELLA_MOLLA, _mezzo(SMORZAMENTO_DELLA_MOLLA), (0.0, 50.0))),
    "gradini": (Parametro("gradini", "intero", GRADINI, _mezzo_intero(GRADINI), (2, 64)),
                Parametro("verso", "scelta", VERSI_DEI_GRADINI, VERSI_DEI_GRADINI[0], (-1, 1))),
    "impulsi": (Parametro("larghezza", "reale", LARGHEZZA_DEGLI_IMPULSI, _mezzo(LARGHEZZA_DEGLI_IMPULSI), (0.01, 0.5)),),
    "doppia": (Parametro("rapporto", "scelta", RAPPORTI_DELLA_DOPPIA, RAPPORTI_DELLA_DOPPIA[0], (0.01, 100.0)),
               Parametro("peso", "reale", PESO_DELLA_DOPPIA, _mezzo(PESO_DELLA_DOPPIA), (0.0, 10.0))),
    "trillo": (Parametro("note", "intero", NOTE_DEL_TRILLO, _mezzo_intero(NOTE_DEL_TRILLO), (1, 64)),
               Parametro("profondita", "reale", PROFONDITA_DEL_TRILLO, _mezzo(PROFONDITA_DEL_TRILLO), (0.0, 2.0))),
}


def _descrizioni(forma):
    """I Parametro di una forma, o ValueError se la forma non c'è."""
    try:
        return PARAMETRI_LFO[forma]
    except (KeyError, TypeError):
        raise ValueError(f"Forma dell'LFO sconosciuta: {forma!r}; le forme sono {', '.join(FORME_LFO)}.") from None


def _valore_valido(forma, parametro, valore):
    """Il valore di un parametro, controllato contro i suoi limiti: intero per gli interi e per le
    scelte fatte di interi, altrimenti float."""
    try:
        numero = float(valore)
    except (TypeError, ValueError):
        raise ValueError(f"Il parametro {parametro.nome!r} della forma {forma!r} vuole un numero, non {valore!r}.") from None
    basso, alto = parametro.limiti
    if not math.isfinite(numero) or not basso <= numero <= alto:
        raise ValueError(f"Il parametro {parametro.nome!r} della forma {forma!r} va da {basso} a {alto}, e {valore!r} è fuori.")
    interi = parametro.tipo == "intero" or (parametro.tipo == "scelta" and all(isinstance(x, int) for x in parametro.intervallo))
    if interi:
        if not numero.is_integer():
            raise ValueError(f"Il parametro {parametro.nome!r} della forma {forma!r} vuole un intero, non {valore!r}.")
        return int(numero)
    return numero


def parametri_di_serie(forma):
    """I parametri di serie di una forma, in un dizionario nuovo."""
    return {parametro.nome: parametro.di_serie for parametro in _descrizioni(forma)}


def parametri_completi(forma, parametri=None):
    """I parametri di una forma, tutti: quelli dati, controllati, e per gli altri il valore di serie.
    Una forma sconosciuta, un nome che la forma non ha o un valore fuori dai limiti sollevano
    ValueError."""
    descrizioni = _descrizioni(forma)
    dati = dict(parametri or {})
    sconosciuti = sorted(set(dati) - {parametro.nome for parametro in descrizioni})
    if sconosciuti:
        nomi = ", ".join(parametro.nome for parametro in descrizioni)
        raise ValueError(f"La forma {forma!r} non ha i parametri {', '.join(map(repr, sconosciuti))}; i suoi sono {nomi}.")
    return {parametro.nome: _valore_valido(forma, parametro, dati.get(parametro.nome, parametro.di_serie)) for parametro in descrizioni}


def parametri_a_caso(forma, rng):
    """I parametri di una forma estratti con rng, un random.Random, nell'ordine di PARAMETRI_LFO: i
    reali con uniform, gli interi con randint, le scelte con choice."""
    estratti = {}
    for parametro in _descrizioni(forma):
        if parametro.tipo == "reale":
            estratti[parametro.nome] = rng.uniform(*parametro.intervallo)
        elif parametro.tipo == "intero":
            estratti[parametro.nome] = rng.randint(*parametro.intervallo)
        else:
            estratti[parametro.nome] = rng.choice(parametro.intervallo)
    return parametri_completi(forma, estratti)


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


def _spinto(valori, esponente):
    """I valori fra meno 1 e 1 piegati dall'esponente, col segno tenuto: sotto 1 vanno verso gli
    estremi. È dispari, quindi una distribuzione simmetrica resta simmetrica, con la media a zero."""
    if esponente == 1.0:
        return valori
    return np.sign(valori) * np.abs(valori) ** esponente


@lru_cache(maxsize=256)
def _picco_del_seno(armonica):
    griglia = np.arange(_PUNTI_PER_CICLO) / _PUNTI_PER_CICLO
    return float(np.max(np.abs(np.sin(2.0 * math.pi * griglia) + armonica * np.sin(4.0 * math.pi * griglia))))


def _seno(fasi, armonica):
    """Il seno con la seconda armonica in fase, diviso per il picco: la somma è una funzione dispari,
    quindi il minimo è il meno del massimo, e la media è zero."""
    valori = np.sin(2.0 * math.pi * fasi)
    if armonica:
        valori = (valori + armonica * np.sin(4.0 * math.pi * fasi)) / _picco_del_seno(armonica)
    return valori


def _triangolo(fasi, pendenza):
    """Il triangolo che sale per la parte pendenza del ciclo e scende nel resto, allineato al seno
    quando la pendenza è 0,5: 0 all'inizio del ciclo, 1 a un quarto, meno 1 a tre quarti. Le due
    rampe vanno da meno 1 a 1, quindi la media resta zero con qualunque pendenza."""
    posti = np.mod(fasi + pendenza / 2.0, 1.0)
    return np.where(posti < pendenza, 2.0 * posti / pendenza - 1.0, 1.0 - 2.0 * (posti - pendenza) / (1.0 - pendenza))


def _quadra(frazioni, frequenza, larghezza, fronte):
    """La quadra in alto nella parte larghezza del ciclo, dall'inizio, e in basso nel resto. Ogni
    fronte è un quarto di seno, largo il raccordo contro i click, o la parte fronte del ciclo se è di
    più, mai oltre metà del tratto più corto: così la quadra arriva sempre a 1 e a meno 1."""
    distanze = np.where(frazioni < larghezza, np.minimum(frazioni, larghezza - frazioni), -np.minimum(frazioni - larghezza, 1.0 - frazioni))
    mezzo = min(max(RACCORDO_DELLA_QUADRA * abs(frequenza) / 2.0, fronte / 2.0), min(larghezza, 1.0 - larghezza) / 2.0)
    if mezzo > 0.0:
        return np.sin(math.pi / 2.0 * np.clip(distanze / mezzo, -1.0, 1.0))
    return np.where(frazioni < larghezza, 1.0, -1.0)


def _deriva(seme, cicli, frazioni, morbidezza, spinta):
    """Il valore a caso dell'inizio del ciclo, tenuto fermo, poi raccordato col coseno a quello del
    ciclo seguente nell'ultima parte morbidezza del ciclo."""
    da = _spinto(_caso(seme, cicli), spinta)
    verso = _spinto(_caso(seme, cicli + 1.0), spinta)
    avanzamento = np.clip((frazioni - (1.0 - morbidezza)) / morbidezza, 0.0, 1.0)
    return da + (verso - da) * (1.0 - np.cos(math.pi * avanzamento)) / 2.0


@lru_cache(maxsize=256)
def _picco_della_molla(oscillazioni, smorzamento):
    griglia = np.arange(_PUNTI_PER_CICLO * oscillazioni + 1) / (_PUNTI_PER_CICLO * oscillazioni)
    return float(np.max(np.abs(np.sin(2.0 * math.pi * oscillazioni * griglia) * np.exp(-smorzamento * griglia))))


def _molla(cicli, frazioni, oscillazioni, smorzamento):
    """Un seno di tante oscillazioni intere per ciclo che si smorza, diviso per il primo picco: parte
    da zero e ci torna a fine ciclo, quindi non salta mai. I cicli pari guizzano in su e i dispari in
    giù, così in due cicli tocca tutti e due gli estremi e la media torna a zero."""
    segni = 1.0 - 2.0 * np.mod(cicli, 2.0)
    onda = np.sin(2.0 * math.pi * oscillazioni * frazioni) * np.exp(-smorzamento * frazioni)
    return segni * onda / _picco_della_molla(oscillazioni, smorzamento)


def _scalini(fasi, frequenza, livelli):
    """Un livello per ogni posto del ciclo, tutti lunghi uguali, coi salti fra l'uno e l'altro
    raccordati da un quarto di seno largo RACCORDO_DELLA_QUADRA, mai oltre un posto intero."""
    livelli = np.asarray(livelli, dtype=float)
    quanti = len(livelli)
    posti = fasi * quanti
    confini = np.round(posti)
    distanze = posti - confini
    prima = livelli[np.mod(confini - 1.0, quanti).astype(np.int64)]
    dopo = livelli[np.mod(confini, quanti).astype(np.int64)]
    mezzo = min(RACCORDO_DELLA_QUADRA * abs(frequenza) * quanti / 2.0, 0.5)
    if mezzo > 0.0:
        pesi = (1.0 + np.sin(math.pi / 2.0 * np.clip(distanze / mezzo, -1.0, 1.0))) / 2.0
    else:
        pesi = (distanze >= 0.0).astype(float)
    return prima + (dopo - prima) * pesi


def _livelli_dei_gradini(gradini, verso):
    """I gradini, equidistanti da meno 1 a 1, in salita; in discesa col verso negativo."""
    livelli = np.linspace(-1.0, 1.0, gradini)
    return livelli if verso >= 0 else livelli[::-1]


def _livelli_del_trillo(note, profondita):
    """Le note del trillo: nel primo mezzo ciclo l'estremo alto e la nota vicina, alternati, a
    cominciare dall'estremo; nel secondo lo stesso in basso, rovesciato."""
    alto = np.array([1.0 - profondita * (posto % 2) for posto in range(note)])
    return np.concatenate((alto, -alto))


def _impulsi(frazioni, larghezza):
    """Due gobbe a coseno rialzato per ciclo, larghe la parte larghezza del ciclo: una in su, centrata
    a un quarto, e una in giù, a tre quarti; fra le due il centro, fermo."""
    mezza = larghezza / 2.0

    def gobba(centro):
        distanze = np.abs(frazioni - centro)
        return np.where(distanze < mezza, (1.0 + np.cos(math.pi * np.minimum(distanze / mezza, 1.0))) / 2.0, 0.0)

    return gobba(0.25) - gobba(0.75)


@lru_cache(maxsize=256)
def _picco_della_doppia(rapporto, peso):
    griglia = np.arange(_PUNTI_PER_CICLO * _CICLI_DELLA_DOPPIA) / _PUNTI_PER_CICLO
    return float(np.max(np.abs(np.sin(2.0 * math.pi * griglia) + peso * np.sin(2.0 * math.pi * rapporto * griglia))))


def _doppia(fasi, rapporto, peso):
    """Due seni insieme, il secondo rapporto volte più svelto e pesato, divisi per il picco della
    somma: è una funzione dispari, quindi il minimo è il meno del massimo, e la media è zero."""
    return (np.sin(2.0 * math.pi * fasi) + peso * np.sin(2.0 * math.pi * rapporto * fasi)) / _picco_della_doppia(rapporto, peso)


def lfo(forma, tempi, frequenza, fase=0.0, seme=0, parametri=None):
    """L'LFO di un bersaglio, fra meno 1 e 1, nei tempi dati in secondi. La fase è una frazione di
    ciclo, da 0 a 1; il seme conta solo per casuale e deriva; parametri sono quelli della forma, in un
    dizionario, e quelli che mancano, o tutti se è None, valgono il loro valore di serie.
    Le forme sono quelle di FORME_LFO, coi parametri di PARAMETRI_LFO:
    seno, con la seconda armonica di ampiezza armonica;
    sega, che sale e ricade, e sega_inversa, che scende e risale, con la rampa piegata da curvatura;
    triangolo, che sale nella parte pendenza del ciclo;
    quadra, in alto nella parte larghezza del ciclo, coi fronti raccordati in RACCORDO_DELLA_QUADRA o
    lunghi fronte;
    casuale, tanti valori a caso per ciclo quanti dice valori, ciascuno tenuto fermo;
    deriva, un valore a caso a ogni inizio di ciclo, raccordato col coseno al seguente nell'ultima
    parte morbidezza del ciclo; casuale e deriva piegano i loro valori con spinta, e deriva passa
    all'inizio di ogni ciclo dallo stesso valore che casuale, con un valore per ciclo e la stessa
    spinta, tiene in quel ciclo;
    molla, un guizzo di oscillazioni intere che si smorza di smorzamento, in su nei cicli pari e in
    giù nei dispari;
    gradini, una scala di gradini livelli da meno 1 a 1, in salita col verso 1 e in discesa col
    verso meno 1, coi salti raccordati;
    impulsi, una gobba in su a un quarto del ciclo e una in giù a tre quarti, larghe larghezza;
    doppia, un seno più un secondo seno rapporto volte più svelto, di peso peso;
    trillo, note note alternate fra l'estremo alto e una nota più bassa di profondità nel primo mezzo
    ciclo, e lo stesso in basso nel secondo, coi salti raccordati.
    Ogni forma tocca gli estremi, o ci arriva vicino, in uno o pochi cicli, e casuale e deriva in
    qualche decina; la media nel tempo è vicina a zero. Scalare in ingresso, float in uscita."""
    return _lfo(forma, tempi, frequenza, fase, seme, parametri_completi(forma, parametri))


def _lfo(forma, tempi, frequenza, fase, seme, parametri):
    """lfo coi parametri già completi e controllati, come quelli di un Bersaglio."""
    frequenza = float(frequenza)
    fasi = np.asarray(tempi, dtype=float) * frequenza + float(fase)
    cicli = np.floor(fasi)
    frazioni = fasi - cicli
    if forma == "seno":
        valori = _seno(fasi, parametri["armonica"])
    elif forma == "sega":
        valori = _spinto(2.0 * frazioni - 1.0, parametri["curvatura"])
    elif forma == "sega_inversa":
        valori = -_spinto(2.0 * frazioni - 1.0, parametri["curvatura"])
    elif forma == "triangolo":
        valori = _triangolo(fasi, parametri["pendenza"])
    elif forma == "quadra":
        valori = _quadra(frazioni, frequenza, parametri["larghezza"], parametri["fronte"])
    elif forma == "casuale":
        valori = _spinto(_caso(seme, np.floor(fasi * parametri["valori"])), parametri["spinta"])
    elif forma == "deriva":
        valori = _deriva(seme, cicli, frazioni, parametri["morbidezza"], parametri["spinta"])
    elif forma == "molla":
        valori = _molla(cicli, frazioni, parametri["oscillazioni"], parametri["smorzamento"])
    elif forma == "gradini":
        valori = _scalini(fasi, frequenza, _livelli_dei_gradini(parametri["gradini"], parametri["verso"]))
    elif forma == "impulsi":
        valori = _impulsi(frazioni, parametri["larghezza"])
    elif forma == "doppia":
        valori = _doppia(fasi, parametri["rapporto"], parametri["peso"])
    elif forma == "trillo":
        valori = _scalini(fasi, frequenza, _livelli_del_trillo(parametri["note"], parametri["profondita"]))
    else:
        raise ValueError(f"Forma dell'LFO sconosciuta: {forma!r}; le forme sono {', '.join(FORME_LFO)}.")
    return fisica.scalare_o_array(np.clip(valori, -1.0, 1.0))


def _portante_valida(portante):
    if portante not in FORME_PORTANTI:
        raise ValueError(f"Portante sconosciuta: {portante!r}; le portanti sono {', '.join(FORME_PORTANTI)}.")
    return portante


def portanti_senza(forma, portanti=BERSAGLIO_PORTANTI):
    """Le portanti per i bersagli, tolta quella del proiettile, perché non si confondano (D18)."""
    return tuple(portante for portante in portanti if portante != forma)


@dataclass(eq=False)
class Bersaglio:
    """Un bersaglio in volo. Si confronta per identità, non per contenuto: due bersagli con gli
    stessi numeri restano due bersagli, e il volo dice quale ha colpito.
    I campi di D18 hanno un valore di serie, quello della 0.1.0: portante, la forma d'onda della sua
    voce, fra sintesi.FORME_PORTANTI, di serie BERSAGLIO_FORMA; escursione, il moltiplicatore della
    sua oscillazione, di serie 1; parametri_lfo, i parametri della sua forma, un dizionario che
    __post_init__ completa coi valori di serie e controlla.
    La zona e, per il mobile, il percorso coi rimbalzi (D28) nascono con lui, da raggio, partenza e
    velocità: chi cambia quei campi dopo ha il bersaglio di prima."""

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
    portante: str = BERSAGLIO_FORMA
    escursione: float = 1.0
    parametri_lfo: dict = field(default_factory=dict)

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
        self.portante = _portante_valida(self.portante)
        self.escursione = float(self.escursione)
        if not math.isfinite(self.escursione) or self.escursione < 0.0:
            raise ValueError(f"L'escursione di un bersaglio è un moltiplicatore positivo, non {self.escursione!r}.")
        self.parametri_lfo = parametri_completi(self.forma, self.parametri_lfo)
        self._zona = fisica.Zona(self.raggio)
        self._percorso = fisica.Percorso(self._zona, self.partenza, self.velocita) if self.mobile and not self.bonus else None

    @property
    def velocita_scalare(self):
        """La velocità in metri al secondo, cioè il modulo del vettore velocità."""
        return float(np.linalg.norm(self.velocita))

    @property
    def mobile(self):
        return self.velocita_scalare > 0.0

    @property
    def zona(self):
        """La fisica.Zona dove può stare il suo centro, quella dei rimbalzi (D28)."""
        return self._zona

    @property
    def tempo_di_uscita(self):
        """Per il bonus il tempo di gioco in cui è uscito del tutto dal settore, dal lato verso cui va,
        cioè quando il suo centro sta a un raggio oltre quel lato; per gli altri None. Un bonus fermo, o
        che non va verso nessun lato, non esce mai: infinito."""
        if not self.bonus:
            return None
        for normale in (self._zona.normale_destra, self._zona.normale_sinistra):
            verso = float(normale @ self.velocita)
            if verso > 0.0:
                return self.nato + max(self.raggio - float(normale @ self.partenza), 0.0) / verso
        return math.inf

    def posizione(self, t):
        """La posizione del centro al tempo di gioco t: (3,) per t scalare, (N, 3) per un array.
        Il fermo sta dov'è nato; il mobile va dritto e rimbalza sulle pareti della sua zona, col suo
        percorso; il bonus va dritto e basta, perché entra ed esce dal settore."""
        tempi = np.asarray(t, dtype=float)
        if self._percorso is not None:
            return self._percorso.posizione(tempi - self.nato)
        return self.partenza + self.velocita * (tempi - self.nato)[..., None]

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
        """Il suo LFO, fra meno 1 e 1, nei tempi dell'audio, coi parametri della sua forma."""
        return _lfo(self.forma, t_audio, self.frequenza_lfo, self.fase_lfo, self.seme, self.parametri_lfo)

    def mezza_escursione_ottave(self):
        """Di quante ottave l'LFO sposta la frequenza del centro, in su e in giù: il raggio per
        FATTORE_ESCURSIONE e per la sua escursione, in ottave, così l'oscillazione copre le altezze
        del bersaglio, un po' di più o un po' di meno secondo il caso."""
        return mezza_escursione_ottave(self.raggio, self.escursione)

    def parametri_sonori(self, tempi_di_gioco, tempi_audio):
        """(frequenze, pan, guadagni lineari) per la voce del motore, tre array lunghi come i tempi:
        la posizione dai tempi di gioco, l'LFO dai tempi dell'audio, BERSAGLIO_DB compreso. Uno dei
        due tempi può essere scalare, per esempio il tempo di gioco fermo durante la mira."""
        gioco, audio = np.broadcast_arrays(np.atleast_1d(np.asarray(tempi_di_gioco, dtype=float)), np.atleast_1d(np.asarray(tempi_audio, dtype=float)))
        return self.suono_in(self.posizione(gioco), audio)

    def suono_in(self, posizioni, tempi_audio, raggio=None):
        """(frequenze, pan, guadagni lineari) del bersaglio nelle posizioni date, (N, 3), invece che in
        quella del suo moto: le stesse regole di parametri_sonori, l'LFO dai tempi dell'audio e
        BERSAGLIO_DB compreso. raggio, se c'è, prende il posto del suo, anche un array di N raggi, per
        un bersaglio che cambia grandezza. Serve agli Ascolti (D23), dove il bersaglio segue un
        cammino che il gioco non conosce."""
        mezza = self.mezza_escursione_ottave() if raggio is None else mezza_escursione_ottave(np.asarray(raggio, dtype=float), self.escursione)
        frequenze, pan, guadagni = fisica.suono_di(posizioni)
        frequenze = np.asarray(frequenze) * 2.0 ** (np.asarray(self.lfo(tempi_audio)) * mezza)
        guadagni = np.asarray(guadagni) * 10.0 ** (BERSAGLIO_DB / 20.0)
        return frequenze, np.asarray(pan, dtype=float), guadagni


def mezza_escursione_ottave(raggio, escursione=1.0):
    """Di quante ottave l'LFO di un bersaglio di quel raggio sposta la frequenza del centro, in su e in
    giù: il raggio per FATTORE_ESCURSIONE e per l'escursione, diviso METRI_PER_OTTAVA. Il raggio può
    essere un array, per un bersaglio che cresce, come negli Ascolti (D23)."""
    return raggio * FATTORE_ESCURSIONE * escursione / METRI_PER_OTTAVA


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


def _portanti_valide(portanti):
    """Le portanti fra cui estrarre, controllate prima di ogni estrazione: un elenco vuoto o una
    portante sconosciuta sollevano ValueError, e rng resta com'era."""
    portanti = tuple(portanti)
    if not portanti:
        raise ValueError("Non c'è nessuna portante fra cui scegliere quella del bersaglio.")
    for portante in portanti:
        _portante_valida(portante)
    return portanti


def _voce_a_caso(rng, portanti):
    """Il suono di un bersaglio, sempre nello stesso ordine di estrazione: la forma dell'LFO, la sua
    frequenza, la fase e il seme; poi i parametri della forma, nell'ordine di PARAMETRI_LFO; poi il
    moltiplicatore dell'escursione; per ultima la portante. Restituisce gli argomenti per Bersaglio."""
    forma = rng.choice(FORME_LFO)
    frequenza = rng.uniform(*FREQUENZA_LFO)
    fase = rng.random()
    seme = rng.getrandbits(32)
    parametri = parametri_a_caso(forma, rng)
    escursione = rng.uniform(*ESCURSIONE_A_CASO)
    portante = rng.choice(portanti)
    return {"forma": forma, "frequenza_lfo": frequenza, "fase_lfo": fase, "seme": seme, "parametri_lfo": parametri, "escursione": escursione,
            "portante": portante}


def genera(livello, rng, numero, tempo, portanti=BERSAGLIO_PORTANTI):
    """Un bersaglio normale del livello, nato al tempo di gioco dato. rng è un random.Random, e il
    bersaglio dipende soltanto da lui: stesso seme, stessi bersagli. Le estrazioni, sempre in
    quest'ordine: la partenza, con fisica.Zona.punto_a_caso, distanza, azimut ed elevazione, ripetute
    finché il punto non sta nella zona del suo raggio (D28); la
    direzione del moto, tre gaussiane, uniforme sulla sfera, estratta anche per i fermi, così la
    sequenza non dipende dal livello; la forma dell'LFO fra FORME_LFO, la sua frequenza in
    FREQUENZA_LFO, la fase e il seme; i parametri della forma, nell'ordine di PARAMETRI_LFO; il
    moltiplicatore dell'escursione in ESCURSIONE_A_CASO; la portante fra portanti (D18). Un elenco di
    portanti vuoto, o con una portante sconosciuta, solleva ValueError prima di ogni estrazione."""
    portanti = _portanti_valide(portanti)
    raggio = raggio_del_livello(livello)
    partenza = fisica.Zona(raggio).punto_a_caso(rng)
    velocita = _direzione_a_caso(rng) * velocita_del_livello(livello)
    return Bersaglio(numero=numero, raggio=raggio, partenza=partenza, velocita=velocita, nato=tempo, bonus=False, **_voce_a_caso(rng, portanti))


def genera_bonus(livello, rng, numero, tempo, portanti=BERSAGLIO_PORTANTI):
    """Il bersaglio bonus del livello: raggio ridotto di RAGGIO_BONUS_FATTORE, centro appena fuori dal
    lato di sinistra o di destra, a caso, su una corda lungo x a quota e profondità costanti che dentro
    il settore resta tutta nella zona del suo raggio (D28); attraversa il settore in
    durata_del_bonus(livello) secondi, dal primo contatto col lato all'uscita completa dall'altro. Le
    estrazioni, sempre in quest'ordine: il lato, poi l'altezza, fra il pavimento della zona e il suo
    raggio esterno, e la profondità, fino al raggio esterno, ripetute finché la corda non sta nella
    zona, poi il suono come in genera, dalla forma dell'LFO alla portante fra portanti. Un elenco di
    portanti vuoto, o con una portante sconosciuta, solleva ValueError prima di ogni estrazione."""
    portanti = _portanti_valide(portanti)
    raggio = raggio_del_livello(livello) * RAGGIO_BONUS_FATTORE
    zona = fisica.Zona(raggio)
    seno, coseno = math.sin(math.radians(SETTORE_AZIMUT)), math.cos(math.radians(SETTORE_AZIMUT))
    da_sinistra = rng.random() < 0.5
    while True:
        altezza = rng.uniform(zona.pavimento, zona.esterno)
        profondita = rng.uniform(0.0, zona.esterno)
        # Dentro il settore la corda va da meno bordo a più bordo: lì sta a un raggio dai lati.
        bordo = (profondita * seno - raggio) / coseno
        if bordo > 0.0 and zona.contiene((0.0, altezza, profondita), tolleranza=0.0) and zona.contiene((bordo, altezza, profondita), tolleranza=0.0):
            break
    # Il centro parte e arriva a un raggio fuori dai lati: la corda intera è lunga tanto.
    fuori = (profondita * seno + raggio) / coseno
    rapidita = 2.0 * fuori / durata_del_bonus(livello)
    if da_sinistra:
        partenza = np.array((-fuori, altezza, profondita))
        velocita = np.array((rapidita, 0.0, 0.0))
    else:
        partenza = np.array((fuori, altezza, profondita))
        velocita = np.array((-rapidita, 0.0, 0.0))
    return Bersaglio(numero=numero, raggio=raggio, partenza=partenza, velocita=velocita, nato=tempo, bonus=True, **_voce_a_caso(rng, portanti))
