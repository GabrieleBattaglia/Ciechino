"""
La fisica di Ciechino: il tiro, la traiettoria del proiettile e il modo in cui un punto dello spazio diventa suono.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Niente wx e niente audio, soltanto numpy: la usano il gioco, il motore audio dentro il callback e
le prove. Ogni funzione di mappatura accetta uno scalare o un array numpy e restituisce la
stessa forma, uno scalare come float di Python e un array come array: così il motore chiede in un
colpo solo i parametri di tutti i campioni di un blocco, e chi lavora con un punto solo non si
porta dietro array di dimensione zero.
Gli assi sono quelli di costanti.py: x verso destra, y verso l'alto, z in avanti; il giocatore sta
nell'origine e il cannone in (0, ALTEZZA_CANNONE, 0). Gli angoli si danno in gradi, le distanze in
metri, le velocità in metri al secondo.
Da D28 lo spazio dei bersagli è la cupola: un guscio di sfera centrato nell'origine, a terra sotto il
cannone, nel settore di ±SETTORE_AZIMUT gradi, sopra ALTEZZA_MINIMA_BERSAGLIO e sotto il cono di
ELEVAZIONE_MASSIMA attorno alla verticale. La classe Zona ne dice la parte dove può stare il centro di
un bersaglio di un certo raggio, coi suoi rimbalzi, e la scala della potenza parte, per ogni alzo, dalla
velocità minima che porta il proiettile nella cupola: così ogni combinazione dei tre valori la
attraversa.
"""

import math
import threading
from functools import lru_cache

import numpy as np

from costanti import (
    ALTEZZA_CANNONE,
    ALTEZZA_MINIMA_BERSAGLIO,
    ALTEZZA_TETTO_DEL_SUONO,
    ALZO_MASSIMO,
    ALZO_MINIMO,
    AZIMUT_DEL_PAN,
    AZIMUT_MASSIMO,
    CUPOLA_RAGGIO,
    CUPOLA_RAGGIO_INTERNO,
    DB_PER_METRO,
    ELEVAZIONE_MASSIMA,
    ESPONENTE_POTENZA,
    FREQUENZA_AL_SUOLO,
    GRAVITA,
    GUADAGNO_DB_MINIMO,
    GUADAGNO_DB_VICINO,
    MARGINE_DI_COPERTURA,
    METRI_PER_OTTAVA,
    SETTORE_AZIMUT,
    VELOCITA_MASSIMA,
)

# Il passo, in gradi, con cui soluzione scandisce l'alzo in cerca del tiro più dolce. Non è una
# taratura del gioco ma la finezza di una ricerca: la velocità di ogni alzo viene da una formula
# chiusa, quindi il tiro trovato passa esattamente per il punto qualunque sia il passo, e il passo
# decide soltanto quanto la potenza scelta si avvicina alla minima possibile.
PASSO_DELL_ALZO_NELLA_SOLUZIONE = 0.02
# La tabella delle velocità minime: tanti punti per grado d'alzo, fra i quali velocita_minima
# interpola; la funzione è liscia, e fra un punto e l'altro l'errore è di millesimi di m/s. Ogni
# velocità della tabella si cerca con una bisezione di tanti passi, provando la traiettoria in tanti
# campioni. Sono finezze di calcolo, non tarature del gioco.
PUNTI_PER_GRADO_DELLE_MINIME = 4
PASSI_DELLA_BISEZIONE = 40
CAMPIONI_DELLA_TRAIETTORIA = 2000
# Il margine aggiunto a ogni velocità minima: la bisezione la trova per eccesso, e il margine tiene
# dentro la cupola anche il tiro allo 0% della potenza fra due punti della tabella, anche dove la curva
# delle minime piega, verso i 72 gradi, e l'interpolazione passerebbe sotto di un centesimo.
MARGINE_DELLA_MINIMA = 0.03
# Il passo, in metri, con cui si campionano gli spigoli della cupola per il punto più lontano da un
# bersaglio, quello della precisione globale (D26).
PASSO_DEGLI_SPIGOLI = 0.25
# Le tolleranze della geometria della zona: quanto un punto può stare oltre una parete ed essere
# ancora dentro, e il tempo più breve dopo cui un rimbalzo vale come nuovo. Contro gli arrotondamenti.
TOLLERANZA_DELLA_ZONA = 1e-6
TEMPO_MINIMO_DEL_RIMBALZO = 1e-9
# Al più tanti tratti per il percorso di un bersaglio: centinaia di ore di gioco ai rimbalzi più fitti.
TRATTI_MASSIMI = 200000
# Il primo orizzonte, in secondi di vita, fin dove si calcolano i tratti di un percorso quando nasce.
ORIZZONTE_DEL_PERCORSO = 60.0

# Il cannone, da dove parte ogni tiro e da dove si misurano distanze e direzioni del suono.
CANNONE = (0.0, ALTEZZA_CANNONE, 0.0)


def scalare_o_array(valore):
    """Uno scalare, anche un array numpy di dimensione zero, diventa un float di Python; un array
    resta com'è. È la regola di uscita di tutte le funzioni di mappatura."""
    if np.ndim(valore) == 0:
        return float(valore)
    return valore


def _array(valore):
    return np.asarray(valore, dtype=float)


def _potenza_con_segno(base, esponente):
    """La potenza che conserva il segno della base: con un esponente frazionario un numero negativo
    darebbe nan, e l'inversa del tiro deve poter dire di quanto un tiro esce dalla scala."""
    return np.sign(base) * np.abs(base) ** esponente


def _passa_nella_cupola(alzi_gradi, velocita):
    """Vero, coppia per coppia, se il tiro a quell'alzo e a quella velocità passa per un punto della
    cupola dove può stare il centro di un bersaglio, cioè della Zona di raggio zero. La direzione non
    conta: la sua scala copre esattamente il settore. Il tiro resta nel suo piano verticale, dove la
    distanza dal centro della cupola è la radice di s al quadrato più y al quadrato, con s la distanza
    orizzontale, e l'elevazione vista dal centro è l'arcotangente di y su s."""
    alzi, velocita = np.broadcast_arrays(np.radians(_array(alzi_gradi)), _array(velocita))
    verticali = velocita * np.sin(alzi)
    a_terra = (verticali + np.sqrt(verticali * verticali + 2.0 * GRAVITA * ALTEZZA_CANNONE)) / GRAVITA
    tempi = a_terra[..., None] * np.linspace(0.0, 1.0, CAMPIONI_DELLA_TRAIETTORIA)
    distanze = (velocita * np.cos(alzi))[..., None] * tempi
    altezze = ALTEZZA_CANNONE + verticali[..., None] * tempi - 0.5 * GRAVITA * tempi * tempi
    raggi = np.hypot(distanze, altezze)
    elevazioni = np.degrees(np.arctan2(altezze, distanze))
    dentro = (raggi >= CUPOLA_RAGGIO_INTERNO) & (raggi <= CUPOLA_RAGGIO) & (altezze >= ALTEZZA_MINIMA_BERSAGLIO) & (elevazioni <= ELEVAZIONE_MASSIMA)
    return dentro.any(axis=-1)


@lru_cache(maxsize=1)
def tabella_delle_minime():
    """(alzi in gradi, velocità minime), la tabella di velocita_minima: per ogni alzo, da ALZO_MINIMO
    ad ALZO_MASSIMO, la velocità più bassa con cui il tiro passa nella cupola, trovata per bisezione,
    più MARGINE_DELLA_MINIMA. Si calcola una volta sola, alla prima chiamata. Solleva ValueError se a
    qualche alzo nemmeno VELOCITA_MASSIMA ci arriva: le costanti della cupola e dell'alzo non vanno
    d'accordo."""
    quanti = round((ALZO_MASSIMO - ALZO_MINIMO) * PUNTI_PER_GRADO_DELLE_MINIME) + 1
    alzi = np.linspace(ALZO_MINIMO, ALZO_MASSIMO, quanti)
    alte = np.full(quanti, VELOCITA_MASSIMA)
    if not _passa_nella_cupola(alzi, alte).all():
        raise ValueError("A qualche alzo nemmeno la velocità massima porta il proiettile nella cupola.")
    basse = np.zeros(quanti)
    for _passo in range(PASSI_DELLA_BISEZIONE):
        medie = (basse + alte) / 2.0
        arriva = _passa_nella_cupola(alzi, medie)
        alte = np.where(arriva, medie, alte)
        basse = np.where(arriva, basse, medie)
    alzi.flags.writeable = False
    minime = np.minimum(alte + MARGINE_DELLA_MINIMA, VELOCITA_MASSIMA)
    minime.flags.writeable = False
    return alzi, minime


def velocita_minima(alzo_gradi):
    """La velocità più bassa con cui un tiro a quell'alzo passa nella cupola (D28): lo 0% della scala
    della potenza a quell'alzo. Interpola la tabella; fuori dalla scala dell'alzo vale quella del capo
    più vicino. Uno scalare per un alzo scalare, un array per un array."""
    alzi, minime = tabella_delle_minime()
    return scalare_o_array(np.interp(_array(alzo_gradi), alzi, minime))


def parametri_del_tiro(direzione, alzo, potenza):
    """I tre valori della mira, da 0 a 1, diventano (azimut in gradi, alzo in gradi, velocità alla
    bocca in metri al secondo). L'azimut è positivo a destra; la potenza va dalla velocità minima di
    quell'alzo, velocita_minima, a VELOCITA_MASSIMA, e passa dall'esponente di costanti.py, che con 1
    tiene la scala lineare (D28)."""
    direzione, alzo, potenza = _array(direzione), _array(alzo), _array(potenza)
    azimut_gradi = (2.0 * direzione - 1.0) * AZIMUT_MASSIMO
    alzo_gradi = ALZO_MINIMO + alzo * (ALZO_MASSIMO - ALZO_MINIMO)
    minima = _array(velocita_minima(alzo_gradi))
    velocita = minima + _potenza_con_segno(potenza, ESPONENTE_POTENZA) * (VELOCITA_MASSIMA - minima)
    return scalare_o_array(azimut_gradi), scalare_o_array(alzo_gradi), scalare_o_array(velocita)


def valori_del_tiro(azimut_gradi, alzo_gradi, velocita):
    """L'inversa di parametri_del_tiro: i tre valori da 0 a 1 che danno quel tiro. Non taglia nulla,
    così un valore fuori da 0 e 1 dice che il tiro non si può scegliere con la mira."""
    azimut_gradi, alzo_gradi, velocita = _array(azimut_gradi), _array(alzo_gradi), _array(velocita)
    direzione = (azimut_gradi / AZIMUT_MASSIMO + 1.0) / 2.0
    alzo = (alzo_gradi - ALZO_MINIMO) / (ALZO_MASSIMO - ALZO_MINIMO)
    minima = _array(velocita_minima(alzo_gradi))
    frazione = (velocita - minima) / (VELOCITA_MASSIMA - minima)
    potenza = _potenza_con_segno(frazione, 1.0 / ESPONENTE_POTENZA)
    return scalare_o_array(direzione), scalare_o_array(alzo), scalare_o_array(potenza)


class Traiettoria:
    """La parabola del proiettile, esatta: gravità costante e niente aria, in funzione del tempo di
    volo contato dallo sparo. Si calcola in qualunque istante senza simulare nulla, così il volo, la
    voce del proiettile e i resoconti vedono la stessa curva, posizione e inclinazione comprese."""

    def __init__(self, azimut_gradi, alzo_gradi, velocita, origine=None):
        self.azimut = float(azimut_gradi)
        self.alzo = float(alzo_gradi)
        self.velocita = float(velocita)
        self.origine = np.array(CANNONE if origine is None else origine, dtype=float)
        azimut = math.radians(self.azimut)
        alzo = math.radians(self.alzo)
        orizzontale = self.velocita * math.cos(alzo)
        # Il vettore velocità alla bocca: l'azimut piega la componente orizzontale verso destra.
        self.iniziale = np.array((orizzontale * math.sin(azimut), self.velocita * math.sin(alzo), orizzontale * math.cos(azimut)))
        self._accelerazione = np.array((0.0, -GRAVITA, 0.0))

    def __repr__(self):
        return f"Traiettoria(azimut={self.azimut:.3f}, alzo={self.alzo:.3f}, velocita={self.velocita:.3f}, origine={self.origine.tolist()})"

    def posizione(self, t):
        """La posizione al tempo di volo t: un array (3,) per t scalare, (N, 3) per t array di N."""
        tempi = _array(t)[..., None]
        return self.origine + self.iniziale * tempi + 0.5 * self._accelerazione * tempi * tempi

    def inclinazione(self, t):
        """L'inclinazione del proiettile al tempo di volo t: l'angolo della sua velocità
        sull'orizzontale, in gradi, positivo in salita, zero nell'apice e negativo in discesa (D21).
        Uno scalare per t scalare, un array per t array."""
        orizzontale = math.hypot(float(self.iniziale[0]), float(self.iniziale[2]))
        verticale = float(self.iniziale[1]) - GRAVITA * _array(t)
        return scalare_o_array(np.degrees(np.arctan2(verticale, orizzontale)))

    def tempo_a_terra(self):
        """Il primo tempo positivo in cui y torna a zero, dalla formula chiusa della parabola: la
        radice positiva di origine_y + vy t - g t al quadrato / 2."""
        partenza_y = float(self.origine[1])
        verticale = float(self.iniziale[1])
        discriminante = max(verticale * verticale + 2.0 * GRAVITA * partenza_y, 0.0)
        return max((verticale + math.sqrt(discriminante)) / GRAVITA, 0.0)

    def apice(self):
        """Il punto più alto, come (tempo di volo, altezza). Un tiro che non sale ha l'apice alla
        partenza."""
        verticale = float(self.iniziale[1])
        if verticale <= 0.0:
            return 0.0, float(self.origine[1])
        return verticale / GRAVITA, float(self.origine[1]) + verticale * verticale / (2.0 * GRAVITA)

    def punto_a_terra(self):
        """Il punto dove il proiettile tocca terra, con y esattamente zero."""
        punto = self.posizione(self.tempo_a_terra())
        punto[1] = 0.0
        return punto


def punto_della_cupola(distanza, azimut_gradi, elevazione_gradi):
    """Il punto della cupola a quella distanza dal suo centro, nell'origine, con quell'azimut, positivo
    a destra, e quell'elevazione sull'orizzonte, viste dal centro: (3,) per scalari, (N, 3) per array."""
    distanza, azimut, elevazione = np.broadcast_arrays(_array(distanza), np.radians(_array(azimut_gradi)), np.radians(_array(elevazione_gradi)))
    orizzontale = distanza * np.cos(elevazione)
    return np.stack((orizzontale * np.sin(azimut), distanza * np.sin(elevazione), orizzontale * np.cos(azimut)), axis=-1)


# Il mezzo della cupola, il punto d'esempio del proiettile e degli Ascolti: a metà fra le due sfere,
# 55 metri, davanti, a 30 gradi d'elevazione, cioè 27,5 metri d'altezza.
MEZZO_DELLA_CUPOLA = tuple(float(c) for c in punto_della_cupola((CUPOLA_RAGGIO_INTERNO + CUPOLA_RAGGIO) / 2.0, 0.0, 30.0))


class Zona:
    """La parte della cupola dove può stare il centro di un bersaglio di quel raggio, con tutto il
    bersaglio dentro (D28): a distanza dal centro della cupola fra CUPOLA_RAGGIO_INTERNO più il raggio e
    CUPOLA_RAGGIO meno il raggio; almeno a un raggio dai due piani verticali dei lati del settore; con
    il centro almeno ad ALTEZZA_MINIMA_BERSAGLIO, e almeno a un raggio da terra; sotto il cono di
    ELEVAZIONE_MASSIMA, ristretto dell'angolo che il raggio sottende alla distanza più corta. Le pareti
    hanno ognuna la sua normale uscente; uscita dice quando un moto rettilineo ne passa una, e
    rimbalzo riflette la velocità su quella parete. Raggio zero è la zona di tutta la cupola, quella
    della scala della potenza e della precisione globale."""

    PARETI = ("esterna", "interna", "pavimento", "destra", "sinistra", "cono")

    def __init__(self, raggio=0.0):
        self.raggio = max(float(raggio), 0.0)
        self.interno = CUPOLA_RAGGIO_INTERNO + self.raggio
        self.esterno = CUPOLA_RAGGIO - self.raggio
        self.pavimento = max(ALTEZZA_MINIMA_BERSAGLIO, self.raggio)
        settore = math.radians(SETTORE_AZIMUT)
        # Le normali uscenti dei due lati, piani verticali per il centro: dentro, n · p ≤ -raggio.
        self.normale_destra = np.array((math.cos(settore), 0.0, -math.sin(settore)))
        self.normale_sinistra = np.array((-math.cos(settore), 0.0, -math.sin(settore)))
        self.elevazione = ELEVAZIONE_MASSIMA - math.degrees(math.asin(min(1.0, self.raggio / self.interno)))
        self._pendenza = math.tan(math.radians(self.elevazione))

    def __repr__(self):
        return f"Zona(raggio={self.raggio})"

    def scarti(self, posizioni):
        """Quanto il punto sta oltre ogni parete, nell'ordine di PARETI: negativo dentro, positivo fuori.
        Per il cono lo scarto è y meno la pendenza per la distanza orizzontale. Un array (..., 6)."""
        p = _array(posizioni)
        raggi = np.sqrt(np.sum(p * p, axis=-1))
        orizzontali = np.hypot(p[..., 0], p[..., 2])
        return np.stack((raggi - self.esterno, self.interno - raggi, self.pavimento - p[..., 1], p @ self.normale_destra + self.raggio,
                         p @ self.normale_sinistra + self.raggio, p[..., 1] - self._pendenza * orizzontali), axis=-1)

    def contiene(self, posizioni, tolleranza=TOLLERANZA_DELLA_ZONA):
        """Vero se il punto sta nella zona, a meno della tolleranza: un booleano per un punto (3,), un
        array di N per posizioni (N, 3)."""
        dentro = np.all(self.scarti(posizioni) <= tolleranza, axis=-1)
        return bool(dentro) if np.ndim(dentro) == 0 else dentro

    def normale(self, parete, punto):
        """La normale uscente, di lunghezza uno, della parete nel punto."""
        p = _array(punto)
        if parete == "esterna":
            return p / np.linalg.norm(p)
        if parete == "interna":
            return -p / np.linalg.norm(p)
        if parete == "pavimento":
            return np.array((0.0, -1.0, 0.0))
        if parete == "destra":
            return self.normale_destra.copy()
        if parete == "sinistra":
            return self.normale_sinistra.copy()
        orizzontale = math.hypot(float(p[0]), float(p[2]))
        if orizzontale <= 0.0:
            return np.array((0.0, 1.0, 0.0))
        gradiente = np.array((-self._pendenza * p[0] / orizzontale, 1.0, -self._pendenza * p[2] / orizzontale))
        return gradiente / np.linalg.norm(gradiente)

    def _tempi_sulle_pareti(self, p, v):
        """I tempi positivi in cui il moto p + v t tocca ogni parete andando verso fuori: coppie
        (tempo, parete), non ancora controllate contro le altre pareti."""
        candidati = []
        a = float(v @ v)
        b = float(p @ v)
        for parete, raggio, segno in (("esterna", self.esterno, 1.0), ("interna", self.interno, -1.0)):
            discriminante = b * b - a * (float(p @ p) - raggio * raggio)
            if discriminante >= 0.0:
                candidati.append(((-b + segno * math.sqrt(discriminante)) / a, parete))
        if v[1] < 0.0:
            candidati.append(((self.pavimento - p[1]) / v[1], "pavimento"))
        for parete, normale in (("destra", self.normale_destra), ("sinistra", self.normale_sinistra)):
            verso = float(normale @ v)
            if verso > 0.0:
                candidati.append(((-self.raggio - float(normale @ p)) / verso, parete))
        k2 = self._pendenza * self._pendenza
        qa = v[1] * v[1] - k2 * (v[0] * v[0] + v[2] * v[2])
        qb = 2.0 * (p[1] * v[1] - k2 * (p[0] * v[0] + p[2] * v[2]))
        qc = p[1] * p[1] - k2 * (p[0] * p[0] + p[2] * p[2])
        if abs(qa) > 1e-15:
            discriminante = qb * qb - 4.0 * qa * qc
            if discriminante >= 0.0:
                radice = math.sqrt(discriminante)
                candidati.extend((((-qb + segno * radice) / (2.0 * qa)), "cono") for segno in (1.0, -1.0))
        elif abs(qb) > 1e-15:
            candidati.append((-qc / qb, "cono"))
        return [(t, parete) for t, parete in candidati if t > TEMPO_MINIMO_DEL_RIMBALZO]

    def uscita(self, punto, velocita):
        """(tempo, parete) in cui il moto rettilineo dal punto, con quella velocità, esce dalla zona, il
        primo istante su una parete in cui il moto va verso fuori e il punto sta dentro tutte le altre;
        (infinito, None) se non esce mai, per esempio da fermo."""
        p, v = _array(punto), _array(velocita)
        if float(v @ v) <= 0.0:
            return math.inf, None
        for tempo, parete in sorted(self._tempi_sulle_pareti(p, v)):
            toccato = p + v * tempo
            if parete == "cono" and toccato[1] <= 0.0:
                continue
            if float(self.normale(parete, toccato) @ v) <= 0.0:
                continue
            if self.contiene(toccato, tolleranza=1e-4):
                return tempo, parete
        return math.inf, None

    def rimbalzo(self, punto, velocita, parete):
        """La velocità riflessa sulla parete toccata nel punto, e su ogni altra parete che il punto tocca
        nello stesso momento verso cui la velocità riflessa andrebbe ancora fuori, come in uno spigolo."""
        v = _array(velocita).copy()
        p = _array(punto)
        toccate = [parete]
        scarti = self.scarti(p)
        toccate.extend(altra for altra, scarto in zip(self.PARETI, scarti, strict=True) if altra != parete and abs(scarto) <= 1e-4)
        for _giro in range(len(self.PARETI)):
            cambiato = False
            for nome in toccate:
                normale = self.normale(nome, p)
                verso = float(normale @ v)
                if verso > 0.0:
                    v = v - 2.0 * verso * normale
                    cambiato = True
            if not cambiato:
                break
        return v

    def punto_a_caso(self, rng):
        """Un punto della zona estratto da rng, un random.Random: la distanza dal centro uniforme fra
        interno ed esterno, l'azimut uniforme nel settore, il seno dell'elevazione uniforme fino a quello
        di ELEVAZIONE_MASSIMA, cioè le direzioni uniformi sulla sfera; si riestrae, sempre in quest'ordine,
        finché il punto non sta dentro. Le distanze uniformi danno ai bersagli vicini e lontani la stessa
        probabilità, come il cubo, dove la profondità era uniforme."""
        seno_massimo = math.sin(math.radians(ELEVAZIONE_MASSIMA))
        while True:
            distanza = rng.uniform(self.interno, self.esterno)
            azimut = rng.uniform(-SETTORE_AZIMUT, SETTORE_AZIMUT)
            elevazione = math.degrees(math.asin(rng.uniform(0.0, seno_massimo)))
            punto = punto_della_cupola(distanza, azimut, elevazione)
            if self.contiene(punto, tolleranza=0.0):
                return punto


class Percorso:
    """Il moto di un bersaglio mobile (D28): dritto, coi rimbalzi sulle pareti della sua zona, a tratti
    rettilinei calcolati una volta sola e tenuti da parte. La posizione di qualunque istante si trova
    senza simulare: il tratto con una ricerca binaria, poi il moto rettilineo dentro il tratto. I tratti
    si allungano quando un istante va oltre, sotto un lucchetto, perché la posizione la chiedono sia il
    callback dell'audio sia la finestra. Un moto che parte fuori dalla zona va dritto e basta."""

    def __init__(self, zona, partenza, velocita):
        self.zona = zona
        self._lucchetto = threading.Lock()
        self._inizi = [0.0]
        self._punti = [np.array(partenza, dtype=float).reshape(3)]
        self._velocita = [np.array(velocita, dtype=float).reshape(3)]
        self._libero = not zona.contiene(self._punti[0], tolleranza=1e-4)
        self._fine = math.inf if self._libero else 0.0
        if not self._libero:
            # Un moto che parte proprio su una parete andando verso fuori rimbalza subito: il contatto a tempo
            # zero uscita lo scarta, e il bersaglio uscirebbe dritto dalla zona.
            punto, velocita = self._punti[0], self._velocita[0]
            for parete, scarto in zip(zona.PARETI, zona.scarti(punto), strict=True):
                if abs(scarto) <= 1e-4 and float(zona.normale(parete, punto) @ velocita) > 0.0:
                    self._velocita[0] = zona.rimbalzo(punto, velocita, parete)
                    break
        self._estendi(ORIZZONTE_DEL_PERCORSO)

    def _estendi(self, fino):
        with self._lucchetto:
            while self._fine < fino and len(self._inizi) < TRATTI_MASSIMI:
                punto, velocita = self._punti[-1], self._velocita[-1]
                durata, parete = self.zona.uscita(punto, velocita)
                if parete is None:
                    self._fine = math.inf
                    break
                toccato = punto + velocita * durata
                self._inizi.append(self._inizi[-1] + durata)
                self._punti.append(toccato)
                self._velocita.append(self.zona.rimbalzo(toccato, velocita, parete))
                self._fine = self._inizi[-1]
            self._tabelle = (np.array(self._inizi), np.array(self._punti), np.array(self._velocita))

    @property
    def rimbalzi(self):
        """Quanti rimbalzi sono già stati calcolati."""
        return len(self._inizi) - 1

    def posizione(self, trascorsi):
        """La posizione dopo tanti secondi di moto, contati dalla partenza, mai prima: (3,) per uno
        scalare, (N, 3) per un array."""
        trascorsi = np.maximum(_array(trascorsi), 0.0)
        if trascorsi.size and float(np.max(trascorsi)) > self._fine:
            self._estendi(float(np.max(trascorsi)))
        inizi, punti, velocita = self._tabelle
        tratti = np.clip(np.searchsorted(inizi, trascorsi, side="right") - 1, 0, len(inizi) - 1)
        return punti[tratti] + velocita[tratti] * (trascorsi - inizi[tratti])[..., None]


@lru_cache(maxsize=1)
def punti_degli_spigoli():
    """I punti degli spigoli della cupola, cioè della Zona di raggio zero, a passi di
    PASSO_DEGLI_SPIGOLI: le curve dove due pareti s'incontrano. Il punto della cupola più lontano da un
    punto qualunque sta sempre su uno di loro, perché ogni parete, piano, sfera o cono, ha il suo punto
    più lontano sul bordo, non nel mezzo: dal di dentro della cupola il punto opposto al centro sta
    sotto terra. Un array (N, 3), calcolato una volta sola."""
    zona = Zona(0.0)
    passo = PASSO_DEGLI_SPIGOLI
    pezzi = []

    def arco_orizzontale(distanza_orizzontale, altezza):
        quanti = max(2, math.ceil(2.0 * math.radians(SETTORE_AZIMUT) * distanza_orizzontale / passo) + 1)
        azimut = np.radians(np.linspace(-SETTORE_AZIMUT, SETTORE_AZIMUT, quanti))
        return np.stack((distanza_orizzontale * np.sin(azimut), np.full(quanti, altezza), distanza_orizzontale * np.cos(azimut)), axis=-1)

    # Gli archi a terra, sulla sfera esterna e sull'interna, e quelli dove le due sfere incontrano il cono.
    for raggio in (CUPOLA_RAGGIO, CUPOLA_RAGGIO_INTERNO):
        pezzi.append(arco_orizzontale(math.sqrt(raggio * raggio - zona.pavimento ** 2), zona.pavimento))
        elevazione = math.radians(zona.elevazione)
        pezzi.append(arco_orizzontale(raggio * math.cos(elevazione), raggio * math.sin(elevazione)))
    for lato in (-SETTORE_AZIMUT, SETTORE_AZIMUT):
        # Sui lati: gli archi delle due sfere, dal pavimento al cono, e i segmenti a terra e sul cono.
        for raggio in (CUPOLA_RAGGIO, CUPOLA_RAGGIO_INTERNO):
            dal_pavimento = math.degrees(math.asin(zona.pavimento / raggio))
            quanti = max(2, math.ceil(math.radians(zona.elevazione - dal_pavimento) * raggio / passo) + 1)
            pezzi.append(punto_della_cupola(raggio, lato, np.linspace(dal_pavimento, zona.elevazione, quanti)))
        quanti = max(2, math.ceil((CUPOLA_RAGGIO - CUPOLA_RAGGIO_INTERNO) / passo) + 1)
        for elevazione_a, elevazione_b in ((math.asin(zona.pavimento / CUPOLA_RAGGIO_INTERNO), math.asin(zona.pavimento / CUPOLA_RAGGIO)),
                                           (math.radians(zona.elevazione), math.radians(zona.elevazione))):
            vicino = punto_della_cupola(CUPOLA_RAGGIO_INTERNO, lato, math.degrees(elevazione_a))
            lontano = punto_della_cupola(CUPOLA_RAGGIO, lato, math.degrees(elevazione_b))
            pezzi.append(vicino + (lontano - vicino) * np.linspace(0.0, 1.0, quanti)[:, None])
    punti = np.concatenate(pezzi)
    punti.flags.writeable = False
    return punti


def distanza_dal_punto_piu_lontano(posizioni):
    """La distanza fra un punto e il punto della cupola più lontano da lui: è la misura della precisione
    globale di D26, la retta che va dal centro del bersaglio al punto più lontano dello spazio colpibile,
    scelta da Gabriele il 20 novembre 2026 per il cubo e passata alla cupola con D28. Vale anche per un
    punto fuori dalla cupola, come il centro di un bonus appena fuori da un lato. Uno scalare per un punto
    (3,), un array di N per posizioni (N, 3)."""
    posizioni = _array(posizioni)
    spigoli = punti_degli_spigoli()
    piatte = posizioni.reshape(-1, 3)
    massime = np.empty(len(piatte))
    for inizio in range(0, len(piatte), 256):
        pezzo = piatte[inizio:inizio + 256]
        differenze = pezzo[:, None, :] - spigoli[None, :, :]
        massime[inizio:inizio + 256] = np.sqrt(np.max(np.sum(differenze * differenze, axis=-1), axis=-1))
    return scalare_o_array(massime.reshape(posizioni.shape[:-1]))


def pan_di(x, z):
    """La panoramica di un punto, da meno 1 a sinistra a 1 a destra: il suo azimut visto dal cannone
    diviso AZIMUT_DEL_PAN, così la panoramica di un bersaglio coincide con la direzione che lo mira.
    La potenza costante la applica sintesi.panoramica."""
    x, z = _array(x), _array(z)
    azimut = np.degrees(np.arctan2(x - CANNONE[0], z - CANNONE[2]))
    return scalare_o_array(np.clip(azimut / AZIMUT_DEL_PAN, -1.0, 1.0))


def frequenza_di(y):
    """La frequenza di un'altezza: FREQUENZA_AL_SUOLO per 2 alla y diviso METRI_PER_OTTAVA, con y
    tenuta fra il suolo e ALTEZZA_TETTO_DEL_SUONO."""
    y = np.clip(_array(y), 0.0, ALTEZZA_TETTO_DEL_SUONO)
    return scalare_o_array(FREQUENZA_AL_SUOLO * 2.0 ** (y / METRI_PER_OTTAVA))


def guadagno_db_di(distanza):
    """Il guadagno in decibel di un suono a quella distanza dal cannone: scende linearmente coi
    metri, fino al pavimento di GUADAGNO_DB_MINIMO."""
    decibel = GUADAGNO_DB_VICINO - DB_PER_METRO * _array(distanza)
    return scalare_o_array(np.maximum(decibel, GUADAGNO_DB_MINIMO))


def guadagno_di(distanza):
    """Il guadagno lineare di un suono a quella distanza dal cannone."""
    return scalare_o_array(10.0 ** (_array(guadagno_db_di(distanza)) / 20.0))


def distanza_dal_cannone(posizioni):
    """La distanza tridimensionale dal cannone: uno scalare per un punto (3,), un array di N per
    posizioni (N, 3)."""
    differenze = _array(posizioni) - np.asarray(CANNONE)
    return scalare_o_array(np.sqrt(np.sum(differenze * differenze, axis=-1)))


def distanza_orizzontale(posizioni):
    """La distanza orizzontale dal cannone, senza l'altezza: quella della gittata e delle misure
    dell'artigliere del sommario (D21). Uno scalare per un punto (3,), un array di N per posizioni
    (N, 3)."""
    posizioni = _array(posizioni)
    return scalare_o_array(np.hypot(posizioni[..., 0] - CANNONE[0], posizioni[..., 2] - CANNONE[2]))


def suono_di(posizioni):
    """Il suono dello spazio, uguale per proiettile, bersagli ed effetti: per posizioni (N, 3)
    restituisce (frequenze, pan, guadagni lineari), tre array di N; per un punto (3,) tre float."""
    posizioni = _array(posizioni)
    frequenze = frequenza_di(posizioni[..., 1])
    pan = pan_di(posizioni[..., 0], posizioni[..., 2])
    guadagni = guadagno_di(distanza_dal_cannone(posizioni))
    return frequenze, pan, guadagni


def scomponi(delta, punto_del_bersaglio):
    """Lo scarto delta = proiettile meno bersaglio, scomposto: (radiale, laterale, verticale). Il
    radiale è lungo la direzione orizzontale del bersaglio vista dal cannone, positivo se lungo e
    negativo se corto; il laterale è orizzontale e perpendicolare, positivo a destra; il verticale è
    positivo sopra. Un bersaglio proprio sopra il cannone, che non ha una direzione orizzontale, si
    legge come se stesse davanti. Il volo lo usa per lo scarto nel punto più vicino, e il sommario
    di D21 per il laterale e il verticale delle misure dell'artigliere."""
    delta, punto = _array(delta), _array(punto_del_bersaglio)
    avanti_x = punto[..., 0] - CANNONE[0]
    avanti_z = punto[..., 2] - CANNONE[2]
    lunghezza = np.hypot(avanti_x, avanti_z)
    davanti = lunghezza > 0.0
    divisore = np.where(davanti, lunghezza, 1.0)
    ux = np.where(davanti, avanti_x / divisore, 0.0)
    uz = np.where(davanti, avanti_z / divisore, 1.0)
    # La destra di chi guarda lungo (ux, 0, uz), con y in alto, è (uz, 0, meno ux).
    radiale = delta[..., 0] * ux + delta[..., 2] * uz
    laterale = delta[..., 0] * uz - delta[..., 2] * ux
    verticale = delta[..., 1]
    return scalare_o_array(radiale), scalare_o_array(laterale), scalare_o_array(verticale)


def velocita_per(punto, alzo_gradi):
    """La velocità alla bocca con cui un tiro a quell'alzo passa per il punto, dalla parabola:
    v al quadrato = g d al quadrato / (2 coseno al quadrato dell'alzo (d tangente dell'alzo meno h)),
    con d la distanza orizzontale e h il dislivello dal cannone. Dove l'alzo non basta a salire fin
    lì la velocità è nan. Accetta l'alzo scalare o array."""
    x, y, z = (float(c) for c in punto)
    distanza = math.hypot(x - CANNONE[0], z - CANNONE[2])
    dislivello = y - CANNONE[1]
    alzo = np.radians(_array(alzo_gradi))
    coseno = np.cos(alzo)
    denominatore = 2.0 * coseno * coseno * (distanza * np.tan(alzo) - dislivello)
    possibile = (denominatore > 0.0) & (distanza > 0.0)
    quadrato = np.where(possibile, GRAVITA * distanza * distanza / np.where(possibile, denominatore, 1.0), np.nan)
    return scalare_o_array(np.sqrt(quadrato))


def soluzione(punto):
    """I tre valori da 0 a 1 di un tiro che passa per il punto, tutti dentro i margini di copertura,
    oppure None se non ce n'è. Serve alla prova della copertura.
    La direzione viene diretta dall'azimut del punto. Fra gli alzi possibili si sceglie quello che
    chiede la potenza più bassa: è il tiro più dolce, e il più indulgente, perché nella velocità
    minima la gittata non cambia al primo ordine con l'alzo, come i 45 gradi sul terreno piano, e un
    piccolo errore nell'alzo sposta il colpo il meno possibile. L'alzo della velocità
    minima ha una formula chiusa, tangente = (h + radice di h al quadrato più d al quadrato) / d, e
    si prova per primo; la scansione dell'alzo serve quando quel tiro esce dai margini."""
    x, y, z = (float(c) for c in punto)
    margine = MARGINE_DI_COPERTURA
    azimut = math.degrees(math.atan2(x - CANNONE[0], z - CANNONE[2]))
    alzo_basso = ALZO_MINIMO + margine * (ALZO_MASSIMO - ALZO_MINIMO)
    alzo_alto = ALZO_MINIMO + (1.0 - margine) * (ALZO_MASSIMO - ALZO_MINIMO)
    alzi = [np.arange(alzo_basso, alzo_alto, PASSO_DELL_ALZO_NELLA_SOLUZIONE), [alzo_alto]]
    distanza = math.hypot(x - CANNONE[0], z - CANNONE[2])
    dislivello = y - CANNONE[1]
    if distanza > 0.0:
        dolce = math.degrees(math.atan((dislivello + math.hypot(dislivello, distanza)) / distanza))
        if alzo_basso <= dolce <= alzo_alto:
            alzi.append([dolce])
    alzi = np.concatenate(alzi)
    velocita = np.atleast_1d(velocita_per((x, y, z), alzi))
    direzione, valori_alzo, potenze = valori_del_tiro(azimut, alzi, velocita)
    if not margine <= direzione <= 1.0 - margine:
        return None
    validi = np.isfinite(velocita) & (potenze >= margine) & (potenze <= 1.0 - margine)
    if not validi.any():
        return None
    scelto = int(np.argmin(np.where(validi, velocita, np.inf)))
    return float(direzione), float(valori_alzo[scelto]), float(potenze[scelto])
