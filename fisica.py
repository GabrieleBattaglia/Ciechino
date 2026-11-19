"""
La fisica di Ciechino: il tiro, la traiettoria del proiettile e il modo in cui un punto dello spazio diventa suono.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Niente wx e niente audio, soltanto numpy: la usano il gioco, il motore audio dentro il callback, le
prove e i banchi. Ogni funzione di mappatura accetta uno scalare o un array numpy e restituisce la
stessa forma, uno scalare come float di Python e un array come array: così il motore chiede in un
colpo solo i parametri di tutti i campioni di un blocco, e chi lavora con un punto solo non si
porta dietro array di dimensione zero.
Gli assi sono quelli di costanti.py: x verso destra, y verso l'alto, z in avanti; il giocatore sta
nell'origine e il cannone in (0, ALTEZZA_CANNONE, 0). Gli angoli si danno in gradi, le distanze in
metri, le velocità in metri al secondo.
"""

import math

import numpy as np

from costanti import (
    ALTEZZA_CANNONE,
    ALTEZZA_MINIMA_BERSAGLIO,
    ALTEZZA_TETTO_DEL_SUONO,
    ALZO_MASSIMO,
    ALZO_MINIMO,
    AZIMUT_DEL_PAN,
    AZIMUT_MASSIMO,
    CUBO_X,
    CUBO_Y,
    CUBO_Z,
    DB_PER_METRO,
    ESPONENTE_POTENZA,
    FREQUENZA_AL_SUOLO,
    GRAVITA,
    GUADAGNO_DB_MINIMO,
    GUADAGNO_DB_VICINO,
    MARGINE_DI_COPERTURA,
    METRI_PER_OTTAVA,
    VELOCITA_MASSIMA,
    VELOCITA_MINIMA,
)

# Il passo, in gradi, con cui soluzione scandisce l'alzo in cerca del tiro più dolce. Non è una
# taratura del gioco ma la finezza di una ricerca: la velocità di ogni alzo viene da una formula
# chiusa, quindi il tiro trovato passa esattamente per il punto qualunque sia il passo, e il passo
# decide soltanto quanto la potenza scelta si avvicina alla minima possibile.
PASSO_DELL_ALZO_NELLA_SOLUZIONE = 0.02

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


def parametri_del_tiro(direzione, alzo, potenza):
    """I tre valori delle scivolate, da 0 a 1, diventano (azimut in gradi, alzo in gradi, velocità
    alla bocca in metri al secondo). L'azimut è positivo a destra; la potenza passa dall'esponente
    di costanti.py, che con 1 tiene la scala lineare."""
    direzione, alzo, potenza = _array(direzione), _array(alzo), _array(potenza)
    azimut_gradi = (2.0 * direzione - 1.0) * AZIMUT_MASSIMO
    alzo_gradi = ALZO_MINIMO + alzo * (ALZO_MASSIMO - ALZO_MINIMO)
    velocita = VELOCITA_MINIMA + _potenza_con_segno(potenza, ESPONENTE_POTENZA) * (VELOCITA_MASSIMA - VELOCITA_MINIMA)
    return scalare_o_array(azimut_gradi), scalare_o_array(alzo_gradi), scalare_o_array(velocita)


def valori_del_tiro(azimut_gradi, alzo_gradi, velocita):
    """L'inversa di parametri_del_tiro: i tre valori da 0 a 1 che danno quel tiro. Non taglia nulla,
    così un valore fuori da 0 e 1 dice che il tiro non si può scegliere con le scivolate."""
    azimut_gradi, alzo_gradi, velocita = _array(azimut_gradi), _array(alzo_gradi), _array(velocita)
    direzione = (azimut_gradi / AZIMUT_MASSIMO + 1.0) / 2.0
    alzo = (alzo_gradi - ALZO_MINIMO) / (ALZO_MASSIMO - ALZO_MINIMO)
    frazione = (velocita - VELOCITA_MINIMA) / (VELOCITA_MASSIMA - VELOCITA_MINIMA)
    potenza = _potenza_con_segno(frazione, 1.0 / ESPONENTE_POTENZA)
    return scalare_o_array(direzione), scalare_o_array(alzo), scalare_o_array(potenza)


class Traiettoria:
    """La parabola del proiettile, esatta: gravità costante e niente aria, in funzione del tempo di
    volo contato dallo sparo. Si calcola in qualunque istante senza simulare nulla, così il volo, la
    voce del proiettile e i resoconti vedono la stessa curva."""

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


def zona_del_centro(raggio):
    """La zona dove può stare il centro di un bersaglio di quel raggio: tutto il bersaglio dentro il
    cubo e il centro almeno ad ALTEZZA_MINIMA_BERSAGLIO da terra. Restituisce (minimi, massimi), due
    array di 3 per x, y e z. Un bersaglio più grande del cubo non ci sta: la zona si stringe al punto
    di mezzo, invece di diventare un intervallo rovesciato."""
    raggio = float(raggio)
    minimi = np.array((CUBO_X[0] + raggio, max(CUBO_Y[0] + raggio, ALTEZZA_MINIMA_BERSAGLIO), CUBO_Z[0] + raggio))
    massimi = np.array((CUBO_X[1] - raggio, CUBO_Y[1] - raggio, CUBO_Z[1] - raggio))
    rovesciati = massimi < minimi
    mezzo = (minimi + massimi) / 2.0
    return np.where(rovesciati, mezzo, minimi), np.where(rovesciati, mezzo, massimi)


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


def suono_di(posizioni):
    """Il suono dello spazio, uguale per proiettile, bersagli ed effetti: per posizioni (N, 3)
    restituisce (frequenze, pan, guadagni lineari), tre array di N; per un punto (3,) tre float."""
    posizioni = _array(posizioni)
    frequenze = frequenza_di(posizioni[..., 1])
    pan = pan_di(posizioni[..., 0], posizioni[..., 2])
    guadagni = guadagno_di(distanza_dal_cannone(posizioni))
    return frequenze, pan, guadagni


def scomponi(delta, punto_del_bersaglio):
    """Lo scarto delta = proiettile meno bersaglio, scomposto come lo dice il resoconto:
    (radiale, laterale, verticale). Il radiale è lungo la direzione orizzontale del bersaglio vista
    dal cannone, positivo se lungo e negativo se corto; il laterale è orizzontale e perpendicolare,
    positivo a destra; il verticale è positivo sopra. Un bersaglio proprio sopra il cannone, che
    non ha una direzione orizzontale, si legge come se stesse davanti."""
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
    oppure None se non ce n'è. Serve alla prova della copertura e ai banchi.
    La direzione viene diretta dall'azimut del punto. Fra gli alzi possibili si sceglie quello che
    chiede la potenza più bassa: è il tiro più dolce, e il più indulgente, perché nella velocità
    minima la gittata non cambia al primo ordine con l'alzo, come i 45 gradi sul terreno piano, e un
    piccolo errore nella seconda scivolata sposta il colpo il meno possibile. L'alzo della velocità
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
