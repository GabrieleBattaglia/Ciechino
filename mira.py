"""
La mira manuale di Ciechino (D29): tre valori, direzione, alzo e potenza, che si muovono finché i loro tasti sono giù.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0; dalla 2.0.0 la mira non è più fatta di scivolate da fermare con lo spazio.
Gabriele, il 22 novembre 2026: «togliere gli scorrimenti dalle barre parametriche e passare ad un vero
puntamento manuale», e poi «i 3 spostamenti 3d si fanno tutti in una unica operazione di puntamento».
Ogni valore va da 0 a 1, parte dal centro della sua scala e, col suo tasto giù, la percorre a
VELOCITA_DELLA_MIRA, da un capo all'altro in DURATA_DELLA_SCALA secondi, fino a fermarsi ai capi; con due
tasti di valori diversi giù, i due valori si muovono insieme. Il tempo è quello dell'audio, in secondi,
il campione udibile diviso la frequenza di campionamento: così il valore di un istante è quello che si
sentiva in quell'istante.
Il modulo non sa niente di audio né di finestre, e non usa numpy fuori da valore, che accetta anche un
array di tempi per il motore: lo importano la partita e le prove.
"""

import numpy as np

from costanti import PARAMETRI, PARTENZA_DELLA_MIRA, TACCHE, VELOCITA_DELLA_MIRA


class Asse:
    """Un valore della mira. Lo stato è una tupla sola, (valore, verso, dal): il valore che aveva al
    tempo dal, e il verso in cui si muove da allora, meno uno, zero o uno. La tupla si sostituisce tutta
    insieme, così il filo dell'audio, che legge lo stato a ogni blocco, non vede mai un cambio a metà."""

    def __init__(self, valore=PARTENZA_DELLA_MIRA):
        self.stato = (min(1.0, max(0.0, float(valore))), 0, 0.0)

    @property
    def verso(self):
        return self.stato[1]

    def valore(self, tempo):
        """Il valore al tempo dato, da 0 a 1: uno scalare per un tempo, un array per un array."""
        valore, verso, dal = self.stato
        tempi = np.asarray(tempo, dtype=float)
        valori = np.clip(valore + verso * VELOCITA_DELLA_MIRA * np.maximum(tempi - dal, 0.0), 0.0, 1.0)
        return float(valori) if valori.ndim == 0 else valori

    def muovi(self, verso, tempo):
        """Da tempo il valore si muove nel verso dato, meno uno verso lo zero, uno verso l'uno, zero fermo:
        il valore di quell'istante resta, e la corsa riparte da lì."""
        self.stato = (self.valore(tempo), int(verso), float(tempo))

    def ferma(self, tempo):
        self.muovi(0, tempo)

    def in_moto(self, tempo):
        """Vero dove il valore si muove, al tempo dato: col verso non zero, dal tempo dal fino all'arrivo al
        capo, escluso. Uno scalare per un tempo, un array di bool per un array; lo stato si legge una volta
        sola, come in valore. Il puntatore suona dove almeno un valore è in moto (D31)."""
        valore, verso, dal = self.stato
        tempi = np.asarray(tempo, dtype=float)
        if verso == 0:
            esito = np.zeros(tempi.shape, dtype=bool)
        else:
            capo = 1.0 if verso > 0 else 0.0
            esito = (tempi >= dal) & (tempi < dal + abs(capo - valore) / VELOCITA_DELLA_MIRA)
        return bool(esito) if esito.ndim == 0 else esito

    def capo(self):
        """Il capo verso cui va, 0 o 1, o None da fermo."""
        if self.verso > 0:
            return 1.0
        if self.verso < 0:
            return 0.0
        return None

    def tempo_al_capo(self):
        """Il tempo in cui arriva al capo verso cui va, o None da fermo. Un valore già al capo ci arriva
        subito, al tempo dal, e allora il capo si sente appena il tasto scende."""
        valore, _verso, dal = self.stato
        capo = self.capo()
        if capo is None:
            return None
        return dal + abs(capo - valore) / VELOCITA_DELLA_MIRA

    def tacche_davanti(self):
        """Le tacche che il valore attraverserà correndo verso il capo, come coppie (tempo, k), con k da
        1 a TACCHE meno 1, cioè dal primo al nono decimo: quella su cui il valore sta già non conta, e i
        capi hanno il loro suono. Vuota da fermo."""
        valore, verso, dal = self.stato
        if verso == 0:
            return []
        tacche = []
        for k in range(1, TACCHE):
            soglia = k / TACCHE
            davanti = soglia > valore + 1e-12 if verso > 0 else soglia < valore - 1e-12
            if davanti:
                tacche.append((dal + abs(soglia - valore) / VELOCITA_DELLA_MIRA, k))
        return sorted(tacche)


class Mira:
    """La mira di un tiro: un Asse per ogni parametro di PARAMETRI, e i tasti giù di ciascuno, nell'ordine
    in cui sono scesi. Un valore va nel verso dell'ultimo tasto suo sceso e ancora giù: tenendo giù la
    freccia sinistra e poi anche la destra va a destra, e lasciata la destra torna a sinistra.
    premi e lascia restituiscono vero se il moto del valore è cambiato, e allora chi suona deve
    rifare il suono di quel valore."""

    def __init__(self, partenza=PARTENZA_DELLA_MIRA):
        self.assi = {parametro: Asse(partenza) for parametro in PARAMETRI}
        self._giu = {parametro: [] for parametro in PARAMETRI}
        # I tasti premuti in tutta la mira, per le statistiche: una pressione vera, non le ripetizioni.
        self.pressioni = 0

    def _controlla(self, parametro, verso):
        if parametro not in self.assi:
            raise ValueError(f"parametro sconosciuto: {parametro!r}")
        if verso not in (-1, 1):
            raise ValueError(f"il verso di un tasto della mira è meno uno o uno, non {verso!r}")

    def premi(self, parametro, verso, tempo):
        """Il tasto del parametro nel verso dato scende al tempo dato. Un tasto già giù, come la sua
        ripetizione automatica, non cambia niente."""
        self._controlla(parametro, verso)
        giu = self._giu[parametro]
        if verso in giu:
            return False
        giu.append(verso)
        self.pressioni += 1
        self.assi[parametro].muovi(verso, tempo)
        return True

    def lascia(self, parametro, verso, tempo):
        """Il tasto del parametro nel verso dato sale al tempo dato: il valore prende il verso dell'altro
        tasto suo ancora giù, o si ferma. Un tasto che non era giù non cambia niente."""
        self._controlla(parametro, verso)
        giu = self._giu[parametro]
        if verso not in giu:
            return False
        giu.remove(verso)
        asse = self.assi[parametro]
        nuovo = giu[-1] if giu else 0
        if nuovo == asse.verso:
            return False
        asse.muovi(nuovo, tempo)
        return True

    def lascia_tutto(self, tempo):
        """Tutti i tasti su al tempo dato, come allo sparo, all'Esc o alla pausa: i valori restano dove
        sono. Restituisce i parametri il cui moto è cambiato."""
        cambiati = []
        for parametro, asse in self.assi.items():
            self._giu[parametro] = []
            if asse.verso != 0:
                asse.ferma(tempo)
                cambiati.append(parametro)
        return cambiati

    def tasti_giu(self):
        """I tasti giù, come coppie (parametro, verso)."""
        return [(parametro, verso) for parametro, giu in self._giu.items() for verso in giu]

    def valori(self, tempo):
        """I tre valori al tempo dato, nell'ordine di PARAMETRI."""
        return [self.assi[parametro].valore(tempo) for parametro in PARAMETRI]
