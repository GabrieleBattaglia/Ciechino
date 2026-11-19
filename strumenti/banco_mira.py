"""
Ciechino, il banco della mira: le scivolate alla cieca e la prova di precisione, per tarare il colpo a orecchio.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di strumenti/banco_spazio.py di MESS. Gabriele l'ha
chiesto nel punto 73 di idee.txt: è necessario fare dei test con i suoni, soprattutto per la
calibrazione del colpo.
La prima parte sono i gruppi alla cieca. Per ogni parametro tre versioni della scivolata: la forma
della direzione, la scala dell'alzo, la scala della potenza, con la domanda di quanto è facile capire
dove si è nella scala e fermarla dove si vuole; poi le tacche e i bip del conto, con la mira intera,
le tre scivolate di fila come in partita. La seconda parte è la prova di precisione, che misura: per
ogni parametro otto tentativi, ciascuno con un breve suono di riferimento, il punto della scala da
raggiungere, e poi la mira vera, la pausa, i bip e la scivolata, fermata con lo spazio letto da key.
Il valore lo dà mira.valore_scelto sul campione udibile del motore audio di Ciechino, la stessa
funzione e lo stesso motore della partita: gli errori che il banco scrive servono a tarare
COMPENSAZIONE_DEL_TASTO e la scala.
Il protocollo è quello delle prove alla cieca di Gabriele, che va trattato come uno strumento di
misura. Prima della prova lo strumento dice soltanto come si esegue e che domanda ci sarà, mai quale
versione sia quella attuale, mai i numeri, mai che c'è un controllo; aspetta il via di ogni gruppo e
lascia qualche secondo di silenzio prima del primo suono, dentro il buffer, così il silenzio è
esatto al campione e la tastiera resta in ascolto. In ogni gruppo una versione, scelta a caso, sta
sotto due lettere ed è presentata come le altre: è il controllo. Le lettere si mescolano con un
seme, che si registra. Ogni gruppo comincia con la rassegna, tutte le lettere una volta, senza
domande, e poi il voto da 1 a 5 lettera per lettera, con n per nessuna preferenza; ogni lettera si
risente quante volte si vuole, e nel voto lo spazio la fa partire e la ferma, come in partita.
Mentre suona non si scrive niente, perché la voce di NVDA non copra il suono. In fondo al gruppo c'è
il menu comune di collaudo_comune di GBUtils: r ripete, c commenta, Invio dà il gruppo per superato,
Escape lo chiude senza giudizio, e allora i suoi voti restano nel file ma non decidono niente.
Gli esiti vanno nel file banco_mira_esiti.txt, accanto al banco, in coda, una riga per voce: in testa
la data e il seme, poi i voti con le sole lettere e i commenti; alla fine della sessione, e soltanto
allora, anche se la sessione si interrompe, quale lettera era quale, il riepilogo dei voti per
versione, il conto del controllo, la scelta di ogni gruppo e i conti della precisione.
Questo modulo tiene anche le parti comuni ai tre banchi, che banco_bersagli e banco_volo importano:
l'ascolto col motore del gioco, il flusso in memoria che rende fuori tempo reale ciò che il motore
suonerebbe, le prove alla cieca col loro voto, i risultati e la rivelazione.
Uso, dalla cartella del programma: python strumenti\\banco_mira.py; con un numero, per esempio
python strumenti\\banco_mira.py 4127, usa quel seme, e quindi le stesse lettere di una sessione passata.
"""

import contextlib
import random
import statistics
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

STRUMENTI = Path(__file__).resolve().parent
RADICE = STRUMENTI.parent
for cartella in (RADICE, STRUMENTI):
    if str(cartella) not in sys.path:
        sys.path.insert(0, str(cartella))

import collaudo_comune as comune  # noqa: E402
from GBUtils import enter_escape, key  # noqa: E402

import mira  # noqa: E402
import sintesi  # noqa: E402
from costanti import (  # noqa: E402
    ALZO_FREQUENZE,
    BIP_DURATA,
    DIREZIONE_FREQUENZA,
    FS,
    MARGINE_DOPO_LA_SCIVOLATA,
    PARAMETRI,
    PAUSA_FRA_BARRE,
    POTENZA_DB_MINIMO,
    POTENZA_FREQUENZA,
    RAMPA_DEL_GUADAGNO,
    SCIVOLATA_DB,
    SCIVOLATA_DURATA,
    SCIVOLATA_FORMA,
    TACCA_DB,
)
from lingua import _, ngettext  # noqa: E402
from motore_audio import Motore  # noqa: E402
from suoni import Suoni  # noqa: E402
from testi import conta, durata, elenco, intero, numero, percento  # noqa: E402

FILE_DEGLI_ESITI = STRUMENTI / "banco_mira_esiti.txt"
LETTERE = "ABCDEFGH"
VOTI = frozenset("12345")
# Il silenzio prima del primo suono di ogni gruppo, dopo il suo via: il protocollo di Gabriele vuole
# qualche secondo per togliere le mani dalla tastiera e mettersi in ascolto.
SILENZIO_INIZIALE = 3.0
# Il silenzio prima di ogni lettera della rassegna, il tempo di sentire il suo nome da NVDA; nel voto
# il suono parte quasi subito, perché il tasto l'ha chiesto.
ANTICIPO_RASSEGNA = 2.0
ANTICIPO_PRIMA_LETTERA = SILENZIO_INIZIALE + ANTICIPO_RASSEGNA
ANTICIPO_VOTO = 0.3
# Mentre si suona si dà un'occhiata alla tastiera a questo passo: key di GBUtils aspetta dentro il
# sistema, quindi un tasto arriva appena premuto, e il passo decide soltanto ogni quanto si guarda
# se il suono è finito. Oltre la fine si aspetta CODA, perché l'ultimo suono non lo copra NVDA.
SGUARDO = 0.05
CODA = 0.3
# Il tempo vero oltre la fine prevista dopo cui si smette comunque di aspettare: un flusso che si è
# fermato non deve tenere il banco in attesa per sempre.
RISERVA_DI_TEMPO = 1.0
# Nella rassegna soltanto Escape ferma il suono, e la interrompe.
TASTI_DELLA_RASSEGNA = frozenset("\x1b")
# I numeri delle versioni da confrontare, dati da Gabriele il 19 novembre 2026: la potenza da meno 30
# decibel, le tacche più forti di tanti decibel di quelle attuali, i bip del conto di 20 millesimi.
# L'alzo da 150 a 2400 hertz ha vinto il primo banco ed è diventato quello attuale (D19): il
# confronto è ora con la scala di prima, da 220 a 1760.
ALZO_DI_PRIMA = (220.0, 1760.0)
POTENZA_DB_MINIMO_RIDOTTO = -30.0
TACCHE_PIU_FORTI_DB = 6.0
BIP_LUNGHI = 0.020
# La prova di precisione: quanti tentativi per serie, fra quali punti della scala cadono i
# riferimenti, uno a caso in ogni ottavo, così coprono tutta la scala senza i capi; quanto dura il
# suono di riferimento, quanto silenzio c'è fra lo spazio che comincia il tentativo e il riferimento.
TENTATIVI = 8
RIFERIMENTI = (0.1, 0.9)
DURATA_DEL_RIFERIMENTO = 0.4
ANTICIPO_TENTATIVO = 1.0
# Le rampe del riferimento, contro i click, le stesse della scala delle scivolate.
RAMPA_DEL_RIFERIMENTO = sintesi.RAMPA_DELLA_SCALA
TASTI_DEL_TENTATIVO = frozenset("\x1b")
# L'orologio delle attese; le prove lo sostituiscono.
orologio = time.monotonic


# L'ascolto col motore del gioco.

class Valore(NamedTuple):
    """Un valore preso da una scivolata: il valore da 0 a 1, l'esito di mira.valore_scelto e se è
    venuto dallo spazio o dalla scivolata arrivata in fondo da sola."""

    valore: float
    esito: str
    premuto: bool


class Ascolto:
    """Il motore audio del gioco al servizio dei banchi: suona i materiali al campione, guarda la
    tastiera mentre suonano e, nelle scivolate, prende il valore come la partita, con
    mira.valore_scelto sul campione udibile. effetti sono i Suoni del motore, per il valore preso.
    partiti conta i materiali fatti partire, barra è la scivolata in corso, (campione d'inizio del
    suo buffer, Scivolata), oppure None: servono a chi guarda il banco da fuori, come le prove."""

    def __init__(self, motore, effetti=None):
        self.motore = motore
        self.effetti = effetti
        self.partiti = 0
        self.barra = None

    @property
    def fs(self):
        return self.motore.fs

    def chiudi(self):
        self.motore.chiudi()

    def silenzio(self, secondi):
        """Tanti secondi di silenzio stereo, arrotondati al campione."""
        return np.zeros((round(secondi * self.fs), 2), dtype=np.float32)

    def suona(self, materiale, anticipo, tasti, ferma_con_spazio=False):
        """Fa sentire un materiale, un buffer o una tupla di scivolate, dopo anticipo secondi di
        silenzio, e restituisce il tasto fra quelli dati che lo ha fermato, o None."""
        if isinstance(materiale, np.ndarray):
            return self.suona_buffer(materiale, anticipo, tasti, ferma_con_spazio)
        tasto, _valori = self.suona_mira(materiale, anticipo, tasti, ferma_con_spazio)
        return tasto

    def suona_buffer(self, buffer, anticipo, tasti, ferma_con_spazio=False):
        """Fa sentire un buffer col silenzio dell'anticipo in testa e aspetta che finisca. Un tasto
        fra quelli dati, premuto mentre suona, lo ferma e torna; con ferma_con_spazio lo spazio lo
        ferma e basta. None se il suono è finito da sé."""
        dati = np.concatenate([self.silenzio(anticipo), np.asarray(buffer, dtype=np.float32)])
        voce = self.motore.suona(dati)
        self.partiti += 1
        fine = voce.inizio + len(dati) + round(CODA * self.fs)
        scadenza = orologio() + len(dati) / self.fs + CODA + RISERVA_DI_TEMPO
        while self.motore.campione_udibile() < fine and orologio() < scadenza:
            tasto = key(attesa=SGUARDO, alla_scadenza=None)
            if tasto == " " and ferma_con_spazio:
                self.motore.ferma(voce)
                return None
            if tasto is not None and tasto in tasti:
                self.motore.ferma(voce)
                return tasto
        return None

    def suona_mira(self, scivolate, anticipo, tasti, ferma_con_spazio=False, teste=None):
        """Le scivolate una dopo l'altra, come nella mira della partita. Ogni barra è il silenzio
        della pausa, che non sta nel buffer della scivolata, poi il buffer; la prima ha il silenzio
        dell'anticipo, mai meno della pausa fra le barre, perché lo spazio che la fa partire è come
        quello che avvia la mira. teste, se c'è, dà la testa di ogni barra al posto del silenzio:
        la prova di precisione ci mette il riferimento.
        Con ferma_con_spazio lo spazio prende il valore con mira.valore_scelto sul campione udibile:
        nella pausa non conta, altrove ferma la scivolata con la sfumatura del gioco e fa suonare il
        valore preso. Senza spazio, quando mira.scaduta lo dice, il valore è il massimo (D3), e suona
        anch'esso il valore preso. Un tasto fra quelli dati ferma tutto e torna.
        Restituisce (tasto o None, lista dei Valore presi)."""
        valori = []
        for posto, scivolata in enumerate(scivolate):
            if teste is not None:
                testa = teste[posto]
            else:
                testa = self.silenzio(max(anticipo, PAUSA_FRA_BARRE) if posto == 0 else PAUSA_FRA_BARRE)
            dati = np.concatenate([testa, scivolata.buffer])
            voce = self.motore.suona(dati)
            if posto == 0:
                self.partiti += 1
            inizio = voce.inizio + len(testa)
            self.barra = (inizio, scivolata)
            scadenza = orologio() + len(dati) / self.fs + MARGINE_DOPO_LA_SCIVOLATA + RISERVA_DI_TEMPO
            try:
                while True:
                    tasto = key(attesa=SGUARDO, alla_scadenza=None)
                    # Il campione udibile si legge subito dopo il tasto: è ciò che usciva dalle casse
                    # quando il dito è sceso, a meno della compensazione che toglie mira.
                    udibile = self.motore.campione_udibile()
                    if tasto == " " and ferma_con_spazio:
                        valore, esito = mira.valore_scelto(udibile, inizio, scivolata)
                        if esito == "pausa":
                            continue
                        self.motore.ferma(voce)
                        self._preso()
                        valori.append(Valore(valore, esito, True))
                        break
                    if tasto is not None and tasto in tasti:
                        self.motore.ferma(voce)
                        return tasto, valori
                    if mira.scaduta(udibile, inizio, scivolata) or orologio() >= scadenza:
                        self._preso()
                        valori.append(Valore(1.0, "massimo", False))
                        break
            finally:
                self.barra = None
        return self.aspetta(CODA, tasti), valori

    def aspetta(self, secondi, tasti):
        """Aspetta tanti secondi guardando la tastiera: un tasto fra quelli dati torna subito."""
        scadenza = orologio() + secondi
        while (resto := scadenza - orologio()) > 0:
            tasto = key(attesa=min(resto, SGUARDO), alla_scadenza=None)
            if tasto is not None and tasto in tasti:
                return tasto
        return None

    def _preso(self):
        if self.effetti is not None:
            self.effetti.suona("valore_preso")


def prepara_ascolto(eventi=("valore_preso",)):
    """L'ascolto col motore vero, avviato sulla scheda audio, e i suoni degli eventi già sintetizzati;
    None se la scheda non si apre, perché un banco d'ascolto muto non serve. Le prove lo sostituiscono."""
    motore = Motore()
    if not motore.avvia():
        motore.chiudi()
        return None
    effetti = Suoni(motore)
    effetti.precarica(eventi)
    return Ascolto(motore, effetti)


# Il rendering fuori tempo reale.

class _TempiInMemoria(NamedTuple):
    """Il time_info del flusso in memoria, coi nomi dei campi di PortAudio."""

    outputBufferDacTime: float
    currentTime: float
    inputBufferAdcTime: float = 0.0


class FlussoInMemoria:
    """Un flusso che non suona: il motore gli dà il suo callback all'apertura, e lui lo chiama a
    blocchi quando gli si chiede di rendere tanti campioni, con un orologio che avanza di un blocco
    a ogni chiamata. Così le voci del gioco, i loro eventi al campione e il tetto della somma si
    rendono in un buffer esattamente come il motore li suonerebbe."""

    def __init__(self, callback, fs, blocco):
        self.callback = callback
        self.fs = int(fs)
        self.blocco = int(blocco)
        self.latency = 0.0
        self.time = 0.0
        self.interfaccia = "in memoria"
        self.device = None

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def rendi(self, campioni):
        """I prossimi campioni del motore, (campioni, 2) float32."""
        blocchi = -(-int(campioni) // self.blocco)
        uscita = np.zeros((blocchi * self.blocco, 2), dtype=np.float32)
        for indice in range(blocchi):
            pezzo = uscita[indice * self.blocco:(indice + 1) * self.blocco]
            self.callback(pezzo, self.blocco, _TempiInMemoria(self.time + self.latency, self.time), None)
            self.time += self.blocco / self.fs
        return uscita[:int(campioni)]


@contextlib.contextmanager
def in_memoria():
    """Un motore del gioco avviato sul flusso in memoria, chiuso all'uscita: (motore, flusso)."""
    motore = Motore(apri=FlussoInMemoria)
    motore.avvia()
    try:
        yield motore, motore.flusso
    finally:
        motore.chiudi()


def entra_ed_esce(voce, inizio, fine, fs=FS):
    """Prepara una voce che sale da zero in RAMPA_DEL_GUADAGNO dal campione inizio e si spegne nella
    stessa rampa fino al campione fine: un materiale che comincia e finisce senza click."""
    rampa = round(RAMPA_DEL_GUADAGNO * fs)
    voce.imposta_guadagno(0.0, inizio, 0)
    voce.imposta_guadagno(1.0, inizio, rampa)
    voce.ferma_a(max(inizio, fine - rampa), rampa)
    return voce


# Le prove alla cieca.

class Candidato(NamedTuple):
    """Una versione da confrontare: il nome, che si rivela soltanto alla fine, la ricetta del suo
    materiale e se è quella che il gioco usa oggi."""

    nome: str
    ricetta: object
    attuale: bool = False


class Gruppo(NamedTuple):
    """Un gruppo alla cieca: la chiave, il titolo, la domanda, ciò che si sente in ogni lettera e le
    versioni. preferenza falso vuol dire che il voto misura una percezione, come la grandezza, e non
    un gradimento: allora il riepilogo segue l'ordine delle versioni e non c'è una scelta."""

    chiave: str
    titolo: str
    domanda: str
    materiale: str
    candidati: tuple
    preferenza: bool = True


def assegna(rng, quanti):
    """Le lettere di un gruppo: per ogni lettera, nell'ordine, l'indice della sua versione, con una
    versione scelta a caso ripetuta sotto due lettere, il controllo, e tutto mescolato.
    Restituisce l'ordine e l'indice della versione ripetuta."""
    ripetuto = rng.randrange(quanti)
    ordine = [*range(quanti), ripetuto]
    rng.shuffle(ordine)
    return tuple(ordine), ripetuto


class Prova:
    """Un gruppo da ascoltare: il gruppo, i materiali delle versioni, le lettere col controllo e i
    voti che si raccolgono."""

    def __init__(self, gruppo, materiali, ordine, ripetuto):
        self.gruppo = gruppo
        self.materiali = materiali
        self.ordine = ordine
        self.ripetuto = ripetuto
        self.lettere = LETTERE[:len(ordine)]
        self.voti = {}
        self.nessuna_preferenza = False
        self.saltata = False
        # Iniziata col via del gruppo, svolta col menu di fine gruppo: un gruppo iniziato e non
        # svolto è stato interrotto, e i suoi voti si scrivono lo stesso alla fine.
        self.iniziata = False
        self.svolta = False
        # Chiusa senza giudizio, con Escape nel menu di fine gruppo: i voti restano nel file ma non decidono.
        self.chiusa = False

    def indice(self, lettera):
        return self.ordine[self.lettere.index(lettera)]

    def candidato(self, lettera):
        return self.gruppo.candidati[self.indice(lettera)]

    def materiale(self, lettera):
        return self.materiali[self.indice(lettera)]

    def lettere_di(self, indice):
        """Le lettere sotto cui sta una versione: due per quella ripetuta, una per le altre."""
        return [lettera for lettera in self.lettere if self.indice(lettera) == indice]


def prepara_prove(gruppi, componi, seme, materiali=None):
    """Le prove dei gruppi, con le lettere mescolate dal seme. componi fa il materiale di una ricetta;
    materiali, se c'è, dà quelli già fatti di ogni gruppo, per chiave: le prove lo usano per non
    rifarli."""
    rng = random.Random(seme)
    prove = []
    for gruppo in gruppi:
        composti = [componi(c.ricetta) for c in gruppo.candidati] if materiali is None else materiali[gruppo.chiave]
        prove.append(Prova(gruppo, composti, *assegna(rng, len(gruppo.candidati))))
    return prove


def rassegna(ascolto, prova):
    """Tutte le lettere una volta, in ordine, ciascuna annunciata prima di suonare, senza domande; la
    prima dopo il silenzio iniziale, perché segue il via del gruppo. Escape la interrompe."""
    print(_("Rassegna."))
    for posto, lettera in enumerate(prova.lettere):
        print(_("Lettera {lettera}.").format(lettera=lettera))
        anticipo = ANTICIPO_PRIMA_LETTERA if posto == 0 else ANTICIPO_RASSEGNA
        if ascolto.suona(prova.materiale(lettera), anticipo, TASTI_DELLA_RASSEGNA) == "\x1b":
            print(_("Rassegna interrotta."))
            return


def tasti_del_voto(prova):
    """I tasti del voto, che valgono anche mentre una lettera suona: spazio, Invio, Escape, n, i voti
    e le lettere del gruppo."""
    return frozenset({" ", "\r", "\x1b", "n", "N", *VOTI, *prova.lettere, *prova.lettere.lower()})


def invito(lettera, voto):
    """Il prompt del voto di una lettera, fra due ritorni carrello ed entro quaranta caratteri, col
    voto già dato se c'è."""
    if voto is None:
        testo = _("{lettera}: spazio ascolta, da 1 a 5 il voto").format(lettera=lettera)
    else:
        testo = _("{lettera}, voto {voto}: spazio ascolta, 1-5 cambia").format(lettera=lettera, voto=voto)
    return f"\r{testo}\r"


def riassunto(prova):
    """I voti del gruppo in una riga, o la mancanza di preferenze."""
    if prova.nessuna_preferenza:
        return _("Nessuna preferenza.")
    voti = [_("{lettera} {voto}").format(lettera=lettera, voto=prova.voti[lettera]) if lettera in prova.voti
            else _("{lettera} senza voto").format(lettera=lettera) for lettera in prova.lettere]
    return _("I voti: {voti}.").format(voti=", ".join(voti))


def voto(ascolto, prova):
    """Il voto, una lettera alla volta, dalla prima. Spazio fa partire la lettera, quante volte si
    vuole, e mentre suona la ferma: nelle scivolate prende il valore come in partita, e la mira
    intera passa alla barra seguente. Il tasto di un'altra lettera fa sentire quella, per confronto;
    da 1 a 5 si vota la lettera del prompt, anche mentre ne suona un'altra, e si passa alla seguente;
    Invio passa oltre senza cambiare il voto; n dichiara nessuna preferenza per tutto il gruppo e
    cancella i voti; Escape chiude il voto. Ogni altro tasto premuto mentre una lettera suona la
    ferma e vale subito."""
    print(_("Il voto, da A a {ultima}: {domanda}; n se non hai preferenze.").format(ultima=prova.lettere[-1], domanda=prova.gruppo.domanda))
    tasti = tasti_del_voto(prova)
    interrompono = tasti - {" "}
    posto = 0
    tasto = None
    while posto < len(prova.lettere):
        lettera = prova.lettere[posto]
        if tasto is None:
            tasto = key(invito(lettera, prova.voti.get(lettera)))
            print()
        scelto, tasto = tasto, None
        if scelto not in tasti:
            continue
        if scelto == " ":
            tasto = ascolto.suona(prova.materiale(lettera), ANTICIPO_VOTO, interrompono, ferma_con_spazio=True)
        elif scelto == "\r":
            posto += 1
        elif scelto == "\x1b":
            break
        elif scelto in ("n", "N"):
            prova.voti.clear()
            prova.nessuna_preferenza = True
            break
        elif scelto in VOTI:
            prova.voti[lettera] = int(scelto)
            prova.nessuna_preferenza = False
            posto += 1
        else:
            tasto = ascolto.suona(prova.materiale(scelto.upper()), ANTICIPO_VOTO, interrompono, ferma_con_spazio=True)
    print(riassunto(prova))


def giro(ascolto, prova):
    """Un giro del gruppo: la rassegna e poi il voto."""
    rassegna(ascolto, prova)
    voto(ascolto, prova)


# I risultati.

class Risultati:
    """Il file degli esiti, accanto al banco: una riga per voce, aggiunta in fondo."""

    def __init__(self, percorso):
        self.percorso = percorso

    def scrivi(self, *righe):
        if righe:
            with open(self.percorso, "a", encoding="utf-8") as file:
                file.writelines(riga + "\n" for riga in righe)


class Annotazioni(comune.Esiti):
    """Il menu di fine gruppo di collaudo_comune, con gli esiti scritti una riga per voce, senza le
    righe vuote del registro comune, e trattenuti finché il gruppo non si chiude: nel file vengono
    dopo i voti del gruppo, a cui si riferiscono."""

    def __init__(self, percorso):
        super().__init__(percorso)
        self.in_sospeso = []

    def registra(self, titolo, commento, misura=""):
        self.in_sospeso.append((titolo, commento, misura))

    def scrivi_in_sospeso(self):
        righe = []
        for titolo, commento, misura in self.in_sospeso:
            riga = _("{titolo}, {data}: {commento}").format(titolo=titolo, data=time.strftime("%Y-%m-%d %H:%M"), commento=commento or _("nessun commento"))
            if misura:
                riga = _("{riga}; misura: {misura}").format(riga=riga, misura=misura)
            righe.append(riga)
        self.in_sospeso.clear()
        Risultati(self.percorso).scrivi(*righe)


def indice_attuale(gruppo):
    """La posizione della versione che il gioco usa oggi, o None se il gruppo non ne ha una."""
    return next((indice for indice, candidato in enumerate(gruppo.candidati) if candidato.attuale), None)


def medie(prova):
    """La media dei voti di ogni versione che ne ha, per indice."""
    risultato = {}
    for indice in range(len(prova.gruppo.candidati)):
        voti = [prova.voti[lettera] for lettera in prova.lettere_di(indice) if lettera in prova.voti]
        if voti:
            risultato[indice] = sum(voti) / len(voti)
    return risultato


def scelta(prova):
    """L'indice della versione preferita, o None. Vince la media più alta, ma una versione diversa
    da quella attuale vince soltanto se l'attuale ha avuto un voto e lei lo supera, perché con i voti
    incompleti una lettera votata da sola non ha battuto niente; a parità resta l'attuale, e fra pari
    vince la prima. Senza voti, senza preferenze o col gruppo chiuso senza giudizio resta l'attuale,
    o nessuna se il gruppo non ne ha."""
    attuale = indice_attuale(prova.gruppo)
    valori = medie(prova)
    if prova.nessuna_preferenza or prova.chiusa or not valori:
        return attuale
    migliore = max(valori.values())
    if attuale is not None and (attuale not in valori or valori[attuale] == migliore):
        return attuale
    return min(indice for indice, media in valori.items() if media == migliore)


def righe_dei_voti(prova):
    """I voti di un gruppo, con le sole lettere: una riga per lettera, o una per la mancanza di preferenze."""
    titolo = prova.gruppo.titolo
    if prova.nessuna_preferenza:
        return [_("{titolo}: nessuna preferenza.").format(titolo=titolo)]
    return [_("{titolo}, lettera {lettera}: {voto}.").format(titolo=titolo, lettera=lettera, voto=prova.voti.get(lettera, _("senza voto")))
            for lettera in prova.lettere]


def _lettere(lettere):
    return ngettext("lettera {lettere}", "lettere {lettere}", len(lettere)).format(lettere=elenco(lettere))


def righe_della_rivelazione(prove):
    """Ciò che si scrive soltanto alla fine, per ogni gruppo iniziato: quale lettera era quale, il
    riepilogo che somma i voti per versione, il conto del controllo e la scelta."""
    righe = []
    for prova in prove:
        if not prova.iniziata:
            continue
        gruppo = prova.gruppo
        titolo = gruppo.titolo
        coppia = prova.lettere_di(prova.ripetuto)
        for lettera in prova.lettere:
            riga = _("{titolo}, la lettera {lettera} era: {nome}").format(titolo=titolo, lettera=lettera, nome=prova.candidato(lettera).nome)
            if lettera in coppia:
                altra = next(x for x in coppia if x != lettera)
                riga = _("{riga}; è la versione ripetuta per controllo, identica alla lettera {altra}").format(riga=riga, altra=altra)
            righe.append(riga + ".")
        if prova.nessuna_preferenza:
            righe.append(_("Riepilogo, {titolo}: nessuna preferenza fra le lettere.").format(titolo=titolo))
        else:
            valori = medie(prova)
            indici = list(range(len(gruppo.candidati)))
            if gruppo.preferenza:
                indici.sort(key=lambda indice, valori=valori: -valori.get(indice, 0.0))
            for indice in indici:
                lettere = prova.lettere_di(indice)
                voti = [prova.voti[lettera] for lettera in lettere if lettera in prova.voti]
                if voti:
                    somma = _("totale {totale} con {quanti}, media {media}").format(totale=sum(voti), quanti=conta(len(voti), "{n} voto", "{n} voti"),
                                                                                media=numero(sum(voti) / len(voti)))
                else:
                    somma = _("nessun voto")
                righe.append(_("Riepilogo, {titolo}, {nome}: {somma}, {lettere}.").format(titolo=titolo, nome=gruppo.candidati[indice].nome, somma=somma,
                                                                                         lettere=_lettere(lettere)))
        if prova.nessuna_preferenza:
            righe.append(_("Controllo, {titolo}: le lettere {coppia} erano identiche campione per campione, e non hai dato preferenze.").format(
                titolo=titolo, coppia=elenco(coppia)))
        else:
            voti_della_coppia = [_("{lettera} {voto}").format(lettera=lettera, voto=prova.voti.get(lettera, _("senza voto"))) for lettera in coppia]
            righe.append(_("Controllo, {titolo}: le lettere {coppia}, identiche campione per campione, hanno avuto {voti}.").format(
                titolo=titolo, coppia=elenco(coppia), voti=elenco(voti_della_coppia)))
        if gruppo.preferenza:
            righe.append(riga_della_scelta(prova))
    return righe


def riga_della_scelta(prova):
    """La scelta di un gruppo, col perché quando resta quella attuale per i voti chiusi, mancanti o
    incompleti."""
    gruppo = prova.gruppo
    titolo = gruppo.titolo
    indice = scelta(prova)
    if indice is None:
        if prova.chiusa:
            return _("Scelta, {titolo}: il gruppo è stato chiuso senza giudizio, e nessuna versione ha la preferenza.").format(titolo=titolo)
        return _("Scelta, {titolo}: senza voti nessuna versione ha la preferenza.").format(titolo=titolo)
    scelto = gruppo.candidati[indice].nome
    if prova.chiusa:
        return _("Scelta, {titolo}: il gruppo è stato chiuso senza giudizio, quindi i voti non contano e resta {scelto}.").format(titolo=titolo, scelto=scelto)
    if prova.nessuna_preferenza or not prova.voti:
        return _("Scelta, {titolo}: senza voti resta {scelto}.").format(titolo=titolo, scelto=scelto)
    senza = [lettera for lettera in prova.lettere if lettera not in prova.voti]
    if not senza:
        return _("Scelta, {titolo}: {scelto}.").format(titolo=titolo, scelto=scelto)
    if len(senza) == 1:
        mancano = _("la lettera {lettera} è rimasta").format(lettera=senza[0])
    else:
        mancano = _("le lettere {lettere} sono rimaste").format(lettere=elenco(senza))
    attuale = indice_attuale(gruppo)
    if attuale is not None and attuale not in medie(prova):
        return _("Scelta, {titolo}: i voti erano incompleti, {mancano} senza voto, e la versione attuale non ne ha avuti, quindi resta lei: {scelto}.").format(
            titolo=titolo, mancano=mancano, scelto=scelto)
    return _("Scelta, {titolo}: {scelto}, anche se i voti erano incompleti: {mancano} senza voto.").format(titolo=titolo, scelto=scelto, mancano=mancano)


# La sessione.

def esegui(ascolto, prova, risultati, annotazioni):
    """Un gruppo intero: la domanda e il materiale, l'annuncio, il giro, il menu di fine gruppo e i
    voti nel file. Se il menu si chiude con Escape, senza giudizio, i voti si scrivono lo stesso ma
    non decidono la scelta."""
    gruppo = prova.gruppo
    print(_("Nel prossimo gruppo la domanda è: {domanda}, con un voto da 1 a 5 per ogni lettera.").format(domanda=gruppo.domanda))
    print(gruppo.materiale)
    if not comune.gruppo(gruppo.titolo, len(prova.lettere), _("lettere")):
        prova.saltata = True
        risultati.scrivi(_("{titolo}: gruppo saltato.").format(titolo=gruppo.titolo))
        return
    prova.iniziata = True
    giro(ascolto, prova)
    prova.chiusa = annotazioni.esito(gruppo.titolo, riproduci=lambda: giro(ascolto, prova)) == "chiuso"
    prova.svolta = True
    risultati.scrivi(*righe_dei_voti(prova))
    annotazioni.scrivi_in_sospeso()


def concludi(prove, risultati, annotazioni):
    """La chiusura della sessione, anche interrotta: i voti del gruppo rimasto a metà, gli esiti in
    sospeso e la rivelazione dei gruppi iniziati."""
    for prova in prove:
        if prova.iniziata and not prova.svolta:
            risultati.scrivi(*righe_dei_voti(prova), _("{titolo}: gruppo interrotto.").format(titolo=prova.gruppo.titolo))
    annotazioni.scrivi_in_sospeso()
    risultati.scrivi(*righe_della_rivelazione(prove))


def leggi_seme(argv, banco):
    """Il seme dalla riga di comando, o uno nuovo; None, con la spiegazione, se non è un numero."""
    if not argv:
        return random.SystemRandom().randrange(1, 1_000_000)
    try:
        return int(argv[0])
    except ValueError:
        print(_("Il seme dev'essere un numero intero, e {valore} non lo è. Uso: python strumenti\\{banco}.py, con un numero per ripetere "
                "le lettere di una sessione passata.").format(valore=argv[0], banco=banco))
        return None


def invito_a_cominciare():
    return "\r{}\r".format(_("Invio per cominciare, Escape per uscire"))


def sessione(nome, seme, percorso, prove, eventi=(), altro=None, righe_in_coda=None):
    """La sessione di un banco, dopo il via: apre l'ascolto, scrive la testa del file, fa i gruppi
    alla cieca e poi altro(ascolto, risultati, annotazioni), se c'è; alla fine, anche se si
    interrompe, scrive la rivelazione e le righe di righe_in_coda(), e chiude il motore.
    Restituisce il codice d'uscita del banco."""
    ascolto = prepara_ascolto(eventi)
    if ascolto is None:
        print(_("La scheda audio non si apre: senza suono il banco non serve, e si chiude."))
        return 1
    risultati = Risultati(percorso)
    annotazioni = Annotazioni(percorso)
    esito = 0
    try:
        risultati.scrivi(_("{nome}, {data}, seme {seme}.").format(nome=nome, data=time.strftime("%Y-%m-%d %H:%M"), seme=seme))
        for prova in prove:
            esegui(ascolto, prova, risultati, annotazioni)
        if altro is not None:
            altro(ascolto, risultati, annotazioni)
    except KeyboardInterrupt:
        print(_("Banco interrotto."))
        esito = 1
    finally:
        try:
            # Anche se la sessione si interrompe, i gruppi iniziati si svelano: senza, i loro voti non
            # direbbero niente.
            concludi(prove, risultati, annotazioni)
            if righe_in_coda is not None:
                risultati.scrivi(*righe_in_coda())
        finally:
            ascolto.chiudi()
    print(_("Fine del banco. Nel file {file}, accanto al banco, trovi i voti, quale lettera era quale e il riepilogo.").format(file=Path(percorso).name))
    return esito


# I gruppi della mira.

DOMANDA_DELLA_SCALA = _("quanto è facile capire dove sei nella scala e fermarla dove vuoi")
MIRA_INTERA = _("In ogni lettera senti la mira intera, le tre scivolate di fila come in partita: la direzione, l'alzo e la potenza.")
NOMI_DEI_PARAMETRI = {"direzione": _("la direzione"), "alzo": _("l'alzo"), "potenza": _("la potenza")}


def _attuale(nome, attuale):
    """Il nome di una versione, con l'avviso che è quella del gioco di oggi."""
    return _("{nome}, quella attuale").format(nome=nome) if attuale else nome


def _direzione():
    candidati = []
    for forma in ("triangolo", "seno", "sega_morbida"):
        attuale = forma == SCIVOLATA_FORMA["direzione"]
        candidati.append(Candidato(_attuale(_("la forma {forma}").format(forma=forma), attuale), (("direzione", {"forma": forma}),), attuale))
    return Gruppo("direzione", _("Mira, la forma della direzione"), DOMANDA_DELLA_SCALA,
                  _("In ogni lettera senti la scivolata della direzione, coi suoi bip, come in partita."), tuple(candidati))


def _alzo():
    candidati = []
    for curva, frequenze in (("esponenziale", ALZO_FREQUENZE), ("esponenziale", ALZO_DI_PRIMA), ("lineare", ALZO_FREQUENZE)):
        attuale = curva == sintesi.CURVE["alzo"][0] and tuple(frequenze) == tuple(ALZO_FREQUENZE)
        nome = _("la scala {curva} da {da} a {a} hertz").format(curva=curva, da=intero(frequenze[0]), a=intero(frequenze[1]))
        candidati.append(Candidato(_attuale(nome, attuale), (("alzo", {"curva": curva, "frequenze": tuple(frequenze)}),), attuale))
    return Gruppo("alzo", _("Mira, la scala dell'alzo"), DOMANDA_DELLA_SCALA, _("In ogni lettera senti la scivolata dell'alzo, coi suoi bip, come in partita."),
                  tuple(candidati))


def _potenza():
    candidati = []
    for curva, minimo in (("decibel", POTENZA_DB_MINIMO), ("decibel", POTENZA_DB_MINIMO_RIDOTTO), ("ampiezza", POTENZA_DB_MINIMO)):
        attuale = curva == sintesi.CURVE["potenza"][0] and minimo == POTENZA_DB_MINIMO
        if curva == "decibel":
            nome = _("la crescita lineare in decibel da {minimo} decibel").format(minimo=intero(minimo))
        else:
            nome = _("la crescita lineare nell'ampiezza da {minimo} decibel").format(minimo=intero(minimo))
        candidati.append(Candidato(_attuale(nome, attuale), (("potenza", {"curva": curva, "db_minimo": minimo}),), attuale))
    return Gruppo("potenza", _("Mira, la scala della potenza"), DOMANDA_DELLA_SCALA,
                  _("In ogni lettera senti la scivolata della potenza, coi suoi bip, come in partita."), tuple(candidati))


def _tacche():
    versioni = (
        (_("le tacche attuali, a {db} decibel").format(db=intero(TACCA_DB)), {}, True),
        (_("le tacche più forti, a {db} decibel").format(db=intero(TACCA_DB + TACCHE_PIU_FORTI_DB)), {"tacca_db": TACCA_DB + TACCHE_PIU_FORTI_DB}, False),
        (_("senza tacche"), {"tacche": False}, False),
    )
    candidati = tuple(Candidato(nome, tuple((parametro, impostazioni) for parametro in PARAMETRI), attuale) for nome, impostazioni, attuale in versioni)
    return Gruppo("tacche", _("Mira, le tacche"), DOMANDA_DELLA_SCALA, MIRA_INTERA, candidati)


def _bip():
    candidati = []
    for secondi in (BIP_DURATA, BIP_LUNGHI):
        attuale = secondi == BIP_DURATA
        nome = _attuale(conta(round(secondi * 1000), "la durata dei bip di {n} millesimo", "la durata dei bip di {n} millesimi"), attuale)
        candidati.append(Candidato(nome, tuple((parametro, {"bip_durata": secondi}) for parametro in PARAMETRI), attuale))
    return Gruppo("bip", _("Mira, i bip del conto"), _("quanto è facile capire, dai bip, quando parte la scivolata e di quale parametro è"),
                  MIRA_INTERA, tuple(candidati))


GRUPPI = (_direzione(), _alzo(), _potenza(), _tacche(), _bip())


def componi(ricetta):
    """Il materiale di una versione della mira: le sue scivolate, fatte da sintesi.scivolata come
    nel gioco."""
    return tuple(sintesi.scivolata(parametro, **impostazioni) for parametro, impostazioni in ricetta)


def prepara(seme, materiali=None):
    return prepara_prove(GRUPPI, componi, seme, materiali)


# La prova di precisione.

class Tentativo(NamedTuple):
    """Un tentativo della prova di precisione: la serie, il posto nella serie, il riferimento, il
    valore preso, il suo esito e se è venuto dallo spazio."""

    serie: int
    posto: int
    riferimento: float
    valore: float
    esito: str
    premuto: bool

    @property
    def errore(self):
        """Il valore scelto meno il riferimento: positivo vuol dire oltre, cioè in ritardo."""
        return self.valore - self.riferimento


class Precisione:
    """La prova di precisione di un parametro: la scivolata del gioco, il generatore dei riferimenti
    e i tentativi fatti."""

    def __init__(self, parametro, rng):
        self.parametro = parametro
        self.rng = rng
        self.titolo = _("Precisione, {parametro}").format(parametro=NOMI_DEI_PARAMETRI[parametro])
        self.scivolata = sintesi.scivolata(parametro)
        self.tentativi = []
        self.serie = 0
        self.saltata = False
        self.iniziata = False
        self.svolta = False
        self.chiusa = False

    def riferimenti(self):
        """I riferimenti di una serie nuova: uno a caso in ogni ottavo dell'intervallo RIFERIMENTI,
        in ordine mescolato, così una serie copre tutta la scala e non si indovina il prossimo."""
        da, a = RIFERIMENTI
        valori = [da + (a - da) * (posto + self.rng.random()) / TENTATIVI for posto in range(TENTATIVI)]
        self.rng.shuffle(valori)
        return valori


def prepara_precisione(seme):
    """Le prove di precisione dei tre parametri, ognuna col suo generatore, nato dal seme."""
    return [Precisione(parametro, random.Random(f"{seme}-precisione-{parametro}")) for parametro in PARAMETRI]


def tono_di_riferimento(parametro, frazione, fs=FS):
    """Il suono di riferimento: un tono fermo di DURATA_DEL_RIFERIMENTO, che è la scivolata attuale
    del parametro ferma nel punto frazione della scala, con le formule di sintesi.scivolata. La
    direzione ha la panoramica di quel punto, l'alzo la sua frequenza sulla scala esponenziale, la
    potenza il suo livello lineare in decibel; la forma e il livello sono quelli della scivolata."""
    pan = 0.0
    livello = SCIVOLATA_DB
    if parametro == "direzione":
        hertz = DIREZIONE_FREQUENZA
        pan = 2.0 * frazione - 1.0
    elif parametro == "alzo":
        da, a = ALZO_FREQUENZE
        hertz = da * (a / da) ** frazione
    else:
        hertz = POTENZA_FREQUENZA
        livello = SCIVOLATA_DB + POTENZA_DB_MINIMO * (1.0 - frazione)
    campioni = round(DURATA_DEL_RIFERIMENTO * fs)
    onda = sintesi.Oscillatore(SCIVOLATA_FORMA[parametro], fs).rendi(hertz, campioni) * sintesi.db_in_lineare(livello)
    rampa = min(round(RAMPA_DEL_RIFERIMENTO * fs), campioni // 2)
    if rampa > 0:
        sale = 0.5 - 0.5 * np.cos(np.pi * (np.arange(rampa) + 0.5) / rampa)
        onda[:rampa] *= sale
        onda[-rampa:] *= sale[::-1]
    return sintesi.panoramica(onda, pan)


def invito_al_tentativo(posto, totale):
    """Il prompt di un tentativo, fra due ritorni carrello ed entro quaranta caratteri."""
    return "\r{}\r".format(_("Tentativo {posto} di {totale}: spazio comincia").format(posto=posto, totale=totale))


def serie(ascolto, precisione):
    """Una serie di tentativi. Per ognuno spazio comincia ed Escape interrompe la serie; poi, dopo
    un poco di silenzio, il riferimento, la pausa fra le barre e la scivolata del gioco, che lo spazio
    ferma col valore di mira.valore_scelto. Il primo tentativo ha anche il silenzio iniziale."""
    precisione.serie += 1
    riferimenti = precisione.riferimenti()
    for posto, riferimento in enumerate(riferimenti, start=1):
        tasto = None
        while tasto not in (" ", "\x1b"):
            tasto = key(invito_al_tentativo(posto, len(riferimenti)))
            print()
        if tasto == "\x1b":
            print(_("Serie interrotta."))
            return
        anticipo = ANTICIPO_TENTATIVO + (SILENZIO_INIZIALE if posto == 1 else 0.0)
        testa = np.concatenate([ascolto.silenzio(anticipo), tono_di_riferimento(precisione.parametro, riferimento, ascolto.fs), ascolto.silenzio(PAUSA_FRA_BARRE)])
        tasto, valori = ascolto.suona_mira((precisione.scivolata,), 0.0, TASTI_DEL_TENTATIVO, ferma_con_spazio=True, teste=(testa,))
        if valori:
            preso = valori[0]
            precisione.tentativi.append(Tentativo(precisione.serie, posto, riferimento, preso.valore, preso.esito, preso.premuto))
        if tasto == "\x1b":
            print(_("Serie interrotta."))
            return


def esegui_precisione(ascolto, precisione, risultati, annotazioni):
    """La prova di precisione di un parametro: la spiegazione, l'annuncio, la serie, il menu di fine
    gruppo, dove r fa un'altra serie, e il conto dei tentativi nel file; i numeri vengono alla fine."""
    print(_("Nella prossima prova, {parametro}, per ogni tentativo spazio comincia: senti un breve suono di riferimento, poi la mira vera, "
            "coi bip e la scivolata, e la fermi con spazio dove senti il riferimento. Escape interrompe la serie.").format(
        parametro=NOMI_DEI_PARAMETRI[precisione.parametro]))
    if not comune.gruppo(precisione.titolo, TENTATIVI, _("tentativi")):
        precisione.saltata = True
        risultati.scrivi(_("{titolo}: prova saltata.").format(titolo=precisione.titolo))
        return
    precisione.iniziata = True
    serie(ascolto, precisione)
    precisione.chiusa = annotazioni.esito(precisione.titolo, riproduci=lambda: serie(ascolto, precisione)) == "chiuso"
    precisione.svolta = True
    risultati.scrivi(_("{titolo}: {quanti}.").format(titolo=precisione.titolo,
                                                   quanti=conta(len(precisione.tentativi), "{n} tentativo fatto", "{n} tentativi fatti")))
    annotazioni.scrivi_in_sospeso()


class Misura(NamedTuple):
    """I conti di un insieme di tentativi. Gli errori contano soltanto i tentativi fermati nella
    scala: uno spazio nei bip o una scivolata arrivata in fondo dicono altro, e si contano a parte.
    medio, assoluto e deviazione sono None senza tentativi fermati; la deviazione standard è quella
    del campione, zero con un tentativo solo."""

    totale: int
    fermati: int
    medio: float | None
    assoluto: float | None
    deviazione: float | None
    minimi: int
    massimi: int


def misura(tentativi):
    errori = [t.errore for t in tentativi if t.esito == "scelto"]
    medio = assoluto = deviazione = None
    if errori:
        medio = statistics.fmean(errori)
        assoluto = statistics.fmean(abs(e) for e in errori)
        deviazione = statistics.stdev(errori) if len(errori) > 1 else 0.0
    return Misura(len(tentativi), len(errori), medio, assoluto, deviazione, sum(1 for t in tentativi if t.esito == "minimo"),
                  sum(1 for t in tentativi if t.esito == "massimo"))


def _con_segno(frazione):
    """Una frazione della scala come percentuale col segno: +3,7%, -1,2%, 0,0%."""
    testo = percento(frazione, 1)
    if frazione > 0 and not testo.startswith("0,0"):
        return "+" + testo
    return testo


def _millesimi(frazione):
    """Una frazione della scala in millesimi di secondo della scivolata."""
    return conta(round(abs(frazione) * SCIVOLATA_DURATA * 1000), "{n} millesimo", "{n} millesimi")


def _ritardo(frazione):
    """Una frazione della scala in millesimi, col verso: in ritardo se positiva, in anticipo se negativa."""
    millesimi = round(abs(frazione) * SCIVOLATA_DURATA * 1000)
    if millesimi == 0:
        return _("nessun ritardo")
    if frazione > 0:
        return _("{millesimi} in ritardo").format(millesimi=_millesimi(frazione))
    return _("{millesimi} in anticipo").format(millesimi=_millesimi(frazione))


def _esito_del_tentativo(tentativo):
    if tentativo.esito == "scelto":
        return _("fermata nella scala")
    if tentativo.esito == "minimo":
        return _("spazio nei bip, vale il minimo")
    if tentativo.premuto:
        return _("spazio dopo la fine, vale il massimo")
    return _("non fermata, vale il massimo")


def riga_del_tentativo(precisione, tentativo):
    return _("{titolo}, serie {serie}, tentativo {posto}: riferimento {riferimento}, scelto {valore}, errore {errore}, cioè {tempo}; {esito}.").format(
        titolo=precisione.titolo, serie=tentativo.serie, posto=tentativo.posto, riferimento=percento(tentativo.riferimento, 1),
        valore=percento(tentativo.valore, 1), errore=_con_segno(tentativo.errore), tempo=_ritardo(tentativo.errore), esito=_esito_del_tentativo(tentativo))


def riga_del_riepilogo(titolo, tentativi):
    conti = misura(tentativi)
    estremi = _("{minimi} al minimo, {massimi} al massimo").format(minimi=conti.minimi, massimi=conti.massimi)
    fermati = conta(conti.fermati, "{n} tentativo fermato nella scala", "{n} tentativi fermati nella scala")
    if not conti.fermati:
        return _("{titolo}, riepilogo: nessun tentativo fermato nella scala su {totale}; {estremi}.").format(titolo=titolo, totale=conti.totale, estremi=estremi)
    return _("{titolo}, riepilogo: {fermati} su {totale}; errore medio {medio}, cioè {medio_tempo}; medio assoluto {assoluto}, cioè {assoluto_tempo}; "
             "deviazione standard {deviazione}, cioè {deviazione_tempo}; {estremi}.").format(
        titolo=titolo, fermati=fermati, totale=conti.totale, medio=_con_segno(conti.medio), medio_tempo=_ritardo(conti.medio),
        assoluto=percento(conti.assoluto, 1), assoluto_tempo=_millesimi(conti.assoluto), deviazione=percento(conti.deviazione, 1),
        deviazione_tempo=_millesimi(conti.deviazione), estremi=estremi)


def righe_della_precisione(precisioni):
    """I conti della precisione, alla fine: come si leggono, ogni tentativo, il riepilogo di ogni
    parametro e quello di tutti insieme."""
    fatte = [p for p in precisioni if p.tentativi]
    if not fatte:
        return []
    righe = [_("Precisione: l'errore è il valore scelto meno il riferimento, in percentuale della scala e in millesimi della scivolata, che dura {durata}; "
               "positivo vuol dire in ritardo, negativo in anticipo. La compensazione del tasto era di {compensazione}.").format(
        durata=durata(SCIVOLATA_DURATA), compensazione=conta(round(mira.COMPENSAZIONE_DEL_TASTO * 1000), "{n} millesimo", "{n} millesimi"))]
    for precisione in fatte:
        righe.extend(riga_del_tentativo(precisione, tentativo) for tentativo in precisione.tentativi)
        righe.append(riga_del_riepilogo(precisione.titolo, precisione.tentativi))
    if len(fatte) > 1:
        righe.append(riga_del_riepilogo(_("Precisione, tutti i parametri"), [t for p in fatte for t in p.tentativi]))
    return righe


# Il banco.

def main(argv=None):
    seme = leggi_seme(sys.argv[1:] if argv is None else argv, "banco_mira")
    if seme is None:
        return 1
    prove = prepara(seme)
    precisioni = prepara_precisione(seme)
    print(_("Banco della mira di Ciechino: {gruppi} alla cieca, poi la prova di precisione della direzione, dell'alzo e della potenza.").format(
        gruppi=conta(len(prove), "{n} gruppo", "{n} gruppi")))
    print(_("Nei gruppi alla cieca senti la stessa cosa in più versioni, ciascuna con la sua lettera. Prima la rassegna, tutte le lettere una volta, "
            "senza domande; Escape la interrompe."))
    print(_("Poi il voto, una lettera alla volta: spazio fa partire la lettera e, mentre la scivolata scorre, la ferma come in partita; il tasto di "
            "un'altra lettera fa sentire quella; da 1 a 5 dai il voto, 5 il migliore, e passi alla seguente; Invio passa oltre; n vuol dire nessuna "
            "preferenza; Escape chiude il voto."))
    print(_("Nella prova di precisione, a ogni tentativo, senti un breve suono di riferimento e poi la mira vera: ferma la scivolata con spazio dove "
            "senti il riferimento."))
    print(_("A fine gruppo: r ripete, c commenta, Invio lo dà per superato, Escape lo chiude senza giudizio, e allora i suoi voti non contano."))
    print(_("Gli esiti vanno nel file {file}, accanto al banco.").format(file=FILE_DEGLI_ESITI.name))
    # enter_escape e gruppo vanno a capo da sé: un print in più farebbe una riga vuota. Il silenzio
    # prima del primo suono sta nei buffer, dopo il via di ogni gruppo.
    if not enter_escape(invito_a_cominciare()):
        return 0

    def precisione(ascolto, risultati, annotazioni):
        for prova in precisioni:
            esegui_precisione(ascolto, prova, risultati, annotazioni)

    return sessione(_("Banco della mira"), seme, FILE_DEGLI_ESITI, prove, ("valore_preso",), precisione, lambda: righe_della_precisione(precisioni))


if __name__ == "__main__":
    sys.exit(main())
