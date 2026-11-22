"""
La partita di Ciechino: la macchina delle fasi, dall'ascolto alla mira, dal volo al resoconto, con la
riserva dei colpi, i livelli, i bonus, la pausa e i conti delle statistiche.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Niente wx: la finestra chiama spazio, pausa, pausa_automatica, nuova e abbandona ai tasti e agli eventi
delle finestre, battito a ogni giro del suo timer, e mette in coda alla vista i Messaggio che riceve.
Il tempo di gioco lo tiene il motore audio, e la partita non ha orologi paralleli: ferma e fa scorrere
il tempo del motore, e legge il campione udibile per sapere che cosa si sente in quel momento.
L'orologio che riceve serve soltanto alla durata vera della partita, per la classifica e le
statistiche, e non conta le pause.
I fili: tutto qui gira nel filo principale, tranne le funzioni passate a motore.programma, che il
callback chiama nel filo dell'audio al campione esatto. Quelle mandano soltanto comandi al motore,
con buffer preparati prima, e non toccano lo stato della partita: la partita si accorge di ciò che è
successo al battito, dall'evento scattato e dal campione udibile, così il resoconto arriva quando
l'impatto si è davvero sentito.
Gli avvisi della partita, come il livello nuovo, il bonus che appare, l'ultimo colpo e la fine,
suonano in fila, dopo l'effetto dell'impatto che li ha causati e uno dopo l'altro: partiti allo
stesso campione, o sopra il rumore dell'impatto, si fonderebbero in un suono solo, e nessuno si
riconoscerebbe. La finestra mette in fila anche l'esito della classifica, dopo la fine, con
dopo_gli_avvisi.
La pausa (D14, decisione di Gabriele del 19 novembre 2026) ferma tutto e fa tacere tutto: il tempo di
gioco si ferma e le voci di gioco vanno a guadagno zero; nella mira i valori si fermano dove sono, i
loro tasti valgono come lasciati, e il tempo della mira, dove c'è, aspetta la ripresa. La chiede chi
gioca con Ctrl+P, col suo suono e il suo messaggio, oppure la mette la
finestra, in silenzio, quando si apre un dialogo o lei perde l'attivazione: questa seconda, e solo
questa, la toglie la finestra al ritorno. Ciò che succede mentre la partita è ferma, come un tiro il
cui impatto era già partito quando la pausa è arrivata, si dice subito nella vista, ma gli avvisi e
il bonus nuovo aspettano la ripresa.
Il proiettile ha la forma d'onda che chi gioca sceglie nelle Impostazioni (D17), e i bersagli una
portante a caso fra BERSAGLIO_PORTANTI, tolta quella del proiettile, perché non si confondano con lui
(D18): la finestra dà la forma con scegli_proiettile, e una partita tiene quella con cui è cominciata
fino alla fine. La forma scelta a partita in corso vale dalla partita seguente.
Il resoconto di un tiro, coi messaggi che lo seguono, arriva quando l'effetto dell'impatto ha finito di
suonare (D22, dal collaudo di Gabriele del 19 novembre 2026): il tiro si risolve quando l'impatto si
sente, ma i messaggi aspettano in fila la fine del jingle, dell'impatto o del tonfo, e li porta il
battito. I primi livelli sono la palestra delle attese, fino a PALESTRA_FINO_AL_LIVELLO, contando il
livello del tiro: dopo ogni tiro la partita passa nella fase «attesa», col tempo di gioco fermo; finiti
l'effetto, gli avvisi in fila e il resoconto, si ferma e tace come in pausa, finché lo spazio non la
fa ripartire, chiamando i bersagli nuovi dopo un centro o riportando all'ascolto dopo un tiro a vuoto.
Soltanto nella palestra Ctrl+P mette la pausa; quella automatica della finestra c'è sempre.
La mira manuale (D29, chiesta da Gabriele il 22 novembre 2026): lo spazio nell'ascolto avvia la mira, una
sola per i tre valori, tutti al centro della loro scala, e la finestra chiama muovi e lascia ai tasti
che scendono e salgono, le frecce orizzontali per la direzione, quelle verticali per l'alzo, Home e Fine
per la potenza: ogni valore si muove finché il suo tasto è giù, col suo suono, le sue tacche a ogni
decimo e il suono fine_scala ai capi. Invio, con spara, fa partire il colpo coi valori dell'istante in
cui si è sentito; Esc, con annulla_mira, annulla la mira e conserva il proiettile. Fuori dalla palestra
la mira è a tempo, tempo_della_mira(livello), con un tick al secondo sempre più acuto, e allo scadere
il colpo parte da solo. Durante la mira il tempo di gioco scorre e i bersagli suonano pieni: i mobili e
il bonus si muovono, e la mira va su dove saranno all'arrivo del proiettile. Da D24 il resoconto arriva
un secondo dopo la fine dell'effetto, e da D26 ogni tiro ha la sua precisione globale, che la partita
somma per la media delle statistiche.
"""

import functools
import math
import random
import time

import numpy as np

import fisica
import mira
import resoconti
from bersagli import genera, genera_bonus, portanti_senza, quanti_bersagli, raggio_del_livello, velocita_del_livello
from costanti import (
    ANTICIPO_DELLO_SPARO,
    CENTRI_PER_LIVELLO,
    COLPI_BONUS,
    COLPI_INIZIALI,
    COLPI_PER_LIVELLO,
    ESPONENTE_DEL_RAGGIO,
    GRAZIA_DELL_ATTESA,
    LIVELLO_DEI_BONUS,
    MOLTIPLICATORE_BONUS,
    PALESTRA_FINO_AL_LIVELLO,
    PARAMETRI,
    PREMIO_PRECISIONE,
    PROBABILITA_BONUS,
    PROIETTILE_FORMA,
    PROIETTILE_FORME,
    PUNTATORE_FORMA,
    PUNTI_BASE,
    RAGGIO_DI_RIFERIMENTO,
    RIDUZIONE_DEL_TEMPO_DELLA_MIRA,
    RITARDO_DEL_RESOCONTO,
    SFUMATURA_STOP,
    STACCO_FRA_GLI_AVVISI,
    TACCHE,
    TEMPO_DELLA_MIRA_INIZIALE,
    TEMPO_DELLA_MIRA_MINIMO,
    VELOCITA_DEL_TEMPO,
    VELOCITA_DI_RIFERIMENTO,
)
from motore_audio import VoceTono
from resoconti import Messaggio
from sintesi import pan_della_direzione, panoramica, parametri_del_puntatore, tacca_mono, tick_del_tempo
from volo import calcola_volo, parametri_del_proiettile

__all__ = ["CONTI_DEL_PROIETTILE", "FASI", "Messaggio", "Partita", "esempio_del_proiettile", "nella_palestra", "punti_del_centro", "tempo_della_mira",
           "volo_d_esempio"]

# Le fasi della partita, come le vuole il contratto. «attesa» è quella dopo un tiro nella palestra
# (D22), fino allo spazio che fa ripartire la partita.
FASI = ("pronta", "ascolto", "mira", "volo", "attesa", "finita")
# Le fasi in cui una partita è in corso: la finestra le usa per chiedere se abbandonarla.
IN_CORSO = ("ascolto", "mira", "volo", "attesa")
# I suoni con cui la finestra dice, dopo la fine, se il punteggio entra in classifica: li mette in fila
# a dopo_gli_avvisi, e la partita nuova chiesta subito dopo li lascia suonare.
ESITI_DELLA_CLASSIFICA = ("in_classifica", "fuori_classifica")
# Il contatore delle statistiche che ogni partita nuova accresce, secondo la forma del suo proiettile:
# così le statistiche dicono quante partite si giocano con ogni forma (D17).
CONTI_DEL_PROIETTILE = {"seno": "partite_col_seno", "triangolo": "partite_col_triangolo", "sega_morbida": "partite_con_la_sega_morbida"}
# L'esempio del proiettile che il dialogo delle Impostazioni fa sentire (D17): tanti secondi di volo
# di un tiro al mezzo della cupola, dallo sparo, e una sfumatura finale, in secondi, perché l'esempio
# non si tronchi con un click.
DURATA_DELL_ESEMPIO = 1.5
SFUMATURA_DELL_ESEMPIO = 0.1


def punti_del_centro(bersaglio, precisione, livello=None):
    """I punti di un centro, con la formula di docs/progetto.txt: PUNTI_BASE, per il rapporto fra
    RAGGIO_DI_RIFERIMENTO e il raggio alla ESPONENTE_DEL_RAGGIO, per 1 più la velocità del bersaglio
    divisa per VELOCITA_DI_RIFERIMENTO, per 1 più PREMIO_PRECISIONE per la precisione, arrotondato
    all'intero. La velocità è quella scalare, così un mobile vale per quanto corre.
    Un bonus vale esattamente MOLTIPLICATORE_BONUS volte i punti di un bersaglio normale del livello in
    cui è nato, col raggio e la velocità di quel livello e lo stesso premio di precisione (D13,
    decisione di Gabriele del 19 novembre 2026): il raggio ridotto e la corsa della traversata non
    contano, così «vale 5 volte» è vero a ogni livello. Per un bonus livello è quindi obbligatorio;
    per gli altri bersagli non serve."""
    precisione = 0.0 if precisione is None else min(1.0, max(0.0, float(precisione)))
    if bersaglio.bonus:
        if livello is None:
            raise ValueError("i punti di un bonus dipendono dal livello in cui è nato: manca livello")
        raggio, velocita = raggio_del_livello(livello), velocita_del_livello(livello)
    else:
        raggio, velocita = bersaglio.raggio, bersaglio.velocita_scalare
    valore = PUNTI_BASE * (RAGGIO_DI_RIFERIMENTO / raggio) ** ESPONENTE_DEL_RAGGIO
    valore *= 1.0 + velocita / VELOCITA_DI_RIFERIMENTO
    valore *= 1.0 + PREMIO_PRECISIONE * precisione
    punti = round(valore)
    if bersaglio.bonus:
        # I punti del normale si arrotondano prima di moltiplicarli: così il bonus vale proprio
        # MOLTIPLICATORE_BONUS volte quelli che il normale avrebbe dato, al punto.
        punti = round(punti * MOLTIPLICATORE_BONUS)
    return punti


def nella_palestra(livello):
    """Vero se il livello è della palestra delle attese (D22), da 1 a PALESTRA_FINO_AL_LIVELLO: lì dopo
    ogni tiro la partita aspetta lo spazio, Ctrl+P mette la pausa e la mira non ha tempo (D29)."""
    return livello <= PALESTRA_FINO_AL_LIVELLO


def tempo_della_mira(livello):
    """I secondi della mira al livello dato, o None nella palestra, dove la mira non ha tempo (D29):
    TEMPO_DELLA_MIRA_INIZIALE al primo livello dopo la palestra, RIDUZIONE_DEL_TEMPO_DELLA_MIRA in meno a
    ogni livello seguente, mai sotto TEMPO_DELLA_MIRA_MINIMO."""
    if nella_palestra(livello):
        return None
    oltre = livello - PALESTRA_FINO_AL_LIVELLO - 1
    return max(TEMPO_DELLA_MIRA_MINIMO, TEMPO_DELLA_MIRA_INIZIALE - RIDUZIONE_DEL_TEMPO_DELLA_MIRA * oltre)


def _sentito(evento, udibile):
    """Vero se l'evento programmato è scattato e il campione che esce dalle casse lo ha passato."""
    return evento is not None and evento.scattato and evento.campione is not None and udibile >= evento.campione


def _forma_del_proiettile(forma):
    """La forma del proiettile, controllata: una di PROIETTILE_FORME, altrimenti ValueError."""
    if forma not in PROIETTILE_FORME:
        raise ValueError(f"Forma del proiettile sconosciuta: {forma!r}; le forme sono {', '.join(PROIETTILE_FORME)}.")
    return forma


@functools.cache
def volo_d_esempio():
    """Il volo dell'esempio del proiettile (D17): il tiro più dolce di fisica.soluzione verso il
    mezzo della cupola, senza bersagli, sparato al tempo di gioco zero, fino a terra. Si calcola una
    volta sola, e chi lo riceve non lo cambia."""
    valori = fisica.soluzione(fisica.MEZZO_DELLA_CUPOLA)
    if valori is None:
        raise ValueError("Il mezzo della cupola non si raggiunge con un tiro dentro i margini: l'esempio del proiettile non ha un volo.")
    traiettoria = fisica.Traiettoria(*fisica.parametri_del_tiro(*valori))
    return calcola_volo(traiettoria, [], 0.0)


def esempio_del_proiettile(motore, forma, durata=DURATA_DELL_ESEMPIO):
    """L'esempio del proiettile di quella forma, per il dialogo delle Impostazioni (D17): la stessa
    voce del gioco, una VoceTono coi parametri sonori del volo d'esempio, che parte dal prossimo
    campione sicuro del motore e dopo durata secondi si spegne, con SFUMATURA_DELL_ESEMPIO. Il volo
    segue il tempo dell'audio, alla VELOCITA_DEL_TEMPO del gioco, non il tempo di gioco, che col
    dialogo aperto è fermo. La voce entra nel motore e si restituisce: chi la vuole fermare prima la
    passa a motore.ferma, e la fermata più vicina vince. Una forma fuori da PROIETTILE_FORME solleva
    ValueError, prima di toccare il motore."""
    forma = _forma_del_proiettile(forma)
    volo = volo_d_esempio()
    suono_del_volo = parametri_del_proiettile(volo)
    inizio = motore.prossimo_campione()
    sparo = inizio / motore.fs

    def parametri(tempi_di_gioco, tempi_audio):
        audio = np.atleast_1d(np.asarray(tempi_audio, dtype=float))
        return suono_del_volo(volo.partenza + (audio - sparo) * VELOCITA_DEL_TEMPO)

    voce = VoceTono(parametri, forma, inizio)
    motore.aggiungi(voce)
    sfumatura = round(SFUMATURA_DELL_ESEMPIO * motore.fs)
    motore.ferma(voce, inizio + max(round(durata * motore.fs) - sfumatura, 0), SFUMATURA_DELL_ESEMPIO)
    return voce


def _suono_del_puntatore(assi, tempi):
    """(frequenze, pan, guadagni lineari, in moto) del puntatore ai tempi dati, dell'audio, per i tre Asse
    nell'ordine di PARAMETRI, senza l'inviluppo che lo apre e lo chiude (D31): i parametri di
    sintesi.parametri_del_puntatore coi valori di quegli istanti, e vero dove almeno un valore si muove.
    La voce del puntatore tiene i suoi assi, e non legge la partita, che a fine mira la dimentica."""
    tempi = np.atleast_1d(np.asarray(tempi, dtype=float))
    frequenze, pan, guadagni = parametri_del_puntatore(*(asse.valore(tempi) for asse in assi))
    in_moto = np.zeros(tempi.shape, dtype=bool)
    for asse in assi:
        in_moto |= asse.in_moto(tempi)
    return frequenze, pan, guadagni, in_moto


def _rampa(bersaglio, partenza, passo):
    """L'inviluppo che insegue il bersaglio, un array di vero e falso, partendo da partenza e cambiando al più
    di passo a ogni campione, verso uno dove il bersaglio è vero e verso zero dove è falso: così il puntatore
    entra ed esce senza click (D31). Restituisce l'inviluppo e il suo ultimo valore, da cui riparte il blocco
    seguente."""
    bersaglio = np.asarray(bersaglio, dtype=float)
    uscita = np.empty(len(bersaglio))
    valore = float(partenza)
    inizio = 0
    for fine in [*(np.flatnonzero(np.diff(bersaglio)) + 1), len(bersaglio)]:
        if fine <= inizio:
            continue
        meta = float(bersaglio[inizio])
        passi = passo * np.arange(1, fine - inizio + 1)
        tratto = np.minimum(meta, valore + passi) if meta >= valore else np.maximum(meta, valore - passi)
        uscita[inizio:fine] = tratto
        valore = float(tratto[-1])
        inizio = fine
    return uscita, valore


class Partita:
    """Una partita di Ciechino, dall'inizio alla fine, e poi la seguente.
    motore è il Motore audio, suoni i Suoni che suonano sul motore; statistiche, se c'è, riceve ogni
    conto e la funzione dei suoni si collega a lei; rng è il random.Random dei bersagli, uno nuovo se
    è None, così le prove ne danno uno col seme; orologio dà i secondi veri, time.monotonic se è None,
    perché una durata non deve saltare quando Windows rimette l'ora.
    Gli attributi pubblici sono quelli del contratto: fase, in_pausa, livello, punti, riserva,
    centri_del_livello, tiri, centri, bersagli (il bonus compreso), valori, i tre valori dell'ultimo
    tiro, e ultimo_volo; e in più serie, bonus, il bersaglio bonus in volo o None, e mira, la
    mira.Mira in corso o None.
    Oltre ai metodi del contratto ci sono pausa_automatica, la pausa silenziosa che mette e toglie la
    finestra (D14), e dopo_gli_avvisi, che dice alla finestra da quale campione far partire l'esito
    della classifica, in fila dopo il suono della fine.
    Il proiettile (D17): proiettile è la forma d'onda dell'unico proiettile della partita in corso, o
    di quella che comincerà se nessuna è in corso; proiettile_scelto quella che chi gioca ha chiesto,
    da cui ogni partita nuova prende la sua. Si sceglie con scegli_proiettile, o all'inizio con
    l'argomento proiettile, PROIETTILE_FORMA se è None.
    La palestra delle attese (D22): nella fase «attesa» attesa_ferma dice se la partita si è già
    fermata e aspetta lo spazio, e nasceranno quanti bersagli nuovi chiamerà lo spazio, zero dopo un
    tiro a vuoto o un centro sul bonus. messaggi_in_attesa dà subito i messaggi che aspettano la fine
    dell'effetto dell'impatto, per la finestra che si chiude o comincia una partita nuova.
    La mira manuale (D29): muovi e lascia ai tasti dei valori, spara all'Invio, annulla_mira all'Esc;
    tasti_giu dice quali tasti la mira crede giù, perché la finestra controlli che lo siano ancora, e
    secondi_della_mira quanti secondi restano a una mira a tempo, None senza tempo."""

    def __init__(self, motore, suoni, statistiche=None, rng=None, orologio=None, proiettile=None):
        self.motore = motore
        self.suoni = suoni
        self.statistiche = statistiche
        self.rng = random.Random() if rng is None else rng
        self.orologio = time.monotonic if orologio is None else orologio
        self.proiettile_scelto = _forma_del_proiettile(PROIETTILE_FORMA if proiettile is None else proiettile)
        self.proiettile = self.proiettile_scelto
        if statistiche is not None:
            suoni.conta = statistiche.suono
        # Le tacche dei tre valori si preparano una volta sola, mono: sintetizzarle a ogni tasto farebbe
        # aspettare il giocatore, e sono sempre le stesse; il punto del puntatore lo prendono in fila (D31).
        self._tacche = {parametro: {k: tacca_mono(parametro, k, fs=motore.fs) for k in range(1, TACCHE)} for parametro in PARAMETRI}
        self.fase = "pronta"
        self.in_pausa = False
        self.livello = 0
        self.punti = 0
        self.riserva = 0
        self.centri_del_livello = 0
        self.tiri = 0
        self.centri = 0
        self.serie = 0
        self.bersagli = []
        self.valori = []
        self.ultimo_volo = None
        self.bonus = None
        # La mira manuale in corso (D29), e i suoi suoni: il puntatore, la voce sola che porta i tre valori
        # (D31), le tacche e il suono del capo messi in fila davanti a ogni valore, i tick del tempo; il
        # campione in cui scade la mira a tempo, i secondi che le restavano all'ultima ripartenza e il
        # campione udibile di quella ripartenza.
        self.mira = None
        self._voce_del_puntatore = None
        self._in_fila_degli_assi = {parametro: [] for parametro in PARAMETRI}
        self._capi_in_fila = {}
        self._tick_in_fila = []
        self._scadenza = None
        self._secondi_rimasti = None
        self._mira_dal = None
        # I secondi di mira già passati prima dell'ultima ripartenza, e il campione da cui la mira corre di
        # nuovo: le pause non contano, come nella durata della partita.
        self._secondi_di_mira = 0.0
        self._mira_corre_dal = None
        # La precisione globale dei tiri della partita (D26), sommata per la media delle statistiche.
        self._somma_globale = 0.0
        self._tiri_globali = 0
        # Le voci dei bersagli, una per bersaglio: Bersaglio si confronta per identità, quindi fa da chiave.
        self._voci = {}
        self._voce_proiettile = None
        self._evento_impatto = None
        # L'effetto dell'impatto in volo e la sua lunghezza in campioni: si conta quando si sente, e gli
        # avvisi che il tiro causa partono dopo di lui.
        self._effetto_impatto = None
        self._lunghezza_impatto = 0
        self._evento_bonus = None
        # Il livello in cui è nato il bonus in volo: i suoi punti sono quelli di un normale di quel livello (D13).
        self._livello_del_bonus = None
        self._numero = 0
        # Il primato della serie torna a ogni centro che allunga la serie oltre il vecchio: si dice una
        # volta sola per serie, e un tiro a vuoto riapre l'annuncio.
        self._serie_detta = False
        # Il campione da cui può partire il prossimo avviso in fila: la fine dell'ultimo, più lo stacco.
        self._fine_degli_avvisi = 0
        # La pausa: se l'ha messa la finestra, e ciò che aspetta la ripresa, gli avvisi e il bonus.
        self._pausa_automatica = False
        self._avvisi_rimandati = []
        self._bonus_rimandato = False
        # L'attesa della palestra (D22): se la partita si è già fermata, quanti bersagli chiamerà lo
        # spazio, e il campione udibile da cui la partita è ferma, per la grazia dello spazio.
        self.attesa_ferma = False
        self.nasceranno = 0
        self._attesa_dal = None
        # I messaggi del tiro che aspettano la fine dell'effetto dell'impatto, e il campione udibile da
        # cui si danno (D22).
        self._da_consegnare = []
        self._consegna_dal = None
        self._inizio = None
        # Il tempo vero in cui la partita è stata ferma, per la pausa o per l'attesa della palestra, e
        # da quando lo è adesso: la durata non lo conta.
        self._pause = 0.0
        self._fermo_dal = None
        self._durata = None

    # Lo stato.

    def in_corso(self):
        """Vero se c'è una partita in corso, in pausa compresa."""
        return self.fase in IN_CORSO

    def scegli_proiettile(self, forma):
        """La forma d'onda del proiettile, una di PROIETTILE_FORME, scelta nelle Impostazioni (D17).
        Vale dalla prossima partita: se nessuna è in corso, cioè subito, e allora cambia anche
        proiettile; con una partita in corso questa tiene la sua fino alla fine. Restituisce vero se
        la forma vale già per la partita che comincerà, falso se aspetta la fine di quella in corso.
        Una forma sconosciuta solleva ValueError e non cambia niente."""
        self.proiettile_scelto = _forma_del_proiettile(forma)
        if self.in_corso():
            return False
        self.proiettile = self.proiettile_scelto
        return True

    def _portanti(self):
        """Le portanti fra cui nascono i bersagli della partita: tutte quelle di BERSAGLIO_PORTANTI,
        tolta la forma del proiettile, perché un bersaglio non suoni come il colpo (D18)."""
        return portanti_senza(self.proiettile)

    def riepilogo(self):
        """{"punteggio", "livello", "tiri", "centri", "durata_s", "precisione_globale"} della partita in
        corso o dell'ultima; la durata è quella vera, senza le pause, al decimo di secondo, e la
        precisione globale è la media dei tiri che l'hanno (D26), al decimillesimo, i perduti a zero,
        o None prima del primo tiro: va nelle statistiche e, da D27, in classifica."""
        durata = self._durata if self._durata is not None else self._durata_adesso()
        media = round(self._somma_globale / self._tiri_globali, 4) if self._tiri_globali else None
        return {"punteggio": self.punti, "livello": self.livello, "tiri": self.tiri, "centri": self.centri, "durata_s": round(durata, 1),
                "precisione_globale": media}

    def righe_della_barra(self):
        """Le righe della barra di stato, da resoconti.righe_della_barra; nella mira coi tre valori, come
        gradi e percentuale, e i secondi che restano a una mira a tempo (D29)."""
        bonus = self.bonus is not None and self.bonus in self.bersagli
        normali = sum(1 for bersaglio in self.bersagli if not bersaglio.bonus)
        valori_della_mira = None
        if self.fase == "mira" and self.mira is not None:
            valori = self.mira.valori(self._adesso())
            azimut, alzo, _velocita = fisica.parametri_del_tiro(*valori)
            valori_della_mira = (azimut, alzo, valori[2], self.secondi_della_mira())
        return resoconti.righe_della_barra(self.fase, in_pausa=self.in_pausa, livello=self.livello, punti=self.punti, riserva=self.riserva,
                                           centri_del_livello=self.centri_del_livello, bersagli=normali, bonus=bonus,
                                           attesa_ferma=self.attesa_ferma, nasceranno=self.nasceranno, mira=valori_della_mira)

    # I comandi.

    def nuova(self):
        """Comincia una partita nuova, abbandonando prima quella in corso: livello 1, riserva
        COLPI_INIZIALI, il proiettile della forma scelta, i bersagli del primo livello che nascono al
        prossimo campione sicuro, col tempo di gioco che scorre da lì. Il suono della partita nuova
        aspetta in fila gli avvisi della partita di prima, e dopo la fine anche l'esito della
        classifica che la finestra vi ha messo, perché lo spazio dato d'abitudine subito dopo
        l'ultimo tiro non li copra. I messaggi dell'ultimo tiro che aspettavano ancora la fine del suo
        effetto vengono per primi: la fine della partita di prima non si perde."""
        finita = self.fase == "finita"
        messaggi = self.messaggi_in_attesa()
        if self.in_corso():
            messaggi.extend(self.abbandona())
        self._ferma_voci()
        self.proiettile = self.proiettile_scelto
        self.livello = 1
        self.riserva = COLPI_INIZIALI
        self.punti = self.tiri = self.centri = self.serie = self.centri_del_livello = 0
        self.valori = []
        self.ultimo_volo = None
        self.mira = None
        self._somma_globale = 0.0
        self._tiri_globali = 0
        self._azzera_la_pausa()
        self._serie_detta = False
        self._numero = 0
        self._inizio = self.orologio()
        self._pause = 0.0
        self._durata = None
        k = self.motore.prossimo_campione()
        self._nuovi_bersagli(k)
        self.motore.fai_scorrere_tempo(k)
        dopo_l_esito = None
        if finita and self._fine_degli_avvisi > k:
            dopo_l_esito = self._fine_degli_avvisi + self._lunghezza_degli_esiti() + self._stacco()
        self._avviso("nuova_partita", dopo_l_esito)
        self._conta("partite_iniziate")
        self._conta(CONTI_DEL_PROIETTILE[self.proiettile])
        self.fase = "ascolto"
        messaggi.append(resoconti.inizio_partita(self.livello, self.riserva))
        messaggi.extend(self._forse_un_bonus())
        return messaggi

    def spazio(self):
        """La barra spaziatrice: senza partita ne comincia una; nell'ascolto avvia la mira; nell'attesa
        della palestra fa ripartire la partita; col proiettile in volo, nella mira, dove spara l'Invio
        (D29), o in pausa, non può fare niente. A partita finita, finché la fine non si è detta, cioè
        finché suona l'effetto dell'ultimo tiro, lo spazio non conta e non suona: ne comincerebbe
        un'altra prima della classifica (D22)."""
        if self.fase in ("pronta", "finita"):
            if self._da_consegnare:
                return []
            return self.nuova()
        if self.in_pausa or self.fase in ("volo", "mira"):
            self.suoni.suona("non_ora")
            return []
        if self.fase == "attesa":
            return self._spazio_nell_attesa()
        return self._avvia_mira()

    def battito(self):
        """Il giro del timer della finestra: segue la mira, si accorge dell'impatto sentito e dei bonus
        usciti, ferma la partita nell'attesa della palestra, e dà i messaggi del tiro quando l'effetto
        dell'impatto ha finito di suonare, coi messaggi nuovi in fila dopo di loro (D22)."""
        if self.fase == "mira":
            nuovi = self._battito_della_mira()
        elif self.fase == "volo":
            nuovi = self._battito_del_volo()
        elif self.fase == "ascolto":
            nuovi = self._controlla_il_bonus()
        elif self.fase == "attesa":
            nuovi = self._battito_dell_attesa()
        else:
            nuovi = []
        return self._consegna(nuovi)

    def messaggi_in_attesa(self):
        """I messaggi del tiro che aspettano la fine dell'effetto dell'impatto (D22), subito, e tolti
        dalla fila: per la partita nuova, per l'abbandono e per la finestra che si chiude, che non li
        devono perdere, la fine della partita men che meno. Una lista, vuota se non ce ne sono."""
        messaggi, self._da_consegnare = self._da_consegnare, []
        self._consegna_dal = None
        return messaggi

    def pausa(self):
        """Ctrl+P, la pausa chiesta da chi gioca (D14): con la partita in corso la ferma e la fa tacere,
        col suono della pausa, e la conta in «pause»; nella pausa chiesta così la fa riprendere col
        suono della ripresa. La pausa si prende soltanto nei livelli della palestra (D22): dopo, Ctrl+P
        suona non_ora, lo dice la vista, e si conta in «pause_rifiutate»; ma una pausa presa
        nell'ultimo livello della palestra, ancora in corso quando il tiro porta al livello dopo, si
        toglie con Ctrl+P come sempre, e non resta incastrata. Nella pausa automatica, che chi gioca
        non vede perché ha un dialogo o un menu davanti, Ctrl+P arriva soltanto dalla voce Pausa del
        menu Partita: nella palestra la pausa diventa quella chiesta, col suo suono, e resta quando il
        menu si chiude; fuori dalla palestra è rifiutata, e la pausa automatica se ne va col menu.
        Il motore resta aperto, così i suoni della pausa e della ripresa si sentono.
        Senza partita non c'è niente da fermare, e lo dice il suono del non ora, come lo spazio fuori
        tempo: con la lettura dei resoconti spenta è l'unico segno che il tasto è arrivato."""
        if self.fase not in IN_CORSO:
            self.suoni.suona("non_ora")
            return [resoconti.nessuna_partita()]
        if self.in_pausa and not self._pausa_automatica:
            # Gli avvisi rimandati dalla pausa vengono in fila dopo il suono della ripresa, non sopra; il bonus
            # che la ripresa fa nascere si dice dopo il resoconto che aspetta ancora la fine dell'effetto (D22).
            self._accoda(self.suoni.suona("ripresa"))
            return [resoconti.ripresa(), *self._consegna(self._riprendi())]
        if not nella_palestra(self.livello):
            self.suoni.suona("non_ora")
            self._conta("pause_rifiutate")
            return [resoconti.pausa_solo_in_palestra(PALESTRA_FINO_AL_LIVELLO)]
        nella_mira = self.fase == "mira"
        if self.in_pausa:
            self._pausa_automatica = False
        else:
            self._sospendi()
        self.suoni.suona("pausa")
        self._conta("pause")
        return [resoconti.in_pausa(nella_mira=nella_mira)]

    def pausa_automatica(self, attiva):
        """La pausa che mette e toglie la finestra (D14): attiva vera quando si apre un dialogo modale o
        un menu (D22), o la finestra perde l'attivazione, falsa quando il dialogo o il menu si chiude e
        la finestra è di nuovo attiva. Vale in tutti i livelli, anche fuori dalla palestra delle
        attese, e anche nell'attesa. Con vero, se la partita è in corso e non è già in pausa, la ferma come Ctrl+P, si
        ricorda che la pausa è automatica e la conta in «pause_automatiche». Con falso la fa
        riprendere, ma soltanto se la pausa è automatica: quella chiesta con Ctrl+P resta finché non
        la si toglie con Ctrl+P. Tutte e due in silenzio, senza suoni né messaggi della pausa, perché
        chi gioca non l'ha chiesta e la finestra ha già il suo dialogo da dire; la barra di stato dice
        comunque la fase. Restituisce una lista di Messaggio, di solito vuota: alla ripresa può
        portare il bonus nato a partita ferma, in fila dopo il resoconto che aspetta ancora la fine
        dell'effetto (D22)."""
        if attiva:
            if self.fase in IN_CORSO and not self.in_pausa:
                self._sospendi()
                self._pausa_automatica = True
                self._conta("pause_automatiche")
            return []
        if self.in_pausa and self._pausa_automatica:
            return self._consegna(self._riprendi())
        return []

    def annulla_mira(self):
        """Esc nella mira (D25, dal collaudo di Gabriele del 20 novembre 2026: «Esc annulla lo sparo in corso
        e il proiettile viene conservato»): i valori tacciono e si perdono, il tempo della mira si ferma,
        la riserva non cambia, e si torna all'ascolto, col tempo di gioco che continua a scorrere come
        nella mira (D29); suona mira_annullata, si conta in mire_annullate, e il messaggio lo dice. Non è
        una fase d'ascolto nuova: non nasce un bonus. Nella mira messa in pausa con Ctrl+P la mira si
        annulla lo stesso, e la pausa resta: alla ripresa si torna ad ascoltare. Fuori dalla mira non fa
        niente, e restituisce una lista vuota."""
        if self.fase != "mira":
            return []
        self._chiudi_la_mira()
        self.fase = "ascolto"
        self.suoni.suona("mira_annullata")
        self._conta("mire_annullate")
        return self._consegna([resoconti.mira_annullata(self.riserva, in_pausa=self.in_pausa)])

    def abbandona(self):
        """Lascia la partita in corso: ferma le voci di gioco, la conta come abbandonata e torna a
        «pronta». I messaggi dell'ultimo tiro che aspettavano la fine del suo effetto vengono prima,
        perché il resoconto non si perda (D22). Senza partita in corso non fa niente."""
        if not self.in_corso():
            return []
        messaggi = [*self.messaggi_in_attesa(), *self._chiudi("abbandonata")]
        self.fase = "pronta"
        return messaggi

    # La pausa.

    def _sospendi(self):
        """Ferma la partita in corso e la fa tacere, per la pausa chiesta e per quella automatica.
        Nell'ascolto, nella mira e nel volo il tempo di gioco si ferma al prossimo campione sicuro;
        nella mira i valori si fermano dove si sentivano, coi loro tasti come lasciati, i loro suoni e
        i tick del tempo tacciono, e il tempo della mira, dove c'è, si ferma coi secondi che le restano
        (D29). Nell'attesa della palestra il tempo è già fermo dall'impatto (D22). Le voci di gioco vanno
        a guadagno zero, con la rampa del motore contro i click."""
        if self.fase == "mira":
            self._ferma_la_mira()
        if self.fase != "attesa":
            self.motore.ferma_tempo(self.motore.prossimo_campione())
        for voce in self._voci_di_gioco():
            self.motore.guadagno_voce(voce, 0.0)
        self.in_pausa = True
        self._pausa_automatica = False
        self._segna_il_fermo()

    def _riprendi(self):
        """La ripresa, dopo l'una o l'altra pausa: nell'ascolto, nella mira e nel volo il tempo riprende
        dal prossimo campione sicuro e le voci tornano a guadagno pieno, e nella mira a tempo riparte il
        conto dei secondi che le restavano, coi suoi tick (D29); nell'attesa della palestra il tempo resta
        fermo, e le voci tornano soltanto se la partita non si era ancora fermata ad aspettare lo spazio
        (D22). Poi suonano in fila gli avvisi rimandati, e nasce il bonus che la pausa ha fatto aspettare:
        i suoi messaggi sono quelli che restituisce."""
        k = self.motore.prossimo_campione()
        if self.fase == "attesa":
            guadagno = 0.0 if self.attesa_ferma else 1.0
        else:
            guadagno = 1.0
            self.motore.fai_scorrere_tempo(k)
        for voce in self._voci_di_gioco():
            self.motore.guadagno_voce(voce, guadagno)
        self.in_pausa = False
        self._pausa_automatica = False
        self._segna_il_fermo()
        if self.fase == "mira":
            self._mira_corre_dal = k
            self._avvia_il_tempo_della_mira(k)
        rimandati, self._avvisi_rimandati = self._avvisi_rimandati, []
        for evento in rimandati:
            self._avviso(evento)
        if self._bonus_rimandato:
            self._bonus_rimandato = False
            return self._forse_un_bonus()
        return []

    def _azzera_la_pausa(self):
        """Nessuna pausa, nessuna attesa della palestra e niente in attesa della ripresa: per la partita
        nuova e per quella che finisce."""
        self.in_pausa = False
        self._pausa_automatica = False
        self._fermo_dal = None
        self._avvisi_rimandati = []
        self._bonus_rimandato = False
        self.attesa_ferma = False
        self.nasceranno = 0
        self._attesa_dal = None

    def _segna_il_fermo(self):
        """Il conto del tempo vero in cui la partita è ferma, per una pausa o per l'attesa della
        palestra (D22), che la durata non conta: si chiama dopo ogni cambio dell'una e dell'altra, e una
        pausa presa durante l'attesa non si conta due volte."""
        ferma = self.in_pausa or self.attesa_ferma
        if ferma and self._fermo_dal is None:
            self._fermo_dal = self.orologio()
        elif not ferma and self._fermo_dal is not None:
            self._pause += max(0.0, self.orologio() - self._fermo_dal)
            self._fermo_dal = None

    # La palestra delle attese (D22).

    def _entra_nell_attesa(self, nasceranno):
        """Dopo un tiro nella palestra la partita entra nella fase «attesa»: il tempo di gioco si ferma
        subito, a partita ferma lo è già, e lo spazio chiamerà nasceranno bersagli nuovi. Le voci dei
        bersagli rimasti si sentono ancora, ferme, finché l'effetto e gli avvisi suonano: poi le zittisce
        il battito, con _battito_dell_attesa."""
        self.fase = "attesa"
        self.nasceranno = nasceranno
        self.attesa_ferma = False
        self._attesa_dal = None
        if not self.in_pausa:
            self.motore.ferma_tempo(self.motore.prossimo_campione())

    def _battito_dell_attesa(self):
        """Nell'attesa della palestra, finiti l'effetto dell'impatto e gli avvisi in fila, stacco
        compreso, e arrivato il resoconto, la partita si ferma e tace come in pausa: le voci di gioco a
        guadagno zero, il tempo dell'attesa fuori dalla durata, e l'attesa si conta. A partita in pausa
        aspetta la ripresa. Un bonus la cui uscita era già partita si dice anche qui."""
        messaggi = self._controlla_il_bonus()
        if self.in_pausa or self.attesa_ferma:
            return messaggi
        udibile = self.motore.campione_udibile()
        if udibile < self._fine_degli_avvisi or (self._da_consegnare and udibile < self._consegna_dal):
            return messaggi
        for voce in self._voci_di_gioco():
            self.motore.guadagno_voce(voce, 0.0)
        self.attesa_ferma = True
        self._attesa_dal = udibile
        self._segna_il_fermo()
        self._conta("attese_della_palestra")
        return messaggi

    def _spazio_nell_attesa(self):
        """Lo spazio nell'attesa della palestra: prima che la partita si fermi, mentre l'effetto o gli
        avvisi suonano ancora, e nella grazia di GRAZIA_DELL_ATTESA che segue, non conta e non suona,
        così un colpo dato in anticipo non salta l'attesa
        per sbaglio; dopo fa ripartire la partita."""
        if not self.attesa_ferma:
            return []
        if self.motore.campione_udibile() < self._attesa_dal + round(GRAZIA_DELL_ATTESA * self.motore.fs):
            return []
        return self._esci_dall_attesa()

    def _esci_dall_attesa(self):
        """La partita riparte dall'attesa della palestra, nella fase d'ascolto, dal prossimo campione
        sicuro: le voci dei bersagli rimasti tornano, nascono i bersagli nuovi, dopo un centro, e il
        tempo di gioco riprende a scorrere da lì. È una nuova fase d'ascolto, e può nascere un bonus:
        i suoi messaggi sono quelli che restituisce."""
        k = self.motore.prossimo_campione()
        nasceranno = self.nasceranno
        self.fase = "ascolto"
        self.attesa_ferma = False
        self.nasceranno = 0
        self._attesa_dal = None
        self._segna_il_fermo()
        for voce in self._voci_di_gioco():
            self.motore.guadagno_voce(voce, 1.0)
        tempo = self.motore.tempo_di_gioco(k)
        for _indice in range(nasceranno):
            self._nasce(genera(self.livello, self.rng, self._nuovo_numero(), tempo, portanti=self._portanti()), k)
        self.motore.fai_scorrere_tempo(k)
        return self._forse_un_bonus()

    # I messaggi del tiro, dopo l'effetto dell'impatto (D22).

    def _rimanda(self, messaggi, dal):
        """Mette in fila i messaggi del tiro, da dare quando il campione udibile arriva a dal."""
        if not messaggi:
            return
        self._da_consegnare.extend(messaggi)
        self._consegna_dal = dal if self._consegna_dal is None else max(self._consegna_dal, dal)

    def _consegna(self, nuovi):
        """I messaggi da dare adesso: i nuovi, se nessuno aspetta; altrimenti i nuovi vanno in fila dopo
        quelli che aspettano la fine dell'effetto, e tutti insieme arrivano quando l'effetto è finito."""
        if not self._da_consegnare:
            return list(nuovi)
        self._da_consegnare.extend(nuovi)
        if self.motore.campione_udibile() < self._consegna_dal:
            return []
        return self.messaggi_in_attesa()

    # I bersagli.

    def _nuovo_numero(self):
        self._numero += 1
        return self._numero

    def _nasce(self, bersaglio, k):
        """Un bersaglio entra in volo, con la sua voce dal campione k, nella sua portante, e si conta. A
        partita ferma nasce muto, come gli altri, e la ripresa gli dà la voce con loro: succede quando
        un impatto partito prima della pausa si risolve durante la pausa."""
        self.bersagli.append(bersaglio)
        voce = VoceTono(bersaglio.parametri_sonori, bersaglio.portante, inizio=k, guadagno=0.0 if self.in_pausa else 1.0)
        self._voci[bersaglio] = voce
        self.motore.aggiungi(voce)
        if self.statistiche is not None:
            self.statistiche.bersaglio_apparso(bersaglio.forma, bersaglio.mobile, bersaglio.bonus, portante=bersaglio.portante)

    def _nuovo_bersaglio(self):
        """Un bersaglio normale del livello, nato al tempo di gioco del prossimo campione sicuro."""
        k = self.motore.prossimo_campione()
        self._nasce(genera(self.livello, self.rng, self._nuovo_numero(), self.motore.tempo_di_gioco(k), portanti=self._portanti()), k)

    def _nuovi_bersagli(self, k):
        """Tutti i bersagli normali del livello, nati insieme al campione k."""
        tempo = self.motore.tempo_di_gioco(k)
        for _indice in range(quanti_bersagli(self.livello)):
            self._nasce(genera(self.livello, self.rng, self._nuovo_numero(), tempo, portanti=self._portanti()), k)

    def _togli(self, bersaglio):
        """Toglie un bersaglio dall'elenco e dimentica la sua voce, senza fermarla."""
        if bersaglio in self.bersagli:
            self.bersagli.remove(bersaglio)
        self._voci.pop(bersaglio, None)
        if bersaglio is self.bonus:
            self.bonus = None
            self._livello_del_bonus = None

    def _ferma_bersagli(self):
        """Ferma subito le voci di tutti i bersagli, annulla l'uscita del bonus e svuota l'elenco."""
        for voce in self._voci.values():
            self.motore.ferma(voce)
        self._voci = {}
        self.bersagli = []
        self.bonus = None
        self._livello_del_bonus = None
        if self._evento_bonus is not None:
            self.motore.annulla(self._evento_bonus)
            self._evento_bonus = None

    def _ferma_voci(self):
        """Ferma tutte le voci di gioco, bersagli, proiettile e suoni della mira, e gli eventi
        programmati. L'effetto di un impatto già partito si conta, perché si è sentito anche se il tiro
        non si risolverà più; quello annullato prima di partire non si conta."""
        self._ferma_bersagli()
        self._chiudi_la_mira()
        if self._voce_proiettile is not None:
            self.motore.ferma(self._voce_proiettile)
        self._voce_proiettile = None
        if self._evento_impatto is not None:
            if self._evento_impatto.scattato:
                self._conta_suono(self._effetto_impatto)
            self.motore.annulla(self._evento_impatto)
            self._evento_impatto = None
        self._effetto_impatto = None

    def _voci_di_gioco(self):
        """Le voci che la pausa zittisce: i bersagli e il proiettile in volo."""
        voci = list(self._voci.values())
        if self._voce_proiettile is not None:
            voci.append(self._voce_proiettile)
        return voci

    # I suoni e i conti.

    def _pronto_senza_conto(self, evento, posizione, pan=None):
        """Il buffer di un effetto preparato senza contarlo: Suoni conta un effetto quando è pronto,
        ma l'uscita del bonus non si sente se il bonus viene colpito prima, e l'impatto non si sente
        se la partita si abbandona col proiettile in volo, e allora il conto sarebbe falso. Lo si
        conta quando il battito sa che è suonato."""
        conta, self.suoni.conta = self.suoni.conta, None
        try:
            return self.suoni.pronto(evento, posizione=posizione, pan=pan)
        finally:
            self.suoni.conta = conta

    def _conta_suono(self, evento):
        """Conta un suono che il motore ha già fatto, con la stessa funzione che usa Suoni."""
        conta = self.suoni.conta
        if conta is not None and evento is not None:
            conta(evento)

    def _conta(self, chiave, quanti=1):
        """Un contatore delle statistiche, se ci sono."""
        if self.statistiche is not None:
            self.statistiche.conta(chiave, quanti)

    def _stacco(self):
        """STACCO_FRA_GLI_AVVISI in campioni."""
        return round(STACCO_FRA_GLI_AVVISI * self.motore.fs)

    def _accoda(self, voce):
        """Mette in fila una voce partita: il prossimo avviso aspetta la sua fine, più lo stacco."""
        if voce is not None:
            self._fine_degli_avvisi = max(self._fine_degli_avvisi, voce.inizio + len(voce.buffer) + self._stacco())

    def _avviso(self, evento, inizio=None):
        """Un avviso della partita, il livello nuovo, il bonus apparso, l'ultimo colpo, la fine o la
        partita nuova: suona dal prossimo campione sicuro, oppure, se l'avviso di prima o l'effetto
        dell'impatto si sentono ancora, in fila dopo di loro, staccato di STACCO_FRA_GLI_AVVISI; mai
        prima di inizio, se c'è. A partita ferma non suona: aspetta la ripresa (D14)."""
        if self.in_pausa:
            self._avvisi_rimandati.append(evento)
            return None
        inizio = max(self.motore.prossimo_campione(), self._fine_degli_avvisi, inizio or 0)
        voce = self.suoni.suona(evento, inizio=inizio)
        self._accoda(voce)
        return voce

    def _lunghezza_degli_esiti(self):
        """I campioni del più lungo fra gli esiti della classifica: la partita non sa quale dei due la
        finestra ha messo in fila, quindi lascia il posto al più lungo."""
        lunghezze = [0]
        for evento in ESITI_DELLA_CLASSIFICA:
            buffer = self._pronto_senza_conto(evento, None)
            if buffer is not None:
                lunghezze.append(len(buffer))
        return max(lunghezze)

    def dopo_gli_avvisi(self):
        """Il campione da cui può partire un suono che deve seguire gli avvisi della partita, come l'esito
        della classifica dopo la fine; None se nessun avviso si deve ancora sentire, e il suono può
        partire subito."""
        if self._fine_degli_avvisi > self.motore.prossimo_campione():
            return self._fine_degli_avvisi
        return None

    # Il bonus.

    def _forse_un_bonus(self):
        """A ogni nuova fase d'ascolto, dal LIVELLO_DEI_BONUS e senza un bonus in volo, nasce un bonus
        con PROBABILITA_BONUS. La sua uscita è un evento programmato al tempo di gioco in cui esce del
        tutto dal settore, che nel filo dell'audio ne ferma la voce e fa suonare l'effetto già pronto,
        dal punto dove esce."""
        if self.livello < LIVELLO_DEI_BONUS or self.bonus is not None:
            return []
        if self.rng.random() >= PROBABILITA_BONUS:
            return []
        k = self.motore.prossimo_campione()
        bonus = genera_bonus(self.livello, self.rng, self._nuovo_numero(), self.motore.tempo_di_gioco(k), portanti=self._portanti())
        self._nasce(bonus, k)
        self.bonus = bonus
        self._livello_del_bonus = self.livello
        self._avviso("bonus_appare")
        uscita = bonus.tempo_di_uscita
        buffer = self._pronto_senza_conto("bonus_uscito", bonus.posizione(uscita))
        voce = self._voci[bonus]
        motore = self.motore

        def esce(campione):
            motore.ferma(voce, campione)
            if buffer is not None and len(buffer):
                motore.suona(buffer, inizio=campione)

        self._evento_bonus = motore.programma(uscita, esce)
        return [resoconti.bonus_apparso(bonus, self.livello)]

    def _controlla_il_bonus(self):
        """Toglie dall'elenco il bonus che non è più vivo, quando la sua uscita si è sentita, e lo dice."""
        bonus = self.bonus
        if bonus is None:
            return []
        udibile = self.motore.campione_udibile()
        # Non basta chiedere bonus.vivo: appena nato, il bonus nasce al prossimo campione sicuro, e
        # al campione udibile, un poco prima, non è ancora vivo. Conta soltanto il tempo dell'uscita.
        if not _sentito(self._evento_bonus, udibile) and self.motore.tempo_di_gioco(udibile) <= bonus.tempo_di_uscita:
            return []
        # La voce l'ha già fermata l'evento al campione esatto; fermarla qui non cambia niente, ma
        # copre un evento che non fosse scattato, perché una seconda fermata più tarda non conta.
        voce = self._voci.get(bonus)
        if voce is not None:
            self.motore.ferma(voce)
        self._togli(bonus)
        self._evento_bonus = None
        self._conta("bonus_usciti")
        self._conta_suono("bonus_uscito")
        return [resoconti.bonus_uscito()]

    # La mira manuale (D29).

    def _adesso(self):
        """Il tempo dell'audio che si sente adesso, in secondi: il tempo della mira, perché un valore
        vale quello che si sentiva nell'istante del tasto."""
        return self.motore.campione_udibile() / self.motore.fs

    def secondi_della_mira(self):
        """I secondi che restano alla mira a tempo, mai sotto zero; None senza mira o nella palestra,
        dove la mira non ha tempo. In pausa sono quelli che restavano quando la pausa è arrivata."""
        if self.fase != "mira" or self._secondi_rimasti is None:
            return None
        if self.in_pausa or self._mira_dal is None:
            return max(0.0, self._secondi_rimasti)
        passati = max(0.0, (self.motore.campione_udibile() - self._mira_dal) / self.motore.fs)
        return max(0.0, self._secondi_rimasti - passati)

    def tasti_giu(self):
        """I tasti della mira che la partita crede giù, come coppie (parametro, verso); vuota fuori dalla
        mira. La finestra controlla che lo siano ancora, perché un rilascio può perdersi con il fuoco
        altrove."""
        if self.fase != "mira" or self.mira is None:
            return []
        return self.mira.tasti_giu()

    def muovi(self, parametro, verso):
        """Il tasto di un valore scende: il valore si muove nel verso dato, meno uno verso il minimo,
        uno verso il massimo, col suo suono e le sue tacche. Fuori dalla mira, o in pausa, non fa niente;
        la ripetizione del tasto tenuto giù nemmeno. Restituisce una lista di Messaggio, vuota."""
        if self.fase != "mira" or self.in_pausa or self.mira is None:
            return []
        if self._scaduta():
            return self._spara_la_mira(self._scadenza / self.motore.fs, scaduta=True)
        if self.mira.premi(parametro, verso, self._adesso()):
            self._suona_l_asse(parametro)
        return []

    def lascia(self, parametro, verso):
        """Il tasto di un valore sale: il valore si ferma dove si sentiva, o va nel verso dell'altro suo
        tasto ancora giù. Restituisce una lista di Messaggio, vuota."""
        if self.fase != "mira" or self.in_pausa or self.mira is None:
            return []
        if self._scaduta():
            return self._spara_la_mira(self._scadenza / self.motore.fs, scaduta=True)
        if self.mira.lascia(parametro, verso, self._adesso()):
            self._suona_l_asse(parametro)
        return []

    def spara(self):
        """Invio nella mira: il colpo parte coi tre valori dell'istante in cui l'Invio si è sentito. In
        pausa suona il non ora, e fuori dalla mira non fa niente. Restituisce una lista di Messaggio."""
        if self.fase != "mira" or self.mira is None:
            return []
        if self.in_pausa:
            self.suoni.suona("non_ora")
            return []
        if self._scaduta():
            return self._spara_la_mira(self._scadenza / self.motore.fs, scaduta=True)
        return self._spara_la_mira(self._adesso(), scaduta=False)

    def _scaduta(self):
        """Vero se la mira a tempo è già scaduta per le orecchie: il battito non se n'è ancora accorto, e un
        tasto o l'Invio arrivati dopo la scadenza non devono spostare il colpo, che parte coi valori di quel
        momento, come la mira scaduta."""
        return self._scadenza is not None and self.motore.campione_udibile() >= self._scadenza

    def _avvia_mira(self):
        """Lo spazio nell'ascolto avvia la mira: suona mira_avviata, i tre valori stanno al centro, fermi e
        muti finché un tasto non li muove, il tempo di gioco continua a scorrere e i bersagli suonano pieni
        (D29); fuori dalla palestra parte il tempo della mira, coi suoi tick."""
        self.suoni.suona("mira_avviata")
        self._conta("mire_avviate")
        self.valori = []
        self.mira = mira.Mira()
        self.fase = "mira"
        k = self.motore.prossimo_campione()
        self._secondi_di_mira = 0.0
        self._mira_corre_dal = k
        self._tempo_della_mira = tempo_della_mira(self.livello)
        self._secondi_rimasti = self._tempo_della_mira
        self._avvia_il_tempo_della_mira(k)
        return []

    def _avvia_il_tempo_della_mira(self, k):
        """Il tempo della mira a tempo parte, o riparte dopo una pausa, dal campione k, coi secondi che le
        restano: la scadenza e i tick dei secondi che mancano, in fila, uno al secondo, sempre più acuti.
        Se gli avvisi in fila dopo l'impatto, come il livello nuovo, o la coda dell'effetto, suonano ancora,
        o se i messaggi del tiro aspettano ancora la fine dell'effetto (D22), il tempo parte dopo di loro:
        così i tick non coprono la fanfara né la voce del resoconto, e il tempo non si consuma mentre li si
        ascolta, come faceva la prima barra. Nella palestra non c'è tempo, e non fa niente."""
        self._ferma_i_tick()
        if self._secondi_rimasti is None:
            self._scadenza = self._mira_dal = None
            return
        fs = self.motore.fs
        k = max(k, self._fine_degli_avvisi)
        if self._da_consegnare and self._consegna_dal is not None:
            k = max(k, round(self._consegna_dal))
        self._mira_dal = k
        self._scadenza = k + round(self._secondi_rimasti * fs)
        totale = self._tempo_della_mira
        passati = totale - self._secondi_rimasti
        for secondo in range(1, math.ceil(totale)):
            if secondo > passati + 1e-9:
                inizio = k + round((secondo - passati) * fs)
                self._tick_in_fila.append(self.motore.suona(tick_del_tempo(secondo, totale, fs=fs), inizio=inizio))

    def _ferma_i_tick(self):
        for voce in self._tick_in_fila:
            self.motore.ferma(voce)
        self._tick_in_fila = []

    def _ferma_in_fila(self, parametro):
        """Le tacche e il suono del capo che il valore aveva davanti, messi in fila e non ancora sentiti,
        non suonano più; quelli già partiti finiscono, perché un tocco brevissimo al capo non tronchi il
        suono fine_scala, che allora si conta subito, perché il battito non lo vedrà più in fila."""
        udibile = self.motore.campione_udibile()
        for voce in self._in_fila_degli_assi.get(parametro, ()):
            if voce.inizio > udibile:
                self.motore.ferma(voce)
        self._in_fila_degli_assi[parametro] = []
        capo = self._capi_in_fila.pop(parametro, None)
        if capo is not None and capo <= udibile:
            self._conta_suono("fine_scala")

    def _suona_l_asse(self, parametro):
        """I suoni della mira dopo un cambio del moto di un valore (D31): il puntatore, la voce sola che porta
        i tre valori, nasce al primo valore che si muove e da lì li segue da solo; davanti al valore, in fila,
        le tacche dei decimi che attraverserà e il suono fine_scala del capo, dove si ferma, ognuno dal punto
        del puntatore nell'istante in cui suona. Se cambia la direzione, cambia anche il punto da cui
        suoneranno le tacche e il capo degli altri valori in corsa, e quelli si rimettono in fila. Il suono
        del capo si conta quando si sente, al battito."""
        self._ferma_in_fila(parametro)
        asse = self.mira.assi[parametro]
        k = self.motore.prossimo_campione()
        if parametro == "direzione":
            for altro in PARAMETRI:
                if altro != parametro:
                    self._rimetti_in_fila(altro, k)
        if asse.verso == 0:
            return
        self._accendi_il_puntatore(k)
        capo = max(k, round(asse.tempo_al_capo() * self.motore.fs))
        self._in_fila_degli_assi[parametro], self._capi_in_fila[parametro] = self._metti_in_fila(parametro, k, capo)

    def _pan_del_puntatore(self, tempo):
        """Il pan del puntatore al tempo dato, dell'audio: quello della direzione in quell'istante."""
        return pan_della_direzione(self.mira.assi["direzione"].valore(tempo))

    def _metti_in_fila(self, parametro, da, capo):
        """Mette in fila, dal campione da in poi, le tacche che il valore attraverserà correndo e, al campione
        capo, se non viene prima di da, il suono fine_scala, ognuno col pan del puntatore nel suo istante.
        Restituisce le voci in fila e il campione del capo, o None se il capo resta fuori."""
        fs = self.motore.fs
        in_fila = []
        for tempo, indice in self.mira.assi[parametro].tacche_davanti():
            inizio = round(tempo * fs)
            if inizio >= da:
                tacca = panoramica(self._tacche[parametro][indice], self._pan_del_puntatore(tempo))
                in_fila.append(self.motore.suona(tacca, inizio=inizio))
        if capo is None or capo < da:
            return in_fila, None
        buffer = self._pronto_senza_conto("fine_scala", None, pan=self._pan_del_puntatore(capo / fs))
        if buffer is not None and len(buffer):
            in_fila.append(self.motore.suona(buffer, inizio=capo))
        return in_fila, capo

    def _rimetti_in_fila(self, parametro, da):
        """Le tacche e il capo di un valore in corsa, quando la direzione cambia moto e li farà suonare da
        un altro punto: quelli che partono dal campione da in poi si fermano e tornano in fila col pan nuovo,
        quelli di prima restano, e il capo resta nello stesso campione."""
        vecchie = self._in_fila_degli_assi.get(parametro, [])
        for voce in vecchie:
            if voce.inizio >= da:
                self.motore.ferma(voce)
        tenute = [voce for voce in vecchie if voce.inizio < da]
        if self.mira.assi[parametro].verso == 0:
            self._in_fila_degli_assi[parametro] = tenute
            return
        nuove, _capo = self._metti_in_fila(parametro, da, self._capi_in_fila.get(parametro))
        self._in_fila_degli_assi[parametro] = tenute + nuove

    def _accendi_il_puntatore(self, k):
        """Il puntatore della mira (D31): una voce sola, in PUNTATORE_FORMA, coi parametri di
        sintesi.parametri_del_puntatore, la direzione nella panoramica, l'alzo nell'altezza e la potenza nel
        volume, che suona soltanto dove almeno un valore si muove, entrando e uscendo in SFUMATURA_STOP. Nasce
        dal campione k al primo valore che si muove, e resta fino alla fine della mira o alla pausa: legge lo
        stato dei tre valori a ogni blocco, così due valori insieme cambiano lo stesso suono, senza voci nuove
        a ogni tasto. Già acceso non fa niente."""
        if self._voce_del_puntatore is not None:
            return
        assi = [self.mira.assi[parametro] for parametro in PARAMETRI]
        passo = 1.0 / max(1.0, SFUMATURA_STOP * self.motore.fs)
        porta = [0.0]

        def parametri(tempi_di_gioco, tempi_audio):
            frequenze, pan, guadagni, in_moto = _suono_del_puntatore(assi, tempi_audio)
            inviluppo, porta[0] = _rampa(in_moto, porta[0], passo)
            return frequenze, pan, guadagni * inviluppo

        voce = VoceTono(parametri, PUNTATORE_FORMA, inizio=k)
        self.motore.aggiungi(voce)
        self._voce_del_puntatore = voce

    def _ferma_la_mira(self):
        """La mira si ferma senza finire, per la pausa: i tasti valgono come lasciati nell'istante che si
        sente, i suoni dei valori e i tick del tempo tacciono, e i secondi che restano aspettano la
        ripresa."""
        if self.mira is None:
            return
        self._secondi_rimasti = self.secondi_della_mira()
        self._secondi_di_mira += self._secondi_che_corrono()
        self._mira_corre_dal = None
        self.mira.lascia_tutto(self._adesso())
        self._zittisci_la_mira()
        self._scadenza = self._mira_dal = None

    def _secondi_che_corrono(self, tempo=None):
        """I secondi di mira dall'ultima ripartenza fino al tempo dato, dell'audio, o fino a adesso; zero se la
        mira è ferma."""
        if self._mira_corre_dal is None:
            return 0.0
        tempo = self._adesso() if tempo is None else tempo
        return max(0.0, tempo - self._mira_corre_dal / self.motore.fs)

    def _zittisci_la_mira(self):
        for parametro in PARAMETRI:
            self._ferma_in_fila(parametro)
        if self._voce_del_puntatore is not None:
            self.motore.ferma(self._voce_del_puntatore)
            self._voce_del_puntatore = None
        self._ferma_i_tick()

    def _chiudi_la_mira(self):
        """La mira finisce, allo sparo, all'Esc o con la partita: niente più suoni, tempo né tasti."""
        if self.mira is not None:
            self.mira.lascia_tutto(self._adesso())
        self._zittisci_la_mira()
        self.mira = None
        self._scadenza = self._secondi_rimasti = self._mira_dal = self._mira_corre_dal = None

    def _battito_della_mira(self):
        """Il battito nella mira: conta il suono del capo che si è sentito, dice il bonus uscito, e allo
        scadere della mira a tempo fa partire il colpo da solo, coi valori dell'istante della scadenza
        (D29). A partita ferma la mira è ferma, e non scade."""
        if self.in_pausa:
            return []
        udibile = self.motore.campione_udibile()
        for parametro, campione in list(self._capi_in_fila.items()):
            if udibile >= campione:
                self._conta_suono("fine_scala")
                del self._capi_in_fila[parametro]
        messaggi = self._controlla_il_bonus()
        if self._scadenza is not None and udibile >= self._scadenza:
            messaggi.extend(self._spara_la_mira(self._scadenza / self.motore.fs, scaduta=True))
        return messaggi

    def _spara_la_mira(self, tempo, scaduta):
        """La mira finisce col colpo: i valori del tempo dato, nell'ordine di PARAMETRI, diventano quelli
        del tiro, le statistiche contano la mira, i suoi secondi e i suoi tasti, e la scaduta in
        mire_scadute, poi parte il colpo."""
        valori = self.mira.valori(tempo)
        secondi = self._secondi_di_mira + self._secondi_che_corrono(tempo)
        pressioni = self.mira.pressioni
        self._chiudi_la_mira()
        self.valori = [float(valore) for valore in valori]
        if self.statistiche is not None:
            for parametro, valore in zip(PARAMETRI, self.valori, strict=True):
                self.statistiche.mira(parametro, valore)
            self.statistiche.mira_sparata(secondi, pressioni, scaduta)
        if scaduta:
            self._conta("mire_scadute")
        return self._spara()

    # Il tiro.

    def _spara(self):
        """Il colpo: la traiettoria dai tre valori, il volo calcolato tutto subito dal tempo di gioco
        dello sparo, lo sparo dal cannone e la voce del proiettile, nella forma della partita, dal
        campione k. L'effetto
        dell'impatto si prepara adesso, perché nel filo dell'audio non c'è tempo per farlo; l'evento
        della fine del volo, al campione esatto, ferma il proiettile e il bersaglio colpito e fa
        partire l'effetto. L'effetto si conta quando si sente, non adesso: se la partita si abbandona
        col proiettile in volo non suona mai.
        Da D29 nella mira il tempo di gioco scorre, e il colpo parte da un k scelto prima del conto del
        volo, ANTICIPO_DELLO_SPARO dopo il prossimo campione sicuro: il volo parte dal tempo di gioco di
        quel campione, coi bersagli dove saranno allora, e il conto, che dura qualche millesimo, finisce
        prima che k arrivi al motore."""
        traiettoria = fisica.Traiettoria(*fisica.parametri_del_tiro(*self.valori))
        k = self.motore.prossimo_campione() + round(ANTICIPO_DELLO_SPARO * self.motore.fs)
        partenza = self.motore.tempo_di_gioco(k)
        vivi = [bersaglio for bersaglio in self.bersagli if bersaglio.vivo(partenza)]
        volo = calcola_volo(traiettoria, vivi, partenza)
        colpito = volo.colpito
        if colpito is None:
            effetto = "a_terra"
        elif colpito.bonus:
            effetto = "colpito_bonus"
        else:
            effetto = "colpito"
        buffer = self._pronto_senza_conto(effetto, volo.punto_finale)
        proiettile = VoceTono(parametri_del_proiettile(volo), self.proiettile, inizio=k)
        self.suoni.suona("sparo", posizione=fisica.CANNONE, inizio=k)
        self.motore.aggiungi(proiettile)
        voce_colpita = self._voci.get(colpito) if colpito is not None else None
        if colpito is not None and colpito is self.bonus and self._evento_bonus is not None:
            # Il bonus colpito non esce più: l'evento della sua uscita si annulla subito, che il volo
            # già sa come andrà, invece di rincorrerlo dopo l'impatto.
            self.motore.annulla(self._evento_bonus)
            self._evento_bonus = None
        motore = self.motore

        def impatto(campione):
            motore.ferma(proiettile, campione)
            if voce_colpita is not None:
                motore.ferma(voce_colpita, campione)
            if buffer is not None and len(buffer):
                motore.suona(buffer, inizio=campione)

        self._evento_impatto = motore.programma(volo.fine, impatto)
        self._effetto_impatto = effetto
        self._lunghezza_impatto = len(buffer) if buffer is not None else 0
        self._voce_proiettile = proiettile
        self.ultimo_volo = volo
        self.tiri += 1
        self.fase = "volo"
        return []

    def _battito_del_volo(self):
        """Il tiro si risolve quando l'impatto si è sentito: l'evento è scattato e il campione udibile
        lo ha passato, oppure il tempo di gioco al campione udibile ha passato la fine del volo. A
        partita ferma il tempo non arriva alla fine del volo, ma un impatto scattato prima della pausa
        si risolve lo stesso: si è sentito, e il resoconto lo dice, alla fine dell'effetto (D22)."""
        udibile = self.motore.campione_udibile()
        if _sentito(self._evento_impatto, udibile) or self.motore.tempo_di_gioco(udibile) >= self.ultimo_volo.fine:
            return self._risolvi()
        return []

    def _livello_di(self, bersaglio):
        """Il livello che fa i punti del bersaglio colpito: per il bonus quello in cui è nato (D13)."""
        if bersaglio.bonus and bersaglio is self.bonus and self._livello_del_bonus is not None:
            return self._livello_del_bonus
        return self.livello

    def _risolvi(self):
        """L'esito del tiro: punti, riserva, serie, livello, il resoconto e i messaggi che seguono, poi
        la nuova fase d'ascolto, o nella palestra l'attesa dello spazio, o la fine della partita. Gli
        avvisi che il tiro causa partono in fila dopo l'effetto dell'impatto, che sta ancora suonando;
        i messaggi aspettano anche loro la fine dell'effetto, più RITARDO_DEL_RESOCONTO, e li dà il
        battito (D22): qui non restituisce niente.
        La palestra (D22) si decide col livello del tiro: dopo il centro che chiude l'ultimo livello
        della palestra c'è ancora l'attesa, il livello nuovo si dice subito, e i suoi bersagli nascono
        allo spazio; dal primo tiro del livello dopo, niente più attese. Nella palestra anche il
        bersaglio nuovo al posto del colpito nasce allo spazio, e così il bonus della fase d'ascolto."""
        volo = self.ultimo_volo
        colpito = volo.colpito
        evento, effetto = self._evento_impatto, self._effetto_impatto
        self._evento_impatto = self._effetto_impatto = None
        self._voce_proiettile = None
        if evento is not None and evento.scattato and evento.campione is not None:
            self._conta_suono(effetto)
            fine_dell_effetto = evento.campione + self._lunghezza_impatto
        else:
            fine_dell_effetto = self.motore.prossimo_campione() + self._lunghezza_impatto
        self._fine_degli_avvisi = max(self._fine_degli_avvisi, fine_dell_effetto + self._stacco())
        livello_del_tiro = self.livello
        palestra = nella_palestra(livello_del_tiro)
        globale = volo.precisione_globale
        if globale is not None:
            # La media della precisione globale della partita, per le statistiche (D26).
            self._somma_globale += globale
            self._tiri_globali += 1
        punti = 0
        colpi_bonus = 0
        if colpito is not None:
            punti = punti_del_centro(colpito, volo.precisione, livello=self._livello_di(colpito))
            self.punti += punti
            self.centri += 1
            self.serie += 1
            self.centri_del_livello += 1
            if colpito.bonus:
                colpi_bonus = COLPI_BONUS
                self.riserva += COLPI_BONUS
                self._conta("colpi_guadagnati", COLPI_BONUS)
            self._togli(colpito)
        else:
            self.riserva = max(0, self.riserva - 1)
            self.serie = 0
            self._serie_detta = False
        # Il livello del tiro decide anche il grado del sommario (D21), bonus compreso: conta il livello
        # in cui si spara, non quello in cui il bonus è nato; le misure le porta il volo.
        messaggi = [resoconti.resoconto_del_tiro(self.tiri, livello_del_tiro, volo, self.valori, punti=punti, totale=self.punti, riserva=self.riserva,
                                                 colpi_bonus=colpi_bonus)]
        battuti = self._statistiche_del_tiro(volo, punti)
        # Il bonus uscito durante il volo si dice e si conta prima che il livello nuovo o la fine
        # svuotino il cielo: la sua uscita si è già sentita, e senza il messaggio sparirebbe dai conti.
        messaggi.extend(self._controlla_il_bonus())
        nasceranno = 0
        if colpito is not None and self.centri_del_livello >= CENTRI_PER_LIVELLO:
            messaggi.extend(self._sali_di_livello(subito=not palestra))
            nasceranno = quanti_bersagli(self.livello)
        elif colpito is not None and not colpito.bonus:
            # D5: un bersaglio nuovo nasce soltanto al posto di uno colpito; nella palestra allo spazio.
            if palestra:
                nasceranno = 1
            else:
                self._nuovo_bersaglio()
        self._seguito_del_tiro(messaggi, battuti, colpito, palestra, nasceranno, fine_dell_effetto)
        return []

    def _seguito_del_tiro(self, messaggi, battuti, colpito, palestra, nasceranno, fine_dell_effetto):
        """Il seguito di ogni tiro risolto: i primati battuti, poi
        la fine a riserva vuota, anche nella palestra, senza attesa; altrimenti l'ultimo colpo dopo un tiro
        a vuoto, e l'attesa nella palestra o, fuori, la nuova fase d'ascolto col bonus che può nascere.
        Tutti i messaggi, quelli ricevuti e questi, aspettano in fila la fine dell'effetto, più
        RITARDO_DEL_RESOCONTO, e li dà il battito (D22, D24)."""
        primati = resoconti.primati_battuti(battuti, self._valori_dei_primati(battuti))
        if primati is not None:
            messaggi.append(primati)
        if self.riserva <= 0:
            # La riserva vuota chiude la partita come sempre, anche nella palestra, senza attesa.
            messaggi.extend(self._chiudi("riserva_vuota"))
            self.fase = "finita"
        else:
            if colpito is None and self.riserva == 1:
                self._avviso("ultimo_colpo")
                messaggi.append(resoconti.ultimo_colpo())
            if palestra:
                self._entra_nell_attesa(nasceranno)
                messaggi.append(resoconti.attesa_della_palestra(nasceranno))
            else:
                self.fase = "ascolto"
                if self.in_pausa:
                    # Un bonus che nascesse adesso entrerebbe a partita ferma: nasce alla ripresa (D14).
                    self._bonus_rimandato = True
                else:
                    messaggi.extend(self._forse_un_bonus())
        self._rimanda(messaggi, fine_dell_effetto + round(RITARDO_DEL_RESOCONTO * self.motore.fs))

    def _statistiche_del_tiro(self, volo, punti):
        """Il tiro nelle statistiche, e i primati battuti da dire. La distanza del primato è dal centro
        del bersaglio, come la dice la pagina delle statistiche, volo.distanza_dal_centro: nel centro
        quella del volo, nel tiro a vuoto la distanza dalla superficie del più vicino più il suo raggio.
        Con lei va la precisione globale di D26."""
        if self.statistiche is None:
            return []
        colpito = volo.colpito
        esito = {"colpito": colpito is not None, "bonus": bool(colpito is not None and colpito.bonus), "forma": colpito.forma if colpito is not None else None,
                 "portante": colpito.portante if colpito is not None else None, "mobile": bool(colpito is not None and colpito.mobile), "precisione": volo.precisione,
                 "punti": punti, "distanza_dal_vicino": volo.distanza_dal_centro, "serie": self.serie, "precisione_globale": volo.precisione_globale}
        prima = self._primati_attuali()
        battuti = self._primati_che_cambiano(self.statistiche.tiro(list(self.valori), esito), prima)
        if "serie" in battuti:
            if self._serie_detta:
                battuti.remove("serie")
            else:
                self._serie_detta = True
        return battuti

    def _primati_attuali(self):
        """Il valore di ogni primato delle statistiche, adesso."""
        if self.statistiche is None:
            return {}
        primati = self.statistiche.dati.get("primati", {})
        return {chiave: dato["valore"] for chiave, dato in primati.items() if isinstance(dato, dict) and "valore" in dato}

    def _primati_che_cambiano(self, battuti, prima):
        """I primati battuti che si dicono: soltanto quelli il cui valore scritto cambia. Le statistiche
        tengono il valore esatto, ma il messaggio lo scrive arrotondato, la precisione al punto
        percentuale e le distanze al centimetro: un miglioramento che non si vede a parole non è un
        nuovo primato da festeggiare, e «100%» si conquista una volta sola."""
        dopo = self._valori_dei_primati(battuti)
        return [chiave for chiave in battuti
                if chiave not in prima or chiave not in dopo or resoconti.primato_scritto(chiave, prima[chiave]) != resoconti.primato_scritto(chiave, dopo[chiave])]

    def _valori_dei_primati(self, chiavi):
        if self.statistiche is None:
            return {}
        primati = self.statistiche.dati.get("primati", {})
        return {chiave: primati[chiave]["valore"] for chiave in chiavi if chiave in primati}

    def _sali_di_livello(self, subito=True):
        """I centri hanno chiuso il livello: il livello sale, arrivano COLPI_PER_LIVELLO colpi, e tutti
        i bersagli, il bonus compreso, lasciano il posto a quelli del livello nuovo, che nascono subito,
        o con subito falso, nella palestra delle attese (D22), allo spazio che chiude l'attesa."""
        self.livello += 1
        self.centri_del_livello = 0
        self.riserva += COLPI_PER_LIVELLO
        self._conta("livelli_superati")
        self._conta("colpi_guadagnati", COLPI_PER_LIVELLO)
        self._ferma_bersagli()
        if subito:
            self._nuovi_bersagli(self.motore.prossimo_campione())
        self._avviso("livello_su")
        return [resoconti.livello_nuovo(self.livello, COLPI_PER_LIVELLO, self.riserva)]

    # La fine.

    def _durata_adesso(self):
        """I secondi veri dall'inizio della partita, senza le pause e le attese della palestra (D22),
        quella in corso compresa."""
        if self._inizio is None:
            return 0.0
        adesso = self.orologio()
        pause = self._pause + (max(0.0, adesso - self._fermo_dal) if self._fermo_dal is not None else 0.0)
        return max(0.0, adesso - self._inizio - pause)

    def _chiudi(self, motivo):
        """Chiude la partita, finita a riserva vuota o abbandonata: voci ferme, durata fissata, conto
        e salvataggio delle statistiche, e il messaggio col riepilogo e i primati battuti."""
        if motivo == "abbandonata" and self._evento_impatto is not None:
            # Il tiro in volo non si è risolto, e le statistiche contano i tiri alla risoluzione: esce
            # anche dal riepilogo, così la partita e le statistiche dicono lo stesso numero di tiri.
            self.tiri = max(0, self.tiri - 1)
        self._ferma_voci()
        self._durata = self._durata_adesso()
        self._azzera_la_pausa()
        riepilogo = self.riepilogo()
        battuti = []
        if self.statistiche is not None:
            prima = self._primati_attuali()
            # Il riepilogo porta già la precisione globale media della partita (D26).
            battuti = self._primati_che_cambiano(self.statistiche.partita_finita({**riepilogo, "motivo": motivo}), prima)
            # Le statistiche non si salvano da sole: la fine di una partita è il momento in cui
            # perderle costerebbe di più.
            self.statistiche.salva()
        valori = self._valori_dei_primati(battuti)
        if motivo == "abbandonata":
            return [resoconti.partita_abbandonata(riepilogo, battuti, valori)]
        self._avviso("partita_finita")
        return [resoconti.fine_partita(riepilogo, battuti, valori)]
