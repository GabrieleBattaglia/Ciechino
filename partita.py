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
La pausa (D14, decisione di Gabriele del 19 novembre 2026) ferma tutto e fa tacere tutto: nell'ascolto
e nel volo il tempo di gioco si ferma e le voci di gioco vanno a guadagno zero; nella mira si ferma la
scivolata in corso, che alla ripresa ricomincia da capo col suo secondo di silenzio, e i valori già
presi restano. La chiede chi gioca con Ctrl+P, col suo suono e il suo messaggio, oppure la mette la
finestra, in silenzio, quando si apre un dialogo o lei perde l'attivazione: questa seconda, e solo
questa, la toglie la finestra al ritorno. Ciò che succede mentre la partita è ferma, come un tiro il
cui impatto era già partito quando la pausa è arrivata, si dice subito nella vista, ma gli avvisi e
il bonus nuovo aspettano la ripresa.
Il proiettile ha la forma d'onda che chi gioca sceglie nelle Impostazioni (D17), e i bersagli una
portante a caso fra BERSAGLIO_PORTANTI, tolta quella del proiettile, perché non si confondano con lui
(D18): la finestra dà la forma con scegli_proiettile, e una partita tiene quella con cui è cominciata
fino alla fine. La forma scelta a partita in corso vale dalla partita seguente.
"""

import functools
import random
import time

import numpy as np

import fisica
import mira
import resoconti
from bersagli import genera, genera_bonus, portanti_senza, quanti_bersagli, raggio_del_livello, velocita_del_livello
from costanti import (
    ATTENUAZIONE_IN_MIRA_DB,
    CENTRI_PER_LIVELLO,
    COLPI_BONUS,
    COLPI_INIZIALI,
    COLPI_PER_LIVELLO,
    CUBO_X,
    CUBO_Y,
    CUBO_Z,
    ESPONENTE_DEL_RAGGIO,
    LIVELLO_DEI_BONUS,
    MOLTIPLICATORE_BONUS,
    PARAMETRI,
    PAUSA_FRA_BARRE,
    PREMIO_PRECISIONE,
    PROBABILITA_BONUS,
    PROIETTILE_FORMA,
    PROIETTILE_FORME,
    PUNTI_BASE,
    RAGGIO_DI_RIFERIMENTO,
    STACCO_FRA_GLI_AVVISI,
    VELOCITA_DEL_TEMPO,
    VELOCITA_DI_RIFERIMENTO,
)
from motore_audio import VoceTono
from resoconti import Messaggio
from sintesi import db_in_lineare, scivolata
from volo import calcola_volo, parametri_del_proiettile

__all__ = ["CONTI_DEL_PROIETTILE", "FASI", "SOTTOFASI", "Messaggio", "Partita", "esempio_del_proiettile", "punti_del_centro", "volo_d_esempio"]

# Le fasi della partita e le parti di una barra della mira, come le vuole il contratto.
FASI = ("pronta", "ascolto", "mira", "volo", "finita")
SOTTOFASI = ("pausa", "conto", "scivolata")
# Le fasi in cui una partita è in corso: la finestra le usa per chiedere se abbandonarla.
IN_CORSO = ("ascolto", "mira", "volo")
# I suoni con cui la finestra dice, dopo la fine, se il punteggio entra in classifica: li mette in fila
# a dopo_gli_avvisi, e la partita nuova chiesta subito dopo li lascia suonare.
ESITI_DELLA_CLASSIFICA = ("in_classifica", "fuori_classifica")
# Il contatore delle statistiche che ogni partita nuova accresce, secondo la forma del suo proiettile:
# così le statistiche dicono quante partite si giocano con ogni forma (D17).
CONTI_DEL_PROIETTILE = {"seno": "partite_col_seno", "triangolo": "partite_col_triangolo", "sega_morbida": "partite_con_la_sega_morbida"}
# L'esempio del proiettile che il dialogo delle Impostazioni fa sentire (D17): tanti secondi di volo
# di un tiro al centro del cubo, dallo sparo, e una sfumatura finale, in secondi, perché l'esempio
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
    centro del cubo, senza bersagli, sparato al tempo di gioco zero, fino a terra. Si calcola una
    volta sola, e chi lo riceve non lo cambia."""
    centro = tuple((basso + alto) / 2 for basso, alto in (CUBO_X, CUBO_Y, CUBO_Z))
    valori = fisica.soluzione(centro)
    if valori is None:
        raise ValueError("Il centro del cubo non si raggiunge con un tiro dentro i margini: l'esempio del proiettile non ha un volo.")
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


class Partita:
    """Una partita di Ciechino, dall'inizio alla fine, e poi la seguente.
    motore è il Motore audio, suoni i Suoni che suonano sul motore; statistiche, se c'è, riceve ogni
    conto e la funzione dei suoni si collega a lei; rng è il random.Random dei bersagli, uno nuovo se
    è None, così le prove ne danno uno col seme; orologio dà i secondi veri, time.monotonic se è None,
    perché una durata non deve saltare quando Windows rimette l'ora.
    Gli attributi pubblici sono quelli del contratto: fase, barra, sottofase, in_pausa, livello,
    punti, riserva, centri_del_livello, tiri, centri, bersagli (il bonus compreso), valori,
    ultimo_volo; e in più serie, bonus, il bersaglio bonus in volo o None, scivolate, i buffer delle
    tre barre, e inizio_barra, il campione del motore dove comincia il buffer della barra in corso.
    Oltre ai metodi del contratto ci sono pausa_automatica, la pausa silenziosa che mette e toglie la
    finestra (D14), e dopo_gli_avvisi, che dice alla finestra da quale campione far partire l'esito
    della classifica, in fila dopo il suono della fine.
    Il proiettile (D17): proiettile è la forma d'onda dell'unico proiettile della partita in corso, o
    di quella che comincerà se nessuna è in corso; proiettile_scelto quella che chi gioca ha chiesto,
    da cui ogni partita nuova prende la sua. Si sceglie con scegli_proiettile, o all'inizio con
    l'argomento proiettile, PROIETTILE_FORMA se è None."""

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
        # I buffer delle tre barre si preparano una volta sola: sintetizzarli a ogni mira farebbe
        # aspettare il giocatore, e il buffer è sempre lo stesso.
        self.scivolate = {parametro: scivolata(parametro, fs=motore.fs) for parametro in PARAMETRI}
        self.fase = "pronta"
        self.barra = None
        self.sottofase = None
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
        self.inizio_barra = None
        # Le voci dei bersagli, una per bersaglio: Bersaglio si confronta per identità, quindi fa da chiave.
        self._voci = {}
        self._voce_barra = None
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
        self._inizio = None
        self._pause = 0.0
        self._pausa_dal = None
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
        """{"punteggio", "livello", "tiri", "centri", "durata_s"} della partita in corso o dell'ultima;
        la durata è quella vera, senza le pause, al decimo di secondo."""
        durata = self._durata if self._durata is not None else self._durata_adesso()
        return {"punteggio": self.punti, "livello": self.livello, "tiri": self.tiri, "centri": self.centri, "durata_s": round(durata, 1)}

    def righe_della_barra(self):
        """Le righe della barra di stato, da resoconti.righe_della_barra."""
        bonus = self.bonus is not None and self.bonus in self.bersagli
        normali = sum(1 for bersaglio in self.bersagli if not bersaglio.bonus)
        parametro = PARAMETRI[self.barra] if self.barra is not None else None
        return resoconti.righe_della_barra(self.fase, in_pausa=self.in_pausa, livello=self.livello, punti=self.punti, riserva=self.riserva,
                                           centri_del_livello=self.centri_del_livello, bersagli=normali, bonus=bonus, parametro=parametro,
                                           sottofase=self.sottofase)

    # I comandi.

    def nuova(self):
        """Comincia una partita nuova, abbandonando prima quella in corso: livello 1, riserva
        COLPI_INIZIALI, il proiettile della forma scelta, i bersagli del primo livello che nascono al
        prossimo campione sicuro, col tempo di gioco che scorre da lì. Il suono della partita nuova
        aspetta in fila gli avvisi della partita di prima, e dopo la fine anche l'esito della
        classifica che la finestra vi ha messo, perché lo spazio dato d'abitudine subito dopo
        l'ultimo tiro non li copra."""
        finita = self.fase == "finita"
        messaggi = self.abbandona() if self.in_corso() else []
        self._ferma_voci()
        self.proiettile = self.proiettile_scelto
        self.livello = 1
        self.riserva = COLPI_INIZIALI
        self.punti = self.tiri = self.centri = self.serie = self.centri_del_livello = 0
        self.valori = []
        self.ultimo_volo = None
        self.barra = self.sottofase = self.inizio_barra = None
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
        """La barra spaziatrice: senza partita ne comincia una; nell'ascolto avvia la mira; nella mira
        prende il valore della barra in corso; col proiettile in volo, o in pausa, non può fare niente."""
        if self.fase in ("pronta", "finita"):
            return self.nuova()
        if self.in_pausa or self.fase == "volo":
            self.suoni.suona("non_ora")
            return []
        if self.fase == "ascolto":
            return self._avvia_mira()
        return self._spazio_nella_mira()

    def battito(self):
        """Il giro del timer della finestra: segue la mira, si accorge dell'impatto sentito e dei bonus
        usciti."""
        if self.fase == "mira":
            return self._battito_della_mira()
        if self.fase == "volo":
            return self._battito_del_volo()
        if self.fase == "ascolto":
            return self._controlla_il_bonus()
        return []

    def pausa(self):
        """Ctrl+P, la pausa chiesta da chi gioca (D14): con la partita in corso la ferma e la fa tacere,
        col suono della pausa, e la conta in «pause»; in pausa, anche in quella automatica, la fa
        riprendere col suono della ripresa, perché la barra di stato dice «Ctrl+P per riprendere» in
        tutte e due. Il motore resta aperto, così i suoni della pausa e della ripresa si sentono.
        Senza partita non c'è niente da fermare, e lo dice il suono del non ora, come lo spazio fuori
        tempo: con la lettura dei resoconti spenta è l'unico segno che il tasto è arrivato."""
        if self.fase not in IN_CORSO:
            self.suoni.suona("non_ora")
            return [resoconti.nessuna_partita()]
        if not self.in_pausa:
            nella_mira = self.fase == "mira"
            self._sospendi()
            self.suoni.suona("pausa")
            self._conta("pause")
            return [resoconti.in_pausa(nella_mira=nella_mira)]
        # Gli avvisi rimandati dalla pausa vengono in fila dopo il suono della ripresa, non sopra.
        self._accoda(self.suoni.suona("ripresa"))
        return [resoconti.ripresa(), *self._riprendi()]

    def pausa_automatica(self, attiva):
        """La pausa che mette e toglie la finestra (D14): attiva vera quando si apre un dialogo modale o
        la finestra perde l'attivazione, falsa quando il dialogo si chiude e la finestra è di nuovo
        attiva. Con vero, se la partita è in corso e non è già in pausa, la ferma come Ctrl+P, si
        ricorda che la pausa è automatica e la conta in «pause_automatiche». Con falso la fa
        riprendere, ma soltanto se la pausa è automatica: quella chiesta con Ctrl+P resta finché non
        la si toglie con Ctrl+P. Tutte e due in silenzio, senza suoni né messaggi della pausa, perché
        chi gioca non l'ha chiesta e la finestra ha già il suo dialogo da dire; la barra di stato dice
        comunque la fase. Restituisce una lista di Messaggio, di solito vuota: alla ripresa può
        portare il bonus nato a partita ferma."""
        if attiva:
            if self.fase in IN_CORSO and not self.in_pausa:
                self._sospendi()
                self._pausa_automatica = True
                self._conta("pause_automatiche")
            return []
        if self.in_pausa and self._pausa_automatica:
            return self._riprendi()
        return []

    def abbandona(self):
        """Lascia la partita in corso: ferma le voci di gioco, la conta come abbandonata e torna a
        «pronta». Senza partita in corso non fa niente."""
        if not self.in_corso():
            return []
        messaggi = self._chiudi("abbandonata")
        self.fase = "pronta"
        return messaggi

    # La pausa.

    def _sospendi(self):
        """Ferma la partita in corso e la fa tacere, per la pausa chiesta e per quella automatica.
        Nell'ascolto e nel volo il tempo di gioco si ferma al prossimo campione sicuro; nella mira, dove
        è già fermo, si ferma la scivolata in corso, che alla ripresa ricomincia da capo, perché un
        valore preso a metà di una barra interrotta non sarebbe una scelta. Le voci di gioco vanno a
        guadagno zero, con la rampa del motore contro i click."""
        if self.fase == "mira":
            if self._voce_barra is not None:
                self.motore.ferma(self._voce_barra)
            self._voce_barra = None
        else:
            self.motore.ferma_tempo(self.motore.prossimo_campione())
        for voce in self._voci_di_gioco():
            self.motore.guadagno_voce(voce, 0.0)
        self.in_pausa = True
        self._pausa_automatica = False
        self._pausa_dal = self.orologio()

    def _riprendi(self):
        """La ripresa, dopo l'una o l'altra pausa: nell'ascolto e nel volo il tempo riprende dal
        prossimo campione sicuro e le voci tornano a guadagno pieno; nella mira i bersagli tornano
        attenuati e la barra in corso riparte da capo, col suo secondo di silenzio e i suoi bip, con i
        valori già presi. Poi suonano in fila gli avvisi rimandati, e nasce il bonus che la pausa ha
        fatto aspettare: i suoi messaggi sono quelli che restituisce."""
        k = self.motore.prossimo_campione()
        if self.fase == "mira":
            guadagno = db_in_lineare(ATTENUAZIONE_IN_MIRA_DB)
        else:
            guadagno = 1.0
            self.motore.fai_scorrere_tempo(k)
        for voce in self._voci_di_gioco():
            self.motore.guadagno_voce(voce, guadagno)
        self.in_pausa = False
        self._pausa_automatica = False
        if self._pausa_dal is not None:
            self._pause += max(0.0, self.orologio() - self._pausa_dal)
        self._pausa_dal = None
        if self.fase == "mira":
            self._avvia_barra(self.barra, k)
            return []
        rimandati, self._avvisi_rimandati = self._avvisi_rimandati, []
        for evento in rimandati:
            self._avviso(evento)
        if self._bonus_rimandato:
            self._bonus_rimandato = False
            return self._forse_un_bonus()
        return []

    def _azzera_la_pausa(self):
        """Nessuna pausa e niente in attesa della ripresa: per la partita nuova e per quella che finisce."""
        self.in_pausa = False
        self._pausa_automatica = False
        self._pausa_dal = None
        self._avvisi_rimandati = []
        self._bonus_rimandato = False

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
        """Ferma tutte le voci di gioco, bersagli, proiettile e scivolata, e gli eventi programmati.
        L'effetto di un impatto già partito si conta, perché si è sentito anche se il tiro non si
        risolverà più; quello annullato prima di partire non si conta."""
        self._ferma_bersagli()
        for voce in (self._voce_proiettile, self._voce_barra):
            if voce is not None:
                self.motore.ferma(voce)
        self._voce_proiettile = self._voce_barra = None
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

    def _pronto_senza_conto(self, evento, posizione):
        """Il buffer di un effetto preparato senza contarlo: Suoni conta un effetto quando è pronto,
        ma l'uscita del bonus non si sente se il bonus viene colpito prima, e l'impatto non si sente
        se la partita si abbandona col proiettile in volo, e allora il conto sarebbe falso. Lo si
        conta quando il battito sa che è suonato."""
        conta, self.suoni.conta = self.suoni.conta, None
        try:
            return self.suoni.pronto(evento, posizione=posizione)
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
        tutto dal cubo, che nel filo dell'audio ne ferma la voce e fa suonare l'effetto già pronto,
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

    # La mira.

    def _scivolata(self):
        return self.scivolate[PARAMETRI[self.barra]]

    def _avvia_mira(self):
        """Lo spazio nell'ascolto: il tempo di gioco si ferma al prossimo campione sicuro, i bersagli
        scendono di ATTENUAZIONE_IN_MIRA_DB con la rampa del motore, e parte la prima barra (D4)."""
        k = self.motore.prossimo_campione()
        self.motore.ferma_tempo(k)
        attenuazione = db_in_lineare(ATTENUAZIONE_IN_MIRA_DB)
        for voce in self._voci.values():
            self.motore.guadagno_voce(voce, attenuazione)
        self.suoni.suona("mira_avviata")
        self._conta("mire_avviate")
        self.valori = []
        self.fase = "mira"
        self._avvia_barra(0, k)
        return []

    def _avvia_barra(self, indice, k):
        """La barra indice comincia dopo PAUSA_FRA_BARRE di silenzio dal campione k: il buffer parte a
        pausa finita, e la partita si ricorda dove, per il conto di mira.valore_scelto. Se gli avvisi
        in fila dopo l'impatto, come il livello nuovo o l'ultimo colpo, o la coda dell'effetto stesso,
        suonano ancora a pausa finita, la pausa si allunga fino alla loro fine, stacco compreso: la
        mira avviata subito dopo l'impatto non copre i bip con gli effetti lunghi di D16. Lo spazio
        nella pausa allungata non conta, come nell'altra."""
        self.barra = indice
        self.sottofase = "pausa"
        self.inizio_barra = max(k + round(PAUSA_FRA_BARRE * self.motore.fs), self._fine_degli_avvisi)
        self._voce_barra = self.motore.suona(self._scivolata().buffer, inizio=self.inizio_barra)

    def _spazio_nella_mira(self):
        """Lo spazio nella mira vale secondo il campione udibile; nella pausa non conta e non suona,
        così assorbe i doppi colpi e le ripetizioni del tasto che arrivano con NVDA."""
        valore, esito = mira.valore_scelto(self.motore.campione_udibile(), self.inizio_barra, self._scivolata())
        if esito == "pausa":
            return []
        return self._prendi(valore, esito)

    def _battito_della_mira(self):
        """Aggiorna la parte della barra che si sente, e prende il massimo della scivolata che nessuno
        ha fermato (D3). A partita ferma la barra è ferma, e non scade."""
        if self.in_pausa:
            return []
        udibile = self.motore.campione_udibile()
        barra = self._scivolata()
        posto = udibile - self.inizio_barra
        if posto < 0:
            self.sottofase = "pausa"
        elif posto < barra.inizio_scivolata:
            self.sottofase = "conto"
        else:
            self.sottofase = "scivolata"
        if mira.scaduta(udibile, self.inizio_barra, barra):
            return self._prendi(1.0, "massimo")
        return []

    def _prendi(self, valore, esito):
        """Un valore preso: si registra, la scivolata si ferma, suona valore_preso, si conta, e la mira
        passa alla barra seguente o allo sparo."""
        parametro = PARAMETRI[self.barra]
        self.valori.append(float(valore))
        if self._voce_barra is not None:
            self.motore.ferma(self._voce_barra)
        self._voce_barra = None
        self.suoni.suona("valore_preso")
        if self.statistiche is not None:
            self.statistiche.mira(parametro, valore, esito)
        if self.barra + 1 < len(PARAMETRI):
            self._avvia_barra(self.barra + 1, self.motore.prossimo_campione())
            return []
        return self._spara()

    # Il tiro.

    def _spara(self):
        """Il colpo: la traiettoria dai tre valori, il volo calcolato tutto subito dal tempo di gioco
        dello sparo, lo sparo dal cannone e la voce del proiettile, nella forma della partita, dal
        campione k, i bersagli di nuovo a guadagno pieno e il tempo che riprende da k. L'effetto
        dell'impatto si prepara adesso, perché nel filo dell'audio non c'è tempo per farlo; l'evento
        della fine del volo, al campione esatto, ferma il proiettile e il bersaglio colpito e fa
        partire l'effetto. L'effetto si conta quando si sente, non adesso: se la partita si abbandona
        col proiettile in volo non suona mai.
        Il conto del volo e la preparazione dell'effetto vengono prima di scegliere k: durano qualche
        millesimo, e un k scelto prima potrebbe essere già passato quando i comandi arrivano al
        motore. Nella mira il tempo di gioco è fermo, quindi la partenza è la stessa a qualunque
        campione la si chieda."""
        traiettoria = fisica.Traiettoria(*fisica.parametri_del_tiro(*self.valori))
        partenza = self.motore.tempo_di_gioco(self.motore.prossimo_campione())
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
        k = self.motore.prossimo_campione()
        proiettile = VoceTono(parametri_del_proiettile(volo), self.proiettile, inizio=k)
        self.suoni.suona("sparo", posizione=fisica.CANNONE, inizio=k)
        self.motore.aggiungi(proiettile)
        for voce in self._voci.values():
            self.motore.guadagno_voce(voce, 1.0)
        self.motore.fai_scorrere_tempo(k)
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
        self.barra = self.sottofase = None
        return []

    def _battito_del_volo(self):
        """Il tiro si risolve quando l'impatto si è sentito: l'evento è scattato e il campione udibile
        lo ha passato, oppure il tempo di gioco al campione udibile ha passato la fine del volo. A
        partita ferma il tempo non arriva alla fine del volo, ma un impatto scattato prima della pausa
        si risolve lo stesso: si è sentito, e il resoconto lo dice."""
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
        la nuova fase d'ascolto, o la fine della partita. Gli avvisi che il tiro causa partono in fila
        dopo l'effetto dell'impatto, che sta ancora suonando."""
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
        messaggi = [resoconti.resoconto_del_tiro(self.tiri, livello_del_tiro, volo, self.valori, punti=punti, totale=self.punti, riserva=self.riserva,
                                                 colpi_bonus=colpi_bonus)]
        battuti = self._statistiche_del_tiro(volo, punti)
        # Il bonus uscito durante il volo si dice e si conta prima che il livello nuovo o la fine
        # svuotino il cielo: la sua uscita si è già sentita, e senza il messaggio sparirebbe dai conti.
        messaggi.extend(self._controlla_il_bonus())
        if colpito is not None and self.centri_del_livello >= CENTRI_PER_LIVELLO:
            messaggi.extend(self._sali_di_livello())
        elif colpito is not None and not colpito.bonus:
            # D5: un bersaglio nuovo nasce soltanto al posto di uno colpito.
            self._nuovo_bersaglio()
        primati = resoconti.primati_battuti(battuti, self._valori_dei_primati(battuti))
        if primati is not None:
            messaggi.append(primati)
        if self.riserva <= 0:
            messaggi.extend(self._chiudi("riserva_vuota"))
            self.fase = "finita"
            return messaggi
        if colpito is None and self.riserva == 1:
            self._avviso("ultimo_colpo")
            messaggi.append(resoconti.ultimo_colpo())
        self.fase = "ascolto"
        if self.in_pausa:
            # Un bonus che nascesse adesso entrerebbe a partita ferma: nasce alla ripresa (D14).
            self._bonus_rimandato = True
        else:
            messaggi.extend(self._forse_un_bonus())
        return messaggi

    def _statistiche_del_tiro(self, volo, punti):
        """Il tiro nelle statistiche, e i primati battuti da dire. La distanza del primato è dal centro
        del bersaglio, come la dice la pagina delle statistiche: nel centro quella del volo, nel tiro a
        vuoto la distanza dalla superficie del più vicino più il suo raggio."""
        if self.statistiche is None:
            return []
        colpito = volo.colpito
        if colpito is not None:
            distanza = volo.distanza_al_centro
        elif volo.vicino is not None and volo.distanza_dal_vicino is not None:
            distanza = volo.distanza_dal_vicino + volo.vicino.raggio
        else:
            distanza = None
        esito = {"colpito": colpito is not None, "bonus": bool(colpito is not None and colpito.bonus), "forma": colpito.forma if colpito is not None else None,
                 "portante": colpito.portante if colpito is not None else None, "mobile": bool(colpito is not None and colpito.mobile), "precisione": volo.precisione, "punti": punti, "distanza_dal_vicino": distanza,
                 "serie": self.serie}
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

    def _sali_di_livello(self):
        """I centri hanno chiuso il livello: il livello sale, arrivano COLPI_PER_LIVELLO colpi, e tutti
        i bersagli, il bonus compreso, lasciano il posto a quelli del livello nuovo."""
        self.livello += 1
        self.centri_del_livello = 0
        self.riserva += COLPI_PER_LIVELLO
        self._conta("livelli_superati")
        self._conta("colpi_guadagnati", COLPI_PER_LIVELLO)
        self._ferma_bersagli()
        self._nuovi_bersagli(self.motore.prossimo_campione())
        self._avviso("livello_su")
        return [resoconti.livello_nuovo(self.livello, COLPI_PER_LIVELLO, self.riserva)]

    # La fine.

    def _durata_adesso(self):
        """I secondi veri dall'inizio della partita, senza le pause, quella in corso compresa."""
        if self._inizio is None:
            return 0.0
        adesso = self.orologio()
        pause = self._pause + (max(0.0, adesso - self._pausa_dal) if self._pausa_dal is not None else 0.0)
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
        self.barra = self.sottofase = None
        riepilogo = self.riepilogo()
        battuti = []
        if self.statistiche is not None:
            prima = self._primati_attuali()
            battuti = self._primati_che_cambiano(self.statistiche.partita_finita({**riepilogo, "motivo": motivo}), prima)
            # Le statistiche non si salvano da sole: la fine di una partita è il momento in cui
            # perderle costerebbe di più.
            self.statistiche.salva()
        valori = self._valori_dei_primati(battuti)
        if motivo == "abbandonata":
            return [resoconti.partita_abbandonata(riepilogo, battuti, valori)]
        self._avviso("partita_finita")
        return [resoconti.fine_partita(riepilogo, battuti, valori)]
