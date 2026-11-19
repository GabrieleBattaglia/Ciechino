"""
I suoni di Ciechino: un preset della collezione condivisa di GBUtils per ogni evento del gioco e della
finestra, sintetizzato una volta sola e suonato dal motore audio di Ciechino.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello dei suoni di MESS.
Le regole di Gabriele: ogni evento ha il suo preset, e nessun preset serve due eventi; nessuno è
un'onda quadra, salvo donazione per l'invito al caffè, come in tutto il parco; gli eventi che
suonano spesso, cioè la mira avviata, il valore preso, F5 e F7, sono brevi e morbidi; lo sparo, il
bersaglio colpito, il bonus colpito e il colpo a terra sono fatti soprattutto di rumore, come vuole
idee.txt; gli esiti diversi si riconoscono a orecchio.
I preset sono tutti nuovi, col prefisso ciechino_, salvo donazione, e portano la firma ciechino nel
campo usato_da (decisione D10): li aggiunge e li firma strumenti/crea_preset.py, che tiene la
tabella dei loro score e le loro descrizioni, e li controlla.
GRUPPI, EVENTI e AZIONI sono le tabelle: i gruppi per l'elenco dei suoni, la mappa da evento a
preset, e l'azione che fa suonare ogni evento, detta a parole e tradotta. L'elenco dei suoni per
Acu_Maker, suoni_di_ciechino.txt, lo scrive strumenti/elenco_suoni.py da queste tabelle, e una prova
controlla che sia aggiornato.
I preset non passano dal mixer di Acusticator: il motore di Ciechino è uno solo, con un flusso e un
orologio (decisione D12), quindi _rendi_preset li sintetizza a FS con Acusticator.preset e
Acusticator.sintetizza e il motore li mette al campione giusto.
La classe Suoni è ciò che usano la partita e la finestra: suona un evento subito, o al campione
chiesto, e prepara il buffer di un effetto che il motore farà partire più tardi dentro un evento
programmato, dove non c'è tempo per sintetizzare né per collocare nulla. Un evento che viene da un
punto dello spazio, come il bersaglio colpito o il colpo a terra, si sente con la stessa panoramica
e lo stesso guadagno della distanza dei bersagli e del proiettile, dati da fisica.suono_di, ma con
l'altezza del suo preset: la frequenza dello spazio la portano già le voci che si muovono.
numpy, fisica e sintesi si importano dentro le funzioni, così chi legge soltanto le tabelle, come
strumenti/elenco_suoni.py, resta leggero.
"""

import sys

from costanti import FS, NOME
from lingua import _

# Gli eventi, gruppo per gruppo: ogni gruppo ha il suo titolo e i suoi eventi con il preset della
# collezione. È la forma del contratto in docs/progetto.txt, e l'ordine dell'elenco dei suoni.
GRUPPI = [
    (_("La partita, l'inizio e la fine"), {
        # Il cannone che si arma apre la partita; il battito di cuore avverte dell'ultimo colpo; la
        # fine e i due esiti della classifica chiudono, con due frasi dello stesso ottone.
        "nuova_partita": "ciechino_nuova_partita",
        "ultimo_colpo": "ciechino_ultimo_colpo",
        "partita_finita": "ciechino_partita_finita",
        "in_classifica": "ciechino_in_classifica",
        "fuori_classifica": "ciechino_fuori_classifica",
    }),
    (_("La mira"), {
        # I due suoni più frequenti del gioco, brevi e morbidi: il respiro che avvia la mira, senza
        # altezza, e lo scatto che scende, mentre i bip delle scivolate salgono.
        "mira_avviata": "ciechino_mira_avviata",
        "valore_preso": "ciechino_valore_preso",
    }),
    (_("Il tiro e i suoi esiti"), {
        # Tutti di rumore, ciascuno col suo colore: bianco lo sparo, rosa il bersaglio colpito,
        # azzurro il bonus, marrone il tonfo a terra.
        "sparo": "ciechino_sparo",
        "colpito": "ciechino_colpito",
        "colpito_bonus": "ciechino_colpito_bonus",
        "a_terra": "ciechino_a_terra",
    }),
    (_("I livelli e i bersagli bonus"), {
        "livello_su": "ciechino_livello_su",
        "bonus_appare": "ciechino_bonus_appare",
        "bonus_uscito": "ciechino_bonus_uscito",
    }),
    (_("La pausa e lo spazio a vuoto"), {
        "pausa": "ciechino_pausa",
        "ripresa": "ciechino_ripresa",
        "non_ora": "ciechino_non_ora",
    }),
    (_("La finestra, l'avvio, l'uscita e gli errori"), {
        "avvio": "ciechino_avvio",
        "uscita": "ciechino_uscita",
        # La finestra della 0.1.0: l'uscita che riapre Ciechino, e le notizie che seguono l'avvio.
        "riavvio": "ciechino_riavvio",
        "primo_piano": "ciechino_primo_piano",
        "errore": "ciechino_errore",
        "crash_precedente": "ciechino_crash_precedente",
        "dati_rovinati": "ciechino_dati_rovinati",
        "salvataggio_non_riuscito": "ciechino_salvataggio_non_riuscito",
    }),
    (_("La finestra, i menu"), {
        "classifica": "ciechino_classifica",
        "statistiche": "ciechino_statistiche",
        "volume": "ciechino_volume",
        # La scelta della forma del proiettile (D17): una piccola parabola, come un colpo in volo.
        "proiettile": "ciechino_proiettile",
        # La lettura dei resoconti con la sintesi: l'interruttore che si accende e quello che si spegne.
        "impostazioni": "ciechino_impostazioni",
        "sintesi_spenta": "ciechino_sintesi_spenta",
        "lingua": "ciechino_lingua",
        "lingua_scelta": "ciechino_lingua_scelta",
        "guida": "ciechino_guida",
        "manuale": "ciechino_manuale",
        "novita": "ciechino_novita",
        "informazioni": "ciechino_informazioni",
        "file_mancante": "ciechino_file_mancante",
        # L'invito al caffè suona come in tutto il parco: l'unica onda quadra ammessa.
        "caffe": "donazione",
        "caffe_paypal": "ciechino_caffe_paypal",
        # Le due voci del PUB, ciascuna col suo suono: il brindisi di chi entra e la campanella di chi chiede aiuto.
        "pub": "ciechino_pub",
        "pub_aiuto": "ciechino_pub_aiuto",
    }),
    (_("La finestra, i dialoghi"), {
        "domanda": "ciechino_domanda",
        "confermato": "ciechino_confermato",
        "annullato": "ciechino_annullato",
        "campo_da_correggere": "ciechino_campo_da_correggere",
    }),
    (_("La finestra, l'aggiornamento automatico"), {
        "aggiornamento_disponibile": "ciechino_aggiornamento_disponibile",
        "aggiornamento_scelto": "ciechino_aggiornamento_scelto",
        "aggiornamento_rimandato": "ciechino_aggiornamento_rimandato",
        "aggiornamento_scaduto": "ciechino_aggiornamento_scaduto",
        "aggiornamento_pronto": "ciechino_aggiornamento_pronto",
        "aggiornamento_non_pronto": "ciechino_aggiornamento_non_pronto",
        "aggiornamento_non_riuscito": "ciechino_aggiornamento_non_riuscito",
    }),
    (_("La finestra, il fuoco"), {
        # F5 sale verso la vista, F7 scende verso la barra: l'uno lo specchio dell'altro.
        "vai_alla_vista": "ciechino_vai_alla_vista",
        "vai_alla_barra": "ciechino_vai_alla_barra",
    }),
]
EVENTI = {evento: preset for _titolo, gruppo in GRUPPI for evento, preset in gruppo.items()}

# Ogni evento detto a parole: l'azione che lo fa suonare, in minuscolo, perché l'elenco dei suoni la
# mette dopo il titolo del gruppo e le dà lui la maiuscola. Una prova controlla che ogni evento abbia
# la sua.
AZIONI = {
    "nuova_partita": _("comincia una partita nuova, con lo spazio quando nessuna partita è in corso, con Ctrl+N o con Nuova partita del menu Partita"),
    "ultimo_colpo": _("nella riserva resta l'ultimo colpo"),
    "partita_finita": _("la riserva è vuota e la partita finisce"),
    "in_classifica": _("a partita finita il punteggio entra nella classifica dei migliori: suona dopo la fine, e quando tace si apre la finestra del nome"),
    "fuori_classifica": _("a partita finita il punteggio non entra nella classifica dei migliori: suona dopo la fine"),
    "mira_avviata": _("lo spazio, mentre si ascoltano i bersagli, avvia la mira: il tempo del gioco si ferma e i bersagli si sentono più piano"),
    "valore_preso": _("la mira prende un valore, la direzione, l'alzo o la potenza, con lo spazio o alla fine della scivolata"),
    "sparo": _("dopo il terzo valore della mira parte il colpo dal cannone"),
    "colpito": _("il proiettile colpisce un bersaglio, e il suono viene dal punto dell'impatto"),
    "colpito_bonus": _("il proiettile colpisce un bersaglio bonus, e il suono viene dal punto dell'impatto"),
    "a_terra": _("il proiettile arriva a terra senza aver colpito niente, e il suono viene dal punto dove cade"),
    "livello_su": _("i centri del livello sono fatti: si sale di livello, con altri colpi nella riserva e bersagli nuovi"),
    "bonus_appare": _("compare un bersaglio bonus, che attraversa lo spazio dei bersagli e poi esce"),
    "bonus_uscito": _("il bersaglio bonus esce dallo spazio dei bersagli senza essere stato colpito"),
    # La pausa automatica, quella di un dialogo aperto o di un'altra finestra, non ha suoni (D14): le
    # due azioni lo dicono, perché chi ritocca i suoni non li cerchi lì.
    "pausa": _("la partita va in pausa, con Ctrl+P o con Pausa del menu Partita, anche durante la mira; la pausa automatica non suona"),
    "ripresa": _("la partita riprende dalla pausa, con Ctrl+P o con Pausa del menu Partita; la ripresa automatica non suona"),
    "non_ora": _("lo spazio premuto quando non può fare niente, col proiettile in volo o in pausa, oppure Ctrl+P senza una partita in corso"),
    "avvio": _("il programma si apre"),
    "uscita": _("l'uscita dal programma, con Ctrl+Q, con Esci del menu Partita o chiudendo la finestra"),
    "riavvio": _("Ciechino si chiude per riaprirsi subito, dopo il sì alla domanda che segue la scelta di una lingua nuova"),
    "primo_piano": _("si apre Ciechino mentre è già aperto, e la finestra aperta torna in primo piano"),
    "errore": _("un errore imprevisto"),
    "crash_precedente": _("all'avvio, dopo il suono d'apertura: la volta scorsa Ciechino si è chiuso per un errore grave, e la vista dice dov'è il resoconto"),
    "dati_rovinati": _("un file dei dati, la classifica, le statistiche o le impostazioni, non si leggeva: Ciechino lo mette da parte e riparte dalla copia o da zero"),
    "salvataggio_non_riuscito": _("la classifica, le statistiche o le impostazioni non si sono potute salvare"),
    "classifica": _("si apre la classifica dei migliori, con Ctrl+K o con Classifica del menu Partita"),
    "statistiche": _("si aprono le statistiche, con Ctrl+T o con Statistiche del menu Partita"),
    "volume": _("Volume del menu Impostazioni, all'apertura e a ogni ritocco, al volume scritto nel campo"),
    "proiettile": _("si apre la scelta della forma del proiettile, con Proiettile del menu Impostazioni"),
    "impostazioni": _("la lettura dei resoconti con la sintesi si accende, con Leggi i resoconti con la sintesi del menu Impostazioni"),
    "sintesi_spenta": _("la lettura dei resoconti con la sintesi si spegne, con Leggi i resoconti con la sintesi del menu Impostazioni"),
    "lingua": _("si apre la scelta della lingua, con Lingua del menu Impostazioni"),
    "lingua_scelta": _("si sceglie una lingua nuova, che vale dal prossimo avvio"),
    "guida": _("F1, la guida ai comandi, dal menu Aiuto"),
    "manuale": _("Maiusc+F1, il manuale, dal menu Aiuto"),
    "novita": _("F2, le novità della versione, dal menu Aiuto"),
    "informazioni": _("F3, le informazioni sul programma, dal menu Aiuto"),
    "file_mancante": _("il manuale o il file delle novità non c'è, o non si apre"),
    "caffe": _("Offrimi un caffè, dal menu Aiuto"),
    "caffe_paypal": _("Dona con PayPal, nell'invito al caffè: si apre PayPal nel browser"),
    "pub": _("Entra al PUB, dal menu Aiuto"),
    "pub_aiuto": _("Chiedi aiuto al PUB, dal menu Aiuto"),
    "domanda": _("una domanda col sì e col no, come Ctrl+N o l'uscita con una partita in corso"),
    "confermato": _("un dialogo confermato con OK, come il nome in classifica, il volume o il proiettile, o una mail del PUB pronta"),
    "annullato": _("un dialogo chiuso con Annulla o con Esc, o il no a una domanda"),
    "campo_da_correggere": _("OK in un dialogo con un campo da correggere, come un nome vuoto in classifica"),
    "aggiornamento_disponibile": _("all'avvio del programma compilato, su GitHub c'è una versione nuova, e la proposta si apre"),
    "aggiornamento_scelto": _("Aggiorna, nella proposta dell'aggiornamento: la versione nuova si scarica"),
    "aggiornamento_rimandato": _("Non adesso, nella proposta dell'aggiornamento, o Esc"),
    "aggiornamento_scaduto": _("la proposta dell'aggiornamento resta senza risposta e si chiude da sola"),
    "aggiornamento_pronto": _("l'aggiornamento è scaricato: Ciechino si chiude per installarlo e si riapre da solo"),
    "aggiornamento_non_pronto": _("la versione nuova c'è, ma il suo pacchetto non è ancora pronto"),
    "aggiornamento_non_riuscito": _("l'aggiornamento non si è scaricato, o non si è installato"),
    "vai_alla_vista": _("F5, il fuoco va nella vista della partita"),
    "vai_alla_barra": _("F7, il fuoco va nella barra di stato"),
}

# I buffer dei preset già sintetizzati, per nome: la sintesi di un rumore o di un'onda sovracampionata
# costa qualche millesimo, troppo per farla a ogni sparo o dentro un evento programmato del motore.
_BUFFER_DEI_PRESET = {}


def _rendi_preset(nome):
    """Il buffer di un preset della collezione, stereo (N, 2) float32 a FS: lo sintetizza la prima
    volta con Acusticator.preset e Acusticator.sintetizza, poi lo restituisce dalla memoria.
    Un preset che la collezione non ha dà un buffer vuoto, (0, 2), una volta sola con l'avviso di
    Acusticator su stderr: il gioco resta muto su quell'evento invece di fermarsi.
    Il buffer è in sola lettura, perché è lo stesso per tutte le chiamate: chi lo vuole cambiare,
    per esempio per collocarlo nello spazio, ne fa una copia. Le prove sostituiscono questa funzione."""
    buffer = _BUFFER_DEI_PRESET.get(nome)
    if buffer is None:
        import numpy as np
        from GBUtils import Acusticator

        score, kind, adsr = Acusticator.preset(nome)
        sintetizzato = Acusticator.sintetizza(score, kind, adsr, fs=FS) if score else None
        if sintetizzato is None:
            buffer = np.zeros((0, 2), dtype=np.float32)
        else:
            buffer = np.array(sintetizzato, dtype=np.float32, copy=True).reshape(-1, 2)
        buffer.flags.writeable = False
        _BUFFER_DEI_PRESET[nome] = buffer
    return buffer


class Suoni:
    """I suoni degli eventi di Ciechino, suonati dal motore audio della partita.
    motore è il Motore che li suona; conta, se c'è, è una funzione chiamata col nome dell'evento a
    ogni suono, e la partita la collega a Statistiche.suono. È un attributo pubblico, perché la
    partita riceve i suoni già fatti: suoni.conta = statistiche.suono la collega anche dopo.
    Si conta ogni evento che suona: quelli di suona quando la voce parte, quelli di pronto, di solito,
    quando il buffer è pronto. Un effetto preparato però può non suonare mai: l'impatto di un
    proiettile in una partita abbandonata in volo, l'uscita di un bonus colpito prima di uscire. Chi
    prepara un effetto così lo chiede a pronto con conta_subito falso, e lo conta con suonato quando
    sa che il motore lo ha fatto partire, perché le statistiche contano i suoni sentiti.
    Un evento sconosciuto, o un preset che manca o non si sintetizza, non fa niente e non si conta:
    il gioco resta muto lì invece di fermarsi, e stderr, cioè il registro degli errori nel
    compilato, lo dice una volta sola, per non riempirlo a ogni battito.
    Ogni preset si chiede a _rendi_preset una volta sola e resta qui in due forme: stereo, così come
    lo dà la collezione, per i suoni senza posizione, e mono, per collocarlo nello spazio. Suoni si
    usa dal filo principale: nel filo dell'audio, dentro un evento programmato, va soltanto il
    buffer già fatto da pronto."""

    def __init__(self, motore, conta=None):
        import numpy as np

        from sintesi import panoramica

        self.motore = motore
        self.conta = conta
        self._stereo = {}
        self._mono = {}
        self._gia_detti = set()
        # Il guadagno di ciascun lato di un suono al centro, secondo la legge di sintesi.panoramica,
        # la stessa di Acusticator. Quasi tutti i preset stanno al centro: dividendo per il doppio di
        # questo guadagno la somma dei due lati, il mono torna quello che il preset aveva prima della
        # panoramica, e un suono collocato proprio davanti al cannone esce identico all'originale,
        # moltiplicato soltanto per il guadagno della distanza.
        self._al_centro = float(panoramica(np.ones(1), 0.0)[0, 0])

    def _avvisa(self, chiave, testo):
        """Scrive il testo su stderr la prima volta che la chiave si presenta, poi tace."""
        if chiave in self._gia_detti:
            return
        self._gia_detti.add(chiave)
        print(f"{NOME}: {testo}", file=sys.stderr)

    def _preset(self, evento):
        """Il preset dell'evento, oppure None per un evento sconosciuto, con l'avviso."""
        try:
            preset = EVENTI.get(evento)
        except TypeError:
            # Un evento che non si può nemmeno cercare, come una lista, è sconosciuto come gli altri.
            preset = None
        if preset is None:
            self._avvisa(("evento", repr(evento)), f"evento sonoro sconosciuto, {evento!r}")
        return preset

    def _stereo_di(self, preset):
        """Il buffer stereo del preset, (N, 2) float32 in sola lettura, sintetizzato una volta sola.
        Un preset che non si sintetizza vale un buffer vuoto, e non si riprova: rifarlo a ogni sparo
        costerebbe il tempo che la memoria serve a risparmiare."""
        buffer = self._stereo.get(preset)
        if buffer is None:
            import numpy as np

            try:
                buffer = np.asarray(_rendi_preset(preset), dtype=np.float32)
                if buffer.ndim != 2 or buffer.shape[1] != 2:
                    raise ValueError(f"buffer di forma {buffer.shape} invece di (N, 2)")
            except Exception as errore:  # noqa: BLE001 - un preset guasto zittisce il suo evento, non ferma la partita
                self._avvisa(("preset", preset), f"il preset {preset} non si sintetizza, {type(errore).__name__}: {errore}")
                buffer = np.zeros((0, 2), dtype=np.float32)
            # Lo stesso buffer va a tutte le voci e torna da pronto: nessuno deve poterlo cambiare.
            buffer.flags.writeable = False
            self._stereo[preset] = buffer
        return buffer

    def _mono_di(self, preset):
        """Il preset mono, da collocare nello spazio: la somma dei due lati divisa per il doppio del
        guadagno al centro, calcolata una volta sola."""
        mono = self._mono.get(preset)
        if mono is None:
            import numpy as np

            stereo = self._stereo_di(preset)
            mono = (stereo[:, 0] + stereo[:, 1]) / np.float32(2.0 * self._al_centro)
            mono.flags.writeable = False
            self._mono[preset] = mono
        return mono

    def _buffer(self, evento, posizione, guadagno_db):
        """Il buffer dell'evento, con la posizione e il guadagno, oppure None se l'evento non c'è."""
        preset = self._preset(evento)
        if preset is None:
            return None
        import numpy as np

        from sintesi import db_in_lineare

        guadagno = db_in_lineare(guadagno_db)
        if posizione is None:
            stereo = self._stereo_di(preset)
            if guadagno == 1.0 or not len(stereo):
                return stereo
            return stereo * np.float32(guadagno)
        from fisica import suono_di
        from sintesi import panoramica

        punto = np.asarray(posizione, dtype=float).reshape(3)
        # La frequenza dello spazio non serve: un effetto tiene l'altezza del suo preset.
        _frequenza, pan, guadagno_della_distanza = suono_di(punto)
        return panoramica(self._mono_di(preset) * np.float32(guadagno * guadagno_della_distanza), pan)

    def _conta(self, evento):
        conta = self.conta
        if conta is None:
            return
        try:
            conta(evento)
        except Exception as errore:  # noqa: BLE001 - una statistica che non si scrive non deve togliere il suono
            self._avvisa(("conta", evento), f"il conteggio del suono {evento} non è riuscito, {type(errore).__name__}: {errore}")

    def suona(self, evento, posizione=None, guadagno_db=0.0, inizio=None):
        """Suona l'evento e restituisce la voce del motore, oppure None se l'evento non ha suono.
        posizione è un punto (x, y, z) in metri: il suono si fa mono e viene da lì, con la panoramica
        e il guadagno della distanza dello spazio; senza, il preset suona com'è, stereo. guadagno_db
        si aggiunge in decibel. inizio è il campione del flusso da cui parte, None vuol dire
        motore.prossimo_campione()."""
        buffer = self._buffer(evento, posizione, guadagno_db)
        if buffer is None or not len(buffer):
            return None
        voce = self.motore.suona(buffer, inizio=inizio)
        self._conta(evento)
        return voce

    def pronto(self, evento, posizione=None, guadagno_db=0.0, conta_subito=True):
        """Il buffer dell'evento già collocato nello spazio e col suo guadagno, (N, 2) float32, da
        passare più tardi a motore.suona dentro un evento programmato; None se l'evento è
        sconosciuto, un buffer vuoto se il suo preset non ha suono. Senza posizione e a zero decibel
        è il buffer della memoria, in sola lettura.
        L'evento si conta subito; con conta_subito falso no, e chi l'ha preparato lo conta con
        suonato quando il motore l'ha fatto partire davvero: è il caso degli effetti che un
        annullamento può lasciare muti."""
        buffer = self._buffer(evento, posizione, guadagno_db)
        if conta_subito and buffer is not None and len(buffer):
            self._conta(evento)
        return buffer

    def suonato(self, evento):
        """Conta un evento che il motore ha fatto partire più tardi, dentro un evento programmato, da
        un buffer preparato con pronto e conta_subito falso. Un evento sconosciuto, o senza suono, non
        si conta, come in suona e in pronto."""
        preset = self._preset(evento)
        if preset is not None and len(self._stereo_di(preset)):
            self._conta(evento)

    def precarica(self, eventi=None):
        """Sintetizza subito i preset degli eventi, tutti se eventi è None, così il primo sparo della
        partita non aspetta la sintesi di un rumore. Restituisce quanti di quegli eventi hanno un suono."""
        con_suono = 0
        for evento in EVENTI if eventi is None else eventi:
            preset = self._preset(evento)
            if preset is not None and len(self._stereo_di(preset)):
                con_suono += 1
        return con_suono
