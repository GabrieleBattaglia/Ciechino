"""
Le costanti di Ciechino: ogni numero della meccanica di gioco, del suono e dei dati sta qui.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0. Gabriele l'ha chiesto nel punto 73 di idee.txt: molte componenti
matematiche della meccanica di gioco si calibreranno con l'esperienza, quindi nessun modulo scrive
un numero di taratura per conto suo, e ogni valore porta accanto il suo significato.
Le misure sono in metri, secondi e gradi (scelta di Gabriele del 19 novembre 2026: lo spazio è
misurabile in metri, qualcosa di realistico, e tutte le indicazioni usano il sistema metrico).
Gli assi: x verso destra, y verso l'alto, z in avanti, con il giocatore nell'origine.
"""

from version import __version__

VERSIONE = __version__
NOME = "Ciechino"

# Il mondo fisico.
GRAVITA = 9.81
# Il tempo di gioco scorre a questa velocità rispetto a quello vero: con 1,5 voli e bersagli
# andrebbero una volta e mezza più svelti, a parità di spazio percorso.
VELOCITA_DEL_TEMPO = 1.0
# Il cannone sta davanti ai piedi del giocatore, un metro sopra il suolo.
ALTEZZA_CANNONE = 1.0
# Lo spazio dei bersagli è un cubo di 50 metri di lato, che comincia 30 metri davanti al
# giocatore: da 25 metri a sinistra a 25 a destra, da terra a 50 metri d'altezza, da 30 a 80
# metri di distanza.
CUBO_LATO = 50.0
CUBO_DISTANZA = 30.0
CUBO_X = (-CUBO_LATO / 2, CUBO_LATO / 2)
CUBO_Y = (0.0, CUBO_LATO)
CUBO_Z = (CUBO_DISTANZA, CUBO_DISTANZA + CUBO_LATO)
# Il centro di un bersaglio sta almeno a questa altezza, e sempre col bersaglio tutto dentro il cubo.
ALTEZZA_MINIMA_BERSAGLIO = 3.0
# Il raggio del proiettile, che conta nel contatto col bersaglio.
RAGGIO_PROIETTILE = 0.1
# Il passo, in secondi di volo, con cui si cerca il contatto fra proiettile e bersagli.
PASSO_DEL_VOLO = 0.001

# Il tiro: i tre valori scelti con le scivolate, da 0 a 1, diventano angoli e velocità.
# La direzione va da 45 gradi a sinistra a 45 a destra: il cubo sta tutto dentro i 40 gradi.
AZIMUT_MASSIMO = 45.0
# L'alzo va da 0 a 85 gradi sull'orizzonte.
ALZO_MINIMO = 0.0
ALZO_MASSIMO = 85.0
# La potenza è la velocità alla bocca del cannone, da 10 a 45 metri al secondo; l'esponente
# piega la scala: 1 la tiene lineare.
VELOCITA_MINIMA = 10.0
VELOCITA_MASSIMA = 45.0
ESPONENTE_POTENZA = 1.0
# Ogni punto del cubo deve essere raggiungibile con i tre valori lontani almeno tanto dai capi
# delle scale: idee.txt vuole che i parametri coprano tutto lo spazio dei bersagli.
MARGINE_DI_COPERTURA = 0.02

# Lo spazio nel suono: idee.txt dà al proiettile e ai bersagli tre parametri sonori, la
# stereofonia per la direzione, l'altezza del suono per l'altezza dal suolo e il volume per la
# distanza dal punto di sparo.
# La panoramica: tutto a sinistra a 45 gradi a sinistra, tutto a destra a 45 a destra, come la
# scivolata della direzione, così chi ferma la barra sulla panoramica del bersaglio mira giusto.
AZIMUT_DEL_PAN = AZIMUT_MASSIMO
# L'altezza: 220 hertz al suolo, un'ottava ogni 50/3 metri, cioè tre ottave sul lato del cubo
# (1760 hertz in cima); sopra il tetto la frequenza non sale più (4 ottave, 3520 hertz).
FREQUENZA_AL_SUOLO = 220.0
METRI_PER_OTTAVA = CUBO_LATO / 3
ALTEZZA_TETTO_DEL_SUONO = 4 * METRI_PER_OTTAVA
# La distanza: il volume scende in decibel, linearmente con i metri dal cannone, fino a un pavimento.
GUADAGNO_DB_VICINO = -6.0
DB_PER_METRO = 0.25
GUADAGNO_DB_MINIMO = -36.0

# La mira: tre scivolate, direzione, alzo e potenza, nell'ordine.
PARAMETRI = ("direzione", "alzo", "potenza")
# Prima di ogni scivolata una pausa muta, che compensa il tempo di reazione: senza, dopo aver
# fermato una barra non si farebbe in tempo a scegliere il minimo della seguente. Lo spazio
# premuto nella pausa non conta.
PAUSA_FRA_BARRE = 1.0
# Poi tre brevi bip a frequenza crescente, distanti 33 millesimi l'uno dall'altro, e 33 millesimi di
# silenzio: lo spazio premuto qui sceglie il minimo.
BIP_DURATA = 0.012
BIP_DISTACCO = 0.033
BIP_SILENZIO_FINALE = 0.033
BIP_DB = -12.0
BIP_FREQUENZE = {
    "direzione": (660.0, 880.0, 1100.0),
    "alzo": (880.0, 1175.0, 1568.0),
    "potenza": (523.0, 698.0, 932.0),
}
# Poi la scivolata, che dura 3 secondi e arriva al massimo se nessuno la ferma.
SCIVOLATA_DURATA = 3.0
SCIVOLATA_DB = -9.0
# Le forme d'onda delle tre scivolate, fra quelle di sintesi.FORME_PORTANTI.
SCIVOLATA_FORMA = {"direzione": "triangolo", "alzo": "seno", "potenza": "triangolo"}
# La direzione: un suono fisso che va da tutto a sinistra a tutto a destra.
DIREZIONE_FREQUENZA = 440.0
# L'alzo: la frequenza sale dal grave all'acuto, su una scala esponenziale.
ALZO_FREQUENZE = (220.0, 1760.0)
# La potenza: un suono fisso che cresce dal quasi silenzio al massimo, linearmente in decibel;
# il minimo è in decibel sotto SCIVOLATA_DB.
POTENZA_FREQUENZA = 330.0
POTENZA_DB_MINIMO = -40.0
# Insieme a ogni scivolata dieci tacche, piccolissimi bip che salgono di frequenza, una ogni decimo
# della scivolata: la decima chiude la scala.
TACCHE = 10
TACCA_DURATA = 0.006
TACCA_FREQUENZE = (1200.0, 2400.0)
TACCA_DB = -18.0
# Secondi da togliere all'istante in cui il programma vede lo spazio, per il ritardo fra il dito e
# il programma: si tara al banco della mira.
COMPENSAZIONE_DEL_TASTO = 0.0
# Durante la mira i bersagli restano fermi e si sentono più piano (scelta di Gabriele del 19
# novembre 2026).
ATTENUAZIONE_IN_MIRA_DB = -9.0
# Ogni cambio di guadagno di una voce si fa con una rampa di tanti secondi, contro i click.
RAMPA_DEL_GUADAGNO = 0.05
# Dopo la fine udibile di una scivolata non fermata, tanti secondi prima di prendere il massimo.
MARGINE_DOPO_LA_SCIVOLATA = 0.05

# I bersagli: il suono è una portante la cui frequenza oscilla con un LFO di pochi hertz attorno
# all'altezza del centro; l'escursione copre le altezze da y meno il raggio a y più il raggio,
# moltiplicata per FATTORE_ESCURSIONE: più grande il bersaglio, più ampia l'oscillazione.
FORME_LFO = ("seno", "sega", "sega_inversa", "triangolo", "quadra", "casuale", "deriva")
FREQUENZA_LFO = (2.0, 6.0)
FATTORE_ESCURSIONE = 1.0
BERSAGLIO_FORMA = "organo"
BERSAGLIO_DB = -3.0
# Il proiettile in volo.
PROIETTILE_FORMA = "sega_morbida"
PROIETTILE_DB = -3.0
# Il silenzio fra due avvisi in fila, come l'impatto, il livello nuovo e il bonus, in secondi:
# quanto basta perché l'orecchio li senta due. Da tarare ascoltando, come i suoni stessi.
STACCO_FRA_GLI_AVVISI = 0.15

# I livelli e i colpi. La riserva parte da 10 colpi; ogni tiro a vuoto ne toglie uno, i centri non
# costano nulla; 10 centri chiudono il livello e danno altri 10 colpi, il bersaglio bonus ne dà 5;
# a riserva vuota la partita finisce (scelta di Gabriele del 19 novembre 2026).
COLPI_INIZIALI = 10
COLPI_PER_LIVELLO = 10
COLPI_BONUS = 5
CENTRI_PER_LIVELLO = 10
# Il raggio dei bersagli: 5 metri al primo livello, poi ogni livello lo moltiplica per 0,85, fino
# al mezzo metro.
RAGGIO_INIZIALE = 5.0
RIDUZIONE_DEL_RAGGIO = 0.85
RAGGIO_MINIMO = 0.5
# Dal quarto livello i bersagli si muovono, prima piano, poi sempre più svelti, rimbalzando
# sulle pareti del cubo.
LIVELLO_DEI_MOBILI = 4
VELOCITA_INIZIALE_BERSAGLI = 1.5
AUMENTO_DI_VELOCITA = 0.75
VELOCITA_MASSIMA_BERSAGLI = 12.0
# A questi livelli i bersagli in volo diventano uno di più: due dall'ottavo, tre dal dodicesimo,
# quattro dal sedicesimo.
LIVELLI_CON_UN_BERSAGLIO_IN_PIU = (8, 12, 16)
# I bersagli bonus, dal terzo livello: attraversano il cubo da un lato all'altro e poi escono.
# A ogni nuova fase d'ascolto, senza un bonus in volo, ne compare uno con questa probabilità.
LIVELLO_DEI_BONUS = 3
PROBABILITA_BONUS = 0.12
# Il tempo della traversata: 14 secondi al livello dei bonus, mezzo secondo meno a ogni livello,
# fino a 6.
DURATA_BONUS_INIZIALE = 14.0
RIDUZIONE_DURATA_BONUS = 0.5
DURATA_BONUS_MINIMA = 6.0
RAGGIO_BONUS_FATTORE = 0.8
# Fino a questo livello il resoconto di un tiro a vuoto dice di quanto e dove si è mancato il
# bersaglio più vicino (scelta di Gabriele: solo nei livelli bassi, poi si deciderà).
AIUTO_MANCATO_FINO_AL_LIVELLO = 3

# I punti di un centro: PUNTI_BASE, per il rapporto fra RAGGIO_DI_RIFERIMENTO e il raggio alla
# ESPONENTE_DEL_RAGGIO, per 1 più la velocità del bersaglio divisa per VELOCITA_DI_RIFERIMENTO, per 1
# più PREMIO_PRECISIONE per la precisione (da 0 di striscio a 1 nel centro pieno: fino al doppio,
# scelta di Gabriele), per MOLTIPLICATORE_BONUS se è un bonus.
PUNTI_BASE = 100
RAGGIO_DI_RIFERIMENTO = RAGGIO_INIZIALE
ESPONENTE_DEL_RAGGIO = 1.0
VELOCITA_DI_RIFERIMENTO = 5.0
PREMIO_PRECISIONE = 1.0
MOLTIPLICATORE_BONUS = 5

# L'audio.
FS = 44100
AUDIO_BLOCCO = 512
AUDIO_LATENZA = "low"
SFUMATURA_STOP = 0.005
# Oltre questo livello la somma delle voci passa da un tetto morbido.
SOGLIA_DEL_TETTO = 0.9
VOLUME_PREDEFINITO = 80
# Il battito della finestra, che fa avanzare la partita, in millesimi.
BATTITO_MS = 10

# La classifica e i dati, accanto al programma.
POSTI_IN_CLASSIFICA = 100
NOME_MINIMO = 1
NOME_MASSIMO = 20
FORMATO_CLASSIFICA = 1
FORMATO_STATISTICHE = 1
FILE_CLASSIFICA = "Ciechino - Classifica.json"
FILE_STATISTICHE = "Ciechino - Statistiche.json"
FILE_IMPOSTAZIONI = "Ciechino - Impostazioni.json"
ULTIME_PARTITE = 10
