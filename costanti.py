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
# Le forme d'onda delle tre scivolate, fra quelle di sintesi.FORME_PORTANTI. La direzione è in seno
# dal banco della mira del 19 novembre 2026 (D19); prima era in triangolo.
SCIVOLATA_FORMA = {"direzione": "seno", "alzo": "seno", "potenza": "triangolo"}
# La direzione: un suono fisso che va da tutto a sinistra a tutto a destra.
DIREZIONE_FREQUENZA = 440.0
# L'alzo: la frequenza sale dal grave all'acuto, su una scala esponenziale, da 150 a 2400 hertz,
# quattro ottave, dal banco della mira del 19 novembre 2026 (D19); prima era da 220 a 1760.
ALZO_FREQUENZE = (150.0, 2400.0)
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
# Le forme dell'LFO: le sette della 0.1.0, poi cinque nuove dal banco dei bersagli del 19 novembre
# 2026, quando Gabriele le ha volute tutte e anche di più (D18): molla, un guizzo che vibra e si
# smorza a ogni ciclo, una volta in su e una in giù; gradini, una scala che sale o scende; impulsi,
# due colpetti a ogni ciclo, uno in su e uno in giù, col centro fermo in mezzo; doppia, due
# oscillazioni di frequenza diversa insieme; trillo, due note vicine che si alternano in alto e poi
# in basso.
FORME_LFO = ("seno", "sega", "sega_inversa", "triangolo", "quadra", "casuale", "deriva", "molla", "gradini", "impulsi", "doppia", "trillo")
FREQUENZA_LFO = (2.0, 6.0)
# I parametri a caso di ogni forma (D18): ogni bersaglio li estrae dentro questi intervalli, i capi
# compresi, così due bersagli della stessa forma suonano sempre un po' diversi. Con qualunque valore
# ogni forma resta fra meno 1 e 1, arriva agli estremi in uno o pochi cicli e ha la media vicina a
# zero: l'escursione dice la grandezza e il centro dice l'altezza.
# Il seno con una seconda armonica, in ampiezza rispetto alla fondamentale: col più le creste
# arrivano prima e scendono piano, col meno il contrario.
ARMONICA_DEL_SENO = (-0.3, 0.3)
# La curvatura di sega e sega inversa, un esponente: sotto 1 la rampa passa svelta dal centro e
# indugia vicino agli estremi, sopra 1 il contrario; 1 è la rampa dritta.
CURVATURA_DELLA_SEGA = (0.6, 1.6)
# La pendenza del triangolo: la parte del ciclo in cui sale; 0,5 è il triangolo simmetrico.
PENDENZA_DEL_TRIANGOLO = (0.3, 0.7)
# La quadra: la larghezza è la parte del ciclo passata in alto, che si scosta poco da metà perché
# la media resti vicina a zero; il fronte è la durata di ogni salto, in parte del ciclo, da quasi
# netto, coi soli millesimi del raccordo contro i click, a un trapezio.
LARGHEZZA_DELLA_QUADRA = (0.46, 0.54)
FRONTE_DELLA_QUADRA = (0.0, 0.2)
# Il casuale: quanti valori nuovi in un ciclo, interi.
VALORI_DEL_CASUALE = (1, 3)
# La deriva: la morbidezza è la parte del ciclo in cui scivola verso il valore seguente; nel resto
# lo tiene fermo. 1 vuol dire che scivola sempre.
MORBIDEZZA_DELLA_DERIVA = (0.4, 1.0)
# Casuale e deriva: l'esponente che spinge i valori a caso verso gli estremi, perché anche questi
# bersagli facciano sentire presto tutta la loro grandezza; 1 li lascerebbe uniformi.
SPINTA_DEL_CASO = (0.4, 0.9)
# La molla: quante oscillazioni in un ciclo, intere, così il ciclo finisce dove il seguente comincia,
# e lo smorzamento, cioè di quanto cala l'ampiezza in un ciclo: alla fine è e alla meno questo.
OSCILLAZIONI_DELLA_MOLLA = (2, 3)
SMORZAMENTO_DELLA_MOLLA = (1.5, 4.0)
# I gradini: quanti, interi, dal più basso al più alto, e il verso della scala, 1 sale e meno 1 scende.
GRADINI = (3, 5)
VERSI_DEI_GRADINI = (1, -1)
# Gli impulsi: quanto dura ciascuno, in parte del ciclo.
LARGHEZZA_DEGLI_IMPULSI = (0.1, 0.3)
# La doppia: il rapporto fra la seconda frequenza e la prima, uno di questi, e il peso della
# seconda oscillazione. Coi rapporti a metà fra due interi la forma si ripete ogni due cicli.
RAPPORTI_DELLA_DOPPIA = (1.5, 2.5, 3.5)
PESO_DELLA_DOPPIA = (0.4, 0.8)
# Il trillo: quante note in ogni mezzo ciclo, intere, e la profondità, cioè di quanto la nota vicina
# si stacca da quella dell'estremo, nella scala dell'LFO che va da meno 1 a 1.
NOTE_DEL_TRILLO = (2, 3)
PROFONDITA_DEL_TRILLO = (0.25, 0.5)
FATTORE_ESCURSIONE = 1.0
# Ogni bersaglio moltiplica la sua escursione per un numero a caso fra questi due: un po' di varietà,
# al prezzo, accettato da Gabriele, di una grandezza un po' meno netta: un bonus, più piccolo di un quinto,
# può suonare largo quanto un bersaglio normale estratto stretto (D18, 19 novembre 2026).
ESCURSIONE_A_CASO = (0.85, 1.15)
# La portante di ogni bersaglio si estrae a caso fra queste (D18), tolta quella che chi gioca ha
# scelto per il proiettile, perché proiettile e bersagli non si confondano. BERSAGLIO_FORMA resta
# la portante di serie, dove una sola serve.
BERSAGLIO_PORTANTI = ("organo", "seno", "triangolo", "sega_morbida")
BERSAGLIO_FORMA = "organo"
BERSAGLIO_DB = -3.0
# Il proiettile in volo: la sua forma d'onda si sceglie nelle Impostazioni fra queste tre (D17,
# 19 novembre 2026); di serie il seno, che ha vinto nel banco del volo.
PROIETTILE_FORME = ("seno", "triangolo", "sega_morbida")
PROIETTILE_FORMA = "seno"
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
# Il sommario in testa al resoconto di ogni tiro, centri compresi (D21, chiesto da Gabriele il 19
# novembre 2026): dove il proiettile è passato rispetto al centro del bersaglio, nel suo passaggio
# più vicino, di lato, in altezza e in profondità, e con quale inclinazione. I primi livelli sono una
# palestra: fino a SOMMARIO_NUMERI_FINO_AL_LIVELLO il sommario dà i numeri, in metri e in gradi con
# un decimale e il segno; fino a SOMMARIO_PAROLE_FINO_AL_LIVELLO soltanto le parole, come «a destra,
# sotto, lungo, in discesa»; dopo, niente sommario. Da ritoccare giocando.
SOMMARIO_NUMERI_FINO_AL_LIVELLO = 3
SOMMARIO_PAROLE_FINO_AL_LIVELLO = 6
# Le soglie delle parole del sommario (D21): uno scarto, in metri, più piccolo della sua soglia si
# dice giusto, «in linea», «all'altezza giusta», «alla distanza giusta»; un'inclinazione, in gradi,
# più piccola della sua soglia, in su o in giù, è il tratto quasi piano vicino alla cima, «in cima».
SOMMARIO_SOGLIA_LATERALE = 0.5
SOMMARIO_SOGLIA_VERTICALE = 0.5
SOMMARIO_SOGLIA_PROFONDITA = 0.5
SOMMARIO_SOGLIA_INCLINAZIONE = 5.0

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
# La scheda audio scelta da chi gioca (D20, 19 novembre 2026): nelle impostazioni si salvano queste due chiavi, il nome
# del dispositivo e il nome intero della sua interfaccia, come «Windows WASAPI», mai l'indice, che cambia da un avvio
# all'altro; il dizionario vuoto è la scelta automatica.
CHIAVI_DELLA_SCHEDA = ("dispositivo", "interfaccia")
# Le cuffie Bluetooth col microfono compaiono due volte, come cuffie e come vivavoce, che Windows apre per le telefonate
# e che suona come un telefono: il vivavoce si riconosce da una di queste parole nel nome, in minuscolo.
NOMI_DEL_VIVAVOCE = ("hands-free", "handsfree", "vivavoce")
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
