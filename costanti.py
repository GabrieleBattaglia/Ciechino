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

import math

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
# Lo spazio dei bersagli è una cupola (D28, dal 22 novembre 2026; fino alla 1.0.0 era un cubo di 50
# metri di lato, da 30 a 80 metri di distanza): un guscio di sfera centrato a terra sotto il cannone,
# da CUPOLA_RAGGIO_INTERNO a CUPOLA_RAGGIO metri, nel settore di ±SETTORE_AZIMUT gradi davanti al
# giocatore. Ne resta fuori il cono stretto attorno alla verticale, sopra ELEVAZIONE_MASSIMA gradi
# d'elevazione vista dal centro: lassù arriverebbe soltanto un alzo vicino ai 90 gradi, dove la
# potenza utile si stringe a niente.
CUPOLA_RAGGIO = 80.0
CUPOLA_RAGGIO_INTERNO = 30.0
SETTORE_AZIMUT = 40.0
ELEVAZIONE_MASSIMA = 80.0
# Il centro di un bersaglio sta almeno a questa altezza, e sempre col bersaglio tutto dentro la cupola.
ALTEZZA_MINIMA_BERSAGLIO = 3.0
# Il raggio del proiettile, che conta nel contatto col bersaglio.
RAGGIO_PROIETTILE = 0.1
# Il passo, in secondi di volo, con cui si cerca il contatto fra proiettile e bersagli.
PASSO_DEL_VOLO = 0.001

# Il tiro: i tre valori della mira, da 0 a 1, diventano angoli e velocità, e da D28 ogni loro
# combinazione attraversa la cupola.
# La direzione copre esattamente il settore della cupola.
AZIMUT_MASSIMO = SETTORE_AZIMUT
# L'alzo va da 15 a 85 gradi sull'orizzonte: sotto i 15 la potenza utile si stringe troppo, a 10
# gradi a 2,4 m/s.
ALZO_MINIMO = 15.0
ALZO_MASSIMO = 85.0
# La potenza è la velocità alla bocca del cannone. La massima è la radice di 2 g R, circa 39,6 m/s:
# l'apice del tiro più alto arriva in cima alla cupola, e nessun tiro ne esce senza attraversarla. La
# minima dipende dall'alzo: è quella che a quell'alzo porta il proiettile nella cupola, e la calcola
# fisica.velocita_minima. L'esponente piega la scala: 1 la tiene lineare.
VELOCITA_MASSIMA = math.sqrt(2.0 * GRAVITA * CUPOLA_RAGGIO)
ESPONENTE_POTENZA = 1.0
# Ogni punto della cupola dove può stare un bersaglio deve essere raggiungibile con i tre valori
# lontani almeno tanto dai capi delle scale: idee.txt vuole che i parametri coprano tutto lo spazio
# dei bersagli. Con le scivolate il margine era del 2%, perché fermarne una proprio in fondo era
# difficile; da D29 ogni valore arriva al suo capo tenendo giù il tasto, e lì si ferma da solo, quindi
# i capi si scelgono come ogni altro valore, e il margine è zero. Con il 2% i bersagli piccoli vicini ai
# lati del settore e quelli più in alto sarebbero rimasti fuori.
MARGINE_DI_COPERTURA = 0.0

# Lo spazio nel suono: idee.txt dà al proiettile e ai bersagli tre parametri sonori, la
# stereofonia per la direzione, l'altezza del suono per l'altezza dal suolo e il volume per la
# distanza dal punto di sparo.
# La panoramica: tutto a sinistra ai capi sinistri del settore, tutto a destra ai capi destri, come
# la direzione della mira, così chi porta la direzione sulla panoramica del bersaglio mira giusto.
AZIMUT_DEL_PAN = AZIMUT_MASSIMO
# L'altezza: 220 hertz al suolo e OTTAVE_DEL_SUONO ottave fino all'apice del tiro più alto, poco più
# di 81 metri, circa 3500 hertz: da D28 nessun tiro sale oltre il tetto, e il suono non è mai piatto.
# Fino alla 1.0.0 le ottave erano 3 sui 50 metri del cubo, col tetto a 66,7, sotto i 103 del tiro più alto.
FREQUENZA_AL_SUOLO = 220.0
OTTAVE_DEL_SUONO = 4
ALTEZZA_TETTO_DEL_SUONO = ALTEZZA_CANNONE + VELOCITA_MASSIMA ** 2 / (2.0 * GRAVITA)
METRI_PER_OTTAVA = ALTEZZA_TETTO_DEL_SUONO / OTTAVE_DEL_SUONO
# La distanza: il volume scende in decibel, linearmente con i metri dal cannone, fino a un pavimento.
GUADAGNO_DB_VICINO = -6.0
DB_PER_METRO = 0.25
GUADAGNO_DB_MINIMO = -36.0

# La mira manuale (D29, dal 22 novembre 2026; fino alla 1.0.0 erano tre scivolate da fermare con lo
# spazio): tre valori, direzione, alzo e potenza, nell'ordine, che si muovono insieme finché i loro
# tasti sono giù, le frecce orizzontali la direzione, quelle verticali l'alzo, Home e Fine la potenza.
PARAMETRI = ("direzione", "alzo", "potenza")
# Ogni valore parte dal centro della sua scala, e la percorre da un capo all'altro in tanti secondi,
# quanto durava la scivolata.
PARTENZA_DELLA_MIRA = 0.5
DURATA_DELLA_SCALA = 3.0
VELOCITA_DELLA_MIRA = 1.0 / DURATA_DELLA_SCALA
# Il puntatore (D31, dalla 2.0.1; Gabriele il 22 novembre 2026: «Quando muovo il mirino, gli altri
# parametri devono rimanere nel suono di puntamento: volume e altezza se muovo il pan eccetera. In pratica
# quel suono è il mio puntatore.»): una voce sola, in PUNTATORE_FORMA, che suona mentre almeno un valore
# si muove e porta sempre tutti e tre: la direzione nella panoramica, da tutto a sinistra a tutto a
# destra; l'alzo nell'altezza, da ALZO_FREQUENZE[0] a ALZO_FREQUENZE[1] hertz, quattro ottave, su scala
# esponenziale; la potenza nel volume, linearmente in decibel, da POTENZA_DB_MINIMO sotto PUNTATORE_DB
# fino a PUNTATORE_DB, così anche al minimo il puntatore si sente. Fino alla 2.0.0 ogni valore aveva la
# voce della sua scivolata di D19, e gli altri due tacevano.
PUNTATORE_DB = -9.0
PUNTATORE_FORMA = "triangolo"
ALZO_FREQUENZE = (150.0, 2400.0)
POTENZA_DB_MINIMO = -20.0
# Le tacche: ogni valore ne suona una a ogni decimo della scala che attraversa, in tutte e due le
# direzioni, piccolissimi bip che salgono di frequenza dal primo decimo al nono, nella forma TACCA_FORMA
# del suo valore e dal punto del puntatore, con la panoramica della direzione di quell'istante. Ai capi,
# 0 e 100%, il valore si ferma col suono fine_scala, anche lui dal punto del puntatore.
TACCA_FORMA = {"direzione": "seno", "alzo": "seno", "potenza": "triangolo"}
TACCHE = 10
TACCA_DURATA = 0.006
TACCA_FREQUENZE = (1200.0, 2400.0)
TACCA_DB = -18.0
# La mira a tempo (D29): dal livello dopo la palestra, 30 secondi al primo, uno in meno a ogni livello,
# mai sotto 10. Nella palestra non c'è tempo. Un tick al secondo scandisce il tempo che scorre,
# sempre più acuto, da TICK_FREQUENZE[0] al primo secondo a TICK_FREQUENZE[1] all'ultimo; allo
# scadere il colpo parte da solo coi valori del momento.
TEMPO_DELLA_MIRA_INIZIALE = 30.0
RIDUZIONE_DEL_TEMPO_DELLA_MIRA = 1.0
TEMPO_DELLA_MIRA_MINIMO = 10.0
TICK_FREQUENZE = (300.0, 1200.0)
TICK_DURATA = 0.02
TICK_DB = -20.0
TICK_FORMA = "triangolo"
# Il colpo parte tanti secondi dopo il prossimo campione sicuro, perché il conto del volo, che parte
# col tempo di gioco che scorre, abbia il tempo di finire prima che il colpo si senta.
ANTICIPO_DELLO_SPARO = 0.05
# Ogni cambio di guadagno di una voce si fa con una rampa di tanti secondi, contro i click.
RAMPA_DEL_GUADAGNO = 0.05

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

# I livelli e i colpi. La riserva parte da 20 colpi; ogni tiro a vuoto ne toglie uno, i centri non
# costano nulla; 10 centri chiudono il livello e danno altri 15 colpi, il bersaglio bonus ne dà 8;
# a riserva vuota la partita finisce (scelte di Gabriele del 19 novembre 2026, D2, e del 22 novembre,
# D30, che ha portato i colpi da 10, 10 e 5 a 20, 15 e 8).
COLPI_INIZIALI = 20
COLPI_PER_LIVELLO = 15
COLPI_BONUS = 8
CENTRI_PER_LIVELLO = 10
# Il primato della precisione globale media vale soltanto per le partite di almeno tanti tiri (D26):
# erano i colpi iniziali di prima, e con D30 restano 10.
TIRI_PER_IL_PRIMATO_DELLA_MEDIA = 10
# Il raggio dei bersagli: 5 metri al primo livello, poi ogni livello lo moltiplica per 0,85, fino
# al mezzo metro.
RAGGIO_INIZIALE = 5.0
RIDUZIONE_DEL_RAGGIO = 0.85
RAGGIO_MINIMO = 0.5
# Dal quarto livello i bersagli si muovono, prima piano, poi sempre più svelti, rimbalzando
# sulle pareti della cupola.
LIVELLO_DEI_MOBILI = 4
VELOCITA_INIZIALE_BERSAGLI = 1.5
AUMENTO_DI_VELOCITA = 0.75
VELOCITA_MASSIMA_BERSAGLI = 12.0
# A questi livelli i bersagli in volo diventano uno di più: due dall'ottavo, tre dal dodicesimo,
# quattro dal sedicesimo.
LIVELLI_CON_UN_BERSAGLIO_IN_PIU = (8, 12, 16)
# I bersagli bonus, dal terzo livello: attraversano la cupola da un lato all'altro e poi escono.
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
# novembre 2026): dove il proiettile è passato rispetto al centro del bersaglio, con le misure
# dell'artigliere scelte da Gabriele, di lato, in altezza e con quale inclinazione quando arriva
# alla distanza del bersaglio, in profondità quando ripassa scendendo alla sua altezza. I primi
# livelli sono una palestra: fino a SOMMARIO_NUMERI_FINO_AL_LIVELLO il sommario dà i numeri, in
# metri e in gradi con un decimale e il segno; fino a SOMMARIO_PAROLE_FINO_AL_LIVELLO soltanto le
# parole, come «a destra, sopra, lungo, in discesa»; dopo, niente sommario. Da ritoccare giocando.
SOMMARIO_NUMERI_FINO_AL_LIVELLO = 3
SOMMARIO_PAROLE_FINO_AL_LIVELLO = 6
# Le soglie delle parole del sommario (D21): uno scarto, in metri, più piccolo della sua soglia si
# dice giusto, «in linea», «all'altezza giusta», «alla distanza giusta»; un'inclinazione, in gradi,
# più piccola della sua soglia, in su o in giù, è il tratto quasi piano vicino alla cima, «in cima».
SOMMARIO_SOGLIA_LATERALE = 0.5
SOMMARIO_SOGLIA_VERTICALE = 0.5
SOMMARIO_SOGLIA_PROFONDITA = 0.5
SOMMARIO_SOGLIA_INCLINAZIONE = 5.0
# La palestra delle attese (D22, dal collaudo di Gabriele del 19 novembre 2026): nei livelli da 1 a
# PALESTRA_FINO_AL_LIVELLO, contando il livello in cui si spara, dopo ogni tiro, centro o a vuoto, la
# partita resta ferma e muta come in pausa, finita l'eco dell'impatto e degli avvisi, finché lo spazio
# non la fa ripartire: dopo un centro lo spazio chiama il bersaglio nuovo, dopo un tiro a vuoto
# riporta all'ascolto. Così si legge il resoconto con calma, e il tempo dell'attesa non conta nella
# durata della partita. Soltanto in questi livelli c'è anche la pausa di Ctrl+P; quella automatica,
# dei dialoghi, dei menu e del cambio di finestra, resta in tutti.
PALESTRA_FINO_AL_LIVELLO = 3
# Lo spazio scioglie l'attesa della palestra soltanto passati tanti secondi da quando la partita si è
# fermata: quello premuto mentre l'effetto o gli avvisi suonano ancora, o appena dopo, non la salta
# per sbaglio. Da tarare giocando, come la grazia dei dialoghi.
GRAZIA_DELL_ATTESA = 0.5
# Il resoconto di ogni tiro compare nella vista, e va alla sintesi, quando l'effetto dell'impatto, il
# jingle, l'impatto o il tonfo, ha finito di suonare (D22), in tutti i livelli; e dopo altri tanti
# secondi di silenzio: un secondo dal collaudo di Gabriele del 20 novembre 2026, «metterei un secondo
# di silenzio» (D24).
RITARDO_DEL_RESOCONTO = 1.0
# Il display braille si legge a blocchi di tanti caratteri (D24): la riga del sommario mette i numeri
# nel primo blocco e la legenda dal carattere seguente.
BLOCCO_DEL_BRAILLE = 40

# Gli Ascolti, il tutorial audio chiesto da Gabriele il 20 novembre 2026 (D23): nel menu Aiuto, dalla 0.11.10
# soltanto lì, un passo alla volta, il test stereo, il bersaglio che si muove nella cupola, i punti a caso, la grandezza,
# le forme, i timbri, la mira manuale e il suo tempo, il volo e gli effetti, con le regole dello spazio sonoro del gioco. Tutti da tarare
# ascoltando. Ogni suono degli Ascolti entra ed esce con una rampa di tanti secondi, contro i click.
ASCOLTI_RAMPA = 0.05
# Il silenzio fra i suoni di uno stesso passo, fra due forme, due timbri, due punti o due effetti, e quanto suona
# ognuno di quei suoni, una forma, un timbro o un punto a caso (D23).
ASCOLTI_PAUSA = 0.6
ASCOLTI_SUONO = 2.5
# Il test stereo (D23): un suono fermo tutto a sinistra per ASCOLTI_STEREO_SOSTA secondi, poi la scivolata fino a
# tutto a destra in ASCOLTI_STEREO_VIAGGIO, poi fermo lì; la forma d'onda, fra quelle di sintesi.FORME_PORTANTI,
# la nota in hertz e il livello in decibel, quello del puntatore della mira con la potenza al massimo.
ASCOLTI_STEREO_SOSTA = 1.0
ASCOLTI_STEREO_VIAGGIO = 4.0
ASCOLTI_STEREO_FORMA = "triangolo"
ASCOLTI_STEREO_FREQUENZA = 440.0
ASCOLTI_STEREO_DB = PUNTATORE_DB
# Il bersaglio degli Ascolti (D23): grande come quelli del primo livello, con la forma dell'LFO più regolare, che
# oscilla a metà dell'intervallo del gioco.
ASCOLTI_RAGGIO = RAGGIO_INIZIALE
ASCOLTI_FORMA = "seno"
ASCOLTI_FREQUENZA_LFO = (FREQUENZA_LFO[0] + FREQUENZA_LFO[1]) / 2
# Le tratte degli Ascolti (D23), da un lato all'altro, in altezza e in profondità: il bersaglio va a tanti metri al
# secondo, partendo e arrivando dolce; si ferma nel mezzo della cupola per ASCOLTI_SOSTA secondi, all'inizio e alla
# fine, e a ogni capo della tratta per ASCOLTI_SOSTA_AI_CAPI. Erano 8 m/s nel cubo; da D28 sono 10, perché gli archi
# della cupola sono più lunghi e un passo resta sotto ASCOLTI_DURATA_MASSIMA.
ASCOLTI_VELOCITA = 10.0
ASCOLTI_SOSTA = 1.5
ASCOLTI_SOSTA_AI_CAPI = 1.0
# Quanti punti a caso nella cupola (D23), dopo i quali il testo dice dove erano.
ASCOLTI_PUNTI = 3
# La grandezza (D23): le tappe del raggio, in metri, dal più piccolo del gioco al primo livello, poi un gigante grande
# il doppio di ogni bersaglio vero, e la velocità con cui il raggio cresce, in metri al secondo. A ogni tappa il
# bersaglio si ferma ASCOLTI_SOSTA secondi.
ASCOLTI_GRANDEZZE = (RAGGIO_MINIMO, RAGGIO_INIZIALE, 2 * RAGGIO_INIZIALE)
ASCOLTI_CRESCITA = 1.5
# Le forme dell'oscillazione, tante per passo, perché un passo non diventi troppo lungo (D23).
ASCOLTI_FORME_PER_PASSO = 6
# Il bonus che attraversa la cupola negli Ascolti (D23) va alla velocità di questo livello, il primo dei bonus.
ASCOLTI_LIVELLO_DEL_BONUS = LIVELLO_DEI_BONUS
# Un passo degli Ascolti dura al massimo tanti secondi: se ne serve di più, si divide in due (D23).
ASCOLTI_DURATA_MASSIMA = 20.0

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
# Il formato 2, dalla 0.12.0 (D27), aggiunge a ogni voce la precisione globale media della partita;
# le voci del formato 1 si leggono ancora, senza precisione.
FORMATO_CLASSIFICA = 2
FORMATO_STATISTICHE = 1
FILE_CLASSIFICA = "Ciechino - Classifica.json"
FILE_STATISTICHE = "Ciechino - Statistiche.json"
FILE_IMPOSTAZIONI = "Ciechino - Impostazioni.json"
ULTIME_PARTITE = 10
