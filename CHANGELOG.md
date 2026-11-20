# Changelog - Ciechino

Tutti i cambiamenti e le novità introdotte nelle versioni di Ciechino.

## [0.11.8] - 2026-11-20

La riga del sommario coi numeri non ha più lo spazio dopo i due punti, come L:+2,5 V:+7,7 P:+12,1 I:-23,2 G:91,4%: così i valori e la precisione globale stanno tutti nei primi 40 caratteri, e la legenda comincia dal quarantunesimo. Dal quarto al sesto livello le parole hanno la loro riga, e la precisione globale col punto di vista va sempre sulla riga sotto.

Lo spazio premuto appena finita una scivolata prende il massimo per tre decimi di secondo, non più 50 millesimi; dopo, la barra ricomincia. Corretto anche un difetto nascosto, che con la compensazione del tasto poteva far ricominciare la barra proprio a chi aveva preso il massimo.

Nelle statistiche il proiettile perduto entra nella precisione globale media come un tiro allo 0%, il primato della media migliore in una partita vale soltanto per le partite di almeno dieci tiri, e i massimi presi con lo spazio hanno una voce loro, separata da quella delle versioni prima della 0.10.0, che contava le scivolate arrivate in fondo senza lo spazio.

## [0.11.0] - 2026-11-20

Arriva la precisione globale, chiesta da Gabriele: ogni resoconto, tranne quello del proiettile perduto, che non ha volo, dice quanto il proiettile si è avvicinato al bersaglio considerando l'intero cubo, in percentuale con un decimale. Si misura dal centro del bersaglio, colpito o passato più vicino, sulla diagonale del cubo, 86,6 metri: un centro al primo livello vale sempre più del 94%, e la precisione di prima, quella che raddoppia i punti, resta com'è. Nei primi tre livelli sta nel sommario coi numeri, come G, nei tre seguenti accanto alle parole, e dopo nella riga dell'esito; la voce e il braille la dicono sempre.

Le statistiche contano la precisione globale media dei tiri e quella per partita, le ultime partite dicono la loro media, e due primati nuovi, la precisione globale migliore in un tiro e la media migliore in una partita, si annunciano quando li batti.

## [0.10.0] - 2026-11-20

Le barre della mira aspettano la barra spaziatrice: quella che arriva in fondo senza che tu la fermi non vale più il massimo, ma ricomincia da capo, col suo secondo di silenzio e i suoi bip. Ogni barra suona al più cinque volte; se anche la quinta passa senza spazio, il colpo non parte e il proiettile è perduto, come un tiro a vuoto, con un suono suo, la miccia che si spegne, e il suo resoconto. Lo spazio premuto appena finita la scivolata vale ancora il massimo.

Durante la mira Esc annulla il tiro: il proiettile resta nella riserva, i bersagli tornano a cantare, e c'è un suono anche per questo, il respiro che si lascia andare. La barra di stato lo ricorda, e dal secondo giro di una barra dice quale giro suona; la guida F1 lo dice. I suoni sono adesso 60 in tutto, in 10 gruppi, e le statistiche contano le mire annullate, le barre ripartite e i proiettili perduti.

## [0.9.2] - 2026-11-20

La riga del sommario si legge meglio sul display braille: comincia coi numeri, ciascuno con la sua iniziale, come L: +1,9 V: +15,8 P: +24,7 I: -18,5, tutti nei primi 40 caratteri, e dal quarantunesimo una legenda breve dice che cosa vuol dire ogni lettera. Una misura che non c'è si dice con la sua lettera e la parola manca, e la legenda spiega perché. Nei livelli delle parole le parole vengono prima, e la legenda dopo. La voce e il braille dicono l'esito e i numeri, senza la legenda.

## [0.9.1] - 2026-11-20

Il resoconto del tiro arriva un secondo dopo la fine del suono dell'impatto, come ha chiesto Gabriele: il jingle o il tonfo si sentono per intero, e poi un attimo di silenzio prima della voce.

## [0.9.0] - 2026-11-20

Arrivano gli Ascolti, il tutorial audio chiesto da Gabriele: nel menu Impostazioni e nel menu Aiuto, una finestra racconta un passo alla volta come suona lo spazio di Ciechino, e la barra spaziatrice fa sentire il passo, quante volte vuoi, Invio passa al seguente, Esc chiude. Si comincia col test stereo, che ti dice se cuffie o altoparlanti sono nel verso giusto; poi un bersaglio al centro del cubo va a sinistra e a destra, scende a terra e sale al soffitto, viene vicino al massimo e va lontano al massimo, e salta in tre punti a caso, che dopo l'ascolto il testo ti dice in metri, così controlli l'orecchio.

Seguono un bersaglio che cresce da piccolissimo a gigantesco, le dodici forme dell'oscillazione e i timbri dei bersagli, ognuno col suo nome, le tre scivolate della mira come in partita, il proiettile in volo dallo sparo al tonfo, il jingle del centro e quello del bonus, vicino e lontano, e un bonus che attraversa il cubo. Il passo del vicino e del lontano serve anche a ritarare l'orecchio sulla distanza quando cambi il volume del computer o le cuffie.

Finché non hai giocato la prima partita, Ciechino apre gli Ascolti da solo all'avvio, la sala briefing dei suoni di Gabriele, dopo i suoni dell'apertura e prima della proposta di un aggiornamento. L'apertura degli Ascolti ha il suo suono, e i suoni sono adesso 58 in tutto, in 10 gruppi; le statistiche contano le aperture degli Ascolti, quelle all'avvio, i passi ascoltati e gli Ascolti seguiti fino in fondo.

## [0.8.0] - 2026-11-20

Ciechino non è ancora pubblicato: la prima versione pubblica sarà la 1.0.0, e fino ad allora il numero sale a ogni novità e a ogni correzione. La 0.8.0 raccoglie il gioco nato il 19 novembre 2026 e le sette novità venute dai banchi d'ascolto e dal primo collaudo di Gabriele: i suoni del colpo, la forma del proiettile, i bersagli sempre diversi, la mira ritoccata, la scelta della scheda audio, il sommario del tiro e la palestra. Questo blocco racconta il gioco intero, com'è oggi, per chi lo prova per la prima volta. Il nome è un gioco di parole fra cieco e cecchino, e l'idea è di Gabriele, che voleva un gioco di mira in cui l'orecchio facesse tutto il lavoro degli occhi.

Davanti a te, a 30 metri, comincia un cubo di 50 metri per lato, e là dentro cantano i bersagli: ognuno con una voce che vibra attorno alla sua nota, in una di dodici forme, dal seno all'onda quadra, dal dente di sega al trillo, e con il timbro, i dettagli della forma e l'ampiezza dell'oscillazione presi un poco a caso, così due bersagli non suonano mai proprio uguali. Il lato si sente nella panoramica, l'altezza nella nota, da 220 hertz al suolo con un'ottava ogni 16,7 metri circa, la distanza nel volume. La grandezza di un bersaglio si sente nell'ampiezza della sua oscillazione: più è largo il vibrato, più è grande il bersaglio.

Per sparare si avvia la mira con la barra spaziatrice: il tempo si ferma, i bersagli restano dove sono, più piano, e tre scivolate scelgono la direzione, l'alzo e la potenza. Ognuna comincia dopo un secondo di silenzio con tre bip, poi scivola per tre secondi, con dieci tacche che ne contano i decimi, e la barra spaziatrice la ferma nel punto esatto che senti, senza gradini. Poi parte il colpo, e lo segui con l'orecchio mentre sale, si allontana e ricade, fino all'impatto o fino al tonfo a terra, ciascuno dal punto in cui avviene. Il proiettile canta in seno, oppure nella forma d'onda che scegli con Proiettile, nel menu Impostazioni, fra seno, triangolo e sega morbida, sentendo un esempio di ciascuna; vale dalla partita seguente, e i bersagli non la usano mai, così il colpo non si confonde con loro. Tutte le misure sono vere: metri, gradi e metri al secondo.

I colpi sono una riserva: si parte con 10, un tiro a vuoto ne costa uno, un centro non costa niente e vale fino al doppio se lo prendi nel cuore. Dieci centri chiudono il livello e danno altri 10 colpi. Salendo, i bersagli si rimpiccioliscono fino a mezzo metro, dal quarto livello si muovono e rimbalzano sulle pareti del cubo, dall'ottavo diventano due, poi tre e quattro, e dal terzo può attraversare il cubo un bersaglio bonus, che vale cinque volte un bersaglio normale del suo livello e regala cinque colpi.

Dopo ogni tiro, quando il suono dell'impatto è finito, arriva il resoconto, che comincia con un sommario: dice dal bersaglio, colpito o mancato, di quanto il proiettile è passato a destra o a sinistra del suo centro, e quanto sopra o sotto, quando arriva alla sua distanza, se lì saliva o scendeva, e di quanto è lungo o corto quando, scendendo, ripassa alla sua altezza: le misure di un artigliere, col segno che dice la correzione. Fino al terzo livello il sommario dà i numeri, in metri e in gradi col segno, fino al sesto soltanto le parole, come a destra, sopra, lungo, in discesa, poi tace; la voce e il braille lo leggono subito dopo l'esito. Poi il resoconto dice l'esito, i valori del colpo, quanto è salito, quanto è andato lontano e di quanto si è spostato di lato; dopo un centro, anche dove stava il bersaglio; nei primi tre livelli, dopo un tiro a vuoto, di quanto e dove hai mancato il bersaglio, a destra o a sinistra, sopra o sotto, lungo o corto. Il resoconto arriva nella vista della partita e, se vuoi, da solo alla voce e al braille del tuo screen reader.

I primi tre livelli sono una palestra: dopo ogni tiro la partita si ferma e tace, per leggere il resoconto con calma, e la barra spaziatrice chiama il bersaglio nuovo dopo un centro, o riporta all'ascolto dopo un tiro a vuoto; il tempo delle attese non conta nella durata della partita. Nella palestra Ctrl+P mette in pausa la partita, anche a metà della mira, e la fa riprendere; dal quarto livello si torna subito ad ascoltare, e la pausa con Ctrl+P non c'è più. La partita si ferma da sola, in silenzio e in tutti i livelli, anche quando apri un menu o una finestra di dialogo o passi a un'altra applicazione, e riprende da sola quando torni. Ctrl+N e l'uscita, con una partita in corso, chiedono prima se abbandonarla.

La scheda audio la sceglie Ciechino da solo, quella di Windows nel modo più pronto, oppure tu, con Scheda audio, nel menu Impostazioni, o con Ctrl+Maiusc+X: l'elenco comprende anche le cuffie Bluetooth accese dopo l'avvio, dice la latenza di ogni scheda e avvisa delle interfacce che possono zittire NVDA, con la via per tornare a orecchio alla scelta automatica, e del vivavoce delle cuffie, che suona come un telefono. La scheda scelta vale da subito, anche a partita in corso, e resta per i prossimi avvii; se non c'è, Ciechino suona su quella automatica e lo dice, senza dimenticare la tua scelta.

La classifica tiene i cento punteggi migliori, con il nome, che si chiede soltanto a chi ci entra, e l'età del record in giorni. Le statistiche contano tutto: le partite, i tiri, i valori presi con ogni scivolata, i bersagli forma per forma e timbro per timbro, i bonus, le pause, i suoni, le ultime dieci partite e i primati, dal punteggio più alto alla serie di centri più lunga, che il resoconto annuncia quando li batti.

Ogni azione ha il suo suono, 57 in tutto, in 10 gruppi, e due azioni non ne condividono mai uno: gli effetti del tiro sono di rumore, lunghi e marcati, bianco lo sparo, grave, rosa il centro, azzurro il bonus, marrone il tonfo a terra, e il centro e il bonus finiscono ciascuno col suo jingle, tre note il primo, otto più acute e brillanti il secondo. Nel menu Aiuto ci sono la guida ai comandi, il manuale, le novità, le informazioni, l'offerta di un caffè e le due voci del PUB, il gruppo di supporto. Ciechino si aggiorna da solo, si apre una volta sola, e annota gli errori imprevisti, anche quelli del motore audio, in un registro accanto al programma, da allegare quando chiedi aiuto; se la scheda audio smette di rispondere, lo dice e continua senza suono, finché Ctrl+Maiusc+X non ridà il suono.

## [0.1.0] - 2026-11-19

La nascita di Ciechino, il tiro a segno a orecchio, dal punto 73 di idee.txt di Gabriele: il gioco di base, la finestra, il motore audio, la classifica e le statistiche.
