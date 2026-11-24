# Ciechino, il tiro a segno a orecchio

Un gioco di tiro accessibile, pensato da un giocatore cieco per chi non vede, e per chi vuole provare a mirare con le orecchie. Il nome è un gioco di parole fra cieco e cecchino.

Davanti a te si apre una cupola, un guscio fra due sfere di 30 e 80 metri di raggio, centrate sotto i tuoi piedi, da 40 gradi a sinistra a 40 gradi a destra, e là dentro cantano i bersagli: la panoramica dice se stanno a destra o a sinistra, la nota quanto sono in alto, il volume quanto sono lontani, e ognuno ha una voce un po' diversa dagli altri. Per sparare punti tre valori, la direzione, l'alzo e la potenza, tenendo giù le frecce e i tasti Home e Fine, mentre i bersagli continuano a cantare e a muoversi; Invio spara, e segui il colpo con l'orecchio, fino all'impatto o fino a terra. Ogni combinazione dei tre valori attraversa la cupola, quindi nessun pezzo delle scale è sprecato, e ogni punto della cupola si raggiunge coi valori dentro le loro scale, capi compresi. I colpi sono una riserva, dieci centri chiudono il livello, i bersagli si rimpiccioliscono, si muovono e si moltiplicano, e ogni tanto passa un bonus che vale cinque volte un bersaglio normale del suo livello. I cento punteggi migliori restano in classifica, una griglia allineata da leggere in braille, che per ogni partita dice anche il livello, i tiri, i centri, la precisione globale media e il tempo di gioco.

## Scaricare e installare

Ciechino gira su Windows 10 e 11 a 64 bit, e si gioca meglio in cuffia. L'ultima versione sta nella pagina delle release su GitHub: https://github.com/GabrieleBattaglia/Ciechino/releases

- **Con l'installatore, Ciechino-Setup:** chiede di accettare la licenza GPL 3 e installa Ciechino nella tua cartella dei programmi, senza permessi di amministratore. Alla disinstallazione chiede se cancellare anche la classifica, le statistiche e le impostazioni.
- **Con lo zip, Ciechino.zip:** si scompatta in una cartella dove puoi scrivere, mai in Programmi, perché Ciechino salva i suoi dati accanto al programma.

Il programma non è firmato, quindi al primo avvio Windows può mostrare l'avviso di SmartScreen: Altre informazioni, poi Esegui comunque.

All'avvio Ciechino controlla se c'è una versione nuova e propone di installarla: l'aggiornamento sostituisce il programma, lascia dove sono classifica, statistiche e impostazioni, e lo riapre.

## Come si gioca

La finestra ha la vista della partita, dove arriva il resoconto di ogni tiro, e sotto la barra di stato, con il livello, i punti e i colpi, e durante la mira i tre valori e il tempo che resta. F5 porta alla vista e F7 alla barra, per leggere: lì le frecce, Home e Fine fanno quello che fanno sempre e non muovono la mira, mentre Invio ed Esc valgono anche da lì; durante la mira F6 riporta sul controllo Mira, da dove i tasti dei valori muovono di nuovo la mira.

1. La barra spaziatrice, o Ctrl+N, comincia la partita, e a partita finita la barra spaziatrice ne comincia un'altra.
2. Ascolti il bersaglio, finché hai capito dove sta.
3. La barra spaziatrice avvia la mira: la direzione, l'alzo e la potenza partono dal centro della loro scala, il cursore va sul controllo Mira, che lascia passare i tasti senza che NVDA li trattenga, e i bersagli continuano a cantare al volume pieno; quelli che si muovono continuano a muoversi, e si mira dove saranno quando arriva il colpo.
4. Col cursore sulla Mira, ogni valore si muove finché tieni giù il suo tasto: le frecce sinistra e destra la direzione, le frecce su e giù l'alzo, Home e Fine la potenza, anche due valori insieme, da un capo all'altro della scala in 3 secondi, con una tacca a ogni decimo e un suono suo ai capi. La barra di stato dice i tre valori, in gradi, sempre con un decimale, e in percentuale. Invio spara; Esc annulla il tiro, e il proiettile resta nella riserva. Nei primi tre livelli la mira non ha tempo; dal quarto ne ha 30 secondi, uno in meno a ogni livello, fino a 10 dal ventiquattresimo, con un tick a ogni secondo sempre più acuto, e allo scadere il colpo parte da solo. Il tempo parte dopo gli avvisi e il resoconto del tiro di prima, così i tick non li coprono.
5. Parte il colpo: senti lo sparo, il proiettile che vola e l'impatto, dal punto in cui avviene; un centro finisce con un breve jingle, e quello del bonus è più lungo e più acuto.
6. Finito il suono dell'impatto, dopo un secondo di silenzio, arriva il resoconto. Nei primi sei livelli comincia con il sommario: dove il proiettile è passato rispetto al centro del bersaglio, visto dal bersaglio, di lato, in altezza e in profondità, e se saliva o scendeva, coi numeri fino al terzo livello, come L:+1,9 V:+15,8 P:+24,7 I:-18,5 G:79,0%, tutti nei primi 40 caratteri della riga, con la legenda dal quarantunesimo per il display braille, e a parole fino al sesto, con la riga sotto per la precisione globale e il punto di vista. Ogni resoconto dice anche la precisione globale, quanto il proiettile si è avvicinato al bersaglio considerando tutta la cupola, in percentuale: la distanza del passaggio più vicino al centro del bersaglio, in proporzione alla retta che va dal centro del bersaglio al punto della cupola più lontano da lui, lunga 63,9 metri per un bersaglio nel mezzo della cupola e 109,3 per uno in un angolo in fondo; così un tiro passato a 10 metri dal centro del bersaglio vale l'84,3% nel primo caso e il 90,8% nel secondo, e uno passato a 64 metri lo 0% e il 41,4%. Nei numeri è la G. Dopo il sommario, o in testa dal settimo livello, la situazione, su due righe: i colpi sparati nel livello e quelli che restano nella riserva, i bersagli presi nel livello e quanti ne mancano per passare al seguente; poi il tempo di gioco e il posto che avresti in classifica coi punti di adesso, col nome, i punti e il distacco di chi ti sta davanti, o al primo posto il vantaggio su chi segue. Poi dice l'esito, i valori del tiro e dove è passato il colpo; nei primi livelli, dopo un tiro a vuoto, anche di quanto e dove hai mancato il bersaglio.
7. Nei primi tre livelli, la palestra, la partita poi si ferma e tace, per leggere il resoconto con calma: la barra spaziatrice chiama il bersaglio nuovo dopo un centro, o riporta all'ascolto dopo un tiro a vuoto. Dal quarto livello si torna subito ad ascoltare.

La riserva parte da venti colpi. Un tiro a vuoto costa un colpo, un centro non costa niente e vale fino al doppio per la precisione. Dieci centri chiudono il livello e danno altri quindici colpi, un centro sul bonus otto; a riserva vuota la partita finisce.

Proiettile, nel menu Impostazioni, sceglie il timbro del proiettile in volo, seno, triangolo o sega morbida, e fa sentire un esempio di ciascuno mentre scorri l'elenco; la forma scelta vale dalla partita seguente, e i bersagli non la usano mai, così il colpo non si confonde con loro.

Scheda audio, nel menu Impostazioni, o Ctrl+Maiusc+X, sceglie dove suona Ciechino: rilegge le schede, così nell'elenco ci sono anche le cuffie Bluetooth accese dopo l'avvio, e dice la latenza di ognuna. La scheda scelta vale da subito e resta per i prossimi avvii; se manca, Ciechino suona su quella automatica, la scheda di Windows, e lo dice. Lo stesso tasto ridà il suono quando la scheda si stacca durante il gioco.

Ascolti, nel menu Aiuto, è il tutorial audio: un passo alla volta, con la barra spaziatrice che lo fa sentire e Invio che passa al seguente, il test stereo, il bersaglio che si muove nella cupola, i punti a caso da indovinare, la grandezza, le forme, i timbri, la mira manuale e il suo tempo, il volo e gli effetti; il passo del vicino e del lontano ritara l'orecchio sulla distanza quando cambi il volume, e finché non hai giocato la prima partita gli Ascolti si aprono da soli all'avvio.

Nella palestra, i primi tre livelli, Ctrl+P mette in pausa la partita, anche durante la mira, e la fa riprendere; dal quarto livello la pausa con Ctrl+P non c'è più. La partita si ferma da sola, in silenzio e in tutti i livelli, quando apri un menu o un dialogo o passi a un'altra finestra, e riprende da sola quando torni; una pausa chiesta con Ctrl+P invece resta, finché non la togli. Con una partita in corso, Ctrl+N e l'uscita chiedono prima se abbandonarla, col No già scelto.

## Aiuto

- Il manuale, con Maiusc+F1, spiega ogni menu, ogni tasto, lo spazio in metri, i suoni dei bersagli, i punti e i livelli.
- F1 dà la guida ai comandi, F2 le novità, F3 le informazioni sul programma.
- Nel menu Aiuto, Entra al PUB prepara la mail per iscriversi alla lista di supporto degli utenti, e Chiedi aiuto al PUB quella con la domanda e le informazioni sul computer, con i registri degli errori se vuoi.
- I problemi si segnalano anche come issue su GitHub: https://github.com/GabrieleBattaglia/Ciechino/issues

Un errore imprevisto, anche del motore audio, si annota in Ciechino - Errori.log, accanto al programma: è il file da allegare quando chiedi aiuto.

## Accessibilità

Ciechino è pensato per lo screen reader e il display braille: la vista e la barra di stato sono testi da leggere con le frecce, senza separatori grafici né tabelle, le finestre interne si aprono massimizzate e Esc le chiude, ogni comando sta in un menu con il suo tasto rapido, ogni azione ha il suo suono, e il resoconto di ogni tiro arriva da solo alla voce e al braille di NVDA o di JAWS, o alla voce di Windows, se l'impostazione Leggi i resoconti con la sintesi è accesa. Le misure sono in metri, gradi e metri al secondo, con i numeri all'italiana.

## Dai sorgenti

Serve Python 3.14, con le librerie di requirements.txt, e GBUtils, la libreria condivisa del parco software di IZ4APU, che non sta su PyPI e va clonata a parte, con la sua cartella in PYTHONPATH:

    git clone https://github.com/GabrieleBattaglia/GBUtils.git
    git clone https://github.com/GabrieleBattaglia/Ciechino.git
    set PYTHONPATH=C:\percorso\di\GBUtils
    cd Ciechino
    pip install -r requirements.txt
    python Ciechino.py

Gli strumenti dello sviluppo, cioè pytest, ruff, PyInstaller, Babel e polib, stanno in requirements-dev.txt. Le prove sono nella cartella tests, con pytest: non aprono mai la scheda audio vera, e le finestre nascono su un desktop di Windows nascosto. Il validatore è ruff check, senza il formattatore.

    pip install -r requirements-dev.txt
    python -m pytest -q
    ruff check .

La cartella strumenti contiene crea_preset.py, che aggiunge i preset di Ciechino alla collezione dei suoni di GBUtils, e elenco_suoni.py, che scrive suoni_di_ciechino.txt, l'elenco delle azioni con il loro preset da ritoccare con Acu_Maker.

Il pacchetto si compila con PyInstaller dalla ricetta Ciechino.spec, a cartella, poi zip_maker.py prepara Ciechino.zip per l'aggiornamento automatico e setup_maker.py l'installatore con Inno Setup 6, dalla ricetta Ciechino.iss:

    python -m PyInstaller --noconfirm Ciechino.spec
    python zip_maker.py
    python setup_maker.py

## Tradurre Ciechino

Ciechino nasce in italiano, ma ogni testo per l'utente passa da lingua.py, con le funzioni _, ngettext e plurale, così si può tradurre senza toccare il programma. babel.cfg dice a pybabel quali file leggere; il modello dei testi si estrae con le parole chiave _, ngettext:1,2 e plurale:2,3, e con i commenti per i traduttori:

    pybabel extract -F babel.cfg -k _ -k ngettext:1,2 -k plurale:2,3 --add-comments=Traduttori: --project=Ciechino --version=2.1.0 --copyright-holder="Gabriele Battaglia" --msgid-bugs-address=https://github.com/GabrieleBattaglia/Ciechino/issues -o locales/messages.pot .
    pybabel init -i locales/messages.pot -d locales -l en
    pybabel update -i locales/messages.pot -d locales
    pybabel compile -d locales

Una traduzione è il file locales/CODICE/LC_MESSAGES/messages.po, dove CODICE è il codice della lingua, come en o es, e si traduce con un programma come Poedit o con polib. I nomi fra graffe, come {n} o {livello}, restano identici, ma si possono spostare nella frase; i testi con un numero hanno una forma per ogni plurale della lingua. pybabel compile crea i messages.mo, che il pacchetto porta con sé. La lingua si sceglie con Lingua, nel menu Impostazioni, e vale dal riavvio; al primo avvio Ciechino parla la lingua di Windows, se qualcuno l'ha tradotta. Una traduzione si propone con una pull request, o mandando il suo messages.po al PUB. Il manuale resta in italiano.

## Dati

Ciechino scrive tutto accanto al programma: Ciechino - Classifica.json, la classifica dei cento migliori; Ciechino - Statistiche.json, le statistiche con i primati e le ultime partite; Ciechino - Impostazioni.json, il volume, la forma del proiettile, la scheda audio scelta, la lettura dei resoconti, l'uscita della sintesi e l'ultimo nome usato in classifica. L'uscita della sintesi non è la scheda audio, che si sceglie con Ctrl+Maiusc+X: dice chi legge i resoconti, automatica, oppure nvda, jaws o sapi5, la voce di Windows, e si cambia soltanto scrivendola nel file, a Ciechino chiuso. Ognuno si salva con la sua copia .bak, e un file rovinato non si sovrascrive mai: si mette da parte come .illeggibile. Accanto stanno anche i registri, Ciechino - Errori.log e i resoconti dei crash, Ciechino - Registri per il PUB.txt, quando lo prepari, e selected_language.json, la lingua scelta. Chi gioca dai sorgenti li trova nella cartella del repository: sono dati personali di chi gioca, e .gitignore li tiene tutti fuori da git, con le loro copie .bak, .tmp e .illeggibile, come in MeTeOra (decisione D15).

I suoni degli eventi vengono dalla collezione comune Acu_Collection.json di GBUtils, con il prefisso ciechino_; l'elenco delle azioni con il loro preset è suoni_di_ciechino.txt. Le voci dei valori della mira, le loro tacche, i tick del tempo della mira, i bersagli e il proiettile li crea il motore audio di Ciechino mentre suonano. Tutti i numeri della meccanica di gioco, dalla cupola ai punti, stanno in costanti.py, per tararli giocando; il progetto e le decisioni stanno in docs/progetto.txt, l'idea di partenza in docs/idea_73.txt.

## Licenza

Ciechino è software libero, con licenza GNU GPL versione 3: vedi il file LICENSE.

## Autori

Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
