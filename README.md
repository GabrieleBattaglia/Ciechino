# Ciechino, il tiro a segno a orecchio

Un gioco di tiro accessibile, pensato da un giocatore cieco per chi non vede, e per chi vuole provare a mirare con le orecchie. Il nome è un gioco di parole fra cieco e cecchino.

Davanti a te, a 30 metri, comincia un cubo di 50 metri per lato, e là dentro cantano i bersagli: la panoramica dice se stanno a destra o a sinistra, la nota quanto sono in alto, il volume quanto sono lontani. Per sparare scegli tre valori, la direzione, l'alzo e la potenza, fermando con la barra spaziatrice tre suoni che scivolano; poi segui il colpo con l'orecchio, fino all'impatto o fino a terra. I colpi sono una riserva, dieci centri chiudono il livello, i bersagli si rimpiccioliscono, si muovono e si moltiplicano, e ogni tanto passa un bonus che vale cinque volte tanto. I cento punteggi migliori restano in classifica.

## Scaricare e installare

Ciechino gira su Windows 10 e 11 a 64 bit, e si gioca meglio in cuffia. L'ultima versione sta nella pagina delle release su GitHub: https://github.com/GabrieleBattaglia/Ciechino/releases

- **Con l'installatore, Ciechino-Setup:** chiede di accettare la licenza GPL 3 e installa Ciechino nella tua cartella dei programmi, senza permessi di amministratore. Alla disinstallazione chiede se cancellare anche la classifica, le statistiche e le impostazioni.
- **Con lo zip, Ciechino.zip:** si scompatta in una cartella dove puoi scrivere, mai in Programmi, perché Ciechino salva i suoi dati accanto al programma.

Il programma non è firmato, quindi al primo avvio Windows può mostrare l'avviso di SmartScreen: Altre informazioni, poi Esegui comunque.

All'avvio Ciechino controlla se c'è una versione nuova e propone di installarla: l'aggiornamento sostituisce il programma, lascia dove sono classifica, statistiche e impostazioni, e lo riapre.

## Come si gioca

La finestra ha la vista della partita, dove arriva il resoconto di ogni tiro, e sotto la barra di stato, con il livello, i punti e i colpi; F5 porta alla vista, F7 alla barra.

1. La barra spaziatrice, o Ctrl+N, comincia la partita.
2. Ascolti il bersaglio, finché hai capito dove sta.
3. La barra spaziatrice avvia la mira: il tempo si ferma, e i bersagli restano dove sono, più piano.
4. Tre scivolate, direzione, alzo e potenza: ognuna comincia dopo un secondo di silenzio con tre bip, poi scivola per tre secondi con dieci tacche, e la barra spaziatrice la ferma sul valore che vuoi. Lo spazio nei bip sceglie il minimo, nessuno spazio il massimo.
5. Parte il colpo: senti lo sparo, il proiettile che vola e l'impatto, dal punto in cui avviene.
6. Il resoconto dice l'esito, i valori del tiro e dove è passato il colpo; nei primi livelli, dopo un tiro a vuoto, anche di quanto e dove hai mancato il bersaglio.

Un tiro a vuoto costa un colpo, un centro non costa niente e vale fino al doppio per la precisione. Dieci centri chiudono il livello e danno altri dieci colpi; a riserva vuota la partita finisce.

## Aiuto

- Il manuale, con Maiusc+F1, spiega ogni menu, ogni tasto, lo spazio in metri, i suoni dei bersagli, i punti e i livelli.
- F1 dà la guida ai comandi, F2 le novità, F3 le informazioni sul programma.
- Nel menu Aiuto, Entra al PUB prepara la mail per iscriversi alla lista di supporto degli utenti, e Chiedi aiuto al PUB quella con la domanda e le informazioni sul computer, con i registri degli errori se vuoi.
- I problemi si segnalano anche come issue su GitHub: https://github.com/GabrieleBattaglia/Ciechino/issues

Un errore imprevisto si annota in Ciechino - Errori.log, accanto al programma: è il file da allegare quando chiedi aiuto.

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

La cartella strumenti contiene i banchi d'ascolto con cui Gabriele tara il gioco, banco_mira.py per le scivolate, banco_bersagli.py per le forme e le grandezze dei bersagli, banco_volo.py per il proiettile e gli effetti, che si giocano alla tastiera e scrivono gli esiti accanto a sé; crea_preset.py, che aggiunge i preset di Ciechino alla collezione dei suoni di GBUtils, e elenco_suoni.py, che scrive suoni_di_ciechino.txt, l'elenco delle azioni con il loro preset da ritoccare con Acu_Maker.

Il pacchetto si compila con PyInstaller dalla ricetta Ciechino.spec, a cartella, poi zip_maker.py prepara Ciechino.zip per l'aggiornamento automatico e setup_maker.py l'installatore con Inno Setup 6, dalla ricetta Ciechino.iss:

    python -m PyInstaller --noconfirm Ciechino.spec
    python zip_maker.py
    python setup_maker.py

## Tradurre Ciechino

Ciechino nasce in italiano, ma ogni testo per l'utente passa da lingua.py, con le funzioni _, ngettext e plurale, così si può tradurre senza toccare il programma. babel.cfg dice a pybabel quali file leggere; il modello dei testi si estrae con le parole chiave _, ngettext:1,2 e plurale:2,3, e con i commenti per i traduttori:

    pybabel extract -F babel.cfg -k _ -k ngettext:1,2 -k plurale:2,3 --add-comments=Traduttori: -o locales/messages.pot .
    pybabel init -i locales/messages.pot -d locales -l en
    pybabel update -i locales/messages.pot -d locales
    pybabel compile -d locales

Una traduzione è il file locales/CODICE/LC_MESSAGES/messages.po, dove CODICE è il codice della lingua, come en o es, e si traduce con un programma come Poedit o con polib. I nomi fra graffe, come {n} o {livello}, restano identici, ma si possono spostare nella frase; i testi con un numero hanno una forma per ogni plurale della lingua. pybabel compile crea i messages.mo, che il pacchetto porta con sé. La lingua si sceglie con Lingua, nel menu Impostazioni, e vale dal riavvio; al primo avvio Ciechino parla la lingua di Windows, se qualcuno l'ha tradotta. Una traduzione si propone con una pull request, o mandando il suo messages.po al PUB. Il manuale resta in italiano.

## Dati

Ciechino scrive tutto accanto al programma: Ciechino - Classifica.json, la classifica dei cento migliori; Ciechino - Statistiche.json, le statistiche con i primati e le ultime partite; Ciechino - Impostazioni.json, il volume, la sintesi, l'uscita audio e l'ultimo nome usato in classifica. Ognuno si salva con la sua copia .bak, e un file rovinato non si sovrascrive mai: si mette da parte come .illeggibile. Accanto stanno anche i registri, Ciechino - Errori.log e i resoconti dei crash, e selected_language.json, la lingua scelta.

I suoni degli eventi vengono dalla collezione comune Acu_Collection.json di GBUtils, con il prefisso ciechino_; l'elenco delle azioni con il loro preset è suoni_di_ciechino.txt. Le scivolate, i bersagli e il proiettile li crea il motore audio di Ciechino mentre suonano. Tutti i numeri della meccanica di gioco, dal cubo ai punti, stanno in costanti.py, per tararli giocando; il progetto e le decisioni stanno in docs/progetto.txt, l'idea di partenza in docs/idea_73.txt.

## Licenza

Ciechino è software libero, con licenza GNU GPL versione 3: vedi il file LICENSE.

## Autori

Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
