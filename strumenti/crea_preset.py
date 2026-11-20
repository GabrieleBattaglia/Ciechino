"""
Ciechino, i preset nuovi dei suoni: li aggiunge alla collezione condivisa di GBUtils e li firma.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello dei passi dei suoni di MESS, con la decisione D10:
i preset fatti per Ciechino entrano in Acu_Collection.json col prefisso ciechino_ e con la firma
ciechino nel campo usato_da, autorizzati da Gabriele il 19 novembre 2026.
La tabella PRESET ha un preset nuovo per ogni evento di suoni.EVENTI, tranne il caffè, che usa
donazione come tutto il parco. Le regole di Gabriele sui suoni stanno qui e le controlla
controlla_tabella prima di ogni scrittura: un preset per evento, mai lo stesso per due eventi;
niente onda quadra; brevi e morbidi gli eventi che suonano spesso; lo sparo, i bersagli colpiti e il
colpo a terra fatti di rumore, perché idee.txt li vuole rappresentati da un effetto costituito da
rumore; esiti diversi riconoscibili a orecchio.
Lo score sta nella forma del file della collezione: quartine [nota, durata, panorama, volume], dove
il volume è lo scarto dalla base 0,5 di Acusticator, un numero o la lista dei due capi di una
scivolata. Per il rumore, kind da 5 a 8, il primo campo è la banda del passabanda.
La scrittura passa soltanto da Acusticator.aggiorna, che rilegge il file sotto lucchetto e cambia
soltanto ciò che serve, e da Acusticator.firma: gli altri preset non si toccano. Lo strumento è
idempotente: un preset che c'è già, uguale alla tabella, non si riscrive, e una firma che c'è già
non si rimette; se non c'è niente da fare il file non si scrive affatto. Un preset che c'è già ma è
diverso dalla tabella, per esempio ritoccato da Gabriele con Acu_Maker, resta com'è, e lo strumento
lo dice: lo sostituisce soltanto con --sostituisci.
Uso, dalla cartella del programma:
python strumenti/crea_preset.py --prova dice soltanto che cosa farebbe sulla collezione condivisa;
python strumenti/crea_preset.py --collezione PERCORSO lavora su un'altra collezione, per esempio
una copia da provare;
python strumenti/crea_preset.py lavora sulla collezione condivisa di GBUtils, e dopo ci vuole un
commit a parte nel repository di GBUtils, con la sua versione alzata.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent
if str(RADICE) not in sys.path:
    sys.path.insert(0, str(RADICE))

from GBUtils import Acusticator  # noqa: E402

import suoni  # noqa: E402

# L'identificativo di Ciechino nelle firme usato_da della collezione.
APP = "ciechino"
PREFISSO = "ciechino_"
# Ogni descrizione dei preset nuovi comincia così, come quelle di MESS cominciano con "MESS, ".
INIZIO_DESCRIZIONE = "Ciechino, "
# L'evento del caffè usa il preset donazione, l'unica onda quadra ammessa, come in tutto il parco.
EVENTO_DEL_CAFFE = "caffe"
PRESET_DEL_CAFFE = "donazione"
ONDA_QUADRA = 2
# Gli eventi che suonano spesso, a ogni valore della mira o a ogni F5 e F7: la regola di Gabriele
# li vuole brevi e morbidi, sotto questa durata in secondi. Non sta in costanti.py perché riguarda
# il disegno dei preset, non il gioco.
EVENTI_FREQUENTI = ("valore_preso", "mira_avviata", "vai_alla_vista", "vai_alla_barra")
DURATA_MASSIMA_FREQUENTI = 0.08

# I preset nuovi, nell'ordine dei gruppi di suoni.GRUPPI. Le note si scrivono col nome inglese, le
# bande del rumore in hertz, col punto per la scivolata.
PRESET = {
    # La partita, l'inizio e la fine.
    "ciechino_nuova_partita": {
        "score": [["2500-9000", 0.012, -0.3, 0.35], ["1200-3500.2500-6000", 0.07, -0.1, 0.1], ["p", 0.04, 0.0, -0.5],
                  ["700-4000", 0.035, 0.2, 0.45]],
        "kind": 5,
        "adsr": [3.0, 30.0, 50.0, 40.0],
        "descrizione": ("Ciechino, comincia una partita nuova: il cannone che si arma, rumore bianco, uno scatto acuto a sinistra fra 2500 e "
                        "9000 Hz, l'otturatore che scorre con la banda che sale da 1200-3500 a 2500-6000 Hz e, dopo una breve pausa, il colpo "
                        "secco della chiusura appena a destra, fra 700 e 4000 Hz; 157 ms. Va bene per un'arma che si carica, un meccanismo "
                        "che si arma o un inizio pronto all'azione."),
    },
    "ciechino_ultimo_colpo": {
        "score": [["e3.c3", 0.07, 0.0, 0.05], ["p", 0.09, 0.0, -0.5], ["d3.a#2", 0.1, 0.0, -0.02]],
        "kind": 4,
        "adsr": [1.0, 60.0, 20.0, 39.0],
        "descrizione": ("Ciechino, nella riserva resta l'ultimo colpo: un battito di cuore, due colpi gravi di dente di sega che cadono, "
                        "dal mi 3 al do 3 e, dopo una breve pausa, dal re 3 al la diesis 2, al centro, in 260 ms. Va bene per un'ultima "
                        "possibilità, una tensione che sale o un avvertimento grave."),
    },
    "ciechino_partita_finita": {
        "score": [["d6", 0.09, 0.5, -0.05], ["a5", 0.09, 0.25, -0.05], ["f5", 0.09, 0.0, -0.05], ["d5", 0.11, -0.25, -0.05],
                  ["a4", 0.42, -0.5, -0.02]],
        "kind": 1,
        "adsr": [4.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, la riserva è vuota e la partita finisce: una breve frase di seno che scende, re 6, la 5, fa 5 e re 5, "
                        "l'arpeggio di re minore da destra verso sinistra, e un la 4 lungo che si posa; 800 ms. È lo specchio di "
                        "ciechino_livello_su, che sale in maggiore. Va bene per una fine, una sconfitta serena o una chiusura."),
    },
    "ciechino_in_classifica": {
        "score": [["g4", 0.09, -0.3, -0.15], ["c5", 0.09, -0.15, -0.15], ["e5", 0.09, 0.0, -0.15], ["g5", 0.2, 0.15, -0.12],
                  ["p", 0.03, 0.0, -0.5], ["e5", 0.08, 0.0, -0.15], ["g5", 0.42, 0.0, -0.1]],
        "kind": 4,
        "adsr": [5.0, 20.0, 70.0, 25.0],
        "descrizione": ("Ciechino, a partita finita il punteggio entra nella classifica dei migliori: una fanfara di dente di sega, "
                        "squillante come un ottone, sol 4, do 5, mi 5 e sol 5, poi la risposta, mi 5 e un sol 5 lungo, al centro, in un "
                        "secondo; suona prima della finestra del nome. Il suo rovescio è ciechino_fuori_classifica. Va bene per un record, "
                        "un ingresso fra i migliori o una vittoria."),
    },
    "ciechino_fuori_classifica": {
        "score": [["g4", 0.16, 0.0, -0.2], ["f#4", 0.16, 0.0, -0.2], ["f4", 0.16, 0.0, -0.2], ["e4.d#4", 0.45, 0.0, [-0.2, -0.32]]],
        "kind": 4,
        "adsr": [8.0, 15.0, 70.0, 30.0],
        "descrizione": ("Ciechino, a partita finita il punteggio non entra nella classifica dei migliori: un ottone di dente di sega, "
                        "piano, che scende per semitoni, sol 4, fa diesis 4 e fa 4, e un mi 4 lungo che cede ancora verso il re diesis 4 "
                        "spegnendosi, al centro, in 930 ms. Un peccato detto con un sorriso. Il suo rovescio è ciechino_in_classifica. Va "
                        "bene per un tentativo non riuscito, una consolazione o un risultato che non basta."),
    },
    # La mira.
    "ciechino_mira_avviata": {
        "score": [["200-1200.400-2000", 0.07, 0.0, 0.15]],
        "kind": 7,
        "adsr": [55.0, 0.0, 100.0, 40.0],
        "descrizione": ("Ciechino, lo spazio avvia la mira e il tempo del gioco si ferma: un respiro trattenuto, un soffio breve di rumore "
                        "marrone che nasce piano e si schiarisce, con la banda che sale da 200-1200 a 400-2000 Hz, al centro, in 70 ms. "
                        "Suona a ogni tiro, quindi è breve e morbido, e non ha altezza, per non confondersi con i bip e le scivolate della "
                        "mira. Va bene per un'attenzione che si concentra, una mira o un fiato sospeso."),
    },
    "ciechino_valore_preso": {
        "score": [["a5", 0.016, 0.0, -0.12], ["e5", 0.044, 0.0, -0.08]],
        "kind": 3,
        "adsr": [5.0, 45.0, 30.0, 40.0],
        "descrizione": ("Ciechino, la mira prende un valore, la direzione, l'alzo o la potenza: uno scatto di fermo, onda triangolare, un "
                        "tocco brevissimo di la 5 che cade sul mi 5, al centro, in 60 ms. Suona tre volte a ogni tiro, quindi è breve e "
                        "morbido, e scende, mentre i bip della mira salgono. Va bene per un valore fissato, una scelta presa o un fermo "
                        "che scatta."),
    },
    # Esc nella mira (D25), il 20 novembre 2026: il respiro di ciechino_mira_avviata lasciato andare.
    "ciechino_mira_annullata": {
        "score": [["500-2400.150-900", 0.18, 0.0, [0.2, -0.15]]],
        "kind": 7,
        "adsr": [5.0, 0.0, 100.0, 60.0],
        "descrizione": ("Ciechino, Esc annulla la mira, e il proiettile resta nella riserva: il respiro trattenuto di ciechino_mira_avviata "
                        "che si lascia andare, un soffio di rumore marrone con la banda che scende da 500-2400 a 150-900 Hz e si spegne, al "
                        "centro, in 180 ms. Va bene per un'azione annullata, un sospiro di sollievo o un'attenzione che si scioglie."),
    },
    # Il tiro e i suoi esiti. Dal banco del volo del 19 novembre 2026 (D16): tutti e quattro più lunghi e più marcati, lo
    # sparo più grave e con un decadimento più lungo, il tonfo con un rilascio più lungo, e i due colpiti col loro jingle,
    # di bande strettissime di rumore, tre note il normale e otto il bonus, che resta più brillante.
    "ciechino_sparo": {
        "score": [["900-6000", 0.008, 0.0, 0.35], ["120-2200.40-200", 0.45, 0.0, 0.5]],
        "kind": 5,
        "adsr": [1.0, 40.0, 45.0, 50.0],
        "descrizione": ("Ciechino, dopo il terzo valore della mira parte il colpo dal cannone: uno sparo pieno di rumore bianco, uno "
                        "schiocco di 8 ms fra 900 e 6000 Hz e il corpo del colpo, con la banda che scende da 120-2200 a 40-200 Hz e si "
                        "spegne piano in un rimbombo grave, al centro, in 458 ms. Va bene per uno sparo, un colpo di cannone o una "
                        "detonazione."),
    },
    "ciechino_colpito": {
        "score": [["1450-1700", 0.06, 0.0, 0.4], ["150-8000.80-900", 0.4, 0.0, 0.5], ["p", 0.02, 0.0, -0.5],
                  ["1295-1342", 0.08, 0.0, 0.25], ["1540-1596", 0.08, 0.0, 0.25], ["2056-2131", 0.18, 0.0, 0.3]],
        "kind": 6,
        "adsr": [1.0, 30.0, 55.0, 45.0],
        "descrizione": ("Ciechino, il proiettile colpisce un bersaglio: rumore rosa, il tocco metallico dell'impatto, una banda stretta "
                        "fra 1450 e 1700 Hz che suona quasi come un sol 6, poi l'esplosione, piena e larga, con la banda che scende da "
                        "150-8000 a 80-900 Hz, e un breve jingle di tre note, bande strettissime attorno al mi 6, al sol 6 e a un do 7 "
                        "più lungo; 820 ms, al centro, perché il gioco lo colloca nel punto dell'impatto. Il jingle del bonus, in "
                        "ciechino_colpito_bonus, è più acuto, più rapido e di otto note. Va bene per un bersaglio centrato, un colpo "
                        "andato a segno o un'esplosione con un premio."),
    },
    "ciechino_colpito_bonus": {
        "score": [["1900-2300", 0.05, 0.0, 0.35], ["200-10000.250-2500", 0.4, 0.0, 0.5], ["p", 0.02, 0.0, -0.5],
                  ["2056-2131", 0.045, 0.0, 0.2], ["2590-2685", 0.045, 0.0, 0.2], ["3080-3192", 0.045, 0.0, 0.2], ["4112-4261", 0.07, 0.0, 0.25],
                  ["p", 0.03, 0.0, -0.5], ["3080-3192", 0.045, 0.0, 0.2], ["3458-3583", 0.045, 0.0, 0.2], ["3881-4022", 0.045, 0.0, 0.2],
                  ["4112-4261", 0.24, 0.0, 0.3]],
        "kind": 8,
        "adsr": [1.0, 30.0, 50.0, 45.0],
        "descrizione": ("Ciechino, il proiettile colpisce un bersaglio bonus: rumore azzurro, più brillante e più ricco del bersaglio "
                        "normale, il tocco dell'impatto fra 1900 e 2300 Hz, l'esplosione con la banda che scende da 200-10000 a "
                        "250-2500 Hz, e un jingle di otto note, bande strettissime un'ottava sopra quelle di ciechino_colpito: "
                        "l'arpeggio rapido di do maggiore che sale, do 7, mi 7, sol 7 e do 8, e dopo una breve pausa la scala che "
                        "risale, sol 7, la 7 e si 7, fino a un do 8 lungo; 1080 ms, al centro, perché il gioco lo colloca nel punto "
                        "dell'impatto. Va bene per un premio colpito, un bersaglio speciale o un festeggiamento."),
    },
    "ciechino_a_terra": {
        "score": [["60-1100.42-220", 0.52, 0.0, 0.5], ["p", 0.03, 0.0, -0.5], ["400-2400", 0.07, 0.0, 0.0],
                  ["300-1800.180-1000", 0.16, 0.0, -0.12]],
        "kind": 7,
        "adsr": [1.0, 30.0, 45.0, 60.0],
        "descrizione": ("Ciechino, il proiettile arriva a terra senza colpire niente: un tonfo grave e lungo di rumore marrone, con la "
                        "banda che scende da 60-1100 a 42-220 Hz e una coda che si spegne piano, poi un poco di terra che ricade, due "
                        "brevi fruscii più piano, uno fra 400 e 2400 Hz e uno con la banda che scende da 300-1800 a 180-1000 Hz; 780 "
                        "ms, al centro, perché il gioco lo colloca nel punto dove cade. Va bene per un tiro a vuoto, un oggetto pesante "
                        "che cade o un colpo nel terreno."),
    },
    # Il proiettile perduto (D25), il 20 novembre 2026: una barra della mira è passata cinque volte senza lo spazio, e il colpo
    # non parte. Rumore bianco come lo sparo, perché è lo sparo che non c'è: la miccia che sfrigola e si spegne senza il colpo.
    "ciechino_proiettile_perduto": {
        "score": [["4000-10000.2500-8000", 0.3, 0.0, [0.0, 0.25]], ["p", 0.05, 0.0, -0.5], ["400-1600.150-500", 0.25, 0.0, [0.15, -0.35]]],
        "kind": 5,
        "adsr": [5.0, 10.0, 80.0, 30.0],
        "descrizione": ("Ciechino, una barra della mira è passata cinque volte senza lo spazio, e il proiettile è perduto: lo sparo che non "
                        "c'è, rumore bianco, una miccia che sfrigola acuta e cresce, con la banda che scende da 4000-10000 a 2500-8000 Hz, "
                        "poi, dopo una breve pausa, un soffio sordo che si spegne, da 400-1600 a 150-500 Hz, senza il colpo, al centro, in "
                        "600 ms. Va bene per un colpo che fa cilecca, una miccia spenta o un'occasione sprecata."),
    },
    # I livelli e i bersagli bonus.
    "ciechino_livello_su": {
        "score": [["a4", 0.065, -0.6, -0.05], ["d5", 0.065, -0.3, -0.05], ["f#5", 0.065, 0.0, -0.05], ["a5", 0.065, 0.3, -0.05],
                  ["d6", 0.32, 0.5, 0.0]],
        "kind": 3,
        "adsr": [4.0, 15.0, 70.0, 30.0],
        "descrizione": ("Ciechino, i centri del livello sono fatti e si sale di livello: una breve frase che sale, onda triangolare, "
                        "l'arpeggio di re maggiore, la 4, re 5, fa diesis 5 e la 5, da sinistra verso destra, e un re 6 lungo che resta in "
                        "alto; 580 ms. Lo specchio è ciechino_partita_finita, che scende in minore. Va bene per un livello superato, un "
                        "traguardo o una promozione."),
    },
    "ciechino_bonus_appare": {
        "score": [["e6", 0.035, -0.5, -0.2], ["g#6", 0.035, -0.25, -0.2], ["e6", 0.035, 0.0, -0.2], ["g#6", 0.035, 0.25, -0.2],
                  ["b6", 0.11, 0.5, -0.15]],
        "kind": 1,
        "adsr": [8.0, 20.0, 70.0, 30.0],
        "descrizione": ("Ciechino, compare un bersaglio bonus, che attraversa lo spazio dei bersagli e poi esce: uno scintillio di seno, "
                        "un trillo veloce fra mi 6 e sol diesis 6 che passa da sinistra a destra e si ferma su un si 6, in 250 ms. Il suo "
                        "seguito, quando il bonus esce senza essere colpito, è ciechino_bonus_uscito. Va bene per un'occasione che si "
                        "presenta, un oggetto raro o un premio di passaggio."),
    },
    "ciechino_bonus_uscito": {
        "score": [["g#6", 0.035, 0.0, -0.12], ["e6", 0.035, 0.0, -0.15], ["g#6", 0.035, 0.0, -0.18], ["e6", 0.035, 0.0, -0.21],
                  ["c#6.g#5", 0.16, 0.0, [-0.24, -0.4]]],
        "kind": 1,
        "adsr": [8.0, 20.0, 70.0, 30.0],
        "descrizione": ("Ciechino, il bersaglio bonus esce senza essere stato colpito: il trillo di ciechino_bonus_appare, fra sol diesis "
                        "6 e mi 6, sempre più piano, che invece di salire scivola giù dal do diesis 6 al sol diesis 5 e svanisce, al "
                        "centro, in 300 ms. Un'occasione persa. Va bene per un premio che se ne va, un'opportunità sfumata o qualcosa "
                        "che scompare."),
    },
    # La pausa e lo spazio a vuoto.
    "ciechino_pausa": {
        "score": [["a5.d5", 0.22, 0.0, [-0.1, -0.35]]],
        "kind": 3,
        "adsr": [5.0, 0.0, 100.0, 30.0],
        "descrizione": ("Ciechino, la partita va in pausa: un nastro che rallenta, onda triangolare che scivola dal la 5 al re 5 e cala "
                        "di volume, al centro, in 220 ms. Il suo specchio è ciechino_ripresa. Va bene per una pausa, un'attesa o "
                        "qualcosa che si ferma."),
    },
    "ciechino_ripresa": {
        "score": [["d5.a5", 0.22, 0.0, [-0.35, -0.1]]],
        "kind": 3,
        "adsr": [30.0, 0.0, 100.0, 10.0],
        "descrizione": ("Ciechino, la partita riprende dalla pausa: il nastro che riparte, onda triangolare che sale dal re 5 al la 5 "
                        "crescendo di volume, al centro, in 220 ms. È lo specchio di ciechino_pausa. Va bene per una ripresa, un "
                        "riavvio o qualcosa che torna in moto."),
    },
    "ciechino_non_ora": {
        "score": [["d4", 0.045, 0.0, -0.1], ["p", 0.035, 0.0, -0.5], ["d4.b3", 0.08, 0.0, -0.15]],
        "kind": 3,
        "adsr": [5.0, 30.0, 50.0, 40.0],
        "descrizione": ("Ciechino, lo spazio premuto quando non può fare niente, per esempio col proiettile in volo, o Ctrl+P senza una partita: "
                        "due tocchi sordi di onda triangolare sul re 4, il secondo più piano, che cede di una terza minore fino al si 3, "
                        "al centro, in 160 ms. Un garbato non adesso. Va bene per un comando che arriva nel momento sbagliato o un tasto "
                        "che deve aspettare."),
    },
    # La finestra, l'avvio, l'uscita e gli errori.
    "ciechino_avvio": {
        "score": [["e5", 0.06, -0.5, -0.1], ["b5", 0.06, -0.3, -0.1], ["a5", 0.08, -0.1, -0.05], ["e6", 0.35, 0.0, -0.05]],
        "kind": 3,
        "adsr": [3.0, 12.0, 75.0, 25.0],
        "descrizione": ("Ciechino, il programma si apre: il motivo di Ciechino, onda triangolare, mi 5, il salto di quinta al si 5, un "
                        "passo indietro sul la 5 e l'arrivo lungo sul mi 6, come una mira che trova il bersaglio, da sinistra verso il "
                        "centro, in 550 ms. Lo rovescia ciechino_uscita. Va bene per un benvenuto, un avvio o una presentazione."),
    },
    "ciechino_uscita": {
        "score": [["e6", 0.08, 0.0, -0.05], ["a5", 0.06, -0.1, -0.05], ["b5", 0.06, -0.3, -0.05], ["e5", 0.4, -0.5, -0.05]],
        "kind": 3,
        "adsr": [3.0, 12.0, 75.0, 30.0],
        "descrizione": ("Ciechino, l'uscita dal programma: il motivo di ciechino_avvio al contrario, onda triangolare, mi 6, la 5, si 5 e "
                        "un mi 5 lungo che si posa, dal centro verso sinistra, in 600 ms. Va bene per un congedo, un'uscita o la fine di "
                        "una sessione."),
    },
    "ciechino_primo_piano": {
        "score": [["b5", 0.04, 0.0, -0.2], ["p", 0.02, 0.0, -0.5], ["b5.e6", 0.1, 0.0, -0.15]],
        "kind": 1,
        "adsr": [3.0, 20.0, 60.0, 30.0],
        "descrizione": ("Ciechino, si apre Ciechino mentre è già aperto, e la finestra aperta torna in primo piano: un eccomi di seno, un "
                        "si 5 breve e la scivolata che sale dal si 5 al mi 6, al centro, in 160 ms. Va bene per una finestra che si "
                        "ripresenta, un richiamo o un ritorno."),
    },
    "ciechino_errore": {
        "score": [["a3.e3", 0.2, 0.0, -0.1], ["p", 0.05, 0.0, -0.5], ["e3.a2", 0.35, 0.0, -0.1]],
        "kind": 4,
        "adsr": [3.0, 20.0, 70.0, 30.0],
        "descrizione": ("Ciechino, un errore imprevisto: dente di sega grave che crolla in due tempi, una scivolata dal la 3 al mi 3 e, "
                        "dopo una pausa, una più lunga dal mi 3 al la 2, al centro, in 600 ms. Va bene per un errore, un guasto o "
                        "un'operazione che non riesce."),
    },
    # La finestra della 0.1.0: la riapertura, le notizie dell'avvio e i salvataggi che non riescono.
    "ciechino_riavvio": {
        "score": [["e6", 0.06, 0.0, -0.08], ["b5", 0.06, -0.2, -0.08], ["e5", 0.1, -0.4, -0.08], ["p", 0.05, 0.0, -0.5], ["e5", 0.05, -0.4, -0.08],
                  ["b5", 0.05, -0.2, -0.08], ["e6", 0.2, 0.0, -0.05]],
        "kind": 3,
        "adsr": [3.0, 12.0, 75.0, 25.0],
        "descrizione": ("Ciechino, il programma si chiude per riaprirsi subito, per la lingua nuova: il motivo dell'uscita che torna "
                        "indietro, onda triangolare, mi 6, si 5 e mi 5 dal centro verso sinistra e, dopo una breve pausa, mi 5, si 5 e un "
                        "mi 6 più lungo che risalgono verso il centro, in 570 ms. Va bene per un riavvio, un arrivederci a subito o un "
                        "ritorno."),
    },
    "ciechino_crash_precedente": {
        "score": [["c4", 0.08, 0.0, -0.12], ["p", 0.06, 0.0, -0.5], ["c4", 0.08, 0.0, -0.12], ["p", 0.06, 0.0, -0.5], ["g3.d3", 0.3, 0.0, -0.12]],
        "kind": 4,
        "adsr": [3.0, 25.0, 60.0, 30.0],
        "descrizione": ("Ciechino, all'avvio, dopo il suono d'apertura: la volta scorsa il programma si è chiuso per un errore grave. "
                        "Dente di sega grave, due colpi di do 4 come chi bussa per avvertire e una scivolata che cade dal sol 3 al re 3, "
                        "al centro, in 580 ms. Va bene per un guasto passato, un avvertimento o un ricordo che non è andato bene."),
    },
    "ciechino_dati_rovinati": {
        "score": [["a5", 0.03, 0.0, -0.15], ["g#5", 0.03, 0.0, -0.15], ["a5", 0.03, 0.0, -0.15], ["p", 0.04, 0.0, -0.5], ["d5.c#5", 0.18, 0.0, -0.15]],
        "kind": 1,
        "adsr": [4.0, 20.0, 60.0, 30.0],
        "descrizione": ("Ciechino, un file dei dati non si leggeva e il programma riparte dalla copia o da zero: un inciampo di seno, "
                        "la 5, sol diesis 5 e la 5 come un disco che salta, poi un re 5 che scivola giù di un semitono, al centro, in "
                        "310 ms. Va bene per dei dati rovinati, un file messo da parte o un ripristino."),
    },
    "ciechino_salvataggio_non_riuscito": {
        "score": [["b3", 0.12, 0.0, -0.15], ["p", 0.04, 0.0, -0.5], ["f3", 0.25, 0.0, -0.15]],
        "kind": 4,
        "adsr": [3.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, la classifica, le statistiche o le impostazioni non si sono potute salvare: dente di sega grave, un "
                        "si 3 e, dopo una breve pausa, un fa 3 più lungo, il tritono che scende, al centro, in 410 ms. Va bene per un "
                        "salvataggio fallito, un disco che non scrive o un lavoro che non resta."),
    },
    # La scelta della scheda audio (D20): la scheda scelta che non c'è o non si apre.
    "ciechino_scheda_non_aperta": {
        "score": [["d5", 0.025, -0.4, -0.2], ["p", 0.025, 0.0, -0.5], ["e5", 0.025, 0.0, -0.2], ["p", 0.06, 0.0, -0.5],
                  ["c5.f#4", 0.22, 0.4, [-0.2, -0.32]]],
        "kind": 4,
        "adsr": [3.0, 30.0, 55.0, 30.0],
        "descrizione": ("Ciechino, la scheda audio scelta non c'è o non si apre, e il suono va su quella automatica: il selettore di "
                        "ciechino_scheda_audio che s'inceppa, dente di sega piano, due scatti brevissimi, re 5 e mi 5, e dopo una breve "
                        "pausa una scivolata che cade dal do 5 al fa diesis 4 spegnendosi, da sinistra a destra, in 355 ms. Va bene per "
                        "un collegamento che non riesce, un dispositivo che manca o un'uscita di ripiego."),
    },
    # La finestra, i menu.
    "ciechino_classifica": {
        "score": [["e5", 0.08, -0.5, -0.05], ["p", 0.03, 0.0, -0.5], ["g5", 0.12, 0.0, 0.0], ["p", 0.03, 0.0, -0.5],
                  ["c5", 0.1, 0.5, -0.1]],
        "kind": 3,
        "adsr": [4.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, si apre la classifica dei migliori: un podio, onda triangolare, tre note staccate alte come i tre "
                        "gradini, il mi 5 del secondo a sinistra, il sol 5 del primo al centro e il do 5 del terzo a destra, più piano, "
                        "in 360 ms. Va bene per una classifica, un podio o un elenco di primati."),
    },
    "ciechino_statistiche": {
        "score": [["a5", 0.025, -0.6, -0.1], ["p", 0.025, 0.0, -0.5], ["b5", 0.025, -0.3, -0.1], ["p", 0.025, 0.0, -0.5],
                  ["c#6", 0.025, 0.0, -0.1], ["p", 0.025, 0.0, -0.5], ["e6", 0.07, 0.4, -0.05]],
        "kind": 1,
        "adsr": [4.0, 40.0, 30.0, 26.0],
        "descrizione": ("Ciechino, si aprono le statistiche: un contatore che scatta, tocchi di seno che salgono, la 5, si 5, do diesis "
                        "6 e un mi 6 più lungo, da sinistra a destra, in 220 ms. Va bene per dei conti, dei numeri che si sommano o un "
                        "resoconto di dati."),
    },
    "ciechino_volume": {
        "score": [["a5", 0.22, 0.0, [-0.35, 0.05]]],
        "kind": 1,
        "adsr": [5.0, 0.0, 100.0, 25.0],
        "descrizione": ("Ciechino, il volume cambiato dal menu Impostazioni: una sola nota di seno, la 5 al centro, che cresce dal "
                        "quasi silenzio al livello pieno, in 220 ms, così si sente il volume scelto. Va bene per provare un volume, un "
                        "livello o un guadagno."),
    },
    # La scelta della scheda audio (D20), con Ctrl+Maiusc+X: un selettore che gira e si ferma.
    "ciechino_scheda_audio": {
        "score": [["d6", 0.02, -0.6, -0.12], ["p", 0.025, 0.0, -0.5], ["e6", 0.02, -0.2, -0.12], ["p", 0.025, 0.0, -0.5],
                  ["f#6", 0.02, 0.2, -0.12], ["p", 0.025, 0.0, -0.5], ["g#6", 0.02, 0.6, -0.12], ["p", 0.04, 0.0, -0.5],
                  ["b5", 0.12, 0.0, -0.08]],
        "kind": 3,
        "adsr": [3.0, 35.0, 35.0, 30.0],
        "descrizione": ("Ciechino, si apre la scelta della scheda audio, con Ctrl+Maiusc+X o dal menu Impostazioni: un selettore che "
                        "gira e si ferma, onda triangolare, quattro scatti brevissimi che salgono per toni interi da sinistra a destra, "
                        "re 6, mi 6, fa diesis 6 e sol diesis 6, e un si 5 più lungo al centro, dove il selettore si ferma, in 315 ms. "
                        "Il suo seguito, quando la scheda scelta non si apre, è ciechino_scheda_non_aperta. Va bene per la scelta di "
                        "un'uscita audio, un collegamento o un selettore."),
    },
    # Gli Ascolti, il tutorial audio (D23): le cuffie che si provano, un orecchio, l'altro e il centro.
    "ciechino_ascolti": {
        "score": [["c6", 0.07, -0.8, -0.12], ["p", 0.05, 0.0, -0.5], ["c6", 0.07, 0.8, -0.12], ["p", 0.05, 0.0, -0.5], ["g5.c6", 0.18, 0.0, -0.1]],
        "kind": 1,
        "adsr": [5.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, si aprono gli Ascolti, il tutorial audio, dal menu Impostazioni, dal menu Aiuto o da soli all'avvio: le "
                        "cuffie che si provano, seno, un do 6 a sinistra e, dopo una breve pausa, lo stesso a destra, poi una scivolata che "
                        "sale dal sol 5 al do 6, al centro, in 420 ms. Va bene per una prova delle cuffie, un tutorial audio o un ascolto "
                        "guidato."),
    },
    "ciechino_proiettile": {
        "score": [["e5.b5", 0.12, -0.6, -0.1], ["b5.f#5", 0.16, 0.6, -0.12]],
        "kind": 1,
        "adsr": [5.0, 10.0, 80.0, 25.0],
        "descrizione": ("Ciechino, si apre la scelta della forma del proiettile, dal menu Impostazioni: una parabola in miniatura, seno, "
                        "una nota che sale dal mi 5 al si 5 partendo da sinistra e ricade al fa diesis 5 arrivando a destra, come un "
                        "colpo in volo, in 280 ms. Va bene per una traiettoria, un lancio o la scelta di un'arma."),
    },
    "ciechino_impostazioni": {
        "score": [["f5", 0.03, -0.2, -0.1], ["p", 0.025, 0.0, -0.5], ["c6", 0.03, 0.2, -0.05], ["g5", 0.05, 0.0, -0.08]],
        "kind": 3,
        "adsr": [3.0, 40.0, 40.0, 30.0],
        "descrizione": ("Ciechino, la lettura dei resoconti con la sintesi si accende, dal menu Impostazioni: un interruttore che scatta "
                        "e si assesta, onda triangolare, un tocco di fa 5 a sinistra, la quinta sopra, do 6, a destra, e un sol 5 al "
                        "centro, in 135 ms. Lo specchio è ciechino_sintesi_spenta. Va bene per un'opzione accesa, un interruttore o una "
                        "preferenza."),
    },
    "ciechino_sintesi_spenta": {
        "score": [["g5", 0.03, 0.2, -0.1], ["p", 0.025, 0.0, -0.5], ["c5", 0.03, -0.2, -0.05], ["f5", 0.05, 0.0, -0.08]],
        "kind": 3,
        "adsr": [3.0, 40.0, 40.0, 30.0],
        "descrizione": ("Ciechino, la lettura dei resoconti con la sintesi si spegne, dal menu Impostazioni: l'interruttore di "
                        "ciechino_impostazioni al contrario, onda triangolare, un tocco di sol 5 a destra, la quinta sotto, do 5, a "
                        "sinistra, e un fa 5 al centro, in 135 ms. Va bene per un'opzione spenta, un interruttore o una preferenza tolta."),
    },
    "ciechino_lingua": {
        "score": [["g5", 0.05, -0.5, -0.12], ["c6", 0.08, -0.5, -0.12], ["p", 0.04, 0.0, -0.5], ["a5", 0.05, 0.5, -0.12], ["d6", 0.08, 0.5, -0.12]],
        "kind": 1,
        "adsr": [5.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, si apre la scelta della lingua, dal menu Impostazioni: due voci che si salutano, seno, sol 5 e do 6 a "
                        "sinistra e, dopo una breve pausa, la risposta un tono più su, la 5 e re 6, a destra, in 300 ms. Va bene per una "
                        "lingua, un dialogo fra due o un saluto."),
    },
    "ciechino_lingua_scelta": {
        "score": [["d5", 0.06, 0.0, -0.08], ["b5", 0.06, 0.0, -0.08], ["p", 0.03, 0.0, -0.5], ["g5.a5", 0.2, 0.0, -0.08]],
        "kind": 3,
        "adsr": [4.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, si sceglie una lingua nuova, che vale dal prossimo avvio: onda triangolare, il salto di sesta dal re 5 "
                        "al si 5 e, dopo una breve pausa, un sol 5 che sale piano al la 5, come una promessa, al centro, in 350 ms. Va "
                        "bene per una scelta che vale più tardi, un cambio annunciato o un appuntamento."),
    },
    "ciechino_guida": {
        "score": [["c5.g5", 0.1, -0.2, -0.1], ["p", 0.04, 0.0, -0.5], ["c6", 0.06, 0.2, -0.08], ["g5", 0.14, 0.0, -0.08]],
        "kind": 1,
        "adsr": [5.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, F1, la guida ai comandi: una domanda e la sua risposta, seno, una scivolata che sale dal do 5 al sol 5 "
                        "come una domanda, poi un do 6 e un sol 5 che rispondono sicuri, in 340 ms. Va bene per un aiuto, una guida o "
                        "un elenco di istruzioni."),
    },
    "ciechino_manuale": {
        "score": [["1500-6000.2500-9000", 0.11, -0.4, 0.2], ["p", 0.06, 0.0, -0.5], ["1500-6000.2500-9000", 0.11, 0.4, 0.1]],
        "kind": 6,
        "adsr": [45.0, 0.0, 100.0, 50.0],
        "descrizione": ("Ciechino, Maiusc+F1, il manuale: due fruscii di pagina che si gira, rumore rosa che nasce piano e si schiarisce "
                        "con la banda da 1500-6000 a 2500-9000 Hz, il primo a sinistra e il secondo, più piano, a destra, in 280 ms. Va "
                        "bene per un libro, un documento che si apre o una pagina che si volta."),
    },
    "ciechino_novita": {
        "score": [["e5", 0.05, -0.3, -0.05], ["e6", 0.05, 0.0, -0.05], ["b5.f#6", 0.16, 0.3, -0.05]],
        "kind": 3,
        "adsr": [4.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, F2, le novità della versione: onda triangolare, un salto d'ottava dal mi 5 al mi 6, poi una scivolata "
                        "che riparte dal si 5 e sale al fa diesis 6, da sinistra a destra, in 260 ms, come qualcosa di nuovo che si "
                        "affaccia. Va bene per delle novità, un aggiornamento o una scoperta."),
    },
    "ciechino_informazioni": {
        "score": [["e5", 0.04, -0.2, -0.15], ["b5", 0.04, -0.1, -0.15], ["a5", 0.05, 0.0, -0.12], ["e6", 0.14, 0.1, -0.12]],
        "kind": 1,
        "adsr": [3.0, 20.0, 60.0, 30.0],
        "descrizione": ("Ciechino, F3, le informazioni sul programma, con nome, versione e autori: il biglietto da visita, il motivo di "
                        "ciechino_avvio in miniatura e in seno, mi 5, si 5, la 5 e un mi 6 breve, quasi al centro, in 270 ms. Va bene "
                        "per dei crediti, una firma o una presentazione."),
    },
    "ciechino_file_mancante": {
        "score": [["e5", 0.08, 0.0, -0.12], ["p", 0.1, 0.0, -0.5], ["b4", 0.08, 0.0, -0.12], ["p", 0.06, 0.0, -0.5],
                  ["b4.a4", 0.12, 0.0, [-0.15, -0.3]]],
        "kind": 3,
        "adsr": [4.0, 30.0, 50.0, 30.0],
        "descrizione": ("Ciechino, il manuale o il file delle novità non c'è, o non si apre: una pagina vuota, onda triangolare, un mi 5 "
                        "e, dopo un silenzio, un si 4 che lo ripete e un si 4 che cede al la 4 spegnendosi, al centro, in 440 ms. Va bene "
                        "per un file che manca, una ricerca a vuoto o un documento introvabile."),
    },
    "ciechino_pub": {
        "score": [["c#7", 0.1, -0.3, -0.12], ["p", 0.03, 0.0, -0.5], ["e7", 0.16, 0.3, -0.15]],
        "kind": 1,
        "adsr": [1.0, 70.0, 10.0, 29.0],
        "descrizione": ("Ciechino, Entra al PUB, nel menu Aiuto: un brindisi, due bicchieri che si toccano, seno acuto che si spegne "
                        "subito, do diesis 7 a sinistra e mi 7 a destra, in 290 ms. Chiedi aiuto al PUB ha il suo, ciechino_pub_aiuto. "
                        "Va bene per una compagnia, un ritrovo o un saluto fra amici."),
    },
    "ciechino_caffe_paypal": {
        "score": [["e7", 0.03, 0.0, -0.15], ["b6", 0.03, 0.0, -0.18], ["e7", 0.03, 0.0, -0.2], ["p", 0.03, 0.0, -0.5], ["g#6.e6", 0.15, 0.0, -0.15]],
        "kind": 1,
        "adsr": [1.0, 50.0, 30.0, 30.0],
        "descrizione": ("Ciechino, Dona con PayPal, nell'invito al caffè: una moneta che tintinna e cade nel salvadanaio, seno acuto, mi 7, "
                        "si 6 e mi 7, poi un sol diesis 6 che scivola al mi 6, al centro, in 270 ms. Va bene per una donazione, un "
                        "pagamento o un grazie."),
    },
    "ciechino_pub_aiuto": {
        "score": [["a6", 0.12, 0.2, -0.1], ["p", 0.06, 0.0, -0.5], ["a6", 0.3, 0.2, -0.12]],
        "kind": 1,
        "adsr": [1.0, 70.0, 10.0, 29.0],
        "descrizione": ("Ciechino, Chiedi aiuto al PUB, nel menu Aiuto: il campanello del banco, suonato due volte da chi chiede, seno "
                        "acuto che si spegne subito, due la 6, il secondo più lungo, appena a destra, in 480 ms. Entra al PUB ha il suo, "
                        "ciechino_pub. Va bene per una richiesta d'aiuto, un servizio o un richiamo garbato."),
    },
    # La finestra, i dialoghi.
    "ciechino_domanda": {
        "score": [["g4", 0.06, 0.0, -0.1], ["p", 0.02, 0.0, -0.5], ["d5.g5", 0.14, 0.0, -0.1]],
        "kind": 3,
        "adsr": [4.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, una domanda col sì e col no, come l'uscita con una partita in corso: onda triangolare, un sol 4 e una "
                        "scivolata che sale dal re 5 al sol 5, come la voce che si alza alla fine di una domanda, al centro, in 220 ms. "
                        "Va bene per una domanda, una conferma da dare o un dubbio."),
    },
    "ciechino_confermato": {
        "score": [["c6", 0.04, 0.0, -0.12], ["e6", 0.09, 0.0, -0.12]],
        "kind": 1,
        "adsr": [3.0, 25.0, 60.0, 30.0],
        "descrizione": ("Ciechino, un dialogo confermato con OK: un sì garbato, seno, do 6 e mi 6, la terza maggiore che sale, al "
                        "centro, in 130 ms. Il suo rovescio è ciechino_annullato. Va bene per una conferma, un'operazione accettata o un "
                        "assenso."),
    },
    "ciechino_annullato": {
        "score": [["e6", 0.04, 0.0, -0.15], ["c#6", 0.1, 0.0, -0.15]],
        "kind": 1,
        "adsr": [3.0, 25.0, 60.0, 30.0],
        "descrizione": ("Ciechino, un dialogo chiuso con Annulla o con Esc, o il no a una domanda: un no garbato, seno, mi 6 e do diesis "
                        "6, la terza minore che scende, al centro, in 140 ms. Il suo rovescio è ciechino_confermato. Va bene per un "
                        "annullamento, un ripensamento o un rifiuto gentile."),
    },
    "ciechino_campo_da_correggere": {
        "score": [["f#4", 0.05, 0.0, -0.2], ["p", 0.03, 0.0, -0.5], ["f#4", 0.05, 0.0, -0.2]],
        "kind": 4,
        "adsr": [3.0, 30.0, 50.0, 30.0],
        "descrizione": ("Ciechino, OK in un dialogo con un campo da correggere, come un nome vuoto in classifica: due tocchi ronzanti "
                        "di dente di sega sul fa diesis 4, piano, al centro, in 130 ms, come un dito che indica il campo. Va bene per "
                        "un dato da correggere, un campo vuoto o un valore non valido."),
    },
    # La finestra, l'aggiornamento automatico.
    "ciechino_aggiornamento_disponibile": {
        "score": [["c5", 0.05, -0.3, -0.1], ["e5", 0.05, -0.1, -0.1], ["g5", 0.05, 0.1, -0.1], ["c6", 0.05, 0.3, -0.1], ["p", 0.04, 0.0, -0.5],
                  ["e6.g6", 0.18, 0.0, -0.08]],
        "kind": 3,
        "adsr": [4.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, su GitHub c'è una versione nuova, e la proposta si apre: onda triangolare, l'arpeggio di do maggiore che "
                        "sale da sinistra a destra, do 5, mi 5, sol 5 e do 6, e dopo una breve pausa un mi 6 che sale al sol 6, al "
                        "centro, in 420 ms. Va bene per una novità che arriva, un pacco in consegna o un'offerta."),
    },
    "ciechino_aggiornamento_scelto": {
        "score": [["c5.c6", 0.12, 0.0, -0.12], ["c6", 0.1, 0.0, -0.12]],
        "kind": 1,
        "adsr": [4.0, 15.0, 70.0, 30.0],
        "descrizione": ("Ciechino, Aggiorna, nella proposta dell'aggiornamento: lo scaricamento che parte, seno, una scivolata d'ottava "
                        "dal do 5 al do 6 che si ferma lassù, al centro, in 220 ms. Va bene per un via, uno scaricamento o qualcosa che "
                        "prende velocità."),
    },
    "ciechino_aggiornamento_rimandato": {
        "score": [["g5", 0.07, 0.0, -0.12], ["p", 0.05, 0.0, -0.5], ["d5", 0.14, 0.0, -0.15]],
        "kind": 3,
        "adsr": [5.0, 20.0, 65.0, 35.0],
        "descrizione": ("Ciechino, Non adesso, nella proposta dell'aggiornamento: un più tardi tranquillo, onda triangolare, un sol 5 e, "
                        "dopo una breve pausa, un re 5 più lungo e più piano, al centro, in 260 ms. Va bene per un rinvio, una proposta "
                        "messa da parte o un a dopo."),
    },
    "ciechino_aggiornamento_scaduto": {
        "score": [["e6", 0.02, -0.3, -0.15], ["p", 0.08, 0.0, -0.5], ["b5", 0.02, 0.3, -0.15], ["p", 0.08, 0.0, -0.5], ["e6", 0.02, -0.3, -0.15],
                  ["p", 0.08, 0.0, -0.5], ["e5.b4", 0.2, 0.0, -0.15]],
        "kind": 1,
        "adsr": [1.0, 40.0, 40.0, 19.0],
        "descrizione": ("Ciechino, la proposta dell'aggiornamento resta senza risposta e si chiude da sola: un orologio, seno, tre "
                        "tic-tac di mi 6 e si 5 fra sinistra e destra, poi un mi 5 che scivola giù al si 4, al centro, in 500 ms. Va bene "
                        "per un tempo scaduto, un'attesa finita o un'occasione lasciata passare."),
    },
    "ciechino_aggiornamento_pronto": {
        "score": [["g5", 0.06, 0.0, -0.08], ["b5", 0.06, 0.0, -0.08], ["d6", 0.06, 0.0, -0.08], ["c6", 0.3, 0.0, -0.05]],
        "kind": 3,
        "adsr": [4.0, 15.0, 70.0, 30.0],
        "descrizione": ("Ciechino, l'aggiornamento è scaricato, e il programma si chiude per installarlo e si riapre da solo: onda "
                        "triangolare, una cadenza che arriva a casa, sol 5, si 5 e re 6, e un do 6 lungo che risolve, al centro, in 480 "
                        "ms. Il suo cugino sospeso è ciechino_aggiornamento_non_pronto. Va bene per un lavoro finito, un pacco arrivato o "
                        "una chiusura soddisfatta."),
    },
    "ciechino_aggiornamento_non_pronto": {
        "score": [["g5", 0.06, 0.0, -0.1], ["b5", 0.06, 0.0, -0.1], ["d6", 0.06, 0.0, -0.1], ["p", 0.05, 0.0, -0.5],
                  ["d6.c#6", 0.2, 0.0, [-0.1, -0.25]]],
        "kind": 3,
        "adsr": [4.0, 15.0, 70.0, 30.0],
        "descrizione": ("Ciechino, la versione nuova c'è, ma il suo pacchetto non è ancora pronto: la cadenza di "
                        "ciechino_aggiornamento_pronto che resta sospesa, onda triangolare, sol 5, si 5, re 6 e, dopo una pausa, un re 6 "
                        "che cala al do diesis 6 spegnendosi, senza risolvere, al centro, in 430 ms. Va bene per un'attesa, un ancora un "
                        "po' o una cosa non finita."),
    },
    "ciechino_aggiornamento_non_riuscito": {
        "score": [["c5.g5", 0.15, 0.0, -0.18], ["p", 0.03, 0.0, -0.5], ["f#4.c4", 0.25, 0.0, -0.18]],
        "kind": 4,
        "adsr": [3.0, 20.0, 65.0, 30.0],
        "descrizione": ("Ciechino, l'aggiornamento non si è scaricato, o non si è installato: uno scaricamento che si spezza, dente di "
                        "sega, una scivolata che sale dal do 5 al sol 5 e, dopo un attimo, una che crolla dal fa diesis 4 al do 4, al "
                        "centro, in 430 ms. Va bene per un trasferimento fallito, un collegamento caduto o un tentativo andato male."),
    },
    # La finestra, il fuoco.
    # Come in tutto il parco, F5 è il tasto grave e F7 quello acuto: spostamento_f5 e spostamento_f7.
    "ciechino_vai_alla_vista": {
        "score": [["c5.f4", 0.05, 0.0, -0.15]],
        "kind": 1,
        "adsr": [20.0, 0.0, 100.0, 40.0],
        "descrizione": ("Ciechino, F5, il fuoco va nella vista della partita: una scivolata breve e morbida di seno che scende dal do 5 al "
                        "fa 4, al centro, in 50 ms; grave, come F5 negli altri programmi del parco. Lo specchio è ciechino_vai_alla_barra. "
                        "Va bene per uno spostamento del fuoco verso l'area principale."),
    },
    "ciechino_vai_alla_barra": {
        "score": [["f5.c6", 0.05, 0.0, -0.25]],
        "kind": 1,
        "adsr": [20.0, 0.0, 100.0, 40.0],
        "descrizione": ("Ciechino, F7, il fuoco va nella barra di stato: una scivolata breve e morbida di seno che sale dal fa 5 al do 6, "
                        "al centro, in 50 ms; acuta, come F7 negli altri programmi del parco. Lo specchio è ciechino_vai_alla_vista. Va "
                        "bene per uno spostamento del fuoco verso una barra o un'area secondaria."),
    },
}


@dataclass
class Esito:
    """Che cosa lo strumento ha fatto, o farebbe con --prova, su una collezione."""

    percorso: str
    prova: bool = False
    # I preset aggiunti, quelli sostituiti con --sostituisci e quelli già presenti ma diversi dalla
    # tabella, lasciati come sono.
    aggiunti: list = field(default_factory=list)
    sostituiti: list = field(default_factory=list)
    diversi: list = field(default_factory=list)
    # I preset di suoni.EVENTI che ricevono la firma di Ciechino, e quelli che non si possono
    # firmare perché la collezione non li ha.
    firmati: list = field(default_factory=list)
    mancanti: list = field(default_factory=list)
    # I problemi della tabella, che fermano la scrittura, e l'errore che l'ha impedita.
    problemi: list = field(default_factory=list)
    errore: str | None = None
    scritto: bool = False


def percorso_condiviso():
    """La collezione condivisa, accanto a GBUtils.py: è quella che Acusticator usa da sorgente
    quando un progetto non ne registra una sua, e Ciechino non ne registra."""
    import GBUtils

    return os.path.join(os.path.dirname(os.path.abspath(GBUtils.__file__)), Acusticator.NOME_COLLEZIONE)


def nomi_da_firmare():
    """Tutti i preset di suoni.EVENTI, donazione compresa, ciascuno una volta, nell'ordine dei gruppi."""
    return list(dict.fromkeys(suoni.EVENTI.values()))


def campi(nome):
    """Il preset della tabella nella forma del file: descrizione, score a quartine, kind e adsr, nell'ordine dei campi della
    collezione. Liste nuove a ogni chiamata, così chi le riceve può scriverle senza toccare la tabella."""
    voce = PRESET[nome]
    score = [[nota, durata, panorama, list(volume) if isinstance(volume, (list, tuple)) else volume] for nota, durata, panorama, volume in voce["score"]]
    return {"descrizione": voce["descrizione"], "score": score, "kind": voce["kind"], "adsr": list(voce["adsr"])}


def durata(score):
    """La durata di uno score a quartine, pause comprese, in secondi."""
    return sum(float(quartina[1]) for quartina in score)


def firmato(preset):
    """Vero se il preset della collezione porta già la firma di Ciechino."""
    firme = preset.get("usato_da") if isinstance(preset, dict) else None
    return isinstance(firme, list) and APP in (str(a).strip().lower() for a in firme)


def _uguale(preset, nome):
    """Vero se il preset del file ha già descrizione, score, kind e adsr della tabella; le firme e gli altri campi non contano."""
    return isinstance(preset, dict) and all(preset.get(chiave) == valore for chiave, valore in campi(nome).items())


def controlla_tabella():
    """I problemi della tabella rispetto alle regole di Gabriele e al contratto degli eventi; vuota se va tutto bene."""
    problemi = []
    usati = {}
    for evento, preset in suoni.EVENTI.items():
        if preset in usati:
            problemi.append(f"il preset {preset} serve due eventi, {usati[preset]} e {evento}")
        usati[preset] = evento
        if evento == EVENTO_DEL_CAFFE:
            if preset != PRESET_DEL_CAFFE:
                problemi.append(f"l'evento {evento} deve usare {PRESET_DEL_CAFFE}, non {preset}")
        elif preset not in PRESET:
            problemi.append(f"l'evento {evento} usa {preset}, che non è nella tabella dei preset nuovi")
    for nome, voce in PRESET.items():
        if nome not in usati:
            problemi.append(f"il preset {nome} della tabella non serve nessun evento")
        if not nome.startswith(PREFISSO):
            problemi.append(f"il preset {nome} non ha il prefisso {PREFISSO}")
        if voce["kind"] == ONDA_QUADRA or voce["kind"] not in range(1, 9):
            problemi.append(f"il preset {nome} ha il kind {voce['kind']}: niente onda quadra, e soltanto da 1 a 8")
        if not voce["descrizione"].startswith(INIZIO_DESCRIZIONE):
            problemi.append(f"la descrizione di {nome} non comincia con {INIZIO_DESCRIZIONE!r}")
        adsr = voce["adsr"]
        if len(adsr) != 4 or not all(0 <= v <= 100 for v in adsr) or adsr[0] + adsr[1] + adsr[3] > 100:
            problemi.append(f"il preset {nome} ha un adsr non valido: {adsr}")
        if not voce["score"] or any(len(quartina) != 4 for quartina in voce["score"]):
            problemi.append(f"lo score di {nome} non è fatto di quartine")
    for evento in EVENTI_FREQUENTI:
        preset = suoni.EVENTI.get(evento)
        if preset in PRESET and durata(PRESET[preset]["score"]) >= DURATA_MASSIMA_FREQUENTI:
            problemi.append(f"l'evento frequente {evento} dura {durata(PRESET[preset]['score']):.3f} secondi, oltre {DURATA_MASSIMA_FREQUENTI}")
    return problemi


def _piano(dati, esito, sostituisci):
    """Riempie l'esito con ciò che c'è da fare sulla collezione dati, e la cambia di conseguenza.
    Vero se ha cambiato dei preset. Un preset nuovo va in coda; uno sostituito resta al suo posto,
    con le sue firme e i campi che lo strumento non conosce."""
    esito.aggiunti.clear()
    esito.sostituiti.clear()
    esito.diversi.clear()
    for nome in PRESET:
        presente = dati.get(nome)
        if presente is None:
            dati[nome] = campi(nome)
            esito.aggiunti.append(nome)
        elif not _uguale(presente, nome):
            if sostituisci and isinstance(presente, dict):
                presente.update(campi(nome))
                esito.sostituiti.append(nome)
            else:
                esito.diversi.append(nome)
    esito.firmati = [nome for nome in nomi_da_firmare() if nome in dati and not firmato(dati[nome])]
    esito.mancanti = [nome for nome in nomi_da_firmare() if nome not in dati]
    return bool(esito.aggiunti or esito.sostituiti)


def crea(percorso=None, prova=False, sostituisci=False):
    """Aggiunge alla collezione i preset della tabella che non ci sono, poi firma per Ciechino tutti i
    preset di suoni.EVENTI, donazione compresa. percorso None vuol dire la collezione condivisa di
    GBUtils. Con prova legge soltanto, e l'esito dice che cosa farebbe; con sostituisci rimette come
    nella tabella anche i preset che ci sono già ma sono diversi. Non scrive se la tabella ha dei
    problemi o se la collezione non c'è: aggiorna creerebbe una collezione nuova con i soli preset di
    Ciechino. Restituisce l'Esito."""
    percorso = os.path.abspath(percorso_condiviso() if percorso is None else percorso)
    esito = Esito(percorso=percorso, prova=prova)
    esito.problemi = controlla_tabella()
    if esito.problemi:
        esito.errore = "la tabella dei preset ha dei problemi: non scrivo niente"
        return esito
    if not os.path.isfile(percorso):
        esito.errore = f"la collezione {percorso} non c'è"
        return esito
    if prova:
        try:
            with open(percorso, encoding="utf-8") as file:
                dati = json.load(file)
        except (OSError, ValueError) as errore:
            esito.errore = f"la collezione non si legge: {errore}"
            return esito
        if not isinstance(dati, dict):
            esito.errore = "la collezione non è un dizionario di preset"
            return esito
        _piano(dati, esito, sostituisci)
        return esito
    cambiati = []

    def cambia(dati):
        cambiato = _piano(dati, esito, sostituisci)
        cambiati.append(cambiato)
        return cambiato

    if Acusticator.aggiorna(cambia, percorso) is None:
        esito.errore = "Acusticator.aggiorna non ha potuto scrivere la collezione: il motivo è su stderr"
        return esito
    esito.scritto = any(cambiati)
    if esito.firmati:
        firmati = Acusticator.firma(esito.firmati, APP, percorso)
        if firmati is None:
            esito.errore = "Acusticator.firma non ha potuto scrivere la collezione: il motivo è su stderr"
            return esito
        esito.scritto = esito.scritto or firmati > 0
    return esito


def _elenco(titolo, nomi):
    """Una riga di titolo col conto, poi un nome per riga, che con NVDA si scorre bene."""
    if not nomi:
        return []
    return [f"{titolo}: {len(nomi)}.", *nomi]


def righe(esito):
    """Il resoconto dell'esito, una voce per riga."""
    uscita = [f"Collezione: {esito.percorso}."]
    if esito.prova:
        uscita.append("Prova: non scrivo niente, dico soltanto che cosa farei.")
    uscita.extend(f"Problema: {problema}." for problema in esito.problemi)
    if esito.errore:
        uscita.append(f"Errore: {esito.errore}.")
        return uscita
    uscita.extend(_elenco("Preset nuovi da aggiungere" if esito.prova else "Preset nuovi aggiunti", esito.aggiunti))
    uscita.extend(_elenco("Preset da sostituire" if esito.prova else "Preset sostituiti", esito.sostituiti))
    uscita.extend(_elenco("Preset già presenti ma diversi dalla tabella, lasciati come sono (--sostituisci li rimette)", esito.diversi))
    uscita.extend(_elenco("Preset da firmare" if esito.prova else "Preset firmati", esito.firmati))
    uscita.extend(_elenco("Preset degli eventi che la collezione non ha, e che non si possono firmare", esito.mancanti))
    if not (esito.aggiunti or esito.sostituiti or esito.firmati):
        uscita.append("Niente da fare: la collezione ha già tutti i preset di Ciechino, firmati.")
    elif not esito.prova:
        uscita.append("Collezione scritta." if esito.scritto else "Collezione non scritta.")
    return uscita


def main(argomenti=None):
    parser = argparse.ArgumentParser(description="Aggiunge alla collezione dei suoni di GBUtils i preset nuovi di Ciechino e li firma.")
    parser.add_argument("--collezione", metavar="PERCORSO", help="la collezione su cui lavorare, per esempio una copia; di serie quella condivisa di GBUtils")
    parser.add_argument("--prova", action="store_true", help="non scrive niente, dice soltanto che cosa farebbe")
    parser.add_argument("--sostituisci", action="store_true", help="rimette come nella tabella anche i preset che ci sono già ma sono diversi")
    opzioni = parser.parse_args(argomenti)
    esito = crea(opzioni.collezione, prova=opzioni.prova, sostituisci=opzioni.sostituisci)
    for riga in righe(esito):
        print(riga)
    return 1 if esito.errore else 0


if __name__ == "__main__":
    sys.exit(main())
