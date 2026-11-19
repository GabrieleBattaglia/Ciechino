"""
Ciechino, il banco dei bersagli: le forme dell'LFO, le grandezze e le portanti, alla cieca.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di strumenti/banco_spazio.py di MESS, con le parti
comuni di strumenti/banco_mira.py.
Tre gruppi alla cieca, ciascuno col controllo. Le sette forme dell'LFO, tutte nello stesso punto e
della stessa grandezza, con la domanda di quanto ogni suono si distingue dagli altri e quanto è
gradevole: nel gioco ci sono tutte, a caso, quindi nessuna è quella attuale. Le grandezze di una
forma sola, dal raggio del primo livello a quello minimo, con la domanda di quanto sembra grande il
bersaglio: se i voti seguono il raggio, la grandezza si sente; anche qui nessuna è l'attuale, e il
riepilogo segue l'ordine dei raggi invece dei voti. Le portanti dei bersagli, sentite prima in basso
e poi in alto, perché il timbro cambia con l'altezza, con la domanda di quanto il suono è gradevole e
quanto fa capire l'altezza.
I bersagli sono quelli del gioco: Bersaglio di bersagli.py, col suo LFO e i suoi parametri sonori,
suonato da una VoceTono del motore audio di Ciechino, resa fuori tempo reale sul flusso in memoria
di banco_mira, coi blocchi, il tetto e la panoramica del motore vero; la voce entra e esce con la
rampa del guadagno, perché una lettera non cominci con un click. Il protocollo, il voto, il menu di
fine gruppo e la rivelazione sono quelli di banco_mira.
Gli esiti vanno nel file banco_bersagli_esiti.txt, accanto al banco, in coda, con la data e il seme.
Uso, dalla cartella del programma: python strumenti\\banco_bersagli.py; con un numero, per esempio
python strumenti\\banco_bersagli.py 4127, usa quel seme, e quindi le stesse lettere di una sessione passata.
"""

import sys
from pathlib import Path

import numpy as np

STRUMENTI = Path(__file__).resolve().parent
if str(STRUMENTI) not in sys.path:
    sys.path.insert(0, str(STRUMENTI))

import banco_mira as banco  # noqa: E402
from GBUtils import enter_escape  # noqa: E402

from bersagli import Bersaglio  # noqa: E402
from costanti import BERSAGLIO_FORMA, CUBO_LATO, CUBO_Y, CUBO_Z, FORME_LFO, FREQUENZA_LFO, FS, RAGGIO_INIZIALE  # noqa: E402
from lingua import _  # noqa: E402
from motore_audio import VoceTono  # noqa: E402
from testi import conta, metri  # noqa: E402

FILE_DEGLI_ESITI = STRUMENTI / "banco_bersagli_esiti.txt"
# Il punto delle forme e delle grandezze: davanti al cannone, a metà altezza del cubo e a un quarto
# della sua profondità, abbastanza vicino da sentirsi bene anche col raggio più grande tutto dentro.
PUNTO_DI_PROVA = (0.0, (CUBO_Y[0] + CUBO_Y[1]) / 2, CUBO_Z[0] + CUBO_LATO / 4)
# I due punti delle portanti, davanti al cannone alla stessa distanza: in basso e in alto, a un
# ottavo dal suolo e dal tetto del cubo, dove il bersaglio del primo livello sta ancora tutto dentro.
PUNTI_DELLE_PORTANTI = ((0.0, CUBO_Y[0] + CUBO_LATO / 8, PUNTO_DI_PROVA[2]), (0.0, CUBO_Y[1] - CUBO_LATO / 8, PUNTO_DI_PROVA[2]))
# Il raggio delle forme e delle portanti, quello del primo livello; le grandezze, in metri, scelte da
# Gabriele il 19 novembre 2026: oggi vanno dal raggio del primo livello a quello minimo.
RAGGIO_DI_PROVA = RAGGIO_INIZIALE
RAGGI_DI_PROVA = (5.0, 2.5, 1.0, 0.5)
# La forma dell'LFO delle grandezze e delle portanti: il seno, la più regolare, perché cambi soltanto
# ciò che il gruppo vuole far sentire.
FORMA_DI_PROVA = "seno"
# Le portanti da confrontare, l'attuale fra loro.
PORTANTI_DI_PROVA = ("organo", "seno", "triangolo", "sega_morbida")
# L'LFO a metà dell'intervallo del gioco, con la fase a zero; il seme conta per casuale e deriva, ed è
# fisso perché tutte le sessioni sentano gli stessi valori a caso.
FREQUENZA_LFO_DI_PROVA = (FREQUENZA_LFO[0] + FREQUENZA_LFO[1]) / 2
SEME_DEL_BERSAGLIO = 73
# Quanto suona un bersaglio: le forme e le grandezze, abbastanza cicli dell'LFO per riconoscerli; le
# portanti, ciascuno dei due punti, con una pausa fra l'uno e l'altro.
DURATA_DEL_BERSAGLIO = 4.0
DURATA_DI_UN_PUNTO = 3.0
PAUSA_FRA_I_PUNTI = 0.6


def rendi_bersaglio(forma, raggio, punto, portante, durata, fs=FS):
    """Il suono di un bersaglio fermo, come lo suonerebbe il motore del gioco: (campioni, 2) float32."""
    bersaglio = Bersaglio(numero=0, forma=forma, raggio=raggio, partenza=punto, frequenza_lfo=FREQUENZA_LFO_DI_PROVA, fase_lfo=0.0, seme=SEME_DEL_BERSAGLIO)
    campioni = round(durata * fs)
    with banco.in_memoria() as (motore, flusso):
        motore.aggiungi(banco.entra_ed_esce(VoceTono(bersaglio.parametri_sonori, portante, 0, guadagno=0.0), 0, campioni, fs))
        return flusso.rendi(campioni)


def componi(ricetta):
    """Il materiale di una versione: il bersaglio della ricetta in ognuno dei suoi punti, in fila,
    con una pausa fra l'uno e l'altro."""
    pezzi = []
    for punto in ricetta["punti"]:
        if pezzi:
            pezzi.append(np.zeros((round(PAUSA_FRA_I_PUNTI * FS), 2), dtype=np.float32))
        pezzi.append(rendi_bersaglio(ricetta["forma"], ricetta["raggio"], punto, ricetta["portante"], ricetta["durata"]))
    return np.concatenate(pezzi)


def _ricetta(forma, raggio, portante, punti, durata):
    return {"forma": forma, "raggio": raggio, "portante": portante, "punti": tuple(punti), "durata": durata}


def _forme():
    candidati = tuple(banco.Candidato(_("la forma {forma} dell'LFO").format(forma=forma),
                                      _ricetta(forma, RAGGIO_DI_PROVA, BERSAGLIO_FORMA, (PUNTO_DI_PROVA,), DURATA_DEL_BERSAGLIO)) for forma in FORME_LFO)
    return banco.Gruppo("forme", _("Bersagli, le forme"), _("quanto il suono si distingue dagli altri e quanto è gradevole"),
                        _("In ogni lettera senti un bersaglio fermo, sempre nello stesso punto e della stessa grandezza."), candidati)


def _grandezze():
    candidati = tuple(banco.Candidato(_("il raggio di {raggio}").format(raggio=metri(raggio)),
                                      _ricetta(FORMA_DI_PROVA, raggio, BERSAGLIO_FORMA, (PUNTO_DI_PROVA,), DURATA_DEL_BERSAGLIO)) for raggio in RAGGI_DI_PROVA)
    return banco.Gruppo("grandezze", _("Bersagli, la grandezza"), _("quanto ti sembra grande il bersaglio, da 1 il più piccolo a 5 il più grande"),
                        _("In ogni lettera senti un bersaglio fermo, sempre nello stesso punto e con lo stesso suono."), candidati, preferenza=False)


def _portanti():
    candidati = []
    for portante in PORTANTI_DI_PROVA:
        attuale = portante == BERSAGLIO_FORMA
        nome = _("la portante {portante}").format(portante=portante)
        if attuale:
            nome = _("{nome}, quella attuale").format(nome=nome)
        candidati.append(banco.Candidato(nome, _ricetta(FORMA_DI_PROVA, RAGGIO_DI_PROVA, portante, PUNTI_DELLE_PORTANTI, DURATA_DI_UN_PUNTO), attuale))
    return banco.Gruppo("portanti", _("Bersagli, la portante"), _("quanto il suono è gradevole e quanto ti fa capire l'altezza del bersaglio"),
                        _("In ogni lettera senti un bersaglio fermo, prima in basso e poi in alto."), tuple(candidati))


GRUPPI = (_forme(), _grandezze(), _portanti())


def prepara(seme, materiali=None):
    return banco.prepara_prove(GRUPPI, componi, seme, materiali)


def main(argv=None):
    seme = banco.leggi_seme(sys.argv[1:] if argv is None else argv, "banco_bersagli")
    if seme is None:
        return 1
    print(_("Preparo i suoni dei bersagli."))
    prove = prepara(seme)
    print(_("Banco dei bersagli di Ciechino: {gruppi} alla cieca.").format(gruppi=conta(len(prove), "{n} gruppo", "{n} gruppi")))
    print(_("In ogni gruppo senti la stessa cosa in più versioni, ciascuna con la sua lettera. Prima la rassegna, tutte le lettere una volta, "
            "senza domande; Escape la interrompe."))
    print(_("Poi il voto, una lettera alla volta: spazio fa partire la lettera e la ferma; il tasto di un'altra lettera fa sentire quella; da 1 a 5 "
            "dai il voto e passi alla seguente; Invio passa oltre; n vuol dire nessuna preferenza; Escape chiude il voto."))
    print(_("A fine gruppo: r ripete, c commenta, Invio lo dà per superato, Escape lo chiude senza giudizio, e allora i suoi voti non contano."))
    print(_("Gli esiti vanno nel file {file}, accanto al banco.").format(file=FILE_DEGLI_ESITI.name))
    if not enter_escape(banco.invito_a_cominciare()):
        return 0
    return banco.sessione(_("Banco dei bersagli"), seme, FILE_DEGLI_ESITI, prove)


if __name__ == "__main__":
    sys.exit(main())
