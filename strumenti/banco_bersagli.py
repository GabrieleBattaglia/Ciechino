"""
Ciechino, il banco dei bersagli: le forme dell'LFO, le grandezze e le portanti, alla cieca.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di strumenti/banco_spazio.py di MESS, con le parti
comuni di strumenti/banco_mira.py.
Tre prove alla cieca, divise in gruppi, ciascuno col controllo. Le forme dell'LFO, tutte nello stesso punto, della
stessa grandezza e con la stessa portante, con la domanda di quanto ogni suono si distingue dagli
altri e quanto è gradevole: nel gioco ci sono tutte, a caso, quindi nessuna è quella attuale. Ogni
forma suona coi suoi parametri estratti a caso come nel gioco (D18), da un seme fisso, così tutte le
sessioni sentono gli stessi, e il nome che si rivela alla fine li dice. Le forme sono più delle
lettere di un gruppo, che ne tiene una di meno, per il controllo: si dividono in più gruppi, nell'ordine
di FORME_LFO, e il primo tiene le sette della 0.1.0. Le grandezze di una forma sola, dal raggio del
primo livello a quello minimo, con la domanda di quanto sembra grande il bersaglio: se i voti seguono
il raggio, la grandezza si sente; anche qui nessuna è l'attuale, e il riepilogo segue l'ordine dei
raggi invece dei voti. Le portanti dei bersagli, sentite prima in basso e poi in alto, perché il
timbro cambia con l'altezza, con la domanda di quanto il suono è gradevole e quanto fa capire
l'altezza: dal 19 novembre 2026 ogni bersaglio estrae la sua fra tutte quelle di BERSAGLIO_PORTANTI
(D18), quindi sono tutte in gioco e nessuna è l'attuale; una che prendesse voti bassi si toglierebbe
dall'elenco. Grandezze e portanti usano la forma più regolare coi parametri di serie, perché cambi
soltanto ciò che il gruppo vuole far sentire.
I bersagli sono quelli del gioco: Bersaglio di bersagli.py, col suo LFO e i suoi parametri sonori,
suonato da una VoceTono del motore audio di Ciechino, resa fuori tempo reale sul flusso in memoria
di banco_mira, coi blocchi, il tetto e la panoramica del motore vero; la voce entra e esce con la
rampa del guadagno, perché una lettera non cominci con un click. Il protocollo, il voto, il menu di
fine gruppo e la rivelazione sono quelli di banco_mira.
Gli esiti vanno nel file banco_bersagli_esiti.txt, accanto al banco, in coda, con la data e il seme.
Uso, dalla cartella del programma: python strumenti\\banco_bersagli.py; con un numero, per esempio
python strumenti\\banco_bersagli.py 4127, usa quel seme, e quindi le stesse lettere di una sessione passata.
"""

import random
import sys
from pathlib import Path

import numpy as np

STRUMENTI = Path(__file__).resolve().parent
if str(STRUMENTI) not in sys.path:
    sys.path.insert(0, str(STRUMENTI))

import banco_mira as banco  # noqa: E402
from GBUtils import enter_escape  # noqa: E402

from bersagli import PARAMETRI_LFO, Bersaglio, parametri_a_caso  # noqa: E402
from costanti import BERSAGLIO_FORMA, BERSAGLIO_PORTANTI, CUBO_LATO, CUBO_Y, CUBO_Z, FORME_LFO, FREQUENZA_LFO, FS, RAGGIO_INIZIALE  # noqa: E402
from lingua import _  # noqa: E402
from motore_audio import VoceTono  # noqa: E402
from testi import conta, elenco, metri, numero  # noqa: E402

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
# La forma dell'LFO delle grandezze e delle portanti: il seno, la più regolare, coi parametri di
# serie, perché cambi soltanto ciò che il gruppo vuole far sentire.
FORMA_DI_PROVA = "seno"
# Le portanti da confrontare: tutte quelle in gioco, che i bersagli si estraggono a caso (D18).
PORTANTI_DI_PROVA = BERSAGLIO_PORTANTI
# L'LFO a metà dell'intervallo del gioco, con la fase a zero; il seme conta per casuale e deriva, ed è
# fisso perché tutte le sessioni sentano gli stessi valori a caso.
FREQUENZA_LFO_DI_PROVA = (FREQUENZA_LFO[0] + FREQUENZA_LFO[1]) / 2
SEME_DEL_BERSAGLIO = 73
# Il seme dei parametri a caso delle forme, fisso per lo stesso motivo: le forme li estraggono da un
# solo generatore, nell'ordine di FORME_LFO, come farebbe un bersaglio del gioco.
SEME_DEI_PARAMETRI = 18
# Quante versioni stanno in un gruppo: le lettere di banco_mira, meno quella del controllo.
VERSIONI_PER_GRUPPO = len(banco.LETTERE) - 1
# Quanto suona un bersaglio: le forme e le grandezze, abbastanza cicli dell'LFO per riconoscerli; le
# portanti, ciascuno dei due punti, con una pausa fra l'uno e l'altro.
DURATA_DEL_BERSAGLIO = 4.0
DURATA_DI_UN_PUNTO = 3.0
PAUSA_FRA_I_PUNTI = 0.6


def parametri_di_prova():
    """I parametri di ogni forma nel gruppo delle forme, estratti come nel gioco da un generatore col
    seme SEME_DEI_PARAMETRI, nell'ordine di FORME_LFO: sempre gli stessi."""
    rng = random.Random(SEME_DEI_PARAMETRI)
    return {forma: parametri_a_caso(forma, rng) for forma in FORME_LFO}


def rendi_bersaglio(forma, raggio, punto, portante, durata, parametri=None, fs=FS):
    """Il suono di un bersaglio fermo, come lo suonerebbe il motore del gioco: (campioni, 2) float32.
    parametri sono quelli della forma; None vuol dire quelli di serie."""
    bersaglio = Bersaglio(numero=0, forma=forma, raggio=raggio, partenza=punto, frequenza_lfo=FREQUENZA_LFO_DI_PROVA, fase_lfo=0.0, seme=SEME_DEL_BERSAGLIO,
                          portante=portante, parametri_lfo=parametri)
    campioni = round(durata * fs)
    with banco.in_memoria() as (motore, flusso):
        motore.aggiungi(banco.entra_ed_esce(VoceTono(bersaglio.parametri_sonori, bersaglio.portante, 0, guadagno=0.0), 0, campioni, fs))
        return flusso.rendi(campioni)


def componi(ricetta):
    """Il materiale di una versione: il bersaglio della ricetta in ognuno dei suoi punti, in fila,
    con una pausa fra l'uno e l'altro."""
    pezzi = []
    for punto in ricetta["punti"]:
        if pezzi:
            pezzi.append(np.zeros((round(PAUSA_FRA_I_PUNTI * FS), 2), dtype=np.float32))
        pezzi.append(rendi_bersaglio(ricetta["forma"], ricetta["raggio"], punto, ricetta["portante"], ricetta["durata"], ricetta["parametri"]))
    return np.concatenate(pezzi)


def _ricetta(forma, raggio, portante, punti, durata, parametri=None):
    return {"forma": forma, "raggio": raggio, "portante": portante, "punti": tuple(punti), "durata": durata, "parametri": parametri}


def _valore(valore):
    if isinstance(valore, int):
        return str(valore)
    return numero(valore, 2)


def nome_della_forma(forma, parametri):
    """Il nome che si rivela alla fine: la forma e i suoi parametri, con i numeri all'italiana."""
    detti = [f"{parametro.nome} {_valore(parametri[parametro.nome])}" for parametro in PARAMETRI_LFO[forma]]
    # Traduttori: forma è il nome di una forma dell'oscillazione, come trillo; parametri l'elenco dei suoi numeri.
    return _("la forma {forma} dell'LFO, con {parametri}").format(forma=forma, parametri=elenco(detti))


def _forme():
    """I gruppi delle forme, al massimo VERSIONI_PER_GRUPPO forme ciascuno, nell'ordine di FORME_LFO."""
    parametri = parametri_di_prova()
    parti = [FORME_LFO[inizio:inizio + VERSIONI_PER_GRUPPO] for inizio in range(0, len(FORME_LFO), VERSIONI_PER_GRUPPO)]
    gruppi = []
    for posto, forme in enumerate(parti, start=1):
        candidati = tuple(banco.Candidato(nome_della_forma(forma, parametri[forma]),
                                          _ricetta(forma, RAGGIO_DI_PROVA, BERSAGLIO_FORMA, (PUNTO_DI_PROVA,), DURATA_DEL_BERSAGLIO, parametri[forma]))
                          for forma in forme)
        if len(parti) == 1:
            chiave, titolo = "forme", _("Bersagli, le forme")
        else:
            chiave = "forme" if posto == 1 else f"forme_{posto}"
            # Traduttori: parte e parti sono numeri: le forme sono divise in più gruppi d'ascolto.
            titolo = _("Bersagli, le forme, parte {parte} di {parti}").format(parte=posto, parti=len(parti))
        gruppi.append(banco.Gruppo(chiave, titolo, _("quanto il suono si distingue dagli altri e quanto è gradevole"),
                                   _("In ogni lettera senti un bersaglio fermo, sempre nello stesso punto, della stessa grandezza e con lo stesso timbro; "
                                     "cambia soltanto il modo in cui il suono oscilla."), candidati))
    return tuple(gruppi)


def _grandezze():
    candidati = tuple(banco.Candidato(_("il raggio di {raggio}").format(raggio=metri(raggio)),
                                      _ricetta(FORMA_DI_PROVA, raggio, BERSAGLIO_FORMA, (PUNTO_DI_PROVA,), DURATA_DEL_BERSAGLIO)) for raggio in RAGGI_DI_PROVA)
    return banco.Gruppo("grandezze", _("Bersagli, la grandezza"), _("quanto ti sembra grande il bersaglio, da 1 il più piccolo a 5 il più grande"),
                        _("In ogni lettera senti un bersaglio fermo, sempre nello stesso punto e con lo stesso suono."), candidati, preferenza=False)


def _portanti():
    candidati = tuple(banco.Candidato(_("la portante {portante}").format(portante=portante),
                                      _ricetta(FORMA_DI_PROVA, RAGGIO_DI_PROVA, portante, PUNTI_DELLE_PORTANTI, DURATA_DI_UN_PUNTO))
                      for portante in PORTANTI_DI_PROVA)
    return banco.Gruppo("portanti", _("Bersagli, la portante"), _("quanto il suono è gradevole e quanto ti fa capire l'altezza del bersaglio"),
                        _("In ogni lettera senti un bersaglio fermo, prima in basso e poi in alto."), candidati)


GRUPPI = (*_forme(), _grandezze(), _portanti())


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
