"""
Ciechino, il banco del volo: il proiettile in volo, la caduta del volume con la distanza e gli effetti del tiro.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di strumenti/banco_spazio.py di MESS, con le parti
comuni di strumenti/banco_mira.py.
In ogni lettera dei gruppi alla cieca si sentono quattro tiri contro lo stesso bersaglio fermo,
sempre in quest'ordine: un centro, uno corto, che cade davanti al bersaglio, uno lungo, che gli passa
sopra e cade oltre, e uno largo, che gli passa accanto a destra. La domanda è quanto si capisce dove
va il colpo. Il primo gruppo confronta il timbro del proiettile, il secondo la caduta del volume con
la distanza, DB_PER_METRO: per sentirne un'altra, il banco cambia la costante dentro fisica.py, il
modulo che la legge a ogni suono dello spazio, e la rimette sempre com'era, anche se qualcosa va
storto. In fondo un gruppo d'ascolto, non alla cieca, degli effetti del tiro, lo sparo, il bersaglio
colpito, il bonus colpito e il colpo a terra, ciascuno annunciato e sentito dal punto dove nasce in
partita, col commento libero.
I tiri sono quelli del gioco: i tre valori di fisica.soluzione, la traiettoria di fisica.Traiettoria
e il volo di volo.calcola_volo, provati alla preparazione, perché ogni tiro faccia davvero ciò che il
suo nome promette. Il suono è reso fuori tempo reale dal motore audio di Ciechino sul flusso in
memoria di banco_mira, come in partita: il bersaglio con la sua voce, lo sparo, la voce del
proiettile, e all'istante dell'impatto un evento programmato, al campione esatto, che ferma il
proiettile, fa partire l'effetto già pronto, colpito o a terra, collocato nello spazio, e se è un
centro ferma il bersaglio. Il protocollo, il voto, il menu di fine gruppo e la rivelazione sono
quelli di banco_mira.
Gli esiti vanno nel file banco_volo_esiti.txt, accanto al banco, in coda, con la data e il seme.
Uso, dalla cartella del programma: python strumenti\\banco_volo.py; con un numero, per esempio
python strumenti\\banco_volo.py 4127, usa quel seme, e quindi le stesse lettere di una sessione passata.
"""

import contextlib
import functools
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

STRUMENTI = Path(__file__).resolve().parent
if str(STRUMENTI) not in sys.path:
    sys.path.insert(0, str(STRUMENTI))

import banco_mira as banco  # noqa: E402
import collaudo_comune as comune  # noqa: E402
from GBUtils import enter_escape  # noqa: E402

import fisica  # noqa: E402
import suoni  # noqa: E402
from bersagli import Bersaglio, raggio_del_livello  # noqa: E402
from costanti import BERSAGLIO_FORMA, DB_PER_METRO, FREQUENZA_LFO, FS, PROIETTILE_FORMA, VELOCITA_DEL_TEMPO  # noqa: E402
from lingua import _  # noqa: E402
from motore_audio import VoceTono  # noqa: E402
from testi import conta, elenco, numero  # noqa: E402
from volo import calcola_volo, parametri_del_proiettile  # noqa: E402

FILE_DEGLI_ESITI = STRUMENTI / "banco_volo_esiti.txt"
# Il bersaglio fermo dei tiri: davanti al cannone, a una dozzina di metri d'altezza, nella prima
# metà del cubo; del primo livello, quello che Gabriele sentirà per primo giocando.
PUNTO_DEL_BERSAGLIO = (0.0, 12.0, 50.0)
LIVELLO_DEL_BERSAGLIO = 1
FORMA_DEL_BERSAGLIO = "seno"
FREQUENZA_LFO_DEL_BERSAGLIO = (FREQUENZA_LFO[0] + FREQUENZA_LFO[1]) / 2
SEME_DEL_BERSAGLIO = 73
# I tiri mancati passano a tanti metri dalla superficie del bersaglio: abbastanza da essere errori
# chiari, abbastanza poco da essere errori veri. Il lungo si cerca alzando il punto di mira a passi,
# finché non passa sopra senza toccare.
SCARTO_DEI_MANCATI = 4.0
PASSO_DEL_LUNGO = 1.0
PASSI_DEL_LUNGO = 30
TIRI = ("centro", "corto", "lungo", "largo")
# Il tempo di un tiro: il bersaglio da solo prima dello sparo, perché l'orecchio lo trovi; dopo
# l'impatto, l'effetto e almeno la coda; fra un tiro e l'altro una pausa.
ASCOLTO_PRIMA_DEL_TIRO = 1.5
CODA_DEL_TIRO = 1.0
PAUSA_FRA_I_TIRI = 1.0
# Le versioni da confrontare, date da Gabriele il 19 novembre 2026, l'attuale fra loro.
FORME_DEL_PROIETTILE = ("sega_morbida", "triangolo", "seno")
DB_PER_METRO_RIPIDO = 0.4
EFFETTI = ("sparo", "colpito", "colpito_bonus", "a_terra")


class Tiro(NamedTuple):
    """Un tiro di prova: la chiave, i tre valori delle scivolate, la traiettoria e il volo contro il
    bersaglio di prova, sparato al tempo di gioco zero."""

    chiave: str
    valori: tuple
    traiettoria: fisica.Traiettoria
    volo: object


@contextlib.contextmanager
def caduta(db_per_metro):
    """Per la durata del blocco, la caduta del volume con la distanza vale db_per_metro decibel per
    metro, dentro fisica.py, che la legge a ogni suono dello spazio; poi torna com'era, sempre."""
    com_era = fisica.DB_PER_METRO
    fisica.DB_PER_METRO = db_per_metro
    try:
        yield
    finally:
        fisica.DB_PER_METRO = com_era


def bersaglio_di_prova():
    return Bersaglio(numero=0, forma=FORMA_DEL_BERSAGLIO, raggio=raggio_del_livello(LIVELLO_DEL_BERSAGLIO), partenza=PUNTO_DEL_BERSAGLIO,
                     frequenza_lfo=FREQUENZA_LFO_DEL_BERSAGLIO, fase_lfo=0.0, seme=SEME_DEL_BERSAGLIO)


def _tiro(chiave, punto, bersaglio):
    """Il tiro che passa per il punto, col tiro più dolce di fisica.soluzione, o None se non c'è."""
    valori = fisica.soluzione(punto)
    if valori is None:
        return None
    traiettoria = fisica.Traiettoria(*fisica.parametri_del_tiro(*valori))
    return Tiro(chiave, valori, traiettoria, calcola_volo(traiettoria, [bersaglio], 0.0))


def _riesce(tiro, bersaglio, distanza):
    """Vero se il tiro fa ciò che il suo nome promette: il centro colpisce, il corto cade prima del
    bersaglio, il lungo cade oltre, il largo gli passa a destra, e nessun mancato lo tocca."""
    if tiro is None:
        return False
    volo = tiro.volo
    if tiro.chiave == "centro":
        return volo.colpito is bersaglio
    if volo.colpito is not None:
        return False
    if tiro.chiave == "corto":
        return volo.gittata < distanza - bersaglio.raggio
    if tiro.chiave == "lungo":
        return volo.gittata > distanza + bersaglio.raggio
    return volo.scarto is not None and volo.scarto[1] > bersaglio.raggio


@functools.cache
def tiri_di_prova():
    """Il bersaglio di prova e i quattro tiri, nell'ordine di TIRI. Il centro mira al centro del
    bersaglio; il corto a un punto a terra davanti a lui; il largo a un punto alla sua destra; il
    lungo a un punto sopra di lui, alzato finché il tiro non gli passa sopra. Solleva ValueError se
    un tiro non fa ciò che deve: allora vanno ritoccati i numeri in testa al modulo."""
    bersaglio = bersaglio_di_prova()
    centro = np.asarray(bersaglio.partenza, dtype=float)
    cannone = np.asarray(fisica.CANNONE, dtype=float)
    distanza = math.hypot(centro[0] - cannone[0], centro[2] - cannone[2])
    avanti = np.array(((centro[0] - cannone[0]) / distanza, 0.0, (centro[2] - cannone[2]) / distanza))
    destra = np.array((avanti[2], 0.0, -avanti[0]))
    oltre = bersaglio.raggio + SCARTO_DEI_MANCATI
    tiri = {
        "centro": _tiro("centro", centro, bersaglio),
        "corto": _tiro("corto", np.array((cannone[0], 0.0, cannone[2])) + avanti * (distanza - oltre), bersaglio),
        "largo": _tiro("largo", centro + destra * oltre, bersaglio),
    }
    for passo in range(PASSI_DEL_LUNGO):
        lungo = _tiro("lungo", centro + np.array((0.0, oltre + passo * PASSO_DEL_LUNGO, 0.0)), bersaglio)
        if _riesce(lungo, bersaglio, distanza):
            tiri["lungo"] = lungo
            break
    sbagliati = [chiave for chiave in TIRI if not _riesce(tiri.get(chiave), bersaglio, distanza)]
    if sbagliati:
        raise ValueError(f"i tiri di prova {', '.join(sbagliati)} non fanno ciò che devono: vanno ritoccati i numeri in testa a banco_volo.py")
    return bersaglio, tuple(tiri[chiave] for chiave in TIRI)


def rendi_tiro(bersaglio, tiro, forma, fs=FS):
    """Il suono di un tiro, come lo suonerebbe il motore del gioco: il bersaglio da solo, poi lo
    sparo e la voce del proiettile di forma forma, e all'impatto l'evento programmato del contratto
    del volo. Restituisce (campioni, 2) float32."""
    with banco.in_memoria() as (motore, flusso):
        effetti = suoni.Suoni(motore)
        sparo = round(ASCOLTO_PRIMA_DEL_TIRO * fs)
        volo = calcola_volo(tiro.traiettoria, [bersaglio], motore.tempo_di_gioco(sparo))
        effetto = effetti.pronto("colpito" if volo.colpito is not None else "a_terra", posizione=volo.punto_finale)
        arrivo = sparo + math.ceil(volo.durata / VELOCITA_DEL_TEMPO * fs)
        fine = arrivo + max(0 if effetto is None else len(effetto), round(CODA_DEL_TIRO * fs))
        voce_del_bersaglio = banco.entra_ed_esce(VoceTono(bersaglio.parametri_sonori, BERSAGLIO_FORMA, 0, guadagno=0.0), 0, fine, fs)
        proiettile = VoceTono(parametri_del_proiettile(volo), forma, sparo)
        motore.aggiungi(voce_del_bersaglio)
        effetti.suona("sparo", inizio=sparo)
        motore.aggiungi(proiettile)

        def impatto(campione):
            # Nel filo dell'audio, al campione esatto: soltanto voci e buffer già pronti.
            motore.ferma(proiettile, campione)
            if effetto is not None and len(effetto):
                motore.suona(effetto, inizio=campione)
            if volo.colpito is not None:
                motore.ferma(voce_del_bersaglio, campione)

        motore.programma(volo.fine, impatto)
        return flusso.rendi(fine)


def componi(ricetta):
    """Il materiale di una versione: i quattro tiri in fila, con la forma del proiettile e la caduta
    del volume della ricetta, e una pausa fra l'uno e l'altro."""
    bersaglio, tiri = tiri_di_prova()
    pezzi = []
    with caduta(ricetta["db_per_metro"]):
        for tiro in tiri:
            if pezzi:
                pezzi.append(np.zeros((round(PAUSA_FRA_I_TIRI * FS), 2), dtype=np.float32))
            pezzi.append(rendi_tiro(bersaglio, tiro, ricetta["forma"]))
    return np.concatenate(pezzi)


def _decimali(valore):
    """Un numero coi soli decimali che servono: 0,25; 0,4."""
    testo = numero(valore, 2)
    return testo.rstrip("0").rstrip(",") if "," in testo else testo


def _con_attuale(nome, attuale):
    return _("{nome}, quella attuale").format(nome=nome) if attuale else nome


DOMANDA = _("quanto capisci dove va il colpo")
MATERIALE = _("In ogni lettera senti quattro tiri contro lo stesso bersaglio fermo, sempre in quest'ordine: un centro, uno corto, uno lungo e uno largo.")


def _timbro():
    candidati = []
    for forma in FORME_DEL_PROIETTILE:
        attuale = forma == PROIETTILE_FORMA
        nome = _con_attuale(_("la forma {forma} del proiettile").format(forma=forma), attuale)
        candidati.append(banco.Candidato(nome, {"forma": forma, "db_per_metro": DB_PER_METRO}, attuale))
    return banco.Gruppo("timbro", _("Volo, il timbro del proiettile"), DOMANDA, MATERIALE, tuple(candidati))


def _caduta():
    candidati = []
    for valore in (DB_PER_METRO, DB_PER_METRO_RIPIDO):
        attuale = valore == DB_PER_METRO
        nome = _con_attuale(_("la caduta di {valore} decibel per metro").format(valore=_decimali(valore)), attuale)
        candidati.append(banco.Candidato(nome, {"forma": PROIETTILE_FORMA, "db_per_metro": valore}, attuale))
    return banco.Gruppo("caduta", _("Volo, la caduta del volume"), DOMANDA, MATERIALE, tuple(candidati))


GRUPPI = (_timbro(), _caduta())
TITOLO_DEGLI_EFFETTI = _("Volo, gli effetti")
NOMI_DEGLI_EFFETTI = {"sparo": _("Lo sparo."), "colpito": _("Il bersaglio colpito."), "colpito_bonus": _("Il bonus colpito."), "a_terra": _("Il colpo a terra.")}


def prepara(seme, materiali=None):
    return banco.prepara_prove(GRUPPI, componi, seme, materiali)


def effetti_da_sentire(ascolto):
    """Gli effetti del gruppo d'ascolto, in ordine, come (nome da annunciare, buffer): lo sparo com'è,
    i due colpiti dal punto d'impatto del centro, il colpo a terra dal punto dove cade il corto."""
    _bersaglio, tiri = tiri_di_prova()
    dove = {"sparo": None, "colpito": tiri[0].volo.punto_finale, "colpito_bonus": tiri[0].volo.punto_finale, "a_terra": tiri[1].volo.punto_finale}
    effetti = []
    for evento in EFFETTI:
        buffer = ascolto.effetti.pronto(evento, posizione=dove[evento])
        effetti.append((NOMI_DEGLI_EFFETTI[evento], np.zeros((0, 2), dtype=np.float32) if buffer is None else buffer))
    return effetti


def ascolta_effetti(ascolto, effetti):
    """Ogni effetto annunciato e poi suonato, il primo dopo il silenzio iniziale; Escape interrompe."""
    for posto, (nome, buffer) in enumerate(effetti):
        print(nome)
        anticipo = banco.ANTICIPO_PRIMA_LETTERA if posto == 0 else banco.ANTICIPO_RASSEGNA
        if ascolto.suona_buffer(buffer, anticipo, banco.TASTI_DELLA_RASSEGNA) == "\x1b":
            print(_("Ascolto interrotto."))
            return


def esegui_effetti(ascolto, risultati, annotazioni):
    """Il gruppo d'ascolto degli effetti, che non è alla cieca: la spiegazione, l'annuncio, l'ascolto,
    il menu di fine gruppo col commento libero e, nel file, quale preset suona ogni effetto."""
    print(_("Nel prossimo gruppo senti gli effetti del tiro, ciascuno annunciato prima di suonare e dal punto dove nasce in partita: lo sparo, "
            "il bersaglio colpito, il bonus colpito e il colpo a terra. Poi puoi commentarli."))
    if not comune.gruppo(TITOLO_DEGLI_EFFETTI, len(EFFETTI), _("suoni")):
        risultati.scrivi(_("{titolo}: gruppo saltato.").format(titolo=TITOLO_DEGLI_EFFETTI))
        return
    effetti = effetti_da_sentire(ascolto)
    ascolta_effetti(ascolto, effetti)
    annotazioni.esito(TITOLO_DEGLI_EFFETTI, riproduci=lambda: ascolta_effetti(ascolto, effetti))
    preset = [_("{evento} col preset {preset}").format(evento=evento, preset=suoni.EVENTI[evento]) for evento in EFFETTI]
    risultati.scrivi(_("{titolo}: {preset}.").format(titolo=TITOLO_DEGLI_EFFETTI, preset=elenco(preset)))
    annotazioni.scrivi_in_sospeso()


def main(argv=None):
    seme = banco.leggi_seme(sys.argv[1:] if argv is None else argv, "banco_volo")
    if seme is None:
        return 1
    print(_("Preparo i suoni dei tiri."))
    prove = prepara(seme)
    print(_("Banco del volo di Ciechino: {gruppi} alla cieca, poi l'ascolto degli effetti del tiro.").format(gruppi=conta(len(prove), "{n} gruppo", "{n} gruppi")))
    print(_("Nei gruppi alla cieca senti la stessa cosa in più versioni, ciascuna con la sua lettera. Prima la rassegna, tutte le lettere una volta, "
            "senza domande; Escape la interrompe."))
    print(_("Poi il voto, una lettera alla volta: spazio fa partire la lettera e la ferma; il tasto di un'altra lettera fa sentire quella; da 1 a 5 "
            "dai il voto e passi alla seguente; Invio passa oltre; n vuol dire nessuna preferenza; Escape chiude il voto."))
    print(_("A fine gruppo: r ripete, c commenta, Invio lo dà per superato, Escape lo chiude senza giudizio, e allora i suoi voti non contano."))
    print(_("Gli esiti vanno nel file {file}, accanto al banco.").format(file=FILE_DEGLI_ESITI.name))
    if not enter_escape(banco.invito_a_cominciare()):
        return 0
    return banco.sessione(_("Banco del volo"), seme, FILE_DEGLI_ESITI, prove, EFFETTI, esegui_effetti)


if __name__ == "__main__":
    sys.exit(main())
