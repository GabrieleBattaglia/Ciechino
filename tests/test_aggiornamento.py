"""
Le prove dell'aggiornamento automatico, senza rete e senza sostituire niente: da sorgente il controllo
non parte; con un sys.frozen finto e gestisci_aggiornamento sostituito, il filo chiama sulla finestra
i metodi giusti, nel filo della finestra e nel loro ordine, e la chiude solo quando l'aggiornamento è
pronto; con la funzione vera di GBUtils, e il controllo su GitHub e lo scaricamento sostituiti, la
conversazione passa dalla finestra; le frasi di GBUtils dette in Ciechino e gli esiti riconosciuti.
La finestra vera la scrive un'altra autrice: qui una finestra finta annota le chiamate.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di test_aggiornamento.py di MESS.
"""

import sys
import threading

import GBUtils
import pytest

import aggiornamento
from costanti import VERSIONE

# Quanti secondi una prova aspetta il filo del controllo: molti più del necessario.
ATTESA = 10
NOTE = "## Le novità\nIl bersaglio bonus."


def _subito(funzione, *argomenti):
    """Al posto di wx.CallAfter: esegue subito, nel filo del controllo, e annota in quale filo."""
    _subito.fili.append(threading.current_thread().name)
    return funzione(*argomenti)


_subito.fili = []


class FinestraFinta:
    """La finestra vista dal filo dell'aggiornamento: annota le chiamate e risponde alla proposta come le si dice."""

    def __init__(self, risposta=True, chiusa=False):
        self.risposta = risposta
        self.chiusa = chiusa
        self.chiamate = []

    def proponi_l_aggiornamento(self, versione_attuale, versione_nuova, note, attesa=None):
        self.chiamate.append(("proponi", versione_attuale, versione_nuova, note, attesa))
        return self.risposta

    def avvisa_dell_aggiornamento(self, testo):
        self.chiamate.append(("avvisa", testo))

    def avanzamento_dell_aggiornamento(self, presi, totale):
        self.chiamate.append(("avanzamento", presi, totale))

    def chiudi_per_aggiornare(self):
        self.chiamate.append(("chiudi",))


@pytest.fixture
def compilato(monkeypatch):
    """Un Ciechino compilato per finta."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _subito.fili = []


@pytest.fixture
def gestione_finta(compilato, monkeypatch):
    """
    gestisci_aggiornamento di GBUtils sostituito da una conversazione finta, che fa quello che fa la
    vera quando c'è una versione nuova e la si accetta: propone, fa avanzare lo scaricamento, dice
    l'esito con la frase di GBUtils passata da traduci, e risponde vero. Restituisce che cosa ha
    ricevuto, e come rispondere.
    """
    ricevuti = {"argomenti": None, "opzioni": None, "pronto": True}

    def gestisci(app, versione, api, **opzioni):
        ricevuti["argomenti"], ricevuti["opzioni"] = (app, versione, api), opzioni
        if not opzioni["proponi"](versione, "0.2.0", NOTE, attesa=120):
            return False
        opzioni["avanzamento"](50, 100)
        if not ricevuti["pronto"]:
            opzioni["avvisa"](opzioni["traduci"]("Aggiornamento non riuscito, si prosegue con questa versione."))
            return False
        opzioni["avvisa"](opzioni["traduci"]("Aggiornamento pronto, il programma si chiude per applicarlo."))
        return True

    monkeypatch.setattr(GBUtils, "gestisci_aggiornamento", gestisci)
    return ricevuti


def test_da_sorgente_il_controllo_non_parte(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    chiamate = []
    monkeypatch.setattr(GBUtils, "gestisci_aggiornamento", lambda *a, **k: chiamate.append(a) or True)
    finestra = FinestraFinta()
    assert aggiornamento.controlla(finestra, chiama_dopo=_subito) is None
    assert chiamate == [] and finestra.chiamate == []


def test_i_dati_della_release():
    assert aggiornamento.APP == "Ciechino"
    assert aggiornamento.API == "https://api.github.com/repos/GabrieleBattaglia/Ciechino/releases/latest"


def test_dall_eseguibile_chiama_i_metodi_giusti(gestione_finta):
    finestra = FinestraFinta()
    filo = aggiornamento.controlla(finestra, chiama_dopo=_subito)
    filo.join(ATTESA)
    assert not filo.is_alive()
    assert gestione_finta["argomenti"] == ("Ciechino", VERSIONE, aggiornamento.API)
    assert gestione_finta["opzioni"]["traduci"] is aggiornamento.traduci
    assert finestra.chiamate == [
        ("proponi", VERSIONE, "0.2.0", NOTE, 120),
        ("avanzamento", 50, 100),
        ("avvisa", aggiornamento.FRASE_PRONTO),
        ("chiudi",),
    ]
    # Tutto passa da chiama_dopo, cioè dal filo della finestra, e niente dal filo principale.
    assert len(_subito.fili) == 4 and threading.main_thread().name not in _subito.fili


def test_non_adesso_non_chiude(gestione_finta):
    finestra = FinestraFinta(risposta=False)
    aggiornamento.controlla(finestra, chiama_dopo=_subito).join(ATTESA)
    assert [c[0] for c in finestra.chiamate] == ["proponi"]


def test_lo_scaricamento_non_riuscito_non_chiude(gestione_finta):
    gestione_finta["pronto"] = False
    finestra = FinestraFinta()
    aggiornamento.controlla(finestra, chiama_dopo=_subito).join(ATTESA)
    assert finestra.chiamate[-1] == ("avvisa", aggiornamento.FRASE_NON_RIUSCITO)
    assert "chiudi" not in [c[0] for c in finestra.chiamate]


def test_la_finestra_chiusa_non_riceve_niente(gestione_finta):
    # La finestra che si chiude mentre il controllo gira non riceve niente, e la risposta vale no.
    chiusa = FinestraFinta(chiusa=True)
    aggiornamento.controlla(chiusa, chiama_dopo=_subito).join(ATTESA)
    assert chiusa.chiamate == []
    assert gestione_finta["argomenti"] is not None


def test_la_conversazione_vera_di_gbutils(compilato, monkeypatch):
    """La funzione vera di GBUtils, con il controllo su GitHub e lo scaricamento sostituiti."""
    ricevuti = []

    def controllo(versione, api, *_argomenti, **_opzioni):
        ricevuti.append(("controllo", versione, api))
        return True, "0.2.0", "https://github.com/GabrieleBattaglia/Ciechino/releases/download/v0.2.0/Ciechino.zip", NOTE

    def scaricamento(indirizzo, app, avanzamento=None, prima_di_sostituire=None, **_opzioni):
        ricevuti.append(("scaricamento", app))
        avanzamento(50, 100)
        prima_di_sostituire()
        return True

    monkeypatch.setattr(GBUtils, "update_checker", controllo)
    monkeypatch.setattr(GBUtils, "perform_update", scaricamento)
    finestra = FinestraFinta()
    aggiornamento.controlla(finestra, chiama_dopo=_subito).join(ATTESA)
    assert ricevuti == [("controllo", VERSIONE, aggiornamento.API), ("scaricamento", "Ciechino")]
    # La proposta riceve l'attesa di GBUtils, due minuti; poi l'avanzamento, l'esito detto in Ciechino prima della sostituzione, e la chiusura.
    assert finestra.chiamate == [
        ("proponi", VERSIONE, "0.2.0", NOTE, 120),
        ("avanzamento", 50, 100),
        ("avvisa", aggiornamento.FRASE_PRONTO),
        ("chiudi",),
    ]


def test_gli_esiti_dalle_frasi_vere_di_gbutils(compilato, monkeypatch):
    # Il pacchetto non ancora pronto: GBUtils mette la versione davanti alla frase.
    monkeypatch.setattr(GBUtils, "update_checker", lambda *_a, **_k: (True, "0.2.0", "", NOTE))
    finestra = FinestraFinta()
    aggiornamento.controlla(finestra, chiama_dopo=_subito).join(ATTESA)
    non_pronto = finestra.chiamate[-1][1]
    assert non_pronto == "Disponibile la versione 0.2.0, ma il pacchetto non è ancora pronto: te la ripropongo a un prossimo avvio."
    # Lo scaricamento che non riesce.
    monkeypatch.setattr(GBUtils, "update_checker", lambda *_a, **_k: (True, "0.2.0", "https://github.com/x/Ciechino.zip", NOTE))
    monkeypatch.setattr(GBUtils, "perform_update", lambda *_a, **_k: False)
    finestra = FinestraFinta()
    aggiornamento.controlla(finestra, chiama_dopo=_subito).join(ATTESA)
    non_riuscito = finestra.chiamate[-1][1]
    assert "chiudi" not in [c[0] for c in finestra.chiamate]
    assert [aggiornamento.esito(t) for t in (aggiornamento.FRASE_PRONTO, non_riuscito, non_pronto, "Un'altra frase.")] == [
        aggiornamento.PRONTO, aggiornamento.NON_RIUSCITO, aggiornamento.NON_PRONTO, None]


def test_le_frasi_di_gbutils_dette_in_ciechino():
    assert aggiornamento.traduci("ma il pacchetto non e' ancora pronto.") == aggiornamento.FRASE_NON_PRONTO
    assert aggiornamento.traduci("Hai gia' l'ultima versione,") == "Hai già l'ultima versione,"
    assert aggiornamento.traduci("Una frase che e' nuova, perche' GBUtils cresce.") == "Una frase che è nuova, perché GBUtils cresce."
    assert aggiornamento.traduci("Disponibile la versione") == "Disponibile la versione"
    # Le frasi dell'esito parlano di Ciechino, e nessuna ha l'apostrofo al posto dell'accento.
    assert "Ciechino" in aggiornamento.FRASE_PRONTO
    assert not any(accento in frase for frase in aggiornamento.FRASI.values() for accento in ("e' ", "a' ", "o' ", "u' ", "i' "))
