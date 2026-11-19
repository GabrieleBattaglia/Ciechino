"""
La classifica di Ciechino: le cento partite migliori per punteggio, il loro file e le righe da leggere.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di gestione_dati.py di Gabryscola e di dati.py di PokerMachine.
La classifica è la top 100 del punto 73 di idee.txt e della decisione D11: posizione, nome,
punteggio ed età del record in giorni, ordinata per punteggio; a parità di punti resta davanti chi
c'era prima, perché un risultato uguale non scavalca mai quello già conquistato. Il nome si chiede
soltanto a chi entra.
Il modulo non stampa e non solleva per un file rovinato: carica restituisce le voci e un avviso, e
chi chiama decide come dirlo. Qui sta anche la scrittura sicura dei file JSON di Ciechino, che usano
pure statistiche.py e impostazioni.py: prima un file .tmp, poi il file vecchio diventa la copia
.bak, poi il .tmp prende il suo posto con os.replace. Un file che non si legge non si sovrascrive
mai: si rinomina in .illeggibile, con un numero se ce n'è già uno, e si riparte dalla copia di
riserva o da zero.
Una voce è {"nome", "punteggio", "data" (ISO, ora locale), "livello", "tiri", "centri",
"durata_s"}; il file è {"formato": FORMATO_CLASSIFICA, "voci": [...]}.
"""

import contextlib
import json
import math
import os
from datetime import datetime

import costanti
import percorsi
import testi
from lingua import _, ngettext, plurale

# Una partita senza punti non entra in classifica, anche se la classifica è vuota: un record da
# zero punti non è un record. Non è un numero di taratura del gioco, ma la regola d'ingresso.
PUNTEGGIO_MINIMO = 1
# I campi numerici di una voce, oltre al punteggio, con il loro tipo.
CAMPI_INTERI = ("livello", "tiri", "centri")
CAMPI_DECIMALI = ("durata_s",)
# Gli errori che dicono un file rovinato: il disco, il JSON, la codifica, una struttura troppo
# annidata. Tutti finiscono in un avviso, mai in un'eccezione.
ERRORI_DI_LETTURA = (OSError, ValueError, RecursionError)


# I file JSON, scritti e letti con cura.

def _leggi_json(percorso):
    # utf-8-sig accetta anche il BOM che il Blocco note di Windows mette davanti.
    with open(percorso, encoding="utf-8-sig") as f:
        return json.load(f)


def _spiega(errore):
    return str(errore) or type(errore).__name__


def _nome_libero(base):
    """Il primo nome che non c'è ancora: base, poi base.2, base.3 e così via."""
    if not os.path.exists(base):
        return base
    n = 2
    while os.path.exists(f"{base}.{n}"):
        n += 1
    return f"{base}.{n}"


def _metti_da_parte(percorso):
    """Rinomina un file che non si legge, senza mai sovrascrivere un altro file messo da parte.
    Restituisce il nome nuovo, o None se Windows non lo lascia rinominare."""
    nuovo = _nome_libero(percorso + ".illeggibile")
    try:
        os.rename(percorso, nuovo)
    except OSError:
        return None
    return nuovo


def _dalle_copie(percorso, controlla, suffissi):
    """I dati dalla prima copia che si legge, fra quelle coi suffissi dati, e la frase che lo
    dice; se nessuna si legge, None e la frase della ripartenza. Una copia di riserva che non si
    legge si mette da parte, così il salvataggio seguente non la sovrascrive."""
    for suffisso in suffissi:
        copia = percorso + suffisso
        if not os.path.exists(copia):
            continue
        try:
            dati = controlla(_leggi_json(copia))
        except ERRORI_DI_LETTURA:
            if suffisso == ".bak":
                _metti_da_parte(copia)
            continue
        if suffisso == ".tmp":
            # Traduttori: dopo un salvataggio interrotto; file è il nome del file della copia.
            return dati, _("Riprendo dalla copia dell'ultimo salvataggio, {file}.").format(file=os.path.basename(copia))
        # Traduttori: file è il nome del file della copia di riserva.
        return dati, _("Riprendo dalla copia di riserva, {file}.").format(file=os.path.basename(copia))
    return None, _("Riparto da zero.")


def leggi_dati(percorso, controlla):
    """Legge un file JSON di Ciechino e restituisce (dati, avviso). controlla(grezzi) restituisce
    i dati puliti, o solleva ValueError se il contenuto non è quello giusto.
    Un file che manca senza copie è il primo avvio: (None, None). Un file che non si legge si
    rinomina in .illeggibile e si prova la copia .bak; un file che manca, per un salvataggio
    interrotto fra le due sostituzioni, si cerca nella copia .tmp e poi nella .bak. Senza niente
    da leggere i dati sono None, e chi chiama riparte dai suoi predefiniti."""
    nome = os.path.basename(percorso)
    if os.path.exists(percorso):
        try:
            return controlla(_leggi_json(percorso)), None
        except ERRORI_DI_LETTURA as e:
            errore = _spiega(e)
        rinominato = _metti_da_parte(percorso)
        # Traduttori: file è il nome del file, errore il motivo, spesso in inglese.
        parti = [_("Il file {file} non si legge: {errore}.").format(file=nome, errore=errore)]
        if rinominato:
            parti.append(_("Il file rovinato ora si chiama {file}, e resta com'era.").format(file=os.path.basename(rinominato)))
        else:
            parti.append(_("Il file rovinato non si è potuto rinominare."))
        dati, frase = _dalle_copie(percorso, controlla, (".bak",))
    elif os.path.exists(percorso + ".tmp") or os.path.exists(percorso + ".bak"):
        parti = [_("Il file {file} manca.").format(file=nome)]
        dati, frase = _dalle_copie(percorso, controlla, (".tmp", ".bak"))
    else:
        return None, None
    parti.append(frase)
    return dati, " ".join(parti)


def scrivi_dati(dati, percorso):
    """Scrive i dati in JSON senza mai lasciare un file a metà: prima il .tmp, poi il file
    vecchio diventa la copia .bak, poi il .tmp prende il suo posto. Vero se è riuscito; un disco
    che non collabora restituisce falso, e il .tmp se ne va."""
    temporaneo = percorso + ".tmp"
    try:
        with open(temporaneo, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dati, f, ensure_ascii=False, indent=1)
        if os.path.exists(percorso):
            os.replace(percorso, percorso + ".bak")
        os.replace(temporaneo, percorso)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(temporaneo)
        return False
    return True


# I nomi e le voci.

def pulisci_nome(testo):
    """Il nome come entra in classifica: i caratteri di controllo e invisibili tolti, gli spazi
    di ogni genere ridotti a uno solo, niente spazi ai bordi, al più NOME_MASSIMO caratteri. Le
    maiuscole restano come le ha scritte chi gioca."""
    if testo is None:
        return ""
    caratteri = []
    for c in str(testo):
        if c.isspace():
            caratteri.append(" ")
        elif c.isprintable():
            caratteri.append(c)
    nome = " ".join("".join(caratteri).split())
    return nome[:costanti.NOME_MASSIMO].rstrip()


def nome_valido(testo):
    """Vero se il nome, pulito, ha almeno NOME_MINIMO caratteri."""
    return len(pulisci_nome(testo)) >= costanti.NOME_MINIMO


def _adesso():
    return datetime.now().isoformat(timespec="seconds")


def nuova_voce(nome, riepilogo, data=None):
    """La voce di una partita finita, dal riepilogo di Partita.riepilogo(): {"punteggio",
    "livello", "tiri", "centri", "durata_s"}. La data è adesso, se non la si dà."""
    voce = {"nome": pulisci_nome(nome), "punteggio": _intero(riepilogo.get("punteggio")), "data": data or _adesso()}
    for campo in CAMPI_INTERI:
        voce[campo] = _intero(riepilogo.get(campo))
    for campo in CAMPI_DECIMALI:
        voce[campo] = _decimale(riepilogo.get(campo))
    return voce


def _numero(valore):
    return isinstance(valore, int | float) and not isinstance(valore, bool) and math.isfinite(valore)


def _intero(valore):
    return max(0, round(valore)) if _numero(valore) else 0


def _decimale(valore):
    return max(0.0, round(float(valore), 3)) if _numero(valore) else 0.0


def _data(testo):
    if not isinstance(testo, str):
        return None
    try:
        return datetime.fromisoformat(testo)
    except ValueError:
        return None


def _voce_pulita(grezza):
    """Una voce del file, pulita, o None se manca di ciò che serve: un nome, un punteggio, una
    data che si legga. I campi secondari rovinati tornano a zero, e la voce resta."""
    if not isinstance(grezza, dict):
        return None
    nome = pulisci_nome(grezza.get("nome")) if isinstance(grezza.get("nome"), str) else ""
    punteggio = grezza.get("punteggio")
    if len(nome) < costanti.NOME_MINIMO or not _numero(punteggio) or punteggio < 0 or _data(grezza.get("data")) is None:
        return None
    voce = {"nome": nome, "punteggio": round(punteggio), "data": grezza["data"]}
    for campo in CAMPI_INTERI:
        voce[campo] = _intero(grezza.get(campo))
    for campo in CAMPI_DECIMALI:
        voce[campo] = _decimale(grezza.get(campo))
    return voce


def _ordinate(voci):
    """Per punteggio, dal più alto; a parità, la più vecchia davanti."""
    return sorted(voci, key=lambda v: (-v["punteggio"], _data(v["data"]).replace(tzinfo=None)))


def _controlla(grezzi):
    if not isinstance(grezzi, dict) or not isinstance(grezzi.get("voci"), list):
        raise ValueError(_("il file non contiene una classifica"))
    voci = [v for v in (_voce_pulita(g) for g in grezzi["voci"]) if v is not None]
    return _ordinate(voci)[:costanti.POSTI_IN_CLASSIFICA]


def _percorso(percorso):
    # Il percorso predefinito si calcola a ogni chiamata, mai all'importazione: da compilato e
    # nelle prove la cartella del programma cambia.
    return percorso or percorsi.percorso_dati(costanti.FILE_CLASSIFICA)


def carica(percorso=None):
    """Le voci della classifica e un avviso, None se è andato tutto bene. Non solleva mai: un
    file rovinato si mette da parte e si riparte dalla copia di riserva, o da una classifica vuota."""
    voci, avviso = leggi_dati(_percorso(percorso), _controlla)
    return (voci or []), avviso


def salva(voci, percorso=None):
    """Scrive la classifica; vero se è riuscito."""
    return scrivi_dati({"formato": costanti.FORMATO_CLASSIFICA, "voci": list(voci)}, _percorso(percorso))


def posizione(voci, punteggio):
    """La posizione, da 1, che avrebbe una voce nuova con quel punteggio, dietro tutte quelle con
    punti uguali o più alti; None se resta fuori dai POSTI_IN_CLASSIFICA o non ha punti."""
    if not _numero(punteggio) or punteggio < PUNTEGGIO_MINIMO:
        return None
    davanti = sum(1 for v in voci if v["punteggio"] >= punteggio)
    return davanti + 1 if davanti < costanti.POSTI_IN_CLASSIFICA else None


def inserisci(voci, voce):
    """Mette la voce al suo posto e taglia la classifica ai POSTI_IN_CLASSIFICA; restituisce (voci
    nuove, posizione), senza toccare la lista ricevuta. La voce entra pulita, col nome ripulito e
    la data di adesso se non ce l'ha; voci_nuove[posizione - 1] è lei, da passare a righe come
    nuova. Una voce che non entra lascia la classifica com'era, con posizione None."""
    voci = list(voci)
    nuova = dict(voce)
    nuova["nome"] = pulisci_nome(nuova.get("nome")) or _("Anonimo")
    if _data(nuova.get("data")) is None:
        nuova["data"] = _adesso()
    dove = posizione(voci, nuova.get("punteggio"))
    if dove is None:
        return voci, None
    voci.insert(dove - 1, nuova)
    return voci[:costanti.POSTI_IN_CLASSIFICA], dove


def _giorni(data, oggi):
    quando = _data(data)
    if quando is None:
        return 0
    giorno = oggi.date() if isinstance(oggi, datetime) else oggi
    return (giorno - quando.date()).days


def righe(voci, oggi=None, nuova=None):
    """Le righe da leggere: l'intestazione, poi una riga per voce, come «1. Gabry, 12.340 punti,
    3 giorni fa.». nuova è la voce appena entrata, o la sua posizione da 1, e la sua riga lo dice;
    oggi è il giorno da cui contare l'età dei record, predefinito adesso. Una classifica vuota è
    una riga sola, che lo dice."""
    voci = list(voci)
    if not voci:
        return [_("La classifica è ancora vuota: la prima partita che fa punti ci entra.")]
    oggi = oggi or datetime.now()
    n = len(voci)
    risultato = [ngettext("Classifica, {n} voce, ordinata per punteggio.", "Classifica, {n} voci, ordinata per punteggio.", n).format(n=testi.intero(n))]
    for numero, voce in enumerate(voci, 1):
        if isinstance(nuova, int) and not isinstance(nuova, bool):
            e_nuova = numero == nuova
        else:
            e_nuova = nuova is not None and (voce is nuova or voce == nuova)
        punti = plurale(voce["punteggio"], "{n} punto", "{n} punti", come=testi.intero(voce["punteggio"]))
        eta = testi.eta_in_giorni(_giorni(voce.get("data"), oggi))
        if e_nuova:
            # Traduttori: una riga della classifica, la voce appena entrata: «1. Gabry, 12.340 punti, oggi, nuova.».
            riga = _("{posizione}. {nome}, {punti}, {eta}, nuova.")
        else:
            # Traduttori: una riga della classifica: «1. Gabry, 12.340 punti, 3 giorni fa.».
            riga = _("{posizione}. {nome}, {punti}, {eta}.")
        risultato.append(riga.format(posizione=numero, nome=voce["nome"], punti=punti, eta=eta))
    return risultato
