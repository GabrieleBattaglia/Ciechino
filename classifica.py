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
riserva o da zero; ripresi i dati da una copia, il file principale torna al suo posto, così
l'avviso si sente una volta sola.
Una voce è {"nome", "punteggio", "data" (ISO, ora locale), "livello", "tiri", "centri",
"durata_s"}; il file è {"formato": FORMATO_CLASSIFICA, "voci": [...]}.
"""

import contextlib
import json
import os
import shutil
import time
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
# Il numero più grande che un file dei dati può contenere, in valore assoluto: 2 alla 53, l'intero
# più grande che un float tiene esatto. Un intero più grande, scritto a mano in un file che resta
# JSON valido, fa sollevare OverflowError a ogni conversione in float, e nessun conto vero di
# Ciechino ci arriva. È il confine fra un numero e un file rovinato, non una taratura del gioco.
NUMERO_MASSIMO = 2 ** 53
# Gli errori che dicono un file rovinato: il disco, il JSON, la codifica, una struttura troppo
# annidata, un numero troppo grande. Tutti finiscono in un avviso, mai in un'eccezione.
ERRORI_DI_LETTURA = (OSError, ValueError, RecursionError, OverflowError)
# Un antivirus o un programma di backup può tenere un file aperto per un attimo, e Windows rifiuta
# l'apertura, la rinomina o la sostituzione con PermissionError: prima di dire rovinato un file
# sano, o di rinunciare a un salvataggio, si riprova qualche volta, da vicino. Sono attese del
# disco, non tarature del gioco.
TENTATIVI_DEL_DISCO = 3
PAUSA_FRA_I_TENTATIVI = 0.2
# I file rovinati che Windows non ha lasciato rinominare, per percorso: finché stanno qui, il
# salvataggio non li ruota nella copia di riserva, che è buona, ma riprova a metterli da parte, e
# se non ci riesce non scrive niente.
_DA_METTERE_DA_PARTE = set()


# I numeri dei file.

def numero_valido(valore):
    """Vero per un numero che un file dei dati può contenere: un int o un float, non booleano,
    finito e non oltre NUMERO_MASSIMO in valore assoluto. Il confronto con abs scarta anche
    l'infinito e il NaN, e non converte mai un intero enorme in float."""
    return isinstance(valore, int | float) and not isinstance(valore, bool) and abs(valore) <= NUMERO_MASSIMO


# I file JSON, scritti e letti con cura.

def _con_pazienza(azione, *argomenti):
    """L'azione sul disco, riprovata fino a TENTATIVI_DEL_DISCO volte se Windows risponde
    PermissionError, cioè un file tenuto aperto da un altro programma; l'ultimo errore passa a chi
    chiama. Gli altri errori passano subito: un file che manca non arriva aspettando."""
    for tentativo in range(1, TENTATIVI_DEL_DISCO + 1):
        try:
            return azione(*argomenti)
        except PermissionError:
            if tentativo >= TENTATIVI_DEL_DISCO:
                raise
            time.sleep(PAUSA_FRA_I_TENTATIVI)
    return None


def _apri_e_leggi(percorso):
    # utf-8-sig accetta anche il BOM che il Blocco note di Windows mette davanti.
    with open(percorso, encoding="utf-8-sig") as f:
        return json.load(f)


def _leggi_json(percorso):
    return _con_pazienza(_apri_e_leggi, percorso)


def _chiave(percorso):
    """Il percorso come chiave di _DA_METTERE_DA_PARTE: assoluto e con le maiuscole di Windows
    ignorate, perché lo stesso file non conti due volte."""
    return os.path.normcase(os.path.abspath(percorso))


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
        _con_pazienza(os.rename, percorso, nuovo)
    except OSError:
        return None
    return nuovo


def _dalle_copie(percorso, controlla, suffissi):
    """I dati dalla prima copia che si legge, fra quelle coi suffissi dati, la frase che lo dice
    e il percorso della copia usata; se nessuna si legge, None, la frase della ripartenza e None.
    Una copia che non si legge si mette da parte: quella di riserva, perché il salvataggio
    seguente non la sovrascriva; il .tmp di un salvataggio interrotto a metà, perché il caricamento
    seguente non ridica ogni volta che il file manca."""
    for suffisso in suffissi:
        copia = percorso + suffisso
        if not os.path.exists(copia):
            continue
        try:
            dati = controlla(_leggi_json(copia))
        except ERRORI_DI_LETTURA:
            _metti_da_parte(copia)
            continue
        if suffisso == ".tmp":
            # Traduttori: dopo un salvataggio interrotto; file è il nome del file della copia.
            return dati, _("Riprendo dalla copia dell'ultimo salvataggio, {file}.").format(file=os.path.basename(copia)), copia
        # Traduttori: file è il nome del file della copia di riserva.
        return dati, _("Riprendo dalla copia di riserva, {file}.").format(file=os.path.basename(copia)), copia
    return None, _("Riparto da zero."), None


def _rimetti_a_posto(copia, percorso):
    """Il file principale, che manca, rifatto dalla copia da cui si sono appena ripresi i dati:
    senza, il caricamento seguente direbbe di nuovo che il file manca, e l'avviso dei dati rovinati
    tornerebbe a ogni avvio e a ogni partita finita. Si sposta il file grezzo, non si riscrivono i
    dati puliti, perché per la classifica sono la lista delle voci e non il file intero. Il .tmp di
    un salvataggio interrotto prende il suo posto, come avrebbe fatto il salvataggio; il .bak
    invece si copia, e resta la copia di riserva. La copia passa dal .tmp, così il file principale
    o c'è intero o non c'è. Un disco che non collabora lascia tutto com'è: il ripiego riuscirà di
    nuovo al prossimo caricamento."""
    if copia.endswith(".tmp"):
        with contextlib.suppress(OSError):
            _con_pazienza(os.replace, copia, percorso)
        return
    temporaneo = percorso + ".tmp"
    try:
        shutil.copyfile(copia, temporaneo)
        _con_pazienza(os.replace, temporaneo, percorso)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(temporaneo)


def leggi_dati(percorso, controlla):
    """Legge un file JSON di Ciechino e restituisce (dati, avviso). controlla(grezzi) restituisce
    i dati puliti, o solleva ValueError se il contenuto non è quello giusto.
    Un file che manca senza copie è il primo avvio: (None, None). Un file che non si legge si
    rinomina in .illeggibile e si prova la copia .bak; un file che manca, per un salvataggio
    interrotto fra le due sostituzioni, si cerca nella copia .tmp e poi nella .bak. Ripresi i dati
    da una copia, il file principale si rimette al suo posto, e l'avviso si dà una volta sola.
    Senza niente da leggere i dati sono None, e chi chiama riparte dai suoi predefiniti."""
    nome = os.path.basename(percorso)
    if os.path.exists(percorso):
        try:
            dati = controlla(_leggi_json(percorso))
        except ERRORI_DI_LETTURA as e:
            errore = _spiega(e)
        else:
            # Un file che ora si legge non va più messo da parte, anche se una volta non si lasciava rinominare.
            _DA_METTERE_DA_PARTE.discard(_chiave(percorso))
            return dati, None
        rinominato = _metti_da_parte(percorso)
        # Traduttori: file è il nome del file, errore il motivo, spesso in inglese.
        parti = [_("Il file {file} non si legge: {errore}.").format(file=nome, errore=errore)]
        if rinominato:
            parti.append(_("Il file rovinato ora si chiama {file}, e resta com'era.").format(file=os.path.basename(rinominato)))
        else:
            # Il file resta col suo nome: il salvataggio seguente lo deve mettere da parte, non ruotare nel .bak.
            _DA_METTERE_DA_PARTE.add(_chiave(percorso))
            parti.append(_("Il file rovinato non si è potuto rinominare."))
        dati, frase, copia = _dalle_copie(percorso, controlla, (".bak",))
    elif os.path.exists(percorso + ".tmp") or os.path.exists(percorso + ".bak"):
        parti = [_("Il file {file} manca.").format(file=nome)]
        dati, frase, copia = _dalle_copie(percorso, controlla, (".tmp", ".bak"))
    else:
        return None, None
    # Il file rovinato che non si è potuto rinominare è ancora lì, col suo nome: non si tocca.
    if dati is not None and not os.path.exists(percorso):
        _rimetti_a_posto(copia, percorso)
    parti.append(frase)
    return dati, " ".join(parti)


def scrivi_dati(dati, percorso):
    """Scrive i dati in JSON senza mai lasciare un file a metà: prima il .tmp, poi il file
    vecchio diventa la copia .bak, poi il .tmp prende il suo posto. Vero se è riuscito; un disco
    che non collabora restituisce falso.
    Il file principale che il caricamento ha trovato rovinato e non ha potuto rinominare non
    diventa mai la copia di riserva: prima si riprova a metterlo da parte, e se ancora non si
    lascia rinominare non si scrive niente. Se la seconda sostituzione fallisce, quando il file
    vecchio è già nel .bak, il .tmp resta: è completo, è l'unica copia dei dati nuovi, e il
    caricamento seguente lo riprende."""
    chiave = _chiave(percorso)
    if chiave in _DA_METTERE_DA_PARTE:
        if os.path.exists(percorso) and _metti_da_parte(percorso) is None:
            return False
        _DA_METTERE_DA_PARTE.discard(chiave)
    temporaneo = percorso + ".tmp"
    spostato = False
    try:
        with open(temporaneo, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dati, f, ensure_ascii=False, indent=1)
        if os.path.exists(percorso):
            _con_pazienza(os.replace, percorso, percorso + ".bak")
            spostato = True
        _con_pazienza(os.replace, temporaneo, percorso)
    except OSError:
        # Il .bak non torna al posto del file: il caricamento leggerebbe i dati vecchi e
        # ignorerebbe il .tmp più nuovo, che il salvataggio seguente sovrascriverebbe.
        if not spostato:
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


def _intero(valore):
    return max(0, round(valore)) if numero_valido(valore) else 0


def _decimale(valore):
    return max(0.0, round(float(valore), 3)) if numero_valido(valore) else 0.0


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
    if len(nome) < costanti.NOME_MINIMO or not numero_valido(punteggio) or punteggio < 0 or _data(grezza.get("data")) is None:
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
    if not numero_valido(punteggio) or punteggio < PUNTEGGIO_MINIMO:
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
