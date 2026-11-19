"""
Le schede audio di Ciechino, per la scelta dell'uscita (D20): l'elenco delle uscite della macchina, la riga che dice ognuna
nell'elenco, la scelta da salvare nelle impostazioni e da ritrovare, e la rilettura delle schede arrivate dopo l'avvio.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, copiato da schede_audio.py di MeTeOra, come ha chiesto Gabriele quel giorno (D20), e
adattato. Ciechino ha un motore solo, il suo, e niente mpv: qui non ci sono la ricerca della scheda nell'elenco di mpv né
applica, automatica e in_uso, che toccano il mixer condiviso di GBUtils. La scheda la apre motore_audio.apri_flusso, e la
cambia Motore.cambia_scheda. L'elenco si chiede a GBUtils nel modo callback, che è quello del motore: a callback WDM-KS si
apre, con una decina di millesimi di latenza, e quindi resta nell'elenco, mentre MeTeOra, che scrive a blocchi, lo lascia
fuori. ASIO e WDM-KS prendono la scheda in esclusiva e possono zittire NVDA, e la loro riga lo dice; lo dice anche la riga
del vivavoce delle cuffie Bluetooth, che suona come un telefono. Le latenze si scrivono all'italiana, con la virgola.
Della scelta si salvano il nome del dispositivo e il nome intero della sua interfaccia, per esempio "Windows WASAPI", mai
l'indice: gli indici di PortAudio cambiano da un avvio all'altro, e lo stesso nome compare in più interfacce. Il dizionario
vuoto è la scelta automatica, quella di scegli_dispositivo_audio di GBUtils.
PortAudio conta le uscite una volta sola, quando parte: le cuffie Bluetooth accese dopo l'avvio compaiono soltanto dopo
rileggi, che chiude e riapre PortAudio con rileggi_dispositivi_audio di GBUtils. Un flusso di sounddevice aperto in quel
momento può far cadere il processo, e per questo rileggi si chiama soltanto col flusso del motore chiuso: lo fa
Motore.cambia_scheda, con rileggi vero.
Funzioni senza finestre e senza suono: non aprono niente e non scrivono in console.
"""

import re

import testi
from costanti import CHIAVI_DELLA_SCHEDA, NOMI_DEL_VIVAVOCE
from lingua import _

_A_CAPO = re.compile(r"[\r\n]+")


def _ripulito(nome):
    """Il nome su una riga sola: qualche driver ci mette un a capo, che romperebbe la riga di un elenco."""
    return _A_CAPO.sub(" ", nome or "").strip()


def uscite():
    """Le uscite fra cui scegliere, senza aprire niente: una lista di dizionari con indice, dispositivo, interfaccia (il
    nome intero, per esempio "Windows WASAPI"), breve ("WASAPI"), latenza in millesimi di secondo, o None, ed esclusiva.
    Ci sono tutte le interfacce, WDM-KS compreso, perché il motore apre il flusso a callback. In ordine di latenza
    crescente; dentro la stessa latenza resta l'ordine di GBUtils: prima la scheda che Windows usa già, poi per preferenza
    d'interfaccia e per indice."""
    from GBUtils import elenco_dispositivi_audio

    tutte = [{"indice": voce["indice"], "dispositivo": _ripulito(voce["dispositivo"]), "interfaccia": voce["interfaccia"], "breve": voce["breve"],
              "latenza": voce["latenza"], "esclusiva": voce["esclusiva"]} for voce in elenco_dispositivi_audio(prova="nessuno", modo="callback")]
    # sorted è stabile: basta la latenza, il resto lo porta l'ordine di GBUtils.
    return sorted(tutte, key=lambda uscita: (uscita["latenza"] is None, uscita["latenza"] or 0.0))


def _millisecondi(latenza):
    """3 ms, 23,2 ms: il decimale solo se serve, con la virgola."""
    decimi = round(latenza, 1)
    numero = testi.numero(decimi, 0 if decimi == int(decimi) else 1)
    # Traduttori: la latenza di una scheda audio, in millisecondi.
    return _("{numero} ms").format(numero=numero)


def e_vivavoce(dispositivo):
    """Vero se il nome è quello del vivavoce di cuffie Bluetooth, che suona come un telefono."""
    nome = _ripulito(dispositivo).lower()
    return any(parola in nome for parola in NOMI_DEL_VIVAVOCE)


def nome_della_scheda(scheda):
    """Il nome di una scheda da leggere, "Altoparlanti (Realtek(R) Audio), WASAPI": il dispositivo e l'interfaccia col nome
    breve, senza Windows davanti. scheda è un'uscita di uscite(), una scelta salvata o l'esito del motore; il testo è vuoto
    se scheda non dice un dispositivo, come la scelta automatica."""
    if not scheda or not scheda.get("dispositivo"):
        return ""
    interfaccia = scheda.get("breve") or (scheda.get("interfaccia") or "").removeprefix("Windows ")
    return ", ".join(parte for parte in (_ripulito(str(scheda["dispositivo"])), interfaccia) if parte)


def etichetta(uscita):
    """La riga che si legge nell'elenco: "Altoparlanti (Realtek(R) Audio), WASAPI, 3 ms"; per le interfacce esclusive,
    ASIO e WDM-KS, in coda l'avviso che possono zittire NVDA, e per il vivavoce delle cuffie Bluetooth l'avviso che suona
    come un telefono."""
    parti = [nome_della_scheda(uscita)]
    if uscita.get("latenza") is not None:
        parti.append(_millisecondi(uscita["latenza"]))
    if uscita.get("esclusiva"):
        # Traduttori: in coda alla riga di una scheda audio con un'interfaccia esclusiva, come ASIO.
        parti.append(_("esclusiva: può zittire NVDA"))
    if e_vivavoce(uscita["dispositivo"]):
        # Traduttori: in coda alla riga del vivavoce di cuffie Bluetooth, che Windows apre per le telefonate.
        parti.append(_("vivavoce: suona come un telefono"))
    return ", ".join(parti)


def valida(scelta):
    """La scelta salvata, ripulita: {"dispositivo", "interfaccia"} coi due nomi su una riga sola, se sono due testi non
    vuoti; altrimenti il dizionario vuoto, la scelta automatica. Serve alle impostazioni, che non si fidano di un file
    scritto a mano, e al motore. Restituisce sempre un dizionario nuovo."""
    if not isinstance(scelta, dict):
        return {}
    pulita = {}
    for chiave in CHIAVI_DELLA_SCHEDA:
        valore = scelta.get(chiave)
        if not isinstance(valore, str) or not _ripulito(valore):
            return {}
        pulita[chiave] = _ripulito(valore)
    return pulita


def da_salvare(uscita):
    """La scelta da scrivere nelle impostazioni: dispositivo e interfaccia dell'uscita, o il dizionario vuoto per
    l'automatica, con uscita None."""
    return {"dispositivo": uscita["dispositivo"], "interfaccia": uscita["interfaccia"]} if uscita else {}


def ritrova(scelta, elenco):
    """L'uscita dell'elenco con lo stesso dispositivo e la stessa interfaccia della scelta salvata, o None: la scheda non c'è
    più, o la scelta è vuota."""
    scelta = valida(scelta)
    if not scelta:
        return None
    for uscita in elenco:
        if _ripulito(uscita["dispositivo"]) == scelta["dispositivo"] and uscita["interfaccia"] == scelta["interfaccia"]:
            return uscita
    return None


def rileggi():
    """Rilegge le uscite di PortAudio con rileggi_dispositivi_audio di GBUtils, che lo chiude e lo riapre, e ne restituisce
    l'esito: prima e dopo, quante uscite c'erano e quante ce ne sono; arrivate e sparite, le coppie (dispositivo,
    interfaccia). Gli indici cambiano: chi se ne teneva uno lo richiede a uscite(). Va chiamata soltanto col flusso del motore
    chiuso, come fa Motore.cambia_scheda con rileggi vero. Nelle prove si sostituisce."""
    from GBUtils import rileggi_dispositivi_audio

    return rileggi_dispositivi_audio()
