"""
Le prove di partita.py: l'inizio con le voci dei bersagli, la mira al campione udibile, lo sparo e
l'impatto al campione esatto, i tiri a vuoto fino alla fine, il livello, i bonus e i loro punti (D13),
gli avvisi in fila dopo l'impatto, la pausa chiesta e quella automatica, anche nella mira (D14),
l'abbandono, i primati, la barra di stato, le statistiche, e il sommario in testa al resoconto vero,
al grado della palestra del livello del tiro (D21).
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
Nessuna prova apre la scheda audio né sintetizza un preset vero: il motore gira sul flusso finto di
aiuti_audio, che chiama il callback soltanto quando la prova lo chiede, e _rendi_preset dà a ogni
preset lo stesso buffer breve. Per premere spazio a un campione preciso la prova fa preparare al
motore il blocco che lo contiene, poi sposta l'orologio finto finché il campione udibile è proprio
quello: così la mira si prova al campione, senza il caso del tempo vero.
"""

import math
import random

import numpy as np
import pytest
from aiuti_audio import motore_finto

import partita as modulo_partita
import resoconti
import suoni
import testi
from bersagli import Bersaglio, durata_del_bonus, genera, genera_bonus, portanti_senza, quanti_bersagli, raggio_del_livello, velocita_del_livello
from costanti import (
    ATTENUAZIONE_IN_MIRA_DB,
    BERSAGLIO_PORTANTI,
    CENTRI_PER_LIVELLO,
    COLPI_BONUS,
    COLPI_INIZIALI,
    COLPI_PER_LIVELLO,
    COMPENSAZIONE_DEL_TASTO,
    ESPONENTE_DEL_RAGGIO,
    FS,
    LIVELLO_DEI_BONUS,
    MARGINE_DOPO_LA_SCIVOLATA,
    MOLTIPLICATORE_BONUS,
    PARAMETRI,
    PAUSA_FRA_BARRE,
    PREMIO_PRECISIONE,
    PROIETTILE_FORMA,
    PROIETTILE_FORME,
    PUNTI_BASE,
    RAGGIO_DI_RIFERIMENTO,
    SFUMATURA_STOP,
    SOMMARIO_NUMERI_FINO_AL_LIVELLO,
    SOMMARIO_PAROLE_FINO_AL_LIVELLO,
    VELOCITA_DEL_TEMPO,
    VELOCITA_DI_RIFERIMENTO,
)
from fisica import Traiettoria, parametri_del_tiro, soluzione
from motore_audio import VoceTono
from partita import Messaggio, Partita, punti_del_centro
from sintesi import db_in_lineare
from statistiche import Statistiche
from suoni import Suoni

SEME = 73
# Il buffer finto di ogni preset: breve e costante, così un effetto si riconosce nell'uscita e non
# copre a lungo le voci dei bersagli.
LUNGHEZZA_FINTA = 256
AMPIEZZA_FINTA = 0.1
# Quanti blocchi al massimo si aspetta un evento, prima di dire che non arriva: 30 secondi di suono.
BLOCCHI_MASSIMI = 30 * FS // 512
SFUMATURA = round(SFUMATURA_STOP * FS)


class Orologio:
    """L'orologio vero finto della partita: va avanti soltanto a mano."""

    def __init__(self, valore=0.0):
        self.valore = float(valore)

    def __call__(self):
        return self.valore


class Banco:
    """Il motore sul flusso finto, i suoni coi preset finti e l'elenco degli eventi sonori contati."""

    def __init__(self):
        self.motore, self.flusso = motore_finto()
        self.contati = []
        self.suoni = Suoni(self.motore, conta=self.contati.append)

    def partita(self, statistiche=None, orologio=None, seme=SEME, proiettile=None):
        return Partita(self.motore, self.suoni, statistiche=statistiche, rng=random.Random(seme), orologio=orologio or Orologio(), proiettile=proiettile)


@pytest.fixture
def banco(monkeypatch):
    buffer = np.full((LUNGHEZZA_FINTA, 2), AMPIEZZA_FINTA, dtype=np.float32)
    buffer.flags.writeable = False
    monkeypatch.setattr(suoni, "_rendi_preset", lambda nome: buffer)
    banco = Banco()
    yield banco
    banco.motore.chiudi()


@pytest.fixture
def bonus_sicuro(monkeypatch):
    """Un bonus a ogni nuova fase d'ascolto, dal suo livello."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 1.0)


# Gli aiuti delle prove.

def porta_udibile_a(banco, campione):
    """Fa preparare al motore il blocco che contiene il campione, poi sposta l'orologio finto perché
    il campione udibile sia proprio quello."""
    banco.flusso.avanza_fino(math.ceil(campione))
    inizio, dac = banco.motore._riferimento
    banco.flusso.time = dac + (campione - inizio) / banco.flusso.fs
    assert banco.motore.campione_udibile() == pytest.approx(campione, abs=1e-6)


def campione_del_valore(partita, valore):
    """Il campione udibile a cui lo spazio sceglie quel valore nella barra in corso."""
    barra = partita.scivolate[PARAMETRI[partita.barra]]
    return partita.inizio_barra + barra.inizio_scivolata + valore * barra.campioni_scivolata + COMPENSAZIONE_DEL_TASTO * barra.fs


def mira_e_spara(banco, partita, valori):
    """Con la mira avviata, preme spazio ai campioni che scelgono i tre valori."""
    for indice, valore in enumerate(valori):
        assert partita.fase == "mira" and partita.barra == indice
        porta_udibile_a(banco, campione_del_valore(partita, valore))
        assert partita.battito() == []
        assert partita.spazio() == []
    assert partita.fase == "volo"


def fino_alla_risoluzione(banco, partita):
    """Fa avanzare il motore col battito finché il tiro si risolve, e restituisce i messaggi."""
    messaggi = []
    for _blocco in range(BLOCCHI_MASSIMI):
        banco.flusso.avanza(1)
        messaggi.extend(partita.battito())
        if partita.fase != "volo":
            return messaggi
    raise AssertionError("il tiro non si è risolto")


def colpisci(banco, partita, bersaglio=None):
    """Avvia la mira e spara al centro di un bersaglio fermo, con la soluzione della fisica."""
    assert partita.fase == "ascolto"
    partita.spazio()
    partenza = banco.motore.tempo_di_gioco(banco.motore.prossimo_campione())
    if bersaglio is None:
        bersaglio = next(b for b in partita.bersagli if not b.bonus)
    valori = soluzione(bersaglio.posizione(partenza))
    assert valori is not None
    mira_e_spara(banco, partita, valori)
    return valori


def tiro_a_vuoto(banco, partita):
    """Un tiro tutto a sinistra, rasoterra e piano, che cade a pochi metri dal cannone, e la sua
    risoluzione."""
    partita.spazio()
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    assert partita.ultimo_volo.colpito is None
    return fino_alla_risoluzione(banco, partita)


def senti_anche_le_statistiche(banco, statistiche):
    """Con le statistiche la partita collega a loro il conto dei suoni: la prova lo vuole anche
    nell'elenco del banco, per l'ordine in cui i suoni arrivano."""
    def conta(evento):
        banco.contati.append(evento)
        statistiche.suono(evento)

    banco.suoni.conta = conta


def mira_sul_bonus(bonus, partenza):
    """I valori che colpiscono il bonus in moto: il punto dove sarà alla fine del volo, ricalcolato
    finché il tempo di volo non si assesta."""
    tempo = 0.0
    valori = None
    for _giro in range(30):
        punto = bonus.posizione(partenza + tempo)
        valori = soluzione(punto)
        assert valori is not None
        traiettoria = Traiettoria(*parametri_del_tiro(*valori))
        orizzontale = math.hypot(traiettoria.iniziale[0], traiettoria.iniziale[2])
        tempo = math.hypot(punto[0], punto[2]) / orizzontale
    return valori


# Le prove.

def test_punti_del_centro():
    pieno = Bersaglio(numero=1, forma="seno", raggio=RAGGIO_DI_RIFERIMENTO, partenza=(0.0, 20.0, 50.0))
    assert punti_del_centro(pieno, 0.0) == PUNTI_BASE
    assert punti_del_centro(pieno, None) == PUNTI_BASE
    assert punti_del_centro(pieno, 1.0) == round(PUNTI_BASE * (1 + PREMIO_PRECISIONE))
    assert punti_del_centro(pieno, 2.0) == punti_del_centro(pieno, 1.0)
    piccolo = Bersaglio(numero=2, forma="seno", raggio=RAGGIO_DI_RIFERIMENTO / 2, partenza=(0.0, 20.0, 50.0), velocita=(VELOCITA_DI_RIFERIMENTO, 0.0, 0.0))
    assert punti_del_centro(piccolo, 0.5) == round(PUNTI_BASE * 2 ** ESPONENTE_DEL_RAGGIO * 2 * (1 + PREMIO_PRECISIONE * 0.5))
    # D13: il bonus vale esattamente MOLTIPLICATORE_BONUS volte un normale del livello in cui è nato, col
    # raggio e la velocità di quel livello; il suo raggio e la sua corsa non contano.
    bonus = Bersaglio(numero=3, forma="seno", raggio=RAGGIO_DI_RIFERIMENTO / 4, partenza=(0.0, 20.0, 50.0), velocita=(9.0, 0.0, 0.0), bonus=True)
    assert raggio_del_livello(1) == RAGGIO_DI_RIFERIMENTO and velocita_del_livello(1) == 0.0
    assert punti_del_centro(bonus, 0.0, livello=1) == PUNTI_BASE * MOLTIPLICATORE_BONUS
    with pytest.raises(ValueError):
        punti_del_centro(bonus, 0.0)
    # Un normale non ha bisogno del livello, e se lo riceve non cambia.
    assert punti_del_centro(piccolo, 0.5, livello=20) == punti_del_centro(piccolo, 0.5)
    rng = random.Random(SEME)
    for livello in (LIVELLO_DEI_BONUS, 4, 6, 10, 15, 25):
        normale = genera(livello, rng, 1, 0.0)
        speciale = genera_bonus(livello, rng, 2, 0.0)
        for precisione in (0.0, 0.37, 0.5, 1.0):
            assert punti_del_centro(speciale, precisione, livello=livello) == MOLTIPLICATORE_BONUS * punti_del_centro(normale, precisione)


def test_inizio_della_partita_e_voci_dei_bersagli(banco):
    partita = banco.partita()
    assert partita.fase == "pronta" and not partita.in_corso()
    k = banco.motore.prossimo_campione()
    messaggi = partita.nuova()
    assert partita.fase == "ascolto" and partita.in_corso()
    assert (partita.livello, partita.riserva, partita.punti, partita.tiri, partita.centri, partita.centri_del_livello, partita.serie) == (1, COLPI_INIZIALI, 0, 0, 0, 0, 0)
    assert len(partita.bersagli) == quanti_bersagli(1)
    assert partita.bonus is None
    for bersaglio in partita.bersagli:
        voce = partita._voci[bersaglio]
        assert isinstance(voce, VoceTono)
        # D18: il bersaglio suona nella sua portante, mai in quella del proiettile, che di serie è PROIETTILE_FORMA.
        assert voce.forma == bersaglio.portante in portanti_senza(PROIETTILE_FORMA) and voce.inizio == k
        assert voce.parametri == bersaglio.parametri_sonori
        assert bersaglio.nato == banco.motore.tempo_di_gioco(k)
    assert banco.motore.tempo_scorre
    assert banco.contati == ["nuova_partita"]
    assert [m.tipo for m in messaggi] == ["info"]
    assert isinstance(messaggi[0], Messaggio)
    assert messaggi[0].annuncio == "Partita nuova: Spazio per mirare."
    # Lo stesso seme dà gli stessi bersagli.
    gemello = genera(1, random.Random(SEME), 1, partita.bersagli[0].nato, portanti=portanti_senza(PROIETTILE_FORMA))
    assert np.allclose(gemello.partenza, partita.bersagli[0].partenza) and gemello.forma == partita.bersagli[0].forma
    assert gemello.portante == partita.bersagli[0].portante and gemello.parametri_lfo == partita.bersagli[0].parametri_lfo
    # Finito l'effetto della partita nuova, ciò che si sente è la voce del bersaglio.
    uscita = banco.flusso.avanza(20)
    assert not uscita[:k].any()
    assert np.abs(uscita[k + LUNGHEZZA_FINTA + 64:]).max() > 0


def test_mira_completa_sul_centro_del_bersaglio(banco):
    partita = banco.partita()
    partita.nuova()
    banco.flusso.avanza(10)
    bersaglio = partita.bersagli[0]
    valori = colpisci(banco, partita)
    volo = partita.ultimo_volo
    assert partita.valori == pytest.approx(list(valori), abs=1e-9)
    assert partita.tiri == 1 and partita.fase == "volo"
    assert volo.colpito is bersaglio and volo.precisione > 0.99
    # L'effetto dell'impatto si conta quando si sente, non allo sparo.
    assert "colpito" not in banco.contati
    messaggi = fino_alla_risoluzione(banco, partita)
    attesi = round(PUNTI_BASE * (RAGGIO_DI_RIFERIMENTO / bersaglio.raggio) ** ESPONENTE_DEL_RAGGIO * (1 + PREMIO_PRECISIONE * volo.precisione))
    assert partita.punti == attesi == punti_del_centro(bersaglio, volo.precisione)
    assert (partita.centri, partita.serie, partita.centri_del_livello, partita.riserva) == (1, 1, 1, COLPI_INIZIALI)
    assert partita.fase == "ascolto"
    assert bersaglio not in partita.bersagli and bersaglio not in partita._voci
    assert len(partita.bersagli) == quanti_bersagli(1) and partita.bersagli[0] in partita._voci
    resoconto = messaggi[0]
    # D21: al livello 1 il sommario coi numeri apre il resoconto e l'annuncio; il centro è pieno, quindi
    # gli scarti dal centro si arrotondano a zero.
    sommario = resoconti.riga_del_sommario(volo, "numeri")
    assert sommario.startswith("Laterale 0,0 metri, verticale 0,0, profondità 0,0, inclinazione ")
    assert resoconto.testo.split("\n")[0] == sommario and resoconto.testo.split("\n")[1].startswith("Tiro 1, livello 1: colpito un bersaglio, ")
    assert resoconto.tipo == "resoconto" and resoconto.annuncio == f"{sommario} Colpito, {attesi} punti."
    assert resoconto.dati["colpito"] and resoconto.dati["punti"] == attesi and resoconto.dati["totale"] == attesi
    assert banco.contati == ["nuova_partita", "mira_avviata", "valore_preso", "valore_preso", "valore_preso", "sparo", "colpito"]


def test_impatto_al_campione_esatto(banco, monkeypatch):
    partita = banco.partita()
    partita.nuova()
    bersaglio = partita.bersagli[0]
    voce_bersaglio = partita._voci[bersaglio]
    chiamate = []
    originale = banco.motore.suona

    def spia(buffer, inizio=None, guadagno=1.0):
        chiamate.append((np.array(buffer), inizio))
        return originale(buffer, inizio=inizio, guadagno=guadagno)

    monkeypatch.setattr(banco.motore, "suona", spia)
    colpisci(banco, partita)
    volo = partita.ultimo_volo
    proiettile = partita._voce_proiettile
    evento = partita._evento_impatto
    k = proiettile.inizio
    assert isinstance(proiettile, VoceTono) and proiettile.forma == PROIETTILE_FORMA == partita.proiettile
    assert evento.tempo == pytest.approx(volo.fine) and volo.partenza == pytest.approx(banco.motore.tempo_di_gioco(k))
    # Lo sparo parte allo stesso campione del proiettile.
    assert any(inizio == k for _buffer, inizio in chiamate)
    while partita.fase == "volo":
        banco.flusso.avanza(1)
        udibile = banco.motore.campione_udibile()
        messaggi = partita.battito()
        if partita.fase == "volo":
            assert not (evento.scattato and udibile >= evento.campione)
        else:
            assert udibile >= evento.campione
    assert messaggi[0].tipo == "resoconto"
    atteso = k + math.ceil((volo.fine - volo.partenza) * FS / VELOCITA_DEL_TEMPO)
    assert evento.scattato and abs(evento.campione - atteso) <= 1
    banco.flusso.avanza(2)
    assert proiettile.finita and proiettile.finita_al == evento.campione + SFUMATURA
    assert voce_bersaglio.finita and voce_bersaglio.finita_al == evento.campione + SFUMATURA
    effetti = [buffer for buffer, inizio in chiamate if inizio == evento.campione]
    assert len(effetti) == 1
    np.testing.assert_allclose(effetti[0], banco.suoni.pronto("colpito", posizione=volo.punto_finale))


def test_spazio_nella_pausa_non_conta(banco):
    partita = banco.partita()
    partita.nuova()
    partita.spazio()
    assert partita.fase == "mira" and partita.barra == 0
    porta_udibile_a(banco, partita.inizio_barra - 100)
    assert partita.battito() == [] and partita.sottofase == "pausa"
    prima = list(banco.contati)
    assert partita.spazio() == []
    assert partita.spazio() == []
    assert partita.valori == [] and partita.barra == 0 and banco.contati == prima


def test_spazio_nei_bip_vale_il_minimo(banco):
    partita = banco.partita()
    partita.nuova()
    partita.spazio()
    barra = partita.scivolate["direzione"]
    porta_udibile_a(banco, partita.inizio_barra + barra.inizio_scivolata / 2)
    assert partita.battito() == [] and partita.sottofase == "conto"
    assert partita.spazio() == []
    assert partita.valori == [0.0] and partita.barra == 1 and partita.sottofase == "pausa"
    assert banco.contati[-1] == "valore_preso"


def test_scivolate_non_fermate_valgono_il_massimo(banco):
    partita = banco.partita()
    partita.nuova()
    partita.spazio()
    barra = partita.scivolate["direzione"]
    fine = partita.inizio_barra + barra.fine_scivolata
    porta_udibile_a(banco, fine + MARGINE_DOPO_LA_SCIVOLATA * FS - 1)
    assert partita.battito() == [] and partita.valori == [] and partita.sottofase == "scivolata"
    porta_udibile_a(banco, fine + MARGINE_DOPO_LA_SCIVOLATA * FS)
    assert partita.battito() == []
    assert partita.valori == [1.0] and partita.barra == 1
    # Le altre due barre arrivano al massimo da sole, e parte il colpo (D3).
    for _blocco in range(BLOCCHI_MASSIMI):
        if partita.fase != "mira":
            break
        banco.flusso.avanza(1)
        partita.battito()
    assert partita.fase == "volo" and partita.valori == [1.0, 1.0, 1.0]
    assert banco.contati.count("valore_preso") == 3


def test_tempo_fermo_e_bersagli_attenuati_nella_mira(banco):
    partita = banco.partita()
    partita.nuova()
    banco.flusso.avanza(20)
    assert banco.motore.tempo_di_gioco() > 0
    voce = partita._voci[partita.bersagli[0]]
    partita.spazio()
    banco.flusso.avanza(4)
    fermo = banco.motore.tempo_di_gioco()
    banco.flusso.avanza(40)
    assert banco.motore.tempo_di_gioco() == fermo and not banco.motore.tempo_scorre
    assert voce.guadagno == pytest.approx(db_in_lineare(ATTENUAZIONE_IN_MIRA_DB))
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    assert partita.ultimo_volo.partenza == pytest.approx(fermo)
    banco.flusso.avanza(4)
    assert voce.guadagno == 1.0 and banco.motore.tempo_scorre
    assert banco.motore.tempo_di_gioco() > fermo


def test_tiri_a_vuoto_fino_alla_fine(banco):
    orologio = Orologio(100.0)
    partita = banco.partita(orologio=orologio)
    partita.nuova()
    voci = list(partita._voci.values())
    for numero in range(1, COLPI_INIZIALI + 1):
        if numero == COLPI_INIZIALI:
            orologio.valore = 160.0
        messaggi = tiro_a_vuoto(banco, partita)
        assert partita.tiri == numero and partita.riserva == COLPI_INIZIALI - numero
        assert messaggi[0].tipo == "resoconto" and not messaggi[0].dati["colpito"]
        sommario = resoconti.riga_del_sommario(partita.ultimo_volo, "numeri")
        assert sommario and messaggi[0].testo.split("\n")[0] == sommario and messaggi[0].annuncio.startswith(f"{sommario} A vuoto, ")
        if partita.riserva > 1:
            assert [m.tipo for m in messaggi] == ["resoconto"]
            assert messaggi[0].annuncio.startswith(f"{sommario} A vuoto, riserva {partita.riserva}, ")
        elif partita.riserva == 1:
            assert [m.tipo for m in messaggi] == ["resoconto", "avviso"]
            assert messaggi[1].annuncio == "Ultimo colpo."
    assert partita.fase == "finita" and not partita.in_corso()
    assert [m.tipo for m in messaggi] == ["resoconto", "fine"]
    assert messaggi[0].annuncio == f"{sommario} A vuoto, riserva vuota."
    riepilogo = {"punteggio": 0, "livello": 1, "tiri": COLPI_INIZIALI, "centri": 0, "durata_s": 60.0}
    assert messaggi[1].dati == riepilogo == partita.riepilogo()
    assert messaggi[1].annuncio == "Partita finita, 0 punti."
    assert banco.contati.count("ultimo_colpo") == 1 and banco.contati.count("partita_finita") == 1
    assert banco.contati.count("a_terra") == COLPI_INIZIALI
    banco.flusso.avanza(2)
    assert all(voce.finita for voce in voci) and partita.bersagli == []
    # Il tempo vero che passa dopo la fine non allunga la partita.
    orologio.valore = 500.0
    assert partita.riepilogo()["durata_s"] == 60.0
    assert partita.spazio()[-1].annuncio == "Partita nuova: Spazio per mirare."
    assert partita.fase == "ascolto" and partita.tiri == 0 and partita.riserva == COLPI_INIZIALI


def test_il_livello_sale_dopo_i_centri(banco):
    partita = banco.partita()
    partita.nuova()
    tiro_a_vuoto(banco, partita)
    for _centro in range(CENTRI_PER_LIVELLO):
        vecchi = list(partita._voci.values())
        colpisci(banco, partita)
        assert partita.ultimo_volo.colpito is not None
        messaggi = fino_alla_risoluzione(banco, partita)
    riserva = COLPI_INIZIALI - 1 + COLPI_PER_LIVELLO
    assert partita.livello == 2 and partita.riserva == riserva
    assert partita.centri == CENTRI_PER_LIVELLO and partita.centri_del_livello == 0 and partita.tiri == CENTRI_PER_LIVELLO + 1
    assert [m.tipo for m in messaggi][:2] == ["resoconto", "livello"]
    assert messaggi[1].annuncio == f"Livello 2, riserva {riserva}."
    assert banco.contati.count("livello_su") == 1
    assert len(partita.bersagli) == quanti_bersagli(2)
    assert all(b.raggio == raggio_del_livello(2) for b in partita.bersagli)
    assert all(b in partita._voci for b in partita.bersagli)
    banco.flusso.avanza(2)
    assert all(voce.finita for voce in vecchi)
    assert partita.righe_della_barra()[1] == f"Riserva {riserva}, centri 0 su {CENTRI_PER_LIVELLO}"


def test_il_bonus_nasce_ed_esce(banco, bonus_sicuro):
    partita = banco.partita()
    partita.nuova()
    partita.livello = LIVELLO_DEI_BONUS
    messaggi = tiro_a_vuoto(banco, partita)
    bonus = partita.bonus
    assert bonus is not None and bonus.bonus
    assert bonus in partita.bersagli and bonus in partita._voci
    assert messaggi[-1].tipo == "bonus" and messaggi[-1].annuncio in ("Bonus da sinistra.", "Bonus da destra.")
    assert banco.contati.count("bonus_appare") == 1 and "bonus_uscito" not in banco.contati
    assert partita.righe_della_barra()[2] == "Bersagli in volo: 1 e il bonus"
    evento = partita._evento_bonus
    assert evento.tempo == pytest.approx(bonus.tempo_di_uscita)
    voce = partita._voci[bonus]
    usciti = []
    for _blocco in range(BLOCCHI_MASSIMI):
        banco.flusso.avanza(1)
        usciti.extend(partita.battito())
        if usciti:
            break
    assert [m.tipo for m in usciti] == ["bonus"] and usciti[0].annuncio == "Bonus uscito."
    assert evento.scattato and banco.motore.campione_udibile() >= evento.campione
    assert partita.bonus is None and bonus not in partita.bersagli and bonus not in partita._voci
    assert banco.contati.count("bonus_uscito") == 1
    assert voce.finita and voce.finita_al == evento.campione + SFUMATURA
    assert partita.righe_della_barra()[2] == "Bersagli in volo: 1"


def test_il_bonus_colpito_da_i_colpi_e_annulla_l_uscita(banco, bonus_sicuro, monkeypatch):
    partita = banco.partita()
    partita.nuova()
    partita.livello = LIVELLO_DEI_BONUS
    tiro_a_vuoto(banco, partita)
    bonus = partita.bonus
    evento = partita._evento_bonus
    assert bonus is not None
    # A metà della traversata il bonus sta in mezzo al cubo, dove ogni punto si raggiunge.
    banco.flusso.avanza_secondi(durata_del_bonus(LIVELLO_DEI_BONUS) / 2)
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    riserva = partita.riserva
    partita.spazio()
    partenza = banco.motore.tempo_di_gioco(banco.motore.prossimo_campione())
    mira_e_spara(banco, partita, mira_sul_bonus(bonus, partenza))
    volo = partita.ultimo_volo
    assert volo.colpito is bonus
    assert evento.annullato
    messaggi = fino_alla_risoluzione(banco, partita)
    assert partita.riserva == riserva + COLPI_BONUS
    # D13: cinque volte un normale del livello in cui il bonus è nato, con la stessa precisione.
    normale = round(PUNTI_BASE * (RAGGIO_DI_RIFERIMENTO / raggio_del_livello(LIVELLO_DEI_BONUS)) ** ESPONENTE_DEL_RAGGIO
                    * (1 + velocita_del_livello(LIVELLO_DEI_BONUS) / VELOCITA_DI_RIFERIMENTO) * (1 + PREMIO_PRECISIONE * volo.precisione))
    attesi = MOLTIPLICATORE_BONUS * normale
    assert partita.punti == attesi == punti_del_centro(bonus, volo.precisione, livello=LIVELLO_DEI_BONUS)
    assert partita.bonus is None and bonus not in partita.bersagli
    assert len(partita.bersagli) == quanti_bersagli(LIVELLO_DEI_BONUS)
    # Il tiro avviene al livello dei bonus, che è ancora della palestra coi numeri (D21).
    assert messaggi[0].annuncio == f"{resoconti.riga_del_sommario(volo, resoconti.grado_della_palestra(LIVELLO_DEI_BONUS))} Colpito il bonus, {testi.intero(attesi)} punti."
    assert messaggi[0].dati["bonus"]
    assert "colpito_bonus" in banco.contati and "bonus_uscito" not in banco.contati
    # L'uscita annullata non scatta mai, nemmeno quando il suo tempo passa.
    while banco.motore.tempo_di_gioco() <= evento.tempo + 0.5:
        banco.flusso.avanza(10)
    assert not evento.scattato


def test_il_sommario_in_testa_al_resoconto_ai_tre_gradi(banco, monkeypatch):
    """D21: il resoconto vero comincia col sommario del passaggio del volo, al grado della palestra del
    livello del tiro: i numeri nei primi livelli, poi le sole parole, poi niente; e apre l'annuncio."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    partita = banco.partita()
    partita.nuova()
    numeri, parole = SOMMARIO_NUMERI_FINO_AL_LIVELLO, SOMMARIO_PAROLE_FINO_AL_LIVELLO
    for livello, grado in ((1, "numeri"), (numeri, "numeri"), (numeri + 1, "parole"), (parole, "parole"), (parole + 1, "niente")):
        partita.livello = livello
        messaggi = tiro_a_vuoto(banco, partita)
        volo = partita.ultimo_volo
        assert volo.vicino is not None and volo.passaggio is not None and volo.scarto == volo.passaggio.scarto
        righe = messaggi[0].testo.split("\n")
        esito = f"Tiro {partita.tiri}, livello {livello}: a vuoto; nella riserva restano {partita.riserva} colpi."
        sommario = resoconti.riga_del_sommario(volo, grado)
        assert resoconti.grado_della_palestra(livello) == grado
        if grado == "niente":
            assert sommario == "" and righe[0] == esito
            assert messaggi[0].annuncio.startswith(f"A vuoto, riserva {partita.riserva}")
            continue
        assert righe[:2] == [sommario, esito]
        assert messaggi[0].annuncio.startswith(f"{sommario} A vuoto, riserva {partita.riserva}")
        if grado == "numeri":
            assert sommario.startswith("Laterale ") and ", inclinazione " in sommario and sommario.endswith(" gradi.")
        else:
            frase = ", ".join(resoconti.parole_del_passaggio(volo.passaggio))
            assert sommario == f"{frase[:1].upper()}{frase[1:]}."


def test_il_sommario_del_bonus_segue_il_livello_del_tiro(banco, bonus_sicuro, monkeypatch):
    """D21: per il bonus il grado del sommario è quello del livello in cui si spara, mentre i punti
    restano quelli del livello in cui il bonus è nato (D13). La prova alza il livello a bonus in volo,
    a mano, per separare i due livelli."""
    partita = banco.partita()
    partita.nuova()
    partita.livello = LIVELLO_DEI_BONUS
    tiro_a_vuoto(banco, partita)
    bonus = partita.bonus
    assert bonus is not None
    banco.flusso.avanza_secondi(durata_del_bonus(LIVELLO_DEI_BONUS) / 2)
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    nato = resoconti.grado_della_palestra(LIVELLO_DEI_BONUS)
    livello_del_tiro = next((n for n in range(LIVELLO_DEI_BONUS + 1, LIVELLO_DEI_BONUS + 30) if resoconti.grado_della_palestra(n) != nato), LIVELLO_DEI_BONUS + 1)
    grado = resoconti.grado_della_palestra(livello_del_tiro)
    partita.livello = livello_del_tiro
    partita.spazio()
    partenza = banco.motore.tempo_di_gioco(banco.motore.prossimo_campione())
    mira_e_spara(banco, partita, mira_sul_bonus(bonus, partenza))
    volo = partita.ultimo_volo
    assert volo.colpito is bonus and volo.passaggio is not None
    messaggi = fino_alla_risoluzione(banco, partita)
    assert partita.punti == punti_del_centro(bonus, volo.precisione, livello=LIVELLO_DEI_BONUS)
    righe = messaggi[0].testo.split("\n")
    esito = f"Tiro {partita.tiri}, livello {livello_del_tiro}: colpito il bonus, "
    sommario = resoconti.riga_del_sommario(volo, grado)
    if sommario:
        assert righe[0] == sommario and righe[1].startswith(esito) and messaggi[0].annuncio.startswith(f"{sommario} Colpito il bonus, ")
        assert sommario != resoconti.riga_del_sommario(volo, nato)
    else:
        assert righe[0].startswith(esito) and messaggi[0].annuncio.startswith("Colpito il bonus, ")


def test_pausa_e_ripresa_nell_ascolto(banco):
    orologio = Orologio(0.0)
    partita = banco.partita(orologio=orologio)
    partita.nuova()
    banco.flusso.avanza(10)
    voce = partita._voci[partita.bersagli[0]]
    orologio.valore = 10.0
    messaggi = partita.pausa()
    assert partita.in_pausa and messaggi[0].annuncio == "Pausa." and banco.contati[-1] == "pausa"
    banco.flusso.avanza(5)
    fermo = banco.motore.tempo_di_gioco()
    banco.flusso.avanza(30)
    assert banco.motore.tempo_di_gioco() == fermo and voce.guadagno == 0.0
    assert partita.spazio() == [] and banco.contati[-1] == "non_ora" and partita.fase == "ascolto"
    assert partita.righe_della_barra()[-1] == "In pausa: Ctrl+P per riprendere"
    orologio.valore = 70.0
    messaggi = partita.pausa()
    assert not partita.in_pausa and messaggi[0].annuncio == "Si riprende." and banco.contati[-1] == "ripresa"
    banco.flusso.avanza(10)
    assert banco.motore.tempo_di_gioco() > fermo and voce.guadagno == 1.0
    orologio.valore = 100.0
    assert partita.riepilogo()["durata_s"] == 40.0
    # Senza partita non c'è niente da mettere in pausa, e lo dice il suono del non ora.
    nessuna = banco.partita()
    messaggi = nessuna.pausa()
    assert [m.annuncio for m in messaggi] == ["Nessuna partita in corso."] and banco.contati[-1] == "non_ora" and not nessuna.in_pausa
    # Lo stesso a partita finita.
    nessuna.fase = "finita"
    prima = len(banco.contati)
    assert nessuna.pausa()[0].annuncio == "Nessuna partita in corso." and banco.contati[prima:] == ["non_ora"]


def test_pausa_nella_mira(banco, cartella_di_prova):
    """Ctrl+P nella mira (D14): la scivolata in corso si ferma, i bersagli tacciono e la barra non
    scade; alla ripresa la barra in corso ricomincia da capo, col suo secondo di silenzio, i bersagli
    tornano attenuati e i valori già presi restano."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    orologio = Orologio(0.0)
    partita = banco.partita(statistiche=statistiche, orologio=orologio)
    senti_anche_le_statistiche(banco, statistiche)
    partita.nuova()
    banco.flusso.avanza(10)
    voce = partita._voci[partita.bersagli[0]]
    partita.spazio()
    porta_udibile_a(banco, campione_del_valore(partita, 0.3))
    assert partita.spazio() == [] and partita.valori == [pytest.approx(0.3)] and partita.barra == 1
    porta_udibile_a(banco, campione_del_valore(partita, 0.2))
    assert partita.battito() == [] and partita.sottofase == "scivolata"
    barra = partita._voce_barra
    fermo = banco.motore.tempo_di_gioco()
    orologio.valore = 5.0
    messaggi = partita.pausa()
    assert partita.in_pausa and partita.fase == "mira" and banco.contati[-1] == "pausa"
    assert [m.annuncio for m in messaggi] == ["Pausa."] and "ricomincia da capo" in messaggi[0].testo
    assert partita.righe_della_barra()[-1] == "In pausa: Ctrl+P per riprendere"
    banco.flusso.avanza(3)
    assert barra.finita and voce.guadagno == 0.0
    # La barra ferma non scade, e lo spazio non sceglie niente: fa sentire il non ora.
    for _blocco in range(10 * FS // banco.motore.blocco):
        banco.flusso.avanza(1)
        assert partita.battito() == []
    assert partita.fase == "mira" and partita.valori == [pytest.approx(0.3)] and partita.barra == 1
    assert banco.motore.tempo_di_gioco() == fermo
    assert partita.spazio() == [] and banco.contati[-1] == "non_ora" and partita.valori == [pytest.approx(0.3)]
    orologio.valore = 65.0
    k = banco.motore.prossimo_campione()
    messaggi = partita.pausa()
    assert not partita.in_pausa and [m.annuncio for m in messaggi] == ["Si riprende."] and banco.contati[-1] == "ripresa"
    # La barra dell'alzo ricomincia da capo: il secondo di silenzio, i bip e la scivolata.
    assert partita.barra == 1 and partita.sottofase == "pausa" and partita.inizio_barra == k + round(PAUSA_FRA_BARRE * FS)
    assert partita._voce_barra is not barra and partita._voce_barra.inizio == partita.inizio_barra
    banco.flusso.avanza(10)
    assert voce.guadagno == pytest.approx(db_in_lineare(ATTENUAZIONE_IN_MIRA_DB)) and not banco.motore.tempo_scorre
    for valore in (0.0, 0.0):
        porta_udibile_a(banco, campione_del_valore(partita, valore))
        partita.spazio()
    assert partita.fase == "volo" and partita.valori[0] == pytest.approx(0.3) and partita.valori[1:] == [0.0, 0.0]
    assert partita.ultimo_volo.partenza == pytest.approx(fermo)
    # La durata vera della partita non conta il minuto di pausa; la pausa si conta fra quelle chieste.
    orologio.valore = 100.0
    assert partita.riepilogo()["durata_s"] == 40.0
    assert statistiche.dati["conti"]["pause"] == 1 and statistiche.dati["conti"].get("pause_automatiche", 0) == 0


def test_pausa_automatica(banco, cartella_di_prova):
    """La pausa che mette la finestra (D14): ferma e zittisce come Ctrl+P, ma in silenzio, senza suoni
    né messaggi; la toglie soltanto se l'aveva messa lei, e Ctrl+P la toglie sempre."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    orologio = Orologio(0.0)
    partita = banco.partita(statistiche=statistiche, orologio=orologio)
    senti_anche_le_statistiche(banco, statistiche)
    # Senza partita non fa niente.
    assert partita.pausa_automatica(True) == [] and not partita.in_pausa
    assert partita.pausa_automatica(False) == []
    partita.nuova()
    banco.flusso.avanza(10)
    voce = partita._voci[partita.bersagli[0]]
    sentiti = len(banco.contati)
    # Nell'ascolto: il tempo si ferma e i bersagli tacciono, in silenzio.
    orologio.valore = 10.0
    assert partita.pausa_automatica(True) == []
    assert partita.in_pausa and banco.contati[sentiti:] == []
    assert partita.righe_della_barra()[-1] == "In pausa: Ctrl+P per riprendere"
    banco.flusso.avanza(5)
    fermo = banco.motore.tempo_di_gioco()
    banco.flusso.avanza(20)
    assert banco.motore.tempo_di_gioco() == fermo and voce.guadagno == 0.0
    # Una seconda chiamata non cambia niente, e non si conta.
    assert partita.pausa_automatica(True) == []
    orologio.valore = 30.0
    assert partita.pausa_automatica(False) == []
    assert not partita.in_pausa and banco.contati[sentiti:] == []
    banco.flusso.avanza(10)
    assert banco.motore.tempo_di_gioco() > fermo and voce.guadagno == 1.0
    assert statistiche.dati["conti"]["pause_automatiche"] == 1
    # La pausa chiesta con Ctrl+P resta finché non la si toglie con Ctrl+P: la finestra non la tocca.
    partita.pausa()
    assert partita.pausa_automatica(True) == [] and partita.in_pausa
    assert partita.pausa_automatica(False) == [] and partita.in_pausa
    partita.pausa()
    assert not partita.in_pausa and banco.contati[-2:] == ["pausa", "ripresa"]
    # Ctrl+P durante la pausa automatica la toglie, col suono della ripresa: la barra dice «Ctrl+P per riprendere».
    partita.pausa_automatica(True)
    messaggi = partita.pausa()
    assert not partita.in_pausa and [m.annuncio for m in messaggi] == ["Si riprende."] and banco.contati[-1] == "ripresa"
    assert partita.pausa_automatica(False) == [] and not partita.in_pausa
    assert (statistiche.dati["conti"]["pause"], statistiche.dati["conti"]["pause_automatiche"]) == (1, 2)
    # Nella mira: la scivolata si ferma, e al ritorno la barra in corso ricomincia da capo, senza suoni della pausa.
    partita.spazio()
    porta_udibile_a(banco, campione_del_valore(partita, 0.5))
    partita.spazio()
    barra = partita._voce_barra
    sentiti = len(banco.contati)
    partita.pausa_automatica(True)
    banco.flusso.avanza(3)
    assert partita.in_pausa and partita._voce_barra is None and voce.guadagno == 0.0
    for _blocco in range(10 * FS // banco.motore.blocco):
        banco.flusso.avanza(1)
        assert partita.battito() == []
    # La barra dell'alzo era ancora nel suo secondo di silenzio: finisce senza aver suonato.
    assert barra.finita and partita.fase == "mira"
    k = banco.motore.prossimo_campione()
    assert partita.pausa_automatica(False) == []
    assert not partita.in_pausa and partita.barra == 1 and partita.inizio_barra == k + round(PAUSA_FRA_BARRE * FS)
    assert banco.contati[sentiti:] == [] and partita.valori == [pytest.approx(0.5)]
    banco.flusso.avanza(10)
    assert voce.guadagno == pytest.approx(db_in_lineare(ATTENUAZIONE_IN_MIRA_DB))
    # Durante il volo, e poi la durata vera, senza le pause.
    for valore in (0.0, 0.0):
        porta_udibile_a(banco, campione_del_valore(partita, valore))
        partita.spazio()
    assert partita.fase == "volo"
    proiettile = partita._voce_proiettile
    partita.pausa_automatica(True)
    banco.flusso.avanza(3)
    assert proiettile.guadagno == 0.0
    evento = partita._evento_impatto
    for _blocco in range(3 * FS // banco.motore.blocco):
        banco.flusso.avanza(1)
        assert partita.battito() == []
    assert partita.fase == "volo" and not evento.scattato
    partita.pausa_automatica(False)
    assert fino_alla_risoluzione(banco, partita)[0].tipo == "resoconto"
    assert statistiche.dati["conti"]["pause_automatiche"] == 4


def test_pausa_durante_il_volo(banco):
    partita = banco.partita()
    partita.nuova()
    colpisci(banco, partita)
    proiettile = partita._voce_proiettile
    evento = partita._evento_impatto
    banco.flusso.avanza(3)
    partita.pausa()
    banco.flusso.avanza(3)
    assert proiettile.guadagno == 0.0
    for _blocco in range(3 * FS // banco.motore.blocco):
        banco.flusso.avanza(1)
        assert partita.battito() == []
    assert partita.fase == "volo" and not evento.scattato
    partita.pausa()
    banco.flusso.avanza(3)
    assert proiettile.guadagno == 1.0
    messaggi = fino_alla_risoluzione(banco, partita)
    assert evento.scattato and messaggi[0].tipo == "resoconto"


def test_abbandona(banco, cartella_di_prova):
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    assert partita.abbandona() == []
    partita.nuova()
    voci = list(partita._voci.values())
    colpisci(banco, partita)
    proiettile = partita._voce_proiettile
    evento = partita._evento_impatto
    messaggi = partita.abbandona()
    assert [m.tipo for m in messaggi] == ["info"] and messaggi[0].annuncio == "Partita abbandonata."
    assert messaggi[0].dati == partita.riepilogo()
    assert partita.fase == "pronta" and not partita.in_corso() and partita.bersagli == []
    assert evento.annullato
    for _blocco in range(5 * FS // banco.motore.blocco):
        banco.flusso.avanza(1)
    assert all(voce.finita for voce in voci) and proiettile.finita
    assert not evento.scattato
    assert statistiche.dati["conti"]["partite_abbandonate"] == 1
    assert statistiche.dati["ultime"][-1]["motivo"] == "abbandonata"
    # Il tiro in volo non si è risolto: non è nel riepilogo né fra i tiri delle statistiche, e
    # l'effetto del suo impatto, mai suonato, non si conta.
    assert partita.riepilogo()["tiri"] == 0 and "su 1 tiro" not in messaggi[0].testo and "nessun tiro" in messaggi[0].testo
    assert statistiche.dati["ultime"][-1]["tiri"] == statistiche.dati["conti"]["tiri"] == 0
    assert "colpito" not in statistiche.dati["suoni"]
    assert partita.righe_della_barra() == ["Nessuna partita: Spazio per cominciare"]
    # Una partita nuova chiesta durante un'altra la abbandona prima.
    partita.nuova()
    messaggi = partita.nuova()
    assert [m.annuncio for m in messaggi] == ["Partita abbandonata.", "Partita nuova: Spazio per mirare."]
    assert statistiche.dati["conti"]["partite_abbandonate"] == 2


def test_abbandono_dopo_un_impatto_gia_partito(banco, cartella_di_prova):
    """Un centro, poi un tiro il cui impatto è partito nel filo dell'audio, e l'abbandono prima che si
    risolva: l'effetto si è sentito e si conta, il tiro no, né nel riepilogo né nelle statistiche."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    colpisci(banco, partita)
    fino_alla_risoluzione(banco, partita)
    partita.spazio()
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    evento = partita._evento_impatto
    for _blocco in range(BLOCCHI_MASSIMI):
        banco.flusso.avanza(1)
        if evento.scattato:
            break
    assert evento.scattato and partita.fase == "volo" and partita.tiri == 2
    messaggi = partita.abbandona()
    assert messaggi[0].dati["tiri"] == 1 and "1 centro su 1 tiro, 100%" in messaggi[0].testo
    assert statistiche.dati["ultime"][-1]["tiri"] == statistiche.dati["conti"]["tiri"] == 1
    assert statistiche.dati["suoni"]["a_terra"] == 1 and statistiche.dati["suoni"]["colpito"] == 1


def test_gli_avvisi_aspettano_l_effetto_dell_impatto(banco, monkeypatch):
    """L'ultimo colpo e la fine partono in fila dopo l'effetto dell'impatto che li causa, staccati, e non
    sopra di lui; la partita nuova chiesta subito dopo la fine aspetta la fine e l'esito della
    classifica che la finestra mette in fila a dopo_gli_avvisi."""
    voci = []
    originale = banco.suoni.suona

    def spia(evento, *argomenti, **opzioni):
        voce = originale(evento, *argomenti, **opzioni)
        voci.append((evento, voce))
        return voce

    monkeypatch.setattr(banco.suoni, "suona", spia)
    partita = banco.partita()
    partita.nuova()
    stacco = round(modulo_partita.STACCO_FRA_GLI_AVVISI * FS)
    partita.riserva = 2
    for avviso in ("ultimo_colpo", "partita_finita"):
        partita.spazio()
        mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
        evento = partita._evento_impatto
        fino_alla_risoluzione(banco, partita)
        voce = next(v for nome, v in voci if nome == avviso)
        assert voce.inizio == evento.campione + LUNGHEZZA_FINTA + stacco
    fine = voci[-1][1]
    assert voci[-1][0] == "partita_finita" and partita.dopo_gli_avvisi() == fine.inizio + LUNGHEZZA_FINTA + stacco
    esito = banco.suoni.suona("fuori_classifica", inizio=partita.dopo_gli_avvisi())
    # Lo spazio dato d'abitudine subito dopo l'ultimo tonfo: la partita comincia, e il suo suono aspetta.
    partita.spazio()
    assert partita.fase == "ascolto" and voci[-1][0] == "nuova_partita"
    assert voci[-1][1].inizio == esito.inizio + LUNGHEZZA_FINTA + stacco


def test_la_prima_barra_aspetta_gli_avvisi(banco, monkeypatch):
    """Con gli effetti lunghi di D16 l'avviso del livello nuovo, in fila dopo l'impatto, suona oltre il
    secondo di silenzio della prima barra: la mira avviata subito dopo la resa allunga la pausa fino
    alla fine degli avvisi, stacco compreso, così la fanfara non copre i bip; lo spazio nella pausa
    allungata non conta, e la barra seguente torna al secondo solito."""
    corto = np.full((LUNGHEZZA_FINTA, 2), AMPIEZZA_FINTA, dtype=np.float32)
    lungo = np.full((round(1.5 * FS), 2), AMPIEZZA_FINTA, dtype=np.float32)
    monkeypatch.setattr(suoni, "_rendi_preset", lambda nome: lungo if nome == suoni.EVENTI["livello_su"] else corto)
    voci = []
    originale = banco.suoni.suona

    def spia(evento, *argomenti, **opzioni):
        voce = originale(evento, *argomenti, **opzioni)
        voci.append((evento, voce))
        return voce

    monkeypatch.setattr(banco.suoni, "suona", spia)
    stacco = round(modulo_partita.STACCO_FRA_GLI_AVVISI * FS)
    pausa = round(PAUSA_FRA_BARRE * FS)
    partita = banco.partita()
    partita.nuova()
    partita.centri_del_livello = CENTRI_PER_LIVELLO - 1
    colpisci(banco, partita)
    messaggi = fino_alla_risoluzione(banco, partita)
    assert [m.tipo for m in messaggi][:2] == ["resoconto", "livello"]
    fanfara = next(v for nome, v in voci if nome == "livello_su")
    fine = fanfara.inizio + len(lungo) + stacco
    # Lo spazio subito dopo la resa: senza l'attesa i bip comincerebbero sotto la fanfara.
    k = banco.motore.prossimo_campione()
    assert k + pausa < fine
    assert partita.spazio() == [] and partita.fase == "mira" and partita.barra == 0
    assert partita.inizio_barra == fine
    porta_udibile_a(banco, fine - 0.2 * FS)
    assert partita.battito() == [] and partita.sottofase == "pausa"
    assert partita.spazio() == [] and partita.valori == [] and partita.barra == 0
    # Preso il primo valore, gli avvisi sono finiti: la seconda barra ha la sua pausa di sempre.
    porta_udibile_a(banco, campione_del_valore(partita, 0.5))
    assert partita.spazio() == [] and partita.barra == 1
    assert partita.inizio_barra - banco.motore.prossimo_campione() == pausa


def _bonus_quasi_uscito(banco, partita, anticipo):
    """Un bonus in volo, e il tempo di gioco che corre fino a tanti secondi prima della sua uscita."""
    partita.livello = LIVELLO_DEI_BONUS
    tiro_a_vuoto(banco, partita)
    bonus = partita.bonus
    assert bonus is not None
    while banco.motore.tempo_di_gioco(banco.motore.prossimo_campione()) < bonus.tempo_di_uscita - anticipo:
        banco.flusso.avanza(1)
        assert partita.battito() == []
    return bonus


def test_il_bonus_uscito_durante_il_volo_che_chiude_il_livello(banco, bonus_sicuro, cartella_di_prova, monkeypatch):
    """Il bonus esce col proiettile in volo, e il centro chiude il livello: l'uscita si dice e si conta
    prima che il livello nuovo svuoti il cielo."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    bonus = _bonus_quasi_uscito(banco, partita, 0.3)
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    partita.centri_del_livello = CENTRI_PER_LIVELLO - 1
    colpisci(banco, partita)
    volo = partita.ultimo_volo
    assert volo.colpito is not None and not volo.colpito.bonus
    assert volo.partenza < bonus.tempo_di_uscita < volo.fine
    messaggi = fino_alla_risoluzione(banco, partita)
    # Dopo il livello possono venire i primati del centro.
    assert [m.tipo for m in messaggi][:3] == ["resoconto", "bonus", "livello"] and messaggi[1].annuncio == "Bonus uscito."
    assert partita.livello == LIVELLO_DEI_BONUS + 1 and partita.bonus is None
    assert statistiche.dati["conti"]["bonus_usciti"] == 1 and statistiche.dati["suoni"]["bonus_uscito"] == 1


def test_il_bonus_uscito_durante_l_ultimo_volo(banco, bonus_sicuro, cartella_di_prova):
    """Il bonus esce mentre vola il tiro a vuoto che svuota la riserva: l'uscita si dice e si conta
    prima della fine."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    bonus = _bonus_quasi_uscito(banco, partita, 0.2)
    partita.riserva = 1
    partita.spazio()
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    volo = partita.ultimo_volo
    assert volo.colpito is None and volo.partenza < bonus.tempo_di_uscita < volo.fine
    messaggi = fino_alla_risoluzione(banco, partita)
    assert [m.tipo for m in messaggi] == ["resoconto", "bonus", "fine"]
    assert statistiche.dati["conti"]["bonus_usciti"] == 1 and statistiche.dati["suoni"]["bonus_uscito"] == 1


def test_impatto_partito_prima_della_pausa(banco, bonus_sicuro, monkeypatch):
    """Ctrl+P quando l'impatto è già partito nel filo dell'audio ma non si è ancora sentito: il tiro si
    risolve e il resoconto arriva, ma a partita ferma i bersagli nuovi nascono muti, e il livello nuovo
    e il bonus aspettano la ripresa, in fila dopo il suono della ripresa."""
    voci = []
    originale = banco.suoni.suona

    def spia(evento, *argomenti, **opzioni):
        voce = originale(evento, *argomenti, **opzioni)
        voci.append((evento, voce))
        return voce

    monkeypatch.setattr(banco.suoni, "suona", spia)
    partita = banco.partita()
    partita.nuova()
    partita.livello = LIVELLO_DEI_BONUS - 1
    partita.centri_del_livello = CENTRI_PER_LIVELLO - 1
    colpisci(banco, partita)
    evento = partita._evento_impatto
    for _blocco in range(BLOCCHI_MASSIMI):
        banco.flusso.avanza(1)
        if evento.scattato:
            break
    assert evento.scattato and banco.motore.campione_udibile() < evento.campione
    partita.pausa()
    assert partita.in_pausa and banco.contati[-1] == "pausa"
    messaggi = fino_alla_risoluzione(banco, partita)
    assert [m.tipo for m in messaggi] == ["resoconto", "livello"]
    assert partita.fase == "ascolto" and partita.in_pausa and partita.livello == LIVELLO_DEI_BONUS and partita.bonus is None
    assert banco.contati[-2:] == ["pausa", "colpito"]
    banco.flusso.avanza(3)
    assert partita.bersagli and all(partita._voci[b].guadagno == 0.0 for b in partita.bersagli)
    assert partita.righe_della_barra()[-1] == "In pausa: Ctrl+P per riprendere"
    messaggi = partita.pausa()
    assert [m.tipo for m in messaggi] == ["info", "bonus"] and partita.bonus is not None
    assert banco.contati[-3:] == ["ripresa", "livello_su", "bonus_appare"]
    stacco = round(modulo_partita.STACCO_FRA_GLI_AVVISI * FS)
    ripresa, livello, bonus = (v for _nome, v in voci[-3:])
    assert livello.inizio == ripresa.inizio + LUNGHEZZA_FINTA + stacco and bonus.inizio == livello.inizio + LUNGHEZZA_FINTA + stacco
    banco.flusso.avanza(10)
    assert all(partita._voci[b].guadagno == 1.0 for b in partita.bersagli)


def test_i_primati_che_a_parole_non_cambiano_non_si_dicono(banco, cartella_di_prova):
    """Le statistiche tengono il primato esatto, ma il messaggio lo scrive arrotondato: un primato
    battuto per meno di quanto si scrive si registra in silenzio."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    detti = []
    for scarto in (0.05, 1e-7):
        colpisci(banco, partita)
        volo = partita.ultimo_volo
        primati = statistiche.dati["primati"]
        primati["precisione"] = {"valore": volo.precisione - scarto, "quando": "2026-11-19T10:00:00"}
        primati["vicino"] = {"valore": volo.distanza_al_centro + scarto, "quando": "2026-11-19T10:00:00"}
        messaggi = fino_alla_risoluzione(banco, partita)
        detti.append([chiave for m in messaggi if m.tipo == "info" for chiave in m.dati.get("primati", [])])
        assert primati["precisione"]["valore"] == volo.precisione and primati["vicino"]["valore"] == volo.distanza_al_centro
    assert {"precisione", "vicino"} <= set(detti[0])
    assert "precisione" not in detti[1] and "vicino" not in detti[1]


def test_righe_della_barra_in_ogni_fase(banco):
    partita = banco.partita()
    viste = []

    def righe():
        viste.extend(partita.righe_della_barra())
        return partita.righe_della_barra()

    assert righe() == ["Nessuna partita: Spazio per cominciare"]
    partita.nuova()
    assert righe() == ["Livello 1, 0 punti", f"Riserva {COLPI_INIZIALI}, centri 0 su {CENTRI_PER_LIVELLO}", "Bersagli in volo: 1", "Ascolto: Spazio per mirare"]
    partita.spazio()
    partita.battito()
    assert righe()[-1] == "Mira: direzione, pausa"
    porta_udibile_a(banco, partita.inizio_barra + 10)
    partita.battito()
    assert righe()[-1] == "Mira: direzione, bip"
    porta_udibile_a(banco, campione_del_valore(partita, 0.001))
    partita.battito()
    assert righe()[-1] == "Mira: direzione, scivolata"
    partita.spazio()
    partita.battito()
    assert righe()[-1] == "Mira: alzo, pausa"
    for valore in (0.0, 0.0):
        porta_udibile_a(banco, campione_del_valore(partita, valore))
        partita.spazio()
    assert righe()[-1] == "Volo"
    partita.pausa()
    assert righe()[-1] == "In pausa: Ctrl+P per riprendere"
    partita.pausa()
    partita.riserva = 1
    fino_alla_risoluzione(banco, partita)
    assert righe() == ["Livello 1, 0 punti", f"Riserva 0, centri 0 su {CENTRI_PER_LIVELLO}", "Partita finita: Spazio per ricominciare"]
    # Righe brevi, per il display braille.
    assert all(len(riga) <= 40 for riga in viste)


def test_statistiche_aggiornate(banco, cartella_di_prova):
    percorso = cartella_di_prova / "statistiche.json"
    statistiche = Statistiche(percorso=str(percorso))
    partita = banco.partita(statistiche=statistiche)
    assert banco.suoni.conta == statistiche.suono
    partita.nuova()
    primo = partita.bersagli[0]
    colpisci(banco, partita)
    fino_alla_risoluzione(banco, partita)
    secondo = partita.bersagli[0]
    tiro_a_vuoto(banco, partita)
    partita.abbandona()
    # I timbri di D18, apparsi e colpiti: il primo bersaglio colpito, il secondo nato al suo posto.
    timbri = {portante: coppia for portante, coppia in statistiche.dati["portanti"].items() if coppia["apparsi"] or coppia["colpiti"]}
    attesi = {}
    for bersaglio, colpiti in ((primo, 1), (secondo, 0)):
        coppia = attesi.setdefault(bersaglio.portante, {"apparsi": 0, "colpiti": 0})
        coppia["apparsi"] += 1
        coppia["colpiti"] += colpiti
    assert timbri == attesi
    conti = statistiche.dati["conti"]
    assert (conti["partite_iniziate"], conti["mire_avviate"], conti["tiri"], conti["centri"], conti["tiri_a_vuoto"], conti["partite_abbandonate"]) == (1, 2, 2, 1, 1, 1)
    assert conti["punti"] == partita.punti > 0
    suonati = statistiche.dati["suoni"]
    assert (suonati["nuova_partita"], suonati["mira_avviata"], suonati["valore_preso"], suonati["sparo"], suonati["colpito"], suonati["a_terra"]) == (1, 2, 6, 2, 1, 1)
    for parametro in PARAMETRI:
        presi = statistiche.dati["mira"][parametro]
        assert presi["minimo"] + presi["scelto"] + presi["massimo"] == 2
        assert presi["nei_centri"] == 1
    assert statistiche.dati["moto"]["fermi"] == {"apparsi": 2, "colpiti": 1}
    assert statistiche.dati["ultime"][-1]["tiri"] == 2 and statistiche.dati["ultime"][-1]["centri"] == 1
    assert statistiche.dati["somme"]["centri_misurati"] == 1
    assert percorso.exists()


# Il proiettile scelto nelle Impostazioni e le portanti dei bersagli (D17 e D18).

def test_la_forma_del_proiettile_vale_dalla_partita_seguente(banco, cartella_di_prova):
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche, proiettile="triangolo")
    assert partita.proiettile == partita.proiettile_scelto == "triangolo"
    # Senza partita in corso la forma vale subito, per la partita che comincerà.
    assert partita.scegli_proiettile("sega_morbida") is True
    assert partita.proiettile == partita.proiettile_scelto == "sega_morbida"
    partita.nuova()
    assert partita.proiettile == "sega_morbida" and all(b.portante in portanti_senza("sega_morbida") for b in partita.bersagli)
    # Con la partita in corso la forma nuova aspetta: questa tiene la sua dall'inizio alla fine.
    assert partita.scegli_proiettile("seno") is False
    assert (partita.proiettile, partita.proiettile_scelto) == ("sega_morbida", "seno")
    colpisci(banco, partita)
    assert partita._voce_proiettile.forma == "sega_morbida"
    fino_alla_risoluzione(banco, partita)
    # Il bersaglio nato al posto del colpito evita ancora la forma della partita in corso, non quella scelta dopo.
    assert partita.bersagli and all(b.portante in portanti_senza("sega_morbida") for b in partita.bersagli)
    # Anche una scelta fatta in pausa aspetta la partita seguente.
    partita.pausa()
    assert partita.scegli_proiettile("triangolo") is False
    partita.pausa()
    assert (partita.proiettile, partita.proiettile_scelto) == ("sega_morbida", "triangolo")
    # La partita nuova, anche quella che abbandona la vecchia, prende la forma scelta per ultima.
    partita.nuova()
    assert partita.proiettile == "triangolo" and all(b.portante in portanti_senza("triangolo") for b in partita.bersagli)
    partita.spazio()
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    assert partita._voce_proiettile.forma == "triangolo"
    fino_alla_risoluzione(banco, partita)
    # Le statistiche contano le partite cominciate con ogni forma.
    conti = statistiche.dati["conti"]
    assert (conti["partite_con_la_sega_morbida"], conti["partite_col_triangolo"], conti["partite_col_seno"]) == (1, 1, 0)
    assert conti["partite_iniziate"] == 2
    # Abbandonata la partita, la forma scelta vale di nuovo subito.
    partita.abbandona()
    assert partita.scegli_proiettile("seno") is True and partita.proiettile == "seno"


def test_una_forma_del_proiettile_sconosciuta_non_vale(banco):
    with pytest.raises(ValueError, match="organo"):
        banco.partita(proiettile="organo")
    partita = banco.partita()
    assert partita.proiettile == partita.proiettile_scelto == PROIETTILE_FORMA
    # L'organo resta ai bersagli; le forme delle LFO e gli altri valori non sono forme del proiettile.
    for sbagliata in ("organo", "quadra", "Seno", "", None):
        with pytest.raises(ValueError):
            partita.scegli_proiettile(sbagliata)
    assert partita.proiettile == partita.proiettile_scelto == PROIETTILE_FORMA
    # Ogni forma ha il suo contatore delle partite.
    assert set(modulo_partita.CONTI_DEL_PROIETTILE) == set(PROIETTILE_FORME)
    assert len(set(modulo_partita.CONTI_DEL_PROIETTILE.values())) == len(PROIETTILE_FORME)


@pytest.mark.parametrize("forma", PROIETTILE_FORME)
def test_i_bersagli_e_il_bonus_nascono_senza_la_portante_del_proiettile(banco, bonus_sicuro, monkeypatch, forma):
    """D18: genera e genera_bonus ricevono le portanti di BERSAGLIO_PORTANTI tolta la forma del proiettile, e ogni voce suona nella portante del suo bersaglio."""
    chieste = []
    for nome in ("genera", "genera_bonus"):
        originale = getattr(modulo_partita, nome)

        def spia(*argomenti, _nome=nome, _originale=originale, **opzioni):
            chieste.append((_nome, opzioni.get("portanti")))
            return _originale(*argomenti, **opzioni)

        monkeypatch.setattr(modulo_partita, nome, spia)
    partita = banco.partita(proiettile=forma)
    partita.nuova()
    partita.livello = LIVELLO_DEI_BONUS
    tiro_a_vuoto(banco, partita)
    bonus = partita.bonus
    assert bonus is not None
    attese = portanti_senza(forma)
    assert set(attese) == set(BERSAGLIO_PORTANTI) - {forma} and len(attese) == len(BERSAGLIO_PORTANTI) - 1
    # Tanti bersagli nuovi del livello, come quelli che nascono al posto dei colpiti: nessuno ha la portante del
    # proiettile, e col seme fisso escono tutte le altre.
    for _volta in range(30):
        partita._nuovo_bersaglio()
    assert {nome for nome, _portanti in chieste} == {"genera", "genera_bonus"}
    assert all(portanti == attese for _nome, portanti in chieste)
    assert bonus.portante in attese and partita._voci[bonus].forma == bonus.portante
    for bersaglio in partita.bersagli:
        assert bersaglio.portante in attese and partita._voci[bersaglio].forma == bersaglio.portante
    assert {b.portante for b in partita.bersagli} == set(attese)


def test_l_esempio_del_proiettile(banco):
    motore, flusso = banco.motore, banco.flusso
    durata = round(modulo_partita.DURATA_DELL_ESEMPIO * FS)
    volo = modulo_partita.volo_d_esempio()
    # Il volo d'esempio è un tiro verso il centro del cubo, che dura più dell'esempio, e passa da lì.
    centro = np.array((0.0, 25.0, 55.0))
    assert volo.colpito is None and volo.durata > modulo_partita.DURATA_DELL_ESEMPIO
    tempi = np.linspace(0.0, volo.durata, 20001)
    assert np.min(np.linalg.norm(volo.traiettoria.posizione(tempi) - centro, axis=1)) < 0.01
    # Il tempo di gioco fermo, come col dialogo aperto, non ferma l'esempio: segue il tempo dell'audio.
    motore.ferma_tempo()
    k = motore.prossimo_campione()
    voce = modulo_partita.esempio_del_proiettile(motore, "triangolo")
    assert isinstance(voce, VoceTono) and voce.forma == "triangolo" and voce.inizio == k
    frequenze, pan, _guadagni = voce.parametri(0.0, np.array((k, k + durata // 2, k + durata)) / FS)
    assert frequenze[0] < frequenze[1] < frequenze[2] and np.allclose(pan, 0.0, atol=1e-9)
    uscita = flusso.avanza_fino(k + durata + 2 * motore.blocco)
    assert not uscita[:k].any()
    # Suona per tutto l'esempio, ogni decimo di secondo, e dopo tace.
    pezzo = FS // 10
    assert all(np.abs(uscita[inizio:inizio + pezzo]).max() > 0 for inizio in range(k, k + durata - pezzo, pezzo))
    assert not uscita[k + durata:].any()
    assert voce.finita and voce.finita_al == k + durata
    # Fermato prima, come quando parte l'esempio di un'altra forma, finisce subito, con la sfumatura del motore.
    seconda = modulo_partita.esempio_del_proiettile(motore, "seno")
    flusso.avanza_fino(seconda.inizio + motore.blocco)
    motore.ferma(seconda)
    flusso.avanza(4)
    assert seconda.finita and seconda.finita_al < seconda.inizio + durata
    with pytest.raises(ValueError, match="organo"):
        modulo_partita.esempio_del_proiettile(motore, "organo")


def test_il_primato_della_serie_si_dice_una_volta(banco, cartella_di_prova):
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    statistiche.dati["primati"]["serie"] = {"valore": 1, "quando": "2026-11-19T10:00:00"}
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    detti = []
    for _centro in range(3):
        colpisci(banco, partita)
        messaggi = fino_alla_risoluzione(banco, partita)
        detti.append([chiave for m in messaggi if m.tipo == "info" for chiave in m.dati.get("primati", [])])
    assert statistiche.dati["primati"]["serie"]["valore"] == 3
    assert "serie" not in detti[0] and "serie" in detti[1] and "serie" not in detti[2]
    assert partita.serie == 3
