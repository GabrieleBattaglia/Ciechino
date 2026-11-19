"""
Le prove di partita.py: l'inizio con le voci dei bersagli, la mira al campione udibile, lo sparo e
l'impatto al campione esatto, i tiri a vuoto fino alla fine, il livello, i bonus, la pausa, l'abbandono,
la barra di stato e le statistiche.
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
import suoni
import testi
from bersagli import Bersaglio, durata_del_bonus, genera, quanti_bersagli, raggio_del_livello
from costanti import (
    ATTENUAZIONE_IN_MIRA_DB,
    BERSAGLIO_FORMA,
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
    PREMIO_PRECISIONE,
    PROIETTILE_FORMA,
    PUNTI_BASE,
    RAGGIO_DI_RIFERIMENTO,
    SFUMATURA_STOP,
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

    def partita(self, statistiche=None, orologio=None, seme=SEME):
        return Partita(self.motore, self.suoni, statistiche=statistiche, rng=random.Random(seme), orologio=orologio or Orologio())


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
    bonus = Bersaglio(numero=3, forma="seno", raggio=RAGGIO_DI_RIFERIMENTO, partenza=(0.0, 20.0, 50.0), bonus=True)
    assert punti_del_centro(bonus, 0.0) == PUNTI_BASE * MOLTIPLICATORE_BONUS


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
        assert voce.forma == BERSAGLIO_FORMA and voce.inizio == k
        assert voce.parametri == bersaglio.parametri_sonori
        assert bersaglio.nato == banco.motore.tempo_di_gioco(k)
    assert banco.motore.tempo_scorre
    assert banco.contati == ["nuova_partita"]
    assert [m.tipo for m in messaggi] == ["info"]
    assert isinstance(messaggi[0], Messaggio)
    assert messaggi[0].annuncio == "Partita nuova: Spazio per mirare."
    # Lo stesso seme dà gli stessi bersagli.
    gemello = genera(1, random.Random(SEME), 1, partita.bersagli[0].nato)
    assert np.allclose(gemello.partenza, partita.bersagli[0].partenza) and gemello.forma == partita.bersagli[0].forma
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
    messaggi = fino_alla_risoluzione(banco, partita)
    attesi = round(PUNTI_BASE * (RAGGIO_DI_RIFERIMENTO / bersaglio.raggio) ** ESPONENTE_DEL_RAGGIO * (1 + PREMIO_PRECISIONE * volo.precisione))
    assert partita.punti == attesi == punti_del_centro(bersaglio, volo.precisione)
    assert (partita.centri, partita.serie, partita.centri_del_livello, partita.riserva) == (1, 1, 1, COLPI_INIZIALI)
    assert partita.fase == "ascolto"
    assert bersaglio not in partita.bersagli and bersaglio not in partita._voci
    assert len(partita.bersagli) == quanti_bersagli(1) and partita.bersagli[0] in partita._voci
    resoconto = messaggi[0]
    assert resoconto.tipo == "resoconto" and resoconto.annuncio == f"Colpito, {attesi} punti."
    assert resoconto.dati["colpito"] and resoconto.dati["punti"] == attesi and resoconto.dati["totale"] == attesi
    # L'effetto dell'impatto si conta quando è pronto, cioè allo sparo: l'ordine fra i due non conta.
    assert sorted(banco.contati) == sorted(["nuova_partita", "mira_avviata", "valore_preso", "valore_preso", "valore_preso", "sparo", "colpito"])


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
    assert isinstance(proiettile, VoceTono) and proiettile.forma == PROIETTILE_FORMA
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
        if partita.riserva > 1:
            assert [m.tipo for m in messaggi] == ["resoconto"]
            assert messaggi[0].annuncio.startswith(f"A vuoto, riserva {partita.riserva}, ")
        elif partita.riserva == 1:
            assert [m.tipo for m in messaggi] == ["resoconto", "avviso"]
            assert messaggi[1].annuncio == "Ultimo colpo."
    assert partita.fase == "finita" and not partita.in_corso()
    assert [m.tipo for m in messaggi] == ["resoconto", "fine"]
    assert messaggi[0].annuncio == "A vuoto, riserva vuota."
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
    attesi = round(PUNTI_BASE * (RAGGIO_DI_RIFERIMENTO / bonus.raggio) ** ESPONENTE_DEL_RAGGIO * (1 + bonus.velocita_scalare / VELOCITA_DI_RIFERIMENTO)
                   * (1 + PREMIO_PRECISIONE * volo.precisione) * MOLTIPLICATORE_BONUS)
    assert partita.punti == attesi == punti_del_centro(bonus, volo.precisione)
    assert partita.bonus is None and bonus not in partita.bersagli
    assert len(partita.bersagli) == quanti_bersagli(LIVELLO_DEI_BONUS)
    assert messaggi[0].annuncio == f"Colpito il bonus, {testi.intero(attesi)} punti."
    assert messaggi[0].dati["bonus"]
    assert "colpito_bonus" in banco.contati and "bonus_uscito" not in banco.contati
    # L'uscita annullata non scatta mai, nemmeno quando il suo tempo passa.
    while banco.motore.tempo_di_gioco() <= evento.tempo + 0.5:
        banco.flusso.avanza(10)
    assert not evento.scattato


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
    # Nella mira il tempo è già fermo, e la pausa lo dice soltanto.
    partita.spazio()
    messaggi = partita.pausa()
    assert [m.tipo for m in messaggi] == ["avviso"] and not partita.in_pausa and partita.fase == "mira"
    # Senza partita non c'è niente da mettere in pausa.
    assert banco.partita().pausa()[0].annuncio == "Nessuna partita in corso."


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
    assert partita.righe_della_barra() == ["Nessuna partita: Spazio per cominciare"]
    # Una partita nuova chiesta durante un'altra la abbandona prima.
    partita.nuova()
    messaggi = partita.nuova()
    assert [m.annuncio for m in messaggi] == ["Partita abbandonata.", "Partita nuova: Spazio per mirare."]
    assert statistiche.dati["conti"]["partite_abbandonate"] == 2


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
    colpisci(banco, partita)
    fino_alla_risoluzione(banco, partita)
    tiro_a_vuoto(banco, partita)
    partita.abbandona()
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
