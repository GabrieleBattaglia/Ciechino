"""
Le prove di partita.py: l'inizio con le voci dei bersagli, la mira al campione udibile, lo sparo e
l'impatto al campione esatto, i tiri a vuoto fino alla fine, il livello, i bonus e i loro punti (D13),
gli avvisi in fila dopo l'impatto, la pausa chiesta e quella automatica, anche nella mira (D14),
l'abbandono, i primati, la barra di stato, le statistiche, e il sommario in testa al resoconto vero,
al grado della palestra del livello del tiro (D21). Da D22 il resoconto che arriva quando l'effetto
dell'impatto ha finito di suonare, e la palestra delle attese: la partita ferma e muta dopo ogni
tiro dei primi livelli, lo spazio che la scioglie con la sua grazia, la durata senza l'attesa, la
pausa automatica durante l'attesa, Ctrl+P soltanto nella palestra e il confine col livello dopo.
I tiri dei primi livelli finiscono nell'attesa: gli aiuti che mirano la fanno sciogliere prima.
Dal secondo collaudo il resoconto un secondo dopo l'effetto (D24), Esc nella mira (D25), e la
precisione globale nelle statistiche (D26).
Dalla 2.0.0 la cupola (D28), la mira manuale (D29) e la riserva più larga (D30): via le scivolate, i
passaggi delle barre, il proiettile perduto, i bersagli attenuati e il tempo fermo nella mira; al loro
posto i tre valori che si muovono finché i loro tasti sono giù, con le tacche e il suono del capo,
l'Invio coi valori dell'istante udibile, il tempo della mira fuori dalla palestra coi suoi tick e la
scadenza che fa partire il colpo da solo, la pausa che conserva i secondi rimasti, Esc, i bersagli che
si muovono durante la mira, e i colpi di D30. Dalla 2.0.1 il puntatore di D31, la voce sola che porta i tre
valori, con le tacche e il capo dal suo punto; dalla 2.0.3 il puntatore di D32, nel timbro del proiettile e
acceso piano per tutta la mira.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0; dalla 2.0.0 le prove mirano tenendo giù i tasti dei valori, non più
con lo spazio sulle scivolate.
Nessuna prova apre la scheda audio né sintetizza un preset vero: il motore gira sul flusso finto di
aiuti_audio, che chiama il callback soltanto quando la prova lo chiede, e _rendi_preset dà a ogni
preset lo stesso buffer breve. Per premere un tasto a un campione preciso la prova fa preparare al
motore il blocco che lo contiene, poi sposta l'orologio finto finché il campione udibile è proprio
quello: così la mira si prova al campione, senza il caso del tempo vero. Gli aiuti che mirano tengono
giù i tasti dei valori il tempo giusto, come farebbe chi gioca, e poi premono Invio.
"""

import math
import random

import numpy as np
import pytest
from aiuti_audio import motore_finto

import mira as modulo_mira
import partita as modulo_partita
import resoconti
import suoni
import testi
from bersagli import Bersaglio, durata_del_bonus, genera, genera_bonus, portanti_senza, quanti_bersagli, raggio_del_livello, velocita_del_livello
from costanti import (
    AIUTO_MANCATO_FINO_AL_LIVELLO,
    ANTICIPO_DELLO_SPARO,
    BERSAGLIO_PORTANTI,
    BLOCCO_DEL_BRAILLE,
    CENTRI_PER_LIVELLO,
    COLPI_BONUS,
    COLPI_INIZIALI,
    COLPI_PER_LIVELLO,
    DURATA_DELLA_SCALA,
    ESPONENTE_DEL_RAGGIO,
    FS,
    GRAZIA_DELL_ATTESA,
    LIVELLO_DEI_BONUS,
    LIVELLO_DEI_MOBILI,
    MOLTIPLICATORE_BONUS,
    PALESTRA_FINO_AL_LIVELLO,
    PARAMETRI,
    PARTENZA_DELLA_MIRA,
    POTENZA_DB_MINIMO,
    PREMIO_PRECISIONE,
    PROIETTILE_FORMA,
    PROIETTILE_FORME,
    PUNTATORE_DB,
    PUNTATORE_FERMO_DB,
    PUNTI_BASE,
    RAGGIO_DI_RIFERIMENTO,
    RAGGIO_PROIETTILE,
    RIDUZIONE_DEL_TEMPO_DELLA_MIRA,
    RITARDO_DEL_RESOCONTO,
    SFUMATURA_STOP,
    SOMMARIO_NUMERI_FINO_AL_LIVELLO,
    SOMMARIO_PAROLE_FINO_AL_LIVELLO,
    TACCHE,
    TEMPO_DELLA_MIRA_INIZIALE,
    TEMPO_DELLA_MIRA_MINIMO,
    TIRI_PER_IL_PRIMATO_DELLA_MEDIA,
    VELOCITA_DEL_TEMPO,
    VELOCITA_DELLA_MIRA,
    VELOCITA_DI_RIFERIMENTO,
)
from fisica import MEZZO_DELLA_CUPOLA, Traiettoria, distanza_dal_punto_piu_lontano, parametri_del_tiro, soluzione
from motore_audio import VoceTono
from partita import Messaggio, Partita, punti_del_centro, tempo_della_mira
from sintesi import db_in_lineare, parametri_del_puntatore, tacca_mono, tick_del_tempo
from statistiche import Statistiche
from suoni import Suoni
from volo import calcola_volo

SEME = 73
# Il buffer finto di ogni preset: breve e costante, così un effetto si riconosce nell'uscita e non
# copre a lungo le voci dei bersagli.
LUNGHEZZA_FINTA = 256
AMPIEZZA_FINTA = 0.1
# Quanti blocchi al massimo si aspetta un evento, prima di dire che non arriva: 30 secondi di suono.
BLOCCHI_MASSIMI = 30 * FS // 512
SFUMATURA = round(SFUMATURA_STOP * FS)
# Quanto dura ogni mira degli aiuti, dal primo tasto all'Invio: un valore che va dal centro a un capo
# della scala ci mette mezza DURATA_DELLA_SCALA, e un decimo di secondo resta di margine.
ATTESA_DELLO_SPARO = DURATA_DELLA_SCALA / 2 + 0.1
# Un valore che deve arrivare a un capo della scala si lascia un poco dopo esserci arrivato: così è
# proprio il capo, e non un capello prima.
OLTRE_IL_CAPO = 0.01
# Il primo livello con la mira a tempo più breve, TEMPO_DELLA_MIRA_MINIMO secondi (D29): le prove della
# scadenza ci aspettano meno. I bersagli restano quelli del primo livello, che la prova non cambia.
LIVELLO_DELLA_MIRA_BREVE = PALESTRA_FINO_AL_LIVELLO + 1 + round((TEMPO_DELLA_MIRA_INIZIALE - TEMPO_DELLA_MIRA_MINIMO) / RIDUZIONE_DEL_TEMPO_DELLA_MIRA)


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
    # Il resoconto arriva alla fine dell'effetto più RITARDO_DEL_RESOCONTO (D22, D24): qui il ritardo è zero, perché le prove
    # contano i campioni dell'effetto; il secondo vero di D24 lo prova test_il_resoconto_arriva_quando_l_effetto_ha_finito_di_suonare,
    # e lo usano le prove d'integrazione.
    monkeypatch.setattr(modulo_partita, "RITARDO_DEL_RESOCONTO", 0.0)
    banco = Banco()
    yield banco
    banco.motore.chiudi()


@pytest.fixture
def bonus_sicuro(monkeypatch):
    """Un bonus a ogni nuova fase d'ascolto, dal suo livello."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 1.0)


# Gli aiuti delle prove.

def assi_della_mira(partita):
    """I tre Asse della mira in corso, nell'ordine di PARAMETRI, come li legge il puntatore."""
    return [partita.mira.assi[parametro] for parametro in PARAMETRI]


def suono_del_puntatore(partita, campioni):
    """(frequenze, pan, guadagni, in moto) del puntatore ai campioni dati, senza l'inviluppo che lo apre e lo chiude (D31)."""
    return modulo_partita._suono_del_puntatore(assi_della_mira(partita), np.asarray(campioni, dtype=float) / FS)


def pan_stimato(stereo):
    """La panoramica di un buffer stereo, da meno uno a uno, rifatta dall'energia dei due lati con la legge di
    sintesi.panoramica."""
    dati = np.asarray(stereo, dtype=np.float64)
    sinistra = np.sqrt(np.mean(dati[:, 0] ** 2))
    destra = np.sqrt(np.mean(dati[:, 1] ** 2))
    return float(np.arctan2(destra, sinistra) * 4.0 / np.pi - 1.0)


def decimo_della_tacca(partita, parametro, buffer):
    """Il decimo di una tacca in fila del parametro, riconosciuta dal suono: con la legge a potenza costante la somma dei
    quadrati dei due lati è il quadrato della tacca mono di quel decimo, qualunque sia il suo pan (D31). None se il buffer
    non è una di quelle tacche."""
    energia = np.sum(np.asarray(buffer, dtype=np.float64) ** 2, axis=1)
    for decimo, mono in partita._tacche[parametro].items():
        if len(mono) == len(energia) and np.allclose(energia, np.asarray(mono, dtype=np.float64) ** 2, rtol=0, atol=1e-7):
            return decimo
    return None


def porta_udibile_a(banco, campione):
    """Fa preparare al motore il blocco che contiene il campione, poi sposta l'orologio finto perché
    il campione udibile sia proprio quello. L'orologio che torna indietro, quando i blocchi sono già
    preparati oltre, come per i tasti della mira, può lasciare il campione udibile un capello sotto
    quello chiesto, per gli arrotondamenti: allora l'orologio avanza del più piccolo passo che c'è,
    finché il campione chiesto non si sente."""
    banco.flusso.avanza_fino(math.ceil(campione))
    inizio, dac = banco.motore._riferimento
    banco.flusso.time = dac + (campione - inizio) / banco.flusso.fs
    for _passo in range(64):
        if banco.motore.campione_udibile() >= campione:
            break
        banco.flusso.time = math.nextafter(banco.flusso.time, math.inf)
    assert banco.motore.campione_udibile() == pytest.approx(campione, abs=1e-6) and banco.motore.campione_udibile() >= campione


def spia_del_motore(monkeypatch, motore):
    """Le voci che il motore suona da un buffer, gli effetti, le tacche, i suoni del capo e i tick, come
    terne (buffer, campione d'inizio, voce), nell'ordine in cui arrivano."""
    suonate = []
    originale = motore.suona

    def spia(buffer, inizio=None, guadagno=1.0):
        voce = originale(buffer, inizio=inizio, guadagno=guadagno)
        suonate.append((buffer, voce.inizio, voce))
        return voce

    monkeypatch.setattr(motore, "suona", spia)
    return suonate


def porta_i_valori(banco, partita, valori):
    """Nella mira porta i tre valori dove servono, come chi gioca coi tasti (D29): preme insieme, al
    campione udibile di adesso, il tasto di ogni valore che deve muoversi, nel suo verso, e lascia
    ognuno quando il suo valore è arrivato, alla VELOCITA_DELLA_MIRA, spostando il campione udibile con
    porta_udibile_a; il valore che va a un capo si lascia OLTRE_IL_CAPO dopo. Fra un tasto e l'altro
    non c'è battito. Restituisce il campione udibile dell'ultimo rilascio."""
    assert partita.fase == "mira" and partita.mira is not None
    inizio = banco.motore.campione_udibile()
    rilasci = []
    for parametro, valore in zip(PARAMETRI, valori, strict=True):
        scarto = float(valore) - partita.mira.assi[parametro].valore(inizio / FS)
        if abs(scarto) < 1e-12:
            continue
        verso = 1 if scarto > 0 else -1
        assert partita.muovi(parametro, verso) == []
        secondi = abs(scarto) / VELOCITA_DELLA_MIRA + (OLTRE_IL_CAPO if valore in (0.0, 1.0) else 0.0)
        rilasci.append((inizio + secondi * FS, parametro, verso))
    fine = inizio
    for campione, parametro, verso in sorted(rilasci):
        porta_udibile_a(banco, campione)
        assert partita.lascia(parametro, verso) == []
        fine = campione
    assert partita.tasti_giu() == []
    assert partita.mira.valori(fine / FS) == pytest.approx([float(valore) for valore in valori], abs=1e-9)
    return fine


def mira_e_spara(banco, partita, valori, invio=None):
    """Con la mira avviata, porta i tre valori dove servono con porta_i_valori e preme Invio al campione
    udibile invio, o, se è None, ATTESA_DELLO_SPARO dopo il primo tasto: così ogni mira degli aiuti che
    parte dal centro dura lo stesso tempo, e il colpo parte coi valori voluti. Una mira che non parte dal
    centro, come quella ripresa dopo una pausa, può chiedere di più: allora Invio segue l'ultimo tasto."""
    inizio = banco.motore.campione_udibile()
    fine = porta_i_valori(banco, partita, valori)
    if invio is None:
        invio = max(inizio + ATTESA_DELLO_SPARO * FS, fine)
    assert fine <= invio
    porta_udibile_a(banco, invio)
    assert partita.spara() == []
    assert partita.fase == "volo" and partita.valori == pytest.approx([float(valore) for valore in valori], abs=1e-9)


def prepara_lo_sparo(banco, secondi=ATTESA_DELLO_SPARO):
    """Per mirare un bersaglio in moto bisogna sapere da quale tempo di gioco partirà il colpo: dal
    prossimo campione sicuro dell'Invio più ANTICIPO_DELLO_SPARO (D29), e il prossimo campione dipende
    dai blocchi che il motore ha già preparato. L'aiuto fa preparare subito i blocchi fino all'Invio,
    tanti secondi dopo il campione udibile, poi rimette l'orologio finto dov'era: i tasti e l'Invio che
    seguono non preparano altri blocchi, e il campione dello sparo si sa prima di muovere i valori.
    Restituisce il campione udibile dell'Invio e il tempo di gioco della partenza del colpo."""
    adesso = banco.motore.campione_udibile()
    invio = adesso + secondi * FS
    banco.flusso.avanza_fino(math.ceil(invio))
    porta_udibile_a(banco, adesso)
    sparo = banco.motore.prossimo_campione() + round(ANTICIPO_DELLO_SPARO * FS)
    return invio, banco.motore.tempo_di_gioco(sparo)


def mira_sul_bersaglio(bersaglio, partenza):
    """I valori che colpiscono il bersaglio, fermo o in moto, col colpo che parte al tempo di gioco
    partenza: il punto dove sarà quando il proiettile ci arriva, ricalcolato finché il tempo di volo non
    si assesta. Per un bersaglio fermo è la soluzione della fisica sul suo centro."""
    tempo = 0.0
    valori = None
    for _giro in range(30):
        punto = bersaglio.posizione(partenza + tempo)
        valori = soluzione(punto)
        assert valori is not None
        traiettoria = Traiettoria(*parametri_del_tiro(*valori))
        orizzontale = math.hypot(traiettoria.iniziale[0], traiettoria.iniziale[2])
        tempo = math.hypot(punto[0], punto[2]) / orizzontale
    return valori


def fino_alla_risoluzione(banco, partita):
    """Fa avanzare il motore col battito finché il tiro si risolve e i suoi messaggi arrivano, quando
    l'effetto dell'impatto ha finito di suonare (D22), e li restituisce."""
    messaggi = []
    for _blocco in range(BLOCCHI_MASSIMI):
        banco.flusso.avanza(1)
        messaggi.extend(partita.battito())
        if partita.fase != "volo" and messaggi:
            return messaggi
    raise AssertionError("il tiro non si è risolto")


def fino_al_fermo(banco, partita):
    """Nella palestra (D22): fa avanzare il motore col battito finché la partita si ferma ad aspettare
    lo spazio; restituisce i messaggi arrivati intanto."""
    assert partita.fase == "attesa"
    messaggi = []
    for _blocco in range(BLOCCHI_MASSIMI):
        if partita.attesa_ferma:
            return messaggi
        banco.flusso.avanza(1)
        messaggi.extend(partita.battito())
    raise AssertionError("l'attesa della palestra non comincia")


def fuori_dall_attesa(banco, partita):
    """Nella palestra (D22): aspetta che la partita si fermi, lascia passare la grazia e preme lo
    spazio, che la fa ripartire; restituisce i messaggi dello spazio, come il bonus che nasce."""
    fino_al_fermo(banco, partita)
    # La grazia si conta dal campione a cui la partita si è fermata: col ritardo del resoconto di D24 può essere quello, a metà
    # di un blocco, a cui la prova ha portato il motore per i messaggi del tiro.
    porta_udibile_a(banco, math.ceil(partita._attesa_dal + round(GRAZIA_DELL_ATTESA * FS)) + 1)
    assert partita.battito() == []
    messaggi = partita.spazio()
    assert partita.fase == "ascolto" and not partita.attesa_ferma
    return messaggi


def pronta_a_mirare(banco, partita):
    """Se la partita aspetta lo spazio nella palestra, la fa ripartire: poi lo spazio avvia la mira."""
    if partita.fase == "attesa":
        return fuori_dall_attesa(banco, partita)
    return []


def colpisci(banco, partita, bersaglio=None):
    """Avvia la mira e spara al centro di un bersaglio, il primo normale se è None, fermo o in moto:
    dove sarà all'arrivo del proiettile, col tempo di gioco della partenza che dà prepara_lo_sparo.
    Restituisce i valori."""
    pronta_a_mirare(banco, partita)
    assert partita.fase == "ascolto"
    partita.spazio()
    if bersaglio is None:
        bersaglio = next(b for b in partita.bersagli if not b.bonus)
    invio, partenza = prepara_lo_sparo(banco)
    valori = mira_sul_bersaglio(bersaglio, partenza)
    mira_e_spara(banco, partita, valori, invio=invio)
    assert partita.ultimo_volo.partenza == pytest.approx(partenza)
    return valori


def tiro_a_vuoto(banco, partita):
    """Un tiro tutto a sinistra, all'alzo più basso e con la potenza minima, che entra appena nella
    cupola sul suo lato sinistro e ricade a una trentina di metri, lontano dai bersagli delle prove, e
    la sua risoluzione."""
    pronta_a_mirare(banco, partita)
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


def finiti_gli_avvisi(banco, partita):
    """Il tempo della mira a tempo parte dopo gli avvisi in fila, come quello della partita nuova (D29, dalle prove del 22
    novembre 2026): le prove che ne contano i secondi dallo spazio aspettano prima che finiscano, stacco compreso."""
    if partita._fine_degli_avvisi > banco.motore.campione_udibile():
        porta_udibile_a(banco, partita._fine_degli_avvisi)
    assert partita._fine_degli_avvisi <= banco.motore.prossimo_campione()


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


def test_la_riserva_piu_larga_di_d30(banco):
    """D30: 20 colpi all'inizio, 15 a ogni livello chiuso, 8 per un centro sul bonus, al posto di 10, 10 e 5 di D2 e D13; il
    primato della precisione globale media vale ancora dalle partite di almeno 10 tiri."""
    assert (COLPI_INIZIALI, COLPI_PER_LIVELLO, COLPI_BONUS, TIRI_PER_IL_PRIMATO_DELLA_MEDIA) == (20, 15, 8, 10)
    partita = banco.partita()
    messaggi = partita.nuova()
    assert partita.riserva == 20 and messaggi[0].dati == {"livello": 1, "riserva": 20}
    assert "nella riserva 20 colpi" in messaggi[0].testo


def test_inizio_della_partita_e_voci_dei_bersagli(banco):
    partita = banco.partita()
    assert partita.fase == "pronta" and not partita.in_corso()
    k = banco.motore.prossimo_campione()
    messaggi = partita.nuova()
    assert partita.fase == "ascolto" and partita.in_corso()
    assert (partita.livello, partita.riserva, partita.punti, partita.tiri, partita.centri, partita.centri_del_livello, partita.serie) == (1, COLPI_INIZIALI, 0, 0, 0, 0, 0)
    assert len(partita.bersagli) == quanti_bersagli(1)
    assert partita.bonus is None and partita.mira is None
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
    assert bersaglio not in partita.bersagli and bersaglio not in partita._voci
    # D22: il livello 1 è della palestra. La partita aspetta lo spazio, che chiama il bersaglio nuovo:
    # finché non arriva, il cielo è vuoto.
    assert partita.fase == "attesa" and partita.nasceranno == 1 and partita.bersagli == []
    assert [m.tipo for m in messaggi] == ["resoconto", "info"]
    assert messaggi[-1].testo == "Palestra: la partita aspetta lo Spazio, che chiama il bersaglio nuovo." and messaggi[-1].annuncio == ""
    fuori_dall_attesa(banco, partita)
    assert len(partita.bersagli) == quanti_bersagli(1) and partita.bersagli[0] in partita._voci
    resoconto = messaggi[0]
    # D21: al livello 1 il sommario coi numeri apre il resoconto e segue l'esito nell'annuncio. Il
    # centro è pieno, col tiro più dolce di fisica.soluzione, che arriva scendendo: le misure
    # dell'artigliere dal centro si arrotondano a zero, profondità compresa.
    # D24: i valori con le loro iniziali nel primo blocco, e con loro la precisione globale di D26, che nel centro pieno è quasi del
    # 100% e ci sta.
    sommario = resoconti.sommario_del_tiro(volo, "numeri")
    assert volo.misure.inclinazione < 0.0 and volo.precisione_globale > 0.99
    assert sommario.primo.startswith("L:0,0 V:0,0 P:0,0 I:-") and sommario.primo.endswith(f" G:{testi.percento(volo.precisione_globale, 1)}")
    assert len(sommario.primo) < BLOCCO_DEL_BRAILLE and sommario.riga[BLOCCO_DEL_BRAILLE:].startswith("L laterale, V verticale")
    # D33: dopo il sommario la situazione, il livello e il tempo; senza la classifica, come qui, il posto tace, anche nell'annuncio.
    righe = resoconto.testo.split("\n")
    assert righe[0] == sommario.riga and righe[3].startswith("Tiro 1, livello 1: colpito un bersaglio, ")
    assert righe[1] == f"Sparato 1 colpo nel livello, {COLPI_INIZIALI} in riserva; preso 1 bersaglio, ne mancano {CENTRI_PER_LIVELLO - 1} al livello 2."
    assert righe[2] == resoconti.riga_del_tempo_e_del_posto(resoconto.dati["durata_s"], attesi, None) and righe[2].startswith("Tempo ")
    assert resoconto.dati["tiri_del_livello"] == 1 and resoconto.dati["centri_del_livello"] == 1 and resoconto.dati["posizione"] is None
    assert resoconto.tipo == "resoconto" and resoconto.annuncio == f"Colpito, {attesi} punti. {sommario.annuncio}"
    assert resoconto.dati["colpito"] and resoconto.dati["punti"] == attesi and resoconto.dati["totale"] == attesi
    # D29: la mira manuale non ha più il suono del valore preso; i valori del centro stanno lontani dai capi.
    assert banco.contati == ["nuova_partita", "mira_avviata", "sparo", "colpito"]


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


# La mira manuale (D29).

def test_la_mira_parte_dal_centro_e_i_valori_si_muovono_coi_tasti(banco, cartella_di_prova):
    """D29: lo spazio nell'ascolto avvia la mira, una sola per i tre valori, tutti al centro della loro scala, fermi e muti.
    Ogni valore si muove finché il suo tasto è giù, alla VELOCITA_DELLA_MIRA, da un capo all'altro in DURATA_DELLA_SCALA
    secondi, e si ferma dove si sentiva quando il tasto sale; con due tasti di valori diversi giù i due valori si muovono
    insieme, nello stesso puntatore di D31, che porta sempre i tre valori e resta per tutta la mira; il tasto opposto,
    premuto col primo ancora giù, lo fa tornare indietro, e lasciato riporta al verso del primo. La ripetizione del tasto
    tenuto giù non cambia niente, e fuori dalla mira i tasti non fanno niente."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    senti_anche_le_statistiche(banco, statistiche)
    assert partita.muovi("direzione", 1) == [] and partita.lascia("direzione", 1) == [] and partita.tasti_giu() == []
    partita.nuova()
    banco.flusso.avanza(10)
    assert partita.muovi("direzione", 1) == [] and partita.mira is None and partita.tasti_giu() == [] and partita.secondi_della_mira() is None
    assert partita.spazio() == []
    assert partita.fase == "mira" and isinstance(partita.mira, modulo_mira.Mira) and banco.contati[-1] == "mira_avviata"
    assert statistiche.dati["conti"]["mire_avviate"] == 1 and partita.valori == []
    inizio = banco.motore.campione_udibile()
    assert partita.mira.valori(inizio / FS) == [PARTENZA_DELLA_MIRA] * 3 == [0.5] * 3
    # Il puntatore si accende con la mira, nel timbro del proiettile, che nessun bersaglio usa (D32).
    k = banco.motore.prossimo_campione()
    voce = partita._voce_del_puntatore
    assert isinstance(voce, VoceTono) and voce.forma == partita.proiettile == PROIETTILE_FORMA and voce.inizio == k
    assert partita.tasti_giu() == [] and all(bersaglio.portante != voce.forma for bersaglio in partita.bersagli)
    # La direzione verso destra e l'alzo verso il basso, insieme, nello stesso puntatore (D31).
    assert partita.muovi("direzione", 1) == [] and partita.muovi("alzo", -1) == []
    assert partita.tasti_giu() == [("direzione", 1), ("alzo", -1)] and partita._voce_del_puntatore is voce
    # Il puntatore porta i tre valori di ogni istante del tempo dell'audio: a 0,3 e a 0,9 secondi dal tasto la direzione al
    # sesto e all'ottavo decimo, l'alzo al quarto e al secondo, la potenza ferma al centro; e si muove.
    frequenze, pan, guadagni, in_moto = suono_del_puntatore(partita, inizio + np.array((0.3, 0.9)) * FS)
    attese = parametri_del_puntatore(np.array((0.6, 0.8)), np.array((0.4, 0.2)), PARTENZA_DELLA_MIRA)
    for avuti, voluti in zip((frequenze, pan, guadagni), attese, strict=True):
        np.testing.assert_allclose(avuti, voluti, atol=1e-9)
    assert in_moto.tolist() == [True, True]
    # La ripetizione del tasto tenuto giù non cambia niente, e non è una pressione.
    assert partita.muovi("direzione", 1) == [] and partita._voce_del_puntatore is voce and partita.mira.pressioni == 2
    porta_udibile_a(banco, inizio + 0.6 * FS)
    assert partita.battito() == []
    assert partita.mira.valori(inizio / FS + 0.6) == pytest.approx([0.7, 0.3, 0.5])
    # Lasciata, la direzione si ferma dove si sentiva, e il puntatore resta lì, a destra, mentre l'alzo va avanti nella nota.
    assert partita.lascia("direzione", 1) == [] and partita._voce_del_puntatore is voce
    porta_udibile_a(banco, inizio + 0.9 * FS)
    assert partita.mira.valori(inizio / FS + 0.9) == pytest.approx([0.7, 0.2, 0.5])
    _frequenze, pan, _guadagni, in_moto = suono_del_puntatore(partita, inizio + np.array((0.9, 1.0)) * FS)
    np.testing.assert_allclose(pan, 2.0 * 0.7 - 1.0, atol=1e-9)
    assert in_moto.tolist() == [True, True]
    # La freccia su, con la giù ancora giù, fa salire l'alzo; lasciata, l'alzo torna a scendere; lasciata anche la giù, si ferma.
    assert partita.muovi("alzo", 1) == [] and partita.tasti_giu() == [("alzo", -1), ("alzo", 1)] and partita._voce_del_puntatore is voce
    porta_udibile_a(banco, inizio + 1.2 * FS)
    assert partita.mira.valori(inizio / FS + 1.2) == pytest.approx([0.7, 0.3, 0.5])
    assert partita.lascia("alzo", 1) == [] and partita.tasti_giu() == [("alzo", -1)]
    porta_udibile_a(banco, inizio + 1.5 * FS)
    assert partita.lascia("alzo", -1) == [] and partita.tasti_giu() == [] and partita._voce_del_puntatore is voce
    assert partita.mira.valori(inizio / FS + 9.0) == pytest.approx([0.7, 0.2, 0.5])
    # Coi valori fermi il puntatore non è in moto, e suona piano (D32): il prossimo tasto lo riapre.
    assert not suono_del_puntatore(partita, inizio + np.array((1.6, 9.0)) * FS)[3].any()
    # Il tasto che non era giù non cambia niente; un parametro o un verso sconosciuti non sono tasti della mira.
    assert partita.lascia("potenza", 1) == [] and partita.mira.pressioni == 3
    with pytest.raises(ValueError):
        partita.muovi("velocita", 1)
    with pytest.raises(ValueError):
        partita.muovi("direzione", 0)
    # Il puntatore resta acceso per tutta la mira, piano coi valori fermi, e finisce con la mira.
    banco.flusso.avanza(4)
    assert not voce.finita and partita.fase == "mira" and partita.valori == []
    partita.annulla_mira()
    banco.flusso.avanza(4)
    assert voce.finita and partita._voce_del_puntatore is None


def test_le_tacche_e_il_suono_del_capo(banco, monkeypatch):
    """D29: il valore che si muove suona una tacca a ogni decimo della scala che attraversa, in tutte e due le direzioni, la
    tacca di quel decimo, dal punto del puntatore (D31), e al capo si ferma col suono fine_scala, dove il puntatore tace; il
    suono del capo si conta
    quando si sente, al battito, una volta sola, e non si conta, né suona, se il tasto sale prima. Al capo il valore resta,
    e il tasto premuto ancora verso lo stesso capo fa sentire subito fine_scala."""
    suonate = spia_del_motore(monkeypatch, banco.motore)
    partita = banco.partita()
    partita.nuova()
    partita.spazio()
    inizio = banco.motore.campione_udibile()
    prima = len(suonate)
    partita.muovi("direzione", 1)
    voce = partita._voce_del_puntatore
    nuove = suonate[prima:]
    # Dal centro verso destra: le tacche del sesto, del settimo, dell'ottavo e del nono decimo, ognuna dal punto del puntatore,
    # quello del suo decimo, poi il capo, tutto a destra (D31).
    assert len(nuove) == TACCHE - 5
    for (buffer, campione, _voce), decimo in zip(nuove[:-1], range(6, TACCHE), strict=True):
        assert decimo_della_tacca(partita, "direzione", buffer) == decimo
        assert pan_stimato(buffer) == pytest.approx(2.0 * decimo / TACCHE - 1.0, abs=1e-3)
        assert abs(campione - round((inizio / FS + (decimo / TACCHE - 0.5) / VELOCITA_DELLA_MIRA) * FS)) <= 1
    np.testing.assert_allclose(partita._tacche["direzione"][7], tacca_mono("direzione", 7))
    buffer_del_capo, capo, voce_del_capo = nuove[-1]
    assert abs(capo - round((inizio / FS + 0.5 / VELOCITA_DELLA_MIRA) * FS)) <= 1
    assert pan_stimato(buffer_del_capo) == pytest.approx(1.0, abs=1e-3)
    # Il suono del capo si conta quando si sente; lì il puntatore smette di muoversi, e tace.
    porta_udibile_a(banco, capo - 1)
    assert partita.battito() == [] and "fine_scala" not in banco.contati
    porta_udibile_a(banco, capo)
    assert partita.battito() == [] and banco.contati.count("fine_scala") == 1
    porta_udibile_a(banco, capo + 0.5 * FS)
    assert partita.battito() == [] and banco.contati.count("fine_scala") == 1
    assert suono_del_puntatore(partita, [capo - 2, capo - 1])[3].tolist() == [True, True]
    assert not suono_del_puntatore(partita, [capo + 1, capo + 0.5 * FS])[3].any()
    assert not voce.finita and voce_del_capo.finita_al == capo + LUNGHEZZA_FINTA
    assert partita.mira.valori(capo / FS + 5.0)[0] == 1.0 and partita.tasti_giu() == [("direzione", 1)]
    assert partita.lascia("direzione", 1) == []
    # Il tasto premuto di nuovo verso il capo dove il valore sta già: fine_scala subito, senza tacche; lasciato prima che si
    # senta, non suona e non si conta.
    prima = len(suonate)
    k = banco.motore.prossimo_campione()
    partita.muovi("direzione", 1)
    assert [campione for _b, campione, _v in suonate[prima:]] == [k]
    di_nuovo = suonate[-1][2]
    assert partita.lascia("direzione", 1) == []
    banco.flusso.avanza_fino(k + banco.motore.blocco)
    assert partita.battito() == [] and di_nuovo.finita and di_nuovo.finita_al == k and banco.contati.count("fine_scala") == 1
    # Verso sinistra le tacche tornano indietro, dal nono decimo al primo; il tasto lasciato dopo un secondo e due decimi, a
    # sei decimi, ferma le tacche che il valore non ha attraversato e il suono del capo, che finiscono senza suonare.
    porta_udibile_a(banco, banco.motore.campione_udibile() + 0.1 * FS)
    partenza = banco.motore.campione_udibile()
    prima = len(suonate)
    partita.muovi("direzione", -1)
    nuove = suonate[prima:]
    assert [decimo_della_tacca(partita, "direzione", buffer) for buffer, _c, _v in nuove[:-1]] == list(range(TACCHE - 1, 0, -1))
    porta_udibile_a(banco, partenza + 1.2 * FS)
    assert partita.lascia("direzione", -1) == []
    assert partita.mira.valori(partenza / FS + 5.0)[0] == pytest.approx(0.6)
    banco.flusso.avanza_fino(nuove[-1][1] + banco.motore.blocco)
    assert partita.battito() == [] and banco.contati.count("fine_scala") == 1
    for _buffer, campione, tacca in nuove:
        assert tacca.finita
        if campione > partenza + 1.2 * FS + 2 * banco.motore.blocco + banco.flusso.latency * FS:
            assert tacca.finita_al == campione


def test_il_puntatore_porta_i_tre_valori(banco, monkeypatch):
    """D31, chiesto da Gabriele il 22 novembre 2026: il suono della mira è il puntatore, una voce sola che porta sempre i tre
    valori. Muovendo l'alzo il puntatore resta nel punto della direzione e nel volume della potenza, che anche al minimo si
    sente, e le tacche e il capo dell'alzo arrivano da lì; quando la direzione cambia moto, le tacche e il capo dell'alzo
    ancora davanti si rimettono in fila negli stessi campioni, col pan della direzione nel loro istante, e quelli di prima
    tacciono senza suonare."""
    suonate = spia_del_motore(monkeypatch, banco.motore)
    partita = banco.partita()
    partita.nuova()
    partita.spazio()
    # La direzione tutta a destra, e la potenza al minimo.
    inizio = banco.motore.campione_udibile()
    partita.muovi("direzione", 1)
    partita.muovi("potenza", -1)
    porta_udibile_a(banco, inizio + (0.5 / VELOCITA_DELLA_MIRA + OLTRE_IL_CAPO) * FS)
    assert partita.battito() == []
    assert partita.lascia("direzione", 1) == [] and partita.lascia("potenza", -1) == []
    voce = partita._voce_del_puntatore
    # L'alzo sale: il puntatore resta tutto a destra e al volume più basso, e cambia soltanto la nota.
    partenza = banco.motore.campione_udibile()
    prima = len(suonate)
    assert partita.muovi("alzo", 1) == [] and partita._voce_del_puntatore is voce
    frequenze, pan, guadagni, in_moto = suono_del_puntatore(partita, partenza + np.array((0.3, 0.9)) * FS)
    attese = parametri_del_puntatore(1.0, np.array((0.6, 0.8)), 0.0)
    for avuti, voluti in zip((frequenze, pan, guadagni), attese, strict=True):
        np.testing.assert_allclose(avuti, voluti, atol=1e-9)
    assert in_moto.all() and np.all(pan == 1.0)
    np.testing.assert_allclose(20.0 * np.log10(guadagni), PUNTATORE_DB + POTENZA_DB_MINIMO)
    davanti = suonate[prima:]
    assert [decimo_della_tacca(partita, "alzo", buffer) for buffer, _c, _v in davanti[:-1]] == list(range(6, TACCHE))
    assert all(pan_stimato(buffer) == pytest.approx(1.0, abs=1e-3) for buffer, _c, _v in davanti)
    # La direzione torna verso sinistra mentre l'alzo sale: le tacche e il capo dell'alzo ancora davanti tornano in fila negli
    # stessi campioni, ognuno col pan della direzione nel suo istante, che scende da destra verso il centro.
    porta_udibile_a(banco, partenza + 0.1 * FS)
    svolta = banco.motore.campione_udibile()
    prima = len(suonate)
    assert partita.muovi("direzione", -1) == [] and partita._voce_del_puntatore is voce
    rimesse = suonate[prima:prima + len(davanti)]
    assert [campione for _b, campione, _v in rimesse] == [campione for _b, campione, _v in davanti]
    assert [decimo_della_tacca(partita, "alzo", buffer) for buffer, _c, _v in rimesse[:-1]] == list(range(6, TACCHE))
    for buffer, campione, _voce in rimesse:
        direzione = max(0.0, 1.0 - VELOCITA_DELLA_MIRA * (campione - svolta) / FS)
        assert pan_stimato(buffer) == pytest.approx(2.0 * direzione - 1.0, abs=1e-3), campione
    # Dopo le rimesse, le tacche della direzione, dal nono decimo al primo, e il suo capo, tutto a sinistra.
    proprie = suonate[prima + len(davanti):]
    assert [decimo_della_tacca(partita, "direzione", buffer) for buffer, _c, _v in proprie[:-1]] == list(range(TACCHE - 1, 0, -1))
    assert pan_stimato(proprie[-1][0]) == pytest.approx(-1.0, abs=1e-3)
    # Quelle di prima tacciono senza suonare; le rimesse suonano intere.
    banco.flusso.avanza_fino(davanti[-1][1] + LUNGHEZZA_FINTA + banco.motore.blocco)
    assert all(vecchia.finita and vecchia.finita_al == campione for _b, campione, vecchia in davanti)
    assert all(nuova.finita and nuova.finita_al == campione + len(buffer) for buffer, campione, nuova in rimesse)


def test_il_puntatore_si_apre_e_si_chiude_senza_click(banco):
    """D31, D32: il puntatore suona pieno dove almeno un valore si muove, e coi valori fermi PUNTATORE_FERMO_DB più piano; il suo
    inviluppo passa dall'uno all'altro in SFUMATURA_STOP, un campione alla volta, senza salti, anche a cavallo di due blocchi. La
    prova fa leggere da sola la voce, un blocco dopo l'altro, con l'alzo che si muove dal decimo blocco e si ferma al trentesimo, e
    finisce prima che il motore la suoni."""
    partita = banco.partita()
    partita.nuova()
    partita.spazio()
    partita.muovi("alzo", 1)
    voce = partita._voce_del_puntatore
    assi = assi_della_mira(partita)
    alzo = partita.mira.assi["alzo"]
    blocco = banco.motore.blocco
    sfumatura = round(SFUMATURA_STOP * FS)
    da, a = 10 * blocco, 30 * blocco
    alzo.stato = (PARTENZA_DELLA_MIRA, 0, 0.0)
    pezzi = []
    for inizio in range(0, 40 * blocco, blocco):
        if inizio == da:
            alzo.stato = (PARTENZA_DELLA_MIRA, 1, da / FS)
        if inizio == a:
            alzo.stato = (alzo.valore(a / FS), 0, a / FS)
        tempi = np.arange(inizio, inizio + blocco) / FS
        _frequenze, _pan, guadagni = voce.parametri(None, tempi)
        pieni = modulo_partita._suono_del_puntatore(assi, tempi)[2]
        pezzi.append(guadagni / pieni)
    inviluppo = np.concatenate(pezzi)
    fermo = float(db_in_lineare(PUNTATORE_FERMO_DB))
    np.testing.assert_allclose(inviluppo[:da], fermo)
    np.testing.assert_allclose(inviluppo[a + sfumatura:], fermo)
    np.testing.assert_allclose(inviluppo[da + sfumatura:a], 1.0)
    assert np.max(np.abs(np.diff(inviluppo))) <= 1.0 / (SFUMATURA_STOP * FS) + 1e-12
    assert fermo < inviluppo[da] < inviluppo[da + sfumatura // 2] < 1.0 and fermo < inviluppo[a + sfumatura // 2] < 1.0


def test_il_puntatore_si_accende_con_la_mira_e_alla_ripresa(banco):
    """D32: il puntatore si accende allo spazio che avvia la mira, nel timbro del proiettile della partita, che nessun bersaglio
    usa, e coi valori fermi suona PUNTATORE_FERMO_DB più piano di come suonerebbe in moto, col volume della potenza; tace alla
    pausa, si riaccende alla ripresa e finisce con la mira."""
    partita = banco.partita(proiettile="triangolo")
    partita.nuova()
    assert partita._voce_del_puntatore is None
    partita.spazio()
    voce = partita._voce_del_puntatore
    assert voce.forma == partita.proiettile == "triangolo" and all(bersaglio.portante != "triangolo" for bersaglio in partita.bersagli)
    campioni = banco.motore.prossimo_campione() + np.arange(banco.motore.blocco)
    frequenze, pan, guadagni = voce.parametri(None, campioni / FS)
    attese, attesi, pieni, in_moto = suono_del_puntatore(partita, campioni)
    np.testing.assert_allclose(frequenze, attese)
    np.testing.assert_allclose(pan, attesi)
    assert not in_moto.any()
    np.testing.assert_allclose(20.0 * np.log10(guadagni / pieni), PUNTATORE_FERMO_DB)
    np.testing.assert_allclose(20.0 * np.log10(pieni), PUNTATORE_DB + POTENZA_DB_MINIMO * (1.0 - PARTENZA_DELLA_MIRA))
    # La pausa lo spegne, e la ripresa lo riaccende, una voce nuova nello stesso timbro.
    assert partita.pausa_automatica(True) == [] and partita.in_pausa and partita._voce_del_puntatore is None
    banco.flusso.avanza(4)
    assert voce.finita
    partita.pausa_automatica(False)
    ripresa = partita._voce_del_puntatore
    assert not partita.in_pausa and ripresa is not None and ripresa is not voce and ripresa.forma == "triangolo"
    # Con la mira annullata finisce.
    partita.annulla_mira()
    banco.flusso.avanza(4)
    assert partita._voce_del_puntatore is None and ripresa.finita


def test_lo_spazio_nella_mira_suona_non_ora(banco):
    """D29: nella mira spara l'Invio, e lo spazio, come col proiettile in volo, fa sentire soltanto il non ora: la mira resta
    com'era, coi suoi valori e i suoi tasti."""
    partita = banco.partita()
    partita.nuova()
    partita.spazio()
    partita.muovi("potenza", 1)
    mira = partita.mira
    prima = len(banco.contati)
    assert partita.spazio() == [] and banco.contati[prima:] == ["non_ora"]
    assert partita.fase == "mira" and partita.mira is mira and partita.tasti_giu() == [("potenza", 1)]


def test_l_invio_spara_coi_valori_dell_istante_udibile(banco, cartella_di_prova, monkeypatch):
    """D29: Invio fa partire il colpo coi tre valori dell'istante in cui si è sentito, anche coi tasti ancora giù, che da lì non
    contano più, e la mira finisce. Il tempo di gioco scorre nella mira, e il colpo parte da un campione scelto prima del conto
    del volo, ANTICIPO_DELLO_SPARO dopo il prossimo campione sicuro: il volo parte dal tempo di gioco di quel campione, e lo
    sparo e la voce del proiettile da lì. Le statistiche contano ogni valore fra quelli della mira manuale, e la mira sparata
    coi suoi secondi e i tasti premuti. Fuori dalla mira Invio non fa niente."""
    suonate = spia_del_motore(monkeypatch, banco.motore)
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    banco.flusso.avanza(10)
    assert partita.spara() == [] and partita.fase == "ascolto" and partita.tiri == 0
    avviata = banco.motore.prossimo_campione()
    partita.spazio()
    inizio = banco.motore.campione_udibile()
    partita.muovi("direzione", 1)
    partita.muovi("alzo", -1)
    porta_udibile_a(banco, inizio + 0.6 * FS)
    voci = [partita._voce_del_puntatore]
    sparo = banco.motore.prossimo_campione() + round(ANTICIPO_DELLO_SPARO * FS)
    assert partita.spara() == []
    valori = [0.5 + 0.6 * VELOCITA_DELLA_MIRA, 0.5 - 0.6 * VELOCITA_DELLA_MIRA, 0.5]
    assert partita.fase == "volo" and partita.valori == pytest.approx(valori, abs=1e-9) and partita.tiri == 1
    assert partita.mira is None and partita.tasti_giu() == [] and partita.secondi_della_mira() is None and partita._voce_del_puntatore is None
    volo = partita.ultimo_volo
    assert volo.partenza == pytest.approx(banco.motore.tempo_di_gioco(sparo)) and volo.partenza > banco.motore.tempo_di_gioco(inizio + 0.6 * FS)
    assert partita._voce_proiettile.inizio == sparo and any(campione == sparo for _b, campione, _v in suonate)
    # Invio di nuovo, col proiettile in volo, non fa niente.
    assert partita.spara() == [] and partita.tiri == 1
    banco.flusso.avanza(4)
    assert all(voce.finita for voce in voci)
    mira = statistiche.dati["mira"]
    assert [mira[parametro]["manuali"] for parametro in PARAMETRI] == [1, 1, 1] and [mira[parametro]["ai_capi"] for parametro in PARAMETRI] == [0, 0, 0]
    assert [mira[parametro]["somma"] for parametro in PARAMETRI] == pytest.approx(valori, abs=1e-6)
    conti = statistiche.dati["conti"]
    assert (conti["mire_sparate"], conti["tasti_della_mira"], conti["mire_scadute"]) == (1, 2, 0)
    assert statistiche.dati["somme"]["secondi_di_mira"] == pytest.approx((inizio + 0.6 * FS - avviata) / FS, abs=1e-3)
    assert fino_alla_risoluzione(banco, partita)[0].tipo == "resoconto"


def test_il_tempo_scorre_e_i_bersagli_suonano_pieni_nella_mira(banco):
    """D29, al posto di D4, che non vale più: durante la mira il tempo di gioco scorre e i bersagli restano a volume pieno,
    perché chi gioca deve capire l'esatta distanza del bersaglio; il colpo parte dal tempo di gioco dello sparo, non da
    quello dello spazio."""
    partita = banco.partita()
    partita.nuova()
    banco.flusso.avanza(20)
    voce = partita._voci[partita.bersagli[0]]
    partita.spazio()
    banco.flusso.avanza(4)
    allo_spazio = banco.motore.tempo_di_gioco()
    banco.flusso.avanza(40)
    assert banco.motore.tempo_scorre and banco.motore.tempo_di_gioco() == pytest.approx(allo_spazio + 40 * banco.motore.blocco / FS * VELOCITA_DEL_TEMPO)
    assert voce.guadagno == 1.0
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    assert partita.ultimo_volo.partenza > allo_spazio + 40 * banco.motore.blocco / FS + DURATA_DELLA_SCALA / 2
    banco.flusso.avanza(4)
    assert voce.guadagno == 1.0 and banco.motore.tempo_scorre


def test_i_mobili_si_muovono_durante_la_mira(banco, monkeypatch):
    """D29: i mobili continuano a muoversi durante la mira, e la mira va su dove il bersaglio sarà all'arrivo del proiettile:
    il colpo mirato così lo prende, mentre quello mirato dove il bersaglio stava allo spazio, come col tempo fermo di D4, lo
    manca."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    partita = banco.partita()
    partita.nuova()
    partita.livello = LIVELLO_DEI_MOBILI
    partita._nuovo_bersaglio()
    mobile = partita.bersagli[-1]
    assert mobile.mobile and not modulo_partita.nella_palestra(LIVELLO_DEI_MOBILI)
    allo_spazio = banco.motore.tempo_di_gioco(banco.motore.prossimo_campione())
    dove_era = mobile.posizione(allo_spazio)
    colpisci(banco, partita, mobile)
    volo = partita.ultimo_volo
    assert volo.colpito is mobile and volo.partenza > allo_spazio + DURATA_DELLA_SCALA / 2
    assert np.linalg.norm(mobile.posizione(volo.fine) - dove_era) > mobile.raggio
    fermo = calcola_volo(Traiettoria(*parametri_del_tiro(*soluzione(dove_era))), [mobile], volo.partenza)
    assert fermo.colpito is None
    messaggi = fino_alla_risoluzione(banco, partita)
    assert messaggi[0].dati["colpito"] and partita.fase == "ascolto"


def test_il_bonus_esce_durante_la_mira(banco, bonus_sicuro, monkeypatch):
    """D29: anche il bonus si muove durante la mira, e può uscire: l'uscita si dice al battito della mira, e la mira va avanti."""
    partita = banco.partita()
    partita.nuova()
    bonus = _bonus_quasi_uscito(banco, partita, 0.5)
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    partita.spazio()
    assert partita.fase == "mira"
    usciti = []
    for _blocco in range(BLOCCHI_MASSIMI):
        banco.flusso.avanza(1)
        usciti.extend(partita.battito())
        if usciti:
            break
    assert [m.annuncio for m in usciti] == ["Bonus uscito."] and banco.motore.tempo_di_gioco() >= bonus.tempo_di_uscita
    assert partita.fase == "mira" and partita.bonus is None and bonus not in partita.bersagli
    assert partita.righe_della_barra()[2] == "Bersagli in volo: 1"
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    assert fino_alla_risoluzione(banco, partita)[0].tipo == "resoconto"


def test_tempo_della_mira():
    """D29: nella palestra la mira non ha tempo; dopo, 30 secondi al primo livello fuori, il quarto, uno in meno a ogni livello,
    10 secondi dal livello 24 in poi."""
    assert PALESTRA_FINO_AL_LIVELLO == 3
    assert [tempo_della_mira(livello) for livello in range(1, PALESTRA_FINO_AL_LIVELLO + 1)] == [None] * PALESTRA_FINO_AL_LIVELLO
    assert [tempo_della_mira(livello) for livello in (4, 5, 6, 23, 24, 25, 60)] == [30.0, 29.0, 28.0, 11.0, 10.0, 10.0, 10.0]
    assert LIVELLO_DELLA_MIRA_BREVE == 24 and tempo_della_mira(LIVELLO_DELLA_MIRA_BREVE) == TEMPO_DELLA_MIRA_MINIMO
    assert "tempo_della_mira" in modulo_partita.__all__


def test_la_mira_a_tempo_e_i_suoi_tick(banco, monkeypatch):
    """D29: fuori dalla palestra la mira è a tempo, per tutti e tre i valori insieme, dallo spazio, finiti gli avvisi: un tick
    al secondo, sempre più acuto, scandisce il tempo che scorre, dal primo secondo all'ultimo prima della scadenza;
    secondi_della_mira dice quanti ne restano, e la barra di stato li dice coi valori. Nella palestra niente tempo e niente
    tick. La mira avviata mentre gli avvisi suonano ancora la prova test_il_tempo_della_mira_parte_dopo_gli_avvisi."""
    suonate = spia_del_motore(monkeypatch, banco.motore)
    partita = banco.partita()
    partita.nuova()
    prima = len(suonate)
    partita.spazio()
    assert len(suonate) == prima + 1 and partita.secondi_della_mira() is None and partita._scadenza is None
    assert partita.righe_della_barra()[-2] == "Direzione davanti, alzo 50,0 gradi, potenza 50%, senza tempo"
    partita.annulla_mira()
    partita.livello = LIVELLO_DELLA_MIRA_BREVE
    totale = round(TEMPO_DELLA_MIRA_MINIMO)
    finiti_gli_avvisi(banco, partita)
    prima = len(suonate)
    k = banco.motore.prossimo_campione()
    partita.spazio()
    tick = suonate[prima + 1:]
    assert len(tick) == totale - 1
    for secondo, (buffer, campione, _voce) in enumerate(tick, start=1):
        assert campione == k + round(secondo * FS)
        np.testing.assert_allclose(buffer, tick_del_tempo(secondo, totale))
    # Sempre più acuti: il primo e l'ultimo non sono lo stesso bip.
    assert not np.allclose(tick[0][0], tick[-1][0])
    assert partita.secondi_della_mira() == pytest.approx(totale) and partita._scadenza == k + round(totale * FS)
    assert partita.righe_della_barra()[-2] == f"Direzione davanti, alzo 50,0 gradi, potenza 50%, {totale} secondi"
    porta_udibile_a(banco, k + 3.5 * FS)
    assert partita.battito() == [] and partita.secondi_della_mira() == pytest.approx(totale - 3.5)
    assert partita.righe_della_barra()[-2] == f"Direzione davanti, alzo 50,0 gradi, potenza 50%, {math.ceil(totale - 3.5)} secondi"
    # I tick si sentono: quelli già passati hanno suonato tutto il loro bip.
    banco.flusso.avanza(2)
    assert tick[0][2].finita and tick[0][2].finita_al == tick[0][1] + len(tick[0][0])
    assert not tick[-1][2].finita


def test_allo_scadere_il_colpo_parte_da_solo(banco, cartella_di_prova):
    """D29: allo scadere della mira a tempo il colpo parte da solo coi valori dell'istante della scadenza, anche col tasto ancora
    giù e il battito che se ne accorge dopo; la mira si conta fra quelle sparate e fra le scadute, coi suoi secondi interi."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    partita.livello = LIVELLO_DELLA_MIRA_BREVE
    finiti_gli_avvisi(banco, partita)
    avviata = banco.motore.prossimo_campione()
    partita.spazio()
    scadenza = partita._scadenza
    assert scadenza == avviata + round(TEMPO_DELLA_MIRA_MINIMO * FS)
    # La direzione si muove verso destra da mezzo secondo prima della scadenza, e il tasto resta giù.
    porta_udibile_a(banco, scadenza - 0.5 * FS)
    assert partita.battito() == []
    partita.muovi("direzione", 1)
    porta_udibile_a(banco, scadenza - 1)
    assert partita.battito() == [] and partita.fase == "mira" and partita.secondi_della_mira() == pytest.approx(1 / FS)
    porta_udibile_a(banco, scadenza + 0.2 * FS)
    assert partita.battito() == []
    assert partita.fase == "volo" and partita.valori == pytest.approx([0.5 + 0.5 * VELOCITA_DELLA_MIRA, 0.5, 0.5], abs=1e-9)
    assert partita.mira is None and partita._scadenza is None and partita.tasti_giu() == []
    conti = statistiche.dati["conti"]
    assert (conti["mire_sparate"], conti["mire_scadute"], conti["tasti_della_mira"]) == (1, 1, 1)
    assert statistiche.dati["somme"]["secondi_di_mira"] == pytest.approx(TEMPO_DELLA_MIRA_MINIMO, abs=1e-3)
    messaggi = fino_alla_risoluzione(banco, partita)
    assert messaggi[0].tipo == "resoconto" and partita.fase == "ascolto" and partita.tiri == 1


def test_l_invio_e_i_tasti_dopo_la_scadenza_non_spostano_il_colpo(banco, cartella_di_prova):
    """D29: la mira a tempo finisce alla scadenza, e il colpo parte coi valori di quell'istante. L'Invio sentito dopo la
    scadenza, prima che il battito se ne accorga, trova la mira già scaduta: il colpo parte coi valori della scadenza, e la
    mira conta fra le scadute, come lo spazio che arrivava dopo la scadenza di una barra la faceva scadere (D25). Così il tasto
    che sale dopo la scadenza, e quello che scende: il valore del colpo resta quello della scadenza."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    partita.livello = LIVELLO_DELLA_MIRA_BREVE
    alla_scadenza = [0.5 + 0.5 * VELOCITA_DELLA_MIRA, 0.5, 0.5]
    # L'Invio due centesimi dopo la scadenza.
    partita.spazio()
    scadenza = partita._scadenza
    porta_udibile_a(banco, scadenza - 0.5 * FS)
    partita.muovi("direzione", 1)
    porta_udibile_a(banco, scadenza + 0.02 * FS)
    assert partita.spara() == [] and partita.fase == "volo"
    assert partita.valori == pytest.approx(alla_scadenza, abs=1e-9)
    assert statistiche.dati["conti"]["mire_scadute"] == 1
    fino_alla_risoluzione(banco, partita)
    # Il tasto che sale due centesimi dopo la scadenza, e il battito dopo di lui.
    partita.spazio()
    scadenza = partita._scadenza
    porta_udibile_a(banco, scadenza - 0.5 * FS)
    partita.muovi("direzione", 1)
    porta_udibile_a(banco, scadenza + 0.02 * FS)
    partita.lascia("direzione", 1)
    assert partita.battito() == [] and partita.fase == "volo"
    assert partita.valori == pytest.approx(alla_scadenza, abs=1e-9)
    assert statistiche.dati["conti"]["mire_scadute"] == 2
    fino_alla_risoluzione(banco, partita)
    # Il tasto di un altro valore che scende due centesimi dopo la scadenza, prima del battito: non muove l'alzo, e il colpo
    # parte coi valori della scadenza, fra le scadute.
    partita.spazio()
    scadenza = partita._scadenza
    porta_udibile_a(banco, scadenza - 0.5 * FS)
    partita.muovi("direzione", 1)
    porta_udibile_a(banco, scadenza + 0.02 * FS)
    assert partita.muovi("alzo", 1) == [] and partita.fase == "volo"
    assert partita.valori == pytest.approx(alla_scadenza, abs=1e-9)
    assert statistiche.dati["conti"]["mire_scadute"] == 3 and statistiche.dati["conti"]["mire_sparate"] == 3


def test_la_pausa_nella_mira_a_tempo_conserva_i_secondi(banco, monkeypatch):
    """D29 e D14: la pausa, qui quella della finestra, che c'è in tutti i livelli, ferma la mira a tempo coi secondi che le
    restano: i tick tacciono, il valore che si muoveva si ferma dove si sentiva, il suo tasto vale come lasciato, e la mira non
    scade, per quanto duri la pausa; in pausa i tasti dei valori non fanno niente, e l'Invio e lo spazio fanno sentire il non
    ora. Alla ripresa il conto riparte dai secondi rimasti, coi tick dei secondi che mancano, e la scadenza arriva dopo di
    loro, coi valori di prima della pausa."""
    suonate = spia_del_motore(monkeypatch, banco.motore)
    partita = banco.partita()
    partita.nuova()
    partita.livello = LIVELLO_DELLA_MIRA_BREVE
    totale = round(TEMPO_DELLA_MIRA_MINIMO)
    voce = partita._voci[partita.bersagli[0]]
    finiti_gli_avvisi(banco, partita)
    k = banco.motore.prossimo_campione()
    partita.spazio()
    assert partita._mira_dal == k
    porta_udibile_a(banco, k + 3.25 * FS)
    partita.muovi("direzione", -1)
    asse = partita._voce_del_puntatore
    porta_udibile_a(banco, k + 3.4 * FS)
    assert partita.pausa_automatica(True) == [] and partita.in_pausa and partita.fase == "mira"
    rimasti = totale - 3.4
    fermi = [0.5 - 0.15 * VELOCITA_DELLA_MIRA, 0.5, 0.5]
    assert partita.secondi_della_mira() == pytest.approx(rimasti) and partita.tasti_giu() == [] and partita._voce_del_puntatore is None
    assert partita.mira.valori(k / FS + 60.0) == pytest.approx(fermi)
    assert partita._tick_in_fila == [] and partita._scadenza is None
    assert partita.righe_della_barra()[-2:] == [resoconti.riga_dei_valori_della_mira(*parametri_del_tiro(*fermi)[:2], fermi[2], rimasti),
                                                "In pausa: Ctrl+P per riprendere"]
    # In pausa: i tasti non fanno niente, Invio e spazio il non ora.
    assert partita.muovi("alzo", 1) == [] and partita.tasti_giu() == [] and partita._voce_del_puntatore is None
    prima = len(banco.contati)
    assert partita.spara() == [] and partita.spazio() == [] and banco.contati[prima:] == ["non_ora", "non_ora"] and partita.fase == "mira"
    # Una pausa lunga il doppio della mira: niente scadenza, e passata la rampa del guadagno tutto tace, tick compresi.
    uscita = []
    for _secondo in range(2 * totale):
        uscita.append(banco.flusso.avanza_secondi(1.0))
        assert partita.battito() == []
    uscita = np.concatenate(uscita)
    assert not uscita[round(0.2 * FS):].any()
    assert partita.fase == "mira" and partita.secondi_della_mira() == pytest.approx(rimasti) and asse.finita and voce.guadagno == 0.0
    # La ripresa: i tick dei secondi che mancano, dal campione della ripresa, e la scadenza dopo i secondi rimasti.
    prima = len(suonate)
    ripresa = banco.motore.prossimo_campione()
    assert partita.pausa_automatica(False) == [] and not partita.in_pausa
    tick = suonate[prima:]
    secondi = [secondo for secondo in range(1, totale) if secondo > totale - rimasti]
    assert len(tick) == len(secondi) == totale - 4
    for secondo, (buffer, campione, _voce) in zip(secondi, tick, strict=True):
        assert abs(campione - (ripresa + round((secondo - (totale - rimasti)) * FS))) <= 1
        np.testing.assert_allclose(buffer, tick_del_tempo(secondo, totale))
    assert abs(partita._scadenza - (ripresa + round(rimasti * FS))) <= 1
    banco.flusso.avanza(10)
    assert voce.guadagno == 1.0 and banco.motore.tempo_scorre and partita.tasti_giu() == []
    porta_udibile_a(banco, partita._scadenza - 1)
    assert partita.battito() == [] and partita.fase == "mira"
    porta_udibile_a(banco, partita._scadenza)
    assert partita.battito() == [] and partita.fase == "volo"
    assert partita.valori == pytest.approx(fermi, abs=1e-9)


@pytest.mark.parametrize("a_tempo", [False, True])
def test_i_secondi_della_mira_nelle_statistiche_non_contano_le_pause(banco, cartella_di_prova, a_tempo):
    """D29 e D14: le statistiche sommano i secondi di ogni mira finita col colpo, per il tempo medio di una mira, e come la durata
    della partita non contano il tempo in cui la partita è stata ferma. Una mira della palestra, un secondo prima e uno dopo
    un minuto di pausa chiesta con Ctrl+P, conta due secondi; una mira a tempo scaduta, con mezzo minuto della pausa della
    finestra nel mezzo, conta i suoi dieci secondi. La mira comincia finiti gli avvisi della partita nuova, perché il suo tempo
    partirebbe soltanto dopo di loro: i secondi della mira che li aspetta la prova test_il_tempo_della_mira_parte_dopo_gli_avvisi."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    if a_tempo:
        partita.livello = LIVELLO_DELLA_MIRA_BREVE
    finiti_gli_avvisi(banco, partita)
    avviata = banco.motore.prossimo_campione()
    partita.spazio()
    porta_udibile_a(banco, avviata + 1.0 * FS)
    if a_tempo:
        partita.pausa_automatica(True)
        banco.flusso.avanza_secondi(30.0)
        partita.pausa_automatica(False)
        porta_udibile_a(banco, partita._scadenza)
        assert partita.battito() == [] and partita.fase == "volo"
        attesi = TEMPO_DELLA_MIRA_MINIMO
    else:
        partita.pausa()
        banco.flusso.avanza_secondi(60.0)
        ripresa = banco.motore.prossimo_campione()
        partita.pausa()
        porta_udibile_a(banco, ripresa + 1.0 * FS)
        assert partita.spara() == [] and partita.fase == "volo"
        attesi = 2.0
    assert statistiche.dati["conti"]["mire_sparate"] == 1
    assert statistiche.dati["somme"]["secondi_di_mira"] == pytest.approx(attesi, abs=0.1)


def test_il_tempo_della_mira_parte_dopo_l_avviso_della_partita_nuova(banco, cartella_di_prova, monkeypatch):
    """D29, dalle prove del 22 novembre 2026: il tempo della mira a tempo, la scadenza e i tick, parte dallo spazio o, se gli
    avvisi in fila suonano ancora, dalla loro fine, stacco compreso: la mira avviata subito dopo nuova() aspetta la fine
    dell'avviso della partita nuova. Fino a lì i secondi della mira restano tutti, e la barra li dice tutti, ma i valori si
    muovono già; allo scadere il colpo parte da solo, e le statistiche contano i secondi della mira dallo spazio, l'attesa
    dell'avviso compresa, perché è tempo di mira anche quello."""
    avviso = np.full((round(1.2 * FS), 2), AMPIEZZA_FINTA, dtype=np.float32)
    corto = np.full((LUNGHEZZA_FINTA, 2), AMPIEZZA_FINTA, dtype=np.float32)
    monkeypatch.setattr(suoni, "_rendi_preset", lambda nome: avviso if nome == suoni.EVENTI["nuova_partita"] else corto)
    suonate = spia_del_motore(monkeypatch, banco.motore)
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    stacco = round(modulo_partita.STACCO_FRA_GLI_AVVISI * FS)
    partita.nuova()
    partita.livello = LIVELLO_DELLA_MIRA_BREVE
    totale = round(TEMPO_DELLA_MIRA_MINIMO)
    nuova_partita = next(voce for buffer, _campione, voce in suonate if buffer is avviso)
    fine = nuova_partita.inizio + len(avviso) + stacco
    avviata = banco.motore.prossimo_campione()
    assert partita._fine_degli_avvisi == fine and fine > avviata + FS
    prima = len(suonate)
    partita.spazio()
    tick = suonate[prima + 1:]
    assert partita.fase == "mira" and partita._mira_dal == fine and partita._scadenza == fine + round(totale * FS)
    assert [campione for _buffer, campione, _voce in tick] == [fine + round(secondo * FS) for secondo in range(1, totale)]
    # Mentre l'avviso suona i secondi restano tutti, e la direzione si muove già.
    assert partita.secondi_della_mira() == pytest.approx(totale)
    assert partita.righe_della_barra()[-2] == f"Direzione davanti, alzo 50,0 gradi, potenza 50%, {totale} secondi"
    porta_udibile_a(banco, fine - 0.5 * FS)
    assert partita.battito() == [] and partita.secondi_della_mira() == pytest.approx(totale)
    assert partita.muovi("direzione", 1) == []
    porta_udibile_a(banco, fine)
    assert partita.lascia("direzione", 1) == [] and partita.battito() == [] and partita.secondi_della_mira() == pytest.approx(totale)
    porta_udibile_a(banco, fine + 2.5 * FS)
    assert partita.battito() == [] and partita.secondi_della_mira() == pytest.approx(totale - 2.5)
    # Allo scadere il colpo parte da solo, coi valori di quel momento.
    porta_udibile_a(banco, partita._scadenza)
    assert partita.battito() == [] and partita.fase == "volo"
    assert partita.valori == pytest.approx([0.5 + 0.5 * VELOCITA_DELLA_MIRA, 0.5, 0.5], abs=1e-9)
    assert statistiche.dati["conti"]["mire_scadute"] == 1
    assert statistiche.dati["somme"]["secondi_di_mira"] == pytest.approx((fine + totale * FS - avviata) / FS, abs=1e-3)
    assert fino_alla_risoluzione(banco, partita)[0].tipo == "resoconto"


def test_il_tempo_della_mira_non_copre_la_fanfara_del_livello(banco, monkeypatch):
    """D29, dalle prove del 22 novembre 2026: fuori dalla palestra il centro che chiude il livello mette in fila, dopo l'effetto
    dell'impatto, la fanfara del livello nuovo; lo spazio dato mentre suona avvia la mira, ma il suo tempo, coi tick, parte alla
    fine della fanfara, stacco compreso, e fino a lì i secondi restano tutti."""
    fanfara = np.full((round(1.5 * FS), 2), AMPIEZZA_FINTA, dtype=np.float32)
    corto = np.full((LUNGHEZZA_FINTA, 2), AMPIEZZA_FINTA, dtype=np.float32)
    monkeypatch.setattr(suoni, "_rendi_preset", lambda nome: fanfara if nome == suoni.EVENTI["livello_su"] else corto)
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    suonate = spia_del_motore(monkeypatch, banco.motore)
    stacco = round(modulo_partita.STACCO_FRA_GLI_AVVISI * FS)
    partita = banco.partita()
    partita.nuova()
    partita.livello = PALESTRA_FINO_AL_LIVELLO + 1
    partita.centri_del_livello = CENTRI_PER_LIVELLO - 1
    colpisci(banco, partita)
    messaggi = fino_alla_risoluzione(banco, partita)
    assert [m.tipo for m in messaggi][:2] == ["resoconto", "livello"] and partita.livello == PALESTRA_FINO_AL_LIVELLO + 2
    assert partita.fase == "ascolto"
    voce = next(voce for buffer, _campione, voce in suonate if buffer is fanfara)
    fine = voce.inizio + len(fanfara) + stacco
    k = banco.motore.prossimo_campione()
    assert partita._fine_degli_avvisi == fine and k + FS < fine
    totale = tempo_della_mira(partita.livello)
    prima = len(suonate)
    partita.spazio()
    tick = suonate[prima + 1:]
    assert partita.fase == "mira" and partita._mira_dal == fine and partita._scadenza == fine + round(totale * FS)
    assert len(tick) == math.ceil(totale) - 1 and tick[0][1] == fine + FS
    porta_udibile_a(banco, fine - 1)
    assert partita.battito() == [] and partita.secondi_della_mira() == pytest.approx(totale)
    porta_udibile_a(banco, fine + 1.5 * FS)
    assert partita.battito() == [] and partita.secondi_della_mira() == pytest.approx(totale - 1.5)


def test_i_suoni_della_mira_gia_partiti_finiscono(banco, monkeypatch):
    """D29, dalle prove del 22 novembre 2026: quando il moto di un valore cambia, tacciono soltanto le tacche e il suono del capo
    messi in fila e non ancora partiti; quelli che si sentono già finiscono. Il tasto lasciato poco dopo che il valore è arrivato
    al capo, o un tocco brevissimo verso il capo dove il valore sta già, non troncano fine_scala; il tasto lasciato prima che il
    suono parta lo ferma, e non suona. Il tasto opposto premuto mentre una tacca suona la lascia finire, e le tacche e il capo
    davanti, che il valore non attraversa più, tacciono senza suonare."""
    capo_lungo = np.full((round(0.3 * FS), 2), AMPIEZZA_FINTA, dtype=np.float32)
    corto = np.full((LUNGHEZZA_FINTA, 2), AMPIEZZA_FINTA, dtype=np.float32)
    monkeypatch.setattr(suoni, "_rendi_preset", lambda nome: capo_lungo if nome == suoni.EVENTI["fine_scala"] else corto)

    def e_il_capo(buffer):
        # Il capo arriva dal punto del puntatore, qui il centro: lo stesso suono, collocato di nuovo (D31).
        return len(buffer) == len(capo_lungo) and np.allclose(buffer, capo_lungo, rtol=0, atol=1e-6)

    suonate = spia_del_motore(monkeypatch, banco.motore)
    partita = banco.partita()
    partita.nuova()
    partita.spazio()
    # La potenza va al massimo, e il tasto sale un centesimo dopo il capo, mentre fine_scala suona: tutto finisce di suonare.
    prima = len(suonate)
    assert partita.muovi("potenza", 1) == []
    buffer, capo, voce_del_capo = suonate[-1]
    tacche = suonate[prima:-1]
    assert e_il_capo(buffer) and len(tacche) == TACCHE - 6
    porta_udibile_a(banco, capo + OLTRE_IL_CAPO * FS)
    contati = banco.contati.count("fine_scala")
    assert partita.lascia("potenza", 1) == [] and partita.mira.valori(capo / FS + 1.0)[2] == 1.0
    # Il suono del capo già partito si conta al rilascio, perché il battito non lo trova più in fila.
    assert banco.contati.count("fine_scala") == contati + 1
    banco.flusso.avanza_fino(capo + len(capo_lungo) + banco.motore.blocco)
    assert voce_del_capo.finita and voce_del_capo.finita_al == capo + len(capo_lungo)
    assert all(voce.finita and voce.finita_al == campione + len(tacca) for tacca, campione, voce in tacche)
    # Il tocco brevissimo verso il capo dove la potenza sta già: fine_scala parte subito, e il tasto che sale cinque centesimi
    # dopo non lo tronca.
    k = banco.motore.prossimo_campione()
    assert partita.muovi("potenza", 1) == []
    buffer, inizio, di_nuovo = suonate[-1]
    assert e_il_capo(buffer) and inizio == k
    porta_udibile_a(banco, k + 0.05 * FS)
    assert partita.lascia("potenza", 1) == []
    banco.flusso.avanza_fino(k + len(capo_lungo) + banco.motore.blocco)
    assert di_nuovo.finita and di_nuovo.finita_al == k + len(capo_lungo)
    # Lasciato prima che parta, invece, fine_scala non suona.
    k = banco.motore.prossimo_campione()
    assert partita.muovi("potenza", 1) == []
    ancora = suonate[-1][2]
    assert e_il_capo(suonate[-1][0]) and partita.lascia("potenza", 1) == []
    banco.flusso.avanza_fino(k + banco.motore.blocco)
    assert ancora.finita and ancora.finita_al == k
    # Il suono del capo che non è partito non si conta.
    assert banco.contati.count("fine_scala") == contati + 2
    # Dal massimo verso il minimo, e a metà della prima tacca il tasto opposto: la tacca finisce, le altre e il capo tacciono.
    prima = len(suonate)
    assert partita.muovi("potenza", -1) == []
    davanti = suonate[prima:]
    assert e_il_capo(davanti[-1][0]) and len(davanti) == TACCHE
    tacca, campione, prima_tacca = davanti[0]
    assert decimo_della_tacca(partita, "potenza", tacca) == TACCHE - 1
    porta_udibile_a(banco, campione + len(tacca) // 2)
    assert partita.muovi("potenza", 1) == [] and partita.lascia("potenza", -1) == [] and partita.lascia("potenza", 1) == []
    banco.flusso.avanza_fino(davanti[-1][1] + banco.motore.blocco)
    assert prima_tacca.finita and prima_tacca.finita_al == campione + len(tacca)
    assert all(voce.finita and voce.finita_al == inizio for _buffer, inizio, voce in davanti[1:])
    assert partita.tasti_giu() == [] and partita.fase == "mira"


def test_la_precisione_globale_nelle_statistiche(banco, cartella_di_prova):
    """D26: ogni tiro con un bersaglio vicino ha la sua precisione globale, dal terzo collaudo 1 meno la distanza più piccola dal
    centro del bersaglio divisa per la distanza fra quel centro e il punto della cupola più lontano da lui (D28): il centro sta
    sopra 1 meno raggio più RAGGIO_PROIETTILE diviso quella distanza. La partita la dà alle statistiche, tiro per tiro, e alla
    fine la sua media, che va fra le ultime partite."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    bersaglio = partita.bersagli[0]
    # Il bersaglio del primo livello è fermo: il punto più lontano si misura da dove sta.
    lontananza = distanza_dal_punto_piu_lontano(bersaglio.partenza)
    colpisci(banco, partita)
    centro = partita.ultimo_volo
    messaggi = fino_alla_risoluzione(banco, partita)
    assert centro.precisione_globale == pytest.approx(1.0 - centro.distanza_al_centro / lontananza)
    assert centro.precisione_globale > 1.0 - (bersaglio.raggio + RAGGIO_PROIETTILE) / lontananza
    assert messaggi[0].dati["precisione_globale"] == centro.precisione_globale
    tiro_a_vuoto(banco, partita)
    vuoto = partita.ultimo_volo
    # Il tiro a vuoto ha per vicino il bersaglio nato al posto del colpito, e la sua lontananza.
    lontananza = distanza_dal_punto_piu_lontano(vuoto.vicino.partenza)
    assert vuoto.precisione_globale == pytest.approx(max(0.0, 1.0 - (vuoto.distanza_dal_vicino + vuoto.vicino.raggio) / lontananza))
    assert vuoto.precisione_globale < centro.precisione_globale
    somme = statistiche.dati["somme"]
    assert somme["tiri_col_globale"] == 2 and somme["precisione_globale"] == pytest.approx(centro.precisione_globale + vuoto.precisione_globale, abs=1e-5)
    assert statistiche.dati["primati"]["precisione_globale"]["valore"] == pytest.approx(centro.precisione_globale)
    partita.abbandona()
    media = (centro.precisione_globale + vuoto.precisione_globale) / 2
    assert statistiche.dati["ultime"][-1]["precisione_globale"] == round(media, 4)
    # Una partita di due tiri entra nella media per partita, ma non prende il primato, che vuole almeno TIRI_PER_IL_PRIMATO_DELLA_MEDIA tiri.
    assert somme["partite_col_globale"] == 1 and somme["globale_delle_partite"] == pytest.approx(media, abs=1e-4)
    assert "precisione_globale_media" not in statistiche.dati["primati"]


def test_la_media_della_partita_prende_il_primato_dai_dieci_tiri(banco, cartella_di_prova):
    """D26, dalla 0.11.8, e D30: il primato precisione_globale_media lo prende soltanto una partita con almeno
    TIRI_PER_IL_PRIMATO_DELLA_MEDIA tiri, 10, che con la riserva di D30 non sono più i colpi iniziali: una partita di nove
    tiri non lo prende, una di dieci sì, anche abbandonata."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    assert TIRI_PER_IL_PRIMATO_DELLA_MEDIA < COLPI_INIZIALI
    for tiri in (TIRI_PER_IL_PRIMATO_DELLA_MEDIA - 1, TIRI_PER_IL_PRIMATO_DELLA_MEDIA):
        partita.nuova()
        globali = []
        for _tiro in range(tiri):
            tiro_a_vuoto(banco, partita)
            globali.append(partita.ultimo_volo.precisione_globale)
        partita.abbandona()
        assert statistiche.dati["ultime"][-1]["tiri"] == tiri
        if tiri < TIRI_PER_IL_PRIMATO_DELLA_MEDIA:
            assert "precisione_globale_media" not in statistiche.dati["primati"]
    assert statistiche.dati["primati"]["precisione_globale_media"]["valore"] == round(sum(globali) / TIRI_PER_IL_PRIMATO_DELLA_MEDIA, 4)


def test_esc_annulla_la_mira_e_conserva_il_proiettile(banco, cartella_di_prova, bonus_sicuro, monkeypatch):
    """D25 e D29: Esc nella mira la annulla: i valori tacciono e si perdono, il tempo della mira si ferma coi suoi tick, la
    riserva non cambia, e si torna all'ascolto col tempo di gioco che scorre come nella mira e i bersagli pieni; suona
    mira_annullata, il messaggio lo dice, e si conta. Non è una fase d'ascolto nuova: non nasce un bonus. Lo spazio seguente
    avvia una mira nuova, dal centro. Nella mira in pausa la pausa resta. Fuori dalla mira Esc non fa niente."""
    suonate = spia_del_motore(monkeypatch, banco.motore)
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    senti_anche_le_statistiche(banco, statistiche)
    assert partita.annulla_mira() == [] and partita.fase == "pronta"
    partita.nuova()
    # Il livello della mira breve viene dopo quello dei bonus: un bonus nascerebbe a ogni fase d'ascolto nuova.
    partita.livello = LIVELLO_DELLA_MIRA_BREVE
    banco.flusso.avanza(10)
    voce = partita._voci[partita.bersagli[0]]
    assert partita.annulla_mira() == [] and partita.fase == "ascolto"
    partita.spazio()
    tick = [voce_del_tick for _b, _c, voce_del_tick in suonate[-(round(TEMPO_DELLA_MIRA_MINIMO) - 1):]]
    inizio = banco.motore.campione_udibile()
    partita.muovi("direzione", 1)
    porta_udibile_a(banco, inizio + 0.3 * FS)
    asse = partita._voce_del_puntatore
    tempo = banco.motore.tempo_di_gioco()
    sentiti = len(banco.contati)
    messaggi = partita.annulla_mira()
    assert [(m.tipo, m.annuncio) for m in messaggi] == [("info", "Mira annullata.")]
    assert messaggi[0].testo == f"Mira annullata con Esc: il proiettile resta nella riserva, che ha {COLPI_INIZIALI} colpi. Spazio per mirare di nuovo."
    assert partita.fase == "ascolto" and partita.valori == [] and partita.mira is None and partita.tasti_giu() == []
    assert partita.secondi_della_mira() is None and partita._scadenza is None and partita._tick_in_fila == []
    assert (partita.riserva, partita.tiri) == (COLPI_INIZIALI, 0) and partita.bonus is None
    assert banco.contati[sentiti:] == ["mira_annullata"] and statistiche.dati["conti"]["mire_annullate"] == 1
    # Il valore e i tick tacciono: il primo tick, che doveva suonare un secondo dopo lo spazio, finisce senza suonare.
    banco.flusso.avanza_fino(tick[0].inizio + banco.motore.blocco)
    assert asse.finita and tick[0].finita and tick[0].finita_al == tick[0].inizio
    assert voce.guadagno == 1.0 and banco.motore.tempo_scorre and banco.motore.tempo_di_gioco() > tempo
    assert partita.battito() == [] and partita.bonus is None and partita.righe_della_barra()[-1] == "Ascolto: Spazio per mirare"
    # Lo spazio avvia una mira nuova, dal centro: qui nella palestra, dove Ctrl+P mette la pausa; Esc nella mira in pausa la
    # annulla, e la pausa resta.
    partita.livello = 1
    partita.spazio()
    assert partita.fase == "mira" and partita.mira.valori(banco.motore.campione_udibile() / FS) == [0.5] * 3
    partita.pausa()
    messaggi = partita.annulla_mira()
    assert partita.fase == "ascolto" and partita.in_pausa and "la partita resta in pausa" in messaggi[0].testo
    banco.flusso.avanza(10)
    assert voce.guadagno == 0.0 and not banco.motore.tempo_scorre
    partita.pausa()
    banco.flusso.avanza(10)
    assert not partita.in_pausa and voce.guadagno == 1.0 and banco.motore.tempo_scorre
    assert statistiche.dati["conti"]["mire_annullate"] == 2 and statistiche.dati["conti"]["tiri"] == 0


def test_tiri_a_vuoto_fino_alla_fine(banco):
    orologio = Orologio(100.0)
    partita = banco.partita(orologio=orologio)
    partita.nuova()
    voci = list(partita._voci.values())
    globali = []
    for numero in range(1, COLPI_INIZIALI + 1):
        if numero == COLPI_INIZIALI:
            orologio.valore = 160.0
        messaggi = tiro_a_vuoto(banco, partita)
        assert partita.tiri == numero and partita.riserva == COLPI_INIZIALI - numero
        assert messaggi[0].tipo == "resoconto" and not messaggi[0].dati["colpito"]
        globali.append(messaggi[0].dati["precisione_globale"])
        # D21: il tiro tutto a sinistra, basso e piano, entra appena nella cupola, a sinistra del bersaglio e sotto di lui, e
        # scendendo resta sempre sotto il suo centro: il sommario lo dice coi numeri, e la legenda dice perché manca la profondità (D24).
        sommario = resoconti.sommario_del_tiro(partita.ultimo_volo, "numeri")
        assert sommario.primo.startswith("L:-") and " V:-" in sommario.primo and " P:manca " in sommario.primo
        assert "P manca: scendendo, il tiro resta sempre sotto il centro del bersaglio." in sommario.secondo
        assert messaggi[0].testo.split("\n")[0] == sommario.riga
        # D22: al livello 1, della palestra, ogni tiro a vuoto finisce con l'attesa dello spazio, che riporta all'ascolto.
        if partita.riserva > 1:
            assert [m.tipo for m in messaggi] == ["resoconto", "info"] and partita.fase == "attesa" and partita.nasceranno == 0
            assert messaggi[0].annuncio == f"A vuoto, riserva {partita.riserva}. {sommario.annuncio}"
            assert messaggi[-1].testo == "Palestra: la partita aspetta lo Spazio, che ti riporta all'ascolto."
        elif partita.riserva == 1:
            assert [m.tipo for m in messaggi] == ["resoconto", "avviso", "info"]
            assert messaggi[1].annuncio == "Ultimo colpo."
    # La riserva vuota chiude la partita come prima, senza attesa.
    assert partita.fase == "finita" and not partita.in_corso()
    assert [m.tipo for m in messaggi] == ["resoconto", "fine"]
    assert messaggi[0].annuncio == f"A vuoto, riserva vuota. {sommario.annuncio}"
    # Il riepilogo porta la precisione globale media dei tiri (D26), che va in classifica (D27).
    riepilogo = {"punteggio": 0, "livello": 1, "tiri": COLPI_INIZIALI, "centri": 0, "durata_s": 60.0,
                 "precisione_globale": round(sum(globali) / len(globali), 4)}
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
    # D22: nella palestra i bersagli del livello nuovo nascono allo spazio che chiude l'attesa.
    assert partita.fase == "attesa" and partita.bersagli == [] and partita.nasceranno == quanti_bersagli(2) and messaggi[-1].dati["nasceranno"] == quanti_bersagli(2)
    fuori_dall_attesa(banco, partita)
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
    tiro_a_vuoto(banco, partita)
    # Il livello dei bonus è ancora della palestra (D22): la nuova fase d'ascolto, e il bonus, li porta lo spazio.
    assert partita.fase == "attesa" and partita.bonus is None
    messaggi = fuori_dall_attesa(banco, partita)
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
    fuori_dall_attesa(banco, partita)
    bonus = partita.bonus
    evento = partita._evento_bonus
    assert bonus is not None
    # A metà della traversata il bonus sta in mezzo al settore, dove ogni punto si raggiunge; la mira lo segue mentre si muove (D29).
    banco.flusso.avanza_secondi(durata_del_bonus(LIVELLO_DEI_BONUS) / 2)
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    riserva = partita.riserva
    colpisci(banco, partita, bonus)
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
    assert messaggi[0].annuncio == f"Colpito il bonus, {testi.intero(attesi)} punti. {resoconti.sommario_del_tiro(volo, resoconti.grado_della_palestra(LIVELLO_DEI_BONUS)).annuncio}"
    assert messaggi[0].dati["bonus"] and messaggi[0].dati["riserva"] == riserva + COLPI_BONUS
    assert "colpito_bonus" in banco.contati and "bonus_uscito" not in banco.contati
    # Dopo un centro sul bonus non nasce un bersaglio nuovo (D5): lo spazio della palestra riporta all'ascolto (D22).
    assert partita.fase == "attesa" and partita.nasceranno == 0
    assert messaggi[-1].testo == "Palestra: la partita aspetta lo Spazio, che ti riporta all'ascolto."
    fuori_dall_attesa(banco, partita)
    assert len(partita.bersagli) == quanti_bersagli(LIVELLO_DEI_BONUS)
    # L'uscita annullata non scatta mai, nemmeno quando il suo tempo passa.
    while banco.motore.tempo_di_gioco() <= evento.tempo + 0.5:
        banco.flusso.avanza(10)
    assert not evento.scattato


def righe_della_situazione(messaggio):
    """Le due righe della situazione di D33 che il resoconto deve avere, dai numeri dei suoi dati; senza
    la classifica, come nelle prove della partita, la seconda dice soltanto il tempo."""
    dati = messaggio.dati
    return [resoconti.riga_del_livello(dati["livello"], dati["tiri_del_livello"], dati["riserva"], dati["centri_del_livello"]),
            resoconti.riga_del_tempo_e_del_posto(dati["durata_s"], dati["totale"], None)]


def tiro_lungo(banco, partita):
    """Un tiro a vuoto vicino, coi valori del tiro giusto sul primo bersaglio normale, un poco più a
    destra e molto più potente: passa a destra, ricade oltre, e ha tutte le misure dell'artigliere
    (D21). Restituisce i messaggi della sua risoluzione."""
    pronta_a_mirare(banco, partita)
    partita.spazio()
    bersaglio = next(b for b in partita.bersagli if not b.bonus)
    invio, partenza = prepara_lo_sparo(banco)
    direzione, alzo, potenza = mira_sul_bersaglio(bersaglio, partenza)
    mira_e_spara(banco, partita, (direzione + 0.02, alzo, min(1.0, potenza + 0.15)), invio=invio)
    volo = partita.ultimo_volo
    assert volo.colpito is None and volo.vicino is bersaglio
    assert volo.misure.arrivo is not None and volo.misure.ricaduta is not None
    assert volo.misure.laterale > 0.0 and volo.misure.profondita > 0.0
    return fino_alla_risoluzione(banco, partita)


def test_il_sommario_in_testa_al_resoconto_ai_tre_gradi(banco, monkeypatch):
    """D21: il resoconto vero comincia col sommario delle misure dell'artigliere del volo, al grado
    della palestra del livello del tiro: i numeri nei primi livelli, poi le sole parole, poi niente.
    L'annuncio dice prima l'esito, senza lo scarto principale quando il sommario ha i numeri, poi il
    sommario; e la riga del mancato di D6 dice le stesse misure."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    partita = banco.partita()
    partita.nuova()
    numeri, parole = SOMMARIO_NUMERI_FINO_AL_LIVELLO, SOMMARIO_PAROLE_FINO_AL_LIVELLO
    for livello, grado in ((1, "numeri"), (numeri, "numeri"), (numeri + 1, "parole"), (parole, "parole"), (parole + 1, "niente")):
        partita.livello = livello
        messaggi = tiro_lungo(banco, partita)
        volo = partita.ultimo_volo
        righe = messaggi[0].testo.split("\n")
        esito = f"Tiro {partita.tiri}, livello {livello}: a vuoto; nella riserva restano {partita.riserva} colpi."
        situazione = righe_della_situazione(messaggi[0])
        assert situazione[0].endswith(f"; presi 0 bersagli, ne mancano {CENTRI_PER_LIVELLO} al livello {livello + 1}.")
        sommario = resoconti.riga_del_sommario(volo, grado)
        pezzi = resoconti.sommario_del_tiro(volo, grado)
        assert resoconti.grado_della_palestra(livello) == grado
        if livello <= AIUTO_MANCATO_FINO_AL_LIVELLO:
            mancato = resoconti.riga_del_mancato(volo)
            assert righe[-1] == mancato and resoconti.scarti(volo.misure.scarto)[0].startswith("a destra di ")
            assert mancato.endswith(f" e lungo di {resoconti.metri(volo.misure.profondita)}.")
        globale = testi.percento(volo.precisione_globale, 1)
        if grado == "niente":
            # Dove il sommario tace, la precisione globale di D26 sta nella riga dell'esito e nell'annuncio.
            assert sommario == "" and pezzi is None
            # D33: dove il sommario tace la situazione apre il resoconto.
            assert righe[:2] == situazione
            assert righe[2] == f"Tiro {partita.tiri}, livello {livello}: a vuoto, precisione globale {globale}; nella riserva restano {partita.riserva} colpi."
            assert messaggi[0].annuncio == resoconti.resoconto_del_tiro(partita.tiri, livello, volo, partita.valori, riserva=partita.riserva, palestra="niente").annuncio
            assert messaggi[0].annuncio.endswith(f" Precisione globale {globale}.")
            continue
        assert sommario == pezzi.riga
        assert messaggi[0].annuncio.startswith(f"A vuoto, riserva {partita.riserva}") and messaggi[0].annuncio.endswith(f". {pezzi.annuncio}")
        if grado == "numeri":
            # Una riga sola, coi valori e G nel primo blocco, senza spazio dopo i due punti (0.11.8).
            assert righe[:4] == [sommario, *situazione, esito] and not pezzi.a_capo
            assert pezzi.primo.startswith("L:+") and " P:+" in pezzi.primo and pezzi.primo.endswith(f" G:{globale}") and pezzi.annuncio == pezzi.primo
            assert len(pezzi.primo) < BLOCCO_DEL_BRAILLE and sommario[BLOCCO_DEL_BRAILLE:].startswith("L laterale, ")
            assert messaggi[0].annuncio == f"A vuoto, riserva {partita.riserva}. {pezzi.annuncio}"
        else:
            # Le parole su una riga, la precisione globale e la legenda sempre sotto, su una riga loro (0.11.8).
            parole = ", ".join(resoconti.parole_delle_misure(volo.misure))
            assert pezzi.a_capo and righe[:5] == [pezzi.primo, pezzi.secondo, *situazione, esito]
            assert pezzi.primo == f"{parole[0].upper()}{parole[1:]}." and pezzi.secondo.startswith(f"Precisione globale {globale}. Dal bersaglio: ")


def test_la_situazione_dopo_ogni_tiro(banco, monkeypatch):
    """D33: dopo ogni tiro il resoconto dice la situazione, dopo il sommario: i colpi sparati nel livello, che ripartono da zero
    col livello nuovo, la riserva, i centri del livello, il tempo vero di gioco e, con la classifica che la finestra mette nella
    partita, il posto e il distacco, anche nell'annuncio. La prova chiude il livello a mano, coi centri a uno dalla fine."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    orologio = Orologio(0.0)
    partita = banco.partita(orologio=orologio)
    voci = [{"nome": "Gabry", "punteggio": 1_000_000}, {"nome": "Bea", "punteggio": 5}]
    letture = []

    def classifica():
        letture.append(len(voci))
        return voci

    partita.classifica = classifica
    partita.nuova()
    orologio.valore = 205.0
    resoconto = tiro_a_vuoto(banco, partita)[0]
    righe = resoconto.testo.split("\n")
    assert righe[1:3] == [f"Sparato 1 colpo nel livello, {COLPI_INIZIALI - 1} in riserva; presi 0 bersagli, ne mancano {CENTRI_PER_LIVELLO} al livello 2.",
                          "Tempo 03:25; fuori classifica, dietro Bea, 5 punti, distacco 5."]
    assert righe[3].startswith("Tiro 1, livello 1: a vuoto") and resoconto.annuncio.startswith(f"A vuoto, riserva {COLPI_INIZIALI - 1}. Fuori classifica, a 5. L:")
    assert letture == [2] and partita.tiri_del_livello == 1
    fuori_dall_attesa(banco, partita)
    partita.centri_del_livello = CENTRI_PER_LIVELLO - 1
    colpisci(banco, partita)
    messaggi = fino_alla_risoluzione(banco, partita)
    resoconto = messaggi[0]
    punti = partita.punti
    righe = resoconto.testo.split("\n")
    assert [m.tipo for m in messaggi][:2] == ["resoconto", "livello"] and partita.livello == 2
    assert righe[1:3] == [f"Sparati 2 colpi nel livello, {COLPI_INIZIALI - 1} in riserva; presi {CENTRI_PER_LIVELLO} bersagli, livello superato.",
                          f"Tempo 03:25; in classifica 2°, dietro Gabry, 1.000.000 punti, distacco {testi.intero(1_000_000 - punti)}."]
    assert resoconto.annuncio.startswith(f"Colpito, {testi.intero(punti)} punti. 2°, a {testi.intero(1_000_000 - punti)}. L:")
    assert resoconto.dati["posizione"] == 2 and resoconto.dati["tiri_del_livello"] == 2
    # Il livello nuovo riparte da zero, e così la partita nuova.
    assert partita.tiri_del_livello == 0 and partita.centri_del_livello == 0 and partita.tiri == 2
    partita.nuova()
    assert partita.tiri_del_livello == 0


def test_il_sommario_del_bonus_segue_il_livello_del_tiro(banco, bonus_sicuro, monkeypatch):
    """D21: per il bonus il grado del sommario è quello del livello in cui si spara, mentre i punti
    restano quelli del livello in cui il bonus è nato (D13). La prova alza il livello a bonus in volo,
    a mano, per separare i due livelli."""
    partita = banco.partita()
    partita.nuova()
    partita.livello = LIVELLO_DEI_BONUS
    tiro_a_vuoto(banco, partita)
    fuori_dall_attesa(banco, partita)
    bonus = partita.bonus
    assert bonus is not None
    banco.flusso.avanza_secondi(durata_del_bonus(LIVELLO_DEI_BONUS) / 2)
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    nato = resoconti.grado_della_palestra(LIVELLO_DEI_BONUS)
    livello_del_tiro = next((n for n in range(LIVELLO_DEI_BONUS + 1, LIVELLO_DEI_BONUS + 30) if resoconti.grado_della_palestra(n) != nato), LIVELLO_DEI_BONUS + 1)
    grado = resoconti.grado_della_palestra(livello_del_tiro)
    partita.livello = livello_del_tiro
    colpisci(banco, partita, bonus)
    volo = partita.ultimo_volo
    assert volo.colpito is bonus and volo.misure is not None
    messaggi = fino_alla_risoluzione(banco, partita)
    assert partita.punti == punti_del_centro(bonus, volo.precisione, livello=LIVELLO_DEI_BONUS)
    righe = messaggi[0].testo.split("\n")
    esito = f"Tiro {partita.tiri}, livello {livello_del_tiro}: colpito il bonus, "
    sommario = resoconti.sommario_del_tiro(volo, grado)
    if sommario is not None:
        # Il sommario a parole ha due righe, quello coi numeri una (0.11.8).
        quante = sommario.riga.count("\n") + 1
        assert quante == (2 if grado == "parole" else 1)
        assert "\n".join(righe[:quante]) == sommario.riga and righe[quante:quante + 2] == righe_della_situazione(messaggi[0])
        assert righe[quante + 2].startswith(esito)
        assert messaggi[0].annuncio.startswith("Colpito il bonus, ") and messaggi[0].annuncio.endswith(f" punti. {sommario.annuncio}")
        assert sommario.riga != resoconti.riga_del_sommario(volo, nato)
    else:
        assert righe[:2] == righe_della_situazione(messaggi[0])
        assert righe[2].startswith(esito) and messaggi[0].annuncio.startswith("Colpito il bonus, ") and " L:" not in messaggi[0].annuncio


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
    """Ctrl+P nella mira della palestra (D14, D29): i valori si fermano dove si sentivano, i loro tasti valgono come lasciati, i
    loro suoni e i bersagli tacciono, e il tempo di gioco si ferma; in pausa i tasti dei valori non fanno niente, e l'Invio e
    lo spazio fanno sentire il non ora. Alla ripresa i valori sono dove erano, fermi finché un tasto non li muove di nuovo, i
    bersagli tornano pieni e il tempo riprende; la durata vera della partita non conta la pausa."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    orologio = Orologio(0.0)
    partita = banco.partita(statistiche=statistiche, orologio=orologio)
    senti_anche_le_statistiche(banco, statistiche)
    partita.nuova()
    banco.flusso.avanza(10)
    voce = partita._voci[partita.bersagli[0]]
    partita.spazio()
    inizio = banco.motore.campione_udibile()
    partita.muovi("direzione", -1)
    partita.muovi("potenza", 1)
    porta_udibile_a(banco, inizio + 0.45 * FS)
    assi = [partita._voce_del_puntatore]
    orologio.valore = 5.0
    messaggi = partita.pausa()
    assert partita.in_pausa and partita.fase == "mira" and banco.contati[-1] == "pausa"
    assert [m.annuncio for m in messaggi] == ["Pausa."] and "i valori restano dove sono" in messaggi[0].testo
    assert partita.tasti_giu() == [] and partita._voce_del_puntatore is None and partita.secondi_della_mira() is None
    fermi = [0.5 - 0.45 * VELOCITA_DELLA_MIRA, 0.5, 0.5 + 0.45 * VELOCITA_DELLA_MIRA]
    assert partita.mira.valori(inizio / FS + 60.0) == pytest.approx(fermi)
    assert partita.righe_della_barra()[-1] == "In pausa: Ctrl+P per riprendere"
    # Il tempo si ferma al prossimo campione sicuro della pausa: da lì in poi il tempo di gioco che si sente non cambia.
    banco.flusso.avanza(5)
    assert all(asse.finita for asse in assi) and voce.guadagno == 0.0
    fermo = banco.motore.tempo_di_gioco()
    for _blocco in range(10 * FS // banco.motore.blocco):
        banco.flusso.avanza(1)
        assert partita.battito() == []
    assert partita.fase == "mira" and banco.motore.tempo_di_gioco() == fermo
    # In pausa i tasti dei valori non fanno niente; l'Invio e lo spazio fanno sentire il non ora.
    assert partita.muovi("alzo", 1) == [] and partita.tasti_giu() == [] and partita._voce_del_puntatore is None
    prima = len(banco.contati)
    assert partita.spara() == [] and partita.spazio() == [] and banco.contati[prima:] == ["non_ora", "non_ora"]
    assert partita.fase == "mira" and partita.tiri == 0
    orologio.valore = 65.0
    messaggi = partita.pausa()
    assert not partita.in_pausa and [m.annuncio for m in messaggi] == ["Si riprende."] and banco.contati[-1] == "ripresa"
    banco.flusso.avanza(10)
    assert voce.guadagno == 1.0 and banco.motore.tempo_scorre and banco.motore.tempo_di_gioco() > fermo
    assert partita.mira.valori(banco.motore.campione_udibile() / FS) == pytest.approx(fermi) and partita.tasti_giu() == []
    # Ripresa la mira, i tasti muovono di nuovo i valori, da dove erano.
    mira_e_spara(banco, partita, (0.2, 0.5, 0.7))
    # La durata vera della partita non conta il minuto di pausa; la pausa si conta fra quelle chieste.
    orologio.valore = 100.0
    assert partita.riepilogo()["durata_s"] == 40.0
    assert statistiche.dati["conti"]["pause"] == 1 and statistiche.dati["conti"]["pause_automatiche"] == 0


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
    # Ctrl+P durante la pausa automatica arriva soltanto da Pausa del menu Partita, col menu che ha messo la
    # pausa (D22): nella palestra la pausa diventa quella chiesta, col suo suono, e resta quando il menu si chiude.
    partita.pausa_automatica(True)
    messaggi = partita.pausa()
    assert partita.in_pausa and [m.annuncio for m in messaggi] == ["Pausa."] and banco.contati[-1] == "pausa"
    assert partita.pausa_automatica(False) == [] and partita.in_pausa
    messaggi = partita.pausa()
    assert not partita.in_pausa and [m.annuncio for m in messaggi] == ["Si riprende."] and banco.contati[-1] == "ripresa"
    assert (statistiche.dati["conti"]["pause"], statistiche.dati["conti"]["pause_automatiche"]) == (2, 2)
    # Nella mira (D29): il valore che si muove si ferma dove si sentiva, il suo tasto vale come lasciato, e al ritorno resta
    # dov'era, senza suoni della pausa.
    partita.spazio()
    inizio = banco.motore.campione_udibile()
    partita.muovi("alzo", 1)
    porta_udibile_a(banco, inizio + 0.3 * FS)
    asse = partita._voce_del_puntatore
    sentiti = len(banco.contati)
    partita.pausa_automatica(True)
    banco.flusso.avanza(3)
    assert partita.in_pausa and partita.tasti_giu() == [] and asse.finita and voce.guadagno == 0.0
    for _blocco in range(10 * FS // banco.motore.blocco):
        banco.flusso.avanza(1)
        assert partita.battito() == []
    assert partita.fase == "mira"
    assert partita.pausa_automatica(False) == []
    assert not partita.in_pausa and banco.contati[sentiti:] == []
    assert partita.mira.valori(banco.motore.campione_udibile() / FS) == pytest.approx([0.5, 0.5 + 0.3 * VELOCITA_DELLA_MIRA, 0.5])
    banco.flusso.avanza(10)
    assert voce.guadagno == 1.0 and banco.motore.tempo_scorre
    # Durante il volo, e poi la durata vera, senza le pause.
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
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
    # Una partita nuova chiesta durante un'altra la abbandona prima; anche durante la mira, che tace.
    partita.nuova()
    partita.spazio()
    partita.muovi("direzione", 1)
    asse = partita._voce_del_puntatore
    messaggi = partita.nuova()
    assert [m.annuncio for m in messaggi] == ["Partita abbandonata.", "Partita nuova: Spazio per mirare."]
    assert partita.fase == "ascolto" and partita.mira is None and partita.tasti_giu() == []
    banco.flusso.avanza(4)
    assert asse.finita
    assert statistiche.dati["conti"]["partite_abbandonate"] == 2 and statistiche.dati["conti"]["mire_sparate"] == 1


def test_abbandono_dopo_un_impatto_gia_partito(banco, cartella_di_prova):
    """Un centro, poi un tiro il cui impatto è partito nel filo dell'audio, e l'abbandono prima che si
    risolva: l'effetto si è sentito e si conta, il tiro no, né nel riepilogo né nelle statistiche."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    colpisci(banco, partita)
    fino_alla_risoluzione(banco, partita)
    fuori_dall_attesa(banco, partita)
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
        pronta_a_mirare(banco, partita)
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


def _bonus_quasi_uscito(banco, partita, anticipo):
    """Un bonus in volo, e il tempo di gioco che corre fino a tanti secondi prima della sua uscita."""
    partita.livello = LIVELLO_DEI_BONUS
    tiro_a_vuoto(banco, partita)
    fuori_dall_attesa(banco, partita)
    bonus = partita.bonus
    assert bonus is not None
    while banco.motore.tempo_di_gioco(banco.motore.prossimo_campione()) < bonus.tempo_di_uscita - anticipo:
        banco.flusso.avanza(1)
        assert partita.battito() == []
    return bonus


def test_il_bonus_uscito_durante_il_volo_che_chiude_il_livello(banco, bonus_sicuro, cartella_di_prova, monkeypatch):
    """Il bonus esce col proiettile in volo, e il centro chiude il livello: l'uscita si dice e si conta
    prima che il livello nuovo svuoti il cielo. Durante la mira il tempo scorre (D29): il bonus esce poco
    dopo lo sparo, che arriva ATTESA_DELLO_SPARO dopo lo spazio."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    partita.nuova()
    bonus = _bonus_quasi_uscito(banco, partita, ATTESA_DELLO_SPARO + 0.3)
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
    bonus = _bonus_quasi_uscito(banco, partita, ATTESA_DELLO_SPARO + 0.3)
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
    e il bonus aspettano la ripresa, in fila. Fuori dalla palestra di D22, dove Ctrl+P non mette la
    pausa, la pausa è quella della finestra, che c'è sempre; la palestra la toglie la costante a zero,
    così il bersaglio fermo del secondo livello si colpisce con la fisica."""
    monkeypatch.setattr(modulo_partita, "PALESTRA_FINO_AL_LIVELLO", 0)
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
    assert partita.pausa_automatica(True) == [] and partita.in_pausa
    messaggi = fino_alla_risoluzione(banco, partita)
    assert [m.tipo for m in messaggi] == ["resoconto", "livello"]
    assert partita.fase == "ascolto" and partita.in_pausa and partita.livello == LIVELLO_DEI_BONUS and partita.bonus is None
    assert banco.contati[-1] == "colpito"
    banco.flusso.avanza(3)
    assert partita.bersagli and all(partita._voci[b].guadagno == 0.0 for b in partita.bersagli)
    assert partita.righe_della_barra()[-1] == "In pausa: Ctrl+P per riprendere"
    k = banco.motore.prossimo_campione()
    messaggi = partita.pausa_automatica(False)
    assert [m.tipo for m in messaggi] == ["bonus"] and partita.bonus is not None
    assert banco.contati[-2:] == ["livello_su", "bonus_appare"]
    stacco = round(modulo_partita.STACCO_FRA_GLI_AVVISI * FS)
    livello, bonus = (v for _nome, v in voci[-2:])
    assert livello.inizio == max(k, evento.campione + LUNGHEZZA_FINTA + stacco) and bonus.inizio == livello.inizio + LUNGHEZZA_FINTA + stacco
    banco.flusso.avanza(10)
    assert all(partita._voci[b].guadagno == 1.0 for b in partita.bersagli)


def test_la_pausa_presa_nell_ultimo_livello_della_palestra(banco, cartella_di_prova, monkeypatch):
    """D22: Ctrl+P nell'ultimo livello della palestra, quando l'impatto del centro che lo chiude è già
    partito: il tiro si risolve a partita ferma e porta al livello dopo, fuori dalla palestra, ma la
    pausa presa prima non resta incastrata, e Ctrl+P la toglie, col livello nuovo in fila dopo la
    ripresa. Il tiro era della palestra, e la sua attesa c'è: allo spazio nascono i bersagli del
    livello nuovo. Da lì Ctrl+P è rifiutato, e il tiro seguente non ha attesa."""
    monkeypatch.setattr(modulo_partita, "PROBABILITA_BONUS", 0.0)
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    senti_anche_le_statistiche(banco, statistiche)
    partita.nuova()
    partita.livello = PALESTRA_FINO_AL_LIVELLO
    partita.centri_del_livello = CENTRI_PER_LIVELLO - 1
    colpisci(banco, partita)
    evento = partita._evento_impatto
    for _blocco in range(BLOCCHI_MASSIMI):
        banco.flusso.avanza(1)
        if evento.scattato:
            break
    assert evento.scattato and banco.motore.campione_udibile() < evento.campione
    assert [m.annuncio for m in partita.pausa()] == ["Pausa."] and partita.in_pausa
    messaggi = fino_alla_risoluzione(banco, partita)
    nuovo = PALESTRA_FINO_AL_LIVELLO + 1
    assert [m.tipo for m in messaggi] == ["resoconto", "livello", "info"] and messaggi[1].annuncio.startswith(f"Livello {nuovo}, ")
    assert partita.livello == nuovo and partita.fase == "attesa" and partita.in_pausa
    assert partita.bersagli == [] and partita.nasceranno == quanti_bersagli(nuovo)
    # Il livello è fuori dalla palestra, ma la pausa presa prima si toglie con Ctrl+P, e il livello nuovo suona dopo la ripresa.
    messaggi = partita.pausa()
    assert not partita.in_pausa and [m.annuncio for m in messaggi] == ["Si riprende."] and banco.contati[-2:] == ["ripresa", "livello_su"]
    # Adesso Ctrl+P è rifiutato, anche nell'attesa del tiro della palestra: suona il non ora, e la vista lo dice.
    messaggi = partita.pausa()
    assert not partita.in_pausa and banco.contati[-1] == "non_ora"
    assert [(m.tipo, m.testo, m.annuncio) for m in messaggi] == [
        ("avviso", f"La pausa con Ctrl+P c'è soltanto nei livelli della palestra, fino al livello {PALESTRA_FINO_AL_LIVELLO}.", "Pausa solo in palestra.")]
    fuori_dall_attesa(banco, partita)
    assert len(partita.bersagli) == quanti_bersagli(nuovo) and all(b.mobile == (nuovo >= LIVELLO_DEI_MOBILI) for b in partita.bersagli)
    assert all(b.raggio == raggio_del_livello(nuovo) for b in partita.bersagli)
    # Il tiro seguente è del livello nuovo, fuori dalla palestra: niente attesa, e subito l'ascolto.
    messaggi = tiro_a_vuoto(banco, partita)
    assert [m.tipo for m in messaggi] == ["resoconto"] and partita.fase == "ascolto"
    conti = statistiche.dati["conti"]
    assert (conti["pause"], conti["pause_rifiutate"], conti["attese_della_palestra"]) == (1, 1, 1)


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
    """La barra di stato in ogni fase, con le righe brevi per il display braille; nella mira (D29) una riga in più, sopra
    quella della fase, coi tre valori, la direzione e l'alzo in gradi e la potenza in percentuale, e i secondi che restano
    alla mira a tempo, o che la mira non ha tempo, nella palestra."""
    partita = banco.partita()
    viste = []

    def righe():
        viste.extend(partita.righe_della_barra())
        return partita.righe_della_barra()

    assert righe() == ["Nessuna partita: Spazio per cominciare"]
    partita.nuova()
    testa = ["Livello 1, 0 punti", f"Riserva {COLPI_INIZIALI}, centri 0 su {CENTRI_PER_LIVELLO}", "Bersagli in volo: 1"]
    assert righe() == [*testa, "Ascolto: Spazio per mirare"]
    partita.spazio()
    assert righe() == [*testa, "Direzione davanti, alzo 50,0 gradi, potenza 50%, senza tempo", "Mira: Invio spara, Esc annulla"]
    porta_i_valori(banco, partita, (1.0, 1.0, 0.0))
    assert righe()[-2:] == ["Direzione 40,0 gradi a destra, alzo 85,0 gradi, potenza 0%, senza tempo", "Mira: Invio spara, Esc annulla"]
    porta_i_valori(banco, partita, (0.0, 0.0, 1.0))
    assert righe()[-2] == "Direzione 40,0 gradi a sinistra, alzo 15,0 gradi, potenza 100%, senza tempo"
    # I valori qualunque passano dalla fisica ai gradi, e da lì alla riga di resoconti.
    porta_i_valori(banco, partita, (0.3, 0.2, 0.65))
    valori = partita.mira.valori(banco.motore.campione_udibile() / FS)
    azimut, alzo, _velocita = parametri_del_tiro(*valori)
    assert righe()[-2] == resoconti.riga_dei_valori_della_mira(azimut, alzo, valori[2], None) and righe()[-2].startswith("Direzione 16,0 gradi")
    # La barra legge i valori della mira una volta sola, così la riga dice la direzione, l'alzo e la potenza dello stesso istante.
    letture = []
    originale = partita.mira.valori
    partita.mira.valori = lambda tempo: letture.append(tempo) or originale(tempo)
    partita.righe_della_barra()
    del partita.mira.valori
    assert len(letture) == 1
    partita.pausa()
    assert righe()[-2:] == [resoconti.riga_dei_valori_della_mira(azimut, alzo, valori[2], None), "In pausa: Ctrl+P per riprendere"]
    partita.pausa()
    assert partita.spara() == []
    assert righe() == [*testa, "Volo"]
    partita.pausa()
    assert righe()[-1] == "In pausa: Ctrl+P per riprendere"
    partita.pausa()
    partita.riserva = 1
    fino_alla_risoluzione(banco, partita)
    assert righe() == ["Livello 1, 0 punti", f"Riserva 0, centri 0 su {CENTRI_PER_LIVELLO}", "Partita finita: Spazio per ricominciare"]
    # Fuori dalla palestra la mira è a tempo, e la riga dice i secondi che restano, contati per eccesso.
    partita.nuova()
    partita.livello = PALESTRA_FINO_AL_LIVELLO + 1
    partita.spazio()
    secondi = round(TEMPO_DELLA_MIRA_INIZIALE)
    assert righe()[-2:] == [f"Direzione davanti, alzo 50,0 gradi, potenza 50%, {secondi} secondi", "Mira: Invio spara, Esc annulla"]
    porta_udibile_a(banco, partita._mira_dal + 4.5 * FS)
    assert righe()[-2] == f"Direzione davanti, alzo 50,0 gradi, potenza 50%, {math.ceil(secondi - 4.5)} secondi"
    # Righe brevi, per il display braille; quella dei valori della mira è più lunga, e non è divisa in blocchi.
    assert all(len(riga) <= BLOCCO_DEL_BRAILLE for riga in viste if not riga.startswith("Direzione "))


def test_statistiche_aggiornate(banco, cartella_di_prova):
    percorso = cartella_di_prova / "statistiche.json"
    statistiche = Statistiche(percorso=str(percorso))
    partita = banco.partita(statistiche=statistiche)
    assert banco.suoni.conta == statistiche.suono
    partita.nuova()
    primo = partita.bersagli[0]
    colpisci(banco, partita)
    fino_alla_risoluzione(banco, partita)
    fuori_dall_attesa(banco, partita)
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
    # D29: due mire finite col colpo, nessuna scaduta; tre tasti ciascuna, perché nessuno dei valori dei due tiri è al centro.
    assert (conti["mire_sparate"], conti["mire_scadute"], conti["tasti_della_mira"], conti["mire_annullate"]) == (2, 0, 6, 0)
    assert statistiche.dati["somme"]["secondi_di_mira"] == pytest.approx(2 * ATTESA_DELLO_SPARO, abs=0.2)
    # D22: l'attesa della palestra dopo il centro; quella dopo il tiro a vuoto non comincia, perché la partita si abbandona prima.
    assert conti["attese_della_palestra"] == 1 and conti["pause_rifiutate"] == 0
    suonati = statistiche.dati["suoni"]
    assert (suonati["nuova_partita"], suonati["mira_avviata"], suonati["sparo"], suonati["colpito"], suonati["a_terra"]) == (1, 2, 2, 1, 1)
    assert "valore_preso" not in suonati
    for parametro in PARAMETRI:
        presi = statistiche.dati["mira"][parametro]
        # Dalla 2.0.0 i valori della mira manuale hanno la chiave loro, e quelle delle scivolate restano ai conti di prima.
        assert presi["manuali"] == 2 and presi["ai_capi"] == 1 and presi["nei_centri"] == 1
        assert (presi["scelto"], presi["minimo"], presi["massimo_con_lo_spazio"], presi["massimo"]) == (0, 0, 0, 0)
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
    fuori_dall_attesa(banco, partita)
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
    fuori_dall_attesa(banco, partita)
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
    # Il volo d'esempio è un tiro verso il mezzo della cupola (D28), che dura più dell'esempio, e passa da lì.
    centro = np.array(MEZZO_DELLA_CUPOLA)
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


# Il resoconto dopo l'effetto e la palestra delle attese (D22).

def tonfo_lungo(monkeypatch, secondi=0.8):
    """Il tonfo del tiro a vuoto lungo come quello vero, gli altri preset brevi: così l'effetto dura più di
    un blocco del motore, e si vede quando finisce. Restituisce il buffer del tonfo."""
    corto = np.full((LUNGHEZZA_FINTA, 2), AMPIEZZA_FINTA, dtype=np.float32)
    lungo = np.full((round(secondi * FS), 2), AMPIEZZA_FINTA, dtype=np.float32)
    monkeypatch.setattr(suoni, "_rendi_preset", lambda nome: lungo if nome == suoni.EVENTI["a_terra"] else corto)
    return lungo


def fino_all_impatto(banco, partita):
    """Fa avanzare il motore col battito finché il tiro si risolve, quando l'impatto si sente: i messaggi
    non sono ancora arrivati. Restituisce il campione dove comincia l'effetto."""
    evento = partita._evento_impatto
    for _blocco in range(BLOCCHI_MASSIMI):
        banco.flusso.avanza(1)
        assert partita.battito() == []
        if partita.fase != "volo":
            assert evento.scattato and banco.motore.campione_udibile() >= evento.campione
            return evento.campione
    raise AssertionError("il tiro non si è risolto")


def test_nella_palestra():
    assert modulo_partita.nella_palestra(1) and modulo_partita.nella_palestra(PALESTRA_FINO_AL_LIVELLO)
    assert not modulo_partita.nella_palestra(PALESTRA_FINO_AL_LIVELLO + 1)
    assert "attesa" in modulo_partita.FASI and "attesa" in modulo_partita.IN_CORSO
    assert "mira" in modulo_partita.FASI and "mira" in modulo_partita.IN_CORSO


@pytest.mark.parametrize("palestra", [True, False])
def test_il_resoconto_arriva_quando_l_effetto_ha_finito_di_suonare(banco, monkeypatch, palestra):
    """D22: il tiro si risolve quando l'impatto si sente, ma il resoconto, coi messaggi che lo seguono,
    arriva quando l'effetto ha finito di suonare, nella palestra e fuori; RITARDO_DEL_RESOCONTO lo
    sposta ancora più in là, anche col suo valore vero, il secondo di D24. Gli avvisi in fila restano
    dove erano, dopo l'effetto."""
    lungo = tonfo_lungo(monkeypatch)
    if not palestra:
        monkeypatch.setattr(modulo_partita, "PALESTRA_FINO_AL_LIVELLO", 0)
    stacco = round(modulo_partita.STACCO_FRA_GLI_AVVISI * FS)
    partita = banco.partita()
    partita.nuova()
    for ritardo in (0.0, 0.25, RITARDO_DEL_RESOCONTO):
        monkeypatch.setattr(modulo_partita, "RITARDO_DEL_RESOCONTO", ritardo)
        pronta_a_mirare(banco, partita)
        partita.spazio()
        mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
        tiri = partita.tiri
        inizio = fino_all_impatto(banco, partita)
        # Il tiro è già risolto: la riserva è scesa, e la fase è quella dopo il tiro.
        assert partita.riserva == COLPI_INIZIALI - tiri and partita.fase == ("attesa" if palestra else "ascolto")
        assert partita._fine_degli_avvisi == inizio + len(lungo) + stacco
        fine = inizio + len(lungo) + round(ritardo * FS)
        porta_udibile_a(banco, fine - 1)
        assert partita.battito() == []
        porta_udibile_a(banco, fine)
        messaggi = partita.battito()
        assert [m.tipo for m in messaggi] == (["resoconto", "info"] if palestra else ["resoconto"]) and messaggi[0].dati["tiro"] == tiri
        assert partita.battito() == []


def test_la_mira_avviata_durante_l_effetto_riceve_il_resoconto(banco, monkeypatch):
    """D22 e D29, fuori dalla palestra: lo spazio dato mentre il tonfo suona ancora avvia la mira, e il resoconto arriva nella
    mira, quando il tonfo ha finito di suonare, più RITARDO_DEL_RESOCONTO. Il tempo della mira, dalle prove del 22 novembre 2026,
    non parte subito: aspetta la fine del tonfo, con lo stacco degli avvisi, e il resoconto che aspetta la fine dell'effetto,
    così i tick non coprono la sua voce, e fino a lì i secondi restano tutti; poi la mira va avanti, coi suoi secondi che
    scorrono."""
    monkeypatch.setattr(modulo_partita, "PALESTRA_FINO_AL_LIVELLO", 0)
    lungo = tonfo_lungo(monkeypatch)
    stacco = round(modulo_partita.STACCO_FRA_GLI_AVVISI * FS)
    partita = banco.partita()
    partita.nuova()
    totale = tempo_della_mira(1)
    assert totale == TEMPO_DELLA_MIRA_INIZIALE
    # Senza ritardo il tempo parte alla fine dello stacco dopo il tonfo; col ritardo più lungo dello stacco, al resoconto.
    for ritardo in (0.0, 0.25):
        assert (ritardo * FS > stacco) == (ritardo > 0.0)
        monkeypatch.setattr(modulo_partita, "RITARDO_DEL_RESOCONTO", ritardo)
        partita.spazio()
        mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
        inizio = fino_all_impatto(banco, partita)
        arrivo = inizio + len(lungo) + round(ritardo * FS)
        k = banco.motore.prossimo_campione()
        prima = len(banco.contati)
        assert partita.fase == "ascolto" and partita.spazio() == [] and partita.fase == "mira" and banco.contati[prima:] == ["mira_avviata"]
        partenza = max(k, inizio + len(lungo) + stacco, arrivo)
        assert partenza > k and partita._fine_degli_avvisi == inizio + len(lungo) + stacco
        assert partita._mira_dal == partenza and partita._scadenza == partenza + round(totale * FS)
        assert partita.secondi_della_mira() == pytest.approx(totale)
        porta_udibile_a(banco, arrivo - 1)
        assert partita.battito() == [] and partita.secondi_della_mira() == pytest.approx(totale)
        porta_udibile_a(banco, arrivo)
        assert [m.tipo for m in partita.battito()] == ["resoconto"] and partita.fase == "mira"
        assert partita.secondi_della_mira() == pytest.approx(totale)
        porta_udibile_a(banco, partenza + 0.5 * FS)
        assert partita.battito() == [] and partita.secondi_della_mira() == pytest.approx(totale - 0.5)
        mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
        fino_alla_risoluzione(banco, partita)
        assert partita.fase == "ascolto"


def test_il_bonus_nato_alla_ripresa_si_dice_dopo_il_resoconto(banco, bonus_sicuro, monkeypatch):
    """D22: fuori dalla palestra, con l'impatto risolto a partita ferma e il tonfo che suona ancora, la
    ripresa fa nascere il bonus, ma il suo messaggio aspetta in fila dopo il resoconto, che arriva alla
    fine del tonfo."""
    monkeypatch.setattr(modulo_partita, "PALESTRA_FINO_AL_LIVELLO", 0)
    lungo = tonfo_lungo(monkeypatch)
    partita = banco.partita()
    partita.nuova()
    partita.livello = LIVELLO_DEI_BONUS
    partita.spazio()
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    evento = partita._evento_impatto
    for _blocco in range(BLOCCHI_MASSIMI):
        banco.flusso.avanza(1)
        if evento.scattato:
            break
    assert evento.scattato and banco.motore.campione_udibile() < evento.campione
    partita.pausa_automatica(True)
    inizio = fino_all_impatto(banco, partita)
    assert partita.in_pausa and partita.fase == "ascolto" and partita.bonus is None
    assert partita.pausa_automatica(False) == [] and partita.bonus is not None
    porta_udibile_a(banco, inizio + len(lungo))
    assert [m.tipo for m in partita.battito()] == ["resoconto", "bonus"]


def test_l_attesa_della_palestra_dopo_un_centro(banco, cartella_di_prova):
    """D22: dopo un centro nella palestra la partita entra nell'attesa col tempo di gioco fermo e il
    cielo vuoto; finiti l'effetto e gli avvisi, stacco compreso, si ferma, e la barra dice che cosa fa
    lo spazio. Lo spazio prima, e nella grazia, non conta e non suona; dopo chiama il bersaglio nuovo,
    che nasce in quel momento, e il tempo riprende. La durata della partita non conta l'attesa."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    orologio = Orologio(0.0)
    partita = banco.partita(statistiche=statistiche, orologio=orologio)
    senti_anche_le_statistiche(banco, statistiche)
    partita.nuova()
    colpisci(banco, partita)
    fino_alla_risoluzione(banco, partita)
    assert partita.fase == "attesa" and partita.in_corso() and not partita.attesa_ferma and not banco.motore.tempo_scorre
    assert partita.bersagli == [] and partita.nasceranno == 1
    assert partita.righe_della_barra()[-1] == "Palestra: ascolta l'esito del tiro"
    # Lo spazio mentre gli avvisi e l'effetto suonano ancora non conta e non suona.
    prima = list(banco.contati)
    assert partita.spazio() == [] and banco.contati == prima and partita.fase == "attesa"
    fine = partita._fine_degli_avvisi
    porta_udibile_a(banco, fine - 1)
    orologio.valore = 10.0
    assert partita.battito() == [] and not partita.attesa_ferma
    porta_udibile_a(banco, fine)
    assert partita.battito() == [] and partita.attesa_ferma
    assert partita.righe_della_barra()[-1] == "Palestra: Spazio per il bersaglio nuovo"
    assert statistiche.dati["conti"]["attese_della_palestra"] == 1
    fermo = banco.motore.tempo_di_gioco()
    # La grazia: lo spazio dato appena la partita si è fermata non la fa ripartire. Si conta dal campione udibile del fermo,
    # quello della fine degli avvisi a meno degli arrotondamenti dell'orologio finto.
    assert partita._attesa_dal == pytest.approx(fine, abs=1e-6)
    grazia = round(GRAZIA_DELL_ATTESA * FS)
    porta_udibile_a(banco, partita._attesa_dal + grazia - 1)
    assert partita.spazio() == [] and partita.fase == "attesa" and banco.contati == prima
    porta_udibile_a(banco, partita._attesa_dal + grazia)
    assert banco.motore.tempo_di_gioco() == fermo and partita.bersagli == []
    orologio.valore = 40.0
    k = banco.motore.prossimo_campione()
    assert partita.spazio() == []
    assert partita.fase == "ascolto" and not partita.attesa_ferma and partita.nasceranno == 0 and banco.motore.tempo_scorre
    assert len(partita.bersagli) == quanti_bersagli(1)
    nuovo = partita.bersagli[0]
    assert partita._voci[nuovo].inizio == k and nuovo.nato == pytest.approx(fermo) == banco.motore.tempo_di_gioco(k)
    assert partita.righe_della_barra()[-1] == "Ascolto: Spazio per mirare"
    # Dieci secondi di gioco prima dell'attesa; i trenta dell'attesa non contano.
    assert partita.riepilogo()["durata_s"] == 10.0
    banco.flusso.avanza(10)
    assert banco.motore.tempo_di_gioco() > fermo
    # Lo spazio seguente avvia la mira, come sempre.
    partita.spazio()
    assert partita.fase == "mira" and banco.contati[-1] == "mira_avviata"


def test_l_attesa_della_palestra_dopo_un_tiro_a_vuoto(banco):
    """D22: dopo un tiro a vuoto nella palestra il bersaglio mancato si sente ancora, fermo, finché
    suonano l'effetto e gli avvisi; poi tutto tace, e la partita aspetta lo spazio, che riporta
    all'ascolto, col bersaglio che torna a suonare; lo spazio seguente avvia la mira."""
    partita = banco.partita()
    partita.nuova()
    bersaglio = partita.bersagli[0]
    voce = partita._voci[bersaglio]
    messaggi = tiro_a_vuoto(banco, partita)
    assert partita.fase == "attesa" and partita.nasceranno == 0 and partita.bersagli == [bersaglio]
    assert messaggi[-1].testo == "Palestra: la partita aspetta lo Spazio, che ti riporta all'ascolto." and messaggi[-1].dati == {"nasceranno": 0}
    banco.flusso.avanza(2)
    assert voce.guadagno == 1.0 and not banco.motore.tempo_scorre
    fermo = banco.motore.tempo_di_gioco(banco.motore.prossimo_campione())
    fino_al_fermo(banco, partita)
    banco.flusso.avanza(10)
    assert voce.guadagno == 0.0 and banco.motore.tempo_di_gioco() == fermo
    assert partita.righe_della_barra()[-2:] == ["Bersagli in volo: 1", "Palestra: Spazio per tornare all'ascolto"]
    banco.flusso.avanza_secondi(GRAZIA_DELL_ATTESA)
    prima = list(banco.contati)
    assert partita.spazio() == [] and partita.fase == "ascolto" and partita.bersagli == [bersaglio] and banco.contati == prima
    banco.flusso.avanza(10)
    assert voce.guadagno == 1.0 and banco.motore.tempo_scorre and banco.motore.tempo_di_gioco() > fermo
    partita.spazio()
    assert partita.fase == "mira" and banco.contati[-1] == "mira_avviata"


def test_la_pausa_automatica_durante_l_attesa(banco, cartella_di_prova):
    """D22: la pausa automatica c'è anche nell'attesa della palestra. Prima che la partita si fermi
    zittisce il bersaglio e la tiene in sospeso, anche a effetto e avvisi finiti; alla ripresa il
    bersaglio torna, e al battito seguente la partita si ferma. A partita ferma la pausa non cambia
    niente di ciò che si sente, e alla ripresa la partita resta ferma ad aspettare lo spazio. La
    durata non conta né le pause né l'attesa, e una pausa dentro l'attesa non si conta due volte."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    orologio = Orologio(0.0)
    partita = banco.partita(statistiche=statistiche, orologio=orologio)
    senti_anche_le_statistiche(banco, statistiche)
    partita.nuova()
    voce = partita._voci[partita.bersagli[0]]
    tiro_a_vuoto(banco, partita)
    assert partita.fase == "attesa" and not partita.attesa_ferma
    orologio.valore = 5.0
    assert partita.pausa_automatica(True) == [] and partita.in_pausa
    banco.flusso.avanza(10)
    assert voce.guadagno == 0.0
    porta_udibile_a(banco, partita._fine_degli_avvisi + FS)
    assert partita.battito() == [] and not partita.attesa_ferma
    assert partita.spazio() == [] and banco.contati[-1] == "non_ora" and partita.fase == "attesa"
    assert partita.righe_della_barra()[-1] == "In pausa: Ctrl+P per riprendere"
    orologio.valore = 15.0
    assert partita.pausa_automatica(False) == [] and not partita.in_pausa and not partita.attesa_ferma
    banco.flusso.avanza(10)
    assert voce.guadagno == 1.0 and not banco.motore.tempo_scorre
    orologio.valore = 20.0
    assert partita.battito() == [] and partita.attesa_ferma
    banco.flusso.avanza(10)
    assert voce.guadagno == 0.0
    # A partita ferma: la pausa automatica, e lo spazio che fa sentire il non ora.
    orologio.valore = 30.0
    partita.pausa_automatica(True)
    banco.flusso.avanza_secondi(GRAZIA_DELL_ATTESA)
    assert partita.spazio() == [] and banco.contati[-1] == "non_ora" and partita.fase == "attesa"
    orologio.valore = 50.0
    assert partita.pausa_automatica(False) == [] and not partita.in_pausa and partita.attesa_ferma and partita.fase == "attesa"
    banco.flusso.avanza(10)
    assert voce.guadagno == 0.0 and not banco.motore.tempo_scorre
    assert partita.righe_della_barra()[-1] == "Palestra: Spazio per tornare all'ascolto"
    orologio.valore = 60.0
    assert partita.spazio() == [] and partita.fase == "ascolto"
    # Fermi dal 5 al 15 per la pausa, dal 20 al 60 per l'attesa, pausa compresa: in gioco dieci secondi.
    assert partita.riepilogo()["durata_s"] == 10.0
    conti = statistiche.dati["conti"]
    assert (conti["pause_automatiche"], conti["attese_della_palestra"], conti["pause"]) == (2, 1, 0)
    assert "pausa" not in banco.contati and "ripresa" not in banco.contati


def test_ctrl_p_soltanto_nella_palestra(banco, cartella_di_prova):
    """D22: Ctrl+P mette la pausa nei livelli della palestra; dopo suona non_ora, la vista dice perché,
    e il rifiuto si conta, anche nella mira, che va avanti, col valore che si muove e il suo tempo che
    scorre (D29). La pausa automatica c'è in tutti i livelli, e Ctrl+P, che allora arriva dal menu, la
    lascia com'è: la toglie la finestra."""
    statistiche = Statistiche(percorso=str(cartella_di_prova / "statistiche.json"))
    partita = banco.partita(statistiche=statistiche)
    senti_anche_le_statistiche(banco, statistiche)
    partita.nuova()
    for livello in (1, PALESTRA_FINO_AL_LIVELLO):
        partita.livello = livello
        assert [m.annuncio for m in partita.pausa()] == ["Pausa."] and partita.in_pausa
        assert [m.annuncio for m in partita.pausa()] == ["Si riprende."] and not partita.in_pausa
    partita.livello = PALESTRA_FINO_AL_LIVELLO + 1
    messaggi = partita.pausa()
    rifiuto = f"La pausa con Ctrl+P c'è soltanto nei livelli della palestra, fino al livello {PALESTRA_FINO_AL_LIVELLO}."
    assert [(m.tipo, m.testo, m.annuncio) for m in messaggi] == [("avviso", rifiuto, "Pausa solo in palestra.")]
    assert not partita.in_pausa and banco.contati[-1] == "non_ora" and banco.motore.tempo_scorre
    # Nella mira il rifiuto non la tocca: il valore si muove ancora, e il tempo della mira scorre.
    partita.spazio()
    inizio = banco.motore.campione_udibile()
    partita.muovi("direzione", 1)
    asse = partita._voce_del_puntatore
    assert partita.pausa()[0].testo == rifiuto and not partita.in_pausa and partita.fase == "mira"
    assert partita.tasti_giu() == [("direzione", 1)] and partita._voce_del_puntatore is asse
    secondi = partita.secondi_della_mira()
    porta_udibile_a(banco, inizio + 0.6 * FS)
    assert partita.battito() == [] and partita.secondi_della_mira() < secondi and not asse.finita
    assert partita.mira.valori(inizio / FS + 0.6)[0] == pytest.approx(0.5 + 0.6 * VELOCITA_DELLA_MIRA)
    # La pausa automatica: Ctrl+P dal menu la lascia, e se ne va col menu.
    assert partita.pausa_automatica(True) == [] and partita.in_pausa
    assert partita.pausa()[0].testo == rifiuto and partita.in_pausa and banco.contati[-1] == "non_ora"
    assert partita.pausa_automatica(False) == [] and not partita.in_pausa
    conti = statistiche.dati["conti"]
    assert (conti["pause"], conti["pause_rifiutate"], conti["pause_automatiche"]) == (2, 3, 1)


def test_la_riserva_vuota_finisce_senza_attesa_e_la_fine_non_si_perde(banco, monkeypatch):
    """D22: l'ultimo colpo a vuoto chiude la partita come sempre, anche nella palestra, senza attesa; la
    fine arriva col resoconto, quando il tonfo ha finito di suonare, e lo spazio dato intanto non
    comincia un'altra partita, che passerebbe davanti alla classifica. La partita nuova e l'abbandono
    chiesti mentre l'effetto suona danno prima i messaggi che aspettavano."""
    lungo = tonfo_lungo(monkeypatch)
    partita = banco.partita()
    partita.nuova()
    partita.riserva = 1
    partita.spazio()
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    inizio = fino_all_impatto(banco, partita)
    assert partita.fase == "finita" and not partita.in_corso()
    prima = list(banco.contati)
    assert partita.spazio() == [] and partita.fase == "finita" and banco.contati == prima
    porta_udibile_a(banco, inizio + len(lungo))
    messaggi = partita.battito()
    assert [m.tipo for m in messaggi] == ["resoconto", "fine"]
    assert partita.spazio()[-1].annuncio == "Partita nuova: Spazio per mirare."
    # La partita nuova chiesta col tonfo che suona ancora: prima la fine di quella di prima.
    partita.riserva = 1
    partita.spazio()
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    fino_all_impatto(banco, partita)
    assert [m.tipo for m in partita.nuova()] == ["resoconto", "fine", "info"]
    assert partita.fase == "ascolto" and partita.battito() == []
    # L'abbandono, nella palestra, col tonfo di un tiro a vuoto che suona ancora: prima il resoconto e l'attesa.
    partita.spazio()
    mira_e_spara(banco, partita, (0.0, 0.0, 0.0))
    fino_all_impatto(banco, partita)
    assert partita.fase == "attesa"
    messaggi = partita.abbandona()
    assert [m.tipo for m in messaggi] == ["resoconto", "info", "info"] and messaggi[-1].annuncio == "Partita abbandonata."
    assert partita.fase == "pronta" and partita.battito() == []
