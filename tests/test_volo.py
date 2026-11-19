"""
Le prove di volo.py: centri pieni e di striscio, tiri corti, lunghi, a destra e sopra, bersagli
mobili, bonus già usciti, la voce del proiettile e il tempo di calcolo; e il passaggio del sommario
(D21), confrontato con un minimo cercato senza i passi del volo, in salita e in discesa, su un
bersaglio che si muove e fra due passi.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
"""

import math
import time

import numpy as np
import pytest

import fisica
import volo
from bersagli import Bersaglio
from costanti import ALTEZZA_CANNONE, CUBO_X, GRAVITA, PROIETTILE_DB, RAGGIO_PROIETTILE

PARTENZA = 20.0
# Quante volte si ripete il calcolo nella prova del tempo: conta il più veloce, che non risente
# degli altri processi in corsa sulla macchina.
RIPETIZIONI_DEL_CRONOMETRO = 5
TEMPO_MASSIMO_DEL_CALCOLO = 0.05
# Con passi di 5 centesimi di secondo, cinquanta volte quelli del gioco, il vertice della parabola
# sbaglia di qualche millimetro: il centimetro basta a distinguerlo dal passo, che sbaglia di decimetri.
TOLLERANZA_FRA_DUE_PASSI = 0.01


def _fermo(punto, raggio=1.0, **altro):
    return Bersaglio(numero=altro.pop("numero", 1), forma="seno", raggio=raggio, partenza=punto, **altro)


def _tiro_per(punto, alzo):
    """La traiettoria che a quell'alzo passa per il punto."""
    velocita = fisica.velocita_per(punto, alzo)
    assert not math.isnan(velocita)
    return fisica.Traiettoria(math.degrees(math.atan2(punto[0], punto[2])), alzo, velocita)


def _tiro_con_apice(punto):
    """La traiettoria che ha l'apice nel punto: lì la tangente è orizzontale."""
    salita = punto[1] - ALTEZZA_CANNONE
    verticale = math.sqrt(2 * GRAVITA * salita)
    orizzontale = math.hypot(punto[0], punto[2]) / (verticale / GRAVITA)
    return fisica.Traiettoria(math.degrees(math.atan2(punto[0], punto[2])), math.degrees(math.atan2(verticale, orizzontale)), math.hypot(verticale, orizzontale))


def _minimo_esatto(traiettoria, bersaglio, da=0.0, a=None):
    """Il tempo di volo della distanza minima fra il proiettile e il centro del bersaglio, dove il
    bersaglio sta in quell'istante, cercato senza i passi del volo: un campionamento a un decimillesimo
    di secondo, poi la sezione aurea attorno al campione migliore."""
    a = traiettoria.tempo_a_terra() if a is None else a

    def distanza(t):
        return float(np.linalg.norm(traiettoria.posizione(t) - bersaglio.posizione(PARTENZA + t)))

    tempi = np.arange(da, a, 1e-4)
    distanze = np.linalg.norm(traiettoria.posizione(tempi) - bersaglio.posizione(PARTENZA + tempi), axis=1)
    migliore = int(np.argmin(distanze))
    basso, alto = float(tempi[max(migliore - 1, 0)]), float(tempi[min(migliore + 1, len(tempi) - 1)])
    aurea = (math.sqrt(5.0) - 1.0) / 2.0
    for _giro in range(80):
        sinistra, destra = alto - aurea * (alto - basso), basso + aurea * (alto - basso)
        if distanza(sinistra) < distanza(destra):
            alto = destra
        else:
            basso = sinistra
    return (basso + alto) / 2.0


def _scarto_vero(traiettoria, bersaglio, tempo):
    """Lo scarto (radiale, laterale, verticale) del proiettile dal centro del bersaglio in quell'istante."""
    centro = bersaglio.posizione(PARTENZA + tempo)
    return fisica.scomponi(traiettoria.posizione(tempo) - centro, centro)


def _controlla_passaggio(esito, traiettoria, bersaglio, tolleranza=1e-4):
    """D21: il passaggio del volo è quello vero, nell'istante della distanza minima, con lo scarto dal
    centro del bersaglio in quello stesso istante e l'inclinazione del proiettile lì."""
    passaggio = esito.passaggio
    tempo = _minimo_esatto(traiettoria, bersaglio)
    assert passaggio.tempo == pytest.approx(tempo, abs=tolleranza / 10)
    assert passaggio.scarto == pytest.approx(_scarto_vero(traiettoria, bersaglio, tempo), abs=tolleranza)
    assert passaggio.inclinazione == pytest.approx(traiettoria.inclinazione(tempo), abs=tolleranza * 10)
    assert (passaggio.laterale, passaggio.verticale, passaggio.profondita) == (passaggio.scarto[1], passaggio.scarto[2], passaggio.scarto[0])
    assert passaggio.distanza == pytest.approx(math.sqrt(sum(c * c for c in passaggio.scarto)))
    return passaggio


def _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio):
    assert esito.colpito is None
    assert esito.precisione is None
    assert esito.distanza_al_centro is None
    assert esito.durata == pytest.approx(traiettoria.tempo_a_terra())
    assert esito.punto_finale[1] == 0.0
    assert esito.vicino is bersaglio
    assert esito.distanza_dal_vicino > RAGGIO_PROIETTILE
    assert esito.fine == pytest.approx(PARTENZA + esito.durata)
    # Lo scarto del tiro a vuoto è quello del passaggio, preso dove la distanza è minima, anche fra due
    # passi, e coerente con la distanza dal vicino.
    passaggio = _controlla_passaggio(esito, traiettoria, bersaglio)
    assert esito.scarto == passaggio.scarto
    assert passaggio.distanza - bersaglio.raggio == pytest.approx(esito.distanza_dal_vicino, abs=1e-6)
    return passaggio


def test_centro_pieno_su_bersaglio_fermo():
    traiettoria = fisica.Traiettoria(6.0, 42.0, 31.0)
    centro = traiettoria.posizione(1.8)
    bersaglio = _fermo(centro, raggio=2.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    assert esito.colpito is bersaglio
    assert esito.precisione > 0.995
    assert esito.distanza_al_centro < 0.01
    assert esito.vicino is None
    assert esito.scarto is None
    assert esito.durata < 1.8
    # La fine è il contatto, a raggio più RAGGIO_PROIETTILE dal centro, sulla traiettoria.
    assert np.allclose(esito.punto_finale, traiettoria.posizione(esito.durata))
    assert np.linalg.norm(esito.punto_finale - centro) == pytest.approx(2.0 + RAGGIO_PROIETTILE, abs=1e-3)
    assert esito.laterale_finale == pytest.approx(esito.punto_finale[0])
    assert esito.gittata == pytest.approx(math.hypot(esito.punto_finale[0], esito.punto_finale[2]))
    assert esito.altezza_massima == pytest.approx(max(traiettoria.posizione(np.linspace(0, esito.durata, 5001))[:, 1]), abs=1e-4)
    assert esito.fine == pytest.approx(PARTENZA + esito.durata)
    # D21: il passaggio del centro pieno è nel centro, dopo il contatto, dove si misura la precisione.
    passaggio = _controlla_passaggio(esito, traiettoria, bersaglio)
    assert passaggio.tempo == pytest.approx(1.8, abs=1e-5) and passaggio.tempo > esito.durata
    assert passaggio.distanza == pytest.approx(esito.distanza_al_centro, abs=1e-6)
    assert passaggio.distanza < 0.01
    assert passaggio.inclinazione == pytest.approx(traiettoria.inclinazione(1.8), abs=1e-3) and passaggio.inclinazione > 0.0


def test_tiro_di_striscio():
    traiettoria = fisica.Traiettoria(-4.0, 38.0, 30.0)
    sul_tiro = traiettoria.posizione(1.5)
    tangente = traiettoria.iniziale + np.array((0.0, -GRAVITA * 1.5, 0.0))
    di_lato = np.cross(tangente, (0.0, 1.0, 0.0))
    di_lato /= np.linalg.norm(di_lato)
    raggio = 1.5
    centro = sul_tiro + di_lato * (raggio + RAGGIO_PROIETTILE) * 0.95
    bersaglio = _fermo(centro, raggio=raggio)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    assert esito.colpito is bersaglio
    assert esito.precisione < 0.1
    assert esito.distanza_al_centro == pytest.approx((raggio + RAGGIO_PROIETTILE) * 0.95, abs=1e-3)
    # D21: il bersaglio sta a sinistra del tiro, di lato rispetto al piano verticale della parabola,
    # quindi il passaggio più vicino è proprio a 1,5 secondi, col proiettile a destra del centro, alla
    # sua altezza, e lo scarto è lungo quanto la distanza della precisione.
    passaggio = _controlla_passaggio(esito, traiettoria, bersaglio)
    assert passaggio.tempo == pytest.approx(1.5, abs=1e-5)
    assert passaggio.distanza == pytest.approx(esito.distanza_al_centro, abs=1e-6)
    assert passaggio.laterale > 0.9 * (raggio + RAGGIO_PROIETTILE) * 0.95
    assert passaggio.verticale == pytest.approx(0.0, abs=1e-6)
    assert passaggio.inclinazione == pytest.approx(traiettoria.inclinazione(1.5), abs=1e-3)
    # Un po' più in là non si tocca più.
    lontano = _fermo(sul_tiro + di_lato * (raggio + RAGGIO_PROIETTILE) * 1.05, raggio=raggio)
    assert volo.calcola_volo(traiettoria, [lontano], PARTENZA).colpito is None


def test_tiro_corto():
    centro = (0.0, 12.0, 55.0)
    bersaglio = _fermo(centro)
    traiettoria = _tiro_per((0.0, 12.0, 52.0), 80.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio)
    radiale, laterale, verticale = esito.scarto
    assert radiale < -2.5
    assert abs(radiale) > abs(verticale)
    assert abs(laterale) < 1e-6
    assert esito.distanza_dal_vicino == pytest.approx(math.sqrt(radiale**2 + laterale**2 + verticale**2) - 1.0, abs=0.01)
    assert esito.passaggio.profondita < -2.5


def test_tiro_lungo():
    centro = (0.0, 12.0, 55.0)
    bersaglio = _fermo(centro)
    traiettoria = _tiro_per((0.0, 12.0, 58.0), 80.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    passaggio = _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio)
    radiale, _, verticale = esito.scarto
    assert radiale > 2.5
    assert abs(radiale) > abs(verticale)
    # D21: lungo, in profondità positivo, e a 80 gradi d'alzo il proiettile ricade quasi a piombo.
    assert passaggio.profondita == radiale
    assert passaggio.inclinazione < -60.0


def test_tiro_a_destra():
    centro = (-5.0, 15.0, 50.0)
    bersaglio = _fermo(centro)
    traiettoria = _tiro_per((-2.0, 15.0, 50.0), 50.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio)
    radiale, laterale, verticale = esito.scarto
    assert laterale > 2.5
    assert abs(laterale) > abs(radiale)
    assert abs(laterale) > abs(verticale)
    # Il tiro passa a 3 metri dal centro, di lato: 2 dalla superficie di un bersaglio di raggio 1.
    assert esito.distanza_dal_vicino == pytest.approx(2.0, abs=0.05)
    # D21: a destra, laterale positivo.
    assert esito.passaggio.laterale == laterale > 2.5


def test_tiro_sopra():
    centro = (4.0, 15.0, 50.0)
    bersaglio = _fermo(centro)
    traiettoria = _tiro_con_apice((4.0, 18.0, 50.0))
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    _controlla_tiro_a_vuoto(esito, traiettoria, bersaglio)
    radiale, laterale, verticale = esito.scarto
    assert verticale == pytest.approx(3.0, abs=0.01)
    assert abs(radiale) < 0.05
    assert abs(laterale) < 0.05
    assert esito.distanza_dal_vicino == pytest.approx(2.0, abs=1e-3)
    assert esito.altezza_massima == pytest.approx(18.0)
    # D21: sopra, e nell'apice, con l'inclinazione nulla.
    assert esito.passaggio.verticale == pytest.approx(3.0, abs=1e-6)
    assert esito.passaggio.tempo == pytest.approx(traiettoria.apice()[0], abs=1e-5)
    assert esito.passaggio.inclinazione == pytest.approx(0.0, abs=1e-3)


def test_tiro_in_salita_e_in_discesa():
    """D21: lo stesso tiro passa sotto due bersagli uguali, 3 metri sopra la traiettoria, uno un
    secondo prima dell'apice e uno un secondo dopo: nel primo il proiettile sale, nel secondo scende,
    con la stessa inclinazione e il segno opposto, perché la parabola è simmetrica."""
    traiettoria = fisica.Traiettoria(0.0, 60.0, 30.0)
    apice, _altezza = traiettoria.apice()
    su = np.array((0.0, 3.0, 0.0))
    prima = _fermo(traiettoria.posizione(apice - 1.0) + su, numero=1)
    dopo = _fermo(traiettoria.posizione(apice + 1.0) + su, numero=2)
    salita = _controlla_tiro_a_vuoto(volo.calcola_volo(traiettoria, [prima], PARTENZA), traiettoria, prima)
    discesa = _controlla_tiro_a_vuoto(volo.calcola_volo(traiettoria, [dopo], PARTENZA), traiettoria, dopo)
    assert salita.inclinazione > 30.0 and discesa.inclinazione < -30.0
    assert salita.inclinazione == pytest.approx(-discesa.inclinazione, abs=1e-3)
    assert salita.tempo < apice < discesa.tempo
    # Il proiettile passa sotto tutti e due. In salita il punto più vicino al bersaglio, che sta sopra,
    # viene un po' più avanti lungo la curva, quindi lungo; in discesa un po' prima, quindi corto.
    assert salita.verticale < -1.0 and discesa.verticale < -1.0
    assert discesa.profondita < 0.0 < salita.profondita
    assert salita.distanza == pytest.approx(discesa.distanza, abs=1e-4)


def test_il_vicino_e_il_piu_vicino_dalla_superficie():
    traiettoria = _tiro_con_apice((0.0, 20.0, 50.0))
    # Il primo è più vicino fra i centri, il secondo dalla superficie, perché è più grande.
    piccolo = _fermo((0.0, 24.0, 50.0), raggio=0.5, numero=1)
    grande = _fermo((0.0, 26.0, 50.0), raggio=4.0, numero=2)
    esito = volo.calcola_volo(traiettoria, [piccolo, grande], PARTENZA)
    assert esito.colpito is None
    assert esito.vicino is grande
    assert esito.distanza_dal_vicino == pytest.approx(2.0, abs=1e-3)


def test_vince_il_primo_contatto():
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    dopo = _fermo(traiettoria.posizione(2.0), numero=1)
    prima = _fermo(traiettoria.posizione(1.0), numero=2)
    esito = volo.calcola_volo(traiettoria, [dopo, prima], PARTENZA)
    assert esito.colpito is prima


def test_bersaglio_mobile_che_attraversa_la_traiettoria():
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    incrocio = traiettoria.posizione(1.6)
    rapidita = 4.0
    # Il bersaglio nasce prima dello sparo e arriva sulla traiettoria proprio quando passa il proiettile.
    nato = PARTENZA - 1.0
    partenza = incrocio - np.array((rapidita * (PARTENZA + 1.6 - nato), 0.0, 0.0))
    mobile = Bersaglio(numero=1, forma="sega", raggio=1.0, partenza=partenza, velocita=(rapidita, 0.0, 0.0), nato=nato)
    assert np.allclose(mobile.posizione(PARTENZA + 1.6), incrocio)
    esito = volo.calcola_volo(traiettoria, [mobile], PARTENZA)
    assert esito.colpito is mobile
    assert esito.precisione > 0.9
    # Fermo dov'era alla partenza non si colpisce, e nemmeno se arriva un secondo dopo.
    fermo = _fermo(mobile.posizione(PARTENZA))
    assert volo.calcola_volo(traiettoria, [fermo], PARTENZA).colpito is None
    in_ritardo = Bersaglio(numero=2, forma="sega", raggio=1.0, partenza=partenza - np.array((rapidita, 0.0, 0.0)), velocita=(rapidita, 0.0, 0.0), nato=nato)
    esito = volo.calcola_volo(traiettoria, [in_ritardo], PARTENZA)
    assert esito.colpito is None
    assert esito.vicino is in_ritardo
    assert esito.scarto[1] > 0.0


def test_passaggio_su_un_bersaglio_che_si_muove():
    """D21: con un bersaglio in moto lo scarto è dal suo centro nello stesso istante, non da dove
    stava allo sparo, nei tiri a vuoto e nei centri."""
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    incrocio = traiettoria.posizione(1.6)
    rapidita = 4.0
    nato = PARTENZA - 1.0
    # Arriva sulla traiettoria un secondo dopo il proiettile, venendo da sinistra: il proiettile gli
    # passa davanti, alla sua destra.
    in_ritardo = Bersaglio(numero=1, forma="sega", raggio=1.0, partenza=incrocio - np.array((rapidita * (PARTENZA + 2.6 - nato), 0.0, 0.0)),
                           velocita=(rapidita, 0.0, 0.0), nato=nato)
    esito = volo.calcola_volo(traiettoria, [in_ritardo], PARTENZA)
    passaggio = _controlla_tiro_a_vuoto(esito, traiettoria, in_ritardo)
    assert passaggio.laterale > 2.5
    # Contro il centro di partenza lo scarto sarebbe tutt'altro.
    allo_sparo = in_ritardo.posizione(PARTENZA)
    fermo = fisica.scomponi(traiettoria.posizione(passaggio.tempo) - allo_sparo, allo_sparo)
    assert abs(fermo[1] - passaggio.laterale) > 1.0
    # Un centro sullo stesso moto, che arriva insieme al proiettile ma un metro più in alto.
    in_orario = Bersaglio(numero=2, forma="sega", raggio=1.5, partenza=incrocio + np.array((-rapidita * (PARTENZA + 1.6 - nato), 1.0, 0.0)),
                          velocita=(rapidita, 0.0, 0.0), nato=nato)
    centro = volo.calcola_volo(traiettoria, [in_orario], PARTENZA)
    assert centro.colpito is in_orario
    passaggio = _controlla_passaggio(centro, traiettoria, in_orario)
    assert passaggio.distanza == pytest.approx(centro.distanza_al_centro, abs=1e-6)
    assert passaggio.verticale < 0.0


def test_passaggio_fra_due_passi(monkeypatch):
    """D21: con passi grossi il minimo cade fra due passi, e il passaggio si prende lo stesso nel punto
    giusto, al vertice della parabola, non sul passo più vicino; vale nei tiri a vuoto e nei centri."""
    monkeypatch.setattr(volo, "PASSO_DEL_VOLO", 0.05)
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    fermo = _fermo((-4.0, 20.0, 45.0))
    in_moto = Bersaglio(numero=2, forma="seno", raggio=1.0, partenza=(10.0, 20.0, 50.0), velocita=(-3.0, 1.0, -2.0), nato=PARTENZA - 1.0)
    for bersaglio in (fermo, in_moto):
        esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
        assert esito.colpito is None
        passaggio = _controlla_passaggio(esito, traiettoria, bersaglio, tolleranza=TOLLERANZA_FRA_DUE_PASSI)
        assert esito.scarto == passaggio.scarto
        # Sul passo più vicino lo scarto sbaglierebbe di decimetri: la prova guarda proprio fra due passi.
        passi = np.linspace(0.0, traiettoria.tempo_a_terra(), math.ceil(traiettoria.tempo_a_terra() / 0.05) + 1)
        sul_passo = passi[int(np.argmin(np.abs(passi - passaggio.tempo)))]
        assert np.abs(np.array(_scarto_vero(traiettoria, bersaglio, sul_passo)) - np.array(passaggio.scarto)).max() > 0.1
    # Un centro col passaggio a 1,53 secondi, fra due passi, a un metro dal centro di un bersaglio
    # abbastanza grande perché il proiettile resti dentro per qualche passo.
    sul_tiro = traiettoria.posizione(1.53)
    tangente = traiettoria.iniziale + np.array((0.0, -GRAVITA * 1.53, 0.0))
    di_lato = np.cross(tangente, (0.0, 1.0, 0.0))
    di_lato /= np.linalg.norm(di_lato)
    largo = _fermo(sul_tiro + di_lato * 1.0, raggio=3.0)
    esito = volo.calcola_volo(traiettoria, [largo], PARTENZA)
    assert esito.colpito is largo
    passaggio = _controlla_passaggio(esito, traiettoria, largo, tolleranza=TOLLERANZA_FRA_DUE_PASSI)
    assert passaggio.tempo == pytest.approx(1.53, abs=TOLLERANZA_FRA_DUE_PASSI / 10)
    assert passaggio.distanza == pytest.approx(esito.distanza_al_centro, abs=1e-3) and passaggio.distanza == pytest.approx(1.0, abs=1e-3)
    passi = np.linspace(0.0, traiettoria.tempo_a_terra(), math.ceil(traiettoria.tempo_a_terra() / 0.05) + 1)
    assert np.abs(passi - 1.53).min() > 0.005


def test_bonus_gia_uscito_non_si_colpisce():
    # Un tiro che passa fuori dal cubo, oltre la parete di destra.
    punto = (CUBO_X[1] + 5.0, 12.0, 45.0)
    traiettoria = _tiro_per(punto, 45.0)
    tempo = math.hypot(punto[0], punto[2]) / (traiettoria.velocita * math.cos(math.radians(45.0)))
    raggio = 1.0
    rapidita = 6.0
    nato = 0.0
    partenza = np.array(punto) - np.array((rapidita * (PARTENZA + tempo - nato), 0.0, 0.0))
    bonus = Bersaglio(numero=1, forma="quadra", raggio=raggio, partenza=partenza, velocita=(rapidita, 0.0, 0.0), nato=nato, bonus=True)
    # Senza l'uscita sarebbe proprio lì quando passa il proiettile.
    assert np.allclose(bonus.posizione(PARTENZA + tempo), punto)
    assert not bonus.vivo(PARTENZA + tempo)
    esito = volo.calcola_volo(traiettoria, [bonus], PARTENZA)
    assert esito.colpito is None
    assert esito.durata == pytest.approx(traiettoria.tempo_a_terra())


def test_bonus_non_ancora_nato_non_si_colpisce():
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    bonus = Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=traiettoria.posizione(1.0), nato=PARTENZA + 1.5, bonus=True)
    esito = volo.calcola_volo(traiettoria, [bonus], PARTENZA)
    assert esito.colpito is None
    # Non era mai vivo durante il volo prima di 1,5 secondi, e dopo era lontano: è il vicino ugualmente.
    assert esito.vicino is bonus


def test_senza_bersagli():
    traiettoria = fisica.Traiettoria(10.0, 30.0, 25.0)
    esito = volo.calcola_volo(traiettoria, [], PARTENZA)
    assert esito.colpito is None
    assert esito.vicino is None
    assert esito.distanza_dal_vicino is None
    assert esito.scarto is None
    assert esito.passaggio is None
    assert esito.durata == pytest.approx(traiettoria.tempo_a_terra())
    assert np.allclose(esito.punto_finale, traiettoria.punto_a_terra())
    assert esito.altezza_massima == pytest.approx(traiettoria.apice()[1])
    assert esito.gittata == pytest.approx(math.hypot(*traiettoria.punto_a_terra()[[0, 2]]))
    assert esito.laterale_finale > 0.0


def test_parametri_del_proiettile():
    traiettoria = fisica.Traiettoria(-8.0, 40.0, 28.0)
    esito = volo.calcola_volo(traiettoria, [], PARTENZA)
    parametri = volo.parametri_del_proiettile(esito)
    tempi_di_gioco = np.linspace(PARTENZA - 1.0, PARTENZA + esito.durata + 1.0, 4000)
    tempi_audio = np.linspace(100.0, 101.0, 4000)
    frequenze, pan, guadagni = parametri(tempi_di_gioco, tempi_audio)
    assert frequenze.shape == pan.shape == guadagni.shape == (4000,)
    tempi_di_volo = np.clip(tempi_di_gioco - PARTENZA, 0.0, esito.durata)
    attese = fisica.suono_di(traiettoria.posizione(tempi_di_volo))
    assert np.allclose(frequenze, attese[0])
    assert np.allclose(pan, attese[1])
    assert np.allclose(guadagni, attese[2] * 10 ** (PROIETTILE_DB / 20))
    # Prima dello sparo sta nel cannone, dopo la fine resta nel punto finale.
    prima = fisica.suono_di(traiettoria.posizione(0.0))
    dopo = fisica.suono_di(esito.punto_finale)
    assert frequenze[0] == pytest.approx(prima[0])
    assert frequenze[-1] == pytest.approx(dopo[0])
    assert pan[-1] == pytest.approx(dopo[1])
    # Il tempo di gioco fermo dà un suono fermo, lungo quanto il blocco dell'audio.
    fermo = parametri(PARTENZA + 0.5, tempi_audio)
    assert all(valori.shape == (4000,) for valori in fermo)
    assert np.all(fermo[0] == fermo[0][0])


def test_parametri_del_proiettile_dopo_un_centro():
    traiettoria = fisica.Traiettoria(0.0, 45.0, 30.0)
    bersaglio = _fermo(traiettoria.posizione(1.2), raggio=2.0)
    esito = volo.calcola_volo(traiettoria, [bersaglio], PARTENZA)
    parametri = volo.parametri_del_proiettile(esito)
    frequenze, pan, _ = parametri(np.array([PARTENZA + 5.0]), np.array([0.0]))
    attese = fisica.suono_di(esito.punto_finale)
    assert frequenze[0] == pytest.approx(attese[0])
    assert pan[0] == pytest.approx(attese[1])


def test_tempo_di_calcolo():
    """Dieci secondi di volo fra cinque bersagli, mobili e bonus compresi, in meno di 50 millesimi:
    il volo si calcola nel filo principale, allo sparo, e non deve farsi sentire."""
    verticale = 49.0
    traiettoria = fisica.Traiettoria(-30.0, math.degrees(math.atan2(verticale, 5.0)), math.hypot(verticale, 5.0))
    assert traiettoria.tempo_a_terra() == pytest.approx(10.0, abs=0.1)
    gruppo = [
        Bersaglio(numero=1, forma="seno", raggio=1.0, partenza=(20.0, 30.0, 60.0), velocita=(6.0, 3.0, -2.0)),
        Bersaglio(numero=2, forma="deriva", raggio=0.5, partenza=(10.0, 40.0, 70.0), velocita=(-4.0, 5.0, 8.0), seme=3),
        Bersaglio(numero=3, forma="quadra", raggio=2.0, partenza=(15.0, 10.0, 40.0)),
        Bersaglio(numero=4, forma="casuale", raggio=1.5, partenza=(5.0, 20.0, 50.0), velocita=(11.0, -2.0, 1.0)),
        Bersaglio(numero=5, forma="sega", raggio=0.8, partenza=(-25.8, 30.0, 60.0), velocita=(4.0, 0.0, 0.0), bonus=True),
    ]
    volo.calcola_volo(traiettoria, gruppo, 0.0)
    durate = []
    for _ in range(RIPETIZIONI_DEL_CRONOMETRO):
        inizio = time.perf_counter()
        esito = volo.calcola_volo(traiettoria, gruppo, 0.0)
        durate.append(time.perf_counter() - inizio)
    assert esito.durata > 9.9
    assert min(durate) < TEMPO_MASSIMO_DEL_CALCOLO, f"il volo più veloce ha chiesto {min(durate) * 1000:.1f} millesimi"
