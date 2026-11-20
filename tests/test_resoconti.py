"""
Le prove di resoconti.py: le frasi dei tiri, dei livelli, dei bonus, dei primati e della fine, i segni
delle direzioni e degli scarti, i singolari e i plurali, le righe della barra, il sommario di D21 ai
tre gradi della palestra, coi segni, le soglie, le misure che mancano e i confini fra i livelli,
l'annuncio con l'esito prima del sommario, i messaggi e la barra della palestra delle attese di D22,
la riga del sommario a blocchi per il display braille di D24, il proiettile perduto e Esc nella mira di
D25, la precisione globale di D26, e nessun separatore grafico. Dalla 0.11.8 i valori senza lo spazio dopo
i due punti, con G nel primo blocco, e il sommario a parole su due righe.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
I voli sono costruiti a mano, coi numeri tondi che le frasi devono dire; i numeri che dipendono dalle
costanti di taratura, come il raggio di un livello, si sostituiscono nel modulo, così le frasi restano
note anche quando Gabriele tara il gioco.
"""

import math
import re

import numpy as np
import pytest

import resoconti
from bersagli import Bersaglio
from costanti import (
    AIUTO_MANCATO_FINO_AL_LIVELLO,
    BLOCCO_DEL_BRAILLE,
    CENTRI_PER_LIVELLO,
    COLPI_BONUS,
    PASSAGGI_DELLA_BARRA,
    SOMMARIO_NUMERI_FINO_AL_LIVELLO,
    SOMMARIO_PAROLE_FINO_AL_LIVELLO,
    SOMMARIO_SOGLIA_INCLINAZIONE,
    SOMMARIO_SOGLIA_LATERALE,
    SOMMARIO_SOGLIA_PROFONDITA,
    SOMMARIO_SOGLIA_VERTICALE,
)
from fisica import Traiettoria, distanza_dall_angolo_piu_lontano
from resoconti import Messaggio
from volo import Misure, Volo

# Le righe fatte di soli simboli ripetuti, o tre simboli uguali di fila: separatori grafici.
SEPARATORE = re.compile(r"([-=_*~#+.])\1{2,}")
TESTI = []


def raccogli(messaggio):
    """Tiene da parte testo e annuncio di ogni messaggio, per la prova dei separatori; un annuncio vuoto,
    come quello della riga della palestra di D22, non è un testo."""
    TESTI.extend(testo for testo in (messaggio.testo, messaggio.annuncio) if testo)
    return messaggio


# Un punto a 48,3 metri dal cannone in orizzontale, 5,2 a sinistra e 22,1 d'altezza.
DOVE_STAVA = (-5.2, 22.1, math.sqrt(48.3**2 - 5.2**2))


def bersaglio(raggio=3.6, posizione=DOVE_STAVA, forma="sega", **altro):
    return Bersaglio(numero=1, forma=forma, raggio=raggio, partenza=posizione, **altro)


def volo(colpito=None, azimut=12.4, alzo=38.2, velocita=27.5, **campi):
    base = {"traiettoria": Traiettoria(azimut, alzo, velocita), "partenza": 10.0, "durata": 2.8, "colpito": colpito, "punto_finale": np.array((3.1, 0.0, 56.2)),
            "precisione": 0.64 if colpito is not None else None, "distanza_al_centro": 0.5 if colpito is not None else None, "vicino": None,
            "distanza_dal_vicino": None, "scarto": None, "altezza_massima": 23.4, "gittata": 56.2, "laterale_finale": 3.1}
    base.update(campi)
    return Volo(**base)


def mancato(scarto=(1.2, -0.8, 1.7), distanza=2.3, **campi):
    return volo(vicino=bersaglio(), distanza_dal_vicino=distanza, scarto=scarto, **campi)


def misure(laterale=13.2, verticale=-2.4, profondita=1.0, inclinazione=-31.4, arriva=True, ricade=True):
    """Le misure dell'artigliere di D21, date nell'ordine del sommario; arriva falso toglie l'arrivo,
    con laterale, verticale e inclinazione, ricade falso la ricaduta, con la profondità."""
    if not arriva:
        laterale = verticale = inclinazione = None
    if not ricade:
        profondita = None
    return Misure(arrivo=2.0 if arriva else None, laterale=laterale, verticale=verticale, inclinazione=inclinazione, ricaduta=2.5 if ricade else None,
                  profondita=profondita)


VALORI = (0.64, 0.45, 0.5)


def blocchi(primo, secondo):
    """La riga in due blocchi di D24: primo, spazi fino al carattere BLOCCO_DEL_BRAILLE compreso, poi secondo; se primo non ci sta,
    secondo dopo uno spazio."""
    return primo.ljust(BLOCCO_DEL_BRAILLE) + secondo if len(primo) < BLOCCO_DEL_BRAILLE else f"{primo} {secondo}"


# La legenda del sommario coi numeri (D24), con la voce della precisione globale (D26), e quella delle parole.
LEGENDA = "L laterale, V verticale, P profondità, in metri; I inclinazione, in gradi; dal bersaglio"
LEGENDA_CON_G = f"{LEGENDA}; G precisione globale"
DAL_BERSAGLIO = "Dal bersaglio: lato, altezza, profondità, inclinazione."
# Le precisioni globali dei voli costruiti a mano (D26): il centro a mezzo metro dal centro del bersaglio, il tiro a vuoto a 2,3 metri
# dalla superficie del bersaglio di 3,6 metri; dal terzo collaudo sulla distanza fra DOVE_STAVA e il più lontano degli angoli del cubo,
# 52,1 metri.
G_DEL_CENTRO = "99,0%"
G_DEL_MANCATO = "88,7%"
# Il bonus dei voli costruiti a mano nasce alla loro fine, a 12,8 secondi di gioco: in quell'istante, quello del passaggio di un volo
# senza tempo_del_passaggio, sta in DOVE_STAVA, come il bersaglio fermo, e la precisione globale è la stessa.
NATO_DEL_BONUS = 12.8
# Il sommario coi numeri delle misure di serie, e quello a parole, nel tiro a vuoto: la riga e l'annuncio. Dalla 0.11.8 i valori non hanno
# lo spazio dopo i due punti, e coi numeri G sta con loro nel primo blocco; a parole la precisione globale e il punto di vista vanno sulla
# seconda riga.
VALORI_NUMERI = "L:+13,2 V:-2,4 P:+1,0 I:-31,4"
NUMERI = blocchi(f"{VALORI_NUMERI} G:{G_DEL_MANCATO}", f"{LEGENDA_CON_G}.")
NUMERI_ANNUNCIO = f"{VALORI_NUMERI} G:{G_DEL_MANCATO}"
PAROLE_SOLE = "A destra, sotto, lungo, in discesa."
SOTTO_LE_PAROLE = f"Precisione globale {G_DEL_MANCATO}. {DAL_BERSAGLIO}"
PAROLE = f"{PAROLE_SOLE}\n{SOTTO_LE_PAROLE}"
PAROLE_ANNUNCIO = f"{PAROLE_SOLE} Precisione globale {G_DEL_MANCATO}."
# La riga del mancato di D6 con le misure di serie, nell'ordine del sommario.
MANCATO = "Il bersaglio più vicino è stato mancato di 2,3 metri: rispetto al suo centro il tiro è passato a destra di 13,2 metri, sotto di 2,4 metri e lungo di 1 metro."


def test_resoconto_di_un_centro():
    # Un volo senza le misure non ha il sommario: la precisione globale di D26 va nella riga dell'esito, e nell'annuncio.
    messaggio = raccogli(resoconti.resoconto_del_tiro(12, 3, volo(bersaglio()), VALORI, punti=230, totale=1230, riserva=8))
    assert messaggio.tipo == "resoconto"
    assert messaggio.testo.split("\n") == [
        f"Tiro 12, livello 3: colpito un bersaglio, forma dente di sega, raggio 3,6 metri, fermo; 230 punti, precisione 64%, precisione globale {G_DEL_CENTRO}, "
        "totale 1.230 punti.",
        "Mira: direzione 12,4 gradi a destra, 64% della scala; alzo 38,2 gradi, 45%; potenza 27,5 metri al secondo, 50%.",
        "Traiettoria: altezza massima 23,4 metri; impatto a 56,2 metri dal cannone, 3,1 metri a destra del centro, dopo 2,8 secondi di volo.",
        "Il bersaglio stava a 48,3 metri dal cannone, a 22,1 metri d'altezza, 5,2 metri a sinistra del centro.",
    ]
    assert messaggio.annuncio == f"Colpito, 230 punti. Precisione globale {G_DEL_CENTRO}."
    assert messaggio.dati == {"tiro": 12, "livello": 3, "colpito": True, "bonus": False, "punti": 230, "totale": 1230, "riserva": 8, "precisione": 0.64,
                              "precisione_globale": pytest.approx(1 - 0.5 / distanza_dall_angolo_piu_lontano(DOVE_STAVA))}


def test_resoconto_di_un_bonus_e_di_un_mobile():
    bonus = bersaglio(raggio=2.9, forma="seno", bonus=True, velocita=(3.9, 0.0, 0.0), nato=NATO_DEL_BONUS)
    messaggio = raccogli(resoconti.resoconto_del_tiro(7, 4, volo(bonus, precisione=0.5), VALORI, punti=1150, totale=2380, riserva=12))
    assert messaggio.testo.split("\n")[0] == (f"Tiro 7, livello 4: colpito il bonus, forma seno, raggio 2,9 metri; 1.150 punti, precisione 50%, "
                                              f"precisione globale {G_DEL_CENTRO}, {COLPI_BONUS} colpi in più, totale 2.380 punti.")
    assert messaggio.annuncio == f"Colpito il bonus, 1.150 punti. Precisione globale {G_DEL_CENTRO}."
    assert messaggio.dati["bonus"]
    uno = raccogli(resoconti.resoconto_del_tiro(7, 4, volo(bonus), VALORI, punti=1, totale=1, riserva=1, colpi_bonus=1))
    assert f"; 1 punto, precisione 64%, precisione globale {G_DEL_CENTRO}, 1 colpo in più, totale 1 punto." in uno.testo
    mobile = bersaglio(raggio=2.0, forma="triangolo", velocita=(2.3, 0.0, 0.0))
    assert resoconti.descrizione_del_bersaglio(mobile) == "forma triangolo, raggio 2 metri, mobile a 2,3 metri al secondo"
    assert resoconti.descrizione_del_bersaglio(bersaglio(raggio=1.0, forma="quadra")) == "forma onda quadra, raggio 1 metro, fermo"


def test_resoconto_di_un_tiro_a_vuoto_con_l_aiuto():
    messaggio = raccogli(resoconti.resoconto_del_tiro(4, 1, mancato(), VALORI, riserva=8))
    assert messaggio.testo.split("\n") == [
        f"Tiro 4, livello 1: a vuoto, precisione globale {G_DEL_MANCATO}; nella riserva restano 8 colpi.",
        "Mira: direzione 12,4 gradi a destra, 64% della scala; alzo 38,2 gradi, 45%; potenza 27,5 metri al secondo, 50%.",
        "Traiettoria: altezza massima 23,4 metri; a terra a 56,2 metri dal cannone, 3,1 metri a destra del centro, dopo 2,8 secondi di volo.",
        "Il bersaglio più vicino è stato mancato di 2,3 metri: rispetto al suo centro il tiro è passato a sinistra di 0,8 metri, sopra di 1,7 metri e lungo di 1,2 metri.",
    ]
    assert messaggio.annuncio == f"A vuoto, riserva 8, sopra di 1,7 metri. Precisione globale {G_DEL_MANCATO}."
    assert not messaggio.dati["colpito"] and messaggio.dati["riserva"] == 8


def test_segni_opposti_e_singolari():
    messaggio = raccogli(resoconti.resoconto_del_tiro(1, 1, mancato(scarto=(-3.4, 0.5, -0.2), distanza=1.0, azimut=-7.0, laterale_finale=-1.0), VALORI, riserva=1))
    righe = messaggio.testo.split("\n")
    assert righe[0] == "Tiro 1, livello 1: a vuoto, precisione globale 91,2%; nella riserva resta 1 colpo."
    assert righe[1].startswith("Mira: direzione 7 gradi a sinistra, 64% della scala;")
    assert ", 1 metro a sinistra del centro," in righe[2]
    assert righe[3] == ("Il bersaglio più vicino è stato mancato di 1 metro: rispetto al suo centro il tiro è passato a destra di 0,5 metri, sotto di 0,2 metri "
                        "e corto di 3,4 metri.")
    assert messaggio.annuncio == "A vuoto, riserva 1, corto di 3,4 metri. Precisione globale 91,2%."
    assert resoconti.scarti((0.04, -0.03, 2.0)) == ["sopra di 2 metri"]
    # Lo scarto arriva nell'ordine di fisica.scomponi, e si dice in quello del sommario (D21); le
    # misure che mancano non si dicono.
    assert resoconti.scarti((1.2, -0.8, 1.7)) == ["a sinistra di 0,8 metri", "sopra di 1,7 metri", "lungo di 1,2 metri"]
    assert resoconti.scarti((-12.3, None, None)) == ["corto di 12,3 metri"]
    assert resoconti.scarti((None, 2.0, -1.0)) == ["a destra di 2 metri", "sotto di 1 metro"]
    assert resoconti.scarto_principale((0.1, -4.0, 2.0)) == "a sinistra di 4 metri"
    assert resoconti.scarto_principale((-12.3, None, None)) == "corto di 12,3 metri"
    assert resoconti.scarto_principale((None, -0.5, 7.0)) == "sopra di 7 metri"
    # A parità vince la prima nell'ordine del sommario: laterale, verticale, profondità.
    assert resoconti.scarto_principale((-2.0, 2.0, 2.0)) == "a destra di 2 metri"
    assert resoconti.scarto_principale((-2.0, None, 2.0)) == "sopra di 2 metri"
    assert resoconti.scarto_principale((None, None, None)) == ""
    assert resoconti.scarto_principale(None) == ""
    assert resoconti.direzione(0.03) == "al centro"
    assert resoconti.direzione(-1.0) == "1 grado a sinistra"
    assert resoconti.lato(0.04) == "al centro"
    assert resoconti.metri(1.04) == "1 metro"
    assert resoconti.metri(-0.01) == "0 metri"
    assert resoconti.metri_al_secondo(1.0) == "1 metro al secondo"
    assert resoconti.metri_al_secondo(10) == "10 metri al secondo"
    assert resoconti.secondi(1.0) == "1 secondo"
    assert resoconti.secondi(0.45) == "0,5 secondi"
    assert resoconti.gradi(1.0) == "1 grado"


def test_tiro_a_vuoto_senza_aiuto_e_a_riserva_vuota():
    oltre = raccogli(resoconti.resoconto_del_tiro(30, AIUTO_MANCATO_FINO_AL_LIVELLO + 1, mancato(), VALORI, riserva=8))
    assert len(oltre.testo.split("\n")) == 3 and "mancato" not in oltre.testo
    assert oltre.annuncio == f"A vuoto, riserva 8. Precisione globale {G_DEL_MANCATO}."
    forzato = raccogli(resoconti.resoconto_del_tiro(30, AIUTO_MANCATO_FINO_AL_LIVELLO + 1, mancato(), VALORI, riserva=8, aiuto=True))
    assert len(forzato.testo.split("\n")) == 4
    vuota = raccogli(resoconti.resoconto_del_tiro(10, 1, mancato(), VALORI, riserva=0))
    assert vuota.testo.split("\n")[0] == f"Tiro 10, livello 1: a vuoto, precisione globale {G_DEL_MANCATO}; la riserva è vuota."
    assert vuota.annuncio == f"A vuoto, riserva vuota. Precisione globale {G_DEL_MANCATO}."
    # Un tiro senza bersagli vivi vicini non ha la riga del mancato, né la precisione globale.
    solo = raccogli(resoconti.resoconto_del_tiro(2, 1, volo(), VALORI, riserva=5))
    assert len(solo.testo.split("\n")) == 3 and solo.annuncio == "A vuoto, riserva 5." and solo.dati["precisione_globale"] is None
    assert solo.testo.split("\n")[0] == "Tiro 2, livello 1: a vuoto; nella riserva restano 5 colpi."


def test_sommario_coi_numeri_in_testa_al_resoconto():
    """D21, i primi livelli: il sommario coi numeri è la prima riga del resoconto; la parte verbosa che
    segue resta quella senza sommario, riga per riga, salvo la precisione globale di D26, che senza il
    sommario sta nella riga dell'esito; i dati non cambiano. Da D24 la riga ha i valori con le loro
    iniziali nei primi 40 caratteri e la legenda dal 41°. L'annuncio dice prima l'esito, poi i valori,
    senza la legenda, e l'esito non ripete lo scarto principale, che il sommario dice già."""
    tiro = mancato(misure=misure())
    for livello in range(1, SOMMARIO_NUMERI_FINO_AL_LIVELLO + 1):
        senza = resoconti.resoconto_del_tiro(4, livello, tiro, VALORI, riserva=8, palestra="niente")
        con = raccogli(resoconti.resoconto_del_tiro(4, livello, tiro, VALORI, riserva=8))
        righe, righe_senza = con.testo.split("\n"), senza.testo.split("\n")
        # Dalla 0.11.8 G sta coi valori nel primo blocco, e il secondo comincia con la legenda.
        assert righe[0] == NUMERI and righe[0][BLOCCO_DEL_BRAILLE:].startswith("L laterale, ")
        assert righe[0][:BLOCCO_DEL_BRAILLE].rstrip() == f"{VALORI_NUMERI} G:{G_DEL_MANCATO}"
        assert righe[1] == f"Tiro 4, livello {livello}: a vuoto; nella riserva restano 8 colpi."
        assert righe_senza[0] == f"Tiro 4, livello {livello}: a vuoto, precisione globale {G_DEL_MANCATO}; nella riserva restano 8 colpi."
        assert righe[2:] == righe_senza[1:]
        assert con.dati == senza.dati
        if livello <= AIUTO_MANCATO_FINO_AL_LIVELLO:
            # La riga del mancato di D6 dice le misure del sommario, nel suo ordine.
            assert righe[-1] == MANCATO
            assert senza.annuncio == f"A vuoto, riserva 8, a destra di 13,2 metri. Precisione globale {G_DEL_MANCATO}."
        assert con.annuncio == f"A vuoto, riserva 8. {NUMERI_ANNUNCIO}"
    # Anche nei centri: di quanto il proiettile è passato lontano dal centro del bersaglio colpito, con la precisione globale nel
    # primo blocco.
    centro = raccogli(resoconti.resoconto_del_tiro(12, 1, volo(bersaglio(), misure=misure(0.3, 0.04, -0.06, 12.0)), VALORI, punti=230, totale=1230, riserva=8))
    righe = centro.testo.split("\n")
    assert righe[0] == blocchi(f"L:+0,3 V:0,0 P:-0,1 I:+12,0 G:{G_DEL_CENTRO}", f"{LEGENDA_CON_G}.")
    assert righe[1].startswith("Tiro 12, livello 1: colpito un bersaglio, ") and "globale" not in righe[1] and len(righe) == 5
    assert centro.annuncio == f"Colpito, 230 punti. L:+0,3 V:0,0 P:-0,1 I:+12,0 G:{G_DEL_CENTRO}"
    pieno = raccogli(resoconti.resoconto_del_tiro(13, 1, volo(bersaglio(), misure=misure(0.0, 0.0, 0.0, -45.8)), VALORI, punti=230, totale=1460, riserva=8))
    assert pieno.testo.split("\n")[0] == blocchi(f"L:0,0 V:0,0 P:0,0 I:-45,8 G:{G_DEL_CENTRO}", f"{LEGENDA_CON_G}.")
    assert pieno.annuncio == f"Colpito, 230 punti. L:0,0 V:0,0 P:0,0 I:-45,8 G:{G_DEL_CENTRO}"
    # E nel bonus, dove l'annuncio ha le sue parole.
    bonus = bersaglio(raggio=2.9, forma="seno", bonus=True, velocita=(3.9, 0.0, 0.0), nato=NATO_DEL_BONUS)
    preso = raccogli(resoconti.resoconto_del_tiro(7, 2, volo(bonus, misure=misure()), VALORI, punti=1150, totale=2380, riserva=12))
    assert preso.testo.split("\n")[:2] == [blocchi(f"{VALORI_NUMERI} G:{G_DEL_CENTRO}", f"{LEGENDA_CON_G}."),
                                            f"Tiro 7, livello 2: colpito il bonus, forma seno, raggio 2,9 metri; 1.150 punti, precisione 64%, "
                                            f"{COLPI_BONUS} colpi in più, totale 2.380 punti."]
    assert preso.annuncio == f"Colpito il bonus, 1.150 punti. {VALORI_NUMERI} G:{G_DEL_CENTRO}"
    # A riserva vuota l'esito resta il suo, e il sommario lo segue.
    vuota = raccogli(resoconti.resoconto_del_tiro(10, 1, tiro, VALORI, riserva=0))
    assert vuota.annuncio == f"A vuoto, riserva vuota. {NUMERI_ANNUNCIO}"


def test_segni_del_sommario():
    """D21: i numeri del sommario hanno sempre un decimale e il segno, + o il trattino; quelli che si
    arrotondano a zero si scrivono 0,0, senza segno."""
    assert resoconti.con_segno(13.24) == "+13,2"
    assert resoconti.con_segno(-2.4) == "-2,4"
    assert resoconti.con_segno(1.0) == "+1,0"
    assert resoconti.con_segno(0.04) == resoconti.con_segno(-0.04) == resoconti.con_segno(0.0) == "0,0"
    assert resoconti.con_segno(-0.06) == "-0,1" and resoconti.con_segno(0.06) == "+0,1"
    assert resoconti.con_segno(1234.56) == "+1.234,6"
    riga = raccogli(resoconti.resoconto_del_tiro(1, 1, mancato(misure=misure(-7.0, 0.0, -0.04, 0.0)), VALORI, riserva=9)).testo.split("\n")[0]
    assert riga == blocchi(f"L:-7,0 V:0,0 P:0,0 I:0,0 G:{G_DEL_MANCATO}", f"{LEGENDA_CON_G}.")
    # Senza un bersaglio vicino non c'è la precisione globale, e nemmeno la sua voce nella legenda.
    assert resoconti.riga_del_sommario(volo(misure=misure(0.0, 25.5, -40.3, 87.2)), "numeri") == blocchi("L:0,0 V:+25,5 P:-40,3 I:+87,2", f"{LEGENDA}.")
    # Il meno è il trattino, che NVDA legge meno e il braille mostra in una cella: niente meno tipografico.
    assert chr(0x2212) not in riga and riga.count("-") == 1


def test_i_valori_e_la_g_nel_primo_blocco():
    """Dalla 0.11.8, per scelta di Gabriele del 20 novembre 2026, i valori del sommario coi numeri non hanno lo spazio dopo i due
    punti, e stanno tutti nel primo blocco, G compresa, come «L:+2,5 V:+7,7 P:+12,1 I:-23,2 G:91,4%»; poi spazi fino al carattere
    40 compreso, e dal 41 la legenda. La riga eccezionale che non ci sta con almeno uno spazio dopo di sé tiene il ripiego di prima:
    G apre il secondo blocco. Gli esempi coi numeri delle docstring di resoconti stanno anche loro nel primo blocco."""
    esempio = resoconti.sommario_del_tiro(mancato(misure=misure(2.5, 7.7, 12.1, -23.2), distanza=0.88), "numeri")
    assert esempio.primo == "L:+2,5 V:+7,7 P:+12,1 I:-23,2 G:91,4%" and esempio.annuncio == esempio.primo
    assert esempio.riga == esempio.primo.ljust(BLOCCO_DEL_BRAILLE) + f"{LEGENDA_CON_G}." and esempio.riga.index("L laterale") == BLOCCO_DEL_BRAILLE
    # Con due cifre prima della virgola in ogni valore la riga ci sta ancora, con uno spazio prima della legenda.
    lunga = resoconti.sommario_del_tiro(mancato(misure=misure(-25.4, -48.6, 76.2, -85.0)), "numeri")
    assert lunga.primo == f"L:-25,4 V:-48,6 P:+76,2 I:-85,0 G:{G_DEL_MANCATO}" and len(lunga.primo) == BLOCCO_DEL_BRAILLE - 1
    assert lunga.riga == f"{lunga.primo} {LEGENDA_CON_G}."
    # Una cifra in più, e il primo blocco resterebbe pieno senza uno spazio dopo di sé: G apre il secondo, come prima della 0.11.8.
    eccezionale = resoconti.sommario_del_tiro(mancato(misure=misure(-25.4, -103.3, 76.2, -85.0)), "numeri")
    assert eccezionale.primo == "L:-25,4 V:-103,3 P:+76,2 I:-85,0"
    assert eccezionale.riga == blocchi(eccezionale.primo, f"G:{G_DEL_MANCATO}. {LEGENDA_CON_G}.")
    assert eccezionale.annuncio == f"{eccezionale.primo} G:{G_DEL_MANCATO}"
    documentati = " ".join(testo or "" for testo in (resoconti.__doc__, resoconti.riga_del_sommario.__doc__, resoconti._sommario_coi_numeri.__doc__))
    esempi = re.findall(r"«(L:[^»]*)»", documentati)
    assert esempi and all(len(esempio) < BLOCCO_DEL_BRAILLE and ": " not in esempio for esempio in esempi), esempi


def test_sommario_a_parole_e_le_soglie(monkeypatch):
    """D21, i livelli di mezzo: soltanto le parole, nell'ordine laterale, verticale, profondità e
    inclinazione; sotto la sua soglia un asse va bene, dalla soglia in su ha la sua direzione. Dalla 0.11.8 la
    precisione globale e il punto di vista vanno sempre sotto le parole, su una seconda riga."""

    def parole(*valori):
        """La prima riga del sommario a parole; la seconda è sempre la precisione globale e il punto di vista (D24, D26)."""
        righe = raccogli(resoconti.resoconto_del_tiro(1, 1, mancato(misure=misure(*valori)), VALORI, riserva=9, palestra="parole")).testo.split("\n")
        assert righe[1] == SOTTO_LE_PAROLE and righe[2].startswith("Tiro 1, livello 1: ")
        assert righe[0] == righe[0].strip() and "Precisione" not in righe[0]
        return righe[0]

    laterale, verticale, profondita, inclinazione = SOMMARIO_SOGLIA_LATERALE, SOMMARIO_SOGLIA_VERTICALE, SOMMARIO_SOGLIA_PROFONDITA, SOMMARIO_SOGLIA_INCLINAZIONE
    assert parole() == PAROLE_SOLE
    testo = raccogli(resoconti.resoconto_del_tiro(1, 1, mancato(misure=misure()), VALORI, riserva=9, palestra="parole")).testo
    assert testo.startswith(f"{PAROLE}\n") and resoconti.riga_del_sommario(mancato(misure=misure()), "parole") == PAROLE
    # Anche senza un bersaglio vicino, cioè senza la precisione globale, il punto di vista sta sotto.
    assert resoconti.riga_del_sommario(volo(misure=misure()), "parole") == f"{PAROLE_SOLE}\n{DAL_BERSAGLIO}"
    assert parole(-2 * laterale, 2 * verticale, -2 * profondita, 2 * inclinazione) == "A sinistra, sopra, corto, in salita."
    giuste = "In linea, all'altezza giusta, alla distanza giusta, in cima."
    assert parole(0.99 * laterale, -0.99 * verticale, 0.99 * profondita, -0.99 * inclinazione) == giuste
    assert parole(0.0, 0.0, 0.0, 0.0) == giuste
    # Sulla soglia l'asse non va più bene.
    assert parole(laterale, -verticale, profondita, -inclinazione) == PAROLE_SOLE
    assert parole(-laterale, verticale, -profondita, inclinazione) == "A sinistra, sopra, corto, in salita."
    assert resoconti.parole_delle_misure(misure()) == ["a destra", "sotto", "lungo", "in discesa"]
    assert resoconti.parole_delle_misure(misure(arriva=False)) == [None, None, "lungo", None]
    assert resoconti.parole_delle_misure(misure(ricade=False)) == ["a destra", "sotto", None, "in discesa"]
    # Le soglie sono quelle di costanti.py: alzate, i numeri di prima vanno bene; a zero, ogni scarto ha la sua direzione.
    monkeypatch.setattr(resoconti, "SOMMARIO_SOGLIA_LATERALE", 20.0)
    monkeypatch.setattr(resoconti, "SOMMARIO_SOGLIA_INCLINAZIONE", 45.0)
    assert parole() == "In linea, sotto, lungo, in cima."
    for nome in ("SOMMARIO_SOGLIA_LATERALE", "SOMMARIO_SOGLIA_VERTICALE", "SOMMARIO_SOGLIA_PROFONDITA", "SOMMARIO_SOGLIA_INCLINAZIONE"):
        monkeypatch.setattr(resoconti, nome, 0.0)
    assert parole(0.01, -0.01, 0.01, -0.01) == PAROLE_SOLE
    assert parole(0.0, 0.0, 0.0, 0.0) == giuste


def test_sommario_con_le_misure_che_mancano():
    """D21: una misura che non esiste non ha un numero. Il tiro che cade prima della distanza del
    bersaglio non ha laterale, verticale né inclinazione; quello che, scendendo, resta sempre sotto il
    centro del bersaglio non ha la profondità. Da D24 al grado dei numeri la misura che manca ha la sua
    iniziale e la parola manca, e la legenda dice perché; a parole si dice come prima. La riga del
    mancato tace le misure che mancano, e l'annuncio resta l'esito più il primo blocco."""
    def sommari(**quali):
        tiro = mancato(misure=misure(-0.6, 3.2, -12.34, 41.0, **quali))
        return resoconti.riga_del_sommario(tiro, "numeri"), resoconti.riga_del_sommario(tiro, "parole")

    sotto = "P manca: scendendo, il tiro resta sempre sotto il centro del bersaglio"
    prima = "L, V e I mancano: il tiro cade prima della distanza del bersaglio"
    globale = f"Precisione globale {G_DEL_MANCATO}. {DAL_BERSAGLIO}"
    assert sommari(ricade=False) == (blocchi(f"L:-0,6 V:+3,2 P:manca I:+41,0 G:{G_DEL_MANCATO}", f"{LEGENDA_CON_G}; {sotto}."),
                                     f"A sinistra, sopra, profondità non misurabile, in salita.\n{globale}")
    # Anche il tiro corto con la profondità di due cifre ci sta, G compresa, con uno spazio prima del secondo blocco.
    assert len(f"L:manca V:manca P:-12,3 I:manca G:{G_DEL_MANCATO}") == BLOCCO_DEL_BRAILLE - 1
    assert sommari(arriva=False) == (blocchi(f"L:manca V:manca P:-12,3 I:manca G:{G_DEL_MANCATO}", f"{LEGENDA_CON_G}; {prima}."),
                                     f"Cade prima della sua distanza, corto.\n{globale}")
    assert sommari(arriva=False, ricade=False) == (blocchi(f"L:manca V:manca P:manca I:manca G:{G_DEL_MANCATO}", f"{LEGENDA_CON_G}; {prima}; {sotto}."),
                                                   f"Cade prima della sua distanza, profondità non misurabile.\n{globale}")
    corto = raccogli(resoconti.resoconto_del_tiro(3, 1, mancato(misure=misure(profondita=-12.34, arriva=False)), VALORI, riserva=8))
    righe = corto.testo.split("\n")
    assert righe[0] == sommari(arriva=False)[0]
    assert righe[-1] == "Il bersaglio più vicino è stato mancato di 2,3 metri: rispetto al suo centro il tiro è passato corto di 12,3 metri."
    assert corto.annuncio == f"A vuoto, riserva 8. L:manca V:manca P:-12,3 I:manca G:{G_DEL_MANCATO}"
    niente = raccogli(resoconti.resoconto_del_tiro(3, 1, mancato(misure=misure(arriva=False, ricade=False)), VALORI, riserva=8))
    assert niente.testo.split("\n")[-1] == "Il bersaglio più vicino è stato mancato di 2,3 metri."
    assert niente.annuncio == f"A vuoto, riserva 8. L:manca V:manca P:manca I:manca G:{G_DEL_MANCATO}"
    # Al grado delle parole, con l'aiuto di D6 chiesto a mano, lo scarto principale resta nell'esito,
    # perché le parole non ne dicono la misura.
    aiutato = raccogli(resoconti.resoconto_del_tiro(3, 1, mancato(misure=misure(-0.6, 3.2, -12.34, 41.0, ricade=False)), VALORI, riserva=8, palestra="parole",
                                                    aiuto=True))
    assert aiutato.annuncio == (f"A vuoto, riserva 8, sopra di 3,2 metri. A sinistra, sopra, profondità non misurabile, in salita. "
                                f"Precisione globale {G_DEL_MANCATO}.")


def test_la_palestra_ai_confini_dei_livelli():
    """D21: i numeri fino a SOMMARIO_NUMERI_FINO_AL_LIVELLO, le parole fino a
    SOMMARIO_PAROLE_FINO_AL_LIVELLO, poi niente sommario; palestra lo sceglie a mano."""
    numeri, parole = SOMMARIO_NUMERI_FINO_AL_LIVELLO, SOMMARIO_PAROLE_FINO_AL_LIVELLO
    assert 1 <= numeri < parole
    assert [resoconti.grado_della_palestra(n) for n in (1, numeri, numeri + 1, parole, parole + 1, 100)] == ["numeri", "numeri", "parole", "parole", "niente", "niente"]
    tiro = mancato(misure=misure())

    def resoconto(livello, **altro):
        return raccogli(resoconti.resoconto_del_tiro(5, livello, tiro, VALORI, riserva=8, **altro))

    assert resoconto(numeri).testo.split("\n")[0] == NUMERI
    for livello in (numeri + 1, parole):
        messaggio = resoconto(livello)
        assert messaggio.testo.split("\n")[:3] == [PAROLE_SOLE, SOTTO_LE_PAROLE, f"Tiro 5, livello {livello}: a vuoto; nella riserva restano 8 colpi."]
        assert messaggio.annuncio.startswith("A vuoto, riserva 8") and messaggio.annuncio.endswith(f". {PAROLE_ANNUNCIO}")
    oltre = resoconto(parole + 1)
    # Senza sommario la precisione globale di D26 sta nella riga dell'esito, e l'annuncio è l'esito più lei.
    assert oltre.testo.split("\n")[0] == f"Tiro 5, livello {parole + 1}: a vuoto, precisione globale {G_DEL_MANCATO}; nella riserva restano 8 colpi."
    assert oltre.annuncio == f"A vuoto, riserva 8. Precisione globale {G_DEL_MANCATO}." and "L:" not in oltre.testo and "Dal bersaglio" not in oltre.testo
    assert oltre.annuncio == resoconti.resoconto_del_tiro(5, parole + 1, mancato(), VALORI, riserva=8).annuncio
    # palestra vince sul livello, in tutti e due i versi.
    assert resoconto(parole + 10, palestra="numeri").testo.split("\n")[0] == NUMERI
    assert resoconto(1, palestra="niente").testo.split("\n")[0].startswith("Tiro 5, livello 1: ")
    with pytest.raises(ValueError):
        resoconti.riga_del_sommario(tiro, "tanti")
    # Senza un bersaglio vivo vicino il volo non ha misure, e nessun grado ha il sommario.
    assert all(resoconti.riga_del_sommario(volo(), grado) == "" for grado in resoconti.GRADI_DELLA_PALESTRA)
    assert resoconto(1).testo.split("\n")[0] == NUMERI and raccogli(resoconti.resoconto_del_tiro(5, 1, volo(), VALORI, riserva=8)).testo.startswith("Tiro 5, ")


def test_livello_e_inizio(monkeypatch):
    monkeypatch.setattr(resoconti, "quanti_bersagli", lambda n: 1 if n < 8 else 2)
    monkeypatch.setattr(resoconti, "raggio_del_livello", lambda n: 3.07 if n > 1 else 5.0)
    monkeypatch.setattr(resoconti, "velocita_del_livello", lambda n: 1.5 if n >= 4 else 0.0)
    livello = raccogli(resoconti.livello_nuovo(4, 10, 17))
    assert livello.tipo == "livello"
    assert livello.testo.split("\n") == ["Livello 4! Il livello 3 è superato.", "10 colpi in più nella riserva, che ora ne ha 17.",
                                         "In volo: 1 bersaglio mobile a 1,5 metri al secondo, raggio 3,1 metri."]
    assert livello.annuncio == "Livello 4, riserva 17."
    assert resoconti.bersagli_del_livello(8) == "2 bersagli mobili a 1,5 metri al secondo, raggio 3,1 metri"
    assert resoconti.bersagli_del_livello(2) == "1 bersaglio fermo, raggio 3,1 metri"
    inizio = raccogli(resoconti.inizio_partita(1, 10))
    assert inizio.tipo == "info"
    assert inizio.testo.split("\n")[:2] == ["Partita nuova, livello 1: nella riserva 10 colpi.", "In volo: 1 bersaglio fermo, raggio 5 metri."]
    assert inizio.annuncio == "Partita nuova: Spazio per mirare."
    assert raccogli(resoconti.livello_nuovo(2, 1, 1)).testo.split("\n")[1] == "1 colpo in più nella riserva, che ora ne ha 1."


def test_bonus(monkeypatch):
    monkeypatch.setattr(resoconti, "durata_del_bonus", lambda n: 14.0)
    monkeypatch.setattr(resoconti, "MOLTIPLICATORE_BONUS", 5)
    monkeypatch.setattr(resoconti, "COLPI_BONUS", 5)
    da_sinistra = bersaglio(raggio=2.9, forma="deriva", bonus=True, velocita=(3.9, 0.0, 0.0))
    apparso = raccogli(resoconti.bonus_apparso(da_sinistra, 3))
    assert apparso.tipo == "bonus"
    assert apparso.testo.split("\n") == ["È apparso un bersaglio bonus, che entra da sinistra: forma deriva, raggio 2,9 metri.",
                                         "Attraversa lo spazio dei bersagli in 14 secondi, poi esce; colpito vale 5 volte i punti di un bersaglio di questo livello "
                                         "e dà 5 colpi in più."]
    assert apparso.annuncio == "Bonus da sinistra."
    da_destra = bersaglio(raggio=2.9, bonus=True, velocita=(-3.9, 0.0, 0.0))
    assert raccogli(resoconti.bonus_apparso(da_destra, 3)).annuncio == "Bonus da destra."
    monkeypatch.setattr(resoconti, "MOLTIPLICATORE_BONUS", 1)
    monkeypatch.setattr(resoconti, "COLPI_BONUS", 1)
    assert "vale 1 volta i punti di un bersaglio di questo livello e dà 1 colpo in più." in raccogli(resoconti.bonus_apparso(da_destra, 3)).testo
    uscito = raccogli(resoconti.bonus_uscito())
    assert (uscito.tipo, uscito.annuncio) == ("bonus", "Bonus uscito.")


def test_primati():
    assert resoconti.primati_battuti([]) is None
    uno = raccogli(resoconti.primati_battuti(["precisione"], {"precisione": 0.98}))
    assert uno.tipo == "info" and uno.testo == "Nuovo primato: il centro più preciso, 98%." and uno.annuncio == uno.testo
    due = raccogli(resoconti.primati_battuti(["serie", "vicino"], {"serie": 5, "vicino": 0.05}))
    assert due.testo.split("\n") == ["Nuovo primato: la serie più lunga di centri consecutivi, 5.",
                                     "Nuovo primato: il tiro passato più vicino al centro di un bersaglio, 0,05 metri dal centro."]
    assert due.annuncio == "2 nuovi primati." and due.dati == {"primati": ["serie", "vicino"]}
    assert resoconti.riga_del_primato("punteggio", 1) == "Nuovo primato: il punteggio più alto in una partita, 1 punto."
    assert resoconti.riga_del_primato("partita_piu_lunga", 252) == "Nuovo primato: la partita più lunga, 4 minuti e 12 secondi."
    assert resoconti.riga_del_primato("partita_piu_lunga") == "Nuovo primato: la partita più lunga."
    assert resoconti.riga_del_primato("tiri_lunghi", 3.5) == "Nuovo primato: tiri lunghi, 3,5."
    # I due primati della precisione globale (D26), col decimale.
    assert resoconti.riga_del_primato("precisione_globale", 0.9734) == "Nuovo primato: la precisione globale migliore in un tiro, 97,3%."
    assert resoconti.riga_del_primato("precisione_globale_media", 0.8) == "Nuovo primato: la precisione globale media migliore in una partita, 80,0%."
    assert resoconti.primato_scritto("precisione_globale", 0.97341) == resoconti.primato_scritto("precisione_globale", 0.97349)


def test_primato_scritto():
    """Il valore come lo scrive il messaggio: due precisioni che si arrotondano allo stesso punto
    percentuale, o due distanze allo stesso centimetro, si scrivono uguali, e la partita non annuncia
    un primato che a parole non cambia."""
    assert resoconti.primato_scritto("precisione", 0.9951) == resoconti.primato_scritto("precisione", 0.9999) == "100%"
    assert resoconti.primato_scritto("precisione", 0.9651) == resoconti.primato_scritto("precisione", 0.9698) == "97%"
    assert resoconti.primato_scritto("precisione", 0.962) != resoconti.primato_scritto("precisione", 0.972)
    assert resoconti.primato_scritto("vicino", 0.012) == resoconti.primato_scritto("vicino", 0.011) == "0,01 metri dal centro"
    assert resoconti.primato_scritto("mancato", 0.2) == "0,20 metri dal centro"
    assert resoconti.primato_scritto("punteggio", 1230) == "1.230 punti"
    assert resoconti.primato_scritto("partita_piu_lunga", 252.0) == resoconti.primato_scritto("partita_piu_lunga", 252.3)
    assert resoconti.primato_scritto("tiri_lunghi", 3.5) == "3,5"


def test_fine_e_abbandono():
    riepilogo = {"punteggio": 1230, "livello": 3, "tiri": 20, "centri": 12, "durata_s": 252.0}
    fine = raccogli(resoconti.fine_partita(riepilogo, ["punteggio"], {"punteggio": 1230}))
    assert fine.tipo == "fine" and fine.dati == riepilogo
    assert fine.testo.split("\n") == ["Partita finita: 1.230 punti, livello 3, 12 centri su 20 tiri, 60%, in 4 minuti e 12 secondi.",
                                      "Nuovo primato: il punteggio più alto in una partita, 1.230 punti."]
    assert fine.annuncio == "Partita finita, 1.230 punti."
    singolare = raccogli(resoconti.fine_partita({"punteggio": 1, "livello": 1, "tiri": 1, "centri": 1, "durata_s": 1.0}))
    assert singolare.testo == "Partita finita: 1 punto, livello 1, 1 centro su 1 tiro, 100%, in 1 secondo."
    nessuno = raccogli(resoconti.fine_partita({"punteggio": 0, "livello": 1, "tiri": 0, "centri": 0, "durata_s": 0.0}))
    assert nessuno.testo == "Partita finita: 0 punti, livello 1, nessun tiro, in 0 secondi."
    abbandonata = raccogli(resoconti.partita_abbandonata(riepilogo))
    assert abbandonata.tipo == "info" and abbandonata.annuncio == "Partita abbandonata."
    assert abbandonata.testo.startswith("Partita abbandonata: 1.230 punti, livello 3,")


def test_avvisi_e_pausa():
    for messaggio in (resoconti.ultimo_colpo(), resoconti.in_pausa(), resoconti.in_pausa(nella_mira=True), resoconti.ripresa(), resoconti.nessuna_partita()):
        raccogli(messaggio)
        assert messaggio.testo and messaggio.annuncio and "\n" not in messaggio.testo
    assert resoconti.ultimo_colpo().tipo == "avviso" and resoconti.ultimo_colpo().annuncio == "Ultimo colpo."
    assert "Ctrl+P" in resoconti.in_pausa().testo
    # Nella mira la pausa sospende anche la barra (D14): il messaggio dice che ricomincia da capo, coi valori già presi.
    nella_mira = resoconti.in_pausa(nella_mira=True)
    assert nella_mira.tipo == "info" and nella_mira.annuncio == "Pausa." and "Ctrl+P" in nella_mira.testo
    assert "ricomincia da capo" in nella_mira.testo and "valori già presi restano" in nella_mira.testo
    assert resoconti.nessuna_partita().annuncio == "Nessuna partita in corso."


def test_righe_della_barra():
    assert resoconti.righe_della_barra("pronta") == ["Nessuna partita: Spazio per cominciare"]
    assert resoconti.righe_della_barra("ascolto", livello=3, punti=1230, riserva=8, centri_del_livello=4, bersagli=2, bonus=True) == [
        "Livello 3, 1.230 punti", f"Riserva 8, centri 4 su {CENTRI_PER_LIVELLO}", "Bersagli in volo: 2 e il bonus", "Ascolto: Spazio per mirare"]
    assert resoconti.righe_della_barra("ascolto", livello=1, punti=1, bersagli=1)[0] == "Livello 1, 1 punto"
    # Nella mira la riga dice che Esc la annulla (D25), nei 40 caratteri del display braille.
    assert resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="alzo", sottofase="conto")[-1] == "Mira: alzo, bip; Esc annulla"
    assert resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="potenza", sottofase="scivolata")[-1] == "Mira: potenza, scivolata; Esc annulla"
    assert resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="direzione", sottofase="pausa")[-1] == "Mira: direzione, pausa; Esc annulla"
    assert resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="direzione")[-1] == "Mira: direzione; Esc annulla"
    assert len(resoconti.riga_della_fase("mira", parametro="direzione", sottofase="scivolata")) <= BLOCCO_DEL_BRAILLE
    # Dal secondo passaggio della barra il passaggio va nel secondo blocco, dal carattere 41.
    assert resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="direzione", sottofase="scivolata", passaggio=1)[-1] == (
        "Mira: direzione, scivolata; Esc annulla")
    riga = resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="direzione", sottofase="scivolata", passaggio=PASSAGGI_DELLA_BARRA)[-1]
    assert riga == blocchi("Mira: direzione, scivolata; Esc annulla", f"passaggio {PASSAGGI_DELLA_BARRA} di {PASSAGGI_DELLA_BARRA}")
    assert resoconti.righe_della_barra("volo", livello=1, bersagli=1)[-1] == "Volo"
    assert resoconti.righe_della_barra("volo", in_pausa=True, livello=1, bersagli=1)[-1] == "In pausa: Ctrl+P per riprendere"
    assert resoconti.righe_della_barra("finita", livello=2, punti=500, riserva=0, centri_del_livello=3) == [
        "Livello 2, 500 punti", f"Riserva 0, centri 3 su {CENTRI_PER_LIVELLO}", "Partita finita: Spazio per ricominciare"]


def test_la_barra_nell_attesa_della_palestra():
    """D22: nell'attesa la barra dice, prima che la partita si fermi, che si ascolta l'esito del tiro;
    poi che cosa farà lo spazio. La pausa, anche dentro l'attesa, dice la sua. Righe da display braille."""
    righe = resoconti.righe_della_barra("attesa", livello=2, punti=230, riserva=9, centri_del_livello=1, bersagli=0)
    assert righe == ["Livello 2, 230 punti", f"Riserva 9, centri 1 su {CENTRI_PER_LIVELLO}", "Bersagli in volo: 0", "Palestra: ascolta l'esito del tiro"]
    attese = {(False, 1): "Palestra: ascolta l'esito del tiro", (True, 0): "Palestra: Spazio per tornare all'ascolto",
              (True, 1): "Palestra: Spazio per il bersaglio nuovo", (True, 2): "Palestra: Spazio per 2 bersagli nuovi",
              (True, 12): "Palestra: Spazio per 12 bersagli nuovi"}
    for (ferma, nasceranno), riga in attese.items():
        assert resoconti.righe_della_barra("attesa", bersagli=1, attesa_ferma=ferma, nasceranno=nasceranno)[-1] == riga
        assert resoconti.riga_dell_attesa(ferma, nasceranno) == riga and len(riga) <= 40
    assert resoconti.righe_della_barra("attesa", in_pausa=True, attesa_ferma=True, nasceranno=1)[-1] == "In pausa: Ctrl+P per riprendere"


def test_i_messaggi_della_palestra():
    """D22: l'ultimo messaggio di un tiro nella palestra dice che cosa farà lo spazio, senza annuncio; il
    rifiuto di Ctrl+P fuori dalla palestra dice fino a che livello la pausa c'è."""
    for nasceranno, testo in ((0, "Palestra: la partita aspetta lo Spazio, che ti riporta all'ascolto."),
                              (1, "Palestra: la partita aspetta lo Spazio, che chiama il bersaglio nuovo."),
                              (3, "Palestra: la partita aspetta lo Spazio, che chiama 3 bersagli nuovi.")):
        messaggio = raccogli(resoconti.attesa_della_palestra(nasceranno))
        assert (messaggio.tipo, messaggio.testo, messaggio.annuncio, messaggio.dati) == ("info", testo, "", {"nasceranno": nasceranno})
    rifiuto = raccogli(resoconti.pausa_solo_in_palestra(3))
    assert rifiuto.tipo == "avviso" and rifiuto.annuncio == "Pausa solo in palestra." and rifiuto.dati == {"fino_al_livello": 3}
    assert rifiuto.testo == "La pausa con Ctrl+P c'è soltanto nei livelli della palestra, fino al livello 3."
    assert resoconti.pausa_solo_in_palestra(1000).testo.endswith("fino al livello 1.000.")


def test_in_blocchi():
    """D24: il secondo blocco comincia dal carattere seguente BLOCCO_DEL_BRAILLE; un primo blocco che non ci sta, con almeno uno
    spazio, si allunga, e il secondo lo segue dopo uno spazio; senza secondo blocco resta il primo."""
    assert resoconti.in_blocchi("abc", "def") == "abc" + " " * (BLOCCO_DEL_BRAILLE - 3) + "def"
    assert resoconti.in_blocchi("abc", "def").index("def") == BLOCCO_DEL_BRAILLE
    pieno = "x" * BLOCCO_DEL_BRAILLE
    assert resoconti.in_blocchi(pieno, "def") == f"{pieno} def"
    assert resoconti.in_blocchi("x" * (BLOCCO_DEL_BRAILLE - 1), "def") == "x" * (BLOCCO_DEL_BRAILLE - 1) + " def"
    assert resoconti.in_blocchi("abc", "") == "abc" and resoconti.in_blocchi("abc", "d", larghezza=5) == "abc  d"


def test_il_proiettile_perduto_e_la_mira_annullata():
    """D25: il resoconto del proiettile perduto dice la barra e i passaggi, conta come un tiro a vuoto e non ha precisione globale;
    la mira annullata con Esc dice che il proiettile resta nella riserva."""
    perduto = raccogli(resoconti.proiettile_perduto(6, 2, "alzo", 5, 7, totale=430))
    assert perduto.tipo == "resoconto" and perduto.testo.split("\n") == [
        "Tiro 6, livello 2: proiettile perduto; nella riserva restano 7 colpi.",
        "La barra dell'alzo è passata 5 volte senza lo Spazio: il colpo non è partito, e il proiettile conta come un tiro a vuoto."]
    assert perduto.annuncio == "Proiettile perduto, riserva 7."
    assert perduto.dati == {"tiro": 6, "livello": 2, "colpito": False, "bonus": False, "punti": 0, "totale": 430, "riserva": 7, "precisione": None,
                            "precisione_globale": None, "perduto": True, "parametro": "alzo", "passaggi": 5}
    ultimo = raccogli(resoconti.proiettile_perduto(1, 1, "direzione", 1, 0))
    assert ultimo.testo.split("\n") == ["Tiro 1, livello 1: proiettile perduto; la riserva è vuota.",
                                        "La barra della direzione è passata 1 volta senza lo Spazio: il colpo non è partito, e il proiettile conta come un "
                                        "tiro a vuoto."]
    assert ultimo.annuncio == "Proiettile perduto, riserva vuota."
    assert raccogli(resoconti.proiettile_perduto(2, 1, "potenza", 5, 1)).testo.split("\n")[0] == "Tiro 2, livello 1: proiettile perduto; nella riserva resta 1 colpo."
    annullata = raccogli(resoconti.mira_annullata(10))
    assert (annullata.tipo, annullata.annuncio, annullata.dati) == ("info", "Mira annullata.", {"riserva": 10})
    assert annullata.testo == "Mira annullata con Esc: il proiettile resta nella riserva, che ha 10 colpi, e i bersagli tornano a cantare. Spazio per mirare di nuovo."
    in_pausa = raccogli(resoconti.mira_annullata(1, in_pausa=True))
    assert in_pausa.testo == "Mira annullata con Esc: il proiettile resta nella riserva, che ha 1 colpo, e la partita resta in pausa. Ctrl+P per riprendere."


def test_messaggio_di_tipo_sconosciuto():
    with pytest.raises(ValueError):
        Messaggio("strano", "testo")
    assert Messaggio("info", "testo").dati == {}


def _campionario():
    """Un messaggio di ogni genere, con numeri piccoli e grandi, per la prova dei separatori."""
    riepilogo = {"punteggio": 1234567, "livello": 12, "tiri": 300, "centri": 120, "durata_s": 4000.0}
    bonus = bersaglio(raggio=0.4, bonus=True, velocita=(-9.0, 0.0, 0.0))
    messaggi = [
        resoconti.resoconto_del_tiro(1, 1, volo(bersaglio()), VALORI, punti=200, totale=200, riserva=10),
        resoconti.resoconto_del_tiro(2, 20, volo(bonus), VALORI, punti=9999, totale=1234567, riserva=3),
        resoconti.resoconto_del_tiro(3, 1, mancato(), VALORI, riserva=9),
        resoconti.resoconto_del_tiro(4, 1, mancato(scarto=(-30.0, 12.0, -0.01)), VALORI, riserva=0),
        resoconti.resoconto_del_tiro(5, 1, mancato(misure=misure(-1234.5, 0.0, 0.04, -89.9)), VALORI, riserva=7),
        resoconti.resoconto_del_tiro(6, SOMMARIO_PAROLE_FINO_AL_LIVELLO, volo(bersaglio(), misure=misure(0.0, 0.0, 0.0, 0.0)), VALORI, punti=200, totale=400, riserva=7),
        resoconti.resoconto_del_tiro(7, SOMMARIO_PAROLE_FINO_AL_LIVELLO + 1, mancato(misure=misure()), VALORI, riserva=6),
        resoconti.resoconto_del_tiro(8, 1, mancato(misure=misure(arriva=False, ricade=False)), VALORI, riserva=5),
        resoconti.resoconto_del_tiro(9, SOMMARIO_PAROLE_FINO_AL_LIVELLO, mancato(misure=misure(arriva=False)), VALORI, riserva=4),
        resoconti.inizio_partita(1, 10), resoconti.livello_nuovo(16, 10, 40), resoconti.bonus_apparso(bonus, 20), resoconti.bonus_uscito(),
        resoconti.primati_battuti(["precisione", "vicino", "mancato", "serie", "punti_in_un_tiro"], {"precisione": 1.0, "vicino": 0.0, "mancato": 0.2,
                                                                                                     "serie": 12, "punti_in_un_tiro": 9999}),
        resoconti.ultimo_colpo(), resoconti.fine_partita(riepilogo, ["punteggio", "livello", "centri_in_una_partita", "partita_piu_lunga"],
                                                         {"punteggio": 1234567, "livello": 12, "centri_in_una_partita": 120, "partita_piu_lunga": 4000.0}),
        resoconti.partita_abbandonata(riepilogo), resoconti.in_pausa(), resoconti.ripresa(), resoconti.in_pausa(nella_mira=True), resoconti.nessuna_partita(),
        resoconti.pausa_solo_in_palestra(3), resoconti.attesa_della_palestra(0), resoconti.attesa_della_palestra(1), resoconti.attesa_della_palestra(4),
        resoconti.proiettile_perduto(3, 4, "potenza", PASSAGGI_DELLA_BARRA, 6), resoconti.proiettile_perduto(3, 4, "direzione", 1, 0),
        resoconti.mira_annullata(12), resoconti.mira_annullata(1, in_pausa=True),
        resoconti.resoconto_del_tiro(10, SOMMARIO_PAROLE_FINO_AL_LIVELLO + 1, volo(bersaglio()), VALORI, punti=200, totale=400, riserva=7),
    ]
    testi_ = [*TESTI]
    for messaggio in messaggi:
        testi_.extend(testo for testo in (messaggio.testo, messaggio.annuncio) if testo)
    for fase in ("pronta", "ascolto", "mira", "volo", "attesa", "finita"):
        testi_.extend(resoconti.righe_della_barra(fase, livello=12, punti=1234567, riserva=40, centri_del_livello=9, bersagli=4, bonus=True, parametro="potenza",
                                                  sottofase="scivolata"))
    for ferma, nasceranno in ((False, 0), (True, 0), (True, 1), (True, 4)):
        testi_.append(resoconti.riga_dell_attesa(ferma, nasceranno))
    for passaggio in range(1, PASSAGGI_DELLA_BARRA + 1):
        testi_.append(resoconti.riga_della_fase("mira", parametro="potenza", sottofase="scivolata", passaggio=passaggio))
    return testi_


def test_nessun_separatore_grafico():
    """Ogni testo, dai messaggi di ogni genere e da quelli delle prove già girate: niente separatori
    grafici, niente righe vuote, niente spazi in testa o in coda, e in ogni riga almeno una parola."""
    for testo in _campionario():
        assert not SEPARATORE.search(testo), testo
        assert "\n\n" not in testo and not testo.startswith("\n") and not testo.endswith("\n")
        for riga in testo.split("\n"):
            assert riga.strip() == riga and re.search(r"\w", riga), riga
