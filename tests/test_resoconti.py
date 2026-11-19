"""
Le prove di resoconti.py: le frasi dei tiri, dei livelli, dei bonus, dei primati e della fine, i segni
delle direzioni e degli scarti, i singolari e i plurali, le righe della barra, e nessun separatore
grafico.
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
from costanti import AIUTO_MANCATO_FINO_AL_LIVELLO, CENTRI_PER_LIVELLO, COLPI_BONUS
from fisica import Traiettoria
from resoconti import Messaggio
from volo import Volo

# Le righe fatte di soli simboli ripetuti, o tre simboli uguali di fila: separatori grafici.
SEPARATORE = re.compile(r"([-=_*~#+.])\1{2,}")
TESTI = []


def raccogli(messaggio):
    """Tiene da parte testo e annuncio di ogni messaggio, per la prova dei separatori."""
    TESTI.extend((messaggio.testo, messaggio.annuncio))
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


VALORI = (0.64, 0.45, 0.5)


def test_resoconto_di_un_centro():
    messaggio = raccogli(resoconti.resoconto_del_tiro(12, 3, volo(bersaglio()), VALORI, punti=230, totale=1230, riserva=8))
    assert messaggio.tipo == "resoconto"
    assert messaggio.testo.split("\n") == [
        "Tiro 12, livello 3: colpito un bersaglio, forma dente di sega, raggio 3,6 metri, fermo; 230 punti, precisione 64%, totale 1.230 punti.",
        "Mira: direzione 12,4 gradi a destra, 64% della scala; alzo 38,2 gradi, 45%; potenza 27,5 metri al secondo, 50%.",
        "Traiettoria: altezza massima 23,4 metri; impatto a 56,2 metri dal cannone, 3,1 metri a destra del centro, dopo 2,8 secondi di volo.",
        "Il bersaglio stava a 48,3 metri dal cannone, a 22,1 metri d'altezza, 5,2 metri a sinistra del centro.",
    ]
    assert messaggio.annuncio == "Colpito, 230 punti."
    assert messaggio.dati == {"tiro": 12, "livello": 3, "colpito": True, "bonus": False, "punti": 230, "totale": 1230, "riserva": 8, "precisione": 0.64}


def test_resoconto_di_un_bonus_e_di_un_mobile():
    bonus = bersaglio(raggio=2.9, forma="seno", bonus=True, velocita=(3.9, 0.0, 0.0))
    messaggio = raccogli(resoconti.resoconto_del_tiro(7, 4, volo(bonus, precisione=0.5), VALORI, punti=1150, totale=2380, riserva=12))
    assert messaggio.testo.split("\n")[0] == (f"Tiro 7, livello 4: colpito il bonus, forma seno, raggio 2,9 metri; 1.150 punti, precisione 50%, "
                                              f"{COLPI_BONUS} colpi in più, totale 2.380 punti.")
    assert messaggio.annuncio == "Colpito il bonus, 1.150 punti."
    assert messaggio.dati["bonus"]
    uno = raccogli(resoconti.resoconto_del_tiro(7, 4, volo(bonus), VALORI, punti=1, totale=1, riserva=1, colpi_bonus=1))
    assert "; 1 punto, precisione 64%, 1 colpo in più, totale 1 punto." in uno.testo
    mobile = bersaglio(raggio=2.0, forma="triangolo", velocita=(2.3, 0.0, 0.0))
    assert resoconti.descrizione_del_bersaglio(mobile) == "forma triangolo, raggio 2 metri, mobile a 2,3 metri al secondo"
    assert resoconti.descrizione_del_bersaglio(bersaglio(raggio=1.0, forma="quadra")) == "forma onda quadra, raggio 1 metro, fermo"


def test_resoconto_di_un_tiro_a_vuoto_con_l_aiuto():
    messaggio = raccogli(resoconti.resoconto_del_tiro(4, 1, mancato(), VALORI, riserva=8))
    assert messaggio.testo.split("\n") == [
        "Tiro 4, livello 1: a vuoto; nella riserva restano 8 colpi.",
        "Mira: direzione 12,4 gradi a destra, 64% della scala; alzo 38,2 gradi, 45%; potenza 27,5 metri al secondo, 50%.",
        "Traiettoria: altezza massima 23,4 metri; a terra a 56,2 metri dal cannone, 3,1 metri a destra del centro, dopo 2,8 secondi di volo.",
        "Il bersaglio più vicino è stato mancato di 2,3 metri: rispetto al suo centro il tiro è passato lungo di 1,2 metri, a sinistra di 0,8 metri e sopra di 1,7 metri.",
    ]
    assert messaggio.annuncio == "A vuoto, riserva 8, sopra di 1,7 metri."
    assert not messaggio.dati["colpito"] and messaggio.dati["riserva"] == 8


def test_segni_opposti_e_singolari():
    messaggio = raccogli(resoconti.resoconto_del_tiro(1, 1, mancato(scarto=(-3.4, 0.5, -0.2), distanza=1.0, azimut=-7.0, laterale_finale=-1.0), VALORI, riserva=1))
    righe = messaggio.testo.split("\n")
    assert righe[0] == "Tiro 1, livello 1: a vuoto; nella riserva resta 1 colpo."
    assert righe[1].startswith("Mira: direzione 7 gradi a sinistra, 64% della scala;")
    assert ", 1 metro a sinistra del centro," in righe[2]
    assert righe[3] == ("Il bersaglio più vicino è stato mancato di 1 metro: rispetto al suo centro il tiro è passato corto di 3,4 metri, a destra di 0,5 metri "
                        "e sotto di 0,2 metri.")
    assert messaggio.annuncio == "A vuoto, riserva 1, corto di 3,4 metri."
    assert resoconti.scarti((0.04, -0.03, 2.0)) == ["sopra di 2 metri"]
    assert resoconti.scarto_principale((0.1, -4.0, 2.0)) == "a sinistra di 4 metri"
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
    assert oltre.annuncio == "A vuoto, riserva 8."
    forzato = raccogli(resoconti.resoconto_del_tiro(30, AIUTO_MANCATO_FINO_AL_LIVELLO + 1, mancato(), VALORI, riserva=8, aiuto=True))
    assert len(forzato.testo.split("\n")) == 4
    vuota = raccogli(resoconti.resoconto_del_tiro(10, 1, mancato(), VALORI, riserva=0))
    assert vuota.testo.split("\n")[0] == "Tiro 10, livello 1: a vuoto; la riserva è vuota."
    assert vuota.annuncio == "A vuoto, riserva vuota."
    # Un tiro senza bersagli vivi vicini non ha la riga del mancato.
    solo = raccogli(resoconti.resoconto_del_tiro(2, 1, volo(), VALORI, riserva=5))
    assert len(solo.testo.split("\n")) == 3 and solo.annuncio == "A vuoto, riserva 5."


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
                                         "Attraversa lo spazio dei bersagli in 14 secondi, poi esce; colpito vale 5 volte i punti e dà 5 colpi in più."]
    assert apparso.annuncio == "Bonus da sinistra."
    da_destra = bersaglio(raggio=2.9, bonus=True, velocita=(-3.9, 0.0, 0.0))
    assert raccogli(resoconti.bonus_apparso(da_destra, 3)).annuncio == "Bonus da destra."
    monkeypatch.setattr(resoconti, "MOLTIPLICATORE_BONUS", 1)
    monkeypatch.setattr(resoconti, "COLPI_BONUS", 1)
    assert "vale 1 volta i punti e dà 1 colpo in più." in raccogli(resoconti.bonus_apparso(da_destra, 3)).testo
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
    for messaggio in (resoconti.ultimo_colpo(), resoconti.in_pausa(), resoconti.ripresa(), resoconti.pausa_nella_mira(), resoconti.nessuna_partita()):
        raccogli(messaggio)
        assert messaggio.testo and messaggio.annuncio and "\n" not in messaggio.testo
    assert resoconti.ultimo_colpo().tipo == "avviso" and resoconti.ultimo_colpo().annuncio == "Ultimo colpo."
    assert "Ctrl+P" in resoconti.in_pausa().testo
    assert resoconti.pausa_nella_mira().tipo == "avviso"


def test_righe_della_barra():
    assert resoconti.righe_della_barra("pronta") == ["Nessuna partita: Spazio per cominciare"]
    assert resoconti.righe_della_barra("ascolto", livello=3, punti=1230, riserva=8, centri_del_livello=4, bersagli=2, bonus=True) == [
        "Livello 3, 1.230 punti", f"Riserva 8, centri 4 su {CENTRI_PER_LIVELLO}", "Bersagli in volo: 2 e il bonus", "Ascolto: Spazio per mirare"]
    assert resoconti.righe_della_barra("ascolto", livello=1, punti=1, bersagli=1)[0] == "Livello 1, 1 punto"
    assert resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="alzo", sottofase="conto")[-1] == "Mira: alzo, bip"
    assert resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="potenza", sottofase="scivolata")[-1] == "Mira: potenza, scivolata"
    assert resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="direzione", sottofase="pausa")[-1] == "Mira: direzione, pausa"
    assert resoconti.righe_della_barra("mira", livello=1, bersagli=1, parametro="direzione")[-1] == "Mira: direzione"
    assert resoconti.righe_della_barra("volo", livello=1, bersagli=1)[-1] == "Volo"
    assert resoconti.righe_della_barra("volo", in_pausa=True, livello=1, bersagli=1)[-1] == "In pausa: Ctrl+P per riprendere"
    assert resoconti.righe_della_barra("finita", livello=2, punti=500, riserva=0, centri_del_livello=3) == [
        "Livello 2, 500 punti", f"Riserva 0, centri 3 su {CENTRI_PER_LIVELLO}", "Partita finita: Spazio per ricominciare"]


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
        resoconti.inizio_partita(1, 10), resoconti.livello_nuovo(16, 10, 40), resoconti.bonus_apparso(bonus, 20), resoconti.bonus_uscito(),
        resoconti.primati_battuti(["precisione", "vicino", "mancato", "serie", "punti_in_un_tiro"], {"precisione": 1.0, "vicino": 0.0, "mancato": 0.2,
                                                                                                     "serie": 12, "punti_in_un_tiro": 9999}),
        resoconti.ultimo_colpo(), resoconti.fine_partita(riepilogo, ["punteggio", "livello", "centri_in_una_partita", "partita_piu_lunga"],
                                                         {"punteggio": 1234567, "livello": 12, "centri_in_una_partita": 120, "partita_piu_lunga": 4000.0}),
        resoconti.partita_abbandonata(riepilogo), resoconti.in_pausa(), resoconti.ripresa(), resoconti.pausa_nella_mira(), resoconti.nessuna_partita(),
    ]
    testi_ = [*TESTI]
    for messaggio in messaggi:
        testi_.extend((messaggio.testo, messaggio.annuncio))
    for fase in ("pronta", "ascolto", "mira", "volo", "finita"):
        testi_.extend(resoconti.righe_della_barra(fase, livello=12, punti=1234567, riserva=40, centri_del_livello=9, bersagli=4, bonus=True, parametro="potenza",
                                                  sottofase="scivolata"))
    return testi_


def test_nessun_separatore_grafico():
    """Ogni testo, dai messaggi di ogni genere e da quelli delle prove già girate: niente separatori
    grafici, niente righe vuote, niente spazi in testa o in coda, e in ogni riga almeno una parola."""
    for testo in _campionario():
        assert not SEPARATORE.search(testo), testo
        assert "\n\n" not in testo and not testo.startswith("\n") and not testo.endswith("\n")
        for riga in testo.split("\n"):
            assert riga.strip() == riga and re.search(r"\w", riga), riga
