"""
Le prove di testi.py: numeri, misure, date, durate, informazioni e novità di Ciechino.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
"""

import datetime

import testi
import version


def test_numero_all_italiana():
    assert testi.numero(3.24) == "3,2"
    assert testi.numero(12340.5) == "12.340,5"
    assert testi.numero(1234567.891, 2) == "1.234.567,89"
    assert testi.numero(7, 0) == "7"
    assert testi.numero(-0.04) == "0,0"
    assert testi.numero(-2.5) == "-2,5"


def test_intero_col_punto_delle_migliaia():
    assert testi.intero(12340) == "12.340"
    assert testi.intero(999) == "999"
    assert testi.intero(1000000) == "1.000.000"
    assert testi.intero(12.7) == "13"


def test_percento_col_simbolo():
    assert testi.percento(0.64) == "64%"
    assert testi.percento(0.015, 1) == "1,5%"
    assert testi.percento(1) == "100%"
    assert testi.percento(0) == "0%"


def test_metri_e_gradi():
    assert testi.metri(3.2) == "3,2 metri"
    assert testi.metri(1) == "1 metro"
    assert testi.metri(1.0) == "1 metro"
    assert testi.metri(5) == "5 metri"
    assert testi.metri(1.04) == "1,0 metri"
    assert testi.metri(0.123, 2) == "0,12 metri"
    assert testi.metri(1500) == "1.500 metri"
    assert testi.gradi(12.4) == "12,4 gradi"
    assert testi.gradi(1) == "1 grado"
    assert testi.gradi(45) == "45 gradi"


def test_conta_con_e_senza_segnaposto():
    assert testi.conta(1, "{n} colpo", "{n} colpi") == "1 colpo"
    assert testi.conta(3, "{n} colpo", "{n} colpi") == "3 colpi"
    assert testi.conta(12340, "{n} punto", "{n} punti") == "12.340 punti"
    assert testi.conta(1, "centro", "centri") == "1 centro"
    assert testi.conta(0, "centro", "centri") == "0 centri"


def test_elenco():
    assert testi.elenco([]) == ""
    assert testi.elenco(["a"]) == "a"
    assert testi.elenco(["a", "", "b"]) == "a e b"
    assert testi.elenco(["a", "b", "c"]) == "a, b e c"


def test_data_lunga_col_giorno_della_settimana():
    assert testi.data_lunga(datetime.date(2026, 11, 19)) == "giovedì 19 novembre 2026"
    assert testi.data_lunga("2026-11-19") == "giovedì 19 novembre 2026"
    assert testi.data_lunga("2026-11-22T10:00:00") == "domenica 22 novembre 2026"
    assert testi.data_lunga(datetime.datetime(2026, 11, 19, 18, 3), ora=True) == "giovedì 19 novembre 2026 alle 18:03"
    assert testi.data_lunga(datetime.datetime(2026, 11, 19, 1, 16), ora=True) == "giovedì 19 novembre 2026 all'1:16"
    assert testi.data_lunga(datetime.date(2026, 11, 19), ora=True) == "giovedì 19 novembre 2026"
    assert testi.data_lunga("non è una data") == "una data che non si legge"
    assert testi.data_lunga(None) == "una data che non si legge"


def test_eta_in_giorni():
    assert testi.eta_in_giorni(0) == "oggi"
    assert testi.eta_in_giorni(-3) == "oggi"
    assert testi.eta_in_giorni(1) == "1 giorno fa"
    assert testi.eta_in_giorni(3) == "3 giorni fa"
    assert testi.eta_in_giorni(1200) == "1.200 giorni fa"


def test_durata_a_parole():
    assert testi.durata(0) == "0 secondi"
    assert testi.durata(None) == "0 secondi"
    assert testi.durata(0.4) == "meno di un secondo"
    assert testi.durata(1) == "1 secondo"
    assert testi.durata(60) == "1 minuto"
    assert testi.durata(192) == "3 minuti e 12 secondi"
    assert testi.durata(3600) == "1 ora"
    assert testi.durata(3600 + 5 * 60 + 20) == "1 ora e 5 minuti"
    assert testi.durata(2 * 86400 + 3 * 3600 + 60) == "2 giorni, 3 ore e 1 minuto"


def test_informazioni():
    righe = testi.informazioni().split("\n")
    assert righe[0] == f"Ciechino, il tiro a segno a orecchio, versione {version.__version__}, rilasciata {testi.data_lunga(version.__date__)}."
    # La data lunga ha il giorno della settimana: la prova non si lega alla versione del momento.
    assert testi.data_lunga("2026-11-19") == "giovedì 19 novembre 2026"
    assert righe[1] == f"Autori: {version.__author__}."
    assert any("GNU GPL versione 3" in riga for riga in righe)
    assert righe[-1].endswith("https://github.com/GabrieleBattaglia/Ciechino")
    assert all(riga.strip() for riga in righe)


CHANGELOG = """# Changelog - Ciechino

Tutti i cambiamenti di **Ciechino**.

---

## [0.1.0] - 2026-11-19

Il primo rilascio, con il suono `nuova_partita` e *tre* scivolate.

### Il gioco

- La mira con [la guida](https://example.org/guida) e la __barra__.
* Un punto elenco con l'asterisco, e un nome_con_sottolineature.
> Una citazione.

## [Non pubblicata]

## [0.0.9] - in lavorazione

[0.1.0]: https://github.com/GabrieleBattaglia/Ciechino/releases/tag/v0.1.0
"""


def test_novita_toglie_il_markdown():
    righe = testi.novita(CHANGELOG).split("\n")
    assert righe == [
        "Changelog - Ciechino.",
        "Tutti i cambiamenti di Ciechino.",
        "Versione 0.1.0, giovedì 19 novembre 2026.",
        "Il primo rilascio, con il suono nuova_partita e tre scivolate.",
        "Il gioco.",
        "La mira con la guida (https://example.org/guida) e la barra.",
        "Un punto elenco con l'asterisco, e un nome_con_sottolineature.",
        "Una citazione.",
        "Versione Non pubblicata.",
        "Versione 0.0.9, in lavorazione.",
    ]


def test_novita_con_la_lineetta_e_i_titoli_sottolineati():
    assert testi.novita("## [0.2.0] \N{EN DASH} 2026-11-20\n===\n**Nota**: niente.") == "Versione 0.2.0, venerdì 20 novembre 2026.\nNota: niente."


def test_novita_vuota():
    assert testi.novita("") == ""
    assert testi.novita(None) == ""
