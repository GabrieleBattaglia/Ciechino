"""
Le prove di mira.py: i quattro esiti di valore_scelto, la compensazione del tasto e la scadenza.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0.
"""

import itertools

import pytest

import mira
from costanti import COMPENSAZIONE_DEL_TASTO, FS, MARGINE_DOPO_LA_SCIVOLATA
from sintesi import Scivolata, scivolata

INIZIO = 100_000


class Finta:
    """Una scivolata con numeri tondi: 1.000 campioni di bip e silenzio, 10.000 di scala."""

    inizio_scivolata = 1000
    campioni_scivolata = 10_000
    fs = FS


def test_pausa():
    assert mira.valore_scelto(INIZIO - 1, INIZIO, Finta(), compensazione=0.0) == (None, "pausa")
    assert mira.valore_scelto(0, INIZIO, Finta(), compensazione=0.0) == (None, "pausa")


def test_minimo_nei_bip_e_nel_silenzio():
    assert mira.valore_scelto(INIZIO, INIZIO, Finta(), compensazione=0.0) == (0.0, "minimo")
    assert mira.valore_scelto(INIZIO + 999.5, INIZIO, Finta(), compensazione=0.0) == (0.0, "minimo")


def test_scelto_continuo_nella_scala():
    assert mira.valore_scelto(INIZIO + 1000, INIZIO, Finta(), compensazione=0.0) == (0.0, "scelto")
    valore, esito = mira.valore_scelto(INIZIO + 1000 + 2500.25, INIZIO, Finta(), compensazione=0.0)
    assert esito == "scelto"
    assert valore == pytest.approx(0.250025)
    valori = [mira.valore_scelto(INIZIO + 1000 + x / 10.0, INIZIO, Finta(), compensazione=0.0)[0] for x in range(0, 100_000, 7)]
    assert all(b > a for a, b in itertools.pairwise(valori))
    valore, esito = mira.valore_scelto(INIZIO + 10_999.9, INIZIO, Finta(), compensazione=0.0)
    assert esito == "scelto" and 0.9999 < valore < 1.0


def test_massimo_dopo_la_fine():
    assert mira.valore_scelto(INIZIO + 11_000, INIZIO, Finta(), compensazione=0.0) == (1.0, "massimo")
    assert mira.valore_scelto(INIZIO + 50_000, INIZIO, Finta(), compensazione=0.0) == (1.0, "massimo")


def test_compensazione():
    """Lo spazio sentito al campione c vale come se fosse stato premuto compensazione secondi prima."""
    secondi = 0.05
    indietro = secondi * FS
    valore, esito = mira.valore_scelto(INIZIO + 6000 + indietro, INIZIO, Finta(), compensazione=secondi)
    assert esito == "scelto" and valore == pytest.approx(0.5)
    assert mira.valore_scelto(INIZIO + 1000 + indietro - 1, INIZIO, Finta(), compensazione=secondi) == (0.0, "minimo")
    assert mira.valore_scelto(INIZIO + indietro - 1, INIZIO, Finta(), compensazione=secondi) == (None, "pausa")
    assert mira.valore_scelto(INIZIO + 11_000 + indietro, INIZIO, Finta(), compensazione=secondi) == (1.0, "massimo")


def test_compensazione_predefinita():
    atteso = mira.valore_scelto(INIZIO + 6000, INIZIO, Finta(), compensazione=COMPENSAZIONE_DEL_TASTO)
    assert mira.valore_scelto(INIZIO + 6000, INIZIO, Finta()) == atteso


def test_con_la_scivolata_vera():
    barra = scivolata("potenza", tacche=False)
    assert isinstance(barra, Scivolata)
    meta = INIZIO + barra.inizio_scivolata + barra.campioni_scivolata / 2.0
    assert mira.valore_scelto(meta, INIZIO, barra, compensazione=0.0) == (0.5, "scelto")
    assert mira.valore_scelto(INIZIO + barra.fine_scivolata, INIZIO, barra, compensazione=0.0) == (1.0, "massimo")


def test_scaduta():
    fine = INIZIO + 11_000
    margine = MARGINE_DOPO_LA_SCIVOLATA * FS
    assert not mira.scaduta(fine, INIZIO, Finta())
    assert not mira.scaduta(fine + margine - 1, INIZIO, Finta())
    assert mira.scaduta(fine + margine, INIZIO, Finta())
    assert mira.scaduta(fine, INIZIO, Finta(), margine=0.0)
    assert set(mira.ESITI) == {"pausa", "minimo", "scelto", "massimo"}
