"""
Le prove dei banchi d'ascolto di Ciechino, strumenti/banco_mira.py, banco_bersagli.py e banco_volo.py,
senza mai suonare e senza mai toccare i file veri degli esiti.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di tests/test_banco_spazio.py di MESS.
Il motore è quello vero, sul flusso finto di aiuti_audio, senza latenza: una tastiera finta da copione,
al posto di key, a ogni occhiata fa avanzare il flusso del tempo dell'occhiata, così i banchi suonano
e misurano come col motore vero, e la prova sa che cosa è uscito e quando. Una voce ("frazione", f)
del copione porta il campione udibile esattamente al punto f della scala della barra in corso e
preme spazio: è la pressione a un campione noto, con cui si controllano il valore scelto e i conti
della precisione. I preset dei suoni sono finti, uno diverso per nome; la cassa di Acusticator e il
motore sulla scheda vera fanno fallire la prova se qualcuno li chiama; gli esiti vanno in tmp_path.
Si controllano le versioni di ogni gruppo sui buffer, il riferimento della precisione contro la
scala, il controllo identico, i tasti della rassegna e del voto, lo spazio che ferma la scivolata
come in partita, la precisione calcolata giusta, e le sessioni intere: i prompt fra due ritorni
carrello entro quaranta caratteri, nessuna riga vuota, nessun nome delle versioni, nessuna parola che
sveli il controllo o l'attuale e nessun numero delle misure a schermo, i voti e la rivelazione nel
file, anche quando la sessione si interrompe, e il motore chiuso alla fine.
"""

import math
import random
import statistics
import sys
import zlib
from pathlib import Path

import collaudo_comune
import numpy as np
import pytest
from aiuti_audio import motore_finto
from GBUtils import Acusticator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "strumenti"))

import banco_bersagli as bb
import banco_mira as bm
import banco_volo as bv

import costanti
import fisica
import mira
import sintesi
import suoni
from bersagli import Bersaglio
from costanti import ALZO_FREQUENZE, FS, PARAMETRI, PAUSA_FRA_BARRE, SCIVOLATA_DURATA, TACCA_DB
from motore_audio import VoceTono

SEME = 4127
# Quella vera, presa prima che la fissatura automatica la vieti: una prova la chiama col motore muto.
PREPARA_ASCOLTO_VERA = bm.prepara_ascolto
# I preset finti: un rumore breve, diverso per ogni nome, al centro.
LUNGHEZZA_FINTA = round(0.05 * FS)
STRUMENTI = Path(bm.__file__).resolve().parent
BANCHI = (bm, bb, bv)
# Le parole che a schermo non devono comparire prima della rivelazione, che a schermo non c'è mai: il
# controllo, l'attuale e le unità delle misure.
VIETATE = ("controllo", "ripetut", "attual", "hertz", "decibel", "millesim", "%")


def _vietato(nome):
    def chiamata(*_argomenti, **_opzioni):
        raise AssertionError(f"una prova avrebbe suonato davvero, con {nome}")

    return chiamata


def _preset_finto(nome):
    rumore = np.random.default_rng(zlib.crc32(nome.encode())).standard_normal(LUNGHEZZA_FINTA) * 0.2
    buffer = sintesi.panoramica(rumore, 0.0)
    buffer.flags.writeable = False
    return buffer


@pytest.fixture(autouse=True)
def niente_di_vero(monkeypatch, tmp_path):
    """Nessun suono vero e nessun file vero: la cassa di Acusticator e il motore sulla scheda fanno
    fallire chi li chiama, i preset sono finti, gli esiti vanno in tmp_path; alla fine i file veri
    degli esiti accanto ai banchi sono quelli di prima."""
    for nome in ("riproduci", "ciclo_di", "play", "stop"):
        monkeypatch.setattr(Acusticator, nome, _vietato(f"Acusticator.{nome}"))
    monkeypatch.setattr(bm, "prepara_ascolto", _vietato("banco_mira.prepara_ascolto"))
    monkeypatch.setattr(suoni, "_rendi_preset", _preset_finto)
    for banco in BANCHI:
        monkeypatch.setattr(banco, "FILE_DEGLI_ESITI", tmp_path / Path(banco.FILE_DEGLI_ESITI).name)
    prima = {p.name: p.stat().st_mtime_ns for p in STRUMENTI.glob("*_esiti.txt")}
    yield
    assert {p.name: p.stat().st_mtime_ns for p in STRUMENTI.glob("*_esiti.txt")} == prima


@pytest.fixture
def ascolto():
    """L'ascolto col motore vero sul flusso finto, senza latenza; voci sono tutte le voci fatte partire."""
    motore, flusso = motore_finto(latenza=0.0)
    voci = []
    originale = motore.suona

    def suona(buffer, inizio=None, guadagno=1.0):
        voce = originale(buffer, inizio=inizio, guadagno=guadagno)
        voci.append(voce)
        return voce

    motore.suona = suona
    finto = bm.Ascolto(motore, suoni.Suoni(motore))
    finto.voci = voci
    finto.flusso = flusso
    yield finto
    motore.chiudi()


def _materiali_partiti(ascolto):
    """Le voci dei materiali, senza quelle dei preset, che sono corte."""
    return [voce for voce in ascolto.voci if len(voce.buffer) > LUNGHEZZA_FINTA]


def _preset_partiti(ascolto):
    return [voce for voce in ascolto.voci if len(voce.buffer) == LUNGHEZZA_FINTA]


class Tastiera:
    """Al posto di key di GBUtils: ai prompt scrive il prompt come quella vera e risponde dal copione,
    e se lo ricorda in prompt. Alle occhiate guarda la testa del copione: ("durante", n, tasto) torna
    il tasto, o lo solleva se è un'eccezione, quando l'ascolto ha fatto partire n materiali;
    ("frazione", f) porta il campione udibile al punto f della scala della barra in corso e torna lo
    spazio. Altrimenti fa passare il tempo dell'occhiata, sull'orologio finto e sul flusso finto."""

    def __init__(self, copione, ascolto):
        self.copione = list(copione)
        self.ascolto = ascolto
        self.adesso = 0.0
        self.prompt = []

    def orologio(self):
        return self.adesso

    def porta_a(self, campione):
        motore, flusso = self.ascolto.motore, self.ascolto.flusso
        flusso.avanza_fino(math.ceil(campione))
        flusso.time -= (motore.campione_scritto - campione) / motore.fs

    def __call__(self, prompt="", attesa=None, alla_scadenza=""):
        if attesa is None:
            print(prompt, end="")
            self.prompt.append(prompt)
            voce = self.copione.pop(0)
            assert isinstance(voce, str), f"il copione aspettava un tasto a un prompt, trova {voce!r}"
            return voce
        voce = self.copione[0] if self.copione else None
        if isinstance(voce, tuple):
            if voce[0] == "durante" and voce[1] == self.ascolto.partiti:
                self.copione.pop(0)
                if isinstance(voce[2], BaseException):
                    raise voce[2]
                return voce[2]
            if voce[0] == "frazione" and self.ascolto.barra is not None:
                inizio, scivolata = self.ascolto.barra
                self.porta_a(inizio + scivolata.inizio_scivolata + voce[1] * scivolata.campioni_scivolata)
                self.copione.pop(0)
                return " "
        self.adesso += attesa
        self.ascolto.flusso.avanza_secondi(attesa)
        return alla_scadenza


def _tastiera(monkeypatch, ascolto, copione):
    tastiera = Tastiera(copione, ascolto)
    monkeypatch.setattr(bm, "key", tastiera)
    monkeypatch.setattr(collaudo_comune, "key", tastiera)
    monkeypatch.setattr(bm, "orologio", tastiera.orologio)
    return tastiera


def _enter_escape_finta(risposte):
    """Al posto di enter_escape di GBUtils: scrive il prompt e va a capo come quella vera, e risponde dal copione."""
    risposte = iter(risposte)

    def enter_escape(prompt="", *_argomenti, **_opzioni):
        print(prompt)
        return next(risposte)

    return enter_escape


def _prompt_giusto(prompt):
    return prompt.startswith("\r") and prompt.endswith("\r") and 0 < len(prompt.strip("\r")) <= 40 and "\n" not in prompt


def _schermo_pulito(scritto, nomi):
    """A schermo nessuna riga vuota, nessun nome delle versioni, nessuna parola vietata."""
    linee = scritto.split("\n")
    vuote = [i for i, riga in enumerate(linee[:-1]) if riga.strip("\r") == ""]
    assert vuote == [], [linee[i - 1] for i in vuote]
    for vietata in (*nomi, *VIETATE):
        assert vietata not in scritto, vietata


def _db(x):
    return 10.0 * math.log10(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _frequenza(mono, centro, mezzo=256, intorno=4096):
    """La frequenza istantanea attorno a un campione, dalla fase del segnale analitico."""
    from scipy.signal import hilbert

    tratto = np.asarray(mono[centro - intorno:centro + intorno], dtype=np.float64)
    fase = np.unwrap(np.angle(hilbert(tratto)))
    istantanea = np.diff(fase) * FS / (2.0 * math.pi)
    return float(np.median(istantanea[intorno - mezzo:intorno + mezzo]))


def _pan(stereo):
    """La panoramica a potenza costante che danno le energie dei due canali."""
    sinistra, destra = float(np.sum(stereo[:, 0].astype(np.float64) ** 2)), float(np.sum(stereo[:, 1].astype(np.float64) ** 2))
    return math.atan2(math.sqrt(destra), math.sqrt(sinistra)) * 4.0 / math.pi - 1.0


def _nella_scala(scivolata, frazione):
    """Il campione del buffer della scivolata al punto frazione della scala."""
    return scivolata.inizio_scivolata + round(frazione * scivolata.campioni_scivolata)


def _nomi(gruppi):
    return [candidato.nome for gruppo in gruppi for candidato in gruppo.candidati]


# I prompt.

def test_i_prompt_stanno_in_quaranta_caratteri():
    for lettera in bm.LETTERE:
        assert _prompt_giusto(bm.invito(lettera, None)), bm.invito(lettera, None)
        assert _prompt_giusto(bm.invito(lettera, 5)), bm.invito(lettera, 5)
    for posto in range(1, bm.TENTATIVI + 1):
        assert _prompt_giusto(bm.invito_al_tentativo(posto, bm.TENTATIVI))
    assert _prompt_giusto(bm.invito_a_cominciare())


# I gruppi della mira.

def test_ogni_gruppo_della_mira_ha_le_sue_versioni_e_una_attuale():
    assert [g.chiave for g in bm.GRUPPI] == ["direzione", "alzo", "potenza", "tacche", "bip"]
    for gruppo in bm.GRUPPI:
        assert [c.attuale for c in gruppo.candidati].count(True) == 1, gruppo.chiave
        assert len({c.nome for c in gruppo.candidati}) == len(gruppo.candidati)
        assert len(gruppo.candidati) == (2 if gruppo.chiave == "bip" else 3)
        materiali = [bm.componi(c.ricetta) for c in gruppo.candidati]
        # L'attuale è proprio ciò che il gioco suona, campione per campione.
        attuale = materiali[bm.indice_attuale(gruppo)]
        parametri = [s.parametro for s in attuale]
        assert parametri == (list(PARAMETRI) if gruppo.chiave in ("tacche", "bip") else [gruppo.chiave])
        for scivolata in attuale:
            assert np.array_equal(scivolata.buffer, sintesi.scivolata(scivolata.parametro).buffer)
        # Le versioni suonano diverse fra loro.
        for i in range(len(materiali)):
            for j in range(i + 1, len(materiali)):
                assert any(len(a.buffer) != len(b.buffer) or not np.array_equal(a.buffer, b.buffer)
                           for a, b in zip(materiali[i], materiali[j], strict=True)), (gruppo.chiave, i, j)


def _materiale(chiave):
    gruppo = next(g for g in bm.GRUPPI if g.chiave == chiave)
    return {c.nome: bm.componi(c.ricetta) for c in gruppo.candidati}, gruppo


def test_le_scale_dell_alzo():
    materiali, gruppo = _materiale("alzo")
    attese = {"esponenziale": lambda da, a, f: da * (a / da) ** f, "lineare": lambda da, a, f: da + (a - da) * f}
    for candidato in gruppo.candidati:
        (_parametro, impostazioni), = candidato.ricetta
        scivolata, = materiali[candidato.nome]
        da, a = impostazioni["frequenze"]
        for frazione in (0.25, 0.45, 0.65):
            misurata = _frequenza(scivolata.buffer[:, 0], _nella_scala(scivolata, frazione))
            assert misurata == pytest.approx(attese[impostazioni["curva"]](da, a, frazione), rel=0.005), (candidato.nome, frazione)
    assert {c.ricetta[0][1]["frequenze"] for c in gruppo.candidati} == {tuple(ALZO_FREQUENZE), bm.ALZO_DI_PRIMA}


def test_le_scale_della_potenza():
    materiali, gruppo = _materiale("potenza")
    intorno = round(0.025 * FS)
    for candidato in gruppo.candidati:
        (_parametro, impostazioni), = candidato.ricetta
        scivolata, = materiali[candidato.nome]
        livelli = [_db(scivolata.buffer[_nella_scala(scivolata, f) - intorno:_nella_scala(scivolata, f) + intorno]) for f in (0.25, 0.75)]
        minimo = impostazioni["db_minimo"]
        if impostazioni["curva"] == "decibel":
            atteso = -minimo * 0.5
        else:
            basso, alto = sintesi.db_in_lineare(costanti.SCIVOLATA_DB + minimo), sintesi.db_in_lineare(costanti.SCIVOLATA_DB)
            atteso = 20.0 * math.log10((basso + (alto - basso) * 0.75) / (basso + (alto - basso) * 0.25))
        assert livelli[1] - livelli[0] == pytest.approx(atteso, abs=0.5), candidato.nome


def test_le_tacche_e_i_bip():
    materiali, gruppo = _materiale("tacche")
    attuali, forti, senza = (materiali[c.nome] for c in gruppo.candidati)
    assert gruppo.candidati[0].attuale
    for a, f, s in zip(attuali, forti, senza, strict=True):
        assert s.tacche == () and len(a.tacche) == costanti.TACCHE
        dove, fine = a.tacche[0]
        # Tolta la scala, che è la stessa, restano le tacche: le più forti di TACCHE_PIU_FORTI_DB.
        assert _db(f.buffer[dove:fine] - s.buffer[dove:fine]) - _db(a.buffer[dove:fine] - s.buffer[dove:fine]) == pytest.approx(bm.TACCHE_PIU_FORTI_DB, abs=0.1)
    assert TACCA_DB + bm.TACCHE_PIU_FORTI_DB < 0
    materiali, gruppo = _materiale("bip")
    corti, lunghi = (materiali[c.nome] for c in gruppo.candidati)
    for corta, lunga in zip(corti, lunghi, strict=True):
        quanti = len(costanti.BIP_FREQUENZE[corta.parametro])
        assert lunga.inizio_scivolata - corta.inizio_scivolata == quanti * (round(bm.BIP_LUNGHI * FS) - round(costanti.BIP_DURATA * FS))


def test_il_controllo_e_il_seme():
    prove = bm.prepara(SEME)
    for prova in prove:
        assert len(prova.lettere) == len(prova.gruppo.candidati) + 1
        a, b = prova.lettere_di(prova.ripetuto)
        assert prova.materiale(a) is prova.materiale(b)
    impronta = [(p.ordine, p.ripetuto) for p in prove]
    assert impronta == [(p.ordine, p.ripetuto) for p in bm.prepara(SEME)]
    assert impronta != [(p.ordine, p.ripetuto) for p in bm.prepara(SEME + 1)]


# Il riferimento della precisione.

def test_il_riferimento_suona_come_la_scala_in_quel_punto():
    intorno = round(0.025 * FS)
    for parametro in PARAMETRI:
        scivolata = sintesi.scivolata(parametro)
        for frazione in (0.25, 0.65):
            tono = bm.tono_di_riferimento(parametro, frazione)
            assert tono.dtype == np.float32 and tono.shape == (round(bm.DURATA_DEL_RIFERIMENTO * FS), 2)
            # Comincia e finisce con le rampe, contro i click.
            assert float(np.max(np.abs(tono[0]))) < 0.01 and float(np.max(np.abs(tono[-1]))) < 0.01
            mezzo = len(tono) // 2
            punto = _nella_scala(scivolata, frazione)
            pezzo_della_scala = scivolata.buffer[punto - intorno:punto + intorno]
            pezzo_del_tono = tono[mezzo - intorno:mezzo + intorno]
            if parametro == "direzione":
                assert _pan(pezzo_del_tono) == pytest.approx(2.0 * frazione - 1.0, abs=0.01)
                assert _pan(pezzo_della_scala) == pytest.approx(_pan(pezzo_del_tono), abs=0.02)
            elif parametro == "alzo":
                attesa = ALZO_FREQUENZE[0] * (ALZO_FREQUENZE[1] / ALZO_FREQUENZE[0]) ** frazione
                assert _frequenza(tono[:, 0], mezzo, intorno=mezzo - 10) == pytest.approx(attesa, rel=0.005)
                assert _frequenza(scivolata.buffer[:, 0], punto) == pytest.approx(attesa, rel=0.005)
            else:
                assert _db(pezzo_del_tono) == pytest.approx(_db(pezzo_della_scala), abs=0.5)


# L'ascolto: rassegna, voto, mira.

def test_la_rassegna_col_silenzio_iniziale(ascolto, monkeypatch, capsys):
    prova = bm.prepara(SEME)[0]
    _tastiera(monkeypatch, ascolto, [])
    bm.rassegna(ascolto, prova)
    partiti = _materiali_partiti(ascolto)
    assert ascolto.partiti == len(prova.lettere) == len(partiti)
    for posto, (lettera, voce) in enumerate(zip(prova.lettere, partiti, strict=True)):
        anticipo = round((bm.ANTICIPO_PRIMA_LETTERA if posto == 0 else bm.ANTICIPO_RASSEGNA) * FS)
        scivolata, = prova.materiale(lettera)
        assert not np.any(voce.buffer[:anticipo])
        assert np.array_equal(voce.buffer[anticipo:], scivolata.buffer)
    # Prima del primo suono qualche secondo di silenzio, proprio all'uscita.
    assert bm.ANTICIPO_PRIMA_LETTERA >= bm.SILENZIO_INIZIALE + bm.ANTICIPO_RASSEGNA
    uscita = ascolto.flusso.uscita
    assert not np.any(uscita[partiti[0].inizio:partiti[0].inizio + round(bm.ANTICIPO_PRIMA_LETTERA * FS)])
    assert np.any(uscita[partiti[0].inizio:])
    # Nessuno ha fermato le scivolate: ognuna è arrivata in fondo e ha preso il massimo, col suo suono.
    assert len(_preset_partiti(ascolto)) == len(prova.lettere)
    assert capsys.readouterr().out.splitlines() == ["Rassegna."] + [f"Lettera {lettera}." for lettera in prova.lettere]
    # Escape mentre suona la seconda lettera interrompe la rassegna; spazio e i voti no.
    _tastiera(monkeypatch, ascolto, [("durante", 5, " "), ("durante", 5, "1"), ("durante", 6, "\x1b")])
    bm.rassegna(ascolto, prova)
    assert ascolto.partiti == 6
    assert capsys.readouterr().out.splitlines()[-1] == "Rassegna interrotta."


def test_il_voto_e_lo_spazio_che_ferma(ascolto, monkeypatch, capsys):
    prova = bm.prepara(SEME)[0]
    a, b, c, d = prova.lettere
    # A: spazio la fa partire, lo spazio a metà scala la ferma come in partita, poi il 4. B: Invio passa
    # senza voto. C: la lettera d la fa sentire per confronto, e il 2 premuto mentre suona vale subito
    # per C. D: un tasto sconosciuto non fa niente, ed Escape chiude il voto.
    copione = [" ", ("frazione", 0.5), "4", "\r", d.lower(), ("durante", 2, "2"), "x", "\x1b"]
    tastiera = _tastiera(monkeypatch, ascolto, copione)
    bm.voto(ascolto, prova)
    assert tastiera.copione == []
    assert prova.voti == {a: 4, c: 2}
    # La fermata arriva al motore al blocco seguente: si lascia scorrere il flusso.
    ascolto.flusso.avanza(4)
    prima, seconda = _materiali_partiti(ascolto)
    assert np.array_equal(prima.buffer[-len(prova.materiale(a)[0].buffer):], prova.materiale(a)[0].buffer)
    assert np.array_equal(seconda.buffer[-len(prova.materiale(d)[0].buffer):], prova.materiale(d)[0].buffer)
    # La prima si è fermata a metà della scala, con la sfumatura del gioco, e ha fatto suonare il valore preso.
    fine_della_scala = prima.inizio + len(prima.buffer)
    assert prima.finita and prima.finita_al < fine_della_scala - prova.materiale(a)[0].campioni_scivolata // 3
    assert seconda.finita and len(_preset_partiti(ascolto)) == 1
    scritto = capsys.readouterr().out
    assert bm.invito(a, None) in scritto
    assert scritto.rstrip().endswith(f"I voti: {a} 4, {b} senza voto, {c} 2, {d} senza voto.")
    assert "\n\n" not in scritto
    # Il secondo giro mostra il voto già dato, e Invio lo lascia com'è; n cancella tutto.
    _tastiera(monkeypatch, ascolto, ["\r", "n"])
    bm.voto(ascolto, prova)
    assert bm.invito(a, 4) in capsys.readouterr().out
    assert prova.nessuna_preferenza and prova.voti == {}


def test_la_mira_intera_e_la_compensazione(ascolto, monkeypatch):
    materiale = bm.componi(next(c for c in bm.GRUPPI[3].candidati if c.attuale).ricetta)
    _tastiera(monkeypatch, ascolto, [("frazione", 0.3), ("frazione", 0.6), ("frazione", 0.9)])
    tasto, valori = ascolto.suona_mira(materiale, bm.ANTICIPO_VOTO, frozenset("\x1b"), ferma_con_spazio=True)
    assert tasto is None
    assert [v.esito for v in valori] == ["scelto"] * 3 and all(v.premuto for v in valori)
    assert [v.valore for v in valori] == pytest.approx([0.3, 0.6, 0.9], abs=1e-6)
    barre = _materiali_partiti(ascolto)
    assert ascolto.partiti == 1 and len(barre) == 3 and len(_preset_partiti(ascolto)) == 3
    # Ogni barra è la pausa fra le barre, che non sta nel buffer della scivolata, poi la scivolata del
    # gioco; la prima almeno la pausa, anche con l'anticipo breve del voto.
    for voce, scivolata in zip(barre, materiale, strict=True):
        testa = len(voce.buffer) - len(scivolata.buffer)
        assert testa == round(PAUSA_FRA_BARRE * FS)
        assert not np.any(voce.buffer[:testa]) and np.array_equal(voce.buffer[testa:], scivolata.buffer)
    # La barra seguente parte dopo che la precedente è stata fermata.
    assert barre[1].inizio > barre[0].finita_al and barre[2].inizio > barre[1].finita_al
    # La compensazione del tasto si toglie al campione udibile: lo stesso campione vale un po' meno.
    monkeypatch.setattr(mira, "COMPENSAZIONE_DEL_TASTO", 0.06)
    _tastiera(monkeypatch, ascolto, [("frazione", 0.5)])
    _tasto, (valore,) = ascolto.suona_mira(materiale[:1], 0.0, frozenset("\x1b"), ferma_con_spazio=True)
    assert valore.valore == pytest.approx(0.5 - 0.06 / SCIVOLATA_DURATA, abs=1e-6)


# La precisione.

def test_la_precisione_su_pressioni_a_campioni_noti(ascolto, monkeypatch, capsys):
    precisione = bm.Precisione("alzo", random.Random(11))
    riferimenti = bm.Precisione("alzo", random.Random(11)).riferimenti()
    assert len(riferimenti) == bm.TENTATIVI and all(bm.RIFERIMENTI[0] <= r <= bm.RIFERIMENTI[1] for r in riferimenti)
    # Uno in ogni ottavo dell'intervallo.
    da, a = bm.RIFERIMENTI
    assert sorted(int((r - da) / (a - da) * bm.TENTATIVI) for r in riferimenti) == list(range(bm.TENTATIVI))
    scarti = [0.05, -0.03, 0.10, 0.0, -0.08, 0.02]
    copione = [" ", ("durante", 1, " ")]
    for riferimento, scarto in zip(riferimenti, scarti, strict=False):
        copione += [("frazione", riferimento + scarto), " "]
    # Il settimo nei bip, l'ottavo senza spazio: il minimo e il massimo.
    copione += [("frazione", -0.01), " "]
    tastiera = _tastiera(monkeypatch, ascolto, copione)
    bm.serie(ascolto, precisione)
    assert tastiera.copione == []
    tentativi = precisione.tentativi
    assert [t.posto for t in tentativi] == list(range(1, 9)) and {t.serie for t in tentativi} == {1}
    assert [t.riferimento for t in tentativi] == riferimenti
    for tentativo, scarto in zip(tentativi, scarti, strict=False):
        assert tentativo.esito == "scelto" and tentativo.premuto
        assert tentativo.errore == pytest.approx(scarto, abs=1e-6)
    assert (tentativi[6].valore, tentativi[6].esito, tentativi[6].premuto) == (0.0, "minimo", True)
    assert (tentativi[7].valore, tentativi[7].esito, tentativi[7].premuto) == (1.0, "massimo", False)
    conti = bm.misura(tentativi)
    assert (conti.totale, conti.fermati, conti.minimi, conti.massimi) == (8, 6, 1, 1)
    assert conti.medio == pytest.approx(statistics.fmean(scarti), abs=1e-6)
    assert conti.assoluto == pytest.approx(statistics.fmean(abs(s) for s in scarti), abs=1e-6)
    assert conti.deviazione == pytest.approx(statistics.stdev(scarti), abs=1e-6)
    # Il primo tentativo: il silenzio iniziale, il riferimento, la pausa e la scivolata del gioco.
    prima = _materiali_partiti(ascolto)[0]
    silenzio = round((bm.ANTICIPO_TENTATIVO + bm.SILENZIO_INIZIALE) * FS)
    tono = bm.tono_di_riferimento("alzo", riferimenti[0])
    assert not np.any(prima.buffer[:silenzio])
    assert np.array_equal(prima.buffer[silenzio:silenzio + len(tono)], tono)
    assert not np.any(prima.buffer[silenzio + len(tono):silenzio + len(tono) + round(PAUSA_FRA_BARRE * FS)])
    assert np.array_equal(prima.buffer[-len(precisione.scivolata.buffer):], sintesi.scivolata("alzo").buffer)
    seconda = _materiali_partiti(ascolto)[1]
    assert len(seconda.buffer) == len(prima.buffer) - round(bm.SILENZIO_INIZIALE * FS)
    # Le righe del file: ogni tentativo coi suoi numeri all'italiana, e il riepilogo.
    riga = bm.riga_del_tentativo(precisione, tentativi[0])
    assert riga == (f"Precisione, l'alzo, serie 1, tentativo 1: riferimento {bm.percento(riferimenti[0], 1)}, scelto {bm.percento(riferimenti[0] + 0.05, 1)}, "
                    f"errore +5,0%, cioè 150 millesimi in ritardo; fermata nella scala.")
    assert "errore -3,0%, cioè 90 millesimi in anticipo" in bm.riga_del_tentativo(precisione, tentativi[1])
    assert bm.riga_del_tentativo(precisione, tentativi[3]).endswith("errore 0,0%, cioè nessun ritardo; fermata nella scala.")
    assert bm.riga_del_tentativo(precisione, tentativi[6]).endswith("; spazio nei bip, vale il minimo.")
    assert bm.riga_del_tentativo(precisione, tentativi[7]).endswith("; non fermata, vale il massimo.")
    riepilogo = bm.riga_del_riepilogo(precisione.titolo, tentativi)
    medio = statistics.fmean(scarti)
    assert riepilogo.startswith(f"Precisione, l'alzo, riepilogo: 6 tentativi fermati nella scala su 8; errore medio +{bm.numero(medio * 100, 1)}%, ")
    assert riepilogo.endswith("; 1 al minimo, 1 al massimo.")
    assert "\n\n" not in capsys.readouterr().out
    # Escape a un prompt interrompe la serie, che conta un'altra serie.
    _tastiera(monkeypatch, ascolto, [" ", ("frazione", 0.5), "\x1b"])
    bm.serie(ascolto, precisione)
    assert len(precisione.tentativi) == 9 and precisione.tentativi[-1].serie == 2
    assert capsys.readouterr().out.splitlines()[-1] == "Serie interrotta."


# La sessione della mira.

def test_la_sessione_della_mira(ascolto, monkeypatch, capsys, tmp_path):
    prove = bm.prepara(SEME)
    monkeypatch.setattr(bm, "prepara", lambda seme, materiali=None: prove if seme == SEME else pytest.fail(f"seme {seme}"))
    monkeypatch.setattr(bm, "prepara_ascolto", lambda eventi=(): ascolto)
    monkeypatch.setattr(bm, "enter_escape", _enter_escape_finta([True]))
    monkeypatch.setattr(collaudo_comune, "enter_escape", _enter_escape_finta([True, False, False, False, True, True, False, False]))
    monkeypatch.setattr(collaudo_comune, "dgt", lambda *_a, **_k: "la seconda è più chiara ")
    riferimenti = bm.prepara_precisione(SEME)[0].riferimenti()
    scarti = [0.02, -0.04, 0.06, 0.01, -0.02, 0.03, 0.0, -0.05]
    direzione, bip = prove[0], prove[4]
    voti = dict(zip(direzione.lettere, (5, 3, 2, 4), strict=True))
    a = direzione.lettere[0]
    # La direzione: la rassegna intera; nel voto A si fa partire, si ferma e si vota; al prompt di B si
    # sente A per confronto e il 3 va a B; poi C e D; nel menu un commento e Invio. I bip: Escape nella
    # rassegna, mentre suona la seconda lettera; n nel voto; Escape nel menu. La precisione della
    # direzione: otto tentativi a scarti noti, e Invio.
    copione = [" ", ("frazione", 0.4), "5", a.lower(), ("durante", 6, "3"), "2", "4", "c", "\r"]
    copione += [("durante", 8, "\x1b"), "n", "\x1b"]
    for riferimento, scarto in zip(riferimenti, scarti, strict=True):
        copione += [" ", ("frazione", riferimento + scarto)]
    copione += ["\r"]
    tastiera = _tastiera(monkeypatch, ascolto, copione)
    assert bm.main([str(SEME)]) == 0
    assert tastiera.copione == []
    assert ascolto.motore.chiuso
    assert all(_prompt_giusto(p) for p in tastiera.prompt), [p for p in tastiera.prompt if not _prompt_giusto(p)]
    assert direzione.voti == voti and bip.nessuna_preferenza and bip.chiusa
    assert [p.saltata for p in prove] == [False, True, True, True, False]

    righe = (tmp_path / "banco_mira_esiti.txt").read_text(encoding="utf-8").splitlines()
    assert righe[0].startswith("Banco della mira, 20") and righe[0].endswith(f", seme {SEME}.")
    titoli = [p.gruppo.titolo for p in prove]
    prima_rivelazione = next(i for i, r in enumerate(righe) if " era: " in r)
    davanti = righe[1:prima_rivelazione]
    attese = [f"{titoli[0]}, lettera {x}: {voti[x]}." for x in direzione.lettere]
    i = davanti.index(attese[0])
    assert davanti[i:i + 4] == attese
    assert davanti[i + 4].startswith(f"{titoli[0]}, ") and davanti[i + 4].endswith(": la seconda è più chiara")
    assert [f"{t}: gruppo saltato." for t in titoli[1:4]] == davanti[i + 5:i + 8]
    assert davanti[i + 8] == f"{titoli[4]}: nessuna preferenza."
    assert davanti[i + 9].endswith(": chiuso senza giudizio")
    assert davanti[i + 10] == "Precisione, la direzione: 8 tentativi fatti."
    assert davanti[i + 11].endswith(": test superato, nessun commento")
    assert davanti[i + 12:] == ["Precisione, l'alzo: prova saltata.", "Precisione, la potenza: prova saltata."]
    nomi = _nomi(bm.GRUPPI)
    assert not any(nome in riga for riga in righe[:prima_rivelazione] for nome in nomi)
    assert not any("%" in riga for riga in righe[:prima_rivelazione])
    rivelate = righe[prima_rivelazione:]
    for prova in (direzione, bip):
        titolo = prova.gruppo.titolo
        for x in prova.lettere:
            assert any(r.startswith(f"{titolo}, la lettera {x} era: {prova.candidato(x).nome}") for r in rivelate)
        x, y = prova.lettere_di(prova.ripetuto)
        assert f"{titolo}, la lettera {x} era: {prova.candidato(x).nome}; è la versione ripetuta per controllo, identica alla lettera {y}." in rivelate
    x, y = direzione.lettere_di(direzione.ripetuto)
    assert f"Controllo, {titoli[0]}: le lettere {x} e {y}, identiche campione per campione, hanno avuto {x} {voti[x]} e {y} {voti[y]}." in rivelate
    assert f"Scelta, {titoli[0]}: {direzione.gruppo.candidati[bm.scelta(direzione)].nome}." in rivelate
    attuale = bip.gruppo.candidati[bm.indice_attuale(bip.gruppo)].nome
    assert f"Scelta, {titoli[4]}: il gruppo è stato chiuso senza giudizio, quindi i voti non contano e resta {attuale}." in rivelate
    assert not any(r.startswith(f"{t}, la lettera") for r in rivelate for t in titoli[1:4])
    # I conti della precisione, in fondo.
    assert any(r.startswith("Precisione: l'errore è il valore scelto meno il riferimento") for r in rivelate)
    tentativi = [r for r in rivelate if r.startswith("Precisione, la direzione, serie 1, tentativo ")]
    assert len(tentativi) == 8
    medio = statistics.fmean(scarti)
    riepilogo = next(r for r in rivelate if r.startswith("Precisione, la direzione, riepilogo: "))
    assert f"8 tentativi fermati nella scala su 8; errore medio +{bm.numero(medio * 100, 1)}%" in riepilogo
    assert not any(r.startswith("Precisione, tutti i parametri") for r in rivelate)
    assert all(r.strip() and not r.startswith(("-", "=", "_")) for r in righe)

    scritto = capsys.readouterr().out
    assert scritto.startswith("Banco della mira di Ciechino: 5 gruppi alla cieca, ")
    assert scritto.rstrip().endswith("trovi i voti, quale lettera era quale e il riepilogo.")
    assert scritto.count("; n se non hai preferenze.\n") == 2
    _schermo_pulito(scritto, nomi)


def test_senza_il_via_non_suona_niente_e_il_seme(monkeypatch, capsys, tmp_path):
    for banco in BANCHI:
        monkeypatch.setattr(banco, "prepara", lambda seme, materiali=None: [])
        monkeypatch.setattr(banco, "enter_escape", _enter_escape_finta([False]))
        assert banco.main([]) == 0
        assert banco.main(["quattro"]) == 1
    scritto = capsys.readouterr().out
    assert scritto.count("Il seme dev'essere un numero intero, e quattro non lo è.") == 3
    assert "\n\n" not in scritto
    assert list(tmp_path.iterdir()) == []


def test_senza_scheda_il_banco_lo_dice(monkeypatch, capsys, tmp_path):
    # Il conftest tiene chiusa la scheda vera: il motore lavorerebbe muto, e un banco d'ascolto muto non serve.
    assert PREPARA_ASCOLTO_VERA() is None
    monkeypatch.setattr(bm, "prepara_ascolto", PREPARA_ASCOLTO_VERA)
    assert bm.sessione("Banco di prova", SEME, tmp_path / "esiti.txt", []) == 1
    assert capsys.readouterr().out == "La scheda audio non si apre: senza suono il banco non serve, e si chiude.\n"
    assert list(tmp_path.iterdir()) == []


# I bersagli.

@pytest.fixture(scope="module")
def materiali_dei_bersagli():
    """I materiali di tutti i gruppi dei bersagli, fatti una volta per il modulo."""
    return {g.chiave: [bb.componi(c.ricetta) for c in g.candidati] for g in bb.GRUPPI}


def test_il_flusso_in_memoria_rende_cio_che_il_motore_suona():
    bersaglio = Bersaglio(numero=0, forma="deriva", raggio=2.0, partenza=bb.PUNTO_DI_PROVA, frequenza_lfo=3.0, seme=5)
    blocchi = 20
    with bm.in_memoria() as (motore, flusso):
        motore.aggiungi(VoceTono(bersaglio.parametri_sonori, "organo", 0))
        in_memoria = flusso.rendi(blocchi * motore.blocco)
    vero, finto = motore_finto(latenza=0.0)
    try:
        vero.aggiungi(VoceTono(bersaglio.parametri_sonori, "organo", 0))
        assert np.array_equal(finto.avanza(blocchi), in_memoria)
    finally:
        vero.chiudi()
    assert motore.chiuso and np.any(in_memoria)


def _centro_in_ottave(mono, centro):
    """Di quante ottave il baricentro dello spettro, nella mezza ottava attorno a centro, sta sopra
    centro: le armoniche stanno fuori, e l'oscillazione dell'LFO, simmetrica, si annulla."""
    mono = np.asarray(mono, dtype=np.float64)
    spettro = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    frequenze = np.fft.rfftfreq(len(mono), 1.0 / FS)
    banda = (frequenze > centro / math.sqrt(2.0)) & (frequenze < centro * math.sqrt(2.0))
    ottave = np.log2(frequenze[banda] / centro)
    pesi = spettro[banda] / np.sum(spettro[banda])
    media = float(np.sum(pesi * ottave))
    return media, math.sqrt(float(np.sum(pesi * (ottave - media) ** 2)))


def test_i_gruppi_dei_bersagli(materiali_dei_bersagli):
    forme, grandezze, portanti = bb.GRUPPI
    assert [c.ricetta["forma"] for c in forme.candidati] == list(costanti.FORME_LFO)
    assert not any(c.attuale for c in (*forme.candidati, *grandezze.candidati)) and not grandezze.preferenza
    assert [c.ricetta["raggio"] for c in grandezze.candidati] == [5.0, 2.5, 1.0, 0.5]
    assert [c.ricetta["portante"] for c in portanti.candidati if c.attuale] == [costanti.BERSAGLIO_FORMA]
    for gruppo in bb.GRUPPI:
        materiali = materiali_dei_bersagli[gruppo.chiave]
        assert len({len(m) for m in materiali}) == 1
        for i, materiale in enumerate(materiali):
            assert materiale.dtype == np.float32 and materiale.shape[1] == 2
            # Entra e esce senza click, e sta davanti al cannone: i due canali uguali.
            assert float(np.max(np.abs(materiale[:3]))) < 1e-3 and float(np.max(np.abs(materiale[-3:]))) < 1e-3
            assert np.allclose(materiale[:, 0], materiale[:, 1], atol=1e-6)
            for altro in materiali[i + 1:]:
                assert not np.array_equal(materiale, altro), gruppo.chiave


def test_la_grandezza_allarga_l_oscillazione(materiali_dei_bersagli):
    centro = fisica.frequenza_di(bb.PUNTO_DI_PROVA[1])
    larghezze = []
    for materiale in materiali_dei_bersagli["grandezze"]:
        media, larghezza = _centro_in_ottave(materiale[round(0.5 * FS):-round(0.5 * FS), 0], centro)
        assert abs(media) < 0.03
        larghezze.append(larghezza)
    assert larghezze == sorted(larghezze, reverse=True)
    # Col seno, lo scarto delle ottave è la mezza escursione divisa per radice di due.
    assert larghezze[0] == pytest.approx(5.0 * costanti.FATTORE_ESCURSIONE / costanti.METRI_PER_OTTAVA / math.sqrt(2.0), rel=0.15)
    assert larghezze[0] > 4 * larghezze[-1]


def test_le_portanti_in_basso_e_in_alto(materiali_dei_bersagli):
    meta = round(bb.DURATA_DI_UN_PUNTO * FS)
    attese = [fisica.frequenza_di(p[1]) for p in bb.PUNTI_DELLE_PORTANTI]
    assert attese[1] > 4 * attese[0]
    for materiale in materiali_dei_bersagli["portanti"]:
        for tratto, attesa in zip((materiale[:meta, 0], materiale[-meta:, 0]), attese, strict=True):
            media, _larghezza = _centro_in_ottave(tratto, attesa)
            assert abs(media) < 0.03
        assert not np.any(materiale[meta:len(materiale) - meta])


def test_la_sessione_dei_bersagli_interrotta(ascolto, materiali_dei_bersagli, monkeypatch, capsys, tmp_path):
    prove = bb.prepara(SEME, materiali_dei_bersagli)
    forme = prove[0]
    assert len(forme.lettere) == 8 and forme.materiale(forme.lettere_di(forme.ripetuto)[0]) is forme.materiale(forme.lettere_di(forme.ripetuto)[1])
    monkeypatch.setattr(bb, "prepara", lambda seme, materiali=None: prove)
    monkeypatch.setattr(bm, "prepara_ascolto", lambda eventi=(): ascolto)
    monkeypatch.setattr(bb, "enter_escape", _enter_escape_finta([True]))
    monkeypatch.setattr(collaudo_comune, "enter_escape", _enter_escape_finta([True]))
    # Ctrl+C mentre suona la seconda lettera della rassegna delle forme.
    tastiera = _tastiera(monkeypatch, ascolto, [("durante", 2, KeyboardInterrupt())])
    assert bb.main([str(SEME)]) == 1
    assert tastiera.copione == [] and ascolto.motore.chiuso
    righe = (tmp_path / "banco_bersagli_esiti.txt").read_text(encoding="utf-8").splitlines()
    titolo = forme.gruppo.titolo
    assert righe[0].startswith("Banco dei bersagli, ") and righe[0].endswith(f", seme {SEME}.")
    assert righe[1:10] == [f"{titolo}, lettera {x}: senza voto." for x in forme.lettere] + [f"{titolo}: gruppo interrotto."]
    assert sum(1 for r in righe if r.startswith(f"{titolo}, la lettera ")) == 8
    assert f"Scelta, {titolo}: senza voti nessuna versione ha la preferenza." in righe
    assert not any(r.startswith(f"{p.gruppo.titolo}, ") for r in righe for p in prove[1:])
    scritto = capsys.readouterr().out
    assert "Banco interrotto." in scritto
    _schermo_pulito(scritto, _nomi(bb.GRUPPI))


def test_le_grandezze_si_riepilogano_per_raggio(materiali_dei_bersagli):
    prova = bb.prepara(SEME, materiali_dei_bersagli)[1]
    prova.iniziata = prova.svolta = True
    for lettera in prova.lettere:
        prova.voti[lettera] = 5 - prova.indice(lettera) if prova.indice(lettera) else 1
    righe = bm.righe_della_rivelazione([prova])
    titolo = prova.gruppo.titolo
    riepilogo = [r for r in righe if r.startswith(f"Riepilogo, {titolo}, ")]
    assert len(riepilogo) == len(prova.gruppo.candidati)
    for riga, candidato in zip(riepilogo, prova.gruppo.candidati, strict=True):
        assert riga.startswith(f"Riepilogo, {titolo}, {candidato.nome}: totale "), riga
    assert not any(r.startswith("Scelta, ") for r in righe)


# Il volo.

@pytest.fixture(scope="module")
def materiali_del_volo():
    """I materiali dei gruppi del volo, fatti una volta per il modulo coi preset finti."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(suoni, "_rendi_preset", _preset_finto)
        return {g.chiave: [bv.componi(c.ricetta) for c in g.candidati] for g in bv.GRUPPI}


def test_i_quattro_tiri_fanno_cio_che_promettono():
    bersaglio, tiri = bv.tiri_di_prova()
    assert [t.chiave for t in tiri] == list(bv.TIRI)
    centro, corto, lungo, largo = (t.volo for t in tiri)
    distanza = math.hypot(bersaglio.partenza[0], bersaglio.partenza[2])
    assert centro.colpito is bersaglio and centro.precisione > 0.9
    for volo in (corto, lungo, largo):
        assert volo.colpito is None and volo.vicino is bersaglio
    assert corto.gittata < distanza - bersaglio.raggio and corto.scarto[0] < 0
    assert lungo.gittata > distanza + bersaglio.raggio and lungo.scarto[0] > 0
    assert largo.scarto[1] > bersaglio.raggio
    for tiro in tiri:
        assert all(costanti.MARGINE_DI_COPERTURA <= v <= 1 - costanti.MARGINE_DI_COPERTURA for v in tiro.valori)


def test_la_caduta_torna_sempre_com_era():
    com_era = fisica.DB_PER_METRO
    with bv.caduta(0.4):
        assert fisica.DB_PER_METRO == 0.4
        assert fisica.guadagno_db_di(10.0) == pytest.approx(costanti.GUADAGNO_DB_VICINO - 4.0)
    assert com_era == fisica.DB_PER_METRO
    with pytest.raises(RuntimeError), bv.caduta(0.4):
        raise RuntimeError("qualcosa va storto")
    assert fisica.DB_PER_METRO == com_era == costanti.DB_PER_METRO


def test_i_materiali_del_volo(materiali_del_volo):
    assert fisica.DB_PER_METRO == costanti.DB_PER_METRO
    timbro, caduta = bv.GRUPPI
    assert [c.ricetta["forma"] for c in timbro.candidati if c.attuale] == [costanti.PROIETTILE_FORMA]
    assert [c.ricetta["db_per_metro"] for c in caduta.candidati] == [costanti.DB_PER_METRO, bv.DB_PER_METRO_RIPIDO]
    sparo = round(bv.ASCOLTO_PRIMA_DEL_TIRO * FS)
    materiali = materiali_del_volo["timbro"]
    # Prima dello sparo c'è soltanto il bersaglio, uguale per tutti; dopo cambia il proiettile.
    for altro in materiali[1:]:
        assert np.array_equal(altro[:sparo], materiali[0][:sparo]) and not np.array_equal(altro, materiali[0])
    # La caduta più ripida abbassa il bersaglio di tanti decibel quanti dice la differenza per la distanza.
    vicino, ripido = materiali_del_volo["caduta"]
    tratto = slice(round(0.3 * FS), sparo - round(0.1 * FS))
    distanza = fisica.distanza_dal_cannone(bv.PUNTO_DEL_BERSAGLIO)
    assert _db(vicino[tratto]) - _db(ripido[tratto]) == pytest.approx((bv.DB_PER_METRO_RIPIDO - costanti.DB_PER_METRO) * distanza, abs=0.3)
    # La caduta attuale usa il timbro attuale: è lo stesso materiale della sua lettera nel timbro.
    attuale, = (indice for indice, candidato in enumerate(timbro.candidati) if candidato.attuale)
    assert np.array_equal(vicino, materiali[attuale])
    # Nel centro il bersaglio tace dall'impatto, finito l'effetto; nel corto continua fino in fondo.
    bersaglio, tiri = bv.tiri_di_prova()
    arrivo = sparo + math.ceil(tiri[0].volo.durata / costanti.VELOCITA_DEL_TEMPO * FS)
    fine = arrivo + round(bv.CODA_DEL_TIRO * FS)
    margine = LUNGHEZZA_FINTA + round(0.02 * FS)
    assert not np.any(vicino[arrivo + margine:fine - 1])
    assert np.any(vicino[arrivo - round(0.05 * FS):arrivo + LUNGHEZZA_FINTA])
    inizio_del_corto = fine + round(bv.PAUSA_FRA_I_TIRI * FS)
    assert not np.any(vicino[fine:inizio_del_corto])
    arrivo_corto = inizio_del_corto + sparo + math.ceil(tiri[1].volo.durata / costanti.VELOCITA_DEL_TEMPO * FS)
    assert np.any(vicino[arrivo_corto + margine:arrivo_corto + round(bv.CODA_DEL_TIRO * FS) - round(0.1 * FS)])
    assert bersaglio.raggio == bv.raggio_del_livello(bv.LIVELLO_DEL_BERSAGLIO)


def test_la_sessione_del_volo(ascolto, materiali_del_volo, monkeypatch, capsys, tmp_path):
    prove = bv.prepara(SEME, materiali_del_volo)
    timbro, caduta = prove
    monkeypatch.setattr(bv, "prepara", lambda seme, materiali=None: prove)
    monkeypatch.setattr(bm, "prepara_ascolto", lambda eventi=(): ascolto)
    monkeypatch.setattr(bv, "enter_escape", _enter_escape_finta([True]))
    monkeypatch.setattr(collaudo_comune, "enter_escape", _enter_escape_finta([False, True, True]))
    monkeypatch.setattr(collaudo_comune, "dgt", lambda *_a, **_k: "lo sparo è troppo secco")
    voti = dict(zip(caduta.lettere, (4, 2, 3), strict=True))
    # Il timbro si salta; la caduta: Escape appena parte la prima lettera della rassegna, i voti, Invio
    # nel menu; gli effetti si sentono tutti, si commentano e si chiudono con Invio.
    copione = [("durante", 1, "\x1b")] + [str(voti[x]) for x in caduta.lettere] + ["\r", "c", "\r"]
    tastiera = _tastiera(monkeypatch, ascolto, copione)
    assert bv.main([str(SEME)]) == 0
    assert tastiera.copione == [] and ascolto.motore.chiuso
    assert caduta.voti == voti and timbro.saltata
    effetti = _materiali_partiti(ascolto)[1:]
    assert ascolto.partiti == 5 and len(effetti) == 4
    assert not np.any(effetti[0].buffer[:round(bm.ANTICIPO_PRIMA_LETTERA * FS)])
    assert np.array_equal(effetti[0].buffer[round(bm.ANTICIPO_PRIMA_LETTERA * FS):], _preset_finto(suoni.EVENTI["sparo"]))
    righe = (tmp_path / "banco_volo_esiti.txt").read_text(encoding="utf-8").splitlines()
    assert righe[0].startswith("Banco del volo, ") and righe[0].endswith(f", seme {SEME}.")
    prima_rivelazione = next(i for i, r in enumerate(righe) if " era: " in r)
    davanti = righe[1:prima_rivelazione]
    assert davanti[0] == f"{timbro.gruppo.titolo}: gruppo saltato."
    assert davanti[1:4] == [f"{caduta.gruppo.titolo}, lettera {x}: {voti[x]}." for x in caduta.lettere]
    assert davanti[4].endswith(": test superato, nessun commento")
    assert davanti[5] == ("Volo, gli effetti: sparo col preset ciechino_sparo, colpito col preset ciechino_colpito, colpito_bonus col preset "
                          "ciechino_colpito_bonus e a_terra col preset ciechino_a_terra.")
    assert davanti[6].startswith("Volo, gli effetti, ") and davanti[6].endswith(": lo sparo è troppo secco")
    nomi = _nomi(bv.GRUPPI)
    assert not any(nome in riga for riga in davanti for nome in nomi)
    rivelate = righe[prima_rivelazione:]
    assert sum(1 for r in rivelate if r.startswith(f"{caduta.gruppo.titolo}, la lettera ")) == 3
    scelto = caduta.gruppo.candidati[bm.scelta(caduta)].nome
    assert f"Scelta, {caduta.gruppo.titolo}: {scelto}." in rivelate
    scritto = capsys.readouterr().out
    assert [riga for riga in scritto.splitlines() if riga in bv.NOMI_DEGLI_EFFETTI.values()] == [bv.NOMI_DEGLI_EFFETTI[e] for e in bv.EFFETTI]
    assert all(_prompt_giusto(p) for p in tastiera.prompt)
    _schermo_pulito(scritto, nomi)
