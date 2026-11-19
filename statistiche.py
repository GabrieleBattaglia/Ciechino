"""
Le statistiche di Ciechino: i conti di ogni caratteristica del gioco, raccolti giocando, e la pagina che li racconta.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di dati.py di PokerMachine, per nuovi_dati, completa e
il caricamento difensivo, e di statistiche.py di MeTeOra, per la pagina: ogni sezione si apre con
«Sezione n di N», poi una voce per riga nella forma «Etichetta: valore.», e le voci a zero si
omettono. Gabriele ama le statistiche e vuole un contatore per ogni caratteristica nuova.

I conti arrivano da sette strade:
- carica(), che conta l'avvio di Ciechino;
- conta(chiave, quanti), per i contatori semplici che solo la partita conosce: partite_iniziate,
  mire_avviate, livelli_superati, colpi_guadagnati, bonus_usciti, e qualunque chiave nuova, che la
  pagina dice nella sezione degli altri conti finché non ha un'etichetta sua;
- suono(evento), per ogni evento sonoro, contato per nome;
- bersaglio_apparso(forma, mobile, bonus), per ogni bersaglio che nasce;
- mira(parametro, valore, esito_della_scivolata), per ogni valore preso con le scivolate;
- tiro(valori, esito), dopo ogni tiro;
- partita_finita(riepilogo), alla fine di ogni partita, finita a riserva vuota o abbandonata.
tiro e partita_finita restituiscono le chiavi dei primati appena battuti, perché la partita possa
dirlo; il primo valore di un primato non è un primato battuto.

Il file è «Ciechino - Statistiche.json», accanto al programma, scritto con scrivi_dati di
classifica.py: un file rovinato non solleva mai, si ripiega sulla copia .bak, e il file si rinomina
in .illeggibile, mai sovrascritto, ripartendo da zero con un avviso.
"""

import math
from datetime import datetime

import costanti
import percorsi
import testi
from classifica import leggi_dati, scrivi_dati
from lingua import _, plurale

# L'istogramma dei valori presi con ogni scivolata ha tante classi quante sono le fasce a decimi
# della scala: è una scelta della pagina, non una taratura del gioco, e un cambio rimette a zero gli
# istogrammi salvati con un altro numero di classi.
CLASSI = 10
# I contatori che la pagina conosce: nuovi_dati li parte da zero, e la sezione degli altri conti
# dice soltanto le chiavi che non stanno qui.
CHIAVI_DEI_CONTI = ("avvii", "partite_iniziate", "partite_finite", "partite_abbandonate", "tiri", "centri", "tiri_a_vuoto", "punti", "bonus_apparsi",
                    "bonus_colpiti", "bonus_usciti", "livelli_superati", "colpi_guadagnati", "mire_avviate")
# Come può finire una scivolata, da mira.valore_scelto: fermata nei bip o subito dopo, scelta nella
# scala, arrivata al massimo senza lo spazio.
ESITI_DELLA_SCIVOLATA = ("minimo", "scelto", "massimo")
# Come può finire una partita, dal riepilogo.
MOTIVI = ("riserva_vuota", "abbandonata")
# I primati: quelli in cui vince il valore più alto e quelli in cui vince il più basso, nell'ordine
# in cui la pagina li dice.
PRIMATI_ALTI = ("punteggio", "livello", "serie", "precisione", "punti_in_un_tiro", "centri_in_una_partita", "partita_piu_lunga")
PRIMATI_BASSI = ("vicino", "mancato")

NOMI_DEI_PARAMETRI = {"direzione": _("direzione"), "alzo": _("alzo"), "potenza": _("potenza")}
NOMI_DELLE_FORME = {
    # Traduttori: da qui, le forme dell'oscillazione che fa vibrare il suono di un bersaglio.
    "seno": _("seno"), "sega": _("dente di sega"), "sega_inversa": _("dente di sega inverso"), "triangolo": _("triangolo"), "quadra": _("onda quadra"),
    "casuale": _("casuale"), "deriva": _("deriva"),
}


# I numeri del file.

def _numero(valore):
    """Vero per un numero vero, non booleano e finito."""
    return isinstance(valore, int | float) and not isinstance(valore, bool) and math.isfinite(valore)


def _contatore(valore):
    return round(valore) if _numero(valore) and valore >= 0 else 0


def _quantita(valore):
    return round(float(valore), 3) if _numero(valore) and valore >= 0 else 0.0


def _numeri(valore):
    """Un dizionario di numeri non negativi, senza le voci sbagliate: le chiavi sconosciute restano."""
    if not isinstance(valore, dict):
        return {}
    return {str(k): v for k, v in valore.items() if _numero(v) and v >= 0}


def _somma(attuale, quanto):
    """Gli interi restano interi; i decimali si tengono al millesimo, perché il file non cresca di cifre inutili."""
    valore = attuale + quanto
    return round(valore, 3) if isinstance(valore, float) else valore


def _data_valida(testo):
    return isinstance(testo, str) and testi.leggi_data(testo) is not None


# La struttura.

def _nuova_mira():
    return {"minimo": 0, "scelto": 0, "massimo": 0, "classi": [0] * CLASSI, "somma": 0.0, "nei_centri": 0, "somma_nei_centri": 0.0}


def _nuova_coppia():
    return {"apparsi": 0, "colpiti": 0}


def nuovi_dati(adesso):
    """Le statistiche di chi comincia adesso, un istante ISO."""
    return {
        "formato": costanti.FORMATO_STATISTICHE,
        "dal": adesso,
        "salvato": None,
        "conti": dict.fromkeys(CHIAVI_DEI_CONTI, 0),
        "tempo_di_gioco": 0.0,
        # Le somme per le medie: la precisione dei centri che l'hanno, i punteggi delle partite.
        "somme": {"precisione": 0.0, "centri_misurati": 0, "punteggi": 0},
        "mira": {parametro: _nuova_mira() for parametro in costanti.PARAMETRI},
        "forme": {forma: _nuova_coppia() for forma in costanti.FORME_LFO},
        "moto": {"fermi": _nuova_coppia(), "mobili": _nuova_coppia()},
        "primati": {},
        "ultime": [],
        "suoni": {},
    }


def _coppia_pulita(valore):
    coppia = _nuova_coppia()
    if isinstance(valore, dict):
        for chiave in coppia:
            coppia[chiave] = _contatore(valore.get(chiave))
    return coppia


def _mira_pulita(valore):
    mira = _nuova_mira()
    if not isinstance(valore, dict):
        return mira
    for chiave in ("minimo", "scelto", "massimo", "nei_centri"):
        mira[chiave] = _contatore(valore.get(chiave))
    for chiave in ("somma", "somma_nei_centri"):
        mira[chiave] = _quantita(valore.get(chiave))
    classi = valore.get("classi")
    if isinstance(classi, list) and len(classi) == CLASSI:
        mira["classi"] = [_contatore(c) for c in classi]
    return mira


def _partita_pulita(valore):
    if not isinstance(valore, dict) or not _data_valida(valore.get("quando")):
        return None
    motivo = valore.get("motivo")
    return {"quando": valore["quando"], "punteggio": _contatore(valore.get("punteggio")), "livello": _contatore(valore.get("livello")),
            "tiri": _contatore(valore.get("tiri")), "centri": _contatore(valore.get("centri")), "durata_s": _quantita(valore.get("durata_s")),
            "motivo": motivo if motivo in MOTIVI else MOTIVI[0]}


def completa(grezzi, adesso):
    """Le statistiche lette dal file, risanate: ciò che manca o ha il tipo sbagliato torna al
    predefinito, le chiavi nuove dei contatori, delle forme e dei suoni restano. Restituisce un
    dizionario nuovo e non tocca quello ricevuto."""
    dati = nuovi_dati(adesso)
    if not isinstance(grezzi, dict):
        return dati
    for chiave in ("dal", "salvato"):
        if _data_valida(grezzi.get(chiave)):
            dati[chiave] = grezzi[chiave]
    dati["conti"].update(_numeri(grezzi.get("conti")))
    dati["tempo_di_gioco"] = _quantita(grezzi.get("tempo_di_gioco"))
    somme = grezzi.get("somme") if isinstance(grezzi.get("somme"), dict) else {}
    dati["somme"] = {"precisione": _quantita(somme.get("precisione")), "centri_misurati": _contatore(somme.get("centri_misurati")),
                     "punteggi": _contatore(somme.get("punteggi"))}
    if isinstance(grezzi.get("mira"), dict):
        for parametro, valore in grezzi["mira"].items():
            dati["mira"][str(parametro)] = _mira_pulita(valore)
    if isinstance(grezzi.get("forme"), dict):
        for forma, valore in grezzi["forme"].items():
            dati["forme"][str(forma)] = _coppia_pulita(valore)
    if isinstance(grezzi.get("moto"), dict):
        for moto in dati["moto"]:
            dati["moto"][moto] = _coppia_pulita(grezzi["moto"].get(moto))
    if isinstance(grezzi.get("primati"), dict):
        dati["primati"] = {str(k): {"valore": v["valore"], "quando": v["quando"]} for k, v in grezzi["primati"].items()
                           if isinstance(v, dict) and _numero(v.get("valore")) and _data_valida(v.get("quando"))}
    if isinstance(grezzi.get("ultime"), list):
        partite = [_partita_pulita(v) for v in grezzi["ultime"]]
        dati["ultime"] = [p for p in partite if p is not None][-costanti.ULTIME_PARTITE:]
    dati["suoni"] = {k: _contatore(v) for k, v in _numeri(grezzi.get("suoni")).items()}
    return dati


def _controlla(grezzi, adesso):
    if not isinstance(grezzi, dict):
        raise ValueError(_("il file non contiene le statistiche"))
    return completa(grezzi, adesso)


# Le parole della pagina.

def _maiuscola(testo):
    return testo[:1].upper() + testo[1:]


def _nome_da_chiave(chiave):
    """Una chiave senza nome nella pagina, detta com'è, con gli spazi al posto delle sottolineature."""
    return _maiuscola(str(chiave).replace("_", " "))


def _con_parte(n, tutto, frase_con, frase_senza):
    """La voce di una parte, con la sua percentuale sul tutto quando il tutto non è zero."""
    if tutto:
        return frase_con.format(n=testi.intero(n), percentuale=testi.percento(n / tutto))
    return frase_senza.format(n=testi.intero(n))


class Statistiche:
    """I conti e il loro file. percorso None vuol dire il file predefinito, calcolato a ogni
    salvataggio; ora dà l'istante di adesso, e le prove la sostituiscono."""

    def __init__(self, percorso=None, dati=None, ora=None):
        self.percorso = percorso
        self._ora = ora or datetime.now
        self.dati = nuovi_dati(self._iso()) if dati is None else completa(dati, self._iso())

    def _iso(self):
        return self._ora().isoformat(timespec="seconds")

    def _file(self):
        return self.percorso or percorsi.percorso_dati(costanti.FILE_STATISTICHE)

    # Il file.

    @classmethod
    def carica(cls, percorso=None, avvio=True, ora=None):
        """Le statistiche dal file e un avviso, None se è andato tutto bene: restituisce
        (statistiche, avviso) e non solleva mai. Con avvio vero, il predefinito, conta un avvio
        di Ciechino: si chiama una volta all'apertura."""
        statistiche = cls(percorso, ora=ora)
        adesso = statistiche._iso()
        dati, avviso = leggi_dati(statistiche._file(), lambda grezzi: _controlla(grezzi, adesso))
        if dati is not None:
            statistiche.dati = dati
        if avvio:
            statistiche.conta("avvii")
        return statistiche, avviso

    def salva(self):
        """Scrive il file; vero se è riuscito."""
        self.dati["salvato"] = self._iso()
        return scrivi_dati(self.dati, self._file())

    # I conti.

    def conta(self, chiave, quanti=1):
        """Aggiunge quanti al contatore chiave; una chiave nuova nasce da zero."""
        conti = self.dati["conti"]
        conti[chiave] = _somma(conti.get(chiave, 0), quanti)

    def suono(self, evento):
        """Un evento sonoro suonato, contato per nome."""
        suoni = self.dati["suoni"]
        suoni[str(evento)] = suoni.get(str(evento), 0) + 1

    def _forma(self, forma):
        return self.dati["forme"].setdefault(str(forma), _nuova_coppia())

    def _mira(self, parametro):
        return self.dati["mira"].setdefault(str(parametro), _nuova_mira())

    def _primato(self, chiave, valore, piu_alto=True):
        """Registra il valore se batte il primato; vero se c'era un primato, e l'ha battuto."""
        primati = self.dati["primati"]
        attuale = primati.get(chiave)
        if attuale is not None and not (valore > attuale["valore"] if piu_alto else valore < attuale["valore"]):
            return False
        primati[chiave] = {"valore": valore, "quando": self._iso()}
        return attuale is not None

    def bersaglio_apparso(self, forma, mobile, bonus):
        """Un bersaglio nato, con la forma del suo LFO, fermo o mobile, normale o bonus."""
        self._forma(forma)["apparsi"] += 1
        self.dati["moto"]["mobili" if mobile else "fermi"]["apparsi"] += 1
        if bonus:
            self.conta("bonus_apparsi")

    def mira(self, parametro, valore, esito_della_scivolata):
        """Un valore preso con una scivolata: esito «minimo», «scelto» o «massimo», come lo dà
        mira.valore_scelto. Un esito diverso, come la pausa, non è un valore e non si conta."""
        if esito_della_scivolata not in ESITI_DELLA_SCIVOLATA:
            return
        mira = self._mira(parametro)
        mira[esito_della_scivolata] += 1
        if _numero(valore):
            v = min(1.0, max(0.0, float(valore)))
            mira["classi"][min(CLASSI - 1, int(v * CLASSI))] += 1
            mira["somma"] = round(mira["somma"] + v, 6)

    def _valori_nei_centri(self, valori):
        """I tre valori di un centro, per il valore medio dei centri di ogni parametro: una
        sequenza nell'ordine di PARAMETRI o un dizionario per parametro."""
        if isinstance(valori, dict):
            coppie = valori.items()
        elif isinstance(valori, list | tuple):
            coppie = zip(costanti.PARAMETRI, valori, strict=False)
        else:
            return
        for parametro, valore in coppie:
            if _numero(valore):
                mira = self._mira(parametro)
                mira["nei_centri"] += 1
                mira["somma_nei_centri"] = round(mira["somma_nei_centri"] + min(1.0, max(0.0, float(valore))), 6)

    def tiro(self, valori, esito):
        """Un tiro sparato. valori sono la direzione, l'alzo e la potenza, da 0 a 1; esito è
        {"colpito", "bonus", "forma", "mobile", "precisione", "punti", "distanza_dal_vicino",
        "serie"}, con precisione e distanza None quando non ci sono. Restituisce i primati battuti."""
        e = esito if isinstance(esito, dict) else {}
        battuti = []
        self.conta("tiri")
        colpito = bool(e.get("colpito"))
        if colpito:
            self.conta("centri")
            if e.get("bonus"):
                self.conta("bonus_colpiti")
            if isinstance(e.get("forma"), str):
                self._forma(e["forma"])["colpiti"] += 1
            self.dati["moto"]["mobili" if e.get("mobile") else "fermi"]["colpiti"] += 1
            precisione = e.get("precisione")
            if _numero(precisione):
                p = min(1.0, max(0.0, float(precisione)))
                somme = self.dati["somme"]
                somme["precisione"] = round(somme["precisione"] + p, 6)
                somme["centri_misurati"] += 1
                if self._primato("precisione", p):
                    battuti.append("precisione")
            punti = e.get("punti")
            if _numero(punti) and punti > 0:
                self.conta("punti", round(punti))
                if self._primato("punti_in_un_tiro", round(punti)):
                    battuti.append("punti_in_un_tiro")
            self._valori_nei_centri(valori)
        else:
            self.conta("tiri_a_vuoto")
        distanza = e.get("distanza_dal_vicino")
        if _numero(distanza) and distanza >= 0:
            if self._primato("vicino", distanza, piu_alto=False):
                battuti.append("vicino")
            if not colpito and self._primato("mancato", distanza, piu_alto=False):
                battuti.append("mancato")
        serie = e.get("serie")
        if _numero(serie) and serie > 0 and self._primato("serie", round(serie)):
            battuti.append("serie")
        return battuti

    def partita_finita(self, riepilogo):
        """Una partita conclusa: riepilogo è {"punteggio", "livello", "tiri", "centri",
        "durata_s", "motivo"}, con motivo «riserva_vuota» o «abbandonata». Restituisce i primati
        battuti."""
        r = riepilogo if isinstance(riepilogo, dict) else {}
        motivo = r.get("motivo") if r.get("motivo") in MOTIVI else MOTIVI[0]
        self.conta("partite_abbandonate" if motivo == "abbandonata" else "partite_finite")
        partita = {"quando": self._iso(), "punteggio": _contatore(r.get("punteggio")), "livello": _contatore(r.get("livello")),
                   "tiri": _contatore(r.get("tiri")), "centri": _contatore(r.get("centri")), "durata_s": _quantita(r.get("durata_s")), "motivo": motivo}
        self.dati["tempo_di_gioco"] = _somma(self.dati["tempo_di_gioco"], partita["durata_s"])
        self.dati["somme"]["punteggi"] += partita["punteggio"]
        battuti = []
        for chiave, valore in (("punteggio", partita["punteggio"]), ("livello", partita["livello"]), ("centri_in_una_partita", partita["centri"]),
                               ("partita_piu_lunga", partita["durata_s"])):
            if valore > 0 and self._primato(chiave, valore):
                battuti.append(chiave)
        self.dati["ultime"] = [*self.dati["ultime"], partita][-costanti.ULTIME_PARTITE:]
        return battuti

    # La pagina.

    def righe(self, oggi=None, nomi_dei_suoni=None):
        """Le righe della finestra delle statistiche: l'intestazione, l'elenco delle sezioni, poi
        ogni sezione aperta da «Sezione n di N», una voce per riga e niente righe vuote. oggi è
        l'istante da cui contare i giorni di raccolta, predefinito adesso; nomi_dei_suoni dà il
        nome a parole di ogni evento sonoro, come suoni.AZIONI, e senza si dice la chiave."""
        return Pagina(self, oggi or self._ora(), nomi_dei_suoni or {}).righe()


def carica(percorso=None, avvio=True):
    """Come Statistiche.carica: (statistiche, avviso)."""
    return Statistiche.carica(percorso, avvio=avvio)


class Pagina:
    """Le righe della finestra delle statistiche, sezione per sezione."""

    def __init__(self, statistiche, oggi, nomi_dei_suoni):
        self.d = statistiche.dati
        self.oggi = oggi
        self.nomi_dei_suoni = nomi_dei_suoni
        self.conti = self.d["conti"]

    def c(self, chiave):
        return self.conti.get(chiave, 0)

    def partite(self):
        return self.c("partite_finite") + self.c("partite_abbandonate")

    def sezioni(self):
        """I titoli delle sezioni con le loro righe. Una sezione facoltativa senza righe non c'è:
        gli altri conti compaiono soltanto quando qualcuno conta una chiave nuova."""
        sezioni = [
            # Traduttori: da qui, i titoli delle sezioni delle statistiche, con l'iniziale minuscola: «Sezione 1 di 8, in breve.».
            (_("in breve"), self.in_breve, False),
            (_("le partite"), self.le_partite, False),
            (_("i tiri"), self.i_tiri, False),
            (_("la mira"), self.la_mira, False),
            (_("i bersagli"), self.i_bersagli, False),
            (_("i primati"), self.i_primati, False),
            (_("le ultime partite"), self.le_ultime_partite, False),
            (_("i suoni"), self.i_suoni, False),
            (_("gli altri conti"), self.gli_altri_conti, True),
        ]
        composte = [(titolo, [r for r in funzione() if r], facoltativa) for titolo, funzione, facoltativa in sezioni]
        return [(titolo, righe) for titolo, righe, facoltativa in composte if righe or not facoltativa]

    def righe(self):
        sezioni = self.sezioni()
        dal = testi.leggi_data(self.d["dal"]) or self.oggi
        giorno = self.oggi.date() if isinstance(self.oggi, datetime) else self.oggi
        inizio = dal.date() if isinstance(dal, datetime) else dal
        giorni = max(1, (giorno - inizio).days + 1)
        righe = [
            # Traduttori: data è come «giovedì 19 novembre 2026», giorni come «5 giorni».
            _("Statistiche di {nome} {versione}, raccolte da {data}: {giorni}, oggi compreso.").format(nome=costanti.NOME, versione=costanti.VERSIONE,
                                                                                                    data=testi.data_lunga(self.d["dal"]),
                                                                                                    giorni=plurale(giorni, "{n} giorno", "{n} giorni",
                                                                                                                   come=testi.intero(giorni))),
            _("Le sezioni: {sezioni}.").format(sezioni="; ".join(f"{n} {_maiuscola(titolo)}" for n, (titolo, _r) in enumerate(sezioni, 1))),
        ]
        for n, (titolo, contenuto) in enumerate(sezioni, 1):
            righe.append(_("Sezione {n} di {totale}, {titolo}.").format(n=n, totale=len(sezioni), titolo=titolo))
            righe.extend(contenuto or [_("Ancora niente da dire.")])
        return righe

    def _voce(self, testo, n):
        """Una voce «Etichetta: n.», o niente se n è zero."""
        return testo.format(n=testi.intero(n)) if n else ""

    # 1.

    def in_breve(self):
        tiri, centri = self.c("tiri"), self.c("centri")
        primati = self.d["primati"]
        righe = [
            self._voce(_("Avvii di Ciechino: {n}."), self.c("avvii")),
            self._voce(_("Partite giocate: {n}."), self.partite()),
            _("Tempo di gioco: {durata}.").format(durata=testi.durata(self.d["tempo_di_gioco"])) if self.d["tempo_di_gioco"] else "",
            self._voce(_("Tiri: {n}."), tiri),
            _con_parte(centri, tiri, _("Centri: {n}, {percentuale} dei tiri."), _("Centri: {n}.")) if centri else "",
            self._voce(_("Punti fatti: {n}."), self.c("punti")),
        ]
        if "punteggio" in primati:
            righe.append(_("Punteggio più alto: {punti}.").format(punti=self._punti(primati["punteggio"]["valore"])))
        if "livello" in primati:
            righe.append(_("Livello più alto: {n}.").format(n=testi.intero(primati["livello"]["valore"])))
        return righe

    @staticmethod
    def _punti(n):
        return plurale(round(n), "{n} punto", "{n} punti", come=testi.intero(n))

    # 2.

    def le_partite(self):
        partite = self.partite()
        aperte = self.c("partite_iniziate") - partite
        righe = [
            self._voce(_("Partite iniziate: {n}."), self.c("partite_iniziate")),
            self._voce(_("Partite finite a riserva vuota: {n}."), self.c("partite_finite")),
            self._voce(_("Partite abbandonate: {n}."), self.c("partite_abbandonate")),
            self._voce(_("Partite iniziate e non concluse, perché sono in corso o il programma si è chiuso: {n}."), max(0, aperte)),
            _("Tempo di gioco: {durata}.").format(durata=testi.durata(self.d["tempo_di_gioco"])) if self.d["tempo_di_gioco"] else "",
        ]
        if partite:
            righe.append(_("Durata media di una partita: {durata}.").format(durata=testi.durata(self.d["tempo_di_gioco"] / partite)))
            righe.append(_("Punteggio medio di una partita: {punti}.").format(punti=self._punti(self.d["somme"]["punteggi"] / partite)))
        righe.append(self._voce(_("Livelli superati: {n}."), self.c("livelli_superati")))
        righe.append(self._voce(_("Colpi guadagnati: {n}."), self.c("colpi_guadagnati")))
        return righe

    # 3.

    def i_tiri(self):
        tiri, centri, vuoto = self.c("tiri"), self.c("centri"), self.c("tiri_a_vuoto")
        somme = self.d["somme"]
        righe = [
            self._voce(_("Tiri: {n}."), tiri),
            _con_parte(centri, tiri, _("Centri: {n}, {percentuale} dei tiri."), _("Centri: {n}.")) if centri else "",
            _con_parte(vuoto, tiri, _("Tiri a vuoto: {n}, {percentuale} dei tiri."), _("Tiri a vuoto: {n}.")) if vuoto else "",
        ]
        if somme["centri_misurati"]:
            righe.append(_("Precisione media dei centri: {percentuale}.").format(percentuale=testi.percento(somme["precisione"] / somme["centri_misurati"])))
        if centri and self.c("punti"):
            righe.append(_("Punti per centro, in media: {n}.").format(n=testi.numero(self.c("punti") / centri)))
        righe.append(self._voce(_("Mire avviate: {n}."), self.c("mire_avviate")))
        return righe

    # 4.

    def la_mira(self):
        mire = self.d["mira"]
        ordine = [p for p in costanti.PARAMETRI if p in mire] + sorted(p for p in mire if p not in costanti.PARAMETRI)
        righe = []
        for parametro in ordine:
            righe.extend(self._righe_del_parametro(parametro, mire[parametro]))
        return righe

    def _righe_del_parametro(self, parametro, m):
        nome = _maiuscola(NOMI_DEI_PARAMETRI.get(parametro, parametro.replace("_", " ")))
        presi = m["minimo"] + m["scelto"] + m["massimo"]
        if not presi:
            return []
        righe = [
            # Traduttori: da qui, le righe della mira; parametro è direzione, alzo o potenza, con l'iniziale maiuscola.
            _("{parametro}, valori presi: {n}.").format(parametro=nome, n=testi.intero(presi)),
        ]
        for chiave, frase in (("scelto", _("{parametro}, scelti con lo spazio nella scivolata: {n}, {percentuale}.")),
                              ("minimo", _("{parametro}, fermati al minimo nei bip: {n}, {percentuale}.")),
                              ("massimo", _("{parametro}, arrivati al massimo senza lo spazio: {n}, {percentuale}."))):
            if m[chiave]:
                righe.append(frase.format(parametro=nome, n=testi.intero(m[chiave]), percentuale=testi.percento(m[chiave] / presi)))
        valori = sum(m["classi"])
        if valori:
            righe.append(_("{parametro}, valore medio: {percentuale} della scala.").format(parametro=nome, percentuale=testi.percento(m["somma"] / valori)))
        if m["nei_centri"]:
            righe.append(_("{parametro}, valore medio nei centri: {percentuale} della scala.").format(parametro=nome,
                                                                                                  percentuale=testi.percento(m["somma_nei_centri"] / m["nei_centri"])))
        fasce = [
            # Traduttori: una fascia della scala con quanti valori ci sono caduti: «da 10 a 20%: 5».
            _("da {da} a {a}%: {n}").format(da=testi.intero(k * 100 / CLASSI), a=testi.intero((k + 1) * 100 / CLASSI), n=testi.intero(quanti))
            for k, quanti in enumerate(m["classi"]) if quanti
        ]
        if fasce:
            righe.append(_("{parametro}, valori per fasce della scala: {fasce}.").format(parametro=nome, fasce="; ".join(fasce)))
        return righe

    # 5.

    @staticmethod
    def _apparsi_e_colpiti(etichetta, coppia):
        apparsi, colpiti = coppia["apparsi"], coppia["colpiti"]
        if not apparsi and not colpiti:
            return ""
        parti = [plurale(apparsi, "{n} apparso", "{n} apparsi", come=testi.intero(apparsi)),
                 plurale(colpiti, "{n} colpito", "{n} colpiti", come=testi.intero(colpiti))]
        if apparsi:
            parti.append(testi.percento(colpiti / apparsi))
        # Traduttori: una riga dei bersagli: «Bersagli fermi: 12 apparsi, 5 colpiti, 42%.».
        return _("{etichetta}: {dati}.").format(etichetta=etichetta, dati=", ".join(parti))

    def i_bersagli(self):
        moto, forme = self.d["moto"], self.d["forme"]
        apparsi = moto["fermi"]["apparsi"] + moto["mobili"]["apparsi"]
        colpiti = moto["fermi"]["colpiti"] + moto["mobili"]["colpiti"]
        righe = [
            self._voce(_("Bersagli apparsi: {n}."), apparsi),
            _con_parte(colpiti, apparsi, _("Bersagli colpiti: {n}, {percentuale} degli apparsi."), _("Bersagli colpiti: {n}.")) if colpiti else "",
            self._apparsi_e_colpiti(_("Bersagli fermi"), moto["fermi"]),
            self._apparsi_e_colpiti(_("Bersagli mobili"), moto["mobili"]),
        ]
        ordine = [f for f in costanti.FORME_LFO if f in forme] + sorted(f for f in forme if f not in costanti.FORME_LFO)
        for forma in ordine:
            # Traduttori: forma è il nome della forma dell'oscillazione, come «dente di sega».
            etichetta = _("Forma {forma}").format(forma=NOMI_DELLE_FORME.get(forma, forma.replace("_", " ")))
            righe.append(self._apparsi_e_colpiti(etichetta, forme[forma]))
        bonus_apparsi, bonus_colpiti = self.c("bonus_apparsi"), self.c("bonus_colpiti")
        righe.append(self._voce(_("Bonus apparsi: {n}."), bonus_apparsi))
        if bonus_colpiti:
            righe.append(_con_parte(bonus_colpiti, bonus_apparsi, _("Bonus colpiti: {n}, {percentuale} degli apparsi."), _("Bonus colpiti: {n}.")))
        righe.append(self._voce(_("Bonus usciti dal cubo senza essere colpiti: {n}."), self.c("bonus_usciti")))
        return righe

    # 6.

    def i_primati(self):
        primati = self.d["primati"]
        frasi = {
            # Traduttori: da qui, i primati; valore è già scritto, quando è una data come «giovedì 19 novembre 2026».
            "punteggio": (_("Punteggio più alto in una partita: {valore}, {quando}."), self._punti),
            "livello": (_("Livello più alto raggiunto: {valore}, {quando}."), testi.intero),
            "serie": (_("Serie più lunga di centri consecutivi: {valore}, {quando}."), testi.intero),
            "precisione": (_("Il centro più preciso: {valore}, {quando}."), testi.percento),
            "punti_in_un_tiro": (_("Più punti in un tiro: {valore}, {quando}."), self._punti),
            "centri_in_una_partita": (_("Più centri in una partita: {valore}, {quando}."), testi.intero),
            "partita_piu_lunga": (_("La partita più lunga: {valore}, {quando}."), testi.durata),
            "vicino": (_("Il tiro passato più vicino al centro di un bersaglio: {valore} dal centro, {quando}."), lambda x: testi.metri(x, 2)),
            "mancato": (_("Il tiro a vuoto passato più vicino al centro di un bersaglio: {valore} dal centro, {quando}."), lambda x: testi.metri(x, 2)),
        }
        righe = []
        for chiave in (*PRIMATI_ALTI, *PRIMATI_BASSI):
            if chiave in primati:
                frase, scrivi = frasi[chiave]
                righe.append(frase.format(valore=scrivi(primati[chiave]["valore"]), quando=testi.data_lunga(primati[chiave]["quando"])))
        for chiave in sorted(set(primati) - set(frasi)):
            righe.append(_("{etichetta}: {valore}, {quando}.").format(etichetta=_nome_da_chiave(chiave), valore=testi.numero(primati[chiave]["valore"]),
                                                                    quando=testi.data_lunga(primati[chiave]["quando"])))
        return righe

    # 7.

    def le_ultime_partite(self):
        motivi = {"riserva_vuota": _("finita a riserva vuota"), "abbandonata": _("abbandonata")}
        righe = []
        for partita in reversed(self.d["ultime"]):
            # Traduttori: una delle ultime partite: «Giovedì 19 novembre 2026 alle 18:03: 1.230 punti, livello 3, 12 centri su 20 tiri, 4 minuti e 12 secondi, abbandonata.».
            righe.append(_("{quando}: {punti}, livello {livello}, {centri} su {tiri}, {durata}, {motivo}.").format(
                quando=_maiuscola(testi.data_lunga(partita["quando"], ora=True)), punti=self._punti(partita["punteggio"]),
                livello=testi.intero(partita["livello"]), centri=plurale(partita["centri"], "{n} centro", "{n} centri", come=testi.intero(partita["centri"])),
                tiri=plurale(partita["tiri"], "{n} tiro", "{n} tiri", come=testi.intero(partita["tiri"])), durata=testi.durata(partita["durata_s"]),
                motivo=motivi.get(partita["motivo"], partita["motivo"])))
        return righe

    # 8.

    def i_suoni(self):
        suoni = self.d["suoni"]
        righe = [self._voce(_("Eventi sonori in tutto: {n}."), sum(suoni.values()))]
        for evento, n in sorted(suoni.items(), key=lambda v: (-v[1], v[0])):
            if n:
                nome = self.nomi_dei_suoni.get(evento)
                righe.append(_("{etichetta}: {n}.").format(etichetta=_maiuscola(nome) if nome else _nome_da_chiave(evento), n=testi.intero(n)))
        return righe

    # 9.

    def gli_altri_conti(self):
        righe = []
        for chiave in sorted(set(self.conti) - set(CHIAVI_DEI_CONTI)):
            valore = self.conti[chiave]
            if valore:
                scritto = testi.intero(valore) if float(valore).is_integer() else testi.numero(valore)
                righe.append(_("{etichetta}: {n}.").format(etichetta=_nome_da_chiave(chiave), n=scritto))
        return righe
