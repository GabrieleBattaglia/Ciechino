"""
Ciechino, l'elenco dei suoni per Acu_Maker: per ogni azione che fa suonare un evento, il preset della
collezione di GBUtils che suona.
Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
Nasce il 2026-11-19 con la 0.1.0, sul modello di strumenti/elenco_suoni.py di MESS: un file di testo
nella cartella del progetto, suoni_di_ciechino.txt, che elenca ogni azione con il nome del suo
preset, così Gabriele può ritoccare i suoni come vuole con Acu_Maker. Il file si genera dalle tabelle
di suoni.py, GRUPPI con AZIONI, e una prova controlla che sia aggiornato: quando una tabella cambia,
si rilancia lo strumento.
Il testo è una riga per voce, pensato per NVDA, senza separatori grafici né righe vuote: in testa due
righe che dicono che cosa è il file, poi per ogni gruppo una riga col suo titolo e quanti suoni ha, e
sotto le sue voci, ciascuna con l'azione detta a parole, i due punti e il nome del preset.
Uso, dalla cartella del programma: python strumenti/elenco_suoni.py
"""

import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent
if str(RADICE) not in sys.path:
    sys.path.insert(0, str(RADICE))

import suoni  # noqa: E402
from lingua import _, plurale  # noqa: E402

NOME_DEL_FILE = "suoni_di_ciechino.txt"


def _quanti(n):
    return plurale(n, _("{n} suono"), _("{n} suoni"))


def _voce(azione, preset):
    """Una riga dell'elenco: l'azione con la maiuscola, i due punti e il preset."""
    return f"{azione[0].upper()}{azione[1:]}: {preset}."


def righe():
    """Le righe del file, senza fine riga: l'intestazione, poi i gruppi con le loro voci."""
    uscita = [
        _("I suoni di Ciechino e i loro preset nella collezione di GBUtils, da ritoccare con Acu_Maker: {quanti} in {gruppi}.").format(
            quanti=_quanti(len(suoni.EVENTI)), gruppi=plurale(len(suoni.GRUPPI), _("{n} gruppo"), _("{n} gruppi"))),
        _("Ogni riga dice l'azione che fa suonare il preset e, dopo i due punti, il suo nome. Il file lo scrive strumenti/elenco_suoni.py "
          "dalle tabelle di suoni.py: non va modificato a mano."),
    ]
    for titolo, gruppo in suoni.GRUPPI:
        uscita.append(_("{titolo}, {quanti}.").format(titolo=titolo, quanti=_quanti(len(gruppo))))
        uscita.extend(_voce(suoni.AZIONI[evento], preset) for evento, preset in gruppo.items())
    return uscita


def testo():
    return "".join(f"{riga}\n" for riga in righe())


def percorso():
    return RADICE / NOME_DEL_FILE


def main():
    destinazione = percorso()
    destinazione.write_text(testo(), encoding="utf-8")
    print(f"Scritto {destinazione.name}: {len(righe())} righe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
