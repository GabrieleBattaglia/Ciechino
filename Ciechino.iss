; Ciechino, l'installatore con Inno Setup 6.
; Autori: Gabriele Battaglia (IZ4APU) & ClaudIA (Claude Opus 5.5, UltraCode).
; Nasce il 2026-11-19 con la 0.1.0, sul modello di MESS.iss e di MeTeOra.iss: il pacchetto ha
; l'installatore, e l'installatore chiede di accettare la licenza.
;
; Si compila con python setup_maker.py, dopo PyInstaller e zip_maker.py: lo script passa la
; versione di version.py. L'installatore affianca lo zip dell'aggiornamento automatico, che usa
; sempre lo zip; si installa per utente, senza amministratore, perché Ciechino scrive classifica,
; statistiche e impostazioni accanto all'eseguibile e l'aggiornamento automatico sostituisce i file
; nella sua cartella; alla disinstallazione chiede se cancellare anche questi dati.
; I nomi dei file dei dati sono quelli di costanti.py, FILE_CLASSIFICA, FILE_STATISTICHE e
; FILE_IMPOSTAZIONI, e quelli dei registri vengono da registro.py: chi li cambia là li cambia anche
; qui, e la prova del pacchetto se ne accorge.

#ifndef Versione
  #error Manca la versione: si compila con python setup_maker.py
#endif

[Setup]
; L'identità dell'installazione: non cambia mai, e non è quella di nessun altro programma del parco.
AppId={{14E6AA49-2875-4C1C-BAD9-ADBFCDB8A746}
AppName=Ciechino
AppVersion={#Versione}
AppVerName=Ciechino {#Versione}
AppPublisher=Gabriele Battaglia (IZ4APU)
AppPublisherURL=https://github.com/GabrieleBattaglia/Ciechino
AppSupportURL=https://github.com/GabrieleBattaglia/Ciechino/issues
AppUpdatesURL=https://github.com/GabrieleBattaglia/Ciechino/releases
VersionInfoVersion={#Versione}
; La licenza GPL 3, da accettare prima di installare.
LicenseFile=LICENSE
; Per utente: con PrivilegesRequired=lowest {autopf} è la cartella Programs di AppData\Local.
PrivilegesRequired=lowest
DefaultDirName={autopf}\Ciechino
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=.
OutputBaseFilename=Ciechino-Setup-{#Versione}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ShowLanguageDialog=no
UninstallDisplayName=Ciechino
UninstallDisplayIcon={app}\Ciechino.exe
; Ciechino aperto tiene questo mutex (istanza.py): installazione e disinstallazione chiedono di chiuderlo.
AppMutex=Ciechino
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "italiano"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Prima di installare, via le cartelle del programma lasciate da una versione precedente. Un Ciechino
; aggiornato da solo ha in _internal il manifesto dello zip: se restasse, al primo avvio
; pulisci_residui cancellerebbe tutti i file nuovi che quel manifesto non elenca. I dati di chi gioca
; stanno fuori da _internal, accanto all'eseguibile, e restano dove sono.
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\_internal_vecchio"

[Files]
; Il pacchetto a cartella di PyInstaller, senza i dati che una prova avesse lasciato in dist, come fa
; zip_maker.py. Ogni motivo comincia con la barra rovesciata, che lo lega alla cartella dell'eseguibile:
; i file che Ciechino scrive cominciano tutti col suo nome e un trattino, poi la lingua scelta e i registri.
Source: "dist\Ciechino\*"; DestDir: "{app}"; Excludes: "\Ciechino - *,\selected_language.json,\*.log,\*.log.1,\*.tmp,\*.bak"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Ciechino"; Filename: "{app}\Ciechino.exe"
Name: "{autodesktop}\Ciechino"; Filename: "{app}\Ciechino.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Ciechino.exe"; Description: "{cm:LaunchProgram,Ciechino}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; L'aggiornamento automatico può portare in _internal file che l'installatore non conosce: la
; cartella del programma si toglie intera. I dati stanno fuori da _internal, accanto all'eseguibile.
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\_internal_vecchio"

[Code]
const
  // I file dei dati, con i nomi di costanti.py.
  FileClassifica = 'Ciechino - Classifica.json';
  FileStatistiche = 'Ciechino - Statistiche.json';
  FileImpostazioni = 'Ciechino - Impostazioni.json';

procedure CancellaIFile(const Cartella, Modello: String);
var
  Trovato: TFindRec;
begin
  if FindFirst(Cartella + '\' + Modello, Trovato) then
  begin
    try
      repeat
        if (Trovato.Attributes and FILE_ATTRIBUTE_DIRECTORY) = 0 then
          DeleteFile(Cartella + '\' + Trovato.Name);
      until not FindNext(Trovato);
    finally
      FindClose(Trovato);
    end;
  end;
end;

// Un file dei dati con tutto ciò che gli nasce accanto: la copia .bak, il .tmp della scrittura
// sicura e le copie illeggibili messe da parte, che cominciano tutti col suo nome senza estensione.
procedure CancellaUnDato(const Cartella, Nome: String);
begin
  CancellaIFile(Cartella, ChangeFileExt(Nome, '') + '*');
end;

// I dati di Ciechino: classifica, statistiche e impostazioni, la lingua scelta, i registri degli
// errori e dei crash, il file dei registri preparato per il PUB e il registro dell'aggiornamento.
procedure CancellaIDati(const Cartella: String);
begin
  CancellaUnDato(Cartella, FileClassifica);
  CancellaUnDato(Cartella, FileStatistiche);
  CancellaUnDato(Cartella, FileImpostazioni);
  DeleteFile(Cartella + '\selected_language.json');
  DeleteFile(Cartella + '\Ciechino - Errori.log');
  DeleteFile(Cartella + '\Ciechino - Errori.log.1');
  DeleteFile(Cartella + '\Ciechino - Crash.log');
  DeleteFile(Cartella + '\Ciechino - Crash precedente.log');
  DeleteFile(Cartella + '\Ciechino - Registri per il PUB.txt');
  DeleteFile(Cartella + '\auto_updater_error.log');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and not UninstallSilent then
  begin
    if MsgBox('Cancello anche i dati di Ciechino? La classifica, le statistiche e le impostazioni, con i registri degli errori. ' +
        'Con No restano nella cartella, e reinstallando Ciechino li ritrovi.',
        mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      CancellaIDati(ExpandConstant('{app}'));
    // La cartella si toglie solo se è rimasta vuota.
    RemoveDir(ExpandConstant('{app}'));
  end;
end;
