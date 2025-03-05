# Installation et Lancement du Projet

## Prérequis
### 1. Installation de MongoDB
Assurez-vous que MongoDB est installé sur votre système.

Vous pouvez télécharger et installer MongoDB à partir du site officiel :
[MongoDB Download](https://www.mongodb.com/try/download/community)

### 2. Configuration de l'environnement Python sous Windows
#### a) Autoriser l'exécution des scripts PowerShell
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

#### b) Installer `pyenv-win`
```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

#### c) Installer et configurer Python
Dans le dossier du projet, exécutez les commandes suivantes :
```powershell
pyenv install 3.10.11
pyenv global 3.10.11
pyenv global
python --version
```
Vérifiez que Python est bien installé et que la version affichée correspond à `3.10.11`.

## Lancement du Projet

### 1. Démarrer l'acquisition des données
```powershell
python .\run.py
```

### 2. Lancer l'application
```powershell
python .\app.py
```

L'application devrait maintenant être en cours d'exécution.

## Remarque
Assurez-vous que toutes les dépendances nécessaires sont installées en exécutant :
```powershell
pip install -r requirements.txt
```

