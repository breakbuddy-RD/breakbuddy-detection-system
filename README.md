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

## Description du Projet

Ce projet est un système de détection BreakBuddy qui permet de surveiller et d'analyser les pauses des utilisateurs. Il utilise des technologies telles que Python et MongoDB pour le traitement des données et le stockage.

## Fonctionnement

### 1. Collecte des Données
Le système collecte des données en temps réel à partir de différentes sources et les stocke dans une base de données MongoDB.

### 2. Analyse des Données
Les données collectées sont analysées pour détecter les pauses et générer des rapports sur l'état de fatigue de la personne. Dans le cas où une fatigue est détectée, un système d'alerte envoie des notifications par mail aux parties concernées.

### 3. Interface Utilisateur
Une interface utilisateur permet de visualiser les données collectées et les rapports générés. Elle est accessible via un navigateur web en lançant l'application avec la commande :
```powershell
python .\app.py
```

## Structure du Projet

- `run.py` : Script pour démarrer l'acquisition des données.
- `app.py` : Script pour lancer l'application web.
- `requirements.txt` : Fichier listant les dépendances Python nécessaires au projet.
- `README.md` : Documentation du projet.

## Contribuer

Si vous souhaitez contribuer à ce projet, veuillez suivre les étapes suivantes :

1. Fork le dépôt.
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`).
3. Commitez vos modifications (`git commit -m 'Add some AmazingFeature'`).
4. Poussez votre branche (`git push origin feature/AmazingFeature`).
5. Ouvrez une Pull Request.