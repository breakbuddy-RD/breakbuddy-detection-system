#!/usr/bin/env python3

import subprocess
import signal
import sys

# Variable globale pour le processus en cours
current_process = None

def signal_handler(_, __):
    global current_process
    if current_process:
        current_process.terminate()
        print("Processus arrêté.")
    sys.exit(0)

def run_script(script_path):
    global current_process
    try:
        # Spécifie la version 3.10.11 de Python pour l'exécution
        current_process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = current_process.communicate()
        print("Output:", stdout)
        print("Error:", stderr)
    except Exception as e:
        print("Erreur lors de l'exécution du script:", e)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    script_path = "bitalino_script.py"  # Remplace par le chemin de ton fichier Python
    run_script(script_path)
