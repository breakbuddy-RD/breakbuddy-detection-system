#!/usr/bin/env python3

import subprocess
import signal
import sys
import argparse

# Variable globale pour le processus en cours
current_process = None

def signal_handler(_, __):
    global current_process
    if current_process:
        current_process.terminate()
        print("Processus arrêté.")
    sys.exit(0)

def run_script(script_path, mac_address):
    global current_process
    try:
        # Spécifie la version 3.10.11 de Python pour l'exécution
        print("Exécution du script...")
        print("MAC address:", mac_address)
        current_process = subprocess.Popen(['python', script_path, mac_address], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Lire les sorties standard et d'erreur en temps réel
        while True:
            output = current_process.stdout.readline()
            if output == '' and current_process.poll() is not None:
                break
            if output:
                print("Output:", output.strip())
        
        stderr = current_process.stderr.read()
        if stderr:
            print("Error:", stderr.strip())
    except Exception as e:
        print("Erreur lors de l'exécution du script:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a Python script with a MAC address argument.')
    parser.add_argument('mac_address', type=str, help='The MAC address to pass to the script')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    script_path = "bitalino_script.py"  # Remplace par le chemin de ton fichier Python
    run_script(script_path, args.mac_address)
