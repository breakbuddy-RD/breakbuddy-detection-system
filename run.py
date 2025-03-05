#!/usr/bin/env python3

import subprocess

def run_script(script_path):
    try:
        # Spécifie la version 3.10.11 de Python pour l'exécution
        subprocess.run(['python3.10', '-m', 'pip', 'install', 'bitalino'], check=True)
        result = subprocess.run(['python3.10', script_path], capture_output=True, text=True)
        print("Output:", result.stdout)
        print("Error:", result.stderr)
    except Exception as e:
        print("Erreur lors de l'exécution du script:", e)

if __name__ == "__main__":
    script_path = "bitalino.py"  # Remplace par le chemin de ton fichier Python
    run_script(script_path)
