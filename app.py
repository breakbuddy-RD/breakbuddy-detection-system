from flask import Flask, request, jsonify, render_template
from datetime import datetime
import sys
import json
import os
from pymongo import MongoClient
import numpy as np
import pandas as pd
import scipy.signal as signal
from pymongo import MongoClient

def load_ppg_data_from_mongo(client, db, collection,user):
 
    # Récupérer les données depuis MongoDB
    data = list(collection.find({"channel": 5,"MAC":user['machine']}))
    df = pd.DataFrame(data)
    
    # Conversion du temps en datetime et tri
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time")
    df["time_ms"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds() * 1000
    return df

def calculate_bpm(df):
    time_ms = df["time_ms"].values
    signal_values = df["valeur"].values
    
    # Détection des pics
    peaks, _ = signal.find_peaks(signal_values, distance=200)
    
    # Calcul du BPM
    peak_intervals = np.diff(time_ms[peaks])
    bpm = 60000 / np.mean(peak_intervals) if len(peak_intervals) > 0 else None
    
    return bpm, time_ms, signal_values, peaks, peak_intervals

def analyze_signal_variability(peak_intervals):
    mean_interval = np.mean(peak_intervals)
    std_interval = np.std(peak_intervals)
    variability_index = std_interval / mean_interval if mean_interval != 0 else None
    return mean_interval, std_interval, variability_index

def detect_ppg_waves(signal_values, sampling_rate=1000):
    # Filtrage du signal
    b, a = signal.butter(3, [0.5, 5], btype='bandpass', fs=sampling_rate)
    filtered_signal = signal.filtfilt(b, a, signal_values)
    
    # Détection des ondes
    systolic_peaks, _ = signal.find_peaks(filtered_signal, distance=200)
    diastolic_peaks, _ = signal.find_peaks(-filtered_signal, distance=200)

    return systolic_peaks, diastolic_peaks

def load_emg_data_from_mongo(client, db, collection,user):
 
    # Récupérer les données depuis MongoDB
    data = list(collection.find({"channel": 4,"MAC":user['machine']}))
    df = pd.DataFrame(data)
    
    # Conversion du temps en datetime et tri
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time")
    df["time_ms"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds() * 1000
    return df

def calculate_emg_metrics(df):
    time_ms = df["time_ms"].values
    signal_values = df["valeur"].values
    
    # Calcul des indicateurs EMG
    mean_emg = float(np.mean(signal_values))
    std_emg = float(np.std(signal_values))
    rms_emg = float(np.sqrt(np.mean(signal_values**2)))
    iemg = float(np.sum(np.abs(signal_values)))
    
    # Détection des pics EMG
    peaks, _ = signal.find_peaks(signal_values, height=np.mean(signal_values) + np.std(signal_values))
    
    return mean_emg, std_emg, rms_emg, iemg, len(peaks), time_ms, signal_values

def calculate_fatigue_metrics(signal_values, sampling_rate=1000):
    # Transformer le signal en domaine fréquentiel (FFT)
    fft_values = np.fft.fft(signal_values)
    freqs = np.fft.fftfreq(len(signal_values), d=1/sampling_rate)
    
    # Prendre uniquement les valeurs positives de fréquence
    positive_freqs = freqs[freqs > 0]
    positive_fft_values = np.abs(fft_values[freqs > 0])
    
    # Calcul de la Fréquence Médiane (MDF)
    cumulative_power = np.cumsum(positive_fft_values)
    total_power = cumulative_power[-1]
    median_freq_index = np.where(cumulative_power >= total_power / 2)[0][0]
    MDF = float(positive_freqs[median_freq_index])
    
    # Calcul de la Fréquence Moyenne (MNF)
    MNF = float(np.sum(positive_freqs * positive_fft_values) / np.sum(positive_fft_values))
    
    return MDF, MNF

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["bitalino"]
collection = db["data"]
collection_users = db["users"]


LIMIT_DISPLAY = 20000

@app.route("/data", methods=["GET"])
def get_all_data():
    data_store = list(collection.find().sort("time", -1).limit(LIMIT_DISPLAY))
    for entry in data_store:
        entry["_id"] = str(entry["_id"])  # Convert ObjectId to string for JSON serialization
    return jsonify(data_store)

@app.route("/photoplethysmography/<int:user>", methods=["GET"])
def get_bpm(user):
    try:
        filtered_user = collection_users.find_one({"id": user})
        if not filtered_user:
            return jsonify({"error": "Utilisateur non trouvé"}), 404
        df = load_ppg_data_from_mongo(client, db, collection, filtered_user)
        bpm, _, signal_values, _, peak_intervals = calculate_bpm(df)
        mean_interval, std_interval, variability_index = analyze_signal_variability(peak_intervals)
        systolic_peaks, diastolic_peaks = detect_ppg_waves(signal_values)
        
        return jsonify({
            "bpm": bpm,
            "mean_interval": mean_interval,
            "std_interval": std_interval,
            "variability_index": variability_index,
            "systolic_peaks_count": len(systolic_peaks),
            "diastolic_peaks_count": len(diastolic_peaks)
        })
    except Exception as e:
        return jsonify({})
    
@app.route("/emg/<int:user>", methods=["GET"])
def get_emg_metrics(user):
    try:
        filtered_user = collection_users.find_one({"id": user})
        if not filtered_user:
            return jsonify({"error": "Utilisateur non trouvé"}), 404
        df = load_emg_data_from_mongo(client, db, collection, filtered_user)
        mean_emg, std_emg, rms_emg, iemg, num_peaks, _, signal_values = calculate_emg_metrics(df)
        MDF, MNF = calculate_fatigue_metrics(signal_values)
        
        # Détermination de la fatigue musculaire
        FATIGUE_THRESHOLD_MDF = 50  # Seuil de Fréquence Médiane indicative de fatigue (Hz)
        FATIGUE_THRESHOLD_MNF = 70  # Seuil de Fréquence Moyenne indicative de fatigue (Hz)
        is_fatigued = MDF < FATIGUE_THRESHOLD_MDF or MNF < FATIGUE_THRESHOLD_MNF
        
        return jsonify({
            "mean_emg": mean_emg,
            "std_emg": std_emg,
            "rms_emg": rms_emg,
            "iemg": iemg,
            "num_peaks": num_peaks,
            "MDF": MDF,
            "MNF": MNF,
            "fatigue_detected": is_fatigued
        })
    except Exception as e:
        return jsonify({})


@app.route("/data/<int:user>/<int:channel>", methods=["GET"])
def get_data_by_channel(user,channel):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    filtered_data = list(collection.find({"channel": channel,"MAC":filtered_user['machine']}).sort("time", -1).limit(LIMIT_DISPLAY))
    if not filtered_data:
        return jsonify({"error": "Aucune donnée trouvée pour ce channel"}), 404

    for entry in filtered_data:
        entry["_id"] = str(entry["_id"])  # Convert ObjectId to string for JSON serialization

    return jsonify(filtered_data)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/channels")
def channels():
    channels = collection.distinct("channel")
    users = collection_users.distinct("id")
    return render_template("channels.html", channels=channels, users=users)

@app.route("/manage")
def manage():
    users = list(collection_users.find())
    team = collection_users.distinct("team")
    return render_template("manage.html",team=team,users=users)

@app.route("/workers")
def workers():
    users = collection_users.find({})
    team = collection_users.distinct("team")
    
    return render_template("workers.html", team=team,users=users)


@app.route("/worker")
def worker():
    return render_template("worker.html")



@app.route("/channel/<int:user>/<string:channel>")
def channel_graph(user,channel):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404

    filtered_data = list(collection.find({"channel": int(channel), "MAC": filtered_user['machine']}).sort("time", -1).limit(LIMIT_DISPLAY))
    if not filtered_data:
        return jsonify({"error": "Aucune donnée trouvée pour ce channel " + channel}), 404

    times = [entry["time"] for entry in filtered_data]
    valeurs = [entry["valeur"] for entry in filtered_data]

    return render_template("channel.html", channel=channel, times=times, valeurs=valeurs, user=filtered_user['id'])

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"Erreur lors de l'exécution du serveur Flask : {e}", file=sys.stderr)
        sys.exit(1)