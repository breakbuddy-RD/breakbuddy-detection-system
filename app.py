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
import threading
import time
from scipy.signal import find_peaks
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

## Fonctions pour récupérer les données depuis MongoDB

def load_emg_data_from_mongo(client, db, collection,user):
 
    # Récupérer les données depuis MongoDB
    data = list(collection.find({"channel": 4,"MAC":user['machine']}))
    df = pd.DataFrame(data)
    
    # Conversion du temps en datetime et tri
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time")
    df["time_ms"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds() * 1000
    return df


def load_ppg_data_from_mongo(client, db, collection, user):
    # Récupérer les 10 000 dernières données depuis MongoDB
    data = list(collection.find({"channel": 5, "MAC": user['machine']}).sort("time", -1).limit(20000))
    df = pd.DataFrame(data)
    
    # Conversion du temps en datetime et tri
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time")
    df["time_ms"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds() * 1000
    return df

def load_accelerometer_data_from_mongo(client, db, collection, user):
    # Récupérer les 10 000 dernières données depuis MongoDB
    data = list(collection.find({"channel": {"$in": [1, 2, 3]}, "MAC": user['machine']}).sort("time", -1).limit(20000))
    df = pd.DataFrame(data)
    
    # Conversion du temps en datetime et tri
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time")
    
    return df

## Fonctions pour calculer les métriques

def calculate_acceleration(df):
    df_pivot = df.pivot_table(index='time', columns='channel', values='valeur', aggfunc='mean')
    df_pivot.columns = ["X", "Y", "Z"]
    
    if len(df_pivot.columns) < 3:
        raise ValueError("Les données ne contiennent pas les trois axes nécessaires (X, Y, Z).")
    
    # Calcul de l'accélération résultante
    df_pivot['Acceleration'] = np.sqrt(df_pivot['X']**2 + df_pivot['Y']**2 + df_pivot['Z']**2)
    
    return df_pivot


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

## A Supprimer

def calculate_rms_table(df, window_size=10000):
    rms_table = []
    df["time"] = pd.to_datetime(df["time"], errors="coerce")  
    df["time"] = df["time"].astype('int64') // 10**6  # Convertir en millisecondes
    # Vérification de la taille minimale du DataFrame
    if len(df) < window_size + 10000:
        raise ValueError("Le DataFrame est trop petit pour générer 20 000 valeurs RMS.")

    # Calculer 20 000 valeurs en faisant glisser la fenêtre de 1 en 1
    start_index = len(df) - window_size - 10000
    for start in range(start_index, start_index + 10000):
        window_values = df["valeur"].values[start:start + window_size]

        # Vérifier que la fenêtre contient bien `window_size` valeurs
        if len(window_values) < window_size:
            continue  

        # Calcul du RMS
        rms_value = float(np.sqrt(np.mean(window_values**2)))

        # Ajouter le résultat au tableau
        rms_table.append({"time": df["time"].iloc[start], "valeur": rms_value})

    # Convertir en DataFrame final
    return pd.DataFrame(rms_table)


def calculate_mpf_table(df, window_size=10000, sampling_rate=1000):
    mpf_table = []
    df["time"] = pd.to_datetime(df["time"], errors="coerce")  
    df["time"] = df["time"].astype('int64') // 10**6  # Convertir en millisecondes
    # Vérification de la taille minimale du DataFrame
    if len(df) < window_size:
        raise ValueError("Le DataFrame est trop petit pour générer des valeurs MPF.")

    # Calculer 20 000 valeurs en faisant glisser la fenêtre de 1 en 1
    start_index = len(df) - window_size - 10000  # Commencer 20 000 valeurs avant la fin
    for start in range(start_index, start_index + 10000):
        window_values = df["valeur"].values[start:start + window_size]

        # Vérifier que la fenêtre contient bien `window_size` valeurs
        if len(window_values) < window_size:
            continue  

        # Calcul de la FFT et des fréquences positives
        fft_values = np.fft.fft(window_values)
        freqs = np.fft.fftfreq(len(window_values), d=1/sampling_rate)
        positive_freqs = freqs[freqs > 0]
        positive_fft_values = np.abs(fft_values[freqs > 0])

        # Calcul du MPF
        MPF = float(np.sum(positive_freqs * positive_fft_values) / np.sum(positive_fft_values))

        # Ajouter le résultat au tableau
        mpf_table.append({"time": df["time"].iloc[start], "valeur": MPF})

    # Convertir en DataFrame final
    return pd.DataFrame(mpf_table)

def calculate_bpm_data(time_ms, signal_values):
    time_ms = np.sort(time_ms)  # Tri des timestamps pour éviter les BPM négatifs
    peaks, _ = find_peaks(signal_values, distance=200)

    if len(peaks) > 1:
        peak_intervals = np.diff(time_ms[peaks])

        # Filtrer les intervalles négatifs ou aberrants
        peak_intervals = peak_intervals[peak_intervals > 0]

        if len(peak_intervals) > 0:
            bpm = 60000 / np.mean(peak_intervals)
        else:
            bpm = None  # Aucun intervalle valide
    else:
        bpm = None

    return bpm


def calculate_bpm_table(df, window_size=10000, num_values=10000):
    if len(df) < window_size + num_values:
        raise ValueError("Le DataFrame est trop petit pour générer 10 000 valeurs BPM.")
    
    # Convertir la colonne `time` en datetime puis en timestamp (millisecondes)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")  
    df["time"] = df["time"].astype('int64') // 10**6  # Convertir en millisecondes

    # Convertir `valeur` en numérique
    df["valeur"] = pd.to_numeric(df["valeur"], errors="coerce")

    time_values = df["time"].values
    signal_values = df["valeur"].values

    start_index = len(df) - window_size - num_values
    
    bpm_table = [
        {"time": time_values[start], "valeur": calculate_bpm_data(time_values[start:start+window_size], signal_values[start:start+window_size])}
        for start in range(start_index, start_index + num_values)
    ]

    return pd.DataFrame(bpm_table)

## A Supprimer
    

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["bitalino"]
collection = db["data"]
collection_users = db["users"]
collection_rms = db["rms"]
collection_mpf = db["mpf"]
collection_bpm = db["bpm"]
collection_surveillance = db["surveillance"]

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

def send_email(userId):
    try:
        filtered_user = collection_users.find_one({"id": int(userId)})
        if not filtered_user:
            return jsonify({"error": "Utilisateur non trouvé"}), 404
        
        recipient = filtered_user["email"]
        
        subject = "Alerte de fatigue musculaire"
        body = "Bonjour,\n\nNous avons détecté des signes de fatigue musculaire dans vos dernières données EMG. Veuillez consulter un professionnel de la santé pour plus d'informations.\n\nCordialement,\nL'équipe de suivi de la santé"
        

        sender_email = "damienboucher25@gmail.com"
        sender_password = "test"

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.example.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, msg.as_string())
        server.quit()

        return jsonify({"message": "Email sent successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/send_email", methods=["POST"])
def send_email():
    try:
        data = request.get_json()
        userId = data.get("userId")
        print(userId)
        filtered_user = collection_users.find_one({"id": int(userId)})
        print(filtered_user)
        if not filtered_user:
            return jsonify({"error": "Utilisateur non trouvé"}), 404
        
        recipient = filtered_user["email"]
        
        subject = "Alerte de fatigue musculaire"
        body = "Bonjour,\n\nNous avons détecté des signes de fatigue musculaire dans vos dernières données EMG. Veuillez consulter un professionnel de la santé pour plus d'informations.\n\nCordialement,\nL'équipe de suivi de la santé"
        

        sender_email = "damienboucher25@gmail.com"
        sender_password = "test"

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.example.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient, msg.as_string())
        server.quit()

        return jsonify({"message": "Email sent successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/data/<int:user>/<int:channel>", methods=["GET"])
def get_data_by_channel(user,channel):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    filtered_data = list(collection.find({"channel": channel,"MAC":filtered_user['machine']}).sort("time", -1).limit(2000))
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

@app.route("/manager")
def manage():
    users = list(collection_users.find())
    team = collection_users.distinct("team")
    return render_template("manage.html",team=team,users=users)

@app.route("/workers")
def workers():
    users = collection_users.find({})
    team = collection_users.distinct("team")
    
    return render_template("workers.html", team=team,users=users)


@app.route("/worker/<int:user>")
def worker(user):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    return render_template("worker.html",user=filtered_user)

@app.route("/worker_alert/<int:user>")
def worker_alert(user):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    return render_template("worker.html")

@app.route("/accelerometer/<int:user>")
def accelerometer(user):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    df = load_accelerometer_data_from_mongo(client, db, collection, filtered_user)
    df_pivot = calculate_acceleration(df)
    df_pivot.index = df_pivot.index.astype(str)
    df_pivot = df_pivot.dropna()  # Supprimer les lignes contenant des valeurs None
    
    result = [{"time": time, "valeur": {"Acceleration": acc, "X": x, "Y": y, "Z": z}} for time, acc, x, y, z in zip(df_pivot.index.tolist(), df_pivot['Acceleration'].values, df_pivot['X'].values, df_pivot['Y'].values, df_pivot['Z'].values)]
    
    return jsonify(result)

@app.route("/accelerometer_graph/<int:user>")
def accelerometer_graph(user):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    df = load_accelerometer_data_from_mongo(client, db, collection, filtered_user)
    df_pivot = calculate_acceleration(df)
    df_pivot.index = df_pivot.index.astype(str)
    df_pivot = df_pivot.dropna()  # Supprimer les lignes contenant des valeurs None
    
    times = df_pivot.index.tolist()
    
    valeurs = {
        "Acceleration": df_pivot["Acceleration"].values.tolist(),
        "X": df_pivot["X"].values.tolist(),
        "Y": df_pivot["Y"].values.tolist(),
        "Z": df_pivot["Z"].values.tolist()
    }
    return render_template("channel_acc.html", times=times, valeurs=valeurs, user=filtered_user['id'])

    


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

@app.route("/data/<int:user>/<string:channel>/<string:type>")
def api_data(user,channel,type):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    filtered_data = list(collection.find({"channel": int(channel), "MAC": filtered_user['machine']}).sort("time", -1).limit(LIMIT_DISPLAY))
    if not filtered_data:
        return jsonify({"error": "Aucune donnée trouvée pour ce channel " + channel}), 404
    df = pd.DataFrame(filtered_data)
    
    if type == "rms":
        table = calculate_rms_table(df)
        
    elif type == "mpf":
        table = calculate_mpf_table(df)
        
    elif type == "bpm":
        table = calculate_bpm_table(df)
        
    return jsonify(table.to_dict(orient="records"))

@app.route("/data/<int:user>/<string:type>/")
def api_data_type(user, type):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    if type == "rms":
        data = list(collection_rms.find({"machine": filtered_user['machine']}).sort("time", -1).limit(LIMIT_DISPLAY))
    elif type == "mpf":
        data = list(collection_mpf.find({"machine": filtered_user['machine']}).sort("time", -1).limit(LIMIT_DISPLAY))
    elif type == "bpm":
        data = list(collection_bpm.find({"machine": filtered_user['machine']}).sort("time", -1).limit(LIMIT_DISPLAY))
    if not data:
        return jsonify({"error": "Aucune donnée trouvée pour ce type " + type}), 404
    for entry in data:
        entry.pop("_id", None)  # Remove the _id field for JSON serialization
    return jsonify(data)

@app.route("/channel/<int:user>/<string:channel>/<string:type>")
def channel_graph_type(user,channel,type):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    name = type

    if type == "rms":
        result = api_data_type(user, type).get_json()
        times = [entry["time"] for entry in result]
        valeurs = [entry["valeur"] for entry in result]
    elif type == "mpf":
        result = api_data_type(user, type).get_json()
        times = [entry["time"] for entry in result]
        valeurs = [entry["valeur"] for entry in result]
    elif type == "bpm":
        result = api_data_type(user, type).get_json()
        times = [entry["time"] for entry in result]
        valeurs = [entry["valeur"] for entry in result]
        
    return render_template("channel_type.html", channel=channel, times=times, valeurs=valeurs, user=filtered_user['id'],type=type,name=name)

@app.route("/graph/<int:user>", methods=["GET"])
def graph(user):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    result = collection_surveillance.find({"machine": filtered_user["machine"]}).sort("time", -1).limit(2000)
    times = [entry["time"] for entry in result]
    valeurs = [entry["fatigue_score"] for entry in result]
    return render_template("graph.html", user=filtered_user['id'], times=times, valeurs=valeurs, name="Score de fatigue")

@app.route("/graph/data/<int:user>", methods=["GET"])
def graph_data(user):
    filtered_user = collection_users.find_one({"id": user})
    if not filtered_user:
        return jsonify({"error": "Utilisateur non trouvé"}), 404
    result = list(collection_surveillance.find({"machine": filtered_user["machine"], "fatigue_score": {"$ne": np.nan}}).sort("time", -1).limit(2000))
    for entry in result:
        entry.pop("_id", None)
        
    for entry in result:
        entry["valeur"] = float(entry["fatigue_score"])
    return jsonify(result)


def calculate_rms(df):
    # Calculer une seule valeur RMS
    window_size = 1000
    if len(df) < window_size:
        raise ValueError("Le DataFrame est trop petit pour générer une valeur RMS.")
    
    window_values = df["valeur"].iloc[-window_size:]
    rms_value = np.sqrt(np.mean(window_values**2))
    
    return {"time": df["time"].iloc[-1], "valeur": rms_value}

def calculate_mpf(df):
    # Calculer une seule valeur MPF
    window_size = 1000
    sampling_rate = 1000
    if len(df) < window_size:
        raise ValueError("Le DataFrame est trop petit pour générer une valeur MPF.")
    
    window_values = df["valeur"].iloc[-window_size:]
    fft_values = np.fft.fft(window_values)
    freqs = np.fft.fftfreq(len(window_values), d=1/sampling_rate)
    positive_freqs = freqs[freqs > 0]
    positive_fft_values = np.abs(fft_values[freqs > 0])
    MPF = np.sum(positive_freqs * positive_fft_values) / np.sum(positive_fft_values)
    
    return {"time": df["time"].iloc[-1], "valeur": MPF}


def calculate_bpm_unique(df):
    # Vérification de la longueur minimale
    if len(df) < 1000:
        return {"time": df["time"].iloc[-1], "valeur": None}

    time_ms = df["time_ms"].values
    signal_values = df["valeur"].values
    
    # Détection des pics avec un seuil dynamique
    peaks, _ = signal.find_peaks(signal_values, distance=200, height=np.mean(signal_values) * 0.5)
    
    if len(peaks) < 2:
        return {"time": df["time"].iloc[-1], "valeur": None}  # Pas assez de pics détectés
    
    # Calcul des intervalles entre pics
    peak_intervals = np.diff(time_ms[peaks])
    
    if len(peak_intervals) == 0:
        return {"time": df["time"].iloc[-1], "valeur": None}

    # Calcul BPM
    bpm = 60000 / np.mean(peak_intervals)
    
    return {"time": df["time"].iloc[-1], "valeur": bpm}



def background_task():
    while True:
        # Votre code à exécuter en continu ici
        print("Tâche en arrière-plan en cours d'exécution...")
        # Calculer les valeurs RMS, MPF et BPM
        ## tresholds
        try:
            users = list(collection_users.find())
            for user in users:
                df_emg = load_emg_data_from_mongo(client, db, collection, user)
                rms_table = calculate_rms(df_emg)
                mpf_table = calculate_mpf(df_emg)
                df_ppg = load_ppg_data_from_mongo(client, db, collection, user)
                bpm_table = calculate_bpm_unique(df_ppg)
                # Ajouter la machine à l'import
                rms_table["machine"] = user["machine"]
                mpf_table["machine"] = user["machine"]
                bpm_table["machine"] = user["machine"]
                
                # Vérifier que la dernière valeur n'est pas à la même heure et que la machine est la même avant d'insérer les résultats dans MongoDB
                last_rms = collection_rms.find_one({"machine": user["machine"]}, sort=[("time", -1)])
                last_mpf = collection_mpf.find_one({"machine": user["machine"]}, sort=[("time", -1)])
                last_bpm = collection_bpm.find_one({"machine": user["machine"]}, sort=[("time", -1)])
                
                if not last_rms or (last_rms["time"] != rms_table["time"]):
                    collection_rms.insert_one(rms_table)
                if not last_mpf or (last_mpf["time"] != mpf_table["time"]):
                    collection_mpf.insert_one(mpf_table)
                if not last_bpm or (last_bpm["time"] != bpm_table["time"]):
                    collection_bpm.insert_one(bpm_table)
                
                df_acc = load_accelerometer_data_from_mongo(client, db, collection, user)
                acceleration_table = calculate_acceleration(df_acc)
                last_acceleration = acceleration_table.iloc[-1].to_dict()
                
                # Surveillance de la fatigue musculaire
                normalized_rms = (rms_table["valeur"] - 500) / (1640-500)  # Normalisation entre 0 et 1 vaut 0.5
                normalized_bpm = (bpm_table["valeur"] -40) / (220-40)  # Normalisation entre 0 et 1 vaut 0.3
                normalized_acceleration = (last_acceleration["Acceleration"] - 600) / (1750-600)  # Normalisation entre 0 et 1 vaut 0.2
                
                
                fatigue_score = normalized_rms * 0.5 + normalized_bpm * 0.3 + normalized_acceleration * 0.2
                
                # Vérifier si une entrée de surveillance existe déjà pour cette machine et cette heure
                existing_entry = collection_surveillance.find_one({"machine": user["machine"], "time": rms_table["time"]})
                if not np.isnan(fatigue_score):
                    if existing_entry:
                        # Mettre à jour l'entrée existante avec le nouveau score de fatigue
                        collection_surveillance.update_one(
                            {"_id": existing_entry["_id"]},
                            {"$set": {"fatigue_score": fatigue_score}}
                        )
                    else:
                        # Insérer une nouvelle entrée de surveillance
                        collection_surveillance.insert_one({
                            "machine": user["machine"],
                            "time": rms_table["time"],
                            "fatigue_score": fatigue_score
                        })
                
                if(fatigue_score > user["treshold"]):
                    send_email(user["id"])
                
        except Exception as e:
            print(f"Erreur lors de l'insertion des résultats dans MongoDB : {e}", file=sys.stderr)
        time.sleep(0.2)  # Délai de 10 secondes

if __name__ == "__main__":
    try:
        # Démarrer la tâche en arrière-plan
        background_thread = threading.Thread(target=background_task)
        background_thread.daemon = True
        background_thread.start()

        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"Erreur lors de l'exécution du serveur Flask : {e}", file=sys.stderr)
        sys.exit(1)
        