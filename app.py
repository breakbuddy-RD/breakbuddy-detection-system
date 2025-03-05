from flask import Flask, request, jsonify, render_template
from datetime import datetime
import sys
import json
import os
from pymongo import MongoClient

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["bitalino"]
collection = db["data"]
LIMIT_DISPLAY = 10000

@app.route("/data", methods=["GET"])
def get_all_data():
    data_store = list(collection.find().sort("time", -1).limit(LIMIT_DISPLAY))
    for entry in data_store:
        entry["_id"] = str(entry["_id"])  # Convert ObjectId to string for JSON serialization
    return jsonify(data_store)

@app.route("/data/<int:channel>", methods=["GET"])
def get_data_by_channel(channel):
    filtered_data = list(collection.find({"channel": channel}).sort("time", -1).limit(LIMIT_DISPLAY))
    if not filtered_data:
        return jsonify({"error": "Aucune donnée trouvée pour ce channel"}), 404

    for entry in filtered_data:
        entry["_id"] = str(entry["_id"])  # Convert ObjectId to string for JSON serialization

    return jsonify(filtered_data)

@app.route("/")
def home():
    channels = collection.distinct("channel")
    return render_template("home.html", channels=channels)

@app.route("/channel/<string:channel>")
def channel_graph(channel):
    filtered_data = list(collection.find({"channel": int(channel)}).sort("time", -1).limit(LIMIT_DISPLAY))
    if not filtered_data:
        return jsonify({"error": "Aucune donnée trouvée pour ce channel " + channel}), 404

    times = [entry["time"] for entry in filtered_data]
    valeurs = [entry["valeur"] for entry in filtered_data]

    return render_template("channel.html", channel=channel, times=times, valeurs=valeurs)

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"Erreur lors de l'exécution du serveur Flask : {e}", file=sys.stderr)
        sys.exit(1)
