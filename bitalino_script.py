import threading
from bitalino import BITalino
from datetime import datetime
import requests
from pymongo import MongoClient
MAX_DATA_SEND_SAVE = 100000
# Adresse MAC de votre appareil BITalino
macAddress = "98:D3:51:FE:84:FC"


# Créer une instance de l'appareil BITalino
device = BITalino(macAddress)

# Démarrer l'acquisition de données
device.start(1000, [0, 1, 2, 3, 4,5])  # 1000 Hz, canaux 0 à 5

tab1 = []
tab2 = []
tab3 = []
tab4 = []
tab5 = []
tab6 = []
nb_data_send = 100

envoi = False

def collect_data():
    global tab1, tab2, tab3,tab4, tab5,tab6, envoi,nb_data_send
    try:
        i = 0
        while True:
            data = device.read(1)  # Lire 1 échantillon
            channel_1_data = data[0, 5]  # Extraire la valeur courante du canal 1
            channel_2_data = data[0, 6]  # Extraire la valeur courante du canal 2
            channel_3_data = data[0, 7]  # Extraire la valeur courante du canal 3
            channel_4_data = data[0, 8]  # Extraire la valeur courante du canal 4
            channel_5_data = data[0, 9]  # Extraire la valeur courante du canal 5
            channel_6_data = data[0, 10]  # Extraire la valeur courante du canal 6
            #print(f"Valeur courante du canal 1: {channel_1_data}")
            #print(f"Valeur courante du canal 2: {channel_2_data}")
            tab1.append({'temps': datetime.now().isoformat(timespec='milliseconds'), 'valeur': channel_1_data})
            tab2.append({'temps': datetime.now().isoformat(timespec='milliseconds'), 'valeur': channel_2_data})
            tab3.append({'temps': datetime.now().isoformat(timespec='milliseconds'), 'valeur': channel_3_data})
            tab4.append({'temps': datetime.now().isoformat(timespec='milliseconds'), 'valeur': channel_4_data})
            tab5.append({'temps': datetime.now().isoformat(timespec='milliseconds'), 'valeur': channel_5_data})
            tab6.append({'temps': datetime.now().isoformat(timespec='milliseconds'), 'valeur': channel_6_data})
            if i % nb_data_send == 0:
                envoi = True
            
    except KeyboardInterrupt:
        pass

def send_data():
    global tab1, tab2,tab3,tab4, tab5, envoi,nb_data_send
    while True:
        if len(tab1) >= nb_data_send and len(tab2) >= nb_data_send and len(tab3) >= nb_data_send and len(tab4) >= nb_data_send and len(tab5)>= nb_data_send and len(tab6) >= nb_data_send and envoi:
            # Préparer les données pour l'envoi
            data_to_send = []
            print("Envoi des données...")

            for entry in tab1[-nb_data_send*2:]:
                data_to_send.append({
                    'channel': 1,
                    'time': entry['temps'],
                    'valeur': int(entry['valeur'])
                })

            for entry in tab2[-nb_data_send*2:]:
                data_to_send.append({
                    'channel': 2,
                    'time': entry['temps'],
                    'valeur': int(entry['valeur'])
                })
                
            for entry in tab3[-nb_data_send*2:]:
                data_to_send.append({
                    'channel': 3,
                    'time': entry['temps'],
                    'valeur': int(entry['valeur'])
                })
            
            for entry in tab4[-nb_data_send*2:]:
                data_to_send.append({
                    'channel': 4,
                    'time': entry['temps'],
                    'valeur': int(entry['valeur'])
                    
                })
                
                
            for entry in tab5[-nb_data_send*2:]:
                data_to_send.append({
                    'channel': 5,
                    'time': entry['temps'],
                    'valeur': int(entry['valeur'])
                })
                
            for entry in tab6[-nb_data_send*2:]:
                data_to_send.append({
                    'channel': 6,
                    'time': entry['temps'],
                    'valeur': int(entry['valeur'])
                })

            # Convertir les données en JSON
            try:
                # Connexion à la base de données MongoDB
                client = MongoClient('mongodb://localhost:27017/')
                db = client['bitalino']
                collection = db['data']
                
                
                
                    
                    

                # Insérer les données dans la collection
                collection.insert_many(data_to_send)
                
                # Compter le nombre de documents dans la collection
                count = collection.count_documents({})
                print(f"Nombre de documents dans la collection: {count}")
                if(count >= MAX_DATA_SEND_SAVE):
                    docs_to_delete = collection.find({}).sort("_id", 1).limit(count - MAX_DATA_SEND_SAVE)
                    ids_to_delete = [doc["_id"] for doc in docs_to_delete]
                    if ids_to_delete:
                        collection.delete_many({"_id": {"$in": ids_to_delete}})

                # Supprimer les données envoyées
                tab1 = tab1[nb_data_send:]
                tab2 = tab2[nb_data_send:]
                tab3 = tab3[nb_data_send:]
                tab4 = tab4[nb_data_send:]
                tab5 = tab5[nb_data_send:]
                envoi = False

                
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                

# Créer et démarrer les threads
thread_collect = threading.Thread(target=collect_data)
thread_send = threading.Thread(target=send_data)

thread_collect.start()
thread_send.start()

# Attendre que les threads se terminent
thread_collect.join()
thread_send.join()


# Arrêter l'acquisition de données
device.stop()

# Fermer la connexion avec l'appareil
device.close()
