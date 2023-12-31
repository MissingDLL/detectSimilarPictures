from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import url_quote
from tensorflow.keras.models import load_model
import numpy as np
import os
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import concurrent.futures
from PIL import Image
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import logging
import psutil
import json  # Für das Speichern und Laden von Datenstrukturen
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, pooling='avg')
# model = load_model('image_similarity_model.keras')

# Funktion zur Berechnung des MD5-Hashes eines Bilds
def calculate_image_hash(image_path):
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_hash = hashlib.md5(image_data).hexdigest()
    return image_hash

# Funktion zum Laden der Features und Hash-Werte
def load_features_and_identifiers(feature_path):
    with open(feature_path, 'r') as f:
        feature_data = json.load(f)
    features = np.array(feature_data['features'])
    image_hashes = feature_data['image_hashes']
    return features, image_hashes

# Funktion zum Berechnen der Features eines Bilds (Beispiel)
def calculate_features(image_path):
    image_path_new = []
    for file in image_path:
        if os.path.exists(file):
            image_path_new.append(file)

    # Check if there are valid image files
    if not image_path_new:
        logger.error("No valid image files found.")
        return None

    images = [tf.keras.preprocessing.image.load_img(file, target_size=(100, 100)) for file in image_path_new]

    # Check if there are valid images after loading
    if not images:
        logger.error("No valid images loaded.")
        return None

    images = [tf.keras.preprocessing.image.img_to_array(img) for img in images]
    images = tf.keras.applications.vgg16.preprocess_input(np.array(images))

    features = base_model.predict(images)
    return features


# Funktion zum Speichern der Features und Hash-Werte
def save_features_and_identifiers(features, identifiers, filename):
    # Add your original check
    if features is not None:
        # Convert features to a list
        features_list = [feature.tolist() for feature in features]

        # Check if the lengths match
        if len(features_list) != len(identifiers):
            logger.error("len(features_list) != len(identifiers)")
            raise Exception("len(features_list) != len(identifiers)")

        # Combine features and identifiers into a dictionary
        data = {'features': features_list, 'identifiers': identifiers}

        # Save the data to a JSON file
        with open(filename, 'w') as file:
            json.dump(data, file)


# Funktion zum Ermitteln der neuen Bildpfade
def get_new_image_paths(image_paths, saved_image_hashes):
    new_image_paths = []
    for image in image_paths:
        image_hash = calculate_image_hash(image)
        if image_hash not in saved_image_hashes:
            new_image_paths.append(image)
    return new_image_paths

# Funktion zur Überwachung des Speicherverbrauchs
def monitor_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss  # Gesamtspeicherverbrauch in Bytes


# Funktion zum Holen der Bildpfade aus einem Verzeichnis
def get_image_paths(directory):
    thumbnail_static_dir = os.path.join(app.static_folder, 'thumbnails')
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                drive, path_without_drive = os.path.splitdrive(root)
                thumbnail_path = os.path.join(thumbnail_static_dir, path_without_drive.lstrip('/'))
                if os.path.exists(thumbnail_path):
                    image_paths.append(os.path.join(thumbnail_path, file))
    return image_paths

class Cluster:
    def __init__(self, clusterName, images, accuracy):
        self.clusterName = clusterName
        self.images = images
        self.accuracy = accuracy

def cluster_images_new_bak(image_path, features):
    #logger.info(f"Clustering images in directory: {image_path}")
    cluster_list = []
    image_path_new=[]
    for file in image_path:
        if os.path.exists(file):
            image_path_new.append(file)

    #try:
    # Finde die Indizes der Features, die den Bildern in image_path entsprechen
    feature_indices = [i for i, path in enumerate(image_path) if path in image_path]

    # Wählen Sie nur die relevanten Features basierend auf den gefundenen Indizes aus
    relevant_features = [features[i] for i in feature_indices]

    # Berechnen Sie die Ähnlichkeitsmatrix nur für die relevanten Features
    similarity_matrix = cosine_similarity(relevant_features)
    #print(similarity_matrix)
    threshold = 0.9
    
    percentage_similarity_matrix = (similarity_matrix + 1) / 2 * 100

    for i in range(len(image_path_new)):
        similar_group = []
        similar_group_accuracy = []
        for j in range(len(image_path_new)):
            if image_path_new[i] != image_path_new[j]:
                if similarity_matrix[i, j] > threshold:
                    similar_group.append(os.path.relpath(image_path_new[j], start=app.static_folder))
                    similar_group_accuracy.append(percentage_similarity_matrix[i][j])
        if len(similar_group) > 0:
            cluster = Cluster(os.path.relpath(image_path_new[i], start=app.static_folder), similar_group, similar_group_accuracy)
            cluster_list.append(cluster)
    # except Exception as e:
    #     logger.error(f"Error during clustering for directory {image_path_new}: {e}")
    #logger.info(f"Finished clustering for directory: {image_path}")
    logger.debug(cluster_list)
    cluster_count = len(cluster_list)
    return cluster_list, cluster_count

def calculate_cluster_accuracy(cluster, similarity_matrix):
    image_paths = cluster.images
    num_images = len(image_paths)
    accuracy_values = []

    for i in range(num_images):
        accuracy_sum = 0.0
        for j in range(num_images):
            if i != j:
                image_i_path = image_paths[i]
                image_j_path = image_paths[j]

                # Ermitteln Sie die Indizes der Bilder in der Ähnlichkeitsmatrix
                index_i = image_paths.index(image_i_path)
                index_j = image_paths.index(image_j_path)

                # Holen Sie sich die Ähnlichkeit aus der Ähnlichkeitsmatrix
                similarity = similarity_matrix[index_i][index_j]

                # Fügen Sie die Ähnlichkeit zur Genauigkeits-Summe hinzu
                accuracy_sum += similarity

        # Berechnen Sie den Durchschnitt der Genauigkeitswerte für dieses Bild
        average_accuracy = accuracy_sum / (num_images - 1)  # Das -1 ist, um das Bild selbst auszuschließen
        accuracy_values.append(average_accuracy)

    return accuracy_values

def cluster_images_new(image_path, features):
    #logger.info(f"Clustering images in directory: {image_path}")
    cluster_list = []
    image_path_new=[]
    for file in image_path:
        if os.path.exists(file):
            image_path_new.append(file)

    try:
        # Finde die Indizes der Features, die den Bildern in image_path entsprechen
        feature_indices = [i for i, path in enumerate(image_path) if path in image_path]

        # Wählen Sie nur die relevanten Features basierend auf den gefundenen Indizes aus
        relevant_features = [features[i] for i in feature_indices]

        # Check if there are relevant features to proceed
        if not relevant_features:
            logger.error("No relevant features found. Check input data.")
            return cluster_list, 0

        # Berechnen Sie die Ähnlichkeitsmatrix nur für die relevanten Features
        similarity_matrix = cosine_similarity(relevant_features)
        #print(similarity_matrix)
        threshold = 0.9
        num_images = len(image_path_new)
        percentage_similarity_matrix = (similarity_matrix + 1) / 2 * 100

        processed_images = set()

        for i in range(num_images):
            if image_path_new[i] not in processed_images:
                similar_group = []
                similar_group_accuracy = []

                for j in range(i, num_images):
                    if similarity_matrix[i, j] > threshold:
                        similar_group.append(os.path.relpath(image_path_new[j], start=app.static_folder))
                        similar_group_accuracy.append(similarity_matrix[i, j])
                        processed_images.add(image_path_new[j])

                if len(similar_group) > 1:
                    cluster = Cluster(os.path.relpath(image_path_new[i], start=app.static_folder), similar_group, similar_group_accuracy)
                    accuracy_values = calculate_cluster_accuracy(cluster, similarity_matrix)
                    cluster.accuracy_values = accuracy_values
                    cluster_list.append(cluster)

    except Exception as e:
        # Print the problematic image path
        logger.error(f"Error during clustering for directory {image_path_new}: {e}")
    
    #logger.info(f"Finished clustering for directory: {image_path}")
    cluster_count = len(cluster_list)
    return cluster_list, cluster_count

def create_thumbnail_dir():
    logger.debug("Creating thumbnail directory.")
    thumbnail_static_dir = os.path.join(app.static_folder, 'thumbnails')
    if not os.path.exists(thumbnail_static_dir):
        os.makedirs(thumbnail_static_dir, exist_ok=True)
        logger.info("Created thumbnail directory.")

def get_directories_with_images(root_directory):
    logger.info(f"Getting directories with images in: {root_directory}")
    directories_with_images = []
    for root, dirs, files in os.walk(root_directory):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.HEIC'))]
        if image_files:
            directories_with_images.append(root)
    return directories_with_images

def get_image_paths(directory):
    #logger.info(f"Getting image paths in directory: {directory}")
    thumbnail_static_dir = os.path.join(app.static_folder, 'thumbnails')
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                drive, path_without_drive = os.path.splitdrive(root)
                thumbnail_path = os.path.join(thumbnail_static_dir, path_without_drive.lstrip('/'))
                image_path=os.path.join(thumbnail_path, file)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
    return image_paths

def create_subfolders_in_static(directories):
    logger.debug(f"Creating subfolders in static for directories: {directories}")
    thumbnail_static_dir = os.path.join(app.static_folder, 'thumbnails')
    
    for directory in directories:
        drive, path_without_drive = os.path.splitdrive(directory)
        thumbnail_subfolder = os.path.join(thumbnail_static_dir, path_without_drive.lstrip('/'))
        if not os.path.exists(thumbnail_subfolder):
            os.makedirs(thumbnail_subfolder, exist_ok=True)

def generate_thumbnails_parallel(image_directories):
    logger.debug(f"Generating thumbnails in parallel for directories: {image_directories}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(generate_thumbnails_for_dir, image_directories)

def generate_thumbnails_for_dir(dir_path):
    logger.debug(f"Generating thumbnails for directory: {dir_path}")
    image_paths = [join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.HEIC'))]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(generate_thumbnail, image_paths)

def generate_thumbnail(image_path):
    logger.debug(f"Generating thumbnail for image: {image_path}")
    thumbnail_size = (500, 500)
    thumbnail_static_dir = os.path.join(app.static_folder, 'thumbnails')
    drive, path_without_drive = os.path.splitdrive(image_path)
    thumbnail_path = os.path.join(thumbnail_static_dir, path_without_drive.lstrip('/'))
    # Replace backslashes with forward slashes in thumbnail path
    thumbnail_path = thumbnail_path.replace('\\', '/')
    logger.debug(f"Generating thumbnail path: {thumbnail_path}")
    if not os.path.exists(thumbnail_path):
        img = Image.open(image_path)
        img.thumbnail(thumbnail_size)
        img.save(thumbnail_path)


def get_features_by_hashes(saved_features, saved_image_hashes, image_hashes):
    # Erstellen Sie ein Dictionary, das den Hash-Wert auf den Index in den Features abbildet
    hash_to_index = {hash_value: i for i, hash_value in enumerate(saved_image_hashes)}
    
    # Initialisieren Sie eine leere Liste für die ausgewählten Features
    selected_features = []

    # Iterieren Sie über die Hash-Werte der Bilder, für die Sie Features extrahieren möchten
    for hash_value in image_hashes:
        if hash_value in hash_to_index:
            feature_index = hash_to_index[hash_value]
            selected_feature = saved_features[feature_index]
            selected_features.append(selected_feature)

    return selected_features




# Funktion, um die Verzeichnisstruktur rekursiv zu durchsuchen und in ein JSON-Format umzuwandeln
def create_directory_structure(path):
    # Initialisieren Sie ein leeres Verzeichnisobjekt
    directory_object = {"text": os.path.basename(path), "children": []}

    # Durchlaufen Sie alle Elemente im aktuellen Verzeichnis
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        # Wenn es sich um ein Verzeichnis handelt, fügen Sie es als Kind hinzu
        if os.path.isdir(item_path):
            directory_object["children"].append(create_directory_structure(item_path))
    
    return directory_object


@app.route('/get_directory_structure')
def get_directory_structure():
    # Hier sollten Sie Ihre Verzeichnisstruktur in ein geeignetes JSON-Format umwandeln
    # Beispiel: [{'text': 'Ordner 1', 'children': [{'text': 'Unterverzeichnis', 'children': []}]}, ...]
    directory_structure = create_directory_structure("/pics") # Ersetzen Sie dies durch Ihre Daten
    
    return jsonify(directory_structure)

@app.route('/')
def index():
    logger.info("Serving index page.")
    create_thumbnail_dir()
    return render_template('index.html')

@app.route('/index', methods=['POST'])
def index_images():
    image_directories_array = request.form['image_directory']

    # Teilen Sie den String anhand des Kommas auf, um eine Liste von Verzeichnispfaden zu erhalten
    image_directories_list = [directory.strip().replace('\\', '/') for directory in image_directories_array.split(',')]
    logger.debug(image_directories_list)

    image_directories = []
    for directory in image_directories_list:
        logger.debug(directory)
        image_directory = directory.replace('\\', '/')
        image_directories.extend(get_directories_with_images(image_directory))

    create_subfolders_in_static(image_directories)
    generate_thumbnails_parallel(image_directories)

    return Response("Done!", content_type='text/event-stream')

@app.route('/cluster', methods=['POST'])
def cluster():
    image_directories_array = request.form['image_directory']

    # Teilen Sie den String anhand des Kommas auf, um eine Liste von Verzeichnispfaden zu erhalten
    image_directories_list = [directory.strip().replace('\\', '/') for directory in image_directories_array.split(',')]
    logger.debug(image_directories_list)

    image_directories = []
    for directory in image_directories_list:
        logger.debug(directory)
        image_directory = directory.replace('\\', '/')
        image_directories.extend(get_directories_with_images(image_directory))

    create_subfolders_in_static(image_directories)
    generate_thumbnails_parallel(image_directories)

    feature_path = 'saved_features.json'
    final_features=[]
    image_pathes = get_image_paths(image_directory)
    image_hashes_from_request=[calculate_image_hash(image_path) for image_path in image_pathes if image_path.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if os.path.exists(feature_path):
        logger.info("Using exsisting saved_features.json")
        saved_features, saved_image_hashes = load_features_and_identifiers(feature_path)
        # print(saved_features)
        # print(saved_image_hashes)
        new_image_paths = get_new_image_paths(image_pathes, saved_image_hashes)
        if len(new_image_paths) > 0:
            logger.info("New images found, which are not part of saved_features.json: " + str(new_image_paths))
            new_features = calculate_features(new_image_paths)
            # new_features = [np.squeeze(feature) for feature in new_features]
            final_features = np.vstack([saved_features, new_features])
            #final_features = saved_features + new_features
            combined_image_hashes = saved_image_hashes + [calculate_image_hash(image_path) for image_path in new_image_paths if image_path.lower().endswith(('.png', '.jpg', '.jpeg'))]
            save_features_and_identifiers(final_features, combined_image_hashes, 'saved_features.json')
            # Beispielaufruf der Funktion
            selected_features = get_features_by_hashes(final_features, combined_image_hashes, image_hashes_from_request)

        else:
            logger.info("No new images found, loading exsisting features")
            selected_features = get_features_by_hashes(saved_features, saved_image_hashes, image_hashes_from_request)
    else:
        logger.info("No saved_features.json found.")
        selected_features = calculate_features(image_pathes)
        save_features_and_identifiers(selected_features, image_hashes_from_request, 'saved_features.json')

    
    cluster_list, cluster_count = cluster_images_new(image_pathes,selected_features)  # Änderung hier
    return render_template('cluster.html', clusters=cluster_list, cluster_count=cluster_count)  # Änderung hier

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
