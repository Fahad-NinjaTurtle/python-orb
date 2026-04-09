from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import os

app = Flask(__name__)
CORS(app)

# In-memory DB: { id: { keypoints: [...], descriptors: np.array } }
image_database = {}

orb = cv2.ORB_create(nfeatures=1000)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def decode_image(b64_string):
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

def extract_features(img):
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def match_score(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0
    matches = bf.match(desc1, desc2)
    if len(matches) == 0:
        return 0
    # Count good matches (low Hamming distance)
    good = [m for m in matches if m.distance < 50]
    return len(good)

# POST /api/register  — called from registration tool
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    img_id    = data['id']
    b64_image = data['imageBase64']  # full image as base64

    img = decode_image(b64_image)
    kp, desc = extract_features(img)

    if desc is None or len(kp) < 10:
        return jsonify({ 'success': False, 'reason': 'not_enough_features' }), 400

    # Store descriptors as list for JSON serialization (persist to disk)
    image_database[img_id] = { 'descriptors': desc }

    # Optional: persist to disk so it survives restarts
    save_database()

    return jsonify({ 'success': True, 'keypoints': len(kp), 'total': len(image_database) })

# POST /api/identify  — called from Unity
@app.route('/api/identify', methods=['POST'])
def identify():
    data = request.get_json()
    b64_frame = data.get('imageBase64')

    if not b64_frame:
        return jsonify({ 'matched': False, 'reason': 'missing_image' }), 400

    img = decode_image(b64_frame)
    _, query_desc = extract_features(img)

    if query_desc is None:
        return jsonify({ 'matched': False, 'reason': 'no_features_detected' })

    best_id    = None
    best_score = 0
    MIN_GOOD_MATCHES = 15  # tune this threshold

    for img_id, entry in image_database.items():
        stored_desc = np.array(entry['descriptors'], dtype=np.uint8)
        score = match_score(query_desc, stored_desc)
        if score > best_score:
            best_score = score
            best_id = img_id

    if best_id and best_score >= MIN_GOOD_MATCHES:
        return jsonify({ 'matched': True, 'imageId': best_id, 'score': best_score })
    else:
        return jsonify({ 'matched': False, 'reason': 'no_match', 'bestScore': best_score })

# ── persistence ──────────────────────────────────────────────────────────────

DB_PATH = 'image_db.json'

def save_database():
    serializable = {}
    for k, v in image_database.items():
        serializable[k] = { 'descriptors': v['descriptors'].tolist() }
    with open(DB_PATH, 'w') as f:
        json.dump(serializable, f)

def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            raw = json.load(f)
        for k, v in raw.items():
            image_database[k] = { 'descriptors': np.array(v['descriptors'], dtype=np.uint8) }
        print(f"Loaded {len(image_database)} images from DB")

load_database()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port)