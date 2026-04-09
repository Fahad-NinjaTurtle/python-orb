from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import os

app = Flask(__name__)
CORS(app)

image_database = {}

orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

DB_PATH = 'image_db.json'


def decode_image(b64_string):
    try:
        img_bytes = base64.b64decode(b64_string)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def extract_features(img):
    if img is None:
        return [], None
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints or [], descriptors


def match_score(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0

    matches = bf.match(desc1, desc2)
    if not matches:
        return 0

    good = [m for m in matches if m.distance < 50]
    return len(good)


def save_database():
    serializable = {}
    for image_id, entry in image_database.items():
        serializable[image_id] = {
            'descriptors': entry['descriptors'].tolist()
        }

    with open(DB_PATH, 'w') as f:
        json.dump(serializable, f)


def load_database():
    if not os.path.exists(DB_PATH):
        return

    with open(DB_PATH, 'r') as f:
        raw = json.load(f)

    for image_id, entry in raw.items():
        image_database[image_id] = {
            'descriptors': np.array(entry['descriptors'], dtype=np.uint8)
        }

    print(f"Loaded {len(image_database)} images from DB")


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'images': len(image_database)
    })


@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}

    img_id = data.get('id')
    b64_image = data.get('imageBase64')

    if not img_id:
        return jsonify({'success': False, 'reason': 'missing_id'}), 400

    if not b64_image:
        return jsonify({'success': False, 'reason': 'missing_image'}), 400

    img = decode_image(b64_image)
    if img is None:
        return jsonify({'success': False, 'reason': 'invalid_image'}), 400

    kp, desc = extract_features(img)
    if desc is None or len(kp) < 10:
        return jsonify({'success': False, 'reason': 'not_enough_features'}), 400

    image_database[img_id] = {
        'descriptors': desc
    }

    save_database()

    return jsonify({
        'success': True,
        'keypoints': len(kp),
        'total': len(image_database)
    })


@app.route('/api/identify', methods=['POST'])
def identify():
    data = request.get_json(silent=True) or {}
    b64_frame = data.get('imageBase64')

    if not b64_frame:
        return jsonify({'matched': False, 'reason': 'missing_image'}), 400

    img = decode_image(b64_frame)
    if img is None:
        return jsonify({'matched': False, 'reason': 'invalid_image'}), 400

    _, query_desc = extract_features(img)
    if query_desc is None:
        return jsonify({'matched': False, 'reason': 'no_features_detected'}), 200

    best_id = None
    best_score = 0
    min_good_matches = 15

    for img_id, entry in image_database.items():
        stored_desc = np.array(entry['descriptors'], dtype=np.uint8)
        score = match_score(query_desc, stored_desc)

        if score > best_score:
            best_score = score
            best_id = img_id

    if best_id and best_score >= min_good_matches:
        return jsonify({
            'matched': True,
            'imageId': best_id,
            'score': best_score
        })

    return jsonify({
        'matched': False,
        'reason': 'no_match',
        'bestScore': best_score
    })


load_database()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port)