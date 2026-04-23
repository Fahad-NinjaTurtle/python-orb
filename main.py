import os
import cv2
import numpy as np
import pickle
import uuid
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ── Supabase (free tier — handles image storage + DB) ──────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── ORB config — grid-based for partial view tracking ─────────────────────────
ORB = cv2.ORB_create(
    nfeatures=800,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31
)

FLANN_INDEX_LSH = 6
INDEX_PARAMS = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2)
FLANN = cv2.FlannBasedMatcher(INDEX_PARAMS, {})


def extract_grid_descriptors(image_bgr, grid=4):
    """
    Extract ORB descriptors from a 4x4 grid to ensure
    keypoints across the whole image — enables partial view matching.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell_h, cell_w = h // grid, w // grid

    all_kp = []
    all_desc = []

    for row in range(grid):
        for col in range(grid):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = gray[y1:y2, x1:x2]

            kp, desc = ORB.detectAndCompute(cell, None)
            if desc is None or len(kp) == 0:
                continue

            # Offset keypoints to full-image coordinates
            for k in kp:
                k.pt = (k.pt[0] + x1, k.pt[1] + y1)

            all_kp.extend(kp)
            all_desc.append(desc)

    if not all_desc:
        return None, None

    stacked = np.vstack(all_desc)
    return all_kp, stacked


def rebuild_flann_index():
    """
    Rebuild FLANN index from all descriptors in DB.
    Called after every new image upload.
    Returns serialized index bytes.
    """
    response = supabase.table("artworks").select("id, descriptors").execute()
    rows = response.data

    if not rows:
        return None, []

    all_desc = []
    id_map = []  # parallel array: descriptor_row_index → artwork_id

    for row in rows:
        desc_list = json.loads(row["descriptors"])
        desc_array = np.array(desc_list, dtype=np.uint8)
        all_desc.append(desc_array)
        id_map.extend([row["id"]] * len(desc_array))

    stacked = np.vstack(all_desc)

    flann = cv2.FlannBasedMatcher(INDEX_PARAMS, {})
    flann.add([stacked])
    flann.train()

    index_data = {
        "id_map": id_map,
        "descriptors": stacked.tolist()
    }

    return index_data, id_map


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/upload", methods=["POST"])
def upload_artwork():
    """
    Receives: multipart form with 'image' file + optional 'video_url' field
    1. Uploads image to Supabase Storage (public bucket)
    2. Extracts ORB descriptors (grid-based)
    3. Saves to artworks table
    4. Returns artwork id + public image url
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    video_url  = request.form.get("video_url", "")
    title      = request.form.get("title", "Untitled")

    # Read image bytes
    img_bytes = image_file.read()
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    img_bgr   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Invalid image"}), 400

    # Extract grid-based ORB descriptors
    _, descriptors = extract_grid_descriptors(img_bgr)
    if descriptors is None:
        return jsonify({"error": "Could not extract features from image"}), 400

    # Upload image to Supabase Storage
    artwork_id  = str(uuid.uuid4())
    file_ext    = image_file.filename.rsplit(".", 1)[-1].lower() if "." in image_file.filename else "jpg"
    storage_path = f"artworks/{artwork_id}.{file_ext}"

    supabase.storage.from_("ar-images").upload(
        path=storage_path,
        file=img_bytes,
        file_options={"content-type": image_file.content_type or "image/jpeg"}
    )

    # Get public URL
    public_url = supabase.storage.from_("ar-images").get_public_url(storage_path)

    # Estimate physical width (default 0.3m — can be overridden)
    physical_width = float(request.form.get("physical_width", 0.3))

    # Save to DB
    supabase.table("artworks").insert({
        "id":             artwork_id,
        "title":          title,
        "image_url":      public_url,
        "video_url":      video_url,
        "physical_width": physical_width,
        "descriptors":    json.dumps(descriptors.tolist()),
        "created_at":     datetime.utcnow().isoformat()
    }).execute()

    return jsonify({
        "success":   True,
        "id":        artwork_id,
        "image_url": public_url,
        "message":   f"Extracted {len(descriptors)} descriptors"
    })


@app.route("/api/descriptors", methods=["GET"])
def get_descriptors():
    """
    Returns the full FLANN-ready descriptor bundle.
    Unity downloads this once at app startup (~5MB for 1000 images).
    Format: { id_map: [...], descriptors: [[...], ...] }
    """
    index_data, _ = rebuild_flann_index()

    if index_data is None:
        return jsonify({"id_map": [], "descriptors": []})

    return jsonify(index_data)


@app.route("/api/manifest", methods=["GET"])
def get_manifest():
    """
    Returns lightweight metadata for all artworks.
    No heavy data — just ids, physical sizes, video urls.
    Unity uses this to know physicalWidth for AddReferenceImage.
    """
    response = supabase.table("artworks") \
        .select("id, title, physical_width, video_url") \
        .execute()

    artworks = [
        {
            "id":             row["id"],
            "title":          row["title"],
            "physicalWidth":  row["physical_width"],
            "videoUrl":       row["video_url"]
        }
        for row in response.data
    ]

    return jsonify({"artworks": artworks})


@app.route("/api/artwork/<artwork_id>", methods=["GET"])
def get_artwork(artwork_id):
    """
    Called after FLANN match confirmed on device.
    Returns only the video URL — no heavy data.
    """
    response = supabase.table("artworks") \
        .select("id, title, video_url, physical_width") \
        .eq("id", artwork_id) \
        .single() \
        .execute()

    if not response.data:
        return jsonify({"error": "Not found"}), 404

    row = response.data
    return jsonify({
        "id":            row["id"],
        "title":         row["title"],
        "videoUrl":      row["video_url"],
        "physicalWidth": row["physical_width"]
    })


@app.route("/api/artworks", methods=["GET"])
def list_artworks():
    """Admin endpoint — list all artworks with full details."""
    response = supabase.table("artworks") \
        .select("id, title, image_url, video_url, physical_width, created_at") \
        .order("created_at", desc=True) \
        .execute()

    return jsonify({"artworks": response.data})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
