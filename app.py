import json
import logging
import requests
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, session
import os
import uuid
import threading
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from app3 import LLMDrivenBusinessOrchestrator

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_default_secret_key')
app.logger.setLevel(logging.INFO)

TASK_STATUS = {}

REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

with open('static/data/cities_with_districts.json', 'r', encoding='utf-8') as f:
    districts_data = json.load(f)

with open('static/data/business_categories.json', 'r', encoding='utf-8') as f:
    categories_data = json.load(f)

TAG_MAPPING = {
    "cafe": {"amenity": "cafe"},
    "books": {"shop": "books"},
    "lawyer": {"amenity": "lawyer"},
    "dessert": {"shop": "confectionery"},
    "restaurant": {"amenity": "restaurant"},
    "hospital": {"amenity": "hospital"},
    "school": {"amenity": "school"},
    "shopping_mall": {"shop": "mall"},
    "pharmacy": {"amenity": "pharmacy"},
    "optician": {"shop": "optician"},
    "beauty": {"shop": "beauty"},
    "hairdresser": {"amenity": "hairdresser"},
    "childcare": {"amenity": "childcare"},
    "college": {"amenity": "college"},
    "university": {"amenity": "university"},
    "training": {"amenity": "training"},
    "library": {"amenity": "library"},
    "museum": {"tourism": "museum"},
    "cinema": {"amenity": "cinema"},
    "theatre": {"amenity": "theatre"},
    "music": {"shop": "musical_instrument"},
    "games": {"shop": "games"},
    "sports": {"sport": "sports"},
    "pet": {"shop": "pet"},
    "second_hand": {"shop": "second_hand"},
    "art": {"shop": "art"},
    "florist": {"shop": "florist"},
}

def get_osm_type_local(category_name):
    for item in categories_data:
        if item["category"].lower() == category_name.lower():
            return item["type"]
    return None

def load_cities_from_json(filepath='static/data/cities_with_districts.json'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        app.logger.error(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        app.logger.error(f"JSON decode error in file: {filepath}")
        return {}

@app.route("/")
def index():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    session['session_id'] = session_id
    return render_template("index.html", session_id=session_id)

@app.route('/get_cities')
def get_cities():
    cities_data = load_cities_from_json()
    return jsonify(cities_data)

@app.route('/get_districts')
def get_districts():
    city = request.args.get('city')
    cities_data = load_cities_from_json()
    if city in cities_data:
        return jsonify(cities_data[city])
    else:
        return jsonify({"error": "City not found"}), 404

@app.route('/get_streets_for_district')
def get_streets_for_district():
    city = request.args.get('city')
    district = request.args.get('district')

    if not all([city, district]):
        app.logger.error("Şehir veya ilçe bilgisi eksik. Sokak sorgusu yapılamadı.")
        return jsonify({"error": "Şehir ve ilçe bilgileri gereklidir."}), 400

    app.logger.info(f"'{district}', '{city}' için sokaklar çekiliyor...")

    streets_query = f"""
    [out:json][timeout:180];
    area["name"="Türkiye"]->.country;
    area["name"="{city}"](area.country)->.city_area;
    area["name"="{district}"](area.city_area)->.district_area;
    (
        way["highway"~"^(residential|service|unclassified|pedestrian|living_street)$"](area.district_area);
    );
    out body;
    >;
    out skel qt;
    """

    street_names = []
    try:
        response = requests.post("https://overpass-api.de/api/interpreter", data=streets_query)
        response.raise_for_status()
        street_results = response.json()

        for element in street_results.get("elements", []):
            if "tags" in element and "name" in element["tags"]:
                name = element["tags"]["name"]
                street_names.append(name)

        app.logger.info(f"\n--- '{district}', '{city}' Sokakları (Frontend'e gönderiliyor) ---")
        for name in street_names:
            app.logger.info(f"Sokak: {name}")
        app.logger.info(f"Toplam {len(street_names)} sokak bulundu.")

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Overpass API'den sokakları çekerken hata oluştu: {e}")
        return jsonify({"error": f"Sokaklar çekilemedi: {e}"}), 500
    except json.JSONDecodeError as e:
        app.logger.error(f"Overpass API yanıtı JSON olarak ayrıştırılamadı (sokaklar): {e}")
        return jsonify({"error": "Sokak verisi formatı hatalı."}), 500

    return jsonify({"streets": street_names})

@app.route("/get_businesses", methods=["POST"])
def get_businesses():
    data = request.get_json()
    city = data.get("city")
    district = data.get("district")
    street = data.get("street")
    category = data.get("business_type")
    session_id = request.headers.get('X-Session-ID')

    if not all([city, district, street, category]):
        return jsonify({"error": "Missing required fields (city, district, street, business_type)"}), 400

    business_type = get_osm_type_local(category) # Use local get_osm_type_local
    if not business_type:
        return jsonify({'error': f'Unknown category: {category}'}), 400

    geolocator = Nominatim(user_agent="entrelocate-app")

    location_string = f"{street}, {district}, {city}, Türkiye"
    location = geolocator.geocode(location_string)

    if not location:
        app.logger.warning(f"Nominatim '{location_string}' için konum bulamadı. Daha geniş bir alan aranacak.")
        location_string_fallback = f"{district}, {city}, Türkiye"
        location = geolocator.geocode(location_string_fallback)
        if not location:
            return jsonify({'error': 'Belirtilen sokak veya ilçe/şehir için konum bulunamadı.'}), 404

    lat, lon, radius = location.latitude, location.longitude, 250  # Radius 250 m

    business_type = business_type.lower().strip()
    query_parts = []

    # Using the global TAG_MAPPING defined in app.py
    GENERIC_OSM_KEYS = ["amenity", "shop", "tourism", "leisure", "healthcare"]

    def _build_overpass_query(lat, lon, radius, business_type):
        query_parts = []
        if business_type in TAG_MAPPING:
            for key, value in TAG_MAPPING[business_type].items():
                query_parts += [
                    f'node["{key}"="{value}"](around:{radius},{lat},{lon});',
                    f'way["{key}"="{value}"](around:{radius},{lat},{lon});',
                    f'relation["{key}"="{value}"](around:{radius},{lat},{lon});',
                ]
        else:
            for key in GENERIC_OSM_KEYS:
                query_parts += [
                    f'node["{key}"="{business_type}"](around:{radius},{lat},{lon});',
                    f'way["{key}"="{business_type}"](around:{radius},{lat},{lon});',
                    f'relation["{key}"="{business_type}"](around:{radius},{lat},{lon});',
                ]
        query = f"""[out:json][timeout:25];({"".join(query_parts)});out center;"""
        return query


    app.logger.info(f"Overpass sorgusu '{category}' için '{location_string}' bölgesinde başlatılıyor.")
    query = _build_overpass_query(lat, lon, radius, business_type)
    try:
        response = requests.post("https://overpass-api.de/api/interpreter", data=query)
        response.raise_for_status()
        results = response.json()
        elements = results.get("elements", [])
    except Exception as e:
        app.logger.error(f"Overpass error in get_businesses: {e}")
        return jsonify({"error": f"Overpass error: {e}"}), 500

    for el in elements:
        if 'lat' not in el or 'lon' not in el:
            if 'center' in el:
                el['lat'] = el['center']['lat']
                el['lon'] = el['center']['lon']
    elements = [el for el in elements if el.get("lat") and el.get("lon")]

    with open("osm_filtered_businesses.json", "w", encoding="utf-8") as f:
        json.dump(elements, f, ensure_ascii=False, indent=2)

    app.logger.info(f"'{location_string}' bölgesinde {len(elements)} adet '{category}' iş yeri bulundu.")

    return jsonify({
        "data": elements,
        "lat": lat,
        "lon": lon,
        "business_count": len(elements)
    })

@app.route("/download", methods=["GET"])
def download():
    filepath = "osm_filtered_businesses.json"
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, as_attachment=True)


# REMOVED OLD /generate_report and REPLACED with the ASYNC FLOW BELOW

# --- NEW: This function runs the long analysis in a background thread ---
def run_analysis_task(task_id, city, district, street, business_type):
    """The target function for the background thread."""
    app.logger.info(f"Task {task_id}: Starting analysis for {business_type} in {district}.")

    # This is the callback function we'll pass to the orchestrator
    def update_progress(message, progress):
        app.logger.info(f"Task {task_id} progress: {progress}% - {message}")
        TASK_STATUS[task_id]['status'] = 'running'
        TASK_STATUS[task_id]['message'] = message
        TASK_STATUS[task_id]['progress'] = progress

    try:
        orchestrator = LLMDrivenBusinessOrchestrator()
        
        # Pass the callback function to the orchestrator
        result = orchestrator.analyze_business_viability(
            city=city, 
            district=district, 
            street=street, 
            business_type=business_type,
            progress_callback=update_progress
        )

        # Log final result and cost to terminal
        app.logger.info(f"\n--- Task {task_id} Final Report Data ---")
        app.logger.info(json.dumps(result, indent=2))
        app.logger.info("-------------------------------------\n")
        orchestrator.log_total_orchestration_cost()

        # Update the task status to 'completed' with the final result
        TASK_STATUS[task_id]['status'] = 'completed'
        TASK_STATUS[task_id]['progress'] = 100
        TASK_STATUS[task_id]['message'] = 'Analysis complete.'
        TASK_STATUS[task_id]['result'] = result

    except Exception as e:
        app.logger.error(f"Task {task_id}: Error generating report: {e}", exc_info=True)
        # Update the task status to 'failed'
        TASK_STATUS[task_id]['status'] = 'failed'
        TASK_STATUS[task_id]['message'] = f"An error occurred: {e}"
        TASK_STATUS[task_id]['progress'] = 100

# --- NEW: Endpoint to start the report generation task ---
@app.route("/start_report_generation", methods=["POST"])
def start_report_generation():
    data = request.get_json()
    city = data.get("city")
    district = data.get("district")
    street = data.get("street")
    business_type = data.get("business_type")

    if not all([city, district, business_type, street]):
        return jsonify({"error": "Missing city, district, business_type, or street"}), 400

    # Generate a unique ID for this task
    task_id = str(uuid.uuid4())

    # Initialize the status for this task
    TASK_STATUS[task_id] = {
        "status": "starting",
        "progress": 0,
        "message": "Initializing analysis...",
        "result": None
    }

    # Start the analysis in a background thread
    thread = threading.Thread(
        target=run_analysis_task,
        args=(task_id, city, district, street, business_type)
    )
    thread.daemon = True  # Allows main app to exit even if threads are running
    thread.start()

    # Return the task_id to the client immediately
    return jsonify({"task_id": task_id})

# --- NEW: Endpoint to get the status of a running task ---
@app.route("/get_report_status/<task_id>", methods=["GET"])
def get_report_status(task_id):
    task = TASK_STATUS.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    # To prevent sending the potentially large result every time, 
    # we can send it only when complete.
    response_data = {
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
    }
    if task["status"] == "completed":
        response_data["result"] = task["result"]

    return jsonify(response_data)

@app.route('/download_report/<filename>')
def download_report(filename):
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
