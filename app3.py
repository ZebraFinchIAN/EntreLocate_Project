# app3.py
import os
import json
import logging
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from google.api_core import exceptions
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import requests
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("Gemini API Key not configured. Please set 'GOOGLE_API_KEY' environment variable.")
    raise SystemExit("Gemini API Key missing.")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logging.info("Gemini API Key configured successfully.")
except Exception as e:
    logging.error(f"Gemini configuration error: {e}")
    raise SystemExit(f"Gemini configuration error: {e}")

# --- RAG & LLM Agent Prompts ---
USER_DEFINED_GENERATION_CONFIG_DICT = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

RAG_SYNTHESIS_ROLE = """
You are an expert data extraction and structuring analyst.
Given a single text block containing all available data for a specific Turkish district, your task is to meticulously extract the factual information and structure it into a JSON object.
The text block is a comma-separated list of key-value pairs derived from a database row.

The JSON object you create MUST have the following keys. Populate them using the data from the provided text block. If a piece of information isn't in the text, use "N/A" or an empty list as appropriate.
- "district_overall_socio_economic_rank_turkiye": (Extract from 'SEGE_2022_Rank_Turkiye')
- "district_socio_economic_rank_within_city": (Extract from 'Rank_Within_City')
- "dominant_age_groups": (Extract from 'Dominant_Age_Groups')
- "population_gender_ratio_comment": (Extract from 'Population_Gender_Ratio')
- "household_expenditure_highlights": (Extract from 'Household_Expenditure_Highlights')
- "general_business_openings_in_district": (Extract the number from 'Business_Openings_2023')
- "general_business_closures_in_district": (Extract the number from 'Business_Closures_2023')
- "trends_in_related_sectors": (Extract from 'Key_Trends')
- "key_opportunities_identified_in_docs": (Extract from 'Key_Opportunities')
- "key_challenges_identified_in_docs": (Extract from 'Key_Challenges')

Focus solely on extracting information directly from the provided text. Do not invent or infer data not present.
"""

COMPREHENSIVE_MARKET_RISK_ROLE = """
You are an expert Business Market and Risk Analyst. Your goal is to provide a comprehensive qualitative analysis for opening a 'target_business_type' in a given 'district_name'.
You will receive:
1. target_business_type: The business being considered.
2. district_name: The district.
3. api_business_data_str: A JSON string from an API lookup showing nearby existing businesses.
4. rag_socio_economic_data_str: A JSON string with factual socio-economic data for the district.

Your Task is to analyze all the provided information and produce a single, well-structured JSON object.
Your primary role is to provide SCORES for the graded factors based on your analysis. For each score, 100 is the best possible outcome (e.g., 100 demand, 0 competition) and 0 is the worst.
- For 'competition_level_score', a higher score means MORE competition (higher risk). So a score of 80 is high risk.
- For all other scores ('demand_potential_score', 'socio_economic_environment_score', 'rag_data_quality_score'), a higher score is BETTER (e.g., a demand score of 80 is very good).

DO NOT provide a 'final_risk_score' or a 'recommendation_level'. These will be calculated by a separate tool.

The final JSON MUST contain these keys:
  "overall_market_narrative" (A 3-5 paragraph professional analysis synthesizing all inputs),
  "graded_factors" (A dictionary with keys: "demand_potential_score", "competition_level_score", "socio_economic_environment_score", "rag_data_quality_score"),
  "key_opportunities_for_target_business" (A list of specific opportunities for the target business),
  "key_challenges_for_target_business" (A list of specific challenges for the target business),
  "risk_score_justification" (A text paragraph explaining your reasoning behind the graded factor scores),
  "strategic_recommendations" (A list of 3-5 actionable recommendations for the business owner),
  "dos_and_donts" (A dictionary with two keys: "dos" and "donts", each containing a list of concise bullet points)

Focus on providing high-quality qualitative text and accurate graded factor scores based on the provided data.
"""

ORCHESTRATOR_LLM_PROMPT = """
You are an expert business analysis orchestrator. Your goal is to create a step-by-step plan to analyze the viability of a new business.

User Request Details:
City: {city}
District: {district}
Business Type: {business_type}
Street: {street}

Available Agents/Tools:
1. BusinessDataFetcher: -> inputs: {{"city","district","category"}} , outputs: {{data, lat, lon, business_count, error, details}}
2. SocioEconomicRAGAgent: -> inputs: {{"city_name", "district_name"}}, outputs: JSON’d RAG synth data
3. ComprehensiveMarketRiskAgent: -> inputs: {{"target_business_type","district_name","api_business_data","rag_socio_economic_data"}}, outputs: JSON with qualitative analysis and graded_factors.
4. ScoreCalculator: -> inputs: {{"graded_factors"}}, outputs: {{"final_risk_score", "recommendation_level"}}
5. ReportGeneratorAgent: -> inputs: {{"analysis_result", "scoring_result", "business_type", "district_name", "street_name"}}, outputs: {{"report_filename", "report_summary", "pdf_file_path"}}

Return a step-by-step plan as a JSON object. The final score MUST be calculated in a separate step AFTER the main analysis.

Return strictly:
{{
  "initial_assessment": "...",
  "steps": [
    {{ "agent_to_call": "BusinessDataFetcher", "inputs_for_agent": {{ "city": "{city}", "district": "{district}", "category": "{business_type}" }} , "purpose_of_this_step": "Fetch existing business data from the local area using an API." }},
    {{ "agent_to_call": "SocioEconomicRAGAgent", "inputs_for_agent": {{ "city_name": "{city}", "district_name": "{district}" }}, "purpose_of_this_step": "Fetch detailed socio-economic data for the specified district from the database." }},
    {{ "agent_to_call": "ComprehensiveMarketRiskAgent", "inputs_for_agent": {{ "target_business_type": "{business_type}", "district_name": "{district}", "api_business_data": "OUTPUT_OF_STEP_0", "rag_socio_economic_data": "OUTPUT_OF_STEP_1" }} , "purpose_of_this_step": "Perform a detailed qualitative analysis and generate graded factor scores by synthesizing API and RAG data." }},
    {{ "agent_to_call": "ScoreCalculator", "inputs_for_agent": {{ "graded_factors": "OUTPUT_OF_STEP_2.graded_factors" }}, "purpose_of_this_step": "Calculate the final risk score and recommendation level based on the graded factors." }},
    {{ "agent_to_call": "ReportGeneratorAgent", "inputs_for_agent": {{ "analysis_result": "OUTPUT_OF_STEP_2", "scoring_result": "OUTPUT_OF_STEP_3", "business_type": "{business_type}", "district_name": "{district}", "street_name": "{street}" }} , "purpose_of_this_step": "Generate a final PDF report combining the qualitative analysis and the final calculated score." }}
  ]
}}
"""

# --- RAG DATA LOADING & PROCESSING ---
def load_and_process_csv_for_rag(csv_path='All_Districts_Data.csv'):
    """Loads the district data from the CSV, creates a document for each row, and prepares it for ChromaDB."""
    if not os.path.exists(csv_path):
        logging.error(f"FATAL: The required data file '{csv_path}' was not found.")
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    # Use a raw string or double backslashes for the path
    df = pd.read_csv(csv_path, encoding='utf-8')
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns] # Clean column names
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # Clean string data
    df.fillna("N/A", inplace=True) # Replace NaN with "N/A" string

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # Create a single text block for the entire row
        doc_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
        documents.append(doc_text)

        # Create metadata for filtering. Ensure values are strings.
        metadata = {
            "city": str(row['City']),
            "district": str(row['District'])
        }
        metadatas.append(metadata)

        # Create a unique ID
        doc_id = f"{row['City']}_{row['District']}_{index}"
        ids.append(doc_id)
        
    logging.info(f"Processed {len(df)} rows from {csv_path} for RAG.")
    return documents, metadatas, ids

# --- CHROMADB SETUP ---
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection_name = "Turkish_Districts_SocioEconomic_Data"

# Clean up old collection if it exists
if collection_name in [col.name for col in chroma_client.list_collections()]:
    logging.warning(f"Deleting existing ChromaDB collection: '{collection_name}'")
    chroma_client.delete_collection(name=collection_name)

chroma_collection = chroma_client.create_collection(
    name=collection_name,
    embedding_function=embedding_function
)

# --- POPULATE CHROMADB FROM CSV ---
try:
    documents, metadatas, ids = load_and_process_csv_for_rag('All_Districts_Data.csv')
    
    # Add data to the collection in batches for efficiency
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        chroma_collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    logging.info(f"Successfully added {len(documents)} district documents to ChromaDB collection '{collection_name}'.")
except Exception as e:
    logging.error(f"Failed to populate ChromaDB collection: {e}", exc_info=True)
    raise

# --- AGENT & ORCHESTRATOR CLASSES ---
class Agent:
    INPUT_PRICE_PER_MILLION = 0.075
    OUTPUT_PRICE_PER_MILLION = 0.30

    def __init__(self, name: str, role: str, model_name: str = "gemini-1.5-flash-latest",
                generation_config_dict=None, safety_settings=None):
        self.name = name
        self.role = role
        self.model_name = model_name
        self.logger = logging.getLogger(f"Agent.{self.name}")
        self.logger.info(f"Initializing Agent '{self.name}' with model {self.model_name}...")
        config_dict = generation_config_dict.copy() if generation_config_dict else USER_DEFINED_GENERATION_CONFIG_DICT.copy()
        for k, v in USER_DEFINED_GENERATION_CONFIG_DICT.items():
            config_dict.setdefault(k, v)
        self._base_generation_config_dict = config_dict.copy()
        self.generation_config_instance = genai.types.GenerationConfig(**self._base_generation_config_dict)
        self.safety_settings_instance = safety_settings if safety_settings else DEFAULT_SAFETY_SETTINGS
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.role,
            generation_config=self.generation_config_instance,
            safety_settings=self.safety_settings_instance
        )
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_total_tokens = 0

    def generate_response(self, prompt_content: str, expect_json: bool = False) -> str:
        self.logger.info(f"[{self.name}] Generating response (ilk 100 char): {prompt_content[:100]}...")
        current_cfg = self._base_generation_config_dict.copy()
        if expect_json:
            current_cfg["response_mime_type"] = "application/json"
        else:
            current_cfg.pop("response_mime_type", None)
        final_cfg = genai.types.GenerationConfig(**current_cfg)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"[{self.name}] API çağrısı deneme {attempt+1}/{max_retries}...")
                response = self.model.generate_content(contents=[prompt_content], generation_config=final_cfg)
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    prompt_toks = getattr(response.usage_metadata, 'prompt_token_count', len(prompt_content.split()))
                    cand_toks = getattr(response.usage_metadata, 'candidates_token_count', len(response.text.split()))
                else:
                    prompt_toks = len(prompt_content.split())
                    cand_toks = len(response.text.split())
                self.session_input_tokens += prompt_toks
                self.session_output_tokens += cand_toks
                self.session_total_tokens += (prompt_toks + cand_toks)
                return response.text
            except exceptions.TooManyRequests as e:
                self.logger.warning(f"[{self.name}] Rate limit: {e}. Retry...")
                time.sleep(5 * (2 ** attempt))
            except Exception as e:
                self.logger.error(f"[{self.name}] API çağrısında hata: {e}", exc_info=True)
                raise
        raise RuntimeError(f"[{self.name}] Üç denemede de başarısız oldu.")

    def calculate_and_log_session_cost(self) -> float:
        cost_in = (self.session_input_tokens / 1_000_000) * Agent.INPUT_PRICE_PER_MILLION
        cost_out = (self.session_output_tokens / 1_000_000) * Agent.OUTPUT_PRICE_PER_MILLION
        total_cost = cost_in + cost_out
        self.logger.info(f"[{self.name}] Session cost report: Input tokens {self.session_input_tokens}, Output tokens {self.session_output_tokens}, Toplam maliyet: {total_cost:.6f} USD")
        return total_cost

    def reset_session_costs_and_tokens(self):
        self.logger.info(f"[{self.name}] Reset session tokens/costs")
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_total_tokens = 0

class BusinessDataFetcherAgent:
    OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
    HEADERS = {'User-Agent': 'entrelocate-app'}
    RADIUS_METERS = 2500
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
    GENERIC_OSM_KEYS = ["amenity", "shop", "tourism", "leisure", "healthcare"]

    def __init__(self, categories_data_path: str = 'static/data/business_categories.json'):
        self.logger = logging.getLogger("Agent.BusinessDataFetcher")
        self.logger.info("Initialized BusinessDataFetcherAgent for Overpass API calls.")
        self.geolocator = Nominatim(user_agent="entrelocate-app")
        try:
            with open(categories_data_path, 'r', encoding='utf-8') as f:
                self.categories_data = json.load(f)
            self.logger.info(f"Loaded business categories from {categories_data_path}")
        except Exception as e:
            self.logger.error(f"Kategori verisi yüklenemedi: {e}")
            self.categories_data = []
        self.requests = requests # Assign requests module

    def _get_osm_type_from_category(self, category_name: str) -> str | None:
        for item in self.categories_data:
            if item.get("category", "").lower() == category_name.lower():
                return item.get("type")
        self.logger.warning(f"Kategori bulunamadı: {category_name}")
        return None

    def _build_overpass_query(self, lat: float, lon: float, radius: int, osm_type: str) -> str:
        query_parts = []
        if osm_type in self.TAG_MAPPING:
            for key, value in self.TAG_MAPPING[osm_type].items():
                query_parts.extend([
                    f'node["{key}"="{value}"](around:{radius},{lat},{lon});',
                    f'way["{key}"="{value}"](around:{radius},{lat},{lon});',
                    f'relation["{key}"="{value}"](around:{radius},{lat},{lon});',
                ])
        else:
            self.logger.info(f"No TAG_MAPPING for '{osm_type}', using generic keys.")
            for key in self.GENERIC_OSM_KEYS:
                query_parts.extend([
                    f'node["{key}"="{osm_type}"](around:{radius},{lat},{lon});',
                    f'way["{key}"="{osm_type}"](around:{radius},{lat},{lon});',
                    f'relation["{key}"="{osm_type}"](around:{radius},{lat},{lon});',
                ])
        query = f"[out:json][timeout:25];({' '.join(query_parts)});out center;"
        return query

    def fetch_business_data(self, city: str, district: str, category: str) -> dict:
        return_data = {"data": [], "lat": None, "lon": None, "business_count": 0, "error": None, "details": None}
        try:
            location = self.geolocator.geocode(f"{district}, {city}", timeout=10)
            if not location:
                return_data["error"] = "Location not found."
                return return_data
            lat, lon = location.latitude, location.longitude
            return_data["lat"], return_data["lon"] = lat, lon
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            return_data["error"] = f"Geocoding error: {e}"
            return return_data
        except Exception as e:
            return_data["error"] = f"Unexpected geocoding error: {e}"
            return return_data

        osm_type = self._get_osm_type_from_category(category)
        if not osm_type:
            return_data["error"] = f"Unknown category: {category}"
            return return_data
        osm_type = osm_type.lower().strip()
        query = self._build_overpass_query(lat, lon, self.RADIUS_METERS, osm_type)

        try:
            response = self.requests.post(self.OVERPASS_API_URL, data=query, headers=self.HEADERS, timeout=30)
            response.raise_for_status()
            results = response.json()
            elements = results.get("elements", [])
        except Exception as e:
            return_data["error"] = f"Overpass hata: {e}"
            return return_data

        processed = []
        for el in elements:
            if 'lat' not in el or 'lon' not in el:
                if 'center' in el:
                    el['lat'] = el['center'].get('lat')
                    el['lon'] = el['center'].get('lon')
            if el.get("lat") is not None and el.get("lon") is not None:
                processed.append(el)
        return_data["data"] = processed
        return_data["business_count"] = len(processed)

        try:
            with open("osm_filtered_businesses.json", "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return_data["error"] = f"JSON kaydedilemedi: {e}"

        return return_data

# --- REFACTORED RAG AGENT ---
class RAGRetrieverAgent:
    def __init__(self, synthesis_model_name: str = "gemini-1.5-flash-latest"):
        self.logger = logging.getLogger("Agent.SocioEconomicRAGAgent")
        self.logger.info("SocioEconomicRAGAgent initialized; using global chroma_collection.")
        # This agent uses the globally defined and populated chroma_collection
        self.collection = chroma_collection
        self.synthesis_llm = Agent(name="RAGSynthesizer", role=RAG_SYNTHESIS_ROLE, model_name=synthesis_model_name)
        self.total_synthesis_llm_cost = 0.0

    def _query_district_data(self, city_name: str, district_name: str) -> list[dict]:
        """Queries ChromaDB for a specific district using precise metadata filters."""
        if not self.collection:
            self.logger.error("ChromaDB collection is not available.")
            return []
        try:
            self.logger.info(f"RAG: Querying for City='{city_name}' AND District='{district_name}' using metadata filter.")
            # Use a precise metadata filter instead of a semantic search
            results = self.collection.get(
                where={
                    "$and": [
                        {"city": {"$eq": city_name}},
                        {"district": {"$eq": district_name}}
                    ]
                },
                include=['documents', 'metadatas']
            )

            if not results or not results.get('ids'):
                self.logger.warning(f"No document found in RAG for {district_name}, {city_name}.")
                return []

            # Since each district is a unique row, we expect one result.
            document = results['documents'][0]
            metadata = results['metadatas'][0]
            self.logger.info(f"RAG: Successfully retrieved 1 document for {district_name}, {city_name}.")
            return [{"document": document, "metadata": metadata}]
        except Exception as e:
            self.logger.error(f"ChromaDB get/query error for {district_name}: {e}", exc_info=True)
            return []

    def fetch_rag_data(self, city_name: str, district_name: str) -> dict:
        """Fetches and synthesizes socio-economic data for a given district."""
        self.logger.info(f"RAG: Fetching socio-economic data for '{district_name}, {city_name}'")

        retrieved_docs = self._query_district_data(city_name, district_name)

        error_response = {
            "error": f"No relevant RAG documents found for {district_name}, {city_name}."
        }

        if not retrieved_docs:
            return error_response

        # We now have a single, rich document text containing all data for the district
        context = retrieved_docs[0]["document"]

        prompt_content = f"""
        District: {district_name}
        City: {city_name}

        Here is the complete data for this district from our database:
        ---
        {context}
        ---

        Based ONLY on the provided data block, extract the information and structure it into the requested JSON format.
        """

        try:
            response_json_str = self.synthesis_llm.generate_response(prompt_content, expect_json=True)
            output = json.loads(response_json_str)
            # Since we found a direct match, confidence is high
            output["rag_synthesis_confidence"] = "High - Direct Match"
            self.total_synthesis_llm_cost += self.synthesis_llm.calculate_and_log_session_cost()
            return output
        except Exception as e:
            self.logger.error(f"RAG synthesis failed for {district_name}: {e}", exc_info=True)
            error_response["error"] = "RAG synthesis failed."
            return error_response
    
    def reset_session_costs_and_tokens(self):
        self.synthesis_llm.reset_session_costs_and_tokens()
        self.total_synthesis_llm_cost = 0.0

# --- REVISED COMPREHENSIVE AGENT ---
class ComprehensiveMarketRiskAgent(Agent):
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", generation_config_dict=None):
        default_cfg = generation_config_dict.copy() if generation_config_dict else USER_DEFINED_GENERATION_CONFIG_DICT.copy()
        super().__init__(name="ComprehensiveMarketRiskAgent", role=COMPREHENSIVE_MARKET_RISK_ROLE,
                        model_name=model_name, generation_config_dict=default_cfg)

    def analyze_and_score_risk(self, target_business_type: str, district_name: str,
                           api_business_data: dict, rag_socio_economic_data: dict) -> dict:
        api_str = json.dumps(api_business_data, ensure_ascii=False)
        rag_str = json.dumps(rag_socio_economic_data, ensure_ascii=False)
        prompt = f"""
        Please perform a comprehensive market analysis for opening a '{target_business_type}'
        in the district '{district_name}'.

        API Business Data (existing nearby businesses): --- {api_str} ---
        RAG Socio-Economic Data (district profile): --- {rag_str} ---

        Produce the final JSON object with all the requested keys as per your role.
        """
        try:
            response_json_str = self.generate_response(prompt, expect_json=True)
            output = json.loads(response_json_str)
            # This agent ONLY produces the qualitative analysis and graded scores.
            required = [
                "overall_market_narrative", "graded_factors", "key_opportunities_for_target_business",
                "key_challenges_for_target_business", "risk_score_justification",
                "strategic_recommendations", "dos_and_donts"
            ]
            # Validate that all required keys are present
            for k in required:
                if k not in output:
                    output[k] = "Error: Key missing from LLM response."
            
            return output
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}", exc_info=True)
            return {"error": f"ComprehensiveAgent JSON parsing failed: {e}"}

# --- SCORING & REPORTING ---
def calculate_final_risk_score(graded_factors: dict) -> int:
    """Calculates a final risk score based on a weighted average of graded factors."""
    weights = {
        'competition_level_score': 0.40,
        'socio_economic_environment_score': 0.25,
        'demand_potential_score': 0.20,
        'rag_data_quality_score': 0.15,
    }
    final_score = 0.0
    competition = graded_factors.get('competition_level_score', 50)
    final_score += competition * weights['competition_level_score']
    demand = 100 - graded_factors.get('demand_potential_score', 50)
    final_score += demand * weights['demand_potential_score']
    socio_economic = 100 - graded_factors.get('socio_economic_environment_score', 50)
    final_score += socio_economic * weights['socio_economic_environment_score']
    rag_data = 100 - graded_factors.get('rag_data_quality_score', 50)
    final_score += rag_data * weights['rag_data_quality_score']
    return int(round(final_score))

class ReportGeneratorAgent:
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        self.logger = logging.getLogger("Agent.ReportGenerator")
        self.model_name = model_name
        self.llm = Agent(name="ReportSummarizer", role="You are a report writer.", model_name=self.model_name)
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)
        self.total_report_llm_cost = 0.0
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='JustifyTurkish', parent=self.styles['Normal'], alignment=4, encoding='utf-8'))
        self.styles.add(ParagraphStyle(name='Heading1Turkish', parent=self.styles['Heading1'], encoding='utf-8', spaceAfter=14))
        self.styles.add(ParagraphStyle(name='Heading2Turkish', parent=self.styles['Heading2'], encoding='utf-8', spaceAfter=6))
        self.style_to_use = 'JustifyTurkish'
        try:
            pdfmetrics.registerFont(TTFont('ArialUnicode', 'ArialUnicodeMS.ttf')) 
            self.styles.add(ParagraphStyle(name='TurkishFont', parent=self.styles['Normal'], alignment=0, encoding='utf-8', fontName='ArialUnicode'))
            self.font_to_use = 'TurkishFont'
        except Exception as e:
            self.logger.warning(f"ArialUnicode font not found. Using default. Error: {e}")
            self.font_to_use = 'Normal'

    def generate_report(self, analysis_result: dict, business_type: str, district_name: str, street_name: str, cost_token_data: dict = None) -> tuple[str, str, str]:
        try:
            # 1. Create the main body of the report
            report_elements = self._create_natural_language_report(analysis_result, business_type, district_name, street_name)
            
            # 2. Generate summary (this has an associated LLM cost)
            report_content = ''.join([element.text for element in report_elements if hasattr(element, 'text')])
            summary = self._summarize_report(report_content) # This populates self.llm token/cost info

            # 3. Aggregate all costs and tokens for the final report
            # Start with the cost from this agent's summarizer call
            summarizer_cost_in = (self.llm.session_input_tokens / 1_000_000) * Agent.INPUT_PRICE_PER_MILLION
            summarizer_cost_out = (self.llm.session_output_tokens / 1_000_000) * Agent.OUTPUT_PRICE_PER_MILLION
            summarizer_cost = summarizer_cost_in + summarizer_cost_out
            
            total_input_tokens = self.llm.session_input_tokens
            total_output_tokens = self.llm.session_output_tokens
            total_cost = summarizer_cost

            # Add costs from previous steps passed in via the orchestrator
            if cost_token_data:
                for agent_name, data in cost_token_data.items():
                    total_input_tokens += data.get('input', 0)
                    total_output_tokens += data.get('output', 0)
                    total_cost += data.get('cost', 0.0)

            total_tokens = total_input_tokens + total_output_tokens

            # 4. Get current timestamp for the report
            generation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 5. Create and add the new metadata section to the report elements
            metadata_elements = self._create_metadata_section(
                total_cost, total_tokens, generation_timestamp
            )
            report_elements.extend(metadata_elements)

            # 6. Create the final PDF from all elements
            pdf_file_path = self._create_pdf_report(report_elements, business_type, district_name, street_name)
            
            return os.path.basename(pdf_file_path), summary, pdf_file_path
        except Exception as e:
            self.logger.error(f"Error generating report: {e}", exc_info=True)
            return None, f"Error generating report: {e}", None

    def _create_metadata_section(self, total_cost: float, total_tokens: int, timestamp: str) -> list:
        """Creates a list of Paragraphs for the report's metadata section."""
        elements = []
        spacer = Spacer(1, 0.2 * inch)
        
        elements.append(spacer)

        metadata_style = ParagraphStyle(
            name='MetadataStyle',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=12,
            textColor='grey'
        )
        
        elements.append(Paragraph("<b>Analysis Execution Details</b>", self.styles['Heading2Turkish']))
        elements.append(Paragraph(f"<b>Report Generated:</b> {timestamp}", metadata_style))
        elements.append(Paragraph(f"<b>Total Tokens Used (Approx.):</b> {total_tokens:,}", metadata_style))
        elements.append(Paragraph(f"<b>Estimated Total LLM Cost:</b> ${total_cost:.6f} USD", metadata_style))
        elements.append(spacer)
        
        return elements

    def _create_natural_language_report(self, analysis_result: dict, business_type: str, district_name: str, street_name: str) -> list:
        elements = []
        spacer = Spacer(1, 0.2 * inch)

        def add_list_section(title, items, bullet="•"):
            elements.append(Paragraph(f"<b>{title}</b>", self.styles['Heading2Turkish']))
            
            if not items or items == ["N/A"]:
                elements.append(Paragraph("No specific data available.", self.styles[self.font_to_use]))
            else:
                for item in items:
                    points = item.split('\n')
                    for point in points:
                        cleaned_point = point.strip().lstrip('* ').lstrip('- ').strip()
                        if cleaned_point:
                            elements.append(Paragraph(f"{bullet} {cleaned_point}", self.styles[self.font_to_use]))
            elements.append(spacer)

        elements.append(Paragraph(f"<b>Market Analysis Report: {business_type} in {district_name}, {street_name}</b>", self.styles['Heading1Turkish']))
        elements.append(spacer)

        elements.append(Paragraph("<b>Overall Market Assessment</b>", self.styles['Heading2Turkish']))
        elements.append(Paragraph(analysis_result.get('overall_market_narrative', 'N/A'), self.styles[self.style_to_use]))
        elements.append(spacer)

        add_list_section("Opportunities", analysis_result.get('key_opportunities_for_target_business', []))
        add_list_section("Challenges", analysis_result.get('key_challenges_for_target_business', []))
        add_list_section("Strategic Recommendations", analysis_result.get('strategic_recommendations', []))
        
        elements.append(Paragraph("<b>Do's and Don'ts</b>", self.styles['Heading2Turkish']))
        dos_and_donts = analysis_result.get('dos_and_donts', {})
        add_list_section("Do", dos_and_donts.get('dos', []), bullet="✅")
        if elements and isinstance(elements[-1], Spacer):
            elements.pop()
        add_list_section("Don't", dos_and_donts.get('donts', []), bullet="❌")

        risk_score = analysis_result.get('final_risk_score', 'N/A')
        risk_justification = analysis_result.get('risk_score_justification', 'N/A')
        recommendation_level = analysis_result.get('recommendation_level', 'N/A')

        elements.append(Paragraph("<b>Risk Evaluation</b>", self.styles['Heading2Turkish']))
        elements.append(Paragraph(f"The final calculated risk score is <b>{risk_score}</b>.", self.styles[self.font_to_use]))
        elements.append(Paragraph(f"{risk_justification}", self.styles['JustifyTurkish']))
        elements.append(spacer)

        elements.append(Paragraph("<b>Recommendation Level</b>", self.styles['Heading2Turkish']))
        elements.append(Paragraph(f"Based on the calculated score, the final recommendation is <b>{recommendation_level}</b>.", self.styles[self.font_to_use]))
        elements.append(spacer)

        elements.append(Paragraph("<b>Graded Factors</b>", self.styles['Heading2Turkish']))
        graded_factors = analysis_result.get('graded_factors', {})
        if graded_factors:
            for key, value in graded_factors.items():
                elements.append(Paragraph(f"{key.replace('_', ' ').title()}: {value}", self.styles[self.font_to_use]))
        else:
            elements.append(Paragraph("No graded factors available.", self.styles[self.font_to_use]))
        elements.append(spacer)

        note_style = ParagraphStyle(
            name='NoteStyle',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            textColor='grey',
            fontName='Times-Italic'
        )
        note_text = ("Note on Scoring: The 'Final Risk Score' is calculated using a weighted formula based on the "
                     "'Graded Factors' provided by the AI analysis. The weights are: Competition (40%), Socio-Economic Environment (25%), "
                     "Demand Potential (20%), and Data Quality (15%).")
        elements.append(Paragraph(note_text, note_style))

        return elements
    
    def _create_pdf_report(self, elements: list, business_type: str, district_name: str, street_name: str) -> str:
        try:
            pdf_file_name = f"{business_type.replace(' ', '_')}_{district_name.replace(' ', '_')}_{street_name.replace(' ', '_')}_report.pdf"
            pdf_file_path = os.path.join(self.report_dir, pdf_file_name)
            doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)
            doc.build(elements)
            return pdf_file_path
        except Exception as e:
            self.logger.error(f"Error generating PDF: {e}", exc_info=True)
            raise

    def _summarize_report(self, report_content: str) -> str:
        prompt = f"Please provide a concise summary of the following report (maximum 5 sentences):\n\n{report_content}"
        try:
            summary = self.llm.generate_response(prompt)
            self.total_report_llm_cost += self.llm.calculate_and_log_session_cost()
            return summary
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}", exc_info=True)
            return "Error generating summary."
    
    def reset_session_costs_and_tokens(self):
        self.llm.reset_session_costs_and_tokens()
        self.total_report_llm_cost = 0.0

# --- ORCHESTRATOR CLASS ---
class LLMDrivenBusinessOrchestrator:
    def __init__(self, orchestrator_model_name: str = "gemini-1.5-flash-latest"):
        self.logger = logging.getLogger("LLMOrchestrator")
        self.orchestrator_model_name = orchestrator_model_name
        self.json_path = 'osm_filtered_businesses.json'

        orchestrator_cfg = USER_DEFINED_GENERATION_CONFIG_DICT.copy()
        orchestrator_cfg["response_mime_type"] = "application/json"
        orchestrator_cfg["temperature"] = 0.3

        self.orchestrator_llm = genai.GenerativeModel(
            model_name=self.orchestrator_model_name,
            generation_config=genai.types.GenerationConfig(**orchestrator_cfg),
            safety_settings=DEFAULT_SAFETY_SETTINGS
        )
        self.logger.info(f"Orchestrator LLM ({self.orchestrator_model_name}) hazır.")

        self.business_fetcher = BusinessDataFetcherAgent(categories_data_path='static/data/business_categories.json')
        self.rag_agent = RAGRetrieverAgent()
        self.comprehensive_analyzer = ComprehensiveMarketRiskAgent()
        self.report_generator = ReportGeneratorAgent()
        self.agent_map = {
            "BusinessDataFetcher": self.business_fetcher,
            "SocioEconomicRAGAgent": self.rag_agent,
            "ComprehensiveMarketRiskAgent": self.comprehensive_analyzer,
            "ReportGeneratorAgent": self.report_generator
        }
        self.step_outputs = {}
        self.total_orchestration_llm_cost = 0.0
        self.individual_agent_costs = {}
        # Attributes to track planner LLM cost and tokens
        self.planner_input_tokens = 0
        self.planner_output_tokens = 0
        self.planner_total_tokens = 0
        self.planner_cost = 0.0

    def _generate_plan_with_llm(self, city: str, district: str, business_type: str, street: str) -> dict:
        prompt = ORCHESTRATOR_LLM_PROMPT.format(city=city, district=district, business_type=business_type, street=street)
        try:
            response = self.orchestrator_llm.generate_content(contents=[prompt])
            
            # Capture tokens and cost for the planner LLM call
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                in_toks = response.usage_metadata.prompt_token_count
                out_toks = response.usage_metadata.candidates_token_count
                
                self.planner_input_tokens = in_toks
                self.planner_output_tokens = out_toks
                self.planner_total_tokens = in_toks + out_toks

                cost_in = (in_toks / 1_000_000) * Agent.INPUT_PRICE_PER_MILLION
                cost_out = (out_toks / 1_000_000) * Agent.OUTPUT_PRICE_PER_MILLION
                self.planner_cost = cost_in + cost_out
                # This is used by the existing logging method
                self.total_orchestration_llm_cost = self.planner_cost
            else:
                self.planner_cost = 0.0
                self.total_orchestration_llm_cost = 0.0

            return json.loads(response.text)
        except Exception as e:
            raise ValueError(f"Plan üretimi başarısız: {e}")

    def execute_plan(self, city: str, district: str, street: str, business_type: str, plan: dict, progress_callback=None) -> dict:
        self.logger.info(f"Executing plan with {len(plan.get('steps', []))} steps.")
        self.step_outputs.clear()
        self.individual_agent_costs.clear()
        
        for ag in self.agent_map.values():
            if hasattr(ag, 'reset_session_costs_and_tokens'):
                ag.reset_session_costs_and_tokens()
        
        num_steps = len(plan.get("steps", []))

        for i, step in enumerate(plan.get("steps", [])):
            agent_name = step.get("agent_to_call")
            self.logger.info(f"--- Step {i+1}/{num_steps}: Starting Agent '{agent_name}' ---")

            if progress_callback:
                progress_percent = int(((i + 0.5) / num_steps) * 100)
                progress_callback(f"Step {i+1}/{num_steps}: Running {agent_name}...", progress_percent)
            
            agent_instance = self.agent_map.get(agent_name)
            if agent_name == "ScoreCalculator":
                agent_instance = calculate_final_risk_score

            if not agent_instance:
                error_msg = f"Agent '{agent_name}' not found in configuration."
                self.step_outputs[i] = {"agent_name": agent_name, "status": "failure", "error": error_msg}
                self.logger.error(f"Step {i+1}: {error_msg}")
                continue

            resolved_inputs = {}
            failed = False
            for key, val in step.get("inputs_for_agent", {}).items():
                if isinstance(val, str) and val.startswith("OUTPUT_OF_STEP_"):
                    parts = val.split('.')
                    idx_src = int(parts[0].split("_")[-1])
                    src_out = self.step_outputs.get(idx_src)
                    if src_out and src_out.get("status") == "success":
                        data = src_out.get("data")
                        if len(parts) > 1:
                            nested_key = parts[1]
                            resolved_inputs[key] = data.get(nested_key, {})
                        else:
                            resolved_inputs[key] = data
                    else:
                        failed = True
                        break
                else:
                    resolved_inputs[key] = val
            
            if failed:
                self.step_outputs[i] = {"agent_name": agent_name, "status": "failure", "error": "Input resolution failed due to dependency failure."}
                continue

            try:
                self.logger.info(f"Calling agent '{agent_name}' with inputs: {json.dumps(resolved_inputs, indent=2, default=str)}")
                
                output_data = {}
                if agent_name == "BusinessDataFetcher":
                    output_data = agent_instance.fetch_business_data(**resolved_inputs)
                elif agent_name == "SocioEconomicRAGAgent":
                    output_data = agent_instance.fetch_rag_data(**resolved_inputs)
                elif agent_name == "ComprehensiveMarketRiskAgent":
                    output_data = agent_instance.analyze_and_score_risk(**resolved_inputs)
                elif agent_name == "ScoreCalculator":
                    graded_factors = resolved_inputs.get("graded_factors", {})
                    final_risk_score = agent_instance(graded_factors)
                    if final_risk_score <= 20: recommendation_level = "Very Low Risk"
                    elif final_risk_score <= 40: recommendation_level = "Low Risk"
                    elif final_risk_score <= 60: recommendation_level = "Medium Risk"
                    elif final_risk_score <= 80: recommendation_level = "High Risk"
                    else: recommendation_level = "Very High Risk"
                    output_data = {"final_risk_score": final_risk_score, "recommendation_level": recommendation_level}
                elif agent_name == "ReportGeneratorAgent":
                    analysis_result = resolved_inputs.pop("analysis_result", {})
                    scoring_result = resolved_inputs.pop("scoring_result", {})
                    
                    # --- GATHER COST/TOKEN DATA FOR REPORT GENERATOR ---
                    planner_data = {
                        'input': self.planner_input_tokens, 'output': self.planner_output_tokens, 'cost': self.planner_cost
                    }
                    rag_agent = self.agent_map["SocioEconomicRAGAgent"]
                    rag_data = {
                        'input': rag_agent.synthesis_llm.session_input_tokens, 'output': rag_agent.synthesis_llm.session_output_tokens, 'cost': rag_agent.total_synthesis_llm_cost
                    }
                    comp_agent = self.agent_map["ComprehensiveMarketRiskAgent"]
                    comp_cost_in = (comp_agent.session_input_tokens / 1_000_000) * Agent.INPUT_PRICE_PER_MILLION
                    comp_cost_out = (comp_agent.session_output_tokens / 1_000_000) * Agent.OUTPUT_PRICE_PER_MILLION
                    comp_data = {
                        'input': comp_agent.session_input_tokens, 'output': comp_agent.session_output_tokens, 'cost': comp_cost_in + comp_cost_out
                    }
                    cost_token_data = {
                        "planner": planner_data, "rag_synthesis": rag_data, "comprehensive_analysis": comp_data
                    }
                    # --- END OF GATHERING ---

                    if analysis_result and scoring_result and not analysis_result.get("error"):
                        full_report_data = {**analysis_result, **scoring_result}
                        resolved_inputs['analysis_result'] = full_report_data
                        resolved_inputs['cost_token_data'] = cost_token_data # Pass data to agent
                        
                        report_filename, summary, pdf_path = agent_instance.generate_report(**resolved_inputs)
                        output_data = {"report_filename": report_filename, "report_summary": summary, "pdf_file_path": pdf_path}
                    else:
                        output_data = {"error": "Analysis or scoring step failed, cannot generate report."}

                status = "success" if not output_data.get("error") else "failure"
                self.step_outputs[i] = {"agent_name": agent_name, "status": status, "data": output_data}
                self.logger.info(f"Step {i+1} ({agent_name}) completed with status: {status}.")

            except Exception as e:
                self.step_outputs[i] = {"agent_name": agent_name, "status": "failure", "error": str(e)}
                self.logger.error(f"Step {i+1} ({agent_name}) encountered a critical error: {e}", exc_info=True)

        return {"initial_assessment": plan.get("initial_assessment"), "step_outputs": self.step_outputs}

    def analyze_business_viability(self, city: str, district: str, street: str, business_type: str, progress_callback=None) -> dict:
        self.logger.info(f"Analyzing viability: {business_type} in {street}, {district}, {city}")
        
        # Reset planner-specific costs and tokens for a clean run
        self.planner_input_tokens = 0
        self.planner_output_tokens = 0
        self.planner_total_tokens = 0
        self.planner_cost = 0.0

        try:
            if progress_callback: progress_callback("Generating analysis plan...", 5)
            plan_dict = self._generate_plan_with_llm(city, district, street, business_type)
            self.logger.info(f"Generated plan: {json.dumps(plan_dict, indent=2)}")
            
            if progress_callback: progress_callback("Executing analysis plan...", 10)
            return self.execute_plan(city, district, street, business_type, plan_dict, progress_callback=progress_callback)
        except Exception as e:
            self.logger.error(f"Analysis failed during plan generation or execution: {e}", exc_info=True)
            if progress_callback: progress_callback(f"Analysis failed: {e}", 100)
            return {"error": f"Analysis failed: {e}"}

    def log_total_orchestration_cost(self):
        """Calculates and logs the total cost of the entire orchestration run."""
        self.logger.info("--- Orchestration Cost Summary ---")
        
        # Cost from the orchestrator LLM (plan generation)
        planner_cost = self.total_orchestration_llm_cost
        self.logger.info(f"Orchestrator LLM (Plan Generation) Cost: {planner_cost:.6f} USD")
        
        total_cost = planner_cost
        
        # Costs from individual agents are calculated after their runs
        # and retrieved from their instances.
        for agent_name, agent_instance in self.agent_map.items():
            if agent_name == "SocioEconomicRAGAgent":
                agent_cost = agent_instance.total_synthesis_llm_cost
                total_cost += agent_cost
            elif agent_name == "ComprehensiveMarketRiskAgent":
                # This agent's cost is logged internally by the method
                agent_cost = agent_instance.calculate_and_log_session_cost()
                total_cost += agent_cost
            elif agent_name == "ReportGeneratorAgent":
                agent_cost = agent_instance.total_report_llm_cost
                total_cost += agent_cost
                
        self.logger.info(f"--- TOTAL ORCHESTRATION COST: {total_cost:.6f} USD ---")
        return total_cost