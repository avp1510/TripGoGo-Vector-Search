import streamlit as st
import pandas as pd
import oracledb
from sentence_transformers import SentenceTransformer
import warnings
import json
import time
import requests # Used for Python API interaction

# Suppress the specific future warning from SentenceTransformer
warnings.filterwarnings("ignore", message="The `device` argument is deprecated", category=FutureWarning)

# --- Configuration (Must match ingest_hotels.py) ---
ORACLE_USER = "ADMIN"
ORACLE_PASSWORD = "TripGoGo@2025"
ORACLE_DSN = "tcps://adb.us-ashburn-1.oraclecloud.com:1522/gf05fe3213cc143_tripgogo_high.adb.oraclecloud.com"

# --- LLM API Configuration ---
# NOTE: The apiKey will be injected by the environment, so we keep it blank.
API_KEY = "AIzaSyBwE9bnkNtyp-A9cjRdHYFqTVajMbzyvpg"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# --- Utility Functions (Cached for Performance) ---

@st.cache_resource
def load_model():
    """Load the SentenceTransformer model and cache it."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_oracle_connection():
    """Establish and return an Oracle database connection."""
    try:
        oracledb.enable_thin_mode()
        conn = oracledb.connect(
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            dsn=ORACLE_DSN
        )
        st.sidebar.success("✅ Oracle DB Connected")
        return conn
    except Exception as e:
        st.sidebar.error(f"❌ Database Connection FAILED: {e}")
        return None

def perform_similarity_search(query_text: str, model: SentenceTransformer, conn: oracledb.Connection, top_k: int) -> pd.DataFrame:
    """Encodes the query and performs a vector similarity search."""
    
    if model is None:
        st.error("Model not loaded. Cannot perform search.")
        return pd.DataFrame()

    query_vec = model.encode(query_text, normalize_embeddings=True)
    vec_str = "[" + ",".join(map(str, query_vec)) + "]"

    sql_query = f"""
        SELECT
            name,
            addr_text,
            city,
            price_usd,
            rating,
            url,
            -- Use COSINE_DISTANCE shorthand function
            COSINE_DISTANCE(
                addr_vec, 
                TO_VECTOR(:query_vec, 384)
            ) as distance_score
        FROM
            hotels
        ORDER BY
            distance_score ASC 
        FETCH FIRST {top_k} ROWS ONLY
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query, {"query_vec": vec_str})
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        # FIX: Check for and convert any LOB objects (like CLOB data) to strings 
        # before creating the DataFrame, which caused the Pyarrow/ArrowInvalid error.
        processed_results = []
        for row in results:
            processed_row = []
            for item in row:
                if isinstance(item, oracledb.LOB):
                    # Convert LOB to string. Read the LOB content and decode to string.
                    try:
                        item_content = item.read()
                        if item_content is not None:
                            # Assuming the data is UTF-8 encoded text
                            processed_row.append(item_content.decode('utf-8'))
                        else:
                            processed_row.append(None)
                    except Exception:
                        processed_row.append(None) # Handle read/decode errors gracefully
                else:
                    processed_row.append(item)
            processed_results.append(processed_row)

        return pd.DataFrame(processed_results, columns=column_names)
        
    except Exception as e:
        st.error(f"❌ Database Search Error: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()

def generate_summary(user_query: str, results_markdown: str) -> str:
    """
    Calls the Gemini LLM to generate a conversational summary of the search results.
    This demonstrates the RAG Generation step.
    """
    
    # 1. Define the RAG Prompt and System Instructions
    system_prompt = (
        "You are 'TripGoGo Concierge,' a friendly and knowledgeable AI travel agent. "
        "Your goal is to provide a concise, engaging, and personalized summary "
        "Alo use proper spacing and same font with proper punctuations "
        "Do not include any links or markdown tables in your final output. "
        "Highlight the one best match and briefly explain why it fits the user's request."
    )

    user_prompt = (
        f"The user searched for: '{user_query}'.\n\n"
        f"Here are the top results retrieved from the database:\n\n"
        f"--- HOTEL DATA ---\n{results_markdown}\n\n"
        f"--- YOUR TASK ---\n"
        f"Based on this data, write a conversational, one-paragraph summary for the user. "
        f"Mention the best hotel and its key features (Match Score, Rating, Price)."
    )

    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # 2. Execute API call with Exponential Backoff
    
    headers = {'Content-Type': 'application/json'}
    # The API key is added to the URL query string
    url = f"{API_URL}?key={API_KEY}" 

    retries = 0
    max_retries = 4
    
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() 
            result = response.json()
            
            # Safely extract text
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Could not generate summary.')
            return text
            
        except requests.exceptions.HTTPError as http_err:
            retries += 1
            error_info = f"LLM HTTP Error {response.status_code}. Response: {response.text[:200]}..."
            # Log the detailed error to the console for debugging
            print(f"LLM API failed (Attempt {retries}): {error_info}")
            
            if retries == max_retries:
                # Provide a more specific error message in the UI
                return "I apologize, but I couldn't generate a personalized summary due to an API error. Check the application console for details on the API status code (403 often means missing API Key)."
            
            # Exponential Backoff
            wait_time = 2 ** retries 
            time.sleep(wait_time) 

        except requests.exceptions.RequestException as req_err:
            retries += 1
            # Log the network/connection error to the console
            print(f"LLM Network Error (Attempt {retries}): {req_err}")
            
            if retries == max_retries:
                return "I apologize, but I couldn't generate a personalized summary due to a network connection error after multiple retries."
            
            # Exponential Backoff
            wait_time = 2 ** retries 
            time.sleep(wait_time) 
            
        except Exception:
            return "An internal error occurred while processing the LLM response."
            
    return "Summary generation failed."

# --- Streamlit UI ---

st.set_page_config(
    page_title="TripGoGo Vector Search",
    layout="wide"
)

st.title("📍 TripGoGo Hotel Recommendation Engine")
st.markdown("Find hotels using descriptive queries with **Oracle Vector Search**.")
st.markdown("---")

# Load model and establish connection
model = load_model()
conn = get_oracle_connection()

# Input and Search Form
with st.form(key='search_form'):
    query = st.text_input(
        "**Where do you want to go?** (e.g., 'A quiet area near the beach')",
        placeholder="Enter a location, landmark, or description..."
    )
    top_k_results = st.slider("Number of results to show:", min_value=1, max_value=10, value=3)
    search_button = st.form_submit_button("Search Hotels 🔍")

# Perform search on button click
if search_button and query:
    if conn:
        with st.spinner(f"Processing query '{query}' and generating recommendations..."):
            results_df = perform_similarity_search(query, model, conn, top_k_results)
        
        if not results_df.empty:
            
            # 1. Format the DataFrame for display and RAG input
            results_df['Match Score'] = ((1 - results_df['DISTANCE_SCORE']) * 100).round(2)
            results_df['Display Score'] = results_df['Match Score'].astype(str) + '%'
            
            results_df.rename(columns={
                'NAME': 'Hotel Name',
                'ADDR_TEXT': 'Address',
                'CITY': 'City',
                'PRICE_USD': 'Price (USD)',
                'RATING': 'Rating',
                'URL': 'Booking Link',
            }, inplace=True)
            
            # Select columns needed for the LLM prompt (RAG context)
            rag_df = results_df[['Hotel Name', 'Match Score', 'Rating', 'Price (USD)', 'Address']].sort_values(by='Match Score', ascending=False)
            
            # Generate the LLM Summary
            with st.spinner("Generating personalized summary..."):
                llm_summary = generate_summary(query, rag_df.to_markdown(index=False))
            
            # 2. Display the LLM Summary
            st.markdown("## ✨ Personalized Concierge Recommendation")
            st.info(llm_summary) # Use st.info for a visually pleasing output
            
            # 3. Display the List of Found Hotels (NEW SECTION)
            st.markdown("---") 
            st.markdown("## 🏨 Hotels Retrieved in Search")
            
            hotel_names = results_df['Hotel Name'].tolist()
            hotel_list_markdown = "\n".join([f"* **{name}**" for name in hotel_names])
            
            st.markdown(f"The following **{len(hotel_names)}** properties were analyzed for your recommendation:\n\n{hotel_list_markdown}")
            
            # 4. Display the Raw Data Table
            st.markdown(f"## ✅ Top {len(results_df)} Recommended Hotels (Raw Data)")
            
            display_cols = ['Hotel Name', 'Display Score', 'Rating', 'Price (USD)', 'Address', 'City', 'Booking Link']
            
            st.dataframe(results_df[display_cols], 
                column_config={
                    "Booking Link": st.column_config.LinkColumn("Booking Link", display_text="Book Now"),
                    "Rating": st.column_config.ProgressColumn("Rating", format="%.1f", min_value=0.0, max_value=5.0)
                },
                hide_index=True,
                width='stretch' # FIX: Replaced deprecated use_container_width=True
            )
            
        else:
            st.warning("No results found or an error occurred during search.")
    else:
        st.error("Cannot perform search. Database connection is not active.")
