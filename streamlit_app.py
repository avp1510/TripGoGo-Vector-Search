import streamlit as st
import pandas as pd
import oracledb
from sentence_transformers import SentenceTransformer
import warnings

# Suppress the specific future warning from SentenceTransformer
warnings.filterwarnings("ignore", message="The `device` argument is deprecated", category=FutureWarning)

# --- Configuration (Must match ingest_hotels.py) ---
ORACLE_USER = "ADMIN"
ORACLE_PASSWORD = "TripGoGo@2025"
ORACLE_DSN = "tcps://adb.us-ashburn-1.oraclecloud.com:1522/gf05fe3213cc143_tripgogo_high.adb.oraclecloud.com"

# --- Utility Functions (Cached for Performance) ---

@st.cache_resource
def load_model():
    """Load the SentenceTransformer model and cache it."""
    # MUST match the model used for insertion
    return SentenceTransformer("all-MiniLM-L6-v2")

# NOTE: Removed @st.cache_resource temporarily for easier debugging of initial connection
# If this works, you can add @st.cache_resource back for better performance.
def get_oracle_connection():
    """Establish and return an Oracle database connection."""
    try:
        oracledb.enable_thin_mode()
        conn = oracledb.connect(
            user=ORACLE_USER,
            password=ORACLE_PASSWORD,
            dsn=ORACLE_DSN
        )
        # Display connection status in the sidebar
        st.sidebar.success("✅ Oracle DB Connected")
        return conn
    except Exception as e:
        # Display error in the sidebar without halting the entire app
        st.sidebar.error(f"❌ Database Connection FAILED: {e}")
        return None

def perform_similarity_search(query_text: str, model: SentenceTransformer, conn: oracledb.Connection, top_k: int) -> pd.DataFrame:
    """Encodes the query and performs a vector similarity search."""
    
    if model is None:
        st.error("Model not loaded. Cannot perform search.")
        return pd.DataFrame()

    # 1. Convert the input query to a vector (embedding)
    query_vec = model.encode(query_text, normalize_embeddings=True)
    # Vector is formatted as the string '[x1, x2, ..., x384]'
    vec_str = "[" + ",".join(map(str, query_vec)) + "]"

    # 2. Define the SQL query for Vector Search
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
            -- COSINE_DISTANCE returns distance, so ASC (lower value) is better match
            distance_score ASC 
        FETCH FIRST {top_k} ROWS ONLY
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query, {"query_vec": vec_str})
        results = cursor.fetchall()
        # Fetch column names, which will be in UPPERCASE from Oracle
        column_names = [desc[0] for desc in cursor.description]
        return pd.DataFrame(results, columns=column_names)
        
    except Exception as e:
        st.error(f"❌ Database Search Error: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()

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
        with st.spinner(f"Processing query '{query}'..."):
            results_df = perform_similarity_search(query, model, conn, top_k_results)
        
        st.markdown(f"## ✅ Top {len(results_df)} Recommended Hotels")
        
        if not results_df.empty:
            
            # CRITICAL FIX: Access the column as 'DISTANCE_SCORE' (uppercase from Oracle)
            # and convert distance (0=perfect match) to similarity percentage (100%=perfect match)
            results_df['Match Score'] = ((1 - results_df['DISTANCE_SCORE']) * 100).round(2).astype(str) + '%'
            
            # Formatting for display
            results_df.rename(columns={
                'NAME': 'Hotel Name',
                'ADDR_TEXT': 'Address',
                'CITY': 'City',
                'PRICE_USD': 'Price (USD)',
                'RATING': 'Rating',
                'URL': 'Booking Link',
                # We don't need to rename the original distance column, as we won't display it
                # We also removed the incorrect 'similarity_score': 'Match Score' line
            }, inplace=True)
            
            # Ensure 'Match Score' (our calculated column) is used
            display_cols = ['Hotel Name', 'Match Score', 'Rating', 'Price (USD)', 'Address', 'City', 'Booking Link']
            
            st.dataframe(results_df[display_cols], 
                column_config={
                    "Booking Link": st.column_config.LinkColumn("Booking Link", display_text="Book Now"),
                    "Rating": st.column_config.ProgressColumn("Rating", format="%.1f", min_value=0.0, max_value=5.0)
                },
                hide_index=True,
                use_container_width=True
            )
            
        else:
            st.warning("No results found or an error occurred during search.")
    else:
        st.error("Cannot perform search. Database connection is not active.")
