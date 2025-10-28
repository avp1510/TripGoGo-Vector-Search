import pandas as pd
import oracledb
from sentence_transformers import SentenceTransformer

# ✅ Load local free embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dim

# ✅ Load your CSV
df = pd.read_csv("/Users/abhishek/Documents/TripGoGo/combined.csv")

# ✅ Oracle thin-mode connection (no wallet, no instant client needed)
oracledb.enable_thin_mode()
conn = oracledb.connect(
    user="ADMIN",
    password="TripGoGo@2025",
    dsn="tcps://adb.us-ashburn-1.oraclecloud.com:1522/gf05fe3213cc143_tripgogo_high.adb.oraclecloud.com"
)
cursor = conn.cursor()

# ✅ Insert with embeddings
for _, row in df.iterrows():
    text = f"{row['addr_text']} {row['city']}"
    vec = "[" + ",".join(map(str, model.encode(text, normalize_embeddings=True))) + "]"

 # 384 dim

    cursor.execute("""
        INSERT INTO hotels (name, addr_text, city, lat, lon, price_usd, rating, url, addr_vec)
        VALUES (:1, :2, :3, :4, :5, :6, :7, :8, TO_VECTOR(:9))
    """, (
        row["name"], row["addr_text"], row["city"],
        row["lat"], row["lon"], row["price_usd"], row["rating"], row["url"],
        vec
    ))

conn.commit()
print("✅ DONE — rows inserted with MiniLM embeddings")
