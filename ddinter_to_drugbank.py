import os
import re
import time
import csv
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"C:\Users\minwo\Desktop\Dataset"
INPUT_FILE = os.path.join(DATA_DIR, "ddinter_drug_list.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "ddinter_to_drugbank_map.csv")

# ------------------------------------------------------------
# LOAD DDINTER DRUG LIST
# ------------------------------------------------------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Missing {INPUT_FILE}. Run the drug list extraction first.")

df = pd.read_csv(INPUT_FILE, sep=None, engine="python")
df.columns = [col.strip() for col in df.columns]
print(f"📂 Loaded {len(df)} DDInter entries")
print("Columns loaded:", df.columns.tolist())

# Resume support: load existing mappings if any
existing = {}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row["DDInterID"]] = row["DrugBankID"]
    print(f"🔁 Resuming from existing {len(existing)} mappings")

results = []
session = requests.Session()

# ------------------------------------------------------------
# SCRAPE LOOP
# ------------------------------------------------------------
for i, row in tqdm(df.iterrows(), total=len(df), desc="Mapping DDInter → DrugBank"):
    ddinter_id = row["DDInterID"]
    drug_name = row["DrugName"]

    if ddinter_id in existing:
        results.append({"DDInterID": ddinter_id, "DrugName": drug_name, "DrugBankID": existing[ddinter_id]})
        continue

    url = f"https://ddinter.scbdd.com/ddinter/drug-detail/{ddinter_id}/"
    drugbank_id = None

    try:
        r = session.get(url, timeout=10, verify=False)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            # find link to drugbank
            links = [a['href'] for a in soup.find_all('a', href=True) if "drugbank.com/drugs/" in a['href']]
            if links:
                match = re.search(r"(DB\d+)", links[0])
                if match:
                    drugbank_id = match.group(1)
        else:
            print(f"⚠️ HTTP {r.status_code} for {ddinter_id}")

    except Exception as e:
        print(f"⚠️ Error fetching {ddinter_id}: {e}")

    if drugbank_id:
        print(f"[{i+1}/{len(df)}] {drug_name} → {drugbank_id}")
    else:
        print(f"[{i+1}/{len(df)}] {drug_name} → ❌ Not found")

    results.append({"DDInterID": ddinter_id, "DrugName": drug_name, "DrugBankID": drugbank_id or "NotFound"})

    # Save incrementally to prevent data loss
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    time.sleep(0.3)  # gentle delay to avoid hammering server

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
found = sum(1 for r in results if r["DrugBankID"] != "NotFound")
total = len(results)
pct = (found / total) * 100
print(f"\n✅ Completed mapping {found}/{total} ({pct:.1f}%)")
print(f"📄 Exported to {OUTPUT_FILE}")