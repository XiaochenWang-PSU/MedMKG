
# MedMKG: Medical Multimodal Knowledge Graph

We introduce **MedMKG**, a **Med**ical **M**ultimodal **K**nowledge **G**raph that seamlessly fuses clinical concepts with medical images.  
MedMKG is constructed via a multi-stage pipeline that accurately identifies and disambiguates medical concepts while extracting their interrelations.  
To ensure the conciseness of the resulting graph, we further employ a pruning strategy based on our novel Neighbor-aware Filtering (NaF) algorithm.

---

## 📂 Provided Files

This repository contains:
- `knowledge_graph.csv` — biomedical triplets: Head, Relation, Tail, Head_Name, Tail_Name
- `image_mapping.csv` — image ID to **relative path** mappings

**Note:** The images themselves are **not included**. Users must download MIMIC-CXR-JPG separately and specify their local path.


## 📦 About MIMIC-CXR-JPG

**MIMIC-CXR-JPG** is a large publicly available dataset of chest radiographs in JPEG format, sourced from the Beth Israel Deaconess Medical Center in Boston.

- **URL:** [https://physionet.org/content/mimic-cxr-jpg/2.1.0/](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- **Total uncompressed size:** 570.3 GB

### Access Instructions

To use the image data, you **must** request access and agree to the data use agreement, which includes:
1. You will **not share the data**.
2. You will **not attempt to reidentify individuals**.
3. Any publication using the data will **make the relevant code available**.

**Download options:**
- [ZIP download](https://physionet.org/files/mimic-cxr-jpg/2.1.0/)
- Google BigQuery access
- Google Cloud Storage Browser access
- Command-line download:
  
  ```bash
  wget -r -N -c -np --user your_username --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/



## 🔧 Usage Example

Below is a demo script to load and link the knowledge graph with your local image data:

```python
from huggingface_hub import hf_hub_download
import pandas as pd

# Add repo_type="dataset" to avoid 404
kg_path = hf_hub_download(repo_id="xcwangpsu/MedMKG", filename="MedMKG.csv", repo_type="dataset")
mapping_path = hf_hub_download(repo_id="xcwangpsu/MedMKG", filename="image_mapping.csv", repo_type="dataset")

# Load CSVs
kg_df = pd.read_csv(kg_path)
mapping_df = pd.read_csv(mapping_path)

# Local path to downloaded MIMIC-CXR images
local_root = "/path/to/your/mimic-cxr-jpg"

# Map image IDs to full paths
iid_to_path = {
    row["IID"]: f"{local_root}/{row['Image_Path']}"
    for _, row in mapping_df.iterrows()
}

# Merge image path info into KG
kg_df["Head_Path"] = kg_df["Head"].map(iid_to_path)
kg_df["Tail_Path"] = kg_df["Tail"].map(iid_to_path)

print(kg_df.head())


```

## Benchmark Instruction

We cover codes for three downstream tasks detailed in our paper, including link prediction, knowledge-augmented visual question answering, and knowledge-augmented text-image retrieval. You may check the three folders and execute the following:

```bash
python main.py
```

after specifying your local paths of data files.
