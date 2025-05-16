
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
import pandas as pd

# User-provided local path to mimic-cxr-jpg root
local_root = "/path/to/your/mimic-cxr-jpg"

# Load datasets
kg_df = pd.read_csv("knowledge_graph.csv")
mapping_df = pd.read_csv("image_mapping.csv")

# Build image ID → full local path mapping
iid_to_path = {
    row["IID"]: f"{local_root}/{row['Image_Path']}"
    for _, row in mapping_df.iterrows()
}

# Add image paths to knowledge graph (if Head or Tail is an image)
kg_df["Head_Path"] = kg_df["Head"].map(iid_to_path)
kg_df["Tail_Path"] = kg_df["Tail"].map(iid_to_path)

# Example display
print("Sample with resolved image paths:")
print(kg_df.head())

# Filter rows with image entities
image_rows = kg_df[
    kg_df["Head"].str.startswith("I") | kg_df["Tail"].str.startswith("I")
]
print("\nRows with image entities:")
print(image_rows.head())

```

## Benchmark Instruction

We cover codes for three downstream tasks detailed in our paper, including link prediction, knowledge-augmented visual question answering, and knowledge-augmented text-image retrieval. You may check the three folders and execute the following:

```bash
python main.py
```

after specifying your local paths of data files.
