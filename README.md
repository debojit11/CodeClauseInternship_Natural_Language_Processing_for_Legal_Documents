# Legal Document NLP Tool (Internship Project)

This project is a Natural Language Processing (NLP) tool developed during my internship. It is designed to process Indian legal documents and extract meaningful information. The tool performs Named Entity Recognition (NER) and Summarization using models from the [OpenNyai Project](https://github.com/OpenNyAI/Opennyai).

---

## 🔧 Installation

### Step 1: Create a Conda environment

```bash
conda create -n opennyai python=3.8
conda activate opennyai
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Some dependencies may require version fixes. If you face errors (e.g., typing-extensions, fastapi), please resolve them manually using pip uninstall/install.

### Step 3: Download the model manually

Download the legal NER model manually from Hugging Face:

**Download link:**  
https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl

Rename the file to something like:

```text
en_legal_ner_trf-0.0.1-py3-none-any.whl
```

Then install it using:

```bash
pip install en_legal_ner_trf-0.0.1-py3-none-any.whl
```

---

## 🚀 Usage

### Step 1: Import required modules

```python
from opennyai import Pipeline
from opennyai.utils import Data
```

### Step 2: Initialize the pipeline

```python
pipeline = Pipeline(
    components=['NER', 'Summarizer'],
    use_gpu=False,
    ner_do_sentence_level=True,
    ner_do_postprocess=True,
    ner_mini_batch_size=128,
    verbose=True
)
```

### Step 3: Pass input text

```python
data = Data("Paste your legal text here")
```

### Step 4: Run the pipeline

```python
results = pipeline(data)
```

### Step 5: Use the outputs

```python
# Get summary
summary = results[0]['summary']

# Get named entities
ner_doc_1 = pipeline._ner_model_output[0]
identified_entites = [(ent, ent.label_) for ent in ner_doc_1.ents]

```

You can refer to the notebook in the repository for detailed usage and output formatting.

---

## ▶️ Running the Gradio App

To run the web demo:

```bash
python app.py
```

Then open the URL shown in your terminal (usually http://127.0.0.1:7860).

---

## 📌 Credits

This project uses models and components from the [OpenNyai Project](https://github.com/OpenNyAI/Opennyai).  
Huge thanks to their team for open-sourcing Indian legal NLP resources.

---

## ⚠️ Disclaimer

This project was developed during an internship. It is intended for **personal portfolio** purposes only.  
All rights reserved. Please **do not reuse or redistribute** without permission.
