import gradio as gr
from opennyai import Pipeline
from opennyai.utils import Data
import spacy
from spacy import displacy
import re
from markupsafe import Markup
from opennyai.ner.ner_utils import ner_displacy_option

# Load SpaCy transformer model
nlp = spacy.load("en_core_web_trf")

# Initialize OpenNyai pipeline
pipeline = Pipeline(
    components=["NER", "Summarizer"],
    use_gpu=False,
    ner_do_sentence_level=True,
    ner_do_postprocess=True,
    ner_mini_batch_size=128,
    verbose=True
)

# Clean the summary sections
def clean_summary(summary_dict):
    cleaned = {}
    for section, text in summary_dict.items():
        text = re.sub(r'\s+\.', '.', text)
        text = text.replace('\n', '')
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned[section] = text
    return cleaned

# Format cleaned summary into readable HTML
def format_summary(cleaned_summary):
    output = ""
    for section, text in cleaned_summary.items():
        output += f"<h4>{section}</h4><p>{text}</p>"
    return Markup(f"<div style='max-height: 400px; overflow-y: auto;'>{output}</div>")

# Render NER using displacy with scrollable box
def visualize_ner(ner_results):
    doc = pipeline._ner_model_output[0]
    
    # Custom readable entity style
    ner_colors = {
        "CASE_NUMBER": "#f3e79b",
        "DATE": "#f5b971",
        "COURT": "#b5ead7",
        "OTHER_PERSON": "#ffdac1",
        "JUDGE": "#f8c291",
        "PROVISION": "#c7ceea",
        "STATUTE": "#e0bbE4",
        "ORG": "#bde0fe",
        "GPE": "#ffb4a2"
    }

    custom_ner_options = {
        "colors": ner_colors,
        "bg": "#ffffff",  # force white background
        "font": "monospace"
    }

    html = displacy.render(doc, style="ent", options=custom_ner_options, jupyter=False)

    # Fix hardcoded white text color
    html = html.replace("color: white;", "color: black;")

    return Markup(f"<div style='max-height: 400px; overflow-y: auto;'>{html}</div>")


# Gradio processing function
def process_text(text):
    if isinstance(text, list):
        text = text[0]

    input_data = Data(text)
    results = pipeline(input_data)

    if results and len(results) > 0:
        result = results[0]
        cleaned_summary = clean_summary(result.get("summary", {}))
        summary_html = format_summary(cleaned_summary)
        ner_html = visualize_ner(result.get("ner", {}))
        return ner_html, summary_html

    return "No named entities found.", "No summary generated."

# Launch Gradio app
demo = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(lines=10, label="Enter Legal Text"),
    outputs=[
        gr.HTML(label="Named Entities (Highlighted, Scrollable)"),
        gr.HTML(label="Cleaned Summary by Section (Scrollable)")
    ],
    title="Legal NLP",
    description="Paste a legal document to extract Named Entities and view a clean summary. Both outputs are scrollable for convenience."
)

demo.launch(share=True)
