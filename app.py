import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer

from model import SciBertMultiLabelConfig, SciBertForMultiLabelClassification


MODEL_DIR = "./best_arxiv_ultra_scibert"
MAX_LENGTH = 512
GLOBAL_THRESHOLD = 0.60
ID2LABEL_FILE = os.path.join(MODEL_DIR, "id2label.json")
ARXIV_CATEGORY_MAPPER = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "econ.TH": "Theoretical Economics",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
    "math.AC": "Commutative Algebra",
    "math.AG": "Algebraic Geometry",
    "math.AP": "Analysis of PDEs",
    "math.AT": "Algebraic Topology",
    "math.CA": "Classical Analysis and ODEs",
    "math.CO": "Combinatorics",
    "math.CT": "Category Theory",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GR": "Group Theory",
    "math.GT": "Geometric Topology",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory and Homology",
    "math.LO": "Logic",
    "math.MG": "Metric Geometry",
    "math.MP": "Mathematical Physics",
    "math.NA": "Numerical Analysis",
    "math.NT": "Number Theory",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RA": "Rings and Algebras",
    "math.RT": "Representation Theory",
    "math.SG": "Symplectic Geometry",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Earth and Planetary Astrophysics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Solar and Stellar Astrophysics",
    "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.other": "Other Condensed Matter",
    "cond-mat.quant-gas": "Quantum Gases",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.supr-con": "Superconductivity",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nlin.AO": "Adaptation and Self-Organizing Systems",
    "nlin.CD": "Chaotic Dynamics",
    "nlin.CG": "Cellular Automata and Lattice Gases",
    "nlin.PS": "Pattern Formation and Solitons",
    "nlin.SI": "Exactly Solvable and Integrable Systems",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics.acc-ph": "Accelerator Physics",
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    "physics.app-ph": "Applied Physics",
    "physics.atm-clus": "Atomic and Molecular Clusters",
    "physics.atom-ph": "Atomic Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis, Statistics and Probability",
    "physics.ed-ph": "Physics Education",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History and Philosophy of Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.soc-ph": "Physics and Society",
    "physics.space-ph": "Space Physics",
    "quant-ph": "Quantum Physics",
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs",
    "q-fin.CP": "Computational Finance",
    "q-fin.EC": "Economics",
    "q-fin.GN": "General Finance",
    "q-fin.MF": "Mathematical Finance",
    "q-fin.PM": "Portfolio Management",
    "q-fin.PR": "Pricing of Securities",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.TR": "Trading and Market Microstructure",
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Statistics Theory"
}


st.set_page_config(
    page_title="Article Classifier",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Классификатор научных статей")
st.markdown(
    """
    Этот сервис решает задачу **multilabel-классификации** научных статей:
    по `title` и `abstract` модель прогнозирует несколько подходящих категорий arXiv

    **Что важно знать:**
    - Используется **148 классов** - это категории из официальной таксономии arXiv
    - Одна статья может относиться сразу к нескольким областям
    - Если `abstract` не указан, классификация выполняется только по `title`
    """
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def format_label(label_code: str) -> str:
    label_name = ARXIV_CATEGORY_MAPPER.get(label_code)
    if label_name:
        return f"{label_code} - {label_name}"
    return label_code


def load_id2label() -> dict:
    with open(ID2LABEL_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {int(k): v for k, v in raw.items()}


@st.cache_resource
def load_artifacts():
    model_dir = Path(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    config = SciBertMultiLabelConfig.from_pretrained(model_dir)
    model = SciBertForMultiLabelClassification.from_pretrained(model_dir, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    id2label = load_id2label()

    return tokenizer, model, device, id2label


def prepare_inputs(title, abstract):
    title = (title or "").strip()
    abstract = (abstract or "").strip()

    if not title and not abstract:
        raise ValueError("Нужно ввести хотя бы title или abstract")

    return title, abstract


def decode_predictions(probs, id2label, global_threshold, top_k=10):
    probs = np.asarray(probs, dtype=np.float32)

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_predictions = [
        {
            "label": format_label(id2label[int(i)]),
            "score": float(probs[i]),
        }
        for i in top_indices
    ]

    pred_indices = np.where(probs >= global_threshold)[0]
    pred_indices = pred_indices[np.argsort(probs[pred_indices])[::-1]]

    predicted_labels = [
        {
            "label": format_label(id2label[int(i)]),
            "score": float(probs[i]),
            "threshold": float(global_threshold),
        }
        for i in pred_indices
    ]

    return top_predictions, predicted_labels


def predict(title: str, abstract: str, top_k: int = 10):
    title, abstract = prepare_inputs(title, abstract)

    tokenizer, model, device, id2label = load_artifacts()

    encoding = tokenizer(
        title,
        abstract if abstract else None,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits.detach().cpu().numpy()[0]

    probs = sigmoid(logits)

    top_predictions, predicted_labels = decode_predictions(
        probs=probs,
        id2label=id2label,
        global_threshold=GLOBAL_THRESHOLD,
        top_k=top_k,
    )

    return {
        "top_predictions": top_predictions,
        "predicted_labels": predicted_labels,
        "global_threshold": GLOBAL_THRESHOLD,
    }


with st.sidebar:
    st.header("О приложении")
    st.markdown(
        """
        **Модель:** SciBERT multilabel classifier  
        **Задача:** отнесение статьи к одной или нескольким arXiv-категориям  
        **Размер таксономии:** 148 классов
        """
    )

    try:
        load_artifacts()
        st.success("Артефакты модели загружены")
        st.write(f"Global threshold: **{round(GLOBAL_THRESHOLD, 2)}**")
    except Exception:
        st.error("Не удалось загрузить артефакты модели")


with st.expander("Что означают результаты"):
    st.markdown(
        f"""
        У каждой категории считается независимая вероятность (`sigmoid` от логита)

        **Блоки на странице:**
        1. **Predicted labels** - категории, прошедшие порог `threshold = {round(GLOBAL_THRESHOLD, 2)}`.
        2. **Top-k labels** - самые вероятные категории, даже если они ниже порога

        Метки выводятся в формате:  
        `код arXiv - человекочитаемое название`, например:  
        `cs.LG - Machine Learning`
        """
    )


with st.form("article_form"):
    title = st.text_input(
        "Название статьи",
        placeholder="Например: Attention Is All You Need",
    )

    abstract = st.text_area(
        "Abstract",
        placeholder="Вставьте abstract статьи. Можно оставить пустым",
        height=220,
    )

    top_k = st.slider("Сколько top-меток показывать", min_value=3, max_value=20, value=10)

    submitted = st.form_submit_button("Классифицировать")


if submitted:
    try:
        with st.spinner("Обрабатываю текст..."):
            result = predict(title, abstract, top_k=top_k)

        st.success("Готово!")

        st.subheader("Predicted labels")
        if result["predicted_labels"]:
            for item in result["predicted_labels"]:
                st.write(
                    f"**{item['label']}** — "
                    f"{item['score']:.4f} "
                    f"(threshold: {item['threshold']:.2f})"
                )
                st.progress(float(min(item["score"], 1.0)))
        else:
            st.info(
                "Ни одна метка не прошла threshold"
                "Посмотрите top-k ниже"
            )

        st.subheader(f"Top-{top_k} labels")
        for item in result["top_predictions"]:
            st.write(f"**{item['label']}** — {item['score']:.4f}")
            st.progress(float(min(item["score"], 1.0)))

    except Exception as e:
        st.error(f"Ошибка: {e}")


st.markdown("---")
