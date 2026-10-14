# -*- coding: utf-8 -*-
"""
PLANT DISEASE DETECTION USING THE PLANTVILLAGE DATASET  (Streamlit app)

Run with:
    pip install streamlit tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python Pillow
    streamlit run plant_disease_streamlit_app.py

Pipeline (one page per step, selectable from the sidebar):
    1. Dataset            - locate / extract the dataset and inspect its structure
    2. Data Cleaning      - build the image inventory, drop invalid files and exact duplicates
    3. EDA                - class distribution, size/mode distributions, samples, correlations
    4. Split & Pipeline   - stratified 70/15/15 split and class mapping
    5. Train              - VGG16 transfer learning, then fine-tuning of block5
    6. Evaluate           - metrics, classification report, confusion matrix
    7. Predict            - classify an uploaded leaf image
"""

import gc
import os
import json
import random
import hashlib
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from PIL import Image, ImageFile

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        CSVLogger,
        Callback,
    )
except ImportError:
    st.error("TensorFlow is not installed. Run `pip install tensorflow` and restart the app.")
    st.stop()

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ImageFile.LOAD_TRUNCATED_IMAGES = False
sns.set_theme(style="whitegrid")

PAGES = [
    "1. Dataset",
    "2. Data Cleaning",
    "3. EDA",
    "4. Split & Pipeline",
    "5. Train",
    "6. Evaluate",
    "7. Predict",
]


# --------------------------------------------------------------------------
# Session state helpers
# --------------------------------------------------------------------------
def init_state():
    defaults = {
        "dataset_root": "",
        "output_dir": str(Path.cwd() / "plantvillage_vgg16_outputs"),
        "metadata_df": None,
        "duplicate_paths": None,
        "clean_df": None,
        "train_df": None,
        "validation_df": None,
        "test_df": None,
        "class_names": None,
        "model": None,
        "history_head": None,
        "history_fine": None,
        "y_true": None,
        "y_pred": None,
        "probabilities": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def output_dir() -> Path:
    path = Path(st.session_state["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def require(key: str, message: str) -> bool:
    """Stop a page early (with a friendly message) if a prerequisite is missing."""
    if st.session_state.get(key) is None:
        st.warning(message)
        return False
    return True


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def class_index_maps():
    names = st.session_state["class_names"]
    class_to_index = {name: i for i, name in enumerate(names)}
    index_to_class = {i: name for name, i in class_to_index.items()}
    return class_to_index, index_to_class


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------
def find_candidate_roots(base: Path, min_classes: int = 10, max_depth: int = 4):
    """Find folders that contain at least `min_classes` sub-folders holding images."""
    candidates = []
    base = Path(base)
    if not base.exists():
        return candidates

    def depth_of(p: Path) -> int:
        return len(p.relative_to(base).parts)

    for folder in [base, *base.rglob("*")]:
        if not folder.is_dir() or depth_of(folder) > max_depth:
            continue
        class_folders = 0
        try:
            for sub in folder.iterdir():
                if sub.is_dir() and not sub.name.startswith("."):
                    if any(
                        f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                        for f in sub.iterdir()
                    ):
                        class_folders += 1
        except PermissionError:
            continue
        if class_folders >= min_classes:
            candidates.append((str(folder), class_folders))
    return candidates


def extract_zip(zip_path: str, extract_dir: str):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            raise ValueError(f"Corrupted file detected: {bad_file}")
        names = zf.namelist()
        progress = st.progress(0.0, text="Extracting...")
        for i, name in enumerate(names, 1):
            zf.extract(name, extract_dir)
            if i % 200 == 0 or i == len(names):
                progress.progress(i / len(names), text=f"Extracting {i}/{len(names)} files")
        progress.empty()
    return len(names)


def build_inventory(dataset_root: Path) -> pd.DataFrame:
    """Image paths, class names, extensions, dimensions, colour modes and file sizes."""
    class_dirs = [
        d for d in sorted(dataset_root.iterdir())
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("__")
    ]
    all_files = []
    for class_dir in class_dirs:
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                all_files.append((class_dir.name, p))

    records = []
    progress = st.progress(0.0, text="Inspecting images...")
    total = max(len(all_files), 1)

    for i, (class_name, image_path) in enumerate(all_files, 1):
        try:
            file_size = image_path.stat().st_size
            with Image.open(image_path) as img:
                width, height = img.size
                image_format = img.format
                colour_mode = img.mode
                img.verify()
            records.append({
                "image_path": str(image_path),
                "class_name": class_name,
                "extension": image_path.suffix.lower(),
                "file_size_bytes": file_size,
                "width": width,
                "height": height,
                "image_format": image_format,
                "colour_mode": colour_mode,
                "image_valid": True,
                "error": None,
            })
        except Exception as e:
            records.append({
                "image_path": str(image_path),
                "class_name": class_name,
                "extension": image_path.suffix.lower(),
                "file_size_bytes": image_path.stat().st_size if image_path.exists() else None,
                "width": None,
                "height": None,
                "image_format": None,
                "colour_mode": None,
                "image_valid": False,
                "error": str(e),
            })
        if i % 100 == 0 or i == total:
            progress.progress(i / total, text=f"Inspected {i}/{total} images")
    progress.empty()
    return pd.DataFrame(records)


def sha256_file(filepath, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(filepath, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicates(metadata_df: pd.DataFrame):
    valid_paths = metadata_df.loc[metadata_df["image_valid"], "image_path"].tolist()
    hash_records = []
    progress = st.progress(0.0, text="Hashing files...")
    total = max(len(valid_paths), 1)
    for i, filepath in enumerate(valid_paths, 1):
        try:
            hash_records.append({"image_path": filepath, "sha256": sha256_file(filepath)})
        except Exception:
            pass
        if i % 100 == 0 or i == total:
            progress.progress(i / total, text=f"Hashed {i}/{total} files")
    progress.empty()

    hash_df = pd.DataFrame(hash_records)
    if hash_df.empty:
        return set()
    mask = hash_df.duplicated(subset="sha256", keep="first")
    return set(hash_df.loc[mask, "image_path"])


# --------------------------------------------------------------------------
# tf.data pipeline + model
# --------------------------------------------------------------------------
def dataframe_to_tf_dataset(dataframe, class_to_index, batch_size, training=False):
    paths = dataframe["image_path"].to_numpy()
    labels = dataframe["class_name"].map(class_to_index).to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(dataframe), 10000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method="bilinear")
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(num_classes: int):
    tf.keras.backend.clear_session()

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.08),
            layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        ],
        name="data_augmentation",
    )

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="feature_batch_normalization")(x)
    x = layers.Dense(
        512, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_512"
    )(x)
    x = layers.Dropout(0.40, name="dropout_1")(x)
    x = layers.Dense(
        256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_256"
    )(x)
    x = layers.Dropout(0.30, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_predictions")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="PlantVillage_VGG16")


def get_base_model(model):
    """Return the nested VGG16 layer of the assembled model."""
    for layer in model.layers:
        if layer.name.startswith("vgg16"):
            return layer
    raise ValueError("VGG16 base model not found in the model.")


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


class StreamlitTrainingCallback(Callback):
    """Updates a progress bar, status line and live accuracy/loss chart during training."""

    def __init__(self, total_epochs, progress_bar, status, chart_slot):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status = status
        self.chart_slot = chart_slot
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.rows.append({k: float(v) for k, v in logs.items()})
        self.progress_bar.progress(
            (epoch + 1) / self.total_epochs,
            text=f"Epoch {epoch + 1}/{self.total_epochs}",
        )
        self.status.info(
            f"Epoch {epoch + 1}/{self.total_epochs} - "
            f"loss {logs.get('loss', 0):.4f} | acc {logs.get('accuracy', 0):.4f} | "
            f"val_loss {logs.get('val_loss', 0):.4f} | val_acc {logs.get('val_accuracy', 0):.4f}"
        )
        df = pd.DataFrame(self.rows)
        cols = [c for c in ["accuracy", "val_accuracy"] if c in df.columns]
        if cols:
            self.chart_slot.line_chart(df[cols])


def plot_training_history(history_dict, title):
    history_df = pd.DataFrame(history_dict)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history_df["accuracy"], label="Training Accuracy")
    axes[0].plot(history_df["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history_df["loss"], label="Training Loss")
    axes[1].plot(history_df["val_loss"], label="Validation Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner="Loading model...")
def load_model_from_path(path: str, mtime: float):
    # `mtime` is part of the cache key so a re-trained file is reloaded.
    return tf.keras.models.load_model(path)


def save_class_names():
    names = st.session_state["class_names"]
    if names:
        with open(output_dir() / "class_names.json", "w") as f:
            json.dump(names, f, indent=2)


# ==========================================================================
# PAGE 1 - DATASET
# ==========================================================================
def page_dataset():
    st.header("1. Locate the dataset")
    st.write(
        "Point the app at the PlantVillage dataset on this machine. If you only have the "
        "ZIP file, extract it first. The dataset root is the folder whose sub-folders "
        "are the disease classes."
    )

    with st.expander("Extract a ZIP file (optional)", expanded=False):
        zip_path = st.text_input("Path to PlantVillage.zip", placeholder="/path/to/PlantVillage.zip")
        extract_to = st.text_input("Extract to", value=str(Path.cwd() / "PlantVillage"))
        if st.button("Verify and extract"):
            if not zip_path or not os.path.isfile(zip_path):
                st.error(f"ZIP file not found: {zip_path}")
            else:
                size_mb = os.path.getsize(zip_path) / (1024 ** 2)
                st.write(f"ZIP found ({size_mb:.2f} MB). Verifying...")
                try:
                    n = extract_zip(zip_path, extract_to)
                    st.success(f"Extracted {n} files to {extract_to}")
                    st.session_state["dataset_root"] = extract_to
                except Exception as e:
                    st.error(str(e))

    st.subheader("Dataset root")
    root_input = st.text_input(
        "Dataset root folder",
        value=st.session_state["dataset_root"],
        placeholder="/path/to/PlantVillage/PlantVillage",
    )
    st.session_state["dataset_root"] = root_input

    col_a, col_b = st.columns(2)
    min_classes = col_a.number_input("Min. class folders for auto-detect", 2, 100, 10)
    if col_b.button("Auto-detect dataset root", use_container_width=True):
        if not root_input or not Path(root_input).exists():
            st.error("Enter an existing folder (e.g. where you extracted the ZIP) first.")
        else:
            with st.spinner("Searching..."):
                found = find_candidate_roots(Path(root_input), int(min_classes))
            if found:
                st.session_state["root_candidates"] = found
            else:
                st.warning("No candidate dataset roots found.")
                st.session_state.pop("root_candidates", None)

    if st.session_state.get("root_candidates"):
        st.write("Possible dataset roots:")
        for path, n_classes in st.session_state["root_candidates"]:
            c1, c2 = st.columns([5, 1])
            c1.code(f"{path}   ({n_classes} classes)")
            if c2.button("Use", key=f"use_{path}"):
                st.session_state["dataset_root"] = path
                st.rerun()

    st.subheader("Output folder")
    st.session_state["output_dir"] = st.text_input(
        "Where to save models, logs and reports", value=st.session_state["output_dir"]
    )

    root = Path(st.session_state["dataset_root"]) if st.session_state["dataset_root"] else None
    if root and root.exists():
        st.subheader("Structure preview")
        rows = []
        for p in sorted(root.iterdir()):
            if p.is_dir():
                n = sum(
                    1 for f in p.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                )
                rows.append({"class_name": p.name, "image_count": n})
        if rows:
            class_df = pd.DataFrame(rows)
            c1, c2, c3 = st.columns(3)
            c1.metric("Class folders", len(class_df))
            c2.metric("Total images", int(class_df["image_count"].sum()))
            c3.metric("Smallest class", int(class_df["image_count"].min()))
            st.dataframe(class_df, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(13, max(6, len(class_df) * 0.4)))
            ax.barh(class_df["class_name"], class_df["image_count"])
            ax.set_xlabel("Number of images")
            ax.set_ylabel("Class")
            ax.set_title("PlantVillage Class Distribution")
            fig.tight_layout()
            show_fig(fig)
        else:
            st.warning("No class sub-folders found here. Check the dataset root.")
    elif root:
        st.error(f"Path does not exist: {root}")


# ==========================================================================
# PAGE 2 - DATA CLEANING
# ==========================================================================
def page_cleaning():
    st.header("2. Data cleaning")
    root = st.session_state["dataset_root"]
    if not root or not Path(root).exists():
        st.warning("Set a valid dataset root on the **Dataset** page first.")
        return

    st.write(
        "Builds an inventory of every image (path, class, extension, dimensions, colour mode, "
        "file size), flags unreadable files, and finds exact duplicates by SHA-256 hash."
    )

    if st.button("Run inventory + duplicate detection", type="primary"):
        metadata_df = build_inventory(Path(root))
        if metadata_df.empty:
            st.error("No images were found. Check the dataset root and folder structure.")
            return
        st.session_state["metadata_df"] = metadata_df
        st.session_state["duplicate_paths"] = find_duplicates(metadata_df)

        clean_df = metadata_df[metadata_df["image_valid"]].copy()
        clean_df = clean_df[~clean_df["image_path"].isin(st.session_state["duplicate_paths"])]
        clean_df = clean_df[
            (clean_df["width"] > 0)
            & (clean_df["height"] > 0)
            & (clean_df["file_size_bytes"] > 0)
        ].reset_index(drop=True)
        st.session_state["clean_df"] = clean_df

        # Any earlier splits/models are now stale.
        for key in ["train_df", "validation_df", "test_df", "class_names"]:
            st.session_state[key] = None

    if not require("metadata_df", "Run the inventory above to continue."):
        return

    metadata_df = st.session_state["metadata_df"]
    clean_df = st.session_state["clean_df"]
    duplicates = st.session_state["duplicate_paths"]

    st.subheader("Dataset summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total files", len(metadata_df))
    c2.metric("Invalid images", int((~metadata_df["image_valid"]).sum()))
    c3.metric("Exact duplicates", len(duplicates))
    c4.metric("Cleaned images", len(clean_df))
    c5.metric("Classes", clean_df["class_name"].nunique())

    st.dataframe(metadata_df.head(50), use_container_width=True)

    with st.expander("Missing values, data types and numerical summary"):
        st.write("Missing values")
        st.dataframe(metadata_df.isnull().sum().to_frame("missing_values"))
        st.write("Data types")
        st.dataframe(metadata_df.dtypes.astype(str).to_frame("data_type"))
        st.write("Numerical summary")
        st.dataframe(metadata_df.describe(include="all").T.astype(str))

    if (~metadata_df["image_valid"]).any():
        with st.expander("Invalid files"):
            st.dataframe(
                metadata_df.loc[~metadata_df["image_valid"], ["image_path", "error"]],
                use_container_width=True,
            )

    st.subheader("Class counts after cleaning")
    class_counts = clean_df["class_name"].value_counts().sort_values(ascending=False)
    class_df = class_counts.rename_axis("class_name").reset_index(name="image_count")
    st.dataframe(class_df, use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Smallest class", int(class_df["image_count"].min()))
    c2.metric("Largest class", int(class_df["image_count"].max()))
    c3.metric("Average per class", round(class_df["image_count"].mean(), 2))


# ==========================================================================
# PAGE 3 - EDA
# ==========================================================================
def page_eda():
    st.header("3. Exploratory data analysis")
    if not require("clean_df", "Run **Data Cleaning** first."):
        return
    clean_df = st.session_state["clean_df"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Class distribution",
        "Numerical distributions",
        "Categorical distributions",
        "Representative images",
        "Correlation heatmap",
    ])

    with tab1:
        class_df = (
            clean_df["class_name"].value_counts().rename_axis("class_name")
            .reset_index(name="image_count")
        )
        fig, ax = plt.subplots(figsize=(14, max(8, len(class_df) * 0.28)))
        sns.barplot(data=class_df, x="image_count", y="class_name", ax=ax)
        ax.set_title("PlantVillage Image Distribution by Class")
        ax.set_xlabel("Number of Images")
        ax.set_ylabel("Plant Disease Class")
        fig.tight_layout()
        show_fig(fig)

    with tab2:
        st.caption("Unusual dimensions, extreme aspect ratios or tiny files may indicate damaged images.")
        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
        for ax, column in zip(axes, ["width", "height", "file_size_bytes"]):
            sns.histplot(clean_df[column].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
        fig.tight_layout()
        show_fig(fig)

    with tab3:
        st.caption("Images are converted to RGB at load time, so grayscale/RGBA files are handled consistently.")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ext_counts = clean_df["extension"].value_counts()
        axes[0].bar(ext_counts.index, ext_counts.values)
        axes[0].set_title("Image File Extensions")
        axes[0].set_xlabel("Extension")
        axes[0].set_ylabel("Count")
        mode_counts = clean_df["colour_mode"].value_counts()
        axes[1].bar(mode_counts.index, mode_counts.values)
        axes[1].set_title("Image Colour Modes")
        axes[1].set_xlabel("Colour Mode")
        axes[1].set_ylabel("Count")
        fig.tight_layout()
        show_fig(fig)

    with tab4:
        sample_classes = sorted(clean_df["class_name"].unique())
        if st.button("Shuffle samples") or "eda_sample_classes" not in st.session_state:
            st.session_state["eda_sample_classes"] = random.sample(
                sample_classes, min(12, len(sample_classes))
            )
        selected = st.session_state["eda_sample_classes"]

        fig, axes = plt.subplots(3, 4, figsize=(15, 11))
        axes = axes.flatten()
        for ax in axes:
            ax.axis("off")
        for ax, class_name in zip(axes, selected):
            rows = clean_df.loc[clean_df["class_name"] == class_name, "image_path"]
            image_path = rows.sample(1).iloc[0]
            with Image.open(image_path) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(class_name, fontsize=8)
            ax.axis("off")
        fig.suptitle("Representative PlantVillage Images", fontsize=15)
        fig.tight_layout()
        show_fig(fig)

    with tab5:
        st.caption(
            "Linear relationships among simple image characteristics, computed on a random sample."
        )
        sample_size = st.slider("Sample size", 50, min(2000, len(clean_df)), min(500, len(clean_df)), step=50)
        if st.button("Compute correlation heatmap"):
            paths = clean_df["image_path"].tolist()
            rng = random.Random(SEED)
            sample_paths = rng.sample(paths, min(sample_size, len(paths)))

            feature_records = []
            progress = st.progress(0.0, text="Extracting features...")
            for i, filepath in enumerate(sample_paths, 1):
                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                        img = img.convert("RGB")
                        img.thumbnail((128, 128))
                        arr = np.asarray(img, dtype=np.float32) / 255.0
                        feature_records.append({
                            "mean_red": arr[:, :, 0].mean(),
                            "mean_green": arr[:, :, 1].mean(),
                            "mean_blue": arr[:, :, 2].mean(),
                            "brightness": arr.mean(),
                            "contrast": arr.std(),
                            "width": width,
                            "height": height,
                        })
                except Exception:
                    continue
                if i % 25 == 0 or i == len(sample_paths):
                    progress.progress(i / len(sample_paths))
            progress.empty()

            feature_df = pd.DataFrame(feature_records)
            st.dataframe(feature_df.head(), use_container_width=True)
            corr = feature_df.corr()
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0, square=True, ax=ax)
            ax.set_title("Correlation Heatmap of Image-Level Features")
            fig.tight_layout()
            show_fig(fig)


# ==========================================================================
# PAGE 4 - SPLIT & PIPELINE
# ==========================================================================
def page_split():
    st.header("4. Train / validation / test split")
    if not require("clean_df", "Run **Data Cleaning** first."):
        return
    clean_df = st.session_state["clean_df"]

    train_pct = st.slider("Training share (%)", 50, 90, 70, step=5)
    st.caption(
        f"Remaining {100 - train_pct}% is divided equally between validation and testing "
        f"({(100 - train_pct) / 2:.1f}% each). Splits are stratified by class."
    )

    if st.button("Create stratified split", type="primary"):
        class_counts = clean_df["class_name"].value_counts()
        insufficient = class_counts[class_counts < 3]
        if len(insufficient) > 0:
            st.error(
                "Some classes have fewer than three images, so a reliable stratified "
                f"three-way split is not possible:\n{insufficient.to_dict()}"
            )
            return

        train_df, temporary_df = train_test_split(
            clean_df,
            test_size=1 - train_pct / 100,
            random_state=SEED,
            stratify=clean_df["class_name"],
        )
        validation_df, test_df = train_test_split(
            temporary_df,
            test_size=0.50,
            random_state=SEED,
            stratify=temporary_df["class_name"],
        )

        st.session_state["train_df"] = train_df.reset_index(drop=True)
        st.session_state["validation_df"] = validation_df.reset_index(drop=True)
        st.session_state["test_df"] = test_df.reset_index(drop=True)
        st.session_state["class_names"] = sorted(clean_df["class_name"].unique())
        save_class_names()

    if not require("train_df", "Create the split above to continue."):
        return

    train_df = st.session_state["train_df"]
    validation_df = st.session_state["validation_df"]
    test_df = st.session_state["test_df"]
    total = len(train_df) + len(validation_df) + len(test_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Training", len(train_df), f"{len(train_df) / total:.2%}", delta_color="off")
    c2.metric("Validation", len(validation_df), f"{len(validation_df) / total:.2%}", delta_color="off")
    c3.metric("Testing", len(test_df), f"{len(test_df) / total:.2%}", delta_color="off")

    all_classes = set(clean_df["class_name"])
    ok = (
        set(train_df["class_name"]) == all_classes
        and set(validation_df["class_name"]) == all_classes
        and set(test_df["class_name"]) == all_classes
    )
    if ok:
        st.success("Every class is represented in all three sets.")
    else:
        st.error("Some classes are missing from one of the sets.")

    split_comparison = pd.DataFrame({
        "Training": train_df["class_name"].value_counts(),
        "Validation": validation_df["class_name"].value_counts(),
        "Testing": test_df["class_name"].value_counts(),
    }).fillna(0)

    fig, ax = plt.subplots(figsize=(16, 6))
    split_comparison.plot(kind="bar", width=0.85, ax=ax)
    ax.set_title("Class Distribution Across Training, Validation and Testing")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    show_fig(fig)

    st.subheader("Class mapping")
    names = st.session_state["class_names"]
    st.write(f"{len(names)} classes. Input size: {IMG_SIZE}x{IMG_SIZE} (VGG16 standard).")
    st.dataframe(
        pd.DataFrame({"index": range(len(names)), "class_name": names}),
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================================
# PAGE 5 - TRAIN
# ==========================================================================
def page_train():
    st.header("5. Train the VGG16 model")
    if not require("train_df", "Create the train/validation/test split first."):
        return

    class_names = st.session_state["class_names"]
    num_classes = len(class_names)
    class_to_index, _ = class_index_maps()
    out = output_dir()

    gpus = tf.config.list_physical_devices("GPU")
    st.caption(f"TensorFlow {tf.__version__} | GPU devices: {len(gpus)}")
    if not gpus:
        st.warning("No GPU detected. Training VGG16 on CPU will be slow; consider reducing epochs.")

    with st.expander("Hyperparameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        batch_size = c1.number_input("Batch size", 8, 256, 32, step=8)
        initial_epochs = c2.number_input("Transfer-learning epochs (max)", 1, 100, 15)
        fine_tune_epochs = c3.number_input("Fine-tuning epochs (max)", 1, 100, 10)
        c4, c5 = st.columns(2)
        initial_lr = c4.number_input("Initial learning rate", value=1e-3, format="%.6f", step=1e-4)
        fine_tune_lr = c5.number_input("Fine-tuning learning rate", value=1e-5, format="%.8f", step=1e-6)

    def get_datasets():
        return (
            dataframe_to_tf_dataset(st.session_state["train_df"], class_to_index, batch_size, training=True),
            dataframe_to_tf_dataset(st.session_state["validation_df"], class_to_index, batch_size),
        )

    # ---- Stage 1: transfer learning ----------------------------------
    st.subheader("Stage 1 - Transfer learning (frozen VGG16 base)")
    st.write(
        "Only the new classification head (global average pooling, batch norm, two dense "
        "layers with dropout/L2, softmax) is trained. Callbacks: checkpointing, early "
        "stopping, learning-rate reduction and CSV logging."
    )

    if st.button("Build model and train classification head", type="primary"):
        with st.spinner("Building model (downloads ImageNet weights on first run)..."):
            model = build_model(num_classes)
            compile_model(model, initial_lr)
        st.session_state["model"] = model

        train_ds, validation_ds = get_datasets()
        callbacks = [
            ModelCheckpoint(
                filepath=str(out / "vgg16_best_transfer_learning.keras"),
                monitor="val_loss", save_best_only=True, verbose=0,
            ),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=0),
            CSVLogger(str(out / "transfer_learning_history.csv")),
        ]

        progress = st.progress(0.0, text="Starting...")
        status = st.empty()
        chart_slot = st.empty()
        callbacks.append(StreamlitTrainingCallback(initial_epochs, progress, status, chart_slot))

        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=int(initial_epochs),
            callbacks=callbacks,
            verbose=0,
        )
        st.session_state["history_head"] = history.history
        st.session_state["model"] = model
        # Invalidate any previous evaluation
        for key in ["y_true", "y_pred", "probabilities"]:
            st.session_state[key] = None
        st.success("Transfer-learning stage finished. Best weights restored.")
        gc.collect()

    if st.session_state["history_head"] is not None:
        show_fig(plot_training_history(st.session_state["history_head"], "VGG16 Transfer Learning"))

    # ---- Stage 2: fine-tuning ----------------------------------------
    st.subheader("Stage 2 - Fine-tune the final convolutional block")
    st.write(
        "Unfreezes `block5_*` of VGG16 (earlier blocks stay frozen), recompiles with a much "
        "lower learning rate and continues training."
    )

    if st.session_state["model"] is None:
        st.info("Train the classification head first (or load a saved model on the Predict page).")
    elif st.button("Fine-tune block5"):
        model = st.session_state["model"]
        base_model = get_base_model(model)

        base_model.trainable = True
        for layer in base_model.layers:
            layer.trainable = layer.name.startswith("block5_")

        trainable_layers = [l.name for l in base_model.layers if l.trainable]
        st.write("Trainable VGG16 layers:", ", ".join(trainable_layers))

        compile_model(model, fine_tune_lr)

        train_ds, validation_ds = get_datasets()
        callbacks = [
            ModelCheckpoint(
                filepath=str(out / "vgg16_best_fine_tuned.keras"),
                monitor="val_loss", save_best_only=True, verbose=0,
            ),
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-8, verbose=0),
            CSVLogger(str(out / "fine_tuning_history.csv")),
        ]

        progress = st.progress(0.0, text="Starting...")
        status = st.empty()
        chart_slot = st.empty()
        callbacks.append(StreamlitTrainingCallback(fine_tune_epochs, progress, status, chart_slot))

        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=int(fine_tune_epochs),
            callbacks=callbacks,
            verbose=0,
        )
        st.session_state["history_fine"] = history.history
        st.session_state["model"] = model
        for key in ["y_true", "y_pred", "probabilities"]:
            st.session_state[key] = None
        st.success("Fine-tuning finished. Best weights restored.")
        gc.collect()

    if st.session_state["history_fine"] is not None:
        show_fig(plot_training_history(st.session_state["history_fine"], "VGG16 Fine-Tuning"))

    if st.session_state["model"] is not None:
        with st.expander("Model summary"):
            lines = []
            st.session_state["model"].summary(print_fn=lines.append)
            st.code("\n".join(lines))


# ==========================================================================
# PAGE 6 - EVALUATE
# ==========================================================================
def page_evaluate():
    st.header("6. Evaluate on the test set")
    if not require("test_df", "Create the train/validation/test split first."):
        return
    if not require("model", "Train a model first (page 5)."):
        return

    class_names = st.session_state["class_names"]
    num_classes = len(class_names)
    class_to_index, _ = class_index_maps()
    out = output_dir()
    model = st.session_state["model"]

    if st.button("Generate test predictions", type="primary"):
        test_ds = dataframe_to_tf_dataset(st.session_state["test_df"], class_to_index, 32)
        y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
        with st.spinner("Predicting..."):
            probabilities = model.predict(test_ds, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        assert len(y_true) == len(y_pred)

        st.session_state["y_true"] = y_true
        st.session_state["y_pred"] = y_pred
        st.session_state["probabilities"] = probabilities

    if st.session_state["y_true"] is None:
        st.info("Click the button above to evaluate the current model.")
        return

    y_true = st.session_state["y_true"]
    y_pred = st.session_state["y_pred"]
    st.write(f"Test examples: **{len(y_true)}**")

    # ---- Summary metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])

    st.subheader("Summary metrics")
    cols = st.columns(4)
    for col, (name, value) in zip(cols, list(metrics.items())[:4]):
        col.metric(name, f"{value:.4f}")
    st.dataframe(metrics_df.style.format({"Score": "{:.4f}"}), use_container_width=True, hide_index=True)

    # ---- Classification report
    st.subheader("Classification report")
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(
        report_df.style.format(
            {"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}
        ),
        use_container_width=True,
    )
    report_df.to_csv(out / "classification_report.csv")

    # ---- Confusion matrix
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(
        figsize=(max(12, num_classes * 0.45), max(10, num_classes * 0.40))
    )
    sns.heatmap(
        cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=True, ax=ax
    )
    ax.set_title("VGG16 Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    show_fig(fig)

    st.success(f"Report and confusion matrix saved to `{out}`.")

    csv_bytes = report_df.to_csv().encode("utf-8")
    st.download_button(
        "Download classification report (CSV)",
        csv_bytes,
        file_name="classification_report.csv",
        mime="text/csv",
    )

    # Save the final model for the Predict page
    if st.button("Save current model as final model"):
        final_path = out / "vgg16_final.keras"
        model.save(final_path)
        save_class_names()
        st.success(f"Saved to {final_path}")


# ==========================================================================
# PAGE 7 - PREDICT
# ==========================================================================
def page_predict():
    st.header("7. Predict a plant disease")

    out = output_dir()
    saved_models = sorted(out.glob("*.keras"))

    # ---- Choose the model source
    options = []
    if st.session_state["model"] is not None:
        options.append("Model trained in this session")
    if saved_models:
        options.append("Saved model from the output folder")
    options.append("Upload a .keras model")

    source = st.radio("Model source", options, horizontal=True)

    model = None
    if source == "Model trained in this session":
        model = st.session_state["model"]
    elif source == "Saved model from the output folder":
        choice = st.selectbox("Saved model", [p.name for p in saved_models])
        path = out / choice
        model = load_model_from_path(str(path), path.stat().st_mtime)
    else:
        uploaded_model = st.file_uploader("Upload a .keras model file", type=["keras"])
        if uploaded_model is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(uploaded_model.getvalue())
                tmp_path = tmp.name
            model = load_model_from_path(tmp_path, Path(tmp_path).stat().st_mtime)

    # ---- Class names
    class_names = st.session_state["class_names"]
    if class_names is None and (out / "class_names.json").exists():
        with open(out / "class_names.json") as f:
            class_names = json.load(f)
    if class_names is None:
        uploaded_classes = st.file_uploader(
            "Upload class_names.json (a JSON list of class names, in training order)", type=["json"]
        )
        if uploaded_classes is not None:
            class_names = json.load(uploaded_classes)

    if model is None:
        st.info("Select or upload a model to continue.")
        return
    if class_names is None:
        st.info("Class names are required to label predictions.")
        return
    if model.output_shape[-1] != len(class_names):
        st.error(
            f"The model outputs {model.output_shape[-1]} classes but {len(class_names)} class "
            "names were found. Provide the matching class_names.json."
        )
        return

    # ---- Image input
    tab_upload, tab_camera = st.tabs(["Upload image", "Use camera"])
    with tab_upload:
        image_file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    with tab_camera:
        camera_file = st.camera_input("Take a photo of a leaf")
    image_file = image_file or camera_file

    if image_file is None:
        return

    image = Image.open(image_file).convert("RGB")

    arr = np.asarray(image, dtype=np.float32)
    arr = tf.image.resize(arr, [IMG_SIZE, IMG_SIZE], method="bilinear")
    arr = preprocess_input(tf.cast(arr, tf.float32).numpy())
    probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]

    top_k = min(5, len(class_names))
    top_idx = np.argsort(probs)[::-1][:top_k]

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Input image", use_container_width=True)
    with right:
        best = top_idx[0]
        label = class_names[best].replace("___", " - ").replace("_", " ")
        st.subheader(label)
        st.metric("Confidence", f"{probs[best]:.2%}")
        if probs[best] < 0.60:
            st.warning("Low confidence. The image may not resemble the training data.")

        top_df = pd.DataFrame({
            "Class": [class_names[i] for i in top_idx],
            "Probability": [float(probs[i]) for i in top_idx],
        })
        st.bar_chart(top_df.set_index("Class"))
        st.dataframe(
            top_df.style.format({"Probability": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    init_state()

    st.title("🌿 Plant Disease Detection - PlantVillage + VGG16")

    with st.sidebar:
        st.header("Pipeline")
        page = st.radio("Step", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Progress")
        st.write(("✅" if st.session_state["clean_df"] is not None else "⬜") + " Cleaned data")
        st.write(("✅" if st.session_state["train_df"] is not None else "⬜") + " Split created")
        st.write(("✅" if st.session_state["model"] is not None else "⬜") + " Model ready")
        st.write(("✅" if st.session_state["y_pred"] is not None else "⬜") + " Evaluated")

    if page == PAGES[0]:
        page_dataset()
    elif page == PAGES[1]:
        page_cleaning()
    elif page == PAGES[2]:
        page_eda()
    elif page == PAGES[3]:
        page_split()
    elif page == PAGES[4]:
        page_train()
    elif page == PAGES[5]:
        page_evaluate()
    elif page == PAGES[6]:
        page_predict()


main()
