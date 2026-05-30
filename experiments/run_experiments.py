import os
import time
import math
import zipfile
import argparse
import urllib.request
import warnings
import traceback

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy import sparse
import openml
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml, fetch_20newsgroups, load_breast_cancer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import umap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RETENTION_RATIOS = [0.10, 0.25, 0.50, 0.75, 0.90]
SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.20

AE_MAX_EPOCHS = 60
AE_BATCH_SIZE = 256
AE_LR = 1e-3
AE_WEIGHT_DECAY = 1e-5
AE_PATIENCE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index(df.columns).astype(str)
    if not df.columns.duplicated().any():
        return df
    counts = {}
    new_cols = []
    for c in df.columns:
        counts[c] = counts.get(c, 0) + 1
        new_cols.append(c if counts[c] == 1 else f"{c}__{counts[c]}")
    df.columns = new_cols
    return df


def k_requested(d: int, r: float) -> int:
    return max(1, int(math.floor(r * d)))


def k_max_pca_like(Xtr_post) -> int:
    n = int(Xtr_post.shape[0])
    d = int(Xtr_post.shape[1])
    return max(1, min(d, n - 1))


def clamp_k(k_req: int, k_max: int) -> int:
    return max(1, min(int(k_req), int(k_max)))


def load_dataset(dataset_id: str):
    dataset_id = dataset_id.upper().strip()
    meta = {}

    if dataset_id == "D1":
        meta["name"] = "Pima Indians Diabetes (OpenML 42608)"
        ds = openml.datasets.get_dataset(42608)
        df = ds.get_data(dataset_format="dataframe")[0]
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]

    elif dataset_id == "D2":
        meta["name"] = "Heart Disease (Cleveland) (UCI)"
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "target",
        ]
        df = pd.read_csv(url, header=None, names=cols).replace("?", pd.NA)
        X = df.drop(columns=["target"])
        y = df["target"]
        mask = X.notna().all(axis=1)
        X = X[mask].astype(float)
        y = y[mask].astype(int)

    elif dataset_id == "D3":
        meta["name"] = "Adult Income (OpenML via sklearn)"
        adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
        df = adult.frame
        X = df.drop(columns=["class"])
        y = df["class"]

    elif dataset_id == "D4":
        meta["name"] = "Parkinsons (UCI ucimlrepo)"
        ds = fetch_ucirepo(name="parkinsons")
        X = ds.data.features.copy()
        y = ds.data.targets.copy()
        for col in ["name", "id", "ID"]:
            if col in X.columns:
                X = X.drop(columns=[col])
        X = X.select_dtypes(include=[np.number])

    elif dataset_id == "D5":
        meta["name"] = "Credit Card Default (OpenML 46374)"
        ds = openml.datasets.get_dataset(46374)
        X, y, _, _ = ds.get_data(
            dataset_format="dataframe", target=ds.default_target_attribute
        )

    elif dataset_id == "D6":
        meta["name"] = "In-Vehicle Coupon Recommendation (UCI 603)"
        ds = fetch_ucirepo(id=603)
        X = ds.data.features
        y = ds.data.targets

    elif dataset_id == "D7":
        meta["name"] = "Breast Cancer Wisconsin (Diagnostic)"
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

    elif dataset_id == "D8":
        meta["name"] = "QSAR Biodegradation (UCI)"
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv"
        )
        urllib.request.urlretrieve(url, "biodeg.csv")
        df = pd.read_csv("biodeg.csv", sep=";")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    elif dataset_id == "D9":
        meta["name"] = "Spambase (UCI 94)"
        ds = fetch_ucirepo(id=94)
        X = ds.data.features
        y = ds.data.targets.iloc[:, 0]

    elif dataset_id == "D10":
        meta["name"] = "MADELON (UCI) train+valid"
        url = "https://archive.ics.uci.edu/static/public/171/madelon.zip"
        urllib.request.urlretrieve(url, "madelon.zip")
        with zipfile.ZipFile("madelon.zip", "r") as z:
            z.extractall("madelon")
        X_train = pd.read_csv(
            "madelon/MADELON/madelon_train.data", sep=" ", header=None
        )
        X_valid = pd.read_csv(
            "madelon/MADELON/madelon_valid.data", sep=" ", header=None
        )
        y_train = pd.read_csv("madelon/MADELON/madelon_train.labels", header=None).iloc[
            :, 0
        ]
        y_valid = pd.read_csv("madelon/madelon_valid.labels", header=None).iloc[:, 0]
        X_train = X_train.dropna(axis=1, how="all")
        X_valid = X_valid.dropna(axis=1, how="all")
        X = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
        y = pd.concat([y_train, y_valid], axis=0, ignore_index=True)

    elif dataset_id == "D11":
        meta["name"] = "ISOLET (OpenML 41966)"
        ds = openml.datasets.get_dataset(41966)
        X, y, _, _ = ds.get_data(
            dataset_format="dataframe", target=ds.default_target_attribute
        )

    elif dataset_id == "D12":
        meta["name"] = "Colon Cancer (OpenML 1432)"
        ds = openml.datasets.get_dataset(1432)
        X, y, _, _ = ds.get_data(
            dataset_format="array", target=ds.default_target_attribute
        )
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = pd.DataFrame(X)
        y = pd.Series(y)

    elif dataset_id == "D13":
        meta["name"] = "AG News (TF-IDF 3000, 6000 samples)"
        meta["sparse"] = True

        from datasets import load_dataset

        dataset = load_dataset("ag_news", split="train")
        texts = dataset["text"]
        labels = dataset["label"]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(texts), size=6000, replace=False)

        texts_sub = [texts[i] for i in idx]
        y = np.array([labels[i] for i in idx])

        vectorizer = TfidfVectorizer(max_features=3000, min_df=5, stop_words="english")

        X = vectorizer.fit_transform(texts_sub)
 

    elif dataset_id == "D14":
        meta["name"] = "GISETTE (UCI) train+valid"
        url = "https://archive.ics.uci.edu/static/public/170/gisette.zip"
        urllib.request.urlretrieve(url, "gisette.zip")
        with zipfile.ZipFile("gisette.zip", "r") as z:
            z.extractall(".")
        X_train = pd.read_csv("GISETTE/gisette_train.data", sep=" ", header=None).iloc[
            :, :-1
        ]
        X_valid = pd.read_csv("GISETTE/gisette_valid.data", sep=" ", header=None).iloc[
            :, :-1
        ]
        y_train = pd.read_csv("GISETTE/gisette_train.labels", header=None).iloc[:, 0]
        y_valid = pd.read_csv("gisette_valid.labels", header=None).iloc[:, 0]
        X = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
        y = pd.concat([y_train, y_valid], axis=0, ignore_index=True)


    elif dataset_id == "D15":
        meta["name"] = "ARCENE (UCI) train+valid"
        url = "https://archive.ics.uci.edu/static/public/167/arcene.zip"
        urllib.request.urlretrieve(url, "arcene.zip")
        with zipfile.ZipFile("arcene.zip", "r") as z:
            z.extractall("arcene")
        X_train = pd.read_csv(
            "arcene/ARCENE/arcene_train.data", sep=r"\s+", header=None
        )
        X_valid = pd.read_csv(
            "arcene/ARCENE/arcene_valid.data", sep=r"\s+", header=None
        )
        y_train = pd.read_csv("arcene/ARCENE/arcene_train.labels", header=None).iloc[
            :, 0
        ]
        y_valid = pd.read_csv("arcene/arcene_valid.labels", header=None).iloc[:, 0]
        X = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
        y = pd.concat([y_train, y_valid], axis=0, ignore_index=True)

    else:
        raise ValueError("dataset_id must be one of D1..D15")

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)
    if isinstance(X, pd.DataFrame):
        X = make_unique_columns(X)

    return X, y, meta


def build_preprocessor(X_train_df: pd.DataFrame) -> ColumnTransformer:
    X_train_df = make_unique_columns(X_train_df)
    cat_cols = X_train_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    num_cols = [c for c in X_train_df.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def preprocess_train_test(X_train, X_test, y_train):
    if sparse.issparse(X_train):
        return X_train.tocsr(), X_test.tocsr()

    if isinstance(X_train, pd.DataFrame):
        X_train = make_unique_columns(X_train)
    if isinstance(X_test, pd.DataFrame):
        X_test = make_unique_columns(X_test)

    pre = build_preprocessor(X_train)
    Xtr = pre.fit_transform(X_train, y_train)
    Xte = pre.transform(X_test)

    if sparse.issparse(Xtr):
        return Xtr.tocsr(), Xte.tocsr()
    return np.asarray(Xtr, dtype=np.float64), np.asarray(Xte, dtype=np.float64)


def pca_or_svd(Xtr, Xte, k_use, seed):
    t0 = time.time()
    if sparse.issparse(Xtr):
        k_use = min(int(k_use), int(Xtr.shape[1]))
        model = TruncatedSVD(n_components=k_use, random_state=seed)
        model.fit(Xtr)
        fit_time = time.time() - t0
        t1 = time.time()
        Xtr_k = model.transform(Xtr)
        Xte_k = model.transform(Xte)
        trans_time = time.time() - t1
        return (
            Xtr_k,
            Xte_k,
            np.nan,
            float(np.sum(model.explained_variance_ratio_)),
            fit_time,
            trans_time,
        )

    k_use = clamp_k(int(k_use), k_max_pca_like(Xtr))
    model = PCA(n_components=k_use, svd_solver="randomized", random_state=seed)
    model.fit(Xtr)
    fit_time = time.time() - t0

    t1 = time.time()
    Xtr_k = model.transform(Xtr)
    Xte_k = model.transform(Xte)
    trans_time = time.time() - t1

    Xte_hat = model.inverse_transform(Xte_k)
    recon = float(np.mean((Xte - Xte_hat) ** 2))
    varexp = float(np.sum(model.explained_variance_ratio_))
    return Xtr_k, Xte_k, recon, varexp, fit_time, trans_time


def umap_proj(Xtr, Xte, k_use, seed):
    # UMAP cannot handle k >= n_samples safely
    n = int(Xtr.shape[0])
    k_use = min(int(k_use), n - 2)
    k_use = max(2, k_use)

    t0 = time.time()
    reducer = umap.UMAP(
        n_components=int(k_use),
        random_state=seed,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
    )
    reducer.fit(Xtr)
    fit_time = time.time() - t0

    t1 = time.time()
    Xtr_k = reducer.transform(Xtr)
    Xte_k = reducer.transform(Xte)
    trans_time = time.time() - t1

    return Xtr_k, Xte_k, np.nan, np.nan, fit_time, trans_time


class TabularAE(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        hidden = int(min(512, max(64, round(math.sqrt(input_dim) * 16))))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return z, xhat


def ae_proj(Xtr, Xte, k_use, seed):
    set_global_seed(seed)

    Xtr_dense = Xtr.toarray() if sparse.issparse(Xtr) else np.asarray(Xtr)
    Xte_dense = Xte.toarray() if sparse.issparse(Xte) else np.asarray(Xte)
    Xtr_dense = Xtr_dense.astype(np.float32)
    Xte_dense = Xte_dense.astype(np.float32)

    k_use = clamp_k(int(k_use), k_max_pca_like(Xtr_dense))

    X_train_part, X_val_part = train_test_split(
        Xtr_dense, test_size=0.10, random_state=seed, shuffle=True
    )

    model = TabularAE(input_dim=Xtr_dense.shape[1], bottleneck_dim=k_use).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_part)),
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_tensor = torch.from_numpy(X_val_part).to(DEVICE)

    t0 = time.time()
    best_val = float("inf")
    best_state = None
    patience = 0

    for _ in range(AE_MAX_EPOCHS):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            _, xhat = model(xb)
            loss = loss_fn(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            _, xhat_val = model(val_tensor)
            val_loss = loss_fn(xhat_val, val_tensor).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= AE_PATIENCE:
                break

    fit_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    t1 = time.time()
    model.eval()
    with torch.no_grad():
        Ztr, _ = model(torch.from_numpy(Xtr_dense).to(DEVICE))
        Zte, Xte_hat = model(torch.from_numpy(Xte_dense).to(DEVICE))
    trans_time = time.time() - t1

    Xtr_k = Ztr.detach().cpu().numpy()
    Xte_k = Zte.detach().cpu().numpy()
    recon = float(np.mean((Xte_dense - Xte_hat.detach().cpu().numpy()) ** 2))

    return Xtr_k, Xte_k, recon, np.nan, fit_time, trans_time


def select_kbest(Xtr, Xte, ytr, k_use):
    t0 = time.time()
    k_use = min(int(k_use), int(Xtr.shape[1]))
    sel = SelectKBest(score_func=f_classif, k=k_use)
    sel.fit(Xtr, ytr)
    fit_time = time.time() - t0

    t1 = time.time()
    Xtr_k = sel.transform(Xtr)
    Xte_k = sel.transform(Xte)
    trans_time = time.time() - t1

    return Xtr_k, Xte_k, np.nan, np.nan, fit_time, trans_time


def rfe_wrap(Xtr, Xte, ytr, k_use, d_post):
    # Scalable RFE using LinearSVC (much better for high-dim)
    k_use = min(int(k_use), int(Xtr.shape[1]))

    t0 = time.time()
    estimator = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=0)
    selector = RFE(estimator=estimator, n_features_to_select=k_use, step=0.2)
    selector.fit(Xtr, ytr)
    fit_time = time.time() - t0

    t1 = time.time()
    Xtr_k = selector.transform(Xtr)
    Xte_k = selector.transform(Xte)
    trans_time = time.time() - t1

    return Xtr_k, Xte_k, np.nan, np.nan, fit_time, trans_time


def classifiers(seed):
    return {
        "LR": LogisticRegression(max_iter=3000, solver="lbfgs"),
        "GB": GradientBoostingClassifier(random_state=seed),
        "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),
    }


def eval_downstream(Xtr_k, Xte_k, ytr, yte, seed):
    if sparse.issparse(Xtr_k):
        Xtr_k = Xtr_k.toarray()
        Xte_k = Xte_k.toarray()

    out = {}
    for name, clf in classifiers(seed).items():
        clf.fit(Xtr_k, ytr)
        pred = clf.predict(Xte_k)
        out[name] = {
            "accuracy": float(accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro")),
        }
    return out


def safe_run(
    method_name,
    fn,
    ytr,
    yte,
    seed,
    dataset_id,
    dataset_name,
    r_req,
    k_req,
    k_use,
    eff_r,
):
    rows = []
    try:
        Xtr_k, Xte_k, recon, varexp, fit_t, trans_t = fn()
        scores = eval_downstream(Xtr_k, Xte_k, np.asarray(ytr), np.asarray(yte), seed)
        for clf_name, m in scores.items():
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "seed": seed,
                    "retention_ratio_requested": float(r_req),
                    "retention_ratio_effective": float(eff_r),
                    "k_requested": int(k_req),
                    "k_used": int(k_use),
                    "method": method_name,
                    "classifier": clf_name,
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "recon_mse": (
                        float(recon)
                        if not (isinstance(recon, float) and np.isnan(recon))
                        else np.nan
                    ),
                    "var_explained": (
                        float(varexp)
                        if not (isinstance(varexp, float) and np.isnan(varexp))
                        else np.nan
                    ),
                    "fit_time_sec": float(fit_t),
                    "transform_time_sec": float(trans_t),
                    "status": "OK",
                    "error": "",
                }
            )
    except Exception as e:
        err = str(e)
        if "RFE_SKIPPED_HIGH_DIM" in err:
            err = "RFE_SKIPPED_HIGH_DIM"
        tb = traceback.format_exc(limit=2).replace("\n", " ")
        for clf_name in ["LR", "GB", "SVM_RBF"]:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "seed": seed,
                    "retention_ratio_requested": float(r_req),
                    "retention_ratio_effective": float(eff_r),
                    "k_requested": int(k_req),
                    "k_used": int(k_use),
                    "method": method_name,
                    "classifier": clf_name,
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "recon_mse": np.nan,
                    "var_explained": np.nan,
                    "fit_time_sec": np.nan,
                    "transform_time_sec": np.nan,
                    "status": "ERROR",
                    "error": (err[:400] + " | " + tb[:600])[:900],
                }
            )
    return rows


def run_dataset(dataset_id: str, out_dir: str, force: bool):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_id}_raw_results.csv")

    if (not force) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(
            f"[{dataset_id}] exists, skipping (use --force to overwrite): {out_path}",
            flush=True,
        )
        return

    X, y, meta = load_dataset(dataset_id)
    dataset_name = meta.get("name", "")
    rows = []

    for seed in SEEDS:
        set_global_seed(seed)
        try:
            if sparse.issparse(X):
                Xtr, Xte, ytr, yte = train_test_split(
                    X,
                    np.asarray(y),
                    test_size=TEST_SIZE,
                    random_state=seed,
                    stratify=np.asarray(y),
                )
            else:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
                )
        except Exception:
            if sparse.issparse(X):
                Xtr, Xte, ytr, yte = train_test_split(
                    X, np.asarray(y), test_size=TEST_SIZE, random_state=seed
                )
            else:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=seed
                )

        Xtr_p, Xte_p = preprocess_train_test(Xtr, Xte, ytr)
        d_post = int(Xtr_p.shape[1])

        for r in RETENTION_RATIOS:
            k_req = k_requested(d_post, r)
            if sparse.issparse(Xtr_p):
                k_use = min(k_req, d_post)
            else:
                k_use = clamp_k(k_req, k_max_pca_like(Xtr_p))
            eff_r = float(k_use) / float(d_post)

            print(
                f"[{dataset_id}] seed={seed} r={r:.2f} k_req={k_req} k_use={k_use} eff_r={eff_r:.4f} d={d_post}",
                flush=True,
            )

            rows += safe_run(
                "PCA_or_SVD",
                lambda: pca_or_svd(Xtr_p, Xte_p, k_use, seed),
                ytr,
                yte,
                seed,
                dataset_id,
                dataset_name,
                r,
                k_req,
                k_use,
                eff_r,
            )
            rows += safe_run(
                "UMAP",
                lambda: umap_proj(Xtr_p, Xte_p, k_use, seed),
                ytr,
                yte,
                seed,
                dataset_id,
                dataset_name,
                r,
                k_req,
                k_use,
                eff_r,
            )
            rows += safe_run(
                "AE",
                lambda: ae_proj(Xtr_p, Xte_p, k_use, seed),
                ytr,
                yte,
                seed,
                dataset_id,
                dataset_name,
                r,
                k_req,
                k_use,
                eff_r,
            )
            rows += safe_run(
                "SelectKBest",
                lambda: select_kbest(Xtr_p, Xte_p, np.asarray(ytr), k_use),
                ytr,
                yte,
                seed,
                dataset_id,
                dataset_name,
                r,
                k_req,
                k_use,
                eff_r,
            )
            rows += safe_run(
                "RFE",
                lambda: rfe_wrap(Xtr_p, Xte_p, np.asarray(ytr), k_use, d_post),
                ytr,
                yte,
                seed,
                dataset_id,
                dataset_name,
                r,
                k_req,
                k_use,
                eff_r,
            )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[{dataset_id}] saved -> {out_path} | rows={len(rows)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="D1", help="D1..D15")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument(
        "--force", action="store_true", help="overwrite existing dataset CSV"
    )
    args = p.parse_args()

    run_dataset(args.dataset, args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
