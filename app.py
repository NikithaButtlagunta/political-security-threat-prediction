import io
import logging
import os
import re
import secrets
import sqlite3
from functools import wraps

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-change-this-before-deployment")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DATABASE = os.path.join(app.root_path, "users.db")
MODEL_NAMES = ["Decision Tree", "Naive Bayes", "SVM"]
MODEL_STORE = {}


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db()
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
    finally:
        connection.close()


init_db()


def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign in to access your dashboard.", "error")
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)
    return wrapped_view


def build_pipeline(model_name):
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced", min_samples_leaf=2
        ),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(class_weight="balanced", random_state=42, max_iter=3000),
    }
    if model_name not in classifiers:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("classifier", classifiers[model_name]),
    ])


def evaluate_model(model_name, x_train, x_test, y_train, y_test, labels):
    pipeline = build_pipeline(model_name)
    logger.info("Training %s on %s rows", model_name, len(x_train))
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    class_report = {}
    for label in labels:
        values = report.get(label, {})
        class_report[label] = {
            "precision": round(float(values.get("precision", 0)), 3),
            "recall": round(float(values.get("recall", 0)), 3),
            "f1": round(float(values.get("f1-score", 0)), 3),
            "support": int(values.get("support", 0)),
        }
    return {
        "pipeline": pipeline,
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "accuracy_percent": round(accuracy * 100, 2),
        "macro_f1": round(float(report.get("macro avg", {}).get("f1-score", 0)), 4),
        "weighted_f1": round(float(report.get("weighted avg", {}).get("f1-score", 0)), 4),
        "rows": int(len(x_train) + len(x_test)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "report": class_report,
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Enter your username and password.", "error")
            return redirect(url_for("login"))
        connection = get_db()
        try:
            user = connection.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        finally:
            connection.close()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("You have signed in successfully.", "success")
            return redirect(url_for("index"))
        flash("Invalid username or password.", "error")
        return redirect(url_for("login"))
    return render_template("auth.html", mode="login")


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get(
            "confirm_password", request.form.get("confirmPassword", "")
        )
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", username):
            flash("Username must be 3–40 characters and may contain letters, numbers, periods, underscores, and hyphens.", "error")
            return redirect(url_for("register"))
        if len(email) > 254 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Enter a valid email address.", "error")
            return redirect(url_for("register"))
        if len(password) < 8:
            flash("Your password must contain at least 8 characters.", "error")
            return redirect(url_for("register"))
        if password != confirm_password:
            flash("The passwords do not match.", "error")
            return redirect(url_for("register"))
        connection = get_db()
        try:
            cursor = connection.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, generate_password_hash(password)),
            )
            connection.commit()
            user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            connection.rollback()
            flash("That username or email is already registered.", "error")
            return redirect(url_for("register"))
        except sqlite3.Error:
            connection.rollback()
            logger.exception("Database registration error")
            flash("Account creation failed. Please try again.", "error")
            return redirect(url_for("register"))
        finally:
            connection.close()
        session.clear()
        session["user_id"] = user_id
        session["username"] = username
        flash("Your account was created successfully.", "success")
        return redirect(url_for("index"))
    return render_template("auth.html", mode="register")


@app.post("/logout")
@login_required
def logout():
    session.clear()
    flash("You have signed out.", "success")
    return redirect(url_for("login"))


@app.get("/")
@login_required
def index():
    model_key = session.get("model_state")
    trained_model = MODEL_STORE.get(model_key) if model_key else None
    result = dict(trained_model) if trained_model else None
    last_prediction = session.pop("last_prediction", None)
    if result is not None:
        result["prediction"] = last_prediction
        result["lexicon_score"] = None
    comparison = result.get("comparison", []) if result else []
    return render_template(
        "index.html",
        username=session.get("username"),
        model_names=MODEL_NAMES,
        trained=bool(trained_model),
        result=result,
        comparison=comparison,
        trained_model=trained_model,
        metrics=trained_model,
        prediction=last_prediction,
    )


@app.post("/train")
@login_required
def train():
    try:
        uploaded = request.files.get("dataset")
        text_column = request.form.get("text_column", "").strip()
        label_column = request.form.get("label_column", "").strip()

        if not uploaded or not uploaded.filename:
            flash("Choose a CSV file first.", "error")
            return redirect(url_for("index"))
        if not uploaded.filename.lower().endswith(".csv"):
            flash("Please upload a .csv file.", "error")
            return redirect(url_for("index"))
        file_bytes = uploaded.read()
        if not file_bytes:
            flash("The uploaded CSV file is empty.", "error")
            return redirect(url_for("index"))
        try:
            frame = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")

        frame.columns = frame.columns.astype(str).str.strip()
        if frame.empty:
            flash("The CSV file contains no data rows.", "error")
            return redirect(url_for("index"))
        if not text_column or not label_column or text_column not in frame.columns or label_column not in frame.columns:
            flash("Select valid text and label columns. Available columns: " + ", ".join(frame.columns), "error")
            return redirect(url_for("index"))
        if text_column == label_column:
            flash("The text column and label column must be different.", "error")
            return redirect(url_for("index"))

        data = frame[[text_column, label_column]].copy()
        data = data.dropna(subset=[text_column, label_column])
        data[text_column] = data[text_column].astype(str).str.strip()
        data[label_column] = data[label_column].astype(str).str.strip()
        data = data[(data[text_column].str.len() > 0) & (data[label_column].str.len() > 0)]
        data = data.drop_duplicates(subset=[text_column, label_column])
        class_counts = data[label_column].value_counts()
        if len(data) < 10 or len(class_counts) < 2:
            flash("Provide at least 10 valid labelled rows and at least two distinct classes.", "error")
            return redirect(url_for("index"))
        stratify_labels = data[label_column] if class_counts.min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            data[text_column], data[label_column], test_size=0.2,
            random_state=42, stratify=stratify_labels
        )
        labels = sorted(data[label_column].unique().tolist())

        # Fit all three classifiers on exactly the same train/test split.
        evaluations = []
        trained_models = {}
        for model_name in MODEL_NAMES:
            evaluation = evaluate_model(model_name, x_train, x_test, y_train, y_test, labels)
            trained_models[model_name] = evaluation
            evaluations.append({
                "model_name": model_name,
                "accuracy_percent": evaluation["accuracy_percent"],
                "macro_f1_percent": round(evaluation["macro_f1"] * 100, 2),
                "weighted_f1_percent": round(evaluation["weighted_f1"] * 100, 2),
            })
        # Select the highest macro-F1 model; use accuracy as the tie-breaker.
        best_name = max(
            MODEL_NAMES,
            key=lambda name: (trained_models[name]["macro_f1"], trained_models[name]["accuracy"])
        )
        best = trained_models[best_name]
        best["comparison"] = sorted(
            evaluations, key=lambda row: (row["macro_f1_percent"], row["accuracy_percent"]), reverse=True
        )
        model_key = secrets.token_hex(12)
        MODEL_STORE[model_key] = best
        session["model_state"] = model_key
        session.pop("last_prediction", None)
        logger.info("All classifiers trained; selected %s at %.2f%% accuracy", best_name, best["accuracy_percent"])
        flash(f"All three models were compared. Best macro-F1 model: {best_name}.", "success")
    except RequestEntityTooLarge:
        flash("The uploaded file exceeds the 10 MB limit.", "error")
    except (BadRequest, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.exception("Could not read training request or CSV")
        flash(f"Could not read the CSV file: {exc}", "error")
    except Exception as exc:
        logger.exception("Model comparison/training failed")
        flash(f"Training failed: {exc}", "error")
    return redirect(url_for("index"))


@app.post("/predict")
@login_required
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        flash("Enter some text to analyze.", "error")
        return redirect(url_for("index"))
    if len(text) > 5000:
        flash("Please keep the text within 5,000 characters.", "error")
        return redirect(url_for("index"))
    model_key = session.get("model_state")
    model_info = MODEL_STORE.get(model_key) if model_key else None
    if not model_info:
        flash("Please upload a labelled CSV and train the models first.", "error")
        return redirect(url_for("index"))
    try:
        predicted_label = model_info["pipeline"].predict([text])[0]
        session["last_prediction"] = {
            "text": text,
            "label": str(predicted_label),
            "prediction": str(predicted_label),
            "model_name": model_info["model_name"],
            "lexicon_score": None,
        }
        flash("Text analysis completed.", "success")
    except Exception as exc:
        logger.exception("Prediction failed")
        flash(f"Prediction failed: {exc}", "error")
    return redirect(url_for("index"))


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    flash("The uploaded file exceeds the 10 MB limit.", "error")
    return redirect(url_for("index")), 303


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    logger.exception("Bad request received: %s", error)
    flash("The server could not read that request. Refresh the page and try again.", "error")
    return redirect(url_for("index")), 303


@app.errorhandler(404)
def handle_not_found(error):
    return "Page not found.", 404


@app.errorhandler(500)
def handle_internal_error(error):
    logger.exception("Internal server error")
    return "An internal server error occurred. Check the terminal.", 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

