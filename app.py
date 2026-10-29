
import io
import os
import re
import logging
import secrets
import sqlite3
from functools import wraps

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from werkzeug.security import generate_password_hash, check_password_hash

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ==================================================
# FLASK CONFIGURATION
# ==================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "dev-key-change-this-before-deployment",
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE = os.path.join(app.root_path, "users.db")


# ==================================================
# DATABASE: USER REGISTRATION AND LOGIN
# ==================================================

def get_db():
    """Create and return a SQLite database connection."""
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create the users table if it does not already exist."""
    connection = get_db()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


# Create the database/table when the application starts.
init_db()


def login_required(view_function):
    """Require a signed-in user before accessing a route."""

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign in to access your dashboard.", "error")
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    return wrapped_view


# ==================================================
# MACHINE-LEARNING CONFIGURATION
# ==================================================

MODEL_NAMES = [
    "Decision Tree",
    "Naive Bayes",
    "SVM",
]

# Models remain in memory while the Flask process is running.
MODEL_STORE = {}


def build_pipeline(model_name):
    """Create a TF-IDF text-classification pipeline."""

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
    }

    if model_name not in classifiers:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose one of {', '.join(MODEL_NAMES)}."
        )

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    max_features=20000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifiers[model_name]),
        ]
    )


# ==================================================
# SIGN-IN ROUTE
# ==================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    # If already signed in, go straight to the dashboard.
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
                """
                SELECT id, username, email, password_hash
                FROM users
                WHERE username = ?
                """,
                (username,),
            ).fetchone()
        finally:
            connection.close()

        if user and check_password_hash(
            user["password_hash"], password
        ):
            # Start a fresh session after successful authentication.
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]

            flash("You have signed in successfully.", "success")
            return redirect(url_for("index"))

        flash("Invalid username or password.", "error")
        return redirect(url_for("login"))

    return render_template("auth.html", mode="login")


# ==================================================
# CREATE-ACCOUNT / REGISTRATION ROUTE
# ==================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # Accept either common confirmation-field name.
        confirm_password = request.form.get(
            "confirm_password",
            request.form.get("confirmPassword", ""),
        )

        # Validate username.
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", username):
            flash(
                "Username must be 3–40 characters and may contain "
                "letters, numbers, periods, underscores, and hyphens.",
                "error",
            )
            return redirect(url_for("register"))

        # Basic email validation.
        if (
            len(email) > 254
            or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email)
        ):
            flash("Enter a valid email address.", "error")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash(
                "Your password must contain at least 8 characters.",
                "error",
            )
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("The passwords do not match.", "error")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)

        connection = get_db()

        try:
            cursor = connection.execute(
                """
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
                """,
                (username, email, password_hash),
            )
            connection.commit()
            user_id = cursor.lastrowid

        except sqlite3.IntegrityError:
            connection.rollback()

            existing_user = connection.execute(
                """
                SELECT username, email
                FROM users
                WHERE username = ? COLLATE NOCASE
                   OR email = ? COLLATE NOCASE
                """,
                (username, email),
            ).fetchone()

            if existing_user:
                if existing_user["username"].lower() == username.lower():
                    flash(
                        "That username is already taken. Choose another.",
                        "error",
                    )
                else:
                    flash(
                        "An account with that email already exists.",
                        "error",
                    )
            else:
                flash(
                    "Could not create the account. Please try again.",
                    "error",
                )

            return redirect(url_for("register"))

        except sqlite3.Error:
            connection.rollback()
            logger.exception("Database registration error")
            flash(
                "Account creation failed. Please try again.",
                "error",
            )
            return redirect(url_for("register"))

        finally:
            connection.close()

        # Sign in the user automatically after successful registration.
        session.clear()
        session["user_id"] = user_id
        session["username"] = username

        flash("Your account was created successfully.", "success")
        return redirect(url_for("index"))

    return render_template("auth.html", mode="register")


# ==================================================
# SIGN-OUT ROUTE
# ==================================================

@app.post("/logout")
@login_required
def logout():
    session.clear()
    flash("You have signed out.", "success")
    return redirect(url_for("login"))


# ==================================================
# DASHBOARD
# ==================================================

@app.get("/")
@login_required
def index():
    model_key = session.get("model_state")
    trained_model = (
        MODEL_STORE.get(model_key) if model_key else None
    )

    # Copy the model information so request-specific results
    # do not alter the stored model.
    result = dict(trained_model) if trained_model else None

    last_prediction = session.pop("last_prediction", None)

    if result is not None:
        result["prediction"] = last_prediction
        result["lexicon_score"] = (
            last_prediction.get("lexicon_score")
            if last_prediction
            else None
        )

    return render_template(
        "index.html",
        username=session.get("username"),
        model_names=MODEL_NAMES,

        # These variables support the existing dashboard template.
        trained=bool(trained_model),
        result=result,

        # Keep these names available for templates that use them.
        trained_model=trained_model,
        metrics=trained_model,
        prediction=last_prediction,
    )


# ==================================================
# TRAIN A MODEL USING AN UPLOADED CSV
# ==================================================

@app.post("/train")
@login_required
def train():
    logger.info("TRAIN ROUTE ENTERED")

    try:
        uploaded = request.files.get("dataset")
        text_column = request.form.get(
            "text_column", ""
        ).strip()
        label_column = request.form.get(
            "label_column", ""
        ).strip()
        model_name = request.form.get(
            "model_name", "Decision Tree"
        ).strip()

    except BadRequest:
        logger.exception("Could not parse the training request")
        flash(
            "The upload request could not be read. "
            "Please select the CSV and try again.",
            "error",
        )
        return redirect(url_for("index"))

    logger.info(
        "Upload: %s | Text column: %s | Label column: %s | Model: %s",
        uploaded.filename if uploaded else None,
        text_column,
        label_column,
        model_name,
    )

    # Validate the uploaded file.
    if not uploaded or not uploaded.filename:
        flash("Choose a CSV file first.", "error")
        return redirect(url_for("index"))

    if not uploaded.filename.lower().endswith(".csv"):
        flash("Please upload a .csv file.", "error")
        return redirect(url_for("index"))

    # Read the CSV file.
    try:
        file_bytes = uploaded.read()

        if not file_bytes:
            flash("The uploaded CSV file is empty.", "error")
            return redirect(url_for("index"))

        try:
            frame = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="utf-8-sig",
            )
        except UnicodeDecodeError:
            frame = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin-1",
            )

    except Exception as exc:
        logger.exception("CSV reading failed")
        flash(f"Could not read the CSV file: {exc}", "error")
        return redirect(url_for("index"))

    # Clean column names and validate the data.
    frame.columns = frame.columns.astype(str).str.strip()

    if frame.empty:
        flash("The CSV file contains no data rows.", "error")
        return redirect(url_for("index"))

    if (
        not text_column
        or not label_column
        or text_column not in frame.columns
        or label_column not in frame.columns
    ):
        available_columns = ", ".join(frame.columns.tolist())
        flash(
            "Select valid text and label columns. "
            f"Available columns: {available_columns}",
            "error",
        )
        return redirect(url_for("index"))

    if text_column == label_column:
        flash(
            "The text column and label column must be different.",
            "error",
        )
        return redirect(url_for("index"))

    if model_name not in MODEL_NAMES:
        flash("Please select a supported classifier.", "error")
        return redirect(url_for("index"))

    # Keep the selected columns and remove invalid rows.
    data = frame[[text_column, label_column]].copy()
    data = data.dropna(subset=[text_column, label_column])

    data[text_column] = data[text_column].astype(str).str.strip()
    data[label_column] = data[label_column].astype(str).str.strip()

    data = data[
        (data[text_column].str.len() > 0)
        & (data[label_column].str.len() > 0)
    ]

    data = data.drop_duplicates(
        subset=[text_column, label_column]
    )

    class_counts = data[label_column].value_counts()

    if len(data) < 10:
        flash(
            "Provide at least 10 valid labelled rows.",
            "error",
        )
        return redirect(url_for("index"))

    if len(class_counts) < 2:
        flash(
            "The label column must contain at least two distinct classes.",
            "error",
        )
        return redirect(url_for("index"))

    # Use stratification only when every class has at least two examples.
    stratify_labels = (
        data[label_column]
        if class_counts.min() >= 2
        else None
    )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            data[text_column],
            data[label_column],
            test_size=0.2,
            random_state=42,
            stratify=stratify_labels,
        )

        pipeline = build_pipeline(model_name)

        logger.info(
            "Training %s on %s rows",
            model_name,
            len(x_train),
        )

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        accuracy = float(
            accuracy_score(y_test, predictions)
        )

        labels = sorted(data[label_column].unique().tolist())

        report = classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ).tolist()

        class_report = {}

        for label in labels:
            values = report.get(label, {})

            class_report[label] = {
                "precision": round(
                    float(values.get("precision", 0)), 3
                ),
                "recall": round(
                    float(values.get("recall", 0)), 3
                ),
                "f1": round(
                    float(values.get("f1-score", 0)), 3
                ),
                "support": int(
                    values.get("support", 0)
                ),
            }

        # Store the trained model and its evaluation metrics.
        model_key = secrets.token_hex(12)

        model_info = {
            "pipeline": pipeline,
            "model_name": model_name,
            "accuracy": round(accuracy, 4),
            "accuracy_percent": round(accuracy * 100, 2),
            "rows": int(len(data)),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "labels": labels,
            "confusion_matrix": matrix,
            "report": class_report,
            "text_column": text_column,
            "label_column": label_column,
            "prediction": None,
            "lexicon_score": None,
        }

        MODEL_STORE[model_key] = model_info
        session["model_state"] = model_key
        session.pop("last_prediction", None)

        logger.info(
            "Training completed. Accuracy: %.2f%%",
            accuracy * 100,
        )

        flash(
            "Model trained successfully. Metrics are based "
            "on a held-out test split.",
            "success",
        )

    except Exception as exc:
        logger.exception("Model training failed")
        flash(f"Training failed: {exc}", "error")

    return redirect(url_for("index"))


# ==================================================
# PREDICT THE CLASS OF SUBMITTED TEXT
# ==================================================

@app.post("/predict")
@login_required
def predict():
    text = request.form.get("text", "").strip()

    if not text:
        flash("Enter some text to analyze.", "error")
        return redirect(url_for("index"))

    if len(text) > 5000:
        flash(
            "Please keep the text within 5,000 characters.",
            "error",
        )
        return redirect(url_for("index"))

    model_key = session.get("model_state")
    model_info = MODEL_STORE.get(model_key) if model_key else None

    if not model_info:
        flash(
            "Please upload a labelled CSV and train a model first.",
            "error",
        )
        return redirect(url_for("index"))

    try:
        predicted_label = model_info["pipeline"].predict([text])[0]

        prediction_result = {
            "text": text,
            "label": str(predicted_label),
            "prediction": str(predicted_label),
            "model_name": model_info["model_name"],
            "lexicon_score": None,
        }

        session["last_prediction"] = prediction_result

        logger.info(
            "Prediction completed using %s",
            model_info["model_name"],
        )

        flash("Text analysis completed.", "success")

    except Exception as exc:
        logger.exception("Prediction failed")
        flash(f"Prediction failed: {exc}", "error")

    return redirect(url_for("index"))


# ==================================================
# ERROR HANDLERS
# ==================================================

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    flash(
        "The uploaded file exceeds the 10 MB limit.",
        "error",
    )
    return redirect(url_for("index")), 303


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    logger.exception("Bad request received: %s", error)
    flash(
        "The server could not read that request. "
        "Refresh the page and try again.",
        "error",
    )
    return redirect(url_for("index")), 303


@app.errorhandler(404)
def handle_not_found(error):
    return "Page not found.", 404


@app.errorhandler(500)
def handle_internal_error(error):
    logger.exception("Internal server error")
    return (
        "An internal server error occurred. Check the terminal.",
        500,
    )


# ==================================================
# RUN THE APPLICATION
# ==================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
