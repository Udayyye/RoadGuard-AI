from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import sqlite3
import uuid
import cv2


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

DATABASE = BASE_DIR / "road_damage.db"

MODEL_PATH = BASE_DIR / "models" / "roadguard_best.pt"


# Create required folders
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully.")


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():

    connection = sqlite3.connect(str(DATABASE))

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            date_time TEXT,

            image_name TEXT,

            damage_type TEXT,

            damage_count INTEGER,

            confidence REAL,

            severity TEXT

        )
    """)

    connection.commit()

    connection.close()


init_db()


# ============================================================
# SAVE DETECTION HISTORY
# ============================================================

def save_detection(
    image_name,
    damage_type,
    damage_count,
    confidence,
    severity
):

    connection = sqlite3.connect(str(DATABASE))

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO detections
        (
            date_time,
            image_name,
            damage_type,
            damage_count,
            confidence,
            severity
        )

        VALUES (?, ?, ?, ?, ?, ?)

    """, (

        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        image_name,

        damage_type,

        damage_count,

        confidence,

        severity

    ))

    connection.commit()

    connection.close()


# ============================================================
# GET DETECTION HISTORY
# ============================================================

def get_history():

    connection = sqlite3.connect(str(DATABASE))

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *

        FROM detections

        ORDER BY id DESC

        LIMIT 20
    """)

    history = cursor.fetchall()

    connection.close()

    return history


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

def get_dashboard_stats():

    connection = sqlite3.connect(str(DATABASE))

    cursor = connection.cursor()


    # Total scans

    cursor.execute("""
        SELECT COUNT(*)

        FROM detections
    """)

    total_scans = cursor.fetchone()[0]


    # Total damage

    cursor.execute("""
        SELECT COALESCE(
            SUM(damage_count),
            0
        )

        FROM detections
    """)

    total_damage = cursor.fetchone()[0]


    # Total potholes

    cursor.execute("""
        SELECT COALESCE(
            SUM(damage_count),
            0
        )

        FROM detections

        WHERE LOWER(damage_type)
        LIKE '%pothole%'
    """)

    total_potholes = cursor.fetchone()[0]


    # Average confidence

    cursor.execute("""
        SELECT COALESCE(
            AVG(confidence),
            0
        )

        FROM detections
    """)

    average_confidence = cursor.fetchone()[0]


    connection.close()


    return {

        "total_scans": total_scans,

        "total_damage": total_damage,

        "total_potholes": total_potholes,

        "average_confidence":
            round(
                average_confidence,
                2
            )

    }


# ============================================================
# MAIN PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result_image = None

    detections = []

    summary = None

    error = None

    upload_filename = None


    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    if request.method == "POST":

        if "image" not in request.files:

            error = "No image selected."

            return render_template(
                "index.html",

                error=error,

                history=get_history(),

                dashboard=get_dashboard_stats()
            )


        file = request.files["image"]


        if file.filename == "":

            error = "Please select an image."

            return render_template(
                "index.html",

                error=error,

                history=get_history(),

                dashboard=get_dashboard_stats()
            )


        # ====================================================
        # CREATE UNIQUE FILE NAME
        # ====================================================

        extension = Path(
            file.filename
        ).suffix.lower()


        # Allow only common image formats

        allowed_extensions = {

            ".jpg",
            ".jpeg",
            ".png",
            ".webp"

        }


        if extension not in allowed_extensions:

            error = (
                "Please upload JPG, JPEG, PNG "
                "or WEBP image."
            )

            return render_template(
                "index.html",

                error=error,

                history=get_history(),

                dashboard=get_dashboard_stats()
            )


        filename = (
            f"{uuid.uuid4().hex}"
            f"{extension}"
        )


        # ====================================================
        # ABSOLUTE FILE PATHS
        # ====================================================

        upload_path = (
            UPLOAD_FOLDER / filename
        ).resolve()


        result_path = (
            RESULT_FOLDER
            / f"result_{filename}"
        ).resolve()


        # ====================================================
        # SAVE UPLOADED IMAGE
        # ====================================================

        file.save(str(upload_path))

        upload_filename = filename


        # Debug information

        print(
            "Uploaded file:",
            upload_path
        )

        print(
            "File exists:",
            upload_path.exists()
        )


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if not upload_path.exists():

            error = (
                "Uploaded file could not "
                "be saved."
            )

            return render_template(
                "index.html",

                error=error,

                history=get_history(),

                dashboard=get_dashboard_stats()
            )


        # ====================================================
        # YOLO DETECTION
        # ====================================================

        try:

            results = model.predict(

                source=str(
                    upload_path
                ),

                conf=0.50,

                iou=0.45,

                imgsz=416,

                verbose=True

            )

        except Exception as e:

            print(
                "YOLO ERROR:",
                str(e)
            )

            error = (
                "AI detection failed. "
                "Check the terminal for details."
            )

            return render_template(

                "index.html",

                error=error,

                history=get_history(),

                dashboard=get_dashboard_stats()
            )


        result = results[0]


        # ====================================================
        # READ IMAGE
        # ====================================================

        image = cv2.imread(
            str(upload_path)
        )


        if image is None:

            error = (
                "The uploaded image could "
                "not be read."
            )

            return render_template(

                "index.html",

                error=error,

                history=get_history(),

                dashboard=get_dashboard_stats()
            )


        image_height, image_width = (
            image.shape[:2]
        )

        image_area = (
            image_width *
            image_height
        )


        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )


            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )


            box_width = max(
                0,
                x2 - x1
            )

            box_height = max(
                0,
                y2 - y1
            )


            box_area = (
                box_width *
                box_height
            )


            area_ratio = (
                box_area /
                image_area
            )


            detections.append({

                "class":
                    model.names[
                        class_id
                    ],

                "confidence":
                    round(
                        confidence * 100,
                        2
                    ),

                "area":
                    round(
                        area_ratio * 100,
                        2
                    )

            })


        # ====================================================
        # SUMMARY
        # ====================================================

        total_damage = len(
            detections
        )


        potholes = sum(

            1

            for d in detections

            if d["class"].lower()
            == "pothole"

        )


        cracks = sum(

            1

            for d in detections

            if "crack"
            in d["class"].lower()

        )


        highest_confidence = max(

            [
                d["confidence"]
                for d in detections
            ],

            default=0

        )


        # ====================================================
        # SEVERITY
        # ====================================================

        if total_damage >= 3:

            severity = "HIGH"

        elif total_damage == 2:

            severity = "MEDIUM"

        elif total_damage == 1:

            severity = "LOW"

        else:

            severity = "NO DAMAGE"


        # ====================================================
        # DAMAGE TYPE
        # ====================================================

        damage_types = sorted(

            set(
                d["class"]
                for d in detections
            )

        )


        if damage_types:

            damage_type = ", ".join(
                damage_types
            )

        else:

            damage_type = "No damage"


        # ====================================================
        # SAVE DATABASE RECORD
        # ====================================================

        save_detection(

            image_name=filename,

            damage_type=damage_type,

            damage_count=total_damage,

            confidence=highest_confidence,

            severity=severity

        )


        # ====================================================
        # SUMMARY OBJECT
        # ====================================================

        summary = {

            "total":
                total_damage,

            "potholes":
                potholes,

            "cracks":
                cracks,

            "confidence":
                highest_confidence,

            "severity":
                severity

        }


        # ====================================================
        # CREATE ANNOTATED IMAGE
        # ====================================================

        annotated_image = result.plot()


        cv2.imwrite(

            str(result_path),

            annotated_image

        )


        # ====================================================
        # URL FOR WEBSITE
        # ====================================================

        result_image = (

            "/static/results/"
            f"result_{filename}"

        )


    # ========================================================
    # LOAD HISTORY + DASHBOARD
    # ========================================================

    history = get_history()

    dashboard = get_dashboard_stats()


    # ========================================================
    # RENDER WEBSITE
    # ========================================================

    return render_template(

        "index.html",

        result_image=result_image,

        detections=detections,

        summary=summary,

        history=history,

        dashboard=dashboard,

        upload_filename=upload_filename,

        error=error

    )


# ============================================================
# PDF INSPECTION REPORT
# ============================================================

@app.route(
    "/report/<filename>"
)
def generate_report(filename):

    upload_path = (
        UPLOAD_FOLDER / filename
    ).resolve()


    result_path = (
        RESULT_FOLDER
        / f"result_{filename}"
    ).resolve()


    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if not upload_path.exists():

        return (
            "Original image not found.",
            404
        )


    # ========================================================
    # GET DATABASE RECORD
    # ========================================================

    connection = sqlite3.connect(
        str(DATABASE)
    )

    connection.row_factory = (
        sqlite3.Row
    )

    cursor = connection.cursor()


    cursor.execute("""

        SELECT *

        FROM detections

        WHERE image_name = ?

        ORDER BY id DESC

        LIMIT 1

    """, (filename,))


    record = cursor.fetchone()


    connection.close()


    if record is None:

        return (
            "Detection record not found.",
            404
        )


    # ========================================================
    # RUN YOLO FOR REPORT
    # ========================================================

    results = model.predict(

        source=str(
            upload_path
        ),

        conf=0.50,

        iou=0.45,

        imgsz=416,

        verbose=False

    )


    result = results[0]


    detections_for_report = []


    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )


        detections_for_report.append({

            "class":
                model.names[
                    class_id
                ],

            "confidence":
                round(
                    confidence * 100,
                    2
                )

        })


    # ========================================================
    # PDF PATH
    # ========================================================

    pdf_filename = (

        "RoadGuard_Report_"

        f"{Path(filename).stem}.pdf"

    )


    pdf_path = (
        RESULT_FOLDER
        / pdf_filename
    ).resolve()


    # ========================================================
    # CREATE PDF
    # ========================================================

    pdf = canvas.Canvas(

        str(pdf_path),

        pagesize=A4

    )


    width, height = A4


    # ========================================================
    # TITLE
    # ========================================================

    pdf.setFont(
        "Helvetica-Bold",
        22
    )


    pdf.drawCentredString(

        width / 2,

        height - 50,

        "ROADGUARD AI"

    )


    pdf.setFont(
        "Helvetica-Bold",
        16
    )


    pdf.drawCentredString(

        width / 2,

        height - 75,

        "AI-Based Road Damage Detection"

    )


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    y = height - 120


    pdf.setFont(
        "Helvetica",
        11
    )


    pdf.drawString(

        50,

        y,

        f"Inspection Date: "
        f"{record['date_time']}"

    )


    y -= 25


    pdf.drawString(

        50,

        y,

        f"Image: "
        f"{record['image_name']}"

    )


    # ========================================================
    # SUMMARY
    # ========================================================

    y -= 45


    pdf.setFont(
        "Helvetica-Bold",
        15
    )


    pdf.drawString(

        50,

        y,

        "Detection Summary"

    )


    y -= 25


    pdf.setFont(
        "Helvetica",
        11
    )


    pdf.drawString(

        60,

        y,

        f"Total Damage: "
        f"{record['damage_count']}"

    )


    y -= 20


    pdf.drawString(

        60,

        y,

        f"Damage Type: "
        f"{record['damage_type']}"

    )


    y -= 20


    pdf.drawString(

        60,

        y,

        f"Highest Confidence: "
        f"{record['confidence']:.2f}%"

    )


    y -= 20


    pdf.drawString(

        60,

        y,

        f"Estimated Severity: "
        f"{record['severity']}"

    )


    # ========================================================
    # INDIVIDUAL DETECTIONS
    # ========================================================

    y -= 40


    pdf.setFont(
        "Helvetica-Bold",
        15
    )


    pdf.drawString(

        50,

        y,

        "Detected Damage"

    )


    y -= 25


    pdf.setFont(
        "Helvetica",
        11
    )


    if detections_for_report:

        for index, detection in enumerate(

            detections_for_report,

            start=1

        ):

            pdf.drawString(

                60,

                y,

                f"{index}. "
                f"{detection['class']} - "
                f"{detection['confidence']:.2f}% "
                f"confidence"

            )


            y -= 20


            # Start a new page if needed

            if y < 100:

                pdf.showPage()

                y = height - 50

                pdf.setFont(
                    "Helvetica",
                    11
                )


    else:

        pdf.drawString(

            60,

            y,

            "No road damage detected."

        )


        y -= 20


    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    if result_path.exists():

        y -= 20


        if y < 420:

            pdf.showPage()

            y = height - 50


        pdf.setFont(

            "Helvetica-Bold",

            15

        )


        pdf.drawString(

            50,

            y,

            "Annotated Detection Image"

        )


        y -= 15


        try:

            img = ImageReader(
                str(result_path)
            )


            pdf.drawImage(

                img,

                50,

                y - 350,

                width=500,

                height=350,

                preserveAspectRatio=True,

                anchor="c"

            )

        except Exception as e:

            print(
                "PDF image error:",
                e
            )


    # ========================================================
    # FOOTER
    # ========================================================

    pdf.setFont(
        "Helvetica",
        9
    )


    pdf.drawString(

        50,

        30,

        "Built with Python • "
        "Flask • YOLOv8 • SQLite"

    )


    pdf.save()


    # ========================================================
    # SEND PDF TO USER
    # ========================================================

    return send_file(

        str(pdf_path),

        as_attachment=True,

        download_name=pdf_filename,

        mimetype="application/pdf"

    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )