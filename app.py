"""
app.py — StockSense v7 (3-Month Forecast Horizon)
====================================================
What's new in v7:
  ✅ Model training & forecasting logic moved to model.py
  ✅ Forecast horizon extended from 1 month → 3 months ahead
     (see model.py's FORECAST_HORIZON / forecast_one_product for details)
  ✅ app.py is now just the Flask routes / upload handling — no ML code here
"""

import os
import json

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

from model import run_multi_product_forecast
from preprocessing import clean_data

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "inventory_forecast_secret"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "csv_file" not in request.files or request.files["csv_file"].filename == "":
        flash("Please select a CSV file before uploading.")
        return redirect(url_for("index"))

    file = request.files["csv_file"]
    if not file.filename.lower().endswith(".csv"):
        flash("Only .csv files are accepted.")
        return redirect(url_for("index"))

    save_path = os.path.join(UPLOAD_FOLDER, "latest_upload.csv")
    file.save(save_path)

    try:
        df = pd.read_csv(save_path)
        required = {"Date", "Sales", "Is_Holiday"}
        if not required.issubset(df.columns):
            flash(f"CSV must have: {required}. Found: {set(df.columns)}")
            return redirect(url_for("index"))

        # ── Clean missing values (simple median/mode fill) before forecasting ──
        df, clean_report = clean_data(df)
        if clean_report["missing_before"] > 0:
            flash(
                f"Note: cleaned {clean_report['missing_before']} missing value(s) "
                f"in the uploaded file (filled with column median/mode)."
            )

        results = run_multi_product_forecast(df)
    except Exception as e:
        flash(f"Error processing file: {e}")
        return redirect(url_for("index"))

    return render_template(
        "dashboard.html",
        sku_results_json    = json.dumps(results["sku_results"]),
        sku_list             = results["sku_list"],
        sku_results          = results["sku_results"],
        categories_json      = json.dumps(results["categories"]),
        cat_revenue_json     = json.dumps(results.get("cat_revenue", {})),
        top_sku              = results["top_sku"],
        bottom_sku           = results["bottom_sku"],
        alerts               = results["alerts"],
        total_rev_fmt        = results.get("total_rev_fmt"),
        total_forecast_fmt   = results.get("total_forecast_fmt"),
        total_lost_fmt       = results.get("total_lost_fmt"),
        is_multi             = results["is_multi"],
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
