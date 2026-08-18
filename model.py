from collections import deque
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Constants ──────────────────────────────────────────────────────────────────
HOLIDAY_MONTHS      = {2, 8, 11, 12}
DAYS_PER_MONTH       = 30.44
SAFETY_BUFFER_SIGMA  = 1.5
FORECAST_HORIZON     = 3   # <-- how many months ahead to predict

# Base features always used
BASE_FEATURES = ["Month", "Year", "Is_Holiday", "Sales_Lag1", "Rolling_3"]

# Extra features added only when those columns exist in the CSV
PRICE_FEATURES = ["Selling_Price", "Discount_Pct", "Is_Promotion"]


def fmt_currency(value: float) -> str:
    """Format a number as ₹ with Indian comma system (e.g. ₹8,50,000)."""
    if value is None:
        return "—"
    if value >= 1_00_00_000:   # crore
        return f"₹{value/1_00_00_000:.2f} Cr"
    if value >= 1_00_000:      # lakh
        return f"₹{value/1_00_000:.1f} L"
    return f"₹{value:,.0f}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STOCK HEALTH CALCULATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_stock_health(current_stock, lead_time_days, avg_monthly_sales,
                            sales_std, future_pred, unit_price=None):
    """future_pred here is the NEXT MONTH (month-1) prediction — stock alerts
    are about the immediate near term, so we deliberately use the 1-month
    figure even though the model now also forecasts 3 months ahead."""
    avg_daily   = avg_monthly_sales / DAYS_PER_MONTH
    safety_buf  = round(SAFETY_BUFFER_SIGMA * sales_std, 1)
    reorder_pt  = round((avg_daily * lead_time_days) + safety_buf, 1)
    days_left   = round(current_stock / avg_daily, 1) if avg_daily > 0 else 999
    reorder_qty = max(0, round(future_pred + safety_buf - current_stock))
    coverage    = current_stock / reorder_pt if reorder_pt > 0 else 999

    if days_left <= lead_time_days:
        status = "CRITICAL"
    elif coverage < 1.0:
        status = "LOW"
    elif coverage < 2.0:
        status = "WARNING"
    else:
        status = "OK"

    urgency = {"CRITICAL": 0, "LOW": 1, "WARNING": 2, "OK": 3}[status]

    # Lost revenue if stockout (units short × unit price)
    lost_revenue = None
    if unit_price and status in ("CRITICAL", "LOW"):
        units_short = max(0, future_pred - current_stock)
        lost_revenue = round(units_short * unit_price)

    msgs = {
        "CRITICAL": f"Only {days_left} days of stock left — less than lead time of {lead_time_days} days. Order {reorder_qty} units IMMEDIATELY.",
        "LOW"     : f"Stock ({int(current_stock)}) is below reorder point ({reorder_pt}). Place order for {reorder_qty} units now.",
        "WARNING" : f"Stock is getting low. Reorder point is {reorder_pt}. Plan to order {reorder_qty} units soon.",
        "OK"      : f"Stock level is healthy. {days_left} days of supply remaining.",
    }

    return {
        "current_stock"  : int(current_stock),
        "lead_time_days" : int(lead_time_days),
        "safety_buffer"  : safety_buf,
        "reorder_point"  : reorder_pt,
        "days_of_stock"  : days_left,
        "reorder_qty"    : int(reorder_qty),
        "avg_daily_sales": round(avg_daily, 1),
        "status"         : status,
        "urgency"        : urgency,
        "status_msg"     : msgs[status],
        "lost_revenue"   : lost_revenue,
        "lost_revenue_fmt": fmt_currency(lost_revenue) if lost_revenue else None,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SINGLE-PRODUCT PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def forecast_one_product(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Detect which optional columns are present ─────────────────────────
    has_price = all(c in df.columns for c in ["Unit_Price", "Cost_Price", "Selling_Price",
                                               "Discount_Pct", "Is_Promotion"])

    # ── Feature engineering ───────────────────────────────────────────────
    df["Month"]      = df["Date"].dt.month
    df["Year"]       = df["Date"].dt.year
    df["Sales_Lag1"] = df["Sales"].shift(1).fillna(df["Sales"].mean())
    df["Rolling_3"]  = df["Sales"].rolling(3, min_periods=1).mean().shift(1).fillna(df["Sales"].mean())

    feature_names = BASE_FEATURES.copy()
    if has_price:
        # Selling_Price = Unit_Price × (1 − Discount/100)  — actual paid price
        # Already in dataset but recalculate to be safe
        df["Selling_Price"] = df["Unit_Price"] * (1 - df["Discount_Pct"] / 100)
        feature_names += PRICE_FEATURES

    X = df[feature_names].values
    y = df["Sales"].values

    # ── Train ─────────────────────────────────────────────────────────────
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    # ── Metrics ───────────────────────────────────────────────────────────
    mae      = round(mean_absolute_error(y, preds), 1)
    rmse     = round(mean_squared_error(y, preds) ** 0.5, 1)
    mape     = float(np.mean(np.abs((y - preds) / y)) * 100)
    accuracy = round(max(0, 100 - mape), 1)

    importance = {
        name: round(float(imp) * 100, 1)
        for name, imp in zip(feature_names, model.feature_importances_)
    }

    # ── Promo vs non-promo analysis ───────────────────────────────────────
    promo_analysis = None
    if has_price and "Is_Promotion" in df.columns:
        promo_df    = df[df["Is_Promotion"] == 1]["Sales"]
        no_promo_df = df[df["Is_Promotion"] == 0]["Sales"]
        if len(promo_df) > 0 and len(no_promo_df) > 0:
            promo_avg    = round(float(promo_df.mean()), 1)
            no_promo_avg = round(float(no_promo_df.mean()), 1)
            lift_pct     = round((promo_avg - no_promo_avg) / no_promo_avg * 100, 1)
            promo_analysis = {
                "promo_avg"   : promo_avg,
                "no_promo_avg": no_promo_avg,
                "lift_pct"    : lift_pct,
                "promo_months": int(len(promo_df)),
            }

    # ── Multi-month-ahead forecast (recursive / walk-forward) ─────────────
    # We predict FORECAST_HORIZON months ahead. Since Sales_Lag1 / Rolling_3
    # depend on recent sales, each step's prediction feeds into the next
    # step's lag/rolling features.
    last_date      = df["Date"].max()
    unit_price_val = float(df["Unit_Price"].iloc[-1]) if has_price else None

    # Seed the rolling window with the last up-to-3 *actual* sales values.
    recent_sales = deque(df["Sales"].iloc[-3:].tolist(), maxlen=3)

    future_forecast = []   # list of per-month forecast dicts
    cur_date = last_date
    for step in range(1, FORECAST_HORIZON + 1):
        cur_date    = cur_date + relativedelta(months=1)
        is_holiday  = 1 if cur_date.month in HOLIDAY_MONTHS else 0
        lag1        = float(recent_sales[-1])
        rolling3    = float(np.mean(recent_sales))

        future_row = [cur_date.month, cur_date.year, is_holiday, lag1, rolling3]

        if has_price:
            # Assume no promotion in future months; selling at full unit price.
            future_row += [unit_price_val, 0, 0]

        future_X   = np.array([future_row])
        step_pred  = round(float(model.predict(future_X)[0]), 1)

        recent_sales.append(step_pred)  # feed prediction back in for next step

        month_data = {
            "step"        : step,
            "date"        : cur_date.strftime("%Y-%m"),
            "label"       : cur_date.strftime("%b %Y"),
            "pred"        : step_pred,
        }

        if has_price:
            cost_price_val = float(df["Cost_Price"].iloc[-1])
            rev    = round(step_pred * unit_price_val)
            margin = round(step_pred * (unit_price_val - cost_price_val))
            month_data.update({
                "revenue"     : rev,
                "margin"      : margin,
                "revenue_fmt" : fmt_currency(rev),
                "margin_fmt"  : fmt_currency(margin),
            })

        future_forecast.append(month_data)

    # Backward-compatible single-month fields (month 1 of the horizon)
    next_dt      = future_forecast[0]
    future_pred  = next_dt["pred"]

    # ── Revenue & margin calculations ─────────────────────────────────────
    revenue_data = None
    if has_price:
        unit_price   = unit_price_val
        cost_price   = float(df["Cost_Price"].iloc[-1])
        future_rev   = future_forecast[0]["revenue"]
        future_margin= future_forecast[0]["margin"]
        margin_pct   = round((unit_price - cost_price) / unit_price * 100, 1)
        total_rev    = round(float((df["Sales"] * df["Selling_Price"]).sum()))
        avg_monthly_rev = round(total_rev / len(df))

        future_revenue_3m = sum(m["revenue"] for m in future_forecast)
        future_margin_3m  = sum(m["margin"] for m in future_forecast)

        revenue_data = {
            "unit_price"          : unit_price,
            "cost_price"          : cost_price,
            "margin_pct"          : margin_pct,
            "future_revenue"      : future_rev,
            "future_margin"       : future_margin,
            "total_revenue"       : total_rev,
            "avg_monthly_rev"     : avg_monthly_rev,
            "future_rev_fmt"      : fmt_currency(future_rev),
            "future_margin_fmt"   : fmt_currency(future_margin),
            "total_rev_fmt"       : fmt_currency(total_rev),
            "avg_monthly_fmt"     : fmt_currency(avg_monthly_rev),
            "unit_price_fmt"      : fmt_currency(unit_price),
            "future_revenue_3m"   : future_revenue_3m,
            "future_margin_3m"    : future_margin_3m,
            "future_revenue_3m_fmt": fmt_currency(future_revenue_3m),
            "future_margin_3m_fmt" : fmt_currency(future_margin_3m),
        }

    # ── Stock health (based on the immediate next-month forecast) ─────────
    avg_sales = round(float(df["Sales"].mean()), 1)
    sales_std = round(float(df["Sales"].std()), 1)

    if "Current_Stock" in df.columns and "Lead_Time_Days" in df.columns:
        stock_health = calculate_stock_health(
            float(df["Current_Stock"].iloc[-1]),
            float(df["Lead_Time_Days"].iloc[-1]),
            avg_sales, sales_std, future_pred, unit_price_val
        )
    else:
        stock_health = {
            "current_stock": None, "lead_time_days": None, "safety_buffer": None,
            "reorder_point": None, "days_of_stock": None,  "reorder_qty": None,
            "avg_daily_sales": None, "lost_revenue": None, "lost_revenue_fmt": None,
            "status": "LOW" if future_pred < 150 else "OK",
            "urgency": 1 if future_pred < 150 else 3,
            "status_msg": "Add Current_Stock and Lead_Time_Days for real alerts.",
        }

    future_pred_3m_total = round(sum(m["pred"] for m in future_forecast), 1)

    return {
        "dates"               : df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "actuals"             : df["Sales"].tolist(),
        "predictions"         : [round(p, 1) for p in preds.tolist()],
        # Backward-compatible 1-month-ahead fields (month 1 of the horizon)
        "future_date"         : next_dt["date"],
        "future_label"        : next_dt["label"],
        "future_pred"         : future_pred,
        # New: full 3-month forecast horizon
        "future_forecast"     : future_forecast,
        "future_pred_3m_total": future_pred_3m_total,
        "forecast_horizon"    : FORECAST_HORIZON,
        "mae"                 : mae,
        "rmse"                : rmse,
        "accuracy"            : accuracy,
        "importance"          : importance,
        "total_sales"         : int(df["Sales"].sum()),
        "avg_sales"           : avg_sales,
        "sales_std"           : sales_std,
        "stock"               : stock_health,
        "revenue"             : revenue_data,
        "promo_analysis"      : promo_analysis,
        "has_price"           : has_price,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MULTI-PRODUCT ORCHESTRATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_multi_product_forecast(df: pd.DataFrame) -> dict:
    has_sku      = "SKU"      in df.columns
    has_category = "Category" in df.columns

    if not has_sku:
        result = forecast_one_product(df)
        result["sku"] = "ALL"
        return {
            "sku_results"  : {"ALL": result},
            "sku_list"     : ["ALL"],
            "categories"   : {},
            "top_sku"      : "ALL",
            "bottom_sku"   : "ALL",
            "alerts"       : [result] if result["stock"]["status"] != "OK" else [],
            "total_revenue": result["revenue"]["total_revenue"] if result["revenue"] else None,
            "total_rev_fmt": result["revenue"]["total_rev_fmt"] if result["revenue"] else None,
            "is_multi"     : False,
        }

    sku_results = {}
    for sku, group in df.groupby("SKU"):
        res        = forecast_one_product(group.copy())
        res["sku"] = sku
        if has_category:
            res["category"] = group["Category"].iloc[0]
        sku_results[sku] = res

    sku_list = sorted(sku_results, key=lambda s: sku_results[s]["total_sales"], reverse=True)

    # Category rollups — units AND revenue
    categories      = {}
    cat_revenue     = {}
    if has_category:
        for sku, res in sku_results.items():
            cat = res.get("category", "Other")
            categories[cat]  = categories.get(cat, 0) + res["total_sales"]
            if res["revenue"]:
                cat_revenue[cat] = cat_revenue.get(cat, 0) + res["revenue"]["total_revenue"]

    top_sku    = max(sku_results, key=lambda s: sku_results[s]["avg_sales"])
    bottom_sku = min(sku_results, key=lambda s: sku_results[s]["avg_sales"])

    alerts = sorted(
        [r for r in sku_results.values() if r["stock"]["status"] != "OK"],
        key=lambda r: r["stock"]["urgency"]
    )

    # Portfolio-level revenue totals
    total_rev = sum(
        r["revenue"]["total_revenue"] for r in sku_results.values() if r["revenue"]
    )
    total_forecast_rev = sum(
        r["revenue"]["future_revenue"] for r in sku_results.values() if r["revenue"]
    )
    total_lost_rev = sum(
        r["stock"]["lost_revenue"] for r in sku_results.values()
        if r["stock"].get("lost_revenue")
    )

    return {
        "sku_results"       : sku_results,
        "sku_list"          : sku_list,
        "categories"        : categories,
        "cat_revenue"       : cat_revenue,
        "top_sku"           : top_sku,
        "bottom_sku"        : bottom_sku,
        "alerts"            : alerts,
        "total_revenue"     : total_rev,
        "total_rev_fmt"     : fmt_currency(total_rev),
        "total_forecast_rev": total_forecast_rev,
        "total_forecast_fmt": fmt_currency(total_forecast_rev),
        "total_lost_rev"    : total_lost_rev,
        "total_lost_fmt"    : fmt_currency(total_lost_rev) if total_lost_rev else None,
        "is_multi"          : True,
    }
