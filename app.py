"""
Investigate Hotel Business using Data Visualization
Streamlit app that follows the project brief stage by stage:

  Stage 0  Problem statement
  Stage 1  Data preprocessing (overview + assessment/cleaning)
  Stage 2  Data analysis (hotel type & seasonality, stay duration, lead time)
  Stage 3  Summary and recommendations

Run:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Hotel Business Analysis", page_icon="🏨", layout="wide")
sns.set_theme(style="whitegrid", context="notebook")

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
COLORS = {"City Hotel": "#1f77b4", "Resort Hotel": "#ff7f0e"}
DEFAULT_FILE = "hotel_bookings_cleaned.csv"

COLUMN_GUIDE = pd.DataFrame(
    [
        ("hotel", "City Hotel or Resort Hotel", "Q1 - hotel type"),
        ("is_canceled", "1 = booking cancelled, 0 = guest stayed", "Q2, Q3 - the outcome we study"),
        ("lead_time", "Days between booking date and arrival date", "Q3 - lead time"),
        ("arrival_date_month / year", "When the guest was due to arrive", "Q1 - seasonality"),
        ("stays_in_weekend_nights", "Weekend nights (Sat/Sun) booked", "Q2 - stay duration"),
        ("stays_in_weekdays_nights", "Week nights (Mon-Fri) booked", "Q2 - stay duration"),
        ("adults / children / babies", "Number of guests", "Cleaning - zero-guest rows"),
        ("meal", "Meal plan booked", "Cleaning - 'Undefined' values"),
        ("agent / company", "ID of travel agent / company that booked", "Cleaning - missing values"),
        ("city", "Guest's city of origin", "Cleaning - missing values"),
        ("adr", "Average daily rate (price per night)", "Cleaning - negative / extreme values"),
        ("deposit_type, customer_type, market_segment", "Booking context", "Supporting detail"),
    ],
    columns=["Column", "Meaning", "Why it matters"],
)


# ----------------------------------------------------------------------------
# Data loading and cleaning
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(content))


def load_raw() -> pd.DataFrame:
    st.sidebar.markdown("### Data")
    upload = st.sidebar.file_uploader("Use a different CSV (optional)", type="csv")
    if upload is not None:
        return read_csv_bytes(upload.getvalue())
    try:
        here = Path(__file__).resolve().parent
    except NameError:  # running inside a notebook: no __file__
        here = Path.cwd()
    for p in [here / DEFAULT_FILE, Path.cwd() / DEFAULT_FILE,
              Path("/mnt/user-data/uploads") / DEFAULT_FILE]:
        if p.exists():
            return read_csv_path(str(p))
    st.error(f"Could not find `{DEFAULT_FILE}`. Put it next to app.py or upload it in the sidebar.")
    st.stop()


@st.cache_data(show_spinner=False)
def clean_data(raw: pd.DataFrame):
    """Apply every cleaning decision. Returns (clean_df, step_log, counts)."""
    d = raw.copy()
    c = {"rows_raw": len(d)}
    c["missing_raw"] = raw.isna().sum()

    # 1. duplicates
    c["duplicates"] = int(d.duplicated().sum())
    d = d.drop_duplicates()

    # 2. missing values
    for col, fill in [("children", 0), ("agent", 0), ("company", 0), ("city", "Unknown")]:
        if col in d.columns:
            d[col] = d[col].fillna(fill)
    for col in ["children", "agent", "company"]:
        if col in d.columns:
            d[col] = d[col].astype(int)

    # 3. inconsistent / unclear categories
    c["meal_raw"] = raw["meal"].value_counts()
    c["segment_raw"] = raw["market_segment"].value_counts()
    c["channel_raw"] = raw["distribution_channel"].value_counts()
    d["meal"] = d["meal"].replace({"Undefined": "No Meal", "SC": "No Meal"})
    d["market_segment"] = d["market_segment"].replace({"Undefined": "Other"})
    d["distribution_channel"] = d["distribution_channel"].replace({"Undefined": "Other"})
    if "customer_type" in d.columns:
        d["customer_type"] = d["customer_type"].replace({"Bussiness": "Business"})

    # 4. anomalies
    d["total_guests"] = d["adults"] + d["children"] + d["babies"]
    d["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_weekdays_nights"]
    c["adr_negative"] = int((d["adr"] < 0).sum())
    c["adr_zero"] = int((d["adr"] == 0).sum())
    c["adr_extreme"] = int((d["adr"] > 1000).sum())
    c["zero_guests"] = int((d["total_guests"] == 0).sum())
    c["zero_nights"] = int((d["total_nights"] == 0).sum())
    c["big_party"] = int((d["adults"] >= 10).sum())
    d = d[(d["adr"] >= 0) & (d["adr"] <= 1000) & (d["total_guests"] > 0)].copy()

    # helper columns
    d["arrival_date_month"] = pd.Categorical(d["arrival_date_month"], MONTHS, ordered=True)
    d["arrival_date"] = pd.to_datetime(
        dict(year=d["arrival_date_year"],
             month=d["arrival_date_month"].map(MONTH_NUM).astype(float),
             day=d["arrival_date_day_of_month"]),
        errors="coerce",
    )
    d["status"] = np.where(d["is_canceled"] == 1, "Cancelled", "Not cancelled")
    c["rows_clean"] = len(d)
    return d.reset_index(drop=True), c


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def show(fig, caption=None):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    if caption:
        st.caption("📝 " + caption)


def binned_rate(d, col, edges):
    """Cancellation rate (%) and bookings count for each bin of `col`, per hotel."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            labels.append(f"{int(lo)}+")
        elif hi - lo == 1:
            labels.append(f"{int(lo)}")
        else:
            labels.append(f"{int(lo)}-{int(hi) - 1}")
    tmp = d.assign(bin=pd.cut(d[col], bins=edges, labels=labels, right=False))
    out = (tmp.groupby(["hotel", "bin"], observed=True)["is_canceled"]
              .agg(rate="mean", n="size").reset_index())
    out["rate"] *= 100
    return out, labels


def rate_line_chart(out, labels, title, xlabel, min_n, rotate=0):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pos = {l: i for i, l in enumerate(labels)}
    for hotel, g in out.groupby("hotel"):
        g = g[g["n"] >= min_n].copy()
        g["x"] = g["bin"].astype(str).map(pos)
        g = g.sort_values("x")
        ax.plot(g["x"], g["rate"], marker="o", lw=2, color=COLORS[hotel], label=hotel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Hotel type")
    fig.tight_layout()
    return fig


def extreme_bins(out, min_n):
    res = {}
    for hotel, g in out.groupby("hotel"):
        g = g[g["n"] >= min_n]
        res[hotel] = (g.loc[g["rate"].idxmin()], g.loc[g["rate"].idxmax()])
    return res


def corr_label(r):
    if r > 0.03:
        return "positive"
    if r < -0.03:
        return "negative"
    return "essentially flat"


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
st.sidebar.title("🏨 Hotel Business Analysis")
raw = load_raw()
df, info = clean_data(raw)

PAGES = [
    "Problem Statement",
    "Data Overview",
    "Data Assessment & Cleaning",
    "Hotel Type & Monthly Bookings",
    "Stay Duration vs Cancellation",
    "Lead Time vs Cancellation",
    "Summary & Recommendations",
]
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.markdown("---")
st.sidebar.metric("Clean bookings analysed", f"{len(df):,}")
st.sidebar.metric("Overall cancellation rate", f"{df['is_canceled'].mean() * 100:.1f}%")

# Pre-computed facts used on several pages -----------------------------------
share = df["hotel"].value_counts(normalize=True) * 100
top_hotel, low_hotel = share.idxmax(), share.idxmin()
cancel_by_hotel = df.groupby("hotel")["is_canceled"].mean() * 100
worse, better = cancel_by_hotel.idxmax(), cancel_by_hotel.idxmin()
stay_df = df[df["total_nights"] > 0]  # stays of zero nights are excluded from stay analysis

per_year = (df.groupby(["hotel", "arrival_date_month", "arrival_date_year"], observed=True)
              .size().rename("bookings").reset_index())
per_year = per_year[per_year["bookings"] > 0]
monthly_avg = (per_year.groupby(["hotel", "arrival_date_month"], observed=True)["bookings"]
                       .mean().reset_index())
monthly_total = (df.groupby(["hotel", "arrival_date_month"], observed=True)
                   .size().rename("bookings").reset_index())
all_month_avg = monthly_avg.groupby("arrival_date_month", observed=True)["bookings"].sum()
all_month_avg = all_month_avg[all_month_avg > 0]
busiest_m, quietest_m = all_month_avg.idxmax(), all_month_avg.idxmin()
top3 = list(all_month_avg.sort_values(ascending=False).index[:3])
bottom3 = list(all_month_avg.sort_values().index[:3])


# ============================================================================
# STAGE 0
# ============================================================================
if page == PAGES[0]:
    st.title("Investigate Hotel Business using Data Visualization")
    st.markdown("**Understanding booking & cancellation behaviour** · Stage 0 - Problem Statement")

    st.subheader("1. Why does understanding booking behaviour matter?")
    st.markdown(
        """
A hotel sells a perishable product: a room that stays empty tonight can never be sold again.
Knowing **when guests book, how long they stay and how often they cancel** lets the hotel
decide in advance rather than react afterwards. Better understanding can improve:

- **Pricing & revenue management** - raise rates in busy months, run offers in quiet ones.
- **Overbooking and inventory** - if we know how many bookings usually cancel, we can safely
  sell a few extra rooms and avoid empty beds.
- **Staffing & purchasing** - roster staff and order food according to expected occupancy.
- **Marketing** - target the right guests, channels and seasons.
- **Cancellation policy** - decide on deposits, refund windows and minimum-stay rules.
        """
    )

    st.subheader("2. The three business questions (in my own words)")
    c1, c2, c3 = st.columns(3)
    c1.info("**Q1 - Hotel type**\n\nAre guests choosing the City Hotel or the Resort Hotel more "
            "often, and how does demand move through the year?")
    c2.info("**Q2 - Length of stay**\n\nDo guests who book longer stays cancel more or less often "
            "than guests who book short stays?")
    c3.info("**Q3 - Lead time**\n\nDoes booking far in advance make a guest more likely to "
            "cancel than booking close to arrival?")

    st.subheader("3. Objective and main deliverable")
    st.success(
        "**Objective:** use data visualisation on 2017-2019 booking records to explain what drives "
        "bookings and cancellations for the two hotel types.\n\n"
        "**Single main deliverable:** a clear, chart-backed summary with business recommendations "
        "that a hotel manager (who does not code) can act on to grow bookings and reduce "
        "revenue lost to cancellations."
    )
    st.caption("Use the sidebar to move through the stages. Every insight on later pages is "
               "computed from the data and shown next to the chart that proves it.")


# ============================================================================
# STAGE 1.1
# ============================================================================
elif page == PAGES[1]:
    st.title("Stage 1.1 - Data Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (as loaded)", f"{raw.shape[0]:,}")
    c2.metric("Columns", raw.shape[1])
    dates = pd.to_datetime(
        dict(year=raw["arrival_date_year"],
             month=raw["arrival_date_month"].map(MONTH_NUM).astype(float),
             day=raw["arrival_date_day_of_month"]), errors="coerce")
    c3.metric("First arrival", dates.min().strftime("%d %b %Y"))
    c4.metric("Last arrival", dates.max().strftime("%d %b %Y"))

    st.markdown(
        f"The dataset has **{raw.shape[0]:,} rows × {raw.shape[1]} columns** and covers arrivals "
        f"from **{dates.min():%B %Y}** to **{dates.max():%B %Y}** "
        f"(calendar years {raw['arrival_date_year'].min()}-{raw['arrival_date_year'].max()}). "
        f"Each row is one booking."
    )
    st.info(
        "ℹ️ The project brief mentions ~119K rows. The file you supplied is a **pre-cleaned version "
        f"with {raw.shape[0]:,} rows** (duplicates and several missing values were already removed). "
        "The app still runs and documents every cleaning step, so it also works on the raw file."
    )

    st.subheader("Columns most relevant to the three questions")
    st.dataframe(COLUMN_GUIDE, hide_index=True, width="stretch")

    st.subheader("Preview of the data")
    st.dataframe(raw.head(20), width="stretch")

    with st.expander("Data types and summary statistics"):
        st.dataframe(raw.dtypes.astype(str).rename("dtype"), width="stretch")
        st.dataframe(raw.describe().T, width="stretch")

    st.subheader("Bookings by arrival year")
    yc = raw["arrival_date_year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(x=yc.index.astype(str), y=yc.values, color="#4c72b0", ax=ax)
    ax.set_title("Number of bookings by arrival year", fontweight="bold")
    ax.set_xlabel("Arrival year")
    ax.set_ylabel("Number of bookings")
    show(fig, "Bookings are spread across 2017-2019, but not every month is present in every "
              "year - this matters when we compare months on the next pages.")


# ============================================================================
# STAGE 1.2
# ============================================================================
elif page == PAGES[2]:
    st.title("Stage 1.2 - Data Assessment & Cleaning")
    st.caption("Every issue below is checked on the loaded file, a decision is made, and the "
               "decision is justified.")

    tabs = st.tabs(["1 Missing values", "2 Duplicates", "3 Inconsistent values",
                    "4 Anomalies", "Before → After"])

    # --- 1 Missing ---------------------------------------------------------
    with tabs[0]:
        st.subheader("1. Missing values")
        miss = info["missing_raw"]
        miss = miss[miss > 0]
        if miss.empty:
            st.success("No missing values found in the loaded file.")
        else:
            mdf = pd.DataFrame({"Missing": miss, "% of rows": (miss / info["rows_raw"] * 100).round(2)})
            c1, c2 = st.columns([1, 2])
            c1.dataframe(mdf, width="stretch")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=mdf["% of rows"], y=mdf.index, color="#c44e52", ax=ax)
            ax.set_title("Missing values by column", fontweight="bold")
            ax.set_xlabel("% of rows missing")
            ax.set_ylabel("Column")
            with c2:
                show(fig, "Agent and company are the most incomplete columns because many guests "
                          "book without either.")

        st.markdown("**Decision for each affected column**")
        st.table(pd.DataFrame(
            [
                ("company", "Fill with 0", "No company was involved: a private (non-corporate) guest."),
                ("agent", "Fill with 0", "No travel agent: the guest booked directly with the hotel."),
                ("children", "Fill with 0", "Blank most likely means no children travelled; 0 is the safe, neutral value."),
                ("city", "Fill with 'Unknown'", "Guest's origin city was not recorded; we keep the booking rather than lose it."),
            ],
            columns=["Column", "Action", "What the chosen value implies about the guest"],
        ))

    # --- 2 Duplicates ------------------------------------------------------
    with tabs[1]:
        st.subheader("2. Duplicate rows")
        n_dup = info["duplicates"]
        st.metric("Duplicate rows found", f"{n_dup:,}")
        if n_dup == 0:
            st.success("No exact duplicate rows in the loaded file (they were already removed). "
                       "The pipeline still calls `drop_duplicates()` so a raw file would be handled.")
        else:
            st.warning(f"{n_dup:,} exact duplicate rows were removed.")
        st.markdown("**Why remove them?** An identical booking counted twice would inflate "
                    "booking totals and distort cancellation rates.")

    # --- 3 Inconsistent ----------------------------------------------------
    with tabs[2]:
        st.subheader("3. Inconsistent or unclear values")
        st.markdown(
            """
- **`meal = 'Undefined'`** means no meal package was recorded. In hotel booking data this is
  equivalent to **'SC' (self-catering / no meal)**, so both are grouped into **`No Meal`**.
- **`market_segment` / `distribution_channel = 'Undefined'`** carry no information, so they are
  grouped into **`Other`** instead of being treated as a real category.
- **`customer_type = 'Bussiness'`** is a spelling mistake and is corrected to **`Business`**.
            """
        )
        c1, c2 = st.columns(2)
        for col, key, cont in [("meal", "meal_raw", c1), ("market_segment", "segment_raw", c2)]:
            vc = info[key]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(x=vc.values, y=vc.index, color="#4c72b0", ax=ax)
            ax.set_title(f"'{col}' categories in loaded file", fontweight="bold")
            ax.set_xlabel("Number of bookings")
            ax.set_ylabel(col)
            with cont:
                show(fig, f"Categories of `{col}` before regrouping. "
                          f"{'Undefined/SC present' if 'Undefined' in vc.index or 'SC' in vc.index else 'Already tidy in this file'}.")
        st.write("**After cleaning - `meal`:**", df["meal"].value_counts().to_dict())

    # --- 4 Anomalies -------------------------------------------------------
    with tabs[3]:
        st.subheader("4. Anomalies")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Negative ADR", info["adr_negative"])
        c2.metric("ADR above 1,000", info["adr_extreme"])
        c3.metric("Zero ADR (kept)", f"{info['adr_zero']:,}")
        c4.metric("Zero-guest bookings", info["zero_guests"])
        c5, c6 = st.columns(2)
        c5.metric("Zero-night stays", f"{info['zero_nights']:,}")
        c6.metric("Parties of 10+ adults", info["big_party"])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.boxplot(y=raw["adr"], ax=axes[0], color="#dd8452")
        axes[0].set_title("ADR as loaded (outliers visible)", fontweight="bold")
        axes[0].set_ylabel("Average daily rate (adr)")
        sns.boxplot(y=df["adr"], ax=axes[1], color="#55a868")
        axes[1].set_title("ADR after cleaning", fontweight="bold")
        axes[1].set_ylabel("Average daily rate (adr)")
        for a in axes:
            a.set_xlabel("All bookings")
        fig.tight_layout()
        show(fig, "Left: a handful of very high ADR values stretch the scale. Right: after removing "
                  "impossible values the distribution is readable.")

        st.table(pd.DataFrame(
            [
                ("Negative adr", "Remove", "A price cannot be negative - it is a data-entry error."),
                ("adr > 1,000", "Remove", "Extreme outlier (one record is 5,400), far above the 99.9th "
                                         "percentile; it would distort any price average."),
                ("adr = 0", "Keep", "Often complimentary / staff / cancelled stays. They are real bookings, "
                                    "and we analyse cancellations (not revenue), so removing them would bias results."),
                ("Zero guests (adults+children+babies = 0)", "Remove", "A booking with nobody staying is invalid."),
                ("Zero-night stays", "Keep in general, exclude from stay-duration chart",
                 "They are real bookings for other questions, but 'length of stay' is undefined for them."),
                ("Parties of 10+ adults", "Keep (flag only)", "Plausibly group bookings; they do not affect cancellation rates."),
            ],
            columns=["Anomaly", "Action", "Why"],
        ))

    # --- Before/After ------------------------------------------------------
    with tabs[4]:
        st.subheader("Cleaning summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows loaded", f"{info['rows_raw']:,}")
        c2.metric("Rows after cleaning", f"{info['rows_clean']:,}",
                  delta=f"{info['rows_clean'] - info['rows_raw']:,}")
        c3.metric("Missing values left", int(df[["children", "agent", "company", "city"]].isna().sum().sum()))
        st.markdown("Added columns: `total_nights` (weekend + weekday nights), `total_guests`, "
                    "`arrival_date`, `status`.")
        st.dataframe(df.head(15), width="stretch")
        st.download_button("⬇️ Download cleaned CSV", df.to_csv(index=False).encode(),
                           "hotel_bookings_clean_final.csv", "text/csv")


# ============================================================================
# STAGE 2.1
# ============================================================================
elif page == PAGES[3]:
    st.title("Stage 2.1 - Monthly Booking Analysis by Hotel Type")

    # Q1 share
    st.subheader("1. Which hotel type is booked more often?")
    c1, c2 = st.columns([1, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(share.values, labels=share.index, autopct="%1.1f%%", startangle=90,
           colors=[COLORS[h] for h in share.index], wedgeprops=dict(width=0.45, edgecolor="white"),
           textprops={"fontsize": 11})
    ax.set_title("Share of bookings by hotel type", fontweight="bold")
    with c1:
        show(fig, "Share of all bookings that went to each hotel type.")
    counts = df["hotel"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(x=counts.index, y=counts.values, palette=[COLORS[h] for h in counts.index],
                hue=counts.index, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title("Number of bookings by hotel type", fontweight="bold")
    ax.set_xlabel("Hotel type")
    ax.set_ylabel("Number of bookings")
    with c2:
        show(fig, "Absolute number of bookings for each hotel type.")
    gap = share[top_hotel] - share[low_hotel]
    st.success(f"**{top_hotel}** is booked more often: **{share[top_hotel]:.1f}%** of bookings vs "
               f"**{share[low_hotel]:.1f}%** for {low_hotel} - about **{gap:.1f} percentage points** "
               f"more ({counts[top_hotel] / counts[low_hotel]:.2f}× as many bookings).")

    # Q2 monthly
    st.subheader("2. Bookings per month for each hotel type")
    view = st.radio("Measure", ["Average bookings per year (fair comparison)",
                                "Total bookings in the dataset"], horizontal=True)
    data = monthly_avg if view.startswith("Average") else monthly_total
    fig, ax = plt.subplots(figsize=(11, 5))
    for hotel, g in data.groupby("hotel"):
        g = g.sort_values("arrival_date_month")
        ax.plot(g["arrival_date_month"].astype(str), g["bookings"], marker="o", lw=2,
                color=COLORS[hotel], label=hotel)
    ax.set_xlim(-0.5, 11.5)
    ax.set_title("Bookings per arrival month by hotel type", fontweight="bold")
    ax.set_xlabel("Arrival month")
    ax.set_ylabel("Average bookings per year" if view.startswith("Average") else "Number of bookings")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Hotel type")
    fig.tight_layout()
    show(fig, "Monthly booking volume. Use the switch above: the 'average per year' view corrects "
              "for months that appear in only two of the three years.")

    cov = (df.groupby("arrival_date_month", observed=True)["arrival_date_year"]
             .agg(lambda s: ", ".join(map(str, sorted(s.unique())))))
    with st.expander("Which years are available for each month?"):
        st.dataframe(cov.rename("Years with data"), width="stretch")

    hm = (df.pivot_table(index="hotel", columns="arrival_date_month", values="is_canceled",
                         aggfunc="size", observed=True))
    fig, ax = plt.subplots(figsize=(11, 2.8))
    sns.heatmap(hm, annot=True, fmt=".0f", cmap="YlOrRd", cbar=False, ax=ax)
    ax.set_title("Total bookings: hotel type × arrival month", fontweight="bold")
    ax.set_xlabel("Arrival month")
    ax.set_ylabel("Hotel type")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    show(fig, "Heatmap of total booking counts; darker cells are busier.")

    per_hotel = {}
    for hotel, g in monthly_avg.groupby("hotel"):
        g = g.set_index("arrival_date_month")["bookings"]
        per_hotel[hotel] = (g.idxmax(), g.idxmin())
    st.success(
        f"**Busiest months (all hotels, avg per year):** {', '.join(top3)}.  \n"
        f"**Quietest months:** {', '.join(bottom3)}.  \n"
        + "  \n".join(f"**{h}** peaks in **{p[0]}** and is quietest in **{p[1]}**."
                      for h, p in per_hotel.items())
    )

    st.subheader("3. What might explain the seasonal pattern?")
    st.markdown(
        f"""
The data shows peaks around **{', '.join(top3)}** and a dip in **{', '.join(bottom3)}**. Plausible explanations
(the data shows *when* demand moves, not *why* - these are reasoned hypotheses):

- **School holidays and public holidays** - families travel in mid-year school breaks and around
  year-end, which lifts leisure demand, especially for the Resort Hotel.
- **Travel habits** - leisure guests choose resort destinations in warm/holiday periods, so the
  Resort Hotel is more seasonal than the City Hotel.
- **Business travel** - city hotels are supported by work trips, which are steadier through the
  year and fall mainly during working weeks; this smooths the City Hotel curve.
- **Weather and festivals** - rainy periods and the absence of long holidays usually mean fewer leisure bookings.
        """
    )
    st.subheader("4. Decisions a hotel could make from this trend")
    st.markdown(
        f"""
- **Dynamic pricing:** higher rates and minimum-stay rules in peak months (**{top3[0]}**), discounts or packages in the quiet months (**{bottom3[0]}**).
- **Staffing & purchasing:** hire seasonal staff and stock supplies ahead of peak months; schedule maintenance and renovations in the quietest months.
- **Marketing calendar:** spend promotion budget before the shoulder months to lift demand, not in months that already sell out.
- **Overbooking & inventory:** plan overbooking buffers more carefully in busy months where cancellations leave rooms that can be resold.
- **Corporate/event deals:** target conferences and corporate contracts in low season to fill rooms.
        """
    )


# ============================================================================
# STAGE 2.2
# ============================================================================
elif page == PAGES[4]:
    st.title("Stage 2.2 - Impact of Stay Duration on Cancellation Rate")

    # Q1
    st.subheader("1. Cancellation rate by hotel type")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=cancel_by_hotel.index, y=cancel_by_hotel.values,
                palette=[COLORS[h] for h in cancel_by_hotel.index],
                hue=cancel_by_hotel.index, legend=False, ax=ax)
    for i, v in enumerate(cancel_by_hotel.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Cancellation rate by hotel type", fontweight="bold")
    ax.set_xlabel("Hotel type")
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_ylim(0, cancel_by_hotel.max() * 1.2)
    show(fig, "Share of bookings that were cancelled, for each hotel type.")
    worse = cancel_by_hotel.idxmax()
    better = cancel_by_hotel.idxmin()
    st.success(f"**{worse}** is cancelled more often: **{cancel_by_hotel[worse]:.1f}%** vs "
               f"**{cancel_by_hotel[better]:.1f}%** for {better} "
               f"({cancel_by_hotel[worse] - cancel_by_hotel[better]:.1f} points higher).")

    # Q2
    st.subheader("2. Cancellation rate as the length of stay increases")
    c1, c2 = st.columns(2)
    cap = c1.slider("Group stays of this many nights or more into one bar", 7, 30, 15)
    min_n = c2.slider("Hide groups with fewer than N bookings (avoids noisy points)", 1, 300, 50)
    edges = list(range(1, cap + 1)) + [np.inf]
    out, labels = binned_rate(stay_df, "total_nights", edges)
    fig = rate_line_chart(out, labels, "Cancellation rate by total length of stay",
                          "Total nights booked (weekend + weekday)", min_n)
    show(fig, "Each point is the % of bookings cancelled for a given stay length. Points with too "
              "few bookings are hidden. Excludes zero-night bookings.")

    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=out[out["n"] >= min_n], x="bin", y="n", hue="hotel", palette=COLORS, ax=ax)
    ax.set_title("How many bookings sit behind each point?", fontweight="bold")
    ax.set_xlabel("Total nights booked")
    ax.set_ylabel("Number of bookings")
    ax.tick_params(axis="x", rotation=60)
    with c1:
        show(fig, "Sample size per stay length - most stays are short, so long-stay rates rest on fewer bookings.")
    bucket = pd.cut(stay_df["total_nights"], [0, 3, 7, np.inf], labels=["1-3 nights", "4-7 nights", "8+ nights"])
    bk = stay_df.assign(bucket=bucket).groupby(["hotel", "bucket"], observed=True)["is_canceled"].mean().mul(100).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=bk, x="bucket", y="is_canceled", hue="hotel", palette=COLORS, ax=ax)
    ax.set_title("Cancellation rate by stay-length band", fontweight="bold")
    ax.set_xlabel("Length of stay")
    ax.set_ylabel("Cancellation rate (%)")
    with c2:
        show(fig, "Same idea in three simple bands (short, medium, long).")

    # Q3
    st.subheader("3. Positive, negative or flat?")
    corr = {h: stay_df.loc[stay_df["hotel"] == h, ["total_nights", "is_canceled"]].corr().iloc[0, 1]
            for h in stay_df["hotel"].unique()}
    short = bk[bk["bucket"] == "1-3 nights"].set_index("hotel")["is_canceled"]
    long_ = bk[bk["bucket"] == "8+ nights"].set_index("hotel")["is_canceled"]
    rows = []
    for h in corr:
        rows.append({"Hotel": h, "Correlation (nights vs cancel)": round(corr[h], 3),
                     "Direction": corr_label(corr[h]),
                     "1-3 nights cancel %": round(short.get(h, np.nan), 1),
                     "8+ nights cancel %": round(long_.get(h, np.nan), 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for h in corr:
        diff = long_.get(h, np.nan) - short.get(h, np.nan)
        st.markdown(f"- **{h}:** relationship is **{corr_label(corr[h])}** (r = {corr[h]:+.3f}); "
                    f"long stays are cancelled **{abs(diff):.1f} points {'more' if diff > 0 else 'less'}** often than short stays.")
    directions = {corr_label(v) for v in corr.values()}
    if len(directions) == 1:
        st.info("Both hotel types show the **same direction**, though the strength differs (see table).")
    else:
        st.info("The two hotel types show **different** relationships between stay length and cancellation.")

    # Q4
    st.subheader("4. Why might longer stays be cancelled more (or less)?")
    st.markdown(
        """
**Reasons longer stays may be cancelled MORE often**
- A long trip needs more time off, more money and more planning, so a change of plans (work, illness, visa, budget) is more likely to break it.
- Guests booking long stays often book several options early and later drop the ones they do not need.
- Higher total price means more incentive to keep shopping for a better deal until the last moment.

**Reasons longer stays may be cancelled LESS often**
- Guests who commit to a long stay are usually more certain (relocation, long holiday, project work).
- Long stays are often discounted or prepaid / non-refundable, which discourages cancelling.
- Short city stays are the easiest to rebook or replace, so impulse cancellations concentrate there.

Look at the chart above to see which story the data supports - the data shows *what* happens, not *why*.
        """
    )


# ============================================================================
# STAGE 2.3
# ============================================================================
elif page == PAGES[5]:
    st.title("Stage 2.3 - Impact of Lead Time on Cancellation Rate")

    st.subheader("1. What is lead time?")
    st.info("**Lead time** is the number of days between the day the booking was made and the day "
            "the guest is due to arrive. A lead time of 0 means a same-day booking; 300 means the "
            "guest booked about ten months ahead.")
    ls = df["lead_time"].describe()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median lead time", f"{ls['50%']:.0f} days")
    c2.metric("Average lead time", f"{ls['mean']:.0f} days")
    c3.metric("75% book within", f"{ls['75%']:.0f} days")
    c4.metric("Longest lead time", f"{ls['max']:.0f} days")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    sns.histplot(data=df, x="lead_time", hue="hotel", palette=COLORS, bins=60, element="step", ax=ax)
    ax.set_title("Distribution of lead time", fontweight="bold")
    ax.set_xlabel("Lead time (days between booking and arrival)")
    ax.set_ylabel("Number of bookings")
    show(fig, "Most bookings are made within a few months of arrival; a long tail books a year or more ahead.")

    st.subheader("Cancellation rate vs lead time")
    c1, c2, c3 = st.columns(3)
    bw = c1.select_slider("Group lead time in bins of (days)", [7, 15, 30, 45, 60], value=30)
    cap = c2.select_slider("Group everything above (days) into one bin", [180, 270, 360, 450, 540], value=360)
    min_n = c3.slider("Hide bins with fewer than N bookings", 1, 300, 50)
    edges = list(range(0, cap, bw)) + [cap, np.inf]
    out, labels = binned_rate(df, "lead_time", edges)
    fig = rate_line_chart(out, labels, "Cancellation rate by lead time",
                          "Lead time (days between booking and arrival)", min_n, rotate=60)
    show(fig, "Each point is the % of bookings cancelled within a lead-time band. A rising line "
              "means bookings made further ahead are cancelled more often.")

    # Q2
    st.subheader("2. Where is cancellation lowest and highest?")
    ext = extreme_bins(out, min_n)
    st.dataframe(pd.DataFrame(
        [{"Hotel": h, "Lowest at (days)": lo["bin"], "Lowest rate %": round(lo["rate"], 1),
          "Highest at (days)": hi["bin"], "Highest rate %": round(hi["rate"], 1)}
         for h, (lo, hi) in ext.items()]), hide_index=True, width="stretch")
    hi_h = max(ext, key=lambda h: ext[h][1]["rate"])
    st.success(f"The single highest cancellation rate is **{ext[hi_h][1]['rate']:.1f}%** for the "
               f"**{hi_h}** at lead times of **{ext[hi_h][1]['bin']} days**.")

    # Q3
    st.subheader("3. Which hotel type is more stable across lead times?")
    piv = out[out["n"] >= min_n].pivot(index="bin", columns="hotel", values="rate")
    spread = (piv.max() - piv.min()).sort_values()
    stable, volatile = spread.index[0], spread.index[-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.barplot(x=spread.index, y=spread.values, palette=[COLORS[h] for h in spread.index],
                hue=spread.index, legend=False, ax=ax)
    for i, v in enumerate(spread.values):
        ax.text(i, v + 0.5, f"{v:.1f} pts", ha="center", fontweight="bold")
    ax.set_title("Spread of cancellation rate across lead-time bins", fontweight="bold")
    ax.set_xlabel("Hotel type")
    ax.set_ylabel("Highest minus lowest cancellation rate (points)")
    show(fig, "Bigger bar = the hotel's cancellation rate swings more with lead time.")
    st.success(f"**{stable}** is more stable (range {spread[stable]:.1f} points) while **{volatile}** "
               f"swings by **{spread[volatile]:.1f} points** across lead times.")

    dep = (df.assign(lt=pd.cut(df["lead_time"], [-1, 30, 90, 180, np.inf],
                               labels=["0-30", "31-90", "91-180", "181+"]))
             .groupby(["hotel", "lt", "deposit_type"], observed=True)["is_canceled"]
             .agg(rate="mean", n="size").reset_index())
    dep["rate"] *= 100
    dep = dep[dep["n"] >= max(min_n, 30)]
    with st.expander("Extra evidence: does deposit type change the picture?"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, hotel in zip(axes, ["City Hotel", "Resort Hotel"]):
            sns.barplot(data=dep[dep["hotel"] == hotel], x="lt", y="rate", hue="deposit_type", ax=ax)
            ax.set_title(f"{hotel}: cancellation by lead time and deposit type", fontweight="bold")
            ax.set_xlabel("Lead time band (days)")
            ax.set_ylabel("Cancellation rate (%)")
        fig.tight_layout()
        show(fig, "Cancellation rate for each deposit type within lead-time bands. Bands with very few bookings are hidden.")

    st.markdown("**What could explain the difference between the hotels?**")
    st.markdown(
        """
- **Booking purpose:** city bookings are often work-related or flexible and are made through online
  travel agents, where free cancellation is common - guests can book early "just in case" and drop later.
- **Resort guests commit:** holiday guests plan around school holidays and often pay a deposit or book
  non-refundable rates, so far-ahead plans tend to hold.
- **Many alternatives in cities:** more hotel options at similar prices make it easy to switch when a better deal appears.
- **Uncertainty grows with time:** the longer the gap, the more can change (work schedule, budget, health, weather).
        """
    )


# ============================================================================
# STAGE 3
# ============================================================================
else:
    st.title("Stage 3 - Summary and Recommendations")
    st.caption("All numbers below are computed live from the cleaned data.")

    # facts -----------------------------------------------------------------
    corr = {h: stay_df.loc[stay_df["hotel"] == h, ["total_nights", "is_canceled"]].corr().iloc[0, 1]
            for h in stay_df["hotel"].unique()}
    nights_short = stay_df[stay_df["total_nights"] <= 3]["is_canceled"].mean() * 100
    nights_long = stay_df[stay_df["total_nights"] >= 8]["is_canceled"].mean() * 100
    long_cancels_more = nights_long > nights_short
    lt_low = df[df["lead_time"] <= 30]["is_canceled"].mean() * 100
    lt_mid = df[(df["lead_time"] > 30) & (df["lead_time"] <= 90)]["is_canceled"].mean() * 100
    lt_high = df[df["lead_time"] > 90]["is_canceled"].mean() * 100
    lt_vhigh = df[df["lead_time"] > 180]["is_canceled"].mean() * 100
    lt_hotel = (df.assign(far=df["lead_time"] > 90).groupby(["hotel", "far"])["is_canceled"]
                  .mean().mul(100).unstack())

    # 1 summary -------------------------------------------------------------
    st.subheader("1. Key findings")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{top_hotel} share", f"{share[top_hotel]:.1f}%")
    c2.metric("Cancel rate (stays 1-3 vs 8+ nights)", f"{nights_short:.1f}% → {nights_long:.1f}%")
    c3.metric("Cancel rate (lead ≤30 vs >90 days)", f"{lt_low:.1f}% → {lt_high:.1f}%")
    st.markdown(
        f"""
- **Hotel type & seasonality (Q1):** {top_hotel} receives **{share[top_hotel]:.1f}%** of bookings vs
  {share[low_hotel]:.1f}% for {low_hotel}. Demand peaks in **{', '.join(top3)}** and is weakest in
  **{', '.join(bottom3)}**. *(Evidence: Stage 2.1 donut and monthly line chart.)*
- **Stay duration (Q2):** {worse} has the higher cancellation rate ({cancel_by_hotel[worse]:.1f}% vs
  {cancel_by_hotel[better]:.1f}%). Stays of 8+ nights are cancelled {nights_long:.1f}% of the time vs
  {nights_short:.1f}% for 1-3 nights - a **{corr_label(np.mean(list(corr.values())))}** relationship overall.
  *(Evidence: Stage 2.2 line chart and stay-band bars.)*
- **Lead time (Q3):** cancellation climbs from **{lt_low:.1f}%** for bookings made within 30 days to
  **{lt_high:.1f}%** beyond 90 days and **{lt_vhigh:.1f}%** beyond 180 days. For the City Hotel the jump is
  {lt_hotel.loc['City Hotel', False]:.1f}% → {lt_hotel.loc['City Hotel', True]:.1f}%, for the Resort Hotel
  {lt_hotel.loc['Resort Hotel', False]:.1f}% → {lt_hotel.loc['Resort Hotel', True]:.1f}%.
  *(Evidence: Stage 2.3 line chart.)*
        """
    )

    # 2 hotel type & season -------------------------------------------------
    st.subheader("2. Recommendations - hotel type and seasonality")
    st.markdown(
        f"""
1. **Grow the {low_hotel}:** it takes only {share[low_hotel]:.1f}% of bookings. Build packages
   (weekend / family / wellness), partner with local attractions and run targeted online campaigns
   in its quiet months (**{bottom3[0]}**, **{bottom3[1]}**) to lift occupancy.
2. **Capitalise on peak season:** raise rates and add minimum-stay rules in **{top3[0]}**, **{top3[1]}**
   and **{top3[2]}**; sell early-bird packages before the peak and schedule extra staff and supplies.
3. **Fill the low season:** offer corporate/event deals, mid-week discounts and loyalty promotions
   and use quiet months for maintenance and renovation.
        """
    )

    # 3 stay duration -------------------------------------------------------
    st.subheader("3. Recommendations - stay duration and cancellations")
    if long_cancels_more:
        st.markdown(
            f"""
Longer stays are cancelled more often ({nights_long:.1f}% vs {nights_short:.1f}%), so the larger
bookings carry more revenue risk:

1. **Require a deposit** (e.g. first night or a % of the total) for stays of 7+ nights.
2. **Tiered cancellation terms:** free cancellation only until ~14 days before arrival for long stays; partial charge afterwards.
3. **Offer a discount for non-refundable long-stay rates** so guests who are certain can choose the cheaper, safer option.
4. **Overbook carefully** on long-stay inventory based on the observed rate.
            """
        )
    else:
        st.markdown(
            f"""
Longer stays are cancelled **less** often ({nights_long:.1f}% vs {nights_short:.1f}%), so the risk sits
in short bookings:

1. **Encourage longer stays** with discounts such as "stay 4 nights, get the 4th at half price".
2. **Stricter terms on 1-3 night bookings** during peak months (small deposit or shorter free-cancellation window).
3. **Minimum-stay rules** at peak weekends to protect revenue from easy-to-cancel short bookings.
4. **Overbook slightly on short-stay inventory** based on the observed rate.
            """
        )

    # 4 lead time -----------------------------------------------------------
    st.subheader("4. Recommendations - far-ahead bookings")
    st.markdown(
        f"""
Bookings made more than 90 days ahead are cancelled **{lt_high:.1f}%** of the time (**{lt_vhigh:.1f}%** beyond 180 days):

1. **Reminders and re-confirmation:** automated emails/WhatsApp at 60, 30 and 7 days asking the guest to confirm - this releases rooms early.
2. **Deposit scaling with lead time:** the earlier the booking, the larger the deposit or the earlier the free-cancellation cut-off.
3. **Offer rescheduling instead of cancelling** (date change credit or voucher) to keep the revenue.
4. **Price incentives:** a lower non-refundable rate for early bookers; a flexible rate at a premium.
5. **Overbooking buffer** sized by lead-time band so cancelled far-ahead rooms can be resold.
        """
    )

    # 5 biggest impact ------------------------------------------------------
    st.subheader("5. Which single recommendation has the biggest impact?")
    st.caption("Estimate of cancellations that would be avoided if each high-risk group's cancellation "
               "rate fell to a lower-risk benchmark group. It is an upper-bound what-if, not a forecast.")

    def lever(seg_mask, bench_mask, d):
        n = int(seg_mask.sum())
        r_seg = d[seg_mask]["is_canceled"].mean()
        r_b = d[bench_mask]["is_canceled"].mean()
        return n, r_seg * 100, r_b * 100, max(0.0, n * (r_seg - r_b))

    levers = []
    n, rs, rb, av = lever(df["lead_time"] > 90, df["lead_time"] <= 30, df)
    levers.append(("Lead time: bookings >90 days ahead → match ≤30-day rate", n, rs, rb, av))
    n, rs, rb, av = lever(stay_df["total_nights"] >= 8, stay_df["total_nights"] <= 3, stay_df)
    levers.append(("Stay length: 8+ nights → match 1-3 night rate", n, rs, rb, av))
    n, rs, rb, av = lever(df["hotel"] == worse, df["hotel"] == better, df)
    levers.append((f"Hotel type: {worse} → match {better} rate", n, rs, rb, av))
    lev = pd.DataFrame(levers, columns=["Lever", "Bookings affected", "Segment cancel %", "Benchmark cancel %",
                                        "Cancellations avoided"]).sort_values("Cancellations avoided", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    sns.barplot(data=lev, y="Lever", x="Cancellations avoided", color="#4c72b0", ax=ax)
    for i, v in enumerate(lev["Cancellations avoided"]):
        ax.text(v, i, f" {v:,.0f}", va="center", fontweight="bold")
    ax.set_title("Cancellations avoided under each what-if scenario", fontweight="bold")
    ax.set_xlabel("Number of cancelled bookings avoided")
    ax.set_ylabel("")
    ax.set_xlim(0, lev["Cancellations avoided"].max() * 1.15)
    fig.tight_layout()
    show(fig, "Bigger bar = bigger potential reduction in cancellations.")
    st.dataframe(lev.round(1), hide_index=True, width="stretch")

    best = lev.iloc[0]
    st.success(
        f"**Biggest impact: {best['Lever']}.** About **{best['Bookings affected']:,.0f}** bookings are in "
        f"this group; they cancel **{best['Segment cancel %']:.1f}%** of the time vs "
        f"**{best['Benchmark cancel %']:.1f}%** for the benchmark group. Closing that gap would avoid roughly "
        f"**{best['Cancellations avoided']:,.0f}** cancellations out of "
        f"{int(df['is_canceled'].sum()):,} in total "
        f"({best['Cancellations avoided'] / df['is_canceled'].sum() * 100:.1f}%)."
    )
    st.markdown(
        "**Recommended action:** "
        + ("introduce **lead-time based deposits, reminders and re-confirmation** for bookings made more than "
           "90 days ahead, with a flexible/non-refundable pricing choice." if "Lead time" in best["Lever"]
           else "apply **stay-length based deposit and cancellation terms** and use pricing to steer guests "
                "toward the lower-risk booking type." if "Stay length" in best["Lever"]
           else f"review the **{worse}** cancellation terms, channel mix and pricing to bring its cancellation rate closer to the {better}.")
    )