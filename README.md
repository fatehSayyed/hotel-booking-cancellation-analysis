🏨 Hotel Booking & Cancellation Analysis
An end-to-end Data Analytics project exploring hotel booking patterns, seasonality, and reservation cancellations. Built with Python, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook, and Streamlit, the project turns booking records into visual insights and practical recommendations for hotel operations.
The interactive Streamlit app follows the analysis from data overview and quality checks through booking trends, cancellation analysis, and business recommendations.
🎯 Business Problem
Hotel cancellations and changing demand can make room inventory, staffing, pricing, and revenue planning difficult. This project investigates booking behaviour to help hotel teams understand when customers book, which hotel type they choose, and how booking characteristics relate to cancellations.
Questions explored
1. Hotel type & seasonality: Which is booked more often—City Hotel or Resort Hotel—and how does demand vary by month?
2. Length of stay: How does the number of nights booked relate to cancellation rates?
3. Booking lead time: Are bookings made further in advance more likely to be cancelled?
4. Business action: What changes to pricing, reminders, deposits, and cancellation policies could be considered based on the analysis?
✨ Project Features
- Interactive Streamlit application with separate pages for each stage of the analysis.
- Data overview: dataset dimensions, arrival-year coverage, data preview, column guide, data types, and summary statistics.
- Data quality assessment: checks for missing values, duplicates, inconsistent categories, unusual average daily rates (ADR), and zero-guest bookings.
- Data cleaning: handles selected missing values, standardizes unclear category labels, and creates helper fields such as total guests, total nights, arrival date, and booking status.
- Booking and seasonality analysis: compares City Hotel and Resort Hotel booking shares, monthly booking volumes, and hotel/month patterns.
- Stay-duration analysis: compares cancellation rates across total-night groups, with adjustable grouping and minimum sample-size controls.
- Lead-time analysis: examines cancellation rates across booking lead-time ranges, compares hotel types, and includes an additional breakdown by deposit type.
- Business recommendations: summarizes evidence from the charts and presents possible actions for seasonal demand and cancellation risk.
- What-if comparison: estimates the potential number of cancellations avoided if selected higher-risk groups matched a lower-risk benchmark. This is an illustrative scenario—not a forecast or guarantee.
- Cleaned data export: download the cleaned data from the app as a CSV file.
🧰 Tools & Technologies
Tool	How it is used
Python	Main programming language for the analysis and app logic
Pandas	Data loading, cleaning, transformation, grouping, and aggregation
NumPy	Numerical operations and helper calculations
Matplotlib	Charts and visual presentation
Seaborn	Statistical charts and distribution analysis
Jupyter Notebook	Exploratory analysis and documented code in hotel_health.ipynb
Streamlit	Interactive, multi-page analytics application


📂 Project Files
.
├── README.md
├── app(4).py                    # Interactive Streamlit application
├── hotel_health.ipynb           # Jupyter Notebook analysis
├── hotel_bookings_data.csv      # Original booking dataset
├── hotel_bookings_cleaned.csv   # Supplied cleaned dataset used by default
└── hotel_business_project(1).pdf # Project brief
Note: The app looks for hotel_bookings_cleaned.csv by default. Keep this CSV in the same directory as the app, or upload another CSV using the sidebar file uploader.

📊 Dataset
The project files contain hotel booking records covering arrival years 2017–2019, with 29 columns describing hotel type, cancellation status, lead time, arrival date, stay duration, guest counts, meal plan, booking channel, deposit type, average daily rate, and other booking details.
File	Rows	Columns	Purpose
hotel_bookings_data.csv	119,390	29	Original dataset supplied with the project
hotel_bookings_cleaned.csv	85,963	29	Pre-cleaned dataset supplied with the project and loaded by default


The app also applies its cleaning function to the selected file. Since the cleaned CSV has already had rows and values processed, checks such as duplicate and missing-value counts may differ from those found when loading the original CSV.
🧹 Data Preparation
The app documents and applies the following cleaning decisions where the relevant columns are available:
- Removes exact duplicate rows.
- Fills missing children, agent, and company values with 0, and missing city values with Unknown.
- Groups Undefined meal values and SC under No Meal, and maps unclear market-segment or distribution-channel values to Other.
- Corrects the Bussiness customer-type spelling to Business.
- Creates total_guests and total_nights helper columns and derives arrival date and cancellation status.
- Excludes records with negative ADR, ADR above 1,000, or no recorded guests from the analysis dataset. Zero-ADR records are retained by the app's cleaning logic.
- Excludes zero-night bookings from the stay-duration analysis, while retaining them for other analyses.
These are the project's chosen handling rules; as with any analysis, assumptions about missing values and unusual records should be reviewed against the business context.
🚀 Getting Started
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/hotel-booking-cancellation-analysis.git
cd hotel-booking-cancellation-analysis
Replace YOUR-USERNAME with your GitHub username after creating the repository.
2. Install dependencies
Use Python 3.10 or later, then run:
python -m pip install streamlit pandas numpy matplotlib seaborn jupyter
3. Run the Streamlit app
From the project directory, run:
streamlit run "app(4).py"
Streamlit will print a local URL in the terminal (usually http://localhost:8501). Open that URL in your browser.
4. Open the notebook (optional)
jupyter notebook hotel_health.ipynb
🧭 How to Use the App
1. Open Data Overview to inspect the dataset and its columns.
2. Visit Data Assessment & Cleaning to review data-quality checks and cleaning choices.
3. Explore Hotel Type & Monthly Bookings to compare booking share and seasonality.
4. Use Stay Duration vs Cancellation to investigate cancellation behaviour by nights booked.
5. Use Lead Time vs Cancellation to compare cancellation rates across booking lead-time ranges.
6. Finish at Summary & Recommendations to review the findings and illustrative what-if scenarios.
Some charts include controls for bin sizes or minimum sample counts. Changing these controls changes how results are grouped or displayed.
💡 Business Value
The analysis is designed to support evidence-based discussion around:
- seasonal pricing, promotions, and staffing;
- demand differences between City Hotels and Resort Hotels;
- cancellation policies for different stay durations;
- reminders, re-confirmation, and deposit options for bookings made well in advance; and
- inventory planning based on observed cancellation patterns.
Recommendations should be validated with current operational data and tested before implementation. Relationships shown in the booking data do not, by themselves, prove why a customer cancelled.
🔭 Possible Future Improvements
- Add a dedicated executive KPI dashboard and downloadable chart/report exports.
- Add automated tests for the cleaning pipeline and edge cases.
- Add a requirements.txt file for reproducible dependency installation.
- Deploy the Streamlit app and add a link to the live demo here.
- Add screenshots or GIFs of the app to the repository README.
👤 Author
Fateh Mohammed Sayyed
Hotel Booking & Cancellation Analysis — a portfolio project demonstrating data cleaning, exploratory data analysis, visualization, interactive app development, and translating analysis into business recommendations.
