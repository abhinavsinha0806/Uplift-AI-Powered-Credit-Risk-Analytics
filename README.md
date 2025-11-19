Uplift | AI-Powered Credit Risk Analytics

Uplift is a modern credit assessment dashboard that uses alternative data points (Digital Footprint, Geo-Stability, and Income Volatility) to assess creditworthiness for applicants who might be overlooked by traditional banking systems.

🚀 Features

Alternative Scoring Engine: Uses a custom machine learning model to analyze non-traditional metrics like battery hygiene and location stability.

Interactive Dashboard: Real-time overview of approval rates, disbursed amounts, and risk distribution.

Applicant Analysis: Deep dive into individual profiles with SHAP value explanations for risk factors.

Batch Processing: Upload CSV/Excel files to process bulk applications.

AI Simulator: "What-if" analysis tool to simulate how changes in income or behavior affect credit scores.

PDF Reports: Auto-generate official credit memos for loan officers.

🛠️ Installation

Clone the repository

git clone [https://github.com/yourusername/uplift-credit-analytics.git](https://github.com/yourusername/uplift-credit-analytics.git)
cd uplift-credit-analytics


Create a virtual environment (Optional but recommended)

python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


🏃‍♂️ Usage

Run the Streamlit application:

streamlit run main.py


The dashboard will open automatically in your default browser at http://localhost:8501.

📂 Project Structure

main.py: The entry point for the Streamlit application. Handles navigation and page rendering.

ml_engine.py: Contains the UpliftMLEngine and EnhancedCreditRiskModel. Handles data generation, training, and predictions.

ui_components.py: Custom CSS styling and reusable UI widgets (cards, navbar).

🧠 Model Logic

The system uses a Neural Network (via TensorFlow/Keras) or a robust heuristic fallback if TF is unavailable. Key inputs include:

Income Volatility: Standard deviation of monthly income.

Battery Hygiene: Measures user conscientiousness via charging habits.

Geo-Stability: Measures lifestyle stability via location entropy.

📄 License

MIT
