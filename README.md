# 📦 AI-Enabled Digital Twin for Supply Chain Management

An AI-powered Digital Twin of a multi-city supply chain that simulates inventory operations, predicts customer demand, and provides real-time decision support through an interactive dashboard.

This project demonstrates how Artificial Intelligence, Simulation, and Data Analytics can work together to improve supply chain efficiency by reducing stockouts, optimizing inventory levels, and enabling data-driven decision making.

Project Demo
Streamlit:https://flipkart-digital-twin-hvannmrredwzjavnegkyy7.streamlit.app/

---

## 🚀 Project Overview

Modern supply chains generate massive amounts of operational data every day. However, making timely decisions based on this data remains a challenge.

To address this, this project builds a Digital Twin—a virtual representation of a supply chain—that continuously monitors inventory, forecasts future demand, and evaluates operational performance across multiple warehouse locations.

The project combines simulation, demand forecasting, visualization, and optimization into a single interactive system.

---

## 🎯 Objectives

- Build a Digital Twin of a supply chain
- Simulate real-world inventory movement
- Forecast future product demand using Machine Learning and Time Series techniques
- Monitor inventory in real time through an interactive dashboard
- Reduce stockouts while maintaining healthy inventory levels
- Support better operational decision-making using predictive analytics

---

## 🛠 Technologies Used

| Category | Tools |
|----------|-------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Forecasting | ARIMA (Statsmodels) |
| Machine Learning | Scikit-learn |
| Development | Google Colab, VS Code |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```
AI-Enabled-Digital-Twin-for-Supply-Chain-Management/
│
├── dashboard.py
├── flipkart_iphone16_multicity.csv
├── requirements.txt
├── README.md
└── assets/
```

---

## 📊 Features

### 📦 Inventory Monitoring

- Live inventory tracking
- Warehouse stock visualization
- Daily inventory updates
- Stock availability monitoring

---

### 📈 Demand Forecasting

The project predicts future customer demand using the ARIMA time-series forecasting model.

Forecast accuracy is evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

These metrics help measure how closely the predicted demand matches actual demand.

---

### 📉 Interactive Dashboard

The Streamlit dashboard provides:

- Inventory trend analysis
- Daily demand visualization
- Stockout monitoring
- Cost analysis
- Forecast comparison
- Supply chain KPIs

---

### 🌍 Multi-City Supply Chain

The simulation models operations across multiple cities including:

- Delhi
- Mumbai
- Bengaluru
- Chennai

This allows comparison of inventory and demand across different warehouse locations.

---

## 🔄 Digital Twin Workflow

```
Historical Data
       │
       ▼
Data Cleaning
       │
       ▼
Inventory Simulation
       │
       ▼
Demand Forecasting (ARIMA)
       │
       ▼
Inventory Prediction
       │
       ▼
Real-Time Dashboard
       │
       ▼
Business Decision Support
```

---

## 📊 Key Performance Indicators (KPIs)

The dashboard tracks important supply chain metrics such as:

- Inventory Level
- Daily Demand
- Stockout Rate
- Holding Cost
- Stockout Cost
- Reorder Quantity
- Forecast Accuracy
- Warehouse Performance

---

## 📈 Results

The developed Digital Twin successfully:

- Simulated inventory movement across multiple warehouses
- Predicted future demand using ARIMA
- Reduced uncertainty in inventory planning
- Improved supply chain visibility
- Enabled proactive inventory management
- Provided an interactive dashboard for real-time monitoring

---

## 💡 Future Improvements

Potential enhancements include:

- Integration with live IoT sensor data
- Real-time API connectivity
- Deep Learning forecasting models (LSTM)
- Reinforcement Learning for inventory optimization
- Automated reorder recommendations
- Cloud deployment on AWS or Azure

---

## ▶️ Running the Project

### Clone the repository

```bash
git clone https://github.com/kaushik-vats/AI-Enabled-Digital-Twin-for-Supply-Chain-Management.git
```

### Move into the project folder

```bash
cd AI-Enabled-Digital-Twin-for-Supply-Chain-Management
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run dashboard.py
```

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Digital Twin development
- Supply Chain Analytics
- Demand Forecasting
- Inventory Optimization
- Interactive Dashboard Development
- Data Visualization
- Time-Series Analysis
- Python for Business Analytics
- End-to-End Data Science Workflow

---

## 👨‍💻 About Me

Hi! I'm **Kaushik Vats**, an undergraduate student at **Birla Institute of Technology (BIT Mesra)** with a strong interest in Data Analytics, Machine Learning, AI, and Supply Chain Analytics.

I enjoy building data-driven solutions that combine analytics, visualization, and real-world business applications.

I'm always looking to learn, collaborate, and work on impactful projects.

---

## 🤝 Contributions

Contributions, suggestions, and feedback are always welcome.

Feel free to fork the repository, open an issue, or submit a pull request.

---

## ⭐ Support

If you found this project helpful or interesting, consider giving it a ⭐ on GitHub.

It motivates me to continue building and sharing more projects!

---

## 📄 License

This project is intended for educational and learning purposes.
