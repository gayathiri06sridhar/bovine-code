# Phase 9: IoT Cattle-Monitoring System Integration Design

## 1. High-Level Architecture Overview
The integration of the ML model for Bovine Mastitis prediction into a real-time IoT cattle-monitoring system involves a multi-tier architecture:
- **Perception Layer (Sensors)**: IoT sensors attached to the cattle (e.g., smart collars, milking robots) collect physiological data (milk yield, temperature, pH, conductivity, somatic cell count).
- **Edge / Gateway Layer**: Data is aggregated at the farm level using an Edge Gateway (e.g., Raspberry Pi or local server). The gateway filters noise, handles data formatting, and sends it to the cloud.
- **Cloud / Processing Layer**: The cloud server runs the ML pipeline (the exported `mastitis_model_pipeline.joblib`) deployed as a REST API or microservice. It ingests data, performs feature engineering, and predicts mastitis probability.
- **Application Layer (Dashboard/Alerts)**: A web or mobile application visualizes the data for the farmer, displaying predicted risk levels, alerts, and historical health trends.

## 2. Data Flow
1. **Data Collection**: Milking robots and wearable sensors capture variables (e.g., `Day`, `Milk_Temperature`, `Milk_pH`, `Milk_Conductivity`, `Somatic_Cell_Count`, `Milk_Yield`, `Clotting`).
2. **Transmission**: Sensors transmit data via low-power networks (e.g., LoRaWAN, BLE, or Wi-Fi) to the farm's edge gateway.
3. **Inference**:
   - The edge gateway forwards the raw JSON payload to the Cloud ML API.
   - The ML API loads the data, formats it into a DataFrame, passes it through the trained pipeline (which applies `StandardScaler` and inference).
   - The API returns a JSON response containing `mastitis_probability` and `risk_level` (e.g., "HIGH RISK").
4. **Actionable Insights**: If the risk level exceeds the threshold (e.g., probability > 0.5), an automated SMS or push notification is sent to the veterinarian or farmer.

## 3. Deployment Strategy
- **Containerization**: The ML inference script and required libraries (Scikit-Learn, Pandas, XGBoost) will be packaged using Docker.
- **API Framework**: FastAPI or Flask will serve the model predictions.
- **Continuous Monitoring**: Model drift will be tracked by periodically evaluating new labelled clinical data against model predictions.

## 4. Hardware and Software Stack
- **IoT Sensors**: Smart collars (temperature), Automated Milking Systems (AMS) for pH, conductivity, yield, and SCC.
- **Communication Protocol**: MQTT for lightweight, reliable sensor data transmission.
- **Backend/Inference**: Python, FastAPI, Docker, Joblib.
- **Frontend**: React or Vue.js web dashboard for farmers.
