# 🔥 Fast-Commerce Inventory Heatmap

## Project Goal & Problem Solved

[cite_start]**Goal:** Design a map-based inventory management system for fast-commerce platforms (like Blinkit or Instamart) [cite: 3, 5] [cite_start]that tracks and visualizes demand per zone (pincode) on a heatmap, automates replenishment based on trends, and optionally predicts future stock needs using AI/ML[cite: 7, 8, 9].

**Problem:** Fast-commerce requires hyperlocal inventory management for 10-15 minute deliveries. [cite_start]Platforms struggle with real-time demand tracking, avoiding under/overstocking, and forecasting future needs[cite: 11, 12, 13, 14]. [cite_start]This project builds a system to visualize, analyze, and act on inventory movements per location[cite: 15].

## ✨ Core Features

* [cite_start]**Demand Heatmaps:** Visualize real-time demand intensity per pincode on an interactive map[cite: 52, 53].
* [cite_start]**Inventory Tracking:** Monitor stock levels for each SKU at every store location[cite: 32, 33, 34, 37, 38].
* [cite_start]**Auto-Restocking:** Trigger replenishment based on defined inventory thresholds and business rules[cite: 64, 65, 66, 67].
* [cite_start]**AI Demand Prediction (Bonus):** Forecast future stock needs using time series models[cite: 71, 72, 73, 74].
* [cite_start]**Admin Dashboard:** Centralized view for store locations, heatmap intensity, low-stock alerts, and restock plans[cite: 81, 82, 84, 85, 86, 87].

## 🛠️ Tech Stack

* [cite_start]**Frontend:** React.js, Leaflet.js [cite: 19, 20]
* [cite_start]**Backend:** Python (FastAPI) [cite: 21, 22]
* [cite_start]**Database:** MongoDB [cite: 23, 24]
* [cite_start]**AI/ML:** Python (Prophet / ARIMA) [cite: 27, 28]
* [cite_start]**Scheduler:** Celery (Python) / Node Cron (Node.js) [cite: 29, 30]

## 📂 Project Structure

This project is organized into modular components to facilitate collaborative development:

* `FastCommerceHeatmap/` (Project Root)
    * `.venv/` - Python virtual environment (ignored by Git)
    * `.gitignore` - Files to ignore in Git
    * `README.md` - Project overview (you are here!)
    * `requirements.txt` - Python dependencies
    * `package.json` - Frontend Node.js dependencies
    * `data/` - Raw and mock data files
        * `raw_inventory_data.csv` - Placeholder for initial inventory data
        * `mock_order_data.json` - Placeholder for mock order data for demand tracking
    * `backend/` - Python FastAPI application
        * `__init__.py`
        * `main.py` - Main FastAPI application entry point
        * `api/` - API routes
            * `__init__.py`
            * `inventory_routes.py` - CRUD operations for inventory
            * `demand_routes.py` - Endpoints for demand data
        * `services/` - Business logic
            * `__init__.py`
            * `inventory_service.py` - Logic for inventory management
            * `demand_service.py` - Logic for demand calculation
        * `database/` - DB connection & models
            * `__init__.py`
            * `mongodb_config.py` - MongoDB connection setup
        * `models/` - Pydantic Models for API Request/Response Validation
            * `__init__.py`
            * `inventory_model.py` - Data models for inventory items
            * `demand_model.py` - Data models for demand
        * `utils/` - Helper functions
            * `__init__.py`
            * `helpers.py`
    * `frontend/` - React.js UI application
        * `public/` - Static assets
            * `index.html`
        * `src/` - React source code
            * `App.js` - Main React application component
            * `index.js` - React entry point
            * `components/` - Reusable UI Components
                * `MapComponent.js` - Leaflet map integration
                * `Dashboard.js` - Admin dashboard layout
                * `AlertDisplay.js` - For low-stock alerts
            * `pages/` - Page-specific Components/Views
                * `InventoryView.js`
                * `HeatmapView.js`
            * `services/` - Frontend API Calls
                * `api.js` - Functions to interact with backend APIs
            * `styles/` - CSS Files (e.g., Tailwind CSS setup)
                * `index.css`
            * `utils/` - Frontend Utility Functions
                * `helpers.js`
        * `.env.development` - Environment variables for development
    * `ml_model/` - Python AI/ML module
        * `__init__.py`
        * `train_model.py` - Script for training the demand prediction model
        * `predict_demand.py` - Script for making predictions
        * `models/` - Saved trained models
        * `data_prep/` - Data preparation scripts
            * `__init__.py`
            * `feature_engineering.py`
    * `scheduler/` - Automated task scheduling
        * `__init__.py` - (If Python based)
        * `celery_worker.py` - (If Python Celery)
        * `cron_jobs.js` - (If Node Cron)
        * `config.py`
    * `tests/` - Unit and Integration Tests
        * `backend/`
            * `__init__.py`
            * `test_inventory_service.py`
        * `frontend/`
            * `__init__.py`
            * `test_map_component.js`
        * `ml_model/`
            * `__init__.py`
            * `test_predict_demand.py`
    * `docs/` - Project documentation & diagrams
        * `architecture.md`
        * `api_spec.md`


## 🚀 Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/FastCommerceHeatmap.git](https://github.com/YOUR_USERNAME/FastCommerceHeatmap.git)
    cd FastCommerceHeatmap
    ```
2.  **Backend Setup:**
    ```bash
    python -m venv .venv
    # Activate virtual environment
    pip install -r requirements.txt
    # Configure MongoDB connection in backend/database/mongodb_config.py
    ```
3.  **Frontend Setup:**
    ```bash
    cd frontend
    npm install # or yarn install
    # Configure backend API URL in frontend/.env.development
    ```

## ▶️ How to Run

* **Backend:** `uvicorn backend.main:app --reload` (from project root after venv activation)
* **Frontend:** `npm start` (from `frontend/` directory)

---

## 🤝 Team Roles & Responsibilities

[cite_start]Our team is structured to efficiently tackle this multi-faceted project[cite: 101, 102]:

* [cite_start]**Frontend Developer:** Builds the interactive heatmap and Admin Dashboard using React + Leaflet[cite: 104, 105].
* [cite_start]**Backend Developer:** Creates APIs for inventory and demand, and manages database interactions[cite: 106, 107].
* [cite_start]**ML Engineer:** Designs and trains the demand prediction model[cite: 108, 109].
* [cite_start]**Integration/DevOps Specialist:** Handles scheduled triggers, deployment, and logging[cite: 110].

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
