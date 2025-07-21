Fast-Commerce Inventory Heatmap
Project Goal (Elevator Pitch)
Design a map-based inventory management system for a fast-commerce platform (like Blinkit or Instamart) that:

Tracks and visualizes demand zone-wise (pincode-wise) on a heatmap.

Automatically triggers replenishment based on demand trends.

(Bonus) Predicts next-day stock needs using AI/ML.

Problem Statement
Fast-commerce platforms need hyperlocal inventory management to deliver within 10-15 minutes. They operate dark stores in specific areas and must:

Track real-time demand across regions.

Avoid understocking/overstocking.

Forecast future demand.

This project aims to build a system that helps visualize, analyze, and act on inventory movements per location, using smart UI and AI.

Core Features
Inventory Per Location:

Each store (warehouse/dark store) is mapped to a pincode.

Maintains a list of inventory SKUs (Stock Keeping Units) and current stock levels for each item.

Example MongoDB Schema:

{
  "store_id": "DEL123",
  "pincode": "110037",
  "inventory": {
    "Milk": 23,
    "Eggs": 5,
    "Bread": 12
  }
}

Heatmap of Demand Per Pincode:

Uses Leaflet.js to render a map of serviceable pincodes.

Pincodes are color-coded based on demand intensity (High, Medium, Low).

Demand is calculated using past 24-48 hours of orders per pincode per item, aggregated in real-time or hourly.

Auto-Restocking Logic:

Threshold-based: If inventory drops below a defined threshold, a restock is triggered.

Business Rules: Allows defining specific rules (e.g., "If Eggs < 10 in pincode 110037, send restock request of 50 units").

Scheduled restock checks (e.g., hourly using a cron job).

Predictive AI (Bonus Feature):

Utilizes time series forecasting (Prophet / ARIMA) to predict:

Which items will likely be in high demand tomorrow.

How much to stock today to meet tomorrow's demand.

Model Inputs: Daily sales per item per pincode, Day-of-week, holidays, weather (optional).

Model Output: Predicted demand auto restock quantity.

Admin Dashboard:

View: All stores on a map, heatmap intensity for each item or overall demand, low-stock alerts, auto-generated restock plans.

Filters: Item-wise heatmap (Milk, Fruits, etc.), Time range filter (Today, Week, Custom).

Example Use Case: Milk in Pincode 400092 (Andheri West) - Inventory today: 7 units, Demand trend: ~35 units/day, SHARP Alert: "Restock 40 units by 9 AM tomorrow".

Suggested Tech Stack
Component

Technology

Frontend UI

React.js + Leaflet.js (for maps)

Backend API

Python (FastAPI)

Database

MongoDB (stores, SKUs, pincodes, inventory)

AI Module

Python + Prophet / ARIMA (time series forecast)

Scheduler

Python (Celery) or Node.js (Node Cron)

Project Structure
FastCommerceHeatmap/
├── .venv/                       # Python virtual environment (ignored by Git)
├── .gitignore                   # Specifies files/folders to be ignored by Git
├── README.md                    # Project overview, setup, usage, and team roles
├── requirements.txt             # Python backend dependencies
├── package.json                 # Frontend Node.js dependencies
├── data/                        # Stores raw and processed data files
│   ├── raw_inventory_data.csv   # Placeholder for initial inventory data
│   └── mock_order_data.json     # Placeholder for mock order data for demand tracking
├── backend/                     # Python FastAPI backend
│   ├── __init__.py              # Marks 'backend' as a Python package
│   ├── main.py                  # Main FastAPI application entry point
│   ├── api/                     # API endpoints
│   │   ├── __init__.py
│   │   └── inventory_routes.py  # Endpoints for inventory (GET, POST, PUT)
│   │   └── demand_routes.py     # Endpoints for demand data
│   ├── services/                # Business logic and data interactions
│   │   ├── __init__.py
│   │   ├── inventory_service.py # Logic for inventory management
│   │   └── demand_service.py    # Logic for demand calculation
│   ├── database/                # Database connection and schema
│   │   ├── __init__.py
│   │   └── mongodb_config.py    # MongoDB connection setup
│   ├── models/                  # Pydantic models for API request/response
│   │   ├── __init__.py
│   │   └── inventory_model.py   # Data models for inventory items
│   │   └── demand_model.py      # Data models for demand
│   └── utils/                   # Utility functions (e.g., data validation)
│       └── __init__.py
│       └── helpers.py
├── frontend/                    # React.js + Leaflet.js UI
│   ├── public/                  # Static assets
│   │   └── index.html
│   ├── src/                     # React source code
│   │   ├── App.js               # Main React application component
│   │   ├── index.js             # React entry point
│   │   ├── components/          # Reusable UI components
│   │   │   ├── MapComponent.js  # Leaflet map integration
│   │   │   └── Dashboard.js     # Admin dashboard layout
│   │   │   └── AlertDisplay.js  # For low-stock alerts
│   │   ├── pages/               # Page-specific components/views
│   │   │   ├── InventoryView.js
│   │   │   └── HeatmapView.js
│   │   ├── services/            # Frontend API calls
│   │   │   └── api.js           # Functions to interact with backend APIs
│   │   ├── styles/              # CSS files (e.g., Tailwind CSS setup)
│   │   │   └── index.css
│   │   └── utils/               # Frontend utility functions
│   │       └── helpers.js
│   └── .env.development         # Environment variables for development
│   └── .env.production          # Environment variables for production
├── ml_model/                    # Python for AI/ML module (Bonus Feature)
│   ├── __init__.py
│   ├── train_model.py           # Script for training the demand prediction model
│   ├── predict_demand.py        # Script for making predictions
│   ├── models/                  # Directory to save trained models
│   │   └── prophet_model.pkl    # Example: A saved Prophet model
│   └── data_prep/               # Scripts for ML-specific data preparation
│       └── __init__.py
│       └── feature_engineering.py
├── scheduler/                   # For auto-restock triggers (Node Cron or Celery)
│   ├── __init__.py              # (If Python Celery)
│   ├── celery_worker.py         # (If Python Celery)
│   ├── cron_jobs.js             # (If Node Cron)
│   └── config.py                # Scheduler configuration
└── docs/                        # Project documentation, diagrams, API specs
    └── architecture.md
    └── api_spec.md

Setup Instructions
Follow these steps to get the project running locally.

1. Clone the Repository
git clone [https://github.com/YOUR_USERNAME/FastCommerceHeatmap.git](https://github.com/YOUR_USERNAME/FastCommerceHeatmap.git)
cd FastCommerceHeatmap

Replace YOUR_USERNAME with your actual GitHub username.

2. Backend Setup (Python)
Create a Python Virtual Environment:

python -m venv .venv

Activate the Virtual Environment:

On Windows:

.\.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate

Install Backend Dependencies:

pip install -r requirements.txt

Database (MongoDB) Setup:

Ensure you have MongoDB installed and running locally, or connect to a cloud MongoDB Atlas instance.

Update backend/database/mongodb_config.py with your MongoDB connection string.

3. Frontend Setup (React.js)
Navigate to the Frontend Directory:

cd frontend

Install Frontend Dependencies:

npm install # or yarn install

Configure Environment Variables:

Create a .env.development file in the frontend/ directory.

Add your backend API URL (e.g., REACT_APP_BACKEND_URL=http://localhost:8000).

4. ML Model Setup (Python - Optional)
Ensure your virtual environment is active (from Backend Setup).

Install ML-specific dependencies (already in requirements.txt).

Data Preparation: Ensure data/raw_inventory_data.csv and data/mock_order_data.json are populated with sample data for training and prediction.

5. Scheduler Setup (Optional - Choose one based on your preference)
If using Python (Celery):

Ensure Celery and a message broker (like Redis or RabbitMQ) are installed and configured.

Refer to scheduler/celery_worker.py for setup details.

If using Node.js (Node Cron):

Ensure Node.js is installed.

Refer to scheduler/cron_jobs.js for setup details.

How to Run the Application
1. Start the Backend API
Open a new terminal session.

Navigate to the project root: cd FastCommerceHeatmap

Activate your virtual environment:

Windows: .\.venv\Scripts\activate

macOS/Linux: source .venv/bin/activate

Run the FastAPI application:

uvicorn backend.main:app --reload

The API will typically run on http://localhost:8000.

2. Start the Frontend Application
Open a separate terminal session.

Navigate to the frontend directory: cd FastCommerceHeatmap/frontend

Start the React development server:

npm start # or yarn start

The frontend will typically open in your browser at http://localhost:3000.

3. Run ML Model (Training/Prediction - as needed)
Open a new terminal session.

Navigate to the project root: cd FastCommerceHeatmap

Activate your virtual environment.

To train the model:

python ml_model/train_model.py

To make predictions:

python ml_model/predict_demand.py

4. Start Scheduler (as needed)
If using Celery:

celery -A scheduler.celery_worker worker -l info

If using Node Cron:

node scheduler/cron_jobs.js

Team Roles & Responsibilities
Frontend Dev: Builds the heatmap using React + Leaflet, develops the Admin Dashboard UI.

Backend Dev: Builds APIs for inventory and demand handling using FastAPI, manages MongoDB interactions.

ML Engineer: Trains and deploys the demand prediction model.

Integration/DevOps: Sets up scheduled triggers (cron/Celery), manages deployment, logging, and overall system integration.

Contribution Guidelines
We welcome contributions! Please follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/your-feature-name).

Make your changes.