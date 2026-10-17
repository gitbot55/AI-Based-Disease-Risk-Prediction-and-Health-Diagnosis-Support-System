# AI-Based Disease Risk Prediction & Health Diagnosis Support System

An educational machine learning project for diabetes risk prediction using health indicators, with a Flask backend and React/TypeScript frontend.

## Features
- Machine learning prediction service
- Health-indicator feature validation
- Risk assessment and recommendation components
- Model evaluation outputs

## Technology Stack
- Python, Flask, pandas, NumPy, scikit-learn
- React, TypeScript, Vite

## Setup

### Backend
From the project root, install dependencies with `python -m pip install -r backend/requirements.txt`. Configure required environment variables locally, then follow the backend application instructions to start the Flask server.

### Frontend
Run `cd Frontend`, `npm install`, then `npm run dev`. The frontend API client defaults to `http://127.0.0.1:5000`; configure `VITE_API_BASE_URL` if needed.

## Dataset
The training scripts expect CSV files in `backend/dataset/`. Verify the original dataset source and license before redistributing dataset files publicly.

## Limitations
This project is for educational and research purposes only. Its predictions are not a medical diagnosis and should not replace professional medical advice.

## License
A license has not yet been specified. Confirm the licensing of the code, model, and dataset before public redistribution.
