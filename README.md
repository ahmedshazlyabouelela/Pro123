# Air Quality Classification System

Machine Learning project for classifying air quality into:

- Healthy
- Not Healthy
- Toxic

The final model is an optimized LightGBM classifier using:

- Pollution measurements
- Weather measurements
- Monitoring station
- Wind direction
- Cyclical time features
- 1-hour and 2-hour lag features

## Run locally

pip install -r requirements.txt

streamlit run streamlit_app.py

## Deployment

The project can be deployed using Streamlit Community Cloud.

Main file:

streamlit_app.py