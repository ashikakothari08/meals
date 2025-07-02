# Healthy Meal Subscription Dashboard

This Streamlit dashboard provides end‑to‑end insight generation for the synthetic **Healthy Meal Subscription** survey dataset.

## Features

1. **Data Visualisation** – 10+ descriptive charts with narrative insights  
2. **Classification Lab** – K‑NN, Decision Tree, Random Forest, Gradient Boosting  
   * Metrics table (accuracy, precision, recall, F1)  
   * Toggle for confusion matrix  
   * Combined ROC curves  
   * Upload new data → predict → download results  
3. **Clustering Studio** – interactive k‑means (k=2‑10), elbow plot, persona summary, downloadable labels  
4. **Association Rule Mining** – Apriori with confidence filter & parameter controls  
5. **Regression Corner** – Linear, Ridge, Lasso, Decision‑Tree regressors with quick insights

## Quick start (local)

```bash
git clone <your‑repo>
cd healthy_meal_dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub (public repo works best).  
2. Go to **share.streamlit.io**, link the repo, and set **`app.py`** as the main file.  
3. Add any `SECRETS` if needed (none for this app).  
4. Click **Deploy** – done! 🎉

## File layout

```
healthy_meal_dashboard/
├── app.py
├── healthy_meal_subscriptions_synthetic.csv
├── requirements.txt
└── README.md
```

Feel free to tweak visual themes, add more models, or plug in real survey data.
