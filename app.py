
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import io

st.set_page_config(page_title="Healthy Meal Subscription Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("healthy_meal_subscriptions_synthetic.csv")

df = load_data()

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def num_cat_split(df_):
    cat_cols = df_.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df_.select_dtypes(include=['int64','float64']).columns.tolist()
    return num_cols, cat_cols

def classification_pipeline(model):
    X = df.drop(columns=['trial_intent'])
    y = (df['trial_intent'] >= 4).astype(int)  # 1 = willing
    num_cols, cat_cols = num_cat_split(X)
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    pipe = Pipeline([('pre', pre), ('model', model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:,1]
    else:
        # fall back to decision_function or zeros
        try:
            y_proba = pipe.decision_function(X_test)
        except:
            y_proba = np.zeros_like(y_pred, dtype=float)
    metrics = dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred)
    )
    return pipe, metrics, y_test, y_pred, y_proba

def regression_pipeline(model):
    X = df.drop(columns=['max_daily_budget_aed'])
    y = df['max_daily_budget_aed']
    num_cols, cat_cols = num_cat_split(X)
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    pipe = Pipeline([('pre', pre), ('model', model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = pipe.score(X_test, y_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return pipe, r2, rmse, y_test, y_pred

def cluster_summary(df_labeled, cluster_col):
    summary = df_labeled.groupby(cluster_col).agg({
        'age':'mean',
        'income_aed':'mean',
        'nutri_knowledge':'mean',
        'trial_intent':'mean',
        'max_daily_budget_aed':'mean'
    }).round(1).rename(columns={
        'age':'avg_age',
        'income_aed':'avg_income',
        'nutri_knowledge':'avg_nutri_knw',
        'trial_intent':'avg_intent',
        'max_daily_budget_aed':'avg_budget'
    })
    return summary

# ------------------------------------------------------------------
# Streamlit Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Visualisation", "Classification", "Clustering",
    "Association Rules", "Regression"
])

with tab1:
    st.header("Descriptive Insights")
    # Chart 1: Age distribution
    fig1 = px.histogram(df, x='age', nbins=25, title='Age Distribution')
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Income distribution (log scale)
    fig2 = px.histogram(df, x='income_aed', nbins=40, title='Income Distribution (AED)')
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Trial intent bar
    # Build counts for trial intent
counts_df = (df['trial_intent']
             .value_counts()
             .sort_index()
             .rename_axis('TrialIntent')
             .reset_index(name='Count'))
fig3 = px.bar(counts_df,
              x='TrialIntent', y='Count',
              labels={'TrialIntent':'Trial intent (1-5)', 'Count':'Count'},
              title='Willingness to Try Subscription')

    # Chart 4: Wellness goal pie
    fig4 = px.pie(df, names='wellness_goal', title='Primary Wellness Goals')
    st.plotly_chart(fig4, use_container_width=True)

    # Chart 5: Budget vs Income scatter
    fig5 = px.scatter(df, x='income_aed', y='max_daily_budget_aed',
                      trendline='ols',
                      title='Daily Budget vs Monthly Income')
    st.plotly_chart(fig5, use_container_width=True)

    # Chart 6: Heatmap correlations for numeric fields
    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=['int64','float64'])
    corr = num_df.corr()
    fig6, ax6 = plt.subplots()
    sns.heatmap(corr, ax=ax6, annot=True, fmt='.2f')
    st.pyplot(fig6)

    # Chart 7: Order channel distribution
    fig7 = px.bar(df['order_channel'].value_counts().reset_index(),
                  x='index', y='order_channel',
                  labels={'index':'Order Channel', 'order_channel':'Count'},
                  title='Preferred Ordering Channel')
    st.plotly_chart(fig7, use_container_width=True)

    # Chart 8: Pain point frequencies
    st.subheader("Top Pain Points")
    pain_series = df['pain_points'].str.get_dummies(sep=', ').sum().sort_values(ascending=False)
    fig8 = px.bar(pain_series, title='Pain Point Frequency')
    st.plotly_chart(fig8, use_container_width=True)

    # Chart 9: Activity level by Trial Intent
    fig9 = px.box(df, x='activity_level', y='trial_intent',
                  title='Intent by Activity Level')
    st.plotly_chart(fig9, use_container_width=True)

    # Chart 10: Budget by Wellness Goal
    fig10 = px.box(df, x='wellness_goal', y='max_daily_budget_aed',
                   title='Daily Budget by Wellness Goal')
    st.plotly_chart(fig10, use_container_width=True)

with tab2:
    st.header("Classification Laboratory")
    alg_option = st.selectbox("Choose algorithm", ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"])
    model_map = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    pipe, metrics, y_test, y_pred, y_proba = classification_pipeline(model_map[alg_option])
    st.subheader("Performance Metrics")
    st.dataframe(pd.DataFrame([metrics]))

    if st.checkbox("Show confusion matrix"):
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Willing','Willing'])
        disp.plot(ax=ax_cm)
        st.pyplot(fig_cm)

    st.subheader("ROC Curve (all models)")
    roc_fig, roc_ax = plt.subplots()
    for name, mdl in model_map.items():
        _, _, y_t, _, y_p = classification_pipeline(mdl)
        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_auc = auc(fpr, tpr)
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    roc_ax.plot([0,1],[0,1],'--')
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.legend()
    st.pyplot(roc_fig)

    st.divider()
    st.subheader("Predict on New Data")
    new_file = st.file_uploader("Upload CSV without 'trial_intent' column", type='csv')
    if new_file:
        new_df = pd.read_csv(new_file)
        preds = pipe.predict(new_df)
        result_df = new_df.copy()
        result_df['predicted_trial_intent_class'] = preds
        st.dataframe(result_df.head())
        download = st.download_button(
            "Download predictions",
            data=result_df.to_csv(index=False).encode('utf-8'),
            file_name="subscription_predictions.csv",
            mime='text/csv'
        )

with tab3:
    st.header("Customer Segmentation (k‑means)")
    num_cols, cat_cols = num_cat_split(df)
    k_default = 4
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=k_default)
    # Simple pre‑processing: encode categoricals
    df_enc = pd.get_dummies(df.drop(columns=['open_comment']), drop_first=True)
    inertia = []
    for k_i in range(2,11):
        km_i = KMeans(n_clusters=k_i, random_state=42, n_init='auto')
        km_i.fit(df_enc)
        inertia.append(km_i.inertia_)
    elbow_fig = px.line(x=list(range(2,11)), y=inertia,
                        labels={'x':'k', 'y':'Inertia'},
                        title='Elbow Plot')
    st.plotly_chart(elbow_fig, use_container_width=True)

    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = km.fit_predict(df_enc)
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    st.subheader("Cluster Persona Summary")
    persona_tbl = cluster_summary(df_clustered, 'cluster')
    st.dataframe(persona_tbl)

    st.download_button(
        "Download data with clusters",
        data=df_clustered.to_csv(index=False).encode('utf-8'),
        file_name='subscription_clustered.csv',
        mime='text/csv'
    )

with tab4:
    st.header("Association Rule Mining")
    cols_options = st.multiselect(
        "Select columns for mining (must be comma‑separated multi‑select fields)",
        options=['pain_points', 'convince_factors'],
        default=['pain_points', 'convince_factors']
    )
    min_support = st.slider("Min support", 0.01, 0.5, 0.05, step=0.01)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.4, step=0.05)

    # Prepare transactions
    transactions = []
    for _, row in df.iterrows():
        basket = []
        for col in cols_options:
            if pd.notnull(row[col]) and row[col]:
                basket.extend([x.strip() for x in row[col].split(',')])
        transactions.append(basket)

    te = TransactionEncoder()
    trans_array = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(trans_array, columns=te.columns_)

    frequent = apriori(trans_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric='confidence', min_threshold=min_conf)
    rules_sorted = rules.sort_values('confidence', ascending=False).head(10)
    st.dataframe(rules_sorted[['antecedents','consequents','support','confidence','lift']])

with tab5:
    st.header("Regression Playground")
    reg_choice = st.selectbox(
        "Choose regressor",
        ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor"]
    )
    reg_map = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
    }
    r_pipe, r2, rmse, y_t, y_p = regression_pipeline(reg_map[reg_choice])
    st.metric("R-squared", f"{r2:.3f}")
    st.metric("RMSE", f"{rmse:.2f} AED")

    fig_reg = px.scatter(x=y_t, y=y_p,
                         labels={'x':'Actual Budget', 'y':'Predicted Budget'},
                         title='Actual vs Predicted Budget')
    st.plotly_chart(fig_reg, use_container_width=True)

    st.caption("Note: budget predictions help gauge realistic price points for new user segments.")
