# 📊 Interactive Dashboard for Consumer Finance Clustering

# 📦 Import required libraries
import pandas as pd
import os
import plotly.express as px
from dash import Input, Output, dcc, html
from dash import Dash
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 📁 Load and filter the data
def wrangle(filepath):  
    """Read SCF data file into a DataFrame.

    Filters for credit-fearful households with net worth below $2 million.
    """
    df = pd.read_csv(filepath)
    mask = (df["TURNFEAR"] == 1) & (df["NETWORTH"] < 2e6)
    # Note: You might want to apply the mask here: df = df[mask]
    return df

# Load the dataset
df = wrangle("data/SCFP2019.csv.gz")
print("df type:", type(df))
print("df shape:", df.shape)
df.head()

# 🚀 Initialize Dash app
app = Dash(__name__)
print("app type:", type(app))

# 🖼️ Define the layout of the dashboard
app.layout = html.Div([
    
    # Dashboard title
    html.H1("Survey of Consumer Finances"),

    # Bar chart section for high variance features
    html.H2("High Variance Features"),
    dcc.Graph(id="bar-chart"),

    # Radio buttons to toggle between trimmed and untrimmed variance
    dcc.RadioItems(
        options=[
            {"label": "Trimmed", "value": True},
            {"label": "Not Trimmed", "value": False}
        ],
        value=True,
        id="trim-button"
    ),

    # Clustering section
    html.H2("K-means Clustering"),
    html.H3("Number of Clusters (k)"),
    dcc.Slider(min=2, max=12, step=1, value=2, id="k-slider"),

    # Metrics (inertia and silhouette score) display
    html.Div(id="metrics")
])

# 🔍 Function to compute top 5 high-variance features
def get_high_var_features(trimmed=True, return_feat_names=True): 
    """Returns the five highest-variance features in the dataset."""
    
    if trimmed:
        # Remove bottom and top 10% of values before calculating variance
        top_five_features = df.apply(trimmed_var).sort_values().tail(5)
    else:
        # Use standard variance
        top_five_features = df.var().sort_values().tail(5)

    # Return either just the names or the full series
    if return_feat_names:
        return top_five_features.index.tolist()
    return top_five_features

# 📊 Callback to update the variance bar chart
@app.callback(
    Output("bar-chart", "figure"),
    Input("trim-button", "value")
)
def serve_bar_chart(trimmed=True): 
    """Returns bar chart of top 5 high-variance features."""
    
    top_features = get_high_var_features(trimmed=trimmed, return_feat_names=False)
    fig = px.bar(
        x=top_features, y=top_features.index,
        orientation="h", title="Top 5 High-Variance Features"
    )
    fig.update_layout(xaxis_title="Variance", yaxis_title="Feature")
    return fig

# 🤖 Function to build KMeans model and compute metrics
def get_model_metrics(trimmed=True, k=2, return_metrics=False):  
    """Builds a KMeans model and optionally returns its metrics."""

    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]

    # Scale features and apply KMeans
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
    model.fit(X)

    # Return either just the model or evaluation metrics
    if return_metrics:
        inertia = model.named_steps["kmeans"].inertia_
        silhouette = silhouette_score(X, model.named_steps["kmeans"].labels_)
        return {
            "inertia": round(inertia),
            "silhouette": round(silhouette, 3)
        }
    return model

# 🧪 Callback to show model metrics (inertia & silhouette)
@app.callback(
    Output("metrics", "children"),
    Input("trim-button", "value"),     # ✅ fixed typo here
    Input("k-slider", "value")
)
def serve_metrics(trimmed=True, k=2):  
    """Returns inertia and silhouette score for KMeans model as text."""
    
    metrics = get_model_metrics(trimmed=trimmed, k=k, return_metrics=True)
    return [
        html.H3(f"Inertia: {metrics['inertia']}"),
        html.H3(f"Silhouette Score: {metrics['silhouette']}")
    ]

# 🌐 Run the app in a hosted environment (e.g., DataCamp or Jupyter)

