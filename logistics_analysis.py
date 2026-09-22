"""
Strategic Planning and Data Exploration in Logistics
Week 1 Task - Logistics Data Analyst Internship

This script illustrates the proposed analytical approach described in the
strategic planning report: data loading & cleaning, KPI calculation,
exploratory data analysis, demand forecasting (regression), delivery-zone
segmentation (clustering), and a route-optimisation sketch.

NOTE: This is illustrative code written against an example schema
(shipments.csv). Replace the file path / column names with your actual
dataset before running.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# 1. Data Loading and Cleaning
# ---------------------------------------------------------------------------
def load_and_clean(path="shipments.csv"):
    """Load raw shipment data and apply basic cleaning."""
    df = pd.read_csv(path, parse_dates=["order_time", "delivery_time", "promised_time"])

    df = df.drop_duplicates()
    df = df.dropna(subset=["order_time", "delivery_time", "distance_km"])

    # Delay in minutes: negative/zero means on-time or early
    df["delay_minutes"] = (
        df["delivery_time"] - df["promised_time"]
    ).dt.total_seconds() / 60

    return df


# ---------------------------------------------------------------------------
# 2. KPI Calculation
# ---------------------------------------------------------------------------
def calculate_kpis(df):
    """Compute the four KPIs defined in the strategic planning report."""
    otd_rate = (df["delay_minutes"] <= 0).mean() * 100
    avg_cost = df["delivery_cost"].mean()
    first_attempt_rate = (df["delivery_attempts"] == 1).mean() * 100
    inventory_turnover = df["cogs"].sum() / df["avg_inventory_value"].mean()

    kpis = {
        "On-Time Delivery Rate (%)": round(otd_rate, 2),
        "Average Delivery Cost per Shipment (Rs.)": round(avg_cost, 2),
        "First-Attempt Delivery Success Rate (%)": round(first_attempt_rate, 2),
        "Inventory Turnover Ratio": round(inventory_turnover, 2),
    }

    for name, value in kpis.items():
        print(f"{name}: {value}")

    return kpis


# ---------------------------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------------------------
def plot_delay_by_zone(df):
    """Visualise average delivery delay by zone."""
    zone_delay = df.groupby("zone")["delay_minutes"].mean().sort_values()
    zone_delay.plot(kind="bar", title="Average Delay by Zone")
    plt.ylabel("Delay (minutes)")
    plt.tight_layout()
    plt.savefig("delay_by_zone.png")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Demand / Delivery-Time Forecasting (Regression)
# ---------------------------------------------------------------------------
def train_delivery_time_model(df):
    """Train a simple linear regression to predict delivery time."""
    features = ["distance_km", "order_hour", "is_weekend", "past_7day_avg_orders"]
    X = df[features]
    y = df["delivery_time_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    print(f"Delivery-time model RMSE: {rmse:.2f} minutes")
    return model


# ---------------------------------------------------------------------------
# 5. Delivery Zone Segmentation (Clustering)
# ---------------------------------------------------------------------------
def cluster_zones(df, n_clusters=5):
    """Group delivery locations into zones with similar characteristics."""
    zone_features = df[["latitude", "longitude", "avg_orders_per_day"]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["zone_cluster"] = kmeans.fit_predict(zone_features)
    return df, kmeans


# ---------------------------------------------------------------------------
# 6. Route Optimisation (conceptual sketch)
# ---------------------------------------------------------------------------
def optimise_routes(df, warehouse_location):
    """
    Conceptual sketch of route optimisation per cluster.
    A real implementation would use a solver such as Google OR-Tools
    or PuLP to solve a vehicle-routing problem (VRP) per zone_cluster.
    """
    routes = {}
    for cluster_id, stops in df.groupby("zone_cluster"):
        # Placeholder: nearest-neighbour ordering by distance from warehouse
        # Replace with an actual VRP solver call, e.g. OR-Tools RoutingModel.
        ordered_stops = stops.sort_values("distance_km")
        routes[cluster_id] = ordered_stops[["order_id", "latitude", "longitude"]]

    return routes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_and_clean("shipments.csv")

    calculate_kpis(data)
    plot_delay_by_zone(data)

    model = train_delivery_time_model(data)
    data, kmeans_model = cluster_zones(data)

    warehouse = {"latitude": 28.7041, "longitude": 77.1025}  # example: Delhi
    routes = optimise_routes(data, warehouse)

    print(f"\nGenerated routes for {len(routes)} delivery zones.")
