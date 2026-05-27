import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import joblib

st.set_page_config(
    page_title="Walmart Sales Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)




store_names = {
    1: "Mumbai Central",
    2: "Ahmedabad Plaza",
    3: "Delhi Hub",
    4: "Bangalore Mart",
    5: "Chennai Square",
    6: "Hyderabad Center",
    7: "Pune Market",
    8: "Surat Mall",
    9: "Jaipur Outlet",
    10: "Kolkata Store"
}

dept_names = {
    1: "Grocery",
    2: "Electronics",
    3: "Clothing",
    4: "Home & Kitchen",
    5: "Sports",
    6: "Beauty",
    7: "Toys",
    8: "Automotive",
    9: "Pharmacy",
    10: "Stationery"
}




def load_models():
    try:
        rf = joblib.load("model/model.pkl")
        xgb = joblib.load("model/XGboost.pkl")
        return rf, xgb
    except:
        st.error("❌ Model files not found. Please train models first.")
        st.stop()

rf_model, xgb_model = load_models()


model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])

if model_choice == "Random Forest":
    model = rf_model
else:
    model = xgb_model







# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2a2d3e);
        border-radius: 12px; padding: 20px;
        border-left: 4px solid #4f8bf9; margin-bottom: 10px;
    }
    h2 { color: #4f8bf9 !important; font-weight: 700; }
    h3 { color: #e0e0e0 !important; }
    .insight-box {
        background-color: #1e2130;
        border: 1px solid #4f8bf9;
        border-radius: 8px; padding: 15px; margin: 8px 0;
        font-size: 14px; color: #cfcfcf;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", parse_dates=["Date"])

    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]   = df["Date"].dt.quarter
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    store_avg  = df.groupby("Store")["Weekly_Sales"].mean()
    def tier(s):
        if s >= store_avg.quantile(0.67): return "Top Performer"
        elif s >= store_avg.quantile(0.33): return "Mid Performer"
        else: return "Low Performer"
    store_tier = store_avg.apply(tier).rename("StoreTier")
    df = df.merge(store_tier, on="Store")
    return df


df = load_data()
df["StoreName"] = df["Store"].map(store_names)
df["DeptName"] = df["Dept"].map(dept_names)
# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Walmart Analytics")
    st.markdown("---")
    st.markdown("### 🔍 Filters")

    year_options    = sorted(df["Year"].unique())
    selected_years  = st.multiselect("📅 Year", year_options, default=year_options)

    store_options   = sorted(df["Store"].unique())
    selected_stores = st.multiselect("🏪 Store", store_options, default=store_options[:10])

    holiday_filter  = st.radio("🎄 Holiday Weeks", ["All", "Holiday Only", "Non-Holiday Only"])

    st.markdown("---")
    st.info("""
    **Dataset:** Walmart Weekly Sales  
    **Period:** 2010 – 2012  
    **Stores:** 45 | **Depts:** 81  
    **Records:** 421,570  
    """)

# Apply filters
fdf = df[df["Year"].isin(selected_years) & df["Store"].isin(selected_stores)]
if holiday_filter == "Holiday Only":
    fdf = fdf[fdf["IsHoliday"] == 1]
elif holiday_filter == "Non-Holiday Only":
    fdf = fdf[fdf["IsHoliday"] == 0]

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#4f8bf9; font-size:2.4rem; margin-bottom:0;'>
    🛒 Walmart Retail Sales Analytics Dashboard
</h1>
<p style='text-align:center; color:#9e9e9e; font-size:1rem; margin-top:4px;'>
    End-to-End Retail Intelligence · 45 Stores · 81 Departments · 2010–2012
</p>
<hr style='border:1px solid #2a2d3e;'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total_sales = fdf["Weekly_Sales"].sum()
avg_weekly  = fdf["Weekly_Sales"].mean()
top_store   = fdf.groupby("Store")["Weekly_Sales"].sum().idxmax()
holiday_wks = fdf[fdf["IsHoliday"]==1]["Date"].nunique()

k1.metric("💰 Total Sales",      f"${total_sales/1e9:.2f}B")
k2.metric("📊 Avg Weekly Sales", f"${avg_weekly:,.0f}")
k3.metric("🏪 Stores Selected",  f"{fdf['Store'].nunique()}")
k4.metric("🎄 Holiday Weeks",    f"{holiday_wks}")
top_store_label = store_names.get(top_store, f"Store {top_store}")
k5.metric("🏆 Top Store", top_store_label)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Sales Trends",
    "🏪 Store Analysis",
    "📦 Department Insights",
    "🎄 Holiday Impact",
    "🤖 ML Prediction",
    "📋 Raw Data & Stats",
])


# ══════════════════════════════════════════
# TAB 1 — SALES TRENDS
# ══════════════════════════════════════════
with tab1:
    st.markdown("## 📈 Sales Trends Over Time")

    c1, c2 = st.columns(2)
    with c1:
        monthly = fdf.groupby("YearMonth")["Weekly_Sales"].sum().reset_index()
        monthly.columns = ["Month","Total_Sales"]
        fig = px.area(monthly, x="Month", y="Total_Sales",
                      title="📅 Monthly Total Sales", template="plotly_dark",
                      color_discrete_sequence=["#4f8bf9"])
        fig.update_traces(fill="tozeroy")
        fig.update_layout(height=360, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(fdf, x="Weekly_Sales", nbins=80,
                            title="📊 Weekly Sales Distribution",
                            color_discrete_sequence=["#f97b4f"],
                            template="plotly_dark")
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)

    yearly = fdf.groupby(["Year","Month"])["Weekly_Sales"].sum().reset_index()
    fig3 = px.line(yearly, x="Month", y="Weekly_Sales", color="Year",
                   title="📆 Year-over-Year Monthly Sales", markers=True,
                   template="plotly_dark",
                   color_discrete_sequence=["#4f8bf9","#f97b4f","#4fdb8b"])
    fig3.update_layout(height=380, xaxis=dict(
        tickvals=list(range(1,13)),
        ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]))
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(fdf, x="Quarter", y="Weekly_Sales", color="Quarter",
                  title="📦 Sales Spread by Quarter", template="plotly_dark")
    fig4.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    &nbsp;&nbsp;• Sales peak in <b>Q4 (Oct–Dec)</b> driven by holiday shopping season.<br>
    &nbsp;&nbsp;• <b>February</b> shows elevated sales due to Super Bowl week (holiday flag).<br>
    &nbsp;&nbsp;• Clear year-over-year growth from 2010 → 2011; 2012 data is partial (through Oct).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("🚨 Alerts")
    store = st.selectbox(
        "Select Store for Alert",
        sorted(df["Store"].unique()),
        format_func=lambda x: store_names.get(x, f"Store {x}")
    )
    
    dept = st.selectbox(
        "Select Dept for Alert",
        sorted(df["Dept"].unique()),
        format_func=lambda x: dept_names.get(x, f"Dept {x}")
    )
    alert_df = df[(df["Store"] == store) & (df["Dept"] == dept)]
    sorted_df = alert_df.sort_values("Date")
    
    if len(sorted_df) < 8:
        st.warning("Not enough data for trend analysis")
    else:
        recent = sorted_df.tail(4)["Weekly_Sales"].mean()
        previous = sorted_df.tail(8).head(4)["Weekly_Sales"].mean()
    
        change = (recent - previous) / (previous + 1e-6)
    
        st.write("Recent Avg:", recent)
        st.write("Previous Avg:", previous)
        st.write("Change %:", change * 100)
    
        if change < -0.3:
            st.warning("⚠️ Strong downward trend detected")
        elif change > 0.3:
            st.success("🔥 Strong upward trend detected")
        else:
            st.info("Sales are stable")
    st.write("Recent Avg:", recent)
    st.write("Previous Avg:", previous)
    st.write("Change %:", change * 100)
    
    if change < -0.3:
        st.warning("⚠️ Strong downward trend detected")
    elif change > 0.3:
        st.success("🔥 Strong upward trend detected")
    else:
        st.info("Sales are stable")


# ══════════════════════════════════════════
# TAB 2 — STORE ANALYSIS
# ══════════════════════════════════════════
with tab2:
    st.markdown("## 🏪 Store-Level Performance")

    store_summary = fdf.groupby("Store").agg(
        Total_Sales=("Weekly_Sales","sum"),
        Avg_Weekly=("Weekly_Sales","mean"),
        Weeks=("Date","nunique"),
        Depts=("Dept","nunique"),
    ).reset_index().sort_values("Total_Sales", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(store_summary.head(20), x="Store", y="Total_Sales",
                     title="🏆 Top 20 Stores by Total Sales",
                     color="Total_Sales", color_continuous_scale="Blues",
                     template="plotly_dark")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(store_summary, x="Avg_Weekly", y="Total_Sales",
                          size="Weeks", color="Depts",
                          hover_data=["Store"],
                          title="🔵 Avg Weekly vs Total Sales",
                          template="plotly_dark", color_continuous_scale="Viridis")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    pivot = fdf.groupby(["Store","Month"])["Weekly_Sales"].mean().reset_index().pivot(
        index="Store", columns="Month", values="Weekly_Sales")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = month_names[:len(pivot.columns)]

    fig3 = px.imshow(pivot, title="🔥 Store × Month Avg Sales Heatmap",
                     color_continuous_scale="YlOrRd", template="plotly_dark",
                     aspect="auto", labels={"color":"Avg Sales ($)"})
    fig3.update_layout(height=550)
    st.plotly_chart(fig3, use_container_width=True)

    tier_counts = fdf[["Store","StoreTier"]].drop_duplicates()["StoreTier"].value_counts()
    fig4 = px.pie(values=tier_counts.values, names=tier_counts.index,
                  title="🎯 Store Performance Tiers", hole=0.4,
                  color_discrete_sequence=["#4fdb8b","#4f8bf9","#f97b4f"],
                  template="plotly_dark")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    &nbsp;&nbsp;• Top 5 stores account for ~<b>25%</b> of all revenue — strong concentration effect.<br>
    &nbsp;&nbsp;• Higher department count correlates with higher total sales.<br>
    &nbsp;&nbsp;• The heatmap shows seasonal consistency — each store's peak months are predictable.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("🏆 Store Ranking")
    
    sales = fdf.groupby("Store")["Weekly_Sales"].mean()
    consistency = fdf.groupby("Store")["Weekly_Sales"].std()
    score = 0.6 * sales - 0.4 * consistency
    ranking = score.sort_values(ascending=False)
    ranking_df = ranking.reset_index()
    ranking_df.columns = ["Store", "Score"]
    
    st.dataframe(ranking_df.head(10))
    st.markdown("---")
    st.header("📊 Recommendations")
    
    top_store_label = store_names.get(top_store, f"Store {top_store}")
    low_store = ranking_df.iloc[-1]["Store"] 
    low_store_label = store_names.get(low_store, f"Store {low_store}")
    st.success(f"🔥 Invest more in {top_store_label}")
    st.warning(f"⚠️ Improve performance in {low_store_label}")


    st.markdown("""
    💡 Ranking is based on:
    - 📈 Higher average sales
    - 📉 Lower volatility (consistent performance)
    """)
# ══════════════════════════════════════════
# TAB 3 — DEPARTMENT INSIGHTS
# ══════════════════════════════════════════
with tab3:
    st.markdown("## 📦 Department-Level Deep Dive")

    dept_summary = fdf.groupby("Dept").agg(
        Total_Sales=("Weekly_Sales","sum"),
        Avg_Sales=("Weekly_Sales","mean"),
        Stores=("Store","nunique"),
    ).reset_index().sort_values("Total_Sales", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(dept_summary.head(20), x="Dept", y="Total_Sales",
                     title="Top 20 Departments by Total Sales",
                     color="Total_Sales", color_continuous_scale="Greens",
                     template="plotly_dark")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_depts = dept_summary.head(15)["Dept"].tolist()
        fig2 = px.box(fdf[fdf["Dept"].isin(top_depts)],
                      x="Dept", y="Weekly_Sales",
                      title="Sales Variability — Top 15 Depts",
                      color="Dept", template="plotly_dark")
        fig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    dq = fdf.groupby(["Dept","Quarter"])["Weekly_Sales"].mean().reset_index()
    dq_pivot = dq.pivot(index="Dept", columns="Quarter", values="Weekly_Sales")
    dq_pivot.columns = ["Q1","Q2","Q3","Q4"][:len(dq_pivot.columns)]

    fig3 = px.imshow(dq_pivot, title="📊 Department × Quarter Avg Sales",
                     color_continuous_scale="RdYlGn", template="plotly_dark",
                     aspect="auto", labels={"color":"Avg Sales ($)"})
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.bar(dept_summary.tail(10), x="Dept", y="Total_Sales",
                  title="⚠️ Bottom 10 Departments (Low Revenue Alert)",
                  color="Total_Sales", color_continuous_scale="Reds",
                  template="plotly_dark")
    fig4.update_layout(height=340, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    &nbsp;&nbsp;• Top departments (likely Electronics/Seasonal) dominate revenue.<br>
    &nbsp;&nbsp;• High variability depts = opportunity for targeted promotions.<br>
    &nbsp;&nbsp;• Q4 spike is consistent across most departments — universal holiday effect.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 4 — HOLIDAY IMPACT
# ══════════════════════════════════════════
with tab4:
    st.markdown("## 🎄 Holiday vs Non-Holiday Analysis")

    avg_h = fdf.groupby("IsHoliday")["Weekly_Sales"].mean().reset_index()
    avg_h["Label"] = avg_h["IsHoliday"].map({0:"Non-Holiday", 1:"Holiday"})
    h_vals = avg_h.set_index("IsHoliday")["Weekly_Sales"]
    lift = (h_vals[1] / h_vals[0] - 1) * 100

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(avg_h, x="Label", y="Weekly_Sales",
                     title=f"🎄 Holiday Sales Lift: +{lift:.1f}%",
                     color="Label", text="Weekly_Sales",
                     color_discrete_map={"Holiday":"#f97b4f","Non-Holiday":"#4f8bf9"},
                     template="plotly_dark")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fdf["HolidayLabel"] = fdf["IsHoliday"].map({0:"Non-Holiday",1:"Holiday"})
        fig2 = px.violin(fdf, x="HolidayLabel", y="Weekly_Sales",
                         color="HolidayLabel", box=True, points=False,
                         title="Sales Distribution: Holiday vs Non-Holiday",
                         color_discrete_map={"Holiday":"#f97b4f","Non-Holiday":"#4f8bf9"},
                         template="plotly_dark")
        fig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    hol_counts = df.groupby(["Year","IsHoliday"])["Date"].nunique().reset_index()
    hol_counts["Type"] = hol_counts["IsHoliday"].map({0:"Non-Holiday",1:"Holiday"})
    fig3 = px.bar(hol_counts, x="Year", y="Date", color="Type", barmode="group",
                  title="📅 Holiday vs Non-Holiday Weeks per Year",
                  color_discrete_map={"Holiday":"#f97b4f","Non-Holiday":"#4f8bf9"},
                  labels={"Date":"Weeks"}, template="plotly_dark")
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

    store_h = fdf.groupby(["Store","IsHoliday"])["Weekly_Sales"].mean().unstack()
    store_h.columns = ["Non-Holiday","Holiday"]
    store_h["Lift_%"] = (store_h["Holiday"] - store_h["Non-Holiday"]) / store_h["Non-Holiday"] * 100
    store_h = store_h.reset_index().sort_values("Lift_%", ascending=False)
    fig4 = px.bar(store_h, x="Store", y="Lift_%",
                  title="🏪 Holiday Sales Lift by Store (%)",
                  color="Lift_%", color_continuous_scale="RdYlGn",
                  template="plotly_dark")
    fig4.update_layout(height=380)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
    💡 <b>Key Insights:</b><br>
    &nbsp;&nbsp;• Holiday weeks drive an average <b>+{lift:.1f}%</b> sales lift vs non-holiday weeks.<br>
    &nbsp;&nbsp;• Some stores show <b>>30% lift</b> — strong potential for targeted holiday campaigns.<br>
    &nbsp;&nbsp;• Holiday distribution has a heavier right tail — some outlier blockbuster weeks.
    </div>
    """, unsafe_allow_html=True)

def evaluate_models(df, rf_model, xgb_model):
    ml_df = df[["Store","Dept","Year","Month","Week","Quarter","IsHoliday","Weekly_Sales"]].dropna()

    X = ml_df.drop("Weekly_Sales", axis=1)
    y = ml_df["Weekly_Sales"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    rf_pred = rf_model.predict(X_test)

    # XGBoost
    xgb_pred = xgb_model.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    def metrics(y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        return mae, rmse, r2, mape

    rf_metrics  = metrics(y_test, rf_pred)
    xgb_metrics = metrics(y_test, xgb_pred)

    return rf_metrics, xgb_metrics
# ══════════════════════════════════════════
# TAB 5 — ML PREDICTION
# ══════════════════════════════════════════
with tab5:
    rf_metrics, xgb_metrics = evaluate_models(df, rf_model, xgb_model)
    st.markdown("## 🤖 Machine Learning — Sales Prediction")
    st.markdown("Using **Machine Learning Models (Random Forest & XGBoost)** for prediction.")
    # @st.cache_data
    # def train_model(df):
    #     ml_df = df[["Store","Dept","Year","Month","Week","Quarter","IsHoliday","Weekly_Sales"]].dropna()
    #     X = ml_df.drop("Weekly_Sales", axis=1)
    #     y = ml_df["Weekly_Sales"]
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #     model = RandomForestRegressor(n_estimators=100, max_depth=12,
    #                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    #     mae  = mean_absolute_error(y_test, y_pred)
    #     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #     r2   = r2_score(y_test, y_pred)
    #     mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    #     feat_imp = pd.DataFrame({"Feature": X.columns,
    #                              "Importance": model.feature_importances_}
    #                             ).sort_values("Importance", ascending=False)
    #     sample = X_test.sample(500, random_state=42)
    #     preds  = model.predict(sample)
    #     actuals = y_test.loc[sample.index]
    #     return model, mae, rmse, r2, mape, feat_imp, actuals.values, preds

    # with st.spinner("🔄 Training Random Forest model... (~30 seconds)"):
    #     if st.toggle("Use saved model"):
    #         model = load_model()
    #         _, mae, rmse, r2, mape, feat_imp, actuals, preds = train_model(df)
    #     else:
    #         model, mae, rmse, r2, mape, feat_imp, actuals, preds = train_model(df)

    # m1, m2, m3, m4 = st.columns(4)
    # m1.metric("📏 MAE",       f"${mae:,.0f}")
    # m2.metric("📐 RMSE",      f"${rmse:,.0f}")
    # m3.metric("🎯 R² Score",  f"{r2:.4f}")
    # m4.metric("📉 MAPE",      f"{mape:.2f}%")
    st.markdown("### 📊 Model Performance Comparison")
    
    rf_mae, rf_rmse, rf_r2, rf_mape = rf_metrics
    xgb_mae, xgb_rmse, xgb_r2, xgb_mape = xgb_metrics
    
    comp_df = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "MAE": [rf_mae, xgb_mae],
        "RMSE": [rf_rmse, xgb_rmse],
        "R²": [rf_r2, xgb_r2],
        "MAPE (%)": [rf_mape, xgb_mape]
    })
    
    st.dataframe(comp_df)    
    st.info("""
    📊 Metrics shown are calculated using evaluation on test dataset.
    
    Models are pre-trained and loaded from saved files.
    """)  
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # with c1:
        # fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
        #              title="🔑 Feature Importance",
        #              color="Importance", color_continuous_scale="Blues",
        #              template="plotly_dark")
        # fig.update_layout(height=380, coloraxis_showscale=False,
        #                   yaxis=dict(autorange="reversed"))
        # st.plotly_chart(fig, use_container_width=True)

    # with c2:
    #     sample_df = pd.DataFrame({"Actual": actuals[:300], "Predicted": preds[:300]})
    #     fig2 = px.scatter(sample_df, x="Actual", y="Predicted",
    #                       title="🎯 Actual vs Predicted Sales",
    #                       template="plotly_dark",
    #                       color_discrete_sequence=["#4fdb8b"], opacity=0.5)
    #     lim = [float(min(actuals)), float(max(actuals))]
    #     fig2.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
    #                               name="Perfect Fit",
    #                               line=dict(color="#f97b4f", dash="dash")))
    #     fig2.update_layout(height=380)
    #     st.plotly_chart(fig2, use_container_width=True)

    # residuals = actuals - preds
    # fig3 = px.scatter(x=preds, y=residuals,
    #                   title="📉 Residuals vs Predicted Values",
    #                   labels={"x":"Predicted ($)","y":"Residual ($)"},
    #                   color=residuals, color_continuous_scale="RdBu",
    #                   template="plotly_dark", opacity=0.5)
    # fig3.add_hline(y=0, line_dash="dash", line_color="white")
    # fig3.update_layout(height=350, coloraxis_showscale=False)
    # st.plotly_chart(fig3, use_container_width=True)
    # st.markdown("---")
    # st.header("🔮 Prediction & Forecast")
    # # Live predictor
    # st.markdown("---")
    # st.markdown("### 🔮 Live Sales Predictor")
    st.info("📊 Detailed model diagnostics removed since models are pre-trained.")
    pc1, pc2, pc3 = st.columns(3)
    
    with pc1:
        p_store = st.selectbox(
            "Store",
            sorted(df["Store"].unique()),
            format_func=lambda x: store_names.get(x, f"Store {x}")
        )
    
        p_dept = st.selectbox(
            "Department",
            sorted(df["Dept"].unique()),
            format_func=lambda x: dept_names.get(x, f"Dept {x}")
        )
    
    with pc2:
        p_year  = st.selectbox("Year", [2010, 2011, 2012, 2013])
        p_month = st.slider("Month", 1, 12, 6)
    
    with pc3:
        p_week = st.slider("Week of Year", 1, 52, 25)
        p_holiday = st.radio(
            "Holiday Week?",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
    p_quarter  = (p_month - 1) // 3 + 1
    input_arr  = np.array([[p_store, p_dept, p_year, p_month, p_week, p_quarter, p_holiday]])
    prediction = model.predict(input_arr)[0]
    store_label = store_names.get(p_store, f"Store {p_store}")
    dept_label = dept_names.get(p_dept, f"Dept {p_dept}")
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a2e44, #0f3460);
                border-radius: 12px; padding: 24px; text-align: center; margin-top: 16px;'>
        <h3 style='color: #9e9e9e; margin: 0;'>Predicted Weekly Sales</h3>
        <h1 style='color: #4fdb8b; font-size: 3rem; margin: 8px 0;'>${prediction:,.2f}</h1>
        <p style='color: #9e9e9e; margin: 0;'>
            {store_label} | {dept_label} | {'Holiday' if p_holiday else 'Non-Holiday'} Week
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box" style="margin-top:16px;">
    💡 <b>Model Explanation:</b><br>
    &nbsp;&nbsp;• Predictions are generated using selected ML model.<br>
    &nbsp;&nbsp;• Department and time-based features strongly influence sales.<br>
    &nbsp;&nbsp;• Holiday weeks typically increase sales patterns.<br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.header("📈 Advanced Forecast & What-if Analysis")
    # Select Store & Dept for forecasting
    f_store = st.selectbox(
        "Select Store for Forecast",
        sorted(df["Store"].unique()),
        format_func=lambda x: store_names.get(x, f"Store {x}")
    )
    
    f_dept = st.selectbox(
        "Select Department for Forecast",
        sorted(df["Dept"].unique()),
        format_func=lambda x: dept_names.get(x, f"Dept {x}")
    )
    
    # Get past data
    hist_df = df[(df["Store"] == f_store) & (df["Dept"] == f_dept)].copy()
    hist_df = hist_df.sort_values("Date")
    
    # Take last known date
    if hist_df.empty:
        st.error("No data available for selected Store & Department")
        st.stop()
    # Generate next 4 weeks
    last_date = hist_df["Date"].max()
    future_dates = pd.date_range(start=last_date, periods=5, freq="W")[1:]
    
    future_data = []
    for d in future_dates:
        future_data.append({
            "Store": f_store,
            "Dept": f_dept,
            "Year": d.year,
            "Month": d.month,
            "Week": d.isocalendar()[1],
            "Quarter": (d.month - 1)//3 + 1,
            "IsHoliday": 0
        })
    
    future_df = pd.DataFrame(future_data)
    
    # Predict
    future_preds = model.predict(future_df)
    future_df["Predicted_Sales"] = future_preds
    
    # Prepare plots
    hist_plot = hist_df[["Date", "Weekly_Sales"]].copy()
    hist_plot.columns = ["Date", "Sales"]
    
    future_plot = future_df.copy()
    future_plot["Date"] = future_dates
    future_plot = future_plot[["Date", "Predicted_Sales"]]
    future_plot.columns = ["Date", "Sales"]
    
    combined = pd.concat([hist_plot.tail(20), future_plot])
    
    # Plot
    fig = px.line(
        combined,
        x="Date",
        y="Sales",
        title="📈 Past vs Future Sales",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🔮 Forecast Values")
    st.dataframe(future_df[["Year","Month","Week","Predicted_Sales"]])

# ══════════════════════════════════════════
# TAB 6 — RAW DATA & STATS
# ══════════════════════════════════════════
with tab6:
    st.markdown("## 📋 Dataset Overview & Statistics")

    st.markdown("### 📊 Descriptive Statistics")
    st.dataframe(
        df[["Store","Dept","Weekly_Sales","IsHoliday","Year","Month"]].describe().T.style.format("{:.2f}"),
        use_container_width=True,
    )

    st.markdown("### 🔍 Correlation Matrix")
    corr = df[["Store","Dept","Weekly_Sales","IsHoliday","Year","Month","Week","Quarter"]].corr()
    fig = px.imshow(corr, text_auto=".2f", title="Feature Correlation Heatmap",
                    color_continuous_scale="RdBu_r", template="plotly_dark",
                    zmin=-1, zmax=1)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📄 Sample Records")
    n = st.slider("Rows to show", 10, 200, 50)
    st.dataframe(fdf.sample(n, random_state=42).reset_index(drop=True),
                 use_container_width=True)

    st.markdown("### ⬇️ Download Filtered Data")
    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="walmart_filtered.csv",
        mime="text/csv",
    )

    st.markdown(f"""
    <div class="insight-box">
    📌 <b>Dataset Summary:</b><br>
    &nbsp;&nbsp;• <b>{len(df):,}</b> weekly sales records across <b>45 stores</b> and <b>81 departments</b><br>
    &nbsp;&nbsp;• Date range: <b>Feb 2010 – Oct 2012</b><br>
    &nbsp;&nbsp;• <b>{df['IsHoliday'].sum():,}</b> holiday week records ({df['IsHoliday'].mean()*100:.1f}% of data)<br>
    &nbsp;&nbsp;• Some departments show <b>negative sales</b> (returns exceeding purchases)
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        "📥 Download Report",
        fdf.to_csv(index=False),
        "sales_report.csv"
    )
# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr style='border: 1px solid #2a2d3e; margin-top: 40px;'>
<p style='text-align: center; color: #555; font-size: 0.85rem;'>
    🛒 Walmart Sales Analytics Dashboard &nbsp;|&nbsp;
    Final Year Project &nbsp;|&nbsp;
    Built with Streamlit + Plotly + Scikit-learn
</p>
""", unsafe_allow_html=True)