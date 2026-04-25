"""
=============================================================
  dashboard.py  --  Defect detection pipeline script
=============================================================
HOW TO USE
----------
python dashboard.py

FLAGS
-----
(No flags defined or --help not available)

OLD EXAMPLES / SETUP
--------------------
# streamlit run dashboard.py
=============================================================
"""

import sqlite3
import pandas as pd
import streamlit as st
import json
import plotly.express as px
import os
from datetime import datetime

# Optional: set Streamlit to run locally by default on port 8501
st.set_page_config(page_title="Defect Insights", layout="wide", page_icon="🔍")

# Path to the database you already have in defect_detector.py
DB_PATH = os.path.join("logs", "inspections.db")

def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM inspections", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()

# Main Dashboard UI
st.title("🏭 Live Defect Dashboards")
st.markdown("Monitor real-time defects, distributions, and histories logged by your YOLO pipeline.")

# Auto-refresh button
if st.button("🔄 Refresh Data"):
    st.rerun()

df = load_data()

if df.empty:
    st.warning(f"No logs found in {DB_PATH}. Run your defect detector to create logs!")
else:
    # --- Data Preprocessing ---
    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # --- Date Filter Sidebar ---
    st.sidebar.header("⚙️ Dashboard Controls")
    min_date = df['timestamp'].dt.date.min()
    max_date = df['timestamp'].dt.date.max()
    
    if min_date and max_date:
        selected_dates = st.sidebar.slider(
            "🗓️ Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )
        
        # Apply Filter
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
            mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
            df = df.loc[mask]
    
    # Extract defect specifics from the JSON column
    # The 'defects' column stores lists of dictionaries like: [{"class": "rust", "confidence": 0.8}, ...]
    flat_defects = []
    for idx, row in df.iterrows():
        try:
            defects_list = json.loads(row['defects'])
            for d in defects_list:
                
                # Parse Zone from bounding box
                # The detector divides the image into 4 slices. We map the normalized X centroid.
                zone_name = "Unknown"
                if 'bbox_normalized' in d:
                    nx, ny, nw, nh = d['bbox_normalized']
                    cx_norm = nx + (nw / 2)  # Centroid X (0.0 to 1.0)
                    zone_idx = min(int(cx_norm * 4), 3)  # Map to 0, 1, 2, 3
                    
                    # Map the index to the physical area names
                    zone_mapping = {0: "Top", 1: "Middle", 2: "Thread Area", 3: "Bottom"}
                    zone_name = zone_mapping.get(zone_idx, "Unknown")
                
                flat_defects.append({
                    "id": row.get('id', 0),
                    "timestamp": row['timestamp'],
                    "defect_class": d.get('class', 'Unknown').capitalize(),
                    "confidence": d.get('confidence', 0),
                    "zone": zone_name
                })
        except Exception as e:
            print(f"Error parsing defects for row {idx}: {e}")
            
    df_defects = pd.DataFrame(flat_defects)

    # --- Top KPIs ---
    col1, col2, col3 = st.columns(3)
    
    total_inspections = len(df)
    total_defects = len(df_defects)
    total_passes = len(df[df['result'] == 'PASS'])
    total_rejects = len(df[df['result'] == 'REJECT'])
    rejection_rate = (total_rejects / total_inspections * 100) if total_inspections > 0 else 0
    
    col1.metric("📦 Total Inspected", f"{total_inspections}")
    col2.metric("❌ Rejection Rate", f"{rejection_rate:.1f}%")
    col3.metric("⚠️ Total Individual Defects", f"{total_defects}")
    
    st.divider()

    # --- Charts ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Defect Types Breakdown")
        if not df_defects.empty:
            type_counts = df_defects['defect_class'].value_counts().reset_index()
            type_counts.columns = ['Defect Type', 'Count']
            
            # Pie Chart
            fig_pie = px.pie(
                type_counts, 
                values='Count', 
                names='Defect Type',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No specific defects logged yet.")
            
    with chart_col2:
        st.subheader("Inspection History (Pass vs Reject)")
        
        # Resample to hourly or daily based on timeframe
        df['hour'] = df['timestamp'].dt.floor('h')
        timeline = df.groupby(['hour', 'result']).size().reset_index(name='count')
        
        if not timeline.empty:
            fig_bar = px.bar(
                timeline, 
                x="hour", 
                y="count", 
                color="result",
                color_discrete_map={"PASS": "green", "REJECT": "red"},
                barmode='group',
                title="System Throughput"
            )
            fig_bar.update_layout(xaxis_title="Time", yaxis_title="Number of Inspections")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Not enough timeline data to plot.")

    st.divider()

    # --- Zone Analysis Chart ---
    st.subheader("📍 Defects by Part Area (Zone Analysis)")
    if not df_defects.empty:
        # Group by Area and Defect Type
        zone_counts = df_defects.groupby(['zone', 'defect_class']).size().reset_index(name='count')
        
        # Spatial sorting order for the chart (from left side of camera to right side)
        order = ["Top", "Middle", "Thread Area", "Bottom"]
        
        fig_zone = px.bar(
            zone_counts, 
            x="zone", 
            y="count", 
            color="defect_class",
            title="Where are defects occurring the most?",
            labels={"zone": "Part Area", "count": "Number of Defects", "defect_class": "Defect Type"},
            category_orders={"zone": order},
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_zone, use_container_width=True)
    else:
        st.info("Not enough location data to map defects yet.")

    st.divider()

    # --- Advanced Analytics ---
    st.divider()
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        st.subheader("🎯 Confidence Score Distribution")
        if not df_defects.empty:
            fig_hist = px.histogram(
                df_defects, 
                x="confidence", 
                color="defect_class",
                nbins=20,
                title="Model Certainty (Are scores hovering near the threshold?)",
                labels={"confidence": "Confidence Score"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No confidence data")
            
    with adv_col2:
        st.subheader("🔗 Defect Co-occurrence Matrix")
        if not df_defects.empty:
            # Create a matrix of which defects appear on the same image/part ID
            defect_dummies = pd.get_dummies(df_defects['defect_class'], dtype=int)
            df_dummies = pd.concat([df_defects[['id']], defect_dummies], axis=1).groupby('id').max()
            co_matrix = df_dummies.T.dot(df_dummies).astype(int)
            # Nullify self-correlations to make the heatmap more readable
            for i in range(len(co_matrix)):
                co_matrix.iloc[i, i] = 0  
                
            fig_heat = px.imshow(
                co_matrix, 
                text_auto=True, 
                color_continuous_scale='Reds',
                title="How often do two different defects occur on the SAME part?"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No co-occurrence data")

    st.divider()

    # --- Visual Audit Trail & Recent Logs Table ---
    st.subheader("📸 Visual Audit Trail & Database Logs")
    
    display_df = df.sort_values(by='timestamp', ascending=False)
    
    # Audit Trail Image Viewer
    if 'image_path' in display_df.columns:
        img_logs = display_df[display_df['image_path'].notna() & (display_df['image_path'] != '')]
        if not img_logs.empty:
            st.markdown("**Select a Log to visually inspect the part:**")
            
            def format_option(log_id):
                row_data = img_logs[img_logs['id'] == log_id].iloc[0]
                return f"Log #{log_id} | {row_data['result']} | {row_data['defect_count']} Defects | {row_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                
            selected_log_id = st.selectbox("Recorded Snapshots", img_logs['id'].tolist(), format_func=format_option)
            
            if selected_log_id:
                img_path = img_logs[img_logs['id'] == selected_log_id]['image_path'].values[0]
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Captured Snapshot for Log ID {selected_log_id}", width=700)
                else:
                    st.error(f"Image piece not found on disk: {img_path}")
        else:
            st.info("No images have been captured yet. Use the 'c' key in live mode to capture some!")

    st.markdown("**Recent Raw Logs**")
    display_df_table = display_df[['id', 'timestamp', 'result', 'defect_count', 'max_confidence', 'inference_time_ms', 'image_path']].head(20)
    st.dataframe(display_df_table, use_container_width=True, hide_index=True)
