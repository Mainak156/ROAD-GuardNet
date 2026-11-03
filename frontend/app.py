import base64
import io
import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

st.set_page_config(page_title="ROAD-GuardNet", page_icon="🛣️", layout="wide")

API_URL = os.getenv("ROAD_GUARDNET_API", "http://127.0.0.1:8000")

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.6rem;}
.metric-card {padding: 14px; border: 1px solid #e5e9f0; border-radius: 12px; background: white;}
.small-muted {color: #64748b; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

def get_json(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

st.sidebar.title("🛣️ ROAD-GuardNet")
page = st.sidebar.radio("Navigation", ["Dashboard", "Detect Damage", "Analytics", "About Model"])
st.sidebar.caption("Road damage detection and inspection support")

health, health_error = get_json("/health")
if health_error:
    st.sidebar.error("Backend offline")
elif health and health.get("model_ready"):
    st.sidebar.success("Backend + model ready")
else:
    st.sidebar.warning("Backend online · waiting for model")

if page == "Dashboard":
    st.title("ROAD-GuardNet Dashboard")
    st.write("AI-assisted road damage inspection for four RDD2022 damage categories.")
    stats, err = get_json("/stats")
    if err:
        st.error(f"Cannot reach backend: {err}")
        st.info("Start the FastAPI backend using the commands in README.md.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Images processed", stats["images_processed"])
        c2.metric("Damage detections", stats["total_detections"])
        c3.metric("Average inference", f'{stats["average_inference_time_ms"]:.1f} ms')
        if stats["images_processed"] == 0:
            st.info("No images analysed in this session yet. Open **Detect Damage** to upload a road image.")
        else:
            st.subheader("Recent inspections")
            st.dataframe(pd.DataFrame(stats["history"]).tail(10), use_container_width=True)

elif page == "Detect Damage":
    st.title("Detect Road Damage")
    st.write("Upload a road image to identify visible cracks and potholes.")
    if health and not health.get("model_ready"):
        st.warning("The backend is running, but the trained model weights are not available yet. You can finish the UI now; predictions will work after `best.pt` is placed in the weights folder and the backend is restarted.")
    uploaded = st.file_uploader("Choose a road image", type=["jpg", "jpeg", "png", "webp"])
    left, right = st.columns(2)
    with left:
        confidence = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    with right:
        use_cafa = st.toggle("Enable CAFA-Lite preprocessing", value=True)
    if uploaded:
        original = Image.open(uploaded).convert("RGB")
        st.image(original, caption="Uploaded image", use_container_width=True)
        if st.button("Run ROAD-GuardNet detection", type="primary", use_container_width=True):
            with st.spinner("Analysing image..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        params={"confidence": confidence, "use_cafa": str(use_cafa).lower()},
                        timeout=180
                    )
                    if response.status_code == 503:
                        st.error("Model weights are not loaded yet. Add `best.pt` to the weights folder and restart the backend.")
                    else:
                        response.raise_for_status()
                        data = response.json()
                        annotated_bytes = base64.b64decode(data["annotated_image_base64"])
                        st.session_state["last_result"] = data
                        st.session_state["last_annotated"] = annotated_bytes
                except requests.RequestException as e:
                    st.error(f"Detection request failed: {e}")
        if "last_result" in st.session_state:
            data = st.session_state["last_result"]
            annotated = st.session_state["last_annotated"]
            st.divider()
            st.subheader("Detection result")
            st.image(annotated, caption="Annotated road image", use_container_width=True)
            m1, m2 = st.columns(2)
            m1.metric("Detections", data["total_detections"])
            m2.metric("Inference time", f'{data["inference_time_ms"]:.1f} ms')
            if data["detections"]:
                det_df = pd.DataFrame(data["detections"])
                det_df["bbox"] = det_df["bbox"].apply(lambda x: ", ".join(f"{v:.1f}" for v in x))
                st.dataframe(det_df, use_container_width=True)
                csv = pd.DataFrame(data["detections"]).to_csv(index=False).encode("utf-8")
                st.download_button("Download detection CSV", csv, "road_guardnet_detections.csv", "text/csv")
            else:
                st.info("No detections passed the selected confidence threshold.")

elif page == "Analytics":
    st.title("Inspection Analytics")
    stats, err = get_json("/stats")
    if err:
        st.error(f"Cannot reach backend: {err}")
    else:
        st.metric("Images analysed this session", stats["images_processed"])
        st.metric("Total detected objects", stats["total_detections"])
        counts = stats["class_counts"]
        if counts:
            chart_df = pd.DataFrame({"Damage class": list(counts.keys()), "Detections": list(counts.values())})
            st.plotly_chart(px.bar(chart_df, x="Damage class", y="Detections",
                                   title="Detected damage by class"), use_container_width=True)
        else:
            st.info("Analytics will appear after the first successful prediction.")
        if stats["history"]:
            hist = pd.DataFrame(stats["history"])
            st.subheader("Session history")
            st.dataframe(hist, use_container_width=True)
            st.download_button("Download session history CSV",
                               hist.to_csv(index=False).encode("utf-8"),
                               "road_guardnet_session_history.csv", "text/csv")

else:
    st.title("About ROAD-GuardNet")
    st.write("ROAD-GuardNet is a road-damage detection prototype based on transfer learning with a YOLO object detector.")
    st.subheader("Target damage categories")
    st.table(pd.DataFrame([
        ["D00", "Longitudinal crack"],
        ["D10", "Transverse crack"],
        ["D20", "Alligator crack"],
        ["D40", "Pothole"]
    ], columns=["Class", "Description"]))
    st.subheader("Proposed components")
    st.markdown("""
    - **CAFA-Lite:** image-level brightness and contrast correction.
    - **PGCB:** class-aware oversampling during training.
    - **UGAD:** uncertainty-triggered tiled inference (to be integrated into the deployment inference path after model validation).
    """)
    st.caption("This interface reports measurements from actual API inference sessions; it does not display fabricated model accuracy.")
