
import base64
import binascii
import hashlib
import io
import os

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ROAD-GuardNet",
    page_icon="🛣️",
    layout="wide"
)

API_URL = os.getenv(
    "ROAD_GUARDNET_API",
    "https://road-guardnet-api.onrender.com"
).rstrip("/")

# Separate connection and response timeouts.
# Render Free may take 50+ seconds to wake up.
HEALTH_TIMEOUT = (10, 120)
STATS_TIMEOUT = (10, 120)
PREDICTION_TIMEOUT = (10, 240)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 1.6rem;
}

.metric-card {
    padding: 14px;
    border: 1px solid #e5e9f0;
    border-radius: 12px;
    background: white;
}

.small-muted {
    color: #64748b;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. API COMMUNICATION
# ============================================================

def get_json(path, timeout=STATS_TIMEOUT):
    """
    Send a GET request to FastAPI.

    Uses longer timeouts to accommodate Render cold starts.
    Returns:
        (response_data, None) on success
        (None, error_message) on failure
    """

    try:
        response = requests.get(
            f"{API_URL}{path}",
            timeout=timeout
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            return None, "Invalid response received from backend."

        return data, None

    except requests.exceptions.ConnectTimeout:
        return None, (
            "Connection to backend timed out. "
            "Please check whether the Render service is running."
        )

    except requests.exceptions.ReadTimeout:
        return None, (
            "Backend took too long to respond. "
            "The Render instance may still be starting up."
        )

    except requests.exceptions.ConnectionError:
        return None, (
            "Could not establish a connection with the backend. "
            "Please check your Render deployment."
        )

    except requests.exceptions.HTTPError as e:
        return None, f"Backend HTTP error: {e}"

    except requests.exceptions.RequestException as e:
        return None, f"API request failed: {e}"

    except ValueError:
        return None, "Backend returned an invalid JSON response."


def as_number(value, default=0.0):
    """Safely convert API values to a number for display."""
    try:
        number = float(value)
        return number if pd.notna(number) else default
    except (TypeError, ValueError):
        return default


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛣️ ROAD-GuardNet")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Detect Damage",
        "Analytics",
        "About Model"
    ]
)

st.sidebar.caption(
    "Road damage detection and inspection support"
)


# ============================================================
# 5. BACKEND HEALTH CHECK
# ============================================================

@st.cache_data(ttl=20, show_spinner=False)
def check_backend_health():
    return get_json("/health", timeout=HEALTH_TIMEOUT)


with st.spinner("Connecting to ROAD-GuardNet backend..."):
    health, health_error = check_backend_health()

if health_error:

    st.sidebar.error("Backend temporarily unavailable")

    st.sidebar.caption(
        "The backend may be waking up on Render Free."
    )

elif health and health.get("model_ready", False):

    st.sidebar.success("Backend + model ready")

else:

    st.sidebar.warning("Backend online · Model not ready")


# ============================================================
# 6. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("ROAD-GuardNet Dashboard")

    st.write(
        "AI-assisted road damage inspection "
        "for four RDD2022 damage categories."
    )

    if health_error:

        st.error(
            f"Cannot reach backend: {health_error}"
        )

        st.info(
            "The Render Free instance may have gone to sleep. "
            "Wait for it to start and refresh the dashboard."
        )

    else:

        with st.spinner("Loading inspection statistics..."):

            stats, stats_error = get_json("/stats")

        if stats_error:

            st.error(
                f"Unable to retrieve statistics: {stats_error}"
            )

        else:

            images_processed = stats.get(
                "images_processed", 0
            )

            total_detections = stats.get(
                "total_detections", 0
            )

            inference_time = stats.get(
                "average_inference_time_ms", 0
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Images processed",
                images_processed
            )

            c2.metric(
                "Damage detections",
                total_detections
            )

            c3.metric(
                "Average inference",
                f"{as_number(inference_time):.1f} ms"
            )

            history = stats.get("history") or []
            if not isinstance(history, list):
                history = []

            if images_processed == 0:

                st.info(
                    "No images analysed in this session yet. "
                    "Open Detect Damage to upload a road image."
                )

            elif history:

                st.subheader("Recent inspections")

                st.dataframe(
                    pd.DataFrame(history).tail(10),
                    use_container_width=True
                )


# ============================================================
# 7. ROAD DAMAGE DETECTION
# ============================================================

elif page == "Detect Damage":

    st.title("Detect Road Damage")

    st.write(
        "Upload a road image to identify visible cracks and potholes."
    )

    if health_error:

        st.warning(
            "The backend is currently unavailable. "
            "You can upload an image, but detection will work "
            "only after the backend becomes accessible."
        )

    elif health and not health.get("model_ready", False):

        st.warning(
            "The backend is online, but the trained model "
            "weights are not ready yet. Verify that best.pt "
            "is available at the configured model path."
        )

    uploaded = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    left, right = st.columns(2)

    with left:

        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.25,
            step=0.05
        )

    with right:

        use_cafa = st.toggle(
            "Enable CAFA-Lite preprocessing",
            value=True
        )

    if uploaded:

        uploaded_bytes = uploaded.getvalue()

        # Identify the uploaded image to prevent stale results.
        upload_id = hashlib.sha256(
            uploaded_bytes
        ).hexdigest()

        if st.session_state.get("last_upload_id") != upload_id:

            st.session_state.pop("last_result", None)
            st.session_state.pop("last_annotated", None)

            st.session_state["last_upload_id"] = upload_id

        try:

            # PIL expects a file-like object, not raw bytes.
            with Image.open(io.BytesIO(uploaded_bytes)) as image:
                image.verify()

            # Reopen after verify() and fully decode the image.
            with Image.open(io.BytesIO(uploaded_bytes)) as image:
                original = image.convert("RGB")
                original.load()

        except (UnidentifiedImageError, OSError, ValueError) as e:

            st.error(
                "Invalid or corrupted image. Please upload a valid "
                f"JPG, PNG, or WEBP file. Details: {e}"
            )

            st.stop()

        st.image(
            original,
            caption="Uploaded road image",
            use_container_width=True
        )

        if st.button(
            "Run ROAD-GuardNet Detection",
            type="primary",
            use_container_width=True
        ):

            # Remove previous prediction before starting.
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_annotated", None)

            with st.spinner(
                "Analysing road image... "
                "Please wait while the model processes your image."
            ):

                try:

                    response = requests.post(
                        f"{API_URL}/predict",

                        files={
                            "file": (
                                uploaded.name,
                                uploaded_bytes,
                                uploaded.type or "application/octet-stream"
                            )
                        },

                        params={
                            "confidence": confidence,
                            "use_cafa": str(use_cafa).lower()
                        },

                        timeout=PREDICTION_TIMEOUT
                    )

                    if response.status_code == 503:

                        st.error(
                            "Model unavailable. "
                            "Check the backend model weights and "
                            "restart the Render service."
                        )

                    else:

                        response.raise_for_status()

                        data = response.json()

                        if not isinstance(data, dict):

                            raise ValueError(
                                "Invalid prediction response."
                            )

                        if not data.get("annotated_image_base64"):

                            raise ValueError(
                                "Annotated image missing from API response."
                            )

                        annotated_bytes = base64.b64decode(
                            data["annotated_image_base64"],
                            validate=True
                        )

                        st.session_state["last_result"] = data

                        st.session_state["last_annotated"] = (
                            annotated_bytes
                        )

                except requests.exceptions.ReadTimeout:

                    st.error(
                        "Detection timed out. The model may be "
                        "loading or inference may be taking too long. "
                        "Please try again."
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to the FastAPI backend. "
                        "Check your Render service."
                    )

                except requests.exceptions.HTTPError as e:

                    st.error(
                        f"Backend returned an HTTP error: {e}"
                    )

                except (requests.exceptions.RequestException,
                        ValueError,
                        KeyError,
                        binascii.Error) as e:

                    st.error(
                        f"Detection request failed: {e}"
                    )

                except Exception as e:

                    st.error(
                        f"Unexpected detection error: {e}"
                    )

        # ----------------------------------------------------
        # DISPLAY DETECTION RESULT
        # ----------------------------------------------------

        if "last_result" in st.session_state:

            data = st.session_state["last_result"]

            annotated = st.session_state["last_annotated"]

            st.divider()

            st.subheader("Detection Result")

            st.image(
                annotated,
                caption="ROAD-GuardNet Annotated Image",
                use_container_width=True
            )

            m1, m2 = st.columns(2)

            m1.metric(
                "Total detections",
                data.get("total_detections", 0)
            )

            m2.metric(
                "Inference time",
                f'{as_number(data.get("inference_time_ms")):.1f} ms'
            )

            detections = data.get("detections") or []
            if not isinstance(detections, list):
                detections = []

            if detections:

                st.subheader("Detected Damage Details")

                det_df = pd.DataFrame(detections)

                if "bbox" in det_df.columns:

                    det_df["bbox"] = det_df["bbox"].apply(
                        lambda x: ", ".join(
                            f"{as_number(v):.1f}" for v in x
                        ) if isinstance(x, (list, tuple)) else str(x)
                    )

                st.dataframe(
                    det_df,
                    use_container_width=True
                )

                csv = pd.DataFrame(
                    detections
                ).to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Detection CSV",
                    csv,
                    "road_guardnet_detections.csv",
                    "text/csv"
                )

            else:

                st.info(
                    "No detections passed the selected confidence threshold."
                )


# ============================================================
# 8. ANALYTICS
# ============================================================

elif page == "Analytics":

    st.title("Inspection Analytics")

    if health_error:

        st.error(
            f"Cannot reach backend: {health_error}"
        )

    else:

        with st.spinner("Loading analytics..."):

            stats, stats_error = get_json("/stats")

        if stats_error:

            st.error(
                f"Unable to retrieve analytics: {stats_error}"
            )

        else:

            images_processed = stats.get(
                "images_processed", 0
            )

            total_detections = stats.get(
                "total_detections", 0
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Images analysed this session",
                images_processed
            )

            c2.metric(
                "Total detected objects",
                total_detections
            )

            counts = stats.get("class_counts") or {}
            if not isinstance(counts, dict):
                counts = {}

            if counts:

                st.subheader("Damage Distribution")

                chart_df = pd.DataFrame({
                    "Damage class": list(counts.keys()),
                    "Detections": list(counts.values())
                })

                fig = px.bar(
                    chart_df,
                    x="Damage class",
                    y="Detections",
                    title="Detected Damage by Class",
                    text="Detections"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Analytics will appear after the first successful prediction."
                )

            history = stats.get("history") or []
            if not isinstance(history, list):
                history = []

            if history:

                hist = pd.DataFrame(history)

                st.subheader("Session History")

                st.dataframe(
                    hist,
                    use_container_width=True
                )

                st.download_button(
                    "Download Session History CSV",
                    hist.to_csv(index=False).encode("utf-8"),
                    "road_guardnet_session_history.csv",
                    "text/csv"
                )


# ============================================================
# 9. ABOUT MODEL
# ============================================================

else:

    st.title("About ROAD-GuardNet")

    st.write(
        "ROAD-GuardNet is a road-damage detection prototype "
        "based on transfer learning with a YOLO object detector."
    )

    st.subheader("Target Damage Categories")

    st.table(
        pd.DataFrame(
            [
                ["D00", "Longitudinal crack"],
                ["D10", "Transverse crack"],
                ["D20", "Alligator crack"],
                ["D40", "Pothole"]
            ],
            columns=[
                "Class",
                "Description"
            ]
        )
    )

    st.subheader("Proposed Components")

    st.markdown("""
    - **CAFA-Lite:** Image-level brightness and contrast correction.
    - **PGCB:** Class-aware oversampling during training.
    - **UGAD:** Uncertainty-triggered tiled inference
      (to be integrated into the deployment inference path
      after model validation).
    """)

    st.caption(
        "This interface reports measurements from actual API "
        "inference sessions; it does not display fabricated model accuracy."
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.caption(
    "ROAD-GuardNet | AI-Assisted Road Damage Inspection"
)