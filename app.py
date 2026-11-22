import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from evaluation import (
    load_dataset,
    run_single_pipeline,
    run_full_benchmark,
)

# Page configuration
st.set_page_config(
    page_title="Face Enhancement & LBPH Panel",
    layout="wide",
)

# Minimalist light theme styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FAF9F6;
        color: #1C1C1C;
    }
    section[data-testid="stSidebar"] {
        background-color: #F2EFE9;
        border-right: 1px solid #E2DDD5;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E6E1D8;
        border-radius: 10px;
        padding: 12px 16px;
    }
    h1, h2, h3 {
        color: #161616;
        letter-spacing: -0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_cached_dataset():
    return load_dataset()


@st.cache_data(show_spinner=True)
def get_cached_benchmark(noise_type, sigma, sp_amount):
    return run_full_benchmark(
        noise_type=noise_type,
        sigma=sigma,
        sp_amount=sp_amount,
    )


def plot_histogram(image, title):
    fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=120)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    ax.fill_between(range(256), hist, color="#2B2B2B", alpha=0.82)
    ax.set_title(title, fontsize=9, fontweight="bold", color="#1C1C1C", pad=6)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel Intensity", fontsize=7.5, color="#555555")
    ax.set_ylabel("Frequency", fontsize=7.5, color="#555555")
    ax.tick_params(labelsize=7, colors="#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D6D1C7")
    ax.spines["bottom"].set_color("#D6D1C7")
    plt.tight_layout()
    return fig


def format_psnr(val):
    return "Reference" if not np.isfinite(val) else f"{val:.2f} dB"


# Header
st.title("Face Image Enhancement & Performance Panel")
st.caption(
    "Extended Yale Face Database B | Modules 1–4 Integration: Alignment, Denoising, Sharpening & LBPH Face Recognition"
)

train_data, test_data, label_map = get_cached_dataset()

# Sidebar Controls
with st.sidebar:
    st.header("Experiment Settings")

    noise_type = st.selectbox("Noise Type", ["Gaussian", "Salt & Pepper"])
    if noise_type == "Gaussian":
        sigma = st.slider("Gaussian Noise Intensity (σ)", 5, 35, 15, step=5)
        sp_amount = 0.02
    else:
        sigma = 15
        sp_amount = st.slider("Salt & Pepper Ratio", 0.01, 0.10, 0.02, step=0.01)

    st.divider()
    st.subheader("Single-Image Pipeline")
    denoise_method = st.selectbox(
        "Denoising Filter",
        ["Bilateral", "Median", "Gaussian", "NL-Means", "None"],
    )
    sharpen_method = st.selectbox(
        "Sharpening Method",
        ["Unsharp Masking", "Laplacian", "None"],
    )

    st.divider()
    st.caption(
        f"Dataset: {len(label_map)} subjects | Train: {len(train_data)} | Test: {len(test_data)} images"
    )

# Main Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "1. Single Image & Histogram Analysis",
        "2. Benchmark & LBPH Evaluation",
        "3. Success & Failure Cases",
    ]
)

# ==========================================================
# TAB 1: SINGLE IMAGE COMPARISON & HISTOGRAMS
# ==========================================================
with tab1:
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        selected_person = st.selectbox("Select Subject", sorted(label_map.values()))
    person_imgs = [item for item in test_data if item["person"] == selected_person]

    with col_sel2:
        selected_idx = st.selectbox(
            "Select Test Image (Sorted by Lighting Angle)",
            range(len(person_imgs)),
            format_func=lambda i: (
                f"{person_imgs[i]['filename']}  (Lighting Angle: {person_imgs[i]['angle']:.1f}°)"
            ),
        )

    chosen = person_imgs[selected_idx]
    res = run_single_pipeline(
        chosen["image"],
        noise_type=noise_type,
        denoise_method=denoise_method,
        sharpen_method=sharpen_method,
        sigma=sigma,
        sp_amount=sp_amount,
        seed=42,
    )

    st.subheader("Stage-by-Stage Visual Comparison")
    stages = [
        ("Original (Aligned)", "original"),
        (f"Noisy ({noise_type})", "noisy"),
        (f"Denoised ({denoise_method})", "denoised"),
        (f"Sharpened ({sharpen_method})", "sharpened"),
    ]

    img_cols = st.columns(4)
    for col, (label, key) in zip(img_cols, stages):
        with col:
            st.image(res[key], caption=label, clamp=True)
            m = res["metrics"][key]
            st.markdown(
                f"**PSNR:** {format_psnr(m['psnr'])}  \n"
                f"**SSIM:** {m['ssim']:.4f}  \n"
                f"**Laplacian Var:** {m['lap_var']:.1f}  \n"
                f"**Tenengrad:** {m['tenengrad']:.1f}  \n"
                f"**Contrast:** {m['contrast']:.1f}"
            )

    st.subheader("Pixel Intensity Histograms")
    hist_cols = st.columns(4)
    for col, (label, key) in zip(hist_cols, stages):
        with col:
            fig = plot_histogram(res[key], label)
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Before / After Metric Summary")
    m_noisy = res["metrics"]["noisy"]
    m_den = res["metrics"]["denoised"]
    m_sharp = res["metrics"]["sharpened"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "PSNR (Denoised)",
        format_psnr(m_den["psnr"]),
        delta=f"{m_den['psnr'] - m_noisy['psnr']:+.2f} dB"
        if np.isfinite(m_den["psnr"]) and np.isfinite(m_noisy["psnr"])
        else None,
    )
    k2.metric(
        "SSIM (Denoised)",
        f"{m_den['ssim']:.4f}",
        delta=f"{m_den['ssim'] - m_noisy['ssim']:+.4f}",
    )
    k3.metric(
        "Laplacian Sharpness (Final)",
        f"{m_sharp['lap_var']:.1f}",
        delta=f"{m_sharp['lap_var'] - m_den['lap_var']:+.1f}",
    )
    k4.metric(
        "Tenengrad Sharpness (Final)",
        f"{m_sharp['tenengrad']:.1f}",
        delta=f"{m_sharp['tenengrad'] - m_den['tenengrad']:+.1f}",
    )

# ==========================================================
# TAB 2: FULL BENCHMARK & LBPH RESULTS
# ==========================================================
with tab2:
    st.subheader(f"Method Comparison Under {noise_type} Noise")
    df, details, _ = get_cached_benchmark(noise_type, sigma, sp_amount)

    st.dataframe(df, hide_index=True)

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        fig1, ax1 = plt.subplots(figsize=(6.2, 4.2), dpi=130)
        fig1.patch.set_facecolor("#FFFFFF")
        ax1.set_facecolor("#FFFFFF")
        colors = [
            "#B89758" if acc == df["LBPH Accuracy (%)"].max() else "#2C2C2C"
            for acc in df["LBPH Accuracy (%)"]
        ]
        bars = ax1.barh(df["Method"][::-1], df["LBPH Accuracy (%)"][::-1], color=colors[::-1])
        ax1.set_xlim(0, 108)
        ax1.set_xlabel("LBPH Recognition Accuracy (%)", fontsize=9)
        ax1.set_title("Face Recognition Accuracy by Method", fontsize=10.5, fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        for bar in bars:
            w = bar.get_width()
            ax1.text(w + 1.2, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with c_chart2:
        df_psnr = df.dropna(subset=["Avg PSNR (dB)"])
        fig2, ax2 = plt.subplots(figsize=(6.2, 4.2), dpi=130)
        fig2.patch.set_facecolor("#FFFFFF")
        ax2.set_facecolor("#FFFFFF")
        colors_p = [
            "#B89758" if p == df_psnr["Avg PSNR (dB)"].max() else "#5C5C5C"
            for p in df_psnr["Avg PSNR (dB)"]
        ]
        bars2 = ax2.barh(df_psnr["Method"][::-1], df_psnr["Avg PSNR (dB)"][::-1], color=colors_p[::-1])
        ax2.set_xlim(0, df_psnr["Avg PSNR (dB)"].max() + 7)
        ax2.set_xlabel("Average PSNR (dB)", fontsize=9)
        ax2.set_title("Image Quality Comparison (PSNR)", fontsize=10.5, fontweight="bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for bar in bars2:
            w = bar.get_width()
            ax2.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

# ==========================================================
# TAB 3: SUCCESS & FAILURE CASES
# ==========================================================
with tab3:
    st.subheader("LBPH Prediction Cases: Successes & Failures")
    method_choice = st.selectbox("Select Method to Inspect", df["Method"].tolist(), index=6)
    records = details[method_choice]

    successes = [r for r in records if r["correct"]]
    failures = [r for r in records if not r["correct"]]

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Test Images", len(records))
    s2.metric("Correct Predictions", len(successes))
    s3.metric("Incorrect Predictions", len(failures))

    st.markdown("### Failure Cases (Error Analysis)")
    if not failures:
        st.success("This method correctly recognized all images in the test set.")
    else:
        fail_cols = st.columns(4)
        for idx, rec in enumerate(failures[:8]):
            with fail_cols[idx % 4]:
                st.image(
                    rec["processed"],
                    caption=(
                        f"True: {rec['true_person']} | Pred: {rec['pred_person']}\n"
                        f"Angle: {rec['angle']:.1f}° | Confidence: {rec['confidence']:.1f}"
                    ),
                    clamp=True,
                )

    st.markdown("### Success Cases")
    ok_cols = st.columns(4)
    for idx, rec in enumerate(successes[:4]):
        with ok_cols[idx % 4]:
            st.image(
                rec["processed"],
                caption=(
                    f"Match: {rec['true_person']}\n"
                    f"Angle: {rec['angle']:.1f}° | Confidence: {rec['confidence']:.1f}"
                ),
                clamp=True,
            )