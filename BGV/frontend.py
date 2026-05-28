"""
frontend.py — Gradio UI for BGV Document Verification System
Run: python frontend.py
Requires: pip install gradio requests
"""

import json
import time
from pathlib import Path

import gradio as gr
import requests

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

SUPPORTED_TYPES = ["Aadhaar", "PAN", "Passport", "Resume", "Graduation Certificate", "Marksheet"]

STATUS_EMOJI = {
    "Likely Genuine": "✅",
    "Suspicious": "⚠️",
    "Likely Fake / Tampered": "❌",
}

# ── API calls ─────────────────────────────────────────────────────────────────

def call_api(endpoint: str, file_path: str) -> dict:
    with open(file_path, "rb") as f:
        filename = Path(file_path).name
        mime = "application/pdf" if filename.endswith(".pdf") else "image/jpeg"
        resp = requests.post(
            f"{API_BASE}/{endpoint}",
            files={"file": (filename, f, mime)},
            timeout=500,
        )
    resp.raise_for_status()
    return resp.json()


def check_server() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Formatters ────────────────────────────────────────────────────────────────

def risk_color(score: int) -> str:
    if score < 30:
        return "#16a34a"
    if score < 60:
        return "#d97706"
    return "#dc2626"


def format_fields(fields: dict) -> str:
    if not fields:
        return "_No fields extracted_"
    lines = []
    for k, v in fields.items():
        if v:
            label = k.replace("_", " ").title()
            lines.append(f"**{label}:** {v}")
    return "\n\n".join(lines) if lines else "_No fields extracted_"


def format_fraud(fraud: dict) -> str:
    icons = {
        "clean": "🟢", "not_detected": "🟢", "sharp": "🟢",
        "normal": "🟢", "verified": "🟢", "valid": "🟢",
        "suspicious": "🟡", "slightly_blurry": "🟡",
        "missing": "🟡", "unknown": "🟡", "decode_error": "🟡",
        "not_found": "🟡",
        "tampered": "🔴", "detected": "🔴", "blurry": "🔴",
        "mismatch": "🔴", "invalid": "🔴", "suspected": "🔴",
    }
    checks = {
        "ela": "ELA (pixel tampering)",
        "blur": "Blur / sharpness",
        "duplicate_regions": "Copy-move detection",
        "metadata": "EXIF metadata",
        "ai_artifacts": "AI artifact detection",
        "qr_validation": "QR cross-validation",
        "layout_validation": "Layout validation",
    }
    lines = []
    for key, label in checks.items():
        val = fraud.get(key, "—")
        icon = icons.get(str(val), "⚪")
        lines.append(f"{icon} **{label}:** {val}")
    ela_score = fraud.get("ela_score")
    blur_score = fraud.get("blur_score")
    if ela_score is not None:
        lines.append(f"\n📊 **ELA score:** {ela_score:.1f} (threshold: 30=suspicious, 60=tampered)")
    if blur_score is not None:
        lines.append(f"📊 **Blur (Laplacian variance):** {blur_score:.0f} (lower = blurrier)")
    return "\n\n".join(lines)


def format_validation(validation: dict) -> str:
    checks = validation.get("checks", [])
    if not checks:
        return "_No validation checks run_"
    lines = []
    for c in checks:
        icon = "✅" if c.get("passed") else "❌"
        sev = c.get("severity", "")
        name = c.get("field_name", "").replace("_", " ").title()
        msg = c.get("message", "")
        lines.append(f"{icon} **{name}** ({sev})\n   _{msg}_")
    summary = validation.get("summary", "")
    if summary:
        lines.append(f"\n**Summary:** {summary}")
    return "\n\n".join(lines)


def format_vlm(vlm: dict) -> str:
    if not vlm or not vlm.get("vlm_available", True):
        return "⚠️ VLM analysis unavailable (check ANTHROPIC_API_KEY in .env)"
    lines = [
        f"**Document confirmed as:** {vlm.get('document_type_confirmed', '—')}",
        f"**VLM confidence:** {vlm.get('vlm_confidence', 0)*100:.0f}%",
        f"**Font consistency:** {vlm.get('font_consistency', '—')}",
        f"**Layout authenticity:** {vlm.get('layout_authenticity', '—')}",
        f"**Seal/signature present:** {'Yes ✅' if vlm.get('seal_signature_present') else 'No ❌'}",
        f"**Visible tampering:** {'Detected ❌' if vlm.get('visible_tampering') else 'None detected ✅'}",
        f"**Overall assessment:** {vlm.get('overall_assessment', '—').upper()}",
    ]
    reasoning = vlm.get("reasoning", "")
    if reasoning:
        lines.append(f"\n**Reasoning:**\n> {reasoning}")
    signals = vlm.get("fraud_signals", [])
    if signals:
        lines.append("\n**Fraud signals detected:**")
        for s in signals:
            sev = s.get("severity", "")
            sig = s.get("signal", "")
            icon = "🔴" if sev == "high" else ("🟡" if sev == "medium" else "🟠")
            lines.append(f"  {icon} {sig} ({sev})")
    inconsistencies = vlm.get("logical_inconsistencies", [])
    if inconsistencies:
        lines.append("\n**Logical inconsistencies:**")
        for i in inconsistencies:
            lines.append(f"  ⚠️ {i}")
    return "\n\n".join(lines)


# ── Main processing functions ─────────────────────────────────────────────────

def run_extract(file):
    if file is None:
        return "Please upload a document first.", "", "", ""
    if not check_server():
        return "❌ BGV server is not running. Start it with: `uvicorn main:app --reload`", "", "", ""
    try:
        t0 = time.time()
        data = call_api("extract", file.name)
        elapsed = time.time() - t0

        doc_type = data.get("document_type", "unknown").upper()
        confidence = data.get("confidence", 0) * 100
        page_count = data.get("page_count", 1)
        proc_time = data.get("processing_time_ms", elapsed * 1000)

        summary = (
            f"## 📄 {doc_type}\n\n"
            f"**Confidence:** {confidence:.0f}%  |  "
            f"**Pages:** {page_count}  |  "
            f"**Time:** {proc_time:.0f}ms\n\n"
            f"_Lightweight extract — no fraud detection. Use Full Verify for complete analysis._"
        )
        fields_md = format_fields(data.get("extracted_fields", {}))
        raw = json.dumps(data, indent=2)
        return summary, fields_md, "", raw
    except requests.HTTPError as e:
        return f"❌ API error {e.response.status_code}: {e.response.text}", "", "", ""
    except Exception as e:
        return f"❌ Error: {e}", "", "", ""


def run_verify(file):
    if file is None:
        return "Please upload a document first.", "", "", "", "", ""
    if not check_server():
        return "❌ BGV server is not running. Start it with: `uvicorn main:app --reload`", "", "", "", "", ""
    try:
        t0 = time.time()
        data = call_api("analyze", file.name)
        elapsed = time.time() - t0

        doc_type = data.get("document_type", "unknown").upper()
        risk_score = data.get("risk_score", 0)
        status_text = data.get("status", "Unknown")
        confidence = data.get("confidence", 0) * 100
        proc_time = data.get("processing_time_ms", elapsed * 1000)
        emoji = STATUS_EMOJI.get(status_text, "❓")
        stages = data.get("pipeline_stages", [])

        summary = (
            f"## {emoji} {status_text}\n\n"
            f"**Document:** {doc_type}  |  "
            f"**Risk Score:** {risk_score}/100  |  "
            f"**OCR Confidence:** {confidence:.0f}%  |  "
            f"**Time:** {proc_time:.0f}ms\n\n"
        )
        if data.get("human_summary"):
            summary += f"```\n{data['human_summary']}\n```"
        if stages:
            summary += f"\n\n**Stages completed:** {' → '.join(stages)}"

        fields_md = format_fields(data.get("extracted_fields", {}))
        fraud_md = format_fraud(data.get("fraud_analysis", {}))
        vlm_md = format_vlm(data.get("vlm_analysis", {}))
        validation_md = format_validation(data.get("validation", {}))
        raw = json.dumps(data, indent=2)

        return summary, fields_md, fraud_md, vlm_md, validation_md, raw
    except requests.HTTPError as e:
        err = f"❌ API error {e.response.status_code}: {e.response.text}"
        return err, "", "", "", "", ""
    except Exception as e:
        return f"❌ Error: {e}", "", "", "", "", ""


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#title-row { text-align: center; padding: 1.5rem 0 0.5rem; }
#title-row h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
#title-row p  { color: #6b7280; font-size: 0.95rem; }
.risk-card { border-radius: 12px; padding: 1rem; margin-top: 0.5rem; }
footer { display: none !important; }
"""

with gr.Blocks(title="BGV Document Verifier") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    with gr.Row(elem_id="title-row"):
        gr.Markdown(
            "# 🔍 BGV Document Verification\n"
            "**Background Verification System** — Upload an Indian identity or educational document to verify authenticity"
        )

    # ── Server status ────────────────────────────────────────────────────────
    with gr.Row():
        server_status = gr.Markdown("⏳ Checking server...", label="")

    # ── Main layout ──────────────────────────────────────────────────────────
    with gr.Row():
        # Left: upload + controls
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📤 Upload Document")
            file_input = gr.File(
                label="Supported: PDF, JPG, PNG, TIFF (max 20 MB)",
                file_types=[".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"],
            )
            gr.Markdown(
                "**Supported document types:**\n"
                + "".join(f"- {t}\n" for t in SUPPORTED_TYPES),
                visible=True,
            )
            with gr.Row():
                extract_btn = gr.Button("⚡ Quick Extract", variant="secondary", size="lg")
                verify_btn = gr.Button("🛡️ Full Verify", variant="primary", size="lg")

            gr.Markdown(
                "_**Quick Extract** — OCR + field parsing only (fast)\n\n"
                "**Full Verify** — All 10 stages: OCR + VLM + Fraud Detection + Validation_"
            )

        # Right: results
        with gr.Column(scale=2):
            with gr.Tabs():

                # ── Extract tab ──────────────────────────────────────────────
                with gr.TabItem("⚡ Extract Results"):
                    ext_summary = gr.Markdown("_Upload a document and click Quick Extract_")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Extracted Fields")
                            ext_fields = gr.Markdown()
                    with gr.Accordion("Raw JSON", open=False):
                        ext_raw = gr.Code(language="json", label="")

                # ── Verify tab ───────────────────────────────────────────────
                with gr.TabItem("🛡️ Verification Results"):
                    ver_summary = gr.Markdown("_Upload a document and click Full Verify_")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🏷️ Extracted Fields")
                            ver_fields = gr.Markdown()
                        with gr.Column():
                            gr.Markdown("#### 🔬 Fraud Analysis")
                            ver_fraud = gr.Markdown()

                    gr.Markdown("#### 🤖 VLM Analysis (Claude Vision)")
                    ver_vlm = gr.Markdown()

                    gr.Markdown("#### ✅ Validation Engine")
                    ver_validation = gr.Markdown()

                    with gr.Accordion("Raw JSON", open=False):
                        ver_raw = gr.Code(language="json", label="")

    # ── Example documents hint ───────────────────────────────────────────────
    gr.Markdown(
        "---\n"
        "💡 **Tips:** Ensure the BGV server is running on `localhost:8000` before verifying. "
        "For best results, upload clear, well-lit scans at 300 DPI or higher."
    )

    # ── Event handlers ────────────────────────────────────────────────────────
    def refresh_status():
        if check_server():
            return "✅ **BGV server is online** — `http://localhost:8000`"
        return "❌ **BGV server is offline** — run `uvicorn main:app --reload` in the project directory"

    demo.load(refresh_status, outputs=[server_status])

    extract_btn.click(
        fn=run_extract,
        inputs=[file_input],
        outputs=[ext_summary, ext_fields, gr.Textbox(visible=False), ext_raw],
    )

    verify_btn.click(
        fn=run_verify,
        inputs=[file_input],
        outputs=[ver_summary, ver_fields, ver_fraud, ver_vlm, ver_validation, ver_raw],
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )