---
title: StabilityArc
sdk: gradio
app_file: app.py
---

# StabilityArc Space

Upload this directory as a separate Hugging Face Gradio Space. The app downloads the decoder and inference code from [aaronfeller/StabilityArc](https://huggingface.co/aaronfeller/StabilityArc); the `esm` package downloads frozen ESMC-600M weights on first request. Choose GPU hardware with sufficient memory for ESMC-600M. A CPU Space may be slow or run out of memory. The displayed table shows the top 100 predicted stabilizing substitutions; the downloadable CSV contains all 19 substitutions per position. Predictions are model scores, not calibrated experimental kcal/mol values.