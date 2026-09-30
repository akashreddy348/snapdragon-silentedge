# SilentEdge: Zero-Radiation Acoustic Threat Hunter

Submission for the **Snapdragon® AI Lab Build & Present Challenge**.

## 1. Problem Statement
Microphone-enabled spyware and nearby eavesdropping rigs can reconstruct sensitive keyboard input (passwords, private keys, emails) via acoustic reflection analysis. Traditional CPU/GPU detection models drain laptop batteries and introduce noticeable system lag.

## 2. Our Solution
SilentEdge is a lightweight background defender tailored for Snapdragon-powered HP PCs. It streams audio micro-buffers directly into a quantized temporal anomaly model running on the **Hexagon NPU** via the **Qualcomm AI Hub** and **ONNX Runtime QNN Provider**. It identifies acoustic signatures in real time while maintaining sub-5ms latency and near-zero CPU drain.

## 3. Architecture & Snapdragon Optimization
* **Hardware Target:** Snapdragon® X-Series Hexagon Tensor Processor (HTP) on HP Laptops.
* **Model Pipeline:** Quantized AST / Autoencoder compiled via `qai-hub` to `.bin` / `.onnx` context binaries.
* **Runtime:** `onnxruntime-qnn` leveraging `QnnHtp.dll`.
* **Resource Profile:** < 5% CPU utilization, no fan activation, continuous offline operation.

## 4. How to Run
```bash
git clone [https://github.com/](https://github.com/)<your-username>/snapdragon-silentedge.git
cd snapdragon-silentedge
pip install -r requirements.txt
streamlit run app.py
