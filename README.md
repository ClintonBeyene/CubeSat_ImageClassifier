---

# **CubeSat Image Classification – VERTECS Mission**  

## **Introduction**  

This repository presents our comprehensive solution for the **CubeSat Image Classification Challenge**, designed to enhance data transmission efficiency for resource-constrained CubeSats. Our approach integrates **data preprocessing, model training, and evaluation** into a single, structured pipeline, ensuring reproducibility and clarity.  

Inspired by the **VERTECS mission**, a collaboration led by the **Kyushu Institute of Technology (Kyutech)**, this project aims to classify astronomical images captured by CubeSats to prioritize the most valuable data for transmission.  

## **Problem Overview**  

### **CubeSats & the VERTECS Mission**  

CubeSats are compact, cost-effective satellites used for space research, but they suffer from limitations in **storage, processing power, and communication bandwidth**. The **VERTECS mission** studies the **optical extragalactic background light (EBL)** to gain insights into star formation history. Equipped with a **small-aperture telescope and high-precision attitude control**, VERTECS captures crucial astronomical data for further ground analysis.  

<p align="center">
  <img src="https://raw.githubusercontent.com/Hack4Dev/CubeSat_ImageClassify/main/pictures/SAT.png" width="700">
</p>  
<p align="center"><i>Figure 1: VERTECS Satellite Design (Source: Hack4Dev)</i></p>  

However, due to **slow data transmission speeds**, not all images can be sent back to Earth. **Onboard machine learning** offers a promising solution by **intelligently filtering and prioritizing images**, ensuring that only high-priority data is transmitted while conserving bandwidth.  

## **Hackathon Challenge**  

The goal of this challenge is to develop an **efficient, lightweight machine learning model** that can accurately classify CubeSat images, enabling **real-time decision-making onboard** while balancing **computational efficiency and classification accuracy**.  

### **Solution Approach**  

Our solution follows a structured pipeline:  

1. **Data Ingestion & Preprocessing** – Cleaning and preparing raw CubeSat image data for training.  
2. **Model Architecture Design** – Developing a resource-efficient model suited for onboard classification.  
3. **Model Training & Optimization** – Training and fine-tuning the model for high accuracy with minimal computational cost.  
4. **Model Quantization & Compression** – Reducing model size for deployment on resource-constrained hardware.  
5. **Evaluation & Validation** – Ensuring robust performance through rigorous testing.  

This repository encapsulates the entire pipeline, ensuring transparency and reproducibility.  

## **Repository Structure**  

```
📂 CubeSat_ImageClassifier
│── 📂 sagemaker_deploy            # Backend API (FastAPI) + Dockerfile
│   ├── app/
│   │   ├── main.py               # FastAPI app (POST /predict)
│   │   └── functions.py          # TFLite load, preprocess, inference
│   ├── Dockerfile                # Backend container image
│   └── requirements.txt         # API dependencies
│── 📂 frontend                   # Next.js (UI)
│   ├── pages/                    # index.tsx, _app.tsx, etc.
│   ├── components/               # Starfield, etc.
│   ├── styles/                   # CSS modules
│   ├── package.json
│   └── .env.example              # Public API URL (NEXT_PUBLIC_API_URL)
│── render.yaml                   # Render blueprint for backend deployment
│── Cubenet_Best_Quantized.tflite # Quantized TFLite model used by API
│── 📂 data                       # NPY datasets (ignored in VCS as needed)
│── 📂 notebooks                  # Experimentation notebooks
│── 📂 scripts / 📂 src            # Training/eval utilities
│── README.md                     # This file
```

## **Getting Started**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/ClintonBeyene/CubeSat_ImageClassifier
cd CubeSat_ImageClassifier
```

### **2. Run the Backend API (FastAPI)**
Install API dependencies and start the server locally:
```bash
pip install -r sagemaker_deploy/requirements.txt
uvicorn sagemaker_deploy.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open API docs: http://localhost:8000/docs

### **3. Run the Frontend (Next.js)**
From `frontend/`:
```bash
copy .env.example .env
# edit .env to point NEXT_PUBLIC_API_URL to your backend (e.g., http://localhost:8000/predict)
npm install
npm run dev
```
Visit UI: http://localhost:3000

### **4. Train & Evaluate (optional)**
Use the provided notebooks under `notebooks/` to run training/evaluation. Any Python utilities reside under `src/` and `scripts/`.

## **Results & Performance**  

Our optimized model achieves **high classification accuracy while maintaining computational efficiency**, making it suitable for **real-world deployment on CubeSats**. Further optimizations include **pruning and quantization** to enhance efficiency for edge computing.  

## **Acknowledgments**  

We acknowledge the **Kyushu Institute of Technology (Kyutech)** and the **Hack4Dev CubeSat Challenge** for providing the dataset and problem statement.  

---  

## Deployment & Operations

### Backend: Docker

Recommended build (uses Dockerfile under `sagemaker_deploy/` and repo root as context):

```bash
docker build -f sagemaker_deploy/Dockerfile -t cubesat-image-api:latest . --provenance=false --output type=docker
```

Your requested command (equivalent if run from repo root):

```bash
docker build -t cubesat-image-api:latest . --provenance=false --output type=docker
```

Run locally:

```bash
docker run --rm -p 8080:80 cubesat-image-api:latest
# API now at http://localhost:8080/docs and POST /predict
```

Push to Docker Hub:

```bash
docker tag cubesat-image-api:latest <your-dockerhub-username>/cubesat-image-api:latest
docker push <your-dockerhub-username>/cubesat-image-api:latest
```

### Backend: Render (already configured)

- Infra file: `render.yaml` (env: docker, healthCheckPath: /docs, region: frankfurt)
- Dockerfile: `sagemaker_deploy/Dockerfile`
- Model file is included in the image as `Cubenet_Best_Quantized.tflite`

### Frontend: Vercel

- Root: `frontend/`
- Env var: `NEXT_PUBLIC_API_URL` must point to your backend `/predict`
- Build: `npm run build`; Start: `npm start`

### Optional: AWS ECS (Fargate) from Docker Hub image

High-level steps:
- Create a task definition (Fargate) exposing container port 80
- Use image `<your-dockerhub-username>/cubesat-image-api:latest`
- Service: 1+ tasks in a public subnet with an ALB or a public IP
- Health check path: `/docs`
- Security group: allow inbound 80/443 from the internet

## API Reference

- POST `/predict`
  - Request: `multipart/form-data` with field `file` (image/*)
  - Response JSON:
    - `predicted_class`: string
    - `class_index`: integer
    - `probability`: float (0..1)

## Environment Variables

- Frontend (`frontend/.env`)
  - `NEXT_PUBLIC_API_URL` — full backend URL including `/predict`
    - Example: `https://cubesat-backend.onrender.com/predict`

## Roadmap / Tasks

- [ ] Create Search API with Fast API
- [ ] Create Docker Image for API — “docker build -t cubesat-image-api:latest . --provenance=false --output type=docker”
- [ ] Push Image to Docker HUB
- [ ] Deploy Container on AWS ECS
- [ ] Build the app and deployment

