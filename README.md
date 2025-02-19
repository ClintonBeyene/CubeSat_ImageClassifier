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
📂 CubeSat_ImageClassify  
│── 📂 src                     # Core scripts for preprocessing, training, and evaluation  
│    ├── data_preprocessing.py  
│    ├── model_architecture.py  
│    ├── model_training.py  
│    ├── model_quantization.py  
│    ├── evaluate.py  
│── 📂 scripts                 # Additional analysis and visualization scripts  
│── 📂 notebooks               # Jupyter notebooks for development & experimentation  
│── 📂 data                    # Dataset used for training & validation  
│── 📂 models                  # Saved trained models  
│── README.md                  # Project documentation  
│── requirements.txt           # List of dependencies  
│── train.py                   # Main training script  
│── evaluate.py                # Model evaluation script  
```

## **Getting Started**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/ClintonBeyene/CubeSat_ImageClassifier
cd CubeSat_ImageClassify
```

### **2. Install Dependencies**  
Ensure you have Python installed, then run:  
```bash
pip install -r requirements.txt
```

### **3. Run the Training Pipeline**  
```bash
Execute the training pipeline process within the provided notebook.
```

### **4. Evaluate the Model**  
```bash
Run the evaluation step inside the designated Jupyter notebook.
```

## **Results & Performance**  

Our optimized model achieves **high classification accuracy while maintaining computational efficiency**, making it suitable for **real-world deployment on CubeSats**. Further optimizations include **pruning and quantization** to enhance efficiency for edge computing.  

## **Acknowledgments**  

We acknowledge the **Kyushu Institute of Technology (Kyutech)** and the **Hack4Dev CubeSat Challenge** for providing the dataset and problem statement.  

---  
