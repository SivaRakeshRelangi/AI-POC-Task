# 🛡️ AI Safety Models POC

This project is a **Proof of Concept (POC)** for a suite of AI safety models aimed at enhancing user safety in a conversational AI platform (e.g., chat or messaging apps).

It includes a **FastAPI-based web service** that detects and classifies harmful or unsafe user-generated text across **multiple safety dimensions**.

---

## 🚀 Features

The system supports real-time detection of the following:

| Feature                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **Abuse Language Detection**     | Detects offensive, harmful, or aggressive language.                        |
| **Escalation Pattern Recognition** | Detects increasingly negative or hostile language over time.               |
| **Crisis Intervention**     | Detects potential signs of emotional distress or self-harm.                |
| **Age-appropriate Content Filtering** | Flags content that may be inappropriate for children or minors.             |

---
### 🧠 Model Info

- Pipeline: TF-IDF + Logistic Regression (multi-label via MultiOutputClassifier)

- Labels: abuse, escalation, crisis, age_inappropriate

- Dataset: Simulated data (can be extended with real-world annotated corpora)

### 🔒 Privacy & Ethics

- ⚠️ This is a prototype intended for research and demonstration.

- Not intended for production use without proper ethical review, safety evaluation, and moderation integration.

## 📁 Project Structure

### AI_POC_Task/

- **data/safety_data.csv**          #Simulated multi-label safety dataset

- **train_safety_model.py**        # Script to train the multi-label safety classification model
- **main.py**                      # FastAPI app for real-time prediction via REST API
- **safety_model.pkl**             # Trained multi-label model (serialized)
- **requirements.txt**             # List of dependencies
- **README.md**                    # Project documentation (this file)



---

## 📦 Installation & Setup

## Setup Instructions
1. **Set Up Virtual Environment**:
   ```bash
   conda create -p envtext python=3.10 -y

   


2. **Activate Environment**:
   ```bash
   conda activate ./envtext

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt


4. **Train the model**:
   ```bash
   python train_safety_model.py

This script trains a multi-label text classification model and saves it as safety_model.pkl.

5. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload

Visit the interactive Swagger docs:
📍 http://127.0.0.1:8000/docs

## 📥 API Usage

### `POST /predict`
Use POST Method

**Request:**

```bash
{
  "text": "I'm thinking about ending it all"
}
```

**Response**
```bash
{
  "input": "I'm thinking about ending it all",
  "safety_flags": {
    "abuse": 0,
    "escalation": 0,
    "crisis": 1,
    "age_inappropriate": 0
  }
}
```
### 📊 Model Evaluation

- Use the evaluation script to generate metrics like **precision**, **recall**, **F1-score**, and exact match ratio for the multi-label classifier.

- Run the evaluation script:
```bash
python evaluate_model.py
```
- Sample Output:
#### 🔍 Classification Report (per label):

                   precision    recall  f1-score   support

           abuse         1.00      1.00      1.00       1
      escalation         1.00      0.50      0.67       2
           crisis        1.00      1.00      1.00       1
age_inappropriate        1.00      1.00      1.00       1

✅ Label-wise Accuracy:
- abuse: 1.00
- escalation: 0.50
- crisis: 1.00
- age_inappropriate: 1.00

🎯 Exact Match Ratio (all labels correct per instance): 0.80
### 🧩 Future Enhancements

- 🤖 Use transformers (BERT, RoBERTa) for better accuracy

- 🔄 WebSocket support for live chat systems

- 🧍 Human-in-the-loop moderation triggers

- 🌍 Multilingual support


- 📊 Real-time monitoring dashboard
