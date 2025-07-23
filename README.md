#  Sentiment Analysis on Financial News and Stock Price Movement Prediction

This project explores how financial news sentiment influences stock price movements, using **natural language processing**, **FinBERT**, and **machine learning models** like **XGBoost** and **Logistic Regression**.

---

##  Workflow Overview

| Step | Task |
|------|------|
| 1️⃣ | Collect high-quality financial news using `NewsAPI` |
| 2️⃣ | Clean and preprocess headlines/text (language detection, translation, lemmatization) |
| 3️⃣ | Apply FinBERT to assign sentiment labels and scores |
| 4️⃣ | Merge with stock price data (adjusted via ±3 day window) |
| 5️⃣ | Label movement: `+1` (Up), `-1` (Down), `0` (Flat) based on next-day returns |
| 6️⃣ | Feature engineering: TF-IDF / BERT, sentiment, hour, weekday |
| 7️⃣ | Train ML models and compare performance |
| 8️⃣ | Visualize & analyze model behavior and errors |
| 9️⃣ | Final discussion + Real-world takeaways |

---

## Final Model Performance (XGBoost with Semantic Embeddings)

### Classification Report:

| Label         | Precision | Recall   | F1-score | Notes                                                                                       |
| ------------- | --------- | -------- | -------- | ------------------------------------------------------------------------------------------- |
| **Down (-1)** | 0.67      | **0.91** | 0.77     | Model is very good at **catching price drops**, but may generate some false positives.      |
| **Flat (0)**  | 0.72      | 0.84     | 0.77     | Excellent performance on neutral/flat days. Balanced.                                       |
| **Up (+1)**   | 0.74      | 0.35     | 0.48     | High **precision** (good when it predicts), but **low recall** (misses many real Up cases). |

---

##  Model Comparison

| Model                | Accuracy | Observations |
|---------------------|----------|--------------|
| **XGBoost (Full, Semantic)** | ~71%     | Best performance using embeddings + full data |
| Logistic Regression (Balanced) | ~42%     | Struggles even on balanced data |
| XGBoost (Balanced)   | ~42%     | Balanced class learning, but lower accuracy |
| XGBoost (TF-IDF only)| ~50–60%  | Decent, but lacks semantic richness |

---

##  Imbalanced vs Balanced Learning

| Aspect                     | High-Accuracy Model (`~71%`)                     | Balanced Model (`~42%`)                   |
| -------------------------- | ------------------------------------------------ | ----------------------------------------- |
| **Class Distribution**     | **Imbalanced** (more `+1` and `0`)               | **Balanced** (equal `-1`, `0`, `+1`)      |
| **Real-World Bias**        | Reflects real-world skew (more `Neutral` / `Up`) | Artificial balancing removed natural bias |
| **Data Volume**            | Used all \~2000+ examples                        | Downsampled to \~353 examples (117/class) |
| **Learning Capacity**      | XGBoost learns better from more data             | Less data limits generalization           |
| **TF-IDF Vectorizer**      | Trained on full vocabulary                       | Trained on subset (lower-quality vectors) |
| **Distribution Alignment** | Test set mirrors training set distribution       | Test set is forced equal (not realistic)  |

---

##  Key Features Used

- `cleaned_text` (TF-IDF or BERT embeddings)
- `sentiment_score` (from FinBERT)
- `hour`, `weekday` (temporal)
- Top keywords: `growth`, `strong`, `earnings`, `blank`, `drop`, etc.

---

##  Real-World Challenges

| Challenge | What We Did |
|----------|--------------|
| **News time mismatch** | Used ±5 day matching with fallback logic |
| **Bias from faulty sentiment** | Switched to NewsAPI dataset + verified FinBERT |
| **Data imbalance** | Compared imbalanced vs downsampled models |
| **Noisy features (e.g., "blank")** | Switched to **semantic embeddings** for better learning |
| **Neutral news causing movement** | Acknowledged as part of market noise & lagged effects |

---

##  Dataset Summary

| Source | Description |
|--------|-------------|
| **NewsAPI** | Apple news for last 30 days (~2900 articles) |
| **Yahoo Finance / yfinance** | Apple stock prices (OHLC + Close) |
| **FinBERT** | Sentiment scores (Positive / Negative / Neutral) |

---

##  Movement Labeling Logic

- `+1`: Next-day % change > +0.5%
- `-1`: Next-day % change < -0.5%
- `0`: Between -0.5% and +0.5%

---

##  Common Errors Observed

- **Lag effect**: Positive news today, price drops due to macro/future guidance
- **Market inertia**: Neutral or old news had delayed impact
- **Same news across articles** (duplication bias)

---

> **Note**: A version of this dataset using faulty Investing.com scraping is archived separately. It is marked as **unreliable** due to timestamp mismatches and is not used in the final model.
