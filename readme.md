# Customer Click Prediction & Offer Ranking

## 🚀 Project Overview

This project tackles a challenging recommendation system task from a hackathon. The goal is to predict whether a customer will click on a given offer and to rank these offers to maximize the **Mean Average Precision @ 7 (MAP@7)**.

The primary challenges in the dataset were its large size, a significant user cold-start problem, and its time-series nature, with data split across consecutive days.

---

## 🛠️ Methodology & Approach

The solution leverages a gradient boosting model with a strong focus on sophisticated feature engineering to capture user session dynamics.

### 1. Data Preprocessing
- **Categorical Feature Handling:** Correctly handled null values and reversed multi-hot encoded features (e.g., `f226`-`f309`) into single, rich categorical columns like `sub_category` and `offer_category`. This was done by combining multiple labels with an underscore (`_`) separator to capture interactions.

### 2. Feature Engineering
This was the key to achieving a high score. The most impactful features created were:
- **Session-Based Features:**
  - `time_since_last`: Time in seconds since the user's last interaction.
  - `session_event_count`: A counter for the number of offers seen by the user in the current session.
- **Click-Based Sequential Features:**
  - `time_since_last_click`: Time in seconds since the user's *last clicked offer*, providing a powerful signal of user engagement.
  - `no_of_clicks_previously`: A cumulative count of clicks within the user's session.
- **Interaction Features:** Created interactions between the new sequential features and existing historical features to capture more complex patterns.

### 3. Modeling
- **Model:** `catboost.CatBoostRanker`
- **Objective:** A listwise ranking approach was used to directly optimize for the ranking task.
  - **Loss Function:** `Lambdamart`
  - **Evaluation Metric:** `MAP:top=7`

---

## 📊 Results

This approach, with a heavy emphasis on sequential feature engineering, achieved a final **MAP@7 score of 0.65** on the validation set. This demonstrates the critical importance of modeling user session behavior over relying solely on historical aggregates.

---
