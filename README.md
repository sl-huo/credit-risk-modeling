# Credit Risk Modeling

Utilizing tree-based model (XGBoost) to predict credit risk (probability of default) for loans, leveraging `interation_constraints` to improve interpretability of the model and tree structures, and exploring scorecard feature development based on feature importance and tree plot.

## Model Result

### Feature Importance
![1](image/feature_importance.png)

### Tree Plot

Credit Score - Node Split
![2](image/cnsscore_tree.png)

LTV - Node Split
![3](image/ltv_tree.png)


---

Dataset source: [Kaggle]('https://www.kaggle.com/datasets/sneharshinde/ltfs-av-data?select=train.csv')

Notebook is inspired by GTC 2021 Building Credit Risk Scorecards [Demo]('https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard/cpu')