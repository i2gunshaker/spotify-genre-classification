# Spotify Song Genre Classification

A comprehensive machine learning project for classifying Spotify songs into genres using audio features and metadata. This project compares multiple classification algorithms and neural network architectures to achieve optimal genre prediction.

## 📋 Project Overview

This project implements and evaluates various machine learning models to classify songs into different genres based on Spotify's audio features. The analysis includes extensive feature engineering, hyperparameter tuning, and model comparison to identify the best performing approach for genre classification.

## 🎯 Objectives

- Classify Spotify songs into genres using audio features
- Compare performance of traditional ML algorithms and neural networks
- Implement feature engineering and dimensionality reduction techniques
- Optimize models through hyperparameter tuning
- Evaluate models using multiple metrics (accuracy, F1-score, ROC-AUC, log loss)

## 🔧 Technologies & Libraries

### Core Libraries
- **Python 3.10+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and preprocessing

### Machine Learning Models
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- XGBoost
- CatBoost
- Naive Bayes
- Linear Discriminant Analysis (LDA)
- Stochastic Gradient Descent (SGD)

### Deep Learning
- **TensorFlow/Keras** - Neural network implementation
- Multiple NN architectures with different configurations

### Additional Tools
- **Category Encoders** - Target encoding for high-cardinality features
- **Seaborn & Matplotlib** - Data visualization
- **SciPy** - Statistical functions

## 📊 Dataset Features

The project uses Spotify's audio features including:
- **Acousticness** - Confidence measure of acoustic content
- **Danceability** - How suitable a track is for dancing
- **Energy** - Intensity and activity measure
- **Instrumentalness** - Predicts whether a track contains vocals
- **Liveness** - Presence of an audience in the recording
- **Loudness** - Overall loudness in decibels
- **Speechiness** - Presence of spoken words
- **Tempo** - Overall estimated tempo (BPM)
- **Valence** - Musical positiveness conveyed
- **Track Artist** - Artist information (encoded)

## 🚀 Methodology

### 1. Data Preprocessing
- Handling missing values
- Feature scaling using StandardScaler
- Target encoding for categorical variables (track_artist)
- Label encoding for genre classification

### 2. Feature Engineering
- Dimensionality reduction using PCA
- Feature interaction creation
- Correlation analysis
- Feature importance evaluation

### 3. Model Training
- Train-test split (80-20)
- Stratified K-Fold cross-validation
- Hyperparameter tuning using RandomizedSearchCV
- Multiple neural network architectures with different activations and depths

### 4. Model Evaluation
- Accuracy score
- F1-score (weighted)
- Classification reports
- Confusion matrices
- ROC curves and AUC scores
- Log loss calculation

## 📈 Results

The project evaluates models based on multiple metrics:

### Top Performing Models
1. **XGBoost** - Gradient boosting implementation
2. **CatBoost** - Categorical boosting algorithm
3. **Neural Networks** - Deep learning with ReLU activation
4. **Extra Trees** - Ensemble of randomized decision trees

### Key Findings
- Ensemble methods (XGBoost, CatBoost) showed superior performance
- Feature engineering significantly improved model accuracy
- Neural networks with wider architectures performed well
- Log loss analysis revealed CatBoost as the most calibrated model

## 💻 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install category_encoders
pip install catboost
pip install xgboost
pip install tensorflow
```

### Running the Notebook
1. Clone this repository
2. Install required dependencies
3. Open `spotify_labwork7_CSS330-2.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells sequentially

```bash
jupyter notebook spotify_labwork7_CSS330-2.ipynb
```

## 📂 Project Structure

```
.
├── spotify_labwork7_CSS330-2.ipynb    # Main analysis notebook
├── README.md                           # Project documentation
└── data/                               # Dataset directory (if applicable)
```

## 🔍 Key Features

- **Comprehensive Model Comparison**: Evaluates 12+ different algorithms
- **Hyperparameter Optimization**: Uses RandomizedSearchCV for efficient tuning
- **Neural Network Experimentation**: Tests multiple architectures and activation functions
- **Robust Evaluation**: Multiple metrics ensure comprehensive assessment
- **Visualization**: ROC curves, confusion matrices, and performance comparisons
- **Reproducibility**: Fixed random seeds for consistent results

## 📊 Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to distinguish between classes
- **Log Loss**: Probabilistic confidence of predictions
- **Confusion Matrix**: Detailed class-wise performance

## 🎓 Course Information

- **Course**: CSS330
- **Assignment**: Lab Work 7
- **Topic**: Machine Learning Classification

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is created for educational purposes as part of CSS330 coursework.

## 👤 Author

**Aitugan Shagyr**

## 🙏 Acknowledgments

- Spotify for providing the audio features dataset
- Scikit-learn documentation and community
- TensorFlow/Keras teams for deep learning frameworks
- CSS330 course instructors

## 📧 Contact

For questions or feedback, please reach out through the course platform.

---

**Note**: This project demonstrates practical application of machine learning techniques for music genre classification and serves as a learning resource for comparative model analysis.
