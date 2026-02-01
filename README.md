# Project Overview

This project is a machine learning application for predicting the price of diamonds and segmenting them into different market categories. It consists of two main components:

1.  **Price Prediction:** A regression model that predicts the price of a diamond based on its characteristics (carat, cut, color, clarity, etc.).
2.  **Market Segmentation:** A clustering model that groups diamonds into different market segments (e.g., "Affordable Small Diamonds", "Mid-range Balanced Diamonds", "Premium Heavy Diamonds").

The project includes a Streamlit web application for interactive predictions and visualizations.

## Technologies Used

*   **Language:** Python
*   **Libraries:**
    *   `scikit-learn`: For building and training the regression and clustering models.
    *   `pandas`, `numpy`: For data manipulation and numerical operations.
    *   `streamlit`: For creating the interactive web application.
    *   `tensorflow`: For training a deep learning model for price prediction.
    *   `joblib`: For saving and loading trained models.
    *   `matplotlib`, `seaborn`: For data visualization.

## Project Structure

*   `app.py`: The main entry point for the Streamlit web application.
*   `src/`: Contains the core source code for the project.
    *   `train_regression.py`: Script for training the price prediction model.
    *   `train_clustering.py`: Script for training the market segmentation model.
    *   `feature_engineering.py`: Contains functions for creating new features from the raw data.
    *   `preprocessing.py`: Contains functions for building the data preprocessing pipeline.
*   `models/`: Stores the trained machine learning models.
*   `data/`: Contains the raw data used for training the models.
*   `notebooks/`: Contains Jupyter notebooks for data exploration and analysis.
*   `requirements-train.txt`: Python dependencies for training the models.
*   `requirements-infer.txt`: Python dependencies for running the Streamlit application.
*   `Dockerfile`: For containerizing the application.

# Building and Running

## Running the Web Application

To run the Streamlit web application, you need to have the required dependencies installed. You can install them using pip:

```bash
pip install -r requirements-infer.txt
```

Once the dependencies are installed, you can run the application using the following command:

```bash
streamlit run app.py
```

This will start a local web server, and you can access the application in your browser at `http://localhost:8501`.

## Training the Models

To train the models, you need to have the required dependencies installed. You can install them using pip:

```bash
pip install -r requirements-train.txt
```

### Training the Regression Model

To train the price prediction model, run the following command:

```bash
python src/train_regression.py
```

This will train a `RandomForestRegressor` and a neural network model and save them to the `models/` directory.

### Training the Clustering Model

To train the market segmentation model, run the following command:

```bash
python src/train_clustering.py
```

This will train a `KMeans` model and save it to the `models/` directory.

# Development Conventions

*   **Code Style:** The code follows the general Python conventions (PEP 8).
*   **Modularity:** The code is organized into separate modules for different functionalities (e.g., feature engineering, preprocessing, training).
*   **Model Persistence:** Trained models are saved to the `models/` directory using `joblib` and `keras`.
*   **Dependency Management:** The project uses separate `requirements.txt` files for training and inference to keep the environments clean.
