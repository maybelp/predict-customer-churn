"""
Customer churn prediction library.

This module contains functions for loading customer churn data,
performing exploratory data analysis, engineering features, training
classification models, evaluating model performance, and saving results.

Author: Ligia Palomo
Date created: 2026-06-01
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import train_test_split


EDA_DIR = Path("./images/eda")
RESULTS_DIR = Path("./images/results")
MODELS_DIR = Path("./models")
DATA_PATH = Path("./data/bank_data.csv")

CATEGORICAL_COLUMNS = [
    "gender",
    "education_level",
    "marital_status",
    "income_category",
    "card_category",
]

EDA_PLOT_CONFIG = {
    "churn": "count",
    "customer_age": "hist",
    "marital_status": "count",
    "total_trans_ct": "hist",
}

MODEL_FEATURES = [
    "customer_age",
    "dependent_count",
    "months_on_book",
    "total_relationship_count",
    "months_inactive_12_mon",
    "contacts_count_12_mon",
    "credit_limit",
    "total_revolving_bal",
    "avg_open_to_buy",
    "total_amt_chng_q4_q1",
    "total_trans_amt",
    "total_trans_ct",
    "total_ct_chng_q4_q1",
    "avg_utilization_ratio",
    "gender_churn",
    "education_level_churn",
    "marital_status_churn",
    "income_category_churn",
    "card_category_churn",
]


def configure_environment() -> None:
    """
    Configure environment variables needed for headless plotting.

    Returns:
        None.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def create_output_directories() -> None:
    """
    Create output directories used by the project.

    Returns:
        None.
    """
    for directory in [EDA_DIR, RESULTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def import_data(pth: Path) -> pd.DataFrame:
    """
    Return a dataframe for the csv found at pth.

    Args:
        pth: Path to the csv file.

    Returns:
        Pandas dataframe with unnamed columns removed and column names lowercased.
    """
    df = pd.read_csv(pth)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.lower()

    return df


def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace blank strings with pandas missing values.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with blank strings replaced by missing values.
    """
    data = df.copy()
    data = data.replace(r"^\s*$", pd.NA, regex=True)

    return data


def get_missing_value_report(df: pd.DataFrame) -> pd.Series:
    """
    Return the number of missing values in each column.

    Args:
        df: Input dataframe.

    Returns:
        Series containing missing-value counts by column.
    """
    return df.isna().sum()


def add_churn_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary churn target column.

    Args:
        df: Input dataframe containing attrition_flag.

    Returns:
        Dataframe with churn column added.

    Raises:
        KeyError: If attrition_flag is missing.
        ValueError: If attrition_flag contains unexpected values.
    """
    if "attrition_flag" not in df.columns:
        raise KeyError("Required column not found: attrition_flag")

    data = df.copy()

    churn_mapping = {
        "Existing Customer": 0,
        "Attrited Customer": 1,
    }

    data["churn"] = data["attrition_flag"].map(churn_mapping)

    if data["churn"].isna().any():
        raise ValueError("Unexpected values found in attrition_flag.")

    return data


def split_data(
    df: pd.DataFrame,
    response: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into training and testing sets.

    Args:
        df: Input dataframe.
        response: Target column name.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Training and testing dataframes.

    Raises:
        KeyError: If the response column is not in the dataframe.
    """
    if response not in df.columns:
        raise KeyError(f"Response column not found: {response}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[response],
    )

    return train_df, test_df


def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform EDA on a prepared dataframe and save figures.

    Args:
        df: Prepared pandas dataframe containing a binary churn column.

    Raises:
        ValueError: If the dataframe does not contain the churn column.
        KeyError: If a required plotting column is missing.

    Returns:
        None.
    """
    create_output_directories()

    data = df.copy()

    if "churn" not in data.columns:
        raise ValueError(
            "perform_eda requires a prepared dataframe with a 'churn' column. "
            "Call add_churn_column before perform_eda."
        )

    missing_columns = [
        column for column in EDA_PLOT_CONFIG
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing EDA columns: {missing_columns}")

    for column, plot_type in EDA_PLOT_CONFIG.items():
        plt.figure(figsize=(10, 6))

        if plot_type == "count":
            sns.countplot(x=column, data=data)
            plt.xticks(rotation=45, ha="right")
        elif plot_type == "hist":
            sns.histplot(data[column], kde=False)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"{column}_distribution.png")
        plt.close()

    plt.figure(figsize=(16, 10))
    numeric_data = data.select_dtypes(include="number")

    sns.heatmap(
        numeric_data.corr(),
        annot=False,
        cmap="viridis",
        linewidths=0.5,
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png")
    plt.close()


def fit_missing_value_imputer(df: pd.DataFrame) -> dict:
    """
    Calculate missing-value fill values from a dataframe.

    Numerical columns are filled with the median.
    Categorical columns are filled with the mode.

    Args:
        df: Input dataframe used to calculate imputation values.

    Returns:
        Dictionary containing numerical medians and categorical modes.
    """
    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(
        include=["object", "category"],
    ).columns

    numeric_fill_values = df[numeric_columns].median().to_dict()

    categorical_fill_values = {}
    for column in categorical_columns:
        mode_values = df[column].mode(dropna=True)

        if mode_values.empty:
            categorical_fill_values[column] = "Unknown"
        else:
            categorical_fill_values[column] = mode_values.iloc[0]

    return {
        "numeric": numeric_fill_values,
        "categorical": categorical_fill_values,
    }


def apply_missing_value_imputer(
    df: pd.DataFrame,
    fill_values: dict,
) -> pd.DataFrame:
    """
    Fill missing values using precomputed fill values.

    Args:
        df: Input dataframe to transform.
        fill_values: Dictionary created by fit_missing_value_imputer.

    Returns:
        Dataframe with missing values filled.
    """
    data = df.copy()

    for column, value in fill_values["numeric"].items():
        if column in data.columns:
            data[column] = data[column].fillna(value)

    for column, value in fill_values["categorical"].items():
        if column in data.columns:
            data[column] = data[column].fillna(value)

    return data


def fit_target_encoder(
    train_df: pd.DataFrame,
    category_lst: list[str],
    response: str,
    smoothing: float = 10.0,
) -> tuple[dict[str, pd.Series], float]:
    """
    Fit smoothed target encoders using only the training data.

    Args:
        train_df: Training dataframe.
        category_lst: Categorical columns to encode.
        response: Target column name.
        smoothing: Strength of smoothing toward the global response mean.

    Returns:
        A dictionary of fitted encodings and the global response mean.

    Raises:
        KeyError: If response or categorical columns are missing.
    """
    if response not in train_df.columns:
        raise KeyError(f"Response column not found: {response}")

    global_mean = train_df[response].mean()
    encodings = {}

    for category in category_lst:
        if category not in train_df.columns:
            raise KeyError(f"Categorical column not found: {category}")

        category_stats = train_df.groupby(category)[response].agg(
            ["mean", "count"],
        )

        smoothed_encoding = (
            category_stats["count"] * category_stats["mean"]
            + smoothing * global_mean
        ) / (category_stats["count"] + smoothing)

        encodings[category] = smoothed_encoding

    return encodings, global_mean


def apply_target_encoder(
    df: pd.DataFrame,
    encodings: dict[str, pd.Series],
    category_lst: list[str],
    response: str,
    global_mean: float,
) -> pd.DataFrame:
    """
    Apply fitted target encoders to a dataframe.

    Args:
        df: Dataframe to transform.
        encodings: Encodings fitted from the training data.
        category_lst: Categorical columns to encode.
        response: Target column name.
        global_mean: Fallback value for unseen categories.

    Returns:
        Dataframe with target-encoded columns added.
    """
    data = df.copy()

    for category in category_lst:
        encoded_column = f"{category}_{response}"

        data[encoded_column] = data[category].map(encodings[category])
        data[encoded_column] = data[encoded_column].fillna(global_mean)

    return data


def perform_feature_engineering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    response: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Select model features and separate train/test data into X and y.

    Args:
        train_df: Training dataframe after cleaning and encoding.
        test_df: Testing dataframe after cleaning and encoding.
        response: Target column name.
        feature_columns: Optional list of feature columns to use.
            If None, MODEL_FEATURES is used.

    Returns:
        x_train: Training feature dataframe.
        x_test: Testing feature dataframe.
        y_train: Training target series.
        y_test: Testing target series.

    Raises:
        KeyError: If the response column or required feature columns are missing.
    """
    if response not in train_df.columns:
        raise KeyError(f"Response column not found in train_df: {response}")

    if response not in test_df.columns:
        raise KeyError(f"Response column not found in test_df: {response}")

    selected_features = feature_columns or MODEL_FEATURES

    missing_train_cols = set(selected_features) - set(train_df.columns)
    missing_test_cols = set(selected_features) - set(test_df.columns)

    if missing_train_cols:
        raise KeyError(f"Missing columns in train_df: {missing_train_cols}")

    if missing_test_cols:
        raise KeyError(f"Missing columns in test_df: {missing_test_cols}")

    x_train = train_df[selected_features]
    x_test = test_df[selected_features]

    y_train = train_df[response]
    y_test = test_df[response]

    return x_train, x_test, y_train, y_test


def classification_report_image(
    y_train: pd.Series,
    y_test: pd.Series,
    model_predictions: dict[str, dict[str, pd.Series]],
    output_path: Path,
) -> None:
    """
    Save classification reports for one or more trained models.

    Args:
        y_train: Training target labels.
        y_test: Testing target labels.
        model_predictions: Dictionary mapping model names to train/test predictions.
        output_path: Path where the report image will be saved.

    Raises:
        KeyError: If a model is missing train or test predictions.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_prediction_keys = {"train", "test"}

    missing_keys_by_model = {
        model_name: required_prediction_keys - set(predictions)
        for model_name, predictions in model_predictions.items()
        if required_prediction_keys - set(predictions)
    }

    if missing_keys_by_model:
        raise KeyError(
            f"Each model must provide train and test predictions. "
            f"Missing keys: {missing_keys_by_model}"
        )

    report_sections = []

    for model_name, predictions in model_predictions.items():
        train_report = classification_report(y_train, predictions["train"])
        test_report = classification_report(y_test, predictions["test"])

        report_sections.append(
            f"{model_name} Train\n{train_report}\n"
            f"{model_name} Test\n{test_report}"
        )

    report_text = "\n\n".join(report_sections)

    plt.figure(figsize=(12, 10))
    plt.text(
        0.01,
        0.99,
        report_text,
        family="monospace",
        fontsize=10,
        va="top",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def feature_importance_plot(
    model: RandomForestClassifier,
    x_data: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save feature importance plot.

    Args:
        model: Trained random forest model.
        x_data: Feature dataframe used for training.
        output_path: Path where the plot will be saved.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importances = pd.Series(
        model.feature_importances_,
        index=x_data.columns,
    )

    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=importances.values,
        y=importances.index,
    )
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curves(
    models: dict[str, object],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """
    Save ROC curves for trained classification models.

    Args:
        models: Dictionary mapping model display names to fitted classifiers.
        x_test: Testing feature dataframe.
        y_test: Testing target labels.
        output_path: Path where the ROC curve image will be saved.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, axis = plt.subplots(figsize=(10, 6))

    for model_name, model in models.items():
        RocCurveDisplay.from_estimator(
            model,
            x_test,
            y_test,
            name=model_name,
            ax=axis,
        )

    axis.set_title("ROC Curves")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """
    Train classification models, save trained models, and save evaluation outputs.

    Args:
        x_train: Training feature dataframe.
        x_test: Testing feature dataframe.
        y_train: Training target labels.
        y_test: Testing target labels.

    Returns:
        Dictionary containing the trained models.
    """
    create_output_directories()

    logistic_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )

    logistic_model.fit(x_train, y_train)
    random_forest_model.fit(x_train, y_train)

    y_train_preds_lr = logistic_model.predict(x_train)
    y_test_preds_lr = logistic_model.predict(x_test)

    y_train_preds_rf = random_forest_model.predict(x_train)
    y_test_preds_rf = random_forest_model.predict(x_test)

    joblib.dump(logistic_model, MODELS_DIR / "logistic_model.pkl")
    joblib.dump(random_forest_model, MODELS_DIR / "random_forest_model.pkl")

    classification_report_image(
        y_train,
        y_test,
        {
            "Logistic Regression": {
                "train": y_train_preds_lr,
                "test": y_test_preds_lr,
            },
            "Random Forest": {
                "train": y_train_preds_rf,
                "test": y_test_preds_rf,
            },
        },
        RESULTS_DIR / "classification_report.png",
    )

    feature_importance_plot(
        random_forest_model,
        x_train,
        RESULTS_DIR / "feature_importance.png",
    )

    plot_roc_curves(
        {
            "Logistic Regression": logistic_model,
            "Random Forest": random_forest_model,
        },
        x_test,
        y_test,
        RESULTS_DIR / "roc_curves.png",
    )

    return {
        "logistic_model": logistic_model,
        "random_forest_model": random_forest_model,
    }


def main() -> None:
    """Run the full churn prediction pipeline."""
    configure_environment()

    df = import_data(DATA_PATH)
    df = standardize_missing_values(df)
    df = add_churn_column(df)

    train_df, test_df = split_data(df, response="churn")

    perform_eda(train_df)

    fill_values = fit_missing_value_imputer(train_df)
    train_df = apply_missing_value_imputer(train_df, fill_values)
    test_df = apply_missing_value_imputer(test_df, fill_values)

    encodings, global_mean = fit_target_encoder(
        train_df,
        CATEGORICAL_COLUMNS,
        response="churn",
    )

    train_df = apply_target_encoder(
        train_df,
        encodings,
        CATEGORICAL_COLUMNS,
        response="churn",
        global_mean=global_mean,
    )

    test_df = apply_target_encoder(
        test_df,
        encodings,
        CATEGORICAL_COLUMNS,
        response="churn",
        global_mean=global_mean,
    )

    x_train, x_test, y_train, y_test = perform_feature_engineering(
        train_df,
        test_df,
        response="churn",
    )

    train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
