"""Tests for the customer churn prediction library."""

from pathlib import Path

import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier

import churn_library as cl


DATA_PATH = Path("data/bank_data.csv")


def make_feature_dataframe() -> pd.DataFrame:
    """Create a small dataframe containing all model feature columns."""
    return pd.DataFrame(
        {
            "customer_age": [45, 49, 51, 40],
            "dependent_count": [3, 5, 3, 4],
            "months_on_book": [39, 44, 36, 34],
            "total_relationship_count": [5, 6, 4, 3],
            "months_inactive_12_mon": [1, 1, 1, 4],
            "contacts_count_12_mon": [3, 2, 0, 1],
            "credit_limit": [12691.0, 8256.0, 3418.0, 3313.0],
            "total_revolving_bal": [777, 864, 0, 2517],
            "avg_open_to_buy": [11914.0, 7392.0, 3418.0, 796.0],
            "total_amt_chng_q4_q1": [1.335, 1.541, 2.594, 1.405],
            "total_trans_amt": [1144, 1291, 1887, 1171],
            "total_trans_ct": [42, 33, 20, 20],
            "total_ct_chng_q4_q1": [1.625, 3.714, 2.333, 2.333],
            "avg_utilization_ratio": [0.061, 0.105, 0.0, 0.76],
            "gender_churn": [0.14, 0.17, 0.14, 0.17],
            "education_level_churn": [0.16, 0.15, 0.15, 0.16],
            "marital_status_churn": [0.12, 0.19, 0.12, 0.20],
            "income_category_churn": [0.13, 0.18, 0.13, 0.18],
            "card_category_churn": [0.15, 0.15, 0.15, 0.21],
            "churn": [0, 1, 0, 1],
        }
    )


def test_import_data_returns_dataframe():
    """Test that import_data loads the churn dataset."""
    df = cl.import_data(DATA_PATH)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "attrition_flag" in df.columns
    assert "customer_age" in df.columns
    assert "Unnamed: 0" not in df.columns


def test_standardize_missing_values_replaces_blank_strings():
    """Test that blank strings are converted to missing values."""
    df = pd.DataFrame(
        {
            "education_level": ["Graduate", "", "   ", "High School"],
            "customer_age": [45, 50, 40, 35],
        }
    )

    result = cl.standardize_missing_values(df)

    assert result["education_level"].isna().sum() == 2


def test_get_missing_value_report_counts_missing_values():
    """Test that missing-value report counts missing values by column."""
    df = pd.DataFrame(
        {
            "credit_limit": [1000.0, None, 3000.0],
            "education_level": ["Graduate", None, "High School"],
        }
    )

    report = cl.get_missing_value_report(df)

    assert report["credit_limit"] == 1
    assert report["education_level"] == 1


def test_add_churn_column_creates_binary_target():
    """Test that add_churn_column creates a binary churn column."""
    df = pd.DataFrame(
        {
            "attrition_flag": [
                "Existing Customer",
                "Attrited Customer",
                "Existing Customer",
            ]
        }
    )

    result = cl.add_churn_column(df)

    assert "churn" in result.columns
    assert result["churn"].tolist() == [0, 1, 0]


def test_add_churn_column_raises_error_for_unexpected_label():
    """Test that unexpected attrition labels raise ValueError."""
    df = pd.DataFrame(
        {
            "attrition_flag": [
                "Existing Customer",
                "Unknown Customer",
            ]
        }
    )

    with pytest.raises(ValueError):
        cl.add_churn_column(df)


def test_split_data_preserves_rows():
    """Test that split_data preserves all rows across train and test sets."""
    df = pd.DataFrame(
        {
            "feature": range(100),
            "churn": [0, 1] * 50,
        }
    )

    train_df, test_df = cl.split_data(df, response="churn")

    assert len(train_df) + len(test_df) == len(df)
    assert "churn" in train_df.columns
    assert "churn" in test_df.columns


def test_split_data_raises_error_for_missing_response():
    """Test that split_data raises KeyError if response column is missing."""
    df = pd.DataFrame({"customer_age": [40, 50, 60]})

    with pytest.raises(KeyError):
        cl.split_data(df, response="churn")


def test_missing_value_imputer_fits_and_applies_train_values():
    """Test that missing values are filled using values fitted from training data."""
    train_df = pd.DataFrame(
        {
            "credit_limit": [1000.0, None, 3000.0],
            "education_level": ["Graduate", None, "Graduate"],
        }
    )

    test_df = pd.DataFrame(
        {
            "credit_limit": [None, 9999.0],
            "education_level": [None, "High School"],
        }
    )

    fill_values = cl.fit_missing_value_imputer(train_df)
    result = cl.apply_missing_value_imputer(test_df, fill_values)

    expected_credit_limit_median = train_df["credit_limit"].median()
    expected_education_level_mode = train_df["education_level"].mode().iloc[0]

    assert result.isna().sum().sum() == 0
    assert result.loc[0, "credit_limit"] == expected_credit_limit_median
    assert result.loc[0, "education_level"] == expected_education_level_mode


def test_target_encoder_creates_encoded_columns():
    """Test that target encoder creates smoothed encoded columns."""
    train_df = pd.DataFrame(
        {
            "gender": ["M", "F", "M", "F"],
            "education_level": [
                "Graduate",
                "Graduate",
                "High School",
                "Graduate",
            ],
            "churn": [0, 1, 0, 1],
        }
    )

    category_columns = ["gender", "education_level"]

    encodings, global_mean = cl.fit_target_encoder(
        train_df,
        category_columns,
        response="churn",
        smoothing=10.0,
    )

    result = cl.apply_target_encoder(
        train_df,
        encodings,
        category_columns,
        response="churn",
        global_mean=global_mean,
    )

    assert "gender_churn" in result.columns
    assert "education_level_churn" in result.columns
    assert result["gender_churn"].isna().sum() == 0
    assert result["education_level_churn"].isna().sum() == 0


def test_target_encoder_uses_global_mean_for_unseen_category():
    """Test that unseen categories are filled using train global mean."""
    train_df = pd.DataFrame(
        {
            "gender": ["M", "F", "M", "F"],
            "churn": [0, 1, 0, 1],
        }
    )

    test_df = pd.DataFrame(
        {
            "gender": ["Unknown"],
            "churn": [0],
        }
    )

    encodings, global_mean = cl.fit_target_encoder(
        train_df,
        ["gender"],
        response="churn",
        smoothing=10.0,
    )

    result = cl.apply_target_encoder(
        test_df,
        encodings,
        ["gender"],
        response="churn",
        global_mean=global_mean,
    )

    assert result.loc[0, "gender_churn"] == global_mean


def test_fit_target_encoder_raises_error_for_missing_category():
    """Test that target encoder raises KeyError for missing category column."""
    train_df = pd.DataFrame({"gender": ["M", "F"], "churn": [0, 1]})

    with pytest.raises(KeyError):
        cl.fit_target_encoder(
            train_df,
            ["education_level"],
            response="churn",
        )


def test_fit_target_encoder_raises_error_for_missing_response():
    """Test that target encoder raises KeyError for missing response column."""
    train_df = pd.DataFrame(
        {
            "gender": ["M", "F"],
        }
    )

    with pytest.raises(KeyError):
        cl.fit_target_encoder(
            train_df,
            ["gender"],
            response="churn",
        )


def test_perform_feature_engineering_returns_expected_outputs():
    """Test feature engineering separates X and y correctly."""
    train_df = make_feature_dataframe()
    test_df = make_feature_dataframe()

    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
        train_df,
        test_df,
        response="churn",
    )

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert "churn" not in x_train.columns
    assert "churn" not in x_test.columns
    assert y_train.tolist() == [0, 1, 0, 1]
    assert y_test.tolist() == [0, 1, 0, 1]


def test_perform_feature_engineering_raises_error_for_missing_feature():
    """Test feature engineering raises KeyError when a feature is missing."""
    train_df = make_feature_dataframe().drop(columns=["credit_limit"])
    test_df = make_feature_dataframe()

    with pytest.raises(KeyError):
        cl.perform_feature_engineering(train_df, test_df, response="churn")


def test_feature_importance_plot_saves_file(tmp_path):
    """Test that feature_importance_plot saves an image."""
    x_data = pd.DataFrame(
        {
            "feature_one": [0.0, 1.0, 0.2, 0.8],
            "feature_two": [1.0, 0.0, 0.8, 0.2],
        }
    )
    y_data = pd.Series([0, 1, 0, 1])

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
    model.fit(x_data, y_data)

    output_path = tmp_path / "feature_importance.png"

    cl.feature_importance_plot(model, x_data, output_path)

    assert output_path.exists()


def test_train_models_saves_models_and_images(tmp_path, monkeypatch):
    """Test that train_models saves model and evaluation artifacts."""
    monkeypatch.setattr(cl, "EDA_DIR", tmp_path / "eda")
    monkeypatch.setattr(cl, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cl, "MODELS_DIR", tmp_path / "models")

    x_train = pd.DataFrame(
        {
            "feature_one": [0.0, 1.0, 0.2, 0.8, 0.1, 0.9],
            "feature_two": [1.0, 0.0, 0.8, 0.2, 0.9, 0.1],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1])

    x_test = pd.DataFrame(
        {
            "feature_one": [0.05, 0.95],
            "feature_two": [0.95, 0.05],
        }
    )
    y_test = pd.Series([0, 1])

    cl.train_models(x_train, x_test, y_train, y_test)

    assert (cl.MODELS_DIR / "logistic_model.pkl").exists()
    assert (cl.MODELS_DIR / "random_forest_model.pkl").exists()
    assert (cl.RESULTS_DIR / "classification_report.png").exists()
    assert (cl.RESULTS_DIR / "feature_importance.png").exists()
    assert (cl.RESULTS_DIR / "roc_curves.png").exists()


def test_create_output_directories_creates_expected_folders(
        tmp_path, monkeypatch):
    """Test that output directories are created."""
    monkeypatch.setattr(cl, "EDA_DIR", tmp_path / "eda")
    monkeypatch.setattr(cl, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cl, "MODELS_DIR", tmp_path / "models")

    cl.create_output_directories()

    assert cl.EDA_DIR.exists()
    assert cl.RESULTS_DIR.exists()
    assert cl.MODELS_DIR.exists()


def test_perform_eda_saves_expected_figures(tmp_path, monkeypatch):
    """Test that perform_eda saves expected EDA figures."""
    monkeypatch.setattr(cl, "EDA_DIR", tmp_path / "eda")
    monkeypatch.setattr(cl, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cl, "MODELS_DIR", tmp_path / "models")

    df = pd.DataFrame(
        {
            "attrition_flag": [
                "Existing Customer",
                "Attrited Customer",
                "Existing Customer",
                "Attrited Customer",
            ],
            "customer_age": [45, 49, 51, 40],
            "marital_status": ["Married", "Single", "Married", "Unknown"],
            "total_trans_ct": [42, 33, 20, 20],
            "credit_limit": [12691.0, 8256.0, 3418.0, 3313.0],
        }
    )

    cl.perform_eda(df)

    assert (cl.EDA_DIR / "churn_distribution.png").exists()
    assert (cl.EDA_DIR / "customer_age_distribution.png").exists()
    assert (cl.EDA_DIR / "marital_status_distribution.png").exists()
    assert (cl.EDA_DIR / "total_trans_ct_distribution.png").exists()
    assert (cl.EDA_DIR / "correlation_heatmap.png").exists()


def test_classification_report_image_saves_file(tmp_path, monkeypatch):
    """Test that classification_report_image saves a report image."""
    monkeypatch.setattr(cl, "EDA_DIR", tmp_path / "eda")
    monkeypatch.setattr(cl, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cl, "MODELS_DIR", tmp_path / "models")

    y_true = {
        "train": pd.Series([0, 1, 0, 1]),
        "test": pd.Series([0, 1]),
    }

    predictions = {
        "lr_train": pd.Series([0, 1, 0, 1]),
        "lr_test": pd.Series([0, 1]),
        "rf_train": pd.Series([0, 1, 0, 1]),
        "rf_test": pd.Series([0, 1]),
    }

    cl.classification_report_image(y_true, predictions)

    assert (cl.RESULTS_DIR / "classification_report.png").exists()
