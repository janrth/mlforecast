__all__ = ['lightgbm_space', 'xgboost_space', 'catboost_space', 'linear_regression_space', 'ridge_space', 'lasso_space',
           'elastic_net_space', 'random_forest_space', 'AutoModel', 'AutoLightGBM', 'AutoXGBoost', 'AutoCatboost',
           'AutoLinearRegression', 'AutoRidge', 'AutoLasso', 'AutoElasticNet', 'AutoRandomForest', 'AutoMLForecast']


import copy
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import optuna
import pandas as pd
import utilsforecast.processing as ufp
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import FunctionTransformer
from utilsforecast.compat import DataFrame
from utilsforecast.losses import smape
from utilsforecast.validation import validate_freq

from mlforecast.target_transforms import (
    Differences,
    GlobalSklearnTransformer,
    LocalStandardScaler,
)

from . import MLForecast
from .core import Freq, _get_model_name, _name_models
from .lag_transforms import ExponentiallyWeightedMean, RollingMean
from .optimization import _TrialToConfig, mlforecast_objective
from .utils import PredictionIntervals, _resolve_num_threads


def lightgbm_space(trial: optuna.Trial):
    return {
        "bagging_freq": 1,
        "learning_rate": 0.05,
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 20, 1000, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 4096, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "objective": trial.suggest_categorical("objective", ["l1", "l2"]),
    }


def xgboost_space(trial: optuna.Trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
    }


def catboost_space(trial: optuna.Trial):
    return {
        "silent": True,
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "depth": trial.suggest_int("depth", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
        "min_data_in_leaf": trial.suggest_float("min_data_in_leaf", 1, 100),
    }


def linear_regression_space(trial: optuna.Trial):
    return {"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])}


def ridge_space(trial: optuna.Trial):
    return {
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "alpha": trial.suggest_float("alpha", 0.001, 10.0),
    }


def lasso_space(trial: optuna.Trial):
    return {
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "alpha": trial.suggest_float("alpha", 0.001, 10.0),
    }


def elastic_net_space(trial: optuna.Trial):
    return {
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "alpha": trial.suggest_float("alpha", 0.001, 10.0),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def random_forest_space(trial: optuna.Trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        "criterion": trial.suggest_categorical(
            "criterion", ["squared_error", "absolute_error"]
        ),
    }


class AutoModel:
    """Structure to hold a model and its search space

    Args:
        model (BaseEstimator): scikit-learn compatible regressor
        config (callable): function that takes an optuna trial and produces a configuration
    """

    def __init__(
        self,
        model: BaseEstimator,
        config: _TrialToConfig,
    ):
        self.model = model
        self.config = config

    def __repr__(self):
        return f"AutoModel(model={_get_model_name(self.model)})"


class AutoLightGBM(AutoModel):
    def __init__(
        self,
        config: Optional[_TrialToConfig] = None,
    ):
        from mlforecast.compat import LGBMRegressor

        super().__init__(
            LGBMRegressor(),
            config if config is not None else lightgbm_space,
        )


class AutoXGBoost(AutoModel):
    def __init__(
        self,
        config: Optional[_TrialToConfig] = None,
    ):
        from mlforecast.compat import XGBRegressor

        super().__init__(
            XGBRegressor(),
            config if config is not None else xgboost_space,
        )


class AutoCatboost(AutoModel):
    def __init__(
        self,
        config: Optional[_TrialToConfig] = None,
    ):
        from mlforecast.compat import CatBoostRegressor

        super().__init__(
            CatBoostRegressor(),
            config if config is not None else catboost_space,
        )


class AutoLinearRegression(AutoModel):
    def __init__(
        self,
        config: Optional[_TrialToConfig] = None,
    ):
        from sklearn.linear_model import LinearRegression

        super().__init__(
            LinearRegression(),
            config if config is not None else linear_regression_space,
        )


class AutoRidge(AutoModel):
    def __init__(
        self,
        config: Optional[_TrialToConfig] = None,
    ):
        from sklearn.linear_model import Ridge

        super().__init__(
            Ridge(),
            config if config is not None else ridge_space,
        )


class AutoLasso(AutoModel):
    def __init__(
        self,
        config: Optional[_TrialToConfig] = None,
    ):
        from sklearn.linear_model import Lasso

        super().__init__(
            Lasso(),
            config if config is not None else lasso_space,
        )


class AutoElasticNet(AutoModel):
    def __init__(
        self,
        config: Optional[_TrialToConfig] = None,
    ):
        from sklearn.linear_model import ElasticNet

        super().__init__(
            ElasticNet(),
            config if config is not None else elastic_net_space,
        )


class AutoRandomForest(AutoModel):
    def __init__(
        self,
        config: Optional[_TrialToConfig] = None,
    ):
        from sklearn.ensemble import RandomForestRegressor

        super().__init__(
            RandomForestRegressor(),
            config if config is not None else random_forest_space,
        )


class AutoMLForecast:
    """Hyperparameter optimization helper

    Args:
        models (list or dict): Auto models to be optimized.
        freq (str or int): pandas' or polars' offset alias or integer denoting the frequency of the series.
        season_length (int, optional): Length of the seasonal period. This is used for producing the feature space.
            Only required if `init_config` is None. Defaults to None.
        init_config (callable, optional): Function that takes an optuna trial and produces a configuration passed to the MLForecast constructor.
            Defaults to None.
        fit_config (callable, optional): Function that takes an optuna trial and produces a configuration passed to the MLForecast fit method.
            Defaults to None.
        num_threads (int): Number of threads to use when computing the features. Use -1 to use all available CPU cores. Defaults to 1.
        reuse_cv_splits (bool): Creates splits for cv once and re-uses them for tuning instead of generating the splits in each tuning round.
            Default is set to False.
    """

    def __init__(
        self,
        models: Union[List[AutoModel], Dict[str, AutoModel]],
        freq: Freq,
        season_length: Optional[int] = None,
        init_config: Optional[_TrialToConfig] = None,
        fit_config: Optional[_TrialToConfig] = None,
        num_threads: int = 1,
        reuse_cv_splits: bool = False,
    ):
        self.freq = freq
        if season_length is None and init_config is None:
            raise ValueError(
                "`season_length` is required when `init_config` is not provided."
            )
        if init_config is not None and not callable(init_config):
            raise ValueError("`init_config` must be a function.")
        if season_length is not None and init_config is not None:
            warnings.warn("`season_length` is not used when `init_config` is provided.")
        self.init_config = init_config
        self.season_length = season_length
        if fit_config is not None:
            if not callable(fit_config):
                raise ValueError("`fit_config` must be a function.")
            self.fit_config = fit_config
        else:
            self.fit_config = lambda trial: {}  # noqa: ARG005
        num_threads = _resolve_num_threads(num_threads)
        self.num_threads = num_threads
        if isinstance(models, list):
            model_names = _name_models([_get_model_name(m) for m in models])
            models_with_names = dict(zip(model_names, models))
        else:
            models_with_names = models
        self.models = models_with_names
        self.reuse_cv_splits = reuse_cv_splits
        self.cv_metrics_: Dict[str, Dict[str, float]] = {}
        self.percentile_correction_: Dict[str, Dict[Any, Any]] = {}
        self.percentile_correction_levels_ = list(range(1, 16))
        self.percentile_correction_enabled_ = False

    def __repr__(self):
        return f"AutoMLForecast(models={self.models})"

    @staticmethod
    def _to_numpy(values: Any) -> np.ndarray:
        if hasattr(values, "to_numpy"):
            return values.to_numpy()
        return np.asarray(values)

    @staticmethod
    def _to_pandas(df: DataFrame) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            return df.copy()
        if hasattr(df, "to_pandas"):
            return df.to_pandas()
        return pd.DataFrame(df)

    def _find_systematic_bias_ids(
        self,
        cv_df: pd.DataFrame,
        id_col: str,
        target_col: str,
        model_col: str,
    ) -> pd.DataFrame:
        errors = cv_df[model_col] - cv_df[target_col]
        bias_by_cutoff = (
            cv_df.assign(_bias=errors)
            .groupby([id_col, "cutoff"], observed=True)["_bias"]
            .mean()
            .reset_index()
        )
        id_signs = (
            bias_by_cutoff.groupby(id_col, observed=True)["_bias"]
            .agg(["min", "max"])
            .reset_index()
        )
        underforecast_ids = id_signs.loc[id_signs["max"] < 0, [id_col]].assign(
            direction="hi"
        )
        overforecast_ids = id_signs.loc[id_signs["min"] > 0, [id_col]].assign(
            direction="lo"
        )
        return pd.concat([underforecast_ids, overforecast_ids], ignore_index=True)

    def _select_percentiles_to_reduce_bias(
        self,
        cv_df: pd.DataFrame,
        systematic_bias_ids: pd.DataFrame,
        id_col: str,
        target_col: str,
        model_col: str,
    ) -> Dict[Any, Dict[str, Any]]:
        if systematic_bias_ids.empty:
            return {}
        cv_df = cv_df.merge(systematic_bias_ids, on=id_col, how="inner")
        hi_rows = cv_df["direction"].eq("hi")
        lo_rows = cv_df["direction"].eq("lo")
        best_by_id: Dict[Any, Dict[str, Any]] = {}

        for percentile in self.percentile_correction_levels_:
            hi_col = f"{model_col}-hi-{percentile}"
            lo_col = f"{model_col}-lo-{percentile}"
            if hi_col not in cv_df.columns or lo_col not in cv_df.columns:
                continue
            if hi_rows.any():
                hi_bias = (
                    (cv_df.loc[hi_rows, hi_col] - cv_df.loc[hi_rows, target_col])
                    .groupby(cv_df.loc[hi_rows, id_col], observed=True)
                    .mean()
                )
                for uid, bias in hi_bias.items():
                    abs_bias = abs(float(bias))
                    current = best_by_id.get(uid)
                    if current is None or abs_bias < current["abs_bias"]:
                        best_by_id[uid] = {
                            "direction": "hi",
                            "percentile": percentile,
                            "abs_bias": abs_bias,
                        }
            if lo_rows.any():
                lo_bias = (
                    (cv_df.loc[lo_rows, lo_col] - cv_df.loc[lo_rows, target_col])
                    .groupby(cv_df.loc[lo_rows, id_col], observed=True)
                    .mean()
                )
                for uid, bias in lo_bias.items():
                    abs_bias = abs(float(bias))
                    current = best_by_id.get(uid)
                    if current is None or abs_bias < current["abs_bias"]:
                        best_by_id[uid] = {
                            "direction": "lo",
                            "percentile": percentile,
                            "abs_bias": abs_bias,
                        }
        return best_by_id

    def _compute_percentile_correction(
        self,
        df: DataFrame,
        model_name: str,
        model: BaseEstimator,
        mlf_init_params: Dict[str, Any],
        mlf_fit_params: Dict[str, Any],
        n_windows: int,
        h: int,
        step_size: Optional[int],
        input_size: Optional[int],
        refit: Union[bool, int],
        prediction_intervals: PredictionIntervals,
        id_col: str,
        time_col: str,
        target_col: str,
        weight_col: Optional[str],
    ) -> Dict[str, Any]:
        base_cv_model = MLForecast(
            models={"model": model},
            freq=self.freq,
            **mlf_init_params,
        )
        base_cv = base_cv_model.cross_validation(
            df=df,
            n_windows=n_windows,
            h=h,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            step_size=step_size,
            input_size=input_size,
            refit=refit,
            weight_col=weight_col,
            **mlf_fit_params,
        )
        base_cv_pd = self._to_pandas(base_cv)
        systematic_bias_ids = self._find_systematic_bias_ids(
            cv_df=base_cv_pd,
            id_col=id_col,
            target_col=target_col,
            model_col="model",
        )
        if systematic_bias_ids.empty:
            return {"id_to_col": {}, "levels": []}

        ids = systematic_bias_ids[id_col].tolist()
        id_mask = ufp.is_in(df[id_col], ids)
        filtered_df = ufp.filter_with_mask(df, id_mask)

        intervals_cv_model = MLForecast(
            models={"model": clone(model)},
            freq=self.freq,
            **mlf_init_params,
        )
        intervals_cv = intervals_cv_model.cross_validation(
            df=filtered_df,
            n_windows=n_windows,
            h=h,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            step_size=step_size,
            input_size=input_size,
            refit=True,
            prediction_intervals=prediction_intervals,
            level=self.percentile_correction_levels_,
            weight_col=weight_col,
            **mlf_fit_params,
        )
        intervals_cv_pd = self._to_pandas(intervals_cv)
        best_by_id = self._select_percentiles_to_reduce_bias(
            cv_df=intervals_cv_pd,
            systematic_bias_ids=systematic_bias_ids,
            id_col=id_col,
            target_col=target_col,
            model_col="model",
        )
        if not best_by_id:
            return {"id_to_col": {}, "levels": []}
        id_to_col = {
            uid: f"{model_name}-{info['direction']}-{info['percentile']}"
            for uid, info in best_by_id.items()
        }
        levels = sorted({info["percentile"] for info in best_by_id.values()})
        return {"id_to_col": id_to_col, "levels": levels}

    def _apply_percentile_correction(
        self,
        preds: DataFrame,
        id_col: str,
        model_col: str,
        id_to_col: Dict[Any, str],
    ) -> DataFrame:
        if not id_to_col:
            return preds
        ids = self._to_numpy(preds[id_col])
        corrected_preds = self._to_numpy(preds[model_col]).copy()
        selected_cols = np.array([id_to_col.get(uid, "") for uid in ids], dtype=object)
        for col in np.unique(selected_cols):
            if not col or col not in preds.columns:
                continue
            mask = selected_cols == col
            percentile_values = self._to_numpy(preds[col])
            corrected_preds[mask] = percentile_values[mask]
        return ufp.assign_columns(preds, model_col, corrected_preds)

    def _seasonality_based_config(
        self,
        h: int,
        min_samples: int,
        min_value: float,
    ) -> _TrialToConfig:
        assert self.season_length is not None
        # target transforms
        candidate_targ_tfms: List[Any] = [
            None,
            [LocalStandardScaler()],
            [Differences([1]), LocalStandardScaler()],
        ]
        log1p_tfm = GlobalSklearnTransformer(
            FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        )
        if min_value >= 0:
            candidate_targ_tfms.extend(
                [
                    [log1p_tfm, LocalStandardScaler()],
                    [log1p_tfm, Differences([1]), LocalStandardScaler()],
                ]
            )
        # we leave two seasonal periods for the features and model
        if self.season_length > 1 and min_samples > 3 * self.season_length + 1:
            candidate_targ_tfms.append(
                [Differences([1, self.season_length]), LocalStandardScaler()]
            )
            if min_value >= 0:
                candidate_targ_tfms.append(
                    [
                        log1p_tfm,
                        Differences([1, self.season_length]),
                        LocalStandardScaler(),
                    ],
                )

        # lags
        candidate_lags = [None, [self.season_length]]
        seasonality2extra_candidate_lags = {
            7: [
                [7, 14],
                [7, 28],
            ],
            12: [range(1, 13)],
            24: [
                range(1, 25),
                range(24, 24 * 7 + 1, 24),
            ],
            52: [
                range(4, 53, 4),
            ],
        }
        if self.season_length in seasonality2extra_candidate_lags:
            candidate_lags.extend(
                seasonality2extra_candidate_lags[self.season_length]  # type: ignore
            )
        if h >= 2 * self.season_length:
            candidate_lags.extend(
                [
                    range(self.season_length, h + 1, self.season_length),  # type: ignore
                    [h],
                    [self.season_length, h],
                ]
            )

        # lag transforms
        candidate_lag_tfms = [None, {1: [ExponentiallyWeightedMean(0.9)]}]
        if self.season_length > 1:
            candidate_lag_tfms.append(
                {
                    1: [ExponentiallyWeightedMean(0.9)],
                    self.season_length: [
                        RollingMean(window_size=self.season_length, min_samples=1),
                    ],
                }
            )
        if self.season_length != h:
            candidate_lag_tfms.append(
                {
                    1: [ExponentiallyWeightedMean(0.9)],
                    self.season_length: [
                        RollingMean(window_size=self.season_length, min_samples=1),
                    ],
                    h: [
                        RollingMean(window_size=self.season_length, min_samples=1),
                    ],
                }
            )

        # date features
        seasonality2date_features = {
            1: ["year"],
            4: ["quarter", "year"],
            7: ["weekday", "month", "year"],
            12: ["month", "year"],
            24: ["hour", "weekday", "month", "year"],
            52: ["week", "year"],
            60: ["weekday", "hour", "second"],
        }
        candidate_date_features = seasonality2date_features.get(self.season_length, [])
        if isinstance(self.freq, int):
            candidate_date_features = []

        def config(trial):
            # target transforms
            targ_tfms_idx = trial.suggest_categorical(
                "target_transforms_idx", range(len(candidate_targ_tfms))
            )
            target_transforms = candidate_targ_tfms[targ_tfms_idx]

            # lags
            lags_idx = trial.suggest_categorical("lags_idx", range(len(candidate_lags)))
            lags = candidate_lags[lags_idx]

            # lag transforms
            if candidate_lag_tfms:
                lag_tfms_idx = trial.suggest_categorical(
                    "lag_transforms_idx", range(len(candidate_lag_tfms))
                )
                lag_transforms = candidate_lag_tfms[lag_tfms_idx]
            else:
                lag_transforms = None

            # date features
            if candidate_date_features:
                use_date_features = trial.suggest_int("use_date_features", 0, 1)
                if use_date_features:
                    date_features = candidate_date_features
                else:
                    date_features = None
            else:
                date_features = None

            return {
                "lags": lags,
                "target_transforms": target_transforms,
                "lag_transforms": lag_transforms,
                "date_features": date_features,
            }

        return config

    def fit(
        self,
        df: DataFrame,
        n_windows: int,
        h: int,
        num_samples: int,
        step_size: Optional[int] = None,
        input_size: Optional[int] = None,
        refit: Union[bool, int] = False,
        loss: Optional[Callable[[DataFrame, DataFrame], float]] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        study_kwargs: Optional[Dict[str, Any]] = None,
        optimize_kwargs: Optional[Dict[str, Any]] = None,
        fitted: bool = False,
        prediction_intervals: Optional[PredictionIntervals] = None,
        weight_col: Optional[str] = None,
        percentile_correction: bool = False,
    ) -> "AutoMLForecast":
        """Carry out the optimization process.
        Each model is optimized independently and the best one is trained on all data

        Args:
            df (pandas or polars DataFrame): Series data in long format.
            n_windows (int): Number of windows to evaluate.
            h (int): Forecast horizon.
            num_samples (int): Number of trials to run
            step_size (int, optional): Step size between each cross validation window. If None it will be equal to `h`.
                Defaults to None.
            input_size (int, optional): Maximum training samples per serie in each window. If None, will use an expanding window.
                Defaults to None.
            refit (bool or int): Retrain model for each cross validation window.
                If False, the models are trained at the beginning and then used to predict each window.
                If positive int, the models are retrained every `refit` windows. Defaults to False.
            loss (callable, optional): Function that takes the validation and train dataframes and produces a float.
                If `None` will use the average SMAPE across series. Defaults to None.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            study_kwargs (dict, optional): Keyword arguments to be passed to the optuna.Study constructor.
                Defaults to None.
            optimize_kwargs (dict, optional): Keyword arguments to be passed to the optuna.Study.optimize method.
                Defaults to None.
            fitted (bool): Whether to compute the fitted values when retraining the best model. Defaults to False.
            prediction_intervals: Configuration to calibrate prediction intervals when retraining the best model.
            weight_col (str, optional): Column that contains sample weights. Defaults to None.
            percentile_correction (bool): Whether to correct model predictions using conformal percentiles (1..15)
                for ids with systematic CV bias. Defaults to False.

        Returns:
            (AutoMLForecast): object with best models and optimization results
        """
        validate_freq(df[time_col], self.freq)
        if self.init_config is not None:
            init_config = self.init_config
        else:
            min_size = ufp.counts_by_id(df, id_col)["counts"].min()
            min_train_size = min_size - n_windows * h
            init_config = self._seasonality_based_config(
                h=h,
                min_samples=min_train_size,
                min_value=df[target_col].min(),
            )
        if loss is None:
            def loss(df, train_df):  # noqa: ARG001
                return smape(
                    df,
                    models=["model"],
                    id_col=id_col,
                    target_col=target_col,
                )["model"].mean()
        if study_kwargs is None:
            study_kwargs = {}
        if "sampler" not in study_kwargs:
            # for reproducibility
            study_kwargs["sampler"] = optuna.samplers.TPESampler(seed=0)
        if optimize_kwargs is None:
            optimize_kwargs = {}
        self.results_ = {}
        self.models_ = {}
        self.cv_metrics_ = {}
        self.percentile_correction_ = {}
        self.percentile_correction_enabled_ = percentile_correction
        if percentile_correction and prediction_intervals is None:
            raise ValueError(
                "`prediction_intervals` must be provided when `percentile_correction=True`."
            )
        cv_splits = None
        if self.reuse_cv_splits:
            cv_splits = list(
                ufp.backtest_splits(
                    df,
                    n_windows=n_windows,
                    h=h,
                    id_col=id_col,
                    time_col=time_col,
                    freq=self.freq,
                    step_size=step_size,
                    input_size=input_size,
                )
            )
        for name, auto_model in self.models.items():
            def config_fn(trial: optuna.Trial) -> Dict[str, Any]:
                return {
                    "model_params": auto_model.config(trial),
                    "mlf_init_params": {
                        **init_config(trial),
                        "num_threads": self.num_threads,
                    },
                    "mlf_fit_params": self.fit_config(trial),
                }
            objective = mlforecast_objective(
                df=df,
                config_fn=config_fn,
                loss=loss,
                model=auto_model.model,
                freq=self.freq,
                n_windows=n_windows,
                h=h,
                step_size=step_size,
                input_size=input_size,
                refit=refit,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                weight_col=weight_col,
                cv_splits=cv_splits,
            )
            study = optuna.create_study(direction="minimize", **study_kwargs)
            study.optimize(objective, n_trials=num_samples, **optimize_kwargs)
            self.results_[name] = study
            best_trial = study.best_trial
            best_config = best_trial.user_attrs["config"]
            self.cv_metrics_[name] = {
                "rmse": float(best_trial.user_attrs.get("cv_rmse", np.nan)),
                "bias": float(best_trial.user_attrs.get("cv_bias", np.nan)),
            }
            best_fit_params = copy.deepcopy(best_config["mlf_fit_params"])
            for arg in (
                "fitted",
                "prediction_intervals",
                "id_col",
                "time_col",
                "target_col",
                "weight_col",
            ):
                best_fit_params.pop(arg, None)
            best_model = clone(auto_model.model)
            best_model.set_params(**best_config["model_params"])
            self.models_[name] = MLForecast(
                models={name: best_model},
                freq=self.freq,
                **best_config["mlf_init_params"],
            )
            
            self.models_[name].fit(
                df,
                fitted=fitted,
                prediction_intervals=prediction_intervals,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                weight_col=weight_col,
                **best_fit_params,
            )
            if percentile_correction:
                assert prediction_intervals is not None
                correction_model = clone(auto_model.model)
                correction_model.set_params(**best_config["model_params"])
                self.percentile_correction_[name] = self._compute_percentile_correction(
                    df=df,
                    model_name=name,
                    model=correction_model,
                    mlf_init_params=best_config["mlf_init_params"],
                    mlf_fit_params=best_fit_params,
                    n_windows=n_windows,
                    h=h,
                    step_size=step_size,
                    input_size=input_size,
                    refit=refit,
                    prediction_intervals=prediction_intervals,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    weight_col=weight_col,
                )
            else:
                self.percentile_correction_[name] = {"id_to_col": {}, "levels": []}
        return self

    def predict(
        self,
        h: int,
        X_df: Optional[DataFrame] = None,
        level: Optional[List[Union[int, float]]] = None,
    ) -> DataFrame:
        """ "Compute forecasts

        Args:
            h (int): Number of periods to predict.
            X_df (pandas or polars DataFrame, optional): Dataframe with the future exogenous features. Should have the id column and the time column.
                Defaults to None.
            level (list of ints or floats, optional): Confidence levels between 0 and 100 for prediction intervals.
                Defaults to None.

        Returns:
            (pandas or polars DataFrame): Predictions for each serie and timestep, with one column per model.
        """
        all_preds = None
        for name, model in self.models_.items():
            correction = self.percentile_correction_.get(name, {})
            correction_levels = correction.get("levels", [])
            predict_levels = level
            if self.percentile_correction_enabled_ and correction_levels:
                if level is None:
                    predict_levels = correction_levels
                else:
                    predict_levels = sorted(set(level) | set(correction_levels))
            preds = model.predict(h=h, X_df=X_df, level=predict_levels)
            if self.percentile_correction_enabled_:
                preds = self._apply_percentile_correction(
                    preds=preds,
                    id_col=model.ts.id_col,
                    model_col=name,
                    id_to_col=correction.get("id_to_col", {}),
                )
                extra_levels = set(correction_levels)
                if level is not None:
                    extra_levels -= set(level)
                drop_cols = []
                for lvl in sorted(extra_levels):
                    drop_cols.extend([f"{name}-lo-{lvl}", f"{name}-hi-{lvl}"])
                drop_cols = [c for c in drop_cols if c in preds.columns]
                if drop_cols:
                    preds = ufp.drop_columns(preds, drop_cols)
            if all_preds is None:
                all_preds = preds
            else:
                model_cols = [c for c in preds.columns if c not in all_preds.columns]
                all_preds = ufp.horizontal_concat([all_preds, preds[model_cols]])
        return all_preds

    def save(self, path: Union[str, Path]) -> None:
        """Save AutoMLForecast objects

        Args:
            path (str or pathlib.Path): Directory where artifacts will be stored.
        """
        for name, model in self.models_.items():
            model.save(f"{path}/{name}")

    def forecast_fitted_values(
        self,
        level: Optional[List[Union[int, float]]] = None,
    ) -> DataFrame:
        """Access in-sample predictions.

        Args:
            level (list of ints or floats, optional): Confidence levels between 0 and 100 for prediction intervals.
                Defaults to None.

        Returns:
            (pandas or polars DataFrame): Dataframe with predictions for the training set
        """
        fitted_vals = None
        for name, model in self.models_.items():
            model_fitted = model.forecast_fitted_values(level=level)
            if fitted_vals is None:
                fitted_vals = model_fitted
            else:
                fitted_vals = ufp.join(
                    fitted_vals,
                    ufp.drop_columns(model_fitted, model.ts.target_col),
                    on=[model.ts.id_col, model.ts.time_col],
                    how="inner",
                )
        return fitted_vals
