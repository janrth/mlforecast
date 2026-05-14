import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean


class LastValueRegressor(BaseEstimator, RegressorMixin):
    """Tiny deterministic model so benchmark time is mostly feature plumbing."""

    def fit(self, X, y):
        self.value_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_, dtype=np.float64)


def make_data(n_series: int, n_periods: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(n_series), n_periods)
    t = np.tile(np.arange(n_periods), n_series)
    seasonal = 3.0 * np.sin(2 * np.pi * t / 28)
    level = rng.normal(10.0, 1.0, size=n_series).repeat(n_periods)
    noise = rng.normal(0.0, 0.5, size=n_series * n_periods)
    return pd.DataFrame(
        {
            "unique_id": ids,
            "ds": np.tile(pd.date_range("2020-01-01", periods=n_periods, freq="D"), n_series),
            "y": (level + seasonal + noise).astype(np.float32),
        }
    )


def make_fcst(global_: bool) -> MLForecast:
    return MLForecast(
        models=LastValueRegressor(),
        freq="D",
        lags=[1],
        lag_transforms={1: [RollingMean(window_size=28, global_=global_)]},
        num_threads=1,
    )


def timed(label: str, fn):
    gc.collect()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(json.dumps({"step": label, "elapsed_s": round(elapsed, 4)}))
    return result, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-series", type=int, default=1000)
    parser.add_argument("--n-periods", type=int, default=700)
    parser.add_argument("--horizon", type=int, default=56)
    args = parser.parse_args()

    df = make_data(args.n_series, args.n_periods)
    print(
        json.dumps(
            {
                "mlforecast_file": __import__("mlforecast").__file__,
                "rows": len(df),
                "n_series": args.n_series,
                "n_periods": args.n_periods,
                "horizon": args.horizon,
            }
        )
    )

    results = {}
    for global_ in [False, True]:
        label = "global" if global_ else "local"
        fcst = make_fcst(global_)

        _, preprocess_s = timed(
            f"{label}_preprocess",
            lambda: fcst.preprocess(df, return_X_y=True, as_numpy=True),
        )

        fcst = make_fcst(global_)
        _, fit_s = timed(f"{label}_fit", lambda: fcst.fit(df, as_numpy=True))
        _, predict_s = timed(f"{label}_predict_h{args.horizon}", lambda: fcst.predict(args.horizon))

        results[label] = {
            "preprocess_s": preprocess_s,
            "fit_s": fit_s,
            "predict_s": predict_s,
        }

    ratios = {
        "global_over_local_preprocess": results["global"]["preprocess_s"]
        / results["local"]["preprocess_s"],
        "global_over_local_fit": results["global"]["fit_s"] / results["local"]["fit_s"],
        "global_over_local_predict": results["global"]["predict_s"]
        / results["local"]["predict_s"],
    }
    print(json.dumps({k: round(v, 3) for k, v in ratios.items()}))


if __name__ == "__main__":
    main()
