import os
import functools
import inspect
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Limit TF threads per worker process. setdefault leaves user-set values intact.
# With spawn workers, DeepLearning_Models is re-imported in each child so these
# take effect before TF initialises its thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import warnings
import tensorflow as tf
from tensorflow.keras.layers import (  # pyright: ignore[reportMissingModuleSource]
    Input,
    Dense,
    BatchNormalization,
    Dropout,
    Add,
    Activation,
    Layer,
    Embedding,
)
from tensorflow.keras.layers import LayerNormalization, Concatenate, MultiHeadAttention  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import PReLU  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import EarlyStopping, Callback  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.models import Model  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.regularizers import L2  # pyright: ignore[reportMissingModuleSource]
from typing import List, Optional
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from scipy.optimize import minimize_scalar, minimize
from keras.src import ops
from itertools import combinations
import pandas as pd
import optuna


# =============================================================================
# Input Preparation
# =============================================================================


class Prep_Input:
    def __init__(
        self,
        Numerical=None,
        Binary_Categorical=None,
        Ordinal_Categorical=None,
        Nominal_Categorical=None,
        Ord_levels=None,
        Selected_Features=None,
    ):
        Numerical = Numerical or []
        Binary_Categorical = Binary_Categorical or []
        Ordinal_Categorical = Ordinal_Categorical or []
        Nominal_Categorical = Nominal_Categorical or []
        Ord_levels = Ord_levels or {}

        selected_set = set(Selected_Features) if Selected_Features is not None else None

        self.Num_Features = (
            Numerical
            if selected_set is None
            else [f for f in Numerical if f in selected_set]
        )
        self.Ord_Cat_Features = (
            Ordinal_Categorical
            if selected_set is None
            else [f for f in Ordinal_Categorical if f in selected_set]
        )
        self.Nom_Cat_Features = (
            Nominal_Categorical
            if selected_set is None
            else [f for f in Nominal_Categorical if f in selected_set]
        )
        self.Binary_Features = (
            Binary_Categorical
            if selected_set is None
            else [f for f in Binary_Categorical if f in selected_set]
        )
        self.Ord_levels = Ord_levels
        self.All_Features = [
            set(self.Num_Features),
            set(self.Ord_Cat_Features),
            set(self.Nom_Cat_Features),
            set(self.Binary_Features),
        ]

        if len(set().union(*self.All_Features)) != sum(
            len(s) for s in self.All_Features
        ):
            shared_features = set()
            for s1, s2 in combinations(self.All_Features, 2):
                shared_features.update(s1 & s2)
            raise AssertionError(
                f"{shared_features} are assigned more than one category, but only one assignment is allowed: Numerical, Binary, Ordinal, or Nominal."
            )

    def validate_input(self, X):
        assert set().union(*self.All_Features).issubset(set(X.columns)), (
            f"{set().union(*self.All_Features) - set(X.columns)} are not in input X"
        )

    def MLP_Input(self, X):
        self.validate_input(X)
        ohe_cols = [
            f"{col}_{c}"
            for col in self.Nom_Cat_Features
            for c in X[col].dropna().unique()
        ]
        if set(ohe_cols).issubset(set(X.columns)):
            selected_features = (
                self.Num_Features
                + self.Binary_Features
                + self.Ord_Cat_Features
                + ohe_cols
            )
            X_clean = X[selected_features].copy()
        else:
            selected_features = (
                self.Num_Features
                + self.Binary_Features
                + self.Ord_Cat_Features
                + self.Nom_Cat_Features
            )
            X_clean = X[selected_features].copy()
            for feature in self.Nom_Cat_Features:
                one_hot_encoded = pd.get_dummies(X_clean[feature], prefix=feature)
                X_clean = pd.concat([X_clean, one_hot_encoded], axis=1).drop(
                    columns=[feature]
                )
        for feature in self.Ord_Cat_Features:
            N = (
                self.Ord_levels[feature]
                if feature in self.Ord_levels
                else len(X_clean[feature].unique())
            )
            assert N > 0, f'Ord_levels for "{feature}" must be positive, got {N}.'
            X_clean[feature] = (X_clean[feature] / N).astype(float)
        return X_clean

    def CatEmb_Input(self, X, test=True):
        self.validate_input(X)
        cat_features = self.Nom_Cat_Features + self.Binary_Features
        assert len(cat_features) > 0, (
            "CatEmb_Input requires at least one nominal categorical or binary feature"
        )

        num_features = self.Num_Features + self.Ord_Cat_Features
        X_cont = X[num_features].copy() if len(num_features) > 0 else None

        for feature in self.Ord_Cat_Features:
            N = (
                self.Ord_levels[feature]
                if feature in self.Ord_levels
                else len(X_cont[feature].unique())
            )
            assert N > 0, f'Ord_levels for "{feature}" must be positive, got {N}.'
            X_cont[feature] = (X_cont[feature] / N).astype(float)

        X_cat = X[cat_features].copy()

        if test:
            X_input = [X_cat] if X_cont is None else [X_cont, X_cat]
            return X_input
        else:
            return [X_cont, X_cat]

    def CatEmb_Input_Params(self, X):
        X_cont, X_cat = self.CatEmb_Input(X, test=False)
        input_shape = (
            X_cat.shape[1] if X_cont is None else X_cont.shape[1] + X_cat.shape[1]
        )
        cat_cardinalities = [
            int(X_cat[col].nunique(dropna=True)) for col in X_cat.columns
        ]
        X_input = [X_cat] if X_cont is None else [X_cont, X_cat]
        return input_shape, cat_cardinalities, X_input

    def FTT_Input(self, X, test=True):
        self.validate_input(X)
        cat_features = self.Nom_Cat_Features
        num_features = self.Num_Features + self.Ord_Cat_Features + self.Binary_Features

        X_cont = X[num_features].copy() if len(num_features) > 0 else None
        X_cat = X[cat_features].copy() if len(cat_features) > 0 else None

        for feature in self.Ord_Cat_Features:
            N = (
                self.Ord_levels[feature]
                if feature in self.Ord_levels
                else len(X_cont[feature].unique())
            )
            assert N > 0, f'Ord_levels for "{feature}" must be positive, got {N}.'
            X_cont[feature] = (X_cont[feature] / N).astype(float)

        return (
            [x for x in [X_cont, X_cat] if x is not None] if test else [X_cont, X_cat]
        )

    def FTT_Input_Params(self, X):
        X_cont, X_cat = self.FTT_Input(X, test=False)
        n_cont_features = 0 if X_cont is None else X_cont.shape[1]
        cat_cardinalities = (
            []
            if X_cat is None
            else [int(X_cat[col].nunique(dropna=True)) for col in X_cat.columns]
        )
        assert n_cont_features + len(cat_cardinalities) > 0, (
            "Need to have at least one feature"
        )
        X_input = [x for x in [X_cont, X_cat] if x is not None]
        return n_cont_features, cat_cardinalities, X_input

    def Scale_Num_Features(self):
        ct = ColumnTransformer(
            transformers=[("num_preprocess", StandardScaler(), self.Num_Features)],
            remainder="passthrough",
        )
        return ct

    def CatEmb_Preprocessor(self):
        if len(self.Num_Features) > 0:
            ct = ColumnTransformer(
                transformers=[
                    ("num_preprocess", StandardScaler(), self.Num_Features),
                    (
                        "cat_preprocess",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                        self.Nom_Cat_Features,
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            )
        else:
            ct = ColumnTransformer(
                transformers=[
                    (
                        "cat_preprocess",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                        self.Nom_Cat_Features,
                    )
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            )
        return ct

    def FTT_Preprocessor(self):
        transformers = []
        if len(self.Num_Features) > 0:
            transformers.append(("num_preprocess", StandardScaler(), self.Num_Features))
        if len(self.Nom_Cat_Features) > 0:
            transformers.append(
                (
                    "cat_preprocess",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    self.Nom_Cat_Features,
                )
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )


# =============================================================================
# Utilities
# =============================================================================


def SavePredictions(y_predicted, X_test, filename):
    assert len(y_predicted) == len(X_test), (
        f"y_predicted length ({len(y_predicted)}) does not match X_test length ({len(X_test)})"
    )
    Test_Predictions = pd.DataFrame(index=X_test.index)
    Test_Predictions["Transported"] = y_predicted.astype(bool)
    Test_Predictions.to_csv(filename, index=True)


# =============================================================================
# Parallel CV helpers
# =============================================================================


def _capture_init_kwargs(init_fn):
    """Wrap a model __init__ so every instance stores its constructor kwargs."""
    sig = inspect.signature(init_fn)

    @functools.wraps(init_fn)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        init_fn(self, *args, **kwargs)
        self._init_kwargs = {k: v for k, v in bound.arguments.items() if k != "self"}

    return wrapper


def _parallel_worker_init():
    """Called once per worker process. Best-effort TF thread limit."""
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass


def _parallel_fold_worker(args):
    (
        fold,
        train_idx,
        val_idx,
        model_class,
        model_kwargs,
        X,
        y,
        X_test,
        fit_params,
        calibration_methods,
        initial_weights,
    ) = args

    # Rebuild model in this process (Keras models aren't picklable) and
    # restore the same starting weights used across all folds.
    model = model_class(**model_kwargs)
    model.model.set_weights(initial_weights)

    eval_metric = fit_params["monitor"].removeprefix("val_")

    if isinstance(X, list):
        X_train = [
            arr.iloc[train_idx] if hasattr(arr, "iloc") else arr[train_idx] for arr in X
        ]
        X_val = [
            arr.iloc[val_idx] if hasattr(arr, "iloc") else arr[val_idx] for arr in X
        ]
    else:
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]

    y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
    y_val = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
    y_val_np = np.asarray(y_val)

    history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_params)

    val_metric = model.evaluate(X_val, y_val_np, metric=eval_metric)
    train_metric = model.evaluate(X_train, np.asarray(y_train), metric=eval_metric)

    val_preds = model.predict(X_val, method="predict_proba")
    test_preds = model.predict(X_test, method="predict_proba")

    calibrated = {
        m: model.calibrate_probs(val_preds, y_val_np, test_preds, m)
        for m in calibration_methods
    }

    # history.history is a plain dict of lists — picklable.
    # The Keras History object itself holds a model reference and is not.
    return fold, calibrated, val_metric, train_metric, history.history


# =============================================================================
# Custom Early Stopping Callback
# =============================================================================


class Custom_EarlyStopping(Callback):
    def __init__(
        self,
        min_delta=0,
        patience=3,
        verbose=0,
        mode="auto",
        monitor="accuracy",
        restore_best_weights=False,
        start_from_epoch=0,
    ):
        super().__init__()
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_epoch = start_from_epoch

        self.metric = monitor.removeprefix("val_")
        self.val_metric = "val_" + self.metric

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"Custom_EarlyStopping mode {mode} is unknown, fallback to auto mode",
                stacklevel=2,
            )
            mode = "auto"

        if mode == "auto":
            if self.metric in ["accuracy", "precision", "recall", "auc"]:
                self.mode = "max"
            elif self.metric in ["loss", "mean_squared_error", "mean_absolute_error"]:
                self.mode = "min"
            else:
                raise ValueError(
                    f"Custom_EarlyStopping could not infer mode for metric {self.metric}. Set mode = max or mode = min"
                )
        else:
            self.mode = mode

        self.monitor_op = None

    def _set_monitor_ops(self):
        if self.mode == "max":
            self.monitor_op = ops.greater
            self.val_best = -np.inf
            self.train_at_best = -np.inf
            self._min_delta = self.min_delta
        elif self.mode == "min":
            self.monitor_op = ops.less
            self.val_best = np.inf
            self.train_at_best = np.inf
            self._min_delta = -self.min_delta
        else:
            raise ValueError(
                f"mode should be set to either 'max' or 'min', got {self.mode = }"
            )

    def get_monitor_values(self, logs):
        if not logs:
            raise ValueError("logs cannot be empty")

        train_value = logs.get(self.metric)
        val_value = logs.get(self.val_metric)

        if (train_value is None) or (val_value is None):
            raise ValueError(
                f"{self.metric} or/and {self.val_metric} not available. Available metrics are: {','.join(list(logs.keys()))}"
            )

        train_value = float(train_value)
        val_value = float(val_value)

        return val_value, train_value

    def _is_improvement(self, val_value, train_value, val_ref, train_ref):
        return self.monitor_op(val_value, val_ref + self._min_delta) or (
            (val_value == val_ref + self._min_delta)
            and (self.monitor_op(train_value, train_ref + self._min_delta))
        )

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op is None:
            self._set_monitor_ops()

        if epoch < self.start_from_epoch:
            return

        val_value, train_value = self.get_monitor_values(logs)

        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        if self._is_improvement(
            val_value, train_value, self.val_best, self.train_at_best
        ):
            self.val_best = val_value
            self.train_at_best = train_value
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            if self.verbose:
                print(
                    f"New improvement at epoch = {epoch + 1}, val_acc = {self.val_best:.4f}, train_acc = {self.train_at_best:.4f}"
                )
            return
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose:
                print(
                    f"restoring best weights from epoch {self.best_epoch} with val_acc = {self.val_best:.6f}, train_acc = {self.train_at_best:.6f}"
                )
            self.model.set_weights(self.best_weights)


# =============================================================================
# Base Classifier
# =============================================================================


class BaseClassifier:
    def __init__(self, output_shape, preprocessor=None):
        self.output_shape = output_shape
        self.preprocessor = preprocessor
        self._fitted_preprocessor = None

    def _make_mixup_dataset(self, X, y, alpha, batch_size):
        """Create a tf.data.Dataset that applies mixup augmentation per batch.

        Continuous inputs are linearly interpolated. Categorical (integer-dtype)
        inputs are selected from the dominant sample (the one with higher lambda).
        Labels are always interpolated.
        """
        is_multi = isinstance(X, (list, tuple))
        y_arr = np.asarray(y, dtype=np.float32)
        n = len(y_arr)

        # Use model input dtypes to identify categorical inputs
        if is_multi:
            model_dtypes = [inp.dtype for inp in self.model.inputs]
            is_cat = [dt in (tf.int32, tf.int64) for dt in model_dtypes]
            x_tensors = tuple(
                tf.constant(np.asarray(x, dtype=np.int32 if cat else np.float32))
                for x, cat in zip(X, is_cat)
            )
            ds = tf.data.Dataset.from_tensor_slices((x_tensors, y_arr))
        else:
            is_cat = [False]
            ds = tf.data.Dataset.from_tensor_slices(
                (tf.constant(np.asarray(X, dtype=np.float32)), y_arr)
            )

        ds = ds.shuffle(n).batch(batch_size)
        alpha_t = tf.constant(alpha, dtype=tf.float32)

        def apply_mixup(inputs, labels):
            bs = tf.shape(labels)[0]
            # Beta(alpha, alpha) via the gamma ratio trick
            g1 = tf.random.gamma([bs], alpha_t)
            g2 = tf.random.gamma([bs], alpha_t)
            lam = g1 / (g1 + g2)
            indices = tf.random.shuffle(tf.range(bs))

            if is_multi:
                mixed = []
                for i, inp in enumerate(inputs):
                    if is_cat[i]:
                        cond = tf.reshape(lam > 0.5, [-1, 1])
                        mixed.append(
                            tf.where(cond, inp, tf.gather(inp, indices))
                        )
                    else:
                        lam_r = tf.reshape(lam, [-1, 1])
                        mixed.append(
                            lam_r * inp + (1 - lam_r) * tf.gather(inp, indices)
                        )
                mixed = tuple(mixed)
            else:
                lam_r = tf.reshape(lam, [-1, 1])
                mixed = lam_r * inputs + (1 - lam_r) * tf.gather(inputs, indices)

            mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
            return mixed, mixed_labels

        ds = ds.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def fit(
        self,
        X,
        y,
        *,
        X_val=None,
        y_val=None,
        epochs: int = 200,
        batch_size: int = 256,
        patience: int = 20,
        verbose: bool = False,
        val_split: float = 0.15,
        monitor: str = "val_accuracy",
        custom_callback: bool = False,
        callback_verbose: bool = True,
        mixup_alpha: float = 0.0,
    ):

        self.validate_data(X, y, X_val, y_val)

        if custom_callback:
            early_stopping = Custom_EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=callback_verbose,
            )
        else:
            early_stopping = EarlyStopping(
                monitor=monitor, patience=patience, restore_best_weights=True
            )

        if X_val is None:
            assert self.preprocessor is None, (
                "If preprocessor is given, X_val needs to be given too"
            )
            if val_split is None:
                history = self.model.fit(
                    X, y, epochs=epochs, batch_size=batch_size, verbose=verbose
                )
            else:
                history = self.model.fit(
                    X,
                    y,
                    epochs=epochs,
                    validation_split=val_split,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=verbose,
                )
        else:
            X_tr, X_vl = self.transform_data(X, X_val)
            if mixup_alpha > 0:
                train_ds = self._make_mixup_dataset(
                    X_tr, y, mixup_alpha, batch_size
                )
                history = self.model.fit(
                    train_ds,
                    validation_data=(X_vl, y_val),
                    epochs=epochs,
                    callbacks=[early_stopping],
                    verbose=verbose,
                )
            else:
                history = self.model.fit(
                    X_tr,
                    y,
                    validation_data=(X_vl, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=verbose,
                )

        return history

    def transform_data(self, X0, X0_val=None):
        if isinstance(X0, list):
            X = X0 if self.preprocessor is None else pd.concat(X0, axis=1)
            X_val = (
                X0_val
                if (self.preprocessor is None or X0_val is None)
                else pd.concat(X0_val, axis=1)
            )
        else:
            X = X0.copy() if hasattr(X0, "copy") else X0
            X_val = X0_val.copy() if hasattr(X0_val, "copy") else X0_val

        if self.preprocessor is None:
            return X if X_val is None else (X, X_val)

        ct = clone(self.preprocessor)
        try:
            ct.set_output(transform="pandas")
        except Exception:
            pass
        X = ct.fit_transform(X)

        X_val = None if X_val is None else ct.transform(X_val)
        self._fitted_preprocessor = ct

        if isinstance(X0, list):
            if len(X0) == 1:
                return [X] if X_val is None else ([X], [X_val])
            elif len(X0) == 2:
                X_cont = X[X0[0].columns]
                X_cat = X[X0[1].columns]
                if X_val is None:
                    return [X_cont, X_cat]
                else:
                    X_val_cont = X_val[X0[0].columns]
                    X_val_cat = X_val[X0[1].columns]
                    return ([X_cont, X_cat], [X_val_cont, X_val_cat])
        else:
            return X if X_val is None else (X, X_val)

    def validate_data(self, X, y, X_val=None, y_val=None):
        if X_val is None:
            assert y_val is None, "X_val and y_val should be both given or both None"
        if isinstance(X, list):
            for i in range(len(X)):
                assert X[i].shape[0] == X[0].shape[0]
            assert len(y) == X[0].shape[0]
            if X_val is not None:
                assert isinstance(X_val, list), (
                    f"X_Val should be a list, got X_val of type: {type(X_val)}"
                )
                assert len(X_val) == len(X), (
                    f"X and X_val should have the same length, got len(X) = {len(X)} but len(X_val) = {len(X_val)}"
                )
                assert y_val is not None, (
                    "X_val and y_val should be both given or both None"
                )

                for i in range(len(X_val)):
                    assert X_val[i].shape[0] == X_val[0].shape[0]
                    assert X_val[i].shape[1] == X[i].shape[1], (
                        f"X{i} (dim = {X[i].shape[1]}) and X_val{i} (dim = {X_val[i].shape[1]}) should have the same dimensions"
                    )
                assert len(y_val) == X_val[0].shape[0], (
                    f"X_val (len = {X_val[0].shape[0]}) and y_val (len = {len(y_val)}) should have the same length"
                )
        else:
            assert len(y) == X.shape[0]
            if X_val is not None:
                assert X_val.shape[1] == X.shape[1], (
                    f"X (dim = {X.shape[1]}) and X_val (dim = {X_val.shape[1]}) should have the same dimensions"
                )
                assert len(y_val) == X_val.shape[0], (
                    f"X_val (len = {X_val.shape[0]}) and y_val (len = {len(y_val)}) should have the same length"
                )

    def predict(self, X, method="predict"):
        ct = self._fitted_preprocessor

        if ct is None:
            X_test = X
        else:
            if isinstance(X, list):
                X_trans = ct.transform(pd.concat(X, axis=1))
                X_test = []
                for i in range(len(X)):
                    X_test.append(X_trans[X[i].columns])
            else:
                X_test = ct.transform(X)

        if method == "predict":
            if self.output_shape == 1:
                return np.round(self.model.predict(X_test)).astype(int).ravel()
            else:
                return self.model.predict(X_test).argmax(axis=1)
        elif method == "predict_proba":
            return self.model.predict(X_test)
        else:
            raise ValueError("Invalid method. Choose 'predict' or 'predict_proba'!")

    def evaluate(self, X, y, metric="accuracy"):
        scoring = {
            "accuracy": accuracy_score,
            "auc": roc_auc_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "loss": log_loss,
            "f1_pos": lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=1),
            "f1_neg": lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=0),
            "precision_pos": lambda y_true, y_pred: precision_score(
                y_true, y_pred, pos_label=1
            ),
            "precision_neg": lambda y_true, y_pred: precision_score(
                y_true, y_pred, pos_label=0
            ),
            "recall_pos": lambda y_true, y_pred: recall_score(
                y_true, y_pred, pos_label=1
            ),
            "recall_neg": lambda y_true, y_pred: recall_score(
                y_true, y_pred, pos_label=0
            ),
        }

        assert metric in scoring, "evaluation metric not in list of available scores"

        if metric in ("auc", "loss"):
            y_pred = self.predict(X, method="predict_proba")
            if getattr(y_pred, "ndim", 1) == 2 and y_pred.shape[1] == 1:
                y_pred = y_pred.ravel()
        else:
            y_pred = self.predict(X, method="predict")

        return scoring[metric](y, y_pred)

    def Ensemble_Learning_CV(
        self,
        X,
        y,
        X_test=None,
        fit_params=None,
        n_folds=5,
        n_repeats=1,
        voting_method="soft",
        output_proba=False,
        val_results=False,
        calibrate=False,
        temperature=1.5,
    ):

        assert voting_method in ["soft", "hard"], (
            "voting_method can only take the values 'soft' and 'hard'"
        )
        if output_proba and (voting_method == "hard"):
            raise ValueError(
                "Use soft voting if you want probabilities for output. Otherwise, set output_proba to false"
            )

        if calibrate:
            assert temperature is not None, (
                "temperature must be supplied when calibrate is True"
            )
            assert voting_method == "soft", (
                "when calibrate is True, the voting method is assumed to be soft"
            )

        # Temperature scaling helper
        def apply_temperature_scaling(probs, T):
            # Binary (shape: (n,1) or (n,))
            if probs.ndim == 1 or probs.shape[1] == 1:
                p = probs.reshape(-1)
                # logit transform
                eps = 1e-8
                p = np.clip(p, eps, 1 - eps)
                logits = np.log(p / (1 - p))
                scaled = 1 / (1 + np.exp(-logits / T))
                return scaled.reshape(-1, 1)

            # Multiclass softmax
            else:
                # Convert back to logits
                eps = 1e-8
                probs = np.clip(probs, eps, 1 - eps)
                logits = np.log(probs)
                logits = logits / T
                exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                scaled = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                return scaled

        if X_test is None:
            self.validate_data(X, y)
            X_test = X.copy()
        else:
            if isinstance(X_test, list):
                dummy_y_test = np.ones(X_test[0].shape[0])
            else:
                dummy_y_test = np.ones(X_test.shape[0])

            self.validate_data(X, y, X_test, dummy_y_test)

        if fit_params is None:
            fit_params = self.default_fit_params()

        eval_metric = fit_params["monitor"].removeprefix("val_")

        initial_weights = self.model.get_weights()
        skf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats)
        histories = []
        val_metric_list = []
        train_metric_list = []
        predictions = []
        if calibrate:
            scaled_predictions = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X[0] if isinstance(X, list) else X, y)
        ):
            print(f"fold # {fold}")
            self.model.set_weights(initial_weights)
            if isinstance(X, list):
                X_train = [
                    arr.iloc[train_idx] if hasattr(arr, "iloc") else arr[train_idx]
                    for arr in X
                ]
                X_val = [
                    arr.iloc[val_idx] if hasattr(arr, "iloc") else arr[val_idx]
                    for arr in X
                ]
            else:
                X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
                X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]

            if hasattr(y, "iloc"):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]

            history = self.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_params)
            histories.append(history)
            val_metric = self.evaluate(X_val, y_val, metric=eval_metric)
            train_metric = self.evaluate(X_train, y_train, metric=eval_metric)
            val_metric_list.append(val_metric)
            train_metric_list.append(train_metric)
            fold_preds = (
                self.predict(X_test, method="predict_proba")
                if voting_method == "soft"
                else self.predict(X_test, method="predict")
            )
            predictions.append(fold_preds)

            if calibrate:
                scaled_fold_preds = apply_temperature_scaling(fold_preds, temperature)
                scaled_predictions.append(scaled_fold_preds)

            print(
                f"for fold{fold}: val_{eval_metric}={val_metric:.4f}, train_{eval_metric}={train_metric:.4f}"
            )

        avg_predictions = np.mean(predictions, axis=0)

        if calibrate:
            avg_scaled_predictions = np.mean(scaled_predictions, axis=0)

        print(f"Mean val_{eval_metric} = {np.mean(val_metric_list):.4f}")
        print(f"Mean train_{eval_metric} = {np.mean(train_metric_list):.4f}")

        if output_proba:
            y_pred = avg_predictions
            if calibrate:
                y_pred_scaled = avg_scaled_predictions
        else:
            y_pred = (
                np.round(avg_predictions).astype(int)
                if self.output_shape == 1
                else avg_predictions.argmax(axis=1)
            )
            if calibrate:
                y_pred_scaled = (
                    np.round(avg_scaled_predictions).astype(int)
                    if self.output_shape == 1
                    else avg_scaled_predictions.argmax(axis=1)
                )

        if val_results:
            if calibrate:
                return (
                    histories,
                    predictions,
                    scaled_predictions,
                    np.squeeze(y_pred),
                    np.squeeze(y_pred_scaled),
                    np.array(train_metric_list),
                    np.array(val_metric_list),
                )
            else:
                return (
                    histories,
                    predictions,
                    np.squeeze(y_pred),
                    np.array(train_metric_list),
                    np.array(val_metric_list),
                )
        else:
            if calibrate:
                return (
                    histories,
                    predictions,
                    scaled_predictions,
                    np.squeeze(y_pred),
                    np.squeeze(y_pred_scaled),
                )
            else:
                return histories, predictions, np.squeeze(y_pred)

    @staticmethod
    def safe_clip(p, eps=1e-12):
        return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)

    @staticmethod
    def pos_softplus(x, eps=1e-12):
        return np.logaddexp(0, x) + eps

    @staticmethod
    def sigmoid(x):
        z = np.clip(x, -1e2, 1e2)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def softmax(x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def apply_temperature_scaling(self, Prob, T):
        p = self.safe_clip(Prob)

        def scale_logits(x, t):
            if np.isscalar(t):
                return x / t
            else:
                return x / t.reshape(-1, 1)

        if self.output_shape == 1:
            logits = np.log(p) - np.log1p(-p)
            return self.sigmoid(scale_logits(logits, T))
        else:
            logits = np.log(p) - np.log(np.max(p, axis=1, keepdims=True))
            return self.softmax(scale_logits(logits, T))

    def log_entropy(self, pi):
        p = self.safe_clip(pi)

        if self.output_shape == 1:
            entropy = -(p * np.log(p) + (1.0 - p) * np.log1p(-p))
            norm_entropy = entropy / np.log(2)
        else:
            entropy = -np.sum(p * np.log(p), axis=1)
            norm_entropy = entropy / np.log(self.output_shape)

        return np.log(self.safe_clip(norm_entropy))

    def negative_log_Likelihood(self, q_theta, y):
        q = self.safe_clip(q_theta)

        if self.output_shape == 1:
            assert y.ndim == 1, (
                f"For binary classification, expecting 1 dimensional label, got {y.shape}"
            )
            q = np.squeeze(q)
            return -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))
        else:
            assert y.shape[-1] == self.output_shape, (
                f"Expecting a one-hot-encoded label of shape (n, {self.output_shape}, got {y.shape})"
            )
            return -np.mean(np.sum(y * np.log(q), axis=1))

    def get_logits(self, prob):
        p = self.safe_clip(prob)
        if self.output_shape == 1:
            return np.log(p) - np.log1p(-p)
        else:
            return np.log(p) - np.log(np.max(p, axis=1, keepdims=True))

    def fit_TS(self, p, y, verbose=False):
        Tmin, Tmax = 0.1, 5.0

        def objective(logT):
            T = np.exp(logT)
            p_cal = self.apply_temperature_scaling(p, T)
            return self.negative_log_Likelihood(p_cal, y)

        try:
            res = minimize_scalar(
                objective,
                bounds=(np.log(Tmin), np.log(Tmax)),
                method="bounded",
                options={"disp": 3 if verbose else 1, "xatol": 1e-6},
            )
            logT_opt = res.x
            return np.exp(logT_opt)
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            return 1.0

    def fit_HTS(self, p, y):
        H = self.log_entropy(p)
        a0 = np.log(np.exp(1) - 1)
        b0 = 0.0

        def objective(params):
            a, b = params
            x = a + b * H
            T = self.pos_softplus(x)
            p_cal = self.apply_temperature_scaling(p, T)
            return self.negative_log_Likelihood(p_cal, y)

        try:
            res = minimize(objective, x0=[a0, b0], method="L-BFGS-B")
            return res.x
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            return a0, b0

    def fit_HnLTS(self, p, y):
        H = self.log_entropy(p)
        z = self.get_logits(p)
        a0 = np.log(np.exp(1) - 1)
        b0 = 0.0
        c0 = 0.0

        def objective(params):
            a, b, c = params
            x = a + b * H + c * z
            T = self.pos_softplus(x)
            p_cal = self.apply_temperature_scaling(p, T)
            return self.negative_log_Likelihood(p_cal, y)

        try:
            res = minimize(objective, x0=[a0, b0, c0], method="L-BFGS-B")
            return res.x
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            return a0, b0, c0

    def calibrate_probs(self, p_val, y_val, p_test, method):
        if method is None or str(method).lower() in {
            "none",
            "no_calibration",
            "no_calib",
            "identity",
        }:
            return p_test

        m = str(method)

        if m == "TS":
            T = self.fit_TS(p_val, y_val)
            print(f"T = {T:.4f}")
            return self.apply_temperature_scaling(p_test, T)
        elif m == "HTS":
            a, b = self.fit_HTS(p_val, y_val)
            H = self.log_entropy(p_test)
            x = a + b * H
            T = self.pos_softplus(x)
            print(f"Tmin = {np.min(T):.4f}, Tmax = {np.max(T):.4f}")
            return self.apply_temperature_scaling(p_test, T)
        elif m == "HnLTS":
            a, b, c = self.fit_HnLTS(p_val, y_val)
            H = self.log_entropy(p_test)
            z = self.get_logits(p_test)
            x = a + b * H + c * z
            T = self.pos_softplus(x)
            print(f"Tmin = {np.min(T):.4f}, Tmax = {np.max(T):.4f}")
            return self.apply_temperature_scaling(p_test, T)
        else:
            raise ValueError(
                "calibration method must be 'TS', 'HTS', 'HnLTS', or no calibration (None, 'none', 'no_calibration', 'no_calib', 'identity')"
            )

    def Ensemble_Learning_Calibrated_CV(
        self,
        X,
        y,
        X_test=None,
        fit_params=None,
        n_folds=5,
        n_repeats=1,
        output_proba=False,
        calibration_methods=None,
    ):
        calibration_methods = calibration_methods or ["none"]

        if X_test is None:
            self.validate_data(X, y)
            X_test = X.copy()
        else:
            if isinstance(X_test, list):
                dummy_y_test = np.ones(X_test[0].shape[0])
            else:
                dummy_y_test = np.ones(X_test.shape[0])

            self.validate_data(X, y, X_test, dummy_y_test)

        if fit_params is None:
            fit_params = self.default_fit_params()

        eval_metric = fit_params["monitor"].removeprefix("val_")

        initial_weights = self.model.get_weights()
        skf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats)
        histories = []
        val_metric_list = []
        train_metric_list = []
        calibrated_probas = {m: [] for m in calibration_methods}

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X[0] if isinstance(X, list) else X, y)
        ):
            print(f"fold # {fold}")
            self.model.set_weights(initial_weights)
            if isinstance(X, list):
                X_train = [
                    arr.iloc[train_idx] if hasattr(arr, "iloc") else arr[train_idx]
                    for arr in X
                ]
                X_val = [
                    arr.iloc[val_idx] if hasattr(arr, "iloc") else arr[val_idx]
                    for arr in X
                ]
            else:
                X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
                X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]

            if hasattr(y, "iloc"):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]

            history = self.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_params)
            histories.append(history)
            val_metric = self.evaluate(X_val, y_val, metric=eval_metric)
            train_metric = self.evaluate(X_train, y_train, metric=eval_metric)
            val_metric_list.append(val_metric)
            train_metric_list.append(train_metric)
            print(
                f"for fold{fold}: val_{eval_metric}={val_metric:.4f}, train_{eval_metric}={train_metric:.4f}"
            )

            val_preds = self.predict(X_val, method="predict_proba")
            test_preds = self.predict(X_test, method="predict_proba")

            for m in calibration_methods:
                print(m)
                test_preds_calibrated = self.calibrate_probs(
                    val_preds, y_val, test_preds, m
                )
                calibrated_probas[m].append(test_preds_calibrated)

        print(f"Mean val_{eval_metric} = {np.mean(val_metric_list):.4f}")
        print(f"Mean train_{eval_metric} = {np.mean(train_metric_list):.4f}")

        avg_probas = {
            m: np.mean(calibrated_probas[m], axis=0) for m in calibration_methods
        }
        if output_proba:
            return histories, avg_probas
        else:
            predictions = {}
            for m in calibration_methods:
                y_pred = (
                    np.round(avg_probas[m]).astype(int)
                    if self.output_shape == 1
                    else avg_probas[m].argmax(axis=1)
                )
                predictions[m] = np.squeeze(y_pred)
        return histories, predictions

    def ensemble_fit(
        self,
        X,
        y,
        X_test,
        fit_params=None,
        n_repeats=3,
        voting_method="soft",
        output_proba=False,
    ):
        assert voting_method in ["soft", "hard"], (
            "voting_method can only take the values 'soft' and 'hard'"
        )
        if output_proba and (voting_method == "hard"):
            raise ValueError(
                "use soft voting if you want probabilities for output. Otherwise, set output_proba to false"
            )

        dummy_y_test = (
            np.ones(X_test[0].shape[0])
            if isinstance(X_test, list)
            else np.ones(X_test.shape[0])
        )
        self.validate_data(X, y, X_test, dummy_y_test)
        fit_params = (
            dict(fit_params)
            if fit_params is not None
            else self.default_fit_params(val=False)
        )
        eval_metric = fit_params.pop("metric", "accuracy")

        histories = []
        train_metric_list = []
        predictions = []

        for trial in range(n_repeats):
            history = self.fit(X, y, **fit_params)
            histories.append(history)
            train_metric = self.evaluate(X, y, metric=eval_metric)
            train_metric_list.append(train_metric)
            trial_preds = (
                self.predict(X_test, method="predict_proba")
                if voting_method == "soft"
                else self.predict(X_test, method="predict")
            )
            predictions.append(trial_preds)
            print(f"for {trial = }: training {eval_metric} = {train_metric:.4f}")

        avg_predictions = np.mean(predictions, axis=0)
        print(f"Mean training {eval_metric} = {np.mean(train_metric_list):.4f}")

        if output_proba:
            y_pred = avg_predictions
        else:
            y_pred = (
                np.round(avg_predictions).astype(int)
                if self.output_shape == 1
                else avg_predictions.argmax(axis=1)
            )

        return histories, predictions, np.squeeze(y_pred)

    @staticmethod
    def default_fit_params(val=True):
        if val:
            return {
                "epochs": 600,
                "batch_size": 64,
                "patience": 100,
                "verbose": False,
                "val_split": 0.15,
                "monitor": "val_accuracy",
            }
        else:
            return {
                "epochs": 200,
                "batch_size": 64,
                "verbose": False,
                "val_split": None,
                "metric": "accuracy",
            }

    def parallel_calibrated_cv(
        self,
        X,
        y,
        X_test=None,
        fit_params=None,
        n_folds=5,
        n_repeats=1,
        output_proba=False,
        calibration_methods=None,
        n_jobs=-1,
    ):

        if not hasattr(self, "_init_kwargs"):
            raise RuntimeError(
                "Model has no _init_kwargs. "
                "Ensure DeepLearning_Models is fully loaded before creating the model."
            )

        model_class = type(self)
        model_kwargs = self._init_kwargs
        calibration_methods = calibration_methods or ["none"]

        X_ref = X[0] if isinstance(X, list) else X
        if X_test is None:
            self.validate_data(X, y)
            X_test = [arr.copy() for arr in X] if isinstance(X, list) else X.copy()
        else:
            dummy = np.ones(
                X_test[0].shape[0] if isinstance(X_test, list) else X_test.shape[0]
            )
            self.validate_data(X, y, X_test, dummy)

        if fit_params is None:
            fit_params = self.default_fit_params()

        eval_metric = fit_params["monitor"].removeprefix("val_")
        initial_weights = self.model.get_weights()

        skf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats)
        fold_splits = list(skf.split(X_ref, y))
        n_total = len(fold_splits)
        n_workers = min(
            multiprocessing.cpu_count() if n_jobs == -1 else n_jobs, n_total
        )

        fold_args = [
            (
                fold,
                train_idx,
                val_idx,
                model_class,
                model_kwargs,
                X,
                y,
                X_test,
                fit_params,
                calibration_methods,
                initial_weights,
            )
            for fold, (train_idx, val_idx) in enumerate(fold_splits)
        ]

        results = [None] * n_total
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp_ctx, initializer=_parallel_worker_init
        ) as executor:
            future_to_fold = {
                executor.submit(_parallel_fold_worker, args): args[0]
                for args in fold_args
            }
            for future in as_completed(future_to_fold):
                fold, calibrated, val_metric, train_metric, history_dict = (
                    future.result()
                )
                results[fold] = (calibrated, val_metric, train_metric, history_dict)
                print(
                    f"fold # {fold}  —  "
                    f"val_{eval_metric}={val_metric:.4f}, "
                    f"train_{eval_metric}={train_metric:.4f}",
                    flush=True,
                )

        val_metrics = [r[1] for r in results]
        train_metrics = [r[2] for r in results]
        history_dicts = [r[3] for r in results]
        calibrated_probas = {m: [r[0][m] for r in results] for m in calibration_methods}

        print(f"Mean val_{eval_metric}   = {np.mean(val_metrics):.4f}", flush=True)
        print(f"Mean train_{eval_metric} = {np.mean(train_metrics):.4f}", flush=True)

        avg_probas = {
            m: np.mean(calibrated_probas[m], axis=0) for m in calibration_methods
        }

        if output_proba:
            return history_dicts, avg_probas

        predictions = {
            m: np.squeeze(
                np.round(avg_probas[m]).astype(int)
                if self.output_shape == 1
                else avg_probas[m].argmax(axis=1)
            )
            for m in calibration_methods
        }
        return history_dicts, predictions


# =============================================================================
# MLP Classifier
# =============================================================================


class MLPClassifier(BaseClassifier):
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        hidden_layers: List[int],
        h_activation: str = "relu",
        momentum: float = 0.99,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        l2_weight: float = 0.0001,
        kernel_initializer: str = "glorot_uniform",
        metrics: List[str] = None,
        preprocessor=None,
    ):

        if not isinstance(output_shape, int) or output_shape < 1:
            raise ValueError(f"output_shape must be a positive int, got {output_shape}")

        if len(hidden_layers) < 1:
            raise ValueError(f"hidden_layers cannot be empty, got {hidden_layers = }")

        super().__init__(output_shape=output_shape, preprocessor=preprocessor)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.h_activation = h_activation
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        self.kernel_initializer = kernel_initializer
        self.metrics = metrics or ["accuracy"]

        # Build model
        self.model = self.build_model()

    def add_hidden_layer(self, h, neurons, layer_num):
        h = Dense(
            neurons,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=L2(self.l2_weight),
            name=f"dense_{layer_num}",
        )(h)
        h = BatchNormalization(momentum=self.momentum, name=f"batch_norm_{layer_num}")(
            h
        )
        h = Activation(self.h_activation, name=f"activation_{layer_num}")(h)
        h = Dropout(rate=self.dropout_rate, name=f"dropout_{layer_num}")(h)
        return h

    def build_model(self) -> Model:
        inputs = Input(shape=(self.input_shape,), name="input_layer")
        h = inputs

        # Add hidden layers
        for i, neurons in enumerate(self.hidden_layers):
            h = self.add_hidden_layer(h, neurons, i + 1)

        # Output layer
        if self.output_shape > 1:
            outputs = Dense(
                self.output_shape, activation="softmax", name="output_layer"
            )(h)
            loss = "categorical_crossentropy"
        else:
            outputs = Dense(1, activation="sigmoid", name="output_layer")(h)
            loss = "binary_crossentropy"

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="MLP_Model")

        # Compile model
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=self.metrics,
        )

        return model


def MLP_Optimize_Parameters(
    X, y, output_shape, N_layers, const_params=None, fit_params=None, preprocessor=None
):
    const_params = const_params or {}
    input_shape = X.shape[1]
    default_fit_params = {
        "epochs": 400,
        "batch_size": 64,
        "patience": 50,
        "monitor": "val_accuracy",
        "verbose": False,
        "custom_callback": True,
        "callback_verbose": False,
    }

    fit_params = fit_params or default_fit_params

    eps = 0.1

    def objective(trial):
        model_params = {}

        if "hidden_layers" in const_params:
            model_params["hidden_layers"] = const_params["hidden_layers"]
        else:
            hidden_layers = np.ones(N_layers).astype(int)
            for layer in range(N_layers):
                hidden_layers[layer] = trial.suggest_int(
                    name=f"hidden_layer{layer}", low=32, high=256
                )
            model_params["hidden_layers"] = hidden_layers

        model_params["h_activation"] = (
            const_params["h_activation"]
            if "h_activation" in const_params
            else trial.suggest_categorical(
                name="h_activation", choices=["relu", "mish", "leaky_relu"]
            )
        )

        model_params["momentum"] = (
            const_params["momentum"]
            if "momentum" in const_params
            else trial.suggest_float(name="momentum", low=0.8, high=0.999)
        )

        model_params["l2_weight"] = (
            const_params["l2_weight"]
            if "l2_weight" in const_params
            else trial.suggest_float(name="l2_weight", low=1e-5, high=0.01, log=True)
        )

        model_params["dropout_rate"] = (
            const_params["dropout_rate"]
            if "dropout_rate" in const_params
            else trial.suggest_float(name="dropout_rate", low=0.0, high=0.5)
        )

        model_params["learning_rate"] = (
            const_params["learning_rate"]
            if "learning_rate" in const_params
            else trial.suggest_float(
                name="learning_rate", low=1e-4, high=1e-2, log=True
            )
        )

        model = MLPClassifier(
            input_shape=input_shape,
            output_shape=output_shape,
            **model_params,
            preprocessor=preprocessor,
        )
        histories, preds, y_pred, train_acc, val_acc = model.Ensemble_Learning_CV(
            X,
            y,
            fit_params=fit_params,
            n_folds=10,
            n_repeats=1,
            voting_method="soft",
            val_results=True,
        )

        return val_acc.mean(axis=0) - eps * max(
            0, train_acc.mean(axis=0) - val_acc.mean(axis=0)
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print(f"Best Accuracy Score: {study.best_value:.5f}")

    return study.best_params


# =============================================================================
# TabResNet Classifier
# Based on the ResNet-like architecture described in 'Revisiting Deep Learning
# Models for Tabular Data', Gorishney et al
# =============================================================================


class TabResNetClassifier(BaseClassifier):
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        hidden_layers: List[int],
        residual_layers: List[int],
        *,
        p_activation: str = "linear",
        h_activation: str = "relu",
        res_activation: str = "relu",
        N_Blocks: int = 1,
        momentum: float = 0.99,
        dropout_rate1: float = 0.3,
        dropout_rate2: float = 0.01,
        learning_rate: float = 0.001,
        l2_weight: float = 1e-4,
        kernel_initializer: str = "glorot_uniform",
        metrics: Optional[List[str]] = None,
        preprocessor=None,
    ):

        if not isinstance(output_shape, int) or output_shape < 1:
            raise ValueError(f"output_shape must be a positive int, got {output_shape}")

        if len(hidden_layers) < 1:
            raise ValueError(f"hidden_layers cannot be empty, got {hidden_layers = }")

        if len(residual_layers) != len(hidden_layers):
            raise ValueError(
                f"residual_layers ({len(residual_layers)}) and hidden_layers ({len(hidden_layers)}) lengths do not match!"
            )

        super().__init__(output_shape=output_shape, preprocessor=preprocessor)

        self.ResNet_Blocks = len(hidden_layers)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.p_activation = p_activation
        self.h_activation = h_activation
        self.res_activation = res_activation
        self.hidden_layers = hidden_layers
        self.residual_layers = residual_layers
        self.N_Blocks = N_Blocks
        self.momentum = momentum
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        self.kernel_initializer = kernel_initializer
        self.metrics = metrics or ["accuracy"]

        # Build model
        self.model = self.build_model()

    def Projection_Block(self, x, ResBlock_Num):
        h = Dense(
            units=self.residual_layers[ResBlock_Num],
            kernel_initializer=self.kernel_initializer,
            activation=self.p_activation,
            kernel_regularizer=L2(self.l2_weight),
            name=f"dense_projection_{ResBlock_Num}",
        )(x)
        return h

    def Residual_Block(self, h, ResBlock_Num, Block_Num):
        h = BatchNormalization(
            momentum=self.momentum, name=f"batch_norm1_{ResBlock_Num}_{Block_Num}"
        )(h)
        h = Dense(
            units=self.hidden_layers[ResBlock_Num],
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=L2(self.l2_weight),
            name=f"dense1_{ResBlock_Num}_{Block_Num}",
        )(h)
        h = Activation(
            self.h_activation, name=f"h_activation_{ResBlock_Num}_{Block_Num}"
        )(h)
        h = Dropout(
            rate=self.dropout_rate1, name=f"dropout1_{ResBlock_Num}_{Block_Num}"
        )(h)
        h = Dense(
            units=self.residual_layers[ResBlock_Num],
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=L2(self.l2_weight),
            name=f"dense2_{ResBlock_Num}_{Block_Num}",
        )(h)
        h = Activation(
            self.p_activation, name=f"p_activation_{ResBlock_Num}_{Block_Num}"
        )(h)
        h = Dropout(
            rate=self.dropout_rate2, name=f"dropout2_{ResBlock_Num}_{Block_Num}"
        )(h)

        return h

    def ResNet_Block(self, x, ResBlock_Num):
        h = self.Projection_Block(x, ResBlock_Num)

        for Block in range(self.N_Blocks):
            h_skip = h
            h = self.Residual_Block(h, ResBlock_Num, Block + 1)
            h = Add()([h, h_skip])

        h = BatchNormalization(
            momentum=self.momentum, name=f"batch_norm_Res{ResBlock_Num}"
        )(h)
        h = Activation(
            activation=self.res_activation, name=f"res_activation_{ResBlock_Num}"
        )(h)

        return h

    def build_model(self) -> Model:
        inputs = Input(shape=(self.input_shape,), name="input_layer")
        h = inputs

        for b in range(self.ResNet_Blocks):
            h = self.ResNet_Block(h, b)

        # Output layer
        if self.output_shape > 1:
            outputs = Dense(
                self.output_shape, activation="softmax", name="output_layer"
            )(h)
            loss = "categorical_crossentropy"
        else:
            outputs = Dense(1, activation="sigmoid", name="output_layer")(h)
            loss = "binary_crossentropy"

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="TabResNet_Model")

        # Compile model
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=self.metrics,
        )

        return model


def TabResNet_Optimal_Parameters(
    X, y, output_shape, N_layers, const_params=None, fit_params=None, preprocessor=None
):
    const_params = const_params or {}
    input_shape = X.shape[1]

    default_fit_params = {
        "epochs": 300,
        "batch_size": 64,
        "patience": 50,
        "monitor": "val_accuracy",
        "verbose": False,
        "custom_callback": True,
        "callback_verbose": False,
    }

    fit_params = fit_params or default_fit_params

    def objective(trial):
        model_params = {}

        if "hidden_layers" in const_params:
            model_params["hidden_layers"] = const_params["hidden_layers"]
        else:
            hidden_layers = np.ones(N_layers).astype(int)
            for layer in range(N_layers):
                hidden_layers[layer] = trial.suggest_int(
                    name=f"hidden_layer{layer}", low=4, high=256
                )
            model_params["hidden_layers"] = hidden_layers

        if "residual_layers" in const_params:
            model_params["residual_layers"] = const_params["residual_layers"]
        else:
            residual_layers = np.ones(N_layers).astype(int)
            for layer in range(N_layers):
                residual_layers[layer] = trial.suggest_int(
                    name=f"residual_layer{layer}", low=4, high=256
                )
            model_params["residual_layers"] = residual_layers

        model_params["p_activation"] = (
            const_params["p_activation"]
            if "p_activation" in const_params
            else trial.suggest_categorical(
                name="p_activation", choices=["linear", "relu", "mish"]
            )
        )

        model_params["h_activation"] = (
            const_params["h_activation"]
            if "h_activation" in const_params
            else trial.suggest_categorical(
                name="h_activation", choices=["relu", "leaky_relu", "mish"]
            )
        )

        model_params["res_activation"] = (
            const_params["res_activation"]
            if "res_activation" in const_params
            else trial.suggest_categorical(
                name="res_activation", choices=["relu", "leaky_relu", "mish"]
            )
        )

        model_params["N_Blocks"] = (
            const_params["N_Blocks"]
            if "N_Blocks" in const_params
            else trial.suggest_int(name="N_Blocks", low=1, high=6)
        )

        model_params["momentum"] = (
            const_params["momentum"]
            if "momentum" in const_params
            else trial.suggest_float(name="momentum", low=0.8, high=0.999)
        )

        model_params["dropout_rate1"] = (
            const_params["dropout_rate1"]
            if "dropout_rate1" in const_params
            else trial.suggest_float(name="dropout_rate1", low=0.1, high=0.5)
        )

        model_params["dropout_rate2"] = (
            const_params["dropout_rate2"]
            if "dropout_rate2" in const_params
            else trial.suggest_float(name="dropout_rate2", low=0.0, high=0.5)
        )

        model_params["learning_rate"] = (
            const_params["learning_rate"]
            if "learning_rate" in const_params
            else trial.suggest_float(
                name="learning_rate", low=1e-4, high=0.01, log=True
            )
        )

        model_params["l2_weight"] = (
            const_params["l2_weight"]
            if "l2_weight" in const_params
            else trial.suggest_float(name="l2_weight", low=1e-5, high=0.01, log=True)
        )

        model = TabResNetClassifier(
            input_shape=input_shape,
            output_shape=output_shape,
            **model_params,
            preprocessor=preprocessor,
        )
        histories, preds, y_pred, train_acc, val_acc = model.Ensemble_Learning_CV(
            X,
            y,
            fit_params=fit_params,
            n_folds=10,
            n_repeats=1,
            voting_method="soft",
            val_results=True,
        )

        return val_acc.mean(axis=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=120)

    print(f"Best Accuracy Score: {study.best_value:.5f}")

    return study.best_params


# =============================================================================
# FTTransformer
# Based on the FTTransformer architecture described in 'Revisiting Deep Learning
# Models for Tabular Data', Gorishney et al.
# This is a TensorFlow version with additional modifications.
# =============================================================================

# --- Embedding Layers ---


class ClsTokenLayer(Layer):
    def __init__(self, d_embedding: int, name="Cls_Token"):
        """
        Args:
            d_embedding: the embedding size
        """
        super().__init__(name=name)

        self.cls_embedding = self.add_weight(
            shape=(1, 1, d_embedding),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-(d_embedding**-0.5), maxval=d_embedding**-0.5
            ),
            trainable=True,
            name="cls_embedding",
        )
        self.d_embedding = d_embedding

    def call(self, x):
        if len(x.shape) != 3:
            raise ValueError(
                f"Input must have shape (batch_size, n_features, d_embedding), got {x.shape}"
            )
        assert x.shape[-1] == self.d_embedding, (
            f"Input's last dimension should be {self.d_embedding}, got {x.shape[-1]}"
        )

        batch_size = tf.shape(x)[0]

        # Tile the cls token for each sample in the batch.
        cls_tokens = tf.tile(self.cls_embedding, [batch_size, 1, 1])

        # Concatenate the cls token at the beginning of the "sequence".
        x = tf.concat([cls_tokens, x], axis=1)

        return x


class LinearEmbeddings(Layer):
    def __init__(self, n_features: int, d_embedding: int, name="Continuous_Embeddings"):
        """
        Args:
            n_features: Number of continuous features
            d_embedding: the embedding size
        """
        super().__init__(name=name)

        if n_features <= 0:
            raise ValueError(f"n_features must be positive, however: {n_features = }")
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, however: {d_embedding = }")

        self.n_features = n_features
        self.d_embedding = d_embedding

        # Initialize weight and bias as learnable parameters
        self.weight = self.add_weight(
            shape=(n_features, d_embedding),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-(d_embedding**-0.5), maxval=d_embedding**-0.5
            ),
            trainable=True,
            name="linear_embedding_weight",
        )

        self.bias = self.add_weight(
            shape=(n_features, d_embedding),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-(d_embedding**-0.5), maxval=d_embedding**-0.5
            ),
            trainable=True,
            name="linear_embedding_bias",
        )

    def call(self, x):
        if len(x.shape) != 2:
            raise ValueError(
                f"Input must have shape (batch_size, n_features), got {x.shape}"
            )

        x = tf.expand_dims(x, axis=-1)
        x = x * self.weight + self.bias

        return x


class PeriodicEmbeddings(Layer):
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        n_frequencies: int = 48,
        sigma: float = 1.0,
        name="Periodic_Embeddings",
    ):
        """
        Periodic (sin/cos) embeddings for continuous features, followed by a
        per-feature linear projection to d_embedding.

        For each feature j and frequency k:
            e_{j,k}(x) = [sin(2π v_{j,k} x), cos(2π v_{j,k} x)]
        where v are learnable frequencies initialized from N(0, sigma).

        Args:
            n_features:    Number of continuous features.
            d_embedding:   Output embedding dimension per feature.
            n_frequencies: Number of sin/cos frequency pairs per feature.
            sigma:         Std dev for frequency initialization.
        """
        super().__init__(name=name)

        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features = }")
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, got {d_embedding = }")
        if n_frequencies <= 0:
            raise ValueError(f"n_frequencies must be positive, got {n_frequencies = }")

        self.n_features = n_features
        self.d_embedding = d_embedding
        self.n_frequencies = n_frequencies

        # Learnable frequencies: shape (n_features, n_frequencies)
        self.frequencies = self.add_weight(
            shape=(n_features, n_frequencies),
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=sigma),
            trainable=True,
            name="frequencies",
        )

        # Per-feature linear projection: (2*n_frequencies) → d_embedding
        self.projection_weight = self.add_weight(
            shape=(n_features, 2 * n_frequencies, d_embedding),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name="projection_weight",
        )
        self.projection_bias = self.add_weight(
            shape=(n_features, d_embedding),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="projection_bias",
        )

    def call(self, x):
        if len(x.shape) != 2:
            raise ValueError(
                f"Input must have shape (batch_size, n_features), got {x.shape}"
            )

        # x: (batch, n_features) → (batch, n_features, 1)
        x_exp = tf.expand_dims(x, axis=-1)

        # (batch, n_features, n_frequencies)
        x_freq = 2.0 * np.pi * x_exp * self.frequencies

        # (batch, n_features, 2*n_frequencies)
        x_periodic = tf.concat([tf.math.sin(x_freq), tf.math.cos(x_freq)], axis=-1)

        # Per-feature projection → (batch, n_features, d_embedding)
        x_out = (
            tf.einsum("bfd,fde->bfe", x_periodic, self.projection_weight)
            + self.projection_bias
        )

        return x_out


class CategoricalEmbeddings(Layer):
    def __init__(
        self,
        cardinalities: List[int],
        d_embedding: int,
        bias: bool = True,
        name="Categorical_Embeddings",
    ):
        """
        Args:
            cardinalities: List of integers, the number of distinct values for each feature.
            d_embedding: Integer, the embedding size for each feature.
            bias: Boolean, whether to add a trainable bias vector for each feature.
        """
        super().__init__(name=name)

        if not cardinalities:
            raise ValueError("cardinalities must not be empty.")
        if any(c <= 0 for c in cardinalities):
            raise ValueError("cardinalities must contain only positive values.")
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, got {d_embedding}.")

        # Create an embedding layer for each feature
        self.embeddings = [
            Embedding(input_dim=c, output_dim=d_embedding) for c in cardinalities
        ]

        # Create trainable biases for each feature if needed
        if bias:
            self.bias = self.add_weight(
                shape=(len(cardinalities), d_embedding),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=-(d_embedding**-0.5), maxval=d_embedding**-0.5
                ),
                trainable=True,
                name="categorical_embedding_bias",
            )
        else:
            self.bias = None

    def call(self, x):
        if len(x.shape) < 2:
            raise ValueError(
                f"Input tensor must have at least two dimensions, got {x.shape}"
            )

        if x.shape[-1] != len(self.embeddings):
            raise ValueError(
                f"The last dimension of the input ({x.shape[-1]}) "
                f"must match the number of categorical features ({len(self.embeddings)})."
            )

        # apply each embedding to its corresponding feature
        embeddings = [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))]
        x = tf.stack(embeddings, axis=-2)

        if self.bias is not None:
            x = x + self.bias

        return x


# --- MLP Backbone ---


class MLP_Backbone(Model):
    def __init__(
        self,
        *,
        hidden_layers: List[int],
        output_shape: Optional[int] = None,
        mlp_activation: str = "relu",
        momentum: float = 0.99,
        mlp_dropout: float = 0.3,
        l2_weight: float = 0.0001,
        name="MLP_Backbone",
    ):
        super().__init__(name=name)

        if len(hidden_layers) < 1:
            raise ValueError(f"hidden_layers cannot be empty, got {hidden_layers = }")

        if output_shape:
            assert isinstance(output_shape, int) and output_shape > 0, (
                f"output shape must be None or positive integer, got {output_shape = }"
            )

        self.mlp_layers = []
        for layer_num in range(len(hidden_layers)):
            if mlp_activation != "pmish":
                activation_function = Activation(
                    mlp_activation, name=f"{mlp_activation}_activation_{layer_num}"
                )
            else:
                activation_function = PMish(name=f"pmish_activation_{layer_num}")

            hidden_layer = {
                "dense": Dense(
                    units=hidden_layers[layer_num],
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=L2(l2_weight),
                    name=f"mlp_dense_{layer_num}",
                ),
                "mlp_activation": activation_function,
                "BatchNorm": BatchNormalization(
                    momentum=momentum, name=f"batch_norm_{layer_num}"
                ),
                "Dropout": Dropout(rate=mlp_dropout, name=f"mlp_dropout_{layer_num}"),
            }
            if (layer_num == len(hidden_layers) - 1) and output_shape:
                if output_shape == 1:
                    hidden_layer["mlp_output"] = Dense(
                        units=1, activation="sigmoid", name="sigmoid_mlp_output"
                    )
                else:
                    hidden_layer["mlp_output"] = Dense(
                        units=output_shape,
                        activation="softmax",
                        name="softmax_mlp_output",
                    )

            self.mlp_layers.append(hidden_layer)

    def add_hidden_layer(self, x, hidden_layer):
        x = hidden_layer["dense"](x)
        x = hidden_layer["BatchNorm"](x)
        x = hidden_layer["mlp_activation"](x)
        x = hidden_layer["Dropout"](x)
        if "mlp_output" in hidden_layer:
            x = hidden_layer["mlp_output"](x)
        return x

    def call(self, x):
        for hidden_layer in self.mlp_layers:
            x = self.add_hidden_layer(x, hidden_layer)
        return x

    @staticmethod
    def default_mlp_params():
        return {
            "hidden_layers": [64, 32],
            "mlp_activation": "relu",
            "momentum": 0.99,
            "mlp_dropout": 0.3,
            "l2_weight": 0.0001,
        }


# --- Additional Activation Functions ---


class ReGLU(Layer):
    def call(self, x):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        return x1 * tf.nn.relu(x2)


class MiGLU(Layer):
    def call(self, x):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        return x1 * tf.keras.activations.mish(x2)


class GeGLU(Layer):
    def call(self, x):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        return x1 * tf.nn.gelu(x2)


class pmish_fixed(Layer):
    def __init__(self, alpha=1.0, beta=1.0, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta

    def call(self, x):
        return x * tf.math.tanh(self.alpha * tf.math.softplus(self.beta * x))


class PMish(Layer):
    """
    This function returns x * tanh(alpha * softplus(beta * x))

    Parameters
    -----------
    channel_shared (bool)
        If True, the same weights will be shared across all channels.
        If False, each channel will have its own weights
    in_channels (int or None)
        When channel_shared is False, this tells the layer how many channels
        there are so it can create a weight per channel.
    a_min and a_max (float)
        alpha value will be contrained between a_min and a_max
    b_min and b_max (float)
        beta value will be contrained between b_min and b_max
    name (str or None)
        A unique layer name
    """

    def __init__(
        self,
        channel_shared=True,
        in_channels=None,
        a_min=0.1,
        a_max=5.0,
        b_min=0.1,
        b_max=5.0,
        name=None,
    ):
        super().__init__(name=name)

        assert a_min < 1.0 and a_max > 1.0
        assert b_min < 1.0 and b_max > 1.0

        self.a_mean_init = np.log((1.0 - a_min) / (a_max - 1.0))
        self.b_mean_init = np.log((1.0 - b_min) / (b_max - 1.0))

        self.channel_shared = channel_shared
        self.in_channels = in_channels
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max

        if self.channel_shared:
            self.build((None,))
            self._built = True
        elif self.in_channels:
            self.build((None, self.in_channels))
            self._built = True

    def build(self, input_shape):
        if self.channel_shared:
            w_shape = (1,)
        else:
            w_shape = (input_shape,)

        self.w_alpha = self.add_weight(
            shape=w_shape,
            initializer=tf.keras.initializers.TruncatedNormal(
                mean=self.a_mean_init, stddev=0.1
            ),
            trainable=True,
            name="w_alpha",
        )

        self.w_beta = self.add_weight(
            shape=w_shape,
            initializer=tf.keras.initializers.TruncatedNormal(
                mean=self.b_mean_init, stddev=0.1
            ),
            trainable=True,
            name="w_beta",
        )

    def call(self, x):
        alpha = self.a_min + (self.a_max - self.a_min) * tf.nn.sigmoid(self.w_alpha)
        beta = self.b_min + (self.b_max - self.b_min) * tf.nn.sigmoid(self.w_beta)
        return x * tf.math.tanh(alpha * tf.math.softplus(beta * x))


# --- Transformer Backbone ---


class Transformer_Backbone(Model):
    def __init__(
        self,
        n_blocks: int = 2,
        d_block: int = 128,
        n_heads: int = 8,
        ffn_d_hidden_multiplier: float = 2,
        ffn_activation: str = "relu",
        tfout_activation: str = "mish",
        residual_dropout: float = 0.0,
        ffn_dropout: float = 0.05,
        alpha: float = 0.25,
        beta: float = 1.75,
        name="Transformer_Backbone",
    ):
        super().__init__(name=name)

        ffn_d_hidden = int(d_block * ffn_d_hidden_multiplier)
        ffn_GLU_variant = ffn_activation in ["ReGLU", "GeGLU", "MiGLU"]
        ffn_units = ffn_d_hidden * (2 if ffn_GLU_variant else 1)

        if ffn_activation == "pmish_fixed":
            assert alpha is not None and beta is not None, (
                "alpha and beta must be provided when ffn_activation is 'pmish_fixed'"
            )

        self.blocks = []
        for block_number in range(n_blocks):
            block_layers = {
                "attention": MultiHeadAttention(
                    num_heads=n_heads,
                    key_dim=d_block,
                    name=f"attention_b{block_number}",
                ),
                "dropout_att_res": Dropout(
                    residual_dropout, name=f"Dropout_att_res_b{block_number}"
                ),
                "ffn_norm": LayerNormalization(
                    epsilon=1e-5, name=f"ffn_LayerNorm_b{block_number}"
                ),
                "ffn_Dense1": Dense(ffn_units, name=f"ffn_Dense1_b{block_number}"),
                "dropout_ffn": Dropout(
                    ffn_dropout, name=f"Dropout_ffn_b{block_number}"
                ),
                "ffn_Dense2": Dense(d_block, name=f"ffn_Dense2_b{block_number}"),
                "dropout_ffn_res": Dropout(
                    residual_dropout, name=f"Dropout_ffn_res_b{block_number}"
                ),
            }

            if ffn_GLU_variant:
                if ffn_activation == "ReGLU":
                    block_layers["ffn_activation"] = ReGLU(
                        name=f"ReGLU_b{block_number}"
                    )
                elif ffn_activation == "GeGLU":
                    block_layers["ffn_activation"] = GeGLU(
                        name=f"GeGLU_b{block_number}"
                    )
                elif ffn_activation == "MiGLU":
                    block_layers["ffn_activation"] = MiGLU(
                        name=f"MiGLU_b{block_number}"
                    )
                else:
                    raise ValueError(
                        f"Activation not a GLU variant. Got {ffn_activation}."
                    )
            elif ffn_activation == "pmish":
                block_layers["ffn_activation"] = PMish(
                    name=f"pmish_activation_b{block_number}"
                )
            elif ffn_activation == "pmish_fixed":
                block_layers["ffn_activation"] = pmish_fixed(
                    alpha=alpha,
                    beta=beta,
                    name=f"pmish_fixed_activation_b{block_number}",
                )
            elif ffn_activation == "prelu":
                block_layers["ffn_activation"] = PReLU(
                    name=f"prelu_activation_b{block_number}"
                )
            else:
                block_layers["ffn_activation"] = Activation(
                    ffn_activation, name=f"Activation_b{block_number}"
                )

            if block_number > 0:
                block_layers["attention_norm"] = LayerNormalization(
                    axis=-1, name=f"attention_LayerNorm_b{block_number}"
                )
            if block_number == n_blocks - 1:
                block_layers["last block"] = "last block"

            self.blocks.append(block_layers)

        self.layer_norm_final = LayerNormalization(axis=-1, name="LayerNorm_Final")
        self.tf_out_activation = Activation(
            tfout_activation, name="TF_output_Activation"
        )

    def add_block(self, x, block):
        is_last = "last block" in block

        # Make a copy for residual
        x_identity = x[:, :1, :] if is_last else x

        # PreNormalize if not 1st block
        if "attention_norm" in block:
            x = block["attention_norm"](x)

        # Determine query for the attention
        query = x[:, :1, :] if is_last else x

        # Multi-head attention
        x_att = block["attention"](query, x)

        # Attention residual dropout
        x_att = block["dropout_att_res"](x_att)

        # ADD & Normalize
        x = Add()([x_att, x_identity])

        x_identity = x
        x = block["ffn_norm"](x)

        # Feed Forward Block: Dense layer with activation, dropout, then another Dense layer with linear activation
        # 1) First Dense Layer
        x = block["ffn_Dense1"](x)
        x = block["ffn_activation"](x)
        # 2) Dropout
        x = block["dropout_ffn"](x)
        # 3) Second Dense Layer with linear activation
        x = block["ffn_Dense2"](x)

        # FFN residual dropout
        x = block["dropout_ffn_res"](x)

        # Add the unnormalized attention residual
        x = Add()([x, x_identity])

        return x

    def call(self, x):
        for block in self.blocks:
            x = self.add_block(x, block)

        x = self.layer_norm_final(tf.squeeze(x, axis=1))
        x = self.tf_out_activation(x)
        return x


# --- FTTransformer Main ---


class FTTransformer(BaseClassifier):
    def __init__(
        self,
        *,
        n_cont_features: int,
        cat_cardinalities: List[int],
        output_shape: int,
        transformer_backbone_kwargs: dict = None,
        mlp_backbone_kwargs: dict = None,
        cont_embedding_type: str = "linear",
        n_frequencies: int = 48,
        frequencies_sigma: float = 1.0,
        learning_rate: float = 0.001,
        metrics: Optional[List[str]] = None,
        preprocessor=None,
        name="FTTransformer",
    ):
        if n_cont_features < 0:
            raise ValueError(
                f"n_cont_features must be non-negative, however: {n_cont_features = }"
            )
        if n_cont_features == 0 and not cat_cardinalities:
            raise ValueError("At least one type of feature must be provided.")

        assert isinstance(output_shape, int) and output_shape > 0, (
            f"output_shape should be a positive integer, got {output_shape = }"
        )

        if cont_embedding_type not in ("linear", "periodic"):
            raise ValueError(
                f"cont_embedding_type must be 'linear' or 'periodic', got {cont_embedding_type = }"
            )
        if transformer_backbone_kwargs is None:
            raise ValueError(
                "transformer_backbone_kwargs must be provided. "
                "Use FTTransformer.default_Transformer_params() for a default configuration."
            )
        if mlp_backbone_kwargs is None:
            raise ValueError(
                "mlp_backbone_kwargs must be provided. "
                "Use MLP_Backbone.default_mlp_params() for a default configuration."
            )

        super().__init__(output_shape=output_shape, preprocessor=preprocessor)

        d_block = transformer_backbone_kwargs["d_block"]

        if n_cont_features > 0:
            if cont_embedding_type == "linear":
                self.cont_embeddings = LinearEmbeddings(
                    n_cont_features, d_block, name="cont_embeddings"
                )
            else:
                self.cont_embeddings = PeriodicEmbeddings(
                    n_cont_features,
                    d_block,
                    n_frequencies=n_frequencies,
                    sigma=frequencies_sigma,
                    name="cont_embeddings",
                )
        else:
            self.cont_embeddings = None
        self.cat_embeddings = (
            CategoricalEmbeddings(cat_cardinalities, d_block, name="cat_embeddings")
            if cat_cardinalities
            else None
        )

        self.add_cls_embedding = ClsTokenLayer(d_block)

        self.n_cont_features = n_cont_features
        self.cat_cardinalities = cat_cardinalities
        self.d_block = d_block
        self.learning_rate = learning_rate
        self.metrics = metrics or ["accuracy"]
        self.output_shape = output_shape

        self.transformer_backbone = Transformer_Backbone(**transformer_backbone_kwargs)
        self.mlp_backbone = MLP_Backbone(
            output_shape=output_shape, **mlp_backbone_kwargs
        )

        if output_shape > 1:
            self.loss = "sparse_categorical_crossentropy"
        else:
            self.loss = "binary_crossentropy"

        self.model = self.build_model()

    def build_model(self):
        inputs = []
        embedding_parts = []

        if self.cont_embeddings is not None:
            x_cont_input = Input(
                shape=(self.n_cont_features,), name="continuous_features", dtype=tf.float32
            )
            inputs.append(x_cont_input)
            embedding_parts.append(self.cont_embeddings(x_cont_input))

        if self.cat_embeddings is not None:
            x_cat_input = Input(
                shape=(len(self.cat_cardinalities),),
                name="categorical_features",
                dtype=tf.int32,
            )
            inputs.append(x_cat_input)
            embedding_parts.append(self.cat_embeddings(x_cat_input))

        if len(embedding_parts) == 1:
            x_embeddings = embedding_parts[0]
        else:
            x_embeddings = Concatenate(axis=1, name="concat_embeddings")(
                embedding_parts
            )
        x_embeddings = self.add_cls_embedding(x_embeddings)

        x = self.transformer_backbone(x_embeddings)
        x = self.mlp_backbone(x)

        outputs = x

        model = Model(inputs=inputs, outputs=outputs, name="FTTransformer_Model")

        model.compile(
            loss=self.loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=self.metrics,
        )

        return model

    @staticmethod
    def default_Transformer_params(n_blocks: int = 3):
        """Get the default hyperparameters.
        Args:
            n_blocks: the number of blocks. The supported values are: 1, 2, 3, 4, 5, 6.
        Returns:
            the default keyword arguments for the constructor.
        """
        if n_blocks < 1 or n_blocks > 6:
            raise ValueError(
                "Default configurations are available"
                " only for the following values of n_blocks: 1, 2, 3, 4, 5, 6."
                f" However, {n_blocks = }"
            )
        return {
            "n_blocks": n_blocks,
            "d_block": [96, 128, 192, 256, 320, 384][n_blocks - 1],
            "n_heads": 8,
            "ffn_d_hidden_multiplier": 4 / 3,
            "ffn_activation": "relu",
            "tfout_activation": "relu",
            "ffn_dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25][n_blocks - 1],
            "residual_dropout": 0.0,
        }


# =============================================================================
# MLP Classifier with Embedded Categorical Features
# =============================================================================


class CatEmbs(Layer):
    def __init__(
        self,
        cardinalities: List[int],
        d_ratios: List[float],
        bias: bool = True,
        name="Categorical_Embeddings",
    ):
        super().__init__(name=name)

        if not cardinalities:
            raise ValueError("cardinalities must not be empty.")
        if any(c <= 0 for c in cardinalities):
            raise ValueError("cardinalities must contain only positive values.")
        if any(d <= 0 for d in d_ratios):
            raise ValueError("d_ratios must contain only positive values.")

        assert len(d_ratios) == len(cardinalities), (
            "d_ratios and cardinalities should have the same length"
        )

        # Compute per-feature embedding dimensions
        self.embedding_dims = [
            max(1, int(np.ceil(d_ratios[i] * cardinalities[i])))
            for i in range(len(cardinalities))
        ]

        # Build one embedding layer per feature
        self.embeddings = [
            Embedding(input_dim=c, output_dim=emb_dim, name=f"embedding_feature_{i}")
            for i, (c, emb_dim) in enumerate(zip(cardinalities, self.embedding_dims))
        ]

        # Optional per-feature biases
        if bias:
            self.biases = [
                self.add_weight(
                    shape=(emb_dim,),
                    initializer=tf.keras.initializers.RandomUniform(
                        minval=-(emb_dim**-0.5), maxval=(emb_dim**-0.5)
                    ),
                    trainable=True,
                    name=f"categorical_embedding_bias_feature_{i}",
                )
                for i, emb_dim in enumerate(self.embedding_dims)
            ]
        else:
            self.biases = None

    def call(self, x):
        if len(x.shape) < 2:
            raise ValueError(
                f"Input tensor must have at least two dimensions, got {x.shape}"
            )

        if x.shape[-1] != len(self.embeddings):
            raise ValueError(
                f"The last dimension of the input ({x.shape[-1]}) "
                f"must match the number of categorical features ({len(self.embeddings)})."
            )

        # Apply embeddings feature by feature
        parts = []
        for i, emb_layer in enumerate(self.embeddings):
            part = emb_layer(x[:, i])  # (batch, emb_dim_i)
            if self.biases is not None:
                part = part + self.biases[i]
            parts.append(part)

        # Concatenate: shape (batch, sum(emb_dim_i))
        return tf.concat(parts, axis=-1)


class CatEmb_MLPClassifier(BaseClassifier):
    def __init__(
        self,
        input_shape: int,
        cat_cardinalities: List[int],
        output_shape: int,
        hidden_layers: List[int],
        d_ratios: List[float],
        h_activation: str = "relu",
        momentum: float = 0.99,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        l2_weight: float = 0.0001,
        kernel_initializer: str = "glorot_uniform",
        metrics: List[str] = None,
        preprocessor=None,
    ):

        if len(hidden_layers) < 1:
            raise ValueError(f"hidden_layers cannot be empty, got {hidden_layers = }")

        if not cat_cardinalities:
            raise ValueError(
                f"CatEmb_MLPClassifier assumes you have at least one categorical feature, got: {cat_cardinalities = }"
            )

        assert isinstance(output_shape, int) and output_shape > 0, (
            f"output_shape should be a positive integer, got {output_shape = }"
        )
        assert isinstance(input_shape, int) and input_shape > 0, (
            f"input_shape should be a positive integer, got {input_shape = }"
        )
        assert input_shape >= len(cat_cardinalities), (
            f"input_shape should be bigger or equal to the number of categorical features, got: {input_shape = }"
        )
        assert len(d_ratios) == len(cat_cardinalities), (
            "d_ratios and cat_cardinalities should be of the same length"
        )

        super().__init__(output_shape=output_shape, preprocessor=preprocessor)

        self.n_cont_features = input_shape - len(cat_cardinalities)
        self.cat_cardinalities = cat_cardinalities
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.h_activation = h_activation
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        self.kernel_initializer = kernel_initializer
        self.metrics = metrics or ["accuracy"]
        self.cat_embeddings = (
            CatEmbs(cat_cardinalities, d_ratios, name="categorical_embeddings")
            if cat_cardinalities
            else None
        )

        # Build model
        self.model = self.build_model()

    def add_hidden_layer(self, h, neurons, layer_num):
        h = Dense(
            neurons,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=L2(self.l2_weight),
            name=f"dense_{layer_num}",
        )(h)
        h = BatchNormalization(momentum=self.momentum, name=f"batch_norm_{layer_num}")(
            h
        )
        h = Activation(self.h_activation, name=f"activation_{layer_num}")(h)
        h = Dropout(rate=self.dropout_rate, name=f"dropout_{layer_num}")(h)
        return h

    def build_model(self) -> Model:
        x_cont_input = (
            Input(
                shape=(self.n_cont_features,),
                name="continuous_features",
                dtype=tf.float32,
            )
            if self.n_cont_features > 0
            else None
        )
        x_cat_input = Input(
            shape=(len(self.cat_cardinalities),),
            name="categorical_features",
            dtype=tf.int32,
        )
        inputs = [x_cat_input] if x_cont_input is None else [x_cont_input, x_cat_input]

        cat_embeddings = self.cat_embeddings(x_cat_input)

        if x_cont_input is None:
            h = cat_embeddings
        else:
            h = Concatenate(axis=1, name="concat_embeddings")(
                [x_cont_input, cat_embeddings]
            )

        for i, neurons in enumerate(self.hidden_layers):
            h = self.add_hidden_layer(h, neurons, i + 1)

        if self.output_shape > 1:
            outputs = Dense(
                self.output_shape, activation="softmax", name="output_layer"
            )(h)
            loss = "categorical_crossentropy"
        else:
            outputs = Dense(1, activation="sigmoid", name="output_layer")(h)
            loss = "binary_crossentropy"

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="CatEmb_MLP_Model")

        # Compile model
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=self.metrics,
        )

        return model


def CatEmb_MLP_Optimize_Parameters(
    X,
    y,
    input_shape,
    cat_cardinalities,
    output_shape,
    N_layers,
    const_params=None,
    fit_params=None,
    preprocessor=None,
):
    const_params = const_params or {}
    default_fit_params = {
        "epochs": 400,
        "batch_size": 64,
        "patience": 100,
        "monitor": "val_accuracy",
        "verbose": False,
        "custom_callback": True,
        "callback_verbose": False,
    }

    fit_params = fit_params or default_fit_params

    def objective(trial):
        model_params = {}

        if "hidden_layers" in const_params:
            model_params["hidden_layers"] = const_params["hidden_layers"]
        else:
            hidden_layers = np.ones(N_layers).astype(int)
            for layer in range(N_layers):
                hidden_layers[layer] = trial.suggest_int(
                    name=f"hidden_layer{layer}", low=16, high=128
                )
            model_params["hidden_layers"] = hidden_layers

        model_params["h_activation"] = (
            const_params["h_activation"]
            if "h_activation" in const_params
            else trial.suggest_categorical(
                name="h_activation", choices=["relu", "mish", "leaky_relu"]
            )
        )

        model_params["momentum"] = (
            const_params["momentum"]
            if "momentum" in const_params
            else trial.suggest_float(name="momentum", low=0.8, high=0.999)
        )

        model_params["l2_weight"] = (
            const_params["l2_weight"]
            if "l2_weight" in const_params
            else trial.suggest_float(name="l2_weight", low=1e-5, high=0.01, log=True)
        )

        model_params["dropout_rate"] = (
            const_params["dropout_rate"]
            if "dropout_rate" in const_params
            else trial.suggest_float(name="dropout_rate", low=0.0, high=0.5)
        )

        model_params["learning_rate"] = (
            const_params["learning_rate"]
            if "learning_rate" in const_params
            else trial.suggest_float(
                name="learning_rate", low=1e-4, high=1e-2, log=True
            )
        )

        if "d_ratios" in const_params:
            model_params["d_ratios"] = const_params["d_ratios"]
        else:
            d_ratios = np.ones(len(cat_cardinalities))
            for d in range(len(cat_cardinalities)):
                d_ratios[d] = trial.suggest_float(name=f"d{d}", low=0.1, high=2)
            model_params["d_ratios"] = d_ratios

        model = CatEmb_MLPClassifier(
            input_shape=input_shape,
            cat_cardinalities=cat_cardinalities,
            output_shape=output_shape,
            **model_params,
            preprocessor=preprocessor,
        )
        histories, preds, y_pred, train_acc, val_acc = model.Ensemble_Learning_CV(
            X,
            y,
            fit_params=fit_params,
            n_folds=10,
            n_repeats=1,
            voting_method="soft",
            val_results=True,
        )

        return val_acc.mean(axis=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)

    print(f"Best Accuracy Score: {study.best_value:.5f}")

    return study.best_params


# =============================================================================
# Patch model __init__ methods to capture constructor kwargs for parallel CV.
# Applied here so all subclasses are fully defined before patching.
# =============================================================================

for _cls in (MLPClassifier, TabResNetClassifier, CatEmb_MLPClassifier, FTTransformer):
    _cls.__init__ = _capture_init_kwargs(_cls.__init__)
