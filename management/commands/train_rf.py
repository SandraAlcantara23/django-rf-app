from django.core.management.base import BaseCommand, CommandError
from api.ml import save_model
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import os

class Command(BaseCommand):
    help = "Entrena un RandomForest desde un CSV y guarda el modelo (clasificación o regresión) con opciones de bajo consumo."

    def add_arguments(self, parser):
        parser.add_argument("--csv", required=True, help="Ruta al CSV")
        parser.add_argument("--target", required=True, help="Nombre de la columna objetivo")
        parser.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
        parser.add_argument("--test-size", type=float, default=0.2)
        parser.add_argument("--random-state", type=int, default=42)
        parser.add_argument("--n-estimators", type=int, default=50)      # menos árboles por defecto
        parser.add_argument("--max-depth", type=int, default=12)         # limita profundidad
        parser.add_argument("--max-samples", type=float, default=0.2,    # 20% del dataset por árbol
                            help="Fracción (0-1) de muestras por árbol (bootstrap).")
        parser.add_argument("--sample-rows", type=int, default=20000,    # entrena con primeras N filas
                            help="Número de filas a usar del CSV para entrenar (para aligerar).")
        parser.add_argument("--threads", type=int, default=1,            # 1 hilo = menos calor
                            help="n_jobs para el RandomForest.")

    def handle(self, *args, **opts):
        # Limitar hilos de BLAS para no saturar CPU
        os.environ.setdefault("OMP_NUM_THREADS", str(opts["threads"]))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(opts["threads"]))
        os.environ.setdefault("MKL_NUM_THREADS", str(opts["threads"]))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(opts["threads"]))

        csv = Path(opts["csv"])
        target = opts["target"]
        if not csv.exists():
            raise CommandError(f"No existe el CSV: {csv}")

        # Leer solo las primeras sample-rows filas para no reventar la RAM
        nrows = None if opts["sample-rows"] <= 0 else opts["sample-rows"]
        df = pd.read_csv(csv, nrows=nrows)

        if target not in df.columns:
            raise CommandError(f"'{target}' no está en columnas: {df.columns.tolist()}")

        # Convertir a float32 para ahorrar RAM
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")

        X = df.drop(columns=[target])
        y = df[target]

        task = opts["task"]
        if task == "auto":
            task = "regression" if np.issubdtype(y.dtype, np.number) else "classification"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=opts["test-size"], random_state=opts["random-state"]
        )

        if task == "regression":
            model = RandomForestRegressor(
                n_estimators=opts["n-estimators"],
                random_state=opts["random-state"],
                n_jobs=opts["threads"],
                max_depth=opts["max-depth"],
                max_samples=opts["max-samples"],
            )
        else:
            model = RandomForestClassifier(
                n_estimators=opts["n-estimators"],
                random_state=opts["random-state"],
                n_jobs=opts["threads"],
                max_depth=opts["max-depth"],
                max_samples=opts["max-samples"],
            )

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        # Guardar
        save_model(model, features=list(X.columns), target=target)

        self.stdout.write(self.style.SUCCESS(
            f"Modelo guardado. Task={task}. Score={score:.4f} "
            f"(R^2 si regresión, accuracy si clasificación). "
            f"Filas usadas: {len(df)}; n_estimators={opts['n-estimators']}; "
            f"max_depth={opts['max-depth']}; max_samples={opts['max-samples']}; threads={opts['threads']}"
        ))
