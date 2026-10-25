import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.preprocessing import StandardScaler
SEED = 2026
ROOT = Path(__file__).resolve().parent
initial_columns = ["x_1_0", "y_1_0", "x_2_0", "y_2_0", "x_3_0", "y_3_0"]
position_columns = ["x_1", "y_1", "x_2", "y_2", "x_3", "y_3"]
features_columns = ["t"] + initial_columns
target_columns = position_columns
features_task3 = ["t", "x_1_0", "x_2_0", "y_2_0"]  # best feature set from Task 3
# verificamos que o rsme do baseline = rsme do modelo com menos features,
# o que signfica que existiam efetivamente features redundantes.
reduced_features = ["t", "x_2_0", "x_3_0", "y_3_0"]
# label das novas features, para cada par de corpos, a distancia entre eles, seu
# inverso e o inverso do quadrado.
distance_features = ["r_12_0", "r_13_0", "r_23_0"]
# StandardScaler é utilizado para realizar a normalização
# Desta forma os valores deixam de estar expressos nas unidades originais e passam
# a indicar a sua distancia à media.
def train(X_train, Y_train, X_val, Y_val, selected_features):
    model = Pipeline([("scaler", StandardScaler()),
                      ("linear_regression", LinearRegression()),])
    model.fit(X_train[selected_features], Y_train)
    predictions = model.predict(X_val[selected_features])
    rmse = root_mean_squared_error(Y_val, predictions)
    return rmse, predictions, model
def load(task, sort=False):
    dt = pd.read_csv("C:\\Users\\vicen\\Documents\\X_train.csv")
    # cada trajetória ocupa 257 linhas no nosso dataset, isto cria uma nova coluna
    # no dataset
    dt["trajectory_id"] = dt["Id"] // 257
    if sort:
        dt = dt.sort_values(["trajectory_id", "t"])
    # target outputs, só queremos os valores iniciais
    dt[initial_columns] = (dt.groupby("trajectory_id")[position_columns]
                           .transform("first"))
    test = pd.read_csv("C:\\Users\\vicen\\Documents\\X_test.csv")
    names = {f"{c}0_{i}": f"{c}_{i}_0" for c in "xy" for i in (1, 2, 3)}
    return dt, test.rename(columns=names)
# split by trajectory: 70% train / 15% validation / 15% test and defines a seed
def split_trajectories(dt, seed=SEED):
    #gets each unique id
    ids = dt["trajectory_id"].unique()
    # 30% para validation e test, 70% para treino
    #splits the data into train and the rest
    train_ids, rest_ids = train_test_split(ids, test_size=0.30, random_state=seed)
    # 15% para cada
    #splits the rest into validation and test
    val_ids, test_ids = train_test_split(rest_ids, test_size=0.50, random_state=seed)
    # Construimos os dataframes depois de criar as colunas iniciais.
    #returns the train, validation and test sets
    return (dt[dt["trajectory_id"].isin(train_ids)],
            dt[dt["trajectory_id"].isin(val_ids)],
            dt[dt["trajectory_id"].isin(test_ids)])
def drop_collisions(dt):
    # check for same position in a row in the same trajectory
    diffs = dt.groupby("trajectory_id")[position_columns].diff()
    #checks if the difference is zero
    colision = (diffs.abs() == 0).all(axis="columns")
    #removes the colisions
    return dt[~colision]
#updated to fit tasks 2 and 3
def make_dev_sets(train_data, val_data, features=features_columns,
                  targets=target_columns, n_train=2000, n_val=1000, seed=SEED):
    #samples the training data
    tr = train_data.sample(n=min(n_train, len(train_data)), random_state=seed)
    #samples the validation data
    va = val_data.sample(n=min(n_val, len(val_data)), random_state=seed)
    #returns 4 subsets (full features and targets)
    return tr[features], tr[targets], va[features], va[targets]
def validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None,
                             degrees=range(1, 10)):
    #sets the infinite, best model and best rmse variables
    best_model, best_rmse = None, np.inf
    for d in degrees:
        #used if we need RidgeCV
        reg = clone(regressor) if regressor is not None else LinearRegression()
        model = Pipeline([("scaler", StandardScaler()),
                          ("polynomialFeatures", PolynomialFeatures(degree=d)),
                          ("regressor", reg)])
        #trains the model
        model.fit(X_train, y_train)
        #gets the number of features after polynomial transformation
        n_feat = model.named_steps["polynomialFeatures"].n_output_features_
        # compares the predictions with the true y_val and returns the typical size
        # of the error, in the same units as the target.
        rmse = root_mean_squared_error(y_val, model.predict(X_val))
        print(f"degree={d}  n_features={n_feat}  val RMSE={rmse:.4f}")
        # Checks whether this degree has a lower validation error than the best so
        # far. On the first degree, best_rmse is np.inf, so it always wins.
        if rmse < best_rmse:
            best_model, best_rmse = model, rmse
    return best_model, best_rmse
#overfiting risk because rmse goes up from n = 7
def best_degree_of(model):
    return model.named_steps["polynomialFeatures"].degree
# ColumnTransformer -> decide quais das colunas é que vão passar pelas
# trasnformações e junta os resultados numa matriz
# FunctionTransformer -> responsavél por calcular as novas features, neste caso
# r_12, r_13, r_23
# Este processo afeta essencialmente a pipeline que antes computava somente as
# PolynomialFeatures e agora
# vai adicionar estas transformações à pipeline
# Função para calcular as distancias, FunctionTransformer vai chamar esta função.
def calculate_distances(X):
    distance_12 = np.sqrt((X["x_1_0"] - X["x_2_0"])**2 + (X["y_1_0"] - X["y_2_0"])**2)
    distance_13 = np.sqrt((X["x_1_0"] - X["x_3_0"])**2 + (X["y_1_0"] - X["y_3_0"])**2)
    distance_23 = np.sqrt((X["x_2_0"] - X["x_3_0"])**2 + (X["y_2_0"] - X["y_3_0"])**2)
    return np.column_stack([distance_12, distance_13, distance_23])
def validate_augmented_poly_regression(X_train, y_train, X_val, y_val,
                                       regressor=None, degrees=range(1, 9)):
    best_model, best_rmse = None, np.inf
    for degree in degrees:
        #passo da pipeline para a polinomial_regression.
        polynomial_transformer = Pipeline([
            ("standardscaler", StandardScaler()),
            ("polynomialfeatures", PolynomialFeatures(degree=degree))])
        # pipeline para aplicar o functionTransformer para obtermos as distancias.
        # aplica a normalização depois de calcular as distancias
        distance_transformer = Pipeline([
            ("functiontransformer", FunctionTransformer(calculate_distances)),
            ("standardscaler", StandardScaler())])
        #permite diferenciar as colunas que afeta.
        preprocessor = ColumnTransformer([
            ("polynomial", polynomial_transformer, reduced_features,),
            ("distances", distance_transformer, initial_columns,)])
        reg = clone(regressor) if regressor is not None else LinearRegression()
        #são criados dois ramos, e juntos numa só pipeline
        model = Pipeline([("columntransformer", preprocessor),
                          ("regressor", reg)])
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, predictions)
        polynomial_pipeline = (model.named_steps["columntransformer"]
                                    .named_transformers_["polynomial"])
        polynomial_count = (polynomial_pipeline.named_steps["polynomialfeatures"]
                                               .n_output_features_)
        total_features = polynomial_count + len(distance_features)
        print(f"degree={degree} "f"n_features={total_features} "
              f"val RMSE={rmse:.6f}")
        if rmse < best_rmse:
            best_model = model
            best_rmse = rmse
    return best_model, best_rmse
def predict_in_batches(model, X, batch_size=2048):
    if batch_size is None:
        return model.predict(X)
    parts = []
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start:start + batch_size]
        parts.append(model.predict(batch))
    return np.vstack(parts)
#Submission
def save_submission(model, test, task, batch_size=2048, cols=features_columns):
    prediction = predict_in_batches(model, test[cols], batch_size)
    submission = pd.DataFrame(prediction, columns=position_columns)
    submission.insert(0, "Id", test["Id"].to_numpy())
    assert submission.shape == (len(test), 7)
    assert np.isfinite(submission[position_columns].to_numpy()).all()
    submission.to_csv(ROOT / f"task{task}.csv", index=False)
def task1():
    print("Task 1: Baseline model:")  #task 1
    dt, test = load(1)
    train_data, validation_data, test_data = split_trajectories(dt)
    baseline_rmse, _, baseline_model = train(
        train_data, train_data[target_columns], validation_data,
        validation_data[target_columns], features_columns)
    print(f"Baseline RMSE: {baseline_rmse:.6f}")
    y_predicted = baseline_model.predict(test_data[features_columns])
    rmse_test = root_mean_squared_error(test_data[target_columns], y_predicted)
    print(f"RMSE on test set: {rmse_test:.6f}")
    save_submission(baseline_model, test, 1, None)
def task2():
    print("Task 2: Polynomial regression:")  #Task 2
    dt, test = load(2)
    train_data, validation_data, _ = split_trajectories(drop_collisions(dt))
    x_train, y_train, x_validation, y_validation = make_dev_sets(
        train_data, validation_data)
    model, best_rmse = validate_poly_regression(
        x_train, y_train, x_validation, y_validation, regressor=None,
        degrees=range(1, 10))
    print("best degree:", best_degree_of(model), " RMSE:", best_rmse)
    #Tentamos com o ridge
    ridge_model, ridge_rmse = validate_poly_regression(
        x_train, y_train, x_validation, y_validation, regressor=RidgeCV(),
        degrees=range(1, 10))
    print("ridge best degree:", best_degree_of(ridge_model), " RMSE:", ridge_rmse)
    save_submission(ridge_model if ridge_rmse < best_rmse else model, test, 2)
def task3():
    print("Task 3: Polynomial regression with feature selection:")  #task 3
    dt, test = load(3)
    train_data, validation_data, _ = split_trajectories(drop_collisions(dt))
    x_train, y_train, x_validation, y_validation = make_dev_sets(
        train_data, validation_data)
    augmented_model, augmented_rmse = validate_augmented_poly_regression(
        x_train, y_train, x_validation, y_validation,
        regressor=RidgeCV(), degrees=range(1, 9))
    print(f"Best augmented validation RMSE: {augmented_rmse:.6f}")
    save_submission(augmented_model, test, 3)
def task4():
    dt, test = load(4, sort=True)
    # initial positions + remove collision rows (as in Task 2)
    diffs = dt.groupby("trajectory_id")[position_columns].diff()
    dt = dt[~(diffs.abs() < 1e-9).all(axis="columns")]
    # split by trajectory: 70/15/15
    ids = dt["trajectory_id"].unique()
    train_ids, rest_ids = train_test_split(ids, test_size=0.30, random_state=SEED)
    val_ids, _ = train_test_split(rest_ids, test_size=0.50, random_state=SEED)
    train_data = dt[dt["trajectory_id"].isin(train_ids)]
    val_data = dt[dt["trajectory_id"].isin(val_ids)].sample(10000, random_state=SEED)
    y_train, y_val = train_data[position_columns], val_data[position_columns]
    # kNN: pick k and feature set using validation RMSE
    best = (np.inf, None, None, None)
    for cols in [features_columns, features_task3]:
        for k in range(1, 15):
            knn = Pipeline([("scaler", StandardScaler()),
                            ("knn", KNeighborsRegressor(n_neighbors=k, n_jobs=-1))])
            knn.fit(train_data[cols], y_train)
            rmse = root_mean_squared_error(y_val, knn.predict(val_data[cols]))
            print(len(cols), "features, k =", k, "RMSE =", round(rmse, 4))
            if rmse < best[0]:
                best = (rmse, k, cols, knn)
    _, best_k, knn_cols, knn_best = best
    print("best k:", best_k, "features:", knn_cols)
    save_submission(knn_best, test, 4, 100000, knn_cols)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    selected_task = parser.parse_args().task
    random.seed(SEED)
    np.random.seed(SEED)
    {1: task1, 2: task2, 3: task3, 4: task4}[selected_task]()