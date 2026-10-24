import numpy as np                                                                      #linear algebra
import pandas as pd                                                                     #data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
def plot_y_yhat(y_test,y_pred, plot_title = "plot"):                                    #Plots
    labels = ['x_1','y_1','x_2','y_2','x_3','y_3']
    MAX = 500
    if len(y_test) > MAX:
        idx = np.random.choice(len(y_test),MAX, replace=False)
    else:
        idx = np.arange(len(y_test))
    plt.figure(figsize=(10,10))
    for i in range(6):
        x0 = np.min(y_test[idx,i])
        x1 = np.max(y_test[idx,i])
        plt.subplot(3,2,i+1)
        plt.scatter(y_test[idx,i],y_pred[idx,i])
        plt.xlabel('True '+labels[i])
        plt.ylabel('Predicted '+labels[i])
        plt.plot([x0,x1],[x0,x1],color='red')
        plt.axis('square')
    plt.savefig(plot_title+'.pdf')
    plt.show()
def train(X_train, Y_train, X_val, Y_val, selected_features):                           # StandardScaler é utilizado para realizar a normalização
    model = Pipeline([("scaler",StandardScaler()),
                      ("linear_regression", LinearRegression()),])                      # Desta forma os valores deixam de estar expressos nas unidades originais e passam a indicar a sua distancia à media.
    model.fit(X_train[selected_features], Y_train)
    predictions = model.predict(X_val[selected_features])
    rmse = root_mean_squared_error(Y_val, predictions)
    return rmse, predictions, model
print("Task 1: Baseline model:")                                                        #task 1
dt = pd.read_csv("C:\\Users\\vicen\\Documents\\X_train.csv")
dt["trajectory_id"] = dt["Id"] // 257                                                   #cada trajetória ocupa 257 linhas no nosso dataset, isto cria uma nova coluna no dataset
trajectory_ids = dt["trajectory_id"].unique()                                           #0..4999
train_ids,remaining_ids=train_test_split(trajectory_ids,test_size=0.3,random_state=2026)#30% para validation e test, 70% para treino
validation_ids,test_ids=train_test_split(remaining_ids,test_size=0.5,random_state=2026) #15% para cada
initial_columns = ["x_1_0", "y_1_0","x_2_0", "y_2_0","x_3_0", "y_3_0"]
position_columns  = ["x_1", "y_1","x_2", "y_2","x_3", "y_3"]                                                   
dt[initial_columns] = (dt.groupby("trajectory_id")[position_columns].transform("first"))#target outputs, só queremos os valores iniciais
train_data = dt[dt["trajectory_id"].isin(train_ids)]                                    #Construimos os dataframes depois de criar as colunas iniciais.
validation_data = dt[dt["trajectory_id"].isin(validation_ids)]
test_data = dt[dt["trajectory_id"].isin(test_ids)]
features_columns = ["t"] + initial_columns
target_columns = position_columns 
x_train = train_data[features_columns]
x_validation = validation_data[features_columns]
x_test = test_data[features_columns]
y_train = train_data[target_columns]
y_validation = validation_data[target_columns]
y_test = test_data[target_columns]
baseline_rmse, baseline_predictions, baseline_model = train(x_train, y_train,
                                    x_validation, y_validation, features_columns,)
print(f"Baseline RMSE: {baseline_rmse:.6f}")
#temp# plot_y_yhat(y_validation.to_numpy(),baseline_predictions,plot_title="Baseline",)
y_predicted = baseline_model.predict(x_test)
rmse_test = root_mean_squared_error(y_test,y_predicted)
print(f"RMSE on test set: {rmse_test:.6f}")
print("Task 2: Polynomial regression:")                                                 #Task 2
diffs = dt.groupby("trajectory_id")[position_columns].diff()                            #check for same position in a row in the same trajectory
colision = (diffs.abs() == 0).all(axis="columns")                                       #checks if the difference is zero
dt = dt[~colision]                                                                      #removes the colisions
def split_trajectories(seed=2026):                                                      # split by trajectory: 70% train / 15% validation / 15% test and defines a seed
    ids = dt["trajectory_id"].unique()                                                  #gets each unique id
    train_ids, rest_ids = train_test_split(ids, test_size=0.30, random_state=seed)      #splits the data into train and the rest
    val_ids, test_ids = train_test_split(rest_ids, test_size=0.50, random_state=seed)   #splits the rest into validation and test
    return (dt[dt["trajectory_id"].isin(train_ids)],
        dt[dt["trajectory_id"].isin(val_ids)], dt[dt["trajectory_id"].isin(test_ids)])  #returns the train, validation and test sets
def make_dev_sets(train_data,val_data,features=features_columns,targets=target_columns, #updated to fit tasks 2 and 3
    reduced_features=None,n_train=2000,n_val=1000,seed=2026):
    tr = train_data.sample(n=min(n_train, len(train_data)), random_state=seed)          #samples the training data
    va = val_data.sample(n=min(n_val, len(val_data)), random_state=seed)                #samples the validation data
    if reduced_features is not None:
        return (tr[features],tr[reduced_features],tr[targets],
                va[features],va[reduced_features],va[targets])                          #returns 6 subsets if reduced features are provided
    return tr[features], tr[targets], va[features], va[targets]                         #returns 4 subsets (full features and targets)                # returns 4 subsets (full features and targets)
def validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None,
                             degrees=range(1, 15), max_features=None):
    best_model, best_rmse = None, np.inf                                                #sets the infinite, best model and best rmse variables
    for d in degrees:
        expected = PolynomialFeatures(degree=d).fit(X_train.iloc[:1]).n_output_features_#checks the number of columns PolynomialFeatures will have (counted on 1 row, before fitting the model)
        if max_features is not None and expected > max_features:                        #checks if the expected number of features exceeds the set maximum allowed
            print(f"degree={d}  n_features={expected} -> skipped (> max_features)")
            continue
        reg = clone(regressor) if regressor is not None else LinearRegression()         #used if we need RidgeCV
        model = Pipeline([("scaler", StandardScaler()),
            ("polynomialFeatures", PolynomialFeatures(degree=d)),("regressor", reg)])   
        model.fit(X_train, y_train)                                                     #trains the model
        n_feat = model.named_steps["polynomialFeatures"].n_output_features_             #gets the number of features after polynomial transformation     
        rmse = root_mean_squared_error(y_val, model.predict(X_val))                     #compares the predictions with the true y_val and returns the typical size of the error, in the same units as the target.
        print(f"degree={d}  n_features={n_feat}  val RMSE={rmse:.4f}")
        if rmse < best_rmse:                                                            #Checks whether this degree has a lower validation error than the best so far. On the first degree, best_rmse is np.inf, so it always wins. 
            best_model, best_rmse = model, rmse                                         
    return best_model, best_rmse                                                                                    
def best_degree_of(model):                                                              #overfiting risk because rmse goes up from n = 7
    return model.named_steps["polynomialFeatures"].degree
train_data, validation_data, test_data = split_trajectories(seed=2026)                  #Splits the data into train, validation, and test sets
x_train, y_train, x_validation, y_validation = make_dev_sets(train_data,
                                                             validation_data, seed=2026)#Creates development sets
model, best_rmse = validate_poly_regression(x_train, y_train, x_validation,
                y_validation,regressor=None, degrees=range(1, 10), max_features=None)    
print("best degree:", best_degree_of(model), " RMSE:", best_rmse)
ridge_model, ridge_rmse = validate_poly_regression(x_train, y_train, x_validation,      #Tentamos com o ridge
             y_validation, regressor=RidgeCV(), degrees=range(1, 10), max_features=None)
print("ridge best degree:", best_degree_of(ridge_model), " RMSE:", ridge_rmse)
y_test = y_validation.to_numpy()                                                        #Use the validation set to compare true vs predicted positions
y_pred = model.predict(x_validation)                                                    #Predictions from the best polynomial regression model
y_pred_ridge = ridge_model.predict(x_validation)                                        #Predictions from the best Ridge regression model
#temp#plot_y_yhat(y_test, y_pred, plot_title="polynomial_regression")                 
#temp#plot_y_yhat(y_test, y_pred_ridge, plot_title="ridge_regression")
print("Task 3: Polynomial regression with feature selection:")                          #task 3, 3.1
analysis_columns = features_columns + target_columns                                    #Só temos em conta as variaveis relevantes, ou seja todas exceto as das velocidades e ID. 
analysis_sample = train_data[analysis_columns].sample(200, random_state=2026)           #Amostra para análise
sns.pairplot(analysis_sample, kind="hist")                                              #Matriz de dispersão
plt.show()                                                                              #Mostra a matriz de dispersão
correlations = analysis_sample.corr()                                                   #Matriz de correlação
plt.figure(figsize=(16, 12))                                                            #temos de ter em conta 3 tipos de relações possiveis e o que elas demonstram:
#feature-feature, para decidir eventuais remoções
#feature- target, util para verificarmos a capacidade preditiva de uma feature.
#target-target, util para uma analise complementar
sns.heatmap(correlations,annot=True,fmt=".2f", annot_kws={"size": 7}, cmap="coolwarm",  #olhando para os valores das correlações validamos a existência de algumas associações lineares
            vmin=-1, vmax=1)                                                            #nomeadamente x_3_0 - x_2_0 ; y_3_0-y_2_0; y_1_0; x_1_0      
plt.show()                                                                              
upper_triangle=correlations.where(np.triu(np.ones(correlations.shape),k=1).astype(bool))#makes a table
ordered_correlations = (upper_triangle.stack().dropna().rename("correlation")
                     .reset_index()
                     .rename(columns={"level_0": "feature_1", "level_1": "feature_2"}))
ordered_correlations["absolute_correlation"]=(ordered_correlations["correlation"].abs())
ordered_correlations=ordered_correlations.sort_values(
                                                "absolute_correlation",ascending=False)
print(ordered_correlations.to_string(index=False))
baseline_rmse_t3, _, _ = train(x_train, y_train, x_validation, y_validation,            #Aqui é onde vamos testar a remoção de cada variavél, e avaliar o rmse
                               features_columns)                                        #numa fase inicial removemos uma a uma,numa segunda fase, removemos mais do que uma.
experiments = [["x_1_0"],["y_1_0"],["x_2_0"],["y_2_0"],["x_3_0"],["y_3_0"],             #seleção de variáveis, removendo uma a uma
               ["x_1_0", "y_1_0"],["x_1_0", "y_1_0", "y_2_0"]]
results = []                                                                            #Lista para armazenar resultados
for removed_features in experiments:
     selected_features = [feature for feature in features_columns
                          if feature not in removed_features]
     rmse, _, _ = train(x_train,y_train,x_validation,y_validation,selected_features)    #get the rmse with those features
     results.append({"removed_features": removed_features,                              #store results
                     "selected_features": selected_features,"rmse": rmse,
                     "distance_to_baseline": rmse - baseline_rmse_t3})
results = pd.DataFrame(results).sort_values("rmse")                                     #makes a table
print(results.to_string(index=False))
reduced_features = ["t", "x_2_0", "x_3_0", "y_3_0"]
reduced_rmse, reduced_predictions, reduced_model = train(x_train,y_train,
                                        x_validation,y_validation,reduced_features)
print(f"Reduced features RMSE: {reduced_rmse:.6f}")                                     #verificamos que o rsme do baseline = rsme do modelo com menos features,
print(f"Selected features: {reduced_features}")                                         #o que signfica que existiam efetivamente features redundantes.
print(f"Baseline RMSE: {baseline_rmse_t3:.6f}")
#temp#plot_y_yhat(y_validation.to_numpy(),reduced_predictions,plot_title="Reduced features")
#3.2-Isto vem da task2, ajustando com as reduced_features e com as normais.
x_tr_poly_all,x_tr_poly_reduced,y_tr_poly,x_va_poly_all,x_va_poly_reduced,y_va_poly = ( #obter os dev sets para aplicar validate_poly_regression().
  make_dev_sets(train_data,validation_data,reduced_features=reduced_features,seed=2026))
ridge_model_reduced, ridge_rmse_reduced = validate_poly_regression(x_tr_poly_reduced,
        y_tr_poly, x_va_poly_reduced,y_va_poly,regressor=RidgeCV(),degrees=range(1, 10)) 
ridge_degree_reduced = ridge_model_reduced.named_steps["polynomialFeatures"].degree
ridge_predictions_reduced = ridge_model_reduced.predict(x_va_poly_reduced)
print(f"Best Ridge degree reduced:{ridge_degree_reduced}")
print(f"Best Ridge validation RMSE reduced: {ridge_rmse_reduced:.6f}")
#temp#plot_y_yhat(y_va_poly.to_numpy(),ridge_predictions_reduced,plot_title="PolyidgeReduced")
def predict_in_batches(model, X, batch_size=512):
    parts = []
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start:start + batch_size]
        parts.append(model.predict(batch))
    return np.vstack(parts)
x_test_local = test_data[reduced_features]
y_test_local = test_data[target_columns]
pred_test_local = predict_in_batches(ridge_model_reduced, x_test_local)
test_rmse = root_mean_squared_error(y_test_local, pred_test_local)
print(f"Grau escolhido: {ridge_degree_reduced}")
print(f"RMSE no teste local: {test_rmse:.4f}")
plot_y_yhat(y_test_local.to_numpy(),pred_test_local, plot_title="Task 3.2-Local test")
#task 3.3: vamos criar novas features, uma delas será no caso a distancia entre dois corpos, a segunda o inverso dessa distancia e o inverso dessa disancia ao quadrado
def add_engineered_features(data):                                                      #função responsavél por adicionar as colunas das novas features ao respetivo df, recebe as starting positions dos corpos e devolve uma copy com 9 new columns:
     result = data.copy()                                                               #para cada par de corpos, a distância entre eles, seu inverso e o inverso do quadrado. 
     pairs = [(1, 2), (1, 3), (2, 3)]
     epsilon = 1e-8                                                                     #para evitar divisão por zero, e não dar infinito.
     for body_1, body_2 in pairs:
         difference_x = (result[f"x_{body_1}_0"] - result[f"x_{body_2}_0"])
         difference_y = (result[f"y_{body_1}_0"] - result[f"y_{body_2}_0"])
         distance = np.sqrt(difference_x**2 + difference_y**2)                          #teorema de petagoras, para calcular a distancia entre dois corpos, a partir das suas coordenadas iniciais.
         distance_name = f"r_{body_1}{body_2}_0"                                        #da o nome à coluna nova.
         result[distance_name] = distance                                               #adiciona a coluna da distancia
         result[f"inv_{distance_name}"] = 1 / (distance + epsilon)                      #adiciona a coluna do inverso da distancia
         result[f"inv2_{distance_name}"] = 1 / (distance**2 + epsilon)                  #adiciona a coluna do inverso do quadrado da distancia- shape of the gravitational force law.
     return result                                                                      #devolve a copia com as novas colunas adicionadas
train_data = add_engineered_features(train_data)                                        #adiciona as novas 9 colunas ao df de treino, validação e teste.
validation_data = add_engineered_features(validation_data)
test_data = add_engineered_features(test_data)
distance_features = ["r_12_0", "r_13_0", "r_23_0"]                                      # label das novas features, para cada par de corpos, a distancia entre eles, seu inverso e o inverso do quadrado.
inverse_distance_features = ["inv_r_12_0", "inv_r_13_0", "inv_r_23_0"]                  # 1/r
inverse_squared_features = ["inv2_r_12_0", "inv2_r_13_0", "inv2_r_23_0"]                # 1/r^2
feature_engineering_experiments = {"Reduced baseline": reduced_features,                #reduced_features = ["t", "x_2_0", "x_3_0", "y_3_0"]
     "Distances": reduced_features + distance_features,                                 #Distances = ["t", "x_2_0", "x_3_0", "y_3_0"] + ["r_12_0", "r_13_0", "r_23_0"]
     "Inverse distances": reduced_features + inverse_distance_features,                 #Inverse distances = ["t", "x_2_0", "x_3_0", "y_3_0"] + ["inv_r_12_0", "inv_r_13_0", "inv_r_23_0"]
     "Inverse squared distances": reduced_features + inverse_squared_features}          #Inverse squared distances = ["t", "x_2_0", "x_3_0", "y_3_0"] + ["inv2_r_12_0", "inv2_r_13_0", "inv2_r_23_0"]
feature_engineering_results = []                                                        #Lista para armazenar os resultados das experimentss de engenharia de features
for experiment_name, experiment_features in feature_engineering_experiments.items():    
     rmse, predictions, model = train(train_data,y_train,validation_data,
                                      y_validation,experiment_features)
     feature_engineering_results.append({
         "experiment": experiment_name,"features": experiment_features,
         "rmse": rmse,"distance_to_reduced_baseline": rmse - reduced_rmse})
     plot_y_yhat(y_validation.to_numpy(),predictions,
                 plot_title=f"Task 3.3 - {experiment_name}")
feature_engineering_results=pd.DataFrame(feature_engineering_results).sort_values("rmse")
print(feature_engineering_results.to_string(index=False))
#Task 3.4
#ColumnTransformer -> decide quais das colunas é que vão passar pelas trasnformações e junta os resultados numa matriz
#FunctionTransformer -> responsavél por calcular as novas features, neste caso r_12, r_13, r_23
#Este processo afeta essencialmente a pipeline que antes computava somente as PolynomialFeatures e agora
#vai adicionar estas transformações à pipeline
#Função para calcular as distancias, FunctionTransformer vai chamar esta função.
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
def calculate_distances(X):
     distance_12 = np.sqrt((X["x_1_0"] - X["x_2_0"])**2 + (X["y_1_0"] - X["y_2_0"])**2)
     distance_13 = np.sqrt((X["x_1_0"] - X["x_3_0"])**2 + (X["y_1_0"] - X["y_3_0"])**2)
     distance_23 = np.sqrt((X["x_2_0"] - X["x_3_0"])**2 + (X["y_2_0"] - X["y_3_0"])**2)
     return np.column_stack([distance_12, distance_13, distance_23])
def validate_augmented_poly_regression(X_train, y_train, X_val, y_val,
                                        regressor=None, degrees=range(1, 9)):
    best_model, best_rmse = None, np.inf
    for degree in degrees:
        polynomial_transformer = make_pipeline(StandardScaler(),                        #passo da pipeline para a polinomial_regression.
                                               PolynomialFeatures(degree=degree))           
        distance_transformer = make_pipeline(FunctionTransformer(calculate_distances),  #pipeline para aplicar o functionTransformer para obtermos as distancias.
                                              StandardScaler())                         #aplica a normalização depois de calcular as distancias                                                                           
        preprocessor = ColumnTransformer([                                              #permite diferenciar as colunas que afeta.         
            ("polynomial", polynomial_transformer, reduced_features,),
            ("distances", distance_transformer, initial_columns,)])
        reg = clone(regressor) if regressor is not None else LinearRegression()
        model = make_pipeline(preprocessor, reg)                                        #são criados dois ramos, e juntos numa só pipeline
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, predictions)
        polynomial_pipeline = (model.named_steps["columntransformer"]
                                               .named_transformers_["polynomial"])
        polynomial_count = (polynomial_pipeline.named_steps["polynomialfeatures"]
                                               .n_output_features_)
        total_features = polynomial_count + len(distance_features)
        print(f"degree={degree} "f"n_features={total_features} "f"val RMSE={rmse:.6f}")
        if rmse < best_rmse:
            best_model = model
            best_rmse = rmse
    return best_model, best_rmse
augmented_model, augmented_rmse = validate_augmented_poly_regression(
    x_train_poly_all, y_train_poly, x_validation_poly_all, y_validation_poly,
    regressor=RidgeCV(), degrees=range(1, 9))
augmented_predictions = augmented_model.predict(x_validation_poly_all)
augmented_polynomial_pipeline = (augmented_model.named_steps["columntransformer"]
                                                 .named_transformers_["polynomial"])
augmented_degree = (augmented_polynomial_pipeline.named_steps["polynomialfeatures"]
                                                  .degree)
print(f"Best augmented degree: {augmented_degree}")
print(f"Best augmented validation RMSE: {augmented_rmse:.6f}")
plot_y_yhat(y_validation_poly.to_numpy(),augmented_predictions,
                       plot_title="Task 3.4 - Polynomial Ridge with distances")
print("\nModel comparison")
print(f"Task 2 - all features: {ridge_rmse_all:.6f}")
print(f"Task 3.2 - reduced features: {ridge_rmse_reduced:.6f}")
print(f"Task 3.4 - augmented features: {augmented_rmse:.6f}")
#Submission
X_kaggle = pd.read_csv("/kaggle/input/competitions/"
"task-3-machine-learning-nova-2026-the-three-body-problem/"
"mlNOVA/mlNOVA/X_test.csv")
X_kaggle_features = X_kaggle.rename(columns={"x0_1": "x_1_0", "y0_1": "y_1_0",
"x0_2": "x_2_0", "y0_2": "y_2_0","x0_3": "x_3_0", "y0_3": "y_3_0",})[features_columns]
prediction = predict_in_batches(augmented_model, X_kaggle_features)
submission_augmented = pd.DataFrame(prediction, columns=position_columns)
submission_augmented.insert(0, "Id", X_kaggle["Id"].to_numpy())
assert submission_augmented.shape == (1_041_621, 7)
assert np.isfinite(submission_augmented[position_columns].to_numpy()).all()
submission_augmented.to_csv("/kaggle/working/task3.csv", index=False)
