# %%
# %%
from quapy.method.aggregative import KDEyML, ExpectationMaximizationQuantifier
from quapy.data import LabelledCollection
from mlquantify.base_aggregative import AggregationMixin
from mlquantify.adjust_counting import *
from mlquantify.likelihood import *
from mlquantify.mixture import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils_qcal.extract_estimates import *
from utils_qcal.protocol import *
from utils_qcal.median_estimates import *
from calibrator.QCal_knn import *
from calibrator.QCal_rf import *
from calibrator.QCal_lr import *
from mlq.mlq import *
import warnings
warnings.filterwarnings('ignore')

# %%
#lista contendo os nomes dos datasets binarios
name_data = ['1460_banana','1462_banknote-authentication','1466_cardiotocography','1475_first-order-theorem-proving',
 '1479_hill-valley','1485_madelon','1489_phoneme','1494_qsar-biodeg','1496_ringnorm','1497_wall-robot-navigation',
 '1504_steel-plates-fault','1507_twonorm','1526_wall-robot-navigation','1535_volcanoes-b5','1538_volcanoes-d1',
 '1539_volcanoes-d2','1540_volcanoes-d3','1541_volcanoes-d4','1566_hill-valley','182_satimage',
 '28_optdigits','294_satellite_image','30_page-blocks','312_scene','375_JapaneseVowels','40733_yeast',
 '44_spambase','60_waveform-5000','679_rmftsa_sleepdata',
 'abalone','AedesQuinx','AedesSex','ailerons','anuranCalls','ArabicDigit','balance.1','balance.3',
 'balloon','bank8FM','BNG','BNG_breast-w','breast-cancer','churn','clean2','click-prediction',
 'cmc.1','cmc.2','cmc.3','cmc','covtype_reduced','cpu_act','ctg.1','ctg.2','ctg.3','default_credit_card_clients',
 'delta_ailerons','eeg','electricity-normalized','elevators','file22f0e4a679','file79b563a1a18',
 'file7b5365fa741c','fried','fri_c2_1000_10','german','gina_agnostic','houses','HTRU','jm1-processed',
 'kdd_JapaneseVowels','kin8nm','letter-recognition','magic','MagicTelescope','mammographic',
 'mozilla','namao','occupancy','page-blocks','pendigits','phoneme','php50jXam','phpGUrE90','phphQEck0','phplE7q6h',
 'phpMD2hR6','phpmPOD5A','phpWfYmlu','phpwRjVjk','phpxijhaP','pol','pollen','puma32H','puma8NH','quake','space_ga','spambase',
 'sylva_prior','tictactoe','transfusion','visualizing_soil','wind','wine-q-red','wine-q-white','winetype','yeast']

# %%
def run_experiment(dataset):
    # Quantifiers for the experimental evaluation
    models_quantifiers = {
        'CC(rf)': CC,
        'ACC(rf)': AC,
        'PCC(rf)': PCC,
        'PACC(rf)': PAC,
        'EMQ(rf)': EMQ,
        'FM(rf)': FM,
        'DyS(rf)': DyS,
        'MS(rf)': MS,
        'MS(bcts(rf))': MS,
        'HDy(rf)': HDy,
        'KDEyML(rf)': KDEyML,
        'MLQ(rf)': MLQ_rf,
        'QuantCal(rf,3)': QCal_rf_3,
        'QuantCal(rf,5)': QCal_rf_5,
        'QuantCal(rf,10)': QCal_rf_10,
        'QuantCal(lr,3)': QCal_lr_3,
        'QuantCal(lr,5)': QCal_lr_5,
        'QuantCal(lr,10)': QCal_lr_10,
        'QuantCal(knn,3)': QCal_knn_3,
        'QuantCal(knn,5)': QCal_knn_5,
        'QuantCal(knn,10)': QCal_knn_10,
    }

    # Names of the CUSTOM quantifiers (own classes).
    # They only accept the `learner` argument in __init__ (no `classifier`
    # nor `fit_classifier`), unlike the mlquantify quantifiers.
    custom_quantifiers = {
        name for name in models_quantifiers
        if name.startswith('QuantCal')
    }

    # Single output folder
    base_results_dir = 'results/calibration_experiment'

    # Loading the dataset
    with open(f'datasets/{dataset}.csv', 'r') as file:
        data = pd.read_csv(file)
        X = data.drop(columns=['target']).values
        y = data['target'].values

    for n in range(30):
        print(f'Starting experiment for dataset: {dataset} - split {n}')
        # Creating the directory to save the results (same path used in to_csv)
        output_path = f'{base_results_dir}/split_{n}/'
        os.makedirs(output_path, exist_ok=True)

        # Train/test split
        # random_state=n ensures reproducibility while keeping 30 distinct splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=n
        )

        # Generating the test batches
        batches_test = APP_protocol_mlquantify(X_test, y_test)

        # Training the base classifier (calibrated and uncalibrated)
        model_classifier_rf_bcts = pre_treined_model(
            X_train, y_train,
            clf=RandomForestClassifier(random_state=42, n_jobs=1),
            calibration=True
        )
        model_classifier_rf = pre_treined_model(
            X_train, y_train,
            clf=RandomForestClassifier(random_state=42, n_jobs=1)
        )

        # Training the quantification methods
        trained_models = {}
        for name, quantifier in models_quantifiers.items():
            try:
                if name in custom_quantifiers:
                    # Custom quantifiers (QuantCal): only the `learner` argument.
                    model = quantifier(learner=model_classifier_rf)

                elif name.startswith('KDEyML'):
                    # KDEyML is from another library: the argument is `classifier`.
                    # Applies to BOTH variants -- KDEyML(rf) and KDEyML(bcts(rf)).
                    if 'bcts' in name:
                        model = quantifier(classifier=model_classifier_rf_bcts, fit_classifier=False)
                    else:
                        model = quantifier(classifier=model_classifier_rf, fit_classifier=False)

                else:
                    # Remaining mlquantify quantifiers: use `learner`.
                    if 'bcts' in name:
                        model = quantifier(learner=model_classifier_rf_bcts)
                    else:
                        model = quantifier(learner=model_classifier_rf)

                model.fit(X_train, y_train)
                trained_models[name] = model

            except Exception as e:
                # A model that fails to build/train does not stop the experiment
                print(f'Error training model {name}: {e}')

        # Initializing the result dictionaries BEFORE the batch loop, so that
        # results and predictions always have the same length (1 entry per batch
        # per model), avoiding KeyError and the "arrays must all be same length" ValueError.
        results = {name: [] for name in trained_models}
        predictions = {name: [] for name in trained_models}
        prevalences = []

        # Evaluating the quantification methods
        for batch, real_prevalences in batches_test:
            batch = batch.drop(columns=['target'])
            prevalences.append(real_prevalences[1])  # true prevalence of the positive class

            for name, model in trained_models.items():
                try:
                    # Prevalence prediction for the test batch
                    pred_test = model.predict(batch)

                    # Adjusting the prediction format
                    if isinstance(pred_test, dict):
                        prevalences_predict = [
                            pred_test.get(0, 0.0),
                            pred_test.get(1, 0.0)
                        ]
                    else:
                        prevalences_predict = list(pred_test)

                    # MAE between true and predicted prevalences
                    mae = mean_absolute_error(real_prevalences, prevalences_predict)

                    results[name].append(mae)
                    predictions[name].append(prevalences_predict[1])

                except Exception as e:
                    # On error: NaN in BOTH dictionaries, keeping the alignment
                    print(f'Error evaluating model {name} on the batch: {e}')
                    results[name].append(np.nan)
                    predictions[name].append(np.nan)

        # Converting results and predictions into DataFrames
        results = pd.DataFrame(results)
        results['true_prevalence'] = prevalences
        predictions = pd.DataFrame(predictions)
        predictions['true_prevalence'] = prevalences

        results.to_csv(f'{output_path}results_calibration_exp_{dataset}.csv', index=False)
        predictions.to_csv(f'{output_path}predictions_calibration_exp_{dataset}.csv', index=False)

    return True


# %%
# %%
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import pool

# create a thread pool with max worker threads
pool = ProcessPoolExecutor(max_workers=28)
threads = []

for dataset in name_data:

    # adiciona experimento na lista de threads: pool
    exp = pool.submit(run_experiment, dataset)

    #adiciona na lista para salvar a referencia da thread
    threads.append(exp)

# aguarda pela finalização das threads
for exp in as_completed(threads):
    exp.result()

# %%


