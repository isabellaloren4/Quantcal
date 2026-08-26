"""
QCalNValSweepAP — variante do sweep que avalia phi diretamente sobre um grid de
prevalências do APP, sem dividir X_val e sem gerar subgrupos de avaliação.

Diferença para o QCalNValSweep (held-out UPP)
---------------------------------------------
No sweep held-out, o erro do regressor era medido em subgrupos UPP reservados,
o que exige gerar dados e acumular um pool que MUDA a cada n (causa da subida
espúria de err em datasets como ctg.3 e mammographic). Aqui não há nada disso:

  - a TABELA que treina phi continua vindo de subgrupos UPP de X_val, exatamente
    como no QCal.fit (a parte de produção não muda);
  - a AVALIAÇÃO de phi é feita passando um grid FIXO de prevalências do APP
    direto pela função phi, e medindo a distância à diagonal identidade:

        grid = linspace(0, 1, n_app_prevalences)
        err_reg(n) = MAE(phi_n(grid), grid) = média |phi_n(p) - p|

O grid é o mesmo para todo n, então a curva err_reg(n) reflete só o efeito do
n_validation sobre phi — sem o viés de um conjunto de avaliação que cresce.

O que err_reg SIGNIFICA aqui (leia com atenção)
-----------------------------------------------
err_reg mede quanto phi se AFASTA da identidade nas prevalências do grid, isto
é, a MAGNITUDE da correção que o regressor aplica. NÃO é o erro de quantificação
sobre dados reais (para isso seriam necessárias estimativas cruas do
quantificador, que só vêm de subgrupos). Consequências:

  - err_reg ~ 0  => phi ~ identidade: o regressor quase não corrige;
  - err_reg alto => phi corrige muito naquelas prevalências.

Como o quantificador não é avaliado (não há subgrupos de teste), NÃO existe um
err_quant comparável neste modo, e o 1º critério de parada (err_reg < err_quant)
NÃO se aplica. A seleção usa a estabilização da curva (2º critério: primeiro n
cujo err_reg fica abaixo da média dos err_reg gerados) e, como último recurso,
o argmin.

Atributos (depois do fit)
-------------------------
results_ : DataFrame com uma linha por n_validation e as colunas
    dataset, method, n_prevalences, n_validation, n_pairs, err_reg.
mean_err_reg_ : float — média dos err_reg dos n regressores gerados.
n_stop_mean_  : int ou None — primeiro n (>= min) com err_reg < mean_err_reg_.
best_n_validation_ : int — n_stop_mean_, senão argmin(err_reg).
regressors_ : dict {n_validation: IsotonicRegression ajustada}.
grid_ : ndarray — o grid de prevalências do APP usado na avaliação.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from utils_qcal.protocol import *              # UPP_protocol_mlquantify
from utils_qcal.median_estimates import *
from utils_qcal.extract_estimates import *
from utils_qcal.extract_estimates import _build_and_fit_quantifier

from calibrator.QCal_iso import QCal           # importado apenas pelo _REGISTRY


def _mae(a, b):
    """MAE entre dois vetores."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


class QCalNValSweepAP:
    """Sweep de n_validation avaliando phi sobre um grid de prevalências APP.

    Parâmetros
    ----------
    method_name : str
        Quantificador base; mesmas chaves do QCal._REGISTRY.
    clf : classificador
        Repassado ao `pre_treined_model`, igual ao QCal (não usado em 'hdx').
    min_n_validation : int, padrão=1
        Piso da seleção.
    max_n_validation : int, padrão=10
        Teto da varredura.
    val_size : float, padrão=0.3
        Fração de validação de cada split — espelha o QCal.fit.
    n_prevalences : int, padrão=1100
        Subgrupos UPP por split usados para TREINAR phi (a tabela). Igual ao
        QCal em produção; reduza (ex. 50) se precisar de menos custo/memória.
    batch_size : int, padrão=100
        Tamanho de cada subgrupo UPP (repassado ao protocolo).
    n_app_prevalences : int, padrão=20
        Tamanho do grid APP (linspace 0..1) onde phi é AVALIADA. Não gera dados;
        é só o vetor de prevalências que passa por phi.
    method, calibrator :
        Repassados ao `pre_treined_model` (mesma semântica do QCal).
    name_data : str ou None
        Nome do dataset, gravado na tabela.
    random_state : int ou None
        Mantido por compatibilidade; os splits do laço ficam sem semente.
    """

    def __init__(self, *, method_name='cc', clf=None,
                 min_n_validation=1, max_n_validation=10, val_size=0.3,
                 n_prevalences=1100, batch_size=100, n_app_prevalences=20,
                 method=None, calibrator=False,
                 name_data=None, random_state=None):
        if method_name not in QCal._REGISTRY:
            raise ValueError(
                f"method_name={method_name!r} desconhecido. "
                f"Opções válidas: {sorted(QCal._REGISTRY)}"
            )
        if not 1 <= min_n_validation <= max_n_validation:
            raise ValueError(
                "é preciso 1 <= min_n_validation <= max_n_validation "
                f"(recebido min={min_n_validation}, max={max_n_validation})"
            )
        if n_app_prevalences < 2:
            raise ValueError("n_app_prevalences deve ser >= 2")
        self.method_name = method_name
        self.clf = clf
        self.min_n_validation = min_n_validation
        self.max_n_validation = max_n_validation
        self.val_size = val_size
        self.n_prevalences = n_prevalences
        self.batch_size = batch_size
        self.n_app_prevalences = n_app_prevalences
        self.method = method
        self.calibrator = calibrator
        self.name_data = name_data
        self.random_state = random_state

        # preenchidos pelo fit()
        self.results_ = None
        self.mean_err_reg_ = None
        self.n_stop_mean_ = None
        self.best_n_validation_ = None
        self.regressors_ = None
        self.grid_ = None

    # ------------------------------------------------------------------ #

    def _novo_regressor(self):
        return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                  out_of_bounds="clip")

    def _first_n(self, mask):
        elegiveis = self.results_.loc[
            mask.to_numpy() & (self.results_['n_validation'] >= self.min_n_validation),
            'n_validation'
        ]
        return int(elegiveis.min()) if len(elegiveis) else None

    # ------------------------------------------------------------------ #

    def fit(self, X_train, y_train, name_data=None):
        if name_data is None:
            name_data = self.name_data

        cfg = QCal._REGISTRY[self.method_name]
        quantifier_cls = cfg['quantifier']
        uses_clf = cfg['uses_clf']

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        _ = check_random_state(self.random_state)  # compat; splits sem semente

        # grid FIXO de prevalências do APP onde phi é avaliada (não gera dados)
        self.grid_ = np.linspace(0.0, 1.0, self.n_app_prevalences)

        estimates_all = []
        true_prevalences_all = []
        linhas = []
        self.regressors_ = {}

        for i in range(self.max_n_validation):
            n_validation = i + 1

            # ---- MESMA divisão do QCal.fit (sem semente) ----
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=self.val_size, stratify=y_train
            )
            # subgrupos UPP de X_val -> TREINO de phi (a tabela)
            subgroups = UPP_protocol_mlquantify(
                X_val, y_val,
                batch_size=self.batch_size,
                n_prevalences=self.n_prevalences,
            )
            model_clf = None
            if uses_clf:
                model_clf = pre_treined_model(
                    X_tr, y_tr, clf=self.clf, method=self.method,
                    calibration=self.calibrator
                )
            estimates, true_prevalences = extract_estimates_from_train_iso(
                X_tr, y_tr, subgroups,
                quantifier_cls=quantifier_cls, model_clf=model_clf
            )

            # ---- tabela CUMULATIVA (nunca zera) ----
            estimates_all.extend(estimates)
            true_prevalences_all.extend(true_prevalences)

            # ---- regressor NOVO (phi_n) sobre a tabela acumulada ----
            regressor = self._novo_regressor()
            regressor.fit(estimates_all, true_prevalences_all)
            self.regressors_[n_validation] = regressor

            # ---- avaliação: phi_n direto no grid APP, distância à identidade ----
            #   passa o grid por phi e compara com o próprio grid (a diagonal).
            #   Nenhum X_val nem subgrupo é usado aqui.
            err_reg = _mae(regressor.predict(self.grid_), self.grid_)

            linhas.append({
                'dataset': name_data,
                'method': self.method_name,
                'n_prevalences': self.n_prevalences,
                'n_validation': n_validation,
                'n_pairs': len(estimates_all),   # tamanho da tabela de phi_n
                'err_reg': err_reg,              # |phi_n(grid) - grid| médio
            })

        # ---- curva e seleção (sem err_quant neste modo) ----
        self.results_ = pd.DataFrame(linhas)
        self.mean_err_reg_ = float(self.results_['err_reg'].mean())

        # seleção: 1º critério (cruzamento) não existe aqui; usa o 2º e o argmin
        self.n_stop_mean_ = self._first_n(
            self.results_['err_reg'] < self.mean_err_reg_
        )
        if self.n_stop_mean_ is not None:
            self.best_n_validation_ = self.n_stop_mean_
        else:
            self.best_n_validation_ = int(
                self.results_.loc[self.results_['err_reg'].idxmin(), 'n_validation']
            )
        return self
