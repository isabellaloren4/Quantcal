"""
QCalNValSweep — variante diagnóstica do QCal usada para escolher, por dataset,
o número de splits de validação (`n_validation`).

A classe de produção `QCal` (QCal_iso.py) não é modificada. Esta classe NÃO é
um quantificador: ela não implementa predict(). Sua única função é responder,
para um dataset e um quantificador base, "qual n_validation compensa aqui?".

Como funciona (tudo em UPP, sem APP, sem teste interno)
-------------------------------------------------------
O laço usa EXATAMENTE a mesma divisão do QCal.fit — a cada validação um
train_test_split(test_size=val_size, stratify=y) sem semente. Os subgrupos UPP
de X_val são então separados em duas frações:

  - FRAÇÃO DE TREINO -> entra na tabela cumulativa que alimenta o regressor
    isotônico phi. A tabela é CUMULATIVA: depois da iteração i ela contém os
    pares dos splits 1..i+1 (a tabela que o QCal montaria com n_validation=i+1),
    e um regressor NOVO é treinado sobre ela (a isotônica não tem partial_fit).
  - FRAÇÃO DE AVALIAÇÃO (held-out) -> reservada só para medir o erro; nunca
    entra no treino de phi. Também é acumulada ao longo dos splits.

Os dois erros saem SEMPRE do mesmo conjunto (comparáveis), e o modo controla
se esse conjunto é held-out ou a própria tabela:

  - eval_mode='held_out' (padrão): erros medidos na fração de avaliação. Como
    phi não viu esses subgrupos, err_reg não é otimista. Tudo UPP, então
    err_quant e err_reg vêm da mesma distribuição de prevalências.
  - eval_mode='in_sample': todos os subgrupos vão para a tabela e os erros são
    medidos nela mesma (comportamento anterior). É barato, mas err_reg fica
    otimista porque phi é avaliada onde foi treinada.

  err_quant(n) = MAE(estimativas_do_conjunto, prevalências_do_conjunto)
  err_reg(n)   = MAE(phi_n(estimativas_do_conjunto), prevalências_do_conjunto)

Critérios de parada (por n_validation há um err_quant e um err_reg)
-------------------------------------------------------------------
  1) principal: o menor n (>= min_n_validation) em que err_reg < err_quant,
     isto é, onde a calibração passa a bater o quantificador base;
  2) fallback (caso (1) nunca ocorra): o menor n (>= min_n_validation) em que
     err_reg fica abaixo da MÉDIA dos err_reg de todos os regressores gerados.
     Por isso a varredura precisa de min_n_validation e max_n_validation: a
     média só existe depois de gerar todos os regressores de 1..max.

O gráfico (plot_error_vs_nval, em qcal_plots.py) mostra as duas curvas
err_quant(n) e err_reg(n), as duas médias horizontais, e destaca o cruzamento
do 1º critério.

Atributos (depois do fit)
-------------------------
results_ : DataFrame com uma linha por n_validation e as colunas
    dataset, method, n_prevalences, eval_mode, n_validation, n_pairs,
    n_eval_pairs, err_quant, err_reg, gain.
mean_err_reg_, mean_err_quant_ : float — médias sobre os n regressores/
    quantificadores gerados (as linhas horizontais do gráfico).
n_stop_cross_ : int ou None — 1º critério (err_reg < err_quant).
n_stop_mean_  : int ou None — 2º critério (err_reg < mean_err_reg_).
best_n_validation_ : int — n_stop_cross_, senão n_stop_mean_, senão argmin(err_reg).
regressors_ : dict {n_validation: IsotonicRegression ajustada}.
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

from calibrator.QCal_iso import QCal           # importado apenas pelo _REGISTRY (method_name -> classe)


def _mae(a, b):
    """MAE entre dois vetores de prevalência da classe 1."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


class QCalNValSweep:
    """Erro do quantificador e do regressor (phi) em função de `n_validation`.

    Parâmetros
    ----------
    method_name : str
        Quantificador base a ser calibrado; mesmas chaves do QCal._REGISTRY
        ('cc', 'acc', 'pac', 'dys', 'ms', 'emq', 'fm', 'kde', 'hdx').
    clf : classificador
        Repassado ao `pre_treined_model`, igual ao QCal (não usado em 'hdx').
    min_n_validation : int, padrão=1
        Piso da seleção: nenhum critério escolhe um n abaixo deste.
    max_n_validation : int, padrão=10
        Teto da varredura; err_quant/err_reg são registrados para
        n = 1, 2, ..., max_n_validation.
    val_size : float, padrão=0.3
        Fração de validação de cada split — espelha o test_size=0.3 do QCal.fit.
    n_prevalences : int, padrão=50
        Número de prevalências (subgrupos) do UPP por split. Mantido baixo
        (20-50) de propósito: o UPP padrão gera ~1100 subgrupos, o que pesa em
        memória e tempo quando o sweep roda em paralelo sobre muitos datasets.
        Use 20 para o teste mais leve, 50 para um MAE um pouco mais estável.
    batch_size : int, padrão=100
        Tamanho de cada subgrupo UPP (repassado ao protocolo).
    eval_mode : {'held_out', 'in_sample'}, padrão='held_out'
        Onde medir os erros. 'held_out' reserva `eval_size` dos subgrupos UPP
        para avaliar phi fora do seu treino (sem viés); 'in_sample' mede na
        própria tabela (mais barato, porém otimista).
    eval_size : float, padrão=0.3
        Só em eval_mode='held_out': fração dos subgrupos UPP de cada split
        reservada para avaliação (o resto vai para a tabela).
    method, calibrator :
        Repassados ao `pre_treined_model` com a mesma semântica do QCal
        (definir `method` sozinho já dispara a calibração do classificador).
    name_data : str ou None
        Nome do dataset, gravado na tabela de resultados.
    random_state : int ou None
        Controla a separação treino/avaliação dos subgrupos. Os splits do laço
        ficam sem semente, exatamente como no QCal.fit.
    """

    def __init__(self, *, method_name='cc', clf=None,
                 min_n_validation=1, max_n_validation=10, val_size=0.3,
                 n_prevalences=1100, batch_size=100,
                 eval_mode='held_out', eval_size=0.3,
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
        if eval_mode not in ('held_out', 'in_sample'):
            raise ValueError(
                f"eval_mode={eval_mode!r} inválido; use 'held_out' ou 'in_sample'"
            )
        if eval_mode == 'held_out' and not 0.0 < eval_size < 1.0:
            raise ValueError(
                f"eval_size deve estar em (0, 1); recebido {eval_size}"
            )
        self.method_name = method_name
        self.clf = clf
        self.min_n_validation = min_n_validation
        self.max_n_validation = max_n_validation
        self.val_size = val_size
        self.n_prevalences = n_prevalences
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self.eval_size = eval_size
        self.method = method
        self.calibrator = calibrator
        self.name_data = name_data
        self.random_state = random_state

        # preenchidos pelo fit()
        self.results_ = None
        self.mean_err_reg_ = None
        self.mean_err_quant_ = None
        self.n_stop_cross_ = None
        self.n_stop_mean_ = None
        self.best_n_validation_ = None
        self.regressors_ = None

    # ------------------------------------------------------------------ #

    def _novo_regressor(self):
        """Um corretor novo, configurado igual ao self.regressor do QCal."""
        return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                  out_of_bounds="clip")

    def _first_n(self, mask):
        """Menor n_validation >= min_n_validation onde `mask` é True; None se nenhum."""
        elegiveis = self.results_.loc[
            mask.to_numpy() & (self.results_['n_validation'] >= self.min_n_validation),
            'n_validation'
        ]
        return int(elegiveis.min()) if len(elegiveis) else None

    # ------------------------------------------------------------------ #

    def fit(self, X_train, y_train, name_data=None):
        # se nada for passado aqui, usa o nome salvo no construtor
        if name_data is None:
            name_data = self.name_data

        cfg = QCal._REGISTRY[self.method_name]
        quantifier_cls = cfg['quantifier']
        uses_clf = cfg['uses_clf']

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        rng = check_random_state(self.random_state)

        # tabela cumulativa (treino de phi)
        estimates_all = []
        true_prevalences_all = []
        # pool de avaliação cumulativo (só usado em eval_mode='held_out')
        eval_est_all = []
        eval_real_all = []

        linhas = []
        self.regressors_ = {}

        for i in range(self.max_n_validation):
            n_validation = i + 1

            # ---- MESMA divisão do QCal.fit (sem semente, como no original) ----
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=self.val_size, stratify=y_train
            )

            model_clf = None
            if uses_clf:
                model_clf = pre_treined_model(
                    X_tr, y_tr, clf=self.clf, method=self.method,
                    calibration=self.calibrator
                )
            # O quantificador base é construído UMA vez por iteração e reusado
            # nas duas frações; é equivalente ao que extract_estimates_from_train_iso
            # faz por dentro no QCal.fit, mas aqui guardamos o objeto.
            quantifier = _build_and_fit_quantifier(
                quantifier_cls, X_tr, y_tr, model_clf=model_clf, fit_classifier=False
            )

            # ---- subgrupos UPP de X_val, separados em treino/avaliação ----
            # n_prevalences pequeno (20-50) mantém custo e memória sob controle:
            # o UPP padrão gera ~1100 subgrupos por split, o que estoura em
            # experimentos paralelos sobre muitos datasets.
            subgroups = UPP_protocol_mlquantify(
                X_val, y_val,
                batch_size=self.batch_size,
                n_prevalences=self.n_prevalences,
            )
            if self.eval_mode == 'held_out':
                perm = rng.permutation(len(subgroups))
                n_eval = max(1, int(round(self.eval_size * len(subgroups))))
                subgroups_fit = [subgroups[j] for j in perm[n_eval:]]
                subgroups_eval = [subgroups[j] for j in perm[:n_eval]]
            else:  # 'in_sample'
                subgroups_fit = subgroups
                subgroups_eval = None

            # ---- tabela CUMULATIVA (fração de treino) ----
            est_fit, real_fit = extract_estimates_from_test_subgroups(
                subgroups_fit, quantifier
            )
            estimates_all.extend(est_fit)
            true_prevalences_all.extend(real_fit)

            # ---- pool held-out CUMULATIVO (fração de avaliação) ----
            if self.eval_mode == 'held_out':
                est_eval, real_eval = extract_estimates_from_test_subgroups(
                    subgroups_eval, quantifier
                )
                eval_est_all.extend(est_eval)
                eval_real_all.extend(real_eval)

            # ---- regressor NOVO (phi_n) treinado sobre a tabela acumulada ----
            regressor = self._novo_regressor()
            regressor.fit(estimates_all, true_prevalences_all)
            self.regressors_[n_validation] = regressor

            # ---- conjunto de avaliação: held-out separado, ou a própria tabela ----
            if self.eval_mode == 'held_out':
                e_est = np.asarray(eval_est_all, dtype=float)
                e_real = np.asarray(eval_real_all, dtype=float)
            else:  # 'in_sample'
                e_est = np.asarray(estimates_all, dtype=float)
                e_real = np.asarray(true_prevalences_all, dtype=float)

            err_quant = _mae(e_est, e_real)                    # quantificador base
            err_reg = _mae(regressor.predict(e_est), e_real)   # QCal (phi_n)

            linhas.append({
                'dataset': name_data,
                'method': self.method_name,
                'n_prevalences': self.n_prevalences,  # subgrupos UPP por split
                'eval_mode': self.eval_mode,          # held_out ou in_sample
                'n_validation': n_validation,
                'n_pairs': len(estimates_all),    # tamanho da tabela de phi_n
                'n_eval_pairs': len(e_est),        # tamanho do conjunto de avaliação
                'err_quant': err_quant,            # erro do quantificador base
                'err_reg': err_reg,                # erro do QCal (regressor phi_n)
                'gain': err_quant - err_reg,       # positivo => a calibração ajudou
            })

        # ---- curva e regras de seleção ----
        self.results_ = pd.DataFrame(linhas)

        # médias sobre os n regressores/quantificadores gerados (linhas do gráfico)
        self.mean_err_reg_ = float(self.results_['err_reg'].mean())
        self.mean_err_quant_ = float(self.results_['err_quant'].mean())

        # 1º critério: menor n (>= min) em que o regressor bate o quantificador
        self.n_stop_cross_ = self._first_n(
            self.results_['err_reg'] < self.results_['err_quant']
        )
        # 2º critério (fallback): menor n (>= min) com err_reg abaixo da média
        # dos err_reg de todos os regressores gerados
        self.n_stop_mean_ = self._first_n(
            self.results_['err_reg'] < self.mean_err_reg_
        )

        # n escolhido: 1º critério, senão 2º, senão o de menor err_reg
        if self.n_stop_cross_ is not None:
            self.best_n_validation_ = self.n_stop_cross_
        elif self.n_stop_mean_ is not None:
            self.best_n_validation_ = self.n_stop_mean_
        else:
            self.best_n_validation_ = int(
                self.results_.loc[self.results_['err_reg'].idxmin(), 'n_validation']
            )
        return self
