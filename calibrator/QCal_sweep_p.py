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
de onde sai esse conjunto:

  - eval_mode='held_out' (padrão): erros medidos na fração de avaliação
    CUMULATIVA (held-out dos splits 1..n). Como phi não viu esses subgrupos,
    err_reg não é otimista. Tudo UPP, então err_quant e err_reg vêm da mesma
    distribuição de prevalências. Ressalva: o held-out do split n vem da MESMA
    partição X_val que gerou a fração de treino do split n, então há uma
    proximidade sutil entre treino e avaliação dentro de um mesmo split.
  - eval_mode='in_sample': todos os subgrupos vão para a tabela e os erros são
    medidos na tabela cumulativa INTEIRA. É barato, mas err_reg fica otimista
    porque phi é avaliada onde foi treinada; piora conforme n cresce.
  - eval_mode='current': todos os subgrupos vão para a tabela (phi cumulativa,
    como no in_sample), mas os erros são medidos SÓ nos subgrupos do split
    atual — n_prevalences batches. Conjunto de avaliação de tamanho FIXO e
    barato; err_reg fica otimista (subestimado), pois esses subgrupos também
    treinaram phi.
  - eval_mode='next_split': variante MAIS honesta do held_out (validação
    forward-chaining). A cada split s, os subgrupos são divididos em fração de
    treino (vai para a tabela cumulativa) e held-out. O held-out do split s
    serve para TESTAR phi_{s-1} — o regressor treinado só com os splits
    1..s-1, que nunca viu NENHUM subgrupo do split s (nem treino, nem held-out).
    Assim err_reg(n) é medido em dados de um split totalmente novo (out-of-sample
    de verdade, sem o vazamento sutil do held_out). O held-out NUNCA entra no
    treino de phi. Precisa de max_n_validation+1 splits: o split (n+1) testa
    phi_n. A avaliação NÃO é cumulativa — cada phi_n é testado no held-out de um
    único split (o n+1), então convém n_prevalences maior (ou suavizar a curva
    com a média móvel do gráfico) para reduzir o ruído por ponto.

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

  ATENÇÃO (eval_mode='current' ou 'in_sample'): como o err_reg é medido onde
  phi foi treinada, ele tende a ficar abaixo do err_quant já em n=1, e o 1º
  critério costuma disparar no min_n_validation. Nesses modos o
  best_n_validation_ pode colapsar para 1. Para SELECIONAR o n de forma
  comparável entre datasets, prefira 'held_out' ou 'next_split'.

O gráfico (plot_error_vs_nval, em qcal_plots.py) mostra as duas curvas
err_quant(n) e err_reg(n), as duas médias horizontais, e destaca o cruzamento
do 1º critério. Ele aceita ma_window>1 para desenhar uma média móvel.

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
batch_results_ : DataFrame ou None — preenchido só quando collect_batches=True.
    UMA linha por batch held-out avaliado (não colapsa em MAE). Colunas:
    dataset, method, eval_mode, n_validation, batch_id, true_prevalence,
    est_raw, est_cal, abs_err_raw, abs_err_cal, gain. É o formato pensado para
    o explorer HTML e para plot_batch_error_vs_prevalence (erro-vs-prevalência
    com faixas de desvio). O results_ (MAE por n) continua saindo igual.
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
        n = 1, 2, ..., max_n_validation. Em eval_mode='next_split' o laço roda
        max_n_validation+1 splits (o split n+1 testa phi_n).
    val_size : float, padrão=0.3
        Fração de validação de cada split — espelha o test_size=0.3 do QCal.fit.
    n_prevalences : int, padrão=50
        Número de prevalências (subgrupos) do UPP por split. Mantido baixo
        (20-50) de propósito: o UPP padrão gera ~1100 subgrupos, o que pesa em
        memória e tempo quando o sweep roda em paralelo sobre muitos datasets.
        Em eval_mode='current' este é o TAMANHO do conjunto de avaliação; em
        'next_split' a avaliação por ponto é ~eval_size*n_prevalences (não
        cumulativa), então convém subir n_prevalences nesse modo.
    batch_size : int, padrão=100
        Tamanho de cada subgrupo UPP (repassado ao protocolo).
    eval_mode : {'held_out', 'in_sample', 'current', 'next_split'}, padrão='held_out'
        Onde medir os erros. 'held_out' reserva `eval_size` dos subgrupos UPP
        para avaliar phi fora do seu treino (cumulativo); 'in_sample' mede na
        tabela cumulativa inteira (otimista); 'current' mede só nos subgrupos do
        split atual (tamanho fixo, otimista); 'next_split' mede phi_n no
        held-out de um split totalmente novo (o n+1) — o mais honesto.
    n_eval : int ou None, padrão=None
        Em 'held_out' e 'next_split': contagem ABSOLUTA de batches held-out por
        split. Quando dado, ignora eval_size e fixa o held-out nesse número
        (limitado a [1, n_prevalences-1]). Ex.: n_prevalences=1100, n_eval=100
        => 100 batches no held-out e 1000 na tabela de treino do phi, por split.
    eval_size : float, padrão=0.3
        Em 'held_out' e 'next_split': fração dos subgrupos UPP de cada split
        (usada só quando n_eval é None)
        reservada para avaliação (o resto vai para a tabela).
    method, calibrator :
        Repassados ao `pre_treined_model` com a mesma semântica do QCal
        (definir `method` sozinho já dispara a calibração do classificador).
    name_data : str ou None
        Nome do dataset, gravado na tabela de resultados.
    collect_batches : bool, padrão=False
        Se True, além do results_ (MAE por n_validation) preenche o
        batch_results_ com UMA linha por batch held-out avaliado — a prevalência
        real do batch, a estimativa crua e a calibrada e os erros absolutos.
        É o insumo do explorer HTML e do gráfico erro-vs-prevalência. Não altera
        o results_ nem a seleção do best_n_validation_. Vale em todos os
        eval_mode; em 'next_split' cada n usa o held-out do split n+1 (fresco).
    random_state : int ou None
        Controla a separação treino/avaliação dos subgrupos. Os splits do laço
        ficam sem semente, exatamente como no QCal.fit.
    """

    def __init__(self, *, method_name='cc', clf=None,
                 min_n_validation=1, max_n_validation=10, val_size=0.3,
                 n_prevalences=1100, batch_size=100,
                 eval_mode='held_out', eval_size=0.3, n_eval=None,
                 method=None, calibrator=False,
                 name_data=None, collect_batches=False, random_state=None):
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
        if eval_mode not in ('held_out', 'in_sample', 'current', 'next_split'):
            raise ValueError(
                f"eval_mode={eval_mode!r} inválido; "
                "use 'held_out', 'in_sample', 'current' ou 'next_split'"
            )
        if eval_mode in ('held_out', 'next_split') and n_eval is None \
                and not 0.0 < eval_size < 1.0:
            raise ValueError(
                f"eval_size deve estar em (0, 1); recebido {eval_size}"
            )
        if n_eval is not None and int(n_eval) < 1:
            raise ValueError(f"n_eval deve ser >= 1; recebido {n_eval}")
        self.method_name = method_name
        self.clf = clf
        self.min_n_validation = min_n_validation
        self.max_n_validation = max_n_validation
        self.val_size = val_size
        self.n_prevalences = n_prevalences
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self.eval_size = eval_size
        # n_eval: contagem ABSOLUTA de batches held-out por split. Quando dado,
        # ignora eval_size e fixa o held-out nesse número (ex.: 1100 gerados,
        # n_eval=100 => 100 held-out e 1000 para a tabela de treino do phi).
        self.n_eval = None if n_eval is None else int(n_eval)
        self.method = method
        self.calibrator = calibrator
        self.name_data = name_data
        self.collect_batches = collect_batches
        self.random_state = random_state

        # preenchidos pelo fit()
        self.results_ = None
        self.batch_results_ = None
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

    def _make_split(self, X_train, y_train, cfg, uses_clf, quantifier_cls):
        """Uma divisão treino/validação + quantificador base do split.

        Reproduz o miolo de uma iteração comum a todos os modos: train_test_split
        SEM semente (como no QCal.fit), treina o classificador se necessário e
        constrói/ajusta o quantificador base sobre X_tr. Devolve (X_val, y_val,
        quantifier) — os subgrupos UPP de X_val são gerados pelo chamador.
        """
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=self.val_size, stratify=y_train
        )
        model_clf = None
        if uses_clf:
            model_clf = pre_treined_model(
                X_tr, y_tr, clf=self.clf, method=self.method,
                calibration=self.calibrator
            )
        quantifier = _build_and_fit_quantifier(
            quantifier_cls, X_tr, y_tr, model_clf=model_clf, fit_classifier=False
        )
        return X_val, y_val, quantifier

    # ------------------------------------------------------------------ #

    def _log_m_grid(self, pool_size, n_points=25):
        """Grade log-espaçada de nº de batches de treino, de 1 até pool_size."""
        grid = np.unique(
            np.round(np.logspace(0, np.log10(max(pool_size, 2)), n_points)).astype(int)
        )
        return grid[(grid >= 1) & (grid <= pool_size)]

    def batch_learning_curve(self, X_train, y_train, *, m_grid=None, n_reps=30,
                             pool_size=1000, eval_count=100, name_data=None,
                             random_state=0):
        """Curva de aprendizado do QCal por NÚMERO DE BATCHES de treino.

        Gera, num split, um POOL de `pool_size` batches (subgrupos UPP) para
        treinar a isotônica, e num split SEPARADO um HELD-OUT de `eval_count`
        batches (que o phi nunca vê — sem vazamento). Para cada m em m_grid,
        repete n_reps vezes: amostra m batches do pool, ajusta a isotônica neles
        e mede o MAE no held-out. Devolve um DataFrame com uma linha por
        (m, rep): colunas dataset, method, m, rep, mae_cal, mae_raw.

        - mae_cal: MAE do QCal (isotônica treinada em m batches) no held-out.
        - mae_raw: MAE do quantificador base no MESMO held-out (constante em m;
          serve de linha de referência).

        A média e o desvio (sobre as n_reps) por m são o que o gráfico desenha:
        a linha mostra quanto o QCal melhora conforme ganha batches, e a faixa,
        quão instável ele é com poucos batches.
        """
        if name_data is None:
            name_data = self.name_data
        cfg = QCal._REGISTRY[self.method_name]
        quantifier_cls = cfg['quantifier']
        uses_clf = cfg['uses_clf']

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        rng = check_random_state(random_state)

        # POOL de treino (split A) — de onde os m batches são amostrados
        Xv, yv, q_pool = self._make_split(X_train, y_train, cfg, uses_clf, quantifier_cls)
        pool = UPP_protocol_mlquantify(Xv, yv, batch_size=self.batch_size,
                                       n_prevalences=pool_size)
        est_pool, real_pool = extract_estimates_from_test_subgroups(pool, q_pool)
        est_pool = np.asarray(est_pool, dtype=float)
        real_pool = np.asarray(real_pool, dtype=float)
        pool_n = len(est_pool)

        # HELD-OUT (split B, novo) — avaliação, nunca entra no treino do phi
        Xe, ye, q_eval = self._make_split(X_train, y_train, cfg, uses_clf, quantifier_cls)
        held = UPP_protocol_mlquantify(Xe, ye, batch_size=self.batch_size,
                                       n_prevalences=eval_count)
        est_eval, real_eval = extract_estimates_from_test_subgroups(held, q_eval)
        est_eval = np.asarray(est_eval, dtype=float)
        real_eval = np.asarray(real_eval, dtype=float)

        mae_raw = float(np.mean(np.abs(est_eval - real_eval)))

        if m_grid is None:
            m_grid = self._log_m_grid(pool_n)
        m_grid = [int(m) for m in m_grid if 1 <= int(m) <= pool_n]

        linhas = []
        for m in m_grid:
            for rep in range(n_reps):
                idx = rng.choice(pool_n, size=m, replace=False)
                reg = self._novo_regressor()
                reg.fit(est_pool[idx], real_pool[idx])
                pred = reg.predict(est_eval)
                mae_cal = float(np.mean(np.abs(pred - real_eval)))
                linhas.append({
                    'dataset': name_data,
                    'method': self.method_name,
                    'm': m,                    # nº de batches de treino do phi
                    'rep': rep,
                    'mae_cal': mae_cal,        # MAE do QCal no held-out
                    'mae_raw': mae_raw,        # MAE do base no held-out (constante)
                })
        return pd.DataFrame(linhas)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _trim_by_prevalence(est, real, k=2.0, n_bins=20, min_bin=3, side='both'):
        """Máscara de quais batches MANTER (pré-processamento antes do phi).

        Corte de outlier POR FAIXA de prevalência. Em cada faixa com >= min_bin
        batches, usa a média e o desvio do erro absoluto AE=|est-real| DAQUELA
        faixa. O que se mantém depende de `side`:
          - 'both'  : mantém AE dentro de [média - k·σ, média + k·σ] (dois lados);
          - 'upper' : mantém AE <= média + k·σ (só corta a cauda de erro ALTO;
                      preserva TODOS os batches abaixo da média — erro baixo).
        Faixas com poucos batches são mantidas inteiras. Devolve um array
        booleano do tamanho de est (True = mantém).
        """
        est = np.asarray(est, dtype=float)
        real = np.asarray(real, dtype=float)
        ae = np.abs(est - real)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(real, edges) - 1, 0, n_bins - 1)
        keep = np.ones(len(est), dtype=bool)
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= min_bin:
                mu = ae[m].mean()
                sd = ae[m].std()
                hi = mu + k * sd
                if side == 'upper':
                    keep[m] = ae[m] <= hi          # mantém tudo abaixo + até +k·σ
                else:
                    lo = mu - k * sd
                    keep[m] = (ae[m] >= lo) & (ae[m] <= hi)
        return keep

    def compare_trim(self, X_train, y_train, *, k=2.0, n_bins=20, min_bin=3,
                     pool_size=1000, eval_count=100, n_reps=30,
                     name_data=None, random_state=0):
        """Compara base / QCal-cheio / QCal-aparado para ESTE method_name.

        Em cada repetição, faz um split de treino (pool de `pool_size` batches
        para o phi) e um split separado de held-out (`eval_count` batches, nunca
        filtrado). Fita a isotônica de duas formas — em TODOS os batches do pool
        (cheio) e só nos que sobrevivem ao corte ±k·σ por faixa de prevalência
        (aparado) — e mede o MAE das três condições no MESMO held-out:

          - mae_base : quantificador base, sem calibração;
          - mae_full : QCal treinado em todos os batches do pool;
          - mae_trim : QCal treinado só nos batches dentro de ±k·σ por faixa.

        Devolve um DataFrame com uma linha por repetição: dataset, method, k,
        rep, mae_base, mae_full, mae_trim, kept_frac (fração do pool que passou
        no corte). O corte NUNCA toca no held-out — só na tabela de treino do phi.
        """
        if name_data is None:
            name_data = self.name_data
        cfg = QCal._REGISTRY[self.method_name]
        quantifier_cls = cfg['quantifier']
        uses_clf = cfg['uses_clf']

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        rng = check_random_state(random_state)

        linhas = []
        for rep in range(n_reps):
            # POOL de treino do phi (split A)
            Xv, yv, q_pool = self._make_split(X_train, y_train, cfg, uses_clf, quantifier_cls)
            pool = UPP_protocol_mlquantify(Xv, yv, batch_size=self.batch_size,
                                           n_prevalences=pool_size)
            est_pool, real_pool = extract_estimates_from_test_subgroups(pool, q_pool)
            est_pool = np.asarray(est_pool, dtype=float)
            real_pool = np.asarray(real_pool, dtype=float)

            # HELD-OUT (split B, novo) — nunca filtrado
            Xe, ye, q_eval = self._make_split(X_train, y_train, cfg, uses_clf, quantifier_cls)
            held = UPP_protocol_mlquantify(Xe, ye, batch_size=self.batch_size,
                                           n_prevalences=eval_count)
            est_eval, real_eval = extract_estimates_from_test_subgroups(held, q_eval)
            est_eval = np.asarray(est_eval, dtype=float)
            real_eval = np.asarray(real_eval, dtype=float)

            mae_base = float(np.mean(np.abs(est_eval - real_eval)))

            # QCal cheio: isotônica em todos os batches do pool
            reg_full = self._novo_regressor()
            reg_full.fit(est_pool, real_pool)
            mae_full = float(np.mean(np.abs(reg_full.predict(est_eval) - real_eval)))

            # QCal aparado: isotônica só nos batches dentro de ±k·σ por faixa
            keep = self._trim_by_prevalence(est_pool, real_pool, k=k,
                                            n_bins=n_bins, min_bin=min_bin)
            if keep.sum() >= 2:
                reg_trim = self._novo_regressor()
                reg_trim.fit(est_pool[keep], real_pool[keep])
                mae_trim = float(np.mean(np.abs(reg_trim.predict(est_eval) - real_eval)))
            else:
                mae_trim = np.nan   # corte agressivo demais nesta repetição

            linhas.append({
                'dataset': name_data,
                'method': self.method_name,
                'k': k,
                'rep': rep,
                'mae_base': mae_base,
                'mae_full': mae_full,
                'mae_trim': mae_trim,
                'kept_frac': float(keep.mean()),
            })
        return pd.DataFrame(linhas)

    # ------------------------------------------------------------------ #

    def evaluate_on_test(self, X, y, *, k=2.0, n_bins=20, min_bin=3, side='both',
                         pool_size=1000, test_prevalences=300, test_size=0.3,
                         n_reps=30, name_data=None, random_state=0):
        """Avaliação NO X_test: base / QCal-cheio / QCal-aparado.

        A avaliação "de verdade" (estilo QCDS), com o phi treinado no X_train e
        medido em batches do X_test — o conjunto de teste do dataset, que não
        entra em treino nenhum. Por repetição:

          1. divide (X, y) em X_train / X_test (estratificado, test_size);
          2. no X_train: treina o quantificador base e monta o pool de validação
             (est, real); ajusta a isotônica em TODOS (phi cheio) e nos que
             passam no corte ±k·σ por faixa (phi aparado);
          3. gera batches de teste do X_test (UPP, test_prevalences) e estima
             com o MESMO quantificador base;
          4. calcula o erro absoluto de cada batch de teste nas três condições:
             base (est cru), cheio (phi_full(est)), aparado (phi_trim(est)).

        Devolve um DataFrame com uma linha por (rep, batch de teste): dataset,
        method, k, rep, batch_id, true_prevalence, ae_base, ae_full, ae_trim,
        kept_frac. O X_test NUNCA é filtrado; o corte só toca a tabela de treino
        do phi.
        """
        from sklearn.model_selection import train_test_split

        if name_data is None:
            name_data = self.name_data
        cfg = QCal._REGISTRY[self.method_name]
        quantifier_cls = cfg['quantifier']
        uses_clf = cfg['uses_clf']

        X = np.asarray(X)
        y = np.asarray(y)
        rng = check_random_state(random_state)

        linhas = []
        for rep in range(n_reps):
            seed = rng.randint(0, 2**31 - 1)
            try:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=seed)
            except ValueError:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size, random_state=seed)

            # --- treino: quantificador base + pool de validação (do X_train) ---
            Xv, yv, quantifier = self._make_split(Xtr, ytr, cfg, uses_clf, quantifier_cls)
            pool = UPP_protocol_mlquantify(Xv, yv, batch_size=self.batch_size,
                                           n_prevalences=pool_size)
            est_pool, real_pool = extract_estimates_from_test_subgroups(pool, quantifier)
            est_pool = np.asarray(est_pool, dtype=float)
            real_pool = np.asarray(real_pool, dtype=float)

            reg_full = self._novo_regressor()
            reg_full.fit(est_pool, real_pool)

            keep = self._trim_by_prevalence(est_pool, real_pool, k=k,
                                            n_bins=n_bins, min_bin=min_bin, side=side)
            reg_trim = None
            if keep.sum() >= 2:
                reg_trim = self._novo_regressor()
                reg_trim.fit(est_pool[keep], real_pool[keep])

            # --- teste: batches do X_test, estimados pelo MESMO quantificador ---
            test = UPP_protocol_mlquantify(Xte, yte, batch_size=self.batch_size,
                                           n_prevalences=test_prevalences)
            est_te, real_te = extract_estimates_from_test_subgroups(test, quantifier)
            est_te = np.asarray(est_te, dtype=float)
            real_te = np.asarray(real_te, dtype=float)

            ae_base = np.abs(est_te - real_te)
            ae_full = np.abs(reg_full.predict(est_te) - real_te)
            ae_trim = (np.abs(reg_trim.predict(est_te) - real_te)
                       if reg_trim is not None else np.full_like(ae_base, np.nan))

            kept_frac = float(keep.mean())
            for b in range(len(est_te)):
                linhas.append({
                    'dataset': name_data,
                    'method': self.method_name,
                    'k': k,
                    'rep': rep,
                    'batch_id': b,
                    'true_prevalence': float(real_te[b]),
                    'ae_base': float(ae_base[b]),
                    'ae_full': float(ae_full[b]),
                    'ae_trim': float(ae_trim[b]),
                    'kept_frac': kept_frac,
                })
        return pd.DataFrame(linhas)

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

        self.regressors_ = {}
        self._batch_acc = [] if self.collect_batches else None
        if self.eval_mode == 'next_split':
            linhas = self._sweep_next_split(
                X_train, y_train, name_data, cfg, uses_clf, quantifier_cls, rng
            )
        else:
            linhas = self._sweep_standard(
                X_train, y_train, name_data, cfg, uses_clf, quantifier_cls, rng
            )
        if self.collect_batches:
            self.batch_results_ = pd.DataFrame(self._batch_acc)

        # ---- curva e regras de seleção (comuns a todos os modos) ----
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

    # ------------------------------------------------------------------ #

    def _linha(self, name_data, n_validation, n_pairs, e_est, e_real, regressor):
        """Monta uma linha de results_ dado o conjunto de avaliação e o phi_n."""
        err_quant = _mae(e_est, e_real)                    # quantificador base
        err_reg = _mae(regressor.predict(e_est), e_real)   # QCal (phi_n)
        return {
            'dataset': name_data,
            'method': self.method_name,
            'n_prevalences': self.n_prevalences,  # subgrupos UPP por split
            'eval_mode': self.eval_mode,
            'n_validation': n_validation,
            'n_pairs': n_pairs,                # tamanho da tabela de phi_n
            'n_eval_pairs': len(e_est),        # tamanho do conjunto de avaliação
            'err_quant': err_quant,            # erro do quantificador base
            'err_reg': err_reg,                # erro do QCal (regressor phi_n)
            'gain': err_quant - err_reg,       # positivo => a calibração ajudou
        }

    def _batch_rows(self, name_data, n_validation, e_est, e_real, regressor):
        """Expande o conjunto de avaliação em UMA linha por batch held-out.

        Ao contrário de _linha (que colapsa e_est/e_real num único MAE), aqui
        cada batch vira um ponto: sua prevalência real, a estimativa crua e a
        calibrada, e os erros absolutos correspondentes. É o formato pensado
        para o explorer HTML e para o gráfico erro-vs-prevalência com faixas de
        desvio. Chamado só quando collect_batches=True.
        """
        e_est = np.asarray(e_est, dtype=float)
        e_real = np.asarray(e_real, dtype=float)
        e_cal = np.asarray(regressor.predict(e_est), dtype=float)
        abs_raw = np.abs(e_est - e_real)
        abs_cal = np.abs(e_cal - e_real)
        return [
            {
                'dataset': name_data,
                'method': self.method_name,
                'eval_mode': self.eval_mode,
                'n_validation': n_validation,
                'batch_id': k,                       # índice do batch dentro deste n
                'true_prevalence': float(e_real[k]), # prevalência real do batch
                'est_raw': float(e_est[k]),          # estimativa do quantificador base
                'est_cal': float(e_cal[k]),          # estimativa calibrada (phi_n)
                'abs_err_raw': float(abs_raw[k]),    # erro absoluto do base
                'abs_err_cal': float(abs_cal[k]),    # erro absoluto do QCal
                'gain': float(abs_raw[k] - abs_cal[k]),  # >0 => calibração ajudou aqui
            }
            for k in range(len(e_est))
        ]

    def _resolve_n_eval(self, n_subgroups):
        """Quantos batches held-out usar neste split.

        Se n_eval (contagem absoluta) foi dado, usa esse número — limitado a
        [1, n_subgroups-1] para sempre sobrar ao menos 1 batch para a tabela de
        treino. Caso contrário, cai na fração eval_size arredondada (comportamento
        antigo). Ex.: n_subgroups=1100, n_eval=100 -> 100 held-out, 1000 treino.
        """
        if self.n_eval is not None:
            return max(1, min(self.n_eval, n_subgroups - 1))
        return max(1, int(round(self.eval_size * n_subgroups)))

    def _sweep_standard(self, X_train, y_train, name_data,
                        cfg, uses_clf, quantifier_cls, rng):
        """Modos 'held_out', 'in_sample' e 'current' (add-then-eval por split)."""
        estimates_all = []
        true_prevalences_all = []
        eval_est_all = []           # pool held-out cumulativo (só 'held_out')
        eval_real_all = []
        linhas = []

        for i in range(self.max_n_validation):
            n_validation = i + 1
            X_val, y_val, quantifier = self._make_split(
                X_train, y_train, cfg, uses_clf, quantifier_cls
            )

            subgroups = UPP_protocol_mlquantify(
                X_val, y_val,
                batch_size=self.batch_size,
                n_prevalences=self.n_prevalences,
            )
            if self.eval_mode == 'held_out':
                perm = rng.permutation(len(subgroups))
                n_eval = self._resolve_n_eval(len(subgroups))
                subgroups_fit = [subgroups[j] for j in perm[n_eval:]]
                subgroups_eval = [subgroups[j] for j in perm[:n_eval]]
            else:  # 'in_sample' ou 'current' -> tudo vai para a tabela
                subgroups_fit = subgroups
                subgroups_eval = None

            # ---- tabela CUMULATIVA (fração de treino) ----
            est_fit, real_fit = extract_estimates_from_test_subgroups(
                subgroups_fit, quantifier
            )
            estimates_all.extend(est_fit)
            true_prevalences_all.extend(real_fit)

            # ---- pool held-out CUMULATIVO (só 'held_out') ----
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

            # ---- conjunto de avaliação, conforme o modo ----
            if self.eval_mode == 'held_out':
                e_est = np.asarray(eval_est_all, dtype=float)
                e_real = np.asarray(eval_real_all, dtype=float)
            elif self.eval_mode == 'in_sample':
                e_est = np.asarray(estimates_all, dtype=float)
                e_real = np.asarray(true_prevalences_all, dtype=float)
            else:  # 'current' -> só os n_prevalences batches do split atual
                e_est = np.asarray(est_fit, dtype=float)
                e_real = np.asarray(real_fit, dtype=float)

            linhas.append(self._linha(
                name_data, n_validation, len(estimates_all),
                e_est, e_real, regressor
            ))
            if self.collect_batches:
                self._batch_acc.extend(self._batch_rows(
                    name_data, n_validation, e_est, e_real, regressor
                ))
        return linhas

    def _sweep_next_split(self, X_train, y_train, name_data,
                          cfg, uses_clf, quantifier_cls, rng):
        """Modo 'next_split' (forward-chaining, o mais honesto).

        A cada split s: separa held-out e fração de treino. PRIMEIRO testa
        phi_{s-1} (treinado só com os splits 1..s-1) no held-out do split s —
        que phi_{s-1} nunca viu. DEPOIS folda a fração de treino do split s na
        tabela e treina phi_s. Assim a linha de n_validation=n usa phi_n
        (tabela = splits 1..n) avaliado no held-out do split n+1.

        O held-out NUNCA entra na tabela (nem cumulativamente). Precisa de
        max_n_validation+1 splits para gerar as linhas n=1..max.
        """
        estimates_all = []
        true_prevalences_all = []
        linhas = []
        phi_prev = None       # phi_{s-1}, treinado nos splits 1..s-1
        n_prev = None         # n_validation de phi_prev

        for s in range(1, self.max_n_validation + 2):   # 1..max+1
            X_val, y_val, quantifier = self._make_split(
                X_train, y_train, cfg, uses_clf, quantifier_cls
            )

            subgroups = UPP_protocol_mlquantify(
                X_val, y_val,
                batch_size=self.batch_size,
                n_prevalences=self.n_prevalences,
            )
            perm = rng.permutation(len(subgroups))
            n_eval = self._resolve_n_eval(len(subgroups))
            subgroups_eval = [subgroups[j] for j in perm[:n_eval]]
            subgroups_fit = [subgroups[j] for j in perm[n_eval:]]

            # 1) TESTA phi_{s-1} no held-out deste split (dados 100% novos p/ ele)
            if phi_prev is not None:
                est_eval, real_eval = extract_estimates_from_test_subgroups(
                    subgroups_eval, quantifier
                )
                e_est = np.asarray(est_eval, dtype=float)
                e_real = np.asarray(real_eval, dtype=float)
                # n_pairs = tamanho da tabela que treinou phi_{n_prev} (splits 1..n_prev)
                linhas.append(self._linha(
                    name_data, n_prev, len(estimates_all),
                    e_est, e_real, phi_prev
                ))
                if self.collect_batches:
                    self._batch_acc.extend(self._batch_rows(
                        name_data, n_prev, e_est, e_real, phi_prev
                    ))

            # 2) FOLDA a fração de treino deste split e treina phi_s (se ainda útil)
            if s <= self.max_n_validation:
                est_fit, real_fit = extract_estimates_from_test_subgroups(
                    subgroups_fit, quantifier
                )
                estimates_all.extend(est_fit)
                true_prevalences_all.extend(real_fit)

                regressor = self._novo_regressor()
                regressor.fit(estimates_all, true_prevalences_all)
                self.regressors_[s] = regressor
                phi_prev = regressor
                n_prev = s
        return linhas
