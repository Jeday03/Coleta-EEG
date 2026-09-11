import argparse
import math
import re
from pathlib import Path

import numpy as np
import pyedflib




CANAIS_SENSORIOMOTORES = [
    "C3", "C4", "Cz",
    "F3", "F4",
    "P3", "P4",
]

FREQ_MIN = 8.0
FREQ_MAX = 30.0


JANELA_INICIO = 0.5
JANELA_FIM = 3.5

N_COMPONENTES_CSP = 4
REG_CSP = 0.10
REG_LDA = 0.10


EXCLUSOES_CONHECIDAS = {
    "001": {(1, 2), (1, 19)},
    "002": {(4, 24)},
    "003": {
        (2, 13),
        (2, 23),
        (3, 4),
        (3, 28),
        (3, 33),
        (4, 26),
    },
    "004": set(),
}


def texto_annotation(valor):
    if isinstance(valor, bytes):
        return valor.decode("utf-8", errors="replace")
    return str(valor)


def codigo_participante(reader, caminho):
    codigo = ""

    try:
        codigo = str(reader.getPatientCode()).strip()
    except Exception:
        pass

    if codigo:
        return codigo

    match = re.match(r"([A-Za-z0-9_-]+)", Path(caminho).stem)
    return match.group(1) if match else Path(caminho).stem


def ler_edf(caminho):
    reader = pyedflib.EdfReader(str(caminho))

    try:
        labels = list(reader.getSignalLabels())
        frequencias = np.asarray(
            reader.getSampleFrequencies(),
            dtype=float,
        )

        if len(labels) != 16:
            raise ValueError(
                f"Esperados 16 canais EEG, mas o EDF tem {len(labels)}."
            )

        if not np.allclose(frequencias, frequencias[0]):
            raise ValueError(
                "Os canais do EDF não têm a mesma frequência de amostragem."
            )

        fs = float(frequencias[0])

        faltando = [
            canal
            for canal in CANAIS_SENSORIOMOTORES
            if canal not in labels
        ]

        if faltando:
            raise ValueError(
                "Canais necessários não encontrados: "
                + ", ".join(faltando)
            )

        indices = [
            labels.index(canal)
            for canal in CANAIS_SENSORIOMOTORES
        ]

        sinais = np.vstack(
            [
                np.asarray(reader.readSignal(indice), dtype=np.float64)
                for indice in indices
            ]
        )

        onsets, _, descriptions = reader.readAnnotations()

        annotations = [
            (float(onset), texto_annotation(desc))
            for onset, desc in zip(onsets, descriptions)
        ]

        codigo = codigo_participante(reader, caminho)

    finally:
        reader.close()

    return {
        "codigo": codigo,
        "fs": fs,
        "sinais": sinais,
        "annotations": annotations,
    }


def localizar_trials(annotations):
    trials = []

    run = 0
    trial = 0

    for onset, descricao in annotations:
        if descricao == "NewRun":
            run += 1
            trial = 0

        elif descricao == "NewTrial":
            trial += 1

        elif descricao in {"LeftExec", "RightExec"}:
            if run == 0 or trial == 0:
                continue

            classe = 0 if descricao == "LeftExec" else 1
            rotulo = "LEFT" if classe == 0 else "RIGHT"

            trials.append(
                {
                    "run": run,
                    "trial": trial,
                    "onset": onset,
                    "classe": classe,
                    "rotulo": rotulo,
                }
            )

    return trials


def filtrar_8_30_fft(epoch, fs):

    epoch = np.asarray(epoch, dtype=np.float64)
    epoch = epoch - np.mean(epoch, axis=1, keepdims=True)

    pad = int(round(fs))

    if epoch.shape[1] <= pad:
        raise ValueError("Epoch curto demais para filtragem.")

    estendido = np.pad(
        epoch,
        ((0, 0), (pad, pad)),
        mode="reflect",
    )

    espectro = np.fft.rfft(estendido, axis=1)
    frequencias = np.fft.rfftfreq(
        estendido.shape[1],
        d=1.0 / fs,
    )

    mascara = (
        (frequencias >= FREQ_MIN)
        & (frequencias <= FREQ_MAX)
    )

    espectro[:, ~mascara] = 0.0

    filtrado = np.fft.irfft(
        espectro,
        n=estendido.shape[1],
        axis=1,
    )

    filtrado = filtrado[:, pad:-pad]

    inicio = int(round(JANELA_INICIO * fs))
    fim = int(round(JANELA_FIM * fs))

    return filtrado[:, inicio:fim]


def criar_dataset(dados, ignorar_exclusoes=False):
    fs = dados["fs"]
    sinais = dados["sinais"]
    trials = localizar_trials(dados["annotations"])
    codigo = dados["codigo"]

    exclusoes = set()

    if not ignorar_exclusoes:
        exclusoes = EXCLUSOES_CONHECIDAS.get(codigo, set())

    X = []
    y = []
    grupos = []
    identificadores = []
    removidos = []

    amostras_exec = int(round(4.0 * fs))

    for item in trials:
        chave = (item["run"], item["trial"])

        if chave in exclusoes:
            removidos.append(
                (
                    item["run"],
                    item["trial"],
                    item["rotulo"],
                )
            )
            continue

        inicio = int(round(item["onset"] * fs))
        fim = inicio + amostras_exec

        if fim > sinais.shape[1]:
            continue

        epoch = sinais[:, inicio:fim]

        if epoch.shape[1] != amostras_exec:
            continue

        epoch = filtrar_8_30_fft(epoch, fs)

        X.append(epoch)
        y.append(item["classe"])
        grupos.append(item["run"])
        identificadores.append(
            (
                item["run"],
                item["trial"],
                item["rotulo"],
            )
        )

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    grupos = np.asarray(grupos, dtype=int)

    if len(X) == 0:
        raise ValueError("Nenhum trial foi extraído do EDF.")

    runs = sorted(np.unique(grupos).tolist())

    if runs != [1, 2, 3, 4]:
        raise ValueError(
            f"Esperadas as runs 1, 2, 3 e 4. Encontrado: {runs}"
        )

    if len(np.unique(y)) != 2:
        raise ValueError("As duas classes LEFT e RIGHT não estão presentes.")

    return X, y, grupos, identificadores, removidos


def covariancia_normalizada(epoch):
    cov = epoch @ epoch.T
    traco = float(np.trace(cov))

    if traco <= 0:
        return cov

    return cov / traco


def ajustar_csp(X, y):
    covariancias = []

    for classe in (0, 1):
        covs = np.asarray(
            [
                covariancia_normalizada(epoch)
                for epoch in X[y == classe]
            ]
        )

        cov = np.mean(covs, axis=0)

        escala = np.trace(cov) / cov.shape[0]

        cov = (
            (1.0 - REG_CSP) * cov
            + REG_CSP * escala * np.eye(cov.shape[0])
        )

        covariancias.append(cov)

    cov0, cov1 = covariancias
    total = cov0 + cov1

    valores, vetores = np.linalg.eigh(total)
    valores = np.maximum(valores, 1e-12)

    whitening = (
        vetores
        @ np.diag(1.0 / np.sqrt(valores))
        @ vetores.T
    )

    cov0_whitened = (
        whitening
        @ cov0
        @ whitening.T
    )

    autovalores, autovetores = np.linalg.eigh(
        cov0_whitened
    )

    filtros = autovetores.T @ whitening

    ordem = np.argsort(autovalores)

    metade = N_COMPONENTES_CSP // 2

    selecionados = np.concatenate(
        [
            ordem[:metade],
            ordem[-metade:],
        ]
    )

    return filtros[selecionados]


def transformar_csp(X, filtros):
    projetado = np.einsum(
        "kc,nct->nkt",
        filtros,
        X,
    )

    variancias = np.var(
        projetado,
        axis=2,
    )

    soma = np.sum(
        variancias,
        axis=1,
        keepdims=True,
    )

    soma[soma <= 0] = 1.0

    variancias = variancias / soma

    return np.log(
        variancias + 1e-12
    )


def ajustar_lda(features, y):
    feat0 = features[y == 0]
    feat1 = features[y == 1]

    media0 = np.mean(feat0, axis=0)
    media1 = np.mean(feat1, axis=0)

    central0 = feat0 - media0
    central1 = feat1 - media1

    denominador = max(
        len(features) - 2,
        1,
    )

    cov = (
        central0.T @ central0
        + central1.T @ central1
    ) / denominador

    escala = np.trace(cov) / cov.shape[0]

    cov = (
        (1.0 - REG_LDA) * cov
        + REG_LDA * escala * np.eye(cov.shape[0])
    )

    inversa = np.linalg.pinv(cov)

    w = inversa @ (media1 - media0)

    prior0 = max(float(np.mean(y == 0)), 1e-12)
    prior1 = max(float(np.mean(y == 1)), 1e-12)

    b = (
        -0.5 * (media1 + media0) @ w
        + math.log(prior1 / prior0)
    )

    return w, float(b)


def predizer_lda(features, w, b):
    scores = features @ w + b
    return (scores > 0).astype(int)


def binomial_superior(acertos, total):

    numerador = sum(
        math.comb(total, k)
        for k in range(acertos, total + 1)
    )

    return numerador / (2 ** total)


def metricas(y, pred):
    acuracia = float(
        np.mean(y == pred)
    )

    recall_left = float(
        np.mean(pred[y == 0] == 0)
    )

    recall_right = float(
        np.mean(pred[y == 1] == 1)
    )

    balanceada = (
        recall_left + recall_right
    ) / 2.0

    ll = int(
        np.sum(
            (y == 0)
            & (pred == 0)
        )
    )

    lr = int(
        np.sum(
            (y == 0)
            & (pred == 1)
        )
    )

    rl = int(
        np.sum(
            (y == 1)
            & (pred == 0)
        )
    )

    rr = int(
        np.sum(
            (y == 1)
            & (pred == 1)
        )
    )

    acertos = int(
        np.sum(y == pred)
    )

    p = binomial_superior(
        acertos,
        len(y),
    )

    return {
        "acuracia": acuracia,
        "balanceada": balanceada,
        "recall_left": recall_left,
        "recall_right": recall_right,
        "ll": ll,
        "lr": lr,
        "rl": rl,
        "rr": rr,
        "acertos": acertos,
        "total": len(y),
        "p_binomial": p,
    }


def validacao_por_run(X, y, grupos):
    predicoes = np.full(
        len(y),
        -1,
        dtype=int,
    )

    acuracias_run = {}

    for run_teste in sorted(
        np.unique(grupos)
    ):
        treino = grupos != run_teste
        teste = grupos == run_teste

        filtros = ajustar_csp(
            X[treino],
            y[treino],
        )

        feat_treino = transformar_csp(
            X[treino],
            filtros,
        )

        feat_teste = transformar_csp(
            X[teste],
            filtros,
        )

        w, b = ajustar_lda(
            feat_treino,
            y[treino],
        )

        pred = predizer_lda(
            feat_teste,
            w,
            b,
        )

        predicoes[teste] = pred

        acuracias_run[int(run_teste)] = float(
            np.mean(
                pred == y[teste]
            )
        )

    if np.any(predicoes < 0):
        raise RuntimeError(
            "Nem todos os trials receberam predição."
        )

    return predicoes, acuracias_run


def salvar_modelo(
    caminho,
    X,
    y,
    fs,
):
    filtros = ajustar_csp(
        X,
        y,
    )

    features = transformar_csp(
        X,
        filtros,
    )

    w, b = ajustar_lda(
        features,
        y,
    )

    np.savez(
        caminho,
        csp_filtros=filtros,
        lda_w=w,
        lda_b=np.asarray([b]),
        canais=np.asarray(
            CANAIS_SENSORIOMOTORES,
            dtype="U",
        ),
        fs=np.asarray([fs]),
        freq_min=np.asarray([FREQ_MIN]),
        freq_max=np.asarray([FREQ_MAX]),
        janela_inicio=np.asarray([JANELA_INICIO]),
        janela_fim=np.asarray([JANELA_FIM]),
    )


def analisar_arquivo(
    caminho,
    output_dir,
    ignorar_exclusoes=False,
):
    dados = ler_edf(caminho)

    X, y, grupos, ids, removidos = criar_dataset(
        dados,
        ignorar_exclusoes=ignorar_exclusoes,
    )

    predicoes, acuracias_run = validacao_por_run(
        X,
        y,
        grupos,
    )

    resultado = metricas(
        y,
        predicoes,
    )

    codigo = dados["codigo"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    arquivo_modelo = (
        output_dir
        / f"modelo_{codigo}_csp_lda.npz"
    )

    arquivo_resultado = (
        output_dir
        / f"resultado_{codigo}.txt"
    )

    salvar_modelo(
        arquivo_modelo,
        X,
        y,
        dados["fs"],
    )

    total_encontrado = len(y) + len(removidos)
    acuracia = resultado["acuracia"] * 100
    acima_acaso = acuracia - 50.0

    linhas = []

    linhas.append("=" * 56)
    linhas.append(f"Participante {codigo}")
    linhas.append("=" * 56)
    linhas.append("")
    linhas.append(
        f"Foram analisados {len(y)} de {total_encontrado} trials."
    )

    if removidos:
        texto_removidos = ", ".join(
            f"Run {run}, Trial {trial} ({rotulo})"
            for run, trial, rotulo in removidos
        )
        linhas.append(
            f"Foram excluídos {len(removidos)} trials por falha de aquisição:"
        )
        linhas.append(f"  {texto_removidos}")
    else:
        linhas.append(
            "Nenhum trial precisou ser excluído por falha de aquisição."
        )

    linhas.append("")
    linhas.append(
        "Treinamento: faixa de 8-30 Hz, CSP + LDA, "
        "com validação leave-one-run-out."
    )
    linhas.append(
        "Canais usados: "
        + ", ".join(CANAIS_SENSORIOMOTORES)
        + "."
    )

    linhas.append("")
    linhas.append(
        f"Resultado geral: {acuracia:.2f}% de acerto "
        f"({resultado['acertos']} de {resultado['total']} trials)."
    )

    if acima_acaso >= 0:
        linhas.append(
            f"Isso ficou {acima_acaso:.2f} pontos percentuais "
            "acima dos 50% esperados ao acaso."
        )
    else:
        linhas.append(
            f"Isso ficou {abs(acima_acaso):.2f} pontos percentuais "
            "abaixo dos 50% esperados ao acaso."
        )

    linhas.append("")
    linhas.append("Resultado por run:")

    for run, valor in acuracias_run.items():
        linhas.append(
            f"  Run {run}: {valor * 100:.2f}%"
        )

    linhas.append("")


    texto_resultado = "\n".join(linhas)

    arquivo_resultado.write_text(
        texto_resultado,
        encoding="utf-8",
    )

    print("\n" + texto_resultado)
    print("")
    print(f"Modelo salvo em: {arquivo_modelo}")
    print(f"Resumo salvo em: {arquivo_resultado}")

    return resultado


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Treina um baseline individual de imaginação motora "
            "LEFT x RIGHT a partir de EDF+."
        )
    )

    parser.add_argument(
        "edfs",
        nargs="+",
        help="Um ou mais arquivos EDF.",
    )

    parser.add_argument(
        "--output-dir",
        default="modelos_mi",
        help="Pasta para resultados e modelos.",
    )

    parser.add_argument(
        "--include-known-bad",
        action="store_true",
        help=(
            "Inclui também os trials conhecidos como atingidos "
            "por falhas de aquisição."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    for caminho in args.edfs:
        analisar_arquivo(
            Path(caminho),
            output_dir,
            ignorar_exclusoes=args.include_known_bad,
        )


if __name__ == "__main__":
    main()
