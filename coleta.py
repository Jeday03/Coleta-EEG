import argparse
import math
import random
import re
import tkinter as tk
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

try:
    import winsound
except ImportError:
    winsound = None

try:
    import pyedflib
except ImportError:
    pyedflib = None


DURACAO_CRUZ = 3
DURACAO_PREPARACAO = 2
DURACAO_IMAGETICA = 4
DURACAO_DESCANSO = 1

MARKERS = {
    "NewRun": 10.0,
    "NewTrial": 20.0,
    "LeftPrepare": 31.0,
    "RightPrepare": 32.0,
    "LeftExec": 41.0,
    "RightExec": 42.0,
    "Resting": 50.0,
}

MARKER_NAMES = {value: name for name, value in MARKERS.items()}

EDF_CHANNEL_LABELS = [
    "Fp1", "Fz", "C3", "C4",
    "T5", "T6", "Cz", "Pz",
    "F7", "F8", "F3", "F4",
    "T3", "T4", "P3", "P4",
]


parser = argparse.ArgumentParser()
parser.add_argument("--serial", default="COM3")
parser.add_argument("--output-dir", default="coletas")
parser.add_argument("--test", action="store_true")
parser.add_argument("--hide-status", action="store_true")
parser.add_argument("--no-pauses", action="store_true")
args = parser.parse_args()


if args.test:
    NUM_RUNS = 1
    TRIALS_POR_RUN = 4
else:
    NUM_RUNS = 4
    TRIALS_POR_RUN = 40

TRIALS_POR_CLASSE = TRIALS_POR_RUN // 2
TOTAL_TRIALS = NUM_RUNS * TRIALS_POR_RUN


board_id = BoardIds.CYTON_DAISY_BOARD.value

params = BrainFlowInputParams()
params.serial_port = args.serial

board = BoardShim(board_id, params)

eeg_ch = BoardShim.get_eeg_channels(board_id)
ts_ch = BoardShim.get_timestamp_channel(board_id)
marker_ch = BoardShim.get_marker_channel(board_id)
package_ch = BoardShim.get_package_num_channel(board_id)
fs = BoardShim.get_sampling_rate(board_id)


root = tk.Tk()
root.configure(bg="black")
root.attributes("-fullscreen", True)

board_prepared = False
board_started = False
coleta_iniciando = False
finalizado = False
em_pausa = False

run_index = 0
trial_index = 0
sample_index = 0

session_started = None
output_edf = None
run_orders = []

recorded_events = Counter()
event_records = []
package_numbers = []
sample_timestamps = []
first_timestamp = None
last_timestamp = None

eeg_buffers = [[] for _ in eeg_ch]

participant = {
    "patient_code": "",
    "birthdate": None,
    "sex": "",
    "handedness": "",
}


def sanitizar_codigo(codigo):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", codigo.strip())


def formatar_data_nascimento(event=None):

    valor = nascimento_entry.get()
    digitos = re.sub(r"\D", "", valor)[:8]

    if len(digitos) <= 2:
        formatado = digitos
    elif len(digitos) <= 4:
        formatado = f"{digitos[:2]}/{digitos[2:]}"
    else:
        formatado = (
            f"{digitos[:2]}/"
            f"{digitos[2:4]}/"
            f"{digitos[4:]}"
        )

    if valor != formatado:
        nascimento_entry.delete(0, tk.END)
        nascimento_entry.insert(0, formatado)


def escolher_caminho_edf(output_dir, codigo):

    candidato = output_dir / f"{codigo}.edf"

    if not candidato.exists():
        return candidato

    contador = 1

    while True:
        candidato = output_dir / f"{codigo}_{contador}.edf"

        if not candidato.exists():
            return candidato

        contador += 1


def criar_ordens():
    ordens = []

    for _ in range(NUM_RUNS):
        trials = (
            ["LEFT"] * TRIALS_POR_CLASSE
            + ["RIGHT"] * TRIALS_POR_CLASSE
        )
        random.shuffle(trials)
        ordens.append(trials)

    return ordens


def marcar(nome):
    board.insert_marker(MARKERS[nome])


def beep():
    if winsound is not None:
        winsound.Beep(1000, 150)
    else:
        root.bell()


def atualizar_status():
    if args.hide_status:
        status_label.place_forget()
        return

    if em_pausa:
        texto = f"Run {run_index + 1}/{NUM_RUNS}"
    else:
        texto = (
            f"Run {run_index + 1}/{NUM_RUNS}\n"
            f"Trial {trial_index + 1}/{TRIALS_POR_RUN}"
        )

    status_label.config(text=texto)
    status_label.place(relx=0.985, rely=0.025, anchor="ne")


def flush_data():
    global sample_index
    global first_timestamp
    global last_timestamp

    if not board_started:
        return

    data = board.get_board_data()

    for sample in range(data.shape[1]):
        timestamp = float(data[ts_ch][sample])
        package_num = int(round(data[package_ch][sample])) % 256
        marker = float(data[marker_ch][sample])
        event = MARKER_NAMES.get(marker, "")

        package_numbers.append(package_num)
        sample_timestamps.append(timestamp)

        if first_timestamp is None:
            first_timestamp = timestamp

        last_timestamp = timestamp

        for idx, channel in enumerate(eeg_ch):
            eeg_buffers[idx].append(
                float(data[channel][sample])
            )

        if event:
            recorded_events[event] += 1
            event_records.append(
                {
                    "sample_index": sample_index,
                    "timestamp": timestamp,
                    "event": event,
                    "marker": marker,
                }
            )

        sample_index += 1


def validar_dados_participante():
    global board_prepared
    global run_orders
    global output_edf

    codigo = sanitizar_codigo(codigo_entry.get())
    nascimento = nascimento_entry.get().strip()
    sexo = sexo_var.get()
    lateralidade = lateralidade_var.get()

    if not codigo:
        erro_label.config(
            text="Informe o código do participante."
        )
        return

    try:
        birthdate = datetime.strptime(
            nascimento,
            "%d/%m/%Y",
        ).date()
    except ValueError:
        erro_label.config(
            text="Informe a data de nascimento como DD/MM/AAAA."
        )
        return

    if birthdate >= datetime.now().date():
        erro_label.config(
            text="Informe uma data de nascimento válida."
        )
        return

    if sexo not in {"male", "female"}:
        erro_label.config(text="Selecione o sexo.")
        return

    if lateralidade not in {"right", "left", "both"}:
        erro_label.config(text="Selecione a lateralidade.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_edf = escolher_caminho_edf(
        output_dir,
        codigo,
    )

    participant["patient_code"] = codigo
    participant["birthdate"] = birthdate
    participant["sex"] = sexo
    participant["handedness"] = lateralidade

    erro_label.config(text="")
    setup_frame.place_forget()

    if pyedflib is None:
        setup_frame.place(relx=0.5, rely=0.5, anchor="center")
        erro_label.config(
            text=(
                "pyedflib não está instalado. "
                "Execute: pip install pyedflib"
            )
        )
        return

    try:
        print("Abrindo Cyton + Daisy...")
        board.prepare_session()
        board_prepared = True
    except Exception as e:
        setup_frame.place(relx=0.5, rely=0.5, anchor="center")
        erro_label.config(
            text=f"Não foi possível abrir a placa: {e}"
        )
        return

    run_orders = criar_ordens()

    print("Placa preparada")
    print("Participante:", participant["patient_code"])
    print("Frequência:", fs, "Hz")
    print("Canais EEG:", len(eeg_ch))
    print("Modo:", "TESTE" if args.test else "COLETA")
    print("EDF:", output_edf)

    instrucoes_label.config(
        text=(
            "Permaneça o mais imóvel possível e evite piscar "
            "durante as tarefas.\n\n"
            "Imagine o movimento de abrir e fechar a mão "
            "indicada pela seta.\n\n"
            "Pressione ESPAÇO para iniciar."
        )
    )
    instrucoes_label.place(
        relx=0.5,
        rely=0.5,
        anchor="center",
    )


def iniciar_primeira_run(event=None):
    global board_started
    global coleta_iniciando

    if (
        not board_prepared
        or board_started
        or em_pausa
        or finalizado
        or coleta_iniciando
    ):
        return

    if not instrucoes_label.winfo_ismapped():
        return

    instrucoes_label.config(text="Iniciando...")
    coleta_iniciando = True

    try:
        board.start_stream(450000)
        board_started = True
    except Exception as e:
        coleta_iniciando = False
        instrucoes_label.config(
            text=f"Não foi possível iniciar a aquisição.\n\n{e}"
        )
        return

    root.after(1000, preparar_inicio)


def preparar_inicio():
    global session_started
    global coleta_iniciando

    board.get_board_data()

    session_started = datetime.now()
    coleta_iniciando = False
    instrucoes_label.place_forget()

    print("Aquisição iniciada")

    iniciar_run()


def iniciar_run():
    global trial_index
    global em_pausa

    em_pausa = False
    trial_index = 0

    marcar("NewRun")

    print(f"\nRun {run_index + 1}/{NUM_RUNS}")
    print("Ordem:", run_orders[run_index])

    iniciar_trial()


def iniciar_trial():
    trial = run_orders[run_index][trial_index]

    marcar("NewTrial")

    stimulus_label.config(text="+", fg="white")
    stimulus_label.place(
        relx=0.5,
        rely=0.5,
        anchor="center",
    )
    atualizar_status()

    print(
        f"Run {run_index + 1}/{NUM_RUNS} | "
        f"Trial {trial_index + 1}/{TRIALS_POR_RUN} | "
        f"{trial}"
    )

    root.after(2000, beep)
    root.after(
        int(DURACAO_CRUZ * 1000),
        mostrar_preparacao,
    )


def mostrar_preparacao():
    trial = run_orders[run_index][trial_index]

    if trial == "LEFT":
        stimulus_label.config(text="←", fg="white")
        marcar("LeftPrepare")
    else:
        stimulus_label.config(text="→", fg="white")
        marcar("RightPrepare")

    root.after(
        int(DURACAO_PREPARACAO * 1000),
        iniciar_imagetica,
    )


def iniciar_imagetica():
    trial = run_orders[run_index][trial_index]

    stimulus_label.config(fg="red")

    if trial == "LEFT":
        marcar("LeftExec")
    else:
        marcar("RightExec")

    root.after(
        int(DURACAO_IMAGETICA * 1000),
        iniciar_descanso,
    )


def iniciar_descanso():
    stimulus_label.config(text="", fg="white")
    marcar("Resting")

    root.after(
        int(DURACAO_DESCANSO * 1000),
        proximo_trial,
    )


def proximo_trial():
    global trial_index

    trial_index += 1

    if trial_index >= TRIALS_POR_RUN:
        finalizar_run()
        return

    iniciar_trial()


def finalizar_run():
    global em_pausa
    global run_index

    flush_data()

    if run_index + 1 >= NUM_RUNS:
        finalizar_experimento()
        return

    if args.no_pauses:
        run_index += 1
        iniciar_run()
        return

    em_pausa = True
    stimulus_label.place_forget()
    atualizar_status()

    pause_label.config(
        text=(
            f"Run {run_index + 1} concluída.\n\n"
            "Descanse o tempo que precisar.\n"
            "Quando estiver pronto, pressione ESPAÇO."
        )
    )
    pause_label.place(
        relx=0.5,
        rely=0.5,
        anchor="center",
    )


def continuar_apos_pausa(event=None):
    global run_index
    global em_pausa

    if not em_pausa or finalizado:
        return

    pause_label.place_forget()

    run_index += 1
    em_pausa = False

    iniciar_run()


def calcular_validacao_aquisicao():
    esperado_eventos = {
        "NewRun": NUM_RUNS,
        "NewTrial": TOTAL_TRIALS,
        "LeftPrepare": NUM_RUNS * TRIALS_POR_CLASSE,
        "RightPrepare": NUM_RUNS * TRIALS_POR_CLASSE,
        "LeftExec": NUM_RUNS * TRIALS_POR_CLASSE,
        "RightExec": NUM_RUNS * TRIALS_POR_CLASSE,
        "Resting": TOTAL_TRIALS,
    }

    eventos_ok = all(
        recorded_events.get(evento, 0) == quantidade
        for evento, quantidade in esperado_eventos.items()
    )

    taxa_efetiva = None
    taxa_ok = False

    if (
        sample_index > 1
        and first_timestamp is not None
        and last_timestamp is not None
        and last_timestamp > first_timestamp
    ):
        duracao = last_timestamp - first_timestamp
        taxa_efetiva = (sample_index - 1) / duracao
        taxa_ok = abs(taxa_efetiva - fs) <= fs * 0.03

    descontinuidades = []

    for i in range(1, len(package_numbers)):
        anterior = package_numbers[i - 1]
        atual = package_numbers[i]
        passo = (atual - anterior) % 256

        if passo != 2:
            descontinuidades.append(
                {
                    "sample_before": i - 1,
                    "sample_after": i,
                    "package_before": anterior,
                    "package_after": atual,
                    "delta_mod_256": passo,
                }
            )

    pacotes_ok = len(descontinuidades) == 0

    limite_gap = (1.0 / fs) * 4
    gaps_grandes = []

    for i in range(1, len(sample_timestamps)):
        gap = (
            sample_timestamps[i]
            - sample_timestamps[i - 1]
        )

        if gap > limite_gap:
            gaps_grandes.append(gap)

    intervalos_esperados = {
        ("NewTrial", "LeftPrepare"): DURACAO_CRUZ,
        ("NewTrial", "RightPrepare"): DURACAO_CRUZ,
        ("LeftPrepare", "LeftExec"): DURACAO_PREPARACAO,
        ("RightPrepare", "RightExec"): DURACAO_PREPARACAO,
        ("LeftExec", "Resting"): DURACAO_IMAGETICA,
        ("RightExec", "Resting"): DURACAO_IMAGETICA,
        ("Resting", "NewTrial"): DURACAO_DESCANSO,
    }

    intervalos = defaultdict(list)

    for anterior, atual in zip(
        event_records,
        event_records[1:],
    ):
        chave = (
            anterior["event"],
            atual["event"],
        )

        if chave not in intervalos_esperados:
            continue

        amostras = (
            atual["sample_index"]
            - anterior["sample_index"]
        )

        esperado_s = intervalos_esperados[chave]
        esperado_amostras = esperado_s * fs
        tolerancia = max(
            5,
            esperado_amostras * 0.04,
        )

        ok = (
            abs(amostras - esperado_amostras)
            <= tolerancia
        )

        intervalos[chave].append(
            {
                "samples": amostras,
                "seconds": amostras / fs,
                "ok": ok,
            }
        )

    intervalos_ok = all(
        item["ok"]
        for valores in intervalos.values()
        for item in valores
    )

    return {
        "expected_event_counts": esperado_eventos,
        "events_ok": eventos_ok,
        "effective_sampling_rate_hz": taxa_efetiva,
        "sampling_rate_ok": taxa_ok,
        "packet_sequence_ok": pacotes_ok,
        "packet_discontinuities": descontinuidades,
        "large_timestamp_gaps": gaps_grandes,
        "intervals": intervalos,
        "intervals_ok": intervalos_ok,
        "all_ok": (
            eventos_ok
            and taxa_ok
            and pacotes_ok
            and intervalos_ok
        ),
    }


def imprimir_validacao_aquisicao(validacao):
    print("\nValidação dos eventos:")

    for evento, esperado in (
        validacao["expected_event_counts"].items()
    ):
        gravado = recorded_events.get(evento, 0)
        status = (
            "OK"
            if gravado == esperado
            else "ERRO"
        )
        print(
            f"{evento}: "
            f"{gravado}/{esperado} [{status}]"
        )

    print("\nTaxa de aquisição:")

    taxa = validacao["effective_sampling_rate_hz"]

    if taxa is None:
        print("Não foi possível calcular. [ERRO]")
    else:
        status = (
            "OK"
            if validacao["sampling_rate_ok"]
            else "ERRO"
        )
        print(
            f"{taxa:.2f} Hz "
            f"(esperado: {fs} Hz) [{status}]"
        )

    print("\nContinuidade dos pacotes:")

    perdas = len(
        validacao["packet_discontinuities"]
    )
    status = (
        "OK"
        if validacao["packet_sequence_ok"]
        else "ERRO"
    )

    print(
        f"Descontinuidades: {perdas} [{status}]"
    )

    if perdas:
        for item in (
            validacao["packet_discontinuities"][:10]
        ):
            print(
                "  amostra "
                f"{item['sample_before']} -> "
                f"{item['sample_after']}: "
                f"pacote {item['package_before']} -> "
                f"{item['package_after']} "
                f"(delta {item['delta_mod_256']})"
            )

    print("\nGaps de timestamp:")

    gaps = validacao["large_timestamp_gaps"]

    if gaps:
        print(
            f"Gaps maiores que {4 / fs:.3f}s: "
            f"{len(gaps)} [AVISO]"
        )
        print(
            "Maior gap:",
            f"{max(gaps):.4f}s",
        )
    else:
        print(
            f"Gaps maiores que {4 / fs:.3f}s: "
            "0 [OK]"
        )

    print("\nIntervalos do protocolo:")

    for chave, valores in (
        validacao["intervals"].items()
    ):
        if not valores:
            continue

        segundos = [
            item["seconds"]
            for item in valores
        ]

        status = (
            "OK"
            if all(
                item["ok"]
                for item in valores
            )
            else "ERRO"
        )

        print(
            f"{chave[0]}->{chave[1]}: "
            f"mín {min(segundos):.3f}s | "
            f"máx {max(segundos):.3f}s "
            f"[{status}]"
        )

    if validacao["all_ok"]:
        print("\nVALIDAÇÃO DA AQUISIÇÃO: OK")
    else:
        print("\nVALIDAÇÃO DA AQUISIÇÃO: ERRO")


def exportar_edf():
    sinais = [
        np.asarray(
            signal,
            dtype=np.float64,
        )
        for signal in eeg_buffers
    ]

    n_samples = len(sinais[0])

    writer_edf = None

    try:
        writer_edf = pyedflib.EdfWriter(
            str(output_edf),
            len(EDF_CHANNEL_LABELS),
            file_type=pyedflib.FILETYPE_EDFPLUS,
        )

        writer_edf.setPatientCode(
            participant["patient_code"]
        )

        sexo_edf = (
            1
            if participant["sex"] == "male"
            else 0
        )

        if hasattr(writer_edf, "setSex"):
            writer_edf.setSex(sexo_edf)
        elif hasattr(writer_edf, "setGender"):
            writer_edf.setGender(sexo_edf)

        writer_edf.setBirthdate(
            participant["birthdate"]
        )

        writer_edf.setPatientAdditional(
            f"hand={participant['handedness']}"
        )

        if hasattr(writer_edf, "setEquipment"):
            writer_edf.setEquipment(
                "OpenBCI Cyton + Daisy"
            )

        if session_started is not None:
            writer_edf.setStartdatetime(
                session_started
            )

        global_min = int(
            math.floor(
                min(
                    float(np.min(signal))
                    for signal in sinais
                )
            )
        )

        global_max = int(
            math.ceil(
                max(
                    float(np.max(signal))
                    for signal in sinais
                )
            )
        )

        if global_min == global_max:
            global_min -= 1
            global_max += 1

        signal_headers = []

        for label in EDF_CHANNEL_LABELS:
            signal_headers.append(
                {
                    "label": label,
                    "dimension": "uV",
                    "sample_frequency": fs,
                    "physical_min": global_min,
                    "physical_max": global_max,
                    "digital_min": -32767,
                    "digital_max": 32767,
                    "transducer": "",
                    "prefilter": "",
                }
            )

        writer_edf.setSignalHeaders(
            signal_headers
        )

        writer_edf.writeSamples(sinais)

        for event in event_records:
            onset = (
                event["sample_index"]
                / float(fs)
            )

            if onset == 0:
                onset = 0.0001

            writer_edf.writeAnnotation(
                onset,
                0,
                event["event"],
            )

        writer_edf.close()
        writer_edf = None

        return validar_edf(n_samples)

    except Exception:
        if writer_edf is not None:
            try:
                writer_edf.close()
            except Exception:
                pass
        raise


def validar_edf(n_samples_originais):
    reader = pyedflib.EdfReader(
        str(output_edf)
    )

    try:
        labels = list(
            reader.getSignalLabels()
        )

        frequencias = [
            float(value)
            for value in (
                reader.getSampleFrequencies()
            )
        ]

        nsamples = [
            int(value)
            for value in reader.getNSamples()
        ]

        onsets, durations, descriptions = (
            reader.readAnnotations()
        )

        descriptions = [
            str(value)
            for value in descriptions
        ]

        eventos_esperados = [
            item["event"]
            for item in event_records
        ]

        esperado_edf = (
            math.ceil(
                n_samples_originais / fs
            )
            * fs
        )

        padding = (
            esperado_edf
            - n_samples_originais
        )

        canais_ok = (
            reader.signals_in_file
            == len(EDF_CHANNEL_LABELS)
        )

        labels_ok = (
            labels
            == EDF_CHANNEL_LABELS
        )

        frequencias_ok = all(
            abs(freq - fs) < 1e-9
            for freq in frequencias
        )

        amostras_ok = all(
            value == esperado_edf
            for value in nsamples
        )

        anotacoes_ok = (
            descriptions
            == eventos_esperados
        )

        tudo_ok = all(
            [
                canais_ok,
                labels_ok,
                frequencias_ok,
                amostras_ok,
                anotacoes_ok,
            ]
        )

        print("\nValidação do EDF:")
        print(
            f"Canais: {reader.signals_in_file}/16 "
            f"[{'OK' if canais_ok else 'ERRO'}]"
        )
        print(
            "Nomes dos canais: "
            f"[{'OK' if labels_ok else 'ERRO'}]"
        )
        print(
            "Frequência dos canais: "
            f"[{'OK' if frequencias_ok else 'ERRO'}]"
        )
        print(
            "Amostras por canal: "
            f"{nsamples[0]} "
            f"[{'OK' if amostras_ok else 'ERRO'}]"
        )
        print(
            "Anotações: "
            f"{len(descriptions)}/"
            f"{len(eventos_esperados)} "
            f"[{'OK' if anotacoes_ok else 'ERRO'}]"
        )

        if padding:
            print(
                "Preenchimento do último registro EDF: "
                f"{padding} amostras "
                f"({padding / fs:.3f}s)"
            )

        if tudo_ok:
            print("\nVALIDAÇÃO DO EDF: OK")
        else:
            print("\nVALIDAÇÃO DO EDF: ERRO")

        return tudo_ok

    finally:
        reader.close()


def finalizar_experimento():
    global finalizado
    global board_started
    global board_prepared

    if finalizado:
        return

    finalizado = True
    stimulus_label.place_forget()
    status_label.place_forget()

    flush_data()

    board.stop_stream()
    board.release_session()

    board_started = False
    board_prepared = False

    validacao = calcular_validacao_aquisicao()
    imprimir_validacao_aquisicao(
        validacao
    )

    if not validacao["all_ok"]:
        print(
            "\nATENÇÃO: a aquisição apresentou falhas "
            "nos critérios de validação."
        )
        print(
            "O EDF será salvo normalmente com todos os dados "
            "que foram recebidos."
        )

    try:
        edf_ok = exportar_edf()
    except Exception as e:
        print("\nErro ao gerar EDF:", e)
        edf_ok = False

    if not edf_ok:
        try:
            if output_edf.exists():
                output_edf.unlink()
        except Exception:
            pass

        print(
            "\nFalha estrutural na geração do EDF. "
            "O arquivo foi removido."
        )

        fim_label.config(
            text=(
                "FIM\n\n"
                "Não foi possível salvar o EDF."
            )
        )
        fim_label.place(
            relx=0.5,
            rely=0.5,
            anchor="center",
        )
        return

    print("\nColeta concluída.")
    print("EDF:", output_edf)

    if not validacao["all_ok"]:
        print(
            "Observação: houve avisos/falhas na validação "
            "da aquisição; consulte o terminal."
        )

    fim_label.config(
        text=(
            "FIM\n\n"
            "Coleta salva com sucesso."
        )
    )

    fim_label.place(
        relx=0.5,
        rely=0.5,
        anchor="center",
    )


def abortar(event=None):
    global finalizado
    global board_started
    global board_prepared

    if finalizado:
        root.destroy()
        return

    finalizado = True

    print("\nColeta interrompida.")

    try:
        if board_started:
            board.stop_stream()
            board_started = False

        if board_prepared:
            board.release_session()
            board_prepared = False
    except Exception as e:
        print(
            "Erro ao encerrar a placa:",
            e,
        )

    print("Nenhum EDF foi salvo.")
    root.destroy()


def tratar_espaco(event=None):
    if em_pausa:
        continuar_apos_pausa()
    else:
        iniciar_primeira_run()


setup_frame = tk.Frame(
    root,
    bg="black",
)

titulo = tk.Label(
    setup_frame,
    text="Coleta EEG",
    font=("Arial", 28),
    fg="white",
    bg="black",
)
titulo.grid(
    row=0,
    column=0,
    columnspan=2,
    pady=(0, 30),
)

tk.Label(
    setup_frame,
    text="Código do participante:",
    font=("Arial", 16),
    fg="white",
    bg="black",
).grid(
    row=1,
    column=0,
    sticky="e",
    padx=10,
    pady=8,
)

codigo_entry = tk.Entry(
    setup_frame,
    font=("Arial", 16),
    width=18,
)
codigo_entry.grid(
    row=1,
    column=1,
    padx=10,
    pady=8,
)

tk.Label(
    setup_frame,
    text="Data de nascimento:",
    font=("Arial", 16),
    fg="white",
    bg="black",
).grid(
    row=2,
    column=0,
    sticky="e",
    padx=10,
    pady=8,
)

nascimento_entry = tk.Entry(
    setup_frame,
    font=("Arial", 16),
    width=18,
)
nascimento_entry.grid(
    row=2,
    column=1,
    padx=10,
    pady=8,
)

nascimento_entry.bind(
    "<KeyRelease>",
    formatar_data_nascimento,
)

tk.Label(
    setup_frame,
    text="DD/MM/AAAA",
    font=("Arial", 10),
    fg="#888888",
    bg="black",
).grid(
    row=2,
    column=2,
    sticky="w",
)

tk.Label(
    setup_frame,
    text="Sexo:",
    font=("Arial", 16),
    fg="white",
    bg="black",
).grid(
    row=3,
    column=0,
    sticky="e",
    padx=10,
    pady=8,
)

sexo_var = tk.StringVar(value="")

sexo_menu = tk.OptionMenu(
    setup_frame,
    sexo_var,
    "male",
    "female",
)
sexo_menu.config(
    font=("Arial", 14),
    width=14,
)
sexo_menu.grid(
    row=3,
    column=1,
    padx=10,
    pady=8,
)

tk.Label(
    setup_frame,
    text="Lateralidade:",
    font=("Arial", 16),
    fg="white",
    bg="black",
).grid(
    row=4,
    column=0,
    sticky="e",
    padx=10,
    pady=8,
)

lateralidade_var = tk.StringVar(
    value=""
)

lateralidade_menu = tk.OptionMenu(
    setup_frame,
    lateralidade_var,
    "right",
    "left",
    "both",
)
lateralidade_menu.config(
    font=("Arial", 14),
    width=14,
)
lateralidade_menu.grid(
    row=4,
    column=1,
    padx=10,
    pady=8,
)

erro_label = tk.Label(
    setup_frame,
    text="",
    font=("Arial", 12),
    fg="red",
    bg="black",
)
erro_label.grid(
    row=5,
    column=0,
    columnspan=3,
    pady=5,
)

start_button = tk.Button(
    setup_frame,
    text="Preparar coleta",
    command=validar_dados_participante,
    font=("Arial", 15),
    padx=18,
    pady=8,
)
start_button.grid(
    row=6,
    column=0,
    columnspan=3,
    pady=20,
)


stimulus_label = tk.Label(
    root,
    text="",
    font=("Arial", 100),
    fg="white",
    bg="black",
)

status_label = tk.Label(
    root,
    text="",
    font=("Arial", 11),
    justify="right",
    fg="#777777",
    bg="black",
)

instrucoes_label = tk.Label(
    root,
    text="",
    font=("Arial", 22),
    justify="center",
    fg="white",
    bg="black",
)

pause_label = tk.Label(
    root,
    text="",
    font=("Arial", 26),
    justify="center",
    fg="white",
    bg="black",
)

fim_label = tk.Label(
    root,
    text="",
    font=("Arial", 32),
    justify="center",
    fg="white",
    bg="black",
)


root.bind(
    "<Escape>",
    abortar,
)
root.bind(
    "<space>",
    tratar_espaco,
)

setup_frame.place(
    relx=0.5,
    rely=0.5,
    anchor="center",
)

codigo_entry.focus_set()

root.mainloop()