import csv
import glob
import json
import os.path
import pickle
import re
from io import StringIO

import chardet
import numpy as np
import pandas as pd

from mapping import DeviceMapping


COLUMN = [
    "type",
    "sequence",
    "timestamp",
    "taget_seq",
    "taget",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data",
]


def extract_csi_data(src: str, dst: str):
    f = open(src, "rb")
    r = f.read()
    f_encoding = chardet.detect(r).get("encoding")
    f.close()

    f_src = open(src, "r", encoding=f_encoding)
    f_dst = open(dst, "w")
    csv_writer = csv.writer(f_dst)
    csv_writer.writerow(COLUMN)

    while True:
        try:
            strings = f_src.readline()
            if not strings:
                break
        except UnicodeError:
            continue

        data_index = strings.find("CSI_DATA")
        if data_index != -1:
            try:
                data = StringIO(strings[data_index:])
                csv_reader = csv.reader(data)
                csi_data = next(csv_reader)

                if len(csi_data) != len(COLUMN):
                    raise ValueError(f"element number is not equal {len(COLUMN)}")

                try:
                    json.loads(csi_data[-1])
                except json.JSONDecodeError:
                    raise ValueError("data is not incomplete")

                csv_writer.writerow(csi_data)
            except ValueError:
                continue

    f_src.close()
    f_dst.close()
    print(f"Saved csi data to {dst}")


def split_csi_data(src: str, dst: str, mode="scan"):
    if mode == "scan":
        csv_list = glob.glob(os.path.join(src, "*.csv"))
        file_path = csv_list[0]
        df = pd.read_csv(file_path)
        df = df.to_csv(os.path.join(dst, "merge.csv"), index=False)

        for i in range(1, len(csv_list)):
            file_path = csv_list[i]
            df = pd.read_csv(file_path)
            df = df.to_csv(
                os.path.join(dst, "merge.csv"),
                index=False,
                header=False,
                mode="a+",
            )

        print(f"Merge csi data to {os.path.join(dst, 'merge.csv')}")

        df = pd.read_csv(os.path.join(dst, "merge.csv"))
        os.remove(os.path.join(dst, "merge.csv"))

    elif mode == "single":
        df = pd.read_csv(src)

    mac_groups = df.groupby(df["mac"])

    for mac_group in mac_groups:
        mac = mac_group[0].replace(":", "-")
        mac_group[1].to_csv(
            os.path.join(dst, mac + "-" + str(mac_group[1].shape[0]) + ".csv"),
            index=False,
        )
        print(
            f"Split csi data to {os.path.join(dst, mac +  '-' + str(mac_group[1].shape[0]) + '.csv')}"
        )


def parse_csi_data(src: str, dst: str):
    # only select useful subcarrier channel
    select_list = [i for i in range(6, 32)]
    select_list += [i for i in range(33, 59)]

    df_csv = pd.read_csv(src)

    raw_list = list()
    phase_list = list()
    amplitude_list = list()
    data_groups = df_csv["data"].groupby(df_csv["len"])

    for data_group in data_groups:
        if data_group[0] not in [128, 256, 376, 380, 384, 612]:
            continue

        df_data = data_group[1].to_frame()
        size_x = len(df_data.index)
        size_y = data_group[0] // 2
        array_data = np.zeros([size_x, size_y], dtype=np.complex64)
        for x, csi in enumerate(df_data.iloc):
            csi_raw_data = json.loads(csi["data"])

            if len(csi_raw_data) != data_group[0]:
                continue

            for y in range(0, len(csi_raw_data), 2):
                array_data[x][y // 2] = complex(csi_raw_data[y + 1], csi_raw_data[y])

        array_data = array_data[~(array_data == 0).all(1)]
        array_data = array_data[:, select_list]
        array_phase = np.zeros([array_data.shape[0], array_data.shape[1]])
        array_amplitude = np.zeros([array_data.shape[0], array_data.shape[1]])
        for i in range(array_data.shape[0]):
            array_phase[i] = np.unwrap(np.angle(array_data[i]))
            array_amplitude[i] = abs(array_data[i])

        # add list to concatenate
        raw_list.append(array_data)
        phase_list.append(array_phase)
        amplitude_list.append(array_amplitude)

    if not raw_list:
        return

    # concatenate numpy
    csi_raw = np.concatenate(raw_list, axis=0)
    csi_amplitude = np.concatenate(amplitude_list, axis=0)
    csi_phase = np.concatenate(phase_list, axis=0)

    with open(dst, "wb") as csi_pkl:
        pickle.dump(csi_raw, csi_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(csi_phase, csi_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(csi_amplitude, csi_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved csi data to {dst}")


def process_csi_data(log_src: str, to_csv=True, to_pkl=True, mode="scan"):
    if os.path.isfile(log_src):
        log_path = os.path.split(log_src)[0]
        csv_path = log_path.replace("log", "csv")
        pkl_path = csv_path.replace("csv", "pkl")
        log_list = [os.path.basename(log_src)]
    else:
        log_path = log_src
        csv_path = log_path.replace("log", "csv")
        pkl_path = csv_path.replace("csv", "pkl")
        log_list = os.listdir(log_path)

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    pattern = re.compile(r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})-(\d+).csv$")

    if mode == "single":
        if to_csv:
            for basename in log_list:
                csv_dst = basename.replace("log", "csv")
                extract_csi_data(
                    os.path.join(log_path, basename),
                    os.path.join(csv_path, csv_dst),
                )
                csv_split_path = os.path.splitext(basename)[0]
                if not os.path.exists(os.path.join(csv_path, csv_split_path)):
                    os.makedirs(os.path.join(csv_path, csv_split_path))
                split_csi_data(
                    os.path.join(csv_path, csv_dst),
                    os.path.join(csv_path, csv_split_path),
                    mode,
                )
                for f in os.listdir(os.path.join(csv_path, csv_split_path)):
                    if not pattern.match(f):
                        os.remove(os.path.join(csv_path, csv_split_path, f))
                try:
                    os.remove(
                        os.path.join(
                            csv_path,
                            csv_split_path,
                            "00-00-00-00-00-00.csv",
                        ),
                    )
                except FileNotFoundError:
                    pass
                os.remove(os.path.join(csv_path, csv_dst))

        if to_pkl:
            csv_list = os.listdir(csv_path)
            for basename in csv_list:
                if not os.path.exists(os.path.join(pkl_path, basename)):
                    os.makedirs(os.path.join(pkl_path, basename))
                for f in os.listdir(os.path.join(csv_path, basename)):
                    pkl_dst = f.replace("csv", "pkl")
                    parse_csi_data(
                        os.path.join(csv_path, basename, f),
                        os.path.join(pkl_path, basename, pkl_dst),
                    )
    elif mode == "scan":
        if to_csv:
            for basename in log_list:
                csv_dst = basename.replace("log", "csv")
                extract_csi_data(
                    os.path.join(log_path, basename),
                    os.path.join(csv_path, csv_dst),
                )
            split_csi_data(csv_path, csv_path, mode)
            for f in os.listdir(csv_path):
                if not pattern.match(f):
                    os.remove(os.path.join(csv_path, f))
            try:
                os.remove(os.path.join(csv_path, "00-00-00-00-00-00.csv"))
            except FileNotFoundError:
                pass

        if to_pkl:
            csv_list = os.listdir(csv_path)
            for basename in csv_list:
                pkl_dst = basename.replace("csv", "pkl")
                parse_csi_data(
                    os.path.join(csv_path, basename),
                    os.path.join(pkl_path, pkl_dst),
                )


def phase_scaling(x: np.ndarray) -> np.ndarray:
    scale = x.shape[-1] * np.pi
    return np.divide(x, scale)


def amplitude_scaling(x: np.ndarray) -> np.ndarray:
    scale = 128 * np.sqrt(2)
    return np.divide(x, scale)


def load_labeled_data(
    pkl_src: str,
    pattern: str,
    sample_per_device: int = None,
) -> dict:
    device_mapping = DeviceMapping(pattern)

    ds = {"sc{}".format(i): {"pha": [], "amp": []} for i in range(52)}
    labels = list()

    if os.path.isfile(pkl_src):
        token = device_mapping.get_Token(
            os.path.splitext(os.path.basename(pkl_src))[0][:17]
            .upper()
            .replace("-", ":")
        )
        if token == -1:
            return

        with open(pkl_src, "rb") as csi_pkl:
            raw = pickle.load(csi_pkl)
            phase = pickle.load(csi_pkl)
            amplitude = pickle.load(csi_pkl)

            N = raw.shape[0]

            if sample_per_device is not None:
                samples = np.linspace(
                    start=0,
                    stop=N - 1,
                    num=sample_per_device,
                    dtype=np.int32,
                )
                raw = raw[samples]
                phase = phase[samples]
                amplitude = amplitude[samples]

                N = sample_per_device

            phase = phase_scaling(phase).astype(np.float32)
            amplitude = amplitude_scaling(amplitude).astype(np.float32)

            labels.append(token * np.ones((N,)).astype(np.int64))

            pha_chunks = np.array_split(phase, 52, axis=1)
            amp_chunks = np.array_split(amplitude, 52, axis=1)

            for i, (pha_chunk, amp_chunk) in enumerate(zip(pha_chunks, amp_chunks)):
                ds["sc{}".format(i)]["pha"].append(pha_chunk)
                ds["sc{}".format(i)]["amp"].append(amp_chunk)

    else:
        for root, dirs, files in os.walk(pkl_src, topdown=False):
            for name in files:
                token = device_mapping.get_Token(
                    os.path.splitext(name)[0][:17].upper().replace("-", ":")
                )
                if token == -1:
                    continue

                with open(os.path.join(root, name), "rb") as csi_pkl:
                    raw = pickle.load(csi_pkl)
                    phase = pickle.load(csi_pkl)
                    amplitude = pickle.load(csi_pkl)

                    N = raw.shape[0]

                    if sample_per_device is not None:
                        if sample_per_device < 0:
                            raw = raw[sample_per_device:]
                            phase = phase[sample_per_device:]
                            amplitude = amplitude[sample_per_device:]

                            N = -sample_per_device
                        else:
                            raw = raw[:sample_per_device]
                            phase = phase[:sample_per_device]
                            amplitude = amplitude[:sample_per_device]

                            N = sample_per_device

                    phase = phase_scaling(phase).astype(np.float32)
                    amplitude = amplitude_scaling(amplitude).astype(np.float32)

                    labels.append(token * np.ones((N,)).astype(np.int64))

                    pha_chunks = np.array_split(phase, 52, axis=1)
                    amp_chunks = np.array_split(amplitude, 52, axis=1)

                    for i, (pha_chunk, amp_chunk) in enumerate(
                        zip(pha_chunks, amp_chunks)
                    ):
                        ds["sc{}".format(i)]["pha"].append(pha_chunk)
                        ds["sc{}".format(i)]["amp"].append(amp_chunk)

    return ds, labels
