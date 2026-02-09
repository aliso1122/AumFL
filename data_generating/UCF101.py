import cv2
from data_generating.utils import basic_dataset
from torch.utils.data import DataLoader
from control.Enums import ParametersForDataSplitType, generate_client_id
import os
import torch
import numpy as np
from data_generating.utils import (
    allocate_data_idx,
    data_generating_store,
    get_datadir,
    init_data_dir,
)
import torch.nn as nn
import random
import torchaudio
from torchvision.models import inception_v3, Inception_V3_Weights
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def frames_extraction(video_path, length=40, image_height=299, image_width=299):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps_video = video_reader.get(cv2.CAP_PROP_FPS)
    skip_frames_window = max(int(video_frames_count / length), 1)
    for frame_counter in range(length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255
        transposed_frame = np.transpose(normalized_frame, (2, 0, 1))
        frames_list.append(transposed_frame)
    video_reader.release()
    return frames_list


def UCF101_create_dataloaders(args):
    def create(
        data_peice_paths,
        Dk_len_,
        dataloader_type,
        M_num=2,
    ):
        random.shuffle(data_peice_paths)
        # data_peice_paths = data_peice_paths[100:]
        data_id_ = [data_peice_paths, data_peice_paths]
        D_kps_id_ = allocate_data_idx(
            client_nums_for_Dk=split_parameters.client_nums_for_Dk,
            data_len=len(data_peice_paths),
            M_sets=split_parameters.M_sets,
            M_num=M_num,
            Dk_len=Dk_len_,
            data_idx_list=data_id_,
        )

        for k, D_k_ in enumerate(D_kps_id_, 0):
            for p, D_p_ in enumerate(D_k_, 0):
                data_X = []
                D_p_X_, D_p_Y_ = D_p_
                data_Y = list(map(lambda x: labels_of_mm_data.index(x), D_p_Y_))
                data_nums = len(data_Y)
                # print(D_p_Y_[:100])
                # print(D_p_X_[0][:100])

                for modality_idx, modality in enumerate(split_parameters.M_sets[k]):
                    features = torch.tensor([]).to(device)
                    for count, data_path in enumerate(D_p_X_[modality_idx], 1):
                        if modality == 0:  # video
                            frame_list = frames_extraction(
                                f"{raw_data_path}/{data_path}"
                            )
                            # if len(frame_list) != 40:
                            #     print(data_path, len(frame_list))
                            # sys.stdout.write(
                            #     f"\r{dataloader_type}:preprocessing UCF videos for client {k},{p} and m{modality} [{count}/{data_nums}]"
                            # )
                            # continue
                            video_feature = torch.tensor(
                                np.array(frame_list), dtype=torch.float32
                            )
                            # print(video_feature.shape)
                            with torch.no_grad():
                                video_feature = img_vector_extractor(
                                    video_feature.to(device)
                                ).logits

                            video_feature = video_feature.unsqueeze(0)
                            features = torch.concatenate(
                                [features, video_feature], dim=0
                            )
                        elif modality == 1:  # audio
                            audio, sr = torchaudio.load(f"{raw_data_path}/{data_path}")

                            if audio.shape[0] != 1:
                                audio = torch.mean(audio, dim=0).unsqueeze(0)

                            if sr != 16000:
                                transform_model = torchaudio.transforms.Resample(
                                    sr, 16000
                                )
                                audio = transform_model(audio)
                            audio_feature = torchaudio.compliance.kaldi.fbank(
                                waveform=torch.Tensor(audio),
                                sample_frequency=16000,
                                frame_length=40,
                                frame_shift=20,
                                num_mel_bins=160,
                                window_type="hamming",
                            )
                            audio_feature = (
                                audio_feature - torch.mean(audio_feature, axis=0)
                            ) / (torch.std(audio_feature, axis=0) + 1e-5)
                            gap = len(audio_feature) // 40
                            audio_feature = audio_feature[::gap][:40]
                            features = torch.concatenate(
                                [features, audio_feature.unsqueeze(0)], dim=0
                            )

                        sys.stdout.write(
                            f"\r{dataloader_type}:preprocessing UCF videos for client {k},{p} and m{modality} [{count}/{data_nums}]"
                        )
                        sys.stdout.flush()
                    print("finished")
                    data_X.append(features)
                client_id = generate_client_id(k, p)
                if client_id not in dataloaders_for_clients:
                    dataloaders_for_clients[client_id] = {
                        "train": None,
                        "test": None,
                        "val": None,
                    }
                if dataloader_type == "train":
                    dataloaders_for_clients[client_id]["train"] = DataLoader(
                        basic_dataset(
                            split_parameters.M_sets[k],
                            data_X,
                            torch.tensor(data_Y, dtype=torch.int64),
                        ),
                        batch_size=args.batch_size,
                        shuffle=False,
                    )
                elif dataloader_type == "test":
                    dataloaders_for_clients[client_id]["test"] = DataLoader(
                        basic_dataset(
                            split_parameters.M_sets[k],
                            data_X,
                            torch.tensor(data_Y, dtype=torch.int64),
                        ),
                        batch_size=args.batch_size,
                        shuffle=False,
                    )
                    dataloaders_for_clients[client_id]["val"] = dataloaders_for_clients[
                        client_id
                    ]["test"]

    datasetDir, picklesDir = get_datadir(args.dataset_name, args.data_split_type)
    init_data_dir(datasetDir, picklesDir)
    raw_data_path = datasetDir + "/raw"
    directories = os.listdir(raw_data_path)
    directories.remove("ucfTrainTestlist")
    test_train_choice = "01"
    train_videos_paths = []
    test_videos_paths = []
    labels_of_mm_data = []
    files_not_use = []

    img_vector_extractor = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    img_vector_extractor.fc = nn.Linear(2048, 2048)
    nn.init.eye_(img_vector_extractor.fc.weight)
    img_vector_extractor.to(device)
    with open("data/UCF101/raw/labels_modalities.txt", "r") as f:
        info = {
            s.strip("\n").split(":")[0]: tuple(
                map(int, s.strip("\n").split(":")[-1].split(","))
            )
            for s in f.readlines()
        }
        labels_of_mm_data = list(
            dict(filter(lambda a: a[1][1] == 0, info.items())).keys()
        )
    with open("data/UCF101/raw/videos_shorter_than_30frames.txt", "r") as f:
        files_not_use = [s.strip() for s in f.readlines()]
    with open(
        f"{raw_data_path}/ucfTrainTestlist/trainlist{test_train_choice}.txt"
    ) as f:
        train_videos_paths = list(
            map(lambda x: x.strip("\n").split(" ")[0], f.readlines())
        )
        mm_data_peice_paths_train = list(
            filter(
                lambda x: (os.path.basename(os.path.dirname(x)) in labels_of_mm_data)
                and (x not in files_not_use),
                train_videos_paths,
            )
        )
    with open(f"{raw_data_path}/ucfTrainTestlist/testlist{test_train_choice}.txt") as f:
        test_videos_paths = list(map(lambda x: x.strip("\n"), f.readlines()))
        mm_data_peice_paths_test = list(
            filter(
                lambda x: (os.path.basename(os.path.dirname(x)) in labels_of_mm_data)
                and (x not in files_not_use),
                test_videos_paths,
            )
        )
    random.shuffle(mm_data_peice_paths_train)
    random.shuffle(mm_data_peice_paths_test)
    mm_data_peice_paths_train = mm_data_peice_paths_train
    mm_data_peice_paths_test = mm_data_peice_paths_test
    N_train = len(mm_data_peice_paths_train)
    N_test = len(mm_data_peice_paths_test)
    split_parameters = ParametersForDataSplitType(args.data_split_type, N_train, N_test)
    dataloaders_for_clients = {}
    # create(
    #     data_peice_paths=mm_data_peice_paths_train,
    #     Dk_len_=split_parameters.D_len_train,
    #     dataloader_type="train",
    # )
    # for client_id in dataloaders_for_clients:
    #     data_generating_store(
    #         picklesDir,
    #         client_id,
    #         dataloaders_for_clients[client_id],
    #         {},
    #     )
    create(
        data_peice_paths=mm_data_peice_paths_test,
        Dk_len_=split_parameters.D_len_test,
        dataloader_type="test",
    )
    for client_id in dataloaders_for_clients:
        data_generating_store(
            picklesDir,
            client_id,
            dataloaders_for_clients[client_id],
            {},
        )
