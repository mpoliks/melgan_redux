#!/usr/bin/env python
import os
import argparse
import time
from pathlib import Path
from argparse import Namespace
from collections import OrderedDict

import numpy as np
from PIL import Image
from matplotlib import cm
import torch
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP

import mel2wav.modules
from mel2wav.dataset import AudioDataset
from mel2wav.modules import Audio2Mel, Discriminator, Generator
from mel2wav.utils import save_sample
from util import seed_everything


def load_state_dict_handleDP(model, filepath):
    try:
        model.load_state_dict(torch.load(filepath))
    except RuntimeError as e:
        print("RuntimeError", e)
        print("Fixing model trained with DataParallel by removing .module prefix")
        # state_dict = torch.load(filepath, map_location=device)
        state_dict = torch.load(filepath)
        # state_dict = state_dict["state_dict.model"]
        # remove the DP() to load the model
        state_dict = OrderedDict((k.split(".", 1)[1], v) for k, v in state_dict.items())
        model.load_state_dict(state_dict)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_from_run_id", default=None)
    parser.add_argument("--resume_run_id", default=None)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--ratios", default=[8, 8, 2, 2])
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--pad_mode", type=str, default="reflect", choices=["reflect", "replicate"]
    )

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--sampling_rate", type=int, default=44100)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)

    parser.add_argument("--notes", type=str)
    args = parser.parse_args()
    return args


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(7)

    args = parse_args()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    entity = "demiurge"
    project = "melgan"
    load_from_run_id = args.load_from_run_id
    resume_run_id = args.resume_run_id
    restore_run_id = load_from_run_id or resume_run_id
    batch_size = args.batch_size

    # Getting initial run steps and epoch
    # if restore run, replace args
    steps = None
    if restore_run_id:
        api = wandb.Api()
        previous_run = api.run(f"{entity}/{project}/{restore_run_id}")
        steps = previous_run.lastHistoryStep
        prev_args = argparse.Namespace(**previous_run.config)
        args = vars(args)
        args.update(vars(prev_args))
        args = Namespace(**args)
        args.batch_size = batch_size

    load_initial_weights = bool(restore_run_id)
    sampling_rate = args.sampling_rate
    ratios = args.ratios
    if isinstance(ratios, str):
        ratios = ratios.replace(" ", "")
        ratios = ratios.strip("][").split(",")
        ratios = [int(i) for i in ratios]
        ratios = np.array(ratios)

    if load_from_run_id and resume_run_id:
        raise RuntimeError("Specify either --load_from_id or --resume_run_id.")

    if resume_run_id:
        print(f"Resuming run ID {resume_run_id}.")
    elif load_from_run_id:
        print(f"Starting new run with initial weights from run ID {load_from_run_id}.")
    else:
        print("Starting new run from scratch.")

    # read 1 line in train files to log dataset location
    train_files = Path(args.data_path) / "train_files.txt"
    with open(train_files, encoding="utf-8", mode="r") as f:
        file = f.readline()
    args.train_file_sample = str(file)

    wandb.init(
        entity=entity,
        project=project,
        id=resume_run_id,
        config=args,
        resume=True if resume_run_id else False,
        save_code=True,
        dir=args.save_path,
        notes=args.notes,
    )

    print("run id: " + str(wandb.run.id))
    print("run name: " + str(wandb.run.name))

    root = Path(wandb.run.dir)
    root.mkdir(parents=True, exist_ok=True)

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    wandb.save("args.yml")

    ###############################################
    # The file modules.py is needed by the unagan #
    ###############################################
    wandb.save(mel2wav.modules.__file__, base_path=".")

    #######################
    # Load PyTorch Models #
    #######################

    netG = Generator(
        args.n_mel_channels, args.ngf, args.n_residual_layers, ratios=ratios
    ).to(device)
    netD = Discriminator(
        args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor
    ).to(device)
    fft = Audio2Mel(
        n_mel_channels=args.n_mel_channels,
        pad_mode=args.pad_mode,
        sampling_rate=sampling_rate,
    ).to(device)

    for model in [netG, netD, fft]:
        wandb.watch(model)

    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    if load_initial_weights:

        for model, filenames in [
            (netG, ["netG.pt", "netG_prev.pt"]),
            (optG, ["optG.pt", "optG_prev.pt"]),
            (netD, ["netD.pt", "netD_prev.pt"]),
            (optD, ["optD.pt", "optD_prev.pt"]),
        ]:
            recover_model = False
            filepath = None
            for filename in filenames:
                try:
                    run_path = f"{entity}/{project}/{restore_run_id}"
                    print(f"Restoring {filename} from run path {run_path}")
                    restored_file = wandb.restore(filename, run_path=run_path)
                    filepath = restored_file.name
                    model = load_state_dict_handleDP(model, filepath)
                    recover_model = True
                    break
                except RuntimeError as e:
                    print("RuntimeError", e)
                    print(f"recover model weight file: '{filename}'' failed")
            if not recover_model:
                raise RuntimeError(
                    f"Cannot load model weight files for component {filenames[0]}."
                )
            else:
                # store successfully recovered model weight file ("***_prev.pt")
                path_parent = Path(filepath).parent
                newfilepath = str(path_parent / filenames[1])
                os.rename(filepath, newfilepath)
                wandb.save(newfilepath)
    if torch.cuda.device_count() > 1:
        netG = DP(netG).to(device)
        netD = DP(netD).to(device)
        fft = DP(fft).to(device)
        print(f"We have {torch.cuda.device_count()} gpus. Use data parallel.")
    else:
        print(f"We have {torch.cuda.device_count()} gpu.")

    #######################
    # Create data loaders #
    #######################
    train_set = AudioDataset(
        Path(args.data_path) / "train_files.txt",
        args.seq_len,
        sampling_rate=sampling_rate,
    )
    test_set = AudioDataset(
        Path(args.data_path) / "test_files.txt",
        sampling_rate * 4,
        sampling_rate=sampling_rate,
        augment=False,
    )
    wandb.save(str(Path(args.data_path) / "train_files.txt"))
    wandb.save(str(Path(args.data_path) / "test_files.txt"))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=1)

    if len(train_loader) == 0:
        raise RuntimeError("Train dataset is empty.")

    if len(test_loader) == 0:
        raise RuntimeError("Test dataset is empty.")

    if not restore_run_id:
        steps = wandb.run.step
    start_epoch = steps // len(train_loader)
    print(f"Starting with epoch {start_epoch} and step {steps}.")

    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []
    samples = []
    melImages = []
    num_fix_samples = args.n_test_samples - (args.n_test_samples // 2)
    cmap = cm.get_cmap("inferno")
    for i, x_t in enumerate(test_loader):
        x_t = x_t.to(device)
        s_t = fft(x_t).detach()

        test_voc.append(s_t.to(device))
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), sampling_rate, audio)
        samples.append(
            wandb.Audio(audio, caption=f"sample {i}", sample_rate=sampling_rate)
        )
        melImage = s_t.squeeze().detach().cpu().numpy()
        # melImage = (melImage - np.amin(melImage)) / (
        #     np.amax(melImage) - np.amin(melImage)
        # )
        # melImage = Image.fromarray(melImage).convert("L")
        # melImage = Image.fromarray(np.uint8(cmap(np.array(melImage))) * 255)
        # melImage = melImage.resize((melImage.width * 4, melImage.height * 4))
        melImages.append(wandb.Image(melImage, caption=f"sample {i}"))

        if i == num_fix_samples - 1:
            break

    if not resume_run_id:
        wandb.log({"audio/original": samples}, step=0)
        wandb.log({"mel/original": melImages}, step=0)
    else:
        print("We are resuming, skipping logging of original audio.")

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000

    for epoch in range(start_epoch, start_epoch + args.epochs + 1):
        for iterno, x_t in enumerate(train_loader):
            x_t = x_t.to(device)
            s_t = fft(x_t).detach()
            x_pred_t = netG(s_t.to(device))

            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()

            #######################
            # Train Discriminator #
            #######################
            D_fake_det = netD(x_pred_t.to(device).detach())
            D_real = netD(x_t.to(device))

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.to(device))

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            wt = D_weights * feat_weights
            for i in range(args.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + args.lambda_feat * loss_feat).backward()
            optG.step()

            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])

            wandb.log(
                {
                    "loss/discriminator": costs[-1][0],
                    "loss/generator": costs[-1][1],
                    "loss/feature_matching": costs[-1][2],
                    "loss/mel_reconstruction": costs[-1][3],
                },
                step=steps,
            )
            steps += 1

            if steps % args.save_interval == 0:
                st = time.time()
                with torch.no_grad():
                    samples = []
                    melImages = []
                    # fix samples
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(
                            root / ("generated_%d.wav" % i), sampling_rate, pred_audio
                        )
                        samples.append(
                            wandb.Audio(
                                pred_audio,
                                caption=f"sample {i}",
                                sample_rate=sampling_rate,
                            )
                        )
                        melImage = voc.squeeze().detach().cpu().numpy()
                        # melImage = (melImage - np.amin(melImage)) / (
                        #     np.amax(melImage) - np.amin(melImage)
                        # )
                        # melImage = Image.fromarray(melImage).convert("L")
                        # melImage = Image.fromarray(
                        #     np.uint8(cmap(np.array(melImage))) * 255
                        # )
                        # melImage = melImage.resize(
                        #     (melImage.width * 4, melImage.height * 4)
                        # )
                        melImages.append(wandb.Image(melImage, caption=f"sample {i}"))
                    wandb.log(
                        {
                            "audio/generated": samples,
                            "mel/generated": melImages,
                            "epoch": epoch,
                        },
                        step=steps,
                    )

                    # var samples
                    source = []
                    pred = []
                    pred_mel = []
                    num_var_samples = args.n_test_samples - num_fix_samples
                    for i, x_t in enumerate(test_loader):
                        # source
                        x_t = x_t.to(device)
                        audio = x_t.squeeze().cpu()
                        source.append(
                            wandb.Audio(
                                audio, caption=f"sample {i}", sample_rate=sampling_rate
                            )
                        )
                        # pred
                        s_t = fft(x_t).detach()
                        voc = s_t.to(device)
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        pred.append(
                            wandb.Audio(
                                pred_audio,
                                caption=f"sample {i}",
                                sample_rate=sampling_rate,
                            )
                        )
                        melImage = voc.squeeze().detach().cpu().numpy()
                        # melImage = (melImage - np.amin(melImage)) / (
                        #     np.amax(melImage) - np.amin(melImage)
                        # )
                        # melImage = Image.fromarray(melImage).convert("L")
                        # melImage = Image.fromarray(
                        #     np.uint8(cmap(np.array(melImage))) * 255
                        # )
                        # melImage = melImage.resize(
                        #     (melImage.width * 4, melImage.height * 4)
                        # )
                        pred_mel.append(wandb.Image(melImage, caption=f"sample {i}"))

                        # stop when reach log sample
                        if i == num_var_samples - 1:
                            break

                    wandb.log(
                        {
                            "audio/var_original": source,
                            "audio/var_generated": pred,
                            "mel/var_generated": pred_mel,
                        },
                        step=steps,
                    )

                print("Saving models ...")
                torch.save(netG.state_dict(), root / "netG.pt")
                torch.save(optG.state_dict(), root / "optG.pt")
                wandb.save(str(root / "netG.pt"))
                wandb.save(str(root / "optG.pt"))

                torch.save(netD.state_dict(), root / "netD.pt")
                torch.save(optD.state_dict(), root / "optD.pt")
                wandb.save(str(root / "netD.pt"))
                wandb.save(str(root / "optD.pt"))

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD.state_dict(), root / "best_netD.pt")
                    torch.save(netG.state_dict(), root / "best_netG.pt")
                    wandb.save(str(root / "best_netD.pt"))
                    wandb.save(str(root / "best_netG.pt"))

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()


if __name__ == "__main__":
    main()
