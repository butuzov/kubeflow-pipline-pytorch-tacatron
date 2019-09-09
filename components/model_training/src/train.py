from network import *
from dataload import get_dataset, DataLoader, collate_fn, get_param_size
from torch import optim
import numpy as np
import os
import time
import torch
import torch.nn as nn

import sys
import argparse

# ## ~~
IS_CUDA = torch.cuda.is_available()

# use_cuda = torch.cuda.is_available()


## ~~ main
def main(args):

    import glob

    if not os.path.isdir(args.dir_data):
        raise Exception(f"data_dir {args.dir_data} isn't found")

    # Get dataset
    dataset = get_dataset(args.dir_data)

    # Construct model
    model = nn.DataParallel(Tacotron().cuda()) if IS_CUDA else Tacotron()
    # Make optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


     # Load checkpoint if exists
    try:
        checkpoint_name = 'checkpoint_{}.pth.tar'.format(args.restore_step)
        checkpoint = torch.load(os.path.join(args.dir_checkpoints, checkpoint_name))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n--------model restored at step {}--------\n".format(args.restore_step))
    except:
        print("\n--------Start New Training--------\n")


    # Training
    model = model.train()

    # Make checkpoint directory if not exists
    if not os.path.exists(args.dir_checkpoints):
        os.mkdir(args.dir_checkpoints)


    # Decide loss function
    criterion = nn.L1Loss().cuda() if IS_CUDA else nn.L1Loss()

    # Loss for frequency of human register
    n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    torched = torch.cuda if IS_CUDA else torch

    print(f"Total {args.epochs} epochs")
    for epoch in range(args.epochs):

        start = time.time()
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=24,
        )

        print("Loaded in: %.2f sec" % (time.time() - start))

        print(f"Epoch - {epoch}")
        print(f"Data - {len(dataloader)}")

        for i, data in enumerate(dataloader):
            print(f"Enum {i}")
            start_time = time.time()

            current_step = i + args.restore_step + epoch * len(dataloader) + 1

            optimizer.zero_grad()

            # Make decoder input by concatenating [GO] Frame
            try:
                mel_input = np.concatenate((
                        np.zeros([args.batch_size, hp.num_mels, 1], dtype=np.float32),
                        data[2][:,:,1:]
                    ), axis=2)
            except:
                raise TypeError("not same dimension")

            characters = Variable(torch.from_numpy(data[0]).type(torched.LongTensor), requires_grad=False)
            mel_input  = Variable(torch.from_numpy(mel_input).type(torched.FloatTensor), requires_grad=False)
            mel_spectrogram = Variable(torch.from_numpy(data[2]).type(torched.FloatTensor), requires_grad=False)
            linear_spectrogram = Variable(torch.from_numpy(data[1]).type(torched.FloatTensor), requires_grad=False)

            if IS_CUDA:
                characters = characters.cuda()
                mel_input  = mel_input.cuda()
                mel_spectrogram = mel_spectrogram.cuda()
                linear_spectrogram = linear_spectrogram.cuda()


            # Forward
            mel_output, linear_output = model.forward(characters, mel_input)

            # Calculate loss
            mel_loss = criterion(mel_output, mel_spectrogram)
            linear_loss = torch.abs(linear_output-linear_spectrogram)
            linear_loss = 0.5 * torch.mean(linear_loss) + 0.5 * torch.mean(linear_loss[:,:n_priority_freq,:])
            loss = mel_loss + linear_loss


            if IS_CUDA:
                loss = loss.cuda()




            # Calculate gradients
            loss.backward()

            # clipping gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.)

            # Update weights
            optimizer.step()

            time_per_step = time.time() - start_time
            # print("time per step: %.2f sec" % time_per_step)

            if current_step % args.log_step == 0:
                print("time per step: %.2f sec" % time_per_step)
                # print("At timestep %d" % current_step)
                # print("linear loss: %.4f" % linear_loss.item())
                # print("mel loss: %.4f" % mel_loss.item())
                print("total loss: %.4f" % loss.item())

            # print("Current Step", current_step)
            # print("Save Step", args.save_step)
            # print("Save", current_step % args.save_step)
            if current_step % args.save_step == 0:
                name = 'checkpoint_%d.pth.tar' % current_step
                save_checkpoint({'model':model.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(args.dir_checkpoints, name))
                print("save model at step %d ..." % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)


    save_checkpoint({
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict()
    },  os.path.join(args.dir_checkpoints, 'model.pth.tar'))



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if step == 500000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


    print(model)
    print(optimizer)
    print(args)



if __name__ == "__main__":

    from collections import namedtuple

    Arg = namedtuple('Param' , 'name type help default')

    args = [
       Arg("dir_data", str, "Raw data directory", "/data/"),
       Arg("dir_checkpoints", str, "Checkpoints directory", "/data/checkpoints"),
       Arg("restore_step", int, "Global step to restore checkpoint", 0),
       Arg("batch_size", int, "Batch size", 32),
       Arg("learning_rate", float, "Batch size", 0.001),
       Arg("epochs", int, "Epochs", 10000),
       Arg("log_step", int, "Log each N steps", 10),
       Arg("save_step", int, "Save each N steps", 10),

    ]


    parser = argparse.ArgumentParser(description="Training")
    for a in args:
        parser.add_argument(f"--{a.name}", type=a.type, help=a.help, default=a.default)


    sys.exit(main(parser.parse_args()))
