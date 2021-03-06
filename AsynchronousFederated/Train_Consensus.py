import numpy as np
import time
import argparse
import resnet
from GraphConstruct import GraphConstruct
from AsyncCommunicator import AsyncDecentralized
from DSGD import decenCommunicator
from ModelAvg import model_avg
from mpi4py import MPI
from DataPartition import partition_dataset
from comm_helpers import flatten_tensors
from Misc import AverageMeter, Recorder, test_accuracy, compute_accuracy

import torch
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def run(rank, size):

    # set random seed
    torch.manual_seed(args.randomSeed + rank)
    np.random.seed(args.randomSeed)

    # Split up final node from all the others for communication purposes
    worker_size = size - 1
    color = int(np.floor(rank / worker_size))
    WORKER_COMM = MPI.COMM_WORLD.Split(color=color, key=rank)

    # select neural network model
    num_class = 10
    model = resnet.ResNet(args.resSize, num_class)

    # split up GPUs
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus

    # initialize the GPU being used
    torch.cuda.set_device(gpu_id)
    model = model.cuda(gpu_id)

    # model loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=5e-4,
                          nesterov=args.nesterov)

    # Designate the consensus node (the final node) and worker nodes
    if rank == size-1:

        # Run consensus node task
        model_avg(worker_size, model, args)

    else:

        # guarantee all local models start from the same point
        init_model = sync_allreduce(model, worker_size, WORKER_COMM)

        # load data
        train_loader, test_loader, val_loader = partition_dataset(rank, worker_size, WORKER_COMM, args)

        # load base network topology
        p = 3/worker_size
        GP = GraphConstruct(rank, worker_size, WORKER_COMM, args.graph, args.weight_type, p=p, num_c=args.num_clusters)

        if args.comm_style == 'async':
            communicator = AsyncDecentralized(rank, worker_size, WORKER_COMM, GP,
                                              args.sgd_steps, args.max_sgd, args.wb, args.memory, init_model)
        elif args.comm_style == 'ld-sgd':
            communicator = decenCommunicator(rank, worker_size, WORKER_COMM, GP, args.i1, args.i2)
        elif args.comm_style == 'pd-sgd':
            communicator = decenCommunicator(rank, worker_size, WORKER_COMM, GP, args.i1, 1)
        elif args.comm_style == 'd-sgd':
            communicator = decenCommunicator(rank, worker_size, WORKER_COMM, GP, 0, 1)
        else:
            # Anything else just default to our algorithm
            communicator = AsyncDecentralized(rank, worker_size, WORKER_COMM, GP,
                                              args.sgd_steps, args.max_sgd, args.wb, args.memory, init_model)

        # init recorder
        comp_time = 0
        comm_time = 0
        recorder = Recorder(args, rank)
        losses = AverageMeter()
        top1 = AverageMeter()
        requests = [MPI.REQUEST_NULL for _ in range(args.epoch)]
        count = 0

        if args.noniid:
            d_epoch = 200
        else:
            d_epoch = 100

        WORKER_COMM.Barrier()
        # start training
        for epoch in range(args.epoch):
            init_time = time.time()
            record_time = 0
            model.train()

            # Start training each epoch
            for batch_idx, (data, target) in enumerate(train_loader):
                start_time = time.time()
                # data loading
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

                # forward pass
                output = model(data)
                loss = criterion(output, target)

                # record training loss and accuracy
                record_start = time.time()
                acc1 = compute_accuracy(output, target)
                losses.update(loss.item(), data.size(0))
                top1.update(acc1[0].item(), data.size(0))
                record_end = time.time() - record_start
                record_time += record_end

                # backward pass
                loss.backward()

                # communication happens here
                comm_start = time.time()
                d_comm_time = communicator.communicate(model)
                comm_t = time.time() - comm_start

                # gradient step
                optimizer.step()
                optimizer.zero_grad()
                end_time = time.time()

                # compute computational time
                comp_time += (end_time - start_time - comm_t)
                # compute communication time
                comm_time += d_comm_time

            # update learning rate here
            update_learning_rate(optimizer, epoch, drop=0.5, epochs_drop=20.0, decay_epoch=d_epoch,
                                 itr_per_epoch=len(train_loader))

            send_start = time.time()
            # send model to the dummy node to compute the overall model accuracy
            tensor_list = list()
            for param in model.parameters():
                tensor_list.append(param)
            send_buffer = flatten_tensors(tensor_list).cpu()

            requests[epoch] = MPI.COMM_WORLD.Isend(send_buffer.detach().numpy(), dest=size - 1,
                                                   tag=rank + 10 * worker_size)
            # Clear the memory from Isend
            if requests[count].Test():
                requests[count].Wait()
                count += 1

            # evaluate test accuracy at the end of each epoch
            # test_acc = test_accuracy(model, test_loader)

            # Remove time spent sending messages to the consensus node
            send_time = time.time() - send_start

            # evaluate validation accuracy at the end of each epoch
            val_acc = test_accuracy(model, val_loader)

            # run personalization if turned on
            if args.personalize and args.comm_style == 'async':
                comm_time += communicator.personalize(epoch+2, val_acc, args.noniid)

            # total time spent in algorithm
            comp_time -= record_time
            epoch_time = comp_time + comm_time

            print("rank: %d, epoch: %.3f, loss: %.3f, train_acc: %.3f, val_acc: %.3f, comp time: %.3f, "
                  "epoch time: %.3f" % (rank, epoch, losses.avg, top1.avg, val_acc, comp_time, epoch_time))

            recorder.add_new(comp_time, comm_time, epoch_time, (time.time() - init_time) - send_time,
                             top1.avg, losses.avg, val_acc)

            # reset recorders
            comp_time, comm_time = 0, 0
            losses.reset()
            top1.reset()

        # Save data to output folder
        recorder.save_to_file()

        # Broadcast/wait until all other neighbors are finished in async algorithm
        if args.comm_style == 'async':
            communicator.wait(model)
            print('Finished from Rank %d' % rank)

    MPI.COMM_WORLD.Barrier()


def update_learning_rate(optimizer, epoch, drop, epochs_drop, decay_epoch, itr=None, itr_per_epoch=None):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially starting at decay_epoch
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    base_lr = 0.1
    lr = args.lr

    if args.warmup and epoch < 5:  # warmup to scaled lr
        if lr > base_lr:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    elif epoch >= decay_epoch:
        lr *= np.power(drop, np.floor((1 + epoch - decay_epoch) / epochs_drop))

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def sync_allreduce(model, size, comm):
    senddata = {}
    recvdata = {}
    for param in model.parameters():
        tmp = param.data.cpu()
        senddata[param] = tmp.numpy()
        recvdata[param] = np.empty(senddata[param].shape, dtype=senddata[param].dtype)
    torch.cuda.synchronize()
    comm.Barrier()

    # comm_start = time.time()
    for param in model.parameters():
        comm.Allreduce(senddata[param], recvdata[param], op=MPI.SUM)
    torch.cuda.synchronize()
    comm.Barrier()

    # comm_end = time.time()
    # comm_t = (comm_end - comm_start)

    tensor_list = list()
    for param in model.parameters():
        tensor_list.append(param)
        param.data = torch.Tensor(recvdata[param]).cuda()
        param.data = param.data / float(size)

    # flatten tensors
    initial_model = flatten_tensors(tensor_list).cpu().detach().numpy()

    return initial_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--name', '-n', default="default", type=str, help='experiment name')
    parser.add_argument('--description', type=str, help='experiment description')

    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--comm_style', default='async', type=str, help='baseline communicator')
    parser.add_argument('--resSize', default=50, type=int, help='res net size')
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='total epoch')
    parser.add_argument('--bs', default=64, type=int, help='batch size on each worker')
    parser.add_argument('--noniid', default=1, type=int, help='use non iid data or not')
    parser.add_argument('--degree_noniid', default=0.7, type=float, help='how distributed are labels (0 is random)')
    parser.add_argument('--weight_type', default='uniform', type=str, help='how do workers average with each other')
    parser.add_argument('--unordered_epochs', default=1, type=int, help='calculate consensus after the first n models')

    # Specific async arguments
    parser.add_argument('--wb', default=0, type=int, help='proportionally increase neighbor weights or self replace')
    parser.add_argument('--memory', default=0, type=int, help='store all neighbor local models')
    parser.add_argument('--max_sgd', default=10, type=int, help='max sgd steps per worker')
    parser.add_argument('--personalize', default=1, type=int, help='use personalization or not')

    parser.add_argument('--i1', default=1, type=int, help='i1 comm set, number of local updates no averaging')
    parser.add_argument('--i2', default=2, type=int, help='i2 comm set, number of d-sgd updates')
    parser.add_argument('--sgd_steps', default=3, type=int, help='baseline sgd steps per worker')
    parser.add_argument('--num_clusters', default=1, type=int, help='number of clusters in graph')
    parser.add_argument('--graph', type=str, help='graph topology')

    parser.add_argument('--warmup', action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov momentum or not')
    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', type=str, help='the path of dataset')
    parser.add_argument('--downloadCifar', default=0, type=int, help='change to 1 if needing to download Cifar')
    parser.add_argument('--p', '-p', action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath', type=str, help='save path')
    parser.add_argument('--outputFolder', type=str, help='save folder')
    parser.add_argument('--randomSeed', default=9001, type=int, help='random seed')

    args = parser.parse_args()

    if not args.description:
        print('No experiment description, exit!')
        exit()

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    run(rank, size)
