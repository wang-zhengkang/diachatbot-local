import numpy as np
import torch
import logging
import time
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.env import Environment

from convlab2.policy.vhus.diachat_DynamicGoal.vhus_diachat import UserPolicyVHUS
from convlab2.dpl.etc.util.dst import RuleDST
from convlab2.dpl.etc.loader.build_data import build_data
from convlab2.dpl.gdpl.diachat.gdpl import GDPL
from convlab2.dpl.gdpl.diachat.estimator import RewardEstimator
from convlab2.dpl.gdpl.diachat.test.evaluate2 import *
from convlab2.dpl.rlmodule import Memory
import matplotlib.pyplot as plt

from argparse import ArgumentParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass


def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)

            # interact with env
            next_s, _, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), 0, next_s_vec.numpy(), mask)

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


def sample(env, policy, batchsz, process_num):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
	:param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_ = queue.get()
        buff0.append(buff_)  # merge current Memory into buff0
    evt.set()

    # now buff saves all the sampled data
    buff = buff0

    return buff.get_batch()


def update(env, policy, batchsz, epoch, process_num, rewarder):
    # sample data asynchronously
    batch = sample(env, policy, batchsz, process_num)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    batchsz_real = s.size(0)

    policy.update(epoch, batchsz_real, s, a, next_s, mask, rewarder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str,
                        default="convlab2/dpl/mle/diachat/load/5_fold_mle.pol.mdl", help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=1024, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=1000, help="number dof epochs to train")
    parser.add_argument("--process_num", type=int, default=8, help="number of processes of trajactory sampling")
    args = parser.parse_args()

    # sys rule DST
    dst_sys = RuleDST()

    policy_sys = GDPL(True)
    rewarder = RewardEstimator(True)

    # not use user dst
    dst_usr = None
    
    # US: VHUS
    policy_usr = UserPolicyVHUS(load_from_zip=True)

    # assemble
    simulator = PipelineAgent(None, None, policy_usr, None, 'usr')

    env = Environment(None, simulator, None, dst_sys)

    # 加载测试数据
    with open('convlab2/dpl/etc/data/test.json', 'r') as f:
        source_data = json.load(f)
        test_data = build_data(source_data)
    F1_list = []
    vector = DiachatVector()

    logging.info("Start training.")
    start = int(time.time())
    for i in range(args.epoch):
        i += 1
        update(env, policy_sys, args.batchsz, i, args.process_num, rewarder)
        if i % 5 == 0:
            predict_target_act = []
            for _, state_act_pair in enumerate(test_data):
                state_vec = state_act_pair[0]
                target_act_vec = state_act_pair[1]
                predict_act = policy_sys.predict(state_vec)
                target_act = vector.action_devectorize(target_act_vec)
                temp = [predict_act, target_act]
                predict_target_act.append(temp)
            precise, recall, F1 = calculateF1(predict_target_act)
            F1_list.append(F1)

    end = int(time.time())
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    logging.info(f"Train model cost time {h:0>2d}:{m:0>2d}:{s:0>2d}.")

    
    # 绘画
    epoch_list = [(i+1)*5 for i in range(len(F1_list))]
    plt.plot(epoch_list, F1_list)
    plt.xlabel('epoch')
    plt.ylabel('F1')
    plt.savefig("convlab2/dpl/gdpl/diachat/log/1_recon_loss.jpg")