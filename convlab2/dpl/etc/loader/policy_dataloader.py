import os
import json
import pickle
import torch
from copy import deepcopy
from convlab2.util.file_util import read_zipped_json
import torch.utils.data as data

from convlab2.dpl.etc.util.dst import RuleDST
from convlab2.dpl.etc.util.state_structure import *
from convlab2.dpl.etc.util.vector_diachat import DiachatVector


class PolicyDataloader():
    def __init__(self):
        self.vector = DiachatVector()
        print('Start preprocessing the dataset')
        self._build_data()

    def _build_data(self):

        def org(da, act_label, dsv_list: list):
            for dsv in dsv_list:
                da_temp = []
                da_temp.append(act_label)
                da_temp.append(dsv["domain"] if dsv["domain"] else "none")
                da_temp.append(dsv["slot"] if dsv["slot"] else "none")
                da_temp.append(dsv["value"] if dsv["value"] else "none")
                da.append(da_temp)

        with open('convlab2/dpl/etc/data/complete_data.json', 'r') as f:
            source_data = json.load(f)
        self.data = []

        dst = RuleDST()
        for session in source_data:
            dst.init_session()
            for i, utterance in enumerate(session["utterances"]):
                da = []  # each turn da
                for annotation in utterance["annotation"]:
                    act_label = annotation["act_label"]
                    dsv = annotation["slot_values"]
                    org(da, act_label, dsv)
                if utterance["agentRole"] == "User":
                    dst.update(da)
                    last_sys_da = dst.state["sys_da"]
                    usr_da = dst.state["usr_da"]
                    cur_domain = dst.state["cur_domain"]
                    askfor_ds = dst.state["askfor_ds"]
                    askforsure_ds = dst.state["askforsure_ds"]
                    belief_state = dst.state["belief_state"]
                    if i == len(session["utterances"]) - 2:
                        terminate = True
                    else:
                        terminate = False
                else:
                    dst.update_by_sysda(da)
                    sys_da = dst.state["sys_da"]

                    state = default_state()

                    state['sys_da'] = last_sys_da
                    state['usr_da'] = usr_da
                    state['cur_domain'] = cur_domain
                    state['askfor_ds'] = askfor_ds
                    state['askforsure_ds'] = askforsure_ds
                    state['belief_state'] = belief_state
                    state['terminate'] = terminate
                    action = sys_da
                    self.data.append([self.vector.state_vectorize(state),
                                      self.vector.action_vectorize(action)])
        return self.data

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'rb') as f:
                self.data[part] = pickle.load(f)

    def create_dataset(self, part, batchsz, part_idx):
        print('Start creating {} dataset'.format(part))
        partdata = [self.data[i]for i in part_idx]
        s = []
        a = []
        for item in partdata:
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
        s = torch.stack(s)
        a = torch.stack(a)
        dataset = Dataset(s, a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        return dataloader


class Dataset(data.Dataset):
    def __init__(self, s_s, a_s):
        self.s_s = s_s
        self.a_s = a_s
        self.num_total = len(s_s)

    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        return s, a

    def __len__(self):
        return self.num_total
