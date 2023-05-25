import os
import torch
import torch.utils.data as data

from convlab2.util.file_util import read_zipped_json
from convlab2.dpl.etc.util.state_structure import default_state
from convlab2.dpl.etc.util.vector_diachat import DiachatVector
from convlab2.dpl.etc.util.dst import RuleDST

class ActStateDataset(data.Dataset):
    def __init__(self, s_s, a_s, next_s):
        self.s_s = s_s
        self.a_s = a_s
        self.next_s = next_s
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        next_s = self.next_s[index]
        return s, a, next_s
    
    def __len__(self):
        return self.num_total

class EstimatorDataLoader():
    def __init__(self) -> None:
        self.vector = DiachatVector()
        self.data = {"train": [], "val": [], "test": []}
    def create_dataset_irl(self, part, batchsz):
        def org(da, act_label, dsv_list: list):
            for dsv in dsv_list:
                da_temp = []
                da_temp.append(act_label)
                da_temp.append(dsv["domain"] if dsv["domain"] else "none")
                da_temp.append(dsv["slot"] if dsv["slot"] else "none")
                da_temp.append(dsv["value"] if dsv["value"] else "none")
                da.append(da_temp)
        data_dir = "convlab2/dpl/etc/data"
        source_data = read_zipped_json(os.path.join(data_dir, f"{part}.json.zip"), f"{part}.json",)

        print(f'Start creating {part} dataset')

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
                    self.data[part].append([self.vector.state_vectorize(state),
                                      self.vector.action_vectorize(action)])


        s = []
        a = []
        next_s = []
        for i, item in enumerate(self.data[part]):
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
            if item[0][-1]:  # terminated
                next_s.append(torch.Tensor(item[0]))
            else:
                next_s.append(torch.Tensor(self.data[part][i + 1][0]))
        s = torch.stack(s)
        a = torch.stack(a)
        next_s = torch.stack(next_s)
        dataset = ActStateDataset(s, a, next_s)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print('Finish creating {} irl dataset'.format(part))
        return dataloader
