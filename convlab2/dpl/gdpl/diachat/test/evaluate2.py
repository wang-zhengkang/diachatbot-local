import json
from convlab2.dpl.etc.util.vector_diachat import DiachatVector
from convlab2.dpl.etc.util.state_structure import *
from convlab2.dpl.etc.util.dst import RuleDST
from convlab2.dpl.gdpl.diachat.gdpl import GDPL
from convlab2.dpl.mle.diachat.mle import MLE


def org(da, act_label, dsv_list: list):
    for dsv in dsv_list:
        da_temp = []
        da_temp.append(act_label)
        da_temp.append(dsv["domain"] if dsv["domain"] else "none")
        da_temp.append(dsv["slot"] if dsv["slot"] else "none")
        da_temp.append(dsv["value"] if dsv["value"] else "none")
        da.append(da_temp)


def create_state_data(data):
    vector = DiachatVector()
    targetdata = []
    dst = RuleDST()
    for session in data:
        dst.init_session()
        for i, utterance in enumerate(session["utterances"]):
            da = []
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
                targetdata.append([vector.state_vectorize(state),
                                   vector.action_vectorize(action)])
    return targetdata


def calculateF1(predict_target_act):
    TP, FP, FN = 0, 0, 0
    for item in predict_target_act:
        labels = item[0]
        predicts = item[1]
        for quad in predicts:
            if quad in labels:
                TP += 1
            else:
                FP += 1
        for quad in labels:
            if quad not in predicts:
                FN += 1
    
    precise = 1.0 * TP / (TP + FP) if (TP + FP) else 0.
    recall = 1.0 * TP / (TP + FN) if (TP + FN) else 0.
    F1 = 2.0 * precise * recall / \
        (precise + recall) if (precise + recall) else 0.
    print(f"TP:{TP}")
    print(f"FP:{FP}")
    print(f"FN:{FN}")
    print(f"F1:{F1: .6f}")
    print(f"precise:{precise: .6f}")
    print(f"recall:{recall: .6f}")
    
    return precise, recall, F1


if __name__ == '__main__':
    with open('convlab2/dpl/etc/data/complete_data.json', 'r') as f:
        source_data = json.load(f)

    # 自定义创建的测试数据范围 这里用全部数据测试
    test_idx = range(len(source_data))
    test_data = [source_data[i] for i in test_idx]
    test_data = create_state_data(test_data)
    print(f"总共{len(test_data)}条状态进行测试")

    vector = DiachatVector()
    sys = GDPL()
    # sys = MLE(True)

    predict_target_act = []
    for _, state_act_pair in enumerate(test_data):
        state_vec = state_act_pair[0]
        target_act_vec = state_act_pair[1]
        predict_act = sys.predict(state_vec)
        target_act = vector.action_devectorize(target_act_vec)
        temp = [predict_act, target_act]
        predict_target_act.append(temp)
    calculateF1(predict_target_act)
