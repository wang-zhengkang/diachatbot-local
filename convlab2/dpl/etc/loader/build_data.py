from convlab2.dpl.etc.util.dst import RuleDST
from convlab2.dpl.etc.util.state_structure import *
from convlab2.dpl.etc.util.vector_diachat import DiachatVector


def build_data(partdata):
    def org(act_label, dsv_list: list):
        da = []
        for dsv in dsv_list:
            da_temp = []
            da_temp.append(act_label)
            da_temp.append(dsv["domain"] if dsv["domain"] else "none")
            da_temp.append(dsv["slot"] if dsv["slot"] else "none")
            da_temp.append(dsv["value"] if dsv["value"] else "none")
            da.append(da_temp)
        return da
    targetdata = []
    vector = DiachatVector()
    dst = RuleDST()
    for session in partdata:
        dst.init_session()
        for i, utterance in enumerate(session["utterances"]):
            da = []
            for annotation in utterance["annotation"]:
                act_label = annotation["act_label"]
                dsv = annotation["slot_values"]
                da_tmep = org(act_label, dsv)
                for temp in da_tmep:
                    da.append(temp)
            if utterance["agentRole"] == "User":
                dst.update(da, True if i == len(session["utterances"]) - 2 else False)
            else:
                state_vec = vector.state_vectorize(dst.state)
                dst.update_by_sysda(da)
                action = dst.state["sys_da"]
                action_vec = vector.action_vectorize(action)
                targetdata.append([state_vec, action_vec])
    return targetdata
