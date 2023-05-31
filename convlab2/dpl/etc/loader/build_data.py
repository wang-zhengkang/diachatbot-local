from convlab2.dpl.etc.util.dst import RuleDST
from convlab2.dpl.etc.util.state_structure import *
from convlab2.dpl.etc.util.vector_diachat import DiachatVector


def build_data(partdata):
    def org(da, act_label, dsv_list: list):
        for dsv in dsv_list:
            da_temp = []
            da_temp.append(act_label)
            da_temp.append(dsv["domain"] if dsv["domain"] else "none")
            da_temp.append(dsv["slot"] if dsv["slot"] else "none")
            da_temp.append(dsv["value"] if dsv["value"] else "none")
            da.append(da_temp)
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
                org(da, act_label, dsv)
                if utterance["agentRole"] == "User":
                    dst.update(da)
                    last_sys_da = dst.state["sys_da"]
                    usr_da = dst.state["usr_da"]
                    cur_domain = dst.state["cur_domain"]
                    inform_ds = dst.state["inform_ds"]
                    askhow_ds = dst.state["askhow_ds"]
                    askwhy_ds = dst.state["askwhy_ds"]
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
                    state['inform_ds'] = inform_ds
                    state['askhow_ds'] = askhow_ds
                    state['askwhy_ds'] = askwhy_ds
                    state['askfor_ds'] = askfor_ds
                    state['askforsure_ds'] = askforsure_ds
                    state['belief_state'] = belief_state
                    state['terminate'] = terminate
                    action = sys_da
                    targetdata.append([vector.state_vectorize(state),
                                       vector.action_vectorize(action)])
    return targetdata
