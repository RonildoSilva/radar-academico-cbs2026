import os, glob, re
import pandas as pd, numpy as np
D = os.environ.get("EPM_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "epm", "EPM Dataset 2", "Data"))
OUT = os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__)))
cols = ["session","student","exercise","activity","start","end","idle","wheel","wheel_click","click_l","click_r","mouse_mov","keys"]
rows = []
for s in range(1,7):
    for f in glob.glob(f"{D}/Processes/Session {s}/*"):
        df = pd.read_csv(f, header=None, names=cols, skipinitialspace=True, dtype=str)
        rows.append(df)
log = pd.concat(rows, ignore_index=True)
for c in ["session","student","idle","wheel","wheel_click","click_l","click_r","mouse_mov","keys"]:
    log[c] = pd.to_numeric(log[c].str.strip(), errors="coerce")
log["activity"] = log["activity"].str.strip()
log["exercise"] = log["exercise"].str.strip()
log["start"] = pd.to_datetime(log["start"].str.strip(), format="%d.%m.%Y %H:%M:%S", errors="coerce")
log["end"] = pd.to_datetime(log["end"].str.strip(), format="%d.%m.%Y %H:%M:%S", errors="coerce")
print("linhas", len(log), "nulos start", log.start.isna().sum())
log = log.dropna(subset=["start","end"]).sort_values(["session","student","start","end"]).reset_index(drop=True)
log["dur"] = (log["end"]-log["start"]).dt.total_seconds() + 1

def base(a):
    return re.sub(r"_\d+_\d+$","",a)  # Study_Es_1_2 -> Study_Es
log["activity"] = log["activity"].str.replace("Fsm_Related","FSM_Related")
log["act_base"] = log["activity"].map(base)
OUT_TASK = {"Other","Blank"}
log["in_task"] = ~log["act_base"].isin(OUT_TASK)
META = {"Study_Es":"Estudo","Study_Materials":"Estudo",
        "Deeds_Es":"Construção","Deeds":"Construção","Properties":"Construção","FSM_Es":"Construção","FSM_Related":"Construção",
        "Diagram":"Simulação",
        "TextEditor_Es":"Relato","TextEditor":"Relato",
        "Aulaweb":"Plataforma","Other":"Fora da tarefa","Blank":"Fora da tarefa"}
log["meta"] = log["act_base"].map(META)
print(log["act_base"].value_counts())
print("sem meta:", log.meta.isna().sum())
log.to_pickle(f"{OUT}/log.pkl")
print(log.groupby("session").agg(eventos=("activity","size"), alunos=("student","nunique")))
print("atividades distintas:", log.activity.nunique())
