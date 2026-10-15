import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_records(name:str)->pd.DataFrame:
    """All rate_*.jsonl files of one experiment -> one DataFrame with a 'rate' column."""
    folder=Path("results/raw")/name
    frames=[]
    for f in sorted(folder.glob("rate_*.jsonl")):
        if f.name.endswith("_stats.jsonl"):
            continue
        df=pd.read_json(f,lines=True,convert_dates=False)
        df["rate"]=float(f.stem.split("_")[1])          
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no rate_*.jsonl files in {folder}")
    df=pd.concat(frames,ignore_index=True)
    df["experiment"]=name
    return df


def summarize_rate(g:pd.DataFrame,slo_ttft_s:float,slo_tpot_s:float)->dict:
    """One rate's records -> one summary row."""
    ok=g[g["status"]=="ok"]
    duration_s=g["t_end"].max()-g["t_send"].min()
    nan=float("nan")

    good=ok[(ok["ttft"]<=slo_ttft_s)&(ok["tpot"].isna()|(ok["tpot"]<=slo_tpot_s))]

    def q(col:str,p:float)->float:
        return ok[col].quantile(p) if len(ok) else nan

    return {
        "n":len(g),
        "ok":len(ok),
        "rejected":int((g["status"]=="rejected").sum()),
        "failed":int(g["status"].isin(["error","timeout"]).sum()),
        "req_per_s":len(ok)/duration_s,
        "out_tok_per_s":ok["max_tokens"].sum()/duration_s,
        "ttft_p50_s":q("ttft",0.5),
        "ttft_p99_s":q("ttft",0.99),
        "tpot_p50_s":q("tpot",0.5),
        "tpot_p99_s":q("tpot",0.99),
        "e2el_p50_s":q("e2el",0.5),
        "goodput_req_per_s":len(good)/duration_s,
        "max_send_lag_ms":g["send_lag"].max()*1000,
    }


def summary_table(df:pd.DataFrame,slo_ttft_s:float,slo_tpot_s:float)->pd.DataFrame:
    """One experiment -> table with one row per rate."""
    rows=[]
    for rate,g in df.groupby("rate"):
        rows.append({"experiment":g["experiment"].iloc[0],"rate":rate,
                     **summarize_rate(g,slo_ttft_s,slo_tpot_s)})
    return pd.DataFrame(rows).sort_values("rate")


def plot_metric(tables:dict[str,pd.DataFrame],y:str,ylabel:str,path:Path,log_y:bool=False)->None:
    """One line per experiment: y vs offered rate."""
    fig,ax=plt.subplots(figsize=(6,4))
    for name,t in tables.items():
        ax.plot(t["rate"],t[y],marker="o",label=name)
    ax.set_xlabel("offered rate (req/s)")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True,alpha=0.3)
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)


def main()->None:
    parser=argparse.ArgumentParser(description="Summarise and compare experiments")
    parser.add_argument("--experiments",nargs="+",required=True,help="names under results/raw/")
    parser.add_argument("--tag",required=True,help="output folder results/<tag>/")
    parser.add_argument("--slo-ttft",type=float,default=1.0,help="TTFT SLO in seconds")
    parser.add_argument("--slo-tpot",type=float,default=0.05,help="TPOT SLO in seconds")
    args=parser.parse_args()

    tables={name:summary_table(load_records(name),args.slo_ttft,args.slo_tpot)
            for name in args.experiments}

    out=Path("results")/args.tag
    out.mkdir(parents=True,exist_ok=True)

    combined=pd.concat(tables.values(),ignore_index=True)
    combined.to_csv(out/"summary.csv",index=False)
    print(combined.round(3).to_string(index=False))

    plot_metric(tables,"ttft_p99_s","TTFT p99 (s)",out/"ttft_p99.png",log_y=True)
    plot_metric(tables,"tpot_p50_s","TPOT p50 (s)",out/"tpot_p50.png")
    plot_metric(tables,"req_per_s","throughput (req/s)",out/"throughput.png")
    plot_metric(tables,"goodput_req_per_s","goodput (req/s)",out/"goodput.png")
    print(f"\nwrote {out}/summary.csv and 4 plots")


if __name__=="__main__":
    main()