#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show result results from the judgment files.
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""

import argparse
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

os.environ["WANDB_SILENT"] = "true"


def get_number_of_lines(file, verbose=False):
    """Gets number of lines in a file."""
    with open(file, "r", encoding="utf-16") as f:
        out = sum(1 for _ in f) - 2
        if verbose:
            logging.info(f"File: {file}, lines: {out}")
        return out


def plot_model_comparison_win_rate(df, args, turn=1, debug=False):
    """Plot win rate results for maj, sc models as two separate lines on same
    plot.
    x-axis: DPO steps, y-axis: win rate over a baseline model
    blue line: maj models, orange line: sc models
    turn can be 1, 2 or 'avg' for average of both turns.
    """
    logging.info("Plotting model comparison win rate for turn %s", turn)


def plot_model_comparison_ten_point_scale(dd, args, turn=1, debug=False):
    """Plot ten point scale results for maj, sc models as two separate lines on same
    plot.
    x-axis: DPO steps, y-axis: average score
    blue line: maj models, orange line: sc models
    turn can be 1, 2 or 'avg' for average of both turns.
    """
    logging.info("Plotting model comparison for turn %s", turn)
    dd["dpo_steps"] = dd["model"].apply(
        lambda x: int(
            x.split("_")[next(i for i, v in enumerate(x.split("_")) if v.isdigit())]
        )
    )
    dd_maj = dd[dd["model"].str.contains("maj")]
    dd_sc = dd[dd["model"].str.contains("sc")]

    if turn == "avg":
        dd_maj = dd_maj.groupby(["model"]).mean()
        dd_sc = dd_sc.groupby(["model"]).mean()
    else:
        dd_maj = dd_maj[dd_maj["turn"] == turn].groupby(["model", "turn"]).mean()
        dd_sc = dd_sc[dd_sc["turn"] == turn].groupby(["model", "turn"]).mean()
    dd_maj = dd_maj.sort_values(by="dpo_steps")
    dd_sc = dd_sc.sort_values(by="dpo_steps")

    logging.info("Refining dpo_steps based on wc output of training data files")
    dpo_dataset_home = os.path.expanduser("~/llm-sct/data/reddit/")
    dpo_dataset = os.path.join(
        dpo_dataset_home, "processed/gpt35/maj_sc_v3/matched_prompts/1000_to_64000/"
    )

    dd_maj["dpo_steps"] = dd_maj["dpo_steps"].apply(
        lambda x: get_number_of_lines(os.path.join(dpo_dataset, f"maj_{int(x)}.json"))
    )
    dd_sc["dpo_steps"] = dd_sc["dpo_steps"].apply(
        lambda x: get_number_of_lines(os.path.join(dpo_dataset, f"sc_{int(x)}.json"))
    )
    logging.info("Done refining dpo_steps")

    if debug:
        logging.info("dd_maj head:")
        logging.info(dd_maj.head())
        logging.info("dd_sc head:")
        logging.info(dd_sc.head())

    logging.info("Plotting model score comparison")
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xticks(dd_maj["dpo_steps"].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.plot(dd_maj["dpo_steps"], dd_maj["score"], label="maj", marker="o")
    ax.plot(dd_sc["dpo_steps"], dd_sc["score"], label="sc", marker="o")
    ax.set_xlabel("DPO steps")
    ax.set_ylabel("Score")
    if turn == "avg":
        ax.set_title("Both turns average scores")
    else:
        ax.set_title(f"Turn {turn} scores")
    ax.legend()

    wandb.log({f"model_comparison_plot_turn_{turn}": wandb.Image(fig)})
    plt.close(fig)
    logging.info("Done plotting model score comparison for turn %s", turn)


def display_result_single(args, create_plots=False):
    """Display the results of single score judgment."""

    if wandb.run is None:
        wanb_name = "__".join([f"{value}_{key}" for key, value in vars(args).items()])
        wandb.init(project="dcpo_evals", name=wanb_name)

    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
    else:
        input_file = args.input_file
    print(f"Input file: {input_file}")

    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]
    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    print("\nFirst turn:")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))
    wandb.log({"first_turn_results": wandb.Table(data=df_1.reset_index())})
    if create_plots:
        fig = create_model_comparison_plot(
            df_1, score_column="score", model_column="model"
        )
        wandb.log({"model_comparison_plot_first_turn_results": wandb.Image(fig)})
        plt.close(fig)

    if args.bench_name == "mt_bench":
        print("\nSecond turn:")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))
        wandb.log({"second_turn_results": wandb.Table(data=df_2.reset_index())})
        if create_plots:
            fig = create_model_comparison_plot(
                df_2, score_column="score", model_column="model"
            )
            wandb.log({"model_comparison_plot_second_turn_results": wandb.Image(fig)})
            plt.close(fig)

        print("\nAverage:")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))
        wandb.log({"average_results": wandb.Table(data=df_3.reset_index())})
        if create_plots:
            fig = create_model_comparison_plot(
                df_3, score_column="score", model_column="model"
            )
            wandb.log({"model_comparison_plot_average_results": wandb.Image(fig)})
            plt.close(fig)

    for turn in [1, 2, "avg"]:
        plot_model_comparison_ten_point_scale(df, args, turn=turn)

    wandb.finish()


def create_model_comparison_plot(df, score_column="score", model_column="model"):
    """Create a bar plot comparing model performance.

    Basically written by Claude LLM."""
    df_plot = df.copy()
    if model_column not in df_plot.index.names:
        df_plot = df_plot.groupby(model_column)[score_column].mean().reset_index()

    df_sorted = df_plot.sort_values(by=score_column, ascending=False)

    fig_height = max(6, 0.5 * len(df_sorted))
    fig_width = 16
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.barplot(x=score_column, y=model_column, data=df_sorted, ax=ax)

    ax.set_title("Model Performance Comparison", fontsize=16)
    ax.set_xlabel("Average Score", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_xlim(left=0)
    ax.tick_params(axis="both", which="major", labelsize=10)

    for i, v in enumerate(df_sorted[score_column]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    return fig


def display_result_pairwise(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df_all = df_all[(df_all["g1_winner"] != "error") & (df_all["g2_winner"] != "error")]

    model_list = (
        df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
    )
    model_list = list(set(model_list))

    list_res = []
    # traverse df row by row
    for index, row in df_all.iterrows():
        if args.model_list is not None and row["model_1"] not in args.model_list:
            continue
        if args.baseline_model is not None:
            if args.baseline_model not in [row["model_1"], row["model_2"]]:
                continue
        if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]:
            list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
            list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
        else:
            if row["g1_winner"] == "model_1":
                winner = row["model_1"]
                loser = row["model_2"]
            else:
                winner = row["model_2"]
                loser = row["model_1"]
            list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
            list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})

    df = pd.DataFrame(list_res)
    df = df.groupby(["model"]).sum()

    # remove baseline model
    if args.baseline_model is not None:
        df = df[df.index != args.baseline_model]
    # add win rate
    df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
    df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
    # each tie counts as 0.5 win + 0.5 loss
    df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (
        df["win"] + df["loss"] + df["tie"]
    )
    # print(df.sort_values(by="win_rate", ascending=False))
    # print(df.sort_values(by="loss_rate", ascending=True))
    print(df.sort_values(by="win_rate_adjusted", ascending=False))


if __name__ == "__main__":

    verbose = True
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"logs/log_time{time.time()}.log"),
        ],
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.colheader_justify", "left")

    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4-turbo")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    args = parser.parse_args()
    if verbose:
        print("Arguments:")
        for key, value in vars(args).items():
            print(f"{key}: {value}")

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)
