import matplotlib.pyplot as plt

scores_first_turn = {
    "AV_llama3-8B_voters_128_max_new_tokens": 2.500000,
    "hb_helpful_base_control_128_max_new_tokens": 2.426829,
    "rv_11_gpt35_voters_dataset_dpo_loss_pythia28_model": 2.425000,
    "mp_11_haiku_voters_dataset_dpo_loss_pythia28_model": 2.400000,
    "mp_11_gpt35_voters_dataset_dpo_loss_pythia28_model": 2.325000,
    "RMP_llama3-8B_voters_128_max_new_tokens": 2.292683,
    "hb_helpful_base_control_512_max_new_tokens": 2.287500,
    "rv_llama3-8B_voters_128_max_new_tokens": 2.280488,
    "mp_llama3-8B_voters_128_max_new_tokens": 2.243902,
    "rv_11_haiku_voters_dataset_dpo_loss_pythia28_model": 2.212500,
    "b_arm_original_dcpo_loss_128_max_new_tokens": 2.200000,
    "mp_3_x_11_voters_dataset_dpo_loss_pythia28_32_batch_size": 2.200000,
    "a_arm_original_dpo_loss_128_max_new_tokens": 2.125000,
    "AV_llama3-8B_voters_512_max_new_tokens": 2.125000,
    "mp_llama3-8B_voters_512_max_new_tokens": 2.062500,
    "RMP_llama3-8B_voters_512_max_new_tokens": 2.025000,
    "rv_llama3-8B_voters_512_max_new_tokens": 2.025000,
    "rv_3_x_11_voters_dataset_dpo_loss_pythia28_32_batch_size": 1.912500,
}

scores_second_turn = {
    "RMP_llama3-8B_voters_512_max_new_tokens": 1.887500,
    "RMP_llama3-8B_voters_128_max_new_tokens": 1.865854,
    "AV_llama3-8B_voters_128_max_new_tokens": 1.804878,
    "hb_helpful_base_control_128_max_new_tokens": 1.780488,
    "mp_11_gpt35_voters_dataset_dpo_loss_pythia28_model": 1.737500,
    "b_arm_original_dcpo_loss_128_max_new_tokens": 1.712500,
    "rv_11_gpt35_voters_dataset_dpo_loss_pythia28_model": 1.687500,
    "mp_llama3-8B_voters_128_max_new_tokens": 1.682927,
    "rv_llama3-8B_voters_128_max_new_tokens": 1.670732,
    "rv_llama3-8B_voters_512_max_new_tokens": 1.662500,
    "mp_11_haiku_voters_dataset_dpo_loss_pythia28_model": 1.662500,
    "mp_llama3-8B_voters_512_max_new_tokens": 1.650000,
    "hb_helpful_base_control_512_max_new_tokens": 1.575000,
    "rv_11_haiku_voters_dataset_dpo_loss_pythia28_model": 1.575000,
    "mp_3_x_11_voters_dataset_dpo_loss_pythia28_32_batch_size": 1.550000,
    "a_arm_original_dpo_loss_128_max_new_tokens": 1.487500,
    "AV_llama3-8B_voters_512_max_new_tokens": 1.450000,
    "rv_3_x_11_voters_dataset_dpo_loss_pythia28_32_batch_size": 1.437500,
}

scores_two_turns_average = {
    "AV_llama3-8B_voters_128_max_new_tokens": 2.152439,
    "hb_helpful_base_control_128_max_new_tokens": 2.103659,
    "RMP_llama3-8B_voters_128_max_new_tokens": 2.079268,
    "rv_11_gpt35_voters_dataset_dpo_loss_pythia28": 2.056250,
    "mp_11_haiku_voters_dataset_dpo_loss_pythia28": 2.031250,
    "mp_11_gpt35_voters_dataset_dpo_loss_pythia28": 2.031250,
    "rv_llama3-8B_voters_128_max_new_tokens": 1.975610,
    "mp_llama3-8B_voters_128_max_new_tokens": 1.963415,
    "b_arm_original_dcpo_loss_128_max_new_tokens": 1.956250,
    "RMP_llama3-8B_voters_512_max_new_tokens": 1.956250,
    "hb_helpful_base_control_512_max_new_tokens": 1.931250,
    "rv_11_haiku_voters_dataset_dpo_loss_pythia28": 1.893750,
    "mp_3_x_11_voters_dataset_dpo_loss_pythia28_32": 1.875000,
    "mp_llama3-8B_voters_512_max_new_tokens": 1.856250,
    "rv_llama3-8B_voters_512_max_new_tokens": 1.843750,
    "a_arm_original_dpo_loss_128_max_new_tokens": 1.806250,
    "AV_llama3-8B_voters_512_max_new_tokens": 1.787500,
    "rv_3_x_11_voters_dataset_dpo_loss_pythia28_32": 1.675000,
}


def plot_pairwise_comparisons(mp_or_rmp="mp"):
    model_win_rates = {
        "MP all": 0.534375,
        "RV all": 0.465625,
        "MP gpt35": 0.51875,
        "RV gpt35": 0.48125,
        "MP haiku": 0.537736,
        "RV haiku": 0.462264,
        "MP llama3": 0.534375,
        "RV llama3": 0.465625,
    }

    model_win_rates_rmp = {
        "RMP llama3": 0.475,
        "AV llama3": 0.525,
    }

    models = (
        list(model_win_rates.keys())
        if mp_or_rmp == "mp"
        else list(model_win_rates_rmp.keys())
    )
    win_rate_adjusted = (
        list(model_win_rates.values())
        if mp_or_rmp == "mp"
        else list(model_win_rates_rmp.values())
    )

    plt.figure(figsize=(12, 8)) if mp_or_rmp == "mp" else plt.figure(figsize=(12, 4))
    plt.barh(
        models[::-1],
        win_rate_adjusted[::-1],
        color=[
            "skyblue",
            "orange",
            "skyblue",
            "orange",
            "skyblue",
            "orange",
            "skyblue",
            "orange",
        ],
    )

    plt.title("Pairwise Comparisons, Win Rate Adjusted, Pythia2.8B")
    plt.xlabel("Win Rate Adjusted")
    plt.ylabel("Model")
    plt.xticks(rotation=90)
    plt.axvline(x=0.5, color="red", linestyle="--")
    plt.legend(["Win Rate = 0.5"], loc="lower right")
    plt.tight_layout()

    plt.savefig(
        f"reports/figures/pairwise_comparisons_win_rate_adjusted_{mp_or_rmp}_mode.png"
    )


def plot_rv_vs_mp_grouped(turn_int=0, mp_or_rmp="mp"):
    scores = [scores_two_turns_average, scores_first_turn, scores_second_turn]
    turn_options = ["Two Turns Average", "First Turn", "Second Turn"]

    selected_scores = scores[turn_int]
    selected_turn = turn_options[turn_int]

    models = list(selected_scores.keys())
    scores = list(selected_scores.values())

    gpt_models = [model for model in models if "gpt35" in model]
    haiku_models = [model for model in models if "haiku" in model]
    llama_models = (
        [
            model
            for model in models
            if "llama" in model
            and ("rv" in model or "mp" in model)
            and "RMP" not in model
            and "512" not in model
        ]
        if mp_or_rmp == "mp"
        else [
            model
            for model in models
            if "llama" in model
            and ("RMP" in model or "AV" in model)
            and "512" not in model
        ]
    )
    ensemble_models = [
        model
        for model in models
        if ("3_x_11" in model or "all" in model or "ensemble" in model)
    ]

    gpt_models = sorted(gpt_models, key=lambda x: x.split("_")[0])
    haiku_models = sorted(haiku_models, key=lambda x: x.split("_")[0])
    llama_models = sorted(llama_models, key=lambda x: x.split("_")[0])
    ensemble_models = sorted(ensemble_models, key=lambda x: x.split(" ")[0])

    gpt_models = gpt_models[::-1] if mp_or_rmp == "rmp" else gpt_models
    haiku_models = haiku_models[::-1] if mp_or_rmp == "rmp" else haiku_models
    llama_models = llama_models[::-1] if mp_or_rmp == "rmp" else llama_models
    ensemble_models = ensemble_models[::-1] if mp_or_rmp == "rmp" else ensemble_models

    def plot_models(models, data):
        search_str_1, search_str_2 = (
            ("mp", "rv") if mp_or_rmp == "mp" else ("RMP", "AV")
        )
        for i, model in enumerate(models):
            if search_str_1 in model:
                plt.barh(
                    model,
                    data[model],
                    color="orange",
                    label=search_str_1 if i == 0 else "",
                )
            elif search_str_2 in model:
                plt.barh(
                    model,
                    data[model],
                    color="skyblue",
                    label=search_str_2 if i == 0 else "",
                )

    plt.figure(figsize=(12, 8)) if mp_or_rmp == "mp" else plt.figure(figsize=(12, 4))

    plot_models(ensemble_models, selected_scores)
    plot_models(haiku_models, selected_scores)
    plot_models(gpt_models, selected_scores)
    plot_models(llama_models, selected_scores)

    plt.xlabel("Score")
    plt.ylabel("Model")
    plt.xlim(0, 2.5)
    plt.title(f"Majority Preference vs Random Voter, {selected_turn}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        f"reports/figures/mp_vs_rv_{selected_turn}_{mp_or_rmp}_mode.png".lower().replace(
            " ", "_"
        )
    )


if __name__ == "__main__":
    plot_rv_vs_mp_grouped_bool = True
    plot_pairwise_comparisons_bool = False

    if plot_rv_vs_mp_grouped_bool:
        for turn_int in range(3):
            plot_rv_vs_mp_grouped(turn_int=turn_int, mp_or_rmp="mp")
            plot_rv_vs_mp_grouped(turn_int=turn_int, mp_or_rmp="rmp")

    if plot_pairwise_comparisons_bool:
        plot_pairwise_comparisons(mp_or_rmp="mp")
        plot_pairwise_comparisons(mp_or_rmp="rmp")
