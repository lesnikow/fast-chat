import matplotlib.pyplot as plt

scores_first_turn = {
    "av_llama3-8B_voters_128_max_new_tokens": 2.500000,
    "hb_helpful_base_control_128_max_new_tokens": 2.426829,
    "rv_11_gpt35_voters_dataset_dpo_loss_pythia28_model": 2.425000,
    "mp_11_haiku_voters_dataset_dpo_loss_pythia28_model": 2.400000,
    "mp_11_gpt35_voters_dataset_dpo_loss_pythia28_model": 2.325000,
    "rmp_llama3-8B_voters_128_max_new_tokens": 2.292683,
    "hb_helpful_base_control_512_max_new_tokens": 2.287500,
    "rv_llama3-8B_voters_128_max_new_tokens": 2.280488,
    "mp_llama3-8B_voters_128_max_new_tokens": 2.243902,
    "rv_11_haiku_voters_dataset_dpo_loss_pythia28_model": 2.212500,
    "b_arm_original_dcpo_loss_128_max_new_tokens": 2.200000,
    "mp_3_x_11_voters_dataset_dpo_loss_pythia28_32_batch_size": 2.200000,
    "a_arm_original_dpo_loss_128_max_new_tokens": 2.125000,
    "av_llama3-8B_voters_512_max_new_tokens": 2.125000,
    "mp_llama3-8B_voters_512_max_new_tokens": 2.062500,
    "rmp_llama3-8B_voters_512_max_new_tokens": 2.025000,
    "rv_llama3-8B_voters_512_max_new_tokens": 2.025000,
    "rv_3_x_11_voters_dataset_dpo_loss_pythia28_32_batch_size": 1.912500,
}

scores_second_turn = {
    "rmp_llama3-8B_voters_512_max_new_tokens": 1.887500,
    "rmp_llama3-8B_voters_128_max_new_tokens": 1.865854,
    "av_llama3-8B_voters_128_max_new_tokens": 1.804878,
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
    "av_llama3-8B_voters_512_max_new_tokens": 1.450000,
    "rv_3_x_11_voters_dataset_dpo_loss_pythia28_32_batch_size": 1.437500,
}

scores_two_turns_average = {
    "av_llama3-8B_voters_128_max_new_tokens": 2.152439,
    "hb_helpful_base_control_128_max_new_tokens": 2.103659,
    "rmp_llama3-8B_voters_128_max_new_tokens": 2.079268,
    "rv_11_gpt35_voters_dataset_dpo_loss_pythia28": 2.056250,
    "mp_11_haiku_voters_dataset_dpo_loss_pythia28": 2.031250,
    "mp_11_gpt35_voters_dataset_dpo_loss_pythia28": 2.031250,
    "rv_llama3-8B_voters_128_max_new_tokens": 1.975610,
    "mp_llama3-8B_voters_128_max_new_tokens": 1.963415,
    "b_arm_original_dcpo_loss_128_max_new_tokens": 1.956250,
    "rmp_llama3-8B_voters_512_max_new_tokens": 1.956250,
    "hb_helpful_base_control_512_max_new_tokens": 1.931250,
    "rv_11_haiku_voters_dataset_dpo_loss_pythia28": 1.893750,
    "mp_3_x_11_voters_dataset_dpo_loss_pythia28_32": 1.875000,
    "mp_llama3-8B_voters_512_max_new_tokens": 1.856250,
    "rv_llama3-8B_voters_512_max_new_tokens": 1.843750,
    "a_arm_original_dpo_loss_128_max_new_tokens": 1.806250,
    "av_llama3-8B_voters_512_max_new_tokens": 1.787500,
    "rv_3_x_11_voters_dataset_dpo_loss_pythia28_32": 1.675000,
}


selected_scores = scores_two_turns_average
turn_options = ["First Turn", "Second Turn", "Two Turns Average"]
selected_turn = turn_options[2]
models = list(selected_scores.keys())
scores = list(selected_scores.values())


def plot_all_model_scores():
    plt.figure(figsize=(12, 8))
    plt.barh(models, scores, color="skyblue")
    plt.xlabel("Score")
    plt.ylabel("Models")
    plt.title(f"Scores of All Models, {selected_turn}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        f"reports/figures/all_model_scores_{selected_turn}.png".lower().replace(
            " ", "_"
        )
    )


def plot_rv_vs_mp_basic():
    rv_models = [model for model in models if model.startswith("rv")]
    mp_models = [model for model in models if model.startswith("mp")]

    rv_models.remove("rv_llama3-8B_voters_512_max_new_tokens")
    mp_models.remove("mp_llama3-8B_voters_512_max_new_tokens")

    plt.figure(figsize=(12, 8))
    plt.barh(
        rv_models,
        [selected_scores[model] for model in rv_models],
        color="skyblue",
        label="rv",
    )
    plt.barh(
        mp_models,
        [selected_scores[model] for model in mp_models],
        color="orange",
        label="mp",
    )
    plt.xlabel("Score")
    plt.title(f"Scores of Models, {selected_turn}")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"reports/figures/rv_vs_mp_scores_{selected_turn}.png".lower().replace(" ", "_")
    )


def plot_rv_vs_mp_grouped():
    gpt_models = [model for model in models if "gpt35" in model]
    haiku_models = [model for model in models if "haiku" in model]
    llama_models = [
        model
        for model in models
        if "llama" in model
        and ("rv" in model or "mp" in model)
        and "rmp" not in model
        and "512" not in model
    ]

    gpt_models = sorted(gpt_models, key=lambda x: x.split("_")[0])
    haiku_models = sorted(haiku_models, key=lambda x: x.split("_")[0])
    llama_models = sorted(llama_models, key=lambda x: x.split("_")[0])
    ensemble_models = [model for model in models if "3_x_11" in model]
    plt.figure(figsize=(12, 8))

    def plot_models(models, data):
        for i, model in enumerate(models):
            if "mp" in model:
                plt.barh(
                    model, data[model], color="orange", label="mp" if i == 0 else ""
                )
            elif "rv" in model:
                plt.barh(
                    model, data[model], color="skyblue", label="rv" if i == 0 else ""
                )

    plot_models(ensemble_models, selected_scores)
    plot_models(haiku_models, selected_scores)
    plot_models(gpt_models, selected_scores)
    plot_models(llama_models, selected_scores)

    plt.xlabel("Score")
    plt.ylabel("Models")
    plt.title(f"Majority Preference vs Random Voter, {selected_turn}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        f"reports/figures/mp_vs_rv_{selected_turn}.png".lower().replace(" ", "_")
    )


def plot_pairwise_comparisons():
    models = [
        "mp_llama3-8B_voters_128_max_new_tokens",
        "rv_llama3-8B_voters_128_max_new_tokens",
        "mp_11_gpt35_voters_dataset_dpo_loss_pythia28",
        "rv_11_gpt35_voters_dataset_dpo_loss_pythia28",
        "mp_11_haiku_voters_dataset_dpo_loss_pythia28",
        "rv_11_haiku_voters_dataset_dpo_loss_pythia28",
        "mp_3_x_11_voters_dpo_loss_pythia28_32_batch_size",
        "rv_3_x_11_voters_dpo_loss_pythia28_32_batch_size",
    ]

    win_rate_adjusted = [
        0.534375,
        0.465625,
        0.51875,
        0.48125,
        0.537736,
        0.462264,
        0.534375,
        0.465625,
    ]

    plt.figure(figsize=(12, 8))
    plt.barh(
        models[::-1],
        win_rate_adjusted[::-1],
        color=[
            "orange",
            "skyblue",
            "orange",
            "skyblue",
            "orange",
            "skyblue",
            "orange",
            "skyblue",
        ],
    )

    # Plot vertical line at 0.5, the threshold for a model to be considered better
    plt.axvline(x=0.5, color="red", linestyle="--")

    plt.legend(["Win Rate = 0.5"])

    plt.title("Win Rate Adjusted for Different Models")
    plt.xlabel("Win Rate Adjusted")
    plt.ylabel("Models")
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig("reports/figures/pairwise_comparisons.png")


if __name__ == "__main__":
    plot_all_model_scores()
    plot_rv_vs_mp_grouped()
    plot_pairwise_comparisons()
