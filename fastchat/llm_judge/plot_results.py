import matplotlib.pyplot as plt

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

models = list(scores_two_turns_average.keys())
scores = list(scores_two_turns_average.values())


def plot_all_model_scores():
    plt.figure(figsize=(12, 8))
    plt.barh(models, scores, color="skyblue")
    plt.xlabel("Score")
    plt.title("Scores of All Models, Two Turn Average")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig("all_model_scores.png")


def plot_rv_vs_mp_basic():
    # Compare only rv and mp models
    rv_models = [model for model in models if model.startswith("rv")]
    mp_models = [model for model in models if model.startswith("mp")]

    # Remove some answers not wanting to plot
    rv_models.remove("rv_llama3-8B_voters_512_max_new_tokens")
    mp_models.remove("mp_llama3-8B_voters_512_max_new_tokens")

    plt.figure(figsize=(12, 8))
    plt.barh(
        rv_models,
        [scores_two_turns_average[model] for model in rv_models],
        color="skyblue",
        label="rv",
    )
    plt.barh(
        mp_models,
        [scores_two_turns_average[model] for model in mp_models],
        color="orange",
        label="mp",
    )
    plt.xlabel("Score")
    plt.title("Scores of Models, Two Turn Average")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()

    plt.savefig("rv_vs_mp_scores.png")


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

    # Sort models to have mp first, then rv second
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

    plot_models(ensemble_models, scores_two_turns_average)
    plot_models(haiku_models, scores_two_turns_average)
    plot_models(gpt_models, scores_two_turns_average)
    plot_models(llama_models, scores_two_turns_average)

    plt.xlabel("Score")
    plt.title("Majority Preference vs Random Voter, Two Turn Average")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig("mp_vs_rv_two_turns.png")


if __name__ == "__main__":
    plot_all_model_scores()
    plot_rv_vs_mp_grouped()
