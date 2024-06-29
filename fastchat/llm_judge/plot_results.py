import matplotlib.pyplot as plt

data = {
    "av_answers": 2.152439,
    "av_llama3-8B_voters_128_max_new_tokens": 2.152439,
    "hb_answers": 2.103659,
    "hb_helpful_base_control_128_max_new_tokens": 2.103659,
    "rmp_llama3-8B_voters_128_max_new_tokens": 2.079268,
    "rmp_answers": 2.079268,
    "rv_11_gpt35_voters_dataset_dpo_loss_pythia28": 2.056250,
    "mp_11_haiku_voters_dataset_dpo_loss_pythia28": 2.031250,
    "mp_11_gpt35_voters_dataset_dpo_loss_pythia28": 2.031250,
    "rv_answers": 1.975610,
    "rv_llama3-8B_voters_128_max_new_tokens": 1.975610,
    "mp_llama3-8B_voters_128_max_new_tokens": 1.963415,
    "mp_answers": 1.963415,
    "b_dcpo_policy_answers_max_new_token_128": 1.956250,
    "b_arm_original_dcpo_loss_128_max_new_tokens": 1.956250,
    "rmp_answers_512_max_new_tokens": 1.956250,
    "rmp_llama3-8B_voters_512_max_new_tokens": 1.956250,
    "hb_answers_512_max_new_tokens": 1.931250,
    "hb_helpful_base_control_512_max_new_tokens": 1.931250,
    "rv_11_haiku_voters_dataset_dpo_loss_pythia28": 1.893750,
    "mp_3_x_11_voters_dataset_dpo_loss_pythia28_32": 1.875000,
    "mp_answers_512_max_new_tokens": 1.856250,
    "mp_llama3-8B_voters_512_max_new_tokens": 1.856250,
    "rv_answers_512_max_new_tokens": 1.843750,
    "rv_llama3-8B_voters_512_max_new_tokens": 1.843750,
    "a_dpo_model_answers_max_token_128": 1.806250,
    "a_arm_original_dpo_loss_128_max_new_tokens": 1.806250,
    "av_llama3-8B_voters_512_max_new_tokens": 1.787500,
    "av_answers_512_max_new_tokens": 1.787500,
    "rv_3_x_11_voters_dataset_dpo_loss_pythia28_32": 1.675000,
}

models = list(data.keys())
scores = list(data.values())

plt.figure(figsize=(12, 8))
plt.barh(models, scores, color="skyblue")
plt.xlabel("Score")
plt.title("Scores of Models")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("all_scores.png")


# Compare only rv and mp models
rv_models = [model for model in models if model.startswith("rv")]
mp_models = [model for model in models if model.startswith("mp")]

# Remove some answers not wanting to plot
rv_models.remove("rv_answers")
mp_models.remove("mp_answers")
rv_models.remove("rv_answers_512_max_new_tokens")
mp_models.remove("mp_answers_512_max_new_tokens")
rv_models.remove("rv_llama3-8B_voters_512_max_new_tokens")
mp_models.remove("mp_llama3-8B_voters_512_max_new_tokens")

plt.figure(figsize=(12, 8))
plt.barh(rv_models, [data[model] for model in rv_models], color="skyblue", label="rv")
plt.barh(mp_models, [data[model] for model in mp_models], color="orange", label="mp")
plt.xlabel("Score")
plt.title("Scores of Models")
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()

plt.savefig("rv_vs_mp_scores.png")


# Do as above but group rv, mp models together that share the same voter groups
# Do two different colors for rv vs mp in same group.

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


def plot_models(models):
    for i, model in enumerate(models):
        if "mp" in model:
            plt.barh(model, data[model], color="orange", label="mp" if i == 0 else "")
        elif "rv" in model:
            plt.barh(model, data[model], color="skyblue", label="rv" if i == 0 else "")


plot_models(ensemble_models)
plot_models(haiku_models)
plot_models(gpt_models)
plot_models(llama_models)


plt.xlabel("Score")
plt.title("Scores of Models")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("grouped_scores.png")
