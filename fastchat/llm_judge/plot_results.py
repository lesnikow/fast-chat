import matplotlib.pyplot as plt

data = {
    "av_answers": 2.152439,
    "av_llama3-8B_voters_128_max_new_tokens": 2.152439,
    "hb_answers": 2.103659,
    "hb_helpful_base_control": 2.103659,
    "rmp_llama3-8B_voters": 2.079268,
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
    "rv_11_haiku_voters_dataset_dpo_loss_pythia28_mo": 1.893750,
    "mp_33_voters_dataset_dpo_loss_pythia28_32_batch": 1.875000,
    "mp_3_x_11_voters_dataset_dpo_loss_pythia28_32_b": 1.875000,
    "mp_answers_512_max_new_tokens": 1.856250,
    "mp_llama3-8B_voters_512_max_new_tokens": 1.856250,
    "rv_answers_512_max_new_tokens": 1.843750,
    "rv_llama3-8B_voters_512_max_new_tokens": 1.843750,
    "a_dpo_model_answers_max_token_128": 1.806250,
    "a_arm_original_dpo_loss_128_max_new_tokens": 1.806250,
    "av_llama3-8B_voters_512_max_new_tokens": 1.787500,
    "av_answers_512_max_new_tokens": 1.787500,
    "rv_33_voters_dataset_dpo_loss_pythia28_32_batch": 1.675000,
    "rv_3_x_11_voters_dataset_dpo_loss_pythia28_32_b": 1.675000,
}

models = list(data.keys())
scores = list(data.values())

plt.figure(figsize=(12, 8))
plt.barh(models, scores, color="skyblue")
plt.xlabel("Score")
plt.title("Scores of Models")
plt.gca().invert_yaxis()
plt.tight_layout()

# Save the plot
plt.savefig("scores.png")
