In this readme file, I explain to you on how to use the Mixture-of-Experts model to make an MoE with your desired model:cc




- `create_moe(...)` sets up two experts from whatever `ExpertClass` you provide. These experts could be like `LSTMAutoencoder` or a different model. 
- `train_moe(...)` does the gating logic and updates each expert’s weights separately, using dynamic thresholds.
- `evaluate_moe(...)` finalizes the gating on your test set, expands window-level anomalies to time steps, and stores results in `moe['final_scores']` and `moe['final_preds']`.
- `plot_expert1(...)`, `plot_expert2(...)`, `plot_final_moe(...)` let you visualize each piece.

You can rename or expand these functions to replicate the exact behavior of your prior big class.