# Repository Notes

## Training Reports

After every training run and every pretraining run, create a short text report with the results.

All reports must be written in Russian.

Reports should be stored in:

- `D:\diploma-thesis\reports`

Use the filename format:

- `YYYY-MM-DD_HH-MM_<experiment-name>.md`

The report should be saved in text form and should include at least:

- date and time of the run
- experiment name
- training type (`pretraining`, `linear evaluation`, `fine-tuning`, or similar)
- reason for stopping or finishing the run
- full launch command
- all important Hydra overrides
- dataset used
- key hyperparameters
- augmentation setup when relevant
- hardware and environment details when relevant
- checkpoint path
- latest checkpoint available at the moment this specific run ended
- recommended checkpoint for future use
- final metrics or loss values
- short interpretation of the result
- extended interpretation of the result when useful, including observations about training dynamics, stability, bottlenecks, plateau behavior, and next recommended steps
- comparison with previous relevant runs when such runs exist

If important training hyperparameters are changed in a meaningful way, treat the new run as a new experiment instead of a simple continuation.

Use resume only when continuing the same experiment logic.

The purpose of this report is to keep a lightweight experiment history in human-readable form.

## Commit Messages

Commit messages for this repository should be written in Russian by default.

Use short, concrete messages that describe the result of the change rather than vague labels like `fix`, `update`, or `misc`.
