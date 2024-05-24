import optuna
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('optuna_results.csv')

best_trial_idx = df['best_val_loss'].idxmin()
best_trial = df.loc[best_trial_idx]

def highlight_best(ax, x, y, text, best_x, best_y):
    ax.scatter(best_x, best_y, color='red', s=100, edgecolor='black', zorder=5, label='Best')
    ax.annotate(text, (best_x, best_y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')

fig, ax = plt.subplots()
ax.plot(df.index, df['best_val_loss'], marker='o')
highlight_best(ax, df.index, df['best_val_loss'], 'Best', best_trial_idx, df.loc[best_trial_idx, 'best_val_loss'])
ax.set_title('Convergence Plot of Validation Loss over 80 Trials')
ax.set_xlabel('Trial')
ax.set_ylabel('Validation Loss')
plt.savefig('convergence_validation_loss.png')
plt.close()

fig, axs = plt.subplots(3, 2, figsize=(15, 20))
fig.suptitle('Tuning of the Hyperparameters with Optuna', fontsize=16)

params = ['batch_size', 'weight_decay', 'learning_rate', 'dropout_rate', 'hidden_units', 'hidden_layers']
for i, param in enumerate(params):
    ax = axs[i//2, i%2]
    ax.scatter(df[param], df['best_val_loss'])
    highlight_best(ax, df[param], df['best_val_loss'], 'Best', df.loc[best_trial_idx, param], df.loc[best_trial_idx, 'best_val_loss'])
    ax.set_title(f'Validation Loss vs {param.replace("_", " ").title()}')
    ax.set_xlabel(param.replace("_", " ").title())
    ax.set_ylabel('Validation Loss')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('tuning_hyperparameters.png')
plt.close()
