import matplotlib.pyplot as plt


def plot_loss(training_loss: list, validation_loss: list, model_folder: str = "", fold: int = 0):
    plt.plot(training_loss, label="Training Loss")

    for metric in validation_loss[0].keys():
        if metric == "Pearson Correlation":
            continue
        val_metric = [x[metric] for x in validation_loss]
        plt.plot(val_metric, label=f"Validation {metric}")

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if model_folder:
        plt.savefig(f"{model_folder}/loss_fold_{fold}.png")
    plt.close()
