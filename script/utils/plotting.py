import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_losses(
    train_losses, val_losses, num_batches: int, output_file: str = "loss_plot.png", y_lim: float = None
):
    """
    Plotta le loss di training e validation. Se train_losses è un dizionario,
    crea una curva per ogni chiave.

    :param train_losses: Lista di valori delle loss di training o dizionario con multiple liste di loss.
    :param val_losses: Lista di valori delle loss di validation o dizionario con multiple liste di loss.
    :param output_file: Nome del file in cui salvare il plot.
    """
    plt.figure(figsize=(12, 8))  # Dimensioni della figura per adattarsi a più curve
    plt.ylim(0, y_lim) if y_lim else None

    freq_ticks = 5  # num of epoch to draw ticks

    # Calcola il numero di epoche in base alla lunghezza delle validation loss

    # Controlla se train_losses è un dizionario
    if isinstance(train_losses, dict):
        # Plotta ogni set di loss nel dizionario

        for key, values in train_losses.items():
            plt.plot(values, label=f"{key}")
            num_training_total = len(values)
            num_epochs = num_training_total // num_batches

    else:
        # Plotta direttamente le loss di training
        plt.plot(train_losses, label="Training Loss")
        num_epochs = len(train_losses) // num_batches
        num_training_total = len(train_losses)

    # Plotta le loss di validation
    # Espande val_losses per eguagliare la lunghezza dei batch se batches_per_epoch è specificato
    if num_batches:
        if isinstance(val_losses, dict):
            for key, values in val_losses.items():
                factor = num_training_total // len(values)
                expanded_val_losses = np.repeat(values, factor)
                plt.plot(expanded_val_losses, label=f"{key}")
        else:
            factor = num_training_total // len(val_losses)
            expanded_val_losses = np.repeat(val_losses, factor)
            plt.plot(expanded_val_losses, label="Validity")

    else:
        if isinstance(val_losses, dict):
            for key, values in val_losses.items():
                plt.plot(values, label=f"{key}")
        else:
            plt.plot(val_losses, label="Validity")

    # Aggiungi barre verticali rosse ogni 5 epoche
    for epoch in range(freq_ticks, num_epochs + 1, freq_ticks):
        plt.axvline(
            x=epoch * num_batches if num_batches else epoch,
            color="red",
            linestyle="--",
            alpha=0.3,
        )

    # Configurazione dell'asse delle x per mostrare solo il numero delle epoche
    if num_batches:

        plt.xticks(
            ticks=np.arange(0, num_training_total + 1, step=num_batches),
            labels=np.arange(0, num_epochs + 1),
        )
    else:
        plt.xticks(ticks=np.arange(0, num_epochs + 1), labels=np.arange(0, num_epochs + 1))

    # Configura il plot
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Salva l'immagine
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def plot_loss_curve(train_losses, val_losses=None, save_path=None):
    """
    Plotta la curva di loss durante l'addestramento e, opzionalmente, la validazione.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label="Training Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_reconstructions(reconstructions, data, num_images=10, save_path=None):
    """
    Plotta le immagini originali e le loro ricostruzioni.
    """

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axes[0, i].imshow(data[i].squeeze().cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructions[i].squeeze().cpu(), cmap="gray")
        axes[1, i].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def generate_diffusion_gif(model_ldm, num_images=64, save_filename=None, seed: int = 42, starting_steps=1000):
    device = next(model_ldm.parameters()).device
    noise_latent = torch.randn(num_images, model_ldm.latent_dim).to(device)
    list_img_gen = []

    # Reverse diffusion process
    tqdm_bar = tqdm(
        reversed(range(starting_steps)),
        total=len(range(starting_steps)),
        leave=False,
        position=1,
        desc="Generate GIF",
    )
    for i in tqdm_bar:
        with torch.no_grad():
            noise_pred = model_ldm(noise_latent, torch.as_tensor(i).unsqueeze(0).to(noise_latent.device))
            # Use scheduler to get x0 and xt-1
            noise_latent, _ = model_ldm.scheduler.sample_prev_timestep(
                noise_latent, noise_pred, torch.as_tensor(i).to(noise_latent.device)
            )

            # Decode latent to image space
            img = model_ldm.encoder_decoder.decode(noise_latent).cpu()
            list_img_gen.append(img)
    create_gif_from_tensors(tensor_list=list_img_gen, output_filename=save_filename, duration=50)


def plot_generated_images(model, latent_dim, num_images=64, save_path=None, seed: int = 42):
    """
    Genera e plotta immagini casuali dal modello.
    """

    model.eval()
    with torch.no_grad():
        # sample = torch.randn(num_images, latent_dim).to(next(model.parameters()).device)
        sample = torch.randn([num_images, latent_dim], generator=torch.Generator().manual_seed(seed)).to(
            next(model.parameters()).device
        )
        generated = model.decode(sample).cpu()

    grid = make_grid(generated, nrow=8, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    print("dio lupo")
    train_losses = [0.5, 0.4, 0.3, 0.2] * 5  # 100 batch con batch_per_epoch=100
    val_losses = [0.6, 0.5, 0.35, 0.25, 0.5]  # Una per ogni epoca

    plot_losses(train_losses, val_losses, batch_size=4, output_file="prova_1.png")
