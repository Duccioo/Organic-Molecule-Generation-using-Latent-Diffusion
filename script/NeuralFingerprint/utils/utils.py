import os
from datetime import datetime


def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def logs_and_print(
    folder_path: str,
    log_message: str = "",
    recrate: bool = False,
    dot_line: bool = False,
    print_message: bool = True,
    file_name: str = "log.txt",
):
    """
    Salva un messaggio di log in un file di testo.

    Args:
        folder_path (str): Percorso della cartella in cui salvare il file di log.
        log_message (str, optional): Messaggio di log. Defaults to "".
        recrate (bool, optional): Se True, il file di log sar  ricreato. Defaults to False.
        dot_line (bool, optional): Se True, il messaggio di log sar  una riga di punti. Defaults to False.
        print_message (bool, optional): Se True, il messaggio di log sar  anche stampato a video. Defaults to True.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Nome del file log (puoi personalizzarlo)
    log_file_path = os.path.join(folder_path, file_name)

    # Ottieni il timestamp corrente
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepara il messaggio di log con il timestamp
    full_log_message = f"[{timestamp}] {log_message}\n"

    if recrate:
        open(log_file_path, "w").close()

    # Scrivi il messaggio nel file di log
    with open(log_file_path, "a") as log_file:
        if dot_line:
            full_log_message = "-" * 20 + "\n"
        elif print_message:
            print(log_message)

        log_file.write(full_log_message)
