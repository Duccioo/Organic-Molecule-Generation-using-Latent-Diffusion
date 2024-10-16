import rdkit.Chem as Chem


def calc_metrics(
    smiles_true: list = ["CCO", "CCN", "CCO", "CCC"],
    smiles_predicted: list = ["CCO", "CCN", "CCC", "CCF"],
):
    # rimuovo gli H dagli smiles veri:
    smiles_true = remove_H(smiles_true)

    # calcolo gli smiles validi:
    validity_smiles_list = [calculate_validity(mol) for mol in smiles_predicted]

    # prendo una lista di smiles unici generati
    unique_smiles_list = list(set(validity_smiles_list))
    unique_smiles_list.remove(False)

    validity = sum(1.0 for elemento in validity_smiles_list if elemento is not False) / len(smiles_predicted)
    uniqueness = calculate_uniqueness(validity_smiles_list)
    novelty = calculate_novelty(validity_smiles_list, smiles_true)

    return unique_smiles_list, validity, uniqueness, novelty


def remove_H(list_smiles: list = []) -> list:
    r_list = []
    for smiles in list_smiles:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # print(mol)
        if mol is None:
            print("mhhhh guess")
            continue

        mol = Chem.RemoveHs(mol=mol, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
        except:
            pass

        new_smiles = Chem.MolToSmiles(mol)
        r_list.append(new_smiles)

    return r_list


def calculate_validity(smiles):
    if smiles == "" or smiles == None:
        return False

    # Convert SMILES to molecule object with sanitize=False
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is not None:
        # print("Molecule object created successfully!")

        # Sanitize the molecule for validation
        try:
            Chem.SanitizeMol(molecule)
            return smiles
        except Exception as inst:
            if len(smiles) <= 8:
                print("Invalid SMILES: %s" % smiles, "\t", inst)
            return False
    else:
        return False


def calculate_uniqueness(list_valid_smiles_generated: list = []) -> float:
    uniqueness: float = 0.0

    list_valid_smiles_generated.remove(False)
    unique_valid_smiles: set = set(list_valid_smiles_generated)
    unique_valid_smiles.discard(False)

    if len(unique_valid_smiles) == 0:
        return 0.0

    else:
        uniqueness = len(unique_valid_smiles) / len(list_valid_smiles_generated)

    return uniqueness


def calculate_novelty(list_valid_smiles_generated: list, molecole_reale: list) -> float:
    novelty: float = 0.0

    list_valid_smiles_generated.remove(False)
    unique_valid_smiles_generated = set(list_valid_smiles_generated)
    unique_valid_smiles_generated.discard(False)

    # print("---???...____//////")
    # print([stringa.ljust(5)[:5] for stringa in molecole_reale])

    if len(unique_valid_smiles_generated) == 0:
        return 0.0
    else:
        unique_molecules_reali = set(molecole_reale)

        # from paper https://arxiv.org/pdf/1802.03480.pdf :
        novelty: float = 1 - (len(unique_valid_smiles_generated & unique_molecules_reali)) / len(
            unique_valid_smiles_generated
        )

    return novelty
