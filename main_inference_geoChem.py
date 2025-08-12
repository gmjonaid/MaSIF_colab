# Standard imports
import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from pathlib import Path

# Custom data loader and model
from data import ProteinPairsSurfaces, CenterPairAtoms, load_protein_pair
from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from model import dMaSIF
from data_iteration import iterate
from Arguments import parser

args = parser.parse_args()
model_path = "models/" + args.experiment_name
save_predictions_path = Path("preds/" + args.experiment_name)

# Ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Dataset
transformations = (
    Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
    if args.random_rotation
    else Compose([NormalizeChemFeatures()])
)

if args.single_pdb != "":
    single_data_dir = Path("./data_preprocessing/npys/")
    test_dataset = [load_protein_pair(args.single_pdb, single_data_dir, single_pdb=True)]
    test_pdb_ids = [args.single_pdb]
elif args.pdb_list != "":
    with open(args.pdb_list) as f:
        pdb_list = f.read().splitlines()
    single_data_dir = Path("./data_preprocessing/npys/")
    test_dataset = [load_protein_pair(pdb, single_data_dir, single_pdb=True) for pdb in pdb_list]
    test_pdb_ids = [pdb for pdb in pdb_list]
else:
    test_dataset = ProteinPairsSurfaces(
        "surface_data", train=False, ppi=args.search, transform=transformations
    )
    test_pdb_ids = (
        np.load("surface_data/processed/testing_pairs_data_ids.npy")
        if args.site
        else np.load("surface_data/processed/testing_pairs_data_ids_ppi.npy")
    )
    test_dataset = [
        (data, pdb_id)
        for data, pdb_id in zip(test_dataset, test_pdb_ids)
        if iface_valid_filter(data)
    ]
    test_dataset, test_pdb_ids = list(zip(*test_dataset))

# DataLoader
batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=batch_vars)

# Model
net = dMaSIF(args)
net.load_state_dict(torch.load(model_path, map_location=args.device)["model_state_dict"])
net = net.to(args.device)

# --- Geometry prediction ---
args.geom = True
args.chem = False
args.site = False
geom_save_path = save_predictions_path / "geometry"
iterate(net, test_loader, None, args, test=True, save_path=geom_save_path, pdb_ids=test_pdb_ids)

# --- Chemical prediction ---
args.geom = False
args.chem = True
args.site = False
chem_save_path = save_predictions_path / "chemical"
iterate(net, test_loader, None, args, test=True, save_path=chem_save_path, pdb_ids=test_pdb_ids)

# --- Binding site prediction ---
args.geom = False
args.chem = False
args.site = True
site_save_path = save_predictions_path / "site"
iterate(net, test_loader, None, args, test=True, save_path=site_save_path, pdb_ids=test_pdb_ids)

print(f"Predictions saved to:\n"
      f" - Geometry: {geom_save_path}\n"
      f" - Chemical: {chem_save_path}\n"
      f" - Site:     {site_save_path}")
