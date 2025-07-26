import torch

import utils as utils


class ExtraMolecularFeatures:
    def __init__(self, molecular_features_type, dataset_infos):
        self.charge = ChargeFeature(remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies)
        self.valency = ValencyFeature()
        self.weight = WeightFeature(max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights)
        # self.atom_feat = AtomicFeatures(atom_rows=dataset_infos.rows,
        #                                 atom_groups=dataset_infos.groups,
        #                                 atom_families=dataset_infos.families)
        self.molecular_features_type = molecular_features_type

    def __call__(self, z_t):
        charge = self.charge(z_t).unsqueeze(-1)      # (bs, n, 1)
        valency = self.valency(z_t).unsqueeze(-1)    # (bs, n, 1)
        weight = self.weight(z_t)                    # (bs, 1)
        # atom_feat = self.atom_feat(z_t)              # (bs, n, 3)

        extra_edge_attr = torch.zeros((*z_t.E.shape[:-1], 0), device=z_t.E.device)
        return utils.PlaceHolder(X=torch.cat((charge, valency), dim=-1), E=extra_edge_attr, y=weight)

    def update_input_dims(self, input_dims):
        if self.molecular_features_type is None:
            return input_dims
        elif self.molecular_features_type == "all":
            input_dims.X += 2
            input_dims.y += 1
            return input_dims
        else:
            raise NotImplementedError(f"'{self.molecular_features_type}' molecular feature type not implemented.")


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, z_t):
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=z_t.E.device).reshape(1, 1, 1, -1)
        weighted_E = z_t.E * bond_orders      # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)   # (bs, n)

        valencies = torch.tensor(self.valencies, device=z_t.X.device).reshape(1, 1, -1)
        X = z_t.X * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)               # (bs, n)

        return (normal_valencies - current_valencies).float()


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, z_t):
        orders = torch.tensor([0, 1, 2, 3, 1.5], device=z_t.E.device).reshape(1, 1, 1, -1)
        E = z_t.E * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies.to(z_t.X.device)


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.Tensor(list(atom_weights.values()))

    def __call__(self, z_t):
        X = torch.argmax(z_t.X, dim=-1)     # (bs, n)
        X_weights = self.atom_weight_list.to(X.device)[X]            # (bs, n)
        return X_weights.sum(dim=-1).unsqueeze(-1).to(z_t.X.device) / self.max_weight     # (bs, 1)


class AtomicFeatures:
    def __init__(self, atom_rows, atom_groups, atom_families):
        self.atom_rows = torch.Tensor(list(atom_rows.values()))
        self.atom_groups = torch.Tensor(list(atom_groups.values()))
        self.atom_families = torch.Tensor(list(atom_families.values()))

    def __call__(self, z_t):
        X = torch.argmax(z_t.X, dim=-1)  # (bs, n)
        X_atom_rows = self.atom_rows.to(X.device)[X]            # (bs, n)
        X_atom_groups = self.atom_groups.to(X.device)[X]        # (bs, n)
        X_atom_families = self.atom_families.to(X.device)[X]        # (bs, n)

        X_atom_feat = torch.stack([X_atom_rows, X_atom_groups, X_atom_families], dim=2)   # (bs, n, 3)

        return X_atom_feat
