import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from typing import Tuple, Union

class BDEInteractionLayer(MessagePassing):
    """
    Message Passing Neural Network layer that updates both bond and atom states.
    Based on the logic from the original Keras implementation.
    """
    def __init__(self, atom_features: int):
        super().__init__(aggr='sum')
        self.atom_features = atom_features

        # Pre-activation Batch Normalization
        self.atom_batch_norm = nn.BatchNorm1d(atom_features)
        self.bond_batch_norm = nn.BatchNorm1d(atom_features)

        # Edge update MLP
        # Input: [source_atom, target_atom, bond_state] -> 3 * atom_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * atom_features, 2 * atom_features),
            nn.ReLU(),
            nn.Linear(2 * atom_features, atom_features)
        )

        # MLP for transforming source atom features before message creation
        self.message_source_transform = nn.Linear(atom_features, atom_features)

        # Node update MLP (state transition function)
        self.node_mlp = nn.Sequential(
            nn.Linear(atom_features, atom_features),
            nn.ReLU(),
            nn.Linear(atom_features, atom_features)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the interaction layer.

        Args:
            x (torch.Tensor): Atom feature tensor of shape [num_atoms, atom_features].
            edge_index (torch.Tensor): Edge index tensor of shape [2, num_edges].
            edge_attr (torch.Tensor): Bond feature tensor of shape [num_edges, atom_features].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated atom features
                                               and updated bond features.
        """
        # Store original states for residual connections
        original_x = x # [num_atoms, atom_features]
        original_edge_attr = edge_attr # [num_edges, atom_features]

        # 1. Pre-activation Batch Normalization
        x = self.atom_batch_norm(x) # [num_atoms, atom_features]
        edge_attr = self.bond_batch_norm(edge_attr) # [num_edges, atom_features]

        # 2. Edge Update
        row, col = edge_index
        source_nodes, target_nodes = x[row], x[col] # [num_edges, atom_features]
        
        edge_mlp_input = torch.cat([source_nodes, target_nodes, edge_attr], dim=-1) # [num_edges, 3 * atom_features]
        new_edge_attr = self.edge_mlp(edge_mlp_input) # [num_edges, atom_features]
        
        # Residual connection for bond state
        edge_attr = original_edge_attr + new_edge_attr # [num_edges, atom_features]

        # 3. Node Update (Message Passing)
        # The propagate call will trigger message(), aggregate(), and update()
        # We pass the updated edge_attr to be used in message creation
        propagated_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr) # [num_atoms, atom_features]

        # Residual connection for atom state
        x = original_x + propagated_messages # [num_atoms, atom_features]
        
        return x, edge_attr

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Message creation. Corresponds to `messages = Multiply()([source_atom, bond_state])`.
        `x_j` are the source node features for each edge.

        Args:
            x_j (torch.Tensor): Source atom features tensor of shape [num_edges, atom_features].
            edge_attr (torch.Tensor): Updated bond feature tensor of shape [num_edges, atom_features].

        Returns:
            torch.Tensor: The message tensor for each edge.
        """
        # Transform source atom features
        transformed_x_j = self.message_source_transform(x_j) # [num_edges, atom_features]
        return transformed_x_j * edge_attr # [num_edges, atom_features]

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """
        Node state update. Corresponds to the state transition function.
        `aggr_out` are the aggregated messages for each node.

        Args:
            aggr_out (torch.Tensor): Aggregated messages tensor of shape [num_atoms, atom_features].

        Returns:
            torch.Tensor: The result of the node update MLP.
        """
        return self.node_mlp(aggr_out) # [num_atoms, atom_features]


class BDEModel(nn.Module):
    """
    The complete BDE Prediction Graph Neural Network model.
    """
    def __init__(self, num_atom_classes: int, num_bond_classes: int, atom_features: int = 128, num_messages: int = 6):
        super().__init__()

        # Embeddings for initial atom and bond states
        self.atom_embedding = nn.Embedding(num_atom_classes, atom_features)
        self.bond_embedding = nn.Embedding(num_bond_classes, atom_features)

        # Per-bond-type bias embedding, as per the original implementation
        self.bond_mean_embedding = nn.Embedding(num_bond_classes, 1)

        # Stack of interaction layers
        self.interaction_layers = nn.ModuleList(
            [BDEInteractionLayer(atom_features) for _ in range(num_messages)]
        )

        # Final MLP to predict BDE value from bond state
        self.output_mlp = nn.Linear(atom_features, 1)

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass for the entire model.

        Args:
            data (Union[Data, Batch]): A PyG Data or Batch object containing atom IDs (`x`),
                                       bond IDs (`edge_attr`), and `edge_index`.

        Returns:
            torch.Tensor: A 1D tensor of shape [num_edges] with the predicted BDE for each bond.
        """
        # 1. Initialize atom and bond states from embeddings
        # data.x are the atom IDs, data.edge_attr are the bond IDs
        atom_state = self.atom_embedding(data.x)  # [num_atoms, atom_features]
        bond_state = self.bond_embedding(data.edge_attr) # [num_edges, atom_features]
        
        # Also get the bond_mean bias for later
        bond_mean = self.bond_mean_embedding(data.edge_attr) # [num_edges, 1]

        # 2. Run through message passing layers
        for layer in self.interaction_layers:
            atom_state, bond_state = layer(atom_state, data.edge_index, bond_state) # atom_state: [num_atoms, atom_features], bond_state: [num_edges, atom_features]

        # 3. Predict BDE from final bond state
        bde_pred = self.output_mlp(bond_state) # [num_edges, 1]
        
        # 4. Add the per-bond-type bias
        bde_pred = bde_pred + bond_mean # [num_edges, 1]

        return bde_pred.squeeze(-1) # [num_edges]
