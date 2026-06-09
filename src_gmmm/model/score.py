import torch
import torch.nn as nn
from typing import Optional

from ..nn.encoder import EquivEncoder
from ..nn.readout import Readout


class EquivariantParameterization(nn.Module):
    def __init__(
        self,
        encoder: EquivEncoder,
        readout: Readout,
    ):
        super(EquivariantParameterization, self).__init__()
        self.encoder = encoder
        self.readout = readout

    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
        prev_preds = None, #self-conditioning: previous predictions
        z_prev = None, #latent recursion: previous latent states
        c: Optional[torch.Tensor] = None,
    ):
        states = self.encoder.forward(
            t=t,
            h=h,
            pos=pos,
            node_index=node_index,
            edge_node_index=edge_node_index,
            prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
            z_prev=z_prev, #latent recursion: pass previous latent states to the model
            c=c,
        )

        preds = self.readout.forward(
            t,
            states,
            h=h,
            pos=pos,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )

        if "z" in states: #equivalent to checking if latent_recursion is True
           preds["z"] = states["z"]
        
        return preds
