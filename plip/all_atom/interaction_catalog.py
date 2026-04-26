"""
Interaction Catalog Module - Simplified Storage

Simple storage for all detected interactions.
Provides filtering and classification methods.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .interaction_detector import Interaction


class InteractionCatalog:
    """
    Simple catalog for storing and querying interactions.
    """
    
    def __init__(self):
        self.interactions: Dict[str, List[Interaction]] = {
            'hydrophobic': [],
            'hbond': [],
            'saltbridge': [],
            'pistacking': [],
            'pication': [],
            'halogen': [],
            'metal': [],
            'water_bridge': [],
        }
    
    def add_interactions(self, interaction_type: str, interactions: List[Interaction]):
        """Add interactions of a specific type"""
        if interaction_type in self.interactions:
            self.interactions[interaction_type].extend(interactions)
    
    def get_by_type(self, interaction_type: str) -> List[Interaction]:
        """Get all interactions of a specific type"""
        return self.interactions.get(interaction_type, [])
    
    def get_all(self) -> Dict[str, List[Interaction]]:
        """Get all interactions"""
        return self.interactions
    
    def get_for_residue(self, resname: str, chain: str, resnum: int) -> List[Tuple[str, Interaction]]:
        """Get all interactions involving a specific residue"""
        result = []
        for itype, interactions in self.interactions.items():
            for inter in interactions:
                if ((inter.res_a_name == resname and inter.res_a_chain == chain and inter.res_a_num == resnum) or
                    (inter.res_b_name == resname and inter.res_b_chain == chain and inter.res_b_num == resnum)):
                    result.append((itype, inter))
        return result
    
    def get_between_components(self, comp_a: str, comp_b: str, 
                               residue_types: Dict[str, str]) -> List[Tuple[str, Interaction]]:
        """
        Get interactions between two component types.
        
        Parameters
        ----------
        comp_a, comp_b : str
            Component types ('protein', 'ligand', 'water', 'ion', etc.)
        residue_types : Dict[str, str]
            Mapping from residue ID to component type
        """
        result = []
        for itype, interactions in self.interactions.items():
            for inter in interactions:
                res_a_id = f"{inter.res_a_name}:{inter.res_a_chain}:{inter.res_a_num}"
                res_b_id = f"{inter.res_b_name}:{inter.res_b_chain}:{inter.res_b_num}"
                
                type_a = residue_types.get(res_a_id, 'unknown')
                type_b = residue_types.get(res_b_id, 'unknown')
                
                if ((type_a == comp_a and type_b == comp_b) or
                    (type_a == comp_b and type_b == comp_a)):
                    result.append((itype, inter))
        
        return result
    
    def count_by_type(self) -> Dict[str, int]:
        """Get count of interactions by type"""
        return {itype: len(inters) for itype, inters in self.interactions.items() if inters}
    
    def get_summary(self) -> str:
        """Get text summary of interactions"""
        lines = ["Interaction Summary:"]
        for itype, interactions in self.interactions.items():
            if interactions:
                lines.append(f"  {itype}: {len(interactions)}")
        return "\n".join(lines)
