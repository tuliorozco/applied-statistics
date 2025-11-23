import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional 

class TabularExpert(nn.Module):
    """Experto especializado para procesamiento de datos tabulares"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Construir capas ocultas
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Capa de salida
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.gate_value = nn.Parameter(torch.tensor(0.0))  # Valor de gate para este experto
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class SwitchTransformerTabular(nn.Module):
    """MoE basado en SwitchTransformer para datos tabulares"""
    
    def __init__(self, input_dim: int, num_experts: int, experts_per_token: int,
                 expert_hidden_dims: List[int], output_dim: int,
                 capacity_factor: float = 1.0):
        super().__init__()
        
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.capacity_factor = capacity_factor
        
        # Capa de proyección para entrada tabular
        self.input_projection = nn.Linear(input_dim, input_dim)
        
        # Capa de routing (gate)
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        
        # Crear expertos
        self.experts = nn.ModuleList([
            TabularExpert(input_dim, expert_hidden_dims, output_dim)
            for _ in range(num_experts)
        ])
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass con routing dinámico"""
        batch_size, seq_len, hidden_dim = x.shape
        N = batch_size * seq_len
        
        # Proyectar entrada
        projected_x = self.input_projection(x)          # [B, L, H]
        x_flat = projected_x.reshape(-1, hidden_dim)    # [N, H]
        
        # Calcular logits de routing
        router_logits = self.gate(x_flat)               # [N, num_experts]
        
        # Routing con top-k expertos
        routing_weights = F.softmax(router_logits, dim=-1)  # [N, num_experts]
        expert_weights, expert_indices = torch.topk(
            routing_weights, self.experts_per_token, dim=-1
        )                                               # expert_weights: [N, k]
                                                        # expert_indices: [N, k]
        # Normalizar pesos dentro del top-k
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Calcular capacidad por experto
        expert_capacity = int(
            self.capacity_factor * N / self.num_experts
        )
        expert_capacity = max(expert_capacity, 1)
        
        # Distribuir tokens a expertos
        outputs = torch.zeros_like(x_flat)              # [N, H]
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        
        for expert_idx, expert in enumerate(self.experts):
            # Máscara (token, posición_en_topk) donde este experto fue seleccionado
            mask = (expert_indices == expert_idx)       # [N, k]
            token_positions, k_positions = mask.nonzero(as_tuple=True)  # [M], [M]
            
            num_tokens = token_positions.numel()
            if num_tokens == 0:
                continue
            
            # Limitar por capacidad: seleccionar tokens con mayor peso PARA ESTE experto
            if num_tokens > expert_capacity:
                # pesos específicos de este experto
                scores = expert_weights[token_positions, k_positions]   # [M]
                top_scores, top_idx = torch.topk(scores, expert_capacity)
                
                token_positions = token_positions[top_idx]
                k_positions = k_positions[top_idx]
                num_tokens = expert_capacity
            
            if num_tokens == 0:
                continue
            
            # Entradas a este experto
            expert_input = x_flat[token_positions]      # [M, H]
            expert_output = expert(expert_input)        # [M, output_dim]
            
            # Peso correspondiente a ESTE experto por token (no todo el vector de k)
            weights = expert_weights[token_positions, k_positions]  # [M]
            
            # Combinar salidas ponderadas
            weighted_output = expert_output * weights.unsqueeze(-1)  # [M, output_dim]
            
            # Acumular en la salida global
            outputs[token_positions] = weighted_output
            expert_usage[expert_idx] = num_tokens
        
        # Reformatear salida
        final_output = outputs.reshape(batch_size, seq_len, -1)
        final_output = self.layer_norm(final_output)
        
        # Métricas de routing
        routing_metrics = {
            'expert_usage': expert_usage,
            'router_logits': router_logits,
            'routing_weights': routing_weights
        }
        
        return final_output, routing_metrics

