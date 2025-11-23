"""
Mixin para gestión centralizada de dispositivos en modelos PyTorch.
Proporciona funcionalidad para manejar el dispositivo (CPU/GPU) de manera consistente.
"""

from typing import Optional
import torch
from torch.nn import Module


class DeviceMixin:
    """
    Mixin que proporciona gestión de dispositivos para modelos PyTorch.
    
    Attributes
    ----------
    _device : torch.device
        Dispositivo donde reside el modelo (CPU/GPU).
    """
    
    def __init__(self, *args, **kwargs):
        """Inicializa el mixin y llama al siguiente clase en MRO."""
        super().__init__(*args, **kwargs)
        self._device: Optional[torch.device] = None
    
    @property
    def device(self) -> torch.device:
        """
        Obtiene el dispositivo del modelo.
        
        Returns
        -------
        torch.device
            Dispositivo donde están los parámetros del modelo.
            
        Raises
        ------
        RuntimeError
            Si el modelo no tiene parámetros o no se puede determinar el dispositivo.
        """
        if self._device is not None:
            return self._device
            
        # Intentar determinar el dispositivo a partir de los parámetros
        try:
            # Para modelos nn.Module, obtener dispositivo del primer parámetro
            if isinstance(self, Module) and list(self.parameters()):
                first_param = next(self.parameters())
                self._device = first_param.device
                return self._device
        except (StopIteration, AttributeError):
            pass
            
        # Fallback a CPU si no se puede determinar
        self._device = torch.device('cpu')
        return self._device
    
    def to(self, device: torch.device) -> 'DeviceMixin':
        """
        Mueve el modelo al dispositivo especificado.
        
        Parameters
        ----------
        device : torch.device
            Dispositivo destino.
            
        Returns
        -------
        DeviceMixin
            Self para method chaining.
        """
        if isinstance(self, Module):
            super().to(device)  # type: ignore
        self._device = device
        return self
    
    def cpu(self) -> 'DeviceMixin':
        """
        Mueve el modelo a CPU.
        
        Returns
        -------
        DeviceMixin
            Self para method chaining.
        """
        return self.to(torch.device('cpu'))
    
    def cuda(self, device: Optional[int] = None) -> 'DeviceMixin':
        """
        Mueve el modelo a GPU.
        
        Parameters
        ----------
        device : int, optional
            Índice de la GPU específica.
            
        Returns
        -------
        DeviceMixin
            Self para method chaining.
        """
        if device is not None:
            return self.to(torch.device(f'cuda:{device}'))
        return self.to(torch.device('cuda'))