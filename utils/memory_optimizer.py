import torch
import gc
import psutil
import numpy as np
from typing import Any, Optional, List
import logging
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from threading import Lock
import time

@dataclass
class MemoryStats:
    """Class for tracking memory usage statistics"""
    cpu_percent: float
    ram_usage: float
    gpu_memory_used: Optional[float]
    gpu_utilization: Optional[float]

class MemoryOptimizer:
    def __init__(self, threshold: float = 0.85):
        """
        Initialize memory optimizer
        Args:
            threshold: Memory usage threshold (0-1) to trigger optimization
        """
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self._monitor_lock = Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval: float = 1.0):
        """
        Start background memory monitoring
        Args:
            interval: Monitoring interval in seconds
        """
        with self._monitor_lock:
            if not self._monitoring:
                self._monitoring = True
                self._monitor_thread = threading.Thread(
                    target=self._monitor_memory,
                    args=(interval,),
                    daemon=True
                )
                self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop background memory monitoring"""
        with self._monitor_lock:
            self._monitoring = False
            if self._monitor_thread is not None:
                self._monitor_thread.join()
                self._monitor_thread = None

    def _monitor_memory(self, interval: float):
        """Background memory monitoring task"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                if self._should_optimize(stats):
                    self.optimize()
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {str(e)}")
            time.sleep(interval)

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory usage statistics
        Returns:
            MemoryStats object containing usage information
        """
        # Get CPU stats
        cpu_percent = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent / 100.0

        # Get GPU stats if available
        gpu_memory_used = None
        gpu_utilization = None
        
        if self.device.type == "cuda":
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                gpu_utilization = torch.cuda.utilization()
            except Exception:
                pass

        return MemoryStats(
            cpu_percent=cpu_percent,
            ram_usage=ram_usage,
            gpu_memory_used=gpu_memory_used,
            gpu_utilization=gpu_utilization
        )

    def _should_optimize(self, stats: MemoryStats) -> bool:
        """
        Determine if memory optimization is needed
        Args:
            stats: Current memory statistics
        Returns:
            True if optimization is needed
        """
        if stats.ram_usage > self.threshold:
            return True
            
        if (stats.gpu_memory_used is not None and 
            stats.gpu_memory_used > self.threshold):
            return True
            
        return False

    def optimize(self):
        """Perform memory optimization"""
        # Clear CUDA cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()

        # Clear unused tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda and not obj.is_required:
                        del obj
            except Exception:
                pass

    @contextmanager
    def optimize_context(self):
        """Context manager for automatic memory optimization"""
        try:
            yield
        finally:
            self.optimize()

    def pin_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pin tensor memory for faster GPU transfer
        Args:
            tensor: Input tensor
        Returns:
            Pinned tensor
        """
        if not tensor.is_pinned() and self.device.type == "cuda":
            return tensor.pin_memory()
        return tensor

    def to_device(self, data: Any, non_blocking: bool = True) -> Any:
        """
        Move data to appropriate device
        Args:
            data: Input data (tensor, list, dict, etc.)
            non_blocking: Whether to perform non-blocking transfer
        Returns:
            Data on target device
        """
        if torch.is_tensor(data):
            return data.to(self.device, non_blocking=non_blocking)
        elif isinstance(data, (list, tuple)):
            return [self.to_device(x, non_blocking) for x in data]
        elif isinstance(data, dict):
            return {k: self.to_device(v, non_blocking) for k, v in data.items()}
        return data

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision"""
        if self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

    def clear_memory(self, tensors: List[torch.Tensor]):
        """
        Explicitly clear tensor memory
        Args:
            tensors: List of tensors to clear
        """
        for tensor in tensors:
            if tensor is not None and torch.is_tensor(tensor):
                tensor.detach_()
                del tensor
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def optimize_batch_size(self, 
                          initial_batch_size: int,
                          input_shape: tuple,
                          target_memory_usage: float = 0.8) -> int:
        """
        Dynamically optimize batch size based on memory constraints
        Args:
            initial_batch_size: Starting batch size
            input_shape: Shape of input tensors
            target_memory_usage: Target memory usage ratio
        Returns:
            Optimized batch size
        """
        if self.device.type != "cuda":
            return initial_batch_size

        try:
            # Create test tensor
            test_tensor = torch.zeros((1, *input_shape), device=self.device)
            memory_per_sample = test_tensor.element_size() * test_tensor.nelement()
            
            # Get available memory
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory = total_memory * target_memory_usage
            
            # Calculate optimal batch size
            optimal_batch_size = int(available_memory / memory_per_sample)
            
            # Clean up test tensor
            del test_tensor
            torch.cuda.empty_cache()
            
            return min(optimal_batch_size, initial_batch_size)
            
        except Exception as e:
            self.logger.warning(f"Error optimizing batch size: {str(e)}")
            return initial_batch_size