from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)
    
class LSTMMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(LSTMMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
    # Note: LSTMCell requires both hidden (h) and cell (c) states
    # For TGN, we use GRU which only needs hidden state. LSTM support is experimental.
    self.memory_updater = nn.LSTMCell(input_size=message_dimension,
                                      hidden_size=memory_dimension)
    self.cell_state = None  # Will be initialized in update_memory
  
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return
    
    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
      "Trying to update memory to time in the past"
    
    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps
    
    # Initialize cell state if needed
    if self.cell_state is None or self.cell_state.shape[0] != self.memory.n_nodes:
      self.cell_state = torch.zeros(self.memory.n_nodes, self.memory.memory_dimension).to(self.device)
    
    cell = self.cell_state[unique_node_ids]
    updated_memory, updated_cell = self.memory_updater(unique_messages, (memory, cell))
    
    self.memory.set_memory(unique_node_ids, updated_memory)
    self.cell_state[unique_node_ids] = updated_cell

class TRANSFRORMERMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(TRANSFRORMERMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.TransformerEncoderLayer(d_model=message_dimension, nhead=4)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "lstm":
    return LSTMMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "transformer":
    return TRANSFRORMERMemoryUpdater(memory, message_dimension, memory_dimension, device)
