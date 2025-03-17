

class Machine:
    def __init__(self, machine_id: int, idle_energy: float):
        self.machine_id = machine_id
        self.idle_energy = idle_energy
        self.end_time = 0  # when the machine is next available, for instance

    def __repr__(self):
        return f"Machine(id={self.machine_id}, idle_energy={self.idle_energy})"