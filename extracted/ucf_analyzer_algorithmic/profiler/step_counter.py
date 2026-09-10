class StepCounter:
    def __init__(self):
        self.steps = 0

    def increment(self, count=1):
        self.steps += count

    def reset(self):
        self.steps = 0

    def get_steps(self):
        return self.steps
