from transformers import TrainerCallback

class LossTrackerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append((state.epoch, logs['loss']))
            if 'eval_loss' in logs:
                self.eval_losses.append((state.epoch, logs['eval_loss']))


import matplotlib.pyplot as plt

def plot_losses(train_losses, eval_losses):
    train_epochs, train_vals = zip(*train_losses)
    eval_epochs, eval_vals = zip(*eval_losses)

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_vals, label='Training Loss')
    plt.plot(eval_epochs, eval_vals, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()
