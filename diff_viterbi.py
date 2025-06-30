import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothedViterbi(nn.Module):
    def __init__(self, regularizer='negentropy', gamma=1.0):
        super(SmoothedViterbi, self).__init__()
        self.regularizer = regularizer
        self.gamma = gamma

    def max_omega(self, x):
        """
        Apply the smoothed max operator using the specified regularizer (negentropy or L2).
        """
        if self.regularizer == 'negentropy':
            # Negative entropy regularizer (log-sum-exp)
            return self.gamma * torch.log(torch.sum(torch.exp(x / self.gamma), dim=-1))
        elif self.regularizer == 'l2':
            # L2 regularizer
            return torch.max(x) - (self.gamma / 2) * torch.norm(x, p=2, dim=-1)
        else:
            raise ValueError("Unsupported regularizer type")

    def forward(self, potentials):
        """
        Perform the smoothed Viterbi algorithm to compute the best path.
        potentials: Tensor of shape (T, S, S) where T is the sequence length and S is the number of states.
        """
        T, S, _ = potentials.shape
        v = torch.zeros(T, S).to(potentials.device)  # Viterbi scores
        backpointers = torch.zeros(T, S, dtype=torch.long).to(potentials.device)  # Backpointers for the path

        # Initialization for t = 0
        v[0] = potentials[0, 0]

        # Forward pass: Dynamic programming for smoothed Viterbi
        for t in range(1, T):
            for s in range(S):
                scores = v[t-1] + potentials[t, s]
                v[t, s] = self.max_omega(scores)  # Use smoothed max for score
                backpointers[t, s] = torch.argmax(scores)  # Backpointer for path reconstruction

        # Backtrack to find the best path
        best_path = torch.zeros(T, dtype=torch.long).to(potentials.device)
        best_path[T-1] = torch.argmax(v[T-1])  # Start with the final state

        for t in range(T-2, -1, -1):
            best_path[t] = backpointers[t+1, best_path[t+1]]

        return best_path, v  # Return the best path and Viterbi scores


# Example of usage
if __name__ == "__main__":
    T = 5  # Sequence length
    S = 3  # Number of states
    potentials = torch.randn(T, S, S)  # Example random potential matrix

    # Instantiate the smoothed Viterbi algorithm with negentropy regularization
    smoothed_viterbi = SmoothedViterbi(regularizer='negentropy', gamma=1.0)
    
    best_path, v = smoothed_viterbi(potentials)
    
    print("Best Path:", best_path)
    print("Viterbi Scores:", v)
