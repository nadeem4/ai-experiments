"""Values shared by every part (spec and plan, Global Constraints)."""
K = 4096
NPROBES = (1, 4, 16, 64)
B_NPROBES = (16, 64)
BUDGETS_S = (10, 30, 60, 120)
EPOCHS = (1, 2, 3, 5)
LLOYD_AT = (1, 2, 3, 5, 10, 25)
DIMS = (64, 128, 256, 512, 768)
ARMS = ("lloyd", "emeans", "minibatch")
