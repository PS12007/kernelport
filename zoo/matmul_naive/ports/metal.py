"""Metal (MLX) port of matmul: C[M, N] = A[M, K] @ B[K, N]; one thread per output, float accumulator."""

SOURCE = """
    uint M_ = uint(M[0]), N_ = uint(N[0]), K_ = uint(K[0]);
    uint col = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (row < M_ && col < N_) {
        float acc = 0.0f;
        for (uint k = 0; k < K_; ++k) acc += static_cast<float>(A[row * K_ + k]) * static_cast<float>(B[k * N_ + col]);
        C[row * N_ + col] = static_cast<T>(acc);
    }
"""
THREADGROUP = (16, 16, 1)


def grid(M, N):
    return (N, M, 1)
