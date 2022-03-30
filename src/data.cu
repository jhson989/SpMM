#include "../include/data.cuh"

DTYPE get_random_number() {return std::rand()%11-5;}

inline DTYPE get_nonzero() {
    if (std::rand()%2 == 0)
        return std::rand()%4+1;
    else
        return -1*(std::rand()%4+1);
}

void make_sparse_matrix(std::vector<DTYPE>& A) {

    int total_nonzero = SPARSITY * M * K;

    std::vector<int> idx(M*K);
    for (int i=0; i<M*K; i++) idx[i] = i;

    // Select matrix indices for nonzero value
    for (int j=0; j<20; j++)
        for (int i=0; i<total_nonzero; i++) {
            int random_idx = std::rand()%idx.size();
            DTYPE temp = idx[i];
            idx[i] = idx[random_idx];
            idx[random_idx] = temp;
        }

    // Fill nonzero value into selected indices
    for (int i=0; i<total_nonzero; i++) {
        A[idx[i]] = get_nonzero();
    }

}

