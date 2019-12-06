
# def _random_acyclic_orientation(B_und):
#     return np.tril(_random_permutation(B_und), k=-1)
function random_acyclic_orientation(M)
    M |> random_permutation |> m -> tril(m, -1)
end

using LinearAlgebra
using Random

# def _random_permutation(M):
#     # np.random.permutation permutes first axis only
#     P = np.random.permutation(np.eye(M.shape[0]))
#     return P.T @ M @ P
function random_permutation(M)
    eye = 1 * Matrix(I, size(M)...)
    # P = Random.randperm(eye)
    P = eye[shuffle(1:end), :]
    transpose(P) * M * P
end

function test()
    randn(3,4) * randn(4,3)

    shuffle([1 2 3;4 5 6])
    [1 2 3;4 5 6][shuffle(1:end), :]
    shuffle([1,2,3; 4,5,6])
    randperm(3)

    random_permutation(randn(3,3))
    random_acyclic_orientation(randn(3,3))
end


function ensure_dag(g)
    # get adj matrix
    m = Matrix(adjacency_matrix(g))
    # FIXME this will remove many edges
    m = random_acyclic_orientation(m)
    m = random_permutation(m)
    # restore adj matrix
    DiGraph(m)
end
