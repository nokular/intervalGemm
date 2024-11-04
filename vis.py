import numpy as np
import matplotlib.pyplot as plt
def tiledGemm(a, b, c):
    M = len(a)
    N = len(b[0])
    K = len(b)
    THREAD_N = 2
    THREAD_M = 2
    WARPSIZE = 32
    # since each warp has 32 threads (organized in 4x8) and 4 quadrants weach warptile N is 2*4*TN and warptile M is 2*8*TM
    WARP_TILE_N = 2*4*THREAD_N
    WARP_TILE_M = 2*8*THREAD_M
    # now we define the blocksize (the amount of threads in a block)
    BLOCK_SIZE = 16*16
    BLOCKDIM = (16, 16)
    # from this we can calculate the block tile M and N (we assume 4 warps in the N dim and 2 in the M dim)
    BLOCK_TILE_N = 4*WARP_TILE_N
    BLOCK_TILE_M = 2*WARP_TILE_M # they should the same size as the warptiles
    BLOCK_TILE_K = 8 #currently works onlz with 16
    # now we can calculate the amount of blocks needed (the gridsize)
    GRID_SIZE_N = (N + BLOCK_TILE_N - 1) // BLOCK_TILE_N
    GRID_SIZE_M = (M + BLOCK_TILE_M - 1) // BLOCK_TILE_M
    GRIDSIZE = GRID_SIZE_N * GRID_SIZE_M
    GRIDDIM = (GRID_SIZE_N, GRID_SIZE_M)
    print("BLOCK_TILE_N: ", BLOCK_TILE_N)
    print("BLOCK_TILE_M: ", BLOCK_TILE_M)
    for grid_n in range(GRID_SIZE_N):
        for grid_m in range(GRID_SIZE_M):
            a_shared_memory = np.zeros((BLOCK_TILE_K, BLOCK_TILE_M), dtype=int)
            b_shared_memory = np.zeros((BLOCK_TILE_N, BLOCK_TILE_K), dtype=int)
            
            for dot_idx in range(0,K,BLOCK_TILE_K):
                # now we simulate the threads for the shared memorz loading
                for blockm in range(BLOCKDIM[0]):
                    for block_n in range(BLOCKDIM[1]):
                        flattened_idx = blockm * BLOCKDIM[1] + block_n
                        warp_idx = flattened_idx // WARPSIZE
                        position_in_warp = flattened_idx % WARPSIZE
                        # each thread has to load TM elements from A into the a_shared_memory
                        row_idx_for_a = grid_m * BLOCK_TILE_M + blockm * THREAD_M
                        col_idx_for_a = dot_idx
                        row_offet_for_this_thread = position_in_warp // 8
                        
                
    
    
    
        

# Sample input matrices
import numpy as np

class Plotter:
    def __init__(self) -> None:
        self.ndim = 8
        self.mdim = 8
        self.visited_counts_a = [[0 for i in range(self.mdim)] for j in range(self.ndim)]
        self.visited_counts_b = [[0 for i in range(self.mdim)] for j in range(self.ndim)]
        self.visited_counts_c = [[0 for i in range(self.mdim)] for j in range(self.ndim)]
    def add_visits(self, ndima, mdima, ndimb, mdimb, ndimc, mdimc):
        self.visited_counts_a[ndima][mdima] += 1
        self.visited_counts_b[ndimb][mdimb] += 1
        self.visited_counts_c[ndimc][mdimc] += 1
    def draw(self):
        fig, axs = plt.subplots(1, 3)
        fig.suptitle('Matrix visualization')
        
        # Plot visited_counts_a
        matrix_a = np.array(self.visited_counts_a)
        axs[0].imshow(matrix_a, cmap='viridis', interpolation='nearest')
        axs[0].set_title('Visited Counts A')
        
        # Plot visited_counts_b
        matrix_b = np.array(self.visited_counts_b)
        axs[1].imshow(matrix_b, cmap='viridis', interpolation='nearest')
        axs[1].set_title('Visited Counts B')
        
        # Plot visited_counts_c
        matrix_c = np.array(self.visited_counts_c)
        axs[2].imshow(matrix_c, cmap='viridis', interpolation='nearest')
        axs[2].set_title('Visited Counts C')
        
        plt.tight_layout()
        plt.show()

def draw_3_matrices_with_active_cell(ndim,mdim,nactive1,mactive1,nactive2,mactive2,nactive3,mactive3):
    # use subplots to plot the matrices
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Matrix visualization')
    # create a matrix of zeros
    matrix = np.zeros((ndim, mdim))
    # set the active cell to 1
    matrix[nactive1][mactive1] = 1
    # plot the matrix
    axs[0].imshow(matrix, cmap='hot', interpolation='nearest')
    # create a matrix of zeros
    matrix = np.zeros((ndim, mdim))
    # set the active cell to 1
    matrix[nactive2][mactive2] = 1
    # plot the matrix
    axs[1].imshow(matrix, cmap='hot', interpolation='nearest')
    # create a matrix of zeros
    matrix = np.zeros((ndim, mdim))
    # set the active cell to 1
    matrix[nactive3][mactive3] = 1
    # plot the matrix
    axs[2].imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()



# Create matrix 'a'
a = np.array([[1, 2, 3, 4, 5, 6, 7, 8], 
              [9, 10, 11, 12, 13, 14, 15, 16], 
              [17, 18, 19, 20, 21, 22, 23, 24], 
              [25, 26, 27, 28, 29, 30, 31, 32],
              [33, 34, 35, 36, 37, 38, 39, 40],
              [41, 42, 43, 44, 45, 46, 47, 48],
              [49, 50, 51, 52, 53, 54, 55, 56],
              [57, 58, 59, 60, 61, 62, 63, 64]])

# Create matrix 'b' (same as 'a' for this example)
b = np.array([[1, 2, 3, 4, 5, 6, 7, 8], 
              [9, 10, 11, 12, 13, 14, 15, 16], 
              [17, 18, 19, 20, 21, 22, 23, 24], 
              [25, 26, 27, 28, 29, 30, 31, 32],
              [33, 34, 35, 36, 37, 38, 39, 40],
              [41, 42, 43, 44, 45, 46, 47, 48],
              [49, 50, 51, 52, 53, 54, 55, 56],
              [57, 58, 59, 60, 61, 62, 63, 64]])

# Create matrix 'c' filled with zeros
c = np.zeros((8, 8), dtype=int)

# Print the matrices to verify
#print("Matrix a:")
#print(a)
#print("\nMatrix b:")
#print(b)
#print("\nMatrix c:")
#print(c)


# Perform tiled GEMM
c = tiledGemm(a, b, c)
c_control = np.dot(a, b)

# check if the result is correct
assert np.all(c == c_control)
print("passed")
exit(0)
# Print the resulting matrix
print("Resulting matrix C:")
for row in c:
    print(row)

# Print the control matrix
print("\nControl matrix C:")
for row in c_control:
    print(row)
