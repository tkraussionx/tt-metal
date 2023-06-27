import numpy as np

# ------ size defines ------
ACT_Y = 6
ACT_X = 6

kernel_size = 3

MATRIX_2D_Y = (ACT_Y - 2) * (ACT_X - 2)
MATRIX_2D_X = 1 * kernel_size * kernel_size
print(" 2d matrix size = ", MATRIX_2D_Y, MATRIX_2D_X)

BLOCK_SIZE_Y = 4
BLOCK_SIZE_X = 3



# --- create the arrays ---
activation = np.zeros((ACT_Y, ACT_X))
matrix_2d =  np.zeros((MATRIX_2D_Y, MATRIX_2D_X))

# --- init activation values ---
num = 1
for y in range(ACT_Y):
    for x in range(ACT_X):
        activation[y][x] = num
        num += 1

print("\n\nActivation")
print(activation)

matrix_idx_y = 0
matrix_idx_x = 0


# --- calc the number of blocks to be processed
NUM_BLOCKS_Y = int(MATRIX_2D_Y / BLOCK_SIZE_Y)
NUM_BLOCKS_X = int(MATRIX_2D_X / BLOCK_SIZE_X)

print("Number of blocks: ", NUM_BLOCKS_Y, NUM_BLOCKS_X)




# 2d slide the kernel index
for act_idx_y in range(1, ACT_Y-1):
    for act_idx_x in range(1, ACT_X-1):

        #print(act_idx_y, act_idx_x)
        matrix_idx_x = 0
        # 3x3 kernel window slide
        for kernel_idx_y in range(-1, 2):
            for kernel_idx_x in range(-1, 2):
                act_aggregate_idx_y = act_idx_y+kernel_idx_y
                act_aggregate_idx_x = act_idx_x+kernel_idx_x
                # print(" -- ", matrix_idx_y, matrix_idx_x, " <== ", act_aggregate_idx_y, act_aggregate_idx_x)
                matrix_2d[matrix_idx_y][matrix_idx_x] = activation[act_aggregate_idx_y][act_aggregate_idx_x]
                matrix_idx_x += 1
        matrix_idx_y += 1



'''
# 2d slide the kernel index - COPY OF ORIGINAL LOOP THAT WORKS
for act_idx_y in range(1, ACT_Y-1):
    for act_idx_x in range(1, ACT_X-1):

        #print(act_idx_y, act_idx_x)
        matrix_idx_x = 0
        # 3x3 kernel window slide
        for kernel_idx_y in range(-1, 2):
            for kernel_idx_x in range(-1, 2):
                act_aggregate_idx_y = act_idx_y+kernel_idx_y
                act_aggregate_idx_x = act_idx_x+kernel_idx_x
                # print(" -- ", matrix_idx_y, matrix_idx_x, " <== ", act_aggregate_idx_y, act_aggregate_idx_x)
                matrix_2d[matrix_idx_y][matrix_idx_x] = activation[act_aggregate_idx_y][act_aggregate_idx_x]
                matrix_idx_x += 1
        matrix_idx_y += 1
'''





print("\n\nComposed 2D Matrix:")
print(matrix_2d)
