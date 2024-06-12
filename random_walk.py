'''import scanpy as sc
import numpy as np
from multiprocessing import shared_memory, Process, Lock
from multiprocessing import current_process
import psutil
from tqdm import tqdm
from scipy.stats import wasserstein_distance as wd

# lock = Lock()
cpu_count = psutil.cpu_count(logical=False)
if cpu_count > 3:
    cpu_count -= 2


def task_allocation(row_num, cpu_count):
    rand_idx = np.array(list(range(row_num)))
    np.random.shuffle(rand_idx)

    per_cpu = int(row_num / cpu_count)
    row_dict = {}
    for i in range(cpu_count - 1):
        idx_list = list(range(i * per_cpu, i * per_cpu + per_cpu))
        row_dict[i] = [rand_idx[idx] for idx in idx_list]

    idx_list = list(range(idx_list[-1] + 1, row_num - 1))
    row_dict[cpu_count - 1] = [rand_idx[idx] for idx in idx_list]

    return row_dict


def create_shared_arr(a):
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)

    # # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(a.shape, dtype=np.float64, buffer=shm.buf)

    # Copy the original data into shared memory
    np_array[:] = a[:]
    return shm, np_array


def random_walks(shm_mat, shape_mat, shm_walks, shape_walks, row_list):
    shm_mat_memory = shared_memory.SharedMemory(name=shm_mat)
    shm_walks_memory = shared_memory.SharedMemory(name=shm_walks)
    mat = np.ndarray(shape_mat, dtype=np.float64, buffer=shm_mat_memory.buf)
    walks = np.ndarray(shape_walks, dtype=np.float64, buffer=shm_walks_memory.buf)

    for i in tqdm(row_list):
        next_idx = i
        walks[i, 0] = i
        for j in range(1, walks.shape[1]):
            draw = np.random.rand()
            next_idx = np.searchsorted(mat[next_idx], draw)
            if next_idx == mat.shape[1]:
                next_idx = mat.shape[1] - 1
            walks[i, j] = next_idx

def main():
    if current_process().name == "MainProcess":
        dataset = 'EB_raw'
        mat_type = 'rcasim'

        mat = np.load('./data/{}_{}_for_walk.npy'.format(dataset, mat_type))
        mat = mat.astype('float64')

        walks = np.zeros((mat.shape[0], mat.shape[0]), dtype=np.float64)

        shape_mat = mat.shape
        shape_walks = walks.shape

        shm_mat, mat = create_shared_arr(mat)
        shm_walks, walks = create_shared_arr(walks)

        print('mat', mat)
        print(shm_mat)

        print('walks', walks)
        print(shm_walks)

        row_dict = task_allocation(walks.shape[0], cpu_count)

        process_list = []
        for i in range(cpu_count):
            process = Process(target=random_walks, args=(shm_mat.name, shape_mat, shm_walks.name, shape_walks, row_dict[i], ))
            process_list.append(process)

        for process in process_list:
            process.start()

        for process in process_list:
            process.join()

        # print('mat', mat)
        # print('walks', walks)

        # np.save('./data/{}_{}_walk_idx.npy'.format(dataset, mat_type), walks)

        shm_mat.close()
        shm_mat.unlink()

        shm_walks.close()
        shm_walks.unlink()

        return walks

walks = main()
print(walks)'''

import numpy as np
from multiprocessing import shared_memory, Process, Lock
from multiprocessing import current_process
from tqdm import tqdm
from scipy.stats import levy


class RandomWalk:
    def __init__(self, adjmat, walk_steps, cpu_count):
        self.float_type = np.float64
        self.int_type = np.int64
        self.cumsum_adjmat = np.cumsum(adjmat, axis=1)
        self.cumsum_adjmat = self.cumsum_adjmat / self.cumsum_adjmat[:, -1].reshape(-1, 1)
        self.cumsum_adjmat = self.cumsum_adjmat.astype(self.float_type)
        self.cpu_count = cpu_count
        self.walk_steps = walk_steps
        self.walk_traj = np.zeros((self.cumsum_adjmat.shape[0], walk_steps), dtype=self.float_type)

    def create_shared_arr(self, a):
        shm = shared_memory.SharedMemory(create=True, size=a.nbytes)

        # Create a NumPy array backed by shared memory
        np_array = np.ndarray(a.shape, dtype=np.float64, buffer=shm.buf)

        # Copy the original data into shared memory
        np_array[:] = a[:]
        return shm, np_array

    def task_allocation(self, row_num, cpu_count):
        rand_idx = np.array(list(range(row_num)))
        np.random.shuffle(rand_idx)

        per_cpu = int(row_num / cpu_count)
        row_dict = {}
        for i in range(cpu_count - 1):
            idx_list = list(range(i * per_cpu, i * per_cpu + per_cpu))
            row_dict[i] = [rand_idx[idx] for idx in idx_list]

        idx_list = list(range(idx_list[-1] + 1, row_num - 1))
        row_dict[cpu_count - 1] = [rand_idx[idx] for idx in idx_list]

        return row_dict

    def random_walks(self, shm_mat, shape_mat, shm_walks, shape_walks, row_list):
        shm_mat_memory = shared_memory.SharedMemory(name=shm_mat)
        shm_walks_memory = shared_memory.SharedMemory(name=shm_walks)
        mat = np.ndarray(shape_mat, dtype=np.float64, buffer=shm_mat_memory.buf)
        walks = np.ndarray(shape_walks, dtype=np.float64, buffer=shm_walks_memory.buf)

        for i in tqdm(row_list):
            next_idx = i
            walks[i, 0] = i
            for j in range(1, walks.shape[1]):
                draw = np.random.rand()
                next_idx = np.searchsorted(mat[next_idx], draw)
                if next_idx == mat.shape[1]:
                    next_idx = mat.shape[1] - 1
                walks[i, j] = next_idx

    def go(self):
        if current_process().name == "MainProcess":
            shape_mat = self.cumsum_adjmat.shape
            shape_walks = self.walk_traj.shape

            shm_mat, mat = self.create_shared_arr(self.cumsum_adjmat)
            shm_walks, walks = self.create_shared_arr(self.walk_traj)

            print('mat', mat)
            print(shm_mat)

            print('walks', walks)
            print(shm_walks)

            row_dict = self.task_allocation(walks.shape[0], self.cpu_count)

            process_list = []
            for i in range(self.cpu_count):
                process = Process(target=self.random_walks,
                                  args=(shm_mat.name, shape_mat, shm_walks.name, shape_walks, row_dict[i],))
                process_list.append(process)

            for process in process_list:
                process.start()

            for process in process_list:
                process.join()

            self.walk_traj = walks.copy().astype(np.int64)

            shm_mat.close()
            shm_mat.unlink()

            shm_walks.close()
            shm_walks.unlink()

            return 0


def random_walk(shmr_name, cumsum_proxim_shape, cumsum_proxim_dtype,
                shmw_name, gene_corpus_shape, gene_corpus_dtype, walk_steps, row_list):
    shmr = shared_memory.SharedMemory(name=shmr_name)
    shmw = shared_memory.SharedMemory(name=shmw_name)
    cumsum_proxim = np.ndarray(cumsum_proxim_shape, dtype=cumsum_proxim_dtype, buffer=shmr.buf)
    gene_corpus = np.ndarray(gene_corpus_shape, dtype=gene_corpus_dtype, buffer=shmw.buf)

    for i in tqdm(row_list):
        next_idx = i
        gene_corpus[i, 0] = i
        for j in range(1, walk_steps):
            draw = np.random.rand()
            next_idx = np.searchsorted(cumsum_proxim[next_idx], draw)
            if next_idx == cumsum_proxim.shape[1]:
                next_idx = cumsum_proxim.shape[1] - 1
            gene_corpus[i, j] = next_idx


def random_walk_lazy_teleport(shmr_name, cumsum_proxim_shape, cumsum_proxim_dtype,
                shmw_name, gene_corpus_shape, gene_corpus_dtype, walk_steps, row_list, x_lazy, x_teleport):
    shmr = shared_memory.SharedMemory(name=shmr_name)
    shmw = shared_memory.SharedMemory(name=shmw_name)
    cumsum_proxim = np.ndarray(cumsum_proxim_shape, dtype=cumsum_proxim_dtype, buffer=shmr.buf)
    gene_corpus = np.ndarray(gene_corpus_shape, dtype=gene_corpus_dtype, buffer=shmw.buf)

    for i in tqdm(row_list):
        next_idx = i
        gene_corpus[i, 0] = i
        for j in range(1, walk_steps):
            if np.random.rand() < x_lazy:
                next_idx = next_idx
            elif np.random.rand() < x_teleport:
                next_idx = np.random.randint(0, cumsum_proxim_shape[0])
            else:
                draw = np.random.rand()
                next_idx = np.searchsorted(cumsum_proxim[next_idx], draw)
                if next_idx == cumsum_proxim.shape[1]:
                    next_idx = cumsum_proxim.shape[1] - 1
            gene_corpus[i, j] = next_idx



def random_walk_levy_flight(shmr_name, cumsum_proxim_shape, cumsum_proxim_dtype,
                shmw_name, gene_corpus_shape, gene_corpus_dtype, walk_steps, row_list):
    shmr = shared_memory.SharedMemory(name=shmr_name)
    shmw = shared_memory.SharedMemory(name=shmw_name)
    cumsum_proxim = np.ndarray(cumsum_proxim_shape, dtype=cumsum_proxim_dtype, buffer=shmr.buf)
    gene_corpus = np.ndarray(gene_corpus_shape, dtype=gene_corpus_dtype, buffer=shmw.buf)

    for i in tqdm(row_list):
        next_idx = i
        gene_corpus[i, 0] = i

        levy_steps = levy.rvs(size=walk_steps+1).astype(np.int32) + 1

        for j in range(1, walk_steps):
            if levy_steps[j] > 1000:
                next_idx = np.random.randint(0, cumsum_proxim.shape[0])
            else:
                for s in range(levy_steps[j]):
                    draw = np.random.rand()
                    next_idx = np.searchsorted(cumsum_proxim[next_idx], draw)
                    if next_idx == cumsum_proxim.shape[1]:
                        next_idx = cumsum_proxim.shape[1] - 1
            gene_corpus[i, j] = next_idx

'''def random_walk(shm_name, shape, dtype, n):
    shm = shared_memory.SharedMemory(name=shm_name)
    np_a = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    for i in tqdm(range(shape[1])):
        np_a[n, i] = n'''

'''import numpy as np
from multiprocessing import shared_memory, Process, Lock
from multiprocessing import current_process
from tqdm import tqdm


class RandomWalkOnGraph:
    def __int__(self, adjmat, walk_steps, cpu_count):
        self.float_type = np.float64
        self.int_type = np.int64
        self.cumsum_adjmat = np.cumsum(adjmat, axis=1)
        self.cumsum_adjmat = self.cumsum_adjmat / self.cumsum_adjmat[:, -1].reshape(-1, 1)
        self.cumsum_adjmat = self.cumsum_adjmat.astype(self.float_type)
        self.cpu_count = cpu_count
        self.walk_steps = walk_steps
        self.walk_traj = np.zeros((self.cumsum_adjmat.shape[0], walk_steps), dtype=self.float_type)

    def create_shared_arr(self, a):
        shm = shared_memory.SharedMemory(create=True, size=a.nbytes)

        # Create a NumPy array backed by shared memory
        np_array = np.ndarray(a.shape, dtype=np.float64, buffer=shm.buf)

        # Copy the original data into shared memory
        np_array[:] = a[:]
        return shm, np_array

    def task_allocation(self, row_num, cpu_count):
        rand_idx = np.array(list(range(row_num)))
        np.random.shuffle(rand_idx)

        per_cpu = int(row_num / cpu_count)
        row_dict = {}
        for i in range(cpu_count - 1):
            idx_list = list(range(i * per_cpu, i * per_cpu + per_cpu))
            row_dict[i] = [rand_idx[idx] for idx in idx_list]

        idx_list = list(range(idx_list[-1] + 1, row_num - 1))
        row_dict[cpu_count - 1] = [rand_idx[idx] for idx in idx_list]

        return row_dict

    def random_walks(self, shm_mat, shape_mat, shm_walks, shape_walks, row_list):
        shm_mat_memory = shared_memory.SharedMemory(name=shm_mat)
        shm_walks_memory = shared_memory.SharedMemory(name=shm_walks)
        mat = np.ndarray(shape_mat, dtype=np.float64, buffer=shm_mat_memory.buf)
        walks = np.ndarray(shape_walks, dtype=np.float64, buffer=shm_walks_memory.buf)

        for i in tqdm(row_list):
            next_idx = i
            walks[i, 0] = i
            for j in range(1, walks.shape[1]):
                draw = np.random.rand()
                next_idx = np.searchsorted(mat[next_idx], draw)
                if next_idx == mat.shape[1]:
                    next_idx = mat.shape[1] - 1
                walks[i, j] = next_idx

    def go(self):
        if current_process().name == "MainProcess":

            shape_mat = self.cumsum_adjmat.shape
            shape_walks = self.walk_traj.shape

            shm_mat, mat = self.create_shared_arr(self.cumsum_adjmat)
            shm_walks, walks = self.create_shared_arr(self.walk_traj)

            print('mat', mat)
            print(shm_mat)

            print('walks', walks)
            print(shm_walks)

            row_dict = self.task_allocation(walks.shape[0], self.cpu_count)

            process_list = []
            for i in range(self.cpu_count):
                process = Process(target=self.random_walks,
                                  args=(shm_mat.name, shape_mat, shm_walks.name, shape_walks, row_dict[i],))
                process_list.append(process)

            for process in process_list:
                process.start()

            for process in process_list:
                process.join()

            # self.walks = walks.copy().astype(np.int64)

            shm_mat.close()
            shm_mat.unlink()

            shm_walks.close()
            shm_walks.unlink()

            return 0


rw = RandomWalkOnGraph(adjmat=[1, 2, 3], walk_steps=1000, cpu_count=10)'''
