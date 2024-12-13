import numpy as np
import simsimd
import time

if __name__ == '__main__':
    d = 256
    n = 8388608
    vec1 = np.random.randn(d).astype(np.float32) # rank 1 tensor
    batch2 = np.random.randn(n, d).astype(np.float32)
    print("Created centroid of size", (n, d))

    t1 = time.time()
    dist_rank1 = simsimd.dot(vec1, batch2)
    t2 = time.time()

    time_taken_sec = t2 - t1
    centroids_size_gb = (n * d * 4.0)/(10.0 ** 9)
    measured_throughput = centroids_size_gb/time_taken_sec
    print("Took", time_taken_sec, "sec to scan", centroids_size_gb, "GB resulting in throughput", measured_throughput)
    print("Got result", dist_rank1)