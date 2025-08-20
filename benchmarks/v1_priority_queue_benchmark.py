#!/usr/bin/env python3
"""Benchmark script for PriorityRequestQueue lazy deletion optimization.

This script compares the performance of remove operations before and after
the lazy deletion optimization.
"""

import argparse
import random
import time
from dataclasses import dataclass
from typing import List

import heapq


@dataclass 
class MockRequest:
    """Mock request for benchmarking."""
    request_id: str
    priority: int
    arrival_time: float
    
    def __hash__(self):
        return hash(self.request_id)
    
    def __eq__(self, other):
        return self.request_id == other.request_id


class OriginalPriorityQueue:
    """Original implementation with O(n) heapify on remove."""
    
    def __init__(self):
        self._heap = []
    
    def add_request(self, request):
        heapq.heappush(self._heap, 
                      (request.priority, request.arrival_time, request))
    
    def pop_request(self):
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request
    
    def remove_requests(self, requests):
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap 
                     if r not in requests_to_remove]
        heapq.heapify(self._heap)  # O(n) operation
    
    def __len__(self):
        return len(self._heap)
    
    def __bool__(self):
        return bool(self._heap)


class OptimizedPriorityQueue:
    """Optimized implementation with lazy deletion."""
    
    def __init__(self, rebuild_threshold=0.5):
        self._heap = []
        self._deleted = set()
        self._size = 0
        self._rebuild_threshold = rebuild_threshold
    
    def add_request(self, request):
        self._deleted.discard(request)
        heapq.heappush(self._heap,
                      (request.priority, request.arrival_time, request))
        self._size += 1
    
    def pop_request(self):
        while self._heap:
            priority, arrival, request = heapq.heappop(self._heap)
            if request not in self._deleted:
                self._size -= 1
                return request
            else:
                self._deleted.discard(request)
        raise IndexError("pop from empty heap")
    
    def remove_requests(self, requests):
        for request in requests:
            if request not in self._deleted:
                self._deleted.add(request)
                self._size -= 1
        self._maybe_rebuild()
    
    def _maybe_rebuild(self):
        if len(self._deleted) > len(self._heap) * self._rebuild_threshold:
            self._rebuild()
    
    def _rebuild(self):
        old_heap = self._heap
        self._heap = []
        for priority, arrival, request in old_heap:
            if request not in self._deleted:
                heapq.heappush(self._heap, (priority, arrival, request))
        self._deleted.clear()
    
    def __len__(self):
        return self._size
    
    def __bool__(self):
        return self._size > 0


def benchmark_remove_operations(queue_class, num_requests: int, 
                               batch_size: int, num_iterations: int) -> dict:
    """Benchmark batch remove operations."""
    
    # Create requests
    all_requests = [
        MockRequest(f"req_{i}", i % 10, float(i))
        for i in range(num_requests * 2)  # Extra for re-adding
    ]
    
    # Initialize queue
    queue = queue_class()
    for req in all_requests[:num_requests]:
        queue.add_request(req)
    
    # Benchmark remove operations
    remove_times = []
    request_pool = all_requests[:num_requests]
    extra_pool = all_requests[num_requests:]
    
    for i in range(num_iterations):
        # Select random batch to remove
        current_size = len(queue)
        if current_size < batch_size:
            # Re-add some requests
            for req in extra_pool[:batch_size]:
                queue.add_request(req)
            current_size = len(queue)
        
        batch = random.sample(request_pool, min(batch_size, current_size))
        
        # Time the remove operation
        start = time.perf_counter()
        queue.remove_requests(batch)
        elapsed = time.perf_counter() - start
        remove_times.append(elapsed)
        
        # Re-add half of removed requests
        for req in batch[::2]:
            queue.add_request(req)
    
    # Calculate statistics
    avg_time = sum(remove_times) / len(remove_times)
    min_time = min(remove_times)
    max_time = max(remove_times)
    
    return {
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'total_time_ms': sum(remove_times) * 1000,
    }


def run_comparison(args):
    """Run performance comparison."""
    
    print("=" * 70)
    print("PriorityRequestQueue Lazy Deletion Benchmark")
    print("=" * 70)
    print(f"Queue size: {args.queue_size} requests")
    print(f"Batch size: {args.batch_size} requests per removal")
    print(f"Iterations: {args.iterations}")
    print()
    
    # Warm up
    for _ in range(10):
        benchmark_remove_operations(OriginalPriorityQueue, 100, 10, 5)
        benchmark_remove_operations(OptimizedPriorityQueue, 100, 10, 5)
    
    # Run benchmarks
    print("Running original implementation...")
    orig_results = benchmark_remove_operations(
        OriginalPriorityQueue, 
        args.queue_size,
        args.batch_size,
        args.iterations
    )
    
    print("Running optimized implementation...")
    opt_results = benchmark_remove_operations(
        OptimizedPriorityQueue,
        args.queue_size,
        args.batch_size,
        args.iterations
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nOriginal Implementation (O(n) heapify):")
    print(f"  Average remove time: {orig_results['avg_time_ms']:.3f} ms")
    print(f"  Min remove time:     {orig_results['min_time_ms']:.3f} ms")
    print(f"  Max remove time:     {orig_results['max_time_ms']:.3f} ms")
    print(f"  Total time:          {orig_results['total_time_ms']:.3f} ms")
    
    print("\nOptimized Implementation (Lazy Deletion):")
    print(f"  Average remove time: {opt_results['avg_time_ms']:.3f} ms")
    print(f"  Min remove time:     {opt_results['min_time_ms']:.3f} ms")
    print(f"  Max remove time:     {opt_results['max_time_ms']:.3f} ms")
    print(f"  Total time:          {opt_results['total_time_ms']:.3f} ms")
    
    # Calculate improvement
    speedup = orig_results['avg_time_ms'] / opt_results['avg_time_ms']
    improvement = (orig_results['avg_time_ms'] - opt_results['avg_time_ms']) / orig_results['avg_time_ms'] * 100
    
    print("\n" + "=" * 70)
    print("PERFORMANCE IMPROVEMENT")
    print("=" * 70)
    print(f"  Speedup:     {speedup:.1f}x faster")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Time saved:  {orig_results['total_time_ms'] - opt_results['total_time_ms']:.3f} ms")
    
    # Test different queue sizes
    if args.test_scaling:
        print("\n" + "=" * 70)
        print("SCALING TEST")
        print("=" * 70)
        print("\nTesting with different queue sizes:")
        print("-" * 50)
        
        test_sizes = [100, 500, 1000, 5000, 10000]
        for size in test_sizes:
            orig = benchmark_remove_operations(
                OriginalPriorityQueue, size, args.batch_size, 20
            )
            opt = benchmark_remove_operations(
                OptimizedPriorityQueue, size, args.batch_size, 20
            )
            speedup = orig['avg_time_ms'] / opt['avg_time_ms']
            print(f"Queue size {size:5d}: {speedup:.1f}x speedup "
                  f"({orig['avg_time_ms']:.3f}ms → {opt['avg_time_ms']:.3f}ms)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PriorityRequestQueue lazy deletion optimization"
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=1000,
        help="Number of requests in the queue (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of requests to remove per batch (default: 20)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of remove operations to perform (default: 100)"
    )
    parser.add_argument(
        "--test-scaling",
        action="store_true",
        help="Test performance with different queue sizes"
    )
    
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()